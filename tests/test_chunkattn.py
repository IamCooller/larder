#!/usr/bin/env python3
"""chunkattn.py: query-chunked prefill attention for head dims with no fused
MLX kernel (full-attention layers with D=256).

    python3 tests/test_chunkattn.py      # toy only, random tensors and a toy qwen3

Nine checks:

 1. numerics: chunked equals a whole-T stock pass (array_equal) on a 16 heads / 2 kv / D=256
    shape and a smaller toy shape across N, chunk width and offset, and widths below the
    calibrated min_rows are shown to diverge so the guard is not decorative.
 2. calibrate: _kv_slope reads only allocator jitter for a fused head_dim (128) and a real
    slope for the unfused one (256), so calibrate() routes the second and leaves the first
    on stock, with _min_safe_rows tight per shape (the width just below it diverges).
 3. dispatch: a toy bf16 qwen3 with D=256 installed stays bit-identical to stock for an empty
    table, an unrouted shape, a routed shape with no budget and plain decode (none call
    chunked()), chunks under a tight budget, and leaves a prompt long enough that stock fuses
    (T >= 1024 on newer MLX) to stock.
 4. peak: chunked() lowers the transient vs one whole-T stock call at a 16 heads / 2 kv / D=256
    shape by forcing each chunk (mx.eval) before the next is built, at T < 1024 where stock is
    unfused.
 5. rows: _wrap's cap -> rows math uses per_key as measured (already summed over every head)
    and falls back to one unbounded pass with one warning when the budget is already spent.
 6. compose: install() wraps around a module another patcher already wrapped (foldattn's
    shared .stock marker), because the wrap loop tests chunkattn's own marker.
 7. save-guard: a calibration cache save that fails (full disk, read-only ~/.cache) does not
    raise out of install(), since an optional memory lever must not stop StreamingLLM() loading.
 8. budget: prefill_cap_bytes() subtracts host_gb from the budget and is capped at its ceiling,
    so a roomy --mem-pct budget cannot leave the bounded transient effectively unbounded.
 9. cold cap: that ceiling is the MoE output of a prefill over all N keys, so on the
    16 heads / 2 kv / D=256 shape the chunks it gives a cold prompt are array_equal to stock
    and a narrow pass (verify, a warm turn) over a long context stays one stock call.
"""

import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chunkattn                                               # noqa: E402
import stream_fast                                             # noqa: E402
from mlx_lm.models import qwen3                                # noqa: E402
from mlx_lm.models.cache import make_prompt_cache               # noqa: E402

SMALL_CLIENT_ROW = 8 * 2048     # top_k x hidden: one position's MoE output


def check_numerics():
    mx.random.seed(0)
    cases = 0
    min_rows_of = {}
    for hq, hkv, d in ((16, 2, 256), (4, 2, 256)):
        table = chunkattn.calibrate({(hq, hkv, d)}, mx.bfloat16)
        assert (hq, hkv, d) in table, (hq, hkv, d, "expected an unfused route")
        _, min_rows, floor = table[(hq, hkv, d)]
        assert floor, (hq, hkv, d, "expected the causal key trim to be exact")
        min_rows_of[(hq, hkv, d)] = min_rows
        scale = d ** -0.5
        for n in (37, 300, 4100):
            for extra in (0, 50):   # extra: keys already in cache (offset > 0)
                t = min(n, 33)
                total_n = n + extra
                q = mx.random.normal((1, hq, t, d)).astype(mx.bfloat16)
                k = mx.random.normal((1, hkv, total_n, d)).astype(mx.bfloat16)
                v = mx.random.normal((1, hkv, total_n, d)).astype(mx.bfloat16)
                offset = total_n - t
                row_pos = offset + mx.arange(t)
                col_pos = mx.arange(total_n)
                full_mask = row_pos[:, None] >= col_pos[None, :]
                ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale,
                                                            mask=full_mask)
                mx.eval(ref)
                for rows in (min_rows, min_rows * 2, min_rows * 5, t, t + 5):
                    for fl in (0, floor):       # every key / the causal key trim
                        out = chunkattn.chunked(q, k, v, scale, rows, min_rows, fl)
                        mx.eval(out)
                        assert mx.array_equal(ref, out), (hq, hkv, d, n, extra, rows, fl)
                        cases += 1
    # min_rows exists because some width below it does not match, so this confirms it
    # on the 16-head shape and proves the guard load-bearing rather than just present.
    hq, hkv, d = 16, 2, 256
    n, t = 1024, 200
    q = mx.random.normal((1, hq, t, d)).astype(mx.bfloat16)
    k = mx.random.normal((1, hkv, n, d)).astype(mx.bfloat16)
    v = mx.random.normal((1, hkv, n, d)).astype(mx.bfloat16)
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=d ** -0.5, mask="causal")
    mx.eval(ref)
    bad_rows = 3   # below the calibrated floor found for this shape above
    assert bad_rows < min_rows_of[(16, 2, 256)]
    out = chunkattn.chunked(q, k, v, d ** -0.5, bad_rows, 1)   # min_rows=1: don't clamp
    mx.eval(out)
    diverges = not mx.array_equal(ref, out)
    print(f"numerics: {cases} exact cases; width={bad_rows} without the "
          f"min_rows floor {'diverges as expected' if diverges else 'happened to match'} "
          f"from stock, confirming the floor is load-bearing")
    assert diverges, "expected an unguarded small chunk to diverge from stock"


def _toy():
    args = qwen3.ModelArgs(
        model_type="qwen3", hidden_size=64, num_hidden_layers=2,
        intermediate_size=128, num_attention_heads=4, rms_norm_eps=1e-6,
        vocab_size=256, num_key_value_heads=2, max_position_embeddings=8192,
        rope_theta=1e6, head_dim=256, tie_word_embeddings=True)
    mx.random.seed(1)
    model = qwen3.Model(args)
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    return model


def check_calibrate():
    # A head_dim MLX fuses (128) against one it does not (256): _kv_slope
    # takes the max of several repeats (see its docstring) so allocator
    # jitter never reads as "fused" when it should not -- on this machine
    # that jitter alone puts D=128 at a few B/row/key, never exactly 0.
    # calibrate()'s threshold (~Hq * dtype.size) must still keep it OUT of
    # the table -- `slope <= 0.0` never excluded it (that jitter is always
    # > 0), which routed a flash kernel through chunking for zero memory
    # bounded (no [Hq, T, N] buffer there to bound in the first place).
    table = chunkattn.calibrate({(16, 2, 128), (16, 2, 256), (4, 2, 256)}, mx.bfloat16)
    assert set(table) >= {(16, 2, 256), (4, 2, 256)}, table
    assert (16, 2, 128) not in table, (
        "fused D=128 got routed -- calibrate()'s fused/unfused threshold regressed", table)
    for (hq, hkv, d), (per_key, rows, _) in table.items():
        assert per_key > 0 and rows in chunkattn.ROW_CANDIDATES, (hq, hkv, d, per_key, rows)
    fused_slope = table.get((16, 2, 128), (0.0, None))[0]
    unfused_slope = table[(16, 2, 256)][0]
    assert unfused_slope > 3 * fused_slope, (fused_slope, unfused_slope)
    # Each shape's min_rows is its own and tight: the candidate just below
    # it diverges for THAT shape (fresh draws). Which width that is moves
    # with the shape and the MLX build (0.32.0: 16 heads 5, 4 heads 16, the
    # 4-head one's narrow chunks hit the vector kernel; 0.32.2+ on M3 and
    # newer: 16 for both, the wide gemv takes 2-15 rows), so reusing one
    # shape's width for another, or one machine's for another, is exactly
    # the mistake calibrating per shape exists to avoid.
    for (hq, hkv, d), (_, rows, _) in sorted(table.items()):
        below = [w for w in chunkattn.ROW_CANDIDATES if w < rows]
        if below:
            assert not chunkattn._exact_at(hq, hkv, d, mx.bfloat16, 256, 256, below[-1]), (
                (hq, hkv, d), rows, below[-1], "the width below min_rows matched too")
    print(f"calibrate: D=128 {fused_slope:.1f} B/row/key vs D=256 {unfused_slope:.1f}; "
          f"min_rows {(16, 2, 256)}={table[(16, 2, 256)][1]} "
          f"{(4, 2, 256)}={table[(4, 2, 256)][1]}, each tight")


def check_dispatch():
    model = _toy()
    table = chunkattn.calibrate({(4, 2, 256)}, mx.bfloat16)
    assert (4, 2, 256) in table

    def logits(prompt):
        cache = make_prompt_cache(model)
        out = model(prompt, cache=cache)
        mx.eval(out)
        return out

    prompt = mx.random.randint(0, 256, (1, 300))
    stock_out = logits(prompt)

    real_chunked = chunkattn.chunked
    calls = []
    chunkattn.chunked = lambda *a, **k: calls.append(a) or real_chunked(*a, **k)
    orig_cap = stream_fast.prefill_cap_bytes
    try:
        chunkattn.install(model, table={})
        assert mx.array_equal(logits(prompt), stock_out) and not calls
        print("dispatch: empty table -- bit-identical, 0 chunk calls")

        chunkattn.install(model, table=table)
        assert mx.array_equal(logits(prompt), stock_out) and not calls
        print("dispatch: routed shape, no budget configured -- bit-identical, "
              "0 chunk calls")

        # A FIXED cap, not set_prefill_budget_gb + live mx.get_active_memory():
        # this process's active memory is already several MB by here (prior
        # checks' own SDPA probes), so a budget meant to land tight-but-
        # positive would silently read as cap<=0 instead -- exactly the
        # "budget already exceeded" case _wrap now bows out of rather than
        # chunks (see check_rows_and_budget). 500000 B is comfortably above
        # this shape's own min_rows*N*per_key floor, so it forces several
        # chunks without tripping that path.
        stream_fast.prefill_cap_bytes = lambda *a: 500_000.0
        out = logits(prompt)
        assert mx.array_equal(out, stock_out), "chunked output differs from stock"
        assert calls, "expected chunking under a tight budget"
        print(f"dispatch: routed shape, tight budget -- bit-identical, "
              f"{len(calls)} chunk calls")

        # At T >= 1024 newer MLX fuses stock, whose bits the chunks' fallback kernels cannot
        # reproduce and which holds no score matrix to bound, so it is left to stock.
        long_prompt = mx.random.randint(0, 256, (1, 1100))
        fused = chunkattn._stock_fused(4, 2, 256, mx.bfloat16, 1100, 1100)
        stream_fast.prefill_cap_bytes = orig_cap
        long_stock = logits(long_prompt)
        calls.clear()
        stream_fast.prefill_cap_bytes = lambda *a: 500_000.0
        assert mx.array_equal(logits(long_prompt), long_stock), "T=1100 differs from stock"
        assert bool(calls) != fused, (fused, len(calls))
        print(f"dispatch: T=1100, tight budget -- bit-identical, stock "
              + ("fuses here: left to it, 0 chunk calls" if fused
                 else f"unfused here: {len(calls)} chunk calls"))
        stream_fast.prefill_cap_bytes = orig_cap

        calls.clear()
        cache = make_prompt_cache(model)
        mx.eval(model(prompt, cache=cache))
        stream_fast.prefill_cap_bytes = lambda *a: 500_000.0
        mx.eval(model(mx.random.randint(0, 256, (1, 1)), cache=cache))
        assert not calls, "decode (T=1) must never chunk"
        print("dispatch: decode T=1 -- 0 chunk calls even under a tight budget")
    finally:
        stream_fast.prefill_cap_bytes = orig_cap
        chunkattn.chunked = real_chunked
        chunkattn.install(model, table={})


def check_peak():
    """chunked() must actually LOWER the transient vs one whole-T stock call (16 heads / 2 kv / D=256, T=1000 rows over N=1400 keys, unfused), since building every chunk lazily and forcing the graph once keeps every score buffer alive and peaks higher than stock."""
    hq, hkv, d = 16, 2, 256
    T, N = 1000, 1400
    assert not chunkattn._stock_fused(hq, hkv, d, mx.bfloat16, T, N), (
        "stock fuses at T=1000 on this MLX -- pick a T it runs unfused")
    scale = d ** -0.5
    mx.random.seed(3)
    q = mx.random.normal((1, hq, T, d)).astype(mx.bfloat16)
    k = mx.random.normal((1, hkv, N, d)).astype(mx.bfloat16)
    v = mx.random.normal((1, hkv, N, d)).astype(mx.bfloat16)
    mx.eval(q, k, v)

    # Chunked first: a buffer freed by a finished kernel can still count as
    # active for a moment after mx.eval returns, so whichever call runs
    # second may start from the first one's leftovers -- stock's score matrix
    # (measured: 73.7 MB still active right after its eval at this shape)
    # would swamp the chunks' own peak, a chunk's few MB cannot swamp stock's.
    mx.reset_peak_memory()
    # min_rows=32: the 8-row tail folds into the chunk before it (an 8-row
    # chunk is under the 16-row floor MLX 0.32.2+ calibrates on M3 and newer)
    out = chunkattn.chunked(q, k, v, scale, rows=32, min_rows=32)
    mx.eval(out)
    chunked_peak = mx.get_peak_memory()

    mx.reset_peak_memory()
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal")
    mx.eval(ref)
    stock_peak = mx.get_peak_memory()

    assert mx.array_equal(ref, out), "chunked output differs from stock"
    assert chunked_peak < stock_peak, (
        f"chunked ({chunked_peak / 1e6:.1f} MB) did not beat stock "
        f"({stock_peak / 1e6:.1f} MB) -- chunking bought nothing")
    print(f"peak: stock {stock_peak / 1e6:.1f} MB, chunked(rows=32) "
          f"{chunked_peak / 1e6:.1f} MB, bit-identical")


def check_rows_and_budget():
    """_wrap's cap -> rows math must not divide by Hq twice (per_key is
    already summed over every head), and must fall back to an unbounded pass
    -- once, with one warning -- rather than collapsing to a useless
    min_rows when the budget is already spent before this call even starts.
    """
    model = _toy()
    hq, hkv, d = 4, 2, 256   # _toy()'s own attention shape
    per_key, min_rows = 100.0, 5   # hand-set, not measured: keeps the math
                                    # fully predictable for this check
    chunkattn.install(model, table={(hq, hkv, d): (per_key, min_rows)})

    real_chunked = chunkattn.chunked
    seen_rows = []

    def spy(q, k, v, scale, rows, mr):
        seen_rows.append(rows)
        return real_chunked(q, k, v, scale, rows, mr)

    chunkattn.chunked = spy
    orig_cap = stream_fast.prefill_cap_bytes
    prompt = mx.random.randint(0, 256, (1, 300))   # T = N = 300, cold cache
    try:
        # correct: 900000 // (300 * 100) = 30. A reverted double-count
        # (dividing by Hq=4 again) would instead read 900000 // (4*300*100)
        # = 7 -- the two are far enough apart that either bug shows plainly.
        stream_fast.prefill_cap_bytes = lambda *a: 900_000.0
        mx.eval(model(prompt, cache=make_prompt_cache(model)))
        # One call per attention layer (_toy() has 2), all at the same rows:
        # same shape, same N, same fixed cap every time.
        assert seen_rows and all(r == 30 for r in seen_rows), (
            "rows should be cap // (N * per_key) == 30 for every call, got", seen_rows)
        print(f"rows: cap=900000 B, N=300, per_key=100 -> rows={seen_rows[0]} "
              f"x{len(seen_rows)} layers (a reverted Hq-double-count would read 7)")

        seen_rows.clear()
        chunkattn._warned_over_budget = False
        warnings = []
        real_print = print

        def spy_print(*a, **k):
            msg = " ".join(str(x) for x in a)
            if "budget already exceeded" in msg:
                warnings.append(msg)
            real_print(*a, **k)

        import builtins
        builtins.print = spy_print
        try:
            stream_fast.prefill_cap_bytes = lambda *a: 0.0
            mx.eval(model(prompt, cache=make_prompt_cache(model)))
            mx.eval(model(prompt, cache=make_prompt_cache(model)))
        finally:
            builtins.print = real_print
        assert not seen_rows, f"over-budget must not chunk at all, saw {seen_rows}"
        assert len(warnings) == 1, f"expected exactly one warning, got {warnings}"
        print("budget: cap<=0 falls back to unbounded (0 chunk calls), warns once")
    finally:
        stream_fast.prefill_cap_bytes = orig_cap
        chunkattn.chunked = real_chunked
        chunkattn._warned_over_budget = False
        chunkattn.install(model, table={})


def check_composes_with_other_patcher():
    """install()'s wrap loop tests its own `.chunkattn` marker, not the shared `.stock` foldattn.py also sets, so a module already wrapped by someone else is wrapped here too, around whatever is currently installed, instead of skipping calibration and wrapping after a swap to a target sharing foldattn's module."""
    model = _toy()
    mods, _, _ = chunkattn._attention_modules(model)
    assert mods, "toy model exposes no attention modules to test against"
    saved = {mod: mod.scaled_dot_product_attention for mod in mods}

    def other_patcher(*a, **k):
        raise AssertionError("never actually called in this check")
    other_patcher.stock = other_patcher   # foldattn's shared marker, no .chunkattn
    for mod in mods:
        mod.scaled_dot_product_attention = other_patcher

    try:
        table = chunkattn.install(model, table={(4, 2, 256): (100.0, 5)})
        assert table == {(4, 2, 256): (100.0, 5)}
        for mod in mods:
            fn = mod.scaled_dot_product_attention
            assert hasattr(fn, "chunkattn") and fn.stock is other_patcher, (
                "install() left a module another patcher wrapped alone "
                "instead of composing around it")
        print("compose: wraps around another patcher's wrapper via its own marker")
    finally:
        for mod, fn in saved.items():
            mod.scaled_dot_product_attention = fn


def check_save_failure_does_not_crash():
    """A full disk / read-only ~/.cache must not block model load: the
    calibration cache save is best-effort, the in-memory table calibrate()
    returned is used regardless (review finding, 2026-09-29)."""
    model = _toy()
    real_read_text, real_mkdir = Path.read_text, Path.mkdir

    def raise_read(self, *a, **k):
        raise OSError(2, "no such file")   # forces the calibrate-and-save path

    def raise_mkdir(self, *a, **k):
        raise OSError(28, "no space left on device")

    Path.read_text, Path.mkdir = raise_read, raise_mkdir
    try:
        table = chunkattn.install(model)   # table=None: the calibrate-and-save path
        assert (4, 2, 256) in table, "calibration must succeed even if saving it fails"
        print("save-guard: mkdir/write failure does not crash install()")
    finally:
        Path.read_text, Path.mkdir = real_read_text, real_mkdir
        chunkattn.install(model, table={})


def check_budget_ceiling():
    """prefill_cap_bytes(): host_gb is actually subtracted, and the MoE-row ceiling dominates a roomy --mem-pct 90 budget, so without a ceiling the cap would always clear T and chunking would never fire."""
    saved = (stream_fast._PREFILL_BUDGET_GB, stream_fast._PREFILL_HOST_GB,
              stream_fast._PREFILL_ROW_ELEMS)
    try:
        # A ceiling well under a huge budget's raw room must win outright,
        # regardless of how much this process has already allocated.
        stream_fast.set_prefill_budget_gb(1000.0, host_gb=0.0, row_elems=SMALL_CLIENT_ROW)
        assert stream_fast.prefill_cap_bytes(5528, 2) == 5528 * SMALL_CLIENT_ROW * 2, (
            "a roomy budget must not swamp the MoE-row ceiling")

        # host_gb's effect, read back to back so mx.get_active_memory() is
        # the same both times -- only host_gb should move the answer: by half
        # of it, the cap being half the room (two chunks' pages).
        stream_fast.set_prefill_budget_gb(10.0, host_gb=0.0, row_elems=None)
        cap_no_host = stream_fast.prefill_cap_bytes(5528, 2)
        stream_fast.set_prefill_budget_gb(10.0, host_gb=2.0, row_elems=None)
        cap_with_host = stream_fast.prefill_cap_bytes(5528, 2)
        assert abs((cap_no_host - cap_with_host) - 1.0e9) < 1e6, (
            "host_gb must reduce prefill_cap_bytes by host_gb*1e9 / 2",
            cap_no_host, cap_with_host)
        print("budget: MoE-row ceiling dominates a roomy budget; host_gb reaches "
              "prefill_cap_bytes()")
    finally:
        (stream_fast._PREFILL_BUDGET_GB, stream_fast._PREFILL_HOST_GB,
         stream_fast._PREFILL_ROW_ELEMS) = saved


def check_cold_front_cap():
    """The ceiling is the MoE output of a prefill over all N keys (top_k x hidden per position), no longer two WIDE_GROUPs of expert scratch, so on the 16 heads / 2 kv / D=256 shape the chunks it gives a cold prompt are array_equal to one stock pass and a narrow pass over a long context (decode, a warm turn) still gets one stock call."""
    table = chunkattn.calibrate({(16, 2, 256)}, mx.bfloat16)
    per_key, min_rows, floor = table[(16, 2, 256)]
    saved = (stream_fast._PREFILL_BUDGET_GB, stream_fast._PREFILL_HOST_GB,
              stream_fast._PREFILL_ROW_ELEMS)
    try:
        stream_fast.set_prefill_budget_gb(1000.0, host_gb=0.0, row_elems=SMALL_CLIENT_ROW)
        rows_of = lambda n: int(stream_fast.prefill_cap_bytes(n, 2) // (n * per_key))
        n = 1000                     # a cold prompt, T = N, under 1024 where newer MLX fuses
                                     # stock and _attend leaves it to it (check_dispatch)
        assert not chunkattn._stock_fused(16, 2, 256, mx.bfloat16, n, n), n
        rows = rows_of(n)
        assert min_rows <= rows < n, (rows, n)
        old_rows = int(503e6 // (n * per_key))
        assert old_rows >= n, "the old ceiling never chunked a prompt this short"
        mx.random.seed(3)
        q = mx.random.normal((1, 16, n, 256)).astype(mx.bfloat16)
        k = mx.random.normal((1, 2, n, 256)).astype(mx.bfloat16)
        v = mx.random.normal((1, 2, n, 256)).astype(mx.bfloat16)
        ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=256 ** -0.5,
                                                    mask="causal")
        out = chunkattn.chunked(q, k, v, 256 ** -0.5, rows, min_rows)
        assert mx.array_equal(ref, out).item(), "cold-front chunks differ from stock"
        out = chunkattn.chunked(q, k, v, 256 ** -0.5, rows, min_rows, floor)
        assert mx.array_equal(ref, out).item(), "causal-trimmed chunks differ from stock"
        for t in (2, 7, 95):         # verify passes, a warm turn's rest, at 5.6k keys
            assert rows_of(5623) >= t, (t, rows_of(5623))
        print(f"cold-front cap: {rows} rows per chunk at N={n} (old ceiling: "
              f"{old_rows}, one stock pass), array_equal to stock; "
              f"{rows_of(5623)} rows at N=5623, so passes up to that stay stock")
    finally:
        (stream_fast._PREFILL_BUDGET_GB, stream_fast._PREFILL_HOST_GB,
         stream_fast._PREFILL_ROW_ELEMS) = saved


def check_row_elems_dense_model():
    """moe_row_elems() must not explode on a model with no streamed experts:
    moe_stream.py's install call site calls it unconditionally, dense verify
    targets included."""
    model = _toy()
    assert stream_fast.moe_row_elems(model) == 0
    print("ceiling: dense model (no packed experts) -> 0, no crash")


if __name__ == "__main__":
    check_numerics()
    check_calibrate()
    check_dispatch()
    check_peak()
    check_rows_and_budget()
    check_composes_with_other_patcher()
    check_save_failure_does_not_crash()
    check_budget_ceiling()
    check_cold_front_cap()
    check_row_elems_dense_model()
    print("ok")
