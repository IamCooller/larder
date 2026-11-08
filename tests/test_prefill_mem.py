#!/usr/bin/env python3
"""Prefill memory levers of the MoE engine, exact on toys (a 2 GB cold-front
profile).

  1. conv state: on a prefill pass each layer's GatedDeltaNet conv state is
                    evaluated with the layer's dense half, so its whole lazy
                    conv input is freed inside the layer, not held to the end
                    of the pass: measured with mx.get_active_memory right after
                    a lazy toy pass, against the input bytes it used to pin.
  2. layer sync: the layer's output is evaluated at its end; with 1, the
                    prefilled cache, the logits and the greedy tokens after it
                    are array_equal with and without (per-instance switch).
  3. wide rows: the grouped and split wide paths gather rows from x and
                    write them straight into one unsorted output; against the
                    old arithmetic (sorted copy of x, concatenate, unsort
                    gather), array_equal on real per-row shapes (hidden 2048,
                    expert width 512, top-8, 4-bit g64, bf16) and through the
                    whole toy engine, with a lower peak.
  4. scratch: the wide scratch is dropped at the end of every prefill.
  6. install: the cache-bias check leaves layer 0's arena empty, so the
                    front no longer depends on its unseeded random pass: two
                    installs seeded differently prefill identical caches.
  7. turn end: MLX's buffer cache is empty after a turn.

A tiny qwen3_5_moe (the 2 GB model's architecture: hybrid GatedDeltaNet/attention, MoE
on every layer, cache bias) is built, packed with repack.py into a temporary
directory, and run through stream_fast.install, no real weights, well
under 200 MB. Fix 7 uses moe_stream's toy engine (the real Qwen tokenizer).

    python3 tests/test_prefill_mem.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models import qwen3_5_moe as q35
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.switch_layers import SwitchGLU

import stream_fast

TEXT = dict(
    model_type="qwen3_5_moe_text", hidden_size=64, num_hidden_layers=8,
    num_attention_heads=2, num_key_value_heads=1, head_dim=32,
    # A wide conv input (2 x 4 x 64 + 8 x 64 = 1024 channels) so the bytes
    # check 1 is about stand well clear of everything else a toy pass holds.
    linear_num_key_heads=4, linear_key_head_dim=64, linear_num_value_heads=8,
    linear_value_head_dim=64, linear_conv_kernel_dim=4,
    num_experts=16, num_experts_per_tok=2, moe_intermediate_size=64,
    shared_expert_intermediate_size=64, decoder_sparse_step=1,
    full_attention_interval=4, vocab_size=256, rms_norm_eps=1e-6,
    tie_word_embeddings=False, max_position_embeddings=4096,
)
CFG = {"model_type": "qwen3_5_moe", "text_config": TEXT}
RESIDENT = 4          # arena slots: under the 16 experts, so wide passes stay wide


def fresh_model():
    m = q35.Model(q35.ModelArgs.from_dict(CFG))
    pred = m.quant_predicate
    nn.quantize(m, group_size=32, bits=4,
                class_predicate=lambda p, mod: hasattr(mod, "to_quantized") and pred(p, mod))
    return m


def save(out, weights, cfg):
    out.mkdir(parents=True)
    mx.save_safetensors(str(out / "model.safetensors"), weights)
    idx = {"metadata": {"total_size": (out / "model.safetensors").stat().st_size},
           "weight_map": {k: "model.safetensors" for k in weights}}
    (out / "model.safetensors.index.json").write_text(json.dumps(idx))
    (out / "config.json").write_text(json.dumps(cfg))
    subprocess.run([sys.executable, str(REPO / "tools" / "repack.py"), "--src", str(out)],
                   check=True, capture_output=True)


def engine(out, seed):
    """The packed toy as moe_stream builds it: fused GLUs, cache bias (so the
    install check runs), an arena. `seed` feeds the check's random pass."""
    m = fresh_model()
    m.load_weights(str(out / "model.safetensors"))
    mx.eval(m.parameters())
    mx.random.seed(seed)
    stream_fast.install(m, out, resident=RESIDENT, cache_bias=0.5)
    return m


def set_sync(m, on):
    for l in m.layers:
        l.__dict__["_prefill_on"] = on


def decode(m, cache, tok, n):
    toks = []
    for _ in range(n):
        tok = int(mx.argmax(m(mx.array([[tok]]), cache=cache)[0, -1]).item())
        toks.append(tok)
    return toks


def session(m, on):
    """A cold prompt (the grouped wide path on every layer), decode (fills the
    arenas), a second prompt over them (the split path), decode again --
    moe_stream._prefill's way: only the cache is evaluated."""
    set_sync(m, on)
    rng = np.random.default_rng(7)
    p1, p2 = rng.integers(0, 256, 700).tolist(), rng.integers(0, 256, 64).tolist()
    cache = make_prompt_cache(m)
    m(mx.array(p1[:-1])[None], cache=cache)
    mx.eval([c.state for c in cache])
    scratch = bool(stream_fast._WIDE_SCRATCH)
    toks = decode(m, cache, p1[-1], 4)
    s0 = stream_fast.STATS["wide_split"]
    logits = m(mx.array(p2[:-1])[None], cache=cache)[0, -1]
    split = stream_fast.STATS["wide_split"] - s0
    toks += decode(m, cache, p2[-1], 4)
    arrays = [a for c in cache for a in c.state]
    mx.eval(arrays, logits)
    return toks, arrays + [logits], scratch, split


def same(a, b):
    return (a[0] == b[0] and len(a[1]) == len(b[1])
            and all(mx.array_equal(x, y).item() for x, y in zip(a[1], b[1])))


def clear_arenas(m):
    for g in stream_fast.glus_of(m):
        g.clear_arena()
    stream_fast._WIDE_SCRATCH.clear()


# The wide path before this change, verbatim but for the signature, for
# check 3's comparison.
def old_grouped(self, x, unique_ids, pos):
    G = stream_fast.WIDE_GROUP
    n = len(unique_ids)
    cuts = np.searchsorted(pos, np.arange(G, n, G))
    bounds = [0, *cuts.tolist(), len(pos)]
    ys = []
    for gi, (r0, r1) in enumerate(zip(bounds, bounds[1:])):
        a = gi * G
        ids = unique_ids[a:a + G]
        pg = pos[r0:r1] - a
        outs = []
        for p in self._projs:
            bufs, lo = p._batch_wide(ids)
            rhs = mx.array(ids[pg] - lo) if lo is not None else mx.array(pg)
            src = x[r0:r1] if len(outs) < 2 else self.activation(outs[1], outs[0])
            outs.append(mx.gather_qmm(src, *bufs, rhs_indices=rhs, transpose=True,
                                      group_size=p.group_size, bits=p.bits,
                                      mode=p.mode, sorted_indices=True))
        ys.append(outs[2])
        if r1 < len(pos):
            mx.eval(ys[-1])
    return mx.concatenate(ys, axis=0)


def old_split(self, x, unique_ids, pos):
    slots = self._slot_np[unique_ids]
    res_u = slots >= 0
    row_res = res_u[pos]
    ir = np.nonzero(row_res)[0]
    im = np.nonzero(~row_res)[0]
    stream_fast.STATS["wide_split"] += 1
    ys = []
    if len(ir):
        rhs = mx.array(slots[pos[ir]])
        prep = [a[0] for a in stream_fast._ARENAS[self._akey]]
        src0 = x[mx.array(ir)]
        outs = []
        for p, bufs in zip(self._projs, prep):
            src = src0 if len(outs) < 2 else self.activation(outs[1], outs[0])
            outs.append(mx.gather_qmm(src, *bufs, rhs_indices=rhs, transpose=True,
                                      group_size=p.group_size, bits=p.bits,
                                      mode=p.mode, sorted_indices=False))
        ys.append(outs[2])
    if len(im):
        miss_ids = unique_ids[~res_u]
        remap = np.cumsum(~res_u) - 1
        ys.append(old_grouped(self, x[mx.array(im)], miss_ids, remap[pos[im]]))
    y = mx.concatenate(ys, axis=0) if len(ys) > 1 else ys[0]
    order = np.argsort(np.concatenate([ir, im]))
    return y[mx.array(order)]


def old_wide(body):
    """Today's signature over the old arithmetic: x sorted in one copy
    (_gather_sort's x[order // k]), the body's sorted rows put back by
    _scatter_unsort's gather (inv_order = argsort(order))."""
    def call(self, x, unique_ids, pos, rows, out, n_rows):
        xs = x[rows // (n_rows // x.shape[0])]
        return body(self, xs, unique_ids, pos)[mx.argsort(rows)]
    return call


class OldWidePath:
    def __enter__(self):
        cls = stream_fast.StreamingSwitchGLU
        self.saved = cls._call_grouped, cls._call_split
        cls._call_grouped, cls._call_split = old_wide(old_grouped), old_wide(old_split)

    def __exit__(self, *exc):
        cls = stream_fast.StreamingSwitchGLU
        cls._call_grouped, cls._call_split = self.saved


def check_engine(tmp):
    out = tmp / "tiny_q35"
    mx.random.seed(0)
    m = fresh_model()
    mx.eval(m.parameters())
    save(out, dict(tree_flatten(m.parameters())),
         {**CFG, "quantization": {"group_size": 32, "bits": 4, "mode": "affine"}})
    del m

    a, b = engine(out, seed=1), engine(out, seed=2)

    # 6. the install check leaves no trace in layer 0's arena ...
    g0 = stream_fast.glus_of(a)[0]
    assert g0._akey not in stream_fast._ARENAS and not g0._lru, \
        "the cache-bias check left experts in layer 0's arena"
    # ... so the front no longer depends on its random pass.
    run_a, run_b = session(a, True), session(b, True)
    assert same(run_a, run_b), "installs seeded differently prefill different caches"
    print(f"6. install check: layer 0 arena empty; two installs seeded 1 and 2 give "
          f"identical caches and tokens {run_a[0]}")

    # 1/2/4. the per-layer prefill sync changes nothing numerically.
    assert run_a[3] > 0, "the second prompt did not take the split wide path"
    clear_arenas(a)
    run_off = session(a, False)
    assert same(run_a, run_off), "prefill sync changed the cache or the tokens"
    assert not run_a[2] and run_off[2], (run_a[2], run_off[2])
    print(f"1/2. prefill sync on == off: {len(run_a[1])} arrays array_equal, "
          f"tokens {run_off[0]}, split passes {run_a[3]}")
    print("4. wide scratch: dropped after the prefill with the sync, kept without")

    # 3. the new wide path against the old arithmetic, grouped and split.
    # With the prefill sync a prefill's union wider than a group no longer
    # splits (its rows all go grouped, out of the whole-layer read-ahead);
    # without it the second prompt still takes the split path.
    saved_group = stream_fast.WIDE_GROUP
    stream_fast.WIDE_GROUP = 3          # 16 experts in six groups
    try:
        for on in (True, False):
            clear_arenas(a)
            run_new = session(a, on)
            clear_arenas(a)
            with OldWidePath():
                run_old = session(a, on)
            assert (run_new[3] > 0) == (not on), (on, run_new[3])
            assert same(run_new, run_old), ("the wide path changed a result", on)
    finally:
        stream_fast.WIDE_GROUP = saved_group
    print(f"3. wide path (groups of 3; grouped + read-ahead, and split) == old "
          f"arithmetic: array_equal, tokens {run_new[0]}")

    # 1. memory: the conv inputs a lazy pass used to pin.
    ids = mx.array(np.random.default_rng(3).integers(0, 256, 1023))[None]
    held = {}
    for on in (False, True):
        clear_arenas(a)
        set_sync(a, on)
        cache = make_prompt_cache(a)
        mx.eval(ids)
        base = mx.get_active_memory()
        lazy = a(ids, cache=cache)      # every layer's routing sync has run
        held[on] = mx.get_active_memory() - base
        mx.eval([c.state for c in cache])
        del lazy, cache
    n_gdn = sum(1 for l in a.layers if l.is_linear)
    t = TEXT
    conv_dim = (2 * t["linear_num_key_heads"] * t["linear_key_head_dim"]
                + t["linear_num_value_heads"] * t["linear_value_head_dim"])
    pinned = n_gdn * (ids.shape[1] + t["linear_conv_kernel_dim"] - 1) * conv_dim * 4
    freed = held[False] - held[True]
    assert freed >= 0.9 * pinned, (held, pinned)
    print(f"1. lazy pass of {ids.shape[1]} tokens holds {held[False] / 1e6:.1f} MB "
          f"without the sync, {held[True] / 1e6:.1f} MB with it: {freed / 1e6:.1f} MB "
          f"freed, {n_gdn} conv inputs = {pinned / 1e6:.1f} MB")


def settled_active():
    """MLX's active memory once the previous run's buffers are really gone:
    a kernel's temporaries are released by its command buffer's completion
    handler, which can still be pending when mx.eval returns. Read at once,
    the base sometimes still held them, and the run after it read that much
    lower: the old grouped path 50-100 MB from one run to the next on MLX
    0.32.3 (it then lost to the new path's steady 68 MB about one run in
    three), the same race rarer on 0.32.0; settled, 99.8 every time on both."""
    mx.synchronize()
    a = mx.get_active_memory()
    for _ in range(200):                # 1 s at most; ~10 ms in practice
        time.sleep(0.005)
        b = mx.get_active_memory()
        if b == a:
            return a
        a = b
    return a


def check_real_rows(tmp):
    """Fix 3 at the 2 GB model's per-row shapes, one packed layer: array_equal and peak."""
    H, I, E, k, T = 2048, 512, 32, 8, 512
    rng = np.random.default_rng(0)
    weights, base = {}, "model.layers.0.mlp.switch_mlp"
    for name, (o, i) in (("gate_proj", (I, H)), ("up_proj", (I, H)), ("down_proj", (H, I))):
        weights[f"{base}.{name}.weight"] = mx.array(
            rng.integers(0, 2 ** 32, (E, o, i // 8), dtype=np.uint32))
        sc = mx.array(rng.uniform(0.002, 0.004, (E, o, i // 64)).astype(np.float32))
        weights[f"{base}.{name}.scales"] = sc.astype(mx.bfloat16)
        weights[f"{base}.{name}.biases"] = (-7.5 * sc).astype(mx.bfloat16)
    out = tmp / "real_rows"
    save(out, weights, {})
    del weights
    proto = SwitchGLU(64, 64, 2)        # only its activation and quantization are read
    nn.quantize(proto, group_size=64, bits=4)
    st = stream_fast.SafetensorsMap(sorted(set(stream_fast.build_shard_map(out).values())))
    st.packed = stream_fast.load_packed(out)
    glu = stream_fast.StreamingSwitchGLU(st, base, proto, resident=8, slots=k)
    x = (mx.random.normal((1, T, H)) * 0.5).astype(mx.bfloat16)
    inds = mx.argpartition(-mx.random.normal((1, T, E)), kth=k - 1, axis=-1)[..., :k]
    mx.eval(x, inds)

    def run():
        stream_fast._WIDE_SCRATCH.clear()
        stream_fast._IDX_MEMO.clear()
        mx.clear_cache()
        start = settled_active()
        mx.reset_peak_memory()
        y = glu(x, inds)
        mx.eval(y)
        return y, mx.get_peak_memory() - start

    saved_group = stream_fast.WIDE_GROUP
    stream_fast.WIDE_GROUP = 16         # 32 experts in two groups, as 256 experts in two of 128
    try:
        for label in ("grouped", "split"):
            if label == "split":        # a few experts resident: the split path
                slots, jobs = glu._claim(np.array([1, 5, 9, 20, 30]))
                for j in jobs:
                    stream_fast._do_read(j)
            s0 = stream_fast.STATS["wide_split"]
            y_new, peak_new = run()
            with OldWidePath():
                y_old, peak_old = run()
            assert (stream_fast.STATS["wide_split"] > s0) == (label == "split")
            assert mx.array_equal(y_new, y_old).item(), f"{label}: differs from the old path"
            assert peak_new < peak_old, (label, peak_new, peak_old)
            print(f"3. real rows ({label}, T={T}, top-{k}, hidden {H}): array_equal; "
                  f"peak {peak_old / 1e6:.1f} -> {peak_new / 1e6:.1f} MB "
                  f"(one [T*k, H] = {T * k * H * 2 / 1e6:.1f} MB)")
    finally:
        stream_fast.WIDE_GROUP = saved_group
        st.packed["fd"].close()


def check_long_prompt(tmp):
    """8. long prompts under the prefill budget (StreamingLLM._prefill,
    --prefill-step 0): a prompt that fits only without the decode
    arena parks it, and the cache, the arena read back and the tokens
    after are array_equal to an unbudgeted run, cache bias 0.5 included; one
    that leaves less than MIN_PIECE positions even then drops kept
    conversations, the least recently used first and only while that holds,
    then cuts what is left into passes that fit; an explicit width is kept,
    and _take_conv counts the conversation it is about to prefill against
    keep_gb."""
    import moe_stream
    from moe_stream import StreamingLLM
    from mlx_lm.models.cache import KVCache
    out = tmp / "tiny_q35_long"
    mx.random.seed(0)
    m = fresh_model()
    mx.eval(m.parameters())
    save(out, dict(tree_flatten(m.parameters())),
         {**CFG, "quantization": {"group_size": 32, "bits": 4, "mode": "affine"}})
    del m
    rng = np.random.default_rng(11)
    p1, p2 = rng.integers(0, 256, 300).tolist(), rng.integers(0, 256, 900).tolist()
    n = len(p2) - 1

    def kept(kb):
        """A kept conversation holding `kb` KB of KV over 256 ids."""
        c = KVCache()
        c.update_and_fetch(mx.random.normal((1, 1, 256, kb // 2)),
                           mx.random.normal((1, 1, 256, kb // 2)))
        mx.eval(c.keys, c.values)
        return {"seen": [-1] * 256, "cache": [c], "marks": [], "draft": (None, ())}

    def arenas(e):
        return [np.array(b.view(mx.uint8)) for g in stream_fast.glus_of(e)
                for bufs, *_ in stream_fast._ARENAS.get(g._akey, ()) for b in bufs]

    def run(spare=None, step=0, convs=()):
        """p1 and decode (the arenas fill), then p2 with room for `spare`
        positions beside what MLX holds then (None: no budget)."""
        e = engine(out, seed=1)
        llm = StreamingLLM.__new__(StreamingLLM)
        llm.model, llm.prefill_step, llm._cache = e, step, make_prompt_cache(e)
        stream_fast.set_prefill_budget_gb(None)
        llm._prefill(p1[:-1])
        toks = decode(e, llm._cache, p1[-1], 4)
        llm._convs = tuple(kept(kb) for kb in convs)
        llm._pieces = ()
        before = arenas(e)
        rows, fixed = stream_fast.moe_row_elems(e), stream_fast.whole_layer_bytes(e)
        stream_fast.set_prefill_budget_gb(None, row_elems=rows, fixed=fixed)
        if spare is not None:
            stream_fast._eval_caches(llm._cache)
            need = (fixed + stream_fast._copy(llm._cache)
                    + spare * stream_fast.prefill_need()[1])
            stream_fast.set_prefill_budget_gb((settled_active() + need) / 1e9,
                                              row_elems=rows, fixed=fixed)
        s0 = stream_fast.STATS["wide_split"]
        llm._prefill(p2[:-1])
        llm.split = stream_fast.STATS["wide_split"] - s0
        after = arenas(e)
        toks += decode(e, llm._cache, p2[-1], 4)
        state = [a for c in llm._cache for a in c.state]
        mx.eval(state)
        stream_fast.set_prefill_budget_gb(None)
        return toks, state, before, after, llm

    saved = moe_stream.MIN_PIECE, stream_fast.WIDE_GROUP
    moe_stream.MIN_PIECE = 400          # the toy's prompt is 899 positions
    try:
        # Short by one position with the arena, room to spare without it. In
        # groups of 3 of the 16 experts a prefill's union is wider than a
        # group, as the 35B's 256 are than 128: it never reads the arena, which
        # stays parked through it. In one group it takes the split path,
        # which reads the arena back first.
        for group in (3, saved[1]):
            stream_fast.WIDE_GROUP = group
            ref = run()
            rate = stream_fast.PREFILL_ROWS * stream_fast.moe_row_elems(ref[4].model) * 2
            arena_b = sum(a.nbytes for a in ref[2])
            assert ref[4]._pieces == (n,) and not ref[4]._parked
            assert arena_b > 2 * rate, (arena_b, rate)
            park = run(spare=n - 1)
            assert park[4]._pieces == (n,) and park[4]._parked == arena_b, park[4]._parked
            assert (park[4].split > 0) == (group != 3) == (ref[4].split > 0)
            assert same(park[:2], ref[:2]), ("parking changed the cache or the tokens", group)
            assert all(np.array_equal(x, y) for x, y in zip(park[2], park[3])), \
                ("the arena came back with other bytes", group)
        stream_fast.WIDE_GROUP = 3
        per_conv = kept(128)["cache"][0].nbytes / rate
        print(f"8. a prompt short of room by the arena: arena parked "
              f"({arena_b / 1e3:.0f} KB) and read back, one pass of {n}: cache, "
              f"arena and tokens {park[0]} array_equal to the unbudgeted run "
              f"(cache bias 0.5), parked through the prefill and read back by a "
              f"split pass alike")
        # Under MIN_PIECE without the arena: the oldest kept conversation goes
        # while that holds, then the passes that fit.
        both = run(spare=150 - arena_b / rate, convs=(128, 128))
        llm = both[4]
        assert llm._dropped == 2 and not llm._convs, llm._dropped
        assert len(llm._pieces) > 1 and sum(llm._pieces) == n, llm._pieces
        # 1 MB of the oldest is room for the whole prompt: the newest stays.
        one = run(spare=350 - arena_b / rate, convs=(1024, 64))
        assert one[4]._dropped == 1 and one[4]._pieces == (n,), one[4]._pieces
        assert [cv["cache"][0].nbytes for cv in one[4]._convs] == [64 * 1024]
        fixed = run(spare=n // 3, step=128)
        assert fixed[4]._pieces == (128,) * 7 + (3,), fixed[4]._pieces
        print(f"   short even without it ({per_conv:.0f} positions a 128 KB kept "
              f"conversation): both dropped, then passes {list(llm._pieces)}; a "
              f"1 MB oldest and a newest: the oldest dropped, one pass; "
              f"--prefill-step 128 kept")
    finally:
        moe_stream.MIN_PIECE, stream_fast.WIDE_GROUP = saved

    # keep_gb counts the conversation this turn grows: 128 KB of KV for 256
    # ids, so 512 more ids are 256 KB: two kept ones and that are 512 KB.
    llm = StreamingLLM.__new__(StreamingLLM)
    llm.keep_convs = 4
    for kb, left in ((513, 2), (500, 1)):
        llm.keep_gb, llm._convs = kb * 1024 / 1e9, (kept(128), kept(128))
        assert llm._take_conv(list(range(512))) == 0
        assert len(llm._convs) == left, (kb, len(llm._convs))
    print("   keep_gb: two kept 128 KB conversations and this one's 256 KB to come "
          "fit 513 KB (both kept), not 500 KB (the older dropped)")


def check_turn_end():
    from support import has_tokenizer, toy_llm
    if not has_tokenizer():
        print("turn end: SKIPPED (no tokenizer: set MOE_TOY_TOKENIZER)")
        return
    llm = toy_llm("keep", max_tokens=4)
    llm.ask("Say the word apple.")
    assert mx.get_cache_memory() == 0, mx.get_cache_memory()
    print("7. turn end: MLX buffer cache empty after a turn")


def main():
    tmp = Path(tempfile.mkdtemp(prefix="prefill_mem_"))
    try:
        check_engine(tmp)
        check_real_rows(tmp)
        check_long_prompt(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    check_turn_end()
    # Default mode's groups are main's and never need the kernel line, so it
    # never pays the probe (0.1 s, a 46 MB MLX peak) at startup.
    assert stream_fast._RHS_LINE is None, "default mode probed the gather kernel line"
    print("test_prefill_mem: ok")


if __name__ == "__main__":
    main()
