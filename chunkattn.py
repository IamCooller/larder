"""Query-chunked prefill attention for full-attention layers whose head_dim has no fused MLX kernel, splitting only the query rows so the fallback's [heads, T, N] score matrix stays bounded, installed like foldattn.py with chunk widths calibrated per machine."""

import hashlib
import io
import json
import math
import os
import statistics
import sys
import time
from functools import lru_cache
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

import stream_fast
from rowqmm import LAYOUT_DIR
from engine_log import note

SAMPLE_T = 512          # rows in the slope probe, wide enough to land in the
                         # kernel a real prefill dispatches to (T<=8 always
                         # takes stock's vector kernel, whose cost this must
                         # not be read as)
SAMPLE_N = 512          # the slope reading is already stable at this many
                         # keys, so a larger probe only pays a bigger
                         # calibration-time score-matrix spike
SAMPLE_N_MULT = 8       # the second probe's key count, as a multiple of
                         # SAMPLE_N -- wide apart so a real O(T*N) score
                         # buffer's growth dwarfs the allocator jitter
                         # between two peak readings (measured 0-12 B/row/
                         # key of jitter on this machine's own D=128, a
                         # fused shape, at a 2x gap; 6-9 at 8x)
SLOPE_REPS = 5          # _kv_slope repeats and takes the max, not the mean:
                         # the risky misread here is UNDER-, not OVER-
                         # estimating a real transient (see _kv_slope)

ROW_CANDIDATES = (5, 8, 16, 32, 64)   # widths _min_safe_rows tries, widest
                                        # first, down to the first one that
                                        # misses, so the finest width with
                                        # every wider one exact wins; 8 is
                                        # foldattn's own MAX_ROWS, the
                                        # documented edge of stock's vector
                                        # kernel, 16 the wide gemv's (2-15
                                        # rows, MLX 0.32.2+)
if stream_fast.BACKEND == "cuda":
    # CUDA's unfused kernel needed 128 before a chunk matched stock's bits (probed on the RTX 5060 Ti: 5-64 all differed), and Metal keeps the narrower search it always had.
    ROW_CANDIDATES += (128, 256)
CALIB = ((256, 256), (900, 1500))      # (query rows, keys) the exactness
                                        # check runs at -- small enough for
                                        # any machine, two of them so a width
                                        # that only happens to match at one
                                        # N is not trusted; T < 1024 keeps
                                        # stock itself on the fallback the
                                        # chunks use (a case where it fuses is
                                        # skipped: chunking never runs there)


@lru_cache(maxsize=4096)
def _stock_fused(hq, hkv, d, dtype, t, n, arr=False):
    """True if stock's own call -- mx.fast.scaled_dot_product_attention,
    mask="causal" (`arr`: a boolean [t, n] array), t query rows over n keys
    -- takes an MLX fused kernel. MLX decides that while it builds the graph
    (fast.cpp, use_fallback), so it is read off a lazy graph of zeros of the
    same shapes, never evaluated: nothing is allocated, ~10-30 us a shape,
    cached."""
    q = mx.zeros((1, hq, t, d), dtype=dtype)
    kv = mx.zeros((1, hkv, n, d), dtype=dtype)
    mask = mx.zeros((1, 1, t, n), dtype=mx.bool_) if arr else "causal"
    out = mx.fast.scaled_dot_product_attention(q, kv, kv, scale=1.0, mask=mask)
    dot = io.StringIO()
    mx.export_to_dot(dot, out)
    return "ScaledDotProductAttention" in dot.getvalue()


def _kv_slope(hq, hkv, d, dtype):
    """Bytes stock's peak holds per extra (query row x key): two peaks at
    the same SAMPLE_T, keys doubled, isolate the per-key term from whatever
    fixed workspace the kernel always pays. Repeated and MAXed rather than
    averaged: allocator jitter between two peak readings can swing several
    bytes either way (measured on this machine's own D=128, a fused shape --
    0 to 12 B/row/key across repeats of the identical probe), and the risky
    direction here is UNDER-reading a real transient, not over-reading a
    fused one -- a false "unfused" costs one wasted _min_safe_rows search at
    install time and, since its own per_key_bytes then stays tiny, chunking
    that almost never actually fires; a false "fused" would leave a real
    peak unbounded."""
    def peak(n):
        k = mx.random.normal((1, hkv, n, d)).astype(dtype)
        v = mx.random.normal((1, hkv, n, d)).astype(dtype)
        q = mx.random.normal((1, hq, SAMPLE_T, d)).astype(dtype)
        mx.eval(k, v, q)
        mx.reset_peak_memory()
        mx.eval(mx.fast.scaled_dot_product_attention(
            q, k, v, scale=d ** -0.5, mask="causal"))
        return mx.get_peak_memory()
    best = 0.0
    for _ in range(SLOPE_REPS):
        lo, hi = peak(SAMPLE_N), peak(SAMPLE_N * SAMPLE_N_MULT)
        gap = SAMPLE_N * (SAMPLE_N_MULT - 1)
        best = max(best, (hi - lo) / (gap * SAMPLE_T))
    return best


def _exact_at(hq, hkv, d, dtype, t, n, width):
    """True if chunked() in `width`-row chunks (the folded tail a little
    wider) gives the same bits for all t query rows over n keys as one stock
    call through mask="causal", checked on every row because a kernel that
    sums in another order flips a bf16 rounding only here and there."""
    q = mx.random.normal((1, hq, t, d)).astype(dtype)
    k = mx.random.normal((1, hkv, n, d)).astype(dtype)
    v = mx.random.normal((1, hkv, n, d)).astype(dtype)
    mx.eval(q, k, v)
    scale = d ** -0.5
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal")
    return mx.array_equal(ref, chunked(q, k, v, scale, width, width)).item()


def _min_safe_rows(hq, hkv, d, dtype):
    """Smallest ROW_CANDIDATES width that, with every wider candidate,
    reproduces stock bit-for-bit at every CALIB case stock itself runs
    unfused; None if not even the widest does -- that shape is left
    entirely on stock (install() reports it), never chunked on a guess."""
    cases = [(t, n) for t, n in CALIB if not _stock_fused(hq, hkv, d, dtype, t, n)]
    best = None
    for w in reversed(ROW_CANDIDATES):
        if not cases or not all(_exact_at(hq, hkv, d, dtype, t, n, w) for t, n in cases):
            break
        best = w
    return best


if stream_fast.BACKEND == "cuda":
    # CUDA reads its attention slopes at 64 rows over 4096 and 32768 keys, which _kv_slope shares, while Metal keeps the 512 rows over 512 and 4096 keys it always had (the narrower rows over more keys read its slope up to 4.5x and cut its prefill attention into narrower chunks).
    SAMPLE_T = 64           # query rows of the attention-slope probe
    SAMPLE_N = 4096         # keys of its narrow case, doubled in the wide one
# The indexer's own peak is fitted over these (rows, keys), read at the two row
# counts a pass uses and over the keys a long prompt reaches: a single key count
# would read its small-row matmul workspace as rows, and the slope measured over
# 8k keys would not hold at 128k.
IDX_GRID = ((128, 32768), (128, 65536), (512, 32768), (512, 65536), (1024, 32768))
PROBE_V = 3             # part of the cache key: 1 and 2 misread the indexer's peak


def _peak_of(fn):
    """Bytes the device held at the peak of `fn`, past what was active when it
    began, since the inputs differ between the probe's cases and their own
    bytes would read as the transient's."""
    mx.synchronize()
    mx.clear_cache()
    a0 = mx.get_active_memory()
    mx.reset_peak_memory()
    mx.eval(fn())
    return float(mx.get_peak_memory() - a0)


def _slope_of(peaks, t, n):
    """(per_row_key, per_row, per_key) from peaks at (t, n), (t, 2n), (2t, n): the
    key term from the first pair, the row term from the second with the key
    term's share taken out, and whatever is left over the keys as a flat part."""
    p1, p2, p3 = peaks
    per_key_row = (p2 - p1) / (t * n)
    per_row = (p3 - p1) / t - per_key_row * n
    flat = (p1 - t * per_key_row * n - t * per_row) / n
    return max(0.0, per_key_row), max(0.0, per_row), max(0.0, flat)


def _sdpa_arrays(hq, hkv, d, dtype, t, n):
    """q, k, v of one layer's attention at these shapes, evaluated."""
    q = mx.random.normal((1, hq, t, d)).astype(dtype)
    k = mx.random.normal((1, hkv, n, d)).astype(dtype)
    v = mx.random.normal((1, hkv, n, d)).astype(dtype)
    mx.eval(q, k, v)
    return q, k, v


def _sdpa_peaks(hq, hkv, d, dtype, t, n):
    """Peaks of one array-masked SDPA at (t, n), (t, 2n), (2t, n), the mask the
    one a sparse prefill builds (a bool [1, 1, t, keys] array), since the
    `mask="causal"` string routes to MLX's fused kernel that holds no scores."""
    out = []
    for tt, nn in ((t, n), (t, 2 * n), (2 * t, n)):
        q, k, v = _sdpa_arrays(hq, hkv, d, dtype, tt, nn)
        mask = mx.zeros((1, 1, tt, nn), dtype=mx.bool_)
        mx.eval(mask)
        out.append(_peak_of(lambda: mx.fast.scaled_dot_product_attention(
            q, k, v, scale=d ** -0.5, mask=mask)))
    return out


def _indexer_peak(ix, dtype, t, n):
    """Bytes a full-attention layer's sparse mask holds at (t, n) past what was
    active when it began: QSAIndexer.__call__'s ops in their order (the pooled
    keys, the einsum into f32 scores over n/compress blocks, their relu and sum
    over the heads, the top-k blocks it keeps, repeated over the keys) and then
    Attention.__call__'s `(linds >= rinds) & sparse`, whose bool [t, keys]
    arrays are what the SDPA below it is handed."""
    cr, kd = ix.compress_ratio, ix.head_dim
    w = next((v for v in tree_flatten(ix.parameters())[1]
              if isinstance(v, mx.array) and v.ndim == 2 and v.shape[-1] == kd), None)
    heads = w.shape[0] if w is not None else 4
    budget = ix.token_budget // cr
    nb = n // cr
    raw = mx.random.normal((1, n, kd)).astype(dtype)
    q = mx.random.normal((1, t, heads, kd)).astype(dtype)
    mx.eval(raw, q)
    pooled = raw.reshape(1, nb, cr, kd).astype(mx.float32).mean(axis=2)
    q_pos = (n - t) + mx.arange(t)

    def run():
        pooled_ = ix.k_layernorm(pooled).astype(dtype)
        scores = mx.einsum("bshd,bnd->bsnh", q, pooled_).astype(mx.float32)
        scores = mx.maximum(scores, 0).sum(axis=-1) / kd ** 0.5
        vis = (mx.arange(nb)[None, None, :]
               < (mx.maximum(q_pos + 1, 0) // cr)[None, :, None])
        scores = mx.where(vis, scores, -mx.inf)
        top = mx.argpartition(-scores, min(budget, nb) - 1, axis=-1)
        keep = mx.put_along_axis(
            mx.zeros((1, t, nb + 1), dtype=mx.bool_), top[..., :min(budget, nb)],
            mx.array(True), axis=-1)[..., :nb]
        sparse = mx.repeat(keep, cr, axis=-1)
        own = (mx.arange(n)[None, None, :]
               >= (mx.maximum(q_pos + 1, 0) // cr * cr)[..., None])
        return (mx.arange(n - t, n)[:, None] >= mx.arange(n)) & (sparse | own)
    return _peak_of(run)


def _indexer_fit(ix, dtype):
    """(per_row_key, per_key, worst_error) of the indexer's own tensors: the
    least squares of `t x keys` and `keys` through the grid's peaks, which is
    what its arrays are, plus the matmul workspace that a small row count holds
    whatever the rows. A slope from two row counts and one key count puts the
    workspace in the row part, which then does not hold at 128k keys."""
    pts = [(_indexer_peak(ix, dtype, t, n), t * n, n) for t, n in IDX_GRID]
    sx = sum(x * x for _, x, _ in pts)
    sy = sum(z * z for _, _, z in pts)
    sxx = sum(x * z for _, x, z in pts)
    bx = sum(x * p for p, x, _ in pts)
    bz = sum(z * p for p, _, z in pts)
    det = sx * sy - sxx * sxx
    if det <= 0:
        return 0.0, 0.0, 0.0
    a = (bx * sy - sxx * bz) / det
    f = (sx * bz - sxx * bx) / det
    a, f = max(0.0, a), max(0.0, f)
    worst = max((abs(a * x + f * z - p) / max(p, 1.0) for p, x, z in pts), default=0.0)
    return a, f, worst


def prefill_attn_slope(model, dtype=None, sdpa_rows=0):
    """(sdpa_per_row_key, indexer_per_row_key, per_row, per_key, sdpa_rows)
    bytes one layer's prefill attention holds at this model's shapes: an
    array-masked SDPA's score matrix plus the QSA indexer's own scores,
    relu/sum and top-k buffers, each measured at two key counts and two row
    counts and kept for the next load. `sdpa_rows` is the width chunkattn
    cuts the SDPA to (layout's min exact rows), so its matrix is counted per
    min(chunk, sdpa_rows) rows while the indexer's, which is not cut, is
    counted per chunk row. (0, 0, 0, 0, 0) where stock's SDPA is fused and
    nothing materialises, and None when the probe cannot run (the caller then
    plans without the term, as before)."""
    try:
        _, shapes, arch = _attention_modules(model)
        if not shapes:
            return (0.0, 0.0, 0.0, 0.0, 0)
        dtype = arch if dtype is None else dtype
        q4 = sys.modules.get("mlx_lm.models.qwen4_exp")
        ix = next((m for m in model.modules()
                   if q4 is not None and isinstance(m, q4.QSAIndexer)), None)
        key = json.dumps([PROBE_V, mx.device_info()["device_name"], mx.__version__,
                          sorted(shapes), str(dtype), int(sdpa_rows),
                          None if ix is None else [ix.compress_ratio, ix.head_dim,
                                                   ix.token_budget]])
        path = LAYOUT_DIR / f"attn-{hashlib.sha1(key.encode()).hexdigest()[:16]}.json"
        try:
            return tuple(float(v) for v in json.loads(path.read_text())["slope"])
        except (OSError, ValueError, KeyError):
            pass
        hq, hkv, d = max(shapes)
        pk, pr, pf = _slope_of(_sdpa_peaks(hq, hkv, d, dtype, SAMPLE_T, SAMPLE_N),
                              SAMPLE_T, SAMPLE_N)
        # Half a head's element per row and key is already far above allocator
        # jitter and below any real score matrix, so the borderline exactly-at
        # hq x size reading a real bf16 matrix gives is not misread as fused.
        if pk >= hq * dtype.size / 2:
            ik, iff, worst = ((0.0, 0.0, 0.0) if ix is None else
                              _indexer_fit(ix, dtype))
            out = (pk, ik, pr, pf + iff, int(sdpa_rows))
            print(f"  prefill attention: indexer fit {ik:.1f} B/row/key + "
                  f"{iff / 1e3:.1f} KB/key over {len(IDX_GRID)} points, worst "
                  f"{100 * worst:.1f}%", flush=True)
        else:
            out = (0.0, 0.0, 0.0, 0.0, 0)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"slope": list(out), "key": key}))
        except OSError:
            pass
        return out
    except Exception as e:
        # A probe that cannot run must not stop a model load, but a plan
        # without the term is the plan that tripped the guard at 64k.
        print(f"  prefill attention: not measured ({type(e).__name__}: {e}); "
              f"a prefill's scores are unbounded", flush=True)
        return None


LOOPED = 4096           # MLX's softmax switches to its looped kernel above
                        # this row length (SOFTMAX_LOOPED_LIMIT, metal
                        # softmax.cpp); the two kernels sum the row in a
                        # different order.
KEY_ALIGN = 64          # a cut is rounded up to this many keys: MLX's GEMM
                        # sums a K that is not a multiple of its block first,
                        # so P @ V over an unaligned cut adds the last keys
                        # in another order than over all N (whose unaligned
                        # tail is masked zeros).


def _cut(nk, n, floor):
    """Keys a chunk whose last row sees `nk` of `n` takes (see chunked)."""
    if n >= floor:
        nk = max(nk, floor)
    return min(n, -(-nk // KEY_ALIGN) * KEY_ALIGN)


def _trim_floor(hq, hkv, d, dtype, min_rows):
    """LOOPED + 1 if a chunk's rows attend to only the keys they can see
    (_cut: k/v cut at the chunk's last visible key, rounded up to KEY_ALIGN,
    and never below LOOPED + 1 keys when the whole pass has more, so a row's
    softmax stays on the kernel the whole-N pass runs) with the same bits as
    the same rows against all N keys, else 0 and chunks keep every key,
    checked on the shape's own kernels at a few widths and N up to 16k because
    a retuned MLX would move the softmax switch or the alignment silently."""
    floor = LOOPED + 1
    scale = d ** -0.5
    for n, p in ((3000, 0), (3000, 1500), (4608, 0), (16384, 6000), (16390, 15000)):
        k = mx.random.normal((1, hkv, n, d)).astype(dtype)
        v = mx.random.normal((1, hkv, n, d)).astype(dtype)
        for w in (min_rows, 256):
            q = mx.random.normal((1, hq, w, d)).astype(dtype)
            mask = (p + mx.arange(w))[:, None] >= mx.arange(n)[None, :]
            nk = _cut(p + w, n, floor)
            full = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
            cut = mx.fast.scaled_dot_product_attention(
                q, k[:, :, :nk], v[:, :, :nk], scale=scale, mask=mask[:, :nk])
            if not mx.array_equal(full, cut).item():
                return 0
    return floor


def _bounds(T, rows, min_rows):
    """chunked()'s chunk edges over T rows: `rows` apart, a last chunk under min_rows folded into the one before."""
    rows = max(rows, min_rows)
    bounds = list(range(0, T, rows)) + [T]
    if len(bounds) > 2 and bounds[-1] - bounds[-2] < min_rows:
        bounds.pop(-2)
    return bounds


def chunked(q, k, v, scale, rows, min_rows, floor=0, mask=None, fused=False):
    """Same result as mx.fast.scaled_dot_product_attention(q, k, v,
    scale=scale, mask="causal") when q's T rows are the last T of k/v's N
    (mlx_lm's own bottom-right convention: row r sees keys 0..N-T+r) --
    split only the query rows into groups of `rows`, each against the SAME,
    unsliced k/v, so every key is still read once per chunk rather than
    once per row.

    `floor` (_trim_floor, nonzero): a chunk takes only the keys up to its
    last row's (_cut: rounded up to KEY_ALIGN, at least `floor` of them when
    N >= floor). The fallback
    kernel computes and then masks every score, so a whole-N chunk paid for
    the future half of the square too: 1.42 TFLOPS on the 35B's one-pass
    16k prefill against 2.24 for the same layers in 2048-wide passes.

    The last group is folded into the one before it whenever the plain
    remainder would land under min_rows: that is the one width
    _min_safe_rows never gets to check, and the caller (_wrap) already
    guarantees rows >= min_rows, so the merged group is at most one `rows`
    short of 2x rows -- still inside the range install() verified.

    `mask`: a boolean [..., T, N] array instead of the causal rule, every
    key it keeps visible to the causal rule too (Flash-Next's sparse
    attention past 2048 keys, qwen4_exp's `(linds >= rinds) & sparse`): each
    chunk takes its rows of it. The causal masks the chunks build are arrays
    as well, so a chunk computes what install() verified either way.

    Each chunk is forced (mx.eval) before the next is built: MLX only frees
    a chunk's own [heads, rows, N] score buffer once nothing lazy still
    references it, and without this every chunk stayed alive until the
    final mx.concatenate, measured (tests/test_chunkattn.py) peaking HIGHER
    than one whole-T stock call at every width from 32 rows up -- chunking
    bought nothing. This is the prefill path (under --mem-peak-gb also a
    pass whose room runs short), where the engine already syncs per layer,
    so an extra sync per chunk is not new latency shape.

    `fused` (MOE_PREFILL_SOFTMAX, causal only): a chunk over more than
    LOOPED keys runs _fused_core, the same fallback with its mask and
    softmax as one in-place pass.
    """
    T, N = q.shape[2], k.shape[2]
    offset = N - T
    bounds = _bounds(T, rows, min_rows)
    outs = []
    for c0, c1 in zip(bounds, bounds[1:]):
        if c0:
            # A shadow render's pass gives way to a request here too: the
            # one stretch of a layer with no other check (0.9 s at 12k).
            stream_fast._prefill_stop()
        nk = _cut(offset + c1, N, floor) if floor else N
        kc, vc = (k, v) if nk == N else (k[:, :, :nk], v[:, :, :nk])
        if fused and mask is None and nk > LOOPED:
            out = _fused_core(q[:, :, c0:c1], kc, vc, scale, offset + c0)
        else:
            if mask is None:
                row_pos = offset + c0 + mx.arange(c1 - c0)
                col_pos = mx.arange(nk)
                m = row_pos[:, None] >= col_pos[None, :]
            else:
                m = mask[..., c0:c1, :nk]
            out = mx.fast.scaled_dot_product_attention(
                q[:, :, c0:c1], kc, vc, scale=scale, mask=m)
        mx.eval(out)
        outs.append(out)
    return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=2)


# (Hq, Hkv, D) -> (per_key_bytes, min_rows, key_floor); a shape lands here
# only once both were measured -- unfused AND a bit-exact chunk width exists.
layout = {}


def _arr_mask(mask, T, N):
    """A boolean [1?, 1?, T, N] mask, the one array mask chunked() takes."""
    return (isinstance(mask, mx.array) and mask.dtype == mx.bool_ and mask.ndim >= 2
            and tuple(mask.shape[-2:]) == (T, N) and all(d == 1 for d in mask.shape[:-2]))


def _route(q, k, cache, mask, sinks):
    T = q.shape[2]
    if (T < 2 or sinks is not None or hasattr(cache, "bits")
            or not ((isinstance(mask, str) and mask == "causal")
                    or _arr_mask(mask, T, k.shape[2]))
            or q.dtype not in (mx.bfloat16, mx.float16) or k.dtype != q.dtype
            or q.shape[1] % k.shape[1]):
        return None
    return layout.get((q.shape[1], k.shape[1], q.shape[3]))


_warned_over_budget = False


def _wrap(stock):
    def scaled_dot_product_attention(queries, keys, values, cache, scale, mask,
                                     sinks=None):
        if stream_fast._KV_ONLY:
            # The last layer of a cache-only prefill pass (stream_fast.LAST_KV): nothing evaluates this attention.
            return stock(queries, keys, values, cache=cache, scale=scale, mask=mask, sinks=sinks)
        grouped = _gqa_route(queries, keys, cache, mask, sinks, stock)
        if grouped is not None:
            return _gqa_first(queries, keys, gqa_sdpa(queries, keys, values, scale, *grouped), grouped,
                              lambda: stock(queries, keys, values, cache=cache, scale=scale, mask=mask, sinks=sinks))
        found = _route(queries, keys, cache, mask, sinks)
        if found is None:
            return stock(queries, keys, values, cache=cache, scale=scale,
                         mask=mask, sinks=sinks)
        bud = stream_fast._MEASURING
        if bud is None:
            return _attend(queries, keys, values, cache, scale, mask, sinks, found,
                           stream_fast.prefill_cap_bytes(keys.shape[2], queries.dtype.size))
        # --mem-peak-gb sizes a prefill from its measured need, so the cap is
        # the T positions' own MoE rows rather than the N keys', and a pass
        # that keeps the pool (a warm turn) may use what its standing room
        # leaves free right now, never less than its rows.
        cap = stream_fast.prefill_cap_bytes(queries.shape[2], queries.dtype.size)
        if cap is not None and not bud.prefill:
            cap = max(cap, bud.room())
        # The measurement leaves this call's transient out, capped instead
        # (stream_fast.MemBudget.attention): taken up to the room, it would
        # be measured into the next pass's need as well. Inputs first, so
        # the work before this call is in the peak kept; the output forced,
        # so the transient is over when the peak is reset.
        mx.eval(queries, keys, values)
        bud.attention(True)
        out = _attend(queries, keys, values, cache, scale, mask, sinks, found, cap)
        mx.eval(out)
        bud.attention(False)
        return out

    def _attend(queries, keys, values, cache, scale, mask, sinks, found, cap):
        per_key, min_rows, *floor = found     # hand-set tables may leave floor out
        T, N = queries.shape[2], keys.shape[2]
        # No budget configured: leave this shape's own transient unbounded,
        # same as before this module existed.
        if cap is None:
            rows = T
        elif cap <= 0:
            # The budget is already spent by steady-state memory alone, and
            # chunking down to min_rows cannot recover it (it only multiplies
            # SDPA calls that re-read the whole K/V), so the transient is left
            # unbounded here too, logged once as a misconfiguration.
            global _warned_over_budget
            if not _warned_over_budget:
                print("  chunkattn: prefill budget already exceeded by "
                      "steady-state memory -- leaving attention unbounded "
                      "instead of chunking to a useless min_rows")
                _warned_over_budget = True
            rows = T
        else:
            # per_key is already summed over every Hq head by _kv_slope, so
            # dividing by queries.shape[1] here too would double-count heads
            # and undersize rows by that factor.
            rows = max(min_rows, int(cap // (N * per_key)) or 1)
        arr = not isinstance(mask, str)
        # MOE_PREFILL_SOFTMAX: a causal call it serves runs fused at the width its install timed, one chunk or many.
        width = None if arr else _sm_width(queries, keys)
        if width:
            rows = min(rows, width)
        # A shape that measured unfused at calibration may still fuse at
        # this T, and stock's own kernel is then bounded already and the only
        # kernel with its bits.
        if (rows >= T and width is None) or _stock_fused(queries.shape[1], keys.shape[1],
                                                         queries.shape[3], queries.dtype, T, N, arr):
            return stock(queries, keys, values, cache=cache, scale=scale,
                         mask=mask, sinks=sinks)
        if arr:
            return chunked(queries, keys, values, scale, rows, min_rows, *floor, mask=mask)
        if width is not None:
            return _sm_first(queries, keys, values, scale, min(rows, T), min_rows, floor)
        return chunked(queries, keys, values, scale, rows, min_rows, *floor)
    scaled_dot_product_attention.stock = stock
    # A second, chunkattn-specific marker: foldattn.py sets .stock too (the
    # shared convention both patchers use so neither re-wraps the other's
    # output), but install() needs to tell "already wrapped by ME" (safe to
    # fall through to its ordinary cache-or-calibrate path, cheap once
    # cached) apart from "already wrapped by SOMEONE ELSE" (this shape is
    # fully handled, calibrating here would only be thrown away).
    scaled_dot_product_attention.chunkattn = True
    return scaled_dot_product_attention


# MOE_SDPA_GQA (Metal, on; 0 turns it off): a decode or MTP-verify call MLX would run with its two-pass vector SDPA runs a replay of
# sdpa_vector_2pass_1 in which one simdgroup serves HPT q heads of a kv group per K/V load, with stock's bits.
GQA_ON = stream_fast.BACKEND == "metal" and os.environ.get("MOE_SDPA_GQA", "1") == "1"
GQA_HPT = (2, 4, 8)         # q heads a simdgroup the install times against stock
GQA_T = (1, 2, 3, 4)        # query rows in the envelope
GQA_DIMS = (96, 256)        # MLX 0.32.2's vector SDPA head dims with no sdpa_vector_2pass_1_gqa kernel of its own
GQA_N0 = 4096               # the timing grid starts here and doubles
GQA_NMAX = 65536            # and ends here, or lower where the K/V buffer would pass GQA_BYTES
GQA_BYTES = 256e6           # the install's K/V buffer, shared by every timed call
GQA_CHAIN = 4               # calls a timed chain, each on its own stretch of the buffer where it is long enough
GQA_REPS = 3                # timed chains an arm and cell, the median kept

# (Hq, Hkv, D, dtype) -> {T: [(keys, q heads a simdgroup), ...]}, timed at install; 1 is stock.
gqa = {}

# sdpa_vector_2pass_1 of MLX 0.32.2 with HPT q heads of a kv group a simdgroup: each head's float ops in stock's order.
_GQA_PASS1 = """
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;
  typedef float U;

  const uint simd_lid = thread_index_in_simdgroup;
  const int kv_head_idx = threadgroup_position_in_grid.x;
  const int block_idx = threadgroup_position_in_grid.z;
  const int q_seq_idx = thread_position_in_threadgroup.z;
  const int h_base = GQA * kv_head_idx + HPT * int(thread_position_in_threadgroup.y);
  const int N = prm[0];
  const int64_t kss = kc_strides[2];
  const int64_t vss = vc_strides[2];

  const device T* keys = kc + kv_head_idx * kc_strides[1] + block_idx * kss + simd_lid * qk_per_thread;
  const device T* values = vc + kv_head_idx * vc_strides[1] + block_idx * vss + simd_lid * v_per_thread;

  U q[HPT][qk_per_thread];
  U o[HPT][v_per_thread];
  U max_score[HPT];
  U sum_exp_score[HPT];
#pragma clang loop unroll(full)
  for (int j = 0; j < HPT; j++) {
    const device T* queries = qr + (h_base + j) * qr_strides[1] + q_seq_idx * qr_strides[2] + simd_lid * qk_per_thread;
    for (int e = 0; e < qk_per_thread; e++) {
      q[j][e] = static_cast<U>(scl[0]) * queries[e];
    }
    for (int e = 0; e < v_per_thread; e++) {
      o[j][e] = 0;
    }
    max_score[j] = Limits<U>::finite_min;
    sum_exp_score[j] = 0;
  }
  const int last = N - QL + q_seq_idx;

  for (int i = block_idx; i < N; i += P) {
    if (!CAUSAL || i <= last) {
      T kr[qk_per_thread];
      T vr[v_per_thread];
      for (int e = 0; e < qk_per_thread; e++) {
        kr[e] = keys[e];
      }
      for (int e = 0; e < v_per_thread; e++) {
        vr[e] = values[e];
      }
#pragma clang loop unroll(full)
      for (int j = 0; j < HPT; j++) {
        U score = 0;
        for (int e = 0; e < qk_per_thread; e++) {
          score += q[j][e] * kr[e];
        }
        score = simd_sum(score);
        U new_max = max(max_score[j], score);
        U factor = fast::exp(max_score[j] - new_max);
        U exp_score = fast::exp(score - new_max);
        max_score[j] = new_max;
        sum_exp_score[j] = sum_exp_score[j] * factor + exp_score;
        for (int e = 0; e < v_per_thread; e++) {
          o[j][e] = o[j][e] * factor + exp_score * vr[e];
        }
      }
    }
    keys += P * kss;
    values += P * vss;
  }

#pragma clang loop unroll(full)
  for (int j = 0; j < HPT; j++) {
    const int row = ((h_base + j) * QL + q_seq_idx) * P + block_idx;
    if (simd_lid == 0) {
      sums[row] = sum_exp_score[j];
      maxs[row] = max_score[j];
    }
    for (int e = 0; e < v_per_thread; e++) {
      part[row * V + simd_lid * v_per_thread + e] = static_cast<T>(o[j][e]);
    }
  }
"""
_gqa_kernel = []


def gqa_sdpa(q, k, v, scale, hpt, parts):
    """mx.fast.scaled_dot_product_attention(q, k, v, scale, mask=None or "causal") of batch 1 and T <= 4 rows, bit for bit where MLX runs its two-pass kernel over `parts` partitions."""
    if not _gqa_kernel:
        _gqa_kernel.append(mx.fast.metal_kernel(
            name="sdpa_gqa_1", input_names=["qr", "kc", "vc", "prm", "scl"],
            output_names=["part", "sums", "maxs"], source=_GQA_PASS1, ensure_row_contiguous=False))
    H, T, D = q.shape[1], q.shape[2], q.shape[3]
    Hk, V = k.shape[1], v.shape[3]
    G = H // Hk
    part, sums, maxs = _gqa_kernel[0](
        inputs=[q, k, v, mx.array([k.shape[2]], mx.int32), mx.array([scale], mx.float32)],
        template=[("T", q.dtype), ("D", D), ("V", V), ("P", parts), ("GQA", G), ("HPT", hpt),
                  ("QL", T), ("CAUSAL", T > 1)],
        grid=(32 * Hk, G // hpt, parts * T), threadgroup=(32, G // hpt, T),
        output_shapes=[(H * T, parts, V), (H * T, parts), (H * T, parts)],
        output_dtypes=[q.dtype, mx.float32, mx.float32])
    return stream_fast._qsa_kernel("pass2")(
        inputs=[part, sums, maxs], template=[("T", q.dtype), ("D", V), ("P", parts)],
        grid=(1024 * H * T, 1, 1), threadgroup=(1024, 1, 1),
        output_shapes=[(1, H, T, V)], output_dtypes=[q.dtype])[0]


# (Hq, Hkv, D, dtype, T, q heads a simdgroup, partitions) whose first engine call gave stock's bits.
_gqa_ok = set()


def _gqa_first(q, k, out, grouped, ref):
    """`out`, except that the first call of each grouping and partition count is also run by stock (`ref`), and a difference returns stock's result and turns the shape off."""
    key = (q.shape[1], k.shape[1], q.shape[3], str(q.dtype), q.shape[2], *grouped)
    if key in _gqa_ok:
        return out
    r = ref()
    if mx.array_equal(out, r).item():
        _gqa_ok.add(key)
        return out
    gqa.pop(key[:4], None)
    print(f"  decode attention: {key[0]}/{key[1]}x{key[2]} T={key[4]} over {key[6]} partitions differs from "
          f"stock at {k.shape[2]} keys; stock from here on", flush=True)
    return r


def _gqa_route(q, k, cache, mask, sinks, stock):
    """(q heads a simdgroup, partitions) for a call inside MOE_SDPA_GQA's envelope that the install timed faster than stock, else None."""
    if not gqa:
        return None
    B, H, T, D = q.shape
    if (T > 4 or B != 1 or sinks is not None or hasattr(cache, "bits")
            or not (mask is None and T == 1 or isinstance(mask, str) and mask == "causal")
            or (T > 1 and hasattr(stock, "foldattn"))):
        return None
    rows = gqa.get((H, k.shape[1], D, str(q.dtype)))
    if rows is None or k.dtype != q.dtype or k.shape[0] != 1 or not rows.get(T):
        return None
    N = k.shape[2]
    parts = stream_fast._sdpa_parts(H, k.shape[1], N)
    if parts is None:
        return None
    hpt = min(rows[T], key=lambda r: abs(math.log2(N / r[0])))[1]
    return None if hpt == 1 else (hpt, parts)


def _gqa_time(fns):
    """Median ms of each fn's chain (lazy outputs, synced by mx.eval), the fns interleaved rep by rep."""
    for f in fns:
        mx.eval(f())
    ts = [[] for _ in fns]
    for _ in range(GQA_REPS):
        for j, f in enumerate(fns):
            t0 = time.perf_counter()
            mx.eval(f())
            ts[j].append((time.perf_counter() - t0) * 1e3)
    return [statistics.median(t) for t in ts]


def _gqa_nmax(hkv, d, dtype, cap):
    """The widest key count of the timing grid whose K/V buffer fits `cap` bytes, or None when not even GQA_N0 does."""
    nmax = GQA_NMAX
    while nmax > GQA_N0 and 2 * hkv * (nmax + 256) * d * dtype.size > cap:
        nmax //= 2
    return nmax if 2 * hkv * (nmax + 256) * d * dtype.size <= cap else None


def _gqa_calibrate(hq, hkv, d, dtype, nmax=None):
    """{T: [(keys, q heads a simdgroup)]} for one shape, {} unless every grouping gives stock's bits at odd key counts on strided slices: each timed against stock at doubling key counts up to `nmax` over distinct stretches of one K/V buffer."""
    g = hq // hkv
    hpts = [h for h in GQA_HPT if g % h == 0]
    nmax = nmax or _gqa_nmax(hkv, d, dtype, GQA_BYTES)
    # Own keys, so the engine's global random state is left as it was.
    ks = mx.random.split(mx.random.key(hq * 1000 + d), 3)
    kb = mx.random.normal((1, hkv, nmax + 256, d), dtype, key=ks[0])
    vb = mx.random.normal((1, hkv, nmax + 256, d), dtype, key=ks[1])
    qs = {t: mx.random.normal((1, hq, t, d), dtype, key=ks[2]) for t in GQA_T}
    for a in (kb, vb, qs):         # one at a time, so the generator's float32 transients never coexist
        mx.eval(a)
    scale = d ** -0.5

    def stock(q, k, v):
        return mx.fast.scaled_dot_product_attention(
            q, k, v, scale=scale, mask=None if q.shape[2] == 1 else "causal")

    for n in (GQA_N0 + 77, nmax - 3):
        parts = stream_fast._sdpa_parts(hq, hkv, n)
        if parts is None:
            continue
        k, v = kb[:, :, :n], vb[:, :, :n]
        for t in GQA_T:
            ref = stock(qs[t], k, v)
            if not all(mx.array_equal(ref, gqa_sdpa(qs[t], k, v, scale, h, parts)).item() for h in hpts):
                return {}
    rows = {}
    n = GQA_N0
    while n <= nmax:
        parts = stream_fast._sdpa_parts(hq, hkv, n)
        if parts is not None:
            spans = [(kb[:, :, o:o + n], vb[:, :, o:o + n]) for o in range(0, nmax + 257 - n, n)]
            chain = [spans[i % len(spans)] for i in range(GQA_CHAIN)]
            for t in GQA_T:
                q = qs[t]
                fns = [lambda: [stock(q, k, v) for k, v in chain]]
                fns += [lambda h=h: [gqa_sdpa(q, k, v, scale, h, parts) for k, v in chain] for h in hpts]
                ms = _gqa_time(fns)
                rows.setdefault(t, []).append((n, ([1] + hpts)[ms.index(min(ms))]))
        n *= 2
    return rows


def _gqa_install(shapes, dtype):
    """MOE_SDPA_GQA's table for the model's shapes in its envelope, reused from disk or measured and kept, and the note saying what was chosen."""
    global gqa
    gqa = {}
    if stream_fast.BACKEND != "metal":
        return None
    if not GQA_ON:
        return "stock (MOE_SDPA_GQA=0)"
    hsum = stream_fast._sdpa_hdr()
    if hsum != stream_fast._SDPA_HDR or mx.__version__ not in stream_fast._SDPA_MLX:
        return f"stock (sdpa_vector.h {hsum}, MLX {mx.__version__}: not the ones replayed)"
    shapes = sorted((hq, hkv, d) for hq, hkv, d in shapes if d in GQA_DIMS and hq % hkv == 0 and hq > hkv)
    if dtype not in (mx.bfloat16, mx.float16) or not shapes:
        return f"stock (no grouped {dtype} shape without MLX's own grouped kernel)"
    t0 = time.perf_counter()
    src = hashlib.sha1((_GQA_PASS1 + stream_fast._QSA_PASS2).encode()).hexdigest()[:12]
    said = []
    for hq, hkv, d in shapes:
        key = json.dumps([mx.device_info()["device_name"], mx.__version__, src, hq, hkv, d, str(dtype),
                          GQA_HPT, GQA_N0, GQA_NMAX, GQA_BYTES, GQA_CHAIN, GQA_REPS])
        path = LAYOUT_DIR / f"sdpagqa-{hashlib.sha1(key.encode()).hexdigest()[:16]}.json"
        try:
            rows = {int(t): [tuple(r) for r in v] for t, v in json.loads(path.read_text())["rows"].items()}
        except (OSError, ValueError, KeyError):
            # Under a memory budget the buffer gets a quarter of the room free now: half, as a prefill chunk takes, and the generator's peak is twice the buffer.
            room = stream_fast.prefill_room()
            nmax = _gqa_nmax(hkv, d, dtype, GQA_BYTES if room is None else min(GQA_BYTES, room / 4))
            if nmax is None:
                said.append(f"{hq}/{hkv}x{d} stock (its smallest timing buffer is "
                            f"{2 * hkv * (GQA_N0 + 256) * d * dtype.size / 1e6:.0f} MB, a quarter of the budget's free "
                            f"room is {max(room, 0) / 4e6:.0f} MB now; a start with room calibrates)")
                continue
            rows = _gqa_calibrate(hq, hkv, d, dtype, nmax)
            mx.clear_cache()
            try:
                # A grid the budget cut short is used by this process but not kept.
                if nmax == _gqa_nmax(hkv, d, dtype, GQA_BYTES):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
                    tmp.write_text(json.dumps({"rows": rows, "key": key}))
                    tmp.replace(path)
            except OSError as e:
                print(f"  decode attention: could not save its calibration to {path}: {e}", flush=True)
        if rows:
            gqa[(hq, hkv, d, str(dtype))] = rows
        said.append(f"{hq}/{hkv}x{d} " + ("; ".join(
            f"T={t} " + " ".join(f"{n >> 10}k:{'stock' if h == 1 else f'h{h}'}" for n, h in r)
            for t, r in sorted(rows.items())) or "stock (no grouping gave stock's bits)"))
    return f"grouped heads {', '.join(said)} ({time.perf_counter() - t0:.1f} s)"


# MOE_PREFILL_SOFTMAX (Metal, on; 0 turns it off): a causal prefill chunk over more than LOOPED keys runs MLX's fallback with
# its mask `where` and precise softmax as one in-place pass in MLX 0.32.2's own order, at the row width the install timed fastest.
SM_ON = stream_fast.BACKEND == "metal" and os.environ.get("MOE_PREFILL_SOFTMAX", "1") == "1"
SM_MLX = ("0.32.2",)        # MLX versions whose fallback (fast.cpp) and softmax dispatch (softmax.cpp) were read
SM_HDR = "596ecf82e84e"     # sha1 of the kernels/softmax.h whose softmax_looped the kernel replays
SM_LSIZE = 1024             # threads a row: MLX's looped softmax runs 1024 on Metal and its 32-slot combine assumes it
SM_ROWS = (64, 128, 256, 512)               # widths the install times under the engine's own
SM_T = 1024                 # rows a timed pass takes: the tail of a pass, every chunk over about N keys
SM_N = (8192, 16384, 32768, 65536)          # keys the timing grid runs at
SM_CHECK = ((300, 4160, 64), (200, 8191, 48), (130, 16390, 100))   # (rows, keys, width) of the install's bit check

# softmax_looped<T, float, 4> of MLX 0.32.2 (kernels/softmax.h) over s's rows in place, each element first replaced as
# where(mask, s, finfo(T).min) replaces it, row r of a chunk seeing keys <= prm[2] + r.
_SM_SRC = """
  constexpr int N_READS = 4;
  constexpr int SIMD_SIZE = 32;
  typedef float AccT;
  const uint gid = threadgroup_position_in_grid.x;
  const uint lid = thread_position_in_threadgroup.x;
  const uint lsize = threads_per_threadgroup.x;
  const uint simd_lane_id = thread_index_in_simdgroup;
  const uint simd_group_id = simdgroup_index_in_threadgroup;
  const int axis_size = prm[0];
  const int lim = prm[2] + int(gid % uint(prm[1]));
  const T fill = Limits<T>::finite_min;

  const device T* in = s + gid * size_t(axis_size);
  device T* out = (device T*)s + gid * size_t(axis_size);

  threadgroup AccT local_max[SIMD_SIZE];
  threadgroup AccT local_normalizer[SIMD_SIZE];
  const int rounds = static_cast<int>(ceildiv(axis_size, N_READS * lsize));

  AccT prevmax;
  AccT maxval = Limits<AccT>::finite_min;
  AccT normalizer = 0;
  for (int r = 0; r < rounds; r++) {
    int offset = r * lsize * N_READS + lid * N_READS;
    AccT vals[N_READS];
    for (int i = 0; i < N_READS; i++) {
      T x = offset + i <= lim ? in[offset + i] : fill;
      vals[i] = (offset + i < axis_size) ? AccT(x) : Limits<AccT>::min;
    }
    prevmax = maxval;
    for (int i = 0; i < N_READS; i++) {
      maxval = (maxval < vals[i]) ? vals[i] : maxval;
    }
    normalizer *= metal::fast::exp(prevmax - maxval);
    for (int i = 0; i < N_READS; i++) {
      normalizer += metal::fast::exp(vals[i] - maxval);
    }
  }
  prevmax = maxval;
  maxval = simd_max(maxval);
  normalizer *= metal::fast::exp(prevmax - maxval);
  normalizer = simd_sum(normalizer);
  prevmax = maxval;
  if (simd_lane_id == 0) {
    local_max[simd_group_id] = maxval;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  maxval = simd_max(local_max[simd_lane_id]);
  normalizer *= metal::fast::exp(prevmax - maxval);
  if (simd_lane_id == 0) {
    local_normalizer[simd_group_id] = normalizer;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  normalizer = simd_sum(local_normalizer[simd_lane_id]);
  normalizer = 1 / normalizer;

  for (int r = 0; r < rounds; r++) {
    int offset = r * lsize * N_READS + lid * N_READS;
    for (int i = 0; i < N_READS; i++) {
      if (offset + i < axis_size) {
        T x = offset + i <= lim ? in[offset + i] : fill;
        out[offset + i] = T(metal::fast::exp(x - maxval) * normalizer);
      }
    }
  }
  if (gid == 0 && lid == 0) {
    p[0] = T(0);
  }
"""
_sm_kernel = []

# (Hq, Hkv, D, dtype) -> [(keys, width)] timed at install, width 0 where the engine's own was fastest.
sm = {}
# Shapes whose first routed call gave the stock fallback's bits.
_sm_ok = set()


def _softmax_hdr():
    """The first 12 hex digits of the sha1 of the installed MLX's softmax.h, or None if it cannot be read."""
    try:
        p = os.path.join(os.path.dirname(mx.__file__), "include", "mlx", "backend", "metal", "kernels", "softmax.h")
        with open(p, "rb") as f:
            return hashlib.sha1(f.read()).hexdigest()[:12]
    except OSError:
        return None


def _fused_core(q, k, v, scale, off0):
    """mx.fast.scaled_dot_product_attention's fallback (MLX 0.32.2 fast.cpp) for one causal chunk whose row r sees keys
    <= off0 + r, with the mask and the precise softmax as one pass written over the score buffer."""
    if not _sm_kernel:
        _sm_kernel.append(mx.fast.metal_kernel(
            name="mask_softmax", input_names=["s", "prm"], output_names=["p"], source=_SM_SRC,
            ensure_row_contiguous=False))
    B, H, R, D = q.shape
    hk = k.shape[1]
    q = mx.array(scale, q.dtype) * q
    if H != hk:
        q = q.reshape(B, hk, H // hk, R, D)
        k, v = mx.expand_dims(k, 2), mx.expand_dims(v, 2)
    s = q @ k.swapaxes(-1, -2)
    nk = s.shape[-1]
    # MLX does not see the kernel write s, so the kernel is evaluated before the value GEMM reads s.
    mx.eval(_sm_kernel[0](
        inputs=[s, mx.array([nk, R, off0], mx.int32)], template=[("T", s.dtype)],
        grid=(s.size // nk * SM_LSIZE, 1, 1), threadgroup=(SM_LSIZE, 1, 1),
        output_shapes=[(1,)], output_dtypes=[s.dtype]))
    return (s @ v).reshape(B, H, R, v.shape[-1])


def _sm_width(q, k):
    """The row width the install timed fastest at this call's keys (0: the engine's own), or None where the fused pass does not run."""
    if not sm or q.shape[0] != 1 or k.shape[2] <= LOOPED:
        return None
    rows = sm.get((q.shape[1], k.shape[1], q.shape[3], str(q.dtype)))
    if not rows:
        return None
    return min(rows, key=lambda r: abs(math.log2(k.shape[2] / r[0])))[1]


def _sm_first(q, k, v, scale, rows, min_rows, floor):
    """chunked()'s fused pass, except that the first call of each shape also runs its last chunk on the stock fallback,
    and a difference returns the stock pass and turns the shape off."""
    out = chunked(q, k, v, scale, rows, min_rows, *floor, fused=True)
    key = (q.shape[1], k.shape[1], q.shape[3], str(q.dtype))
    if key in _sm_ok:
        return out
    T = q.shape[2]
    c0 = _bounds(T, rows, min_rows)[-2]
    ref = chunked(q[:, :, c0:], k, v, scale, T - c0, min_rows, *floor)
    if mx.array_equal(out[:, :, c0:], ref).item():
        _sm_ok.add(key)
        return out
    sm.pop(key, None)
    print(f"  prefill softmax: {key[0]}/{key[1]}x{key[2]} differs from the stock fallback at {k.shape[2]} keys; "
          f"stock from here on", flush=True)
    return chunked(q, k, v, scale, rows, min_rows, *floor)


def _sm_exact(hq, hkv, d, dtype, min_rows, floor):
    """True if the fused pass gives the stock fallback's chunks bit for bit at SM_CHECK's shapes: q a transposed view as
    mlx_lm hands it, K/V strided slices of a longer cache buffer, odd key counts and a folded last chunk."""
    ks = mx.random.split(mx.random.key(hq * 1000 + d + 7), 3)
    scale = d ** -0.5
    for t, n, w in SM_CHECK:
        q = mx.random.normal((1, t, hq, d), dtype, key=ks[0]).transpose(0, 2, 1, 3)
        kb = mx.random.normal((1, hkv, n + 256, d), dtype, key=ks[1])
        vb = mx.random.normal((1, hkv, n + 256, d), dtype, key=ks[2])
        mx.eval(q, kb, vb)
        k, v = kb[:, :, :n], vb[:, :, :n]
        w = max(w, min_rows)
        ref = chunked(q, k, v, scale, w, min_rows, *floor)
        if not mx.array_equal(chunked(q, k, v, scale, w, min_rows, *floor, fused=True), ref).item():
            return False
    return True


def _sm_calibrate(hq, hkv, d, dtype, per_key, min_rows, floor):
    """([(keys, width)], whether every SM_ROWS width ran) for one shape: at each SM_N key count, a pass's last SM_T rows
    through the fused chunks at each width under the engine's own (the rows _attend cuts with the budget's room now) and
    at its own, the fastest kept and 0 where its own won."""
    ks = mx.random.split(mx.random.key(hq * 1000 + d), 3)
    nmax = max(SM_N)
    kb = mx.random.normal((1, hkv, nmax, d), dtype, key=ks[0])
    vb = mx.random.normal((1, hkv, nmax, d), dtype, key=ks[1])
    q = mx.random.normal((1, hq, SM_T, d), dtype, key=ks[2])
    for a in (kb, vb, q):          # one at a time, so the generator's transients never coexist
        mx.eval(a)
    scale = d ** -0.5
    out, full = [], True
    for n in SM_N:
        cap = stream_fast.prefill_cap_bytes(n, dtype.size)
        own = SM_T if cap is None else min(SM_T, max(min_rows, int(cap // (n * per_key))))
        cands = [w for w in SM_ROWS if min_rows <= w < own] + [own]
        full = full and own > max(SM_ROWS)
        k, v = kb[:, :, :n], vb[:, :, :n]
        ms = _gqa_time([lambda w=w: chunked(q, k, v, scale, w, min_rows, *floor, fused=True) for w in cands])
        best = cands[ms.index(min(ms))]
        out.append((n, 0 if best == own else best))
    return out, full


def _sm_install(table, dtype):
    """MOE_PREFILL_SOFTMAX's widths for the shapes chunkattn routes, reused from disk or checked, timed and kept, and the
    note saying what was chosen."""
    global sm
    sm = {}
    _sm_ok.clear()
    if stream_fast.BACKEND != "metal":
        return None
    if not SM_ON:
        return "stock (MOE_PREFILL_SOFTMAX=0)"
    hsum = _softmax_hdr()
    if hsum != SM_HDR or mx.__version__ not in SM_MLX:
        return f"stock (softmax.h {hsum}, MLX {mx.__version__}: not the ones replayed)"
    if dtype not in (mx.bfloat16, mx.float16) or not table:
        return f"stock (no {dtype} shape chunkattn routes)"
    t0 = time.perf_counter()
    src = hashlib.sha1(_SM_SRC.encode()).hexdigest()[:12]
    said = []
    for (hq, hkv, d), (per_key, min_rows, *floor) in sorted(table.items()):
        key = json.dumps([mx.device_info()["device_name"], mx.__version__, src, hq, hkv, d, str(dtype), per_key,
                          min_rows, floor, SM_ROWS, SM_T, SM_N, SM_CHECK, GQA_REPS])
        path = LAYOUT_DIR / f"smwidth-{hashlib.sha1(key.encode()).hexdigest()[:16]}.json"
        try:
            rows = [tuple(r) for r in json.loads(path.read_text())["rows"]]
        except (OSError, ValueError, KeyError):
            rows, full = [], True
            if _sm_exact(hq, hkv, d, dtype, min_rows, floor):
                rows, full = _sm_calibrate(hq, hkv, d, dtype, per_key, min_rows, floor)
            mx.clear_cache()
            try:
                # A grid the budget cut short is used by this process but not kept.
                if full:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
                    tmp.write_text(json.dumps({"rows": rows, "key": key}))
                    tmp.replace(path)
            except OSError as e:
                print(f"  prefill softmax: could not save its calibration to {path}: {e}", flush=True)
        if rows:
            sm[(hq, hkv, d, str(dtype))] = rows
        said.append(f"{hq}/{hkv}x{d} " + (" ".join(f"{n >> 10}k:{w or 'own'}" for n, w in rows)
                                         or "stock (the fused pass differs from the stock fallback)"))
    return f"fused mask + softmax, rows {', '.join(said)} ({time.perf_counter() - t0:.1f} s)"


def _heads(m):
    """mlx_lm spells an attention module's head counts differently across
    model files: qwen3.py aliases them to n_heads/n_kv_heads, while
    qwen3_next.py -- which Qwen3.6's full-attention layers use -- keeps the
    config names num_attention_heads/num_key_value_heads. Both are tried."""
    if hasattr(m, "n_heads") and hasattr(m, "n_kv_heads"):
        return m.n_heads, m.n_kv_heads
    if hasattr(m, "num_attention_heads") and hasattr(m, "num_key_value_heads"):
        return m.num_attention_heads, m.num_key_value_heads
    return None


def _attention_modules(model):
    """The python modules whose attention calls the shared sdpa helper, the
    (Hq, Hkv, D) of every attention layer found, and the activation dtype
    they run in."""
    mods, shapes, dtype = set(), set(), mx.bfloat16
    for m in model.modules():
        mod = sys.modules.get(type(m).__module__)
        if mod is not None and hasattr(mod, "scaled_dot_product_attention"):
            mods.add(mod)
        heads = _heads(m)
        if heads is not None and hasattr(m, "scale"):
            hq, hkv = heads
            d = round(m.scale ** -2)
            shapes.add((hq, hkv, d))
            dtype = next((v.dtype for _, v in tree_flatten(m.parameters())
                          if v.dtype in (mx.float16, mx.bfloat16)), dtype)
    return mods, shapes, dtype


def calibrate(shapes, dtype=mx.bfloat16, verbose=False):
    """{(Hq, Hkv, D): (per_key_bytes, min_rows, key_floor)} for every shape
    whose stock pass measured unfused (_kv_slope comparable to a real [Hq, T,
    N] score matrix) AND has a bit-exact chunk width (_min_safe_rows found
    one in ROW_CANDIDATES); everything else stays on stock, untouched.
    key_floor is _trim_floor's (0: chunks keep every key)."""
    out = {}
    for hq, hkv, d in sorted(shapes):
        slope = _kv_slope(hq, hkv, d, dtype)
        # `slope <= 0.0` let a genuinely fused kernel's own allocator jitter
        # (measured 7.1 B/row/key at D=128, 16 heads, bf16 -- never exactly
        # 0) read as "unfused" and route a flash kernel through chunking for
        # zero memory bounded (there is no [Hq, T, N] buffer there to bound).
        # Hq * dtype.size is the scale of one real element of that buffer
        # per head; a fused shape's jitter sits at a few B/row/key regardless
        # of Hq, an unfused one's real cost (~37 at D=256) is many times it.
        if slope < hq * dtype.size:
            if verbose:
                print(f"  chunkattn {hq}/{hkv}x{d}: fused kernel ({slope:.1f} "
                      f"B/row/key, below the {hq * dtype.size} B/row/key "
                      f"noise floor) -> stock")
            continue
        rows = _min_safe_rows(hq, hkv, d, dtype)
        if rows is None:
            print(f"  chunkattn {hq}/{hkv}x{d}: unfused ({slope:.1f} B/row/key) "
                  f"but no width in {ROW_CANDIDATES} reproduced stock bit-for-bit "
                  f"-- leaving prefill unchunked, its peak is NOT bounded here")
            continue
        floor = _trim_floor(hq, hkv, d, dtype, rows)
        out[(hq, hkv, d)] = (slope, rows, floor)
        if verbose or not floor:
            print(f"  chunkattn {hq}/{hkv}x{d}: {slope:.1f} B/row/key, "
                  f"min exact chunk {rows} rows, "
                  + (f"causal key trim (>= {floor} keys past {LOOPED})" if floor
                     else "no exact key trim: every chunk takes all keys"))
    return out


def _layout_path(shapes, dtype):
    key = json.dumps([mx.device_info()["device_name"], mx.__version__,
                      hashlib.sha1(Path(__file__).read_bytes()).hexdigest(),
                      sorted(shapes), str(dtype)])
    return LAYOUT_DIR / f"chunkattn-{hashlib.sha1(key.encode()).hexdigest()[:16]}.json"


def install(model, table=None, verbose=False):
    """Route the model's prefill attention through query-chunking wherever a
    fused MLX kernel is missing for that head shape, reusing the saved
    calibration or measuring and caching one, and composing with foldattn by
    testing chunkattn's own `.chunkattn` marker rather than the `.stock`
    marker every patcher sets."""
    global layout
    mods, shapes, dtype = _attention_modules(model)
    if table is None:
        path = _layout_path(shapes, dtype)
        try:
            table = {tuple(int(x) for x in k.split(",")): tuple(v)
                     for k, v in json.loads(path.read_text()).items()}
            note(f"  chunkattn: calibrated routes reused from {path}")
        except (OSError, ValueError):
            table = calibrate(shapes, dtype, verbose=verbose)
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".tmp")
                tmp.write_text(json.dumps({",".join(map(str, k)): list(v)
                                           for k, v in table.items()}))
                tmp.replace(path)
            except OSError as e:
                # An optional memory lever's own calibration cache is not
                # worth failing the model load over: a full disk or a
                # read-only ~/.cache would otherwise raise straight out of
                # StreamingLLM(), real weights included (review finding,
                # 2026-09-29). The table just computed is still used below;
                # only the next process's install() re-pays this probe.
                print(f"  chunkattn: could not save calibration to {path}: {e}")
    for mod in mods:
        if not hasattr(mod.scaled_dot_product_attention, "chunkattn"):
            mod.scaled_dot_product_attention = _wrap(mod.scaled_dot_product_attention)
    layout = table
    said = _gqa_install(shapes, dtype)
    if said:
        note(f"  decode attention: {said}")
    said = _sm_install(table, dtype)
    if said:
        note(f"  prefill softmax: {said}")
    note("  prefill attention: chunkattn "
          + (", ".join(f"{hq}/{hkv}x{d} >= {v[1]} rows" + (", causal keys" if v[2:3] and v[2] else "")
                        for (hq, hkv, d), v in sorted(table.items()))
             or "nothing (every shape has a fused kernel, or none was exact)"))
    return table
