"""Query-chunked prefill attention for full-attention layers whose head_dim has no fused MLX kernel, splitting only the query rows so the fallback's [heads, T, N] score matrix stays bounded, installed like foldattn.py with chunk widths calibrated per machine."""

import hashlib
import io
import json
import sys
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


def chunked(q, k, v, scale, rows, min_rows, floor=0, mask=None):
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
    """
    T, N = q.shape[2], k.shape[2]
    offset = N - T
    rows = max(rows, min_rows)
    bounds = list(range(0, T, rows)) + [T]
    if len(bounds) > 2 and bounds[-1] - bounds[-2] < min_rows:
        bounds.pop(-2)
    outs = []
    for c0, c1 in zip(bounds, bounds[1:]):
        if c0:
            # A shadow render's pass gives way to a request here too: the
            # one stretch of a layer with no other check (0.9 s at 12k).
            stream_fast._prefill_stop()
        nk = _cut(offset + c1, N, floor) if floor else N
        if mask is None:
            row_pos = offset + c0 + mx.arange(c1 - c0)
            col_pos = mx.arange(nk)
            m = row_pos[:, None] >= col_pos[None, :]
        else:
            m = mask[..., c0:c1, :nk]
        kc, vc = (k, v) if nk == N else (k[:, :, :nk], v[:, :, :nk])
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
        # A shape that measured unfused at calibration may still fuse at
        # this T, and stock's own kernel is then bounded already and the only
        # kernel with its bits.
        arr = not isinstance(mask, str)
        if rows >= T or _stock_fused(queries.shape[1], keys.shape[1],
                                     queries.shape[3], queries.dtype, T, N, arr):
            return stock(queries, keys, values, cache=cache, scale=scale,
                         mask=mask, sinks=sinks)
        if arr:
            return chunked(queries, keys, values, scale, rows, min_rows, *floor, mask=mask)
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
    note("  prefill attention: chunkattn "
          + (", ".join(f"{hq}/{hkv}x{d} >= {v[1]} rows" + (", causal keys" if v[2:3] and v[2] else "")
                        for (hq, hkv, d), v in sorted(table.items()))
             or "nothing (every shape has a fused kernel, or none was exact)"))
    return table
