"""QSA-sparse prefill attention in tile form (MOE_QSA_TILES=1, Metal, off by default): each G-row group of a chunkattn chunk runs both GEMMs over only the TK-key tiles one of its rows keeps, with its scores put back into the full row for stock's where and softmax, so a row's bits are the chunk's stock call's."""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

import chunkattn
import stream_fast
from engine_log import note
from rowqmm import LAYOUT_DIR

ON = os.environ.get("MOE_QSA_TILES", "0") == "1"
G = 16              # query rows a group (a GEMM's M is its query heads times these)
TK = 16             # keys a tile, a multiple of 8: MLX's GEMM sums the keys 8 at a time
ALIGN = chunkattn.KEY_ALIGN     # a GEMM's K a multiple of it keeps MLX summing the keys in order
COST = {}           # (hq, hkv, d) -> (stock_s, stock_k, stock_rk, tile_s, tile_rk, tile_rl): seconds, install's fit
STATS = {"calls": 0, "flagged": 0, "chunks": 0, "tiled": 0,      # array-masked calls, those whose flags were read, their chunks
         "batches": 0, "host_s": 0.0, "eval_s": 0.0, "flags_s": 0.0, "stock_s": 0.0}     # where a call's time goes


def plan(flags, nk):
    """[groups, L] key indices from each group's kept tiles below nk: the kept tiles in key order, padded with
    unkept ones (every key of them masked for the group's rows) to a common L that is a multiple of ALIGN."""
    f = flags[:, :nk // TK]
    lt = -(-int(f.sum(axis=1).max()) // (ALIGN // TK)) * (ALIGN // TK)
    tiles = np.sort(np.argsort(~f, axis=1, kind="stable")[:, :lt], axis=1)
    return (tiles[:, :, None] * TK + np.arange(TK, dtype=np.int32)).reshape(len(f), lt * TK).astype(np.int32)


def starts(c0, c1):
    """[(first row, rows)] of the groups covering [c0, c1): G rows each and the rest as a group of its own, or,
    for a rest of one row (a GEMM of rep rows, MLX's small-M kernel), a G-row group moved back to end at c1."""
    s = [(r, G) for r in range(c0, c1 - G + 1, G)]
    rest = (c1 - c0) % G
    return s + [(c1 - rest, rest) if rest > 1 else (c1 - G, G)] if rest else s


# Three copy kernels move the tile form's data (a group's kept K and V tiles, its scores into the full row with
# the mask's select, the kept probabilities back out) while MLX's ops do its arithmetic, so the bits are the ops'.
_KV_SRC = """
  const int ng = prm[0], lt = prm[1] / TK;
  const uint x = thread_position_in_grid.x;
  const uint g = thread_position_in_grid.y;
  const uint hz = thread_position_in_grid.z;
  const uint key = x / (D / 8);
  if (key >= uint(lt * TK)) {
    return;
  }
  const uint h = hz >> 1;
  const int64_t pos = int64_t(tiles[g * lt + key / TK]) * TK + key % TK;
  const device T* src = (hz & 1) ? v + h * v_strides[1] + pos * v_strides[2]
                                 : k + h * k_strides[1] + pos * k_strides[2];
  device T* dst = ((hz & 1) ? vg : kg) + ((int64_t(h) * ng + g) * lt * TK + key) * D;
  reinterpret_cast<device uint4*>(dst)[x % (D / 8)] = reinterpret_cast<const device uint4*>(src)[x % (D / 8)];
"""
_EXPAND_SRC = """
  const int ng = prm[0], L = prm[1], nk = prm[2], rg = prm[3], gr = prm[4];
  const uint x = thread_position_in_grid.x;
  const uint y = thread_position_in_grid.y;
  const uint z = thread_position_in_grid.z;
  if (x >= uint(nk / 8)) {
    return;
  }
  const uint g = z % ng;
  const int j0 = x * 8;
  const int slot = inv[g * (nk / TK) + j0 / TK];
  const int64_t row = int64_t(z) * rg + y;
  device T* dst = full + row * nk + j0;
  const T neg = negv[0];
  if (slot < 0) {
    for (int e = 0; e < 8; e++) {
      dst[e] = neg;
    }
    return;
  }
  const device T* src = s + row * L + slot * TK + j0 % TK;
  const device bool* m = mask + int64_t(rows[g * gr + y % gr]) * mask_strides[0] + int64_t(j0) * mask_strides[1];
  for (int e = 0; e < 8; e++) {
    dst[e] = m[e * mask_strides[1]] ? src[e] : neg;
  }
"""
_P_SRC = """
  const int L = prm[1], nk = prm[2], rg = prm[3], ng = prm[0];
  const uint x = thread_position_in_grid.x;
  const uint y = thread_position_in_grid.y;
  const uint z = thread_position_in_grid.z;
  if (x >= uint(L / 8)) {
    return;
  }
  const int l0 = x * 8;
  const int t = tiles[(z % ng) * (L / TK) + l0 / TK];
  const int64_t row = int64_t(z) * rg + y;
  reinterpret_cast<device uint4*>(pg + row * L)[x] =
      *reinterpret_cast<const device uint4*>(p + row * nk + t * TK + l0 % TK);
"""
_FLAGS_SRC = """
  const uint t = thread_position_in_grid.x;
  const uint g = thread_position_in_grid.y;
  if (t >= uint(mask_shape[1] / TK)) {
    return;
  }
  bool any = false;
  for (int i = 0; i < cnt[g]; i++) {
    const device bool* m = mask + int64_t(st[g] + i) * mask_strides[0] + int64_t(t) * TK * mask_strides[1];
    for (int e = 0; e < TK; e++) {
      any = any || m[e * mask_strides[1]];
    }
  }
  flags[g * (mask_shape[1] / TK) + t] = any;
"""
_KERNELS = {}


def _kernel(name):
    k = _KERNELS.get(name)
    if k is None:
        if name == "kv":
            k = mx.fast.metal_kernel(name="qsa_tiles_kv", input_names=["k", "v", "tiles", "prm"],
                                     output_names=["kg", "vg"], source=_KV_SRC, ensure_row_contiguous=False)
        elif name == "flags":
            k = mx.fast.metal_kernel(name="qsa_tiles_flags", input_names=["mask", "st", "cnt"], output_names=["flags"],
                                     source=_FLAGS_SRC, ensure_row_contiguous=False)
        elif name == "expand":
            k = mx.fast.metal_kernel(name="qsa_tiles_expand",
                                     input_names=["s", "mask", "rows", "inv", "prm", "negv"],
                                     output_names=["full"], source=_EXPAND_SRC, ensure_row_contiguous=False)
        else:
            k = mx.fast.metal_kernel(name="qsa_tiles_p", input_names=["p", "tiles", "prm"],
                                     output_names=["pg"], source=_P_SRC)
        _KERNELS[name] = k
    return k


COPIES = True       # the copy kernels (False: the same moves in MLX's take / put_along_axis, ~3x slower)


def tile_groups(q, k, v, scale, mask2, st, nk, kidx, copies=None, G=G):
    """[1, Hq, len(st) * G, D]: the stock call's rows st[i] .. st[i] + G over keys :nk, group i on its kept keys
    kidx[i] (plan()'s), q being the call's [1, Hq, T, D] and mask2 its [T, N] mask, built in a function so its
    intermediates are dropped before the caller's eval and MLX can reuse their buffers."""
    copies = COPIES if copies is None else copies
    hq, d, hkv = q.shape[1], q.shape[3], k.shape[1]
    rep = hq // hkv
    n_g, L = kidx.shape
    rows = mx.array((np.asarray(st, np.int32)[:, None] + np.arange(G, dtype=np.int32)).reshape(-1))
    # MLX's fallback (fast.cpp): q * scale, GQA by reshape, scores, where with the dtype's min, precise softmax, P @ V.
    qs = mx.multiply(mx.array(scale, q.dtype), mx.take(q, rows, axis=2))
    # A group's rep query heads and G rows are one GEMM's M, as one K/V head's keys serve them all.
    qs = qs.reshape(1, hkv, rep, n_g, G, d).transpose(0, 1, 3, 2, 4, 5).reshape(1, hkv, n_g, rep * G, d)
    neg = mx.array(mx.finfo(q.dtype).min, q.dtype)
    if copies:
        tiles = kidx[:, ::TK] // TK
        prm = mx.array([n_g, L, nk, rep * G, G], mx.int32)
        mt = mx.array(tiles.astype(np.int32))
        kg, vg = _kernel("kv")(inputs=[k, v, mt, prm], template=[("T", k.dtype), ("D", d), ("TK", TK)],
                               grid=(L * d // 8, n_g, 2 * hkv), threadgroup=(256, 1, 1),
                               output_shapes=[(1, hkv, n_g, L, d)] * 2, output_dtypes=[k.dtype] * 2)
        s = qs @ mx.swapaxes(kg, -1, -2)
        inv = np.full((n_g, nk // TK), -1, np.int32)
        inv[np.arange(n_g)[:, None], tiles] = np.arange(L // TK, dtype=np.int32)
        full = _kernel("expand")(inputs=[s, mask2, rows, mx.array(inv), prm, neg.reshape(1)],
                                 template=[("T", q.dtype), ("TK", TK)],
                                 grid=(nk // 8, rep * G, hkv * n_g),
                                 threadgroup=(256, 1, 1), output_shapes=[(1, hkv, n_g, rep * G, nk)],
                                 output_dtypes=[q.dtype])[0]
        p = mx.softmax(full, axis=-1, precise=True)
        p = _kernel("p")(inputs=[p, mt, prm], template=[("T", q.dtype), ("TK", TK)],
                         grid=(L // 8, rep * G, hkv * n_g),
                         threadgroup=(256, 1, 1), output_shapes=[(1, hkv, n_g, rep * G, L)],
                         output_dtypes=[q.dtype])[0]
    else:
        ki = mx.array(kidx)
        kg = mx.take(k, ki.reshape(-1), axis=2).reshape(1, hkv, n_g, L, d)
        vg = mx.take(v, ki.reshape(-1), axis=2).reshape(1, hkv, n_g, L, d)
        s = (qs @ mx.swapaxes(kg, -1, -2)).reshape(1, hkv, n_g, rep, G, L)
        mg = mx.take_along_axis(mx.take(mask2, rows, axis=0).reshape(n_g, G, -1), ki[:, None, :], axis=-1)
        s = mx.where(mg.reshape(1, 1, n_g, 1, G, L), s, neg)
        at = ki.reshape(1, 1, n_g, 1, 1, L)
        full = mx.put_along_axis(mx.full((1, hkv, n_g, rep, G, nk), neg), at, s, axis=-1)
        p = mx.take_along_axis(mx.softmax(full, axis=-1, precise=True), at, axis=-1)
        p = p.reshape(1, hkv, n_g, rep * G, L)
    o = (p @ vg).reshape(1, hkv, n_g, rep, G, d)
    return o.transpose(0, 1, 3, 2, 4, 5).reshape(1, hq, n_g * G, d)


def group_bytes(hq, hkv, d, size, b, L, nk, r=None):
    """Bytes b groups of r rows (G) and L kept keys over nk keys hold at once (one command buffer keeps every
    intermediate): the full rows, the scores and probabilities on the kept keys, and the gathered K and V."""
    return size * (hq * (r or G) * b * (nk + 2 * L) + 2 * hkv * b * L * d)


def batches(flags, rows, nk, shape, size, budget):
    """[(i, j, L, r)]: groups i..j-1 (of r rows each, `rows` theirs) evaluated together on L keys each, as many
    as fit `budget` bytes, one at least: a group is never split by K/V head, since one head's one group is a
    GEMM of batch 1, where MLX sums P @ V's keys in split-K order."""
    lt = -(-flags[:, :nk // TK].sum(axis=1) // (ALIGN // TK)) * ALIGN
    out, i = [], 0
    while i < len(lt):
        j, L, r = i + 1, int(lt[i]), rows[i]
        while (j < len(lt) and rows[j] == r
               and group_bytes(*shape, size, j + 1 - i, max(L, int(lt[j])), nk, r) <= budget):
            L = max(L, int(lt[j]))
            j += 1
        out.append((i, j, L, r))
        i = j
    return out


def tile_chunk(q, k, v, scale, mask2, c0, c1, nk, flags, budget=None, copies=None):
    """Rows c0..c1 of the stock call over keys :nk on the tile form, flags the chunk's groups' (group_flags),
    in batches of groups whose transient stays under `budget` bytes, each forced before the next (one batch,
    unforced, without a budget)."""
    gs = starts(c0, c1)
    st, rows = [s for s, _ in gs], [r for _, r in gs]
    shape = (q.shape[1], k.shape[1], q.shape[3])
    if budget is None:      # the full groups in one batch, a short last group in its own
        n = len(gs) - (rows[-1] != G)
        bs = ([(0, n, 0, G)] if n else []) + ([(n, len(gs), 0, rows[-1])] if n < len(gs) else [])
    else:
        bs = batches(flags, rows, nk, shape, q.dtype.size, budget)
    outs = []
    for i, j, _, r in bs:
        t0 = time.perf_counter()
        o = tile_groups(q, k, v, scale, mask2, st[i:j], nk, plan(flags[i:j], nk), copies, r)
        if budget is not None:
            t1 = time.perf_counter()
            mx.eval(o)
            STATS["batches"] += 1
            STATS["host_s"] += t1 - t0
            STATS["eval_s"] += time.perf_counter() - t1
        outs.append(o)
    o = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=2)
    # Each row from the first group holding it (a moved-back last group overlaps the one before).
    at = np.full(c1 - c0, -1, np.int32)
    off = 0
    for s, r in gs:
        seg = at[s - c0:s - c0 + r]
        seg[seg < 0] = off + np.nonzero(seg < 0)[0]
        off += r
    return mx.take(o, mx.array(at), axis=2)


def group_flags(mask2, bounds):
    """Host [groups, N // TK] bools: which key tiles any row of each group keeps, for every chunk's groups."""
    nt = mask2.shape[1] // TK
    st = np.array([g for c0, c1 in zip(bounds, bounds[1:]) for g in starts(c0, c1)], np.int32).reshape(-1, 2)
    return np.array(_kernel("flags")(inputs=[mask2, mx.array(st[:, 0].copy()), mx.array(st[:, 1].copy())], template=[("TK", TK)],
                                     grid=(nt, len(st), 1), threadgroup=(256, 1, 1),
                                     output_shapes=[(len(st), nt)], output_dtypes=[mx.bool_])[0])


def _wins(cost, c, nk, bs):
    """True if install's fit puts the tile form's batches bs (batches()) below stock for c rows over nk keys."""
    ss, sk, srk, ts, trk, trl = cost
    tile = sum(ts + r * (j - i) * (trk * nk + trl * L) for i, j, L, r in bs)
    return tile < ss + sk * nk + srk * c * nk


def chunked(q, k, v, scale, rows, min_rows, floor=0, mask=None):
    """chunkattn.chunked with each chunk of an array-masked call on the tile form where install's timing puts
    it below the chunk's stock call, its transient held to what chunkattn sized the chunk for; the chunks,
    their keys and every other call as chunkattn's."""
    shape = (q.shape[1], k.shape[1], q.shape[3])
    cost = COST.get(shape)
    T, N = q.shape[2], k.shape[2]
    if mask is None or cost is None or T < G:
        return _stock_chunked(q, k, v, scale, rows, min_rows, floor, mask)
    STATS["calls"] += 1
    offset = N - T
    rows = max(rows, min_rows)
    bounds = list(range(0, T, rows)) + [T]
    if len(bounds) > 2 and bounds[-1] - bounds[-2] < min_rows:
        bounds.pop(-2)
    cuts = [chunkattn._cut(offset + c1, N, floor) if floor else N for c1 in bounds[1:]]
    per_key = chunkattn.layout[shape][0]
    # With no key kept and a batch a group the tile form would still lose: no flags read, stock's chunks.
    if not any(_wins(cost, c1 - c0, nk, [(g, g + 1, 0, r) for g, (_, r) in enumerate(starts(c0, c1))])
               for c0, c1, nk in zip(bounds, bounds[1:], cuts) if c1 - c0 >= G and nk % ALIGN == 0):
        return _stock_chunked(q, k, v, scale, rows, min_rows, floor, mask)
    mask2 = mask.reshape(T, N)
    t0 = time.perf_counter()
    flags = group_flags(mask2, bounds)
    STATS["flags_s"] += time.perf_counter() - t0
    STATS["flagged"] += 1
    return _tile_call(q, k, v, scale, mask, mask2, flags, bounds, cuts, cost, shape, per_key, N)


def _tile_call(q, k, v, scale, mask, mask2, flags, bounds, cuts, cost, shape, per_key, N):
    """chunked()'s chunk loop once a call's flags are read: each chunk on the tile form or stock's call."""
    outs, g0 = [], 0
    for c0, c1, nk in zip(bounds, bounds[1:], cuts):
        if c0:
            stream_fast._prefill_stop()
        STATS["chunks"] += 1
        gs = starts(c0, c1)
        f = flags[g0:g0 + len(gs)]
        g0 += len(gs)
        budget = per_key * (c1 - c0) * N         # what chunkattn sized this chunk's call for
        if nk % ALIGN == 0 and _wins(cost, c1 - c0, nk,
                                     batches(f, [r for _, r in gs], nk, shape, q.dtype.size, budget)):
            out = tile_chunk(q, k, v, scale, mask2, c0, c1, nk, f, budget)
            STATS["tiled"] += 1
        else:
            t0 = time.perf_counter()
            kc, vc = (k, v) if nk == N else (k[:, :, :nk], v[:, :, :nk])
            out = mx.fast.scaled_dot_product_attention(q[:, :, c0:c1], kc, vc, scale=scale,
                                                       mask=mask[..., c0:c1, :nk])
            mx.eval(out)
            STATS["stock_s"] += time.perf_counter() - t0
        mx.eval(out)
        outs.append(out)
    return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=2)


_stock_chunked = chunkattn.chunked


def _synthetic(hq, hkv, d, dtype, c, nk, kept, rng):
    """q [1, hq, c, d], k, v [1, hkv, nk, d] and a QSA-like [c, nk] mask: c rows ending at key nk, each keeping
    its causal keys in ~`kept` of the 16-key blocks (rows sharing most picks) and its own tail."""
    q = mx.random.normal((1, hq, c, d)).astype(dtype)
    k = mx.random.normal((1, hkv, nk, d)).astype(dtype)
    v = mx.random.normal((1, hkv, nk, d)).astype(dtype)
    nt = nk // 16
    base = rng.random(nt) < kept
    m = np.zeros((c, nk), bool)
    for r in range(c):
        t = base ^ (rng.random(nt) < 0.02)
        m[r] = np.repeat(t, 16)
        last = nk - c + r
        m[r, last - 3:] = True
        m[r, last + 1:] = False
    return q, k, v, mx.array(m)


def _time(fn, reps=5):
    """Median seconds of fn's evaluation after one warm-up."""
    mx.eval(fn())
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def _group_mask(c, nk, share, rng):
    """A causal [c, nk] mask whose G-row groups each keep the same `share` of the TK-key tiles, every row its
    own position too: a chunk whose kept tiles are known for timing."""
    m = np.zeros((c, nk), bool)
    for g0 in range(0, c, G):
        m[g0:g0 + G] = np.repeat(rng.random(nk // TK) < share, TK)
    last = nk - c + np.arange(c)
    m[np.arange(c), last] = True
    return mx.array(m & (np.arange(nk)[None, :] <= last[:, None]))


def _fit(rows, ys):
    """Least squares of ys on the columns of rows, a column whose coefficient comes out negative dropped (0)."""
    a, y = np.array(rows, float), np.array(ys, float)
    keep = list(range(a.shape[1]))
    while True:
        x = np.linalg.lstsq(a[:, keep], y, rcond=None)[0]
        if (x >= 0).all() or len(keep) == 1:
            out = np.zeros(a.shape[1])
            out[keep] = np.maximum(x, 0)
            return [float(v) for v in out]
        keep.pop(int(np.argmin(x)))


def calibrate(hq, hkv, d, dtype, per_key=None):
    """(stock_s, stock_k, stock_rk, tile_s, tile_rk, tile_rl) for this shape: seconds of a stock chunk (fixed, a
    key, a row and key) and of a tile chunk (a batch, a row and key, a row and kept key), from whole chunks as a
    prefill cuts them (112 and 48 rows over 16k keys, 80, 40 and 24 over 40k), the tile form's groups keeping 20,
    45 or 70% of the tiles and run as the engine runs it; None where it does not give synthetic QSA-like chunks'
    stock bits."""
    rng = np.random.default_rng(0)
    scale = d ** -0.5
    shape = (hq, hkv, d)
    per_key = per_key or chunkattn.layout.get(shape, (2.0 * hq,))[0]
    for nk in (16384, 40960):
        for c, kept in ((48, 0.25), (39, 0.7)):
            q, k, v, m = _synthetic(hq, hkv, d, dtype, c, nk, kept, rng)
            mx.eval(q, k, v, m)
            f = group_flags(m, [0, c])
            ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=m)
            for budget in (None, per_key * c * nk):
                if not mx.array_equal(ref, tile_chunk(q, k, v, scale, m, 0, c, nk, f, budget)).item():
                    return None
    st, tl = [], []
    for nk, c in ((16384, 112), (16384, 48), (40960, 80), (40960, 40), (40960, 24)):
        q = mx.random.normal((1, hq, c, d)).astype(dtype)
        k = mx.random.normal((1, hkv, nk, d)).astype(dtype)
        v = mx.random.normal((1, hkv, nk, d)).astype(dtype)
        budget = per_key * c * nk
        for share in ((0.2, 0.45, 0.7) if c > 24 else (0.45,)):
            m = _group_mask(c, nk, share, rng)
            mx.eval(q, k, v, m)
            if share == 0.45:
                st.append(([1.0, nk, c * nk],
                           _time(lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=m))))
            if c > 24:
                bs = batches(group_flags(m, [0, c]), [r for _, r in starts(0, c)], nk, shape, dtype.size, budget)
                tl.append(([len(bs), sum(r * (j - i) for i, j, _, r in bs) * nk,
                            sum(r * (j - i) * L for i, j, L, r in bs)],
                           _time(lambda: tile_chunk(q, k, v, scale, m, 0, c, nk, group_flags(m, [0, c]), budget))))
    return tuple(_fit(*zip(*st)) + _fit(*zip(*tl)))


def _calibrate_apart(layout, cache_bytes):
    """{shape: calibrate()} measured in a child process under the server's MLX cache limit, since its chunks
    would raise this process's MLX peak, which the memory budget reads the host against (8k bits moved through
    a bigger expert pool); {} if the child fails."""
    src = ("import json, sys; import mlx.core as mx; import qsa_tiles as q; mx.set_cache_limit(int(sys.argv[2])); "
           "lay = json.loads(sys.argv[1]); print(json.dumps({s: q.calibrate(*map(int, s.split(',')), mx.bfloat16, "
           "v) for s, v in lay.items()}))")
    arg = json.dumps({",".join(map(str, sh)): v[0] for sh, v in layout.items()})
    try:
        r = subprocess.run([sys.executable, "-c", src, arg, str(int(cache_bytes))], cwd=Path(__file__).parent,
                           capture_output=True, text=True, timeout=600)
        return {tuple(int(x) for x in sh.split(",")): v for sh, v in json.loads(r.stdout.splitlines()[-1]).items()}
    except (OSError, ValueError, IndexError, subprocess.SubprocessError) as e:
        print(f"  qsa tiles: calibration failed ({type(e).__name__}: {e}); stock attention", flush=True)
        return {}


def install(cache_gb=0.05):
    """With MOE_QSA_TILES=1 on Metal: the tile form's cost fit per chunkattn shape (cached like chunkattn's
    calibration, measured apart), and chunkattn's chunked wrapped wherever the tile form gave stock's bits."""
    if not ON or stream_fast.BACKEND != "metal" or chunkattn.chunked is chunked:
        return
    key = json.dumps([mx.device_info()["device_name"], mx.__version__,
                      hashlib.sha1(Path(__file__).read_bytes()).hexdigest(), sorted(chunkattn.layout)])
    path = LAYOUT_DIR / f"qsatiles-{hashlib.sha1(key.encode()).hexdigest()[:16]}.json"
    try:
        table = {tuple(int(x) for x in s.split(",")): v for s, v in json.loads(path.read_text()).items()}
    except (OSError, ValueError):
        table = _calibrate_apart(chunkattn.layout, cache_gb * 1e9)
        try:
            if table:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({",".join(map(str, s)): v for s, v in table.items()}))
        except OSError as e:
            print(f"  qsa tiles: could not save calibration to {path}: {e}")
    COST.clear()
    COST.update({s: tuple(v) for s, v in table.items() if v is not None})
    if COST:
        chunkattn.chunked = chunked
    note("  prefill attention: qsa tiles " + (", ".join(
        f"{hq}/{hkv}x{d}" + (" on" if v else " off (not stock's bits)") for (hq, hkv, d), v in sorted(table.items()))
        or "nothing (no chunked shape)"))
