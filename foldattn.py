"""Folded GQA attention for the multi-row verify passes of a dense target: the G*T query rows that share a kv head are folded into one tile so each K and V element is read once per pass with the stock kernel's fp32 arithmetic, and install() routes a context bucket to it only where calibration timed it faster than stock."""

import hashlib
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from rowqmm import LAYOUT_DIR
from engine_log import note

BK = 16                    # keys per tile, small so many threadgroups fit a core
CHUNKS = (4, 8, 16, 32)    # key chunks per kv head that calibration picks from
MIN_LOG2 = 4               # context buckets calibrated: [2^4, 2^16), one tile up
MAX_LOG2 = 15
MAX_ROWS = 8               # stock's vector kernel covers T <= 8; wider is prefill
CAL_BYTES = 1 << 29        # cap on the K/V buffers calibration allocates

FOLD_HEADER = "#include <metal_simdgroup_matrix>\n"

FOLD_SOURCE = """
    constexpr int KS = D + 8;    // K/V tile row stride, padded off the banks
    constexpr int ND = D / 8;
    constexpr int MP = NSG * 8;  // folded rows, padded to whole simdgroups
    constexpr int CPL = BK / 4;  // score columns per lane, 4 lanes per row

    threadgroup DT ktile[BK * KS];
    threadgroup DT vtile[BK * KS];
    threadgroup float stile[NSG * 8 * BK];
    threadgroup float ftile[NSG * 64];

    const int n = k_shape[2];
    const int hkv = k_shape[1];
    const int bh = threadgroup_position_in_grid.z;
    const int b = bh / hkv, h = bh % hkv;
    const int C = threadgroups_per_grid.y;
    const int c = threadgroup_position_in_grid.y;
    const ushort lane = thread_index_in_simdgroup;
    const ushort sg = simdgroup_index_in_threadgroup;
    const ushort tid = sg * 32 + lane;
    const float scale = scl[0];

    // this threadgroup's keys: whole tiles, only the last chunk runs short
    const int per = ((n + C - 1) / C + BK - 1) / BK * BK;
    const int j0 = c * per;
    const int j1 = min(j0 + per, n);

    // K and V are read through their strides: a KVCache hands out a slice of
    // a longer buffer, and copying it to contiguous would cost more than the
    // attention itself
    const device DT* kb = k + b * k_strides[0] + h * k_strides[1];
    const device DT* vb = v + b * v_strides[0] + h * v_strides[1];
    // 4-wide loads need D contiguous and every row start 4-aligned
    const bool dense = k_strides[3] == 1 && v_strides[3] == 1
        && (k_strides[0] | k_strides[1] | k_strides[2]) % 4 == 0
        && (v_strides[0] | v_strides[1] | v_strides[2]) % 4 == 0;

    // this simdgroup's 8 query rows, all of D, live in registers for the pass
    const int r0 = sg * 8;
    simdgroup_matrix<DT, 8, 8> qm[ND];
    for (short i = 0; i < ND; i++) {
        simdgroup_load(qm[i], q + ((size_t)bh * MP + r0) * D + i * 8, D);
    }
    simdgroup_matrix<float, 8, 8> om[ND];
    for (short i = 0; i < ND; i++) {
        om[i] = simdgroup_matrix<float, 8, 8>(0);
    }

    // lane owns row rr of the simdgroup's 8 and score columns cq*CPL..+CPL
    const ushort rr = lane / 4, cq = lane % 4;
    const int grow = r0 + rr;
    const int last = n - T + grow % T;   // last key this row may see
    float mrow = -INFINITY, lrow = 0.0f;
    threadgroup float* st = stile + sg * 8 * BK;
    threadgroup float* ft = ftile + sg * 64;
    for (short i = lane; i < 64; i += 32) {
        ft[i] = 0.0f;
    }

    for (int jt = j0; jt < j1; jt += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (dense) {
            // 4 elements per load; lanes walk D so a row comes in one transaction
            for (int i = tid; i < BK * D / 4; i += NSG * 32) {
                const int kr = i / (D / 4), d = i % (D / 4) * 4;
                const int j = jt + kr;
                vec<DT, 4> kx = vec<DT, 4>(0), vx = vec<DT, 4>(0);
                if (j < j1) {
                    kx = *(const device vec<DT, 4>*)(kb + j * k_strides[2] + d);
                    vx = *(const device vec<DT, 4>*)(vb + j * v_strides[2] + d);
                }
                *(threadgroup vec<DT, 4>*)(ktile + kr * KS + d) = kx;
                *(threadgroup vec<DT, 4>*)(vtile + kr * KS + d) = vx;
            }
        } else {
            for (int i = tid; i < BK * D; i += NSG * 32) {
                const int kr = i / D, d = i % D;
                const int j = jt + kr;
                const bool in = j < j1;
                ktile[kr * KS + d] = in ? kb[j * k_strides[2] + d * k_strides[3]] : DT(0);
                vtile[kr * KS + d] = in ? vb[j * v_strides[2] + d * v_strides[3]] : DT(0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // S = Q K^T for this tile
        for (short t = 0; t < BK / 8; t++) {
            simdgroup_matrix<float, 8, 8> sm = simdgroup_matrix<float, 8, 8>(0);
            for (short i = 0; i < ND; i++) {
                simdgroup_matrix<DT, 8, 8> km;
                simdgroup_load(km, ktile + t * 8 * KS + i * 8, KS, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(sm, qm[i], km, sm);
            }
            simdgroup_store(sm, st + t * 8, BK);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        float s[CPL];
        float tmax = -INFINITY;
        for (short u = 0; u < CPL; u++) {
            const int j = jt + cq * CPL + u;
            s[u] = (j < j1 && j <= last) ? st[rr * BK + cq * CPL + u] * scale : -INFINITY;
            tmax = max(tmax, s[u]);
        }
        tmax = max(tmax, simd_shuffle_xor(tmax, 1));
        tmax = max(tmax, simd_shuffle_xor(tmax, 2));
        const float mnew = max(mrow, tmax);
        // a row that has seen no key yet keeps m = -inf: exp(-inf - -inf) is NaN
        const bool none = mnew == -INFINITY;
        const float fac = none ? 1.0f : exp(mrow - mnew);
        float psum = 0.0f;
        for (short u = 0; u < CPL; u++) {
            const float p = none ? 0.0f : exp(s[u] - mnew);
            st[rr * BK + cq * CPL + u] = p;
            psum += p;
        }
        psum += simd_shuffle_xor(psum, 1);
        psum += simd_shuffle_xor(psum, 2);
        lrow = lrow * fac + psum;
        mrow = mnew;
        if (cq == 0) {
            ft[rr * 9] = fac;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // rescale O's rows by diag(fac) on the MMA unit; once every row's max
        // has settled the factors are all 1 and this is skipped
        if (simd_any(fac != 1.0f)) {
            simdgroup_matrix<float, 8, 8> fm;
            simdgroup_load(fm, ft, 8);
            for (short i = 0; i < ND; i++) {
                simdgroup_matrix<float, 8, 8> o = om[i];
                simdgroup_multiply(om[i], fm, o);
            }
        }

        // O += P V
        for (short t = 0; t < BK / 8; t++) {
            simdgroup_matrix<float, 8, 8> pm;
            simdgroup_load(pm, st + t * 8, BK);
            for (short i = 0; i < ND; i++) {
                // V widened to fp32 element by element: both fragments of an
                // 8x8 share one lane layout, whatever it is
                simdgroup_matrix<DT, 8, 8> vh;
                simdgroup_matrix<float, 8, 8> vm;
                simdgroup_load(vh, vtile + t * 8 * KS + i * 8, KS);
                vm.thread_elements()[0] = float(vh.thread_elements()[0]);
                vm.thread_elements()[1] = float(vh.thread_elements()[1]);
                simdgroup_multiply_accumulate(om[i], pm, vm, om[i]);
            }
        }
    }

    device float* po = part + (((size_t)bh * C + c) * MP + r0) * D;
    for (short i = 0; i < ND; i++) {
        simdgroup_store(om[i], po + i * 8, D);
    }
    if (cq == 0) {
        device float* pm = ml + (((size_t)bh * C + c) * MP + grow) * 2;
        pm[0] = mrow;
        pm[1] = lrow;
    }
"""

MERGE_SOURCE = """
    // one thread per output element: merge the C chunks of one folded row
    const int d = thread_position_in_grid.x;
    const int r = thread_position_in_grid.y;
    const int bh = thread_position_in_grid.z;
    const int M = threads_per_grid.y;
    const int C = part_shape[1];
    const int MP = part_shape[2];
    const int D = part_shape[3];

    const device float* mlr = ml + ((size_t)bh * C * MP + r) * 2;
    const device float* pr = part + ((size_t)bh * C * MP + r) * D + d;
    float mmax = -INFINITY;
    for (int c = 0; c < C; c++) {
        mmax = max(mmax, mlr[c * MP * 2]);
    }
    float num = 0.0f, den = 0.0f;
    for (int c = 0; c < C; c++) {
        const float w = exp(mlr[c * MP * 2] - mmax);  // an empty chunk weighs 0
        num += w * pr[(size_t)c * MP * D];
        den += w * mlr[c * MP * 2 + 1];
    }
    out[((size_t)bh * M + r) * D + d] = static_cast<DT>(num / den);
"""

_fold_kernel = mx.fast.metal_kernel(
    name="foldattn",
    input_names=["q", "k", "v", "scl"],
    output_names=["part", "ml"],
    source=FOLD_SOURCE,
    header=FOLD_HEADER,
    ensure_row_contiguous=False,
)

_merge_kernel = mx.fast.metal_kernel(
    name="foldattn_merge",
    input_names=["part", "ml"],
    output_names=["out"],
    source=MERGE_SOURCE,
)


def fold(q, k, v, scale, chunks):
    """Causal (bottom-right) GQA attention of q (B, Hq, T, D) against k, v
    (B, Hkv, N, D), the last T of the N positions being q's own; same result
    as the stock kernel with mask="causal"."""
    B, Hq, T, D = q.shape
    Hkv = k.shape[1]
    M = Hq // Hkv * T
    nsg = (M + 7) // 8
    qf = q.reshape(B, Hkv, M, D)
    if nsg * 8 > M:
        qf = mx.concatenate([qf, mx.zeros((B, Hkv, nsg * 8 - M, D), q.dtype)], axis=2)
    else:
        qf = mx.contiguous(qf)
    part, ml = _fold_kernel(
        inputs=[qf, k, v, mx.array([scale], dtype=mx.float32)],
        template=[("DT", q.dtype), ("D", D), ("NSG", nsg), ("T", T), ("BK", BK)],
        grid=(32, nsg * chunks, B * Hkv),
        threadgroup=(32, nsg, 1),
        output_shapes=[(B * Hkv, chunks, nsg * 8, D), (B * Hkv, chunks, nsg * 8, 2)],
        output_dtypes=[mx.float32, mx.float32],
    )
    out = _merge_kernel(
        inputs=[part, ml],
        template=[("DT", q.dtype)],
        grid=(D, M, B * Hkv),
        threadgroup=(D, 1, 1),
        output_shapes=[(B, Hkv, M, D)],
        output_dtypes=[q.dtype],
    )[0]
    return out.reshape(B, Hq, T, D)


# (Hq, Hkv, D, T, floor(log2 N)) -> chunks, only where calibration measured a
# win (everything else is stock); process-wide like rowqmm's layout, so one
# process drives one target.
layout = {}


def _route(q, k, cache, mask, sinks):
    T = q.shape[2]
    if (T < 2 or T > MAX_ROWS or sinks is not None or hasattr(cache, "bits")
            or not isinstance(mask, str) or mask != "causal"
            or q.dtype not in (mx.bfloat16, mx.float16) or k.dtype != q.dtype
            or q.shape[1] % k.shape[1] or q.shape[3] % 8):
        return None
    return layout.get((q.shape[1], k.shape[1], q.shape[3], T, k.shape[2].bit_length() - 1))


def _wrap(stock):
    def scaled_dot_product_attention(queries, keys, values, cache, scale, mask,
                                     sinks=None):
        chunks = _route(queries, keys, cache, mask, sinks)
        if chunks is None:
            return stock(queries, keys, values, cache=cache, scale=scale,
                         mask=mask, sinks=sinks)
        return fold(queries, keys, values, scale, chunks)

    scaled_dot_product_attention.stock = stock
    # A second, foldattn-specific marker, because the shared `.stock` convention
    # cannot tell "wrapped by ME" from "wrapped by chunkattn" (they share the
    # qwen3_next module), so install() tests this one and composes on top of an
    # earlier wrapper instead of skipping it.
    scaled_dot_product_attention.foldattn = True
    return scaled_dot_product_attention


def _attention_modules(model):
    """The python modules whose attention calls the shared sdpa helper (e.g.
    mlx_lm.models.qwen3), the (Hq, Hkv, D) of the attention layers, how many
    there are, and the activation dtype they run in."""
    mods, shapes, layers, dtype = set(), set(), 0, mx.bfloat16
    for m in model.modules():
        mod = sys.modules.get(type(m).__module__)
        if mod is not None and hasattr(mod, "scaled_dot_product_attention"):
            mods.add(mod)
        if hasattr(m, "n_kv_heads") and hasattr(m, "scale"):
            shapes.add((m.n_heads, m.n_kv_heads, round(m.scale ** -2)))
            layers += 1
            dtype = next((v.dtype for _, v in tree_flatten(m.parameters())
                          if v.dtype in (mx.float16, mx.bfloat16)), dtype)
    return mods, shapes, layers, dtype


def _interleaved(fns, reps):
    """Each fn timed reps times, round-robin, so drift lands on all of them."""
    for f in fns:
        mx.eval(f())
        mx.eval(f())
    ts = [[] for _ in fns]
    for _ in range(reps):
        for i, f in enumerate(fns):
            t0 = time.perf_counter()
            mx.eval(f())
            ts[i].append((time.perf_counter() - t0) * 1000)
    return [sorted(t) for t in ts]


def calibrate(shapes, layers, dtype=mx.bfloat16, rows=range(2, MAX_ROWS + 1),
              buckets=range(MIN_LOG2, MAX_LOG2 + 1), reps=9, verbose=False):
    """{(Hq, Hkv, D, T, log2 N): chunks} for every bucket where the fold beat
    stock. One bucket is timed at N = 2^b over `layers` distinct K/V (fewer if
    they would pass CAL_BYTES), so every call streams from memory as decode
    does. A route has to win by more than the spread of the timings: the
    fold's upper quartile must sit below stock's lower quartile."""
    out = {}
    q1, q3 = reps // 4, reps - 1 - reps // 4
    for hq, hkv, d in sorted(shapes):
        scale = d ** -0.5
        for b in buckets:
            n = 1 << b
            copies = max(1, min(layers, CAL_BYTES // (2 * hkv * n * d * dtype.size)))
            ks = [mx.random.normal((1, hkv, n, d)).astype(dtype) for _ in range(copies)]
            vs = [mx.random.normal((1, hkv, n, d)).astype(dtype) for _ in range(copies)]
            mx.eval(ks, vs)
            for T in rows:
                q = mx.random.normal((1, hq, T, d)).astype(dtype)
                fns = [lambda: [mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=scale, mask="causal") for k, v in zip(ks, vs)]]
                fns += [lambda c=c: [fold(q, k, v, scale, c) for k, v in zip(ks, vs)]
                        for c in CHUNKS]
                ts = _interleaved(fns, reps)
                best = min(range(1, len(fns)), key=lambda i: ts[i][reps // 2])
                won = ts[best][q3] < ts[0][q1]
                if won:
                    out[(hq, hkv, d, T, b)] = CHUNKS[best - 1]
                if verbose:
                    print(f"  foldattn {hq}/{hkv}x{d} N={n:6d} T={T}: stock "
                          f"{ts[0][reps // 2]:7.2f} fold/{CHUNKS[best - 1]} "
                          f"{ts[best][reps // 2]:7.2f} ms x{copies} -> "
                          f"{'fold' if won else 'stock'}", flush=True)
            ks = vs = None
            mx.clear_cache()
    return out


def _layout_path(shapes, layers, dtype):
    key = json.dumps([mx.device_info()["device_name"], mx.__version__,
                      hashlib.sha1(Path(__file__).read_bytes()).hexdigest(),
                      sorted(shapes), layers, str(dtype)])
    return LAYOUT_DIR / f"foldattn-{hashlib.sha1(key.encode()).hexdigest()[:16]}.json"


def install(model, table=None, verbose=False):
    """Route the model's multi-row attention through the fold where it was
    timed faster on this machine, reusing the saved calibration or calibrating
    and saving one (rowqmm's cache directory and keying), composing with
    chunkattn.py by testing foldattn's OWN marker rather than the shared
    `.stock`, and return the installed {(Hq, Hkv, D, T, log2 N): chunks}."""
    global layout
    mods, shapes, layers, dtype = _attention_modules(model)
    if table is None:
        path = _layout_path(shapes, layers, dtype)
        try:
            table = {tuple(int(x) for x in k.split(",")): c
                     for k, c in json.loads(path.read_text()).items()}
            note(f"  foldattn: calibrated routes reused from {path}")
        except (OSError, ValueError):
            t0 = time.perf_counter()
            table = calibrate(shapes, layers, dtype, verbose=verbose)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({",".join(map(str, k)): c for k, c in table.items()}))
            tmp.replace(path)
            note(f"  foldattn: calibrated in {time.perf_counter() - t0:.0f} s, "
                  f"saved to {path}")
    for mod in mods:
        if not hasattr(mod.scaled_dot_product_attention, "foldattn"):
            mod.scaled_dot_product_attention = _wrap(mod.scaled_dot_product_attention)
    layout = table
    note("  target attention: foldattn " + _describe(table))
    return table


def _describe(table):
    """'T=5 N 256-65535; T=8 N 16-65535': the context ranges routed per T."""
    runs = []
    for t in sorted({key[3] for key in table}):
        bs = sorted(key[4] for key in table if key[3] == t)
        spans, lo = [], bs[0]
        for a, b in zip(bs, bs[1:] + [None]):
            if b != a + 1:
                spans.append(f"{1 << lo}-{(2 << a) - 1}")
                lo = b
        runs.append(f"T={t} N {','.join(spans)}")
    return "; ".join(runs) or "nothing (stock was faster)"
