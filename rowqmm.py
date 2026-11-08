"""4-bit quantized matmul kernels for MLX on Apple silicon (rowqmv, a row-blocked scalar kernel that beats stock from T=5, and qmm8, eight rows exactly on the simdgroup matrix units), plus the calibration that times the model's own forward and routes each (row count, K, N) to whichever path measured fastest."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from engine_log import note

# Quantization format this kernel decodes (affine, 4-bit, 64-wide groups).
GROUP_SIZE = 64
BITS = 4

# Thread layout: 2 simdgroups, 8 lanes reducing K per output row, and each
# thread owning a block of output rows; calibrate() measures which block size
# wins at each row count rather than fixing one.
SIMDGROUPS = 2
K_LANES = 8
ROW_BLOCKS = (1, 2)

# What calibrate() is allowed to route to: a row-block int for this kernel, or
# any callable(flat_x, module) another module registers.
CANDIDATES = list(ROW_BLOCKS)


def _rows_per_tg(rb):
    return (32 // K_LANES) * SIMDGROUPS * rb

ROW_SOURCE = """
    constexpr int slots_per_sg = 32 / KL;
    typedef vec<DT, 4> xvec;

    const int K = x_shape[1];
    const int N = scales_shape[0];
    const int Kg = scales_shape[1];

    const ushort lane = thread_index_in_simdgroup;
    const ushort klane = lane % KL;
    const ushort slot = lane / KL;

    // One thread owns RB consecutive output rows, so each x load feeds RB dot
    // products instead of one. That ratio is what keeps the inner loop on the
    // arithmetic units instead of the load unit.
    const int row0 = (threadgroup_position_in_grid.y * (slots_per_sg * NSG)
                      + simdgroup_index_in_threadgroup * slots_per_sg + slot) * RB;

    const device uint32_t* wr[RB];
    const device DT* sr[RB];
    const device DT* br[RB];
    #pragma unroll
    for (int r = 0; r < RB; r++) {
        const int row = min(row0 + r, N - 1);
        wr[r] = w + (size_t)row * (Kg * 8);
        sr[r] = scales + (size_t)row * Kg;
        br[r] = biases + (size_t)row * Kg;
    }

    float acc[RB][NV];
    #pragma unroll
    for (int r = 0; r < RB; r++) {
        #pragma unroll
        for (int v = 0; v < NV; v++) {
            acc[r][v] = 0.0f;
        }
    }

    // Each lane walks a strided subset of the 64-wide groups. A group is 8
    // words of 8 nibbles; every word is read once and applied to all NV rows,
    // which stay live in registers for the whole pass over K. Everything past
    // the load is fp32: bf16 arithmetic is emulated on this GPU and half math
    // only pays off when the model itself is fp16.
    for (int g = klane; g < Kg; g += KL) {
        float sc[RB], bi[RB];
        #pragma unroll
        for (int r = 0; r < RB; r++) {
            sc[r] = static_cast<float>(sr[r][g]);
            bi[r] = static_cast<float>(br[r][g]);
        }
        #pragma unroll
        for (int q = 0; q < 2; q++) {
            uint4 words[RB];
            #pragma unroll
            for (int r = 0; r < RB; r++) {
                words[r] = ((const device uint4*)(wr[r] + g * 8))[q];
            }
            #pragma unroll
            for (int c = 0; c < 4; c++) {
                float4 w0[RB], w1[RB];
                #pragma unroll
                for (int r = 0; r < RB; r++) {
                    const uint32_t word = words[r][c];
                    w0[r].x = fma(sc[r], float((word >> 0) & 0xF), bi[r]);
                    w0[r].y = fma(sc[r], float((word >> 4) & 0xF), bi[r]);
                    w0[r].z = fma(sc[r], float((word >> 8) & 0xF), bi[r]);
                    w0[r].w = fma(sc[r], float((word >> 12) & 0xF), bi[r]);
                    w1[r].x = fma(sc[r], float((word >> 16) & 0xF), bi[r]);
                    w1[r].y = fma(sc[r], float((word >> 20) & 0xF), bi[r]);
                    w1[r].z = fma(sc[r], float((word >> 24) & 0xF), bi[r]);
                    w1[r].w = fma(sc[r], float((word >> 28) & 0xF), bi[r]);
                }
                const int k0 = g * 64 + q * 32 + c * 8;
                #pragma unroll
                for (int v = 0; v < NV; v++) {
                    const device xvec* xc = (const device xvec*)(x + (size_t)v * K + k0);
                    const float4 a0 = static_cast<float4>(xc[0]);
                    const float4 a1 = static_cast<float4>(xc[1]);
                    #pragma unroll
                    for (int r = 0; r < RB; r++) {
                        acc[r][v] += dot(a0, w0[r]) + dot(a1, w1[r]);
                    }
                }
            }
        }
    }

    // Reduce over the KL lanes that share an output row. simd_sum would mix the
    // slots packed into one simdgroup, so shuffle by hand.
    #pragma unroll
    for (int r = 0; r < RB; r++) {
        #pragma unroll
        for (int v = 0; v < NV; v++) {
            if (KL >= 32) acc[r][v] += simd_shuffle_down(acc[r][v], 16);
            if (KL >= 16) acc[r][v] += simd_shuffle_down(acc[r][v], 8);
            if (KL >= 8)  acc[r][v] += simd_shuffle_down(acc[r][v], 4);
            if (KL >= 4)  acc[r][v] += simd_shuffle_down(acc[r][v], 2);
            if (KL >= 2)  acc[r][v] += simd_shuffle_down(acc[r][v], 1);
        }
    }

    if (klane == 0) {
        #pragma unroll
        for (int r = 0; r < RB; r++) {
            if (row0 + r < N) {
                #pragma unroll
                for (int v = 0; v < NV; v++) {
                    y[(size_t)v * N + row0 + r] = static_cast<DT>(acc[r][v]);
                }
            }
        }
    }
"""

_row_kernel = mx.fast.metal_kernel(
    name="rowqmv",
    input_names=["x", "w", "scales", "biases"],
    output_names=["y"],
    source=ROW_SOURCE,
)


def rowqmv(x, w, scales, biases, rb=1):
    """x (T, K) fp16 times dequant(w) (N, K) transposed -> (T, N) fp16."""
    rows, K = x.shape
    N = scales.shape[0]
    per_tg = _rows_per_tg(rb)
    tiles = (N + per_tg - 1) // per_tg
    return _row_kernel(
        inputs=[x, w, scales, biases],
        template=[("DT", x.dtype), ("NV", rows), ("KL", K_LANES),
                  ("NSG", SIMDGROUPS), ("RB", rb)],
        grid=(32, SIMDGROUPS * tiles, 1),
        threadgroup=(32, SIMDGROUPS, 1),
        output_shapes=[(rows, N)],
        output_dtypes=[x.dtype],
    )[0]


def usable(x, module):
    """True when the kernel's contract holds for this call."""
    return (
        module.mode == "affine"
        and module.group_size == GROUP_SIZE
        and module.bits == BITS
        and x.dtype in (mx.float16, mx.bfloat16)
        and x.shape[-1] % GROUP_SIZE == 0
    )


def _pick(layout, x, module):
    """Per-shape route if calibration found one, else the per-row-count route."""
    rows = x.size // x.shape[-1]
    if not layout:
        return None
    return layout.get((rows, x.shape[-1], module["scales"].shape[0]),
                      layout.get(rows))


def _apply(x, module, cand):
    flat = x.reshape(-1, x.shape[-1])
    if callable(cand):
        y = cand(flat, module)
    else:
        y = rowqmv(flat, module["weight"], module["scales"], module["biases"], cand)
    return y.reshape(*x.shape[:-1], y.shape[-1])


class RowQuantizedLinear(nn.QuantizedLinear):
    layout = {}  # rows -> row block, only for the rows where we win

    def __call__(self, x):
        cand = _pick(self.layout, x, self)
        if cand is None or not usable(x, self):
            return super().__call__(x)
        x = _apply(x, self, cand)
        if "bias" in self:
            x = x + self["bias"]
        return x


class RowQuantizedEmbedding(nn.QuantizedEmbedding):
    layout = {}

    def as_linear(self, x):
        cand = _pick(self.layout, x, self)
        if cand is None or not usable(x, self):
            return super().as_linear(x)
        return _apply(x, self, cand)


def _targets(model):
    """Every affine 4-bit projection in the model, paired with its wrapper."""
    out = []
    for m in model.modules():
        if isinstance(m, nn.QuantizedLinear):
            wrap = RowQuantizedLinear
        elif isinstance(m, nn.QuantizedEmbedding):
            wrap = RowQuantizedEmbedding
        else:
            continue
        if m.mode == "affine" and m.group_size == GROUP_SIZE and m.bits == BITS:
            out.append((m, wrap))
    return out


def _label(cand):
    return getattr(cand, "label", None) or "rb%d" % cand


def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def _time(fn, reps=10, warmup=3):
    for _ in range(warmup):
        mx.eval(fn())
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append((time.perf_counter() - t0) * 1000)
    return _median(ts)


def _set_layout(layout):
    # Known limit: class-level, so one process drives one model, since
    # nn.Module's __setattr__ does not take a plain dict on an instance.
    RowQuantizedLinear.layout = layout
    RowQuantizedEmbedding.layout = layout


# Calibrated layouts, one file per (device, mlx version, this file, the
# model's projection shapes): calibration times the model's own forward and
# takes ~100 s on a 4B, too long to pay on every load.
LAYOUT_DIR = Path.home() / ".cache" / "moe-ssd-offload"


def _layout_path(model):
    shapes = sorted({(m["weight"].shape[1] * 32 // BITS, m["scales"].shape[0])
                     for m, _ in _targets(model)})
    key = json.dumps([mx.device_info()["device_name"], mx.__version__,
                      hashlib.sha1(Path(__file__).read_bytes()).hexdigest(),
                      shapes])
    return LAYOUT_DIR / f"rowqmm-{hashlib.sha1(key.encode()).hexdigest()[:16]}.json"


def _load_layout(path):
    """The saved layout, or None if there is none or a label no longer names
    a candidate."""
    try:
        saved = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    by_label = {_label(c): c for c in CANDIDATES}
    if not all(lbl in by_label for lbl in saved.values()):
        return None
    return {tuple(int(x) for x in k.split(",")): by_label[lbl]
            for k, lbl in saved.items()}


def _save_layout(path, layout):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({",".join(map(str, k)): _label(c)
                               for k, c in layout.items()}))
    tmp.replace(path)


def install(model, layout=None, verbose=False):
    """Route every quantized projection through the kernel for the row counts
    calibration picked. Returns the {rows: row_block} map that was installed.
    Without `layout`, reuses this machine's saved calibration for the model's
    shapes, or calibrates and saves one; prints which."""
    if layout is None:
        path = _layout_path(model)
        layout = _load_layout(path)
        if layout is not None:
            note(f"  rowqmm: calibrated layout reused from {path}")
        else:
            t0 = time.perf_counter()
            layout = calibrate(model, verbose=verbose)
            _save_layout(path, layout)
            note(f"  rowqmm: calibrated in {time.perf_counter() - t0:.0f} s, "
                  f"saved to {path}")
    for m, wrap in _targets(model):
        m.__class__ = wrap
    _set_layout(layout)
    # The rows one qmm8 pass covers, for a caller sizing a verify pass (past
    # them every projection falls back to stock), set only when calibration
    # actually routed something through qmm8.
    if any(callable(c) for c in layout.values()):
        model.rowqmm_rows = ROWS
    return layout


def uninstall(model):
    _set_layout({})


# --------------------------------------------------------------------------
# selftest


def _layout_check():
    """Confirm nibble order and per-group scale/bias empirically."""
    a = mx.arange(GROUP_SIZE * 2, dtype=mx.float32).reshape(2, GROUP_SIZE) / 7.0
    w, s, b = mx.quantize(a, group_size=GROUP_SIZE, bits=BITS)
    assert w.shape == (2, GROUP_SIZE // 8) and w.dtype == mx.uint32, w.shape
    assert s.shape == (2, 1) and b.shape == (2, 1), (s.shape, b.shape)
    word = int(w[0, 0].item())
    nibbles = mx.array([(word >> (4 * i)) & 0xF for i in range(8)], dtype=mx.float32)
    manual = nibbles * s[0, 0].astype(mx.float32) + b[0, 0].astype(mx.float32)
    ref = mx.dequantize(w, s.astype(mx.float32), b.astype(mx.float32),
                        group_size=GROUP_SIZE, bits=BITS)[0, :8]
    # A swapped nibble order would be off by a whole quantization step; the
    # residual here is just fma-vs-mul-add rounding.
    err = mx.max(mx.abs(manual - ref)).item()
    assert err < 0.01 * abs(float(s[0, 0])), (err, manual, ref)
    print("layout: uint32 holds 8 values, element i in bits 4i..4i+3; "
          "one fp16 scale+bias per 64 values along K")


def _warmup(rows, dtype=mx.bfloat16):
    """Both paths jit-compile a variant per row count; do that off the clock."""
    w, sc, b = mx.quantize(mx.random.normal((64, 64)).astype(dtype),
                           group_size=GROUP_SIZE, bits=BITS)
    for n in rows:
        x = mx.random.normal((n, 64)).astype(dtype)
        mx.eval(*[_apply(x, {"weight": w, "scales": sc, "biases": b}, c)
                  for c in CANDIDATES],
                mx.quantized_matmul(x, w, sc, b, transpose=True,
                                    group_size=GROUP_SIZE, bits=BITS))
    mx.clear_cache()


def _numerics(shapes, rows, dtype=mx.bfloat16):
    _warmup(rows, dtype)
    # max rel is the max absolute error over the peak |output| of that matrix;
    # dividing elementwise is meaningless when logits cross zero.
    print(f"\n{'K':>6} {'N':>7} {'T':>3} {'max abs':>10} {'max rel':>10} "
          f"{'mlx abs':>10} {'ours ms':>9} {'mlx ms':>9} {'speedup':>8}  layout")
    worst = 0.0
    for K, N in shapes:
        w, s, b = mx.quantize(
            mx.random.normal((N, K)).astype(dtype), group_size=GROUP_SIZE, bits=BITS)
        mx.eval(w, s, b)
        mx.clear_cache()
        # fp32 reference: dequantize with fp32 scales so the yardstick itself
        # is not rounded to fp16.
        ref_w = mx.dequantize(w, s.astype(mx.float32), b.astype(mx.float32),
                              group_size=GROUP_SIZE, bits=BITS)
        mx.eval(ref_w)
        for n in rows:
            x = mx.random.normal((n, K)).astype(dtype)
            mx.eval(x)
            ref = x.astype(mx.float32) @ ref_w.T
            rb = min(ROW_BLOCKS, key=lambda r: _time(lambda: rowqmv(x, w, s, b, r)))
            ours = rowqmv(x, w, s, b, rb)
            mlx = mx.quantized_matmul(x, w, s, b, transpose=True, group_size=GROUP_SIZE, bits=BITS)
            mx.eval(ref, ours, mlx)
            peak = mx.max(mx.abs(ref)).item()
            aerr = mx.max(mx.abs(ours.astype(mx.float32) - ref)).item()
            rerr = aerr / peak
            merr = mx.max(mx.abs(mlx.astype(mx.float32) - ref)).item()
            t_ours = _time(lambda: rowqmv(x, w, s, b, rb))
            t_mlx = _time(
                lambda: mx.quantized_matmul(
                    x, w, s, b, transpose=True, group_size=GROUP_SIZE, bits=BITS
                )
            )
            worst = max(worst, aerr / max(merr, 1e-6))
            print(f"{K:>6} {N:>7} {n:>3} {aerr:>10.4f} {rerr:>10.6f} {merr:>10.4f} "
                  f"{t_ours:>9.3f} {t_mlx:>9.3f} {t_mlx / t_ours:>8.2f}x  rb{rb}", flush=True)
        del w, s, b, ref_w
        mx.clear_cache()
    # Products are fp16 where mlx's qmv widens first, so we cost a little
    # accuracy at T>=2 and gain some at T=1; both stay far inside one fp16 ulp
    # chain. Guard against a real regression, not against the last bit.
    assert worst < 4.0, f"error {worst:.2f}x worse than mlx"


# --------------------------------------------------------------------------
# synthetic tokens for calibration and end-to-end timing (no real text: only
# the model's own shapes and a warm KV cache matter here)

_ID_KEY = mx.random.key(0)  # fixed seed so calibration reproduces run to run


def _vocab_size(model):
    """Largest output width among the model's quantized projections -- the
    embedding and lm_head both have N == vocab size, and vocab is always the
    biggest projection in practice."""
    return max(m["scales"].shape[0] for m, _ in _targets(model))


def _new_ids(n, vocab):
    return mx.random.randint(0, vocab, (1, n), key=_ID_KEY)


def _prefill(model, vocab):
    """Warm a KV cache with synthetic tokens. GROUP_SIZE is just reused as a
    convenient nonzero prompt length, not because its value matters here."""
    from mlx_lm.models.cache import make_prompt_cache

    ids = _new_ids(GROUP_SIZE, vocab)
    cache = make_prompt_cache(model)
    mx.eval(model(ids, cache=cache))
    return cache, GROUP_SIZE


def _forward_ms(model, cache, n, vocab, reps=20, warmup=3):
    from mlx_lm.models.cache import trim_prompt_cache

    ids = _new_ids(n, vocab)

    def step():
        mx.eval(model(ids, cache=cache))

    for _ in range(warmup):
        step()
        trim_prompt_cache(cache, n)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        step()
        ts.append((time.perf_counter() - t0) * 1000)
        trim_prompt_cache(cache, n)
    return _median(ts)


def _greedy(model, steps, vocab, forced=None):
    """Greedy argmax for `steps` decode steps. With `forced`, the input at each
    step is taken from that sequence instead of our own prediction, so two runs
    see identical hidden states and a disagreement means the kernel actually
    changed the argmax rather than the run having diverged earlier."""
    cache, _ = _prefill(model, vocab)
    tokid = _new_ids(1, vocab)
    out = []
    for i in range(steps):
        logits = model(tokid, cache=cache)
        pred = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        out.append(pred)
        tokid = mx.array([[pred if forced is None else forced[i]]])
    return out


MM8_HEADER = "#include <metal_simdgroup_matrix>\n"

MM8_SOURCE = """
    constexpr short NACC = BN / 8;   // 8x8 accumulators per simdgroup
    constexpr short WPL = BN / 4;    // packed words this lane decodes per group
    constexpr short LPR = 8 / WPL;   // lanes cooperating on one output row

    const int K = x_shape[1];
    const int N = scales_shape[0];
    const int Kg = scales_shape[1];

    const ushort lane = thread_index_in_simdgroup;
    const ushort sg = simdgroup_index_in_threadgroup;
    const int col0 = (int(threadgroup_position_in_grid.y) * NSG + sg) * BN;

    // TS is a multiple of 4 so every tile row is 8-byte aligned for the vec4
    // stores; the pad keeps consecutive lanes off the same bank.
    threadgroup vec<MT, 4> tilev[NSG * BN * TS / 4];
    // XSH stages x's 8 x 64 block once per threadgroup instead of letting every
    // simdgroup re-read it from device. Without it each simdgroup streams the
    // whole of x, which is 32/BN times the weight bytes -- 2x at BN=16.
    threadgroup vec<MT, 4> xtilev[XSH ? 8 * TS / 4 : 1];
    threadgroup float ostore[NSG * 64];
    threadgroup MT* tile = ((threadgroup MT*)tilev) + sg * (BN * TS);
    threadgroup float* myout = ostore + sg * 64;
    threadgroup MT* xtile = (threadgroup MT*)xtilev;
    const ushort tid = sg * 32 + lane;

    const uint magic2 = (uint(MAGIC) << 16) | uint(MAGIC);

    simdgroup_matrix<float, 8, 8> acc[NACC];
    #pragma unroll
    for (short j = 0; j < NACC; j++) {
        acc[j] = simdgroup_matrix<float, 8, 8>(0);
    }

    // BN below 32 splits an output row across LPR lanes, which is what buys
    // occupancy on small N: the whole matmul launches 32*N/BN threads.
    const short row_l = lane / LPR;
    const short woff = (lane % LPR) * WPL;
    const int row = min(col0 + row_l, N - 1);

    // One group of weights per lane lives in registers; with PRE the loads for
    // group g+1 are issued before the MMAs of group g, so the DRAM round trip
    // overlaps the matrix work instead of stalling the top of the next group.
    uint wv[8];
    float scr, bir;

#define LOADW(gg) { \
        const device uint* wp = \
            w + (size_t)row * (Kg * 8) + (size_t)(gg) * 8 + woff; \
        if (WPL >= 4) { \
            _Pragma("unroll") \
            for (short i = 0; i < WPL; i += 4) { \
                const uint4 t = *(const device uint4*)(wp + i); \
                wv[i] = t.x; wv[i + 1] = t.y; wv[i + 2] = t.z; wv[i + 3] = t.w; \
            } \
        } else { \
            const uint2 t = *(const device uint2*)wp; \
            wv[0] = t.x; wv[1] = t.y; \
        } \
        scr = static_cast<float>(scales[(size_t)row * Kg + (gg)]); \
        bir = fma(-float(OFFSET), scr, \
                  static_cast<float>(biases[(size_t)row * Kg + (gg)])); \
    }

    if (PRE) LOADW(0)

    for (int g = 0; g < Kg; g++) {
        if (!PRE) LOADW(g)
        if (XSH) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        } else {
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        {
            const vec<AT, 4> scv = vec<AT, 4>(static_cast<AT>(scr));
            const vec<AT, 4> biv = vec<AT, 4>(static_cast<AT>(bir));
            threadgroup vec<MT, 4>* dst =
                (threadgroup vec<MT, 4>*)(tile + row_l * TS + woff * 8);
            #pragma unroll
            for (short c = 0; c < WPL; c++) {
                const uint word = wv[c];
                // each mask yields nibble i and nibble i+4 already shaped as
                // two MT values sharing one 32-bit register
                const vec<MT, 2> v0 = as_type<vec<MT, 2>>((word & 0x000F000F) | magic2);
                const vec<MT, 2> v1 = as_type<vec<MT, 2>>(((word >> 4) & 0x000F000F) | magic2);
                const vec<MT, 2> v2 = as_type<vec<MT, 2>>(((word >> 8) & 0x000F000F) | magic2);
                const vec<MT, 2> v3 = as_type<vec<MT, 2>>(((word >> 12) & 0x000F000F) | magic2);
                const vec<MT, 4> lo = {v0.x, v1.x, v2.x, v3.x};
                const vec<MT, 4> hi = {v0.y, v1.y, v2.y, v3.y};
                dst[c * 2] = vec<MT, 4>(fma(vec<AT, 4>(lo), scv, biv));
                dst[c * 2 + 1] = vec<MT, 4>(fma(vec<AT, 4>(hi), scv, biv));
            }
        }
        if (XSH) {
            #pragma unroll
            for (short u = tid; u < 128; u += NSG * 32) {
                const short t = u >> 4, col = (u & 15) * 4;
                xtilev[t * (TS / 4) + (col >> 2)] =
                    *(const device vec<MT, 4>*)(x + (size_t)t * K + g * 64 + col);
            }
        }
        if (PRE && g + 1 < Kg) LOADW(g + 1)
        if (XSH) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        } else {
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }

        {
            const device MT* xg = x + g * 64;
            #pragma unroll
            for (short kk = 0; kk < 8; kk++) {
                simdgroup_matrix<MT, 8, 8> P;
                // SWAP picks which operand gets transposed on load. x is 8 x K
                // and read straight from device; the tile is BN x 64 in
                // threadgroup memory and read NACC times as often, so it is the
                // one worth keeping untransposed.
                if (XSH) {
                    simdgroup_load(P, xtile + kk * 8, TS, ulong2(0, 0), SWAP);
                } else {
                    simdgroup_load(P, xg + kk * 8, K, ulong2(0, 0), SWAP);
                }
                #pragma unroll
                for (short j = 0; j < NACC; j++) {
                    simdgroup_matrix<MT, 8, 8> Q;
                    simdgroup_load(Q, tile + j * 8 * TS + kk * 8, TS, ulong2(0, 0), !SWAP);
                    if (SWAP) {
                        simdgroup_multiply_accumulate(acc[j], Q, P, acc[j]);
                    } else {
                        simdgroup_multiply_accumulate(acc[j], P, Q, acc[j]);
                    }
                }
            }
        }
    }

    #pragma unroll
    for (short j = 0; j < NACC; j++) {
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_store(acc[j], myout, 8, ulong2(0, 0), SWAP);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (short i = lane; i < 64; i += 32) {
            const int n = col0 + j * 8 + (i & 7);
            if (n < N) {
                y[(size_t)(i >> 3) * N + n] = static_cast<OT>(myout[i]);
            }
        }
    }
"""

_mm8_kernel = mx.fast.metal_kernel(
    name="rowqmm8",
    input_names=["x", "w", "scales", "biases"],
    output_names=["y"],
    source=MM8_SOURCE,
    header=MM8_HEADER,
)

ROWS = 8
TS = 68
# magic constant and the integer it encodes: q | MAGIC is exactly OFFSET + q
MAGIC = {mx.bfloat16: (0x4300, 128), mx.float16: (0x6400, 1024)}

# (compute dtype, output cols/simdgroup, simdgroups/threadgroup, swap = W is
# the left MMA operand, pre = prefetch next group's weights, xsh = stage x in
# threadgroup memory), with BN=16 winning across the model's real shapes on
# this GPU.
CFGS = [(mx.bfloat16, 16, 2, True, False, False),
        (mx.bfloat16, 16, 8, True, False, False)]


def qmm8(x, w, scales, biases, cfg):
    """x (T<=8, K) times dequant(w) (N, K) transposed -> (T, N), x's dtype."""
    mt, bn, nsg, swap, pre, xsh = cfg
    T, K = x.shape
    N = scales.shape[0]
    out_dtype = x.dtype
    if T != ROWS:
        x = mx.concatenate([x, mx.zeros((ROWS - T, K), x.dtype)], axis=0)
    if mt != x.dtype:
        x = x.astype(mt)
    magic, offset = MAGIC[mt]
    # scale/bias always in fp32: the magic constant puts the nibble 1024 (half)
    # or 128 (bfloat) up the number line, and undoing that in the tile type
    # itself cancels away the low bits -- 8% error for half, measured.
    at = mx.float32
    per_tg = bn * nsg
    y = _mm8_kernel(
        inputs=[x, w, scales, biases],
        template=[("MT", mt), ("AT", at), ("OT", out_dtype), ("MAGIC", magic),
                  ("OFFSET", offset), ("BN", bn), ("NSG", nsg), ("TS", TS), ("SWAP", swap),
                  ("PRE", pre), ("XSH", xsh)],
        grid=(32, nsg * ((N + per_tg - 1) // per_tg), 1),
        threadgroup=(32, nsg, 1),
        output_shapes=[(ROWS, N)],
        output_dtypes=[out_dtype],
    )[0]
    return y if T == ROWS else y[:T]


def _make_apply(cfg):
    def apply(flat, module):
        return qmm8(flat, module["weight"], module["scales"], module["biases"], cfg)

    apply.label = "mm8-%s%d.%d%s%s%s" % (
        "bf" if cfg[0] == mx.bfloat16 else "f16", cfg[1], cfg[2],
        "w" if cfg[3] else "x", "p" if cfg[4] else "", "s" if cfg[5] else "")
    return apply


MM8_CANDS = [_make_apply(c) for c in CFGS]
# what calibrate() chooses between: stock (None), the scalar row blocks, qmm8
CANDIDATES = list(ROW_BLOCKS) + MM8_CANDS


# --------------------------------------------------------------------------
# selftest (mm8)

SHAPES = [(2560, 9728), (9728, 2560), (2560, 4096),
          (2560, 1024), (4096, 2560), (2560, 151936)]


def _mod(w, s, b):
    return {"weight": w, "scales": s, "biases": b}


def numerics(rows=(8,), cands=None, dtype=mx.bfloat16):
    """Per-op error and ms against mlx and against the scalar row kernel."""
    cands = cands or MM8_CANDS
    print(f"\n{'K':>6} {'N':>7} {'T':>3} {'cand':>12} {'max abs':>9} {'max rel':>9} "
          f"{'mlx abs':>9} {'ours ms':>9} {'mlx ms':>8} {'scalar ms':>10} {'vs mlx':>7}")
    worst = 0.0
    for K, N in SHAPES:
        w, s, b = mx.quantize(mx.random.normal((N, K)).astype(dtype),
                              group_size=GROUP_SIZE, bits=BITS)
        mx.eval(w, s, b)
        mx.clear_cache()
        ref_w = mx.dequantize(w, s.astype(mx.float32), b.astype(mx.float32),
                              group_size=GROUP_SIZE, bits=BITS)
        mx.eval(ref_w)
        for n in rows:
            x = mx.random.normal((n, K)).astype(dtype)
            mx.eval(x)
            ref = x.astype(mx.float32) @ ref_w.T
            mlx = mx.quantized_matmul(x, w, s, b, transpose=True,
                                      group_size=GROUP_SIZE, bits=BITS)
            mx.eval(ref, mlx)
            peak = mx.max(mx.abs(ref)).item()
            merr = mx.max(mx.abs(mlx.astype(mx.float32) - ref)).item()
            t_mlx = _time(lambda: mx.quantized_matmul(
                x, w, s, b, transpose=True, group_size=GROUP_SIZE, bits=BITS))
            t_sc = _time(lambda: rowqmv(x, w, s, b, 1))
            for f in cands:
                ours = f(x, _mod(w, s, b))
                mx.eval(ours)
                aerr = mx.max(mx.abs(ours.astype(mx.float32) - ref)).item()
                t = _time(lambda: f(x, _mod(w, s, b)))
                worst = max(worst, aerr / max(merr, 1e-6))
                print(f"{K:>6} {N:>7} {n:>3} {f.label:>12} {aerr:>9.4f} "
                      f"{aerr / peak:>9.6f} {merr:>9.4f} {t:>9.3f} {t_mlx:>8.3f} "
                      f"{t_sc:>10.3f} {t_mlx / t:>6.2f}x", flush=True)
        del w, s, b, ref_w
        mx.clear_cache()
    print(f"worst error ratio vs mlx: {worst:.2f}x")
    # mlx widens every product to fp32; our tile is bf16, so we carry one extra
    # rounding per weight and land ~1.5x its absolute error, still ~3e-3
    # relative, i.e. the same bf16 ulp. Guard the level, not the last bit.
    assert worst < 2.5, f"error {worst:.2f}x worse than mlx"
    return worst


def _forced(model, vocab, seq, n):
    """Argmax at all 64 positions, fed n tokens per forward from `seq`, so two
    runs see identical hidden states and any disagreement is the kernel's."""
    cache, _ = _prefill(model, vocab)
    first = int(_new_ids(1, vocab).item())
    inp = [first] + list(seq[:-1])
    out = []
    for i in range(0, len(inp), n):
        logits = model(mx.array([inp[i:i + n]]), cache=cache)
        out += mx.argmax(logits[0], axis=-1).tolist()
    return out


def endtoend(path, rows=(1, 2, 3, 4, 5, 6, 7, 8)):
    from mlx_lm import load

    model, _ = load(path)
    vocab = _vocab_size(model)
    _warmup(rows)
    # stock greedy continuation on synthetic tokens, then the stock argmax at
    # every position when that same sequence is fed back 1 and 8 tokens at a time
    base = _greedy(model, 64, vocab)
    ref = {n: _forced(model, vocab, base, n) for n in (1, 8)}

    cache, plen = _prefill(model, vocab)
    print(f"\nend-to-end on {path} (synthetic prompt, {plen} tokens)")

    print("calibrating on the model's own shapes:")
    fast = install(model, verbose=True)

    # stock and routed are timed back to back for each row count: the two
    # passes are minutes apart otherwise and this machine drifts several
    # percent over that, which is the same size as some of the wins
    print(f"\n{'T':>3} {'stock ms':>9} {'ours ms':>9} {'speedup':>8} {'ms/token':>9}  routed")
    before, after = {}, {}
    for n in rows:
        _set_layout({})
        before[n] = _forward_ms(model, cache, n, vocab)
        _set_layout(fast)
        after[n] = _forward_ms(model, cache, n, vocab)
        paths = sorted({_label(v) for k, v in fast.items()
                        if isinstance(k, tuple) and k[0] == n})
        tag = ("+".join(paths) + " x%d shapes" % sum(1 for k in fast if k[0] == n)
               ) if paths else "stock"
        print(f"{n:>3} {before[n]:>9.2f} {after[n]:>9.2f} "
              f"{before[n] / after[n]:>7.2f}x {after[n] / n:>9.2f}  {tag}", flush=True)

    for n in (1, 8):
        install(model, layout={n: MM8_CANDS[0]})
        diff = sum(a != b for a, b in zip(ref[n], _forced(model, vocab, base, n)))
        print(f"teacher-forced argmax vs stock, 64 steps at T={n}: {diff} differing "
              f"(mm8 forced on)")
    install(model, layout=fast)
    return before, after, fast


def shapes_of(model):
    """(K, N) of every quantized projection the model actually runs."""
    seen = {}
    for m, _ in _targets(model):
        seen[(m["scales"].shape[1] * GROUP_SIZE, m["scales"].shape[0])] = None
    return sorted(seen)


def per_shape(model, n, cands, layout=None, verbose=True, margin=0.01):
    """Greedy per-shape route, timed on the model's own forward.

    Isolated evals of a 1.5-14 MB matmul sit on the launch floor, so the only
    honest measurement of a small shape is in the graph it runs in. One shape
    is fixed at a time and the next shape is measured on top of that choice, so
    interactions between them are in the number too.
    """
    ids = _new_ids(n, _vocab_size(model))
    layout = dict(layout or {})
    _set_layout(layout)
    base = _time(lambda: model(ids))
    if verbose:
        print(f"T={n}: stock forward {base:.2f} ms", flush=True)
    for K, N in shapes_of(model):
        key = (n, K, N)
        cost = {None: base}
        for c in cands:
            layout[key] = c
            _set_layout(layout)
            cost[c] = _time(lambda: model(ids))
        layout.pop(key, None)
        best = min(cost, key=cost.get)
        # a route has to beat stock by more than the run-to-run spread of the
        # forward itself, or calibration locks in noise and the "fast" path
        # measures slower than stock when it is re-timed
        if best is not None and cost[best] > cost[None] * (1.0 - margin):
            best = None
        if best is not None:
            layout[key] = best
        _set_layout(layout)
        if verbose:
            parts = " ".join(f"{_label(c)}={cost[c]:6.2f}" for c in cands)
            print(f"  {K:>6}->{N:<7} stock={base:6.2f} {parts} -> "
                  f"{_label(best) if best is not None else 'stock'} "
                  f"({base - cost[best]:+.2f} ms)", flush=True)
        base = cost[best]
    return layout, base


# --------------------------------------------------------------------------
# calibration over both kernels, per row count and per shape


def calibrate(model, rows=range(1, 9), cands=None, verbose=False):
    """Time the model's own forward and return {(rows, K, N): candidate};
    per_shape does the actual per-row-count, per-shape timing."""
    for m, wrap in _targets(model):
        m.__class__ = wrap
    saved = RowQuantizedLinear.layout

    layout = {}
    for n in rows:
        part, total = per_shape(model, n, cands or CANDIDATES, verbose=verbose)
        layout.update(part)
        if verbose:
            print(f"  T={n}: {total:.2f} ms after routing", flush=True)

    _set_layout(saved)
    return layout


def selftest():
    _layout_check()
    _numerics(SHAPES, [1, 8])
    numerics(rows=(8,))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None)
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()
    if args.selftest:
        selftest()
    if args.model:
        endtoend(args.model)
