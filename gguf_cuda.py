"""gguf_experts' kernels on CUDA: the backend prelude of their shared source."""
import mlx.core as mx

# NVRTC contracts a * b + c into one FMA by default, so the dequant's products and sums are rounded alone.
PRELUDE = r"""
#define GPTR(T) const T*
#define LOCAL
#define FN __device__ __forceinline__
#define CFN __host__ __device__ constexpr
#define H2F(u) __half2float(__ushort_as_half((unsigned short)(u)))
#define SHFL(v, m) __shfl_xor_sync(0xffffffffu, v, m)
#define POPC(v) __popc(v)
#define EXP(v) expf(v)
#define MUL(a, b) __fmul_rn(a, b)
#define ADD(a, b) __fadd_rn(a, b)
#define SUB(a, b) __fsub_rn(a, b)
#define U4(a, b, c, d) make_uint4(a, b, c, d)
#define TGMEM __shared__
#define BARRIER __syncthreads()
"""

BODY = r"""
const uint32_t _tid = threadIdx.x;
const uint3 _bid = blockIdx;
const uint32_t _gid = blockIdx.x * blockDim.x + threadIdx.x;
"""


# The decode chain on rows sorted by slot (CUDA only, gguf_experts._gemv_runs): a threadgroup stages TR rows and decodes
# a weight task once a run of equal slots, each row on the chain's lanes, task order, FMA order and shuffle tree.
RUNS_HEADER = r"""
// A staged row: 32-value chunks of 16-byte segments, swizzled so a quarter warp's chunks hit distinct banks.
template <typename TX> FN uint32_t gr_seg(uint32_t c, uint32_t s) {
  constexpr uint32_t SP = 2u * sizeof(TX);
  return c * SP + (s ^ ((c / (8u / SP)) % SP));
}
FN void gr_cvt(uint4 u, float *v, const __nv_bfloat16 *) {
  const uint32_t a[4] = {u.x, u.y, u.z, u.w};
  for (uint32_t i = 0u; i < 4u; i++) {
    v[2u * i] = __uint_as_float(a[i] << 16u);
    v[2u * i + 1u] = __uint_as_float(a[i] & 0xFFFF0000u);
  }
}
FN void gr_cvt(uint4 u, float *v, const __half *) {
  const uint32_t a[4] = {u.x, u.y, u.z, u.w};
  for (uint32_t i = 0u; i < 4u; i++) {
    v[2u * i] = H2F(a[i] & 0xFFFFu);
    v[2u * i + 1u] = H2F(a[i] >> 16u);
  }
}
FN void gr_cvt(uint4 u, float *v, const float *) {
  v[0] = __uint_as_float(u.x); v[1] = __uint_as_float(u.y); v[2] = __uint_as_float(u.z); v[3] = __uint_as_float(u.w);
}
// Values e..e+15 of a staged row (e a multiple of 16) as floats.
template <typename TX> FN void gr_x16(const TX *row, uint32_t e, float *v) {
  constexpr uint32_t V = 16u / sizeof(TX);
  const uint4 *p = (const uint4 *)row;
#pragma unroll
  for (uint32_t q = 0u; q < 16u / V; q++) gr_cvt(p[gr_seg<TX>(e >> 5u, (e & 31u) / V + q)], v + q * V, row);
}
// Rows e0.. of x (one a slot entry) into shared memory and their slots into sl; returns how many there are.
template <typename TX, int IN, int TR, int NW>
FN uint32_t gr_stage(const TX *x, const uint32_t *slots, uint32_t n, uint32_t e0, TX *xs, uint32_t (&sl)[TR]) {
  constexpr uint32_t G = IN * sizeof(TX) / 16u, SP = 2u * sizeof(TX);
  const uint32_t ne = n - e0 < uint32_t(TR) ? n - e0 : uint32_t(TR);
  for (uint32_t i = threadIdx.x; i < ne * G; i += 32u * NW) {
    const uint32_t r = i / G, g = i % G;
    ((uint4 *)xs)[r * G + gr_seg<TX>(g / SP, g % SP)] = ((const uint4 *)(x + (size_t)(e0 + r) * IN))[g];
  }
#pragma unroll
  for (uint32_t r = 0u; r < uint32_t(TR); r++) sl[r] = slots[e0 + (r < ne ? r : 0u)];
  __syncthreads();
  return ne;
}
// One projection over a window: acc[r][j] += w . x on the lane's tasks, a task decoded once a run of equal slots.
template <int T, int IN, int OUT, int LPR, int RPW, int TR, typename TX>
FN void gr_rows(const uint8_t *wb, const TX *xs, const uint32_t (&sl)[TR], uint32_t ne, uint32_t r0, uint32_t sub,
                float (&acc)[TR][RPW], const uint32_t *tab) {
  float w[RPW][32], xv[16];
  for (uint32_t k = sub; k < uint32_t(IN) / 32u; k += LPR) {
    uint32_t i0 = 0u;
#pragma unroll
    for (uint32_t r = 0u; r < uint32_t(TR); r++) {
      if (r >= ne) break;
      if (r == 0u || sl[r] != sl[r - 1u]) {
#pragma unroll
        for (uint32_t j = 0u; j < uint32_t(RPW); j++)
          if (r0 + j < uint32_t(OUT)) i0 = gg_task32<T>(wb + ((size_t)sl[r] * OUT + r0 + j) * gg_row(T, IN), k, w[j], tab);
      }
#pragma unroll
      for (uint32_t hf = 0u; hf < 2u; hf++) {
        gr_x16(xs + r * IN, i0 + hf * gg_run32(T), xv);
#pragma unroll
        for (uint32_t j = 0u; j < uint32_t(RPW); j++)
          if (r0 + j < uint32_t(OUT))
#pragma unroll
            for (uint32_t t = 0u; t < 16u; t++) acc[r][j] += w[j][16u * hf + t] * xv[t];
      }
    }
  }
}
"""

# Gate and up of a window of TR rows: h = SwiGLU, f32.
GATE_UP_RUNS_SRC = r"""
__shared__ __align__(16) TX xs[TR * IN];
const uint32_t lane = _tid & 31u, warp = _tid >> 5u, sub = lane & (LPR - 1u);
const uint32_t r0 = ((_bid.x * NW + warp) * (32u / LPR) + lane / LPR) * RPW, e0 = _bid.y * TR;
uint32_t sl[TR];
const uint32_t ne = gr_stage<TX, IN, TR, NW>(x, slots, uint32_t(slots_shape[0]), e0, xs, sl);
float ag[TR][RPW], au[TR][RPW];
#pragma unroll
for (uint32_t r = 0u; r < TR; r++)
#pragma unroll
  for (uint32_t j = 0u; j < RPW; j++) ag[r][j] = au[r][j] = 0.0f;
gr_rows<TG, IN, OUT, LPR, RPW, TR>(wg, xs, sl, ne, r0, sub, ag, tab);
gr_rows<TU, IN, OUT, LPR, RPW, TR>(wu, xs, sl, ne, r0, sub, au, tab);
#pragma unroll
for (uint32_t r = 0u; r < TR; r++)
#pragma unroll
  for (uint32_t j = 0u; j < RPW; j++) {
    for (uint32_t m = LPR / 2u; m > 0u; m >>= 1u) {
      ag[r][j] += SHFL(ag[r][j], m);
      au[r][j] += SHFL(au[r][j], m);
    }
    if (sub == 0u && r < ne && r0 + j < OUT) h[(size_t)(e0 + r) * OUT + r0 + j] = gg_silu_mul(ag[r][j], au[r][j]);
  }
"""

# Down of a window of TR rows, out in the output dtype, unweighted.
DOWN_RUNS_SRC = r"""
__shared__ __align__(16) float hs[TR * IN];
const uint32_t lane = _tid & 31u, warp = _tid >> 5u, sub = lane & (LPR - 1u);
const uint32_t r0 = ((_bid.x * NW + warp) * (32u / LPR) + lane / LPR) * RPW, e0 = _bid.y * TR;
uint32_t sl[TR];
const uint32_t ne = gr_stage<float, IN, TR, NW>(h, slots, uint32_t(slots_shape[0]), e0, hs, sl);
float a[TR][RPW];
#pragma unroll
for (uint32_t r = 0u; r < TR; r++)
#pragma unroll
  for (uint32_t j = 0u; j < RPW; j++) a[r][j] = 0.0f;
gr_rows<TD, IN, OUT, LPR, RPW, TR>(wd, hs, sl, ne, r0, sub, a, tab);
#pragma unroll
for (uint32_t r = 0u; r < TR; r++)
#pragma unroll
  for (uint32_t j = 0u; j < RPW; j++) {
    for (uint32_t m = LPR / 2u; m > 0u; m >>= 1u) a[r][j] += SHFL(a[r][j], m);
    if (sub == 0u && r < ne && r0 + j < OUT) y[(size_t)(e0 + r) * OUT + r0 + j] = static_cast<TY>(a[r][j]);
  }
"""


def kernel(name, ins, outs, source, header):
    return mx.fast.cuda_kernel(name=name, input_names=ins, output_names=outs, source=BODY + source,
                               header=PRELUDE + header)
