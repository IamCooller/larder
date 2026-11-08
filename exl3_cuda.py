"""exl3_experts.py's kernels as CUDA sources, with the same names, inputs, outputs, template
arguments, grids and threadgroups, so only _kernel() picks the source."""
# The ports' licences are in THIRD_PARTY_NOTICES.md (ddalcu/sushi, turboderp-org/exllamav3, MIT);
# the port has a simdgroup for a warp, simd_shuffle_xor for __shfl_xor_sync, threadgroup memory for
# __shared__ and vectors as scalars in the same order.
import mlx.core as mx

HEADER = r"""
// One codeword to a float: MCG adds the hashed word's two halves in half; mul1 (WIN + 256) sums its four bytes into a half, then one half fma.
template <int WIN>
__device__ __forceinline__ float exl3_val(uint32_t cw) {
  if ((WIN & 255) < 16) cw &= (1u << (WIN & 15)) - 1u;
  if (WIN & 256) {
    const uint32_t x = cw * 0x83DCD12Du;
    const uint32_t s = (x & 255u) + ((x >> 8u) & 255u) + ((x >> 16u) & 255u) + (x >> 24u) + 0x6400u;
    return __half2float(__hfma(__ushort_as_half((unsigned short)s), __ushort_as_half((unsigned short)0x1EEEu),
                               __ushort_as_half((unsigned short)0xC931u)));
  }
  const uint32_t r =((cw * 0xCBAC1FEDu) & 0x8FFF8FFFu) ^ 0x3B603B60u;
  return __half2float(__hadd(__ushort_as_half((unsigned short)(r & 0xffffu)),
                             __ushort_as_half((unsigned short)(r >> 16))));
}
__host__ __device__ constexpr uint32_t exl3_end(uint32_t N, uint32_t j) { return ((j + 1u) * N) >> 4u; }
__device__ __forceinline__ uint64_t exl3_window(const uint32_t *words, uint32_t end, uint32_t nwords, bool full) {
  const uint32_t last = (end - 1u) >> 5u;
  const uint32_t prev = last == 0u ? nwords - 1u : last - 1u;
  const uint32_t s = (0u - end) & 31u;
  uint64_t bits = (((uint64_t)words[prev] << 32u) | (uint64_t)words[last]) >> s;
  if (full) bits |= ((uint64_t)words[prev == 0u ? nwords - 1u : prev - 1u] << 32u) << (32u - s);
  return bits;
}
__host__ __device__ constexpr uint32_t exl3_lane_sh(uint32_t N, uint32_t j) { return N / 2u - exl3_end(N, j); }
template <uint32_t N>
__device__ __forceinline__ uint64_t exl3_lane(const uint32_t *words, uint32_t lane) {
  constexpr uint32_t W = N / 2u;
  constexpr uint32_t low = W & (0u - W);
  return exl3_window(words, W * lane + W, W, exl3_lane_sh(N, 0u) + 16u > 32u + (low < 32u ? low : 32u));
}
__device__ __forceinline__ float exl3_shfl(float v, uint32_t bit) { return __shfl_xor_sync(0xffffffffu, v, bit); }
"""

# One 128-wide Hadamard of v0..v3 (lane holds elements lane, +32, +64, +96), in place.
_FWHT = r"""
  #pragma unroll
  for (uint32_t bit = 1u; bit <= 16u; bit <<= 1u) {
    const float p0 = exl3_shfl(v0, bit);
    const float p1 = exl3_shfl(v1, bit);
    const float p2 = exl3_shfl(v2, bit);
    const float p3 = exl3_shfl(v3, bit);
    const bool lower = (lane & bit) == 0u;
    v0 = lower ? v0 + p0 : p0 - v0;
    v1 = lower ? v1 + p1 : p1 - v1;
    v2 = lower ? v2 + p2 : p2 - v2;
    v3 = lower ? v3 + p3 : p3 - v3;
  }
"""

_TILES = r"""
const uint32_t prow = (lane & 3u) * 2u;
const uint32_t pcol = lane >> 2u;
const uint32_t row0 = prow;
const uint32_t row1 = prow + 1u;
const uint32_t row2 = prow + 8u;
const uint32_t row3 = prow + 9u;
float acc[OTPT][8];
#pragma unroll
for (int o = 0; o < OTPT; o++) {
  #pragma unroll
  for (int i = 0; i < 8; i++) acc[o][i] = 0.0f;
}
if (N == 64u) {
  for (uint32_t tk = tk0 + sg; tk < tk1; tk += SGS) {
    const uint32_t *words = trellis_e + ((size_t)tk * (size_t)OT + ot) * 32u;
    const uint64_t merged = ((uint64_t)words[(lane + 31u) & 31u] << 32) | (uint64_t)words[lane];
    const float in0 = __half2float(%(prep)s[(tk - %(pbase)s) * TILE + row0]);
    const float in1 = __half2float(%(prep)s[(tk - %(pbase)s) * TILE + row1]);
    const float in2 = __half2float(%(prep)s[(tk - %(pbase)s) * TILE + row2]);
    const float in3 = __half2float(%(prep)s[(tk - %(pbase)s) * TILE + row3]);
    const uint32_t sh[8] = {28u, 24u, 20u, 16u, 12u, 8u, 4u, 0u};
    const float ins[8] = {in0, in1, in2, in3, in0, in1, in2, in3};
    #pragma unroll
    for (uint32_t p = 0u; p < 8u; p++) {
      const float w = exl3_val<WIN>(uint32_t(merged >> sh[p]) & 0xffffu);
      acc[0][p] = fmaf(ins[p], w, acc[0][p]);
    }
  }
} else {
  // Both k-tiles' words load before either decodes; a k range is whole H128
  // blocks (8 tiles), so the second k-tile of an iteration is always in range.
  const uint32_t *wp = trellis_e + ((size_t)(tk0 + sg) * (size_t)OT + ot) * PACKED_W;
  const __half *pp = %(prep)s + (tk0 - %(pbase)s + sg) * TILE;
  const uint32_t sh[8] = {exl3_lane_sh(N, 0u), exl3_lane_sh(N, 1u), exl3_lane_sh(N, 2u), exl3_lane_sh(N, 3u),
                          exl3_lane_sh(N, 4u), exl3_lane_sh(N, 5u), exl3_lane_sh(N, 6u), exl3_lane_sh(N, 7u)};
  for (uint32_t tk = tk0 + sg; tk < tk1; tk += 2u * SGS) {
    uint64_t merged[2][OTPT];
    #pragma unroll
    for (uint32_t u = 0u; u < 2u; u++) {
      #pragma unroll
      for (uint32_t o = 0u; o < uint32_t(OTPT); o++) {
        merged[u][o] = exl3_lane<N>(wp + u * SGS * OT * PACKED_W + o * PACKED_W, lane);
      }
    }
    #pragma unroll
    for (uint32_t u = 0u; u < 2u; u++) {
      const float in0 = __half2float(pp[u * SGS * TILE + row0]);
      const float in1 = __half2float(pp[u * SGS * TILE + row1]);
      const float in2 = __half2float(pp[u * SGS * TILE + row2]);
      const float in3 = __half2float(pp[u * SGS * TILE + row3]);
      const float ins[8] = {in0, in1, in2, in3, in0, in1, in2, in3};
      #pragma unroll
      for (uint32_t o = 0u; o < uint32_t(OTPT); o++) {
        #pragma unroll
        for (uint32_t p = 0u; p < 8u; p++) {
          const float w = exl3_val<WIN>(uint32_t(merged[u][o] >> sh[p]) & 0xffffu);
          acc[o][p] = fmaf(ins[p], w, acc[o][p]);
        }
      }
    }
    wp += 2u * SGS * OT * PACKED_W;
    pp += 2u * SGS * TILE;
  }
}
#pragma unroll
for (uint32_t o = 0u; o < uint32_t(OTPT); o++) {
  float clo = (acc[o][0] + acc[o][1]) + (acc[o][2] + acc[o][3]);
  float chi = (acc[o][4] + acc[o][5]) + (acc[o][6] + acc[o][7]);
  clo += exl3_shfl(clo, 1u);
  chi += exl3_shfl(chi, 1u);
  clo += exl3_shfl(clo, 2u);
  chi += exl3_shfl(chi, 2u);
  if ((lane & 3u) == 0u) {
    partial[(o * SGS + sg) * 16u + pcol] = clo;
    partial[(o * SGS + sg) * 16u + pcol + 8u] = chi;
  }
}
__syncthreads();
"""


def _tiles(prep, pbase):
    """The shared tile loop, its inputs read from prep at tile pbase."""
    return _TILES % {"prep": prep, "pbase": pbase}


_GEMV_HEAD = r"""
__shared__ float partial[4 * 16 * OTPT];
const uint32_t ot = blockIdx.x * uint32_t(OTPT);
const uint32_t slot = blockIdx.y;
const uint32_t split = blockIdx.z;
const uint32_t lid = threadIdx.x;
const uint32_t sg = lid >> 5u;
const uint32_t lane = lid & 31u;
constexpr uint32_t TILE = 16u;
constexpr uint32_t N = uint32_t(NHW);
constexpr uint32_t PACKED_HW = N;
constexpr uint32_t PACKED_W = N / 2u;
constexpr uint32_t IT = uint32_t(IDIM) / TILE;
constexpr uint32_t OT = uint32_t(ODIM) / TILE;
constexpr uint32_t SGS = 4u;
const uint32_t eid = slots[slot];
const float sc = 0.08838834764831845f;
"""

PAIR_SRC = _GEMV_HEAD + r"""
const uint32_t tiles_per_split = (IT + uint32_t(NSPLIT) - 1u) / uint32_t(NSPLIT);
const uint32_t tk0 = split * tiles_per_split;
const uint32_t tk1 = min(tk0 + tiles_per_split, IT);
constexpr uint32_t KSPAN = uint32_t(IDIM) / uint32_t(NSPLIT);
__shared__ __half prepared[2u * KSPAN];
{
  const uint32_t xrow = slot / uint32_t(TOPK);
  for (uint32_t b = sg; b < KSPAN / 128u; b += SGS) {
    const uint32_t pbase = tk0 * TILE + b * 128u;
    const size_t xr = (size_t)xrow * (size_t)(IDIM) + pbase;
    const size_t sr = (size_t)eid * (size_t)(IDIM) + pbase;
    const float x0 = static_cast<float>(x[xr + lane]);
    const float x1 = static_cast<float>(x[xr + lane + 32u]);
    const float x2 = static_cast<float>(x[xr + lane + 64u]);
    const float x3 = static_cast<float>(x[xr + lane + 96u]);
    for (uint32_t pj = 0u; pj < 2u; pj++) {
      const __half *suh = (pj == 0u) ? suhg : suhu;
      float v0 = x0 * __half2float(suh[sr + lane]);
      float v1 = x1 * __half2float(suh[sr + lane + 32u]);
      float v2 = x2 * __half2float(suh[sr + lane + 64u]);
      float v3 = x3 * __half2float(suh[sr + lane + 96u]);
""" + _FWHT + r"""
      const float s0 = v0 + v1;
      const float s1 = v0 - v1;
      const float s2 = v2 + v3;
      const float s3 = v2 - v3;
      const uint32_t pw = pj * KSPAN + b * 128u;
      prepared[pw + lane] = __float2half((s0 + s2) * sc);
      prepared[pw + lane + 32u] = __float2half((s1 + s3) * sc);
      prepared[pw + lane + 64u] = __float2half((s0 - s2) * sc);
      prepared[pw + lane + 96u] = __float2half((s1 - s3) * sc);
    }
  }
}
__syncthreads();
for (uint32_t proj = 0u; proj < 2u; proj++) {
  const uint16_t *trellis = (proj == 0u) ? tg : tu;
  float *y = (proj == 0u) ? yg : yu;
  const uint32_t *trellis_e = reinterpret_cast<const uint32_t *>(trellis + ((size_t)eid * (size_t)IT * (size_t)OT) * PACKED_HW);
  const __half *PREP0 = prepared + proj * KSPAN;
""" + _tiles("PREP0", "tk0") + r"""
  if (lid < 16u * uint32_t(OTPT)) {
    const uint32_t o = lid >> 4u;
    float sum = 0.0f;
    for (uint32_t g = 0u; g < SGS; g++) sum += partial[(o * SGS + g) * 16u + (lid & 15u)];
    y[(size_t)(slot * uint32_t(NSPLIT) + split) * (size_t)(ODIM) + (ot + o) * TILE + (lid & 15u)] = sum;
  }
  __syncthreads();
}
"""

# SwiGLU as Metal writes it: sigmoid(g) = 1 / (1 + exp(|g|)) for g < 0, else 1 minus that.
_SILU_MUL = r"""
  {
    const float ys = 1.0f / (1.0f + expf(fabsf(%(g)s)));
    %(h)s = %(g)s * (%(g)s < 0.0f ? ys : 1.0f - ys) * %(u)s;
  }
"""


def _silu_mul(i):
    """SwiGLU of component i: h(i) = g(i) * sigmoid(g(i)) * u(i)."""
    return _SILU_MUL % {"g": f"g{i}", "u": f"u{i}", "h": f"h{i}"}

DOWN_SRC = _GEMV_HEAD + r"""
__shared__ __half prepared[uint32_t(IDIM)];
const uint32_t nblocks = uint32_t(IDIM) / 128u;
for (uint32_t block = sg; block < nblocks; block += SGS) {
  const uint32_t base = block * 128u;
  const size_t sb = (size_t)eid * (size_t)(IDIM) + base;
  float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
  for (uint32_t sp = 0u; sp < uint32_t(NSPLIT); sp++) {
    const size_t xb = ((size_t)slot * uint32_t(NSPLIT) + sp) * (size_t)(IDIM) + base;
    v0 += ig[xb + lane];
    v1 += ig[xb + lane + 32u];
    v2 += ig[xb + lane + 64u];
    v3 += ig[xb + lane + 96u];
  }
""" + _FWHT + r"""
  float s0 = v0 + v1;
  float s1 = v0 - v1;
  float s2 = v2 + v3;
  float s3 = v2 - v3;
  const float g0 = (s0 + s2) * sc * __half2float(svhg[sb + lane]);
  const float g1 = (s1 + s3) * sc * __half2float(svhg[sb + lane + 32u]);
  const float g2 = (s0 - s2) * sc * __half2float(svhg[sb + lane + 64u]);
  const float g3 = (s1 - s3) * sc * __half2float(svhg[sb + lane + 96u]);
  v0 = 0.0f; v1 = 0.0f; v2 = 0.0f; v3 = 0.0f;
  for (uint32_t sp = 0u; sp < uint32_t(NSPLIT); sp++) {
    const size_t xbu = ((size_t)slot * uint32_t(NSPLIT) + sp) * (size_t)(IDIM) + base;
    v0 += iu[xbu + lane];
    v1 += iu[xbu + lane + 32u];
    v2 += iu[xbu + lane + 64u];
    v3 += iu[xbu + lane + 96u];
  }
""" + _FWHT + r"""
  s0 = v0 + v1;
  s1 = v0 - v1;
  s2 = v2 + v3;
  s3 = v2 - v3;
  const float u0 = (s0 + s2) * sc * __half2float(svhu[sb + lane]);
  const float u1 = (s1 + s3) * sc * __half2float(svhu[sb + lane + 32u]);
  const float u2 = (s0 - s2) * sc * __half2float(svhu[sb + lane + 64u]);
  const float u3 = (s1 - s3) * sc * __half2float(svhu[sb + lane + 96u]);
  float h0, h1, h2, h3;
""" + "".join(_silu_mul(i) for i in range(4)) + r"""
  v0 = h0 * __half2float(suhd[sb + lane]);
  v1 = h1 * __half2float(suhd[sb + lane + 32u]);
  v2 = h2 * __half2float(suhd[sb + lane + 64u]);
  v3 = h3 * __half2float(suhd[sb + lane + 96u]);
""" + _FWHT + r"""
  s0 = v0 + v1;
  s1 = v0 - v1;
  s2 = v2 + v3;
  s3 = v2 - v3;
  prepared[base + lane] = __float2half((s0 + s2) * sc);
  prepared[base + lane + 32u] = __float2half((s1 + s3) * sc);
  prepared[base + lane + 64u] = __float2half((s0 - s2) * sc);
  prepared[base + lane + 96u] = __float2half((s1 - s3) * sc);
}
__syncthreads();
const uint32_t tk0 = 0u;
const uint32_t tk1 = IT;
const uint32_t *trellis_e = reinterpret_cast<const uint32_t *>(trellis + ((size_t)eid * (size_t)IT * (size_t)OT) * PACKED_HW);
""" + _tiles("prepared", "0u") + r"""
if (lid < 16u * uint32_t(OTPT)) {
  const uint32_t o = lid >> 4u;
  float sum = 0.0f;
  for (uint32_t g = 0u; g < SGS; g++) sum += partial[(o * SGS + g) * 16u + (lid & 15u)];
  y[(size_t)slot * (size_t)(ODIM) + (ot + o) * TILE + (lid & 15u)] = __float2half(sum);
}
"""

REDUCE_SRC = r"""
__shared__ float vals[TOPK * 128];
const uint32_t block = blockIdx.x;
const uint32_t row = blockIdx.y;
const uint32_t sg = threadIdx.x >> 5u;
const uint32_t lane = threadIdx.x & 31u;
const uint32_t base = block * 128u;
const uint32_t slot = row * uint32_t(TOPK) + sg;
const uint32_t eid = slots[slot];
const size_t xb = (size_t)slot * (size_t)(ODIM) + base;
const size_t sb = (size_t)eid * (size_t)(ODIM) + base;
const float scv = 0.08838834764831845f;
float v0 = __half2float(inner[xb + lane]);
float v1 = __half2float(inner[xb + lane + 32u]);
float v2 = __half2float(inner[xb + lane + 64u]);
float v3 = __half2float(inner[xb + lane + 96u]);
""" + _FWHT + r"""
const float s0 = v0 + v1;
const float s1 = v0 - v1;
const float s2 = v2 + v3;
const float s3 = v2 - v3;
vals[sg * 128u + lane] = (s0 + s2) * scv * __half2float(svh[sb + lane]);
vals[sg * 128u + lane + 32u] = (s1 + s3) * scv * __half2float(svh[sb + lane + 32u]);
vals[sg * 128u + lane + 64u] = (s0 - s2) * scv * __half2float(svh[sb + lane + 64u]);
vals[sg * 128u + lane + 96u] = (s1 - s3) * scv * __half2float(svh[sb + lane + 96u]);
__syncthreads();
if (sg == 0u) {
  float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
  for (uint32_t k = 0u; k < uint32_t(TOPK); k++) {
    const float w = sc[row * uint32_t(TOPK) + k];
    a0 += vals[k * 128u + lane] * w;
    a1 += vals[k * 128u + lane + 32u] * w;
    a2 += vals[k * 128u + lane + 64u] * w;
    a3 += vals[k * 128u + lane + 96u] * w;
  }
  const size_t yb = (size_t)row * (size_t)(ODIM) + base;
  y[yb + lane] = static_cast<T>(a0);
  y[yb + lane + 32u] = static_cast<T>(a1);
  y[yb + lane + 64u] = static_cast<T>(a2);
  y[yb + lane + 96u] = static_cast<T>(a3);
}
"""

DEQ_SRC = r"""
const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
constexpr uint32_t N = uint32_t(NHW);
constexpr uint32_t IT = uint32_t(IDIM) / 16u;
constexpr uint32_t OT = uint32_t(ODIM) / 16u;
const uint32_t j = gid & 255u;
const uint32_t tile = gid >> 8u;
const uint32_t tn = tile % OT;
const uint32_t tk = (tile / OT) % IT;
const uint32_t c = tile / (OT * IT);
const uint32_t eid = slots[c];
const uint32_t *words = reinterpret_cast<const uint32_t *>(trellis + (((size_t)eid * IT + tk) * OT + tn) * N);
const uint32_t end = ((uint32_t(perm[j]) + 1u) * N) >> 4u;
const uint32_t last = (end - 1u) >> 5u;
const uint32_t prev = last == 0u ? N / 2u - 1u : last - 1u;
const uint64_t bits = (((uint64_t)words[prev] << 32u) | (uint64_t)words[last]) >> ((0u - end) & 31u);
const float v = exl3_val<WIN>(uint32_t(bits) & 0xffffu);
q[((size_t)c * IDIM + tk * 16u + (j >> 4u)) * ODIM + tn * 16u + (j & 15u)] = static_cast<T>(v);
"""

# The dense path's prefill kernels on CUDA, each checked by
# exl3_experts against the path it replaces.

# MOE_EXL3_CDEQ: DEQ_SRC's Q, eight weights a lane from one window into shared memory then a 16 x
# 128 strip of eight tiles out in whole rows, as exl3_experts' DEQ2_SRC.
DEQ2_SRC = r"""
const uint32_t tid = threadIdx.x;
const uint32_t tg = blockIdx.x;
constexpr uint32_t N = uint32_t(NHW);
constexpr uint32_t IT = uint32_t(IDIM) / 16u;
constexpr uint32_t OT = uint32_t(ODIM) / 16u;
constexpr uint32_t OG = OT / 8u;
const uint32_t og = tg % OG;
const uint32_t tk = (tg / OG) % IT;
const uint32_t c = tg / (OG * IT);
const uint32_t ti = tid >> 5u;
const uint32_t lane = tid & 31u;
const uint32_t eid = slots[c];
const uint32_t *words = reinterpret_cast<const uint32_t *>(trellis + (((size_t)eid * IT + tk) * OT + og * 8u + ti) * N);
const uint64_t merged = exl3_lane<N>(words, lane);
__shared__ __align__(16) T tile[16u * 128u];
// Stream weight j of lane l is entry (r0 + {0, 1, 8, 9}[j & 3], c0 + 8 (j >> 2)) of its tile (_tc_perm_inv).
const uint32_t r0 = (lane & 3u) * 2u;
const uint32_t c0 = ti * 16u + (lane >> 2u);
#pragma unroll
for (uint32_t j = 0u; j < 8u; j++) {
  const float v = exl3_val<WIN>(uint32_t(merged >> exl3_lane_sh(N, j)) & 0xffffu);
  tile[(r0 + (j & 1u) + ((j & 2u) ? 8u : 0u)) * 128u + c0 + ((j & 4u) ? 8u : 0u)] = static_cast<T>(v);
}
__syncthreads();
const uint32_t orow = tid >> 4u;
const uint32_t ocol = (tid & 15u) * 8u;
const uint4 *src = reinterpret_cast<const uint4 *>(tile + orow * 128u + ocol);
uint4 *dst = reinterpret_cast<uint4 *>(q + ((size_t)c * IDIM + tk * 16u + orow) * ODIM + og * 128u + ocol);
#pragma unroll
for (uint32_t k = 0u; k < uint32_t(sizeof(T)) / 2u; k++) dst[k] = src[k];
"""

# MOE_EXL3_CFUSE: the dense call's preps, finishes and SwiGLU as three kernels doing
# exl3_experts._proj's ops in its order, with MLX's hadamard_n stages and every product rounded
# alone (__fmul_rn).
_UNIT = r"""
const uint32_t lane = threadIdx.x & 31u;
const uint32_t unit = blockIdx.x * 8u + (threadIdx.x >> 5u);
constexpr uint32_t NB = uint32_t(DIM) / 128u;
if (unit >= uint32_t(loc_shape[0]) * NB) return;
const uint32_t r = unit / NB;
const uint32_t base = (unit % NB) * 128u;
const size_t sb = (size_t)slots[loc[r]] * (size_t)(DIM) + base;
const size_t ob = (size_t)r * (size_t)(DIM) + base;
const float sc = 0.08838834764831845f;
"""
_HAD = _FWHT + r"""
  {
    const float s0 = v0 + v1, s1 = v0 - v1, s2 = v2 + v3, s3 = v2 - v3;
    v0 = __fmul_rn(s0 + s2, sc);
    v1 = __fmul_rn(s1 + s3, sc);
    v2 = __fmul_rn(s0 - s2, sc);
    v3 = __fmul_rn(s1 - s3, sc);
  }
"""
# gate and up inputs: rows sel of x times each projection's suh, transformed, rounded to f16, out as
# O.
CPREP_GU_SRC = _UNIT + r"""
const size_t xb = (size_t)sel[r] * (size_t)(DIM) + base;
const float x0 = static_cast<float>(x[xb + lane]);
const float x1 = static_cast<float>(x[xb + lane + 32u]);
const float x2 = static_cast<float>(x[xb + lane + 64u]);
const float x3 = static_cast<float>(x[xb + lane + 96u]);
#pragma unroll
for (uint32_t pj = 0u; pj < 2u; pj++) {
  const __half *su = (pj == 0u) ? sug : suu;
  O *o = (pj == 0u) ? xg : xu;
  float v0 = __fmul_rn(x0, __half2float(su[sb + lane]));
  float v1 = __fmul_rn(x1, __half2float(su[sb + lane + 32u]));
  float v2 = __fmul_rn(x2, __half2float(su[sb + lane + 64u]));
  float v3 = __fmul_rn(x3, __half2float(su[sb + lane + 96u]));
""" + _HAD + r"""
  o[ob + lane] = static_cast<O>(__float2half_rn(v0));
  o[ob + lane + 32u] = static_cast<O>(__float2half_rn(v1));
  o[ob + lane + 64u] = static_cast<O>(__float2half_rn(v2));
  o[ob + lane + 96u] = static_cast<O>(__float2half_rn(v3));
}
"""
# gate and up outputs finished (transform, svh), SwiGLU as MLX's sigmoid writes it, then the down
# input times suh, transformed and rounded to f16.
_SIG = "(G < 0.0f ? 1.0f / (1.0f + expf(fabsf(G))) : 1.0f - 1.0f / (1.0f + expf(fabsf(G))))"
CFIN_GU_D_SRC = _UNIT + r"""
float g[4], u[4];
#pragma unroll
for (uint32_t pj = 0u; pj < 2u; pj++) {
  const float *p = (pj == 0u) ? pg : pu;
  const __half *sv = (pj == 0u) ? svg : svu;
  float v0 = p[ob + lane], v1 = p[ob + lane + 32u], v2 = p[ob + lane + 64u], v3 = p[ob + lane + 96u];
""" + _HAD + r"""
  float *o = (pj == 0u) ? g : u;
  o[0] = __fmul_rn(__half2float(sv[sb + lane]), v0);
  o[1] = __fmul_rn(__half2float(sv[sb + lane + 32u]), v1);
  o[2] = __fmul_rn(__half2float(sv[sb + lane + 64u]), v2);
  o[3] = __fmul_rn(__half2float(sv[sb + lane + 96u]), v3);
}
float h[4];
#pragma unroll
for (uint32_t i = 0u; i < 4u; i++) {
  const float G = g[i];
  h[i] = __fmul_rn(__fmul_rn(G, """ + _SIG + r"""), u[i]);
}
float v0 = __fmul_rn(h[0], __half2float(sud[sb + lane]));
float v1 = __fmul_rn(h[1], __half2float(sud[sb + lane + 32u]));
float v2 = __fmul_rn(h[2], __half2float(sud[sb + lane + 64u]));
float v3 = __fmul_rn(h[3], __half2float(sud[sb + lane + 96u]));
""" + _HAD + r"""
xd[ob + lane] = __float2half_rn(v0);
xd[ob + lane + 32u] = __float2half_rn(v1);
xd[ob + lane + 64u] = __float2half_rn(v2);
xd[ob + lane + 96u] = __float2half_rn(v3);
"""
# down output: f16 in, transformed, times svh, cast to the output dtype.
CFIN_D_SRC = _UNIT + r"""
float v0 = __half2float(p[ob + lane]), v1 = __half2float(p[ob + lane + 32u]);
float v2 = __half2float(p[ob + lane + 64u]), v3 = __half2float(p[ob + lane + 96u]);
""" + _HAD + r"""
y[ob + lane] = static_cast<T>(__fmul_rn(__half2float(sv[sb + lane]), v0));
y[ob + lane + 32u] = static_cast<T>(__fmul_rn(__half2float(sv[sb + lane + 32u]), v1));
y[ob + lane + 64u] = static_cast<T>(__fmul_rn(__half2float(sv[sb + lane + 64u]), v2));
y[ob + lane + 96u] = static_cast<T>(__fmul_rn(__half2float(sv[sb + lane + 96u]), v3));
"""

# MOE_EXL3_CMM: the gate and up GEMMs (grid z) on f16 rows and Q as f16 m16n8k8 MMAs, which give an
# f32 gather_mm's TF32 bits from half the bytes.
GMM16_SRC = r"""
constexpr int BM = 64, BN = 128, BK = 32, AS = BK + 8, BS = BN + 8;
__shared__ __align__(16) __half sa[2][BM * AS];
__shared__ __align__(16) __half sb[2][BK * BS];
const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
const int tl = blockIdx.x, n0 = blockIdx.y * BN, z = blockIdx.z;
const __half *A = z == 0 ? ag : au;
const __half *B = (z == 0 ? qg : qu) + (size_t)tiles[tl * 3] * KD * ND;
float *C = z == 0 ? cg : cu;
const int m0 = tiles[tl * 3 + 1], m1 = tiles[tl * 3 + 2];
const int wm = (warp >> 1) * 32, wn = (warp & 1) * 64;
float acc[2][8][4];
#pragma unroll
for (int i = 0; i < 2; i++)
  #pragma unroll
  for (int j = 0; j < 8; j++)
    #pragma unroll
    for (int v = 0; v < 4; v++) acc[i][j][v] = 0.0f;
#define GMM_LOAD(STAGE, K0) { \
  _Pragma("unroll") for (int i = 0; i < 2; i++) { \
    const int ch = tid + i * 128, r = ch >> 2, cc = (ch & 3) * 8; \
    const __half *src = A + (size_t)min(m0 + r, m1 - 1) * KD + (K0) + cc; \
    const uint32_t dst = (uint32_t)__cvta_generic_to_shared(&sa[STAGE][r * AS + cc]); \
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src)); } \
  _Pragma("unroll") for (int i = 0; i < 4; i++) { \
    const int ch = tid + i * 128, r = ch >> 4, cc = (ch & 15) * 8; \
    const __half *src = B + (size_t)((K0) + r) * ND + n0 + cc; \
    const uint32_t dst = (uint32_t)__cvta_generic_to_shared(&sb[STAGE][r * BS + cc]); \
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src)); } \
  asm volatile("cp.async.commit_group;"); }
GMM_LOAD(0, 0)
for (int kt = 0; kt < KD / BK; kt++) {
  const int s = kt & 1;
  if (kt + 1 < KD / BK) {
    GMM_LOAD(s ^ 1, (kt + 1) * BK)
    asm volatile("cp.async.wait_group 1;");
  } else {
    asm volatile("cp.async.wait_group 0;");
  }
  __syncthreads();
  #pragma unroll
  for (int ks = 0; ks < BK; ks += 8) {
    uint32_t af[2][2], bf[8];
    {
      const uint32_t addr = (uint32_t)__cvta_generic_to_shared(&sa[s][(wm + (lane & 7) + (lane >> 3) * 8) * AS + ks]);
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                   : "=r"(af[0][0]), "=r"(af[0][1]), "=r"(af[1][0]), "=r"(af[1][1]) : "r"(addr));
    }
    #pragma unroll
    for (int h = 0; h < 2; h++) {
      const uint32_t addr = (uint32_t)__cvta_generic_to_shared(&sb[s][(ks + (lane & 7)) * BS + wn + (h * 4 + (lane >> 3)) * 8]);
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                   : "=r"(bf[h * 4]), "=r"(bf[h * 4 + 1]), "=r"(bf[h * 4 + 2]), "=r"(bf[h * 4 + 3]) : "r"(addr));
    }
    #pragma unroll
    for (int i = 0; i < 2; i++)
      #pragma unroll
      for (int j = 0; j < 8; j++)
        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                     : "+f"(acc[i][j][0]), "+f"(acc[i][j][1]), "+f"(acc[i][j][2]), "+f"(acc[i][j][3])
                     : "r"(af[i][0]), "r"(af[i][1]), "r"(bf[j]));
  }
  __syncthreads();
}
const int g = lane >> 2, t = lane & 3;
#pragma unroll
for (int i = 0; i < 2; i++) {
  const int r0 = m0 + wm + i * 16 + g;
  #pragma unroll
  for (int j = 0; j < 8; j++) {
    const int col = n0 + wn + j * 8 + 2 * t;
    if (r0 < m1) *reinterpret_cast<float2 *>(C + (size_t)r0 * ND + col) = make_float2(acc[i][j][0], acc[i][j][1]);
    if (r0 + 8 < m1) *reinterpret_cast<float2 *>(C + (size_t)(r0 + 8) * ND + col) = make_float2(acc[i][j][2], acc[i][j][3]);
  }
}
#undef GMM_LOAD
"""

SOURCES = {"exl3_pair_gemv": PAIR_SRC, "exl3_down_fused": DOWN_SRC, "exl3_reduce": REDUCE_SRC,
           "exl3_dequant": DEQ_SRC, "exl3_dequant2": DEQ2_SRC, "exl3_cprep_gu": CPREP_GU_SRC,
           "exl3_cfin_gu_d": CFIN_GU_D_SRC, "exl3_cfin_d": CFIN_D_SRC, "exl3_gmm16": GMM16_SRC}
# MOE_PIPE_HITS_FIRST's chain: a block whose slot is 0xFFFFFFFF leaves at once and writes nothing, the other slots' math unchanged.
for _n in ("exl3_pair_gemv", "exl3_down_fused", "exl3_reduce"):
    SOURCES[_n + "_skip"] = "if (slots[blockIdx.y] == 0xFFFFFFFFu) return;\n" + SOURCES[_n]


def kernel(name, ins, outs):
    """exl3_experts' kernel `name` as a CUDA kernel, the same inputs and outputs."""
    return mx.fast.cuda_kernel(name=name, input_names=ins, output_names=outs, source=SOURCES[name], header=HEADER)
