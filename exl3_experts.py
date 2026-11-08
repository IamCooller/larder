"""EXL3 trellis-coded routed experts: the engine's gathers for them, in a pack sushi and
exllamav3 define."""
# The licences of the ports: THIRD_PARTY_NOTICES.md (ddalcu/sushi and turboderp-org/exllamav3, MIT).
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

import mlx.core as mx
from engine_log import note

# MLX on CUDA: the same kernels as CUDA sources (exl3_cuda.py, never imported on Metal).
_CUDA = not mx.metal.is_available() and getattr(mx, "cuda", None) is not None and mx.cuda.is_available()
if _CUDA:
    import exl3_cuda

# A mul1 pack's WIN template argument is its window plus MUL1, which only exl3_cuda's decode knows (CUDA).
MUL1 = 256


def spec(eq):
    """The WIN template argument of config.json's expert_quant: the window (16 when absent), plus MUL1 for mul1."""
    return int(eq.get("window", 16)) + (MUL1 if eq.get("codebook") == "mul1" else 0)


HEADER = r"""
// MCG codebook, two codewords to two halves; a narrower search window masks the stream window first.
template <int WIN>
static inline half2 exl3_pairh(uint2 cw) {
  if (WIN < 16) cw &= uint2((1u << WIN) - 1u);
  const uint2 r = ((cw * uint2(0xCBAC1FEDu)) & uint2(0x8FFF8FFFu)) ^ uint2(0x3B603B60u);
  const half4 h = as_type<half4>(r);
  return half2(h.x + h.y, h.z + h.w);
}
template <int WIN>
static inline float2 exl3_decode2(uint2 cw) { return float2(exl3_pairh<WIN>(cw)); }
// Eight weights span N/2 whole bits, so weight j of an eight-weight group
// ends exl3_end(N, j) bits past the group's first bit.
static inline constexpr uint exl3_end(uint N, uint j) { return ((j + 1u) * N) >> 4u; }
// The 64 stream bits ending at bit `end` of a tile of `nwords` words (the tile
// wraps). Two words carry only the last 32 + end % 32 of them; `full` reads a third.
static inline ulong exl3_window(const device uint *words, uint end, uint nwords, bool full) {
  const uint last = (end - 1u) >> 5u;
  const uint prev = last == 0u ? nwords - 1u : last - 1u;
  const uint s = (0u - end) & 31u;
  ulong bits = (((ulong)words[prev] << 32u) | (ulong)words[last]) >> s;
  if (full) bits |= ((ulong)words[prev == 0u ? nwords - 1u : prev - 1u] << 32u) << (32u - s);
  return bits;
}
// Lane l reads weights 8l..8l+7 from the bits ending at its group's last bit; weight
// j's codeword ends exl3_lane_sh(N, j) bits before that.
static inline constexpr uint exl3_lane_sh(uint N, uint j) { return N / 2u - exl3_end(N, j); }
template<uint N>
static inline ulong exl3_lane(const device uint *words, uint lane) {
  constexpr uint W = N / 2u;
  constexpr uint low = W & (0u - W);
  return exl3_window(words, W * lane + W, W, exl3_lane_sh(N, 0u) + 16u > 32u + (low < 32u ? low : 32u));
}
"""

# One 128-wide Hadamard of v (lane holds elements lane, +32, +64, +96), in place.
_FWHT = r"""
  for (ushort bit = 1u; bit <= 16u; bit <<= 1u) {
    const float p0 = simd_shuffle_xor(v.x, bit);
    const float p1 = simd_shuffle_xor(v.y, bit);
    const float p2 = simd_shuffle_xor(v.z, bit);
    const float p3 = simd_shuffle_xor(v.w, bit);
    const bool lower = (lane & bit) == 0u;
    v.x = lower ? v.x + p0 : p0 - v.x;
    v.y = lower ? v.y + p1 : p1 - v.y;
    v.z = lower ? v.z + p2 : p2 - v.z;
    v.w = lower ? v.w + p3 : p3 - v.w;
  }
"""

# The tile loop both GEMVs share, its inputs in %(space)s memory from tile %(pbase)s of %(prep)s on:
# K4 reads two whole words a lane, lower rates read through the funnel.
_TILES = r"""
const uint prow = (lane & 3u) * 2u;
const uint pcol = lane >> 2u;
const uint row0 = prow;
const uint row1 = prow + 1u;
const uint row2 = prow + 8u;
const uint row3 = prow + 9u;
float acc[OTPT][8] = {};
if (N == 64u) {
  for (uint tk = tk0 + sg; tk < tk1; tk += SGS) {
    const device uint *words = trellis_e + ((size_t)tk * (size_t)OT + ot) * 32u;
    const ulong merged = ((ulong)words[(lane + 31u) & 31u] << 32) | (ulong)words[lane];
    const float in0 = float(%(prep)s[(tk - %(pbase)s) * TILE + row0]);
    const float in1 = float(%(prep)s[(tk - %(pbase)s) * TILE + row1]);
    const float in2 = float(%(prep)s[(tk - %(pbase)s) * TILE + row2]);
    const float in3 = float(%(prep)s[(tk - %(pbase)s) * TILE + row3]);
    const uint sh[8] = {28u, 24u, 20u, 16u, 12u, 8u, 4u, 0u};
    const float ins[8] = {in0, in1, in2, in3, in0, in1, in2, in3};
    for (uint p = 0u; p < 4u; p++) {
      const uint2 cw = uint2(uint(merged >> sh[p * 2u]), uint(merged >> sh[p * 2u + 1u])) & uint2(0xffffu);
      const float2 w = exl3_decode2<WIN>(cw);
      acc[0][p * 2u] = fma(ins[p * 2u], w.x, acc[0][p * 2u]);
      acc[0][p * 2u + 1u] = fma(ins[p * 2u + 1u], w.y, acc[0][p * 2u + 1u]);
    }
  }
} else {
  // Both k-tiles' words load before either decodes; a k range is whole H128
  // blocks (8 tiles), so the second k-tile of an iteration is always in range.
  const device uint *wp = trellis_e + ((size_t)(tk0 + sg) * (size_t)OT + ot) * PACKED_W;
  const %(space)s half *pp = %(prep)s + (tk0 - %(pbase)s + sg) * TILE;
  const uint sh[8] = {exl3_lane_sh(N, 0u), exl3_lane_sh(N, 1u), exl3_lane_sh(N, 2u), exl3_lane_sh(N, 3u), exl3_lane_sh(N, 4u), exl3_lane_sh(N, 5u), exl3_lane_sh(N, 6u), exl3_lane_sh(N, 7u)};
  for (uint tk = tk0 + sg; tk < tk1; tk += 2u * SGS) {
    ulong merged[2][OTPT];
    for (uint u = 0u; u < 2u; u++) {
      for (uint o = 0u; o < uint(OTPT); o++) {
        const device uint *words = wp + u * SGS * OT * PACKED_W + o * PACKED_W;
        merged[u][o] = exl3_lane<N>(words, lane);
      }
    }
    for (uint u = 0u; u < 2u; u++) {
      const float in0 = float(pp[u * SGS * TILE + row0]);
      const float in1 = float(pp[u * SGS * TILE + row1]);
      const float in2 = float(pp[u * SGS * TILE + row2]);
      const float in3 = float(pp[u * SGS * TILE + row3]);
      const float ins[8] = {in0, in1, in2, in3, in0, in1, in2, in3};
      for (uint o = 0u; o < uint(OTPT); o++) {
        for (uint p = 0u; p < 4u; p++) {
          const uint2 cw = uint2(uint(merged[u][o] >> sh[p * 2u]), uint(merged[u][o] >> sh[p * 2u + 1u])) & uint2(0xffffu);
          const float2 w = exl3_decode2<WIN>(cw);
          acc[o][p * 2u] = fma(ins[p * 2u], w.x, acc[o][p * 2u]);
          acc[o][p * 2u + 1u] = fma(ins[p * 2u + 1u], w.y, acc[o][p * 2u + 1u]);
        }
      }
    }
    wp += 2u * SGS * OT * PACKED_W;
    pp += 2u * SGS * TILE;
  }
}
for (uint o = 0u; o < uint(OTPT); o++) {
  float clo = (acc[o][0] + acc[o][1]) + (acc[o][2] + acc[o][3]);
  float chi = (acc[o][4] + acc[o][5]) + (acc[o][6] + acc[o][7]);
  clo += simd_shuffle_xor(clo, 1u);
  chi += simd_shuffle_xor(chi, 1u);
  clo += simd_shuffle_xor(clo, 2u);
  chi += simd_shuffle_xor(chi, 2u);
  if ((lane & 3u) == 0u) {
    partial[(o * SGS + sg) * 16u + pcol] = clo;
    partial[(o * SGS + sg) * 16u + pcol + 8u] = chi;
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
"""


def _tiles(prep, pbase, space="threadgroup"):
    """The shared tile loop, its inputs read from prep at tile pbase."""
    return _TILES % {"prep": prep, "pbase": pbase, "space": space}


_GEMV_HEAD = r"""
threadgroup float partial[4 * 16 * uint(OTPT)];
const uint ot = uint(threadgroup_position_in_grid.x) * uint(OTPT);
uint slot = uint(threadgroup_position_in_grid.y);
uint split = uint(threadgroup_position_in_grid.z);
uint sg = uint(simdgroup_index_in_threadgroup);
uint lane = uint(thread_index_in_simdgroup);
uint lid = uint(thread_index_in_threadgroup);
constexpr uint TILE = 16u;
constexpr uint N = uint(NHW);
constexpr uint PACKED_HW = N;
constexpr uint PACKED_W = N / 2u;
constexpr uint IT = uint(IDIM) / TILE;
constexpr uint OT = uint(ODIM) / TILE;
constexpr uint SGS = 4u;
const uint eid = uint(slots[slot]);
const float sc = 0.08838834764831845f;
"""

# x * suh of one projection pj, its FWHT and the four sums.
_IN_HAD = r"""
      float4 v = float4(x0 * float(suh[sr + lane]), x1 * float(suh[sr + lane + 32u]),
                        x2 * float(suh[sr + lane + 64u]), x3 * float(suh[sr + lane + 96u]));
""" + _FWHT + r"""
      const float s0 = v.x + v.y;
      const float s1 = v.x - v.y;
      const float s2 = v.z + v.w;
      const float s3 = v.z - v.w;
"""


def _had_store(arr, at):
    """The last FWHT's four sums into arr at at + lane, + 32, + 64, + 96."""
    return "".join(f"  {arr}[{at} + lane{off}] = half(({val}) * sc);\n"
                   for off, val in (("", "s0 + s2"), (" + 32u", "s1 + s3"),
                                    (" + 64u", "s0 - s2"), (" + 96u", "s1 - s3")))


_PAIR_SPLIT = r"""
const uint tiles_per_split = (IT + uint(NSPLIT) - 1u) / uint(NSPLIT);
const uint tk0 = split * tiles_per_split;
const uint tk1 = min(tk0 + tiles_per_split, IT);
"""

# Each slot's input Hadamards into threadgroup memory, once for both projections.
_PAIR_PREP = r"""
constexpr uint KSPAN = uint(IDIM) / uint(NSPLIT);
threadgroup half prepared[2u * KSPAN];
{
  const uint xrow = slot / uint(TOPK);
  for (uint b = sg; b < KSPAN / 128u; b += SGS) {
    const uint pbase = tk0 * TILE + b * 128u;
    const size_t xr = (size_t)xrow * (size_t)(IDIM) + pbase;
    const size_t sr = (size_t)eid * (size_t)(IDIM) + pbase;
    const float x0 = float(x[xr + lane]);
    const float x1 = float(x[xr + lane + 32u]);
    const float x2 = float(x[xr + lane + 64u]);
    const float x3 = float(x[xr + lane + 96u]);
    for (uint pj = 0u; pj < 2u; pj++) {
      const device half *suh = (pj == 0u) ? suhg : suhu;
""" + _IN_HAD + r"""
      const uint pw = pj * KSPAN + b * 128u;
""" + _had_store("prepared", "pw") + r"""
    }
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
"""

# The gate and up GEMVs one after the other, each reading its half of the prepared input.
_PAIR_HEAD = r"""
for (uint proj = 0u; proj < 2u; proj++) {
  const device ushort *trellis = (proj == 0u) ? tg : tu;
  device float *y = (proj == 0u) ? yg : yu;
  const device uint *trellis_e = (const device uint *)(trellis + ((size_t)eid * (size_t)IT * (size_t)OT) * PACKED_HW);
"""

_PAIR_EPI = r"""
  if (lid < 16u * uint(OTPT)) {
    const uint o = lid >> 4u;
    float sum = 0.0f;
    for (uint g = 0u; g < SGS; g++) sum += partial[(o * SGS + g) * 16u + (lid & 15u)];
    y[(size_t)(slot * uint(NSPLIT) + split) * (size_t)(ODIM) + (ot + o) * TILE + (lid & 15u)] = sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
"""

PAIR_SRC = (_GEMV_HEAD + _PAIR_SPLIT + _PAIR_PREP + _PAIR_HEAD +
            r"""  const threadgroup half *PREP0 = prepared + proj * KSPAN;
""" + _tiles("PREP0", "tk0") + _PAIR_EPI)

# g's and u's Hadamards, the svh scales, SwiGLU, suh and the Hadamard of the product: DOWN's block
# body.
_DOWN_BODY = r"""
  const uint base = block * 128u;
  const size_t sb = (size_t)eid * (size_t)(IDIM) + base;
  float4 v = float4(0.0f);
  for (uint sp = 0u; sp < uint(NSPLIT); sp++) {
    const size_t xb = ((size_t)slot * uint(NSPLIT) + sp) * (size_t)(IDIM) + base;
    v += float4(float(ig[xb + lane]), float(ig[xb + lane + 32u]), float(ig[xb + lane + 64u]), float(ig[xb + lane + 96u]));
  }
""" + _FWHT + r"""
  float s0 = v.x + v.y;
  float s1 = v.x - v.y;
  float s2 = v.z + v.w;
  float s3 = v.z - v.w;
  const float g0 = (s0 + s2) * sc * float(svhg[sb + lane]);
  const float g1 = (s1 + s3) * sc * float(svhg[sb + lane + 32u]);
  const float g2 = (s0 - s2) * sc * float(svhg[sb + lane + 64u]);
  const float g3 = (s1 - s3) * sc * float(svhg[sb + lane + 96u]);
  v = float4(0.0f);
  for (uint sp = 0u; sp < uint(NSPLIT); sp++) {
    const size_t xbu = ((size_t)slot * uint(NSPLIT) + sp) * (size_t)(IDIM) + base;
    v += float4(float(iu[xbu + lane]), float(iu[xbu + lane + 32u]), float(iu[xbu + lane + 64u]), float(iu[xbu + lane + 96u]));
  }
""" + _FWHT + r"""
  s0 = v.x + v.y;
  s1 = v.x - v.y;
  s2 = v.z + v.w;
  s3 = v.z - v.w;
  const float u0 = (s0 + s2) * sc * float(svhu[sb + lane]);
  const float u1 = (s1 + s3) * sc * float(svhu[sb + lane + 32u]);
  const float u2 = (s0 - s2) * sc * float(svhu[sb + lane + 64u]);
  const float u3 = (s1 - s3) * sc * float(svhu[sb + lane + 96u]);
  const float4 g = float4(g0, g1, g2, g3);
  const float4 ys = 1 / (1 + exp(abs(g)));
  const float4 sig = select(1 - ys, ys, g < 0);
  const float4 h = g * sig * float4(u0, u1, u2, u3);
  v = h * float4(float(suhd[sb + lane]), float(suhd[sb + lane + 32u]), float(suhd[sb + lane + 64u]), float(suhd[sb + lane + 96u]));
""" + _FWHT + r"""
  s0 = v.x + v.y;
  s1 = v.x - v.y;
  s2 = v.z + v.w;
  s3 = v.z - v.w;
"""

_DOWN_HEAD = r"""
threadgroup half prepared[uint(IDIM)];
const uint nblocks = uint(IDIM) / 128u;
for (uint block = sg; block < nblocks; block += SGS) {
"""
_DOWN_LOOP = r"""
const uint tk0 = 0u;
const uint tk1 = IT;
const device uint *trellis_e = (const device uint *)(trellis + ((size_t)eid * (size_t)IT * (size_t)OT) * PACKED_HW);
"""
_DOWN_EPI = r"""
if (lid < 16u * uint(OTPT)) {
  const uint o = lid >> 4u;
  float sum = 0.0f;
  for (uint g = 0u; g < SGS; g++) sum += partial[(o * SGS + g) * 16u + (lid & 15u)];
  y[(size_t)slot * (size_t)(ODIM) + (ot + o) * TILE + (lid & 15u)] = half(sum);
}
"""

DOWN_SRC = (_GEMV_HEAD + _DOWN_HEAD + _DOWN_BODY + _had_store("prepared", "base") +
            r"""}
threadgroup_barrier(mem_flags::mem_threadgroup);
""" + _DOWN_LOOP + _tiles("prepared", "0u") + _DOWN_EPI)

REDUCE_SRC = r"""
threadgroup float vals[uint(TOPK) * 128u];
uint block = uint(threadgroup_position_in_grid.x);
uint row = uint(threadgroup_position_in_grid.y);
uint sg = uint(simdgroup_index_in_threadgroup);
ushort lane = thread_index_in_simdgroup;
const uint base = block * 128u;
const uint slot = row * uint(TOPK) + sg;
const uint eid = uint(slots[slot]);
const size_t xb = (size_t)slot * (size_t)(ODIM) + base;
const size_t sb = (size_t)eid * (size_t)(ODIM) + base;
const float scv = 0.08838834764831845f;
float4 v = float4(float(inner[xb + lane]), float(inner[xb + lane + 32u]), float(inner[xb + lane + 64u]), float(inner[xb + lane + 96u]));
""" + _FWHT + r"""
const float s0 = v.x + v.y;
const float s1 = v.x - v.y;
const float s2 = v.z + v.w;
const float s3 = v.z - v.w;
vals[sg * 128u + lane] = (s0 + s2) * scv * float(svh[sb + lane]);
vals[sg * 128u + lane + 32u] = (s1 + s3) * scv * float(svh[sb + lane + 32u]);
vals[sg * 128u + lane + 64u] = (s0 - s2) * scv * float(svh[sb + lane + 64u]);
vals[sg * 128u + lane + 96u] = (s1 - s3) * scv * float(svh[sb + lane + 96u]);
threadgroup_barrier(mem_flags::mem_threadgroup);
if (sg == 0u) {
  float4 a = float4(0.0f);
  for (uint k = 0u; k < uint(TOPK); k++) {
    const float w = float(sc[row * uint(TOPK) + k]);
    a += float4(vals[k * 128u + lane], vals[k * 128u + lane + 32u], vals[k * 128u + lane + 64u], vals[k * 128u + lane + 96u]) * w;
  }
  const size_t yb = (size_t)row * (size_t)(ODIM) + base;
  y[yb + lane] = T(a.x);
  y[yb + lane + 32u] = T(a.y);
  y[yb + lane + 64u] = T(a.z);
  y[yb + lane + 96u] = T(a.w);
}
"""

# MOE_GLUE (Metal, on by default): each slot's input Hadamards once, in the GEMVs' own prep
# fragments (PREP_IN for PAIR, DOWN_PREP for DOWN), instead of in every threadgroup of PAIR (20 a
# slot) and DOWN (80 a slot), so every value stays bit-equal.
PREP = mx.metal.is_available() and os.environ.get("MOE_GLUE", "1") != "0"   # on unless MOE_GLUE=0


_SLOT = r"""
const uint slot = uint(threadgroup_position_in_grid.y);
const uint lane = uint(thread_index_in_simdgroup);
const uint eid = uint(slots[slot]);
const float sc = 0.08838834764831845f;
"""
# One 128-block b of slot's input for projection pj (gate 0, up 1): PAIR's prep loop body at b.
PREP_IN_SRC = _SLOT + r"""
const uint b = uint(threadgroup_position_in_grid.x);
const uint pj = uint(threadgroup_position_in_grid.z);
const uint xrow = slot / uint(TOPK);
const size_t xr = (size_t)xrow * (size_t)(IDIM) + b * 128u;
const size_t sr = (size_t)eid * (size_t)(IDIM) + b * 128u;
const float x0 = float(x[xr + lane]);
const float x1 = float(x[xr + lane + 32u]);
const float x2 = float(x[xr + lane + 64u]);
const float x3 = float(x[xr + lane + 96u]);
{
  const device half *suh = (pj == 0u) ? suhg : suhu;
""" + _IN_HAD + r"""
  const size_t pw = ((size_t)slot * 2u + pj) * (size_t)(IDIM) + b * 128u;
""" + _had_store("prep", "pw") + r"""
}
"""
# One 128-block of slot's down input (the split sums, svh, SwiGLU, suh, the Hadamards): DOWN's own
# block body.
DOWN_PREP_SRC = _SLOT + r"""
const uint block = uint(threadgroup_position_in_grid.x);
{
""" + _DOWN_BODY + r"""
  const size_t pb = (size_t)slot * (size_t)(IDIM) + base;
""" + _had_store("prepd", "pb") + r"""
}
"""
# The GEMVs as before after their prep: the tile loops read the prepared inputs from device memory.
PAIR2_SRC = (_GEMV_HEAD + _PAIR_SPLIT + _PAIR_HEAD +
             r"""  const device half *PREP0 = prep + ((size_t)slot * 2u + proj) * (size_t)(IDIM) + tk0 * TILE;
""" + _tiles("PREP0", "tk0", "device") + _PAIR_EPI)
DOWN2_SRC = (_GEMV_HEAD + _DOWN_LOOP +
             r"""const device half *PD = prepd + (size_t)slot * (size_t)(IDIM);
""" + _tiles("PD", "0u", "device") + _DOWN_EPI)

_K = {}


def _kernel(name, ins, outs, src):
    if name not in _K and _CUDA:
        _K[name] = exl3_cuda.kernel(name, ins, outs)
    if name not in _K:
        _K[name] = mx.fast.metal_kernel(name=name, input_names=ins, output_names=outs, source=src, header=HEADER)
    return _K[name]

# One weight a thread: tile (c, tk, tn) of slot slots[c], weight j of the tile in (in, out)
# row-major order, which is stream position perm[j]; written to Q [c, in, out].
DEQ_SRC = r"""
const uint gid = uint(thread_position_in_grid.x);
constexpr uint N = uint(NHW);
constexpr uint IT = uint(IDIM) / 16u;
constexpr uint OT = uint(ODIM) / 16u;
const uint j = gid & 255u;
const uint tile = gid >> 8u;
const uint tn = tile % OT;
const uint tk = (tile / OT) % IT;
const uint c = tile / (OT * IT);
const uint eid = uint(slots[c]);
const device uint *words = (const device uint *)(trellis + (((size_t)eid * IT + tk) * OT + tn) * N);
const uint end = ((uint(perm[j]) + 1u) * N) >> 4u;
const uint last = (end - 1u) >> 5u;
const uint prev = last == 0u ? N / 2u - 1u : last - 1u;
const ulong bits = (((ulong)words[prev] << 32u) | (ulong)words[last]) >> ((0u - end) & 31u);
const float v = float(exl3_pairh<WIN>(uint2(uint(bits) & 0xffffu, 0u)).x);
q[((size_t)c * IDIM + tk * 16u + (j >> 4u)) * ODIM + tn * 16u + (j & 15u)] = T(v);
"""


# MOE_EXL3_DEQ2 (E2, on by default, 0 = as before): dequant() through DEQ2_SRC, a 16-row strip of 8
# tiles a threadgroup, each lane's 8 weights from one window into threadgroup memory and out in
# whole 512-byte rows.
DEQ2 = os.environ.get("MOE_EXL3_DEQ2", "1") != "0"
DEQ2_SRC = r"""
const uint tid = uint(thread_index_in_threadgroup);
const uint tg = uint(threadgroup_position_in_grid.x);
constexpr uint N = uint(NHW);
constexpr uint IT = uint(IDIM) / 16u;
constexpr uint OT = uint(ODIM) / 16u;
constexpr uint OG = OT / 8u;
const uint og = tg % OG;
const uint tk = (tg / OG) % IT;
const uint c = tg / (OG * IT);
const uint ti = tid >> 5u;
const uint lane = tid & 31u;
const uint eid = uint(slots[c]);
const device uint *words = (const device uint *)(trellis + (((size_t)eid * IT + tk) * OT + og * 8u + ti) * N);
const ulong merged = exl3_lane<N>(words, lane);
threadgroup T tile[16u * 128u];
// Stream weight j of lane l is entry (r0 + {0, 1, 8, 9}[j & 3], c0 + 8 (j >> 2)) of its tile
// (exllamav3's tensor-core order, _tc_perm_inv).
const uint r0 = (lane & 3u) * 2u;
const uint c0 = ti * 16u + (lane >> 2u);
for (uint p = 0u; p < 4u; p++) {
  const uint j0 = p * 2u;
  const uint2 cw = uint2(uint(merged >> exl3_lane_sh(N, j0)), uint(merged >> exl3_lane_sh(N, j0 + 1u))) & uint2(0xffffu);
  const half2 w = exl3_pairh<WIN>(cw);
  const uint row = r0 + ((p & 1u) ? 8u : 0u);
  const uint col = c0 + ((p & 2u) ? 8u : 0u);
  tile[row * 128u + col] = T(float(w.x));
  tile[(row + 1u) * 128u + col] = T(float(w.y));
}
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint orow = tid >> 4u;
const uint ocol = (tid & 15u) * 8u;
device T *dst = q + ((size_t)c * IDIM + tk * 16u + orow) * ODIM + og * 128u + ocol;
for (uint k = 0u; k < 8u; k++) dst[k] = tile[orow * 128u + ocol + k];
"""

# MOE_EXL3_FUSE (E3, on by default, 0 = as before): the dense path's Hadamard preps and finishes as
# one kernel each, every op of _proj's chain in its order, one simdgroup a (row, 128-block).
FUSE = os.environ.get("MOE_EXL3_FUSE", "1") != "0"
_UNIT = r"""
const uint lane = uint(thread_index_in_simdgroup);
// 8 simdgroups a threadgroup (_rows); threads_per_threadgroup is smaller in the grid's last one.
const uint unit = uint(threadgroup_position_in_grid.x) * 8u + uint(simdgroup_index_in_threadgroup);
constexpr uint NB = uint(DIM) / 128u;
const uint r = unit / NB;
const uint base = (unit % NB) * 128u;
const size_t sb = (size_t)uint(slots[loc[r]]) * (size_t)(DIM) + base;
const size_t ob = (size_t)r * (size_t)(DIM) + base;
const float sc = 0.08838834764831845f;
float4 v;
float s0, s1, s2, s3;
"""
_BFLY = _FWHT + r"""
  s0 = v.x + v.y;
  s1 = v.x - v.y;
  s2 = v.z + v.w;
  s3 = v.z - v.w;
"""
# gate and up inputs: rows sel of x, scaled by each projection's suh, transformed, rounded to f16
# (O out: f32 for gather_mm, f16 for E4's kernel).
PREP_GU_SRC = _UNIT + r"""
const size_t xb = (size_t)uint(sel[r]) * (size_t)(DIM) + base;
const float4 xv = float4(float(x[xb + lane]), float(x[xb + lane + 32u]), float(x[xb + lane + 64u]), float(x[xb + lane + 96u]));
for (uint pj = 0u; pj < 2u; pj++) {
  const device half *su = (pj == 0u) ? sug : suu;
  device O *o = (pj == 0u) ? xg : xu;
  v.x = xv.x * float(su[sb + lane]);
  v.y = xv.y * float(su[sb + lane + 32u]);
  v.z = xv.z * float(su[sb + lane + 64u]);
  v.w = xv.w * float(su[sb + lane + 96u]);
""" + _BFLY + r"""
  const float t0 = (s0 + s2) * sc;
  const float t1 = (s1 + s3) * sc;
  const float t2 = (s0 - s2) * sc;
  const float t3 = (s1 - s3) * sc;
  o[ob + lane] = O(half(t0));
  o[ob + lane + 32u] = O(half(t1));
  o[ob + lane + 64u] = O(half(t2));
  o[ob + lane + 96u] = O(half(t3));
}
"""
# gate and up outputs: transformed, scaled, times svh (f32 in, f32 out).
FIN_GU_SRC = _UNIT + r"""
for (uint pj = 0u; pj < 2u; pj++) {
  const device float *p = (pj == 0u) ? pg : pu;
  const device half *sv = (pj == 0u) ? svg : svu;
  device float *o = (pj == 0u) ? g : u;
  v = float4(p[ob + lane], p[ob + lane + 32u], p[ob + lane + 64u], p[ob + lane + 96u]);
""" + _BFLY + r"""
  const float t0 = (s0 + s2) * sc;
  const float t1 = (s1 + s3) * sc;
  const float t2 = (s0 - s2) * sc;
  const float t3 = (s1 - s3) * sc;
  o[ob + lane] = t0 * float(sv[sb + lane]);
  o[ob + lane + 32u] = t1 * float(sv[sb + lane + 32u]);
  o[ob + lane + 64u] = t2 * float(sv[sb + lane + 64u]);
  o[ob + lane + 96u] = t3 * float(sv[sb + lane + 96u]);
}
"""
# down input: h scaled by suh, transformed, rounded to f16.
PREP_D_SRC = _UNIT + r"""
v.x = h[ob + lane] * float(su[sb + lane]);
v.y = h[ob + lane + 32u] * float(su[sb + lane + 32u]);
v.z = h[ob + lane + 64u] * float(su[sb + lane + 64u]);
v.w = h[ob + lane + 96u] * float(su[sb + lane + 96u]);
""" + _BFLY + r"""
const float t0 = (s0 + s2) * sc;
const float t1 = (s1 + s3) * sc;
const float t2 = (s0 - s2) * sc;
const float t3 = (s1 - s3) * sc;
xd[ob + lane] = half(t0);
xd[ob + lane + 32u] = half(t1);
xd[ob + lane + 64u] = half(t2);
xd[ob + lane + 96u] = half(t3);
"""
# down output: f16 in, transformed, scaled, times svh, cast to the output dtype.
FIN_D_SRC = _UNIT + r"""
v = float4(float(p[ob + lane]), float(p[ob + lane + 32u]), float(p[ob + lane + 64u]), float(p[ob + lane + 96u]));
""" + _BFLY + r"""
const float t0 = (s0 + s2) * sc;
const float t1 = (s1 + s3) * sc;
const float t2 = (s0 - s2) * sc;
const float t3 = (s1 - s3) * sc;
y[ob + lane] = T(t0 * float(sv[sb + lane]));
y[ob + lane + 32u] = T(t1 * float(sv[sb + lane + 32u]));
y[ob + lane + 64u] = T(t2 * float(sv[sb + lane + 64u]));
y[ob + lane + 96u] = T(t3 * float(sv[sb + lane + 96u]));
"""


# MOE_EXL3_F16MM (E4, on by default with FUSE, 0 = as before): the gate and up GEMMs on f16 inputs
# through the installed MLX's own gather_mm_rhs with only its input type changed, checked against
# mx.gather_mm at the first use (_f16mm_ok) and off for good if one bit differs.
F16MM = os.environ.get("MOE_EXL3_F16MM", "1") != "0"
_F16MM = {}


def _f16mm_source():
    """(header, body, layout) of the installed MLX's gather_mm_rhs patched to f16 A and B with an
    f32 C, or None where its layout is neither "walk" (rhs_indices walked per row tile, 0.32.2)
    nor "offsets" (each group's first row, tiles within a group, 0.32.3)."""
    import re
    inc = Path(mx.__file__).parent / "include"
    k = inc / "mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_gather.h"
    seen = set()

    def inline(rel):
        out = []
        for line in (inc / rel).read_text().splitlines():
            m = re.match(r'\s*#include\s+"(.+)"', line)
            if m:
                if m.group(1) not in seen:
                    seen.add(m.group(1))
                    out.append(inline(m.group(1)))
            elif line.strip() != "#pragma once":
                out.append(line)
        return "\n".join(out)
    try:
        src = k.read_text()
        header = inline("mlx/backend/metal/kernels/steel/gemm/gemm.h") + "\nusing namespace mlx::steel;\n"
    except OSError:
        return None
    i = src.find("void gather_mm_rhs(")
    sig = src[i:src.find("{", i)]
    if i < 0:
        return None
    if "const device uint32_t* rhs_indices" in sig:
        layout = "walk"
    elif "const device int32_t* offsets" in sig and "const constant int& num_groups" in sig:
        layout = "offsets"
    else:
        return None
    j = src.find("{", src.find(")", src.find("[[threadgroup_position_in_grid]]", i)))
    depth, e = 0, j
    for e in range(j, len(src)):
        depth += {"{": 1, "}": -1}.get(src[e], 0)
        if depth == 0:
            break
    body, n = re.subn(r"GEMMKernel<\s*T,\s*T,", "GEMMKernel<T, float,", src[j + 1:e])
    if n != 1:
        return None
    body = body.replace("params->", "prm.").replace("num_groups,", "dims[10],")
    pre = r"""
typedef half T;
typedef float AccumType;
constexpr int BM = 16, BN = 64, BK = 16, WM = 1, WN = 2;
constexpr bool transpose_a = false, transpose_b = false;
constexpr bool align_M = AM, align_N = AN, align_K = AK;
const uint simd_lane_id = uint(thread_index_in_simdgroup);
const uint simd_group_id = uint(simdgroup_index_in_threadgroup);
const uint3 tid = uint3(threadgroup_position_in_grid);
const GEMMParams prm = {dims[0], dims[1], dims[2], dims[3], dims[4], dims[5], dims[6], dims[7],
                        0, int64_t(dims[8]), 0, 0, dims[9], 0};
"""
    return header, pre + body, layout


def _gather_f16(x, q, loc):
    """gather_mm(x[:, None, :], q, rhs_indices=loc, sorted_indices=True)[:, 0] in f32 for f16 x
    [R, K] and q [c, K, N] (MLX's dispatch: bm 16, bn 64, bk 16, wm 1, wn 2)."""
    R, K = x.shape
    c, _, N = q.shape
    header, body, layout = _F16MM["src"]
    tn, tm = -(-N // 64), -(-R // 16)
    if layout == "offsets":
        # MLX 0.32.3's gather_mm_offsets: each group's first row; a row tile never spans two groups.
        tm = min(R, tm + c - 1)
        # Padded: metal_kernel passes a short input in `constant` space, where schedule_row_tile
        # reads device.
        idx = np.zeros(max(c, 64), np.int32)
        idx[:c] = np.searchsorted(np.array(loc), np.arange(c))
        idx = mx.array(idx)
    else:
        idx = loc
    dims = mx.array([R, N, K, K, N, N, tn, tm, K * N, K // 16, c], dtype=mx.int32)
    kern = _F16MM.get("k")
    if kern is None:
        kern = _F16MM["k"] = mx.fast.metal_kernel(
            name="exl3_gather_f16",
            input_names=["A", "B", "rhs_indices" if layout == "walk" else "offsets", "dims"],
            output_names=["C"], source=body, header=header)
    return kern(inputs=[x, q, idx, dims],
                template=[("AM", R % 16 == 0), ("AN", N % 64 == 0), ("AK", K % 16 == 0)],
                grid=(32 * tn, 2 * tm, 1), threadgroup=(32, 2, 1),
                output_shapes=[(R, N)], output_dtypes=[mx.float32])[0]


_CHECKS = {}


def _checked(key, what, ref, check, applies=True):
    """Run check() once and keep its verdict: a lever stays on only if it matched `ref` bit for
    bit."""
    if key not in _CHECKS:
        _CHECKS[key] = False        # the check's reference takes today's path
        try:
            ok, why = check()
        except Exception as e:      # a kernel that does not build here
            ok, why = False, f"{type(e).__name__}: {str(e)[:200]}"
        _CHECKS[key] = ok
        if applies:
            note(f"  exl3: {what} {'on: bit-equal to the ' + ref if ok else 'off: ' + why}", flush=True)
    return _CHECKS[key]


def _f16mm_ok():
    """E4's kernel checked once: bit-equal to mx.gather_mm's f32 call on random f16-exact rows at
    the pack's shapes and uneven routings, else off for good."""

    def check():
        src = None if _CUDA else _f16mm_source()
        if src is None:
            return False, "this MLX's gather_mm_rhs is not the layout patched here"
        _F16MM["src"] = src
        rng = np.random.default_rng(1)
        for R, K, N, c in ((7, 256, 128, 3), (300, 2560, 640, 8), (2501, 2560, 640, 8), (40, 640, 2560, 2)):
            x = mx.array(rng.standard_normal((R, K)).astype(np.float16))
            q = mx.array((rng.standard_normal((c, K, N)) * K ** -0.5).astype(np.float16))
            loc = mx.array(np.sort(rng.integers(0, c, R)).astype(np.uint32))
            ref = mx.gather_mm(x.astype(mx.float32)[:, None, :], q.astype(mx.float32), rhs_indices=loc,
                               sorted_indices=True)[:, 0]
            if not bool(mx.array_equal(ref, _gather_f16(x, q, loc))):
                return False, f"differs from mx.gather_mm at {R}x{K}x{N}"
        return True, ""

    return _checked("f16mm", "f16 gate/up GEMM", "f32 gather_mm", check, applies=not _CUDA)


def _rows(name, ins, outs, src, inputs, dim, out_dtypes, template=()):
    """One of the fused kernels over every (row, 128-block) of loc's rows."""
    rows = inputs[ins.index("loc")].size
    return _kernel(name, ins, outs, src)(
        inputs=inputs, template=[("DIM", dim), *template],
        grid=(rows * (dim // 128) * 32, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(rows, dim)] * len(outs), output_dtypes=out_dtypes)


def _tc_perm_inv():
    """exllamav3's tensor-core order of a 16x16 tile, inverted: stream position of (in, out) entry
    j."""
    p = []
    for t in range(32):
        r0, c0 = (t % 4) * 2, t // 4
        for c in (c0, c0 + 8):
            p += [r0 * 16 + c, (r0 + 1) * 16 + c, (r0 + 8) * 16 + c, (r0 + 9) * 16 + c]
    return np.argsort(np.array(p)).astype(np.uint16)


_PERM_I = None


def _had(v):
    """128-wide Hadamard (Sylvester order, orthonormal) over the last axis, as a butterfly: a f32
    matmul by H loses 5e-4 of relative precision (MLX 0.32.3)."""
    n = v.shape[-1]
    return mx.hadamard_transform(v.reshape(-1, n // 128, 128), scale=128 ** -0.5).reshape(v.shape)


def dequant(t, slots, win, dtype):
    """Q [len(slots), in, out] of the trellis tensor t [E, in/16, out/16, n] at rows `slots`."""
    _, it, ot, n = t.shape
    two = ot % 8 == 0 and (DEQ2 and not _CUDA or
                           CDEQ and _CUDA and _cuda_ok("deq", max(it, ot) * 16, min(it, ot) * 16, n, win))
    return _deq(t, slots, win, dtype, two)


def _deq(t, slots, win, dtype, two):
    """dequant() through DEQ2_SRC (two) or DEQ_SRC: the same Q."""
    global _PERM_I
    if _PERM_I is None:
        _PERM_I = mx.array(_tc_perm_inv())
    _, it, ot, n = t.shape
    c = slots.size
    if two:
        return _kernel("exl3_dequant2", ["trellis", "slots"], ["q"], DEQ2_SRC)(
            inputs=[t, slots],
            template=[("T", dtype), ("NHW", n), ("IDIM", it * 16), ("ODIM", ot * 16), ("WIN", win)],
            grid=(c * it * (ot // 8) * 256, 1, 1), threadgroup=(256, 1, 1),
            output_shapes=[(c, it * 16, ot * 16)], output_dtypes=[dtype])[0]
    return _kernel("exl3_dequant", ["trellis", "slots", "perm"], ["q"], DEQ_SRC)(
        inputs=[t, slots, _PERM_I],
        template=[("T", dtype), ("NHW", n), ("IDIM", it * 16), ("ODIM", ot * 16), ("WIN", win)],
        grid=(c * it * ot * 256, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(c, it * 16, ot * 16)], output_dtypes=[dtype])[0]


def gemv(x, w, slots, topk, win, out_dtype, skip=False):
    """Every slot's expert output, unweighted: x (rows, hidden) and slots (rows * topk,) uint32
    with slot s on row s // topk, w three (trellis, suh, svh) for gate, up and down, returns
    (rows * topk, hidden); skip (CUDA): a slot 0xFFFFFFFF is left out and its output row is garbage."""
    (tg, sug, svg), (tu, suu, svu), (td, sud, svd) = w
    hid = x.shape[-1]
    nslots = slots.size
    inter = tg.shape[2] * 16
    n = tg.shape[-1]
    funnel = n < 64

    def otpt(out_dim):
        return 2 if funnel and (out_dim // 16) % 2 == 0 else 1

    nsplit = 2 if hid % 256 == 0 else 1
    t = [("NHW", n), ("FUNNEL", int(funnel)), ("WIN", win)]
    o1 = otpt(inter)
    sfx = "_skip" if skip else ""
    assert not (skip and (PREP or not _CUDA)), "skip slots: the CUDA chain only"
    if PREP:
        return _gemv_prep(x, w, slots, topk, out_dtype, hid, inter, nslots, nsplit, t, o1, otpt(hid))
    ig, iu = _kernel("exl3_pair_gemv" + sfx, ["x", "suhg", "suhu", "tg", "tu", "slots"], ["yg", "yu"], PAIR_SRC)(
        inputs=[x, sug, suu, tg, tu, slots],
        template=[("IDIM", hid), ("ODIM", inter), ("NSPLIT", nsplit), ("TOPK", topk), ("OTPT", o1)] + t,
        grid=(inter // 16 // o1 * 128, nslots, nsplit), threadgroup=(128, 1, 1),
        output_shapes=[(nslots * nsplit, inter)] * 2, output_dtypes=[mx.float32] * 2)
    o2 = otpt(hid)
    d = _kernel("exl3_down_fused" + sfx, ["ig", "iu", "trellis", "svhg", "svhu", "suhd", "slots"], ["y"], DOWN_SRC)(
        inputs=[ig, iu, td, svg, svu, sud, slots],
        template=[("IDIM", inter), ("ODIM", hid), ("NSPLIT", nsplit), ("OTPT", o2)] + t,
        grid=(hid // 16 // o2 * 128, nslots, 1), threadgroup=(128, 1, 1),
        output_shapes=[(nslots, hid)], output_dtypes=[mx.float16])[0]
    # The finish at top-k 1 and weight 1.0 is sushi's reduce per slot; the caller weights and sums.
    return _kernel("exl3_reduce" + sfx, ["inner", "svh", "slots", "sc"], ["y"], REDUCE_SRC)(
        inputs=[d, svd, slots, mx.ones((nslots,), dtype=mx.float32)],
        template=[("T", out_dtype), ("ODIM", hid), ("TOPK", 1)],
        grid=(32 * (hid // 128), nslots, 1), threadgroup=(32, 1, 1),
        output_shapes=[(nslots, hid)], output_dtypes=[out_dtype])[0]


_ONES = {}


def _gemv_prep(x, w, slots, topk, out_dtype, hid, inter, nslots, nsplit, t, o1, o2):
    """gemv() under MOE_GLUE: PREP_IN, PAIR2, DOWN_PREP, DOWN2, REDUCE (bit-equal to gemv's chain)."""
    (tg, sug, svg), (tu, suu, svu), (td, sud, svd) = w
    prep = _kernel("exl3_prep_in", ["x", "suhg", "suhu", "slots"], ["prep"], PREP_IN_SRC)(
        inputs=[x, sug, suu, slots], template=[("IDIM", hid), ("TOPK", topk)],
        grid=(32 * (hid // 128), nslots, 2), threadgroup=(32, 1, 1),
        output_shapes=[(nslots * 2, hid)], output_dtypes=[mx.float16])[0]
    ig, iu = _kernel("exl3_pair2", ["prep", "tg", "tu", "slots"], ["yg", "yu"], PAIR2_SRC)(
        inputs=[prep, tg, tu, slots],
        template=[("IDIM", hid), ("ODIM", inter), ("NSPLIT", nsplit), ("OTPT", o1)] + t,
        grid=(inter // 16 // o1 * 128, nslots, nsplit), threadgroup=(128, 1, 1),
        output_shapes=[(nslots * nsplit, inter)] * 2, output_dtypes=[mx.float32] * 2)
    prepd = _kernel("exl3_down_prep", ["ig", "iu", "svhg", "svhu", "suhd", "slots"], ["prepd"], DOWN_PREP_SRC)(
        inputs=[ig, iu, svg, svu, sud, slots], template=[("IDIM", inter), ("NSPLIT", nsplit)],
        grid=(32 * (inter // 128), nslots, 1), threadgroup=(32, 1, 1),
        output_shapes=[(nslots, inter)], output_dtypes=[mx.float16])[0]
    d = _kernel("exl3_down2", ["prepd", "trellis", "slots"], ["y"], DOWN2_SRC)(
        inputs=[prepd, td, slots],
        template=[("IDIM", inter), ("ODIM", hid), ("NSPLIT", nsplit), ("OTPT", o2)] + t,
        grid=(hid // 16 // o2 * 128, nslots, 1), threadgroup=(128, 1, 1),
        output_shapes=[(nslots, hid)], output_dtypes=[mx.float16])[0]
    ones = _ONES.get(nslots)
    if ones is None:            # computed with its first use, then kept (decode passes: one size)
        ones = mx.ones((nslots,), dtype=mx.float32)
        if nslots <= 64:
            _ONES[nslots] = ones
    return _kernel("exl3_reduce", ["inner", "svh", "slots", "sc"], ["y"], REDUCE_SRC)(
        inputs=[d, svd, slots, ones],
        template=[("T", out_dtype), ("ODIM", hid), ("TOPK", 1)],
        grid=(32 * (hid // 128), nslots, 1), threadgroup=(32, 1, 1),
        output_shapes=[(nslots, hid)], output_dtypes=[out_dtype])[0]


# Experts decoded at once on the dense path: ~16 MB of Q an expert at Flash-Next's shapes.
DENSE_CHUNK = 8

# E1 (MOE_EXL3_KEEP_MB) is closed (docs/CLOSED.md), and stream_fast's layer-major hooks still call
# both.
KEEP_MB = 0.0


def pass_begin():
    """A layer-major pass on EXL3 experts starts, and E1 is closed, so MLX's cache limit stands."""


def pass_end():
    """A layer-major pass ends, and E1 is closed, so nothing is flushed."""


def _proj(x, w, slots, loc, win, dtype):
    """svh * H(Q (H(suh * x))) for rows x (f32) of experts slots[loc], the input rounded to f16 and
    the product's plane kept in `dtype` as the decode chain keeps them (f32 gate/up, f16 down)."""
    t, su, sv = w
    xin = _had(x * su[slots][loc]).astype(mx.float16).astype(dtype)
    q = dequant(t, slots, win, dtype)
    # MLX #4727: on CUDA the sorted gather_mm is wrong when x's size-1 axis is an expand_dims view,
    # not a reshape.
    xin = xin.reshape(xin.shape[0], 1, xin.shape[1]) if _CUDA else xin[:, None, :]
    p = mx.gather_mm(xin, q, rhs_indices=loc, sorted_indices=True)[:, 0]
    return sv[slots][loc].astype(mx.float32) * _had(p.astype(mx.float32))


def _dense_fused(x, w, sel, slots, loc, win, out_dtype):
    """dense()'s chunk with the preps and finishes fused (FUSE): the same GEMMs on the same values."""
    (tg, sug, svg), (tu, suu, svu), (td, sud, svd) = w
    hid, inter = x.shape[-1], tg.shape[2] * 16
    f16 = F16MM and _f16mm_ok()
    dt = mx.float16 if f16 else mx.float32
    xg, xu = _rows("exl3_prep_gu", ["x", "sel", "loc", "slots", "sug", "suu"], ["xg", "xu"], PREP_GU_SRC,
                   [x, sel, loc, slots, sug, suu], hid, [dt] * 2, [("O", dt)])
    if f16:
        pg = _gather_f16(xg, dequant(tg, slots, win, dt), loc)
        pu = _gather_f16(xu, dequant(tu, slots, win, dt), loc)
    else:
        pg = mx.gather_mm(xg[:, None, :], dequant(tg, slots, win, dt), rhs_indices=loc, sorted_indices=True)[:, 0]
        pu = mx.gather_mm(xu[:, None, :], dequant(tu, slots, win, dt), rhs_indices=loc, sorted_indices=True)[:, 0]
    g, u = _rows("exl3_fin_gu", ["pg", "pu", "loc", "slots", "svg", "svu"], ["g", "u"], FIN_GU_SRC,
                 [pg, pu, loc, slots, svg, svu], inter, [mx.float32] * 2)
    xd = _rows("exl3_prep_d", ["h", "loc", "slots", "su"], ["xd"], PREP_D_SRC,
               [g * mx.sigmoid(g) * u, loc, slots, sud], inter, [mx.float16])[0]
    pd = mx.gather_mm(xd[:, None, :], dequant(td, slots, win, mx.float16), rhs_indices=loc, sorted_indices=True)
    return _rows("exl3_fin_d", ["p", "loc", "slots", "sv"], ["y"], FIN_D_SRC,
                 [pd[:, 0], loc, slots, svd], hid, [out_dtype], [("T", out_dtype)])[0]


def _chunk(x, w, sel, slots, loc, win, out_dtype):
    """dense()'s chunk as MLX ops: rows sel of x through experts slots[loc]."""
    xs = x[sel].astype(mx.float32)
    g = _proj(xs, w[0], slots, loc, win, mx.float32)
    u = _proj(xs, w[1], slots, loc, win, mx.float32)
    return _proj(g * mx.sigmoid(g) * u, w[2], slots, loc, win, mx.float16).astype(out_dtype)


# MOE_EXL3_CDEQ / MOE_EXL3_CFUSE (CUDA, on by default, 0 = as before): the dense path's dequant and
# its preps, finishes and SwiGLU as exl3_cuda's kernels, each checked against the MLX path at first
# use (_cuda_ok) and off for good if one bit differs.
CDEQ = os.environ.get("MOE_EXL3_CDEQ", "1") != "0"
CFUSE = os.environ.get("MOE_EXL3_CFUSE", "1") != "0"
# MOE_EXL3_CMM (with CFUSE, on by default): the gate and up GEMMs on f16 rows and Q
# (exl3_cuda.GMM16_SRC), the f32 gather_mm's bits from half the bytes.
CMM = os.environ.get("MOE_EXL3_CMM", "1") != "0"


def _gmm16(xg, xu, qg, qu, counts):
    """gather_mm(x, q, rows of expert e = counts[e] in order) in f32 for f16 x [R, K] and q [c, K,
    N], gate and up in one call."""
    R, K = xg.shape
    N = qg.shape[-1]
    ends = np.cumsum(counts)
    tiles = [(e, m, min(m + 64, end)) for e, (start, end) in enumerate(zip(ends - counts, ends))
             for m in range(int(start), int(end), 64)]
    return _kernel("exl3_gmm16", ["ag", "au", "qg", "qu", "tiles"], ["cg", "cu"], None)(
        inputs=[xg, xu, qg, qu, mx.array(np.array(tiles, np.int32).reshape(-1))],
        template=[("KD", K), ("ND", N)], grid=(len(tiles) * 128, N // 128, 2), threadgroup=(128, 1, 1),
        output_shapes=[(R, N)] * 2, output_dtypes=[mx.float32] * 2)


def _dense_cuda(x, w, sel, slots, loc, win, out_dtype, counts=None):
    """_chunk on CUDA with the preps, finishes and SwiGLU fused on the same GEMMs and values, the
    gate and up GEMMs on f16 when given the rows an expert (counts)."""
    (tg, sug, svg), (tu, suu, svu), (td, sud, svd) = w
    hid, inter, R = x.shape[-1], tg.shape[2] * 16, loc.size
    mm = counts is not None
    dt = mx.float16 if mm else mx.float32
    xg, xu = _rows("exl3_cprep_gu", ["x", "sel", "loc", "slots", "sug", "suu"], ["xg", "xu"], None,
                   [x, sel, loc, slots, sug, suu], hid, [dt] * 2, [("O", dt)])
    if mm:
        pg, pu = _gmm16(xg, xu, dequant(tg, slots, win, dt), dequant(tu, slots, win, dt), counts)
    else:
        pg = mx.gather_mm(xg.reshape(R, 1, hid), dequant(tg, slots, win, dt), rhs_indices=loc, sorted_indices=True)
        pu = mx.gather_mm(xu.reshape(R, 1, hid), dequant(tu, slots, win, dt), rhs_indices=loc, sorted_indices=True)
    xd = _rows("exl3_cfin_gu_d", ["pg", "pu", "loc", "slots", "svg", "svu", "sud"], ["xd"], None,
               [pg.reshape(R, inter), pu.reshape(R, inter), loc, slots, svg, svu, sud], inter, [mx.float16])[0]
    pd = mx.gather_mm(xd.reshape(R, 1, inter), dequant(td, slots, win, mx.float16), rhs_indices=loc,
                      sorted_indices=True)
    return _rows("exl3_cfin_d", ["p", "loc", "slots", "sv"], ["y"], None, [pd.reshape(R, hid), loc, slots, svd],
                 hid, [out_dtype], [("T", out_dtype)])[0]


def _same(a, b):
    """Bit for bit (-0 against +0 and NaN payloads too)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    u = {4: mx.uint32, 2: mx.uint16}[a.dtype.size]
    return bool(mx.array_equal(a.view(u), b.view(u)))


_CUDA_WHAT = {"deq": "CUDA dequant", "fuse": "CUDA fused preps / finishes", "mm": "CUDA f16 gate/up GEMM"}


def _cuda_ok(name, hid, inter, n, win):
    """CUDA lever name ("deq", "fuse" or "mm"), checked once against today's path at these shapes."""

    def check():
        rng = np.random.default_rng(3)
        w = []
        for din, dout in ((hid, inter), (hid, inter), (inter, hid)):
            w.append((mx.array(rng.integers(0, 1 << 16, (12, din // 16, dout // 16, n), dtype=np.uint16)),
                      mx.array((rng.standard_normal((12, din)) * din ** -0.5).astype(np.float16)),
                      mx.array((1 + 0.1 * rng.standard_normal((12, dout))).astype(np.float16))))
        slots = mx.array(np.array([5, 0, 11, 3, 7, 2, 9, 1], np.uint32))
        if name == "deq":
            for t, _, _ in w:
                for dt in (mx.float32, mx.float16):
                    if not _same(_deq(t, slots, win, dt, False), _deq(t, slots, win, dt, True)):
                        return False, f"differs at {t.shape} {dt}"
            return True, ""
        for R, c, scale in ((7, 3, 1.0), (300, 8, 1.0), (2501, 8, 1.0), (640, 8, 64.0)):
            x = mx.array((rng.standard_normal((R + 5, hid)) * scale).astype(np.float32)).astype(mx.bfloat16)
            sel = mx.array(rng.permutation(R + 5)[:R].astype(np.int32))
            host = np.sort(np.concatenate([np.arange(c), rng.integers(0, c, R - c)])).astype(np.uint32)
            loc, counts = mx.array(host), np.bincount(host, minlength=c)
            if name == "mm":
                # The GEMMs alone on the gate's and up's Q, then the whole chunk.
                xs = [mx.array((rng.standard_normal((R, hid)) * scale).astype(np.float16)) for _ in range(2)]
                qs = [_deq(t, slots[:c], win, mx.float16, False) for t, _, _ in w[:2]]
                got = _gmm16(*xs, *qs, counts)
                for xi, qi, gi in zip(xs, qs, got):
                    ref = mx.gather_mm(xi.astype(mx.float32).reshape(R, 1, hid), qi.astype(mx.float32),
                                       rhs_indices=loc, sorted_indices=True).reshape(R, -1)
                    if not _same(ref, gi):
                        return False, f"the GEMM differs at {R} rows, {c} experts, x * {scale}"
            for dt in (mx.float32, mx.bfloat16):
                if not _same(_chunk(x, w, sel, slots[:c], loc, win, dt),
                             _dense_cuda(x, w, sel, slots[:c], loc, win, dt, counts if name == "mm" else None)):
                    return False, f"differs at {R} rows, {c} experts, x * {scale}, {dt}"
        return True, ""

    return _checked(name, _CUDA_WHAT[name], "MLX path", check)


def dense(x, w, host, win, out_dtype):
    """Every row's expert output, unweighted: x (rows, hidden) and host (rows,) numpy slot per row
    in any order, DENSE_CHUNK experts at a time with their rows sorted by slot."""
    order = np.argsort(host, kind="stable")
    hs = host[order]
    uniq, first = np.unique(hs, return_index=True)
    bounds = list(first) + [len(hs)]
    shapes = (x.shape[-1], w[0][0].shape[2] * 16, w[0][0].shape[-1], win)      # hidden, expert width, n, window
    y = None
    for a in range(0, len(uniq), DENSE_CHUNK):
        b = min(a + DENSE_CHUNK, len(uniq))
        r0, r1 = bounds[a], bounds[b]
        sel = mx.array(order[r0:r1].astype(np.int32))
        slots = mx.array(uniq[a:b].astype(np.uint32))
        loc = mx.array(np.searchsorted(uniq[a:b], hs[r0:r1]).astype(np.uint32))
        if FUSE and not _CUDA:
            part = _dense_fused(x, w, sel, slots, loc, win, out_dtype)
        elif CFUSE and _CUDA and _cuda_ok("fuse", *shapes):
            mm = CMM and shapes[0] % 32 == 0 and shapes[1] % 128 == 0 and _cuda_ok("mm", *shapes)
            part = _dense_cuda(x, w, sel, slots, loc, win, out_dtype, np.diff(bounds[a:b + 1]) if mm else None)
        else:
            part = _chunk(x, w, sel, slots, loc, win, out_dtype)
        if y is None:
            y = mx.zeros((len(hs), x.shape[-1]), dtype=out_dtype)
        y[sel] = part
        if b < len(uniq):
            mx.eval(y)          # one chunk's Q at a time
    return y


# Per-device costs of the two paths, as calibrate() measures them: chain fixed s, chain s a row,
# dense fixed s, dense s an expert, dense s a row; until then every pass takes the decode chain.
_COST = None
CALIB_DIR = Path.home() / ".cache" / "moe-ssd-offload"


def _measure(hid, inter, n, win):
    """The cost model's five terms, timed on random codewords at these shapes (the kernels have no
    data-dependent branches) as the best of 5 and fitted by least squares, the chain over three
    widths and the dense path over five (experts, rows) points."""
    rng = np.random.default_rng(0)
    E = 32
    w = []
    for din, dout in ((hid, inter), (hid, inter), (inter, hid)):
        w.append((mx.array(rng.integers(0, 1 << 16, (E, din // 16, dout // 16, n), dtype=np.uint16)),
                  mx.array(np.full((E, din), din ** -0.5, np.float16)), mx.array(np.ones((E, dout), np.float16))))
    x = mx.array(rng.standard_normal((80 * E, 1, hid)).astype(np.float32)).astype(mx.bfloat16)
    mx.eval(w, x)

    def secs(u, p, dense_path):
        host = (np.arange(p) % u).astype(np.uint32)
        f = (lambda: dense(x[:p, 0], w, host, win, x.dtype)) if dense_path else \
            (lambda: gemv(x[:p, 0], w, mx.array(host), 1, win, x.dtype))
        mx.eval(f())
        best = float("inf")
        for _ in range(5):
            t0 = time.perf_counter()
            mx.eval(f())
            best = min(best, time.perf_counter() - t0)
        return best

    def fit(rows, ts):
        return np.linalg.lstsq(np.array(rows, dtype=np.float64), np.array(ts), rcond=None)[0]

    chain = [(p, secs(E, p, False)) for p in (10, 10 * E, 80 * E)]
    g = fit([[1.0, p] for p, _ in chain], [t for _, t in chain])
    pts = [(u, p, secs(u, p, True)) for u, p in ((1, 10), (10, 100), (10, 800), (E, 10 * E), (E, 80 * E))]
    d = fit([[1.0, u, p] for u, p, _ in pts], [t for *_, t in pts])
    mx.synchronize()
    mx.clear_cache()
    return [max(float(v), 0.0) for v in (*g, *d)]


def calibrate(hid, inter, n, win):
    """This device's cost model at these shapes, measured once in about a second and cached like
    chunkattn's routes, always on the path without the prefill levers, whose switch their
    timings would move."""
    global _COST, DEQ2, FUSE, F16MM, PREP, CDEQ, CFUSE, CMM
    key = json.dumps([mx.device_info()["device_name"], mx.__version__,
                      hashlib.sha1(Path(__file__).read_bytes()).hexdigest(), hid, inter, n, win])
    path = CALIB_DIR / f"exl3-{hashlib.sha1(key.encode()).hexdigest()[:16]}.json"
    try:
        _COST = json.loads(path.read_text())["cost"]
        note(f"  exl3: prefill switch reused from {path}", flush=True)
    except (OSError, ValueError, KeyError):
        flags = DEQ2, FUSE, F16MM, PREP, CDEQ, CFUSE, CMM
        DEQ2 = FUSE = F16MM = PREP = CDEQ = CFUSE = CMM = False     # every lever off: the switch is measured on today's kernels
        try:
            _COST = _measure(hid, inter, n, win)
        finally:
            DEQ2, FUSE, F16MM, PREP, CDEQ, CFUSE, CMM = flags
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"cost": _COST}))
            tmp.replace(path)
        except OSError as e:
            print(f"  exl3: could not save the prefill switch to {path}: {e}", flush=True)
    if F16MM:
        _f16mm_ok()     # its check's arrays here, not inside a pass the budget measures
    for name, on in (("deq", CDEQ), ("fuse", CFUSE), ("mm", CFUSE and CMM)):
        if on and _CUDA:
            _cuda_ok(name, hid, inter, n, win)
    _, g1, _, de, d1 = _COST
    ratio = de / (g1 - d1) if g1 > d1 else float("inf")
    note(f"  exl3: chain {g1 * 1e6:.1f} us a row, dense {de * 1e3:.2f} ms an expert + {d1 * 1e6:.1f} us a row: "
          f"dense from ~{ratio:.0f} rows an expert", flush=True)
    return _COST


def dense_pays(u, p):
    """Whether dense() beats gemv() on p rows of u experts, by this device's cost model."""
    c = _COST
    return c is not None and c[2] + c[3] * u + c[4] * p < c[0] + c[1] * p


def experts(x, w, rhs, win, host=None, skip=False):
    """The routed experts' outputs as the gather_qmm chain gives them, x (..., 1, hidden) broadcast
    against rhs (...) slot ids, returning rhs.shape + (1, hidden) in x's dtype, with `host` (rhs
    on the host, one row of x a slot) letting dense_pays() pick the path over the decode chain
    (gemv), and skip: gemv's skipped slots."""
    hid = x.shape[-1]
    x2 = x.reshape(-1, hid)
    if host is not None:
        host = np.asarray(host).reshape(-1)
        assert host.size == x2.shape[0] == rhs.size, "dense: one row of x per slot"
        if dense_pays(len(np.unique(host)), host.size):
            return dense(x2, w, host, win, x.dtype).reshape(tuple(rhs.shape) + (1, hid))
    y = gemv(x2, w, rhs.reshape(-1).astype(mx.uint32), rhs.size // x2.shape[0], win, x.dtype, skip)
    return y.reshape(tuple(rhs.shape) + (1, hid))
