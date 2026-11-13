"""Routed experts in llama.cpp's GGUF block formats (an unsloth pack's experts beside our trunk): the engine's
gathers on Metal and CUDA, decoding as gguf_ref.ref_dequant does."""
# The dequant is ggml's dequantize_row_* (ggml-quants.c, llama.cpp b1c902f0, MIT; THIRD_PARTY_NOTICES.md).
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

import mlx.core as mx

from gguf_ref import (TYPES, NAME_OF, KVALUES_IQ4NL, IQ3XXS_GRID, IQ2S_GRID, IQ2XXS_GRID, IQ2XS_GRID,  # noqa: F401
                      IQ3S_GRID, IQ1S_GRID, KVALUES_FP4, HALVES, ref_dequant, random_experts)

# ---- GPU kernels: one source for Metal and CUDA, over a backend prelude (gguf_metal.py, gguf_cuda.py) that
# defines GPTR, LOCAL, FN, CFN, H2F, SHFL, POPC, EXP, MUL/ADD/SUB (rounded alone, never fused) and the
# indices _tid, _bid, _gid.

# Type traits, a block's 32 or 256 weights decoded 8 at a time ("tasks"): a task's weights 0-3 go with
# x[i0 + 0..3] and 4-7 with x[i0 + gg_run(T) + 0..3].
HEADER = r"""
CFN uint32_t gg_be(int T) {
  return (T == 12 || T == 13 || T == 18 || T == 22 || T == 23 || T == 10 || T == 11 || T == 14 || T == 16 ||
          T == 17 || T == 19 || T == 21 || T == 29 || T == 34 || T == 35) ? 256u :
         T == 41 ? 128u : (T == 42 || T == 40) ? 64u : (T == 0 || T == 1 || T == 30 || T == 43) ? 1u : 32u;
}
CFN uint32_t gg_bb(int T) {
  return T == 3 ? 20u : T == 6 ? 22u : T == 7 ? 24u : T == 8 ? 34u : T == 12 ? 144u : T == 13 ? 176u :
         T == 18 ? 98u : T == 22 ? 82u : T == 23 ? 136u : T == 10 ? 84u : (T == 11 || T == 21) ? 110u :
         T == 14 ? 210u : (T == 16 || T == 35) ? 66u : T == 17 ? 74u : T == 19 ? 50u : T == 29 ? 56u :
         T == 34 ? 54u : T == 39 ? 17u : T == 40 ? 36u : T == 0 ? 4u : (T == 1 || T == 30) ? 2u : T == 43 ? 1u : 18u;
}
CFN uint32_t gg_run(int T) {
  return (T == 12 || T == 13) ? 32u : (T == 2 || T == 3 || T == 6 || T == 7 || T == 20 || T == 23) ? 16u : 4u;
}
CFN uint32_t gg_row(int T, uint32_t n) { return n / gg_be(T) * gg_bb(T); }

FN uint32_t gg_ld16(GPTR(uint8_t) p) { return uint32_t(*((GPTR(uint16_t))(p))); }
FN uint32_t gg_ld32(GPTR(uint8_t) p) { return gg_ld16(p) | (gg_ld16(p + 2) << 16u); }
FN uint32_t gg_ld32a(GPTR(uint8_t) p) { return *((GPTR(uint32_t))(p)); }
FN float gg_h(GPTR(uint8_t) p) { return H2F(gg_ld16(p)); }
FN int gg_kv(GPTR(uint32_t) tab, uint32_t i) { return int(tab[i]); }
FN float gg_sgn(float v, uint32_t s, uint32_t bit) { return ((s >> bit) & 1u) ? -v : v; }

// ggml's get_scale_min_k4 for sub-blocks 2c and 2c + 1 of the 12 scale bytes, read as words s0, s1, s2.
FN uint32_t gg_byte(uint32_t w, uint32_t j) { return (w >> (8u * j)) & 0xFFu; }
FN void gg_scale_min(uint32_t s0, uint32_t s1, uint32_t s2, uint32_t c, LOCAL uint32_t *sc0, LOCAL uint32_t *m0,
                     LOCAL uint32_t *sc1, LOCAL uint32_t *m1) {
  if (c < 2u) {
    *sc0 = gg_byte(s0, 2u * c) & 63u;
    *m0 = gg_byte(s1, 2u * c) & 63u;
    *sc1 = gg_byte(s0, 2u * c + 1u) & 63u;
    *m1 = gg_byte(s1, 2u * c + 1u) & 63u;
  } else {
    const uint32_t j0 = 2u * c - 4u, j1 = j0 + 1u;
    *sc0 = (gg_byte(s2, j0) & 0xFu) | ((gg_byte(s0, j0) >> 6u) << 4u);
    *m0 = (gg_byte(s2, j0) >> 4u) | ((gg_byte(s1, j0) >> 6u) << 4u);
    *sc1 = (gg_byte(s2, j1) & 0xFu) | ((gg_byte(s0, j1) >> 6u) << 4u);
    *m1 = (gg_byte(s2, j1) >> 4u) | ((gg_byte(s1, j1) >> 6u) << 4u);
  }
}

// Table offsets in tab (uint32 words): kvalues_iq4nl, iq3xxs_grid, iq2s_grid / iq2xxs_grid / iq2xs_grid / iq1s_grid
// as (low, high) word pairs, iq3s_grid, kvalues_fp4.
#define TB_IQ2XXS 2320u
#define TB_IQ2XS 2832u
#define TB_IQ3S 3856u
#define TB_IQ1S 4368u
#define TB_FP4 8464u
FN uint32_t gg_parity(uint32_t s7) { return s7 | ((POPC(s7) & 1u) << 7u); }
FN float gg_e4m3(uint32_t e, uint32_t m, uint32_t bias) {      // a normal or subnormal E4M3 magnitude times 2^(7 - bias)
  return e == 0u ? MUL(float(m), U2F((127u - bias - 2u) << 23u)) : U2F(((e + 127u - bias) << 23u) | (m << 20u));
}

// The types whose task k is weights 8k .. 8k + 7 of the row, in order (all those added after the first twelve).
template <int T>
FN uint32_t gg_task8(GPTR(uint8_t) row, uint32_t k, LOCAL float *w, GPTR(uint32_t) tab) {
  const uint32_t e0 = 8u * k, o = e0 % gg_be(T);
  GPTR(uint8_t) b = row + (e0 / gg_be(T)) * gg_bb(T);
  if (T == 10 || T == 11 || T == 14) {    // Q2_K, Q3_K, Q6_K: weight 128h + 32j + l
    const uint32_t h = o >> 7u, j = (o >> 5u) & 3u, l0 = o & 31u, s = 8u * h + 2u * j + (l0 >> 4u);
    if (T == 10) {                        // d * sc * q - dmin * m, 2 bits at 2j of qs[32h + l]
      const uint32_t sc = uint32_t(b[s]);
      const float dl = MUL(gg_h(b + 80), float(sc & 0xFu)), ml = MUL(gg_h(b + 82), float(sc >> 4u));
      const uint32_t q0 = gg_ld32a(b + 16u + 32u * h + l0), q1 = gg_ld32a(b + 20u + 32u * h + l0);
      for (uint32_t t = 0u; t < 4u; t++) {
        w[t] = SUB(MUL(dl, float((q0 >> (8u * t + 2u * j)) & 3u)), ml);
        w[4u + t] = SUB(MUL(dl, float((q1 >> (8u * t + 2u * j)) & 3u)), ml);
      }
    } else if (T == 11) {                 // d * (sc - 32) * (q - 4 where bit 4h + j of hmask[l] is clear)
      const uint32_t lo = s < 8u ? uint32_t(b[96u + s]) & 0xFu : uint32_t(b[88u + s]) >> 4u;
      const uint32_t hi = (uint32_t(b[104u + (s & 3u)]) >> (2u * (s >> 2u))) & 3u;
      const float dl = MUL(gg_h(b + 108), float(int(lo | (hi << 4u)) - 32));
      const uint32_t q0 = gg_ld32(b + 32u + 32u * h + l0), q1 = gg_ld32(b + 36u + 32u * h + l0);
      const uint32_t m0 = gg_ld32(b + l0), m1 = gg_ld32(b + 4u + l0), hb = 4u * h + j;
      for (uint32_t t = 0u; t < 4u; t++) {
        w[t] = MUL(dl, float(int((q0 >> (8u * t + 2u * j)) & 3u) - (((m0 >> (8u * t + hb)) & 1u) ? 0 : 4)));
        w[4u + t] = MUL(dl, float(int((q1 >> (8u * t + 2u * j)) & 3u) - (((m1 >> (8u * t + hb)) & 1u) ? 0 : 4)));
      }
    } else {                              // Q6_K: d * sc * (q - 32), here g = j
      const float ds = MUL(gg_h(b + 208), float(int(int8_t(b[192u + s]))));
      GPTR(uint8_t) pl = b + 64u * h + 32u * (j & 1u) + l0;
      const uint32_t l0w = gg_ld32(pl), l1w = gg_ld32(pl + 4), sh = 4u * (j >> 1u);
      const uint32_t h0 = gg_ld32(b + 128u + 32u * h + l0), h1 = gg_ld32(b + 132u + 32u * h + l0);
      for (uint32_t t = 0u; t < 4u; t++) {
        w[t] = MUL(ds, float(int(((l0w >> (8u * t + sh)) & 0xFu) | (((h0 >> (8u * t + 2u * j)) & 3u) << 4u)) - 32));
        w[4u + t] = MUL(ds, float(int(((l1w >> (8u * t + sh)) & 0xFu) | (((h1 >> (8u * t + 2u * j)) & 3u) << 4u)) - 32));
      }
    }
    return e0;
  }
  if (T == 16 || T == 17 || T == 21) {    // IQ2_XXS, IQ2_XS, IQ3_S: db * grid, signed; weight 32ib + 8l + t
    const uint32_t ib = o >> 5u, l = (o >> 3u) & 3u;
    const float d = gg_h(b);
    float db;
    uint32_t lo, hi, sg;
    if (T == 16) {
      const uint32_t aux = gg_ld32(b + 6u + 8u * ib);
      db = MUL(MUL(d, ADD(0.5f, float(aux >> 28u))), 0.25f);
      const uint32_t g = TB_IQ2XXS + 2u * uint32_t(b[2u + 8u * ib + l]);
      lo = tab[g];
      hi = tab[g + 1u];
      sg = gg_parity((aux >> (7u * l)) & 127u);
    } else if (T == 17) {
      const uint32_t q = gg_ld16(b + 2u + 8u * ib + 2u * l), sc = uint32_t(b[66u + ib]);
      db = MUL(MUL(d, ADD(0.5f, float(l < 2u ? (sc & 0xFu) : (sc >> 4u)))), 0.25f);
      lo = tab[TB_IQ2XS + 2u * (q & 511u)];
      hi = tab[TB_IQ2XS + 2u * (q & 511u) + 1u];
      sg = gg_parity(q >> 9u);
    } else {
      const uint32_t qh = uint32_t(b[66u + ib]);
      db = MUL(d, float(1u + 2u * ((uint32_t(b[106u + ib / 2u]) >> (4u * (ib % 2u))) & 0xFu)));
      lo = tab[TB_IQ3S + (uint32_t(b[2u + 8u * ib + 2u * l]) | ((qh << (8u - 2u * l)) & 256u))];
      hi = tab[TB_IQ3S + (uint32_t(b[3u + 8u * ib + 2u * l]) | ((qh << (7u - 2u * l)) & 256u))];
      sg = uint32_t(b[74u + 4u * ib + l]);
    }
    for (uint32_t t = 0u; t < 4u; t++) {
      w[t] = gg_sgn(MUL(db, float((lo >> (8u * t)) & 0xFFu)), sg, t);
      w[4u + t] = gg_sgn(MUL(db, float((hi >> (8u * t)) & 0xFFu)), sg, 4u + t);
    }
    return e0;
  }
  if (T == 19 || T == 29) {               // IQ1_S, IQ1_M: dl * (grid + delta)
    const uint32_t ib = o >> 5u, l = (o >> 3u) & 3u;
    float dl, delta;
    uint32_t idx;
    if (T == 19) {
      const uint32_t qh = gg_ld16(b + 34u + 2u * ib);
      dl = MUL(gg_h(b), float(2u * ((qh >> 12u) & 7u) + 1u));
      delta = (qh & 0x8000u) ? -0.125f : 0.125f;
      idx = uint32_t(b[2u + 4u * ib + l]) | (((qh >> (3u * l)) & 7u) << 8u);
    } else {
      const uint32_t s0 = gg_ld16(b + 48), s1 = gg_ld16(b + 50), s2 = gg_ld16(b + 52), s3 = gg_ld16(b + 54);
      const float d = H2F((s0 >> 12u) | ((s1 >> 8u) & 0xF0u) | ((s2 >> 4u) & 0xF00u) | (s3 & 0xF000u));
      const uint32_t sw = ib / 2u == 0u ? s0 : ib / 2u == 1u ? s1 : ib / 2u == 2u ? s2 : s3;
      dl = MUL(d, float(2u * ((sw >> (6u * (ib % 2u) + 3u * (l >> 1u))) & 7u) + 1u));
      const uint32_t nb = (uint32_t(b[32u + 2u * ib + (l >> 1u)]) >> (4u * (l & 1u))) & 0xFu;
      delta = (nb & 8u) ? -0.125f : 0.125f;
      idx = uint32_t(b[4u * ib + l]) | ((nb & 7u) << 8u);
    }
    const uint32_t lo = tab[TB_IQ1S + 2u * idx], hi = tab[TB_IQ1S + 2u * idx + 1u];
    for (uint32_t t = 0u; t < 4u; t++) {
      w[t] = MUL(dl, ADD(float(int(lo << (24u - 8u * t)) >> 24), delta));
      w[4u + t] = MUL(dl, ADD(float(int(hi << (24u - 8u * t)) >> 24), delta));
    }
    return e0;
  }
  if (T == 34 || T == 35) {               // TQ1_0 (base-3 digits), TQ2_0 (2 bits): (q - 1) * d
    const float d = gg_h(b + (T == 34 ? 52u : 64u));
    // TQ1_0: weights 32n + m of qs[m] (o < 160), 160 + 16n + m of qs[32 + m] (o < 240), 240 + 4n + j of qh[j]
    const uint32_t at = o < 160u ? o % 32u : o < 240u ? 32u + (o - 160u) % 16u : 48u;
    const uint32_t n = o < 160u ? o / 32u : o < 240u ? (o - 160u) / 16u : (o - 240u) / 4u;
    const uint32_t p0 = n == 0u ? 1u : n == 1u ? 3u : n == 2u ? 9u : n == 3u ? 27u : 81u;
    const uint32_t p1 = o < 240u ? p0 : 3u * p0;
    for (uint32_t t = 0u; t < 8u; t++) {
      uint32_t q;
      if (T == 35) {
        q = (uint32_t(b[32u * (o >> 7u) + (o & 31u) + t]) >> (2u * ((o >> 5u) & 3u))) & 3u;
      } else {
        const uint32_t x = uint32_t(b[o < 240u ? at + t : at + (t & 3u)]);
        q = (((x * (t < 4u ? p0 : p1)) & 0xFFu) * 3u) >> 8u;
      }
      w[t] = MUL(float(int(q) - 1), d);
    }
    return e0;
  }
  if (T == 39 || T == 40) {               // MXFP4 (e8m0 / 2), NVFP4 (ue4m3 / 2 a 16): kvalues_fp4[q] * d
    float d;
    uint32_t q;
    if (T == 39) {
      const uint32_t e = uint32_t(b[0]);
      q = 1u + (o & 15u);
      if (e < 2u) {                       // d subnormal, which Metal flushes: the product's bits built exactly
        for (uint32_t t = 0u; t < 8u; t++) {
          const int kv = gg_kv(tab, TB_FP4 + ((uint32_t(b[q + t]) >> (o >= 16u ? 4u : 0u)) & 0xFu));
          const uint32_t m = uint32_t(kv < 0 ? -kv : kv) << e;                 // |kv * d| = m * 2^-128
          w[t] = U2F((m >= 4u ? F2U(float(m)) - (128u << 23u) : m << 21u) | (kv < 0 ? 0x80000000u : 0u));
        }
        return e0;
      }
      d = U2F((e - 1u) << 23u);
    } else {
      const uint32_t x = uint32_t(b[o >> 4u]);
      d = (x == 0u || x == 0x7Fu) ? 0.0f : gg_e4m3((x >> 3u) & 0xFu, x & 7u, 8u);
      q = 4u + 8u * (o >> 4u);
    }
    const uint32_t sh = (T == 39 ? o >= 16u : ((o >> 3u) & 1u) == 1u) ? 4u : 0u;
    for (uint32_t t = 0u; t < 8u; t++) w[t] = MUL(float(gg_kv(tab, TB_FP4 + ((uint32_t(b[q + t]) >> sh) & 0xFu))), d);
    return e0;
  }
  if (T == 41) {                          // Q1_0: d where the bit is set, -d where not
    const float d = gg_h(b);
    const uint32_t q = uint32_t(b[2u + o / 8u]);
    for (uint32_t t = 0u; t < 8u; t++) w[t] = ((q >> t) & 1u) ? d : -d;
    return e0;
  }
  for (uint32_t t = 0u; t < 8u; t++) {   // F32, F16, BF16, F8_E4M3: 8 values from b
    if (T == 0) {
      w[t] = U2F(gg_ld32a(b + 4u * t));
    } else if (T == 1) {
      w[t] = H2F(gg_ld16(b + 2u * t));
    } else if (T == 30) {
      w[t] = U2F(gg_ld16(b + 2u * t) << 16u);
    } else {
      const uint32_t x = uint32_t(b[t]), a = x & 0x7Fu;
      const float v = gg_e4m3(a >> 3u, a & 7u, 7u);
      w[t] = a == 0x7Fu ? U2F(0x7FC00000u) : (x & 0x80u) ? -v : v;
    }
  }
  return e0;
}

// Task k of the row at `row` (ggml type T) into w[0..7], every weight as ggml's dequantize_row_* rounds it;
// returns i0. tab: kvalues_iq4nl [0, 16), iq3xxs_grid [16, 272), iq2s_grid as (low, high) words [272, 2320).
template <int T>
FN uint32_t gg_task(GPTR(uint8_t) row, uint32_t k, LOCAL float *w, GPTR(uint32_t) tab) {
  if (T == 12 || T == 13) {               // Q4_K, Q5_K: d * sc * q - dmin * m
    const uint32_t i = k >> 5u, r = k & 31u, c = r >> 3u, l = (r & 7u) * 4u;
    GPTR(uint8_t) b = row + i * gg_bb(T);
    const uint32_t dd = gg_ld32a(b);
    const float d = H2F(dd & 0xFFFFu), dmin = H2F(dd >> 16u);
    uint32_t sc0, m0, sc1, m1;
    gg_scale_min(gg_ld32a(b + 4), gg_ld32a(b + 8), gg_ld32a(b + 12), c, &sc0, &m0, &sc1, &m1);
    const float d1 = MUL(d, float(sc0)), n1 = MUL(dmin, float(m0));
    const float d2 = MUL(d, float(sc1)), n2 = MUL(dmin, float(m1));
    const uint32_t q = gg_ld32a(b + (T == 13 ? 48u : 16u) + 32u * c + l);
    const uint32_t h = T == 13 ? gg_ld32a(b + 16u + l) : 0u;
    for (uint32_t t = 0u; t < 4u; t++) {
      const uint32_t qb = (q >> (8u * t)) & 0xFFu, hb = (h >> (8u * t)) & 0xFFu;
      const uint32_t lo = (qb & 0xFu) + (((hb >> (2u * c)) & 1u) << 4u);
      const uint32_t hi = (qb >> 4u) + (((hb >> (2u * c + 1u)) & 1u) << 4u);
      w[t] = SUB(MUL(d1, float(lo)), n1);
      w[4u + t] = SUB(MUL(d2, float(hi)), n2);
    }
    return 256u * i + 64u * c + l;
  }
  if (T == 2 || T == 3) {                 // Q4_0: (q - 8) * d; Q4_1: q * d + m
    const uint32_t bi = k >> 2u, j0 = (k & 3u) * 4u;
    GPTR(uint8_t) b = row + bi * gg_bb(T);
    const uint32_t dm = T == 3 ? gg_ld32a(b) : gg_ld16(b);
    const float d = H2F(dm & 0xFFFFu);
    const float m = T == 3 ? H2F(dm >> 16u) : 0.0f;
    const uint32_t q = T == 3 ? gg_ld32a(b + 4u + j0) : gg_ld32(b + 2u + j0);
    for (uint32_t t = 0u; t < 4u; t++) {
      const uint32_t lo = (q >> (8u * t)) & 0xFu, hi = (q >> (8u * t + 4u)) & 0xFu;
      if (T == 3) {
        w[t] = ADD(MUL(float(lo), d), m);
        w[4u + t] = ADD(MUL(float(hi), d), m);
      } else {
        w[t] = MUL(float(int(lo) - 8), d);
        w[4u + t] = MUL(float(int(hi) - 8), d);
      }
    }
    return 32u * bi + j0;
  }
  if (T == 6 || T == 7) {                 // Q5_0: (q - 16) * d; Q5_1: q * d + m
    const uint32_t bi = k >> 2u, j0 = (k & 3u) * 4u;
    GPTR(uint8_t) b = row + bi * gg_bb(T);
    const uint32_t dm = T == 7 ? gg_ld32a(b) : gg_ld16(b);
    const float d = H2F(dm & 0xFFFFu);
    const float m = T == 7 ? H2F(dm >> 16u) : 0.0f;
    const uint32_t qh = T == 7 ? gg_ld32a(b + 4) : gg_ld32(b + 2);
    const uint32_t q = T == 7 ? gg_ld32a(b + 8u + j0) : gg_ld32(b + 6u + j0);
    for (uint32_t t = 0u; t < 4u; t++) {
      const uint32_t j = j0 + t, qb = (q >> (8u * t)) & 0xFFu;
      const uint32_t lo = (qb & 0xFu) | (((qh >> j) << 4u) & 0x10u);
      const uint32_t hi = (qb >> 4u) | ((qh >> (j + 12u)) & 0x10u);
      if (T == 7) {
        w[t] = ADD(MUL(float(lo), d), m);
        w[4u + t] = ADD(MUL(float(hi), d), m);
      } else {
        w[t] = MUL(float(int(lo) - 16), d);
        w[4u + t] = MUL(float(int(hi) - 16), d);
      }
    }
    return 32u * bi + j0;
  }
  if (T == 8) {                           // Q8_0: q * d
    const uint32_t bi = k >> 2u, j0 = (k & 3u) * 8u;
    GPTR(uint8_t) b = row + bi * 34u;
    const float d = gg_h(b);
    const uint32_t q0 = gg_ld32(b + 2u + j0), q1 = gg_ld32(b + 6u + j0);
    for (uint32_t t = 0u; t < 4u; t++) {
      w[t] = MUL(float(int(q0 << (24u - 8u * t)) >> 24), d);
      w[4u + t] = MUL(float(int(q1 << (24u - 8u * t)) >> 24), d);
    }
    return 32u * bi + j0;
  }
  if (T == 42) {                          // Q2_0: (q - 1) * d
    const uint32_t bi = k >> 3u, r = k & 7u;
    GPTR(uint8_t) b = row + bi * 18u;
    const float d = gg_h(b);
    const uint32_t q = gg_ld16(b + 2u + 2u * r);
    for (uint32_t t = 0u; t < 8u; t++) w[t] = MUL(float(int((q >> (2u * t)) & 3u) - 1), d);
    return 64u * bi + 8u * r;
  }
  if (T == 20 || T == 23) {               // IQ4_NL: d * kv[q]; IQ4_XS: d * (ls - 32) * kv[q]
    uint32_t q, i0;
    float dl;
    if (T == 20) {
      const uint32_t bi = k >> 2u, j0 = (k & 3u) * 4u;
      GPTR(uint8_t) b = row + bi * 18u;
      dl = gg_h(b);
      q = gg_ld32(b + 2u + j0);
      i0 = 32u * bi + j0;
    } else {
      const uint32_t i = k >> 5u, r = k & 31u, ib = r >> 2u, j0 = (r & 3u) * 4u;
      GPTR(uint8_t) b = row + i * 136u;
      const uint32_t sh = gg_ld16(b + 2), sl = uint32_t(b[4u + ib / 2u]);
      const uint32_t ls = ((sl >> (4u * (ib % 2u))) & 0xFu) | (((sh >> (2u * ib)) & 3u) << 4u);
      dl = MUL(gg_h(b), float(int(ls) - 32));
      q = gg_ld32a(b + 8u + 16u * ib + j0);
      i0 = 256u * i + 32u * ib + j0;
    }
    for (uint32_t t = 0u; t < 4u; t++) {
      w[t] = MUL(dl, float(gg_kv(tab, (q >> (8u * t)) & 0xFu)));
      w[4u + t] = MUL(dl, float(gg_kv(tab, (q >> (8u * t + 4u)) & 0xFu)));
    }
    return i0;
  }
  if (T == 18) {                          // IQ3_XXS: d * (0.5 + s) * 0.5 * grid, signed
    const uint32_t i = k >> 5u, r = k & 31u, ib = r >> 2u, l = r & 3u;
    GPTR(uint8_t) b = row + i * 98u;
    const uint32_t aux = gg_ld32(b + 66u + 4u * ib);
    const float db = MUL(MUL(gg_h(b), ADD(0.5f, float(aux >> 28u))), 0.5f);
    const uint32_t s7 = (aux >> (7u * l)) & 127u;
    const uint32_t sg = s7 | ((POPC(s7) & 1u) << 7u);
    const uint32_t g1 = tab[16u + uint32_t(b[2u + 8u * ib + 2u * l])];
    const uint32_t g2 = tab[16u + uint32_t(b[3u + 8u * ib + 2u * l])];
    for (uint32_t t = 0u; t < 4u; t++) {
      w[t] = gg_sgn(MUL(db, float((g1 >> (8u * t)) & 0xFFu)), sg, t);
      w[4u + t] = gg_sgn(MUL(db, float((g2 >> (8u * t)) & 0xFFu)), sg, 4u + t);
    }
    return 256u * i + 32u * ib + 8u * l;
  }
  if (T != 22) return gg_task8<T>(row, k, w, tab);
  {                                       // IQ2_S (22): d * (0.5 + s) * 0.25 * grid, signed
    const uint32_t i = k >> 5u, r = k & 31u, ib = r >> 2u, l = r & 3u;
    GPTR(uint8_t) b = row + i * 82u;
    const uint32_t sc = uint32_t(b[74u + ib]);
    const float dl = MUL(MUL(gg_h(b), ADD(0.5f, float(l < 2u ? (sc & 0xFu) : (sc >> 4u)))), 0.25f);
    const uint32_t idx = uint32_t(b[2u + 4u * ib + l]) | ((uint32_t(b[66u + ib]) << (8u - 2u * l)) & 0x300u);
    const uint32_t lo = tab[272u + 2u * idx], hi = tab[273u + 2u * idx];
    const uint32_t sg = uint32_t(b[34u + 4u * ib + l]);
    for (uint32_t t = 0u; t < 4u; t++) {
      w[t] = gg_sgn(MUL(dl, float((lo >> (8u * t)) & 0xFFu)), sg, t);
      w[4u + t] = gg_sgn(MUL(dl, float((hi >> (8u * t)) & 0xFFu)), sg, 4u + t);
    }
    return 256u * i + 32u * ib + 8u * l;
  }
}

// 32 weights, tasks 4K..4K+3 of a row: w[0..15] go with x[i0 + 0..15] and w[16..31] with x[i0 + gg_run32(T) +
// 0..15]. Q4_0, Q4_1, Q4_K, Q5_K and Q5_1 decode them from wide loads and one header; the rest assemble four
// tasks.
CFN uint32_t gg_run32(int T) { return (T == 12 || T == 13) ? 32u : 16u; }
FN uint32_t gg_wb(uint4 v, uint32_t t) {
  const uint32_t w = (t >> 2u) == 0u ? v.x : (t >> 2u) == 1u ? v.y : (t >> 2u) == 2u ? v.z : v.w;
  return (w >> (8u * (t & 3u))) & 0xFFu;
}
template <int T>
FN uint32_t gg_task32(GPTR(uint8_t) row, uint32_t K, LOCAL float *w, GPTR(uint32_t) tab) {
  if (T == 12 || T == 13) {
    const uint32_t i = K >> 3u, c = (K & 7u) >> 1u, l = (K & 1u) * 16u;
    GPTR(uint8_t) b = row + i * gg_bb(T);
    const uint4 hd = *((GPTR(uint4))(b));
    const float d = H2F(hd.x & 0xFFFFu), dmin = H2F(hd.x >> 16u);
    uint32_t sc0, m0, sc1, m1;
    gg_scale_min(hd.y, hd.z, hd.w, c, &sc0, &m0, &sc1, &m1);
    const float d1 = MUL(d, float(sc0)), n1 = MUL(dmin, float(m0));
    const float d2 = MUL(d, float(sc1)), n2 = MUL(dmin, float(m1));
    const uint4 q = *((GPTR(uint4))(b + (T == 13 ? 48u : 16u) + 32u * c + l));
    const uint4 qh = T == 13 ? *((GPTR(uint4))(b + 16u + l)) : U4(0u, 0u, 0u, 0u);
    for (uint32_t t = 0u; t < 16u; t++) {
      const uint32_t qb = gg_wb(q, t), hb = gg_wb(qh, t);
      w[t] = SUB(MUL(d1, float((qb & 0xFu) + (((hb >> (2u * c)) & 1u) << 4u))), n1);
      w[16u + t] = SUB(MUL(d2, float((qb >> 4u) + (((hb >> (2u * c + 1u)) & 1u) << 4u))), n2);
    }
    return 256u * i + 64u * c + l;
  }
  if (T == 2 || T == 3) {
    GPTR(uint8_t) b = row + K * gg_bb(T);
    const uint32_t dm = T == 3 ? gg_ld32a(b) : gg_ld16(b);
    const float d = H2F(dm & 0xFFFFu), m = T == 3 ? H2F(dm >> 16u) : 0.0f;
    const uint4 q = T == 3 ? U4(gg_ld32a(b + 4), gg_ld32a(b + 8), gg_ld32a(b + 12), gg_ld32a(b + 16))
                           : U4(gg_ld32(b + 2), gg_ld32(b + 6), gg_ld32(b + 10), gg_ld32(b + 14));
    for (uint32_t j = 0u; j < 16u; j++) {
      const uint32_t qb = gg_wb(q, j);
      if (T == 3) {
        w[j] = ADD(MUL(float(qb & 0xFu), d), m);
        w[16u + j] = ADD(MUL(float(qb >> 4u), d), m);
      } else {
        w[j] = MUL(float(int(qb & 0xFu) - 8), d);
        w[16u + j] = MUL(float(int(qb >> 4u) - 8), d);
      }
    }
    return 32u * K;
  }
  if (T == 7) {
    GPTR(uint8_t) b = row + K * 24u;
    const uint32_t dm = gg_ld32a(b), qh = gg_ld32a(b + 4);
    const float d = H2F(dm & 0xFFFFu), m = H2F(dm >> 16u);
    const uint4 q = U4(gg_ld32a(b + 8), gg_ld32a(b + 12), gg_ld32a(b + 16), gg_ld32a(b + 20));
    for (uint32_t j = 0u; j < 16u; j++) {
      const uint32_t qb = gg_wb(q, j);
      w[j] = ADD(MUL(float((qb & 0xFu) | (((qh >> j) << 4u) & 0x10u)), d), m);
      w[16u + j] = ADD(MUL(float((qb >> 4u) | ((qh >> (j + 12u)) & 0x10u)), d), m);
    }
    return 32u * K;
  }
  float v[8];
  uint32_t i0 = 0u;
  for (uint32_t q = 0u; q < 4u; q++) {
    const uint32_t j = gg_task<T>(row, 4u * K + q, v, tab);
    if (q == 0u) i0 = j;
    for (uint32_t u = 0u; u < 8u; u++)
      w[gg_run(T) == 4u ? 8u * q + u : u < 4u ? 4u * q + u : 12u + 4u * q + u] = v[u];
  }
  return i0;
}

FN float gg_silu_mul(float g, float u) {
  const float ys = 1.0f / (1.0f + EXP(g < 0.0f ? -g : g));
  return g * (g < 0.0f ? ys : 1.0f - ys) * u;
}
"""

# Decode, gate and up of every slot: x staged in threadgroup memory, LPR lanes RPW rows of both on 32-weight tasks,
# each task's x read once for all of them; h = SwiGLU, f32.
GATE_UP_SRC = r"""
TGMEM float xs[IN];
const uint32_t lane = _tid & 31u, warp = _tid >> 5u, sub = lane & (LPR - 1u);
const uint32_t slot = _bid.y, r0 = ((_bid.x * 4u + warp) * (32u / LPR) + lane / LPR) * RPW;
const uint32_t e = slots[slot];
const size_t xb = (size_t)(slot / TOPK) * IN;
for (uint32_t i = _tid; i < IN; i += 128u) xs[i] = static_cast<float>(x[xb + i]);
BARRIER;
GPTR(uint8_t) pg = wg + ((size_t)e * OUT + r0) * gg_row(TG, IN);
GPTR(uint8_t) pu = wu + ((size_t)e * OUT + r0) * gg_row(TU, IN);
float ag[RPW], au[RPW], w[32], xv[32];
for (uint32_t j = 0u; j < RPW; j++) ag[j] = au[j] = 0.0f;
for (uint32_t k = sub; k < IN / 32u; k += LPR) {
  for (uint32_t j = 0u; j < RPW; j++) {
    if (r0 + j >= OUT) break;
    const uint32_t i0 = gg_task32<TG>(pg + j * gg_row(TG, IN), k, w, tab);
    if (j == 0u) {
      for (uint32_t t = 0u; t < 16u; t++) {
        xv[t] = xs[i0 + t];
        xv[16u + t] = xs[i0 + gg_run32(TG) + t];
      }
    }
    for (uint32_t t = 0u; t < 32u; t++) ag[j] += w[t] * xv[t];
  }
  for (uint32_t j = 0u; j < RPW; j++) {
    if (r0 + j >= OUT) break;
    const uint32_t i0 = gg_task32<TU>(pu + j * gg_row(TU, IN), k, w, tab);
    if (j == 0u && TU != TG) {
      for (uint32_t t = 0u; t < 16u; t++) {
        xv[t] = xs[i0 + t];
        xv[16u + t] = xs[i0 + gg_run32(TU) + t];
      }
    }
    for (uint32_t t = 0u; t < 32u; t++) au[j] += w[t] * xv[t];
  }
}
for (uint32_t j = 0u; j < RPW; j++) {
  for (uint32_t m = LPR / 2u; m > 0u; m >>= 1u) {
    ag[j] += SHFL(ag[j], m);
    au[j] += SHFL(au[j], m);
  }
  if (sub == 0u && r0 + j < OUT) h[(size_t)slot * OUT + r0 + j] = gg_silu_mul(ag[j], au[j]);
}
"""

# Decode, down of every slot: h staged in threadgroup memory, LPR lanes RPW rows on 32-weight tasks, out in the
# output dtype, unweighted.
DOWN_SRC = r"""
TGMEM float hs[IN];
const uint32_t lane = _tid & 31u, warp = _tid >> 5u, sub = lane & (LPR - 1u);
const uint32_t slot = _bid.y, r0 = ((_bid.x * 4u + warp) * (32u / LPR) + lane / LPR) * RPW;
const uint32_t e = slots[slot];
for (uint32_t i = _tid; i < IN; i += 128u) hs[i] = h[(size_t)slot * IN + i];
BARRIER;
GPTR(uint8_t) pd = wd + ((size_t)e * OUT + r0) * gg_row(TD, IN);
float a[RPW], w[32], xv[32];
for (uint32_t j = 0u; j < RPW; j++) a[j] = 0.0f;
for (uint32_t k = sub; k < IN / 32u; k += LPR) {
  for (uint32_t j = 0u; j < RPW; j++) {
    if (r0 + j >= OUT) break;
    const uint32_t i0 = gg_task32<TD>(pd + j * gg_row(TD, IN), k, w, tab);
    if (j == 0u) {
      for (uint32_t t = 0u; t < 16u; t++) {
        xv[t] = hs[i0 + t];
        xv[16u + t] = hs[i0 + gg_run32(TD) + t];
      }
    }
    for (uint32_t t = 0u; t < 32u; t++) a[j] += w[t] * xv[t];
  }
}
for (uint32_t j = 0u; j < RPW; j++) {
  for (uint32_t m = LPR / 2u; m > 0u; m >>= 1u) a[j] += SHFL(a[j], m);
  if (sub == 0u && r0 + j < OUT) y[(size_t)slot * OUT + r0 + j] = static_cast<TY>(a[j]);
}
"""

# Prefill: experts `slots` of a [S, OUT rows] buffer as Q [len(slots), OUT, IN] in T, a task a thread.
DEQ_SRC = r"""
const uint32_t k = _gid % (IN / 8u), r = (_gid / (IN / 8u)) % OUT, c = _gid / (IN / 8u * OUT);
if (c >= uint32_t(slots_shape[0])) return;
float w[8];
const uint32_t i0 = gg_task<TT>(src + ((size_t)slots[c] * OUT + r) * gg_row(TT, IN), k, w, tab);
const size_t o = ((size_t)c * OUT + r) * IN + i0;
for (uint32_t t = 0u; t < 4u; t++) {
  q[o + t] = static_cast<T>(w[t]);
  q[o + gg_run(TT) + t] = static_cast<T>(w[4u + t]);
}
"""

SOURCES = {"gguf_gate_up": GATE_UP_SRC, "gguf_down": DOWN_SRC, "gguf_dequant": DEQ_SRC}

_CUDA = not mx.metal.is_available() and getattr(mx, "cuda", None) is not None and mx.cuda.is_available()
if _CUDA:
    import gguf_cuda as _backend
else:
    import gguf_metal as _backend
# CUDA only: the decode chain on rows sorted by slot, a weight task decoded once a run (gguf_cuda.py, _gemv_runs).
RUN_SOURCES = {"gguf_gate_up_runs": _backend.GATE_UP_RUNS_SRC, "gguf_down_runs": _backend.DOWN_RUNS_SRC} if _CUDA else {}
_K = {}
_TAB = []


def _kernel(name, ins, outs):
    if name not in _K:
        _K[name] = (_backend.kernel(name, ins, outs, RUN_SOURCES[name], HEADER + _backend.RUNS_HEADER)
                    if name in RUN_SOURCES else _backend.kernel(name, ins, outs, SOURCES[name], HEADER))
    return _K[name]


def _tab():
    """The kernels' tables as uint32 words, at the TB_* offsets of HEADER."""
    if not _TAB:
        words = [KVALUES_IQ4NL.astype(np.int32).view(np.uint32)] + [
            np.ascontiguousarray(g).view("<u4").reshape(-1)
            for g in (IQ3XXS_GRID, IQ2S_GRID, IQ2XXS_GRID, IQ2XS_GRID, IQ3S_GRID, IQ1S_GRID)]
        words.append(KVALUES_FP4.astype(np.int32).view(np.uint32))
        assert np.cumsum([len(w) for w in words]).tolist() == [16, 272, 2320, 2832, 3856, 4368, 8464, 8480]
        _TAB.append(mx.array(np.concatenate(words)))
    return _TAB[0]


def dims(buf, t):
    """(rows, row length) of an expert buffer [S, rows, row bytes] in ggml type t."""
    _, be, bb = TYPES[t]
    return buf.shape[1], buf.shape[2] // bb * be


def dequant(buf, slots, t, dtype):
    """Q [len(slots), rows, row length] in dtype of the experts at `slots` (uint32) of buf [S, rows, row bytes]."""
    out, inp = dims(buf, t)
    c = slots.size
    return _kernel("gguf_dequant", ["src", "slots", "tab"], ["q"])(
        inputs=[buf, slots, _tab()], template=[("T", dtype), ("TT", TYPES[t][0]), ("IN", inp), ("OUT", out)],
        grid=(c * out * inp // 8, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(c, out, inp)], output_dtypes=[dtype])[0]


# Rows a warp (gate/up) and a lane group (down) of the decode chain: each task's x is read once for all of them.
RPW = [int(v) for v in os.environ.get("MOE_GGUF_RPW", "1,1").split(",")]
# The rows a warp calibrate() times.
RPW_TRIES = ((1, 1), (2, 1), (2, 2), (4, 1), (4, 2))
# MOE_GGUF_RUNS (CUDA, on unless =0): rows of one slot each go through the run kernels, checked once against the chain.
RUNS = _CUDA and os.environ.get("MOE_GGUF_RUNS", "1") != "0"
# The run kernels' tile: rows a lane group of gate/up and of down, rows a window, warps a threadgroup (no bit depends on it).
RUN_TILE = [int(v) for v in os.environ.get("MOE_GGUF_RUN_TILE", "2,2,8,8").split(",")]
_RUNS_OK = {}


def gemv(x, w, slots, topk, types, out_dtype):
    """Every slot's expert output, unweighted: x (rows, hidden), slots (rows * topk,) uint32 with slot s on row
    s // topk, w the gate, up and down buffers [S, rows, row bytes] in `types`; returns (rows * topk, hidden), through
    the run kernels under RUNS at one slot a row."""
    inter, hid = dims(w[0], types[0])
    if RUNS and topk == 1 and slots.size > 1 and _runs_ok(hid, inter, types, x.dtype, out_dtype):
        return _gemv_runs(x, w, slots, types, out_dtype)[1]
    return _chain(x, w, slots, topk, types, out_dtype)[1]


def _chain(x, w, slots, topk, types, out_dtype):
    """gemv() through the decode chain, a threadgroup row a slot; returns (h, out)."""
    (bg, bu, bd), (tg, tu, td) = w, types
    inter, hid = dims(bg, tg)
    n = slots.size
    lg, ld = _lpr(hid), _lpr(inter)
    # Rows a threadgroup covers; the last one's rows past the end are skipped in the kernels.
    ng, nd = 4 * 32 // lg * RPW[0], 4 * 32 // ld * RPW[1]
    h = _kernel("gguf_gate_up", ["x", "wg", "wu", "slots", "tab"], ["h"])(
        inputs=[x, bg, bu, slots, _tab()],
        template=[("TG", TYPES[tg][0]), ("TU", TYPES[tu][0]), ("IN", hid), ("OUT", inter), ("TOPK", topk),
                  ("LPR", lg), ("RPW", RPW[0])],
        grid=(-(-inter // ng) * 128, n, 1), threadgroup=(128, 1, 1),
        output_shapes=[(n, inter)], output_dtypes=[mx.float32])[0]
    return h, _kernel("gguf_down", ["h", "wd", "slots", "tab"], ["y"])(
        inputs=[h, bd, slots, _tab()],
        template=[("TD", TYPES[td][0]), ("IN", inter), ("OUT", hid), ("LPR", ld), ("TY", out_dtype),
                  ("RPW", RPW[1])],
        grid=(-(-hid // nd) * 128, n, 1), threadgroup=(128, 1, 1),
        output_shapes=[(n, hid)], output_dtypes=[out_dtype])[0]


def _gemv_runs(x, w, slots, types, out_dtype):
    """gemv() on rows of one slot each, best sorted by slot, through the run kernels; returns (h, out)."""
    (bg, bu, bd), (tg, tu, td) = w, types
    inter, hid = dims(bg, tg)
    n = slots.size
    lg, ld = _lpr(hid), _lpr(inter)
    rg, rd, tr, nw = RUN_TILE
    # A window's rows fit the 48 KB of static shared memory a threadgroup has.
    trg, trd = max(1, min(tr, 49152 // (hid * x.dtype.size))), max(1, min(tr, 49152 // (inter * 4)))
    ng, nd = nw * 32 // lg * rg, nw * 32 // ld * rd
    h = _kernel("gguf_gate_up_runs", ["x", "wg", "wu", "slots", "tab"], ["h"])(
        inputs=[x, bg, bu, slots, _tab()],
        template=[("TG", TYPES[tg][0]), ("TU", TYPES[tu][0]), ("IN", hid), ("OUT", inter), ("LPR", lg), ("RPW", rg),
                  ("TR", trg), ("NW", nw), ("TX", x.dtype)],
        grid=(-(-inter // ng) * 32 * nw, -(-n // trg), 1), threadgroup=(32 * nw, 1, 1),
        output_shapes=[(n, inter)], output_dtypes=[mx.float32])[0]
    return h, _kernel("gguf_down_runs", ["h", "wd", "slots", "tab"], ["y"])(
        inputs=[h, bd, slots, _tab()],
        template=[("TD", TYPES[td][0]), ("IN", inter), ("OUT", hid), ("LPR", ld), ("RPW", rd), ("TR", trd),
                  ("NW", nw), ("TY", out_dtype)],
        grid=(-(-hid // nd) * 32 * nw, -(-n // trd), 1), threadgroup=(32 * nw, 1, 1),
        output_shapes=[(n, hid)], output_dtypes=[out_dtype])[0]


def _same_bits(a, b):
    u = {4: mx.uint32, 2: mx.uint16}[a.dtype.size]
    return a.shape == b.shape and bool(mx.array_equal(a.view(u), b.view(u)))


def _runs_ok(hid, inter, types, x_dtype, out_dtype):
    """The run kernels at these shapes and dtypes, checked once: on only if bit-equal to the chain on random experts
    and sorted runs of 1 to 16 rows."""
    key = (hid, inter, tuple(types), str(x_dtype), str(out_dtype))
    if key not in _RUNS_OK:
        _RUNS_OK[key] = False
        try:
            rng = np.random.default_rng(5)
            w = tuple(mx.array(random_experts(rng, 6, r, c, t))
                      for (r, c), t in zip(((inter, hid), (inter, hid), (hid, inter)), types))
            lens = rng.permutation(np.arange(1, 17))
            slots = mx.array(np.repeat(rng.integers(0, 6, len(lens)), lens).astype(np.uint32))
            x = mx.array(rng.standard_normal((slots.size, hid)).astype(np.float32)).astype(x_dtype)
            ok = all(map(_same_bits, _chain(x, w, slots, 1, types, out_dtype), _gemv_runs(x, w, slots, types, out_dtype)))
            why = "" if ok else "differs from the decode chain"
        except Exception as e:      # a kernel that does not build here
            ok, why = False, f"{type(e).__name__}: {str(e)[:200]}"
        _RUNS_OK[key] = ok
        print(f"  gguf: run kernels {'on, bit-equal to the decode chain' if ok else 'off: ' + why}", flush=True)
    return _RUNS_OK[key]


def _lpr(length):
    """Lanes a row of `length` weights: the largest power of two up to 32 dividing its 32-weight tasks."""
    tasks = length // 32
    return min(tasks & -tasks, 32)


# Experts dequantized at once on the dense path: ~10 MB of f16 an expert at Flash-Next's shapes.
DENSE_CHUNK = 8


def _mm(xin, q, loc):
    """Rows xin (R, 1, in) f16 times their experts' Q[loc] (q [c, out, in]) transposed: (R, out) f32."""
    return mx.gather_mm(xin, q.swapaxes(1, 2), rhs_indices=loc, sorted_indices=True)[:, 0].astype(mx.float32)


def dense(x, w, host, types, out_dtype):
    """Every row's expert output, unweighted: x (rows, hidden), host (rows,) numpy slot per row in any order;
    DENSE_CHUNK experts at a time decoded to f16 and multiplied by MLX's gather_mm, rows sorted by slot."""
    (bg, bu, bd), (tg, tu, td) = w, types
    order = np.argsort(host, kind="stable")
    hs = host[order]
    uniq, first = np.unique(hs, return_index=True)
    bounds = list(first) + [len(hs)]
    hid = x.shape[-1]
    y = mx.zeros((len(hs), hid), dtype=out_dtype)
    for a in range(0, len(uniq), DENSE_CHUNK):
        b = min(a + DENSE_CHUNK, len(uniq))
        r0, r1 = int(bounds[a]), int(bounds[b])
        sel = mx.array(order[r0:r1].astype(np.int32))
        slots = mx.array(uniq[a:b].astype(np.uint32))
        loc = mx.array(np.searchsorted(uniq[a:b], hs[r0:r1]).astype(np.uint32))
        # MLX #4727: on CUDA the sorted gather_mm wants the size-1 axis from a reshape, not expand_dims.
        xin = x[sel].astype(mx.float16).reshape(r1 - r0, 1, hid)
        g = _mm(xin, dequant(bg, slots, tg, mx.float16), loc)
        u = _mm(xin, dequant(bu, slots, tu, mx.float16), loc)
        hh = (g * mx.sigmoid(g) * u).astype(mx.float16).reshape(r1 - r0, 1, -1)
        y[sel] = _mm(hh, dequant(bd, slots, td, mx.float16), loc).astype(out_dtype)
        if b < len(uniq):
            mx.eval(y)          # one chunk's Q at a time
    return y


# Per-device costs of the two paths, as calibrate() measures them: chain fixed s, chain s a row, dense fixed s,
# dense s an expert, dense s a row; until then every pass takes the decode chain.
_COST = None
CALIB_DIR = Path.home() / ".cache" / "moe-ssd-offload"


def _measure(hid, inter, types):
    """The cost model's five terms, timed on random experts at these shapes as the best of 5 and fitted by least
    squares, the chain over three widths and the dense path over five (experts, rows) points."""
    rng = np.random.default_rng(0)
    E = 32
    w = tuple(mx.array(random_experts(rng, E, r, c, t))
              for (r, c), t in zip(((inter, hid), (inter, hid), (hid, inter)), types))
    x = mx.array(rng.standard_normal((80 * E, hid)).astype(np.float32)).astype(mx.bfloat16)
    mx.eval(w, x)

    def secs(u, p, dense_path):
        host = (np.arange(p) % u).astype(np.uint32)
        f = (lambda: dense(x[:p], w, host, types, x.dtype)) if dense_path else \
            (lambda: _chain(x[:p], w, mx.array(host), 1, types, x.dtype)[1])
        mx.eval(f())
        best = float("inf")
        for _ in range(5):
            t0 = time.perf_counter()
            mx.eval(f())
            best = min(best, time.perf_counter() - t0)
        return best

    def fit(rows, ts):
        return np.linalg.lstsq(np.array(rows, dtype=np.float64), np.array(ts), rcond=None)[0]

    # The decode chain's rows a warp first (CUDA: Q4_0 at the 35B's shapes 7.66 us an expert at 1,1, 5.31 at 4,2).
    tries = {}
    for rpw in RPW_TRIES:
        RPW[:] = rpw
        tries[rpw] = secs(E, 80, False)
    RPW[:] = min(tries, key=tries.get)
    chain = [(p, secs(E, p, False)) for p in (10, 10 * E, 80 * E)]
    g = fit([[1.0, p] for p, _ in chain], [t for _, t in chain])
    pts = [(u, p, secs(u, p, True)) for u, p in ((1, 10), (10, 100), (10, 800), (E, 10 * E), (E, 80 * E))]
    d = fit([[1.0, u, p] for u, p, _ in pts], [t for *_, t in pts])
    mx.synchronize()
    mx.clear_cache()
    return [max(float(v), 0.0) for v in (*g, *d)]


def calibrate(hid, inter, types):
    """This device's decode rows a warp and cost model at these shapes and types, measured once (a few seconds)
    and cached; MOE_GGUF_RPW set by hand wins over the measured rows."""
    global _COST
    key = json.dumps([mx.device_info()["device_name"], mx.__version__,
                      hashlib.sha1(Path(__file__).read_bytes()).hexdigest(), hid, inter, list(types)])
    path = CALIB_DIR / f"gguf-{hashlib.sha1(key.encode()).hexdigest()[:16]}.json"
    try:
        got = json.loads(path.read_text())
        _COST, rpw = got["cost"], got["rpw"]
    except (OSError, ValueError, KeyError):
        _COST = _measure(hid, inter, types)
        rpw = list(RPW)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"cost": _COST, "rpw": rpw}))
            tmp.replace(path)
        except OSError as e:
            print(f"  gguf: could not save the prefill switch to {path}: {e}", flush=True)
    RPW[:] = [int(v) for v in os.environ["MOE_GGUF_RPW"].split(",")] if "MOE_GGUF_RPW" in os.environ else rpw
    if RUNS:
        _runs_ok(hid, inter, tuple(types), mx.bfloat16, mx.bfloat16)     # built and checked at load, not on the first prompt
    return _COST


def dense_pays(u, p):
    """Whether dense() beats gemv() on p rows of u experts, by this device's cost model."""
    c = _COST
    return c is not None and c[2] + c[3] * u + c[4] * p < c[0] + c[1] * p


def experts(x, bufs, rhs, types, host=None):
    """The routed experts' outputs as the gather_qmm chain gives them, x (..., 1, hidden) broadcast against rhs
    (...) slot ids, bufs the engine's (gguf part, [pad part]) buffers of gate, up and down and types their ggml
    types (a sequence or one "Q4_K/Q4_K/Q5_1" string); returns rhs.shape + (1, hidden) in x's dtype, `host` (rhs
    on the host, one row of x a slot) letting dense_pays() pick the dense path over the decode chain."""
    if isinstance(types, str):
        types = tuple(types.split("/"))
    hid = x.shape[-1]
    x2 = x.reshape(-1, hid)
    w = tuple(b[0] for b in bufs)
    if host is not None:
        host = np.asarray(host).reshape(-1)
        assert host.size == x2.shape[0] == rhs.size, "dense: one row of x per slot"
        if dense_pays(len(np.unique(host)), host.size):
            return dense(x2, w, host, types, x.dtype).reshape(tuple(rhs.shape) + (1, hid))
    y = gemv(x2, w, rhs.reshape(-1).astype(mx.uint32), rhs.size // x2.shape[0], types, x.dtype)
    return y.reshape(tuple(rhs.shape) + (1, hid))
