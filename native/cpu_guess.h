// MOE_CPU_GUESS: the staging guess on a CPU core, the next layer's 8-bit affine router on this layer's MoE input and route()'s selection.
// No MLX here, so tests/test_cpu_guess.py builds it alone on any Mac.
#pragma once

#include <math.h>
#include <stdint.h>
#include <string.h>

#include <algorithm>
#include <numeric>
#include <vector>

#if defined(__ARM_FEATURE_FP16_FML)
#include <arm_neon.h>
#endif

namespace cpu_guess {

// Element i of a bf16 (bf true) or fp32 array.
inline float load(const void* p, size_t i, bool bf) {
  if (!bf)
    return static_cast<const float*>(p)[i];
  uint32_t u = uint32_t(static_cast<const uint16_t*>(p)[i]) << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}

// out[e] = sum over groups of scale * (q . x) + bias * sum(x), MLX's w = scale * q + bias with q the bytes of row e
// (8-bit words read as bytes, little endian), accumulated in fp32 but not in MLX's order. x16: x came from bf16, so
// the dots can take x in fp16 (exact for |x| in fp16's normal range) and FMLAL's fp16 products into fp32.
inline void logits8(const uint8_t* w, const void* scales, const void* biases, bool bf, const float* x,
                    int E, int D, int gs, float* out, bool x16 = false) {
  int G = D / gs;
  std::vector<float> xs(G, 0.0f);
  float amax = 0.0f;
  for (int g = 0; g < G; g++)
    for (int i = 0; i < gs; i++) {
      xs[g] += x[g * gs + i];
      amax = std::max(amax, fabsf(x[g * gs + i]));
    }
#if defined(__ARM_FEATURE_FP16_FML)
  // Known limit: 1.6x the fp32 loop on one core (~54 us at 512 x 2560 on an M-series core); int8 dots would be faster but change the guess.
  std::vector<float16_t> xh;
  if (x16 && gs % 16 == 0 && amax < 32768.0f) {
    xh.resize(D);
    for (int i = 0; i < D; i += 4)
      vst1_f16(&xh[i], vcvt_f16_f32(vld1q_f32(x + i)));
  }
#endif
  for (int e = 0; e < E; e++) {
    const uint8_t* q = w + size_t(e) * D;
    float acc = 0.0f;
    for (int g = 0; g < G; g++) {
      float d = 0.0f;
#if defined(__ARM_FEATURE_FP16_FML)
      if (!xh.empty()) {
        float32x4_t a0 = vdupq_n_f32(0.0f), a1 = a0, a2 = a0, a3 = a0;
        for (int i = g * gs; i < (g + 1) * gs; i += 16) {
          uint8x16_t b = vld1q_u8(q + i);
          float16x8_t lo = vcvtq_f16_u16(vmovl_u8(vget_low_u8(b))), hi = vcvtq_f16_u16(vmovl_high_u8(b));
          float16x8_t xl = vld1q_f16(&xh[i]), xr = vld1q_f16(&xh[i + 8]);
          a0 = vfmlalq_low_f16(a0, lo, xl);
          a1 = vfmlalq_high_f16(a1, lo, xl);
          a2 = vfmlalq_low_f16(a2, hi, xr);
          a3 = vfmlalq_high_f16(a3, hi, xr);
        }
        d = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
      } else
#endif
      {
#pragma clang fp reassociate(on)
        for (int i = 0; i < gs; i++)
          d += x[g * gs + i] * float(q[g * gs + i]);
      }
      size_t k = size_t(e) * G + g;
      acc += load(scales, k, bf) * d + load(biases, k, bf) * xs[g];
    }
    out[e] = acc;
  }
}

// The k largest of v, best first, the larger index on a tie (route()'s rule).
inline void top(const std::vector<float>& v, int k, std::vector<int>& out) {
  out.resize(v.size());
  std::iota(out.begin(), out.end(), 0);
  std::partial_sort(out.begin(), out.begin() + k, out.end(),
                    [&](int a, int b) { return v[a] != v[b] ? v[a] > v[b] : a > b; });
  out.resize(k);
}

// fused_router.route_logits' guess on fp32 logits: softmax, rank = gates + lam * (max - min) * mask, the top k by
// rank, and under tau > 0 the honest top k where the bias gives up more than tau of its mass; best first into out.
inline void pick(const float* lg, const uint8_t* mask, int E, int k, float lam, float tau, int32_t* out) {
  std::vector<float> g(lg, lg + E);
  float m = *std::max_element(g.begin(), g.end()), s = 0.0f;
  for (auto& v : g)
    s += v = expf(v - m);
  float inv = 1.0f / s, hi = -INFINITY, lo = INFINITY;
  for (auto& v : g) {
    v *= inv;
    hi = std::max(hi, v);
    lo = std::min(lo, v);
  }
  std::vector<int> b, h;
  if (lam > 0.0f) {
    std::vector<float> r(g);
    float ls = lam * (hi - lo);
    for (int e = 0; e < E; e++)
      if (mask[e])
        r[e] += ls;
    top(r, k, b);
  } else {
    top(g, k, b);
  }
  if (tau > 0.0f && lam > 0.0f) {
    top(g, k, h);
    // The gathered gates summed from the smallest pick up, as the kernel does.
    float mh = 0.0f, mb = 0.0f;
    for (int j = k - 1; j >= 0; j--) {
      mh += g[h[j]];
      mb += g[b[j]];
    }
    if (mh - mb > tau * mh)
      b = h;
  }
  std::copy(b.begin(), b.end(), out);
}

} // namespace cpu_guess
