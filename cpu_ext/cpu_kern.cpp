// Threaded AVX-512 kernels for MLX's CPU device: affine-quantized matmul (transpose=True) and its expert gather.
// Math per output: sum over groups of scale * dot(x, q) + bias * sum(x), f32 accumulation, as MLX's GPU qmv does.
#include <immintrin.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <tuple>
#include <stdexcept>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "mlx/backend/cpu/encoder.h"
#include "mlx/mlx.h"
#include "mlx/primitives.h"

namespace mx = mlx::core;
namespace nb = nanobind;
using namespace nb::literals;

namespace {

// ---- dtype helpers ---------------------------------------------------------------------------------------------

enum DT { F32 = 0, BF16 = 1, F16 = 2 };

DT dt_of(mx::Dtype d) {
  if (d == mx::float32) return F32;
  if (d == mx::bfloat16) return BF16;
  if (d == mx::float16) return F16;
  throw std::invalid_argument("cpu_kern: float32, bfloat16 or float16 only");
}

inline float bf2f(uint16_t h) {
  uint32_t u = uint32_t(h) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// Round to nearest even, as MLX's bfloat16 conversion does.
inline uint16_t f2bf(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  if ((u & 0x7fffffffu) > 0x7f800000u) return uint16_t((u >> 16) | 0x40);
  u += 0x7fffu + ((u >> 16) & 1u);
  return uint16_t(u >> 16);
}

inline float ld(const void* p, size_t i, DT t) {
  switch (t) {
    case F32: return static_cast<const float*>(p)[i];
    case BF16: return bf2f(static_cast<const uint16_t*>(p)[i]);
    default: return _cvtsh_ss(static_cast<const uint16_t*>(p)[i]);
  }
}

inline void st(void* p, size_t i, float v, DT t) {
  switch (t) {
    case F32: static_cast<float*>(p)[i] = v; break;
    case BF16: static_cast<uint16_t*>(p)[i] = f2bf(v); break;
    default: static_cast<uint16_t*>(p)[i] = _cvtss_sh(v, _MM_FROUND_TO_NEAREST_INT); break;
  }
}

inline float hsum(__m512 v) { return _mm512_reduce_add_ps(v); }

int g_threads = 6;
size_t g_bf_rows = 2;  // rows a job (on average) from which bf16 x takes the VDPBF16PS path

// ---- the row kernel ----------------------------------------------------------------------------------------------
//
// x rows arrive pre-arranged in f32 (prep_x): for 4-bit, each 32-wide chunk as [16 even k, 16 odd k], matching the
// low and high nibbles of 16 packed bytes; for 8-bit, natural order. xs holds each row's per-group sums.

constexpr int MB = 8;  // x rows per pass: the unpacked weights are reused across them

// BF: x is bf16 (exact in the dot) and the 4-bit codes become bf16 too, two products an instruction (VDPBF16PS),
// for passes of several rows where the kernel is compute-bound; one row stays on the f32 path (memory-bound anyway).
template <int BITS, bool BF>
void rows_kernel(const uint8_t* w, const void* sc, const void* bi, DT sdt, int K, int gs, int n0, int n1, int M,
                 const void* const* xp, const float* const* xs, void* const* yp, DT odt) {
  const int G = K / gs;
  const size_t row_bytes = size_t(K) * BITS / 8;
  float sf[1024], bf[1024];
  const __m512i m0f = _mm512_set1_epi32(0x0f);
  const __m512i m0f16 = _mm512_set1_epi16(0x0f);
  // bf16 of 0..15 as a vpermw table.
  alignas(64) static const uint16_t tab_[32] = {0x0000, 0x3f80, 0x4000, 0x4040, 0x4080, 0x40a0, 0x40c0, 0x40e0,
                                                0x4100, 0x4110, 0x4120, 0x4130, 0x4140, 0x4150, 0x4160, 0x4170};
  const __m512i tab = _mm512_load_si512(tab_);
  for (int n = n0; n < n1; ++n) {
    const uint8_t* wr = w + size_t(n) * row_bytes;
    for (int g = 0; g < G; ++g) {
      sf[g] = ld(sc, size_t(n) * G + g, sdt);
      bf[g] = ld(bi, size_t(n) * G + g, sdt);
    }
    for (int m0 = 0; m0 < M; m0 += MB) {
      const int mb = std::min(MB, M - m0);
      __m512 acc[MB];
      for (int i = 0; i < MB; ++i) acc[i] = _mm512_setzero_ps();
      for (int g = 0; g < G; ++g) {
        __m512 t[MB];
        for (int i = 0; i < MB; ++i) t[i] = _mm512_setzero_ps();
        const int k0 = g * gs;
        if constexpr (BF) {
          for (int c = 0; c < gs; c += 64) {
            __m512i w16 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(wr + (k0 + c) / 2)));
            __m512bh lo = (__m512bh)_mm512_permutexvar_epi16(_mm512_and_si512(w16, m0f16), tab);
            __m512bh hi = (__m512bh)_mm512_permutexvar_epi16(_mm512_srli_epi16(w16, 4), tab);
            for (int i = 0; i < mb; ++i) {
              const uint16_t* x = static_cast<const uint16_t*>(xp[m0 + i]) + k0 + c;
              t[i] = _mm512_dpbf16_ps(t[i], lo, (__m512bh)_mm512_loadu_si512(x));
              t[i] = _mm512_dpbf16_ps(t[i], hi, (__m512bh)_mm512_loadu_si512(x + 32));
            }
          }
        } else if constexpr (BITS == 4) {
          for (int c = 0; c < gs; c += 32) {
            __m512i d = _mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(wr + (k0 + c) / 2)));
            __m512 lo = _mm512_cvtepi32_ps(_mm512_and_si512(d, m0f));
            __m512 hi = _mm512_cvtepi32_ps(_mm512_srli_epi32(d, 4));
            for (int i = 0; i < mb; ++i) {
              const float* x = static_cast<const float*>(xp[m0 + i]) + k0 + c;
              t[i] = _mm512_fmadd_ps(lo, _mm512_loadu_ps(x), t[i]);
              t[i] = _mm512_fmadd_ps(hi, _mm512_loadu_ps(x + 16), t[i]);
            }
          }
        } else {
          for (int c = 0; c < gs; c += 16) {
            __m512 q = _mm512_cvtepi32_ps(
                _mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(wr + k0 + c))));
            for (int i = 0; i < mb; ++i)
              t[i] = _mm512_fmadd_ps(q, _mm512_loadu_ps(static_cast<const float*>(xp[m0 + i]) + k0 + c), t[i]);
          }
        }
        const __m512 s = _mm512_set1_ps(sf[g]);
        for (int i = 0; i < mb; ++i) acc[i] = _mm512_fmadd_ps(t[i], s, acc[i]);
      }
      for (int i = 0; i < mb; ++i) {
        float b = 0.f;
        const float* s = xs[m0 + i];
        for (int g = 0; g < G; ++g) b += bf[g] * s[g];
        st(yp[m0 + i], n, hsum(acc[i]) + b, odt);
      }
    }
  }
}

// One x row into the kernel's layout, read with a stride of 1 along k. f32 layout: 4-bit chunks of 32 as [16 even k,
// 16 odd k] (the low and high nibbles of 16 bytes), 8-bit natural order. bf16 layout (bf): groups of 64 as 16 pairs
// (k 4j, 4j+2) then 16 pairs (4j+1, 4j+3), the pairs VDPBF16PS meets against the nibbles of 32 bytes.
void prep_x(const void* x, DT xdt, int K, int gs, int bits, bool bf, void* out, float* sums) {
  if (bf) {
    const uint16_t* xs = static_cast<const uint16_t*>(x);
    uint16_t* o = static_cast<uint16_t*>(out);
    for (int c = 0; c < K; c += 64)
      for (int j = 0; j < 16; ++j) {
        o[c + 2 * j] = xs[c + 4 * j];
        o[c + 2 * j + 1] = xs[c + 4 * j + 2];
        o[c + 32 + 2 * j] = xs[c + 4 * j + 1];
        o[c + 32 + 2 * j + 1] = xs[c + 4 * j + 3];
      }
  } else if (bits == 4) {
    float* o = static_cast<float*>(out);
    for (int c = 0; c < K; c += 32)
      for (int j = 0; j < 16; ++j) {
        o[c + j] = ld(x, c + 2 * j, xdt);
        o[c + 16 + j] = ld(x, c + 2 * j + 1, xdt);
      }
  } else {
    float* o = static_cast<float*>(out);
    for (int k = 0; k < K; ++k) o[k] = ld(x, k, xdt);
  }
  for (int g = 0; g < K / gs; ++g) {
    float s = 0.f;
    for (int k = g * gs; k < (g + 1) * gs; ++k) s += ld(x, k, xdt);
    sums[g] = s;
  }
}

// ---- jobs: a weight matrix and the x rows that meet it ---------------------------------------------------------

struct Job {
  const uint8_t* w;
  const void* sc;
  const void* bi;
  std::vector<size_t> rows;
  std::vector<const void*> xp;
  std::vector<const float*> xs;
  std::vector<void*> yp;
};

void run_jobs(std::vector<Job>& jobs, DT sdt, DT odt, int N, int K, int gs, int bits, bool bf) {
  // Row chunks small enough to balance 6 threads on a 512-row expert, big enough to amortize the task.
  const int chunk = std::max(8, std::min(64, N / 8));
  struct Task { int j, n0, n1; };
  std::vector<Task> tasks;
  for (int j = 0; j < int(jobs.size()); ++j)
    for (int n = 0; n < N; n += chunk) tasks.push_back({j, n, std::min(N, n + chunk)});
  const int nt = int(tasks.size());
  auto one = [&](int i) {
    const Task& t = tasks[i];
    Job& jb = jobs[t.j];
    const int M = int(jb.xp.size());
    auto args = std::make_tuple(jb.w, jb.sc, jb.bi, sdt, K, gs, t.n0, t.n1, M, jb.xp.data(), jb.xs.data(),
                                jb.yp.data(), odt);
    if (bf)
      std::apply(rows_kernel<4, true>, args);
    else if (bits == 4)
      std::apply(rows_kernel<4, false>, args);
    else
      std::apply(rows_kernel<8, false>, args);
  };
  if (nt == 1 || g_threads == 1) {
    for (int i = 0; i < nt; ++i) one(i);
    return;
  }
#pragma omp parallel for schedule(dynamic, 1) num_threads(g_threads)
  for (int i = 0; i < nt; ++i) one(i);
}

// Element offset of flat row r of an array whose last axis is k (stride 1), from its shape and strides.
size_t row_offset(const mx::Shape& shape, const mx::Strides& strides, size_t r) {
  size_t off = 0;
  for (int d = int(shape.size()) - 2; d >= 0; --d) {
    off += (r % shape[d]) * strides[d];
    r /= shape[d];
  }
  return off;
}

// ---- the primitive ----------------------------------------------------------------------------------------------

class CpuQmm : public mx::Primitive {
 public:
  CpuQmm(mx::Stream s, int gs, int bits, bool gather) : mx::Primitive(s), gs_(gs), bits_(bits), gather_(gather) {}

  void eval_gpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error("cpu_kern: CPU device only");
  }
  const char* name() const override { return gather_ ? "CpuGatherQmm" : "CpuQmm"; }
  bool is_equivalent(const mx::Primitive& o) const override {
    auto& p = static_cast<const CpuQmm&>(o);
    return gs_ == p.gs_ && bits_ == p.bits_ && gather_ == p.gather_;
  }

  void eval_cpu(const std::vector<mx::array>& in, std::vector<mx::array>& outs) override {
    auto& y = outs[0];
    y.set_data(mx::allocator::malloc(y.nbytes()));
    const mx::array &x = in[0], &w = in[1], &sc = in[2], &bi = in[3];
    for (auto* a : {&w, &sc, &bi})
      if (!a->flags().row_contiguous) throw std::runtime_error("cpu_kern: weights, scales and biases must be contiguous");
    if (x.strides().back() != 1 && x.shape().back() != 1) throw std::runtime_error("cpu_kern: x needs stride 1 along k");
    // Everything the task needs, by value: the eval loop keeps the buffers alive until a later task runs.
    struct Args {
      const void* x; DT xdt; mx::Shape xshape; mx::Strides xstrides;
      const uint8_t* w; const void* sc; const void* bi; DT sdt;
      void* y; DT ydt; int N, K, M, E; size_t rows;
      std::vector<uint32_t> idx;  // gather: the expert of each batch element
      const void* ip; mx::Shape ishape; mx::Strides istrides;
    } a;
    a.x = x.data<void>();
    a.xdt = dt_of(x.dtype());
    a.xshape = x.shape();
    a.xstrides = x.strides();
    a.w = w.data<uint8_t>();
    a.sc = sc.data<void>();
    a.bi = bi.data<void>();
    a.sdt = dt_of(sc.dtype());
    if (dt_of(bi.dtype()) != a.sdt) throw std::runtime_error("cpu_kern: scales and biases must share a dtype");
    a.y = y.data<void>();
    a.ydt = dt_of(y.dtype());
    a.K = x.shape().back();
    a.N = w.shape(-2);
    a.E = gather_ ? w.shape(0) : 1;
    a.M = gather_ ? x.shape(-2) : 1;
    a.rows = x.size() / a.K;  // gather: batch elements x M
    a.ip = nullptr;
    if (gather_) {
      const mx::array& ix = in[4];
      a.ip = ix.data<void>();
      a.ishape = ix.shape();
      a.istrides = ix.strides();
    }
    const int gs = gs_, bits = bits_;
    const bool gather = gather_;
    auto& enc = mx::cpu::get_command_encoder(stream());
    enc.dispatch([a = std::move(a), gs, bits, gather]() mutable {
      const int K = a.K, N = a.N, G = K / gs;
      const size_t R = a.rows;
      const size_t osz = a.ydt == F32 ? 4 : 2;
      const size_t wbytes = size_t(N) * K * bits / 8, sbytes = size_t(N) * G * (a.sdt == F32 ? 4 : 2);
      // Which rows meet which weights: one job for a plain matmul, one per expert for the gather.
      std::vector<Job> jobs;
      if (!gather) {
        jobs.push_back(Job{a.w, a.sc, a.bi, {}, {}, {}, {}});
        for (size_t r = 0; r < R; ++r) jobs[0].rows.push_back(r);
      } else {
        const size_t B = R / a.M;
        std::vector<int> job_of(a.E, -1);
        const int nd = int(a.ishape.size());
        for (size_t b = 0; b < B; ++b) {
          size_t off = 0, r = b;
          for (int d = nd - 1; d >= 0; --d) {
            off += (r % a.ishape[d]) * a.istrides[d];
            r /= a.ishape[d];
          }
          uint32_t e = static_cast<const uint32_t*>(a.ip)[off];
          if (e >= uint32_t(a.E)) {  // out of range: zeros, since a throw here would end MLX's stream thread
            std::memset(static_cast<char*>(a.y) + b * a.M * N * osz, 0, a.M * N * osz);
            continue;
          }
          if (job_of[e] < 0) {
            job_of[e] = int(jobs.size());
            jobs.push_back(Job{a.w + e * wbytes, static_cast<const char*>(a.sc) + e * sbytes,
                               static_cast<const char*>(a.bi) + e * sbytes, {}, {}, {}, {}});
          }
          for (int m = 0; m < a.M; ++m) jobs[job_of[e]].rows.push_back(b * a.M + m);
        }
      }
      const bool bf = bits == 4 && gs % 64 == 0 && a.xdt == BF16 && R >= g_bf_rows * jobs.size();
      const size_t xb = bf ? 2 : 4;
      std::vector<char> xbuf(R * K * xb);
      std::vector<float> sbuf(R * G);
      // x rows into the kernel's layout, threaded when there are many (prefill).
      auto prep = [&](size_t r) {
        const char* xr = static_cast<const char*>(a.x) + row_offset(a.xshape, a.xstrides, r) * (a.xdt == F32 ? 4 : 2);
        prep_x(xr, a.xdt, K, gs, bits, bf, xbuf.data() + r * K * xb, sbuf.data() + r * G);
      };
      if (R >= 64) {
#pragma omp parallel for num_threads(g_threads)
        for (size_t r = 0; r < R; ++r) prep(r);
      } else {
        for (size_t r = 0; r < R; ++r) prep(r);
      }
      for (auto& jb : jobs)
        for (size_t r : jb.rows) {
          jb.xp.push_back(xbuf.data() + r * K * xb);
          jb.xs.push_back(sbuf.data() + r * G);
          jb.yp.push_back(static_cast<char*>(a.y) + r * N * osz);
        }
      run_jobs(jobs, a.sdt, a.ydt, N, K, gs, bits, bf);
    });
  }

 private:
  int gs_, bits_;
  bool gather_;
};

void check(const mx::array& x, const mx::array& w, const mx::array& sc, int gs, int bits) {
  if (bits != 4 && bits != 8) throw std::invalid_argument("cpu_kern: bits 4 or 8");
  if (gs % 32 != 0) throw std::invalid_argument("cpu_kern: group size a multiple of 32");
  int K = x.shape().back();
  if (K % gs != 0 || K / gs > 1024) throw std::invalid_argument("cpu_kern: K must be a multiple of the group size");
  if (w.shape().back() * 32 / bits != K) throw std::invalid_argument("cpu_kern: w does not match x's last axis");
  if (sc.shape().back() != K / gs) throw std::invalid_argument("cpu_kern: scales do not match");
  if (w.dtype() != mx::uint32) throw std::invalid_argument("cpu_kern: w must be uint32");
}

mx::array qmm(const mx::array& x, const mx::array& w, const mx::array& sc, const mx::array& bi, int gs, int bits,
              mx::StreamOrDevice s) {
  check(x, w, sc, gs, bits);
  if (w.ndim() != 2) throw std::invalid_argument("cpu_kern.qmm: w must be 2-D");
  auto shape = x.shape();
  shape.back() = w.shape(0);
  auto dt = mx::promote_types(x.dtype(), sc.dtype());
  return mx::array(shape, dt, std::make_shared<CpuQmm>(mx::to_stream(s), gs, bits, false), {x, w, sc, bi});
}

// rhs_indices only (no lhs): x [..., M, K], w [E, N, K*bits/32], idx broadcast with x's batch dims.
mx::array gather_qmm(const mx::array& x, const mx::array& w, const mx::array& sc, const mx::array& bi,
                     const mx::array& idx, int gs, int bits, mx::StreamOrDevice s) {
  check(x, w, sc, gs, bits);
  if (w.ndim() != 3) throw std::invalid_argument("cpu_kern.gather_qmm: w must be [E, N, K']");
  if (x.ndim() < 2) throw std::invalid_argument("cpu_kern.gather_qmm: x must be [..., M, K]");
  mx::Shape xb(x.shape().begin(), x.shape().end() - 2);
  auto batch = mx::broadcast_shapes(xb, idx.shape());
  mx::Shape xs = batch;
  xs.push_back(x.shape(-2));
  xs.push_back(x.shape(-1));
  auto st = mx::to_stream(s);
  auto xbc = mx::broadcast_to(x, xs, st);
  auto ibc = mx::broadcast_to(mx::astype(idx, mx::uint32, st), batch, st);
  mx::Shape out = batch;
  out.push_back(x.shape(-2));
  out.push_back(w.shape(1));
  auto dt = mx::promote_types(x.dtype(), sc.dtype());
  return mx::array(out, dt, std::make_shared<CpuQmm>(st, gs, bits, true), {xbc, w, sc, bi, ibc});
}

// ---- GatedDeltaNet recurrence ----------------------------------------------------------------------------------
//
// mlx_lm's gated_delta_ops for scalar gating and no mask, one pass over each head's state a step: decay, read k,
// write the delta, read q. The state update is a multiply then an add (no FMA), as MLX's elementwise ops round it.

class CpuGdn : public mx::Primitive {
 public:
  explicit CpuGdn(mx::Stream s) : mx::Primitive(s) {}
  void eval_gpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error("cpu_kern: CPU device only");
  }
  const char* name() const override { return "CpuGdn"; }
  bool is_equivalent(const mx::Primitive&) const override { return true; }

  void eval_cpu(const std::vector<mx::array>& in, std::vector<mx::array>& outs) override {
    for (auto& a : in)
      if (!a.flags().row_contiguous) throw std::runtime_error("cpu_kern.gdn: inputs must be contiguous");
    auto &y = outs[0], &so = outs[1];
    y.set_data(mx::allocator::malloc(y.nbytes()));
    so.set_data(mx::allocator::malloc(so.nbytes()));
    const auto& q = in[0];
    const auto& v = in[2];
    struct A {
      const void *q, *k, *v, *g, *b; const float* s0; void* y; float* s;
      DT qdt, vdt, gdt, bdt, ydt; int B, T, Hk, Dk, Hv, Dv;
    } a{q.data<void>(), in[1].data<void>(), v.data<void>(), in[3].data<void>(), in[4].data<void>(),
        in[5].data<float>(), y.data<void>(), so.data<float>(), dt_of(q.dtype()), dt_of(v.dtype()),
        dt_of(in[3].dtype()), dt_of(in[4].dtype()), dt_of(y.dtype()), q.shape(0), q.shape(1), q.shape(2),
        q.shape(3), v.shape(2), v.shape(3)};
    auto& enc = mx::cpu::get_command_encoder(stream());
    enc.dispatch([a]() {
      const int rf = a.Hv / a.Hk, Dk = a.Dk, Dv = a.Dv;
      const int tasks = a.B * a.Hv;
#pragma omp parallel for schedule(dynamic, 1) num_threads(g_threads)
      for (int bh = 0; bh < tasks; ++bh) {
        const int b = bh / a.Hv, h = bh % a.Hv, hk = h / rf;
        float* S = a.s + size_t(bh) * Dv * Dk;
        std::memcpy(S, a.s0 + size_t(bh) * Dv * Dk, sizeof(float) * Dv * Dk);
        float kq[2][512];
        for (int t = 0; t < a.T; ++t) {
          const size_t qo = ((size_t(b) * a.T + t) * a.Hk + hk) * Dk;
          const size_t vo = ((size_t(b) * a.T + t) * a.Hv + h) * Dv;
          const size_t go = (size_t(b) * a.T + t) * a.Hv + h;
          for (int i = 0; i < Dk; ++i) {
            kq[0][i] = ld(a.k, qo + i, a.qdt);
            kq[1][i] = ld(a.q, qo + i, a.qdt);
          }
          const __m512 g = _mm512_set1_ps(ld(a.g, go, a.gdt));
          const float beta = ld(a.b, go, a.bdt);
          for (int i = 0; i < Dv; ++i) {
            float* row = S + size_t(i) * Dk;
            __m512 kv = _mm512_setzero_ps();
            for (int j = 0; j < Dk; j += 16) {
              __m512 sr = _mm512_mul_ps(_mm512_loadu_ps(row + j), g);
              _mm512_storeu_ps(row + j, sr);
              kv = _mm512_fmadd_ps(sr, _mm512_loadu_ps(kq[0] + j), kv);
            }
            const float delta = (ld(a.v, vo + i, a.vdt) - hsum(kv)) * beta;
            const __m512 dv = _mm512_set1_ps(delta);
            __m512 yv = _mm512_setzero_ps();
            for (int j = 0; j < Dk; j += 16) {
              __m512 sr = _mm512_add_ps(_mm512_loadu_ps(row + j), _mm512_mul_ps(_mm512_loadu_ps(kq[0] + j), dv));
              _mm512_storeu_ps(row + j, sr);
              yv = _mm512_fmadd_ps(sr, _mm512_loadu_ps(kq[1] + j), yv);
            }
            st(a.y, vo + i, hsum(yv), a.ydt);
          }
        }
      }
    });
  }
};

// q, k [B, T, Hk, Dk]; v [B, T, Hv, Dv]; g, beta [B, T, Hv]; state [B, Hv, Dv, Dk] f32 -> (y [B, T, Hv, Dv], state).
std::vector<mx::array> gdn(const mx::array& q, const mx::array& k, const mx::array& v, const mx::array& g,
                           const mx::array& beta, const mx::array& state, mx::StreamOrDevice s) {
  if (q.ndim() != 4 || v.ndim() != 4 || state.ndim() != 4 || g.ndim() != 3 || beta.ndim() != 3)
    throw std::invalid_argument("cpu_kern.gdn: shapes");
  if (q.shape(3) % 16 || q.shape(3) > 512 || v.shape(2) % q.shape(2) || state.dtype() != mx::float32)
    throw std::invalid_argument("cpu_kern.gdn: unsupported head sizes or state dtype");
  if (k.dtype() != q.dtype()) throw std::invalid_argument("cpu_kern.gdn: q and k must share a dtype");
  auto st = mx::to_stream(s);
  std::vector<mx::array> ins;
  for (auto* a : {&q, &k, &v, &g, &beta, &state}) ins.push_back(mx::contiguous(*a, false, st));
  return mx::array::make_arrays({v.shape(), state.shape()}, {q.dtype(), mx::float32}, std::make_shared<CpuGdn>(st),
                                ins);
}

// ---- attention ---------------------------------------------------------------------------------------------------
//
// softmax(scale * q k^T + mask) v with f32 scores, an online softmax and f32 accumulation. A task is one query position
// t with the G query heads that share a KV head, over a range of keys; with few such rows (decode) the keys are split
// into ranges and the partial results merged, so every thread has work.

inline __m512 ld16(const void* p, size_t i, DT t) {
  if (t == F32) return _mm512_loadu_ps(static_cast<const float*>(p) + i);
  __m256i h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(static_cast<const uint16_t*>(p) + i));
  if (t == BF16) return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(h), 16));
  return _mm512_cvtph_ps(h);
}

struct Part {
  float m, l;  // running max and sum of exp
};

class CpuAttn : public mx::Primitive {
 public:
  CpuAttn(mx::Stream s, float scale, int mask_kind) : mx::Primitive(s), scale_(scale), mask_(mask_kind) {}
  void eval_gpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error("cpu_kern: CPU device only");
  }
  const char* name() const override { return "CpuAttn"; }
  bool is_equivalent(const mx::Primitive& o) const override {
    auto& p = static_cast<const CpuAttn&>(o);
    return scale_ == p.scale_ && mask_ == p.mask_;
  }

  void eval_cpu(const std::vector<mx::array>& in, std::vector<mx::array>& outs) override {
    auto& o = outs[0];
    o.set_data(mx::allocator::malloc(o.nbytes()));
    const auto &q = in[0], &k = in[1], &v = in[2];
    for (auto* a : {&q, &k, &v})
      if (a->strides().back() != 1) throw std::runtime_error("cpu_kern.attn: last axis must have stride 1");
    struct A {
      const void *q, *k, *v; const bool* m; void* o;
      DT qdt, kdt, vdt, odt; int B, Hq, T, D, Hkv, N, Dv;
      int64_t qs[3], ks[3], vs[3], ms[2];
      float scale; int mask;
    } a{};
    a.q = q.data<void>(); a.k = k.data<void>(); a.v = v.data<void>(); a.o = o.data<void>();
    a.qdt = dt_of(q.dtype()); a.kdt = dt_of(k.dtype()); a.vdt = dt_of(v.dtype()); a.odt = dt_of(o.dtype());
    a.B = q.shape(0); a.Hq = q.shape(1); a.T = q.shape(2); a.D = q.shape(3);
    a.Hkv = k.shape(1); a.N = k.shape(2); a.Dv = v.shape(3);
    for (int i = 0; i < 3; ++i) { a.qs[i] = q.strides()[i]; a.ks[i] = k.strides()[i]; a.vs[i] = v.strides()[i]; }
    a.scale = scale_; a.mask = mask_;
    if (mask_ == 2) {
      const auto& m = in[3];
      a.m = m.data<bool>();
      a.ms[0] = m.strides()[m.ndim() - 2];
      a.ms[1] = m.strides()[m.ndim() - 1];
    }
    auto& enc = mx::cpu::get_command_encoder(stream());
    enc.dispatch([a]() {
      const int G = a.Hq / a.Hkv, D = a.D, Dv = a.Dv, N = a.N;
      const int rows = a.B * a.Hkv * a.T;
      int P = 1;  // key ranges per row group
      if (rows < 2 * g_threads) P = std::max(1, std::min({(2 * g_threads + rows - 1) / rows, N / 256, 64}));
      const int span = (N + P - 1) / P;
      std::vector<float> acc(size_t(rows) * P * G * Dv);
      std::vector<Part> parts(size_t(rows) * P * G);
#pragma omp parallel for schedule(dynamic, 1) num_threads(g_threads)
      for (int task = 0; task < rows * P; ++task) {
        const int p = task % P, row = task / P;
        const int t = row % a.T, hk = (row / a.T) % a.Hkv, b = row / (a.T * a.Hkv);
        const int lim = a.mask == 1 ? std::min(N, N - a.T + t + 1) : N;  // causal: keys 0 .. N - T + t
        const int n0 = p * span, n1 = std::min(lim, n0 + span);
        float* ac = acc.data() + (size_t(row) * P + p) * G * Dv;
        Part* pt = parts.data() + (size_t(row) * P + p) * G;
        float qf[16][512];
        for (int g = 0; g < G; ++g) {
          const size_t qo = b * a.qs[0] + (hk * G + g) * a.qs[1] + t * a.qs[2];
          for (int d = 0; d < D; ++d) qf[g][d] = ld(a.q, qo + d, a.qdt) * a.scale;
          pt[g] = {-INFINITY, 0.f};
          std::fill(ac + g * Dv, ac + (g + 1) * Dv, 0.f);
        }
        constexpr int KB = 32;
        float s[16][KB];
        for (int j0 = n0; j0 < n1; j0 += KB) {
          const int nb = std::min(KB, n1 - j0);
          for (int j = 0; j < nb; ++j) {
            const int n = j0 + j;
            const size_t ko = b * a.ks[0] + hk * a.ks[1] + n * a.ks[2];
            __m512 dot[16];
            for (int g = 0; g < G; ++g) dot[g] = _mm512_setzero_ps();
            for (int d = 0; d < D; d += 16) {
              __m512 kk = ld16(a.k, ko + d, a.kdt);
              for (int g = 0; g < G; ++g) dot[g] = _mm512_fmadd_ps(kk, _mm512_loadu_ps(qf[g] + d), dot[g]);
            }
            const bool keep = a.mask != 2 || a.m[t * a.ms[0] + n * a.ms[1]];
            for (int g = 0; g < G; ++g) s[g][j] = keep ? hsum(dot[g]) : -INFINITY;
          }
          for (int g = 0; g < G; ++g) {
            float mx_ = pt[g].m;
            for (int j = 0; j < nb; ++j) mx_ = std::max(mx_, s[g][j]);
            if (mx_ == -INFINITY) continue;
            const float c = std::exp(pt[g].m - mx_);
            float l = pt[g].l * c;
            for (int j = 0; j < nb; ++j) {
              s[g][j] = std::exp(s[g][j] - mx_);
              l += s[g][j];
            }
            pt[g] = {mx_, l};
            float* ag = ac + g * Dv;
            const __m512 cv = _mm512_set1_ps(c);
            for (int d = 0; d < Dv; d += 16) _mm512_storeu_ps(ag + d, _mm512_mul_ps(_mm512_loadu_ps(ag + d), cv));
          }
          for (int j = 0; j < nb; ++j) {
            const size_t vo = b * a.vs[0] + hk * a.vs[1] + (j0 + j) * a.vs[2];
            for (int d = 0; d < Dv; d += 16) {
              __m512 vv = ld16(a.v, vo + d, a.vdt);
              for (int g = 0; g < G; ++g) {
                float* ag = ac + g * Dv + d;
                _mm512_storeu_ps(ag, _mm512_fmadd_ps(vv, _mm512_set1_ps(s[g][j]), _mm512_loadu_ps(ag)));
              }
            }
          }
        }
      }
      // Merge the key ranges of each row group and write [B, Hq, T, Dv].
#pragma omp parallel for num_threads(g_threads)
      for (int row = 0; row < rows; ++row) {
        const int t = row % a.T, hk = (row / a.T) % a.Hkv, b = row / (a.T * a.Hkv);
        for (int g = 0; g < G; ++g) {
          float m = -INFINITY;
          for (int p = 0; p < P; ++p) m = std::max(m, parts[(size_t(row) * P + p) * G + g].m);
          float l = 0.f, w[64];
          for (int p = 0; p < P; ++p) {
            const Part& pp = parts[(size_t(row) * P + p) * G + g];
            w[p] = pp.m == -INFINITY ? 0.f : std::exp(pp.m - m);
            l += pp.l * w[p];
          }
          const float inv = l > 0.f ? 1.f / l : 0.f;
          const size_t oo = ((size_t(b) * a.Hq + hk * G + g) * a.T + t) * Dv;
          for (int d = 0; d < Dv; ++d) {
            float x = 0.f;
            for (int p = 0; p < P; ++p) x += acc[((size_t(row) * P + p) * G + g) * Dv + d] * w[p];
            st(a.o, oo + d, x * inv, a.odt);
          }
        }
      }
    });
  }

 private:
  float scale_;
  int mask_;  // 0 none, 1 causal (bottom-right), 2 boolean array [..., T, N]
};

mx::array attn(const mx::array& q, const mx::array& k, const mx::array& v, float scale,
               const std::optional<std::string>& causal, const std::optional<mx::array>& mask, mx::StreamOrDevice s) {
  if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4) throw std::invalid_argument("cpu_kern.attn: 4-D q, k, v");
  const int G = q.shape(1) / k.shape(1);
  if (q.shape(1) % k.shape(1) || G > 16 || q.shape(3) % 16 || q.shape(3) > 512 || v.shape(3) % 16 ||
      q.shape(3) != k.shape(3) || k.shape(2) != v.shape(2) || q.shape(0) != k.shape(0))
    throw std::invalid_argument("cpu_kern.attn: unsupported shapes");
  auto st = mx::to_stream(s);
  mx::Shape os = q.shape();
  os.back() = v.shape(3);
  std::vector<mx::array> ins{q, k, v};
  int kind = causal ? 1 : 0;
  if (mask) {
    if (mask->dtype() != mx::bool_ || mask->ndim() < 2 || mask->shape(-2) != q.shape(2) ||
        mask->shape(-1) != k.shape(2))
      throw std::invalid_argument("cpu_kern.attn: a boolean [..., T, N] mask only");
    for (int d = 0; d < mask->ndim() - 2; ++d)
      if (mask->shape(d) != 1) throw std::invalid_argument("cpu_kern.attn: mask leading dims must be 1");
    ins.push_back(*mask);
    kind = 2;
  }
  return mx::array(os, q.dtype(), std::make_shared<CpuAttn>(st, scale, kind), ins);
}

}  // namespace

NB_MODULE(cpu_kern, m) {
  m.def("attn", &attn, "q"_a, "k"_a, "v"_a, "scale"_a, "causal"_a = nb::none(), "mask"_a = nb::none(), nb::kw_only(),
        "stream"_a = nb::none());
  m.def("gdn", &gdn, "q"_a, "k"_a, "v"_a, "g"_a, "beta"_a, "state"_a, nb::kw_only(), "stream"_a = nb::none());
  m.doc() = "Threaded AVX-512 kernels for MLX's CPU device";
  m.def("qmm", &qmm, "x"_a, "w"_a, "scales"_a, "biases"_a, "group_size"_a = 64, "bits"_a = 4, nb::kw_only(),
        "stream"_a = nb::none());
  m.def("gather_qmm", &gather_qmm, "x"_a, "w"_a, "scales"_a, "biases"_a, "rhs_indices"_a, "group_size"_a = 64,
        "bits"_a = 4, nb::kw_only(), "stream"_a = nb::none());
  m.def("set_threads", [](int n) { g_threads = std::max(1, n); }, "n"_a);
  m.def("threads", []() { return g_threads; });
  m.def("set_bf_rows", [](size_t n) { g_bf_rows = n; }, "n"_a);
}
