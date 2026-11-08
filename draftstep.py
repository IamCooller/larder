"""Fused T=1 decode step for a small dense draft, one mx.compile'd pure function of arrays
in which every piece but the screened head is the same arithmetic as the MLX kernel it
replaces."""

import argparse
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.activations import swiglu
from mlx_lm.models.cache import KVCache

GROUP_SIZE = 64
BITS = 4
PIECES = ("attn", "qkv", "o", "mlp", "down", "head", "screen")   # attn first: the rest ride on it
EXACT = PIECES[:-1]      # bit-identical to stock; "screen" is exact only on its candidates

# ---------------------------------------------------------------- kernels
# nq_dot is MLX's qdot<float, 16, 4> (kernels/quantized.h) verbatim; the
# caller replays load_vector<T, float, 16, 4> in front of it.
HEADER = """
inline float nq_dot(const device uint8_t* w, thread const float* x_thread,
                    float scale, float bias, float sum) {
  float accum = 0;
  const device uint16_t* ws = (const device uint16_t*)w;
  for (int i = 0; i < 4; i++) {
    accum +=
        (x_thread[4 * i] * (ws[i] & 0x000f) +
         x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
         x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
         x_thread[4 * i + 3] * (ws[i] & 0xf000));
  }
  return scale * accum + sum * bias;
}

// unary_ops.h Sigmoid and binary_ops.h Multiply, as the compiled SwiGLU calls them
struct NqSigmoid {
  template <typename T>
  T operator()(T x) {
    auto y = 1 / (1 + metal::exp(metal::abs(x)));
    return (x < 0) ? y : 1 - y;
  }
};
struct NqMultiply {
  template <typename T>
  T operator()(T x, T y) {
    return x * y;
  }
};

// rms_single_row's inverse RMS for a K-long row, in its reduction order:
// thread t of the (K/4)-thread group sums x[4t..4t+3]^2, each simdgroup of
// the group simd_sums, then one simd_sum over the per-simdgroup sums padded
// with zeros. One simdgroup here plays all K/128 of them.
template <typename T>
inline float nq_inv_rms(const device T* x, int K, float eps, uint lane) {
  float mine = 0;
  for (int s = 0; s < K / 128; s++) {
    const device T* xs = x + 4 * (s * 32 + lane);
    float acc = 0;
    for (int i = 0; i < 4; i++) {
      float xi = xs[i];
      acc += xi * xi;
    }
    acc = simd_sum(acc);
    if (lane == uint(s)) {
      mine = acc;
    }
  }
  float tot = simd_sum(mine);
  return metal::precise::rsqrt(tot / K + eps);
}
"""

# qmv_fast for NV input vectors at once, each weight word loaded once for all of them,
# with the optional NORM, NMAT, PAIR and RESID pieces around it.
NQMV_SOURCE = """
  const int KW = K / 2;
  const int KG = K / 64;
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const int out_row = int(threadgroup_position_in_grid.y) * (NSG * RPS) + int(sg) * RPS;

  const device uint32_t* wa = w0;
  const device T* sa = s0;
  const device T* ba = b0;
  int r0 = out_row;
  MATSEL
  RESIDLOAD

  NORMINV

  const device uint8_t* ws = (const device uint8_t*)wa + r0 * KW + lane * 8;
  const device T* sc = sa + r0 * KG + lane / 4;
  const device T* bi = ba + r0 * KG + lane / 4;
  PAIRPTR
  const device T* xp = x + lane * 16;
  NORMPTR

  float result[NV][RPS];
  float result2[NV][RPS];
  for (int v = 0; v < NV; v++) {
    for (int row = 0; row < RPS; row++) {
      result[v][row] = 0;
      result2[v][row] = 0;
    }
  }
  for (int k = 0; k < K; k += 512) {
    // load_vector<T, float, 16, 4>, with x possibly normalised on the way in
    float x_thread[NV][16];
    float sum[NV];
    for (int v = 0; v < NV; v++) {
      sum[v] = 0;
      for (int i = 0; i < 16; i += 4) {
        const T a0 = XV(v, i), a1 = XV(v, i + 1), a2 = XV(v, i + 2), a3 = XV(v, i + 3);
        sum[v] += a0 + a1 + a2 + a3;
        x_thread[v][i] = a0;
        x_thread[v][i + 1] = a1 / 16.0f;
        x_thread[v][i + 2] = a2 / 256.0f;
        x_thread[v][i + 3] = a3 / 4096.0f;
      }
    }
    for (int row = 0; row < RPS; row++) {
      float s = sc[row * KG];
      float b = bi[row * KG];
      for (int v = 0; v < NV; v++) {
        result[v][row] += nq_dot(ws + row * KW, x_thread[v], s, b, sum[v]);
        PAIRDOT
      }
    }
    ws += 256;
    sc += 8;
    bi += 8;
    xp += 512;
    PAIRADV
    NORMADV
  }

  for (int v = 0; v < NV; v++) {
    for (int row = 0; row < RPS; row++) {
      result[v][row] = simd_sum(result[v][row]);
      PAIRSUM
      if (lane == 0) {
        STORE
      }
    }
  }
"""


def _nqmv_source(nmat, pair, norm, resid):
    matsel = ""
    if nmat >= 2:
        matsel += """
  if (out_row >= N0) { wa = w1; sa = s1; ba = b1; r0 = out_row - N0; }"""
    if nmat == 3:
        matsel += """
  if (out_row >= N0 + N1) { wa = w2; sa = s2; ba = b2; r0 = out_row - N0 - N1; }"""
    rep = {
        "MATSEL": matsel,
        "RESIDLOAD": """T rv[NV][RPS];
  for (int v = 0; v < NV; v++)
    for (int row = 0; row < RPS; row++) rv[v][row] = r[v * NOUT + out_row + row];""" if resid else "",
        "NORMINV": """float inv[NV];
  for (int v = 0; v < NV; v++) inv[v] = nq_inv_rms<T>(x + v * K, K, prm[0], lane);""" if norm else "",
        "PAIRPTR": """
  const device uint8_t* ws2 = (const device uint8_t*)w1 + r0 * KW + lane * 8;
  const device T* sc2 = s1 + r0 * KG + lane / 4;
  const device T* bi2 = b1 + r0 * KG + lane / 4;""" if pair else "",
        "PAIRDOT": """
        result2[v][row] += nq_dot(ws2 + row * KW, x_thread[v], float(sc2[row * KG]),
                                  float(bi2[row * KG]), sum[v]);""" if pair else "",
        "PAIRADV": "ws2 += 256; sc2 += 8; bi2 += 8;" if pair else "",
        "PAIRSUM": "result2[v][row] = simd_sum(result2[v][row]);" if pair else "",
        "NORMPTR": "const device T* gp = g + lane * 16;" if norm else "",
        "NORMADV": "gp += 512;" if norm else "",
    }
    if pair:
        store = """T gt = static_cast<T>(result[v][row]);
        T ut = static_cast<T>(result2[v][row]);
        y[v * NOUT + out_row + row] = NqMultiply()(NqMultiply()(gt, NqSigmoid()(gt)), ut);"""
    elif resid:
        store = "y[v * NOUT + out_row + row] = rv[v][row] + static_cast<T>(result[v][row]);"
    else:
        store = "y[v * NOUT + out_row + row] = static_cast<T>(result[v][row]);"
    rep["STORE"] = store
    src = NQMV_SOURCE
    for k, v in rep.items():
        src = src.replace(k, v)
    xv = ("(gp[J] * static_cast<T>(static_cast<float>(xp[(V) * K + (J)]) * inv[V]))" if norm
          else "(xp[(V) * K + (J)])")
    return "#define XV(V, J) " + xv + "\n" + src + "\n#undef XV\n"


_NQMV = {}


def _nqmv_kernel(nmat, pair, norm, resid):
    key = (nmat, pair, norm, resid)
    if key not in _NQMV:
        names = ["x"]
        if norm:
            names += ["g", "prm"]
        for i in range(2 if pair else nmat):
            names += [f"w{i}", f"s{i}", f"b{i}"]
        if resid:
            names += ["r"]
        _NQMV[key] = mx.fast.metal_kernel(
            name="nqmv_%d%d%d%d" % key,
            input_names=names, output_names=["y"],
            source=_nqmv_source(*key), header=HEADER)
    return _NQMV[key]


def _qparts(m):
    return [m["weight"], m["scales"], m["biases"]]


# rows per simdgroup, simdgroups per threadgroup: any split gives the same
# numbers, so this is purely a speed knob; stock qmv_fast uses (4, 2)
TILE = (4, 2)


def nqmv(x, mods, norm=None, prm=None, pair=False, resid=None, tile=None):
    """x (NV, K) -> (NV, N) with every piece of NQMV_SOURCE the arguments ask
    for. mods: quantized linears (1..3 stacked by rows, or [gate, up] with
    pair)."""
    K = x.shape[-1]
    x = x.reshape(-1, K)
    nv = x.shape[0]
    rps, nsg = tile or TILE
    rows = [m["scales"].shape[0] for m in mods]
    N = rows[0] if pair else sum(rows)
    # 512: qmv_fast's K block; 4096 and 128: rms_single_row's row layout
    assert K % 512 == 0 and (norm is None or K <= 4096), K
    assert all(n % (rps * nsg) == 0 for n in rows), (rows, rps, nsg)
    inputs = [x]
    if norm is not None:
        inputs += [norm, prm]
    for m in mods:
        inputs += _qparts(m)
    if resid is not None:
        inputs.append(resid.reshape(nv, N))
    nmat = 1 if pair else len(mods)
    tmpl = [("T", x.dtype), ("K", K), ("NV", nv), ("NOUT", N), ("RPS", rps), ("NSG", nsg)]
    if nmat >= 2:
        tmpl += [("N0", rows[0]), ("N1", rows[1])]
    return _nqmv_kernel(nmat, pair, norm is not None, resid is not None)(
        inputs=inputs, template=tmpl,
        grid=(32, nsg * (N // (rps * nsg)), 1), threadgroup=(32, nsg, 1),
        output_shapes=[(nv, N)], output_dtypes=[x.dtype])[0]


# sdpa_vector's layout with its arithmetic carried over unchanged and only who runs it
# moved: VS virtual simdgroups per real one, QH query heads per threadgroup, NV new
# rows.
ATTN_SOURCE = """
  constexpr int BN = 32;
  constexpr int D = 128;
  constexpr int NSG = BN / VS;
  constexpr int M = NV * QH;          // query slots: new row m / QH, head m % QH
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const int hq0 = int(threadgroup_position_in_grid.x) * QH;
  const int hk = hq0 / GQA;
  const int p = off[0];
  const int N = p + NV;               // row r sees positions < p + r + 1
  const float eps = prm[0];
  const float scale = prm[1];
  const float lbase = prm[2];

  const int64_t khs = kc_strides[1], kss = kc_strides[2];
  const int64_t vhs = vc_strides[1], vss = vc_strides[2];
  const int64_t qrs = q_strides[0], krs = k_strides[0], vrs = v_strides[0];

  // RMSNorm(128) * w, then RoPE at its own position, for the 4 dims this lane
  // owns: the QH query heads of every new row, and every new row's key
  T qn[M][4];
  T kn[NV][4];
  T vn[NV][4];
  for (int t = 0; t < M + NV; t++) {
    const int j = t < M ? t / QH : t - M;           // which new row
    const device T* src = t < M ? q + j * qrs + (hq0 + t % QH) * D : k + j * krs + hk * D;
    const device T* w = t < M ? qw : kw;
    const float L = 1.0f * static_cast<float>(p + j);
    float xf[4];
    float acc = 0;
    for (int e = 0; e < 4; e++) {
      xf[e] = src[4 * lane + e];
      acc += xf[e] * xf[e];
    }
    acc = simd_sum(acc);
    float inv = metal::precise::rsqrt(acc / 128 + eps);
    for (int e = 0; e < 4; e++) {
      float d = static_cast<float>((4 * int(lane) + e) % 64) / 64.0f;
      float inv_freq = metal::exp2(-d * lbase);
      float theta = L * inv_freq;
      float cs = metal::fast::cos(theta);
      float sn = metal::fast::sin(theta);
      T nv = w[4 * lane + e] * static_cast<T>(xf[e] * inv);
      float me = static_cast<float>(nv);
      float other = simd_shuffle_xor(me, ushort(16));
      float rx;
      if (lane < 16) {
        rx = me * cs - other * sn;
      } else {
        rx = other * sn + me * cs;
      }
      if (t < M) {
        qn[t][e] = static_cast<T>(rx);
      } else {
        kn[t - M][e] = static_cast<T>(rx);
      }
    }
  }
  for (int j = 0; j < NV; j++) {
    for (int e = 0; e < 4; e++) {
      vn[j][e] = v[j * vrs + hk * D + 4 * lane + e];
    }
  }

  // the new rows go into the cache, once per kv head
  if (hq0 % GQA == 0 && sg == 0) {
    for (int j = 0; j < NV; j++) {
      device T* kd = (device T*)kc + hk * khs + (p + j) * kss + 4 * lane;
      device T* vd = (device T*)vc + hk * vhs + (p + j) * vss + 4 * lane;
      for (int e = 0; e < 4; e++) {
        kd[e] = kn[j][e];
        vd[e] = vn[j][e];
      }
    }
  }

  float qs[M][4];
  float o[VS][M][4];
  float max_score[VS][M];
  float sum_exp_score[VS][M];
  for (int m = 0; m < M; m++) {
    for (int e = 0; e < 4; e++) {
      qs[m][e] = static_cast<float>(scale) * qn[m][e];
    }
    for (int s = 0; s < VS; s++) {
      for (int e = 0; e < 4; e++) o[s][m][e] = 0;
      max_score[s][m] = Limits<float>::finite_min;
      sum_exp_score[s][m] = 0;
    }
  }

  // Each key/value row is read once for every slot that sees it; a slot's
  // positions, and the simdgroup each lands on, are those of its own
  // one-token step, so its arithmetic is that step's.
  const device T* keys = kc + hk * khs + 4 * lane;
  const device T* values = vc + hk * vhs + 4 * lane;
  for (int i0 = sg; i0 < N; i0 += BN) {
    T kb[VS][4], vb[VS][4];
    for (int s = 0; s < VS; s++) {
      const int i = i0 + s * NSG;
      if (i < p) {
        for (int e = 0; e < 4; e++) {
          kb[s][e] = keys[i * kss + e];
          vb[s][e] = values[i * vss + e];
        }
      } else {
        for (int j = 0; j < NV; j++) {
          if (i == p + j) {
            for (int e = 0; e < 4; e++) {
              kb[s][e] = kn[j][e];
              vb[s][e] = vn[j][e];
            }
          }
        }
      }
    }
    for (int s = 0; s < VS; s++) {
      const int i = i0 + s * NSG;
      if (i >= N) {
        break;
      }
      float kk[4];
      for (int e = 0; e < 4; e++) kk[e] = kb[s][e];
      for (int m = 0; m < M; m++) {
        if (i > p + m / QH) {
          continue;
        }
        float score = 0;
        for (int e = 0; e < 4; e++) {
          score += qs[m][e] * kk[e];
        }
        score = simd_sum(score);
        float new_max = max(max_score[s][m], score);
        float factor = fast::exp(max_score[s][m] - new_max);
        float exp_score = fast::exp(score - new_max);
        max_score[s][m] = new_max;
        sum_exp_score[s][m] = sum_exp_score[s][m] * factor + exp_score;
        for (int e = 0; e < 4; e++) {
          o[s][m][e] = o[s][m][e] * factor + exp_score * vb[s][e];
        }
      }
    }
  }

  // sdpa_vector's combine across the 32 (virtual) simdgroups, per slot
  threadgroup float outputs[D * BN];
  threadgroup float max_scores[BN];
  threadgroup float sum_exp_scores[BN];
  for (int m = 0; m < M; m++) {
    if (lane == 0) {
      for (int s = 0; s < VS; s++) {
        max_scores[sg + s * NSG] = max_score[s][m];
        sum_exp_scores[sg + s * NSG] = sum_exp_score[s][m];
      }
    }
    for (int s = 0; s < VS; s++) {
      for (int i = 0; i < 4; i++) {
        outputs[(4 * lane + i) * 32 + sg + s * NSG] = o[s][m][i];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float ms = max_scores[lane];
    float new_max = simd_max(ms);
    float factor = fast::exp(ms - new_max);
    float se = simd_sum(sum_exp_scores[lane] * factor);
    for (int s = 0; s < VS; s++) {
      const int dg = sg + s * NSG;     // this simdgroup's dim group: dims 4dg..4dg+3
      float oh[4];
      for (int i = 0; i < 4; i++) {
        oh[i] = simd_sum(outputs[(4 * dg + i) * 32 + lane] * factor);
        oh[i] = se == 0 ? oh[i] : (oh[i] / se);
      }
      if (lane == 0) {
        for (int i = 0; i < 4; i++) {
          out[(m / QH) * NQ * D + (hq0 + m % QH) * D + dg * 4 + i] = static_cast<T>(oh[i]);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
"""

_attn_kernel = mx.fast.metal_kernel(
    name="draft_attn", input_names=["q", "k", "v", "qw", "kw", "kc", "vc", "off", "prm"],
    output_names=["out"], source=ATTN_SOURCE, header=HEADER,
    ensure_row_contiguous=False)

# (VS, QH): virtual simdgroups per real one and query heads per threadgroup, speed
# knobs only.
ATILE = (1, 1)


def attn(q, k, v, a, kc, vc, off, prm, tile=None):
    """NV new rows of attention against the cache buffers kc/vc, written in at off[0]."""
    vs, qh = tile or ATILE
    nq, gqa = a.n_heads, a.n_heads // a.n_kv_heads
    qh = qh or gqa
    assert gqa % qh == 0 and 32 % vs == 0
    nv = q.shape[0]
    return _attn_kernel(
        inputs=[q, k, v, a.q_norm.weight, a.k_norm.weight, kc, vc, off, prm],
        template=[("T", q.dtype), ("GQA", gqa), ("VS", vs), ("QH", qh), ("NV", nv),
                  ("NQ", nq)],
        grid=(1024 // vs * (nq // qh), 1, 1), threadgroup=(1024 // vs, 1, 1),
        output_shapes=[(nv, nq * 128)], output_dtypes=[q.dtype])[0]


# Top-3 of a row as three rounds of mx.argmax would give, in two kernels, with an order
# total enough that neither the slicing nor the merge can change the answer.
SCAN3_HEADER = """
inline bool nq_ahead(float v, int i, float w, int j) {
  return v > w || (v == w && i < j);
}
inline void nq_ins(thread float* bv, thread int* bi, float v, int i) {
  if (nq_ahead(v, i, bv[2], bi[2])) {
    if (nq_ahead(v, i, bv[1], bi[1])) {
      bv[2] = bv[1]; bi[2] = bi[1];
      if (nq_ahead(v, i, bv[0], bi[0])) {
        bv[1] = bv[0]; bi[1] = bi[0]; bv[0] = v; bi[0] = i;
      } else {
        bv[1] = v; bi[1] = i;
      }
    } else {
      bv[2] = v; bi[2] = i;
    }
  }
}
inline void nq_merge(thread float* bv, thread int* bi) {
  for (ushort o = 16; o >= 1; o /= 2) {
    float cv[3];
    int ci[3];
    for (int j = 0; j < 3; j++) {
      cv[j] = simd_shuffle_xor(bv[j], o);
      ci[j] = simd_shuffle_xor(bi[j], o);
    }
    for (int j = 0; j < 3; j++) {
      nq_ins(bv, bi, cv[j], ci[j]);
    }
  }
}
"""

CHUNK = 4096      # elements per threadgroup of the first pass

# PROLOGUE: extra work a caller can put in (the screen's error variance)
TOP3P_SOURCE = """
  constexpr int NT = 256;
  const int tid = int(thread_position_in_threadgroup.x);
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const int g = int(threadgroup_position_in_grid.x);
  const int vr = int(threadgroup_position_in_grid.y);
  const device T* xr = xs + vr * N;
  PROLOGUE
  const int lo = g * CHUNK;
  const int hi = min(N, lo + CHUNK);
  float bv[3] = {-INFINITY, -INFINITY, -INFINITY};
  int bi[3] = {N, N, N};
  int done = lo;
  if (VEC) {                        // 16-byte aligned rows: vector loads
    constexpr int PV = 16 / sizeof(T);
    const int nvec = (hi - lo) / PV;
    const device uint4* xv = (const device uint4*)(xr + lo);
    for (int k = tid; k < nvec; k += NT) {
      uint4 w = xv[k];
      thread const T* e = reinterpret_cast<thread const T*>(&w);
      for (int j = 0; j < PV; j++) {
        nq_ins(bv, bi, static_cast<float>(e[j]), lo + k * PV + j);
      }
    }
    done = lo + nvec * PV;
  }
  for (int i = done + tid; i < hi; i += NT) {
    nq_ins(bv, bi, static_cast<float>(xr[i]), i);
  }
  nq_merge(bv, bi);
  threadgroup float tv[NT / 32 * 3];
  threadgroup int ti[NT / 32 * 3];
  if (lane == 0) {
    for (int j = 0; j < 3; j++) {
      tv[sg * 3 + j] = bv[j];
      ti[sg * 3 + j] = bi[j];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    for (int j = 0; j < 3; j++) {
      bv[j] = lane < NT / 32 ? tv[lane * 3 + j] : -INFINITY;
      bi[j] = lane < NT / 32 ? ti[lane * 3 + j] : N;
    }
    nq_merge(bv, bi);
    if (lane == 0) {
      for (int j = 0; j < 3; j++) {
        pv[(vr * P + g) * 3 + j] = bv[j];
        pi[(vr * P + g) * 3 + j] = bi[j];
      }
    }
  }
"""

TOP3M_SOURCE = """
  const uint lane = thread_index_in_simdgroup;
  const int vr = int(threadgroup_position_in_grid.y);
  float bv[3] = {-INFINITY, -INFINITY, -INFINITY};
  int bi[3] = {N, N, N};
  for (int e = int(lane); e < P * 3; e += 32) {
    nq_ins(bv, bi, pv[vr * P * 3 + e], pi[vr * P * 3 + e]);
  }
  nq_merge(bv, bi);
  if (lane == 0) {
    for (int j = 0; j < 3; j++) {
      y[vr * 3 + j] = uint(bi[j]);
    }
  }
"""


def _top3p_kernel(name, inputs, outputs, prologue=""):
    return mx.fast.metal_kernel(name=name, input_names=inputs, output_names=outputs,
                                source=TOP3P_SOURCE.replace("PROLOGUE", prologue),
                                header=HEADER + SCAN3_HEADER)


_top3p = _top3p_kernel("draft_top3p", ["xs"], ["pv", "pi"])
_top3m = mx.fast.metal_kernel(name="draft_top3m", input_names=["pv", "pi"], output_names=["y"],
                              source=TOP3M_SOURCE, header=SCAN3_HEADER)


def _vec(row):
    """rows of this length start 16-byte aligned in any array they are rows of"""
    return int(row.shape[-1] * row.itemsize % 16 == 0)


def _top3_partials(rows, kernel=_top3p, extra=(), extra_out=()):
    """First pass over (n, N) rows: (n, P, 3) values and ids, plus any extra
    outputs the kernel's prologue writes."""
    n, N = rows.shape
    P = (N + CHUNK - 1) // CHUNK
    return kernel(inputs=[rows, *extra],
                  template=[("T", rows.dtype), ("N", N), ("P", P), ("CHUNK", CHUNK),
                            ("VEC", _vec(rows))] + [(k, v) for k, v, _ in extra_out],
                  grid=(256 * P, n, 1), threadgroup=(256, 1, 1),
                  output_shapes=[(n, P, 3), (n, P, 3)] + [s for _, _, s in extra_out],
                  output_dtypes=[mx.float32, mx.int32] + [mx.float32] * len(extra_out))


def top3(row):
    """ids (uint32) of the three largest entries of a 1-D row, best first and ties to the
    lower index."""
    if row.size < 3:
        return mx.argsort(-row)[:3].astype(mx.uint32)
    if row.size < 64:       # MLX hands tiny inputs over as `constant`
        return top3_masked(row.reshape(-1))
    row = row.reshape(1, -1)
    pv, pi = _top3_partials(row)
    return _top3m(inputs=[pv, pi], template=[("N", row.shape[1]), ("P", pv.shape[1])],
                  grid=(32, 1, 1), threadgroup=(32, 1, 1), output_shapes=[(3,)],
                  output_dtypes=[mx.uint32])[0]


# A 2-bit copy of the lm_head screens its rows so that only the ones the copy cannot
# rule out get the exact 4-bit logit.
SCREEN_Z = 6.0    # statistical margin, not a speed knob; 6 sigma ~ 1e-5 per token

# The approx row's top-3 values come from top3's first pass; its threadgroup
# 0 also reduces the error variance sum_j c[j / 64] * x_j^2 of the same row.
VAR_PROLOGUE = """
  if (g == 0) {
    float acc = 0;
    for (int j = tid; j < K; j += NT) {
      float xj = x[vr * K + j];
      acc += c[j / 64] * xj * xj;
    }
    acc = simd_sum(acc);
    threadgroup float ta[NT / 32];
    if (lane == 0) {
      ta[sg] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
      float v = 0;
      for (int i = 0; i < NT / 32; i++) {
        v += ta[i];
      }
      var[vr] = v;
    }
  }"""

# T is merged once per threadgroup (in every simdgroup it cost ~15 us), then
# one simdgroup per 32 rows: a ballot of which rows pass T, then qmv_fast's
# per-row arithmetic (16 values per lane, 512-wide K blocks, one simd_sum)
# for those rows, four at a time so their loads overlap
RESCORE_SOURCE = """
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const int row0 = int(thread_position_in_grid.y) * 32;
  const int vr = int(thread_position_in_grid.z);
  // T = the approx 3rd largest (merged from the first pass) - Z sigma, once
  // per threadgroup
  threadgroup float tthr[1];
  if (sg == 0) {
    float bv[3] = {-INFINITY, -INFINITY, -INFINITY};
    int bi[3] = {0, 0, 0};
    for (int e = int(lane); e < P * 3; e += 32) {
      nq_ins(bv, bi, pv[vr * P * 3 + e], e);
    }
    nq_merge(bv, bi);
    if (lane == 0) {
      tthr[0] = bv[2] - z[0] * metal::precise::sqrt(var[vr]);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (row0 >= V) {
    return;
  }
  const float thr = tthr[0];
  const device T* ar = a + vr * V;
  device T* yr = y + vr * V;
  const int r = row0 + int(lane);
  bool cand = false;
  if (r < V) {
    T av = ar[r];
    cand = static_cast<float>(av) >= thr;
    if (!cand) {
      yr[r] = av;
    }
  }
  uint m = uint(static_cast<simd_vote::vote_t>(simd_ballot(cand)));
  if (m == 0) {
    return;
  }
  constexpr int KB = K / 512;
  constexpr int KW = K / 2;
  constexpr int KG = K / 64;
  float xt[KB][16];
  float xs[KB];
  const device T* xp = x + vr * K + lane * 16;
  for (int kb = 0; kb < KB; kb++) {
    xs[kb] = 0;
    for (int i = 0; i < 16; i += 4) {
      const T a0 = xp[i], a1 = xp[i + 1], a2 = xp[i + 2], a3 = xp[i + 3];
      xs[kb] += a0 + a1 + a2 + a3;
      xt[kb][i] = a0;
      xt[kb][i + 1] = a1 / 16.0f;
      xt[kb][i + 2] = a2 / 256.0f;
      xt[kb][i + 3] = a3 / 4096.0f;
    }
    xp += 512;
  }
  while (m != 0) {
    int rows[4];
    int nr = 0;
    for (; nr < 4 && m != 0; nr++) {
      rows[nr] = row0 + int(ctz(m));
      m &= m - 1;
    }
    float res[4] = {0, 0, 0, 0};
    for (int kb = 0; kb < KB; kb++) {
      for (int q = 0; q < nr; q++) {
        const device uint8_t* ws = (const device uint8_t*)w + rows[q] * KW + lane * 8 + kb * 256;
        float sv = s[rows[q] * KG + lane / 4 + kb * 8];
        float bv = b[rows[q] * KG + lane / 4 + kb * 8];
        res[q] += nq_dot(ws, xt[kb], sv, bv, xs[kb]);
      }
    }
    for (int q = 0; q < nr; q++) {
      res[q] = simd_sum(res[q]);
      if (lane == 0) {
        yr[rows[q]] = static_cast<T>(res[q]);
      }
    }
  }
"""

_var_top3p = _top3p_kernel("draft_screen_top3p", ["xs", "x", "c"], ["pv", "pi", "var"],
                           VAR_PROLOGUE)
_rescore_kernel = mx.fast.metal_kernel(
    name="draft_screen_rescore", input_names=["a", "x", "pv", "var", "z", "w", "s", "b"],
    output_names=["y"], source=RESCORE_SOURCE, header=HEADER + SCAN3_HEADER)


class _Screen:
    """The 2-bit copy of a 4-bit gs64 head and its error model."""

    def __init__(self, head, rows=16384):
        w, s, b = head["weight"], head["scales"], head["biases"]
        V, G = s.shape
        shifts = 4 * mx.arange(8, dtype=mx.uint32)
        w2, b2, c = [], [], mx.zeros((G,), mx.float32)
        for i in range(0, V, rows):       # chunked: unpacked codes are 32x the bytes
            q = ((w[i:i + rows, :, None] >> shifts) & 0xF).reshape(-1, G, 64)
            lo = (q & 3).astype(mx.float32)
            mean = lo.mean(-1)                                   # (n, G)
            var = ((lo - mean[..., None]) ** 2).mean(-1)
            sf = s[i:i + rows].astype(mx.float32)
            hi = (q >> 2).reshape(q.shape[0], -1, 16)
            w2.append((hi << (2 * mx.arange(16, dtype=mx.uint32))).sum(-1).astype(mx.uint32))
            b2.append((b[i:i + rows].astype(mx.float32) + sf * mean).astype(b.dtype))
            c = c + (sf * sf * var).sum(0)
            mx.eval(w2[-1], b2[-1], c)
        self.w2 = mx.concatenate(w2)
        self.s2 = (s.astype(mx.float32) * 4).astype(s.dtype)     # exact: a power of two
        self.b2 = mx.concatenate(b2)
        self.c = c / V
        self.w, self.s, self.b = w, s, b
        self.z = mx.array([SCREEN_Z], mx.float32)
        mx.eval(self.w2, self.s2, self.b2, self.c)


def screened_head(x, sc):
    """x (NV, K) final-normed -> (NV, V) logits: exact (bit-identical to the
    stock head) on every row the 2-bit copy cannot rule out, the copy's
    approximation elsewhere."""
    nv, K = x.shape
    V = sc.s.shape[0]
    assert K % 512 == 0
    a = mx.quantized_matmul(x, sc.w2, sc.s2, sc.b2, transpose=True, group_size=GROUP_SIZE,
                            bits=2)
    pv, _, var = _top3_partials(a, _var_top3p, extra=(x, sc.c),
                                extra_out=[("K", K, (nv,))])
    return _rescore_kernel(
        inputs=[a, x, pv, var, sc.z, sc.w, sc.s, sc.b],
        template=[("T", x.dtype), ("V", V), ("K", K), ("P", pv.shape[1])],
        grid=(32, (V + 31) // 32, nv), threadgroup=(32, 32, 1),
        output_shapes=[(nv, V)], output_dtypes=[x.dtype])[0]


def top3_masked(row):
    """The reference top3() must match: three masked argmax passes."""
    ids, r = [], row
    neg = mx.array(-float("inf"), dtype=row.dtype)
    ar = mx.arange(r.size)
    for _ in range(3):
        i = mx.argmax(r)
        ids.append(i)
        r = mx.where(ar == i, neg, r)
    return mx.stack(ids).astype(mx.uint32)


# The fused step is built around `attn`, which makes it a pure function of arrays that
# mx.compile can take, with every other piece optional and its stock twin
# numerically identical.

def _grow(c, like, n_kv, head_dim, n=1):
    """KVCache.update_and_fetch's growth, for n more rows."""
    prev = c.offset
    if prev + n > c.keys.shape[2]:
        shape = (1, n_kv, (c.step + n - 1) // c.step * c.step, head_dim)
        new_k = mx.zeros(shape, like.dtype)
        new_v = mx.zeros(shape, like.dtype)
        if prev % c.step != 0:
            c.keys = c.keys[..., :prev, :]
            c.values = c.values[..., :prev, :]
        c.keys = mx.concatenate([c.keys, new_k], axis=2)
        c.values = mx.concatenate([c.values, new_v], axis=2)


def _lin(f, x):
    """A stock quantized matmul the way one-token steps run it: one row at a
    time (stock switches kernels, and rounding, from two rows up)."""
    if x.shape[0] == 1:
        return f(x)
    return mx.concatenate([f(x[i:i + 1]) for i in range(x.shape[0])])


def _layer(l, h, kc, vc, off, prm, on):
    """h (NV, hidden) -> (NV, hidden)."""
    a, mlp = l.self_attn, l.mlp
    nq = a.q_proj["scales"].shape[0]
    nk = a.k_proj["scales"].shape[0]
    if "qkv" in on:
        qkv = nqmv(h, [a.q_proj, a.k_proj, a.v_proj], l.input_layernorm.weight, prm)
        q, k, v = qkv[:, :nq], qkv[:, nq:nq + nk], qkv[:, nq + nk:]
    else:
        x = l.input_layernorm(h)
        q, k, v = _lin(a.q_proj, x), _lin(a.k_proj, x), _lin(a.v_proj, x)
    o = attn(q, k, v, a, kc, vc, off, prm)
    if "o" in on:
        h = nqmv(o, [a.o_proj], resid=h)
    else:
        h = h + _lin(a.o_proj, o)
    if "mlp" in on:
        u = nqmv(h, [mlp.gate_proj, mlp.up_proj], l.post_attention_layernorm.weight,
                 prm, pair=True)
    else:
        x = l.post_attention_layernorm(h)
        u = swiglu(_lin(mlp.gate_proj, x), _lin(mlp.up_proj, x))
    if "down" in on:
        return nqmv(u, [mlp.down_proj], resid=h)
    return h + _lin(mlp.down_proj, u)


def _step_fn(model, on, lo=0, hi=None):
    """Layers [lo, hi) of the step as a pure function of arrays:
    (x, offset (1,), k_lo, v_lo, k_lo+1, ...) -> h (NV, hidden), where x is
    the tokens (1, NV) when lo == 0 and the hidden states otherwise; hi == None
    also runs the final norm and the head and returns logits (1, NV, V)."""
    m = model.model
    prm = model._draftstep.prm
    head = _lm_head(model)
    hi = len(m.layers) if hi is None else hi
    last = hi == len(m.layers)

    def step(x, off, *bufs):
        h = m.embed_tokens(x)[0] if lo == 0 else x
        for i, l in enumerate(m.layers[lo:hi]):
            h = _layer(l, h, bufs[2 * i], bufs[2 * i + 1], off, prm, on)
        if not last:
            return h
        if "screen" in on:
            y = screened_head(m.norm(h), model._draftstep.screen)
        elif "head" in on:
            y = nqmv(h, [head], m.norm.weight, prm)
        elif head is m.embed_tokens:
            y = _lin(head.as_linear, m.norm(h))
        else:
            y = _lin(head, m.norm(h))
        return y[None]
    return step


# Layers in the first compiled segment, which is sent to the GPU (async_eval)
# before the rest is even built: the GPU starts after a few kernels instead of
# after MLX has encoded a full command buffer (40 ops / 40 MB). 0 = one piece.
SPLIT = 0


def usable(model, inputs, cache):
    """True when the fused step's contract holds for this call."""
    if cache is None or inputs.ndim != 2 or inputs.shape[0] != 1 \
            or inputs.shape[1] not in model._draftstep.rows:
        return False
    return all(type(x).update_and_fetch is KVCache.update_and_fetch
               and x.keys is not None for x in cache) and cache[0].offset > 0


def _norms(model):
    m = model.model
    out = [m.norm]
    for l in m.layers:
        a = l.self_attn
        out += [l.input_layernorm, l.post_attention_layernorm, a.q_norm, a.k_norm]
    return out


def _lm_head(model):
    return (model.model.embed_tokens if model.args.tie_word_embeddings
            else model.lm_head)


def _check_model(model):
    """A Qwen3-shaped dense decoder with 4-bit gs64 affine projections, head_dim 128 and
    q/k norms."""
    m = model.model
    if not hasattr(m, "layers") or not hasattr(m.layers[0].self_attn, "q_norm"):
        return False
    qs = [m.embed_tokens, _lm_head(model)]
    for l in m.layers:
        a = l.self_attn
        qs += [a.q_proj, a.k_proj, a.v_proj, a.o_proj,
               l.mlp.gate_proj, l.mlp.up_proj, l.mlp.down_proj]
        r = a.rope
        if not (type(r) is nn.RoPE and not r.traditional and r.scale == 1.0
                and r.dims == 128 and a.q_proj["scales"].shape[0] == a.n_heads * 128
                and a.n_heads % a.n_kv_heads == 0):
            return False
    eps = {n.eps for n in _norms(model)}
    # K % 512 and N % 8: stock's own qmv_fast condition (below it stock runs
    # qmv, which rounds differently); 4096: rms_single_row's row limit
    return len(eps) == 1 and m.norm.weight.shape[0] <= 4096 and all(
        isinstance(q, (nn.QuantizedLinear, nn.QuantizedEmbedding))
        and q.mode == "affine" and q.group_size == GROUP_SIZE and q.bits == BITS
        and "bias" not in q and q["scales"].shape[0] % 8 == 0
        and q["scales"].shape[1] * GROUP_SIZE % 512 == 0 for q in qs)


class _State:
    """Per-model settings, kept off the nn.Module dict so they never show up
    as parameters."""

    def __init__(self, prm):
        self.prm = prm
        self.on = frozenset()
        self.rows = {1}         # token counts per call the fused step takes
        self.steps = {}
        self.screen = None      # the lm_head's 2-bit copy, built when "screen" may run


def forward(model, inputs, cache, on):
    """model(inputs, cache=cache) for 1..NV tokens through the fused step with
    the pieces in `on` (which include "attn")."""
    st = model._draftstep
    key = (on, ATILE, TILE, SPLIT)     # all read when the step is traced
    if key not in st.steps:
        cut = SPLIT if 0 < SPLIT < len(model.model.layers) else 0
        st.steps[key] = [mx.compile(_step_fn(model, on, 0, cut)),
                         mx.compile(_step_fn(model, on, cut))] if cut else \
                        [mx.compile(_step_fn(model, on))]
    a = model.model.layers[0].self_attn
    nv = inputs.shape[1]
    for c in cache:
        _grow(c, c.keys, a.n_kv_heads, c.keys.shape[3], nv)
    off = mx.array([cache[0].offset], dtype=mx.int32)
    bufs = [b for c in cache for b in (c.keys, c.values)]
    parts = st.steps[key]
    if len(parts) == 2:
        cut = SPLIT
        h = parts[0](inputs, off, *bufs[:2 * cut])
        mx.async_eval(h)
        y = parts[1](h, off, *bufs[2 * cut:])
    else:
        y = parts[0](inputs, off, *bufs)
    for c in cache:
        c.offset += nv
    # The KV rows go in as a side effect of y's kernels, so tie the cache to y for a
    # caller that evaluates only the cache state.
    cache[0].keys = mx.depends(cache[0].keys, y)
    return y


def install(model, pieces=None, verbose=False, ctx=512):
    """Route the model's decode steps through the fused step where calibration found them
    faster, keeping the stock forward for anything else."""
    if not _check_model(model):
        return set()
    a = model.model.layers[0].self_attn
    base = float(np.log2(np.float32(a.rope.base)))
    prm = mx.array([model.model.norm.eps, a.scale, base], dtype=mx.float32)
    cls = type(model)
    if not getattr(cls, "_draftstep_cls", False):
        stock = cls.__call__

        def __call__(self, inputs, cache=None, input_embeddings=None):
            on = self._draftstep.on
            if on and input_embeddings is None and usable(self, inputs, cache):
                return forward(self, inputs, cache, on)
            return stock(self, inputs, cache=cache, input_embeddings=input_embeddings)

        model.__class__ = type(cls.__name__, (cls,),
                               {"__call__": __call__, "_draftstep_cls": True})
    model._draftstep = _State(prm)
    if pieces is None or "screen" in pieces:
        model._draftstep.screen = _Screen(_lm_head(model))
    if pieces is None:
        pieces = calibrate(model, ctx=ctx, verbose=verbose)
    pieces = frozenset(pieces)
    assert not pieces or "attn" in pieces, "the fused step is built on attn"
    if verbose:
        print(f"  draftstep: fused {sorted(pieces) or 'nothing'}, qmv tile "
              f"{TILE}, attn tile {ATILE}, split after {SPLIT} layers, "
              f"token counts {sorted(model._draftstep.rows)}", flush=True)
    model._draftstep.on = pieces
    return set(pieces)


def uninstall(model):
    if hasattr(model, "_draftstep"):
        model._draftstep.on = frozenset()


# ---------------------------------------------------------------- timing

def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def _prefill(model, ctx, seed=0):
    from mlx_lm.models.cache import make_prompt_cache
    cache = make_prompt_cache(model)
    ids = mx.random.randint(0, model.args.vocab_size, (1, ctx), key=mx.random.key(seed))
    for i in range(0, ctx, 512):
        mx.eval(model(ids[:, i:i + 512], cache=cache))
    return cache


def _restore(cache, base):
    for c in cache:
        if c.offset > base:
            c.trim(c.offset - base)


def chain(model, cache, base, S, asy=False):
    """A draft chain: S sequential T=1 forwards, top-3 each, the argmax fed back as
    an array and one eval at the end."""
    def f():
        _restore(cache, base)
        d, alts = mx.array([[1234]]), []
        for _ in range(S):
            t3 = top3(model(d, cache=cache)[0, -1])
            if asy:
                mx.async_eval(t3)
            alts.append(t3)
            d = t3[:1][None]
        return alts
    return f


def interleaved(variants, reps=15, warmup=3):
    """{name: fn} timed round-robin, returning {name: median wall ms of build + eval}."""
    names = list(variants)
    for _ in range(warmup):
        for n in names:
            mx.eval(variants[n]())
    t = {n: [] for n in names}
    for _ in range(reps):
        for n in names:
            t0 = time.perf_counter()
            mx.eval(variants[n]())
            t[n].append((time.perf_counter() - t0) * 1e3)
    return {n: _median(v) for n, v in t.items()}


def _with(model, on, fn):
    def g():
        model._draftstep.on = on
        return fn()
    return g


def calibrate(model, ctx=512, verbose=False, margin=0.01, reps=15):
    """What to fuse, timed on the model's own draft chain and kept only where it measured
    faster by more than `margin`."""
    global SPLIT, TILE, ATILE
    cache = _prefill(model, ctx)
    base = cache[0].offset
    f1 = chain(model, cache, base, 1)
    f4 = chain(model, cache, base, 4, asy=True)
    st = model._draftstep

    def arm(on, tile, split, atile):
        def g(f):
            def run():
                global SPLIT, TILE, ATILE
                st.on, TILE, SPLIT, ATILE = on, tile, split, atile
                return f()
            return run
        return g(f1), g(f4)

    def race(name, cur, new):
        """ms per step (mean of the S=1 step and the S=4 chain's steps)."""
        (a1, a4), (b1, b4) = arm(*cur), arm(*new)
        t = interleaved({"a1": a1, "a4": a4, "b1": b1, "b4": b4}, reps=reps)
        c, n = (t["a1"] + t["a4"] / 4) / 2, (t["b1"] + t["b4"] / 4) / 2
        win = n < c * (1.0 - margin)
        if verbose:
            print(f"  {name:12s} {c:7.3f} -> {n:7.3f} ms/step  "
                  f"{'take' if win else 'keep'}", flush=True)
        return new if win else cur

    tile0, at0 = (4, 2), (1, 1)
    rows = {1}
    best = race("attn", (frozenset(), tile0, 0, at0), (frozenset({"attn"}), tile0, 0, at0))
    if best[0]:
        for p in PIECES[1:]:
            best = race(p, best, (best[0] | {p}, tile0, 0, at0))
        if best[0] != frozenset(PIECES):
            best = race("everything", best, (frozenset(PIECES), tile0, 0, at0))
        per = tile0[0] * tile0[1]           # rows per threadgroup, as stock
        for t in [(r, per // r) for r in (2, 1)]:
            best = race(f"tile {t[0]}x{t[1]}", best, (best[0], t, 0, at0))
        best = race("attn by kv", best, (best[0], best[1], 0, (1, 0)))
        n = len(model.model.layers)
        for cand in sorted({d for d in (1, 2, n // 8, n // 4) if 0 < d < n}):
            best = race(f"split@{cand}", best, (best[0], best[1], cand, best[3]))
    kept, TILE, SPLIT, ATILE = best
    st.rows = {1}
    if kept:
        st.on = kept
        for nv in (2, 3):
            def call(fused, nv=nv):
                def f():
                    st.rows = {1, nv} if fused else {1}
                    _restore(cache, base)
                    return top3(model(mx.array([[1234] * nv]), cache=cache)[0, -1])
                return f
            t = interleaved({"stock": call(False), "fused": call(True)}, reps=reps)
            win = t["fused"] < t["stock"] * (1.0 - margin)
            if verbose:
                print(f"  {nv} tokens     {t['stock']:7.3f} -> {t['fused']:7.3f} ms/call  "
                      f"{'take' if win else 'keep'}", flush=True)
            if win:
                rows.add(nv)
    st.rows = rows
    if "screen" not in kept:
        st.screen = None        # 48 MB not worth keeping
    st.on = frozenset()
    st.steps.clear()
    _restore(cache, base)
    del cache
    mx.clear_cache()
    return kept


# ---------------------------------------------------------------- selftest

def _tiny_model(layers=2, vocab=4096, seed=0):
    """Qwen3-0.6B's layer shapes with random weights, few layers, small vocab."""
    from mlx_lm.models import qwen3
    args = qwen3.ModelArgs(
        model_type="qwen3", hidden_size=1024, num_hidden_layers=layers,
        intermediate_size=3072, num_attention_heads=16, rms_norm_eps=1e-6,
        vocab_size=vocab, num_key_value_heads=8, max_position_embeddings=40960,
        rope_theta=1000000.0, head_dim=128, tie_word_embeddings=True)
    mx.random.seed(seed)
    model = qwen3.Model(args)
    # random norms so a wrong weight index cannot hide behind all-ones
    for l in model.model.layers:
        for n in (l.input_layernorm, l.post_attention_layernorm,
                  l.self_attn.q_norm, l.self_attn.k_norm):
            n.weight = (1.0 + 0.3 * mx.random.normal(n.weight.shape))
    model.model.norm.weight = 1.0 + 0.3 * mx.random.normal((1024,))
    model.set_dtype(mx.bfloat16)
    nn.quantize(model, group_size=GROUP_SIZE, bits=BITS)
    mx.eval(model.parameters())
    return model


def _same(a, b):
    a = np.array(a.astype(mx.float32))
    b = np.array(b.astype(mx.float32))
    return int((a != b).sum()), float(np.abs(a - b).max())


def _row(c, i):
    return np.array(mx.concatenate([c.keys[..., i, :], c.values[..., i, :]])
                    .astype(mx.float32))


def _screen_diff(ref, y):
    """For a screened row: (top-3 ids differ, stock's top-3 logits not
    bit-identical, rows exact = the candidate count, roughly)."""
    ref, y = ref.reshape(-1), y.reshape(-1)
    t = top3(ref)
    ids = int(not mx.array_equal(t, top3(y)).item())
    vals = int(not mx.array_equal(ref[t], y[t]).item())
    return ids, vals, int((ref == y).sum().item())


def selftest_pieces(model, ctx=300, combos=None):
    """Each fused piece alone and then all of them against the stock forward on the same
    cache state, with logits and the written KV row bit-identical."""
    install(model, pieces=combos and () or ("attn", "screen"))
    model._draftstep.on = frozenset()
    cache = _prefill(model, ctx, seed=1)
    base = cache[0].offset
    tok = mx.array([[7]])
    ref = model(tok, cache=cache)
    mx.eval(ref)
    ref_kv = [_row(c, base) for c in cache]
    ok = True
    for on in combos or [("attn",)] + [("attn", p) for p in PIECES[1:]] + [EXACT, PIECES]:
        _restore(cache, base)
        for c in cache:
            c.keys[..., base, :] = 0
            c.values[..., base, :] = 0
        mx.eval([c.state for c in cache])
        model._draftstep.on = frozenset(on)
        y = model(tok, cache=cache)
        mx.eval([c.state for c in cache])     # state only, as prefill does: the rows must land
        kv = sum(int((_row(c, base) != r).sum()) for c, r in zip(cache, ref_kv))
        name = "+".join(p for p in PIECES if p in on)
        if "screen" in on:
            ids, vals, exact = _screen_diff(ref, y)
            print(f"  {name:34s} top-3 differs {ids}, their logits differ {vals}, "
                  f"{exact} rows exact, kv row values differing {kv}", flush=True)
            good = ids == 0 and vals == 0
        else:
            n, d = _same(ref, y)
            print(f"  {name:34s} logits differing {n:6d} (max {d:.3g}), "
                  f"kv row values differing {kv}", flush=True)
            good = n == 0
        ok &= good and kv == 0 and all(c.offset == base + 1 for c in cache)
    _restore(cache, base)
    model._draftstep.on = frozenset()
    return ok


def selftest_sequence(model, steps=40):
    """A decode of T=1 chains with T>1 catch-up forwards across a cache growth boundary,
    the exact pieces giving stock's logits at every step."""
    def run(on):
        model._draftstep.on = frozenset(on)
        cache = _prefill(model, 250, seed=2)   # 6 short of the 256-step growth
        out, tok = [], mx.array([[11]])
        rng = np.random.default_rng(0)
        for i in range(steps):
            y = model(tok, cache=cache)[0, -1]
            mx.eval(y)
            out.append(y)
            tok = mx.array([[int(mx.argmax(y).item())]])
            if i % 7 == 6:        # a rejected round: drop 2, re-feed 3 at once
                for c in cache:
                    c.trim(2)
                ids = mx.array([[int(x) for x in rng.integers(0, 4000, 3)]])
                mx.eval(model(ids, cache=cache))
        return out
    install(model, pieces=("attn", "screen"))
    ref, got, scr = run(()), run(EXACT), run(PIECES)
    bad = sum(int(not mx.array_equal(a, b).item()) for a, b in zip(ref, got))
    bad3 = sum(_screen_diff(a, b)[0] for a, b in zip(ref, scr))
    print(f"  {steps}-step decode with trims, T=3 catch-ups and a cache growth: "
          f"{bad} steps differ (exact pieces), {bad3} top-3 differ (with screen)",
          flush=True)
    return bad == 0 and bad3 == 0


def selftest_rows(model, ctx=300):
    """A fused call of 2 and 3 tokens against as many one-token stock steps:
    every row's logits and every new KV row bit-identical (exact pieces)."""
    install(model, pieces=())
    st = model._draftstep
    cache = _prefill(model, ctx, seed=3)
    base = cache[0].offset
    ok = True
    for nv in (2, 3):
        ids = [7, 11, 13][:nv]
        st.on = frozenset()
        _restore(cache, base)
        ref = [model(mx.array([[t]]), cache=cache)[0, -1] for t in ids]
        mx.eval(ref)
        ref_kv = [[_row(c, base + j) for j in range(nv)] for c in cache]
        _restore(cache, base)
        st.on, st.rows = frozenset(EXACT), {1, nv}
        y = model(mx.array([ids]), cache=cache)[0]
        mx.eval(y)
        n = sum(_same(r, y[j])[0] for j, r in enumerate(ref))
        kv = sum(int((_row(c, base + j) != r[j]).sum()) for c, r in zip(cache, ref_kv)
                 for j in range(nv))
        print(f"  {nv} tokens in one fused call vs {nv} stock steps: logits differing {n}, "
              f"kv values differing {kv}", flush=True)
        ok &= n == 0 and kv == 0 and all(c.offset == base + nv for c in cache)
    _restore(cache, base)
    st.on, st.rows = frozenset(), {1}
    return ok


def selftest_top3(n=2000):
    """top3() against three masked argmax passes on random rows full of ties
    (bf16 of a narrow range) and on plain random rows."""
    bad = 0
    for i in range(n):
        key = mx.random.key(i)
        size = [151936, 4096, 1000, 70, 151935][i % 5]
        shape = (3, size) if i % 3 == 0 else (size,)     # a row at an offset, too
        if i % 2:
            row = mx.random.randint(0, 20, shape, key=key).astype(mx.bfloat16)
        else:
            row = (mx.random.normal(shape, key=key) * 4).astype(mx.bfloat16)
        row = row[-1] if len(shape) == 2 else row
        bad += int(not mx.array_equal(top3(row), top3_masked(row)).item())
    print(f"  top3 vs masked argmax on {n} rows (half full of ties): {bad} differ")
    return bad == 0


def selftest():
    global ATILE, TILE
    assert selftest_top3()
    model = _tiny_model()
    ok = selftest_pieces(model)
    ok &= selftest_sequence(model)
    ok &= selftest_rows(model)
    keep = ATILE, TILE
    for ATILE in [(vs, qh) for vs in (1, 2, 4, 8) for qh in (1, 0)]:
        print(f"  attn tile VS={ATILE[0]} QH={ATILE[1] or 'GQA'}:")
        ok &= selftest_pieces(model, ctx=300, combos=[("attn",)])
        model._draftstep.steps.clear()
    ATILE = keep[0]
    for TILE in [(r, n) for r in (1, 2, 4, 8) for n in (1, 2, 4)]:
        print(f"  qmv tile {TILE[0]} rows x {TILE[1]} simdgroups:")
        ok &= selftest_pieces(model, ctx=300, combos=[EXACT])
        model._draftstep.steps.clear()
    TILE = keep[1]
    assert ok, "fused step differs from stock"
    print("selftest: bit-identical to stock")


def agreement(model, prompt_ids, n=1000):
    """A real draft run, stock greedy then the fused step teacher-forced over the same
    tokens."""
    from mlx_lm.models.cache import make_prompt_cache

    def run(on, forced=None):
        model._draftstep.on = frozenset(on)
        cache = make_prompt_cache(model)
        ids = list(prompt_ids)
        for i in range(0, len(ids) - 1, 512):          # prefill, stock (T > 1)
            mx.eval(model(mx.array([ids[i:min(i + 512, len(ids) - 1)]]), cache=cache))
        tok, t3s, rows = ids[-1], [], []
        for i in range(n):
            row = model(mx.array([[tok]]), cache=cache)[0, -1]
            t3 = top3(row)
            mx.eval(t3)
            t3s.append(t3.tolist())
            rows.append(row)
            tok = t3s[-1][0] if forced is None else forced[i]
        return t3s, rows

    keep = model._draftstep.on
    ref, ref_rows = run(())
    got, got_rows = run(keep or PIECES, forced=[t[0] for t in ref])
    model._draftstep.on = keep
    agree = sum(a[0] == b[0] for a, b in zip(ref, got))
    agree3 = sum(a == b for a, b in zip(ref, got))
    differ = sum(int(not mx.array_equal(a, b).item()) for a, b in zip(ref_rows, got_rows))
    return n, agree, agree3, differ


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--model", default=None)
    p.add_argument("--text", default=None,
                   help="with --model: a text file whose tokens start a real "
                        "--tokens-long draft run, checked position by position")
    p.add_argument("--tokens", type=int, default=1000)
    args = p.parse_args()
    if args.selftest:
        selftest()
    if args.model:
        from mlx_lm import load
        model, tok = load(args.model)
        print("real weights:")
        ok = selftest_pieces(model, ctx=300) and selftest_pieces(model, ctx=2000) \
            and selftest_rows(model)
        print("calibrating:")
        print("installed:", sorted(install(model, verbose=True)))
        if args.text:
            ids = tok.encode(open(args.text).read())[:2048]
            n, agree, agree3, differ = agreement(model, ids, args.tokens)
            print(f"real run: {n} draft tokens after a {len(ids)}-token prompt, "
                  f"argmax agrees at {agree}/{n}, top-3 at {agree3}/{n}, "
                  f"logits differ at {differ} positions")
            ok &= agree >= 0.999 * n
        assert ok
