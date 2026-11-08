#!/usr/bin/env python3
"""Flash-Next's decode-row glue: the small ops between a qwen4_exp layer's matvecs, each
stretch in one Metal kernel."""
# MOE_GLUE (Metal only, on by default, MOE_GLUE=0 turns it off) wires them in: install(), called by
# stream_fast.install.
# A T=1 row runs ~60 small kernels between the matvecs, ~1.1 us each, and each kernel here takes
# over one stretch with MLX's dtype and op order (tests/test_layer_glue.py).
# The sigmoid and silu tables are MLX's own kernels': a hand-written exp differs from mlx.metallib's
# in the last bit for a few inputs.
# The kernels' licence is MLX's (MIT, Copyright (c) 2023 Apple Inc.): THIRD_PARTY_NOTICES.md.
import os
import sys

import mlx.core as mx

import route_k

ON = os.environ.get("MOE_GLUE", "1") != "0"
# MOE_GLUE_PREFILL (Metal, on by default, =0 off): the same glue over a layer-major prefill pass's rows.
PREFILL = os.environ.get("MOE_GLUE_PREFILL", "1") != "0"

# ----------------------------------------------------------------------------- tables
_TABLES = {}


def _table(key, fn, f32):
    t = _TABLES.get((key, f32))
    if t is None:
        import numpy as np
        x = mx.array(np.arange(65536, dtype=np.uint16)).view(mx.bfloat16)
        t = _TABLES[(key, f32)] = fn(x.astype(mx.float32) if f32 else x)
        mx.eval(t)
    return t


def sig_table(f32=False):
    """[65536]: mx.sigmoid of each bf16 bit pattern, in bf16, or (f32) of its float32 value."""
    return _table("sigmoid", mx.sigmoid, f32)


def silu_table(f32=False):
    """[65536]: nn.silu (compiled) of each bf16 bit pattern, as sig_table."""
    import mlx.nn as nn
    return _table("silu", nn.silu, f32)


_CONST = {}


def _const(key, make):
    a = _CONST.get(key)
    if a is None:
        a = _CONST[key] = make()       # computed with its first use, then kept: no eval inside a pass
    return a


# --------------------------------------------------------------------------- kernels
# rms_single_row<T, 4> over D elements with 32 lanes a row and the weight w (w_stride 0: a scalar),
# this lane's 4 outputs read from x at off, with MLX's two simd_sums.
_HEAD = r"""
template <typename T, int D, typename X, typename W>
METAL_FUNC void rms_4(X x, int off, W w, int w_stride, float eps, uint lane, thread T* out) {
  float acc = 0;
  float tx[4];
  for (int i = 0; i < 4; i++) {
    tx[i] = x[off + i];
    acc += tx[i] * tx[i];
  }
  acc = simd_sum(acc);
  acc = simd_sum((lane == 0) ? acc : 0.0f);
  float inv = metal::precise::rsqrt(acc / D + eps);
  for (int i = 0; i < 4; i++) {
    out[i] = w[w_stride * (lane * 4 + i)] * static_cast<T>(tx[i] * inv);
  }
}
"""

# rms_single_row<T, 4> over each stream of D, then * weight: one threadgroup a stream, NT threads
# (MLX's threadgroup for D).
_PRE = r"""
  uint lid = thread_position_in_threadgroup.x;
  uint gid = threadgroup_position_in_grid.x;
  uint lane = thread_index_in_simdgroup;
  uint sg = simdgroup_index_in_threadgroup;
  threadgroup float local_inv_mean[1];
  threadgroup float local_sums[32];
  float acc = 0;
  float tx[4];
  size_t base = size_t(gid) * D + lid * 4;
%(load)s
  if (lid * 4 + 4 <= D) {
    for (int i = 0; i < 4; i++) {
      tx[i] = %(src)s;
      acc += tx[i] * tx[i];
    }
  } else {
    for (int i = 0; i < 4; i++) {
      tx[i] = (lid * 4 + i < D) ? (float)%(src)s : 0;
      acc += tx[i] * tx[i];
    }
  }
  acc = simd_sum(acc);
  if (sg == 0) { local_sums[lane] = 0; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) { local_sums[sg] = acc; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    acc = simd_sum(local_sums[lane]);
    if (lane == 0) { local_inv_mean[0] = metal::precise::rsqrt(acc / D + eps[0]); }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const T one = T(1);
  for (int i = 0; i < 4; i++) {
    if (lid * 4 + i < D) {
      T v = one * static_cast<T>(tx[i] * local_inv_mean[0]);
      normed[base + i] = v * w[base + i];
    }
  }
"""
# The decoder layer's residual combine in front: h = hyper + (x[..., None, :] * inject[..., None])
# flattened, two MLX ops, two roundings; h written out, then normed.
_LOAD_COMB = r"""
  T hv[4];
  for (int i = 0; i < 4; i++) {
    if (lid * 4 + i < D) {
      T p = static_cast<T>(xr[lid * 4 + i] * inj[gid]);
      hv[i] = static_cast<T>(hyper[base + i] + p);
      h[base + i] = hv[i];
    }
  }
"""



def _pre(comb=False):
    """The pre kernel: its source with the residual combine and the combined h as the input, or
    not."""
    return _PRE % {"load": _LOAD_COMB if comb else "", "src": "hv[i]" if comb else "x[base + i]"}

# The down rows' silu into s, and with %(inj)s the inject rows' 2 * sigmoid(inject / hc).
_MID_HEAD = r"""
  uint t = thread_position_in_grid.x;
  if (t < ND) {
    T v = y[t] / T(HC);
    s[t] = silu[as_type<ushort>(v)];
  }%(inj)s
"""
_MID = _MID_HEAD % {"inj": ""}
_MID_INJ = _MID_HEAD % {"inj": r""" else if (t < ND + HC) {
    T v = y[t] / T(HC);
    inj[t - ND] = T(2) * sig[as_type<ushort>(v)];
  }"""}

# col_reduce_small's order over the hc streams: t3 + (t2 + (t1 + (t0 + 0))), then times 1/hc
# (hc is a power of two: exact) as MLX's mean takes it.
_MIX = r"""
  uint j = thread_position_in_grid.x;
  if (j < D) {
    T tot = (sig[as_type<ushort>(u[j])] * nrm[j]) + T(0);
    for (int r = 1; r < HC; r++) {
      T p = (sig[as_type<ushort>(u[r * D + j])] * nrm[r * D + j]) + T(0);
      tot = p + tot;
    }
    mixed[j] = tot * T(1.0f / float(HC));
  }
"""

# The delta net's conv step and norms, one simdgroup a block of 128 channels (a q head, a k head or
# a v head): depthwise_conv_1d (float acc += float(in) * w, cast to T), nn.silu's values, the next
# conv state; q and k blocks then rms_single_row (no weight) and the scale as a T scalar.
_GDN_PRE = r"""
  const uint lane = thread_index_in_simdgroup;
  const uint g = threadgroup_position_in_grid.x;
  T sv[4];
  for (int i = 0; i < 4; i++) {
    int c = g * 128 + lane * 4 + i;
    float acc = 0.0;
    acc += static_cast<float>(cs[c]) * cw[c * 4 + 0];
    acc += static_cast<float>(cs[C + c]) * cw[c * 4 + 1];
    acc += static_cast<float>(cs[2 * C + c]) * cw[c * 4 + 2];
    acc += static_cast<float>(qkv[c]) * cw[c * 4 + 3];
    T cv = static_cast<T>(acc);
    sv[i] = silu[as_type<ushort>(cv)];
    cs_out[c] = cs[C + c];
    cs_out[C + c] = cs[2 * C + c];
    cs_out[2 * C + c] = qkv[c];
  }
  if (g < 2 * NK) {
    T o[4];
    rms_4<T, 128>(sv, 0, one, 0, 1e-6f, lane, o);
    T sc = (g < NK) ? scales[0] : scales[1];
    device T* dst = (g < NK) ? q : k;
    for (int i = 0; i < 4; i++) { dst[(g % NK) * 128 + lane * 4 + i] = sc * o[i]; }
  } else {
    for (int i = 0; i < 4; i++) { v[(g - 2 * NK) * 128 + lane * 4 + i] = sv[i]; }
  }
"""

# RMSNormGated (qwen4_exp): rms_norm(y, w), then T(sigmoid(float(z)) * float(.)), the float sigmoid
# looked up by z's bits (mx.sigmoid's values on float32).
_GDN_POST = r"""
  const uint lane = thread_index_in_simdgroup;
  const uint hv = threadgroup_position_in_grid.x;
  T o[4];
  rms_4<T, DV>(y + hv * DV, lane * 4, nw, 1, eps[0], lane, o);
  for (int i = 0; i < 4; i++) {
    int e = lane * 4 + i;
    float gz = sig32[as_type<ushort>(z[hv * DV + e])];
    out[hv * DV + e] = static_cast<T>(gz * static_cast<float>(o[i]));
  }
"""

_SILU_MUL = r"""
  uint i = thread_position_in_grid.x;
  if (i < N) { out[i] = silu[as_type<ushort>(g[i])] * u[i]; }
"""

# SparseMoeBlock's end, (y * scores[..., None]).sum(axis=-2).astype(T) + sigmoid(sg) * sh, as f32
# products and col_reduce_small over 8 rows of threads (row r on thread r % 8, then thread j's total
# added in j order).
_MOE_END_HEAD = "#pragma clang fp contract(off)\n"
_MOE_END = r"""
  uint i = thread_position_in_grid.x;
  if (i < D) {
    float part[8];
    for (int j = 0; j < 8; j++) { part[j] = 0.0f; }
    for (int r = 0; r < K; r++) {
      float p = static_cast<float>(y[r * D + i]) * sc[r];
      part[r % 8] = p + part[r % 8];
    }
    float tot = part[0];
    for (int j = 1; j < 8; j++) { tot = part[j] + tot; }
    T m = sig[as_type<ushort>(sg[0])] * sh[i];
    out[i] = static_cast<T>(tot) + m;
  }
"""

# Attention's q and k heads: rms_norm(x, w) per head of D in MLX's rms_single_row threads, then the
# partial rope on the first R dims as _rope_partial has it (cos / sin cast to T, rot = (-x2, x1)),
# one threadgroup a head.
_QK = r"""
  uint lid = thread_position_in_threadgroup.x;
  uint hd = threadgroup_position_in_grid.x;
  uint lane = thread_index_in_simdgroup;
  uint sg = simdgroup_index_in_threadgroup;
  threadgroup float local_inv_mean[1];
  threadgroup float local_sums[32];
  threadgroup T rin[R];
  const bool isq = hd < NQ;
  const size_t xb = isq ? size_t(hd) * 2 * D : size_t(hd - NQ) * D;
  float acc = 0;
  float tx[4];
  for (int i = 0; i < 4; i++) {
    tx[i] = isq ? qg[xb + lid * 4 + i] : kr[xb + lid * 4 + i];
    acc += tx[i] * tx[i];
  }
  acc = simd_sum(acc);
  if (sg == 0) { local_sums[lane] = 0; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) { local_sums[sg] = acc; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    acc = simd_sum(local_sums[lane]);
    if (lane == 0) { local_inv_mean[0] = metal::precise::rsqrt(acc / D + eps[0]); }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  T o[4];
  for (int i = 0; i < 4; i++) {
    uint j = lid * 4 + i;
    T wj = isq ? qw[j] : kw[j];
    o[i] = wj * static_cast<T>(tx[i] * local_inv_mean[0]);
    if (j < R) { rin[j] = o[i]; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = 0; i < 4; i++) {
    uint j = lid * 4 + i;
    T r = o[i];
    if (j < R) {
      T rot = (j < R / 2) ? T(-rin[j + R / 2]) : rin[j - R / 2];
      T a = o[i] * static_cast<T>(cs[j]);
      T b = rot * static_cast<T>(sn[j]);
      r = a + b;
    }
    if (isq) { q[size_t(hd) * D + j] = r; } else { k[size_t(hd - NQ) * D + j] = r; }
  }
"""

# The attention output times sigmoid(gate), the gate being q_proj's second half of each head.
_AGATE = r"""
  uint i = thread_position_in_grid.x;
  if (i < N) {
    uint h = i / D;
    uint j = i % D;
    out[i] = o[i] * sig[as_type<ushort>(qg[h * 2 * D + D + j])];
  }
"""

def _sub(src, *pairs):
    """src with each (old, new) replaced, every old present."""
    for a, b in pairs:
        assert a in src, a
        src = src.replace(a, b)
    return src


# ------------------------------------------------------------ prefill rows (MOE_GLUE_PREFILL)
# Each decode kernel above over R rows: the same arithmetic per row, R read from an input's shape so
# one compiled kernel serves every width; MLX's own ops pick their kernels by row length only.
_ROWW = ("w[base + i]", "w[(gid % HC) * D + lid * 4 + i]")      # stream gid's weight
_PRE_ROWS = _sub(_pre(), _ROWW)
_PRE_COMB_ROWS = _sub(_pre(True), _ROWW, ("xr[lid * 4 + i]", "xr[(gid / HC) * D + lid * 4 + i]"))

# The down / inject rows of fuse_proj's stack (row stride NY): silu(down / hc), 2 * sigmoid(inject / hc).
_MID_ROWS = r"""
  uint t = thread_position_in_grid.x;
  uint r = t / (ND + HC);
  uint c = t - r * (ND + HC);
  if (r < uint(y_shape[1])) {
    T v = y[r * NY + c] / T(HC);
    if (c < ND) {
      s[r * ND + c] = silu[as_type<ushort>(v)];
    } else {
      inj[r * HC + c - ND] = T(2) * sig[as_type<ushort>(v)];
    }
  }
"""

# _MIX a row of D at a time.
_MIX_ROWS = r"""
  uint t = thread_position_in_grid.x;
  uint r = t / D;
  uint j = t - r * D;
  if (r < uint(u_shape[1])) {
    size_t b = size_t(r) * HC * D + j;
    T tot = (sig[as_type<ushort>(u[b])] * nrm[b]) + T(0);
    for (int q = 1; q < HC; q++) {
      T p = (sig[as_type<ushort>(u[b + q * D])] * nrm[b + q * D]) + T(0);
      tot = p + tot;
    }
    mixed[t] = tot * T(1.0f / float(HC));
  }
"""

# The decoder layer's residual combine, hyper + (x[..., None, :] * inj[..., None]) flattened.
_COMB_ROWS = r"""
  uint t = thread_position_in_grid.x;
  uint rs = t / D;
  if (rs < uint(hyper_shape[1]) * HC) {
    T p = static_cast<T>(x[(rs / HC) * D + (t - rs * D)] * inj[rs]);
    out[t] = static_cast<T>(hyper[t] + p);
  }
"""

# _GDN_PRE on row r of the conv input [state rows 0..2, qkv rows], the next state from row R's group.
_GDN_ROWS_HEAD = _HEAD + r"""
template <typename T>
METAL_FUNC T conv_in(const device T* cs, const device T* qkv, uint j, int c, int C) {
  return (j < 3) ? cs[j * C + c] : qkv[(j - 3) * C + c];
}
"""
_GDN_PRE_ROWS = r"""
  const uint lane = thread_index_in_simdgroup;
  const uint tg = threadgroup_position_in_grid.x;
  const uint r = tg / (C / 128);
  const uint g = tg - r * (C / 128);
  const uint R = qkv_shape[1];
  const uint NV = C / 128 - 2 * NK;
  T sv[4];
  for (int i = 0; i < 4; i++) {
    int c = g * 128 + lane * 4 + i;
    float acc = 0.0;
    acc += static_cast<float>(conv_in(cs, qkv, r + 0, c, C)) * cw[c * 4 + 0];
    acc += static_cast<float>(conv_in(cs, qkv, r + 1, c, C)) * cw[c * 4 + 1];
    acc += static_cast<float>(conv_in(cs, qkv, r + 2, c, C)) * cw[c * 4 + 2];
    acc += static_cast<float>(conv_in(cs, qkv, r + 3, c, C)) * cw[c * 4 + 3];
    T cv = static_cast<T>(acc);
    sv[i] = silu[as_type<ushort>(cv)];
    if (r == 0) {
      for (uint j = 0; j < 3; j++) { cs_out[j * C + c] = conv_in(cs, qkv, R + j, c, C); }
    }
  }
  if (g < 2 * NK) {
    T o[4];
    rms_4<T, 128>(sv, 0, one, 0, 1e-6f, lane, o);
    T sc = (g < NK) ? scales[0] : scales[1];
    device T* dst = (g < NK) ? q : k;
    for (int i = 0; i < 4; i++) { dst[(r * NK + g % NK) * 128 + lane * 4 + i] = sc * o[i]; }
  } else {
    for (int i = 0; i < 4; i++) { v[(r * NV + g - 2 * NK) * 128 + lane * 4 + i] = sv[i]; }
  }
"""

_MOE_END_ROWS = r"""
  uint t = thread_position_in_grid.x;
  uint r = t / D;
  uint i = t - r * D;
  if (r < uint(sh_shape[1])) {
    float part[8];
    for (int j = 0; j < 8; j++) { part[j] = 0.0f; }
    for (int q = 0; q < K; q++) {
      float p = static_cast<float>(y[(r * K + q) * D + i]) * sc[r * K + q];
      part[q % 8] = p + part[q % 8];
    }
    float tot = part[0];
    for (int j = 1; j < 8; j++) { tot = part[j] + tot; }
    T m = sig[as_type<ushort>(sg[r])] * sh[t];
    out[t] = static_cast<T>(tot) + m;
  }
"""


def _route_rows_src():
    """fused_router's logit-routed router (route_logits) with row `row` of [R, NE] logits a threadgroup."""
    import fused_router as fr
    src = _sub(fr._SRC_LOGITS + fr._TAIL_ROUTE, ("logits[", "logits[row * NE + "),
               ("inds[lane]", "inds[row * KT + lane]"), ("ids[lane]", "ids[row * KT + lane]"),
               ("scores[b2 + i]", "scores[row * KT + b2 + i]"))
    return "  uint row = threadgroup_position_in_grid.y;\n" + src


_K = {}

# Each kernel: its inputs, its outputs, its source and the header it needs.
_SPECS = {
    "pre": (["x", "w", "eps"], ["normed"], _pre(), ""),
    "pre_comb": (["hyper", "xr", "inj", "w", "eps"], ["h", "normed"], _pre(True), ""),
    "mid": (["y", "silu"], ["s"], _MID, ""),
    "mid_inj": (["y", "sig", "silu"], ["s", "inj"], _MID_INJ, ""),
    "mix": (["u", "nrm", "sig"], ["mixed"], _MIX, ""),
    "gdn_pre": (["qkv", "cs", "cw", "scales", "one", "silu"], ["q", "k", "v", "cs_out"],
                _GDN_PRE, _HEAD),
    "gdn_post": (["y", "z", "nw", "eps", "sig32"], ["out"], _GDN_POST, _HEAD),
    "silu_mul": (["g", "u", "silu"], ["out"], _SILU_MUL, ""),
    "moe_end": (["y", "sc", "sh", "sg", "sig"], ["out"], _MOE_END, _MOE_END_HEAD),
    "qk": (["qg", "kr", "qw", "kw", "eps", "cs", "sn"], ["q", "k"], _QK, ""),
    "agate": (["o", "qg", "sig"], ["out"], _AGATE, ""),
    "pre_rows": (["x", "w", "eps"], ["normed"], _PRE_ROWS, ""),
    "pre_comb_rows": (["hyper", "xr", "inj", "w", "eps"], ["h", "normed"], _PRE_COMB_ROWS, ""),
    "mid_rows": (["y", "sig", "silu"], ["s", "inj"], _MID_ROWS, ""),
    "mix_rows": (["u", "nrm", "sig"], ["mixed"], _MIX_ROWS, ""),
    "comb_rows": (["hyper", "x", "inj"], ["out"], _COMB_ROWS, ""),
    "gdn_pre_rows": (["qkv", "cs", "cw", "scales", "one", "silu"], ["q", "k", "v", "cs_out"],
                     _GDN_PRE_ROWS, _GDN_ROWS_HEAD),
    "moe_end_rows": (["y", "sc", "sh", "sg", "sig"], ["out"], _MOE_END_ROWS, _MOE_END_HEAD),
    "route_rows": (["logits", "mask", "lam", "tau"], ["inds", "scores", "ids"], _route_rows_src(), ""),
}


def _kern(name):
    k = _K.get(name)
    if k is None:
        ins, outs, src, head = _SPECS[name[0]]
        k = _K[name] = mx.fast.metal_kernel(name="lg_" + "_".join(map(str, name)), input_names=ins,
                                            output_names=outs, source=src, header=head)
    return k


def _eps(v):
    return _const(("eps", v), lambda: mx.array([v], dtype=mx.float32))


# ------------------------------------------------------------------- hyper-connections
def _rms_threads(d):
    need = -(-d // 4)                       # MLX's rms threadgroup: 4 reads a thread, whole simdgroups
    return 32 * -(-need // 32)


def hc_pre(s, hyper, comb=None):
    """GatedResidual's hc_norm of one bf16 row: `normed`, or with comb = (hyper, x, inject) of the
    layer's previous half (hyper arg then None) the residual combine first: (h, normed)."""
    hc, d = s.hc, s.d
    T = (hyper if comb is None else comb[0]).dtype
    nt = _rms_threads(d)
    kw = dict(template=[("T", T), ("D", d)], grid=(nt * hc, 1, 1), threadgroup=(nt, 1, 1))
    if comb is None:
        return _kern(("pre",))(inputs=[hyper, s.hc_norm.weight, _eps(s.hc_norm.eps)],
                               output_shapes=[hyper.shape], output_dtypes=[T], **kw)[0]
    hp, xr, inj = comb
    return tuple(_kern(("pre_comb",))(inputs=[hp, xr, inj, s.hc_norm.weight, _eps(s.hc_norm.eps)],
                                      output_shapes=[hp.shape, hp.shape], output_dtypes=[T, T], **kw))


def hc_rest(s, normed):
    """GatedResidual after its norm, down (with the inject rows, fuse_proj's stack), mid, up and
    mix, returning mixed or (mixed, inject)."""
    hc, d, T = s.hc, s.d, normed.dtype
    up = s.input_mix_weight_up
    nd = up.weight.shape[-1] * 32 // up.bits
    if s.block_inject_weight is None:
        y = s.input_mix_weight_down(normed)
        outs = _kern(("mid",))(inputs=[y, silu_table()], template=[("T", T), ("ND", nd), ("HC", hc)],
                               grid=(-(-nd // 32) * 32, 1, 1), threadgroup=(32, 1, 1),
                               output_shapes=[(1, 1, nd)], output_dtypes=[T])
    else:
        y = s.fused_hc(normed)             # down's rows, inject's, zero rows
        outs = _kern(("mid_inj",))(inputs=[y, sig_table(), silu_table()],
                                   template=[("T", T), ("ND", nd), ("HC", hc)],
                                   grid=(-(-(nd + hc) // 32) * 32, 1, 1), threadgroup=(32, 1, 1),
                                   output_shapes=[(1, 1, nd), (1, 1, hc)], output_dtypes=[T, T])
    u = up(outs[0])
    mixed = _kern(("mix",))(inputs=[u, normed, sig_table()], template=[("T", T), ("D", d), ("HC", hc)],
                               grid=(-(-d // 256) * 256, 1, 1), threadgroup=(256, 1, 1),
                               output_shapes=[(1, 1, d)], output_dtypes=[T])[0]
    return mixed if s.block_inject_weight is None else (mixed, outs[1])


def hc_call(s, hyper):
    """GatedResidual.__call__ for one bf16 row [1, 1, hc * d]."""
    r = hc_rest(s, hc_pre(s, hyper))
    return r if s.block_inject_weight is None else (r[0], hyper, r[1])


def hc_eligible(s):
    """A GatedResidual these kernels compute exactly: bf16, hc a power of two, d in one threadgroup,
    4 reads a thread, the inject stacked by fuse_proj."""
    w = s.hc_norm.weight
    return (w.dtype == mx.bfloat16 and s.hc & (s.hc - 1) == 0 and s.d <= 4096 and s.d % 4 == 0
            and hasattr(s.input_mix_weight_up, "bits") and s.hc_norm.group_size == s.d
            and (s.block_inject_weight is None or "fused_hc" in s))


# --------------------------------------------------------------------------- delta net
def gdn_dims(s):
    return s.n_v, s.n_k, s.dk, s.dv


def gdn_eligible(s):
    """qwen4_exp's GatedDeltaNet at the shapes these kernels copy MLX on."""
    nv, nk, dk, dv = gdn_dims(s)
    w = s.conv1d.weight
    return (dk == 128 and dv == 128 and nv % nk == 0 and s.conv_kernel_size == 4
            and w.dtype == mx.bfloat16 and w.shape == (s.conv_dim, 4, 1)
            and s.conv_dim == 2 * nk * dk + nv * dv and getattr(s.norm, "activation", "") == "sigmoid"
            and s.norm.weight.dtype == mx.bfloat16)


def gdn_step(s, x, cache):
    """GatedDeltaNet.__call__ for one bf16 row after a prefill (conv and delta states exist, mask
    None): the projections, gdn_pre, MLX's gated_delta_update on its outputs, gdn_post and
    out_proj, updating `cache` as the stock lines do."""
    nv, nk, dk, dv = gdn_dims(s)
    T = x.dtype
    qkv = s.in_proj_qkv(x)
    z = s.in_proj_z(x)
    b = s.in_proj_b(x)
    a = s.in_proj_a(x)
    inv = dk ** -0.5
    # The scalars as MLX's ops take a python float: array(value, dtype of x).
    scales = _const(("sc", inv, T), lambda: mx.concatenate(
        [mx.array([inv ** 2], dtype=T), mx.array([inv], dtype=T)]))
    one = _const(("one", T), lambda: mx.array([1], dtype=T))
    C = s.conv_dim
    q, k, v, conv = _kern(("gdn_pre",))(
        inputs=[qkv, cache[0], s.conv1d.weight, scales, one, silu_table()],
        template=[("T", T), ("C", C), ("NK", nk)],
        grid=(32 * (C // 128), 1, 1), threadgroup=(32, 1, 1),
        output_shapes=[(1, 1, nk, dk), (1, 1, nk, dk), (1, 1, nv, dv), cache[0].shape],
        output_dtypes=[T, T, T, cache[0].dtype])
    cache[0] = conv
    gdu = sys.modules[type(s).__module__].gated_delta_update
    y, state = gdu(q, k, v, a, b, s.A_log, s.dt_bias, cache[1], None, use_kernel=not s.training)
    cache[1] = state
    cache.advance(1)
    out = _kern(("gdn_post",))(
        inputs=[y, z, s.norm.weight, _eps(s.norm.eps), sig_table(True)],
        template=[("T", T), ("DV", dv)], grid=(32 * nv, 1, 1), threadgroup=(32, 1, 1),
        output_shapes=[(1, 1, nv * dv)], output_dtypes=[T])[0]
    return s.out_proj(out)


# ----------------------------------------------------------------------------- attention
def attn_eligible(s):
    """qwen4_exp's Attention at the shapes the qk kernel copies MLX on."""
    D = s.head_dim
    return (D % 128 == 0 and D <= 1024 and s.q_norm.group_size is None and s.k_norm.group_size is None
            and s.q_norm.eps == s.k_norm.eps and s.q_norm.weight.dtype == mx.bfloat16
            and s.k_norm.weight.dtype == mx.bfloat16 and s.q_norm.weight.shape == (D,))


def attn_step(s, x, rope, mask, cache, idx_cache):
    """Attention.__call__ for one bf16 row: the projections, the indexer, the cache and SDPA as
    stock, q / k's norms and rope in one kernel and the output gate in another."""
    mod = sys.modules[type(s).__module__]
    B, S, _ = x.shape
    offset = cache.offset if cache is not None else 0
    sparse = s.indexer(x, rope, idx_cache, offset, mod._left_padding(cache))
    qg = s.q_proj(x)
    kr = s.k_proj(x)
    v = s.v_proj(x).reshape(B, S, s.n_kv_heads, -1).transpose(0, 2, 1, 3)
    cos, sin = rope(mod._positions(offset, S))
    T, D, nq, nk, R = x.dtype, s.head_dim, s.n_heads, s.n_kv_heads, cos.shape[-1]
    if R % 2 or R > D:
        raise ValueError(f"layer_glue: rope dim {R} for head dim {D}")
    q, k = _kern(("qk",))(inputs=[qg, kr, s.q_norm.weight, s.k_norm.weight, _eps(s.q_norm.eps), cos, sin],
                          template=[("T", T), ("D", D), ("R", R), ("NQ", nq)],
                          grid=(D // 4 * (nq + nk), 1, 1), threadgroup=(D // 4, 1, 1),
                          output_shapes=[(B, nq, S, D), (B, nk, S, D)], output_dtypes=[T, T])
    if cache is not None:
        k, v = cache.update_and_fetch(k, v)
    if sparse is not None:          # Attention.__call__'s lines
        if mask is None:
            mask = sparse
        elif isinstance(mask, str):
            kv_len = k.shape[2]
            rinds = mx.arange(kv_len)
            linds = mx.arange(kv_len - S, kv_len)[:, None]
            mask = (linds >= rinds) & sparse
        elif mask.dtype == mx.bool_:
            mask = mask & sparse
        else:
            neg = mx.finfo(mask.dtype).min
            mask = mask + mx.where(sparse, mx.array(0, mask.dtype), neg)
    out = mod.scaled_dot_product_attention(q, k, v, cache=cache, scale=s.scale, mask=mask)
    out = out.transpose(0, 2, 1, 3).reshape(B, S, -1)
    n = out.size
    out = _kern(("agate",))(inputs=[out, qg, sig_table()], template=[("T", T), ("D", D), ("N", n)],
                            grid=(-(-n // 256) * 256, 1, 1), threadgroup=(256, 1, 1),
                            output_shapes=[out.shape], output_dtypes=[T])[0]
    return s.o_proj(out)


# ----------------------------------------------------------------------------- the MoE end
def silu_mul(g, u):
    """nn.silu(g) * u, bf16 rows."""
    n = g.size
    return _kern(("silu_mul",))(inputs=[g, u, silu_table()], template=[("T", g.dtype), ("N", n)],
                                grid=(-(-n // 256) * 256, 1, 1), threadgroup=(256, 1, 1),
                                output_shapes=[g.shape], output_dtypes=[g.dtype])[0]


def moe_end(y, scores, sh, sg):
    """(y * scores[..., None]).sum(axis=-2).astype(sh.dtype) + mx.sigmoid(sg) * sh for one row:
    y [1, 1, K, D] the experts' unweighted outputs, scores [1, 1, K] f32, sh [1, 1, D] the shared
    expert's output, sg [1, 1, 1] its gate's logit."""
    K, D = y.shape[-2], y.shape[-1]
    T = sh.dtype
    return _kern(("moe_end",))(inputs=[y, scores, sh, sg, sig_table()],
                               template=[("T", T), ("D", D), ("K", K)],
                               grid=(-(-D // 256) * 256, 1, 1), threadgroup=(256, 1, 1),
                               output_shapes=[(1, 1, D)], output_dtypes=[T])[0]


def moe_tail(m, x, inds, scores):
    """stream_fast.CacheBiasedMoE's native decode row after its route kernel: the experts and the shared
    expert as there (the shared one still between the handshake and the wait, MOE_SHARED_EARLY), the
    weighting, sum and shared gate in moe_end."""
    b = m.block
    sg = []

    def shared(x_):
        sg.append(b.shared_expert_gate(x_))
        return b.shared_expert(x_)
    early = m._mark_shared_early(x)
    if early:
        b.switch_mlp.__dict__["_shared_early"] = shared
    y = b.switch_mlp(x, inds)
    sh = b.switch_mlp.__dict__.pop("_shared_out", None) if early else None
    if sh is None:
        sh = shared(x)
    return moe_end(y, scores, sh, sg[-1])


# ------------------------------------------------------------- prefill rows: the calls
def rows_ok(x):
    """A layer-major chunk these kernels take: bf16 [1, R, n]."""
    return x.ndim == 3 and x.shape[0] == 1 and x.dtype == mx.bfloat16


def hc_rows(s, hyper, comb=None):
    """GatedResidual.__call__ over a chunk's rows [1, R, hc * d]: (mixed, hyper, inject); with
    comb = (hyper, x, inject) of the layer's previous half (hyper arg None) the residual combine
    first, its h returned as `hyper`."""
    hc, d = s.hc, s.d
    src = hyper if comb is None else comb[0]
    T, R = src.dtype, src.shape[1]
    nt = _rms_threads(d)
    kw = dict(template=[("T", T), ("D", d), ("HC", hc)], grid=(nt * hc * R, 1, 1),
              threadgroup=(nt, 1, 1))
    w, eps = s.hc_norm.weight, _eps(s.hc_norm.eps)
    if comb is None:
        normed = _kern(("pre_rows",))(inputs=[hyper, w, eps], output_shapes=[hyper.shape],
                                      output_dtypes=[T], **kw)[0]
    else:
        hyper, normed = _kern(("pre_comb_rows",))(inputs=[*comb, w, eps], output_shapes=[src.shape] * 2,
                                                  output_dtypes=[T, T], **kw)
    y = s.fused_hc(normed)                 # down's rows, inject's, zero rows
    up = s.input_mix_weight_up
    nd = up.weight.shape[-1] * 32 // up.bits
    m, inj = _kern(("mid_rows",))(inputs=[y, sig_table(), silu_table()],
                                  template=[("T", T), ("ND", nd), ("HC", hc), ("NY", y.shape[-1])],
                                  grid=(R * (nd + hc), 1, 1), threadgroup=(256, 1, 1),
                                  output_shapes=[(1, R, nd), (1, R, hc)], output_dtypes=[T, T])
    mixed = _kern(("mix_rows",))(inputs=[up(m), normed, sig_table()],
                                 template=[("T", T), ("D", d), ("HC", hc)], grid=(R * d, 1, 1),
                                 threadgroup=(256, 1, 1), output_shapes=[(1, R, d)], output_dtypes=[T])[0]
    return mixed, hyper, inj


def comb_rows(hyper, inj, x):
    """hyper + (x[..., None, :] * inj[..., None]).reshape(hyper.shape) over a chunk's rows."""
    return _kern(("comb_rows",))(inputs=[hyper, x, inj],
                                 template=[("T", hyper.dtype), ("D", x.shape[-1]), ("HC", inj.shape[-1])],
                                 grid=(hyper.size, 1, 1), threadgroup=(256, 1, 1),
                                 output_shapes=[hyper.shape], output_dtypes=[hyper.dtype])[0]


def gdn_rows_ok(s, x, cache):
    return (cache is not None and getattr(cache, "lengths", None) is None
            and (cache[0] is None or cache[0].dtype == x.dtype))


def gdn_rows(s, x, cache):
    """GatedDeltaNet.__call__ over a chunk's rows (mask None, a cache): the projections, gdn_pre_rows
    (conv over the state and the rows, silu, q / k norms, the next conv state), MLX's
    gated_delta_update, gdn_post and out_proj, updating `cache` as the stock lines do."""
    nv, nk, dk, dv = gdn_dims(s)
    T, (B, S, _) = x.dtype, x.shape
    C = s.conv_dim
    qkv = s.in_proj_qkv(x)
    z = s.in_proj_z(x)
    b = s.in_proj_b(x)
    a = s.in_proj_a(x)
    inv = dk ** -0.5
    scales = _const(("sc", inv, T), lambda: mx.concatenate(
        [mx.array([inv ** 2], dtype=T), mx.array([inv], dtype=T)]))
    one = _const(("one", T), lambda: mx.array([1], dtype=T))
    cs = cache[0] if cache[0] is not None else mx.zeros((B, s.conv_kernel_size - 1, C), dtype=T)
    q, k, v, conv = _kern(("gdn_pre_rows",))(
        inputs=[qkv, cs, s.conv1d.weight, scales, one, silu_table()],
        template=[("T", T), ("C", C), ("NK", nk)],
        grid=(32 * (C // 128) * S, 1, 1), threadgroup=(32, 1, 1),
        output_shapes=[(B, S, nk, dk), (B, S, nk, dk), (B, S, nv, dv), cs.shape],
        output_dtypes=[T, T, T, T])
    cache[0] = conv
    gdu = sys.modules[type(s).__module__].gated_delta_update
    y, state = gdu(q, k, v, a, b, s.A_log, s.dt_bias, cache[1], None, use_kernel=not s.training)
    cache[1] = state
    cache.advance(S)
    out = _kern(("gdn_post",))(
        inputs=[y, z, s.norm.weight, _eps(s.norm.eps), sig_table(True)],
        template=[("T", T), ("DV", dv)], grid=(32 * nv * S, 1, 1), threadgroup=(32, 1, 1),
        output_shapes=[(B, S, nv * dv)], output_dtypes=[T])[0]
    return s.out_proj(out)


def route_rows(logits, mask, lam, tau, k):
    """fused_router.route_logits over a chunk's rows, logits fp32 [1, R, E]: (inds uint32 [1, R, k]
    ascending by value as argpartition gives them, scores fp32 [1, R, k])."""
    import fused_router as fr
    _, R, e = logits.shape
    need = -(-e // 4)                       # MLX's softmax threadgroup for e, as route_logits
    nt = 32 * -(-need // 32)
    tmpl = [("NE", e), ("KT", k), ("NPL", e // 32), ("BUDGET", 1 if tau > 0.0 else 0)]
    inds, scores, _ = _kern(("route_rows",))(
        inputs=[logits, mask, fr._scalar(lam), fr._scalar(tau)], template=tmpl, grid=(nt, R, 1),
        threadgroup=(nt, 1, 1), output_shapes=[(1, R, k)] * 3,
        output_dtypes=[mx.uint32, mx.float32, mx.int32])
    return inds, scores


def route_rows_ok(logits, k):
    e = logits.shape[-1]
    return logits.dtype == mx.float32 and e % 32 == 0 and e <= 4096 and k <= 32


def moe_end_rows(y, scores, x, b):
    """SparseMoeBlock's end over a chunk's rows: (y * scores[..., None]).sum(axis=-2).astype(x.dtype)
    + mx.sigmoid(b.shared_expert_gate(x)) * b.shared_expert(x), the shared expert's silu * up in
    silu_mul."""
    se = b.shared_expert
    sh = se.down_proj(silu_mul(se.gate_proj(x), se.up_proj(x)))
    sg = b.shared_expert_gate(x)
    K, D = y.shape[-2], y.shape[-1]
    return _kern(("moe_end_rows",))(inputs=[y, scores, sh, sg, sig_table()],
                                    template=[("T", sh.dtype), ("D", D), ("K", K)],
                                    grid=(sh.size, 1, 1), threadgroup=(256, 1, 1),
                                    output_shapes=[sh.shape], output_dtypes=[sh.dtype])[0]


# --------------------------------------------------------------------------- the engine's hook
_STOCK = {}


def _row(x):
    return x.ndim == 3 and x.shape[:2] == (1, 1) and x.dtype == mx.bfloat16


def _layer(s, h, rope, mask, conv_mask, cache, idx_cache, ids, prev_ctx):
    """DecoderLayer.__call__ of one bf16 row: the HC kernels, the attention half's combine inside the MLP
    half's norm kernel; the mixer, the MoE block and the PLE as the model has them."""
    if s.ple is not None:
        h = h + s.ple(h, ids, prev_ctx, cache)
    ah, mh = s.attn_hyper_connection, s.mlp_hyper_connection
    x, inj = hc_rest(ah, hc_pre(ah, h))
    if s.layer_type == "linear_attention":
        x = s.linear_attn(x, conv_mask, cache)
    else:
        x = s.self_attn(x, rope, mask, cache, idx_cache)
    h, n = hc_pre(mh, None, comb=(h, x, inj))
    x, inj = hc_rest(mh, n)
    x = s.mlp(x)
    return h + (x[..., None, :] * inj[..., None]).reshape(*x.shape[:-1], -1)


def _wrap(q4):
    """Route the classes' calls by an instance mark (`_lg`), once a process, with unmarked
    instances (an MTP head's layer, another model) and every other row (prefill, MTP verify,
    batches) running stock."""
    if _STOCK:
        return
    dl, gdn, mlp, gr, moe = q4.DecoderLayer, q4.GatedDeltaNet, q4.MLP, q4.GatedResidual, q4.SparseMoeBlock
    _STOCK.update(dl=dl.__call__, gdn=gdn.__call__, mlp=mlp.__call__, gr=gr.__call__, moe=moe.__call__,
                  attn=q4.Attention.__call__)

    def layer_call(self, *a, **k):
        if k or not (self.__dict__.get("_lg") and _row(a[0]) and a[3] is None):
            return _STOCK["dl"](self, *a, **k)
        return _layer(self, *a)

    def gdn_call(self, x, mask, cache):
        if (self.__dict__.get("_lg") and _row(x) and mask is None and cache is not None
                and cache[0] is not None and cache[1] is not None and getattr(cache, "lengths", None) is None):
            return gdn_step(self, x, cache)
        return _STOCK["gdn"](self, x, mask, cache)

    def attn_call(self, x, rope, mask, cache, idx_cache):
        if self.__dict__.get("_lg") and _row(x):
            return attn_step(self, x, rope, mask, cache, idx_cache)
        return _STOCK["attn"](self, x, rope, mask, cache, idx_cache)

    def mlp_call(self, x):
        if self.__dict__.get("_lg") and _row(x):
            return self.down_proj(silu_mul(self.gate_proj(x), self.up_proj(x)))
        return _STOCK["mlp"](self, x)

    def gr_call(self, hyper):
        if self.__dict__.get("_lg") and _row(hyper):
            return hc_call(self, hyper)
        return _STOCK["gr"](self, hyper)

    def moe_call(self, x):
        # SparseMoeBlock's own lines (cache bias 0: no CacheBiasedMoE in front), the end in moe_end.
        if self.__dict__.get("_lgp") and "_lm" in self.switch_mlp.__dict__ and rows_ok(x):
            logits = self.gate(x.astype(mx.float32))
            idx = mx.argpartition(-logits, self.top_k - 1, axis=-1)[..., : self.top_k]
            w = route_k.weights(logits, idx, self.__dict__.get("_k2") or self.top_k)
            return moe_end_rows(self.switch_mlp(x, idx), w, x, self)
        if not (self.__dict__.get("_lg") and _row(x)):
            return _STOCK["moe"](self, x)
        logits = self.gate(x.astype(mx.float32))
        idx = mx.argpartition(-logits, self.top_k - 1, axis=-1)[..., : self.top_k]
        w = route_k.weights(logits, idx, self.__dict__.get("_k2") or self.top_k)
        return moe_end(self.switch_mlp(x, idx), w, self.shared_expert(x), self.shared_expert_gate(x))

    dl.__call__, gdn.__call__, mlp.__call__ = layer_call, gdn_call, mlp_call
    gr.__call__, moe.__call__, q4.Attention.__call__ = gr_call, moe_call, attn_call


def install(model):
    """MOE_GLUE (on unless =0) and MOE_GLUE_PREFILL (=1), Metal: mark a qwen4_exp model's eligible
    layers for these kernels (after fuse_proj's HC stacking) and return how many, 0 when both are
    off, on another backend or without qwen4_exp layers."""
    q4 = sys.modules.get("mlx_lm.models.qwen4_exp")
    if not (ON or PREFILL) or not mx.metal.is_available() or q4 is None:
        return 0
    mods = list(model.modules())
    layers = [m for m in mods if isinstance(m, q4.DecoderLayer)
              and hc_eligible(m.attn_hyper_connection) and hc_eligible(m.mlp_hyper_connection)]
    if not layers:
        return 0
    _wrap(q4)
    sig_table(), sig_table(True), silu_table()      # built now, not by an eval inside a decode pass
    for m in layers if PREFILL else ():
        # Read by stream_fast's layer-major pass (_lm_parts, CacheBiasedMoE) and moe_call.
        m.__dict__["_lgp"] = True
        if m.layer_type == "linear_attention" and gdn_eligible(m.linear_attn):
            m.linear_attn.__dict__["_lgp"] = True
        if isinstance(m.mlp, q4.SparseMoeBlock):
            m.mlp.__dict__["_lgp"] = True
    for m in layers if ON else ():
        m.__dict__["_lg"] = True
        if m.layer_type == "linear_attention" and gdn_eligible(m.linear_attn):
            m.linear_attn.__dict__["_lg"] = True
        elif m.layer_type != "linear_attention" and attn_eligible(m.self_attn):
            m.self_attn.__dict__["_lg"] = True
        if isinstance(m.mlp, q4.SparseMoeBlock) and m.mlp.top_k <= 64:
            m.mlp.__dict__["_lg"] = True
            m.mlp.shared_expert.__dict__["_lg"] = True
    for m in mods if ON else ():
        if isinstance(m, q4.Qwen4ExpModel) and hc_eligible(m.hyper_connection_mixer):
            m.hyper_connection_mixer.__dict__["_lg"] = True
    return len(layers)
