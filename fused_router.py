#!/usr/bin/env python3
"""The router glue of `CacheBiasedMoE.__call__` and the residency decision of `StreamingSwitchGLU._call_defer` in ONE `mx.fast.metal_kernel` launch per MoE layer, copying MLX's dtypes, op order and ascending-by-value argpartition tie order exactly so the result is bit-identical, and off unless MOE_FUSED_ROUTER=1 or a "+fused" arm."""
import os

import mlx.core as mx

# The pass widths the fused path serves: decode (T=1) and an MTP verify pass
# (T=2..gamma+1). Prefill is wide and goes the ordinary route, as it does for
# the logical LRU.
MAX_T = 5

# A Metal kernel: MOE_FUSED_ROUTER=1 is ignored on CUDA and CPU.
_ON = os.environ.get("MOE_FUSED_ROUTER", "0") == "1" and mx.metal.is_available()
_KERNELS = {}
_SCALARS = {}

_SRC = r"""
  uint tid  = thread_position_in_threadgroup.x;
  uint lane = tid & 31u;
  uint sg   = tid >> 5;

  threadgroup float tg[NE];      // this row's gates, widened to float
  threadgroup float tr[NE];      // the array being ranked, picks knocked out
  threadgroup float sv[NSGT];    // per-simdgroup reduction slots
  threadgroup uint  sx[NSGT];
  threadgroup uint  tsel[KT];    // biased top-k, ascending by value
  threadgroup uint  thon[KT];    // honest top-k (budget arms only)
  threadgroup uint  flg[2];      // [0] every slot resident, [1] fall back

  if (tid == 0) { flg[0] = 1u; flg[1] = 0u; }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const float lamv = lam[0];
  const float tauv = tau[0];

  for (uint r = 0; r < NROWS; ++r) {
    // ---- the row, its max and its min ------------------------------------
    float lmax = -FLT_MAX;
    float lmin =  FLT_MAX;
    for (uint e = tid; e < NE; e += NT) {
      float v = (float)gates[r * NE + e];
      tg[e] = v;
      lmax = fmax(lmax, v);
      lmin = fmin(lmin, v);
    }
    lmax = simd_max(lmax);
    lmin = simd_min(lmin);
    if (lane == 0) { sv[sg] = lmax; sx[sg] = as_type<uint>(lmin); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float gmax = sv[0];
    float gmin = as_type<float>(sx[0]);
    for (uint s = 1; s < NSGT; ++s) {
      gmax = fmax(gmax, sv[s]);
      gmin = fmin(gmin, as_type<float>(sx[s]));
    }

    // ---- the bias, in the gates' own dtype and MLX's op order -------------
    //   spread = (T)max - (T)min ;  ls = (T)lam * spread ;
    //   rank   = float(gates) + float(ls) * mask          (mask is 0 or 1)
    T sp = (T)gmax - (T)gmin;
    T ls = (T)lamv * sp;
    float lsf = (float)ls;
    for (uint e = tid; e < NE; e += NT) { tr[e] = tg[e] + lsf * mask[e]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- top-k: pass 0 the biased ranking, pass 1 the honest one ----------
    for (uint p = 0; p < (BUDGET ? 2u : 1u); ++p) {
      threadgroup uint *dst = (p == 0u) ? &tsel[0] : &thon[0];
      if (p == 1u) {
        for (uint e = tid; e < NE; e += NT) { tr[e] = tg[e]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      for (uint j = 0; j < KT; ++j) {
        // Largest value, largest index on a tie -- MLX's ascending sort by
        // (value, index) read from the back. Ascending `e` and `>=` give the
        // largest index within a thread.
        float bv = -FLT_MAX;
        uint  bi = 0u;
        for (uint e = tid; e < NE; e += NT) {
          float v = tr[e];
          if (v >= bv) { bv = v; bi = e; }
        }
        float mv = simd_max(bv);
        uint  mi = simd_max((bv == mv) ? bi : 0u);
        if (lane == 0) { sv[sg] = mv; sx[sg] = mi; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float wv = sv[0];
        uint  wi = sx[0];
        for (uint s = 1; s < NSGT; ++s) {
          if (sv[s] > wv || (sv[s] == wv && sx[s] > wi)) { wv = sv[s]; wi = sx[s]; }
        }
        if (tid == 0) { dst[KT - 1u - j] = wi; tr[wi] = -FLT_MAX; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
    }

    // ---- bounded-loss budget: honest mass against biased mass -------------
    // MLX's bf16 sum over the last axis is a sequential accumulation in the
    // element order (measured), so this adds in the same order.
    if (BUDGET && tid == 0) {
      T mh = (T)tg[thon[0]];
      T mb = (T)tg[tsel[0]];
      for (uint j = 1; j < KT; ++j) { mh = mh + (T)tg[thon[j]]; }
      for (uint j = 1; j < KT; ++j) { mb = mb + (T)tg[tsel[j]]; }
      flg[1] = ((mh - mb) > ((T)tauv * mh)) ? 1u : 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- the residency decision _call_defer used to take -----------------
    if (tid < KT) {
      uint sel = (BUDGET && flg[1] == 1u) ? thon[tid] : tsel[tid];
      inds[r * KT + tid] = sel;
      int s = slot_of[sel];
      rhs[r * KT + tid] = (s > 0) ? s : 0;
      if (s < 0) { flg[0] = 0u; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0) { ok[0] = flg[0]; }
"""


def on():
    return _ON


def set_fused(flag):
    """The "+fused" arm switch, flipped per arm on one model like every other
    knob."""
    global _ON
    _ON = bool(flag)
    return _ON


def _scalar(v):
    a = _SCALARS.get(v)
    if a is None:
        a = mx.array([float(v)], dtype=mx.float32)
        mx.eval(a)
        _SCALARS[v] = a
    return a


def _kernel(key):
    kern = _KERNELS.get(key)
    if kern is None:
        kern = mx.fast.metal_kernel(
            name="fused_router",
            input_names=["gates", "mask", "slot_of", "lam", "tau"],
            output_names=["inds", "ok", "rhs"],
            source=_SRC,
        )
        _KERNELS[key] = kern
    return kern


def route(gates, mask, slot_of, lam, tau, k):
    """One launch: bias, top-k, budget, slot lookup, residency flag.

    gates   [1, T, E]  the block's own softmax, bf16 or float32
    mask    [E] f32    1.0 for the experts the host LRU held at routing time
    slot_of [E] i32    expert -> arena slot, -1 when absent
    returns (inds uint32 [1, T, k], ok uint32 [1], rhs int32 [1, T, k])

    `inds` is uint32 because that is what `mx.argpartition` returns and what
    every reader downstream (take_along_axis, _decode_indices, _claim) has
    been fed since the start.
    """
    b, t, e = gates.shape
    assert b == 1, "fused router: batch 1 only"
    nt = 32 if e < 128 else 128
    budget = 1 if tau > 0.0 else 0
    kern = _kernel((gates.dtype, e, k, t, budget, nt))
    return kern(
        inputs=[gates, mask, slot_of, _scalar(lam), _scalar(tau)],
        template=[("T", gates.dtype), ("NE", e), ("KT", k), ("NROWS", t),
                  ("NT", nt), ("NSGT", nt // 32), ("BUDGET", budget)],
        grid=(nt, 1, 1),
        threadgroup=(nt, 1, 1),
        output_shapes=[(1, t, k), (1,), (1, t, k)],
        output_dtypes=[mx.uint32, mx.uint32, mx.int32],
    )


# ---------------------------------------------------------------------------
# The whole logit-routed (qwen4_exp) router of one decode row in one launch
# (softmax over the raw fp32 logits, bias, top-k and budget above, then the
# softmax over the k selected logits), on by default on native passes and on
# the sync path only with MOE_ROUTE_KERNEL=1 (=0 turns it off), with both
# softmaxes copying MLX 0.32.2's softmax_single_row thread for thread so the
# probabilities are bit-equal and the selection runs on the very numbers
# argpartition saw; with `prev` it takes MOE_STAGING's guess instead, written
# best first after the layer's own ids (the array _call_native sends).

_ROUTE_ENV = os.environ.get("MOE_ROUTE_KERNEL", "")
_ROUTE_ON = _ROUTE_ENV != "0" and mx.metal.is_available()     # Metal kernels; CUDA native passes route as sync does
_ROUTE_SYNC = _ROUTE_ENV == "1"

_SRC_LOGITS = r"""
  uint tid  = thread_position_in_threadgroup.x;
  uint lane = tid & 31u;
  uint sg   = tid >> 5;

  threadgroup float tg[NE];      // softmax(logits), MLX's numbers
  threadgroup float lmax[32];    // softmax_single_row's local_max
  threadgroup float lnorm[32];   // and local_normalizer

  // ---- softmax_single_row<float, float, 4> over the NE logits -------------
  float ld[4];
  int base = int(tid) * 4;
  if (base + 4 <= NE) {
    for (int i = 0; i < 4; i++) { ld[i] = float(logits[base + i]); }
  } else {
    for (int i = 0; i < 4; i++) {
      ld[i] = ((base + i) < NE) ? float(logits[base + i]) : Limits<float>::min;
    }
  }
  if (sg == 0) { lmax[lane] = Limits<float>::min; lnorm[lane] = 0; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float maxval = Limits<float>::finite_min;
  for (int i = 0; i < 4; i++) { maxval = (maxval < ld[i]) ? ld[i] : maxval; }
  maxval = simd_max(maxval);
  if (lane == 0) { lmax[sg] = maxval; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    maxval = simd_max(lmax[lane]);
    if (lane == 0) { lmax[0] = maxval; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  maxval = lmax[0];
  float normalizer = 0;
  for (int i = 0; i < 4; i++) {
    float exp_x = metal::fast::exp(ld[i] - maxval);
    ld[i] = exp_x;
    normalizer += exp_x;
  }
  normalizer = simd_sum(normalizer);
  if (lane == 0) { lnorm[sg] = normalizer; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    normalizer = simd_sum(lnorm[lane]);
    if (lane == 0) { lnorm[0] = normalizer; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  normalizer = 1 / lnorm[0];
  for (int i = 0; i < 4; i++) {
    if ((base + i) < NE) { tg[base + i] = ld[i] * normalizer; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // ---- the rest runs on simdgroup 0 alone, with no barrier: lane l keeps
  // experts l, l + 32, ... in registers (NPL of them) -----------------------
  if (sg != 0) { return; }
  float gl[NPL];
  float lmx = -FLT_MAX;
  float lmn =  FLT_MAX;
  for (int i = 0; i < NPL; i++) {
    gl[i] = tg[lane + 32 * i];
    lmx = fmax(lmx, gl[i]);
    lmn = fmin(lmn, gl[i]);
  }
  // route()'s bias: rank = gates + (lam * (max - min)) * mask
  float ls = lam[0] * (simd_max(lmx) - simd_min(lmn));
  uint pick[2][KT];
  for (uint p = 0; p < (BUDGET ? 2u : 1u); ++p) {
    float r[NPL];
    for (int i = 0; i < NPL; i++) {
      r[i] = (p == 0u) ? gl[i] + ls * mask[lane + 32 * i] : gl[i];
    }
    // KT rounds of argmax, the largest index on a tie: MLX's ascending sort
    // by (value, index) read from the back (route()'s rule); the j-th pick
    // goes to KT - 1 - j, ascending by value as argpartition returns them.
    for (uint j = 0; j < KT; ++j) {
      float bv = -FLT_MAX;
      uint  bi = 0u;
      for (int i = 0; i < NPL; i++) {
        if (r[i] >= bv) { bv = r[i]; bi = lane + 32 * i; }
      }
      float mv = simd_max(bv);
      uint  mi = simd_max((bv == mv) ? bi : 0u);
      pick[p][KT - 1u - j] = mi;
      if ((mi & 31u) == lane) { r[mi >> 5] = -FLT_MAX; }
    }
  }
  bool back = false;
  if (BUDGET) {
    // MLX's sum of the k gathered gates: sequential, in their order.
    float mh = tg[pick[1][0]];
    float mb = tg[pick[0][0]];
    for (uint j = 1; j < KT; ++j) { mh = mh + tg[pick[1][j]]; }
    for (uint j = 1; j < KT; ++j) { mb = mb + tg[pick[0][j]]; }
    back = (mh - mb) > (tau[0] * mh);
  }
  uint tsel[KT];
  for (uint j = 0; j < KT; ++j) { tsel[j] = back ? pick[1][j] : pick[0][j]; }
"""
# The guess: [the layer's own ids, this guess best first], as _call_native sends them.
_TAIL_GUESS = r"""
  if (lane < KT) {
    ids[lane] = prev[lane];
    ids[KT + lane] = int(tsel[KT - 1u - lane]);
  }
"""
# The layer's own routing: the selection, then softmax_single_row<float> over
# the KT selected logits (one simdgroup, as MLX dispatches KT <= 128).
_TAIL_ROUTE = r"""
  if (lane < KT) {
    inds[lane] = tsel[lane];
    ids[lane] = int(tsel[lane]);
  }
  float sl[4];
  int b2 = int(lane) * 4;
  for (int i = 0; i < 4; i++) {
    sl[i] = ((b2 + i) < KT) ? float(logits[tsel[b2 + i]]) : Limits<float>::min;
  }
  float mv = Limits<float>::finite_min;
  for (int i = 0; i < 4; i++) { mv = (mv < sl[i]) ? sl[i] : mv; }
  mv = simd_max(mv);
  mv = simd_max((lane == 0) ? mv : Limits<float>::min);
  float nz = 0;
  for (int i = 0; i < 4; i++) {
    float exp_x = metal::fast::exp(sl[i] - mv);
    sl[i] = exp_x;
    nz += exp_x;
  }
  nz = simd_sum(nz);
  nz = simd_sum((lane == 0) ? nz : 0.0f);
  float inv = 1 / nz;
  for (int i = 0; i < 4; i++) {
    if ((b2 + i) < KT) { scores[b2 + i] = sl[i] * inv; }
  }
"""
_ZEROS = {}


def route_on():
    return _ROUTE_ON


def route_sync():
    return _ROUTE_SYNC


def set_route(flag):
    global _ROUTE_ON
    _ROUTE_ON = bool(flag)
    return _ROUTE_ON


def zeros_mask(e):
    a = _ZEROS.get(e)
    if a is None:
        a = _ZEROS[e] = mx.zeros((e,), dtype=mx.float32)
        mx.eval(a)
    return a


def _route_kernel(guess):
    key = ("logits", guess)
    kern = _KERNELS.get(key)
    if kern is None:
        ins = ["logits", "mask", "lam", "tau"] + (["prev"] if guess else [])
        outs = ["ids"] if guess else ["inds", "scores", "ids"]
        kern = _KERNELS[key] = mx.fast.metal_kernel(
            name="route_logits_guess" if guess else "route_logits",
            input_names=ins, output_names=outs,
            source=_SRC_LOGITS + (_TAIL_GUESS if guess else _TAIL_ROUTE))
    return kern


def route_logits(logits, mask, lam, tau, k, prev=None):
    """One decode row's routing from the raw fp32 logits [1, 1, E].

    prev None: (inds uint32 [1,1,k] ascending by value as argpartition gives
    them, scores fp32 [1,1,k] = softmax of the selected logits, ids int32).
    prev given (int32 [1,1,k], the layer's own ids): the staging guess,
    int32 [1,1,2k] = [prev, the selection best first]."""
    b, t, e = logits.shape
    assert b == 1 and t == 1 and logits.dtype == mx.float32, logits.shape
    need = -(-e // 4)                       # MLX's softmax threadgroup for e:
    nt = 32 * -(-need // 32)                # 4 reads a thread, whole simdgroups
    guess = prev is not None
    budget = 1 if tau > 0.0 else 0
    ins = [logits, mask, _scalar(lam), _scalar(tau)] + ([prev] if guess else [])
    assert e % 32 == 0 and k <= 32, (e, k)
    tmpl = [("NE", e), ("KT", k), ("NPL", e // 32), ("BUDGET", budget)]
    if guess:
        return _route_kernel(True)(
            inputs=ins, template=tmpl, grid=(nt, 1, 1), threadgroup=(nt, 1, 1),
            output_shapes=[(1, 1, 2 * k)], output_dtypes=[mx.int32])[0]
    return _route_kernel(False)(
        inputs=ins, template=tmpl, grid=(nt, 1, 1), threadgroup=(nt, 1, 1),
        output_shapes=[(1, 1, k), (1, 1, k), (1, 1, k)],
        output_dtypes=[mx.uint32, mx.float32, mx.int32])


def glue_logits(logits, mask, lam, tau, k):
    """CacheBiasedMoE's ops for a logit-routed row, in its order: the
    reference route_logits must match bit for bit."""
    gates = mx.softmax(logits, axis=-1, precise=True)
    if lam > 0.0:
        spread = mx.max(gates, axis=-1, keepdims=True) - mx.min(gates, axis=-1, keepdims=True)
        rank = gates + (lam * spread) * mask
    else:
        rank = gates
    inds = mx.argpartition(rank, kth=-k, axis=-1)[..., -k:]
    if tau > 0.0 and lam > 0.0:
        honest = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        m_h = mx.take_along_axis(gates, honest, axis=-1).sum(axis=-1, keepdims=True)
        m_b = mx.take_along_axis(gates, inds, axis=-1).sum(axis=-1, keepdims=True)
        inds = mx.where(m_h - m_b > tau * m_h, honest, inds)
    scores = mx.softmax(mx.take_along_axis(logits, inds, axis=-1), axis=-1, precise=True)
    return inds, scores


def glue(gates, mask, slot_of, lam, tau, k):
    """The MLX ops the kernel replaces, in the engine's order. The reference
    for the unit test and the other arm of the timing ladder."""
    spread = (mx.max(gates, axis=-1, keepdims=True)
              - mx.min(gates, axis=-1, keepdims=True))
    rank = gates + (lam * spread) * mask
    inds = mx.argpartition(rank, kth=-k, axis=-1)[..., -k:]
    if tau > 0.0:
        honest = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        m_h = mx.take_along_axis(gates, honest, axis=-1).sum(axis=-1, keepdims=True)
        m_b = mx.take_along_axis(gates, inds, axis=-1).sum(axis=-1, keepdims=True)
        inds = mx.where(m_h - m_b > tau * m_h, honest, inds)
    slots = mx.take(slot_of, inds)
    return inds, mx.all(slots >= 0), mx.maximum(slots, 0)


# ---------------------------------------------------------------------------
# The same for a probability-routed (Qwen3.5/3.6/Next) router, T <= MAX_T rows,
# one threadgroup a row: MLX's precise softmax over the bf16 logits
# (softmax_single_row<bf16, float>, the gates rounded to bf16), route()'s bias,
# top-k and budget in the gates' dtype, then the selected gates renormalised
# (bf16 sum in element order, bf16 divide) when the block says norm_topk_prob.
_SRC_PROBS = r"""
  uint tid  = thread_position_in_threadgroup.x;
  uint lane = tid & 31u;
  uint sg   = tid >> 5;
  uint r    = threadgroup_position_in_grid.z;

  threadgroup float tg[NE];      // float of the bf16 gates
  threadgroup float lmax[32];
  threadgroup float lnorm[32];

  float ld[4];
  int base = int(tid) * 4;
  if (base + 4 <= NE) {
    for (int i = 0; i < 4; i++) { ld[i] = float(logits[r * NE + base + i]); }
  } else {
    for (int i = 0; i < 4; i++) {
      ld[i] = ((base + i) < NE) ? float(logits[r * NE + base + i]) : Limits<float>::min;
    }
  }
  if (sg == 0) { lmax[lane] = Limits<float>::min; lnorm[lane] = 0; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float maxval = Limits<float>::finite_min;
  for (int i = 0; i < 4; i++) { maxval = (maxval < ld[i]) ? ld[i] : maxval; }
  maxval = simd_max(maxval);
  if (lane == 0) { lmax[sg] = maxval; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    maxval = simd_max(lmax[lane]);
    if (lane == 0) { lmax[0] = maxval; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  maxval = lmax[0];
  float normalizer = 0;
  for (int i = 0; i < 4; i++) {
    float exp_x = metal::fast::exp(ld[i] - maxval);
    ld[i] = exp_x;
    normalizer += exp_x;
  }
  normalizer = simd_sum(normalizer);
  if (lane == 0) { lnorm[sg] = normalizer; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    normalizer = simd_sum(lnorm[lane]);
    if (lane == 0) { lnorm[0] = normalizer; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  normalizer = 1 / lnorm[0];
  for (int i = 0; i < 4; i++) {
    if ((base + i) < NE) { tg[base + i] = float(T(ld[i] * normalizer)); }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sg != 0) { return; }
  float gl[NPL];
  float lmx = -FLT_MAX;
  float lmn =  FLT_MAX;
  for (int i = 0; i < NPL; i++) {
    gl[i] = tg[lane + 32 * i];
    lmx = fmax(lmx, gl[i]);
    lmn = fmin(lmn, gl[i]);
  }
  // route()'s bias in the gates' dtype: spread and lam * spread in T, then
  // rank = float(gates) + float(ls) * mask
  T sp = (T)simd_max(lmx) - (T)simd_min(lmn);
  T ls = (T)lam[0] * sp;
  float lsf = (float)ls;
  uint pick[2][KT];
  for (uint p = 0; p < (BUDGET ? 2u : 1u); ++p) {
    float rk[NPL];
    for (int i = 0; i < NPL; i++) {
      rk[i] = (p == 0u) ? gl[i] + lsf * mask[lane + 32 * i] : gl[i];
    }
    for (uint j = 0; j < KT; ++j) {
      float bv = -FLT_MAX;
      uint  bi = 0u;
      for (int i = 0; i < NPL; i++) {
        if (rk[i] >= bv) { bv = rk[i]; bi = lane + 32 * i; }
      }
      float mv = simd_max(bv);
      uint  mi = simd_max((bv == mv) ? bi : 0u);
      pick[p][KT - 1u - j] = mi;
      if ((mi & 31u) == lane) { rk[mi >> 5] = -FLT_MAX; }
    }
  }
  bool back = false;
  if (BUDGET) {
    T mh = (T)tg[pick[1][0]];
    T mb = (T)tg[pick[0][0]];
    for (uint j = 1; j < KT; ++j) { mh = mh + (T)tg[pick[1][j]]; }
    for (uint j = 1; j < KT; ++j) { mb = mb + (T)tg[pick[0][j]]; }
    back = (mh - mb) > ((T)tau[0] * mh);
  }
  uint tsel[KT];
  for (uint j = 0; j < KT; ++j) { tsel[j] = back ? pick[1][j] : pick[0][j]; }
  T sum = (T)tg[tsel[0]];
  if (NORM) {
    for (uint j = 1; j < KT; ++j) { sum = sum + (T)tg[tsel[j]]; }
  }
  if (lane < KT) {
    inds[r * KT + lane] = tsel[lane];
    ids[r * KT + lane] = int(tsel[lane]);
    T s = (T)tg[tsel[lane]];
    scores[r * KT + lane] = NORM ? (T)(s / sum) : s;
  }
"""


def route_probs(logits, mask, lam, tau, k, norm):
    """A probability-routed router's selection and weights for [1, T, E]
    bf16 logits (T <= MAX_T): (inds uint32, scores in the logits' dtype,
    ids int32), each [1, T, k], equal to CacheBiasedMoE's ops on
    softmax(logits, precise=True)."""
    b, t, e = logits.shape
    assert b == 1 and t <= MAX_T and e % 32 == 0 and k <= 32, logits.shape
    key = ("probs",)
    kern = _KERNELS.get(key)
    if kern is None:
        kern = _KERNELS[key] = mx.fast.metal_kernel(
            name="route_probs", input_names=["logits", "mask", "lam", "tau"],
            output_names=["inds", "scores", "ids"], source=_SRC_PROBS)
    need = -(-e // 4)
    nt = 32 * -(-need // 32)
    return kern(
        inputs=[logits, mask, _scalar(lam), _scalar(tau)],
        template=[("T", logits.dtype), ("NE", e), ("KT", k), ("NPL", e // 32),
                  ("BUDGET", 1 if tau > 0.0 else 0), ("NORM", 1 if norm else 0)],
        grid=(nt, 1, t), threadgroup=(nt, 1, 1),
        output_shapes=[(1, t, k)] * 3, output_dtypes=[mx.uint32, logits.dtype, mx.int32])


def glue_probs(logits, mask, lam, tau, k, norm):
    """CacheBiasedMoE's ops for a probability-routed block, in its order: the
    reference route_probs must match bit for bit."""
    gates = mx.softmax(logits, axis=-1, precise=True)
    rank = gates + (lam * (mx.max(gates, axis=-1, keepdims=True)
                           - mx.min(gates, axis=-1, keepdims=True))) * mask if lam > 0.0 else gates
    inds = mx.argpartition(rank, kth=-k, axis=-1)[..., -k:]
    if tau > 0.0 and lam > 0.0:
        honest = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        m_h = mx.take_along_axis(gates, honest, axis=-1).sum(axis=-1, keepdims=True)
        m_b = mx.take_along_axis(gates, inds, axis=-1).sum(axis=-1, keepdims=True)
        inds = mx.where(m_h - m_b > tau * m_h, honest, inds)
    scores = mx.take_along_axis(gates, inds, axis=-1)
    if norm:
        scores = scores / scores.sum(axis=-1, keepdims=True)
    return inds, scores
