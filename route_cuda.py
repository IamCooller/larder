"""MOE_CUDA_SLOTS / MOE_CUDA_ROUTE (CUDA only; never imported on Metal or CPU): fused_router.route()'s
cache-bias route as one mx.fast.cuda_kernel launch a MoE layer, bit-identical to CacheBiasedMoE's MLX ops on CUDA.

What it replaces, per layer (~16 kernels on CUDA, two of them full 512-wide block sorts):
    spread = max(gates) - min(gates); rank = gates + (lam * spread) * mask
    inds = argpartition(rank, -k)[..., -k:]
    [budget: the honest argpartition, 2x take_along_axis, 2x sum, sub, mul, gt, where]
    slots = take(slot_of, inds); ok = all(slots >= 0); rhs = max(slots, 0)

BIT-IDENTITY on CUDA (MLX 0.32.4's kernels, mlx/backend/cuda):
1. dtypes and op order as fused_router.py: spread and lam * spread in the gates' dtype, then
   rank = float(gates) + float(lam * spread) * mask in fp32 (mask is 0/1, so a contraction changes
   nothing).
2. argpartition is a full stable merge sort on CUDA (sort.cu, gpu_merge_sort for ArgPartition):
   ascending by (value, index), so the last k come ascending by value and a tie puts the larger
   index last. k rounds of argmax, the larger index on a tie, the j-th pick written to k-1-j.
3. The budget's two sums over k gates are row_reduce_simple (reduce/row_reduce.cu): lane l adds
   elements [l*N, l*N + N) in order from 0 (N = 16 / sizeof(T)), then cg::reduce's butterfly over
   the warp (xor 16, 8, 4, 2, 1), in the gates' dtype. Copied here lane for lane.

One warp a row (T <= MAX_T rows, one block); lane l keeps experts l, l + 32, ... in registers.
`table` mode (lever B): the mask is read from the slot table itself, m[e] = 0 <= slot_of[e] < lim,
instead of a host-made array.
"""
import mlx.core as mx

MAX_T = 5
# MOE_CUDA_ROWS (stream_fast sets it): the widest pass the unbiased block's probs() / tail() take, 1 the decode row only.
ROWS = 1
_KERNELS = {}
_SCALARS = {}

_HEADER = """
template <typename U>
__device__ __forceinline__ U route_warp_sum(U x) {
  // cg::reduce on a 32-thread tile (coalesced_reduce): the xor butterfly MLX's row reduce runs.
  for (int m = 16; m > 0; m >>= 1) x = x + __shfl_xor_sync(0xffffffffu, x, m);
  return x;
}
"""

_SRC = """
  const unsigned FULL = 0xffffffffu;
  const int lane = threadIdx.x & 31;
  const int r = threadIdx.x >> 5;               // one warp a row
  __shared__ float sg[NROWS * NE];              // the rows' gates, widened to float
  __shared__ unsigned pk[NROWS * 2 * KT];       // [row][biased, honest][KT], ascending by value
  __shared__ unsigned okf;
  if (threadIdx.x == 0) okf = 1u;

  float gl[NPL];
  float lmx = -INFINITY, lmn = INFINITY;
  for (int i = 0; i < NPL; ++i) {
    const int e = lane + 32 * i;
    const float v = static_cast<float>(gates[r * NE + e]);
    gl[i] = v;
    sg[r * NE + e] = v;
    lmx = fmaxf(lmx, v);
    lmn = fminf(lmn, v);
  }
  for (int m = 16; m > 0; m >>= 1) {
    lmx = fmaxf(lmx, __shfl_xor_sync(FULL, lmx, m));
    lmn = fminf(lmn, __shfl_xor_sync(FULL, lmn, m));
  }
  // spread = (T)max - (T)min ; ls = (T)lam * spread ; rank = float(g) + float(ls) * m
  const T sp = static_cast<T>(lmx) - static_cast<T>(lmn);
  const T ls = static_cast<T>(lam[0]) * sp;
  const float lsf = static_cast<float>(ls);
  float mk[NPL];
  for (int i = 0; i < NPL; ++i) {
    const int e = lane + 32 * i;
    @MASK@
  }

  for (int p = 0; p < (BUDGET ? 2 : 1); ++p) {
    float rk[NPL];
    for (int i = 0; i < NPL; ++i) rk[i] = (p == 0) ? __fadd_rn(gl[i], __fmul_rn(lsf, mk[i])) : gl[i];
    for (int j = 0; j < KT; ++j) {
      // the largest value, the largest index on a tie: MLX's stable ascending sort read from the back
      float bv = -INFINITY;
      int bi = -1;
      for (int i = 0; i < NPL; ++i) {
        if (rk[i] >= bv) { bv = rk[i]; bi = lane + 32 * i; }
      }
      for (int m = 16; m > 0; m >>= 1) {
        const float ov = __shfl_xor_sync(FULL, bv, m);
        const int oi = __shfl_xor_sync(FULL, bi, m);
        if (ov > bv || (ov == bv && oi > bi)) { bv = ov; bi = oi; }
      }
      if (lane == 0) pk[(r * 2 + p) * KT + KT - 1 - j] = (unsigned)bi;
      for (int i = 0; i < NPL; ++i) {
        if (lane + 32 * i == bi) rk[i] = -INFINITY;
      }
    }
  }
  __syncthreads();

  // The budget: honest mass against biased mass, each as row_reduce_simple sums k gates.
  bool back = false;
  if (BUDGET) {
    constexpr int NR = 16 / sizeof(T);
    T mh = static_cast<T>(0.0f), mb = static_cast<T>(0.0f);
    for (int i = 0; i < NR; ++i) {
      const int j = lane * NR + i;
      const T vh = (j < KT) ? static_cast<T>(sg[r * NE + pk[(r * 2 + 1) * KT + j]]) : static_cast<T>(0.0f);
      const T vb = (j < KT) ? static_cast<T>(sg[r * NE + pk[(r * 2) * KT + j]]) : static_cast<T>(0.0f);
      mh = mh + vh;
      mb = mb + vb;
    }
    mh = route_warp_sum(mh);
    mb = route_warp_sum(mb);
    back = (mh - mb) > (static_cast<T>(tau[0]) * mh);
  }

  if (lane < KT) {
    const unsigned sel = pk[(r * 2 + (back ? 1 : 0)) * KT + lane];
    inds[r * KT + lane] = sel;
    @SLOT@
  }
  @OK@
"""
_MASK = {False: "mk[i] = mask[e];",
         True: "const int s = slot_of[e]; mk[i] = (s >= 0 && s < lim[0]) ? 1.0f : 0.0f;"}
_SLOT = ("const int s = slot_of[sel]; rhs[r * KT + lane] = s > 0 ? s : 0; if (s < 0) okf = 0u;",
         "__syncthreads(); if (threadIdx.x == 0) ok[0] = okf;")


def _scalar(v, dt):
    a = _SCALARS.get((v, dt))
    if a is None:
        a = _SCALARS[(v, dt)] = mx.array([v], dtype=dt)
        mx.eval(a)
    return a


def _kernel(table, slots):
    key = (table, slots)
    k = _KERNELS.get(key)
    if k is None:
        ins = ["gates"] + ([] if table else ["mask"]) + ["slot_of", "lam", "tau"] + (["lim"] if table else [])
        outs = ["inds"] + (["ok", "rhs"] if slots else [])
        k = _KERNELS[key] = mx.fast.cuda_kernel(
            name="route_cuda" + ("_table" if table else "") + ("_slots" if slots else ""),
            input_names=ins, output_names=outs, header=_HEADER,
            source=_SRC.replace("@MASK@", _MASK[table]).replace("@SLOT@", _SLOT[0] if slots else "")
            .replace("@OK@", _SLOT[1] if slots else ""))
    return k


def fits(gates, k):
    """The shapes the kernel serves: batch 1, T <= MAX_T rows, E a multiple of 32 up to 1024, k <= 32,
    fp32 / bf16 / fp16 gates."""
    b, t, e = gates.shape
    return (b == 1 and t <= MAX_T and e % 32 == 0 and e <= 1024 and k <= 32
            and gates.dtype in (mx.float32, mx.bfloat16, mx.float16))


def route(gates, mask, slot_of, lam, tau, k, lim=None, slots=True):
    """fused_router.route()'s contract on CUDA: gates [1, T, E], mask [E] f32 (or None with `lim`:
    the mask is 0 <= slot_of < lim), slot_of [E] i32 -> (inds uint32 [1, T, k], ok uint32 [1],
    rhs int32 [1, T, k]); slots=False -> (inds,) only. tau > 0 turns the budget on."""
    _, t, e = gates.shape
    table = lim is not None
    ins = [gates] + ([] if table else [mask]) + [slot_of, _scalar(float(lam), mx.float32),
                                                   _scalar(float(tau), mx.float32)]
    if table:
        ins.append(_scalar(int(lim), mx.int32))
    tmpl = [("T", gates.dtype), ("NE", e), ("KT", k), ("NPL", e // 32), ("NROWS", t), ("BUDGET", tau > 0.0)]
    shapes, dts = [(1, t, k)], [mx.uint32]
    if slots:
        shapes += [(1,), (1, t, k)]
        dts += [mx.uint32, mx.int32]
    return _kernel(table, slots)(inputs=ins, template=tmpl, grid=(32 * t, 1, 1), threadgroup=(32 * t, 1, 1),
                                 output_shapes=shapes, output_dtypes=dts)


# ---------------------------------------------------------------------------
# MOE_CUDA_COMBINE (stream_fast): the decode row of mlx_lm's probability-routed MoE block (Qwen3-Next /
# Qwen3.5 / 3.6 with 256 experts, bf16, top-k <= 8) as two launches instead of ten, at cache bias 0:
#   probs(): softmax(gate logits, precise), argpartition top-k, take_along_axis, sum, divide;
#   tail():  (y * scores).sum(-2) + sigmoid(shared_expert_gate) * shared_expert, after the experts.
# Each copies MLX 0.32.4's CUDA kernel for that op (softmax.cu, sort.cu, row_reduce_simple,
# col_reduce_small, the unary functor): one warp, lane l holding logits [8l, 8l + 8) as
# softmax<bf16, float, 32, 8> does; the stable sort read from the back; the k scores summed by lane 0 in
# order from 0 in bf16 (row_reduce_simple's lane 0 holds all 8, the other lanes add zeros); the column sum
# in expert order from 0 in bf16 (col_reduce_small); each bf16 product and sum rounded on its own (the _rn
# intrinsics keep the compiler from contracting them, as separate kernels never are).
_TAIL_HEADER = """
#include <cooperative_groups/reduce.h>
#include "mlx/backend/cuda/device/unary_ops.cuh"
"""

_SRC_PROBS = """
  // one warp a row (NR rows, MOE_CUDA_ROWS), each row as the one-row kernel
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<32>(block);
  const int lane = threadIdx.x & 31;
  const int r = threadIdx.x >> 5;
  __shared__ T sgs[NR * 256];
  __shared__ unsigned pks[NR * KT];
  T* sg = sgs + r * 256;
  unsigned* pk = pks + r * KT;
  float v[8];
  for (int i = 0; i < 8; ++i) v[i] = static_cast<float>(logits[r * 256 + lane * 8 + i]);
  float prevmax = Limits<float>::finite_min();
  float maxval = prevmax;
  float normalizer = 0.0f;
  for (int i = 0; i < 8; ++i) maxval = cg::greater<float>()(maxval, v[i]);
  normalizer = normalizer * __expf(prevmax - maxval);
  for (int i = 0; i < 8; ++i) normalizer = normalizer + __expf(v[i] - maxval);
  prevmax = maxval;
  maxval = cg::reduce(warp, maxval, cg::greater<float>());
  normalizer = normalizer * __expf(prevmax - maxval);
  normalizer = cg::reduce(warp, normalizer, cg::plus<float>());
  // softmax.cu's second stage with one warp: the same max (its correction exp(0) is 1), lane 0's sum plus zeros.
  normalizer = __shfl_sync(0xffffffffu, normalizer, 0);
  normalizer = 1 / normalizer;
  float gl[8];
  for (int i = 0; i < 8; ++i) {
    const T g = static_cast<T>(__expf(v[i] - maxval) * normalizer);
    sg[lane * 8 + i] = g;
    gl[i] = static_cast<float>(g);
  }
  // argpartition is a stable ascending sort (sort.cu): the k largest by (value, index), the larger index
  // on a tie, the j-th pick written to k - 1 - j.
  for (int j = 0; j < KT; ++j) {
    float bv = -INFINITY;
    int bi = -1;
    for (int i = 0; i < 8; ++i) {
      if (gl[i] >= bv) { bv = gl[i]; bi = lane * 8 + i; }
    }
    for (int m = 16; m > 0; m >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, bv, m);
      const int oi = __shfl_xor_sync(0xffffffffu, bi, m);
      if (ov > bv || (ov == bv && oi > bi)) { bv = ov; bi = oi; }
    }
    if (lane == 0) pk[KT - 1 - j] = (unsigned)bi;
    for (int i = 0; i < 8; ++i) {
      if (lane * 8 + i == bi) gl[i] = -INFINITY;
    }
  }
  __syncthreads();
  if (lane < KT) {
    const unsigned e = pk[lane];
    T s = sg[e];
    if (NORM) {
      T sum = static_cast<T>(0.0f);
      for (int j = 0; j < KT; ++j) sum = __hadd_rn(sum, sg[pk[j]]);
      s = s / sum;
    }
    inds[r * KT + lane] = e;
    scores[r * KT + lane] = s;
  }
"""

_SRC_TAIL = """
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= NR * D) return;
  const int r = j / D, c = j - r * D;
  T acc = static_cast<T>(0.0f);
  for (int e = 0; e < KT; ++e) acc = __hadd_rn(acc, __hmul_rn(y[(r * KT + e) * D + c], scores[r * KT + e]));
  const T sig = Sigmoid{}(gate[r]);
  out[j] = __hadd_rn(acc, __hmul_rn(sig, shared[j]));
"""


def _tail_kernel(name):
    k = _KERNELS.get(name)
    if k is None:
        if name == "probs":
            k = mx.fast.cuda_kernel(name="moe_probs", input_names=["logits"], output_names=["inds", "scores"],
                                    header=_TAIL_HEADER, source=_SRC_PROBS)
        else:
            k = mx.fast.cuda_kernel(name="moe_tail", input_names=["y", "scores", "shared", "gate"],
                                    output_names=["out"], header=_TAIL_HEADER, source=_SRC_TAIL)
        _KERNELS[name] = k
    return k


def probs(logits, k, norm):
    """mlx_lm's router on T rows of 256 bf16 logits [1, T, 256] -> (inds uint32 [1, T, k], scores bf16
    [1, T, k]), each row equal to softmax(precise), argpartition(-k)[..., -k:], take_along_axis and, with `norm`,
    scores / scores.sum(-1)."""
    t = logits.shape[1]
    return _tail_kernel("probs")(inputs=[logits], template=[("T", logits.dtype), ("KT", k), ("NORM", bool(norm)),
                                                            ("NR", t)],
                                 grid=(32 * t, 1, 1), threadgroup=(32 * t, 1, 1), output_shapes=[(1, t, k)] * 2,
                                 output_dtypes=[mx.uint32, logits.dtype])


def tail(y, scores, shared, gate):
    """(y * scores[..., None]).sum(-2) + sigmoid(gate) * shared for T rows, each as the one-row ops: y [1, T, k, D],
    scores [1, T, k], shared [1, T, D], gate [1, T, 1], all bf16."""
    t, k, d = y.shape[-3:]
    return _tail_kernel("tail")(inputs=[y, scores, shared, gate],
                                template=[("T", y.dtype), ("KT", k), ("D", d), ("NR", t)],
                                grid=(-(-t * d // 256) * 256, 1, 1), threadgroup=(256, 1, 1),
                                output_shapes=[(1, t, d)], output_dtypes=[y.dtype])[0]


# MOE_CUDA_BIAS_TAIL (stream_fast): the same two-launch form for CacheBiasedMoE's decode row at a cache bias above
# 0, where the router is softmax, then route()'s table-mode bias, top-k and budget, then take_along_axis and the
# renormalisation (ten launches with the combine). bias_probs() is probs()' softmax (lane l holding [8l, 8l + 8))
# followed by _SRC's selection on those gates, each step copied from the kernel that was checked bit for bit; the
# combine is tail() as at bias 0. One warp a row (T <= MAX_T rows, one block).
_SRC_BPROBS = """
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<32>(block);
  const unsigned FULL = 0xffffffffu;
  const int lane = threadIdx.x & 31;
  const int r = threadIdx.x >> 5;
  __shared__ T sg[NROWS * 256];
  __shared__ unsigned pk[NROWS * 2 * KT];
  int sl[8];
  for (int i = 0; i < 8; ++i) sl[i] = slot_of[lane * 8 + i];     // the table lives in host memory: read it first
  float v[8];
  for (int i = 0; i < 8; ++i) v[i] = static_cast<float>(logits[r * 256 + lane * 8 + i]);
  float prevmax = Limits<float>::finite_min();
  float maxval = prevmax;
  float normalizer = 0.0f;
  for (int i = 0; i < 8; ++i) maxval = cg::greater<float>()(maxval, v[i]);
  normalizer = normalizer * __expf(prevmax - maxval);
  for (int i = 0; i < 8; ++i) normalizer = normalizer + __expf(v[i] - maxval);
  prevmax = maxval;
  maxval = cg::reduce(warp, maxval, cg::greater<float>());
  normalizer = normalizer * __expf(prevmax - maxval);
  normalizer = cg::reduce(warp, normalizer, cg::plus<float>());
  normalizer = __shfl_sync(FULL, normalizer, 0);
  normalizer = 1 / normalizer;
  float gl[8];
  float lmx = -INFINITY, lmn = INFINITY;
  for (int i = 0; i < 8; ++i) {
    const T g = static_cast<T>(__expf(v[i] - maxval) * normalizer);
    sg[r * 256 + lane * 8 + i] = g;
    gl[i] = static_cast<float>(g);
    lmx = fmaxf(lmx, gl[i]);
    lmn = fminf(lmn, gl[i]);
  }
  for (int m = 16; m > 0; m >>= 1) {
    lmx = fmaxf(lmx, __shfl_xor_sync(FULL, lmx, m));
    lmn = fminf(lmn, __shfl_xor_sync(FULL, lmn, m));
  }
  const T sp = static_cast<T>(lmx) - static_cast<T>(lmn);
  const T ls = static_cast<T>(lam[0]) * sp;
  const float lsf = static_cast<float>(ls);
  float mk[8];
  for (int i = 0; i < 8; ++i) mk[i] = (sl[i] >= 0 && sl[i] < lim[0]) ? 1.0f : 0.0f;
  for (int p = 0; p < (BUDGET ? 2 : 1); ++p) {
    float rk[8];
    for (int i = 0; i < 8; ++i) rk[i] = (p == 0) ? __fadd_rn(gl[i], __fmul_rn(lsf, mk[i])) : gl[i];
    for (int j = 0; j < KT; ++j) {
      float bv = -INFINITY;
      int bi = -1;
      for (int i = 0; i < 8; ++i) {
        if (rk[i] >= bv) { bv = rk[i]; bi = lane * 8 + i; }
      }
      for (int m = 16; m > 0; m >>= 1) {
        const float ov = __shfl_xor_sync(FULL, bv, m);
        const int oi = __shfl_xor_sync(FULL, bi, m);
        if (ov > bv || (ov == bv && oi > bi)) { bv = ov; bi = oi; }
      }
      if (lane == 0) pk[(r * 2 + p) * KT + KT - 1 - j] = (unsigned)bi;
      for (int i = 0; i < 8; ++i) {
        if (lane * 8 + i == bi) rk[i] = -INFINITY;
      }
    }
  }
  __syncthreads();
  bool back = false;
  if (BUDGET) {
    constexpr int NR = 16 / sizeof(T);
    T mh = static_cast<T>(0.0f), mb = static_cast<T>(0.0f);
    for (int i = 0; i < NR; ++i) {
      const int j = lane * NR + i;
      const T vh = (j < KT) ? sg[r * 256 + pk[(r * 2 + 1) * KT + j]] : static_cast<T>(0.0f);
      const T vb = (j < KT) ? sg[r * 256 + pk[(r * 2) * KT + j]] : static_cast<T>(0.0f);
      mh = mh + vh;
      mb = mb + vb;
    }
    for (int m = 16; m > 0; m >>= 1) {
      mh = mh + __shfl_xor_sync(FULL, mh, m);
      mb = mb + __shfl_xor_sync(FULL, mb, m);
    }
    back = (mh - mb) > (static_cast<T>(tau[0]) * mh);
  }
  const unsigned* sel = pk + (r * 2 + (back ? 1 : 0)) * KT;
  if (lane < KT) {
    const unsigned e = sel[lane];
    T s = sg[r * 256 + e];
    if (NORM) {
      T sum = static_cast<T>(0.0f);
      for (int j = 0; j < KT; ++j) sum = __hadd_rn(sum, sg[r * 256 + sel[j]]);
      s = s / sum;
    }
    inds[r * KT + lane] = e;
    scores[r * KT + lane] = s;
  }
"""


def bias_probs(logits, slot_of, lam, tau, k, lim, norm):
    """CacheBiasedMoE's decode router at a cache bias above 0 for [1, T, 256] bf16 gate logits (T <= MAX_T, k <= 8):
    softmax(precise), the bias toward 0 <= slot_of < lim, top-k, the budget, take_along_axis and with `norm` the
    renormalisation -> (inds uint32 [1, T, k], scores [1, T, k] in the logits' dtype)."""
    _, t, e = logits.shape
    assert e == 256 and t <= MAX_T and 0 < k <= 8 and lam > 0.0, (logits.shape, k, lam)
    kern = _KERNELS.get("bprobs")
    if kern is None:
        kern = _KERNELS["bprobs"] = mx.fast.cuda_kernel(
            name="moe_bias_probs", input_names=["logits", "slot_of", "lam", "tau", "lim"],
            output_names=["inds", "scores"], header=_TAIL_HEADER, source=_SRC_BPROBS)
    return kern(inputs=[logits, slot_of, _scalar(float(lam), mx.float32), _scalar(float(tau), mx.float32),
                        _scalar(int(lim), mx.int32)],
                template=[("T", logits.dtype), ("KT", k), ("NROWS", t), ("BUDGET", tau > 0.0), ("NORM", bool(norm))],
                grid=(32 * t, 1, 1), threadgroup=(32 * t, 1, 1), output_shapes=[(1, t, k)] * 2,
                output_dtypes=[mx.uint32, logits.dtype])


def _tail_fits(block):
    g = getattr(block, "gate", None)
    return (getattr(block, "sharding_group", None) is None and isinstance(g, dict) and "weight" in g
            and g.weight.shape[0] == 256 and 0 < block.top_k <= 8
            and getattr(block, "shared_expert", None) is not None
            and getattr(block, "shared_expert_gate", None) is not None)


_TAIL_CLASSES = {}


def _tail_class(cls):
    """cls with its decode row through probs() and tail(), every other call as cls has it."""
    if cls not in _TAIL_CLASSES:
        def __call__(self, x):
            rows = x.size // x.shape[-1]
            if x.ndim != 3 or x.shape[0] != 1 or not (rows == 1 or rows <= ROWS) or x.dtype != mx.bfloat16:
                return cls.__call__(self, x)
            inds, scores = probs(self.gate(x), self.top_k, self.norm_topk_prob)
            y = self.switch_mlp(x, inds)
            shared = self.shared_expert(x)
            gate = self.shared_expert_gate(x)
            if not (y.dtype == shared.dtype == gate.dtype == mx.bfloat16):
                y = (y * scores[..., None]).sum(axis=-2)
                return y + mx.sigmoid(gate) * shared
            return tail(y, scores, shared, gate)
        _TAIL_CLASSES[cls] = type(cls.__name__, (cls,), {"__call__": __call__, "_cuda_tail": True})
    return _TAIL_CLASSES[cls]


def install_tail(blocks):
    """Route each probability-routed block that fits through probs() and tail() and return how many."""
    n = 0
    for b in blocks:
        if (type(b).__name__ == "Qwen3NextSparseMoeBlock" and not getattr(type(b), "_cuda_tail", False)
                and _tail_fits(b)):
            b.__class__ = _tail_class(type(b))
            n += 1
    return n


# ---------------------------------------------------------------------------
# MOE_CUDA_ROWS: a 4-bit affine matmul of T bf16 rows, each row with the arithmetic of MLX 0.32.4's one-row qmv_kernel
# (qmm/qmv.cu: a warp an output row, 16 bf16 partial sums a lane, a float slot-order sum, the xor butterfly). MLX runs
# up to 7 rows as one-row passes over the weights, and here each k step's weights are dequantized once for all T rows.
_SRC_QMV = """
  const int N = w_shape[0];
  const int K = w_shape[1] * 8;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * RPB + (threadIdx.x >> 5);
  if (row >= N) return;
  const uint32_t* wr = w + (long)row * (K / 8);
  const __nv_bfloat16* sr = reinterpret_cast<const __nv_bfloat16*>(scales) + (long)row * (K / G);
  const __nv_bfloat16* br = reinterpret_cast<const __nv_bfloat16*>(biases) + (long)row * (K / G);
  const __nv_bfloat16* xb = reinterpret_cast<const __nv_bfloat16*>(x);
  const __nv_bfloat162 z2 = __halves2bfloat162(__ushort_as_bfloat16(0u), __ushort_as_bfloat16(0u));
  __nv_bfloat162 sums[NR][8];
  for (int r = 0; r < NR; ++r)
    for (int i = 0; i < 8; ++i) sums[r][i] = z2;
  for (int base = 0; base < K; base += 512) {
    const int idx = base + lane * 16;
    if (idx >= K) break;
    const uint2 wv = *reinterpret_cast<const uint2*>(wr + (idx >> 3));
    const __nv_bfloat162 ss = __halves2bfloat162(sr[idx / G], sr[idx / G]);
    const __nv_bfloat162 bb = __halves2bfloat162(br[idx / G], br[idx / G]);
    const uint32_t words[2] = {wv.x, wv.y};
    __nv_bfloat162 wd[8];
    for (int p = 0; p < 8; ++p) {
      const uint32_t byte2 = (words[p >> 2] >> ((p & 3) * 8)) & 0xffu;
      const __nv_bfloat162 q = __halves2bfloat162(__int2bfloat16_rn((int)(byte2 & 0xfu)),
                                                 __int2bfloat16_rn((int)(byte2 >> 4)));
      wd[p] = __hadd2(__hmul2(q, ss), bb);
    }
    for (int r = 0; r < NR; ++r) {
      const uint4 xa = *reinterpret_cast<const uint4*>(xb + (long)r * K + idx);
      const uint4 xc = *reinterpret_cast<const uint4*>(xb + (long)r * K + idx + 8);
      const uint32_t xs[8] = {xa.x, xa.y, xa.z, xa.w, xc.x, xc.y, xc.z, xc.w};
      for (int p = 0; p < 8; ++p) {
        const __nv_bfloat162 xp = __halves2bfloat162(__ushort_as_bfloat16((unsigned short)(xs[p] & 0xffffu)),
                                                    __ushort_as_bfloat16((unsigned short)(xs[p] >> 16)));
        sums[r][p] = __hfma2(xp, wd[p], sums[r][p]);
      }
    }
  }
  for (int r = 0; r < NR; ++r) {
    float s = 0.0f;
    for (int i = 0; i < 8; ++i) {
      s += __bfloat162float(__low2bfloat16(sums[r][i]));
      s += __bfloat162float(__high2bfloat16(sums[r][i]));
    }
    for (int m = 16; m > 0; m >>= 1) s += __shfl_xor_sync(0xffffffffu, s, m);
    if (lane == 0) out[(long)r * N + row] = __float2bfloat16(s);
  }
"""
QMV_RPB = 4     # output rows a block, the fastest of 4, 8 and 16 measured; no bit depends on it


def qmv_fits(x, w, scales, group_size, bits, mode, biases):
    """x [..., T, K] bf16 times a 4-bit affine bf16-scaled weight that qmv_rows() serves (K a multiple of 16 and of G)."""
    k = x.shape[-1]
    return (bits == 4 and mode == "affine" and biases is not None and x.dtype == mx.bfloat16
            and scales.dtype == mx.bfloat16 and biases.dtype == mx.bfloat16 and w.dtype == mx.uint32
            and k % 16 == 0 and k % group_size == 0 and group_size % 16 == 0 and w.shape[-1] * 8 == k
            and w.ndim == 2)


def qmv_rows(x, w, scales, biases, group_size):
    """mx.quantized_matmul(x, w, scales, biases, transpose=True, group_size, bits=4) for x [..., T, K] bf16, each row
    bit-equal to the one-row call, the weights read once for all rows."""
    key = (x.shape, w.shape[0], group_size)
    call = _QMV_CALLS.get(key)
    if call is None:
        # the launch's arguments by shape, made once: a verify pass makes ~300 of these calls on the host's critical path
        if "qmv" not in _KERNELS:
            _KERNELS["qmv"] = mx.fast.cuda_kernel(
                name="rows_qmv4", input_names=["x", "w", "scales", "biases"], output_names=["out"], source=_SRC_QMV,
                header="#include <cuda_bf16.h>\n")
        n = w.shape[0]
        call = _QMV_CALLS[key] = dict(
            template=[("NR", x.size // x.shape[-1]), ("G", group_size), ("RPB", QMV_RPB)],
            grid=(-(-n // QMV_RPB) * 32 * QMV_RPB, 1, 1), threadgroup=(32 * QMV_RPB, 1, 1),
            output_shapes=[x.shape[:-1] + (n,)], output_dtypes=[mx.bfloat16])
    return _KERNELS["qmv"](inputs=[x, w, scales, biases], **call)[0]


_QMV_CALLS = {}


# MOE_CUDA_ROWS: RoPE of T positions, each with the source lines of MLX 0.32.4's rope_single (rope.cu), the kernel a
# row-contiguous decode row takes. Its several-position kernel rounds otherwise now and then (197 of 24,576 query rows).
_SRC_ROPE = """
  const uint32_t px = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t r = blockIdx.y * blockDim.y + threadIdx.y;
  if (px >= HALF || r >= ROWS) return;
  const T* in = x + (long)r * D;
  T* o = out + (long)r * D;
  for (uint32_t c = 2 * HALF + px; c < D; c += HALF) o[c] = in[c];
  float d = static_cast<float>(px) / static_cast<float>(HALF);
  float inv_freq = exp2(-d * base[0]);
  float L = scale[0] * static_cast<float>(offset[0] + static_cast<int>(r % NR));
  float theta = L * inv_freq;
  float costheta = cos(theta);
  float sintheta = sin(theta);
  float x1 = static_cast<float>(in[px]);
  float x2 = static_cast<float>(in[px + HALF]);
  float rx1 = x1 * costheta - x2 * sintheta;
  float rx2 = x1 * sintheta + x2 * costheta;
  o[px] = static_cast<T>(rx1);
  o[px + HALF] = static_cast<T>(rx2);
"""
_LOG2F = []


def rope_rows(x, dims, base, scale, offset):
    """mx.fast.rope(x, dims, traditional=False, base=base, scale=scale, offset=offset) for x [1, H, T, D], each position
    with the bits of the one-position call (offset + t)."""
    if not _LOG2F:
        import ctypes
        f = ctypes.CDLL("libm.so.6").log2f      # std::log2(float), as rope.cu passes log2(base) from the host
        f.restype, f.argtypes = ctypes.c_float, [ctypes.c_float]
        _LOG2F.append(f)
    k = _KERNELS.get("rope")
    if k is None:
        k = _KERNELS["rope"] = mx.fast.cuda_kernel(name="rows_rope", input_names=["x", "offset", "base", "scale"],
                                                  output_names=["out"], source=_SRC_ROPE)
    _, h, t, d = x.shape
    half = dims // 2
    lb = _scalar(_LOG2F[0](float(base)), mx.float32)
    off = offset if isinstance(offset, mx.array) else mx.array(offset, dtype=mx.int32)
    return k(inputs=[x, off.astype(mx.int32), lb, _scalar(float(scale), mx.float32)],
             template=[("T", x.dtype), ("NR", t), ("HALF", half), ("D", d), ("ROWS", h * t)],
             grid=(-(-half // 32) * 32, -(-h * t // 8) * 8, 1), threadgroup=(32, 8, 1), output_shapes=[x.shape],
             output_dtypes=[x.dtype])[0]
