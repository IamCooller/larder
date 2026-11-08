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
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<32>(block);
  const int lane = threadIdx.x;
  __shared__ T sg[256];
  __shared__ unsigned pk[KT];
  float v[8];
  for (int i = 0; i < 8; ++i) v[i] = static_cast<float>(logits[lane * 8 + i]);
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
    inds[lane] = e;
    scores[lane] = s;
  }
"""

_SRC_TAIL = """
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= D) return;
  T acc = static_cast<T>(0.0f);
  for (int e = 0; e < KT; ++e) acc = __hadd_rn(acc, __hmul_rn(y[e * D + j], scores[e]));
  const T sig = Sigmoid{}(gate[0]);
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
    """mlx_lm's router on one row of 256 bf16 logits [1, 1, 256] -> (inds uint32 [1, 1, k], scores bf16
    [1, 1, k]), equal to softmax(precise), argpartition(-k)[..., -k:], take_along_axis and, with `norm`,
    scores / scores.sum(-1)."""
    return _tail_kernel("probs")(inputs=[logits], template=[("T", logits.dtype), ("KT", k), ("NORM", bool(norm))],
                                 grid=(32, 1, 1), threadgroup=(32, 1, 1), output_shapes=[(1, 1, k)] * 2,
                                 output_dtypes=[mx.uint32, logits.dtype])


def tail(y, scores, shared, gate):
    """(y * scores[..., None]).sum(-2) + sigmoid(gate) * shared for one row: y [1, 1, k, D], scores [1, 1, k],
    shared [1, 1, D], gate [1, 1, 1], all bf16."""
    k, d = y.shape[-2:]
    return _tail_kernel("tail")(inputs=[y, scores, shared, gate], template=[("T", y.dtype), ("KT", k), ("D", d)],
                                grid=(-(-d // 256) * 256, 1, 1), threadgroup=(256, 1, 1),
                                output_shapes=[(1, 1, d)], output_dtypes=[y.dtype])[0]


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
            if x.size != x.shape[-1] or x.dtype != mx.bfloat16:
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
