"""MOE_CUDA_GDN (on by default when MLX runs on CUDA, MOE_CUDA_GDN=0 turns it off; never on Metal or
CPU, where nothing here is imported): mlx_lm's gated delta recurrence (GatedDeltaNet, Qwen3-Next /
Qwen3.5 / 3.6 / Flash-Next and the other hybrids) as one CUDA kernel with the Metal kernel's math (one
warp per value row, state in float32 registers), because the Python-loop fallback keeps a float32 state
per position in the graph and its memory forces narrow prefill passes."""
import inspect

import mlx.core as mx
from engine_log import note

_HEADER = """
__device__ __forceinline__ float gdn_warp_sum(float x) {
  for (int o = 16; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffff, x, o);
  return x;
}
"""

_SRC = """
  // One warp per (sequence, value head, value row); lane l keeps state[l * NPT, (l + 1) * NPT).
  constexpr int NPT = Dk / 32;
  const long T = q_shape[1];
  const int n = blockIdx.z;
  const long b_idx = n / Hv;
  const int hv = n % Hv;
  const int hk = hv / (Hv / Hk);
  const int lane = threadIdx.x;
  const int dv = blockIdx.y * blockDim.y + threadIdx.y;
  if (dv >= Dv) return;
  const long bt = b_idx * T;
  auto q_ = q + (bt * Hk + hk) * Dk + lane * NPT;
  auto k_ = k + (bt * Hk + hk) * Dk + lane * NPT;
  auto v_ = v + (bt * Hv + hv) * Dv + dv;
  auto y_ = y + (bt * Hv + hv) * Dv + dv;
  auto beta_ = beta + bt * Hv + hv;
  @G_SETUP@
  const long so = ((long)n * Dv + dv) * Dk + lane * NPT;
  float s[NPT];
  for (int i = 0; i < NPT; ++i) s[i] = static_cast<float>(state_in[so + i]);
  for (long t = 0; t < T; ++t) {
    if (@MASK@) {
      float kv = 0.0f;
      for (int i = 0; i < NPT; ++i) {
        s[i] = s[i] * @G_AT@;
        kv += s[i] * static_cast<float>(k_[i]);
      }
      kv = gdn_warp_sum(kv);
      const float delta = (static_cast<float>(*v_) - kv) * static_cast<float>(*beta_);
      float out = 0.0f;
      for (int i = 0; i < NPT; ++i) {
        s[i] = s[i] + static_cast<float>(k_[i]) * delta;
        out += s[i] * static_cast<float>(q_[i]);
      }
      out = gdn_warp_sum(out);
      if (lane == 0) *y_ = static_cast<InT>(out);
    } else if (lane == 0) {
      *y_ = static_cast<InT>(0.0f);
    }
    q_ += Hk * Dk;
    k_ += Hk * Dk;
    v_ += Hv * Dv;
    y_ += Hv * Dv;
    beta_ += Hv;
    @G_STEP@
  }
  for (int i = 0; i < NPT; ++i) state_out[so + i] = static_cast<StT>(s[i]);
"""

# g: [B, T, Hv] (one gate a head) or [B, T, Hv, Dk] (vectorized).
_GATE = {False: ("auto g_ = g + bt * Hv + hv;", "static_cast<float>(*g_)", "g_ += Hv;"),
         True: ("auto g_ = g + (bt * Hv + hv) * Dk + lane * NPT;", "static_cast<float>(g_[i])",
                "g_ += Hv * Dk;")}
_KERNELS = {}


def _kernel(masked, vec):
    key = (masked, vec)
    if key not in _KERNELS:
        setup, at, step = _GATE[vec]
        src = (_SRC.replace("@G_SETUP@", setup).replace("@G_AT@", at).replace("@G_STEP@", step)
               .replace("@MASK@", "mask[bt + t]" if masked else "true"))
        _KERNELS[key] = mx.fast.cuda_kernel(
            name="gdn_step" + ("_vec" if vec else "") + ("_mask" if masked else ""),
            input_names=["q", "k", "v", "g", "beta", "state_in"] + (["mask"] if masked else []),
            output_names=["y", "state_out"], source=src, header=_HEADER)
    return _KERNELS[key]


def recurrence(q, k, v, g, beta, state, mask=None):
    """gated_delta_ops / gated_delta_kernel's contract: q, k [B, T, Hk, Dk], v [B, T, Hv, Dv], g
    [B, T, Hv] or [B, T, Hv, Dk], beta [B, T, Hv], state [B, Hv, Dv, Dk], mask [B, T] or None ->
    y [B, T, Hv, Dv] in q's dtype, the state in its own. Dk a multiple of 32."""
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    inputs = [q, k, v, g, beta, state] + ([mask] if mask is not None else [])
    return _kernel(mask is not None, g.ndim == 4)(
        inputs=inputs,
        template=[("InT", q.dtype), ("StT", state.dtype), ("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv)],
        grid=(32, Dv, B * Hv), threadgroup=(32, 4, 1),
        output_shapes=[(B, T, Hv, Dv), state.shape], output_dtypes=[q.dtype, state.dtype])


# MOE_CUDA_GDN_GLUE: qwen3_5's GatedDeltaNet decode row (one position, no mask) with the glue around the recurrence in two
# kernels instead of ten. step_glue(): nn.silu of the conv output (compiled: Multiply(x, Sigmoid(x)) in its dtype),
# normalize_qk's two rms_norms (rms_norm_small<T, 64, 1, false, 8>: lanes < Dk / 8 sum 8 squares each as fused
# multiply-adds, cg::reduce over the warp, rsqrt(sum / Dk + eps)) and scales, mx.sigmoid(b), compute_g (compiled, its
# tape: Exp, Negative, Add, LogAddExp with 0, AsType, Multiply, Exp) and _SRC's recurrence for one position. A block of
# 16 value rows makes its head's q and k once in shared memory. norm_gate(): Qwen3NextRMSNormGated as rms_norm (with its weight) and precise_swiglu
# (compiled: AsType, Sigmoid, Multiply, AsType, Multiply, AsType). The functors are MLX's own (unary_ops.cuh,
# binary_ops.cuh), as its compiled kernels call them.
_GLUE_HEADER = _HEADER + """
#include <cooperative_groups/reduce.h>
#include "mlx/backend/cuda/device/unary_ops.cuh"
#include "mlx/backend/cuda/device/binary_ops.cuh"
template <typename U>
__device__ __forceinline__ U gdn_silu(U x) { return mlx::core::cu::Multiply{}(x, mlx::core::cu::Sigmoid{}(x)); }
"""

_SRC_STEP_GLUE = """
  // A block: RB value rows of one value head. Its threads first make the head's q and k (silu, norm, scale) once,
  // into shared memory, then each warp runs _SRC's recurrence for its row.
  constexpr int NPT = Dk / 32;
  __shared__ InT sv[2 * Dk];
  __shared__ float qk[2 * Dk];
  __shared__ float nrm[2], gb[2];
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<32>(block);
  const int hv = blockIdx.z;
  const int hk = hv / (Hv / Hk);
  const int lane = threadIdx.x;
  const int w = threadIdx.y;
  const int t = w * 32 + lane;
  const int dv = blockIdx.y * blockDim.y + w;
  // the row's state and v first, so their loads run under the q / k work below
  const long so = ((long)hv * Dv + (dv < Dv ? dv : 0)) * Dk + lane * NPT;
  float s[NPT];
  for (int i = 0; i < NPT; ++i) s[i] = static_cast<float>(state_in[so + i]);
  const InT vr = conv[2 * Hk * Dk + hv * Dv + (dv < Dv ? dv : 0)];
  if (t < 2 * Dk) sv[t] = gdn_silu(conv[(t < Dk ? hk : Hk + hk - 1) * Dk + t]);
  if (t == 2 * Dk) {
    gb[1] = static_cast<float>(Sigmoid{}(b[hv]));
    const float en = Negative{}(Exp{}(static_cast<float>(A_log[hv])));
    const TS s0 = Add{}(static_cast<TS>(a[hv]), static_cast<TS>(dt_bias[hv]));
    const TS sp = LogAddExp{}(s0, static_cast<TS>(0));
    gb[0] = Exp{}(Multiply{}(en, static_cast<float>(sp)));
  }
  __syncthreads();
  if (w < 2) {
    // rms_norm_small's sum of squares for q (warp 0) and k (warp 1)
    float n = 0.0f;
    if (lane < Dk / 8) {
      for (int j = 0; j < 8; ++j) {
        const float x = static_cast<float>(sv[w * Dk + lane * 8 + j]);
        n += x * x;
      }
    }
    n = cg::reduce(warp, n, cg::plus<float>());
    if (lane == 0) nrm[w] = rsqrt(n / Dk + eps[0]);
  }
  __syncthreads();
  if (t < 2 * Dk) {
    const InT y = static_cast<InT>(static_cast<float>(sv[t]) * nrm[t < Dk ? 0 : 1]);
    qk[t] = static_cast<float>(Multiply{}(y, t < Dk ? sq[0] : sk[0]));
  }
  __syncthreads();
  if (dv >= Dv) return;
  float qf[NPT], kf[NPT];
  for (int i = 0; i < NPT; ++i) {
    qf[i] = qk[lane * NPT + i];
    kf[i] = qk[Dk + lane * NPT + i];
  }
  const float vf = static_cast<float>(gdn_silu(vr));
  const float bf = gb[1];
  const float gf = gb[0];
  float kv = 0.0f;
  for (int i = 0; i < NPT; ++i) {
    s[i] = s[i] * gf;
    kv += s[i] * kf[i];
  }
  kv = gdn_warp_sum(kv);
  const float delta = (vf - kv) * bf;
  float out = 0.0f;
  for (int i = 0; i < NPT; ++i) {
    s[i] = s[i] + kf[i] * delta;
    out += s[i] * qf[i];
  }
  out = gdn_warp_sum(out);
  if (lane == 0) y[hv * Dv + dv] = static_cast<InT>(out);
  for (int i = 0; i < NPT; ++i) state_out[so + i] = static_cast<StT>(s[i]);
"""

_SRC_NORM_GATE = """
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<32>(block);
  const int lane = threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= R) return;
  const T* xr = x + (long)row * D;
  float ns = 0.0f;
  if (lane < D / 8) {
    for (int j = 0; j < 8; ++j) {
      const float t = static_cast<float>(xr[lane * 8 + j]);
      ns += t * t;
    }
  }
  ns = cg::reduce(warp, ns, cg::plus<float>());
  ns = rsqrt(ns / D + eps[0]);
  for (int e = lane; e < D; e += 32) {
    const T n = w[e] * static_cast<T>(static_cast<float>(xr[e]) * ns);
    const float gz = static_cast<float>(z[(long)row * D + e]);
    const float gs = Multiply{}(gz, Sigmoid{}(gz));
    out[(long)row * D + e] = static_cast<T>(Multiply{}(gs, static_cast<float>(n)));
  }
"""
_SCALARS = {}


def _scalar(v, dt):
    """A one-element array as MLX makes a Python scalar in an op on dtype dt (1 x v, so the same rounding)."""
    a = _SCALARS.get((v, dt))
    if a is None:
        a = _SCALARS[(v, dt)] = mx.multiply(mx.ones((1,), dtype=dt), v)
        mx.eval(a)
    return a


def step_glue(conv, a, b, A_log, dt_bias, state, Hk, Dk, Hv, Dv):
    """One position of qwen3_5's GatedDeltaNet from the conv output (before silu): conv [1, 1, 2 Hk Dk + Hv Dv], a / b
    [1, 1, Hv], A_log / dt_bias [Hv], state [1, Hv, Dv, Dk] -> (y [1, 1, Hv, Dv] in conv's dtype, the new state),
    equal to silu, the split, normalize_qk, gated_delta_update's sigmoid / compute_g and recurrence."""
    assert conv.shape[:2] == (1, 1) and Dk % 32 == 0 and Dk <= 128 and Hv % Hk == 0, (conv.shape, Dk)
    inv = Dk ** -0.5
    k = _KERNELS.get("step_glue")
    if k is None:
        k = _KERNELS["step_glue"] = mx.fast.cuda_kernel(
            name="gdn_step_glue", input_names=["conv", "a", "b", "A_log", "dt_bias", "state_in", "sq", "sk", "eps"],
            output_names=["y", "state_out"], source=_SRC_STEP_GLUE, header=_GLUE_HEADER)
    ts = (mx.zeros((1,), dtype=a.dtype) + mx.zeros((1,), dtype=dt_bias.dtype)).dtype
    return k(inputs=[conv, a, b, A_log, dt_bias, state, _scalar(inv ** 2, conv.dtype), _scalar(inv, conv.dtype),
                     _scalar(1e-6 * inv ** 2, mx.float32)],
             template=[("InT", conv.dtype), ("StT", state.dtype), ("TS", ts), ("Dk", Dk), ("Dv", Dv), ("Hk", Hk),
                       ("Hv", Hv)],
             grid=(32, -(-Dv // 16) * 16, Hv), threadgroup=(32, 16, 1), output_shapes=[(1, 1, Hv, Dv), state.shape],
             output_dtypes=[conv.dtype, state.dtype])


def norm_gate(x, z, weight, eps):
    """Qwen3NextRMSNormGated(x, z) for rows of D <= 256: rms_norm with `weight`, then precise_swiglu(x, z, normed)."""
    d = x.shape[-1]
    assert d % 8 == 0 and d <= 256 and z.shape == x.shape and weight.dtype == x.dtype, (x.shape, weight.dtype)
    r = x.size // d
    k = _KERNELS.get("norm_gate")
    if k is None:
        k = _KERNELS["norm_gate"] = mx.fast.cuda_kernel(
            name="gdn_norm_gate", input_names=["x", "z", "w", "eps"], output_names=["out"], source=_SRC_NORM_GATE,
            header=_GLUE_HEADER)
    return k(inputs=[x, z, weight, _scalar(float(eps), mx.float32)], template=[("T", x.dtype), ("D", d), ("R", r)],
             grid=(32, -(-r // 4) * 4, 1), threadgroup=(32, 4, 1), output_shapes=[x.shape], output_dtypes=[x.dtype])[0]


def install_glue():
    """MOE_CUDA_GDN_GLUE: qwen3_5.GatedDeltaNet's decode row through step_glue() and norm_gate(), its other calls as they are."""
    import mlx_lm.models.qwen3_5 as q35

    cls = q35.GatedDeltaNet
    if getattr(cls, "_gdn_glue", False):
        return
    orig = cls.__call__

    def bf16_out(layer):
        p = getattr(layer, "scales", None)
        p = getattr(layer, "weight", None) if p is None else p
        return p is not None and p.dtype == mx.bfloat16

    def __call__(self, inputs, mask=None, cache=None):
        B, S, _ = inputs.shape
        if not (B == 1 and S == 1 and mask is None and cache is not None and cache[1] is not None
                and getattr(cache, "lengths", None) is None and self.sharding_group is None and not self.training
                and inputs.dtype == mx.bfloat16 and self.head_k_dim % 32 == 0 and self.head_k_dim <= 128
                and self.head_v_dim <= 256 and self.head_v_dim % 8 == 0 and self.norm.weight.dtype == mx.bfloat16
                and all(bf16_out(m) for m in (self.in_proj_qkv, self.in_proj_a, self.in_proj_b, self.conv1d))):
            return orig(self, inputs, mask, cache)
        # mlx_lm's lines up to the conv, for S = 1 and no mask
        qkv = self.in_proj_qkv(inputs)
        z = self.in_proj_z(inputs).reshape(B, S, self.num_v_heads, self.head_v_dim)
        b = self.in_proj_b(inputs)
        a = self.in_proj_a(inputs)
        conv_state = cache[0] if cache[0] is not None else mx.zeros(
            (B, self.conv_kernel_size - 1, self.conv_dim), dtype=inputs.dtype)
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        cache[0] = mx.contiguous(conv_input[:, -(self.conv_kernel_size - 1):, :])
        conv = self.conv1d(conv_input)
        out, state = step_glue(conv, a, b, self.A_log, self.dt_bias, cache[1], self.num_k_heads, self.head_k_dim,
                               self.num_v_heads, self.head_v_dim)
        cache[1] = state
        cache.advance(S)
        out = norm_gate(out, z, self.norm.weight, self.norm.eps)
        return self.out_proj(out.reshape(B, S, -1))

    cls.__call__ = __call__
    cls._gdn_glue = True
    cls._gdn_glue_orig = orig


_PARAMS = ["q", "k", "v", "a", "b", "A_log", "dt_bias", "state", "mask", "use_kernel", "lower_bound",
           "allow_neg_eigval"]


def install():
    """Route mlx_lm's gated_delta_update to the kernel wherever its Metal kernel would have run: in
    gated_delta itself, which every model module imports the name from, and in the modules that
    already did. An mlx_lm whose gated_delta_update takes other arguments is left alone (said once)."""
    import sys
    import mlx_lm.models.gated_delta as gd

    orig = gd.gated_delta_update
    if getattr(orig, "_gdn_cuda", False):
        return
    if list(inspect.signature(orig).parameters) != _PARAMS:
        note("  gdn_cuda: mlx_lm's gated_delta_update has other arguments; its ops loop stays", flush=True)
        return

    def gated_delta_update(q, k, v, a, b, A_log, dt_bias, state=None, mask=None, *, use_kernel=True,
                           lower_bound=None, allow_neg_eigval=False):
        # mlx_lm 0.32's own gating and dispatch, with this kernel where Metal's would run.
        if not use_kernel or mx.default_device() != mx.gpu or k.shape[-1] < 32 or k.shape[-1] % 32:
            return orig(q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel=use_kernel,
                        lower_bound=lower_bound, allow_neg_eigval=allow_neg_eigval)
        beta = mx.sigmoid(b)
        if allow_neg_eigval:
            beta = beta * 2.0
        if lower_bound is None:
            g = gd.compute_g(A_log, a, dt_bias)
        else:
            g = gd.compute_lower_bound_g(A_log, a, dt_bias, lower_bound)
        if state is None:
            B, _, Hk, Dk = q.shape
            Hv, Dv = v.shape[-2:]
            state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
        return recurrence(q, k, v, g, beta, state, mask)

    gated_delta_update._gdn_cuda = True
    gated_delta_update.ops = orig
    gd.gated_delta_update = gated_delta_update
    for name, mod in list(sys.modules.items()):
        if name.startswith("mlx_lm.models.") and getattr(mod, "gated_delta_update", None) is orig:
            mod.gated_delta_update = gated_delta_update
