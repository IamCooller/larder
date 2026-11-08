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
