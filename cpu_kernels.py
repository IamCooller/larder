"""MLX's CPU device with cpu_ext's threaded AVX-512 kernels: install() routes mx.quantized_matmul, mx.gather_qmm,
mx.fast.scaled_dot_product_attention and mlx_lm's GatedDeltaNet scan to cpu_kern for the cases it covers, stock MLX
for everything else. CPU device only."""
import os
import sys

import mlx.core as mx

_ORIG = {}
_FLOAT = (mx.float32, mx.bfloat16, mx.float16)
cpu_kern = None


def physical_cores():
    try:
        with open("/proc/cpuinfo") as f:
            for ln in f:
                if ln.startswith("cpu cores"):
                    return int(ln.split(":")[1])
    except OSError:
        pass
    return os.cpu_count() or 1


def _ok(x, w, scales, biases, transpose, gs, bits, mode):
    return (transpose and mode == "affine" and biases is not None and bits in (4, 8) and gs % 32 == 0
            and w.dtype == mx.uint32 and x.dtype in _FLOAT and scales.dtype in _FLOAT and biases.dtype == scales.dtype
            and x.ndim >= 1 and x.shape[-1] % gs == 0 and x.shape[-1] == w.shape[-1] * 32 // bits)


def quantized_matmul(x, w, scales, biases=None, transpose=True, group_size=None, bits=None, mode="affine", *,
                     stream=None):
    gs, b = group_size or 64, bits or 4
    if w.ndim == 2 and _ok(x, w, scales, biases, transpose, gs, b, mode):
        return cpu_kern.qmm(x, w, scales, biases, gs, b, stream=stream)
    return _ORIG["quantized_matmul"](x, w, scales, biases, transpose, group_size, bits, mode, stream=stream)


def gather_qmm(x, w, scales, biases=None, lhs_indices=None, rhs_indices=None, transpose=True, group_size=None,
               bits=None, mode="affine", global_scale=None, *, sorted_indices=False, stream=None):
    gs, b = group_size or 64, bits or 4
    if (lhs_indices is None and rhs_indices is not None and global_scale is None and w.ndim == 3 and x.ndim >= 2
            and _ok(x, w, scales, biases, transpose, gs, b, mode)):
        return cpu_kern.gather_qmm(x, w, scales, biases, rhs_indices, gs, b, stream=stream)
    return _ORIG["gather_qmm"](x, w, scales, biases, lhs_indices, rhs_indices, transpose, group_size, bits, mode,
                               global_scale, sorted_indices=sorted_indices, stream=stream)


def gated_delta_ops(q, k, v, g, beta, state=None, mask=None):
    """mlx_lm's GatedDeltaNet scan (scalar gating, no mask) in one threaded pass per step; anything else to stock."""
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
    if (mask is None and g.ndim == 3 and state.dtype == mx.float32 and k.dtype == q.dtype and Dk % 16 == 0
            and Dk <= 512 and Hv % Hk == 0 and all(a.dtype in _FLOAT for a in (q, v, g, beta))):
        y, s = cpu_kern.gdn(q, k, v, g, beta, state)
        return y, s
    return _ORIG["gated_delta_ops"](q, k, v, g, beta, state, mask)


def scaled_dot_product_attention(q, k, v, *, scale, mask=None, sinks=None, stream=None):
    """mx.fast's SDPA for no mask, "causal" or a boolean [1.., T, N] mask; anything else (sinks, additive) to stock."""
    if (sinks is None and q.ndim == 4 and k.ndim == 4 and v.ndim == 4 and q.shape[1] % k.shape[1] == 0
            and q.shape[1] // k.shape[1] <= 16 and q.shape[3] % 16 == 0 and q.shape[3] <= 512
            and v.shape[3] % 16 == 0 and all(a.dtype in _FLOAT for a in (q, k, v))):
        if mask is None:
            return cpu_kern.attn(q, k, v, scale, stream=stream)
        if isinstance(mask, str) and mask == "causal":
            return cpu_kern.attn(q, k, v, scale, "causal", stream=stream)
        if (isinstance(mask, mx.array) and mask.dtype == mx.bool_ and mask.ndim >= 2
                and tuple(mask.shape[-2:]) == (q.shape[2], k.shape[2]) and all(d == 1 for d in mask.shape[:-2])):
            return cpu_kern.attn(q, k, v, scale, mask=mask, stream=stream)
    return _ORIG["sdpa"](q, k, v, scale=scale, mask=mask, sinks=sinks, stream=stream)


def install(threads=None):
    """Patch mx.quantized_matmul, mx.gather_qmm and mlx_lm's GatedDeltaNet scan (idempotent); CPU device only."""
    global cpu_kern
    if _ORIG:
        return
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, "cpu_ext", "build"))
    try:
        import cpu_kern as ck
    except ImportError as e:
        raise ImportError(f"cpu_kern is not built for this python: run {here}/cpu_ext/build.sh {sys.executable} "
                          "(or MOE_CPU_KERNELS=0 for MLX's own, ~40-100x slower CPU kernels)") from e
    from mlx_lm.models import gated_delta
    cpu_kern = ck
    cpu_kern.set_threads(threads or int(os.environ.get("MOE_CPU_THREADS", 0)) or physical_cores())
    _ORIG["quantized_matmul"], _ORIG["gather_qmm"] = mx.quantized_matmul, mx.gather_qmm
    mx.quantized_matmul, mx.gather_qmm = quantized_matmul, gather_qmm
    _ORIG["gated_delta_ops"] = gated_delta.gated_delta_ops
    gated_delta.gated_delta_ops = gated_delta_ops
    _ORIG["sdpa"] = mx.fast.scaled_dot_product_attention
    mx.fast.scaled_dot_product_attention = scaled_dot_product_attention
