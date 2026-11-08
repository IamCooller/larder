"""MOE_ROUTE_K=k1:k2 (lossy, off when unset): Flash-Next's MoE blocks activate their top k1 experts, weighed by exp(logit) over the top k2 logits' sum of exp (arXiv 2609.04575), in every layer or only in MOE_ROUTE_K_LAYERS ("0-3,7")."""
import os
import sys
from pathlib import Path

import mlx.core as mx


def parse(v):
    """(k1, k2) from "k1:k2", or None when unset or empty."""
    if not v:
        return None
    try:
        k1, k2 = (int(p) for p in v.split(":"))
    except ValueError:
        raise ValueError(f"MOE_ROUTE_K={v!r}: expected k1:k2, e.g. 5:10") from None
    if not 1 <= k1 <= k2:
        raise ValueError(f"MOE_ROUTE_K={v!r}: needs 1 <= k1 <= k2")
    return k1, k2


def parse_layers(v):
    """The set of model.layers indices in "0-3,7,12", or None when unset or empty (every layer)."""
    if not v:
        return None
    try:
        out = set()
        for part in v.split(","):
            a, _, b = part.partition("-")
            out.update(range(int(a), int(b or a) + 1))
    except ValueError:
        raise ValueError(f"MOE_ROUTE_K_LAYERS={v!r}: expected indices and ranges, e.g. 0-3,7") from None
    if not out or min(out) < 0:
        raise ValueError(f"MOE_ROUTE_K_LAYERS={v!r}: needs indices >= 0")
    return out


K = parse(os.environ.get("MOE_ROUTE_K", ""))
LAYERS = parse_layers(os.environ.get("MOE_ROUTE_K_LAYERS", ""))


def weights(logits, inds, k2):
    """The selected experts' weights: exp(z_i) over the top k2 logits' sum of exp, the plain selected softmax when k2 is the selection's width."""
    sel = mx.take_along_axis(logits, inds, axis=-1)
    if k2 == inds.shape[-1]:
        return mx.softmax(sel, axis=-1, precise=True)
    ref = mx.topk(logits, k2, axis=-1)
    return mx.exp(sel - mx.logsumexp(ref, axis=-1, keepdims=True))


def _patch(cls):
    """The block's own lines with k2's weights for a marked instance, the stock call for every other one."""
    if cls.__dict__.get("_route_k"):
        return
    stock = cls.__call__

    def call(self, x):
        k2 = self.__dict__.get("_k2")
        if k2 is None:
            return stock(self, x)
        logits = self.gate(x.astype(mx.float32))
        idx = mx.argpartition(-logits, self.top_k - 1, axis=-1)[..., : self.top_k]
        w = weights(logits, idx, k2)
        out = (self.switch_mlp(x, idx) * w[..., None]).sum(axis=-2).astype(x.dtype)
        return out + mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)
    cls.__call__ = call
    cls._route_k = True


def install(model, moe_of, k=None, layers=None):
    """Mark the qwen4_exp MoE blocks of `model.layers` (those in `layers` when given) with k1 and k2 and return how many, 0 when off or a no-op."""
    k = K if k is None else k
    layers = LAYERS if layers is None else layers
    if k is None:
        return 0
    k1, k2 = k
    blocks = {i: b for i, b in enumerate(map(moe_of, model.layers))
              if b is not None and type(b).__module__.endswith("qwen4_exp") and hasattr(b, "top_k")}
    if not blocks:
        print(f"  route k: MOE_ROUTE_K={k1}:{k2} is for qwen4_exp MoE blocks; none here, off", flush=True)
        return 0
    first = next(iter(blocks.values()))
    trained, n_exp = first.top_k, first.gate.weight.shape[0]
    if k2 > n_exp:
        raise ValueError(f"MOE_ROUTE_K: k2 {k2} exceeds the {n_exp} experts")
    if layers is not None and not layers <= blocks.keys():
        raise ValueError(f"MOE_ROUTE_K_LAYERS: {sorted(layers - blocks.keys())} are not MoE layers")
    if k1 == k2 == trained:
        return 0
    marked = [b for i, b in blocks.items() if layers is None or i in layers]
    _patch(type(first))
    for b in marked:
        b.top_k = k1
        b.__dict__["_k2"] = k2
    if len(marked) < len(blocks):
        _mixed(blocks.values(), k1, trained)
    print(f"  route k: top {k1} of the trained {trained} experts, weighed over the top {k2}'s mass "
          f"(MOE_ROUTE_K, lossy) on {len(marked)} layers"
          + (f" ({','.join(map(str, sorted(layers)))})" if len(marked) < len(blocks) else ""), flush=True)
    return len(marked)


def _mixed(blocks, k1, trained):
    """Layers at two k: the engine sizes by the widest, and the native CUDA handshake has its kernels for both widths before an agent exists."""
    # Known limit: stream_fast.top_k_of reads `k` before `top_k`, which sizes every shared buffer by the widest layer at bias 0 (a cache-bias wrapper's own top_k would hide it, untested).
    for b in blocks:
        b.k = max(k1, trained)
    cu = getattr(mx, "cuda", None)
    if os.environ.get("MOE_NATIVE") == "1" and cu is not None and cu.is_available():
        sys.path.append(str(Path(__file__).parent / "native"))
        import moe_pipe_cuda
        for k in sorted({k1, trained}):
            moe_pipe_cuda.warm(k)
