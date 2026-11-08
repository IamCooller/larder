"""A GatedResidual's 4-row inject matrix stacked under its down matrix, one quantized_matmul
for both, on by default and MOE_FUSE_HC=0 turns it off; and on CUDA the projections a block calls
on one input stacked the same way (install_fuse_proj, stream_fast's MOE_FUSE_PROJ)."""
import os

import mlx.core as mx
import mlx.nn as nn


def _fusable(parts, dim):
    return all(
        isinstance(p, nn.QuantizedLinear)
        and "bias" not in p
        and p.weight.shape[1] * 32 // p.bits == dim
        and (p.bits, p.group_size) == (parts[0].bits, parts[0].group_size)
        for p in parts
    )


HC_ON = os.environ.get("MOE_FUSE_HC", "1") == "1"


class _HCFused:
    """down's rows, inject's, then zero rows in one quantized matmul, down's call leaving the
    inject slice for inject's call on the same input."""

    def __init__(self, fused, n, k):
        self.fused, self.n, self.k = fused, n, k
        self._held = None

    def down(self, x):
        y = self.fused(x)
        self._held = (x, y[..., self.n:self.n + self.k])
        return y[..., :self.n]

    def inject(self, x):
        held, self._held = self._held, None
        if held is not None and held[0] is x:
            return held[1]
        return self.fused(x)[..., self.n:self.n + self.k]


def install_fuse_hc(model):
    """Stack every hyper-connection's inject rows under its down matrix, the originals leaving the
    tree, and return how many were stacked."""
    if not HC_ON:
        return 0
    hcs = [m for _, m in model.named_modules()
           if isinstance(m.get("input_mix_weight_down"), nn.QuantizedLinear)
           and isinstance(m.get("block_inject_weight"), nn.QuantizedLinear)]
    for m in hcs:
        down, inj = m.input_mix_weight_down, m.block_inject_weight
        dim = down.weight.shape[1] * 32 // down.bits
        if not (_fusable([down, inj], dim) and down.mode == inj.mode
                and ("biases" in down) == ("biases" in inj)):
            return 0
    for m in hcs:
        down, inj = m.input_mix_weight_down, m.block_inject_weight
        n, k = down.weight.shape[0], inj.weight.shape[0]
        pad = -(n + k) % 8
        fused = nn.QuantizedLinear(down.weight.shape[1] * 32 // down.bits, n + k + pad,
                                   bias=False, group_size=down.group_size, bits=down.bits,
                                   mode=down.mode)
        fused.pop("biases", None)
        for f in ("weight", "scales", "biases"):
            if f in down:
                a, b = down[f], inj[f]
                z = [mx.zeros((pad,) + a.shape[1:], dtype=a.dtype)] if pad else []
                setattr(fused, f, mx.concatenate([a, b] + z, axis=0))
        mx.eval(fused.parameters())
        hc = _HCFused(fused, n, k)
        m.fused_hc = fused                  # the weights, in the tree
        m.input_mix_weight_down = hc.down   # not Modules: the originals leave it
        m.block_inject_weight = hc.inject
    return len(hcs)


# Projections a block calls on the same input, stacked in this order (stream_fast._FUSE_PROJ, CUDA only).
_GROUPS = (
    ("in_proj_qkvz", "in_proj_ba"),                             # Qwen3-Next GatedDeltaNet
    ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"),     # Qwen3.5 / 3.6 GatedDeltaNet
    ("q_proj", "k_proj", "v_proj"),                             # attention
    ("gate_proj", "up_proj"),                                   # a dense MLP or MoE shared expert
)
# MLX CUDA's qmv runs one warp a row whatever the row count, and takes calls of fewer than 8 rows.
QMV_ROWS = 8


class _Stack:
    """The stacked rows' one matmul and its split, computed once a distinct input."""

    def __init__(self, fused, cuts):
        self.fused, self.cuts = fused, cuts
        self._x = self._parts = None

    def part(self, x, i):
        if self._x is not x:
            self._x, self._parts = x, mx.split(self.fused(x), self.cuts, axis=-1)
        return self._parts[i]


class _Stacked(nn.QuantizedLinear):
    """A projection whose weights are row views of its stack: qmv-wide calls read its slice of the stack's
    matmul (the same bits, a warp a row on CUDA), wider calls its own rows as before."""

    def __call__(self, x):
        if x.size // x.shape[-1] < QMV_ROWS:
            return self._stack.part(x, self._at)
        return super().__call__(x)


def _stack(block, names):
    parts = [block[n] for n in names]
    dim = parts[0].weight.shape[1] * 32 // parts[0].bits
    if not (_fusable(parts, dim) and len({p.mode for p in parts}) == 1
            and len({"biases" in p for p in parts}) == 1):
        return 0
    outs = [p.weight.shape[0] for p in parts]
    # Each part's view starts 16-byte aligned in every array, as a vectorized kernel reads it.
    if any(sum(outs[:i]) * p[f].shape[1] * p[f].itemsize % 16
           for i, p in enumerate(parts) for f in ("weight", "scales", "biases") if f in p):
        return 0
    fused = nn.QuantizedLinear(dim, sum(outs), bias=False, group_size=parts[0].group_size,
                               bits=parts[0].bits, mode=parts[0].mode)
    fused.pop("biases", None)
    for f in ("weight", "scales", "biases"):
        if f in parts[0]:
            setattr(fused, f, mx.concatenate([p[f] for p in parts], axis=0))
    mx.eval(fused.parameters())
    stack = _Stack(fused, [sum(outs[:i + 1]) for i in range(len(outs) - 1)])
    lo = 0
    for i, (p, o) in enumerate(zip(parts, outs)):
        for f in ("weight", "scales", "biases"):
            if f in p:
                setattr(p, f, fused[f][lo:lo + o])     # a row slice shares the stack's buffer
        mx.eval(p.parameters())
        p.__class__ = _Stacked
        p._stack, p._at = stack, i                      # plain attributes: outside the parameter tree
        lo += o
    return 1


def install_fuse_proj(model):
    """Stack every group of projections this model's blocks call on one input and return how many; the
    projections stay in the tree as modules over row views, so name-based hooks still find them."""
    n = 0
    for _, m in model.named_modules():
        for names in _GROUPS:
            if all(isinstance(m.get(x), nn.QuantizedLinear) and type(m[x]) is nn.QuantizedLinear
                   for x in names):
                n += _stack(m, names)
    return n


def _selftest_hc():
    """A GatedResidual's down / inject pair at Flash-Next's shapes, 10240 to 320 and to 4 at 4-bit
    g32: down bit-equal at T=1, inject within a bf16 ulp, the originals out of the tree."""

    class HC(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_mix_weight_down = nn.Linear(10240, 320, bias=False)
            self.block_inject_weight = nn.Linear(10240, 4, bias=False)

        def __call__(self, x):
            return self.input_mix_weight_down(x), self.block_inject_weight(x)

    mx.random.seed(0)
    hc = HC()
    nn.quantize(hc, group_size=32, bits=4)
    mx.eval(hc.parameters())
    xs = [mx.random.normal((1, t, 10240)).astype(mx.bfloat16) for t in (1, 1, 7)]
    ref = [hc(x) for x in xs]
    mx.eval(ref)
    assert install_fuse_hc(hc) == 1
    names = {k for k, _ in nn.utils.tree_flatten(hc.parameters())}
    assert names == {"fused_hc.weight", "fused_hc.scales", "fused_hc.biases"}, names
    assert hc.fused_hc.weight.shape[0] == 328
    for i, (x, (d0, j0)) in enumerate(zip(xs, ref)):
        d1, j1 = hc(x)
        if x.shape[1] == 1:
            assert mx.array_equal(d0, d1), f"input {i}: down's rows moved"
        tol = mx.maximum(mx.abs(j0), 1.0) * 2.0 ** -7
        assert mx.all(mx.abs(j1 - j0) <= tol).item(), f"input {i}: inject off by more than an ulp"
    other = xs[2] + 1
    want = hc.fused_hc(other)[..., 320:324]
    assert mx.array_equal(hc.block_inject_weight(other), want), "inject on a fresh input"
    print("ok: hyper-connection inject stacked under down (328 rows), down bit-equal at T=1")


def _selftest_proj():
    """The 35B's GatedDeltaNet inputs (2048 to 8192 / 4096 / 32 / 32, 4-bit g64): one stack, the
    projections still in the tree as row views of it, 1-7 rows through the stack and 8+ through their own
    rows; bit-equal on CUDA (a warp a row) and on the CPU, within a bf16 ulp elsewhere."""

    class GDN(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_proj_qkv = nn.Linear(2048, 8192, bias=False)
            self.in_proj_z = nn.Linear(2048, 4096, bias=False)
            self.in_proj_b = nn.Linear(2048, 32, bias=False)
            self.in_proj_a = nn.Linear(2048, 32, bias=False)

        def __call__(self, x):
            return [self.in_proj_qkv(x), self.in_proj_z(x), self.in_proj_b(x), self.in_proj_a(x)]

    mx.random.seed(1)
    m = GDN()
    m.set_dtype(mx.bfloat16)
    nn.quantize(m, group_size=64, bits=4)
    mx.eval(m.parameters())
    xs = [mx.random.normal((1, t, 2048)).astype(mx.bfloat16) for t in (1, 3, 8, 20)]
    ref = [m(x) for x in xs]
    mx.eval(ref)
    names = sorted(k for k, _ in nn.utils.tree_flatten(m.parameters()))
    assert install_fuse_proj(m) == 1 and install_fuse_proj(m) == 0
    assert sorted(k for k, _ in nn.utils.tree_flatten(m.parameters())) == names, "the tree changed"
    exact = mx.default_device() == mx.cpu or not mx.metal.is_available()
    for x, r in zip(xs, ref):
        got = m(x)
        for a, b in zip(r, got):
            if exact or x.shape[1] >= QMV_ROWS:
                assert mx.array_equal(a, b), f"{x.shape[1]} rows: bits moved"
            else:
                assert mx.all(mx.abs(a - b) <= mx.maximum(mx.abs(a), 1.0) * 2.0 ** -7).item()
    st = m.in_proj_z._stack
    assert m.in_proj_qkv._stack is st and st.fused.weight.shape == (12352, 256)
    y = m(xs[0])
    assert st._x is xs[0] and y[2] is st._parts[2], "one matmul an input"
    print(f"ok: 35B GatedDeltaNet inputs stacked (12352 rows), tree unchanged, "
          f"{'bit-equal' if exact else 'bit-equal at 8+ rows, an ulp below'} on {mx.default_device()}")


if __name__ == "__main__":
    _selftest_hc()
    _selftest_proj()
