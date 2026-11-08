"""cpu_kern (cpu_ext/) against a float64 numpy reference and against MLX's own CPU ops; no model, CPU device only.

    python3 tests/test_cpu_kern.py        # needs cpu_ext/build.sh run for this python

Tolerance: f32 outputs within 1e-5 of the row's sum |x| |w| (f32 accumulation in another order); bf16/f16 outputs
within one output ulp of the float64 value plus that.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cpu_ext", "build"))
import mlx.core as mx  # noqa: E402
try:
    import cpu_kern  # noqa: E402
except ImportError:
    print("test_cpu_kern: skipped (needs cpu_ext/build.sh run for this python)")
    sys.exit(0)

mx.set_default_device(mx.cpu)
cpu_kern.set_threads(2)
STOCK = {}  # MLX's own ops once cpu_kernels.install() has patched them
ULP = {mx.float32: 0.0, mx.bfloat16: 2.0 ** -8, mx.float16: 2.0 ** -11}


def dequant64(w, s, b, gs, bits):
    """[.., N, K] float64 weights from MLX's affine packing, read with numpy only."""
    w = np.array(w).view(np.uint32)
    per = 32 // bits
    q = (w[..., None] >> (np.arange(per, dtype=np.uint32) * bits)) & ((1 << bits) - 1)
    q = q.reshape(*w.shape[:-1], -1).astype(np.float64)
    s = np.array(s.astype(mx.float32)).astype(np.float64).repeat(gs, -1)
    b = np.array(b.astype(mx.float32)).astype(np.float64).repeat(gs, -1)
    return s * q + b


def check(y, x64, w64, label, strict=True):
    ref = x64 @ np.swapaxes(w64, -1, -2)
    scale = np.abs(x64) @ np.abs(np.swapaxes(w64, -1, -2))
    got = np.array(y.astype(mx.float32)).astype(np.float64)
    err = np.abs(got - ref) - (ULP[y.dtype] * np.abs(ref) + 1e-5 * scale)
    worst = float(err.max())
    assert worst <= 0 or not strict, f"{label}: off by {worst:.3g} beyond tolerance"
    return float((np.abs(got - ref) / (scale + 1e-30)).max())


def quant(n, k, gs, bits, dt, e=None, seed=0):
    rng = np.random.default_rng(seed)
    shape = (n, k) if e is None else (e, n, k)
    w = mx.array(rng.normal(0, 0.05, shape).astype(np.float32)).astype(dt)
    return mx.quantize(w, group_size=gs, bits=bits)


def test_qmm():
    rng = np.random.default_rng(1)
    stock_rel = 0.0
    for bits in (4, 8):
        for gs in (32, 64, 128):
            for xdt in (mx.float32, mx.bfloat16, mx.float16):
                for rows in ((1,), (3,), (2, 9), (1, 1, 17)):
                    n, k = 72, 512
                    w, s, b = quant(n, k, gs, bits, mx.bfloat16)
                    x = mx.array(rng.normal(0, 1, (*rows, k)).astype(np.float32)).astype(xdt)
                    y = cpu_kern.qmm(x, w, s, b, gs, bits)
                    stock = STOCK.get('quantized_matmul', mx.quantized_matmul)(x, w, s, b, transpose=True, group_size=gs, bits=bits)
                    assert y.shape == stock.shape and y.dtype == stock.dtype, (y.shape, stock.shape, y.dtype)
                    x64 = np.array(x.astype(mx.float32)).astype(np.float64)
                    rel = check(y, x64, dequant64(w, s, b, gs, bits), f"qmm bits {bits} gs {gs} {xdt} {rows}")
                    stock_rel = max(stock_rel, check(stock, x64, dequant64(w, s, b, gs, bits), "stock", False))
    # A strided x (a slice of a wider array) and f32 scales.
    w, s, b = quant(40, 256, 64, 4, mx.float32)
    xw = mx.array(rng.normal(0, 1, (5, 2 * 256)).astype(np.float32))
    x = xw[:, 128:384]
    check(cpu_kern.qmm(x, w, s, b, 64, 4), np.array(x).astype(np.float64), dequant64(w, s, b, 64, 4), "strided")
    print(f"qmm ok, last rel err {rel:.2e}; MLX's own CPU qmm worst rel err {stock_rel:.2e} (reported, not checked)")


def test_gather():
    rng = np.random.default_rng(2)
    E, n, k, gs = 16, 64, 256, 64
    w, s, b = quant(n, k, gs, 4, mx.bfloat16, e=E)
    w64 = dequant64(w, s, b, gs, 4)
    # SwitchGLU's shapes: x [B, L, 1, 1, K] with indices [B, L, top_k]; then x [B, L, top_k, 1, K'].
    for x_shape, idx_shape in (((1, 1, 1, 1, k), (1, 1, 8)), ((2, 3, 1, 1, k), (2, 3, 4)), ((1, 5, 4, 1, k), (1, 5, 4)),
                               ((7, 1, k), (7,)), ((64, 1, k), (64,))):
        x = mx.array(rng.normal(0, 1, x_shape).astype(np.float32)).astype(mx.bfloat16)
        # The last case is a sorted prefill gather, many rows an expert (the bf16 dot-product path).
        hi = 4 if x_shape[0] == 64 else E
        idx = mx.array(np.sort(rng.integers(0, hi, idx_shape)).astype(np.uint32))
        y = cpu_kern.gather_qmm(x, w, s, b, idx, gs, 4)
        stock = STOCK.get('gather_qmm', mx.gather_qmm)(x, w, s, b, rhs_indices=idx, transpose=True, group_size=gs, bits=4)
        assert y.shape == stock.shape and y.dtype == stock.dtype, (y.shape, stock.shape)
        xb = np.broadcast_to(np.array(x.astype(mx.float32)).astype(np.float64), (*y.shape[:-2], x_shape[-2], k))
        wb = w64[np.array(idx)]
        wb = np.broadcast_to(wb, (*y.shape[:-2], n, k))
        check(y, xb, wb, f"gather {x_shape} {idx_shape}")
        stock_rel = check(stock, xb, wb, "stock", False)
    print(f"gather ok; MLX's own CPU gather_qmm rel err {stock_rel:.2e} (reported, not checked)")


def test_gdn():
    """The GatedDeltaNet scan against a float64 numpy loop, the 35B's head shapes (Hk 16, Hv 32, 128 x 128)."""
    from mlx_lm.models.gated_delta import gated_delta_ops
    gated_delta_ops = STOCK.get("gated_delta_ops", gated_delta_ops)
    rng = np.random.default_rng(3)
    B, T, Hk, Hv, D = 1, 5, 4, 8, 128
    f = lambda *s: rng.normal(0, 1, s).astype(np.float32)  # noqa: E731
    q = mx.array(f(B, T, Hk, D) * 0.1).astype(mx.bfloat16)
    k = mx.array(f(B, T, Hk, D) * 0.1).astype(mx.bfloat16)
    v = mx.array(f(B, T, Hv, D)).astype(mx.bfloat16)
    g = mx.array(rng.uniform(0.5, 1.0, (B, T, Hv)).astype(np.float32))
    beta = mx.array(rng.uniform(0, 1, (B, T, Hv)).astype(np.float32)).astype(mx.bfloat16)
    s0 = mx.array(f(B, Hv, D, D))
    y, s = cpu_kern.gdn(q, k, v, g, beta, s0)
    ys, ss = gated_delta_ops(q, k, v, g, beta, s0)
    assert y.dtype == ys.dtype and y.shape == ys.shape and s.shape == ss.shape
    c = lambda a: np.array(a.astype(mx.float32)).astype(np.float64)  # noqa: E731
    Q, K, V, G, Bt, S = (c(a) for a in (q, k, v, g, beta, s0))
    Y = np.zeros((B, T, Hv, D))
    for t in range(T):
        for h in range(Hv):
            st = S[0, h] * G[0, t, h]
            kk, qq = K[0, t, h // (Hv // Hk)], Q[0, t, h // (Hv // Hk)]
            st = st + np.outer((V[0, t, h] - st @ kk) * Bt[0, t, h], kk)
            S[0, h] = st
            Y[0, t, h] = st @ qq
    s_err = float(np.abs(c(s) - S).max() / np.abs(S).max())
    assert s_err < 1e-5, s_err
    yscale = np.abs(Y).max()
    y_err = np.abs(c(y) - Y) - (2.0 ** -8 * np.abs(Y) + 1e-5 * yscale)
    assert float(y_err.max()) <= 0, float(y_err.max())
    print(f"gdn ok: state rel err {s_err:.1e}; stock state rel err {float(np.abs(c(ss) - S).max() / np.abs(S).max()):.1e}")


def test_attn():
    """Attention against float64: decode (one query, split keys), causal prefill rows, a boolean mask, strided K/V."""
    rng = np.random.default_rng(4)

    def ref(q, k, v, scale, mask):
        q, k, v = (np.array(a.astype(mx.float32)).astype(np.float64) for a in (q, k, v))
        G = q.shape[1] // k.shape[1]
        k, v = np.repeat(k, G, 1), np.repeat(v, G, 1)
        s = scale * q @ np.swapaxes(k, -1, -2)
        if mask is not None:
            s = np.where(mask, s, -np.inf)
        p = np.exp(s - s.max(-1, keepdims=True))
        return (p / p.sum(-1, keepdims=True)) @ v

    for B, Hq, Hkv, T, N, D, dt in ((1, 16, 2, 1, 700, 256, mx.bfloat16), (1, 16, 2, 9, 300, 256, mx.bfloat16),
                                    (2, 4, 4, 3, 40, 64, mx.float32), (1, 8, 1, 1, 5000, 128, mx.float16)):
        q = mx.array(rng.normal(0, 1, (B, Hq, T, D)).astype(np.float32)).astype(dt)
        # K/V as a cache slice: [B, Hkv, cap, D] cut to N keys, so the head stride is not N * D.
        kc = mx.array(rng.normal(0, 1, (B, Hkv, N + 16, D)).astype(np.float32)).astype(dt)
        vc = mx.array(rng.normal(0, 1, (B, Hkv, N + 16, D)).astype(np.float32)).astype(dt)
        k, v = kc[:, :, :N], vc[:, :, :N]
        scale = D ** -0.5
        causal = np.arange(T)[:, None] + N - T >= np.arange(N)[None, :]
        rnd = rng.random((T, N)) < 0.7
        rnd[:, 0] = True
        for kind, mask, m64 in (("none", None, None), ("causal", "causal", causal),
                                ("array", mx.array(rnd)[None, None], rnd)):
            y = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
            r = ref(q, k, v, scale, m64)
            got = np.array(y.astype(mx.float32)).astype(np.float64)
            tol = ULP[dt] * np.abs(r) + 1e-5 * np.abs(r).max()
            worst = float((np.abs(got - r) - tol).max())
            assert worst <= 0, (B, Hq, Hkv, T, N, D, dt, kind, worst)
    print("attn ok")


if __name__ == "__main__":
    import cpu_kernels
    cpu_kernels.install(2)
    STOCK.update(cpu_kernels._ORIG)
    test_qmm()
    test_gather()
    test_gdn()
    test_attn()
    print("all ok")
