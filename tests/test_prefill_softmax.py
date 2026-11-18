#!/usr/bin/env python3
"""chunkattn's fused prefill mask + softmax (MOE_PREFILL_SOFTMAX): MLX 0.32.2's unfused fallback with the causal
`where` and the precise looped softmax as one in-place pass, bit for bit, at a row width timed at install.

  1. Through the attention wrapper as the engine calls it (q a transposed view, K/V strided slices of a cache
     buffer wider than the context, mask "causal"): 16 q heads over 2 kv heads x 256, bf16 and f16, 4,160-20,000
     keys, widths 64 / 256 / the engine's own, with and without a budget cap: 0 differing elements against stock,
     and the fused pass ran.
  2. Calls outside it (an array mask, 4,096 keys or fewer, batch 2, an empty table) never reach the fused pass.
  3. The first-call check: a fused pass that differs from the stock fallback returns stock's bits and turns the
     shape off.
  4. The identity guard: MOE_PREFILL_SOFTMAX off, another softmax.h or MLX version leave the table empty; the
     install's bit check fails on a kernel that differs.
  5. The install's timing: a width table for the shape, kept on disk and then reused.

No checkpoint; needs Metal and MLX 0.32.2 (the replayed kernel's version).

    perl -e 'alarm shift; exec @ARGV' 280 python3 tests/test_prefill_softmax.py
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

import chunkattn as ca
import stream_fast as sf

if sf.BACKEND != "metal" or mx.__version__ not in ca.SM_MLX:
    print(f"test_prefill_softmax: skipped (backend {sf.BACKEND}, MLX {mx.__version__})")
    sys.exit(0)

H, HK, D = 16, 2, 256
SHAPE = (H, HK, D)
CALLS = {"stock": 0, "fused": 0}
_real_core = ca._fused_core


def stock(queries, keys, values, cache, scale, mask, sinks=None):
    CALLS["stock"] += 1
    return mx.fast.scaled_dot_product_attention(queries, keys, values, scale=scale, mask=mask, sinks=sinks)


def counted(*a, **k):
    CALLS["fused"] += 1
    return _real_core(*a, **k)


ca._fused_core = counted
SDPA = ca._wrap(stock)


class Cache:
    pass


def force(width, dtype=mx.bfloat16):
    ca.layout = {SHAPE: (37.0, 16, 4097)}
    ca.sm = {(*SHAPE, str(dtype)): [(8192, width)]}
    ca._sm_ok.clear()


def make(n, t, dtype=mx.bfloat16, b=1):
    cap = -(-n // 256) * 256 + 256
    kb = mx.random.normal((b, HK, cap, D)).astype(dtype)
    vb = mx.random.normal((b, HK, cap, D)).astype(dtype)
    q = mx.random.normal((b, t, H, D)).astype(dtype).transpose(0, 2, 1, 3)
    mx.eval(kb, vb, q)
    return q, kb[:, :, :n], vb[:, :, :n]


def ndiff(a, b):
    return int((a.view(mx.uint16) != b.view(mx.uint16)).sum().item())


def test_exact():
    mx.random.seed(0)
    cap0 = sf.prefill_cap_bytes
    total = 0
    try:
        for dtype in (mx.bfloat16, mx.float16):
            for t, n in ((300, 4160), (1000, 8191), (777, 20000)):
                q, k, v = make(n, t, dtype)
                ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=D ** -0.5, mask="causal")
                for width in (64, 256, 0):
                    for cap in (None, 300 * n * 37.0):
                        sf.prefill_cap_bytes = lambda *a, cap=cap: cap
                        force(width, dtype)
                        CALLS["fused"] = 0
                        SDPA(q, k, v, cache=Cache(), scale=D ** -0.5, mask="causal")   # the first call, checked
                        assert ca.sm, f"{dtype} T={t} N={n} w={width}: the first-call check turned the shape off"
                        CALLS["fused"] = 0
                        out = SDPA(q, k, v, cache=Cache(), scale=D ** -0.5, mask="causal")
                        assert CALLS["fused"], f"{dtype} T={t} N={n} w={width}: the fused pass did not run"
                        d = ndiff(out, ref)
                        assert d == 0, f"{dtype} T={t} N={n} w={width} cap={cap}: {d} elements differ"
                        total += out.size
                mx.clear_cache()
    finally:
        sf.prefill_cap_bytes = cap0
    print(f"  exact: 0 of {total} elements differ (bf16 and f16, N 4160-20000, widths 64/256/own, capped or not)")


def test_outside():
    mx.random.seed(1)
    force(128)
    cases = {}
    q, k, v = make(8192, 300)
    cases["array mask"] = (q, k, v, mx.ones((1, 1, 300, 8192), mx.bool_))
    cases["4096 keys"] = make(4096, 300) + ("causal",)
    cases["batch 2"] = make(8192, 300, b=2) + ("causal",)
    for name, (q, k, v, mask) in cases.items():
        CALLS["fused"] = 0
        ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=D ** -0.5, mask=mask)
        out = SDPA(q, k, v, cache=Cache(), scale=D ** -0.5, mask=mask)
        assert CALLS["fused"] == 0, f"{name}: reached the fused pass"
        assert ndiff(out, ref) == 0, f"{name}: differs from stock"
    ca.sm = {}
    CALLS["fused"] = CALLS["stock"] = 0
    q, k, v = make(8192, 300)
    mx.eval(SDPA(q, k, v, cache=Cache(), scale=D ** -0.5, mask="causal"))
    assert CALLS["fused"] == 0 and CALLS["stock"] == 1, "an empty table still routed"
    print(f"  outside: {len(cases) + 1} kinds of call never reach the fused pass")


def test_first_call():
    mx.random.seed(2)
    q, k, v = make(8192, 300)
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=D ** -0.5, mask="causal")
    force(64)
    ca._fused_core = lambda q, k, v, scale, off0: _real_core(q, k, v, scale * 1.01, off0)
    try:
        out = SDPA(q, k, v, cache=Cache(), scale=D ** -0.5, mask="causal")
    finally:
        ca._fused_core = counted
    assert ndiff(out, ref) == 0, "a differing first call did not return stock's bits"
    assert not ca.sm and not ca._sm_ok, "a differing first call left the shape on"
    CALLS["fused"] = 0
    mx.eval(SDPA(q, k, v, cache=Cache(), scale=D ** -0.5, mask="causal"))
    assert CALLS["fused"] == 0, "a shape turned off still ran fused"
    print("  first call: a differing fused pass returns stock's bits and turns the shape off")


def test_guard():
    table = {SHAPE: (37.0, 16, 4097)}
    on, hdr = ca.SM_ON, ca._softmax_hdr
    try:
        ca.SM_ON = False
        said = ca._sm_install(table, mx.bfloat16)
        assert not ca.sm and "MOE_PREFILL_SOFTMAX=0" in said, said
        ca.SM_ON = True
        ca._softmax_hdr = lambda: "000000000000"
        said = ca._sm_install(table, mx.bfloat16)
        assert not ca.sm and "not the ones replayed" in said, said
    finally:
        ca.SM_ON, ca._softmax_hdr = on, hdr
    assert ca._sm_exact(H, HK, D, mx.bfloat16, 16, [4097]), "the install's bit check failed on the real kernel"
    ca._fused_core = lambda q, k, v, scale, off0: _real_core(q, k, v, scale * 1.01, off0)
    try:
        assert not ca._sm_exact(H, HK, D, mx.bfloat16, 16, [4097]), "the bit check passed a kernel that differs"
    finally:
        ca._fused_core = counted
    print("  guard: off, another softmax.h -> stock; the bit check passes the kernel and fails a wrong one")


def test_install():
    saved = ca.SM_ON, ca.SM_N, ca.SM_T, ca.SM_ROWS, ca.LAYOUT_DIR, ca._sm_calibrate
    cap0 = sf.prefill_cap_bytes
    with tempfile.TemporaryDirectory() as tmp:
        try:
            ca.SM_ON, ca.SM_N, ca.SM_T, ca.SM_ROWS = True, (8192,), 512, (64, 128)
            ca.LAYOUT_DIR = Path(tmp)
            sf.prefill_cap_bytes = lambda *a: 300 * 8192 * 37.0     # the engine's own: 300 rows at 8192 keys
            said = ca._sm_install({SHAPE: (37.0, 16, 4097)}, mx.bfloat16)
            rows = ca.sm.get((*SHAPE, "mlx.core.bfloat16"))
            assert rows and rows[0][0] == 8192 and rows[0][1] in (0, 64, 128), (said, rows)
            assert len(list(Path(tmp).glob("smwidth-*.json"))) == 1, "the timed table was not kept"

            def boom(*a):
                raise AssertionError("timed again instead of reusing the kept table")
            ca._sm_calibrate = boom
            ca._sm_install({SHAPE: (37.0, 16, 4097)}, mx.bfloat16)
            assert ca.sm.get((*SHAPE, "mlx.core.bfloat16")) == rows, "the kept table was not reused"
        finally:
            ca.SM_ON, ca.SM_N, ca.SM_T, ca.SM_ROWS, ca.LAYOUT_DIR, ca._sm_calibrate = saved
            sf.prefill_cap_bytes = cap0
            ca.sm = {}
    print(f"  install: {said}; kept and reused")


if __name__ == "__main__":
    test_exact()
    test_outside()
    test_first_call()
    test_guard()
    test_install()
    print("test_prefill_softmax: ok")
