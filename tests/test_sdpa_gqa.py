#!/usr/bin/env python3
"""chunkattn's grouped-head decode attention (MOE_SDPA_GQA): MLX 0.32.2's two-pass vector SDPA with one
simdgroup serving several q heads of a kv group, bit for bit.

  1. Through the attention wrapper as the engine calls it (q a transposed view, K/V strided slices of a
     cache buffer wider than the context, mask None at T=1 and "causal" above): 16 q heads over 2 kv heads x
     256, bf16 and f16, N 4096-81917, T 1-4, 2/4/8 heads a simdgroup: 0 differing elements against stock.
  2. Every call outside the envelope reaches the wrapped function untouched.
  3. The first-call check: a grouping that differs from stock in the engine returns stock's bits and turns the shape off.
  4. The identity guard: another sdpa_vector.h or MLX version, or MOE_SDPA_GQA=0, leaves the table empty;
     a grouping that differs from stock turns the shape off.
  5. The install's timing: a table for the 35B's shape within its budget, then reused from disk.

No checkpoint; needs Metal and MLX 0.32.2 (the replayed kernel's version).

    perl -e 'alarm shift; exec @ARGV' 280 python3 tests/test_sdpa_gqa.py
"""
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

import chunkattn as ca
import stream_fast as sf

if sf.BACKEND != "metal" or mx.__version__ not in sf._SDPA_MLX:
    print(f"test_sdpa_gqa: skipped (backend {sf.BACKEND}, MLX {mx.__version__})")
    sys.exit(0)

H, HK, D = 16, 2, 256
NS = (4096, 8191, 32768 + 77, 65536 + 129, 81920 - 3)
CALLS = {"n": 0}


def stock(queries, keys, values, cache, scale, mask, sinks=None):
    CALLS["n"] += 1
    return mx.fast.scaled_dot_product_attention(queries, keys, values, scale=scale, mask=mask, sinks=sinks)


SDPA = ca._wrap(stock)


class Cache:
    pass


class Bits:
    bits = 8


def force(hpt, shape=(H, HK, D), dtype=mx.bfloat16):
    ca.gqa = {(*shape, str(dtype)): {t: [(4096, hpt)] for t in ca.GQA_T}}


def make(n, t, dtype, h=H, hk=HK, d=D, b=1):
    cap = -(-n // 256) * 256 + 256
    kb = mx.random.normal((b, hk, cap, d)).astype(dtype)
    vb = mx.random.normal((b, hk, cap, d)).astype(dtype)
    q = mx.random.normal((b, t, h, d)).astype(dtype).transpose(0, 2, 1, 3)
    mx.eval(kb, vb, q)
    return q, kb[:, :, :n], vb[:, :, :n]


def test_exact():
    mx.random.seed(0)
    total = 0
    for dtype in (mx.bfloat16, mx.float16):
        for n in NS:
            for t in ca.GQA_T:
                q, k, v = make(n, t, dtype)
                mask = None if t == 1 else "causal"
                ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=D ** -0.5, mask=mask)
                for hpt in ca.GQA_HPT:
                    force(hpt, dtype=dtype)
                    ca._gqa_ok.clear()
                    SDPA(q, k, v, cache=Cache(), scale=D ** -0.5, mask=mask)    # the first call, checked against stock
                    assert ca.gqa, f"{dtype} N={n} T={t} h{hpt}: the first-call check turned the shape off"
                    CALLS["n"] = 0
                    out = SDPA(q, k, v, cache=Cache(), scale=D ** -0.5, mask=mask)
                    assert CALLS["n"] == 0, "an envelope call reached stock after its first-call check"
                    diff = int((out.view(mx.uint16) != ref.view(mx.uint16)).sum().item())
                    assert diff == 0, f"{dtype} N={n} T={t} h{hpt}: {diff} elements differ"
                    total += out.size
        mx.clear_cache()
    print(f"  exact: 0 of {total} elements differ (bf16 and f16, N {NS[0]}-{NS[-1]}, T 1-4, h2/h4/h8)")


def test_outside():
    mx.random.seed(1)
    force(4)
    q2, k, v = make(8192, 2, mx.bfloat16)
    q1 = q2[:, :, :1]
    arr = mx.ones((1, 1, 2, 8192), mx.bool_)
    q5, k5, v5 = make(8192, 5, mx.bfloat16)
    qs, ks, vs = make(1000, 1, mx.bfloat16)
    q32, k32, v32 = make(8192, 1, mx.float32)
    qd, kd, vd = make(8192, 1, mx.bfloat16, d=128)
    qg, kg, vg = make(8192, 1, mx.bfloat16, hk=H)
    qb, kb, vb = make(8192, 1, mx.bfloat16, b=2)
    sinks = mx.zeros((H,), mx.bfloat16)

    def folded(queries, keys, values, cache, scale, mask, sinks=None):
        return stock(queries, keys, values, cache, scale, mask, sinks)
    folded.foldattn = True
    cases = {
        "T=5": (SDPA, q5, k5, v5, Cache(), "causal", None),
        "T=2 mask None": (SDPA, q2, k, v, Cache(), None, None),
        "T=2 array mask": (SDPA, q2, k, v, Cache(), arr, None),
        "T=1 array mask": (SDPA, q1, k, v, Cache(), arr[:, :, :1], None),
        "sinks": (SDPA, q1, k, v, Cache(), None, sinks),
        "bits cache": (SDPA, q1, k, v, Bits(), None, None),
        "1000 keys (one-pass kernel)": (SDPA, qs, ks, vs, Cache(), None, None),
        "float32": (SDPA, q32, k32, v32, Cache(), None, None),
        "D=128": (SDPA, qd, kd, vd, Cache(), None, None),
        "no grouping": (SDPA, qg, kg, vg, Cache(), None, None),
        "batch 2": (SDPA, qb, kb, vb, Cache(), None, None),
        "T=2 under foldattn": (ca._wrap(folded), q2, k, v, Cache(), "causal", None),
    }
    for name, (fn, q, kk, vv, cache, mask, sk) in cases.items():
        CALLS["n"] = 0
        out = fn(q, kk, vv, cache=cache, scale=q.shape[3] ** -0.5, mask=mask, sinks=sk)
        mx.eval(out)
        assert CALLS["n"] == 1, f"{name}: did not reach the wrapped function"
    ca.gqa = {}
    CALLS["n"] = 0
    mx.eval(SDPA(q1, k, v, cache=Cache(), scale=D ** -0.5, mask=None))
    assert CALLS["n"] == 1, "an empty table still routed"
    print(f"  outside the envelope: {len(cases) + 1} kinds of call reach the wrapped function")


def test_first_call():
    mx.random.seed(2)
    q, k, v = make(8192, 2, mx.bfloat16)
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=D ** -0.5, mask="causal")
    force(4)
    ca._gqa_ok.clear()
    real = ca.gqa_sdpa
    ca.gqa_sdpa = lambda q, k, v, scale, hpt, parts: real(q, k, v, scale * 1.01, hpt, parts)
    try:
        out = SDPA(q, k, v, cache=Cache(), scale=D ** -0.5, mask="causal")
    finally:
        ca.gqa_sdpa = real
    assert bool(mx.array_equal(out, ref)), "a differing first call did not return stock's bits"
    assert not ca.gqa and not ca._gqa_ok, "a differing first call left the shape on"
    CALLS["n"] = 0
    mx.eval(SDPA(q, k, v, cache=Cache(), scale=D ** -0.5, mask="causal"))
    assert CALLS["n"] == 1, "the shape still routed after its first call differed"
    print("  first-call check: a differing grouping returns stock's bits and turns the shape off")


def test_guard():
    shapes = {(H, HK, D)}
    on, hdr, ver = ca.GQA_ON, sf._SDPA_HDR, sf._SDPA_MLX
    try:
        ca.GQA_ON = False
        assert ca._gqa_install(shapes, mx.bfloat16) == "stock (MOE_SDPA_GQA=0)" and not ca.gqa
        ca.GQA_ON = True
        sf._SDPA_HDR = "000000000000"
        said = ca._gqa_install(shapes, mx.bfloat16)
        assert "not the ones replayed" in said and not ca.gqa, said
        sf._SDPA_HDR, sf._SDPA_MLX = hdr, ("0.0.0",)
        said = ca._gqa_install(shapes, mx.bfloat16)
        assert "not the ones replayed" in said and not ca.gqa, said
        sf._SDPA_MLX = ver
        said = ca._gqa_install({(8, 2, 128), (16, 16, 256)}, mx.bfloat16)
        assert said.startswith("stock (no grouped") and not ca.gqa, said
        real = ca.gqa_sdpa
        ca.gqa_sdpa = lambda q, k, v, scale, hpt, parts: real(q, k, v, scale * 1.01, hpt, parts)
        try:
            assert ca._gqa_calibrate(H, HK, D, mx.bfloat16) == {}
        finally:
            ca.gqa_sdpa = real
    finally:
        ca.GQA_ON, sf._SDPA_HDR, sf._SDPA_MLX = on, hdr, ver
    print("  guard: off for MOE_SDPA_GQA=0, another header, another MLX, shapes outside, a grouping with other bits")


def test_install():
    on, ldir = ca.GQA_ON, ca.LAYOUT_DIR
    with tempfile.TemporaryDirectory() as tmp:
        try:
            ca.GQA_ON, ca.LAYOUT_DIR = True, Path(tmp)
            room = sf.prefill_room
            try:
                sf.prefill_room = lambda: 10e6
                tight = ca._gqa_install({(H, HK, D)}, mx.bfloat16)
                assert "a start with room calibrates" in tight and not ca.gqa, tight
                sf.prefill_room = lambda: 200e6
                cut = ca._gqa_install({(H, HK, D)}, mx.bfloat16)
                rows = ca.gqa[(H, HK, D, str(mx.bfloat16))]
                assert max(n for n, _ in rows[1]) < ca.GQA_NMAX, rows
            finally:
                sf.prefill_room = room
            assert not list(Path(tmp).iterdir()), "a calibration under a tight budget was kept"
            print(f"  budget: off with 10 MB of room ({tight}); a 200 MB room times a shorter grid, not kept")
            mx.synchronize()
            mx.reset_peak_memory()
            a0 = mx.get_active_memory()
            t0 = time.perf_counter()
            said = ca._gqa_install({(H, HK, D)}, mx.bfloat16)
            first = time.perf_counter() - t0
            peak = (mx.get_peak_memory() - a0) / 1e6
            rows = ca.gqa[(H, HK, D, str(mx.bfloat16))]
            assert sorted(rows) == list(ca.GQA_T) and all(len(r) == 5 for r in rows.values()), rows
            t0 = time.perf_counter()
            again = ca._gqa_install({(H, HK, D)}, mx.bfloat16)
            cached = time.perf_counter() - t0
            assert ca.gqa[(H, HK, D, str(mx.bfloat16))] == rows and len(list(Path(tmp).iterdir())) == 1
        finally:
            ca.GQA_ON, ca.LAYOUT_DIR = on, ldir
            ca.gqa = {}
    print(f"  install: {said}\n    first {first:.2f} s, peak +{peak:.0f} MB; cached {cached * 1e3:.1f} ms: {again}")
    # The first start's seconds are printed, not held to a bound: they follow the GPU (2.1 s alone, 5.8 s once inside a toy run on the same Mac).
    assert peak < 300 and cached < 0.1, (first, peak, cached)


if __name__ == "__main__":
    test_exact()
    test_outside()
    test_first_call()
    test_guard()
    test_install()
    print("test_sdpa_gqa: ok")
