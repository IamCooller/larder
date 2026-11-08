#!/usr/bin/env python3
"""MOE_CPU_GUESS's arithmetic (native/cpu_guess.h) against numpy and MLX, no model, any Mac with clang++:

  1. logits8: the 8-bit affine router GEMV on Flash-Next's router shape (512 x 2560, group 64, bf16 scales) and an
     fp32 toy one, against a float64 numpy reference built from MLX's own quantize, and against mx.quantized_matmul
     on the GPU (the staging guess's matvec), with the top-10 sets compared; its time a call.
  2. pick: route()'s selection on fp32 logits (softmax, bias, top k, budget, best first) against
     fused_router.route_logits(prev=...), the GPU guess kernel, at bias 0 and 0.5, budget 0 and 0.2, ties included.

    perl -e 'alarm shift; exec @ARGV' 280 python3 tests/test_cpu_guess.py
"""
import ctypes
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SHIM = r"""
#include "cpu_guess.h"
extern "C" void logits8(const uint8_t* w, const void* s, const void* b, int bf, const float* x, int E, int D,
                        int gs, float* out, int x16) { cpu_guess::logits8(w, s, b, bf, x, E, D, gs, out, x16); }
extern "C" void pick(const float* lg, const uint8_t* mask, int E, int k, float lam, float tau, int32_t* out) {
  cpu_guess::pick(lg, mask, E, k, lam, tau, out);
}
"""


def build():
    d = Path(tempfile.mkdtemp())
    (d / "shim.cpp").write_text(SHIM)
    lib = d / "libcg.dylib"
    # moe_native's Release flags (CMake: -O3 -DNDEBUG), C++20 as there; an older SDK where the default one's
    # stubs do not link (the laptop's Command Line Tools on macOS 27).
    cmd = ["clang++", "-O3", "-DNDEBUG", "-std=c++20", "-shared", "-fPIC", "-I", str(ROOT / "native"),
           str(d / "shim.cpp"), "-o", str(lib)]
    sdks = sorted(Path("/Library/Developer/CommandLineTools/SDKs").glob("MacOSX[0-9]*.*.sdk"), reverse=True)
    for extra in [[]] + [["-isysroot", str(s)] for s in sdks]:
        if subprocess.run(cmd + extra, capture_output=True).returncode == 0:
            break
    else:
        subprocess.run(cmd, check=True)
    cg = ctypes.CDLL(str(lib))
    p = ctypes.c_void_p
    cg.logits8.argtypes = [p, p, p, ctypes.c_int, p, ctypes.c_int, ctypes.c_int, ctypes.c_int, p, ctypes.c_int]
    cg.pick.argtypes = [p, p, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, p]
    return cg


def ptr(a):
    return a.ctypes.data_as(ctypes.c_void_p)


def cpu_logits(cg, wq, sc, bi, x, gs):
    """cpu_guess::logits8 on MLX's own arrays' bytes: wq uint32, scales/biases bf16 or fp32, x fp32."""
    import mlx.core as mx
    bf = sc.dtype == mx.bfloat16
    E, D = wq.shape[0], wq.shape[1] * 4
    w = np.ascontiguousarray(np.array(wq).view(np.uint8))
    s = np.ascontiguousarray(np.array(sc.view(mx.uint16) if bf else sc))
    b = np.ascontiguousarray(np.array(bi.view(mx.uint16) if bf else bi))
    xf = np.ascontiguousarray(np.array(x.astype(mx.float32)).reshape(-1))
    out = np.empty(E, np.float32)
    # x16 as the agent sets it: the MoE input came in bf16.
    cg.logits8(ptr(w), ptr(s), ptr(b), int(bf), ptr(xf), E, D, gs, ptr(out), int(x.dtype == mx.bfloat16))
    return out, (w, s, b, xf, bf, E, D)


def check_logits(cg):
    import mlx.core as mx
    rng = np.random.default_rng(0)
    for E, D, gs, dt, n in ((512, 2560, 64, mx.bfloat16, 40), (32, 64, 64, mx.float32, 40)):
        worst, diff_sets, diff_order = 0.0, 0, 0
        for t in range(n):
            w = mx.array(rng.standard_normal((E, D)).astype(np.float32) * 0.02).astype(dt)
            wq, sc, bi = mx.quantize(w, group_size=gs, bits=8)
            x = mx.array(rng.standard_normal((1, 1, D)).astype(np.float32) * 3).astype(dt)
            got, (wb, s, b, xf, bf, _, _) = cpu_logits(cg, wq, sc, bi, x, gs)
            # The bytes read as MLX dequantizes them (scale * q + bias, in float64 here).
            f = (lambda a: (a.astype(np.uint32) << 16).view(np.float32)) if bf else (lambda a: a)
            deq = (f(s).astype(np.float64).repeat(gs, axis=1) * wb.astype(np.float64)
                   + f(b).astype(np.float64).repeat(gs, axis=1))
            ref = deq @ xf.astype(np.float64)
            mxdeq = np.array(mx.dequantize(wq, sc, bi, group_size=gs, bits=8).astype(mx.float32), dtype=np.float64)
            assert np.allclose(mxdeq, deq, rtol=1e-2, atol=1e-6), "byte order or scale layout differs from MLX's"
            rel = float(np.max(np.abs(got - ref)) / np.max(np.abs(ref)))
            assert rel < 1e-5, (E, D, t, rel)
            gpu = np.array(mx.quantized_matmul(x.astype(mx.float32), wq, sc, bi, transpose=True,
                                               group_size=gs, bits=8)).reshape(-1)
            worst = max(worst, float(np.max(np.abs(got - gpu))))
            k = 10 if E > 32 else 4
            a, g = np.argsort(-got, kind="stable")[:k], np.argsort(-gpu, kind="stable")[:k]
            diff_sets += set(a.tolist()) != set(g.tolist())
            diff_order += a.tolist() != g.tolist()
        print(f"1. logits8 {E}x{D} g{gs} {dt}: |cpu - float64| < 1e-5 rel, {n} draws; |cpu - MLX GPU qmv| max "
              f"{worst:.2e}; top-{k} set differs in {diff_sets}/{n}, order in {diff_order}/{n}")
    # Its time a call on Flash-Next's shape, this machine's core (no other load).
    w = mx.array(rng.standard_normal((512, 2560)).astype(np.float32) * 0.02).astype(mx.bfloat16)
    wq, sc, bi = mx.quantize(w, group_size=64, bits=8)
    x = mx.array(rng.standard_normal((1, 1, 2560)).astype(np.float32)).astype(mx.bfloat16)
    _, (wb, s, b, xf, bf, E, D) = cpu_logits(cg, wq, sc, bi, x, 64)
    out = np.empty(E, np.float32)
    for x16 in (0, 1):
        ts = []
        for _ in range(400):
            t = time.perf_counter_ns()
            cg.logits8(ptr(wb), ptr(s), ptr(b), 1, ptr(xf), E, D, 64, ptr(out), x16)
            ts.append(time.perf_counter_ns() - t)
        ts.sort()
        print(f"   logits8 512x2560 on one core, {'fp16 dots' if x16 else 'fp32 loop'}: median "
              f"{ts[200] / 1e3:.1f} us, p90 {ts[360] / 1e3:.1f} us (ctypes call included)")


def check_pick(cg):
    import mlx.core as mx
    import fused_router
    rng = np.random.default_rng(1)
    E, k, n = 512, 10, 400
    for lam, tau in ((0.0, 0.0), (0.5, 0.0), (0.5, 0.2)):
        bad = 0
        for t in range(n):
            lg = (rng.standard_normal(E) * 2).astype(np.float32)
            if t % 4 == 1:
                lg[rng.integers(0, E, 40)] = lg.max()          # ties at the top
            if t % 4 == 2:
                lg = np.round(lg * 4) / 4                       # ties everywhere (bf16-like logits)
            mask = (rng.random(E) < 0.1).astype(np.uint8)
            got = np.empty(k, np.int32)
            cg.pick(ptr(lg), ptr(mask), E, k, lam, tau, ptr(got))
            prev = mx.zeros((1, 1, k), mx.int32)
            ref = np.array(fused_router.route_logits(
                mx.array(lg).reshape(1, 1, E), mx.array(mask.astype(np.float32)), lam, tau, k,
                prev=prev)).reshape(-1)[k:]
            if got.tolist() != ref.tolist():
                bad += 1
                if bad <= 3:
                    print(f"   lam {lam} tau {tau} draw {t}: cpu {got.tolist()} gpu {ref.tolist()}")
        print(f"2. pick lam {lam} tau {tau}: {n - bad}/{n} draws give the GPU guess kernel's ids in its order")
        assert bad <= n // 100, f"pick differs from route_logits in {bad}/{n} draws"


def main():
    if subprocess.run(["which", "clang++"], capture_output=True).returncode:
        print("SKIP: no clang++")
        return
    cg = build()
    check_logits(cg)
    check_pick(cg)
    print("OK")


if __name__ == "__main__":
    main()
