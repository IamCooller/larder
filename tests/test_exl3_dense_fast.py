#!/usr/bin/env python3
"""The EXL3 dense path's fast kernels against today's, bit for bit, no model:

  1. E2 dequant2 (MOE_EXL3_DEQ2) writes the Q dequant() writes: f32 and f16, K3 at windows 14 / 16,
     K2 and K4, the pack's shapes (2560 x 640, 640 x 2560) and small ones, 1-8 experts.
  2. E3 each fused kernel (MOE_EXL3_FUSE) against the MLX ops it replaces (_proj's chain:
     x * suh, mx.hadamard_transform, the casts; svh * H(p)), random rows at many widths and row counts,
     bf16 / f16 / f32 out; normal, tiny and large values, zeros and signed zeros.
  3. dense() with each flag combination against dense() with none, on uneven routings of 1-24 experts
     (one and two DENSE_CHUNK chunks), rows sorted or not; E4 (MOE_EXL3_F16MM) too where its own
     first-use check passes (MLX's f32 gather_mm on the steel kernel: not on newer Apple GPUs with TF32 on).

    perl -e 'alarm shift; exec @ARGV' 280 python3 tests/test_exl3_dense_fast.py [--big]

--big adds the pack's 16k-pass call (10 experts, ~312 rows each) and needs ~1 GB of buffers.
"""
import faulthandler
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import mlx.core as mx

import exl3_experts as ex

faulthandler.dump_traceback_later(270, exit=True)
RNG = np.random.default_rng(7)


def same(a, b, what):
    mx.eval(a, b)
    assert a.dtype == b.dtype and a.shape == b.shape, (what, a.dtype, b.dtype, a.shape, b.shape)
    ua = np.array(a.view(mx.uint16) if a.dtype in (mx.bfloat16, mx.float16) else a.view(mx.uint32))
    ub = np.array(b.view(mx.uint16) if b.dtype in (mx.bfloat16, mx.float16) else b.view(mx.uint32))
    bad = int((ua != ub).sum())
    assert bad == 0, f"{what}: {bad} of {ua.size} values differ"


def trellis(E, din, dout, n):
    return mx.array(RNG.integers(0, 1 << 16, (E, din // 16, dout // 16, n), dtype=np.uint16))


def check_dequant():
    k = 0
    for din, dout, n, win in ((2560, 640, 48, 14), (640, 2560, 48, 14), (256, 128, 48, 16),
                              (128, 256, 64, 16), (256, 128, 32, 14), (128, 384, 48, 14)):
        t = trellis(9, din, dout, n)
        for c in (1, 3, 8):
            slots = mx.array(RNG.choice(9, c, replace=False).astype(np.uint32))
            for dt in (mx.float32, mx.float16):
                ex.DEQ2 = False
                a = ex.dequant(t, slots, win, dt)
                ex.DEQ2 = True
                b = ex.dequant(t, slots, win, dt)
                same(a, b, f"dequant {din}x{dout} n{n} win{win} c{c} {dt}")
                k += 1
    ex.DEQ2 = False
    print(f"1. dequant2: {k} cases bit-equal")


def rows(R, D, dtype, kind):
    if kind == "normal":
        a = RNG.standard_normal((R, D))
    elif kind == "tiny":
        a = RNG.standard_normal((R, D)) * 1e-5
    elif kind == "big":
        a = RNG.standard_normal((R, D)) * 3e3
    else:       # zeros, signed zeros and a few values
        a = np.zeros((R, D))
        a[:, ::7] = -0.0
        a[:, ::5] = RNG.standard_normal((R, len(range(0, D, 5))))
    return mx.array(a.astype(np.float32)).astype(dtype)


def check_fused():
    k = 0
    for D in (128, 640, 2560):
        for R in (1, 5, 37, 300):
            for kind in ("normal", "tiny", "big", "zeros"):
                E, c = 12, 4
                slots = mx.array(RNG.choice(E, c, replace=False).astype(np.uint32))
                loc = mx.array(np.sort(RNG.integers(0, c, R)).astype(np.uint32))
                su = [mx.array((RNG.standard_normal((E, D)) * D ** -0.5).astype(np.float16)) for _ in range(2)]
                sv = [mx.array((1 + 0.1 * RNG.standard_normal((E, D))).astype(np.float16)) for _ in range(2)]
                x = rows(R + 3, D, mx.bfloat16, kind)
                sel = mx.array(RNG.permutation(R + 3)[:R].astype(np.int32))
                # gate / up inputs
                xs = x[sel].astype(mx.float32)
                ref = [ex._had(xs * s[slots][loc]).astype(mx.float16).astype(mx.float32) for s in su]
                got = ex._rows("exl3_prep_gu", ["x", "sel", "loc", "slots", "sug", "suu"], ["xg", "xu"],
                               ex.PREP_GU_SRC, [x, sel, loc, slots, su[0], su[1]], D, [mx.float32] * 2,
                               [("O", mx.float32)])
                for i in range(2):
                    same(ref[i], got[i], f"prep_gu D{D} R{R} {kind} {i}")
                # gate / up outputs
                p = [rows(R, D, mx.float32, kind) for _ in range(2)]
                ref = [s[slots][loc].astype(mx.float32) * ex._had(q.astype(mx.float32)) for q, s in zip(p, sv)]
                got = ex._rows("exl3_fin_gu", ["pg", "pu", "loc", "slots", "svg", "svu"], ["g", "u"],
                               ex.FIN_GU_SRC, [p[0], p[1], loc, slots, sv[0], sv[1]], D, [mx.float32] * 2)
                for i in range(2):
                    same(ref[i], got[i], f"fin_gu D{D} R{R} {kind} {i}")
                # down input
                h = rows(R, D, mx.float32, kind)
                ref = ex._had(h * su[0][slots][loc]).astype(mx.float16).astype(mx.float16)
                got = ex._rows("exl3_prep_d", ["h", "loc", "slots", "su"], ["xd"], ex.PREP_D_SRC,
                               [h, loc, slots, su[0]], D, [mx.float16])[0]
                same(ref, got, f"prep_d D{D} R{R} {kind}")
                # down output, each output dtype
                pd = rows(R, D, mx.float16, kind)
                for dt in (mx.bfloat16, mx.float16, mx.float32):
                    ref = (sv[1][slots][loc].astype(mx.float32) * ex._had(pd.astype(mx.float32))).astype(dt)
                    got = ex._rows("exl3_fin_d", ["p", "loc", "slots", "sv"], ["y"], ex.FIN_D_SRC,
                                   [pd, loc, slots, sv[1]], D, [dt], [("T", dt)])[0]
                    same(ref, got, f"fin_d D{D} R{R} {kind} {dt}")
                k += 1
    print(f"2. fused kernels: {k} cases x 4 kernels bit-equal")


def pack(E, hid, inter, n=48):
    w = []
    for din, dout in ((hid, inter), (hid, inter), (inter, hid)):
        w.append((trellis(E, din, dout, n),
                  mx.array((RNG.standard_normal((E, din)) * din ** -0.5).astype(np.float16)),
                  mx.array((1 + 0.1 * RNG.standard_normal((E, dout))).astype(np.float16))))
    return w


def run_dense(x, w, host, deq2, fuse, f16=False):
    ex.DEQ2, ex.FUSE, ex.F16MM = deq2, fuse, f16
    try:
        return ex.dense(x, w, host, 14, x.dtype)
    finally:
        ex.DEQ2 = ex.FUSE = ex.F16MM = False


def check_dense(big):
    f16 = ex._f16mm_ok()
    cases = [(256, 128, 16, u, p) for u, p in ((1, 3), (3, 40), (8, 64), (9, 90), (16, 200))]
    cases += [(2560, 640, 24, u, p) for u, p in ((2, 7), (10, 400), (17, 900))]
    if big:
        cases += [(2560, 640, 32, 10, 3120)]
    k = 0
    for hid, inter, E, u, p in cases:
        w = pack(E, hid, inter)
        x = rows(p, hid, mx.bfloat16, "normal")
        ids = RNG.choice(E, u, replace=False)
        # uneven: a few experts take most rows
        host = ids[np.minimum(RNG.geometric(0.35, p) - 1, u - 1)].astype(np.uint32)
        if RNG.random() < 0.5:
            host = np.sort(host)
        ref = run_dense(x, w, host, False, False)
        arms = ((True, False, False), (False, True, False), (True, True, False))
        if f16:
            arms += ((True, True, True),)
        for deq2, fuse, f in arms:
            same(ref, run_dense(x, w, host, deq2, fuse, f),
                 f"dense {hid}x{inter} u{u} p{p} deq2={deq2} fuse={fuse} f16mm={f}")
        k += 1
    print(f"3. dense(): {k} routings x {len(arms)} flag sets bit-equal")


if __name__ == "__main__":
    print(f"MLX {mx.__version__}, {mx.device_info()['device_name']}")
    check_dequant()
    check_fused()
    check_dense("--big" in sys.argv)
    print("ok")
