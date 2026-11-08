"""ane_experts' GPU kernels (no ANE): its prep, transposed dequant, middle and output around fp32 GEMMs that stand in for
the ANE programs give the dense path's expert outputs to fp16 precision, the bound's scale included.
    python3 tests/test_ane_experts.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import mlx.core as mx

import ane_experts as AE
import exl3_experts as X

HID, INTER, NHW, WIN = 512, 256, 48, 14


class _P:
    """A stand-in for one ANE program's surfaces: x rows then weight rows in, y out."""

    def __init__(self, B, R, K, N):
        self.B, self.R, self.K, self.N = B, R, K, N
        self.x = mx.zeros((B * R * K,), mx.float16)
        self.w = mx.zeros((B * N * K,), mx.float16)

    def run(self, toks):
        mx.eval(toks)
        x = self.x.reshape(self.B, self.R, self.K).astype(mx.float32)
        w = self.w.reshape(self.B, self.N, self.K).astype(mx.float32)
        return (x @ w.transpose(0, 2, 1)).astype(mx.float16).reshape(-1)


def run(x_scale, bound_log2=AE.BOUND_LOG2):
    rng = np.random.default_rng(0)
    G, E, R = 6, 3, 64
    w = []
    for din, dout in ((HID, INTER), (HID, INTER), (INTER, HID)):
        w.append((mx.array(rng.integers(0, 1 << 16, (G, din // 16, dout // 16, NHW), dtype=np.uint16)),
                  mx.array((rng.uniform(0.5, 1.5, (G, din)) * din ** -0.5).astype(np.float16)),
                  mx.array(rng.uniform(0.5, 1.5, (G, dout)).astype(np.float16))))
    T = 200
    x = (mx.array(rng.standard_normal((T, HID)).astype(np.float32)) * x_scale).astype(mx.bfloat16)
    r = np.sort(rng.choice([0, 2, 3, 5], 150, p=[0.3, 0.3, 0.2, 0.2])).astype(np.uint32)
    sel = rng.integers(0, T, len(r))
    ref = X.dense(x[mx.array(sel.astype(np.int32))], w, r, WIN, mx.bfloat16)

    cnt = np.bincount(r, minlength=G)
    a = AE._takes(cnt, E, R)
    left = np.setdiff1d(np.flatnonzero(cnt), a)
    assert len(a) == E and cnt[a].min() >= cnt[left].max(), (a, cnt)      # the E largest of the four
    ia = np.flatnonzero(np.isin(r, a))
    ra = r[ia]
    lut = np.zeros(G, np.int64)
    lut[a] = np.arange(len(a))
    j = np.arange(len(ra)) - np.searchsorted(ra, ra)
    arow = mx.array((lut[ra] * R + j).astype(np.int32))
    srow = mx.array(lut[ra].astype(np.uint32))
    slots = mx.array(a.astype(np.uint32))
    (tg, sug, svg), (tu, suu, svu), (td, sud, svd) = w
    sug, suu, svg, svu, sud, svd = (v[slots] for v in (sug, suu, svg, svu, sud, svd))
    qb = mx.array([AE.qmax(WIN), float(2 ** bound_log2)], dtype=mx.float32)
    one = mx.ones((1,), mx.float32)
    gu, dp = _P(2 * E, R, HID, INTER), _P(E, R, INTER, HID)

    class _S:
        win = WIN
    lay = AE.Layer.__new__(AE.Layer)
    lay.ps = _S()
    inv, tok = AE._kern("ane_exp_prep", ["x", "xrow", "srow", "arow", "sug", "suu", "qb", "dst", "dep"], ["inv", "tok"],
                        AE.PREP_SRC)(
        inputs=[x, mx.array(sel[ia].astype(np.int32)), srow, arow, sug, suu, qb, gu.x, one],
        template=[("DIM", HID), ("NB", HID // 128), ("XOFF", E * R * HID)],
        grid=(len(ia) * HID // 4, 1, 1), threadgroup=(HID // 4, 1, 1),
        output_shapes=[(len(ia), 2), (1,)], output_dtypes=[mx.float32, mx.int32])
    y = gu.run([tok, lay._deq(tg, slots, gu.w, 0, one), lay._deq(tu, slots, gu.w, E, one)])
    # The transposed dequant is dequant()'s Q, [out, in] an expert.
    q = X.dequant(tg, slots, WIN, mx.float16)
    assert mx.array_equal(gu.w.reshape(2 * E, INTER, HID)[:E], q.transpose(0, 2, 1)).item()
    isd, tok = AE._kern("ane_exp_mid", ["y", "isc", "arow", "srow", "svg", "svu", "sud", "qb", "dst"], ["inv", "tok"],
                        AE.MID_SRC)(
        inputs=[y, inv, arow, srow, svg, svu, sud, qb, dp.x],
        template=[("DIM", INTER), ("NB", INTER // 128), ("YOFF", E * R * INTER)],
        grid=(len(ia) * INTER // 4, 1, 1), threadgroup=(INTER // 4, 1, 1),
        output_shapes=[(len(ia),), (1,)], output_dtypes=[mx.float32, mx.int32])
    yd = dp.run([tok, lay._deq(td, slots, dp.w, 0, one)])
    o = AE._kern("ane_exp_fin", ["y", "isd", "arow", "srow", "svd"], ["o"], AE.FIN_SRC)(
        inputs=[yd, isd, arow, srow, svd], template=[("DIM", HID), ("T", mx.bfloat16)],
        grid=(len(ia) * HID // 4, 1, 1), threadgroup=(HID // 4, 1, 1),
        output_shapes=[(len(ia), HID)], output_dtypes=[mx.bfloat16])[0]
    want = ref.reshape(-1, HID)[mx.array(ia.astype(np.int32))].astype(mx.float32)
    err = (mx.abs(o.astype(mx.float32) - want).max() / mx.abs(want).max()).item()
    scaled = int((inv[:, 0] > 1).sum().item())
    return err, scaled, len(ia)


def main():
    err, scaled, n = run(1.0)
    print(f"x ~ N(0, 1): max rel err {err:.2e}, rows scaled {scaled} of {n}")
    assert err < 2e-2 and scaled == 0, (err, scaled)
    err, scaled, n = run(1.0, 4)        # a bound of 2^4 lowers every row's scale; the outputs still match
    print(f"bound 2^4: max rel err {err:.2e}, rows scaled {scaled} of {n}")
    assert err < 2e-2 and scaled == n, (err, scaled)
    # A closed pass hands its surface bytes to the budget once (MemBudget.close), so decode's rate never counts them.
    ps = AE._Pass.__new__(AE._Pass)
    ps.bufs, ps.nbytes = [], 123
    AE._PASS = ps
    AE.close()
    assert AE._PASS is None and AE.take_bytes() == 123 and AE.take_bytes() == 0
    print("ok")


if __name__ == "__main__":
    main()
