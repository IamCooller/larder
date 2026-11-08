"""MOE_ANE_EXPERTS=1 (off by default, Metal): in a layer-major prefill pass the Neural Engine computes the EXL3 experts' GEMMs for each
group's largest experts a group ahead of the GPU, which runs the rest of the group meanwhile; not bit-exact."""
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

import mlx.core as mx

import ane_split as A
import exl3_experts as X
from engine_log import note

ON = os.environ.get("MOE_ANE_EXPERTS", "0") == "1"
BOUND_LOG2 = 14         # each ANE output and partial sum stays under 2^14, the ANE's fp16 range is 2^15
STATS = {"layers": 0, "groups": 0, "ane_groups": 0, "ane_rows": 0, "gpu_rows": 0, "scaled_rows": 0, "plans": {}}
_V = 1                  # calibration format
_CAL = None             # {"ane": {"E.R": ms of gate/up + down}, "gemm_us_row": GPU GEMM us a row}
_QMAX = {}              # win -> max |decoded weight| of the MCG codebook
_PASS = None            # this pass's plan and programs (closed at the pass's end)
_SIDE = 0               # the closed pass's surface bytes, until MemBudget.close takes them
E_CAND, R_CAND = (2, 4, 6), (128, 256, 384, 512, 768)

_PRE = r"""
const uint lane = uint(thread_index_in_simdgroup);
const uint sg = uint(simdgroup_index_in_threadgroup);
const uint r = uint(threadgroup_position_in_grid.x);
const uint base = sg * 128u;
const float sc = 0.08838834764831845f;
float4 v;
float s0, s1, s2, s3;
"""
_BF = X._BFLY + r"""
  v = float4((s0 + s2) * sc, (s1 + s3) * sc, (s0 - s2) * sc, (s1 - s3) * sc);
"""
# The row's power of two: the bound l1 x QMAX on every output and partial sum kept under 2^BOUND.
_SCALE = r"""
  float tot = 0.0f;
  for (uint i = 0u; i < NB; i++) tot += red[i];
  const float bnd = tot * qb[0] * 1.00390625f;
  const float s = bnd > qb[1] ? metal::exp2(metal::floor(metal::log2(qb[1] / bnd))) : 1.0f;
"""

# Gate and up inputs of the ANE's rows: x[xrow] times suh, transformed, rounded to f16 as the GPU's prep does, then times the
# row's power of two into the surface (gate at arow, up E*R rows further); one threadgroup a row, one simdgroup a 128-block.
PREP_SRC = _PRE + r"""
const size_t xb = (size_t)uint(xrow[r]) * DIM + base;
const size_t sb = (size_t)uint(srow[r]) * DIM + base;
const float4 xv = float4(float(x[xb + lane]), float(x[xb + lane + 32u]), float(x[xb + lane + 64u]), float(x[xb + lane + 96u]));
threadgroup float red[NB];
device half *d = (device half*)dst;
for (uint pj = 0u; pj < 2u; pj++) {
  const device half *su = pj == 0u ? sug : suu;
  v = xv * float4(float(su[sb + lane]), float(su[sb + lane + 32u]), float(su[sb + lane + 64u]), float(su[sb + lane + 96u]));
""" + _BF + r"""
  const half4 h = half4(v);
  float l1 = simd_sum(metal::abs(float(h.x)) + metal::abs(float(h.y)) + metal::abs(float(h.z)) + metal::abs(float(h.w)));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0u) red[sg] = l1;
  threadgroup_barrier(mem_flags::mem_threadgroup);
""" + _SCALE + r"""
  const size_t ob = (size_t)(pj == 0u ? 0u : XOFF) + (size_t)uint(arow[r]) * DIM + base;
  d[ob + lane] = half(float(h.x) * s);
  d[ob + lane + 32u] = half(float(h.y) * s);
  d[ob + lane + 64u] = half(float(h.z) * s);
  d[ob + lane + 96u] = half(float(h.w) * s);
  if (sg == 0u && lane == 0u) inv[r * 2u + pj] = 1.0f / s;
}
if (r == 0u && sg == 0u && lane == 0u) tok[0] = 1;
"""

# dequant()'s DEQ2 kernel with its 16 x 128 tile stored transposed, [out, in] an expert, at batch BOFF + c of the surface.
DEQT_SRC = X.DEQ2_SRC.split("const uint orow")[0] + r"""
const uint ocol = tid >> 1u;
const uint orow = (tid & 1u) * 8u;
device half *d = (device half*)dst + ((size_t)(BOFF + c) * ODIM + og * 128u + ocol) * IDIM + tk * 16u + orow;
for (uint k = 0u; k < 8u; k++) d[k] = half(tile[(orow + k) * 128u + ocol]);
if (tid == 0u && tg == 0u) tok[0] = 1;
"""

# Between the ANE's two programs: gate and up out of the surface, unscaled, transformed, times svh (the GPU's FIN_GU),
# g sigmoid(g) u, then the down input as the GPU's PREP_D, its row's power of two, into the down surface.
MID_SRC = _PRE + r"""
const size_t sb = (size_t)uint(srow[r]) * DIM + base;
const size_t yb = (size_t)uint(arow[r]) * DIM + base;
threadgroup float red[NB];
float4 gu[2];
for (uint pj = 0u; pj < 2u; pj++) {
  const size_t o = yb + (pj == 0u ? 0u : YOFF);
  const float iv = isc[r * 2u + pj];
  v = float4(float(y[o + lane]), float(y[o + lane + 32u]), float(y[o + lane + 64u]), float(y[o + lane + 96u])) * iv;
""" + _BF + r"""
  const device half *sv = pj == 0u ? svg : svu;
  gu[pj] = v * float4(float(sv[sb + lane]), float(sv[sb + lane + 32u]), float(sv[sb + lane + 64u]), float(sv[sb + lane + 96u]));
}
const float4 g = gu[0];
v = g * (1.0f / (1.0f + metal::exp(-g))) * gu[1];
v = v * float4(float(sud[sb + lane]), float(sud[sb + lane + 32u]), float(sud[sb + lane + 64u]), float(sud[sb + lane + 96u]));
""" + _BF + r"""
const half4 h = half4(v);
float l1 = simd_sum(metal::abs(float(h.x)) + metal::abs(float(h.y)) + metal::abs(float(h.z)) + metal::abs(float(h.w)));
if (lane == 0u) red[sg] = l1;
threadgroup_barrier(mem_flags::mem_threadgroup);
""" + _SCALE + r"""
device half *d = (device half*)dst + yb;
d[lane] = half(float(h.x) * s);
d[lane + 32u] = half(float(h.y) * s);
d[lane + 64u] = half(float(h.z) * s);
d[lane + 96u] = half(float(h.w) * s);
if (sg == 0u && lane == 0u) inv[r] = 1.0f / s;
if (r == 0u && sg == 0u && lane == 0u) tok[0] = 1;
"""

# The down output: out of the surface, unscaled, transformed, times svh, in the output dtype (the GPU's FIN_D), a row of out each.
FIN_SRC = _PRE + r"""
const size_t sb = (size_t)uint(srow[r]) * DIM + base;
const size_t yb = (size_t)uint(arow[r]) * DIM + base;
const float iv = isd[r];
v = float4(float(y[yb + lane]), float(y[yb + lane + 32u]), float(y[yb + lane + 64u]), float(y[yb + lane + 96u])) * iv;
""" + _BF + r"""
const size_t ob = (size_t)r * DIM + base;
o[ob + lane] = T(v.x * float(svd[sb + lane]));
o[ob + lane + 32u] = T(v.y * float(svd[sb + lane + 32u]));
o[ob + lane + 64u] = T(v.z * float(svd[sb + lane + 64u]));
o[ob + lane + 96u] = T(v.w * float(svd[sb + lane + 96u]));
"""

_K = {}


def _kern(name, ins, outs, src):
    if name not in _K:
        _K[name] = mx.fast.metal_kernel(name=name, input_names=ins, output_names=outs, source=src, header=X.HEADER)
    return _K[name]


def qmax(win):
    """The largest |weight| the MCG codebook decodes at this window: every 16-bit codeword through exl3_pairh in numpy."""
    if win not in _QMAX:
        cw = np.arange(1 << 16, dtype=np.uint64)
        if win < 16:
            cw &= (1 << win) - 1
        r = ((cw * 0xCBAC1FED) & 0x8FFF8FFF) ^ 0x3B603B60
        h = r.astype(np.uint32).view(np.float16).reshape(-1, 2)
        _QMAX[win] = float(np.abs(h[:, 0] + h[:, 1]).max())
    return _QMAX[win]


class _Buf:
    """One stage's two ANE programs (gate + up as 2E products, down as E) with their surfaces, and the work in flight on them."""

    def __init__(self, E, R, hid, inter):
        self.gu = A._Prog(R, inter, hid * 2 * E, 2 * E)
        self.d = A._Prog(R, hid, inter * E, E)
        self.gu_free = self.d_free = None      # tokens: the last reads of each program's output

    def close(self):
        self.gu.close()
        self.d.close()


class _Pass:
    def __init__(self, E, R, hid, inter, win):
        self.E, self.R, self.hid, self.inter, self.win = E, R, hid, inter, win
        t = time.perf_counter()
        self.bufs = [_Buf(E, R, hid, inter) for _ in range(2)]
        STATS["compile_s"] = STATS.get("compile_s", 0.0) + time.perf_counter() - t
        self.n = 0              # groups started on the ANE this pass
        self.nbytes = sum(sum(p.nbytes) for b in self.bufs for p in (b.gu, b.d))     # the surfaces MLX counts as active

    def close(self):
        for b in self.bufs:
            b.close()
        self.bufs = []


def close():
    """End of a layer-major pass: the programs and their surfaces go."""
    global _PASS, _SIDE
    if _PASS is not None:
        _SIDE = max(_SIDE, _PASS.nbytes)
        _PASS.close()
        _PASS = None


def take_bytes():
    """The surfaces' bytes of the pass closed last, once (MemBudget.close keeps them out of a position's growth)."""
    global _SIDE
    n, _SIDE = _SIDE, 0
    return n


def _cal_path(hid, inter, win):
    dev = mx.device_info().get("device_name", "")
    tag = json.dumps([_V, dev, mx.__version__, hid, inter, win, E_CAND, R_CAND,
                      hashlib.sha1(Path(__file__).read_bytes()).hexdigest()[:12]])
    return Path.home() / ".cache" / "moe-ssd-offload" / f"aneexperts-{hashlib.sha1(tag.encode()).hexdigest()[:16]}.json"


def calibrate(hid, inter, win):
    """The ANE's time for gate + up + down at each candidate (experts, rows) and the GPU's GEMM time a row, timed once and cached."""
    global _CAL
    if _CAL is not None:
        return _CAL
    path = _cal_path(hid, inter, win)
    try:
        _CAL = json.loads(path.read_text())
        return _CAL
    except (OSError, ValueError):
        pass
    ane = {}
    for E in E_CAND:
        for R in R_CAND:
            ms = 0.0
            for B, N, K in ((2 * E, inter, hid), (E, hid, inter)):
                p = A._Prog(R, N, K * B, B)
                mx.eval(A._write(mx.zeros((1, R * K * B), mx.float16), p.x.reshape(1, -1), A._one(), 1, 1),
                        A._write(mx.zeros((1, N * K * B), mx.float16), p.w.reshape(1, -1), A._one(), 1, 1))
                ms += A._time(p.p.run)
                p.close()
            ane[f"{E}.{R}"] = ms
    # The GPU's three GEMMs (E4's f16 gate / up, MLX's f16 down) on 8 experts of 400 rows, per row.
    rng = np.random.default_rng(0)
    c, rows = 8, 3200
    loc = mx.array(np.repeat(np.arange(c), rows // c).astype(np.uint32))
    xg = mx.array(rng.standard_normal((rows, hid)).astype(np.float16))
    xd = mx.array(rng.standard_normal((rows, inter)).astype(np.float16))
    qg = mx.array((rng.standard_normal((c, hid, inter)) * 0.02).astype(np.float16))
    qd = mx.array((rng.standard_normal((c, inter, hid)) * 0.02).astype(np.float16))
    f16 = X._f16mm_ok()
    gmm = (lambda a, q: X._gather_f16(a, q, loc)) if f16 else \
        (lambda a, q: mx.gather_mm(a.astype(mx.float32)[:, None, :], q.astype(mx.float32), rhs_indices=loc, sorted_indices=True))
    us = A._time(lambda: mx.eval(gmm(xg, qg), gmm(xg, qg),
                                 mx.gather_mm(xd[:, None, :], qd, rhs_indices=loc, sorted_indices=True))) * 1e3 / rows
    _CAL = {"ane": ane, "gemm_us_row": us}
    mx.clear_cache()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_CAL))
    return _CAL


def _takes(cnt, E, R):
    """The ANE's experts in a group (local slots): the E largest with at most R rows each."""
    ok = np.flatnonzero((cnt > 0) & (cnt <= R))
    return ok[np.argsort(-cnt[ok], kind="stable")[:E]]


def plan(groups_cnt):
    """(E, R) for this pass from the first layer's groups: the candidate whose predicted layer time is lowest, a group costing
    max(the GPU's rest, the ANE's programs); None when no candidate beats the GPU alone."""
    c = X._COST
    if c is None or _CAL is None:
        return None
    full = [c[2] + c[3] * (cnt > 0).sum() + c[4] * cnt.sum() for cnt in groups_cnt]
    best, t_best = None, sum(full)
    gemm = _CAL["gemm_us_row"] * 1e-6
    for key, ms in _CAL["ane"].items():
        E, R = map(int, key.split("."))
        t = 0.0
        for cnt, f in zip(groups_cnt, full):
            q = cnt[_takes(cnt, E, R)].sum()
            t += max(f - gemm * q, ms * 1e-3) if q else f
        if t < t_best:
            best, t_best = (E, R), t
    STATS["plans"][str(best)] = STATS["plans"].get(str(best), 0) + 1
    return best


def engaged(win, hid, inter, groups_cnt):
    """This pass's programs, made at the first layer that asks (None when off, not Metal, or not worth it)."""
    global _PASS
    if not ON or not mx.metal.is_available():
        return None
    if _PASS is not None:
        return _PASS if _PASS.win == win else None
    try:
        import _ane_mlx  # noqa: F401
    except ImportError as e:
        note(f"  ane experts: the extension is not built ({e}); run native/build.sh", flush=True)
        return None
    calibrate(hid, inter, win)
    p = plan(groups_cnt)
    if p is None:
        return None
    _PASS = _Pass(*p, hid, inter, win)
    return _PASS


class Layer:
    """One layer's expert phase in _call_ahead's groups: group i's ANE inputs and its GPU rows at step i, its middle at
    step i + 1 and its output at step i + 2, so the GPU reads the ANE's work a whole group after queueing it."""

    def __init__(self, ps, x2, out_dtype):
        self.ps, self.x2, self.dt = ps, x2, out_dtype
        self.mid = None         # (buf, rows, inv, arow, srow, sv, dst) waiting for its middle
        self.fin = None         # (buf, rows, isd, arow, srow, svd, dst) waiting for its output
        self.scaled = []        # rows (gate, up, down counted apart) where the bound lowered the scale
        STATS["layers"] += 1

    def group(self, out, n_rows, sel, r, dst, bufs):
        """One group: sel (x rows) and r (local slots, sorted, numpy) of its rows, dst their rows of out; returns out with the
        GPU's rows and the output of the group two steps back written."""
        ps, hid, inter, E, R = self.ps, self.ps.hid, self.ps.inter, self.ps.E, self.ps.R
        cnt = np.bincount(r, minlength=bufs[0][0].shape[0])
        # A group the GPU would run on the decode chain stays whole on the GPU, so its rows keep that chain's arithmetic.
        dense = X.dense_pays(int((cnt > 0).sum()), len(r))
        a = _takes(cnt, E, R) if dense else np.zeros(0, np.int64)
        STATS["groups"] += 1
        out = self._fin(out, n_rows)
        out = self._mid(out, n_rows)
        on = np.isin(r, a)
        if len(a):
            buf = ps.bufs[ps.n % 2]
            ps.n += 1
            ia = np.flatnonzero(on)
            ra = r[ia]
            lut = np.zeros(len(cnt), np.int64)
            lut[a] = np.arange(len(a))
            j = np.arange(len(ra)) - np.searchsorted(ra, ra)     # each row's place among its expert's rows (r is sorted)
            arow = mx.array((lut[ra] * R + j).astype(np.int32))
            srow = mx.array(lut[ra].astype(np.uint32))
            slots = mx.array(a.astype(np.uint32))
            (tg, sug, svg), (tu, suu, svu), (td, sud, svd) = bufs
            # The scales of the ANE's experts copied out: the reader takes the region back before the middle and output run.
            sug, suu, svg, svu, sud, svd = (v[slots] for v in (sug, suu, svg, svu, sud, svd))
            gu, dp = buf.gu, buf.d
            qb = self._qb()
            dep = buf.gu_free if buf.gu_free is not None else A._one()
            inv, pre = _kern("ane_exp_prep", ["x", "xrow", "srow", "arow", "sug", "suu", "qb", "dst", "dep"], ["inv", "tok"],
                             PREP_SRC)(
                inputs=[self.x2, mx.array(sel[ia].astype(np.int32)), srow, arow, sug, suu, qb, gu.x, dep],
                template=[("DIM", hid), ("NB", hid // 128), ("XOFF", E * R * hid)],
                grid=(len(ia) * hid // 4, 1, 1), threadgroup=(hid // 4, 1, 1),
                output_shapes=[(len(ia), 2), (1,)], output_dtypes=[mx.float32, mx.int32])
            toks = [pre, self._deq(tg, slots, gu.w, 0, dep), self._deq(tu, slots, gu.w, E, dep)]
            wd = self._deq(td, slots, dp.w, 0, buf.d_free if buf.d_free is not None else A._one())
            tok, v = gu.p.submit(toks)
            mx.async_eval(tok)              # the ANE starts once these writes are done
            y = gu.p.wait(gu.y, v, [tok])
            self.mid = (buf, len(ia), y, inv, arow, srow, (svg, svu, sud, svd), dst[mx.array(ia.astype(np.int32))], wd)
            self.scaled.append((inv > 1).sum())
            STATS["ane_groups"] += 1
            STATS["ane_rows"] += len(ia)
        ig = np.flatnonzero(~on)
        if len(ig):
            xs = self.x2[mx.array(sel[ig].astype(np.int32))]
            yg = X.dense(xs, bufs, r[ig], ps.win, self.dt) if dense else X.experts(xs, bufs, mx.array(r[ig]), ps.win, host=r[ig])
            out = self._out(out, n_rows, self.dt)
            out[dst[mx.array(ig.astype(np.int32))]] = yg.reshape(-1, 1, hid)
            STATS["gpu_rows"] += len(ig)
        out = self._out(out, n_rows, self.dt)
        mx.eval([out] + ([wd, tok, inv] + list(self.mid[6]) if len(a) else []))     # the region's weights are read before the reader takes it back
        return out

    def _out(self, out, n_rows, dt):
        return mx.zeros((n_rows, 1, self.ps.hid), dtype=dt) if out is None else out

    def _qb(self):
        if getattr(self.ps, "qb", None) is None:
            self.ps.qb = mx.array([qmax(self.ps.win), float(2 ** BOUND_LOG2)], dtype=mx.float32)
        return self.ps.qb

    def _deq(self, t, slots, dst, boff, dep):
        _, it, ot, n = t.shape
        return _kern("ane_exp_deqt", ["trellis", "slots", "dst", "dep"], ["tok"], DEQT_SRC)(
            inputs=[t, slots, dst, dep],
            template=[("T", mx.float16), ("NHW", n), ("IDIM", it * 16), ("ODIM", ot * 16), ("WIN", self.ps.win), ("BOFF", boff)],
            grid=(slots.size * it * (ot // 8) * 256, 1, 1), threadgroup=(256, 1, 1),
            output_shapes=[(1,)], output_dtypes=[mx.int32])[0]

    def _mid(self, out, n_rows):
        """The group queued a step back: its gate and up from the ANE into the down program's input, and the down queued."""
        if self.mid is None:
            return out
        ps, hid, inter, E, R = self.ps, self.ps.hid, self.ps.inter, self.ps.E, self.ps.R
        buf, n, y, inv, arow, srow, sv, dst, wd = self.mid
        self.mid = None
        svg, svu, sud, svd = sv
        dp = buf.d
        isd, tok_m = _kern("ane_exp_mid", ["y", "isc", "arow", "srow", "svg", "svu", "sud", "qb", "dst"], ["inv", "tok"],
                           MID_SRC)(
            inputs=[y, inv, arow, srow, svg, svu, sud, self._qb(), dp.x],
            template=[("DIM", inter), ("NB", inter // 128), ("YOFF", E * R * inter)],
            grid=(n * inter // 4, 1, 1), threadgroup=(inter // 4, 1, 1),
            output_shapes=[(n,), (1,)], output_dtypes=[mx.float32, mx.int32])
        buf.gu_free = tok_m             # gate / up's surfaces may be written again after this read
        self.scaled.append((isd > 1).sum())
        tok, v = dp.p.submit([tok_m, wd])
        mx.async_eval(tok)
        yd = dp.p.wait(dp.y, v, [tok])
        self.fin = (buf, n, yd, isd, arow, srow, svd, dst)
        return out

    def _fin(self, out, n_rows):
        """The group queued two steps back: its down output from the ANE into its rows of out."""
        if self.fin is None:
            return out
        hid = self.ps.hid
        buf, n, yd, isd, arow, srow, svd, dst = self.fin
        self.fin = None
        o = _kern("ane_exp_fin", ["y", "isd", "arow", "srow", "svd"], ["o"], FIN_SRC)(
            inputs=[yd, isd, arow, srow, svd], template=[("DIM", hid), ("T", self.dt)],
            grid=(n * hid // 4, 1, 1), threadgroup=(hid // 4, 1, 1),
            output_shapes=[(n, hid)], output_dtypes=[self.dt])[0]
        buf.d_free = o
        out = self._out(out, n_rows, self.dt)
        out[dst] = o.reshape(-1, 1, hid)
        return out

    def drain(self, out, n_rows):
        """After the last group: the two groups still on the ANE finish into out."""
        out = self._fin(out, n_rows)
        out = self._mid(out, n_rows)
        out = self._fin(out, n_rows)
        if out is not None:
            mx.eval(out)
        if self.scaled:
            STATS["scaled_rows"] += int(sum(self.scaled).item())
        return out
