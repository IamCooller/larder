"""MOE_ANE_OVERLAP (on by default on Metal, 0 turns it off): in Flash-Next's layer-major prefill the Neural Engine computes the next chunk's input projections and this chunk's output projection while the GPU runs the scan or the attention core, closer to fp32 than the GPU's kernels."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

import ane_split as A
import layer_glue

ON = os.environ.get("MOE_ANE_OVERLAP", "1") == "1"     # engages only on qwen4_exp's layer-major prefill of two chunks or more (takes())
KC = int(os.environ.get("MOE_ANE_OVERLAP_KC", "0"))      # 0: inputs 1, the rest the fastest of 1, 2, 4 measured at load
SHARED = os.environ.get("MOE_ANE_OVERLAP_SHARED", "1") == "1"   # the shared expert's gate and up on the ANE too
# Probe-only knobs: EMU=fp32 swaps each ANE program for an fp32 GPU product of its surfaces, UNITS
# keeps only the named unit classes on the ANE (the rest stock), PSTATS records the scaled operands' ranges.
EMU = os.environ.get("MOE_ANE_OVERLAP_EMU", "")
UNITS = set(os.environ["MOE_ANE_OVERLAP_UNITS"].split(",")) if os.environ.get("MOE_ANE_OVERLAP_UNITS") else None
PSTATS = os.environ.get("MOE_ANE_OVERLAP_STATS", "0") == "1"
AUDIT = os.environ.get("MOE_ANE_OVERLAP_AUDIT", "0") == "1"   # each run's outputs against fp32 and stock (probe only)
ROW = os.environ.get("MOE_ANE_OVERLAP_SCALE", "") == "row"     # per-row input and per-channel weight powers of two
# The scaled input's |max| lands in [2^(X_TOP-1), 2^X_TOP), the weights' in (2^(W_TOP-1), 2^W_TOP]; moving bits from x to w
# keeps every product and lifts small weights out of fp16's subnormals, which the ANE flushes.
X_TOP = int(os.environ.get("MOE_ANE_OVERLAP_XTOP", A.X_TOP))
W_TOP = int(os.environ.get("MOE_ANE_OVERLAP_WTOP", A.W_TOP))
# Row scaling keeps s_r t_j sum_k |x_rk w_jk| <= 2^BOUND_LOG2, so no output or partial sum nears the ANE's 32768 (0: no bound, probe only).
BOUND_LOG2 = int(os.environ.get("MOE_ANE_OVERLAP_BOUND", "14")) or None
BOUND_KIND = os.environ.get("MOE_ANE_OVERLAP_BOUND_KIND", "fused")     # fused | rows | max (_x_scale)
_V = 1                  # calibration format
_KC = {}                # (rows, N, K) -> kc
STATS = {"layers": 0, "runs": 0, "loop_s": 0.0, "ane": {}, "compiles": 0, "binds": 0}
_LIVE = False           # set around chunks(): the output projections' hook queues on the ANE only inside it


def _prog(rows, N, K, kc):
    """The loaded program for this shape (compiled once; one chunk width loaded at a time), its surfaces bound for this pass."""
    for k in [k for k in A._PROGS if k[0] != rows]:
        A._PROGS.pop(k).close()
    p = A._PROGS.get((rows, N, K, kc))
    if p is None:
        t = time.perf_counter()
        p = A._prog(rows, N, K, kc)
        STATS["compiles"] += 1
        STATS["compile_s"] = STATS.get("compile_s", 0.0) + time.perf_counter() - t
    elif not p.bound:
        p.bind()
        STATS["binds"] += 1
    return p


def _kc(u, rows):
    # Inputs at split-K 1: kc multiplies the output surface, theirs are the big ones, and k1 ran as fast in the pipeline (README §2).
    return KC or (1 if u.name == "in" else _KC.get((rows, u.N, u.K), 1))


class Unit:
    """Projections reading the same input, rows stacked, all on the ANE: run() queues x and returns each member's lazy bf16 output."""

    def __init__(self, mods, name, cls=None):
        self.mods, self.name, self.cls = mods, name, cls or name
        self.parts = [(m.weight, m.scales, m.get("biases"), m.group_size, m.bits) for m in mods]
        self.ns = [m.weight.shape[0] for m in mods]
        self.N = sum(self.ns)
        self.K = mods[0].weight.shape[1] * 32 // mods[0].bits
        self._wn = None

    def q4(self):
        return all(part[4] == 4 and part[2] is not None for part in self.parts)

    def sw(self):
        """The weights' powers of two: one an output channel under row scaling, else one for the unit; the max |w| to (2^(W_TOP-1), 2^W_TOP]."""
        if self.q4() and ROW:
            return A._pow2(mx.concatenate([A._wmax(part, True) for part in self.parts]), W_TOP)
        if self.q4():
            wm = [A._wmax(part) for part in self.parts]
            return A._pow2(wm[0] if len(wm) == 1 else mx.maximum(*wm) if len(wm) == 2 else mx.stack(wm).max(), W_TOP)
        d = mx.concatenate([mx.abs(A._deq(part, n)).max(axis=1) for part, n in zip(self.parts, self.ns)])
        return A._pow2(d if ROW else d.max(), W_TOP)

    def wnorms(self):
        """[max_j t_j ||w_j||_1, max_j t_j ||w_j||_2] under the weights' powers of two t_j (sw()), once, raised by 2^-8 for the fp16 weights and fp32 sums."""
        if self._wn is None:
            t = self.sw()
            l1, l2, r = [], [], 0
            for (w, s, b, g, bits), n in zip(self.parts, self.ns):
                for a in range(0, n, 2048):      # row blocks: no fp32 copy of the whole layer
                    z = min(a + 2048, n)
                    d = mx.abs(mx.dequantize(w[a:z], s[a:z].astype(mx.float32), None if b is None else b[a:z].astype(mx.float32),
                                             group_size=g, bits=bits)) * (t if t.size == 1 else t[r + a:r + z, None])
                    l1.append(d.sum(axis=1).max())
                    l2.append(mx.sqrt((d * d).sum(axis=1)).max())
                    mx.eval(l1[-1], l2[-1])
                r += n
            self._wn = mx.stack([mx.stack(l1).max(), mx.stack(l2).max()]) * (1 + 2 ** -8)
            mx.eval(self._wn)
        return self._wn

    def run(self, x, rows):
        S, K = x.shape[1], self.K
        kc = _kc(self, rows)
        p = _prog(rows, self.N, K, kc)
        # The surfaces are rewritten only after the previous run's outputs were read out of them (in the GPU's order).
        dep = mx.concatenate([o.reshape(-1)[:1] for o in p.last]) if p.last else None
        toks = []
        if p.owner is not self:
            if self.q4():
                # 4-bit straight into the surface: no fp32 copy of the layer's weights (~0.5 GB transient before).
                p.sw = self.sw()
                r0 = 0
                for part, n in zip(self.parts, self.ns):
                    toks.append(A._write_q(part, p.w, p.sw, kc, self.N, r0, dep, int(ROW)))
                    r0 += n
            else:
                ds = [A._deq(part, n) for part, n in zip(self.parts, self.ns)]
                d = ds[0] if len(ds) == 1 else mx.concatenate(ds, axis=0)
                p.sw = A._pow2(mx.abs(d).max(axis=1 if ROW else None), W_TOP)
                toks.append(A._write(d, p.w, p.sw, kc, self.N, dep, int(ROW)))
            p.owner = self
            A.STATS["weights"] += 1
        x2 = x.reshape(S, K)
        sx = _x_scale(self, x2)
        toks.append(A._write(x2, p.x, sx, kc, rows, dep, int(ROW)))
        if EMU:
            y = _emu(p, toks, kc, rows)
        else:
            tok, v = p.p.submit(toks)
            mx.async_eval(tok)              # the ANE starts as soon as the GPU has written its input
            y = p.p.wait(p.y, v, [tok])
        inv = None if ROW else mx.reciprocal(sx * p.sw)
        if PSTATS and not ROW:
            _range(self, x2, sx, p.sw, y, S, inv)
        outs, c0 = [], 0
        for n in self.ns:
            # Row scaling: each row's and each channel's power of two undone apart (both exact).
            outs.append(_out_row(y, mx.reciprocal(sx), mx.reciprocal(p.sw), S, c0, n, x.dtype) if ROW
                        else _out(y, inv, S, c0, n, x.dtype))
            c0 += n
        if AUDIT:
            _audit(self, x, outs)
        p.last = outs
        STATS["runs"] += 1
        return outs


_XST = None


def _xstats(x2):
    """Each row's max |x|, sum |x| and sum x^2 [S, 3] in fp32, one threadgroup a row and one read of the bf16 rows."""
    global _XST
    if _XST is None:
        _XST = mx.fast.metal_kernel(
            name="ane_overlap_xstats", input_names=["x"], output_names=["o"],
            source="uint r = threadgroup_position_in_grid.x, t = thread_position_in_threadgroup.x;\n"
                   "uint K = x_shape[1];\n"
                   "float m = 0.0f, s1 = 0.0f, s2 = 0.0f;\n"
                   "for (uint k = t; k < K; k += 256) {\n"
                   "  float v = metal::abs(static_cast<float>(x[r * K + k]));\n"
                   "  m = metal::max(m, v); s1 += v; s2 += v * v;\n"
                   "}\n"
                   "m = simd_max(m); s1 = simd_sum(s1); s2 = simd_sum(s2);\n"
                   "threadgroup float sh[24];\n"
                   "uint w = t / 32, l = t % 32;\n"
                   "if (l == 0) { sh[w] = m; sh[8 + w] = s1; sh[16 + w] = s2; }\n"
                   "threadgroup_barrier(mem_flags::mem_threadgroup);\n"
                   "if (t == 0) {\n"
                   "  for (uint i = 1; i < 8; i++) { m = metal::max(m, sh[i]); s1 += sh[8 + i]; s2 += sh[16 + i]; }\n"
                   "  o[r * 3] = m; o[r * 3 + 1] = s1; o[r * 3 + 2] = s2;\n"
                   "}\n")
    S = x2.shape[0]
    return _XST(inputs=[x2], grid=(S * 256, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(S, 3)],
                output_dtypes=[mx.float32])[0]


def _x_scale(u, x2):
    """The inputs' powers of two (one a row under row scaling, else one for the run): the max to [2^(X_TOP-1), 2^X_TOP), lowered where
    s_r times a bound on sum_k |x_rk| t_j |w_jk| over every j would pass 2^BOUND_LOG2. MOE_ANE_OVERLAP_BOUND_KIND: fused (the least of
    ||x_r||_2 W2, max |x_r| W1, ||x_r||_1 2^W_TOP from _xstats), rows (the same from MLX reductions), max (max |x| W1 alone, no extra pass)."""
    if BOUND_KIND == "max" and not ROW:
        m = mx.abs(x2).max()
        s = A._pow2(m, X_TOP)
        if BOUND_LOG2 is None:
            return s
        b = m.astype(mx.float32) * u.wnorms()[0]
        cap = A._pow2(b, BOUND_LOG2)
        if AUDIT:
            e = STATS.setdefault("bound", {}).setdefault(u.cls, [0, 0, 0])
            e[0] += int((cap < s).sum().item())
            e[1] += 1
            e[2] += int(mx.maximum(mx.log2(s / cap), 0).sum().item())
        return mx.minimum(s, cap)
    if BOUND_KIND == "fused" and BOUND_LOG2 is not None:
        st = _xstats(x2)
        m, l1, l2 = st[:, 0], st[:, 1], mx.sqrt(st[:, 2])
    else:
        a = mx.abs(x2).astype(mx.float32)
        m = a.max(axis=1)
        if BOUND_LOG2 is not None:
            l1, l2 = a.sum(axis=1), mx.sqrt((a * a).sum(axis=1))
    s = A._pow2(m if ROW else m.max(), X_TOP)
    if BOUND_LOG2 is None:
        return s
    w1, w2 = u.wnorms()
    c = 1 + 2 ** -8         # the fp32 sums' rounding
    if BOUND_KIND == "max":
        b = m * w1
    else:
        b = mx.minimum(mx.minimum(l2 * c * w2, m * w1), l1 * c * 2.0 ** W_TOP)
    cap = A._pow2(b if ROW else b.max(), BOUND_LOG2)
    if AUDIT:               # rows (runs without row scaling) where the bound lowers the scale, and by how many bits in all
        e = STATS.setdefault("bound", {}).setdefault(u.cls, [0, 0, 0])
        e[0] += int((cap < s).sum().item())
        e[1] += x2.shape[0] if ROW else 1
        e[2] += int(mx.maximum(mx.log2(s / cap), 0).sum().item())
    return mx.minimum(s, cap)


_RD = None


def _emu(p, toks, kc, rows):
    """MOE_ANE_OVERLAP_EMU: the ANE program's product [kc, rows, N] from the same fp16 surfaces after the same writes, in fp32 on the GPU;
    flush / flushw also zero the inputs' and weights' (or only the weights') subnormals as the ANE does, fp16 rounds the output to fp16."""
    global _RD
    if _RD is None:
        _RD = mx.fast.metal_kernel(
            name="ane_overlap_rd", input_names=["src", "dep"], output_names=["o"],
            source="uint i = thread_position_in_grid.x;\n"
                   "if (i < src_shape[0]) {\n"
                   "  float v = static_cast<float>(src[i]);\n"
                   "  o[i] = FL && fabs(v) < 6.103515625e-05f ? 0.0f : v;\n"
                   "}\n")
    dep = mx.concatenate(toks)
    rd = lambda a, fl: _RD(inputs=[a, dep], template=[("FL", int(fl))], grid=(a.size, 1, 1), threadgroup=(256, 1, 1),
                           output_shapes=[a.shape], output_dtypes=[mx.float32])[0]
    y = mx.matmul(rd(p.x, EMU == "flush").reshape(kc, rows, -1),
                  rd(p.w, EMU in ("flush", "flushw")).reshape(kc, p.Na, -1).transpose(0, 2, 1))
    return y.astype(mx.float16) if EMU == "fp16" else y


def _audit(u, x, outs):
    """MOE_ANE_OVERLAP_AUDIT: per unit class, this run's bf16 outputs, the stock kernel's and bf16(truth) against the truth x @ (q s + b)
    in fp32 and against bf16(truth) (the reference arm's output): squared error, error along the reference's sign, elements off."""
    import numpy as np
    e = STATS.setdefault("audit", {}).setdefault(u.cls, {"calls": 0, "n": 0, "t2": 0.0})
    x32 = x.astype(mx.float32)
    vals = []
    for m, part, n, o in zip(u.mods, u.parts, u.ns, outs):
        t = x32 @ A._deq(part, n).T
        tb = t.astype(x.dtype).astype(mx.float32)
        st = nn.QuantizedLinear.__call__(m, x).astype(mx.float32)
        row = [(t * t).sum()]
        for v in (o.astype(mx.float32), st, tb):
            for r in (t, tb):
                d = v - r
                row += [(d * d).sum(), (d * mx.sign(r)).sum(), (d != 0).sum().astype(mx.float32)]
        vals.append(mx.stack(row))
    a = np.array(mx.stack(vals)).astype(np.float64).sum(0)
    e["calls"] += 1
    e["n"] += int(sum(x.shape[1] * n for n in u.ns))
    e["t2"] += float(a[0])
    keys = [f"{v}~{r}.{q}" for v in ("ane", "stock", "bf16") for r in ("t", "ref") for q in ("d2", "dsign", "off")]
    for k, val in zip(keys, a[1:]):
        e[k] = e.get(k, 0.0) + float(val)


TINY = 2.0 ** -14      # fp16's smallest normal: the ANE flushes smaller inputs and all-tiny groups of products


def _range(u, x2, sx, sw, y, S, inv):
    """MOE_ANE_OVERLAP_STATS: per unit class, scaled inputs and weights under fp16's normal range, a lower bound on 32-wide K groups of all-tiny products, max |y| against 32768, and the same under per-row scaling."""
    import numpy as np
    e = STATS.setdefault("range", {}).setdefault(u.cls, {"calls": 0, "x_n": 0, "x_sub": 0, "x_zero": 0, "w_n": 0,
                                                         "w_sub": 0, "groups": 0, "tiny_groups": 0, "y_max": 0.0,
                                                         "tiny_groups_row": 0, "y_max_row": 0.0})
    K = x2.shape[1]
    if getattr(u, "_gw", None) is None:          # the unit's weights, once: per-group max |w| sorted per group, plain and per-channel scaled
        w = mx.concatenate([A._deq(part, n) for part, n in zip(u.parts, u.ns)]) * sw
        a = mx.abs(w)
        e["w_n"] += w.size
        e["w_sub"] += ((a > 0) & (a < TINY)).sum().item()
        gw = np.array(a.reshape(u.N, K // 32, 32).max(-1))
        cw = gw.max(-1)
        u._swc = np.where(cw > 0, 2.0 ** (W_TOP - np.ceil(np.log2(np.maximum(cw, 1e-30)))), 1.0)
        u._gw, u._gwr = np.sort(gw, 0), np.sort(gw * u._swc[:, None], 0)
        del w, a
    xs = mx.abs(x2.astype(mx.float32) * sx)
    gx = np.array(xs.reshape(S, K // 32, 32).max(-1))
    e["calls"] += 1
    e["x_n"] += xs.size
    e["x_sub"] += ((xs > 0) & (xs < TINY)).sum().item()
    e["x_zero"] += (xs == 0).sum().item()
    ys = mx.abs(y[:, :S].astype(mx.float32).sum(0))                    # scaled |output|, [S, N]
    e["y_max"] = max(e["y_max"], ys.max().item())
    # Per-row scaling on top: each row's max to [2^7, 2^8), each output channel's max |w| to (2^-1, 2^0].
    rx = gx.max(1)
    sxr = np.where(rx > 0, 2.0 ** (X_TOP - np.ceil(np.log2(np.maximum(rx, 1e-30)))), 1.0)
    e["y_max_row"] = max(e["y_max_row"], (ys * mx.array((sxr[:, None] * u._swc[None, :]).astype(np.float32))).max().item())
    gw = u._gw
    for g in range(gw.shape[1]):
        e["tiny_groups"] += int(np.searchsorted(gw[:, g], TINY / np.maximum(gx[:, g], 1e-30)).sum())
        e["tiny_groups_row"] += int(np.searchsorted(u._gwr[:, g], TINY / np.maximum(gx[:, g] * sxr, 1e-30)).sum())
    e["groups"] += S * u.N * gw.shape[1]


_OUTR = None


def _out_row(y, ix, iw, S, c0, n, dtype):
    """_out with one inverse scale a row (ix [S]) and one a weight row (iw [N])."""
    global _OUTR
    if _OUTR is None:
        _OUTR = mx.fast.metal_kernel(
            name="ane_overlap_out_row", input_names=["y", "ix", "iw"], output_names=["o"],
            source="uint i = thread_position_in_grid.x;\n"
                   "if (i >= S_ * NM) return;\n"
                   "uint r = i / NM, c = C0 + i % NM;\n"
                   "float s = 0.0f;\n"
                   "for (uint k = 0; k < KCH; k++) s += static_cast<float>(y[(k * ROWS + r) * NT + c]);\n"
                   "o[i] = static_cast<OT>(s * ix[r] * iw[c]);\n")
    kc, rows, N = y.shape
    return _OUTR(inputs=[y, ix, iw], template=[("S_", S), ("NM", n), ("C0", c0), ("KCH", kc), ("ROWS", rows),
                                                ("NT", N), ("OT", dtype)],
                 grid=(S * n, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(1, S, n)], output_dtypes=[dtype])[0]


_OUT = None


def _out(y, inv, S, c0, n, dtype):
    """Columns [c0, c0 + n) of the ANE's output y [kc, rows, N] (fp16, kc partial sums) for its first S rows, summed in fp32, unscaled and rounded once."""
    global _OUT
    if _OUT is None:
        _OUT = mx.fast.metal_kernel(
            name="ane_overlap_out", input_names=["y", "inv"], output_names=["o"],
            source="uint i = thread_position_in_grid.x;\n"
                   "if (i >= S_ * NM) return;\n"
                   "uint r = i / NM, c = C0 + i % NM;\n"
                   "float s = 0.0f;\n"
                   "for (uint k = 0; k < KCH; k++) s += static_cast<float>(y[(k * ROWS + r) * NT + c]);\n"
                   "o[i] = static_cast<OT>(s * inv[0]);\n")
    kc, rows, N = y.shape
    return _OUT(inputs=[y, inv], template=[("S_", S), ("NM", n), ("C0", c0), ("KCH", kc), ("ROWS", rows),
                                           ("NT", N), ("OT", dtype)],
                grid=(S * n, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(1, S, n)], output_dtypes=[dtype])[0]


class _In(nn.QuantizedLinear):
    """A projection whose output for this x was queued on the ANE earlier (chunks()) is handed over, else stock."""

    def __call__(self, x):
        held = self.__dict__.get("_ov_held")
        for i, (hx, y) in enumerate(held or ()):
            if hx is x:
                del held[i]
                return y
        return super().__call__(x)


_HELD = {}              # id -> module holding outputs, emptied at the pass's end (close())


def _hold(mods, x, ys):
    for m, y in zip(mods, ys):
        m.__dict__.setdefault("_ov_held", []).append((x, y))
        _HELD[id(m)] = m


def _drop_held():
    """Unread outputs go and every hooked projection is a stock QuantizedLinear again (decode never pays the hooks)."""
    for m in _HELD.values():
        STATS["unread"] = STATS.get("unread", 0) + len(m.__dict__.get("_ov_held") or ())
        m.__dict__["_ov_held"] = []
    _HELD.clear()
    for m in _HOOKED.values():
        m.__class__ = m.__dict__["_ov_base"]
    _HOOKED.clear()


_HOOKED = {}            # id -> projection whose class is _In / _Out for the layer running now


def _hook(mods):
    for m in mods:
        m.__dict__.setdefault("_ov_base", m.__class__)
        m.__class__ = m.__dict__["_ov_cls"]
        _HOOKED[id(m)] = m


class _Out(nn.QuantizedLinear):
    """An output projection: inside chunks() queued on the ANE (read a chunk later), else stock."""

    def __call__(self, x):
        u = self.__dict__.get("_ov_unit")
        if _LIVE and _on(u) and x.ndim == 3 and x.shape[0] == 1 and x.shape[1] <= _LIVE:
            return u.run(x, _LIVE)[0]
        return super().__call__(x)


def _on(u):
    """A unit that runs on the ANE (MOE_ANE_OVERLAP_UNITS names the classes; all when unset)."""
    return u is not None and (UNITS is None or u.cls in UNITS)


def takes(layer, h):
    """Whether this layer's chunk loop runs through chunks(): flag on, a qwen4_exp layer on layer_glue's prefill rows with its units installed."""
    if not ON or not layer.__dict__.get("_lgp") or not layer_glue.rows_ok(h):
        return False
    mixer = layer.linear_attn if layer.layer_type == "linear_attention" else layer.self_attn
    return "_ov_in" in mixer.__dict__


def chunks(layer, args, bounds, step):
    """_lm_layer_call's chunk loop with the ANE a chunk ahead: chunk i+1's mix and input projections are queued before chunk i's core, chunk i's output projection after it, and chunk i's MLP half (`step`) runs after chunk i+1's core is queued."""
    global _LIVE
    h, rope, mask, conv_mask, cache, idx_cache, ids, prev_ctx = args
    lin = layer.layer_type == "linear_attention"
    s = layer.linear_attn if lin else layer.self_attn
    uin = s.__dict__["_ov_in"]
    rows = bounds[0][1] - bounds[0][0]
    ctx = prev_ctx.shape[1] if prev_ctx is not None else 0
    gdn = lin and s.__dict__.get("_lgp")

    def mix(a, b):
        hc = h[:, a:b]
        if layer.ple is not None:
            hc = hc + layer.ple(hc, ids[:, a:b], prev_ctx if a == 0 else ids[:, a - ctx:a], cache)
        x, hyper, inject = layer_glue.hc_rows(layer.attn_hyper_connection, hc)
        return x, hyper, inject, (uin.run(x, rows) if _on(uin) else None)

    def core(x, a, b):
        if gdn and layer_glue.gdn_rows_ok(s, x, cache):
            return layer_glue.gdn_rows(s, x, cache)
        if lin:
            return s(x, conv_mask, cache)
        return s(x, rope, mask if b - a > 1 else None, cache, idx_cache)

    _drop_held()            # the previous layer's combine loop is over: what it left unread goes
    se = getattr(getattr(layer.mlp, "block", layer.mlp), "shared_expert", None)     # through CacheBiasedMoE too
    ush = se.__dict__.get("_ov_unit") if SHARED and se is not None else None
    ush = ush if _on(ush) else None
    _hook(s.__dict__["_ov_mods"] + (ush.mods if ush is not None else []))

    def shared(xr):
        # The shared expert's gate and up for this chunk's MLP input, read in the combine loop after the experts.
        if ush is not None and xr is not None:
            _hold(ush.mods, xr, ush.run(xr, rows))

    t0 = time.perf_counter()
    _LIVE = rows
    try:
        nxt, prev = mix(*bounds[0]), None
        for i, (a, b) in enumerate(bounds):
            x, hyper, inject, ys = nxt
            if i + 1 < len(bounds):
                nxt = mix(*bounds[i + 1])
            if ys is not None:
                _hold(uin.mods, x, ys)
            try:
                out = core(x, a, b)
            finally:
                for m in uin.mods:
                    m.__dict__["_ov_held"] = []
            if prev is not None:
                shared(step(prev))
            prev = (hyper, out, inject)
        shared(step(prev))
    finally:
        _LIVE = False
    STATS["layers"] += 1
    STATS["loop_s"] += time.perf_counter() - t0


def close():
    """End of a pass: the programs' run times go to STATS (flag on) and every ANE program closes, its surfaces leaving the budget."""
    if not ON:
        A.close_all()
        return
    _drop_held()
    for k, p in A._PROGS.items():
        st = p.p.stats()
        e = STATS["ane"].setdefault("x".join(map(str, k)), {"runs": 0, "ane_ms": 0.0, "wait_ms": 0.0, "errors": 0})
        e["runs"] += st["submitted"]
        for key in ("ane_ms", "wait_ms", "errors"):
            e[key] += st[key]
    # Closed, not kept: the 4 loaded programs held ~0.26-0.30 GB wired outside the footprint between prompts (README
    # round 2) and compiling them again costs ~0.2 s a pass.
    A.close_all()


def calibrate(u, rows):
    """The split-K (MOE_ANE_OVERLAP_KC, else the fastest of 1, 2, 4) at which the ANE alone runs this shape fastest, timed at load, cached on disk."""
    key = (rows, u.N, u.K)
    if key in _KC:
        return
    if KC:
        _KC[key] = KC
        return
    dev = mx.device_info().get("device_name", "")
    tag = json.dumps([_V, dev, mx.__version__, list(key), [list(p[0].shape) + [p[4]] for p in u.parts]])
    path = Path.home() / ".cache" / "moe-ssd-offload" / f"aneoverlap-{hashlib.sha1(tag.encode()).hexdigest()[:16]}.json"
    try:
        _KC[key] = json.loads(path.read_text())["kc"]
        return
    except (OSError, ValueError, KeyError):
        pass
    x = (mx.random.normal((1, rows, u.K)) * 0.5).astype(u.parts[0][1].dtype)
    mx.eval(x)
    ms = {}
    for kc in (1, 2, 4):
        if u.K % (kc * 32):
            continue
        _KC[key] = kc
        mx.eval(u.run(x, rows))
        p = A._PROGS.pop((rows, u.N, u.K, kc))
        ms[kc] = A._time(p.p.run)
        p.close()           # one program at a time: each holds wired memory outside the footprint
        mx.clear_cache()
    _KC[key] = min(ms, key=ms.get)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"kc": _KC[key], "ms": ms, "key": tag}))


def install(model, rows):
    """Units for a qwen4_exp model's delta nets and attention layers (input group, output projection), calibrated; returns how many layers (0 when off)."""
    q4 = sys.modules.get("mlx_lm.models.qwen4_exp")
    if not ON or not mx.metal.is_available() or q4 is None:
        return 0
    if A.ON:
        print("  ane overlap: off, MOE_ANE_SPLIT is on (one ANE path at a time)", flush=True)
        return 0
    try:
        import _ane_mlx  # noqa: F401
    except ImportError as e:
        print(f"  ane overlap: the extension is not built ({e}); run native/build.sh", flush=True)
        return 0
    q = lambda ms: all(isinstance(m, nn.QuantizedLinear) and m.mode == "affine" for m in ms)
    n, seen, allu = 0, {}, []
    for m in model.modules():
        if isinstance(m, q4.GatedDeltaNet):
            ins, out, kind = [m.in_proj_qkv, m.in_proj_z], m.out_proj, "gdn"
        elif isinstance(m, q4.Attention):
            ins, out, kind = [m.indexer.index_qk_proj, m.q_proj, m.k_proj, m.v_proj], m.o_proj, "attn"
        else:
            continue
        if not q(ins + [out]):
            continue
        u_in, u_out = Unit(ins, "in", kind + "_in"), Unit([out], "out", kind + "_out")
        units = [u_in, u_out]
        for x in ins:
            x.__dict__["_ov_cls"] = _In
        out.__dict__["_ov_cls"] = _Out
        out.__dict__["_ov_unit"] = u_out
        m.__dict__["_ov_in"] = u_in
        m.__dict__["_ov_mods"] = ins + [out]         # hooked only while this layer runs (chunks())
        n += 1
        allu += units
        for u in units:
            seen.setdefault((u.N, u.K), u)
    for m in model.modules():
        if isinstance(m, q4.SparseMoeBlock) and SHARED:
            se = m.shared_expert
            if q([se.gate_proj, se.up_proj]):
                u = Unit([se.gate_proj, se.up_proj], "shared")
                se.gate_proj.__dict__["_ov_cls"] = se.up_proj.__dict__["_ov_cls"] = _In
                se.__dict__["_ov_unit"] = u
                allu.append(u)
                seen.setdefault((u.N, u.K), u)
    if BOUND_LOG2 is not None:
        for u in allu:          # the bound's weight norms at load, not in the first pass
            u.wnorms()
        mx.clear_cache()
    import atexit
    atexit.register(A.close_all)
    for u in seen.values():
        if u.name != "in":
            calibrate(u, rows)
    return n


def summary():
    return {f"{k[1]}x{k[2]}": {"kc": v} for k, v in _KC.items()}
