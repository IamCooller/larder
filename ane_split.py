"""MOE_ANE_SPLIT=1 (off by default, Metal): in a layer-major prefill pass the Neural Engine computes the first output channels of
each big dense projection while the GPU computes the rest, not bit-exact."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

AUDIT_ON = os.environ.get("MOE_ANE_AUDIT", "0") == "1"
ON = os.environ.get("MOE_ANE_SPLIT", "0") == "1" or AUDIT_ON
SCALE = os.environ.get("MOE_ANE_SCALE", "1") == "1"      # on: same speed, lower error
KC = int(os.environ.get("MOE_ANE_SPLITK", "0"))     # 0: the fastest of 1, 2, 4 measured at load
SYNC = os.environ.get("MOE_ANE_SYNC", "0") == "1"       # the host waits for the input and runs the ANE itself
ACTIVE = False          # set by stream_fast._lm_layer_call around a layer-major pass's chunks
_V = 3                  # calibration format
_PROGS = {}             # (rows, n_ane, k, kc) -> _Prog
_CAL = {}               # (rows, n, k) -> (n_ane, kc); ("t", rows, n, k) -> (gpu ms, split ms)
STATS = {"calls": 0, "stock": 0, "ane_ms": 0.0, "wait_ms": 0.0, "weights": 0}
AUDIT = {}              # "<layer>.<kind>" -> per module, per variant: errors and ranges (MOE_ANE_AUDIT)
VARIANTS = ((1, False), (1, True), (2, False), (4, False), (8, False))      # (split-K chunks, power-of-two scaling)
X_TOP, W_TOP = 8, 0     # the scaled input's |max| in [2^7, 2^8), the weights' in (2^-1, 2^0]

HDR = ('program(1.3)\n[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3520.4.1"}, '
       '{"coremlc-version", "3520.5.1"}})]\n{\n')


def _mil(S, Na, K, kc):
    """MIL for y [1, kc, S, Na], kc partial products over K's chunks of x [kc, S, K / kc] and w [kc, Na, K / kc] in one input."""
    c = K // kc
    t = lambda *d: f"tensor<fp16, [{', '.join(map(str, d))}]>"
    i4 = lambda n, v: (f'        tensor<int32, [4]> {n} = const()[name = string("{n}"), '
                       f'val = tensor<int32, [4]>([{", ".join(map(str, v))}])];\n')
    return (HDR + f"    func main<ios18>({t(1, kc * (S + Na), 1, c)} x) {{\n"
            + i4("b0", (0, 0, 0, 0)) + i4("s0", (1, kc * S, 1, c))
            + i4("b1", (0, kc * S, 0, 0)) + i4("s1", (1, kc * Na, 1, c))
            + i4("r1", (1, kc, S, c)) + i4("r2", (1, kc, Na, c))
            + f'        {t(1, kc * S, 1, c)} xa = slice_by_size(x = x, begin = b0, size = s0)[name = string("xa")];\n'
            f'        {t(1, kc * Na, 1, c)} wa = slice_by_size(x = x, begin = b1, size = s1)[name = string("wa")];\n'
            f'        {t(1, kc, S, c)} xm = reshape(x = xa, shape = r1)[name = string("xm")];\n'
            f'        {t(1, kc, Na, c)} wm = reshape(x = wa, shape = r2)[name = string("wm")];\n'
            '        bool tx = const()[name = string("tx"), val = bool(false)];\n'
            '        bool ty = const()[name = string("ty"), val = bool(true)];\n'
            f'        {t(1, kc, S, Na)} y = matmul(transpose_x = tx, transpose_y = ty, x = xm, y = wm)[name = string("y")];\n'
            "    } -> (y);\n}\n")


_COPY = None


def _write(src, dst, scale, kc, rt, dep=None, rs=0):
    """A lazy token that leaves src [R, K] times scale (one, or one a row with rs=1) in fp16 in the surface alias dst as [kc, rt, K / kc], after dep."""
    global _COPY
    if _COPY is None:
        _COPY = mx.fast.metal_kernel(
            name="ane_copy_in", input_names=["src", "sc", "dst", "dep"], output_names=["tok"],
            source="uint i = thread_position_in_grid.x;\n"
                   "uint R = src_shape[0], K = src_shape[1];\n"
                   "if (i < R * K) {\n"
                   "  uint r = i / K, c = i % K, j = c / KC;\n"
                   "  ((device half*)dst)[(j * RT + r) * KC + c % KC] = static_cast<half>(static_cast<float>(src[i]) * sc[RS * r]);\n"
                   "}\n"
                   "if (i == 0) tok[0] = 1;\n")
    n = src.shape[0] * src.shape[1]
    return _COPY(inputs=[src, scale, dst, _one() if dep is None else dep], template=[("KC", src.shape[1] // kc), ("RT", rt), ("RS", rs)], grid=(n, 1, 1),
                 threadgroup=(256, 1, 1), output_shapes=[(1,)], output_dtypes=[mx.int32])[0]


_COPYQ = None


def _write_q(part, dst, scale, kc, rt, r0, dep=None, rs=0):
    """A lazy token that leaves the 4-bit affine rows of `part` dequantized (q * s + b in fp32, as mx.dequantize), times scale, in fp16 in dst's rows [r0, r0 + n) of [kc, rt, K / kc]: no fp32 copy of the weights."""
    global _COPYQ
    if _COPYQ is None:
        _COPYQ = mx.fast.metal_kernel(
            name="ane_copy_q4", input_names=["w", "s", "b", "sc", "dst", "dep"], output_names=["tok"],
            source="uint i = thread_position_in_grid.x;\n"
                   "uint R = w_shape[0], K = w_shape[1] * 8;\n"
                   "if (i < R * K) {\n"
                   "  uint r = i / K, c = i % K, j = c / KC, g = c / G;\n"
                   "  uint q = (w[r * (K / 8) + c / 8] >> (4 * (c % 8))) & 0xF;\n"
                   "  float v = static_cast<float>(q) * static_cast<float>(s[r * (K / G) + g]) + static_cast<float>(b[r * (K / G) + g]);\n"
                   "  ((device half*)dst)[(j * RT + R0 + r) * KC + c % KC] = static_cast<half>(v * sc[RS * (R0 + r)]);\n"
                   "}\n"
                   "if (i == 0) tok[0] = 1;\n")
    w, s, b, g, bits = part
    n = w.shape[0] * w.shape[1] * 8
    return _COPYQ(inputs=[w, s, b, scale, dst, _one() if dep is None else dep],
                  template=[("KC", w.shape[1] * 8 // kc), ("RT", rt), ("R0", r0), ("G", g), ("RS", rs)], grid=(n, 1, 1),
                  threadgroup=(256, 1, 1), output_shapes=[(1,)], output_dtypes=[mx.int32])[0]


def _wmax(part, rows=False):
    """A bound on max |q * s + b| from the scales and biases alone (q in [0, 2^bits - 1]), for the power-of-two scaling; one a row with rows=True."""
    w, s, b, g, bits = part
    s, b = s.astype(mx.float32), b.astype(mx.float32)
    return mx.maximum(mx.abs(b), mx.abs(s * (2 ** bits - 1) + b)).max(axis=1 if rows else None)


def _pow2(a, top):
    """2^(top - ceil(log2 a)) as fp32 [1] (or one per element of a vector a), exact (built from its bits), for a > 0 (1 for a == 0)."""
    e = mx.ceil(mx.log2(mx.maximum(a.astype(mx.float32), 1e-30)))
    n = mx.clip(top - e, -126, 127).astype(mx.int32)
    n = mx.where(a > 0, n, 0)
    return ((n + 127) * (1 << 23)).astype(mx.int32).view(mx.float32).reshape(-1)


_ONE = None


def _one():
    global _ONE
    if _ONE is None:
        _ONE = mx.ones((1,), mx.float32)
    return _ONE


class _Prog:
    """One ANE program [S rows, Na channels, K, kc chunks] with its surfaces: x rows then weight rows in, y out."""

    def __init__(self, S, Na, K, kc):
        import _ane_mlx
        self.S, self.Na, self.K, self.kc = S, Na, K, kc
        self.nbytes = ((S + Na) * K * 2, kc * S * Na * 2)
        self.p = _ane_mlx.Program(_mil(S, Na, K, kc).encode(), *self.nbytes)
        self._alias()

    def _alias(self):
        # One alias a surface: MLX counts each alias's whole buffer, so x and w are views of the input's.
        S, K = self.S, self.K
        xin = self.p.alias(0, [(S + self.Na) * K], 0)
        self.x, self.w = xin[:S * K], xin[S * K:]
        self.y = self.p.alias(1, [self.kc, S, self.Na], 0)
        mx.eval(self.x, self.w, self.y)
        self.owner, self.last, self.sw, self.dep = None, [], _one(), None
        self.bound = True

    def free(self):
        """The surfaces go (MOE_ANE_OVERLAP at a pass's end); the program stays loaded for the next pass."""
        self.x = self.w = self.y = None
        self.owner, self.last, self.bound = None, [], False
        self.p.release()

    def bind(self):
        self.p.bind(*self.nbytes)
        self._alias()

    def close(self):
        self.x = self.w = self.y = None
        self.p.close()


def _prog(S, Na, K, kc):
    key = (S, Na, K, kc)
    if key not in _PROGS:
        _PROGS[key] = _Prog(S, Na, K, kc)
    return _PROGS[key]


def close_all():
    for p in _PROGS.values():
        p.close()
    _PROGS.clear()


def _qmm(x, part, r0=0, r1=None):
    w, s, b, g, bits = part
    return mx.quantized_matmul(x, w[r0:r1], scales=s[r0:r1], biases=None if b is None else b[r0:r1],
                               transpose=True, group_size=g, bits=bits)


def _deq(part, r1):
    w, s, b, g, bits = part
    return mx.dequantize(w[:r1], s[:r1].astype(mx.float32),
                         None if b is None else b[:r1].astype(mx.float32), group_size=g, bits=bits)


class Group:
    """Projections reading the same input, rows stacked: the ANE computes rows [0, Na), the GPU the rest."""

    def __init__(self, mods, rows, name):
        self.mods, self.rows, self.name = mods, rows, name
        self.parts = [(m.weight, m.scales, m.get("biases"), m.group_size, m.bits) for m in mods]
        self.ns = [m.weight.shape[0] for m in mods]
        self.N = sum(self.ns)
        self.K = mods[0].weight.shape[1] * 32 // mods[0].bits
        self.held = None
        for i, m in enumerate(mods):
            m.__dict__["_ane"] = (self, i)
            m.__class__ = AneQuantizedLinear

    @property
    def Na(self):
        return _CAL.get((self.rows, self.N, self.K), (0, 1))[0]

    @property
    def kc(self):
        return _CAL.get((self.rows, self.N, self.K), (0, 1))[1]

    def takes(self, Na):
        """Rows [0, a_i) of each member that the ANE computes at a share of Na rows."""
        out, r = [], 0
        for n in self.ns:
            out.append(min(n, max(0, Na - r)))
            r += n
        return out

    def worth(self, S):
        """The split for S rows (the program always runs `rows`), when the times measured at load say it wins."""
        c = _CAL.get(("t", self.rows, self.N, self.K))
        if not self.Na or not c or S > self.rows:
            return False
        t_gpu, t_split = c
        return t_split < t_gpu * S / self.rows

    def write_weights(self, p, scale):
        """The ANE's rows of this group's weights, dequantized in fp32, scaled and cast to fp16 into the surface."""
        ds = [_deq(part, a) for part, a in zip(self.parts, self.takes(p.Na)) if a]
        d = ds[0] if len(ds) == 1 else mx.concatenate(ds, axis=0)
        p.sw = _pow2(mx.abs(d).max(), W_TOP) if scale else _one()
        STATS["weights"] += 1
        return [_write(d, p.w, p.sw, p.kc, p.Na, p.dep)]

    def compute(self, x, kc=None, scale=None):
        """Every member's output for x [1, S, K] (the ANE and the GPU at once) and the ANE's columns in fp32."""
        kc = self.kc if kc is None else kc
        scale = SCALE if scale is None else scale
        S, K = x.shape[1], self.K
        p = _prog(self.rows, self.Na, K, kc)
        if SYNC and p.last:
            mx.eval(p.last)             # the previous call's reads of this surface are done before it is rewritten
        # Async: the rewrites wait in the GPU stream for the previous run and its readers (p.last).
        p.dep = None if SYNC or not p.last else mx.concatenate([o.reshape(-1)[:1] for o in p.last])
        toks = self.write_weights(p, scale) if p.owner != (self, scale) else []
        p.owner = (self, scale)
        x2 = x.reshape(S, K)
        sx = _pow2(mx.abs(x2).max(), X_TOP) if scale else _one()
        toks.append(_write(x2, p.x, sx, kc, p.S, p.dep))
        takes = self.takes(p.Na)
        gpu = [_qmm(x, part, a) if a < n else None for part, a, n in zip(self.parts, takes, self.ns)]
        STATS["calls"] += 1
        if SYNC:
            t0 = time.perf_counter()
            mx.async_eval(toks)
            mx.async_eval([g for g in gpu if g is not None])
            mx.eval(toks)
            t1 = time.perf_counter()
            p.p.run()
            STATS["wait_ms"] += (t1 - t0) * 1e3
            STATS["ane_ms"] += (time.perf_counter() - t1) * 1e3
            yall = p.y
        else:
            # No host wait: the writes and the ANE's start go to the GPU now, so its own rows queue behind them,
            # and the GPU waits for the ANE only where the output is read.
            tok, v = p.p.submit(toks)
            mx.async_eval(tok)
            yall = p.p.wait(p.y, v, [tok] + [g for g in gpu if g is not None])
        y = yall[:, :S]
        if kc == 1 and not scale:
            y32 = y.astype(mx.float32)          # lazy: read only by audit()
        else:
            y32 = y.astype(mx.float32).sum(axis=0, keepdims=True) if kc > 1 else y.astype(mx.float32)
            if scale:
                y32 = y32 * mx.reciprocal(sx * p.sw)
            y = y32
        outs, r = [], 0
        for g, a, n in zip(gpu, takes, self.ns):
            ya = y[..., r:r + a].astype(x.dtype) if a else None
            outs.append(ya if g is None else (g if ya is None else mx.concatenate([ya, g], axis=-1)))
            r += n
        p.last = list(outs)
        return outs, y32

    def audit(self, x):
        """Stock outputs, with each variant's errors on the ANE's rows and the ranges folded into AUDIT (MOE_ANE_AUDIT)."""
        ref = [_qmm(x, part) for part in self.parts]
        takes = self.takes(self.Na)
        x32 = x.astype(mx.float32)
        true = [x32 @ _deq(part, a).T if a else None for part, a in zip(self.parts, takes)]
        mx.eval(ref, true)
        rec = AUDIT.setdefault(self.name, {"x_absmax": 0.0, "rows": 0, "mods": {}})
        rec["x_absmax"] = max(rec["x_absmax"], mx.abs(x).max().item())
        rec["rows"] += x.shape[1]

        def stat(a, b):
            d = mx.abs(a.astype(mx.float32) - b.astype(mx.float32))
            return [d.max(), mx.abs(b).max(), d.mean(), mx.abs(b).astype(mx.float32).mean(),
                    (~mx.isfinite(a)).sum()]
        vals = {}
        for i, (a, t) in enumerate(zip(takes, true)):
            if a:
                vals[f"{i}.mlx~true"] = stat(ref[i][..., :a], t)
        for kc, sc in [v for v in VARIANTS if v[0] <= 4]:      # k8 is the probe's (its surfaces are 8x)
            outs, y32 = self.compute(x, kc, sc)
            raw = y32
            r = 0
            for i, (a, t) in enumerate(zip(takes, true)):
                if a:
                    v = f"k{kc}{'s' if sc else ''}"
                    vals[f"{i}.{v}~mlx"] = stat(outs[i][..., :a], ref[i][..., :a])
                    vals[f"{i}.{v}~true"] = stat(outs[i][..., :a], t)
                    vals[f"{i}.{v}.fp32~true"] = stat(raw[0, ..., r:r + a], t)
                r += self.ns[i]
            mx.eval(list(vals.values()))
        for k, (dmax, rmax, dmean, rmean, bad) in vals.items():
            i, v = k.split(".", 1)
            m = rec["mods"].setdefault(i, {"n_ane": takes[int(i)], "n": self.ns[int(i)], "y_absmax": 0.0})
            e = m.setdefault(v, {"max_abs": 0.0, "ref_absmax": 0.0, "sum_mean_abs": 0.0, "sum_ref_mean": 0.0,
                                 "chunks": 0, "nonfinite": 0})
            e["max_abs"] = max(e["max_abs"], dmax.item())
            e["ref_absmax"] = max(e["ref_absmax"], rmax.item())
            e["sum_mean_abs"] += dmean.item()
            e["sum_ref_mean"] += rmean.item()
            e["chunks"] += 1
            e["nonfinite"] += int(bad.item())
            if v == "mlx~true":
                m["y_absmax"] = max(m["y_absmax"], rmax.item())
        return ref

    def call(self, i, x, mod):
        if self.held is None or self.held[0] is not x:
            if not self.worth(x.shape[1]):
                STATS["stock"] += 1
                return nn.QuantizedLinear.__call__(mod, x)
            self.held = [x, self.audit(x) if AUDIT_ON else self.compute(x)[0]]
        outs = self.held[1]
        y, outs[i] = outs[i], None
        if all(o is None for o in outs):
            self.held = None
        return y


class AneQuantizedLinear(nn.QuantizedLinear):
    """A QuantizedLinear in a Group: inside a layer-major pass its call goes through the split, else stock."""

    def __call__(self, x):
        g = self.__dict__.get("_ane")
        if not ACTIVE or g is None or x.ndim != 3 or x.shape[0] != 1:
            return super().__call__(x)
        return g[0].call(g[1], x, self)


def _time(fn, n=5):
    fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return sorted(ts)[len(ts) // 2] * 1e3


def _search(g, x, kc, t_gpu):
    """(n_ane, split ms, tried): the balance from the two rates alone, then neighbours 2.5% of N apart (6 splits at most)."""
    key = (g.rows, g.N, g.K)
    r64 = lambda v: max(64, min(g.N - 64, int(round(v / 64)) * 64))
    na0 = r64(g.N / 2)
    _CAL[key] = (na0, kc)
    mx.eval(g.compute(x, kc, False)[0])
    t_a = _time(_PROGS[(g.rows, na0, g.K, kc)].p.run)
    _PROGS.pop((g.rows, na0, g.K, kc)).close()
    mx.clear_cache()
    c_g, c_a = t_gpu / g.N, t_a / na0
    step = r64(g.N / 40)
    res, best, na = {}, None, r64(g.N * c_g / (c_g + c_a))
    for _ in range(6):
        if na in res:
            break
        _CAL[key] = (na, kc)
        res[na] = _time(lambda: mx.eval(g.compute(x, kc, False)[0]))
        g.held = None
        # One program at a time: each holds wired memory outside the footprint (and at most ~117 load a process).
        _PROGS.pop((g.rows, na, g.K, kc)).close()
        mx.clear_cache()
        best = na if best is None or res[na] < res[best] else best
        na = r64(best + step) if r64(best + step) not in res else r64(best - step)
    return best, res[best], {"ane_alone_ms_half": t_a, "splits": res}


def calibrate(g):
    """Keeps the fastest share and split-K (MOE_ANE_SPLITK, else 1, 2, 4) timed on this group's weights, cached on disk."""
    key = (g.rows, g.N, g.K)
    if key in _CAL:
        return
    dev = mx.device_info().get("device_name", "")
    tag = json.dumps([_V, dev, mx.__version__, list(key), KC, SCALE,
                      [list(p[0].shape) + [p[4]] for p in g.parts]])
    path = Path.home() / ".cache" / "moe-ssd-offload" / f"anesplit-{hashlib.sha1(tag.encode()).hexdigest()[:16]}.json"
    try:
        c = json.loads(path.read_text())
        _CAL[key], _CAL[("t",) + key] = tuple(c["cal"]), tuple(c["t"])
        return
    except (OSError, ValueError, KeyError):
        pass
    x = (mx.random.normal((1, g.rows, g.K)) * 0.5).astype(g.parts[0][1].dtype)
    mx.eval(x)
    t_gpu = _time(lambda: mx.eval([_qmm(x, p) for p in g.parts]))
    found = {kc: _search(g, x, kc, t_gpu) for kc in ((KC,) if KC else (1, 2, 4))}
    kc = min(found, key=lambda k: found[k][1])
    _CAL[key], _CAL[("t",) + key] = (found[kc][0], kc), (t_gpu, found[kc][1])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"cal": [found[kc][0], kc], "t": [t_gpu, found[kc][1]], "key": tag,
                                "tried": {str(k): v for k, v in found.items()}}))


def install(model, rows):
    """Groups and calibrates a qwen4_exp model's big input and output projections; returns how many (0 when off)."""
    q4 = sys.modules.get("mlx_lm.models.qwen4_exp")
    if not ON or not mx.metal.is_available() or q4 is None:
        return 0
    try:
        import _ane_mlx  # noqa: F401
    except ImportError as e:
        print(f"  ane split: the extension is not built ({e}); run native/build.sh", flush=True)
        return 0
    groups, li = [], -1
    q = lambda ms: all(isinstance(m, nn.QuantizedLinear) and m.mode == "affine" for m in ms)
    for m in model.modules():
        if isinstance(m, q4.GatedDeltaNet):
            sets = {"gdn_in": [m.in_proj_qkv, m.in_proj_z], "gdn_out": [m.out_proj]}
        elif isinstance(m, q4.Attention):
            sets = {"attn_in": [m.indexer.index_qk_proj, m.q_proj, m.k_proj, m.v_proj], "attn_out": [m.o_proj]}
        else:
            continue
        li += 1
        groups += [Group(s, rows, f"{li}.{k}") for k, s in sets.items() if q(s)]
    import atexit
    atexit.register(close_all)
    seen = {}
    for g in groups:
        seen.setdefault((g.N, g.K), g)
    for g in seen.values():
        calibrate(g)
    return len(groups)


def summary():
    return {f"{k[1]}x{k[2]}": {"n_ane": v[0], "kc": v[1], "t_gpu_ms": round(_CAL[("t",) + k][0], 2),
                                "t_split_ms": round(_CAL[("t",) + k][1], 2)}
            for k, v in _CAL.items() if k[0] != "t"}
