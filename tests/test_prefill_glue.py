#!/usr/bin/env python3
"""layer_glue's prefill-row kernels (MOE_GLUE_PREFILL) against the stock lines they replace, bit for
bit, at Flash-Next's shapes over 1-2048 rows (the chunk widths of a 16 GB Mac), each kernel with a mutant the
check must catch; then the wired layer-major path (stream_fast._lm_parts) on a 4-layer model at
bias 0 and 0.5. Needs mlx_lm with qwen4_exp. No model.

    python tests/test_prefill_glue.py [--rows 1,2,3,7,1306,1687,2048]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["MOE_FUSE_HC"] = "1"

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

import fuse_proj
import fused_router
import layer_glue as lg
import stream_fast
try:
    from mlx_lm.models import qwen4_exp as q4
except ImportError:
    print("test_prefill_glue: skipped (needs mlx-lm with qwen4_exp, see requirements.txt)")
    sys.exit(0)
from mlx_lm.models.cache import ArraysCache

ARGS = q4.ModelArgs.from_dict({"model_type": "qwen4_exp", "text_config": {}}).text


def ndiff(a, b):
    """Elements whose bits differ (shape or dtype apart: all of them)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return max(a.size, b.size)
    v = mx.uint16 if a.dtype in (mx.bfloat16, mx.float16) else mx.uint32
    return int((a.view(v) != b.view(v)).sum().item())


def perturb(m, scale=0.02):
    """Random weights at a layer's scale, every Linear 4-bit g32 as the pack's projections."""
    new = []
    for k, v in tree_flatten(m.parameters()):
        if k.endswith("norm.weight"):
            v = 1 + 0.2 * mx.random.normal(v.shape)
        elif k.endswith("conv1d.weight"):
            v = 0.4 * mx.random.normal(v.shape)
        elif k.endswith(("A_log", "dt_bias", "embed_tokens.weight")):
            v = mx.random.normal(v.shape)
        else:
            v = scale * mx.random.normal(v.shape)
        new.append((k, v.astype(mx.bfloat16)))
    m.update(tree_unflatten(new))
    nn.quantize(m, group_size=32, bits=4,
                class_predicate=lambda p, x: hasattr(x, "to_quantized") and not p.endswith("mlp.gate"))
    m.eval()
    mx.eval(m.parameters())


def rand(shape, lo, hi):
    """bf16 normals, each row at its own log-uniform scale in [lo, hi]."""
    s = mx.exp(mx.random.uniform(np.log(lo), np.log(hi), (*shape[:2], *([1] * (len(shape) - 2)))))
    return (mx.random.normal(shape) * s).astype(mx.bfloat16)


# ------------------------------------------------------------------------- the checks
def check_hc(R, hc):
    hyper = rand((1, R, 10240), 0.1, 30)
    ref, got = hc(hyper), lg.hc_rows(hc, hyper)
    return {"mixed": ndiff(got[0], ref[0]), "inject": ndiff(got[2], ref[2])}


def check_comb(R, hc, hc2):
    """The attention half's combine inside the MLP half's norm kernel, and the layer's last combine."""
    hp, x = rand((1, R, 10240), 0.1, 30), rand((1, R, 2560), 0.1, 10)
    inj = (2 * mx.sigmoid(mx.random.normal((1, R, 4)))).astype(mx.bfloat16)
    h = hp + (x[..., None, :] * inj[..., None]).reshape(*x.shape[:-1], -1)
    ref, got = hc2(h), lg.hc_rows(hc2, None, comb=(hp, x, inj))
    return {"h": ndiff(got[1], h), "mixed": ndiff(got[0], ref[0]), "inject": ndiff(got[2], ref[2]),
            "comb": ndiff(lg.comb_rows(hp, inj, x), h)}


def check_gdn(R, m):
    out = {"out": 0, "conv": 0, "state": 0}
    for warm in (False, True):
        ca, cb = ArraysCache(size=2), ArraysCache(size=2)
        if warm:                # a later chunk: the conv and delta states of the one before
            cs = rand((1, 3, m.conv_dim), 0.3, 3)
            st = mx.random.normal((1, m.n_v, m.dv, m.dk)) * 0.05
            ca[0], ca[1], cb[0], cb[1] = cs, st, cs, st
        x = rand((1, R, 2560), 0.3, 3)
        ya, yb = m(x, None, ca), lg.gdn_rows(m, x, cb)
        mx.eval(ya, yb, ca.cache, cb.cache)
        out["out"] += ndiff(yb, ya)
        out["conv"] += ndiff(cb[0], ca[0])
        out["state"] += ndiff(cb[1], ca[1])
    return out


def check_moe(R, blk):
    x = rand((1, R, 2560), 0.1, 5)
    y = rand((1, R + 5, 10, 2560), 0.01, 5)[:, 3:3 + R]       # a chunk's slice of the pass's rows
    sc = mx.softmax(mx.random.normal((1, R, 10)) * 2, axis=-1, precise=True)
    ref = ((y * sc[..., None]).sum(axis=-2).astype(mx.bfloat16)
           + mx.sigmoid(blk.shared_expert_gate(x)) * blk.shared_expert(x))
    return {"end": ndiff(lg.moe_end_rows(y, sc, x, blk), ref)}


def check_route(R):
    out = {"inds": 0, "scores": 0}
    logits = mx.random.normal((1, R, 512)) * 3
    mask = (mx.random.uniform(shape=(512,)) < 0.3).astype(mx.float32)
    for lam, tau in ((0.0, 0.0), (0.02, 0.2), (0.5, 0.2), (0.5, 0.0)):
        ri, rs = fused_router.glue_logits(logits, mask if lam else mx.zeros((512,)), lam, tau, 10)
        gi, gs = lg.route_rows(logits, mask if lam else mx.zeros((512,)), lam, tau, 10)
        out["inds"] += ndiff(gi, ri.astype(mx.uint32))
        out["scores"] += ndiff(gs, rs)
    return out


# --------------------------------------------------------------------------- mutants
MUTANTS = [     # kernel, its (old, new) text, the check that must see it
    ("pre_rows", [("normed[base + i] = v * w[(gid % HC) * D + lid * 4 + i];",
                   "normed[base + i] = static_cast<T>(tx[i] * local_inv_mean[0] * "
                   "float(w[(gid % HC) * D + lid * 4 + i]));")], "hc"),
    ("mid_rows", [("s[r * ND + c] = silu[as_type<ushort>(v)];",
                   "s[r * ND + c] = static_cast<T>(float(v) / (1.0f + metal::exp(-float(v))));")], "hc"),
    ("mix_rows", [("T tot = (sig[as_type<ushort>(u[b])] * nrm[b]) + T(0);",
                   "float tot = float(sig[as_type<ushort>(u[b])]) * float(nrm[b]);"),
                  ("mixed[t] = tot * T(1.0f / float(HC));",
                   "mixed[t] = static_cast<T>(tot * (1.0f / float(HC)));")], "hc"),
    ("pre_comb_rows", [("hv[i] = static_cast<T>(hyper[base + i] + p);",
                        "hv[i] = static_cast<T>(float(hyper[base + i]) + "
                        "float(xr[(gid / HC) * D + lid * 4 + i]) * float(inj[gid]));")], "comb"),
    ("comb_rows", [("T p = static_cast<T>(x[(rs / HC) * D + (t - rs * D)] * inj[rs]);",
                    "float p = float(x[(rs / HC) * D + (t - rs * D)]) * float(inj[rs]);")], "comb"),
    ("gdn_pre_rows", [("sv[i] = silu[as_type<ushort>(cv)];",
                       "sv[i] = static_cast<T>(float(cv) / (1.0f + metal::exp(-float(cv))));")], "gdn"),
    ("gdn_post", [("out[hv * DV + e] = static_cast<T>(gz * static_cast<float>(o[i]));",
                   "out[hv * DV + e] = static_cast<T>(gz) * o[i];")], "gdn"),
    ("moe_end_rows", [("part[q % 8] = p + part[q % 8];", "part[q % 4] = p + part[q % 4];")], "moe"),
    ("route_rows", [("scores[row * KT + b2 + i] = sl[i] * inv;", "scores[row * KT + b2 + i] = sl[i] / nz;")],
     "route"),
]


def mutated(kname, pairs, fn):
    """fn() with kernel kname's source changed (a kernel of its own name, so MLX compiles it)."""
    ins, outs, src, head = lg._SPECS[kname]
    lg._SPECS[kname + "_mut"] = (ins, outs, lg._sub(src, *pairs), head)
    orig = lg._kern
    lg._kern = lambda name: orig((name[0] + "_mut",) + tuple(name[1:]) if name[0] == kname else name)
    try:
        return fn()
    finally:
        lg._kern = orig


# ---------------------------------------------------------------- the wired layer-major path
def check_wired(bias, widths=(5, 1, 13)):
    """stream_fast._lm_parts per layer, a prompt in chunks of each width (the MoE block called with
    the experts live, as the layer-major combine runs it), stock marks off against MOE_GLUE_PREFILL's
    marks on: the layer stack's output and every cache array bit-equal, and the hooks taken."""
    args = q4.ModelArgs.from_dict({"model_type": "qwen4_exp", "text_config": dict(
        num_hidden_layers=4, num_experts=32, vocab_size=1024, ple_layer_ids=[])})
    model = q4.Model(args)
    perturb(model)
    assert fuse_proj.install_fuse_hc(model) == 8
    lg.PREFILL, on = True, lg.ON
    lg.ON = False
    try:
        assert lg.install(model) == 4, "every layer eligible"
    finally:
        lg.ON = on
    inner = model.language_model.model if hasattr(model, "language_model") else model.model
    layers = inner.layers
    if bias:
        stream_fast._BIAS_BUDGET = 0.2
        for l in layers:
            l.mlp = stream_fast.CacheBiasedMoE(l.mlp, bias)
            l.mlp.block.switch_mlp.__dict__["_lru"] = list(np.random.choice(32, 8, replace=False))
    marks = [m for m in model.modules() if m.__dict__.get("_lgp")]
    blocks = [l.mlp.block if bias else l.mlp for l in layers]
    calls = dict.fromkeys(["hc_rows", "comb_rows", "gdn_rows", "moe_end_rows", "route_rows"], 0)
    saved = {n: getattr(lg, n) for n in calls}
    for n in calls:
        setattr(lg, n, lambda *a, _f=saved[n], _n=n, **k: (calls.__setitem__(_n, calls[_n] + 1), _f(*a, **k))[1])

    def run(ids, w, glue):
        for m in marks:
            m.__dict__["_lgp"] = glue
        cache = model.make_cache()
        h = mx.tile(inner.embed_tokens(ids), (1, 1, inner.hc))
        for l, c, b in zip(layers, cache, blocks):
            idx_c = getattr(c, "indexer", None)
            T, pre, moe_in, moe_out, _ = stream_fast._lm_parts(
                l, (h, inner.rope, "causal", None, c, idx_c, ids, None), {})
            outs = []
            b.switch_mlp.__dict__["_lm"] = True        # the layer-major mark the MoE hooks read
            try:
                for a in range(0, T, w):
                    x, keep = moe_in(pre(a, min(a + w, T)))
                    outs.append(moe_out(l.mlp(x), keep))
                    mx.eval(outs[-1], [v for _, v in tree_flatten(c.state) if isinstance(v, mx.array)])
            finally:
                b.switch_mlp.__dict__.pop("_lm")
            h = mx.concatenate(outs, axis=1)
        arrs = [v for c in cache for _, v in tree_flatten(c.state) if isinstance(v, mx.array)]
        mx.eval(h, arrs)
        return h, arrs

    bad = {"out": 0, "caches": 0}
    try:
        for w in widths:
            ids = mx.array(np.random.randint(0, 1024, (1, 13)))
            (ha, ca), (hb, cb) = run(ids, w, False), run(ids, w, True)
            bad["out"] += ndiff(hb, ha)
            bad["caches"] += sum(ndiff(x, y) for x, y in zip(ca, cb)) + abs(len(ca) - len(cb))
    finally:
        for n, f in saved.items():
            setattr(lg, n, f)
    chunks = sum(-(-13 // w) for w in widths)
    want = {"hc_rows": 8 * chunks, "comb_rows": 4 * chunks, "gdn_rows": 3 * chunks,
            "moe_end_rows": 4 * chunks, "route_rows": 4 * chunks if bias else 0}
    bad["hooks"] = sum(calls[k] != n for k, n in want.items())
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="1,2,3,7,1306,1687,2048")
    ap.add_argument("--mutant-rows", type=int, default=512)
    a = ap.parse_args()
    mx.random.seed(0)
    np.random.seed(0)
    hc, hc2 = q4.GatedResidual(ARGS), q4.GatedResidual(ARGS)
    gdn = q4.GatedDeltaNet(ARGS)
    blk = q4.SparseMoeBlock(q4.ModelArgs.from_dict(
        {"model_type": "qwen4_exp", "text_config": {"num_experts": 16}}).text)
    for m in (hc, hc2, gdn, blk):
        perturb(m)
    assert fuse_proj.install_fuse_hc(hc) == 1 and fuse_proj.install_fuse_hc(hc2) == 1
    assert lg.hc_eligible(hc) and lg.gdn_eligible(gdn)
    checks = {"hc": lambda R: check_hc(R, hc), "comb": lambda R: check_comb(R, hc, hc2),
              "gdn": lambda R: check_gdn(R, gdn), "moe": lambda R: check_moe(R, blk),
              "route": check_route}
    res = {}
    for R in [int(r) for r in a.rows.split(",")]:
        res[R] = {k: f(R) for k, f in checks.items()}
        print(f"rows {R}: {res[R]}", flush=True)
    mut = {}
    for kname, pairs, chk in MUTANTS:
        got = mutated(kname, pairs, lambda: checks[chk](a.mutant_rows))
        mut[kname] = sum(got.values())
    print(f"mutants (values differing, each must be > 0): {mut}", flush=True)
    wired = {f"bias {b}": check_wired(b) for b in (0.0, 0.5)}
    print(f"wired: {wired}", flush=True)
    print(f"mlx {mx.__version__}: rows {a.rows}")
    assert all(v == 0 for r in res.values() for c in r.values() for v in c.values()), \
        "a kernel differs from the stock lines"
    assert all(v > 0 for v in mut.values()), "a mutant went unseen"
    assert all(v == 0 for r in wired.values() for v in r.values()), "the wired path differs"
    print("ok: every kernel bit-equal at every row count, every mutant caught, the wired path bit-equal")


if __name__ == "__main__":
    main()
