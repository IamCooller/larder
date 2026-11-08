#!/usr/bin/env python3
"""layer_glue's kernels against the stock lines they replace, bit for bit, at Flash-Next's shapes
(hidden 2560, hc 4, low rank 320, 4-bit g32; delta net 48 v / 16 k heads of 128; top-10 experts).
Uses qwen4_exp's own modules when mlx_lm has them, else copies of their lines.
No model.

    python tests/test_layer_glue.py [--rows 100]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["MOE_FUSE_HC"] = "1"

import mlx.core as mx
import mlx.nn as nn
import numpy as np

import fuse_proj
import layer_glue as lg
from mlx_lm.models.gated_delta import gated_delta_update

try:
    from mlx_lm.models import qwen4_exp as q4
    SRC = "qwen4_exp"
except ImportError:
    q4 = None
    SRC = "copy"


class Args:
    hidden_size, hc_count, hc_lowrank, rms_norm_eps = 2560, 4, 320, 1e-6
    linear_num_value_heads, linear_num_key_heads = 48, 16
    linear_key_head_dim = linear_value_head_dim = 128
    linear_conv_kernel_dim, output_gate_type = 4, "sigmoid"


class RMSNorm(nn.Module):               # qwen4_exp.RMSNorm
    def __init__(self, dim, group_size=None, eps=1e-6):
        super().__init__()
        self.weight = mx.ones(dim)
        self.eps = eps
        self.group_size = group_size

    def __call__(self, x):
        if self.group_size is None:
            return mx.fast.rms_norm(x, self.weight, self.eps)
        shape = x.shape
        x = x.reshape(*shape[:-1], -1, self.group_size)
        x = mx.fast.rms_norm(x, None, self.eps).reshape(shape)
        return x * self.weight


class GatedResidual(nn.Module):         # qwen4_exp.GatedResidual
    def __init__(self, args, use_combine=True):
        super().__init__()
        self.hc = args.hc_count
        self.d = args.hidden_size
        hc_dim = self.hc * self.d
        self.hc_norm = RMSNorm(hc_dim, group_size=self.d, eps=args.rms_norm_eps)
        self.input_mix_weight_down = nn.Linear(hc_dim, args.hc_lowrank, bias=False)
        self.input_mix_weight_up = nn.Linear(args.hc_lowrank, hc_dim, bias=False)
        self.block_inject_weight = nn.Linear(hc_dim, self.hc, bias=False) if use_combine else None

    def __call__(self, hyper):
        normed = self.hc_norm(hyper)
        w = nn.silu(self.input_mix_weight_down(normed) / self.hc)
        w = mx.sigmoid(self.input_mix_weight_up(w))
        w = w.reshape(*w.shape[:-1], self.hc, self.d)
        mixed = (w * normed.reshape(*normed.shape[:-1], self.hc, self.d)).mean(axis=-2)
        if self.block_inject_weight is None:
            return mixed
        inject = 2 * mx.sigmoid(self.block_inject_weight(normed) / self.hc)
        return mixed, hyper, inject


class RMSNormGated(nn.Module):          # qwen4_exp.RMSNormGated
    def __init__(self, dim, eps=1e-6, activation="sigmoid"):
        super().__init__()
        self.weight = mx.ones(dim)
        self.eps = eps
        self.activation = activation

    def __call__(self, x, gate=None):
        out = mx.fast.rms_norm(x, self.weight, self.eps)
        g = mx.sigmoid(gate.astype(mx.float32))
        return (g * out.astype(mx.float32)).astype(x.dtype)


class GatedDeltaNet(nn.Module):         # qwen4_exp.GatedDeltaNet, decode lines (no mask, a cache)
    def __init__(self, args):
        super().__init__()
        self.n_v, self.n_k = args.linear_num_value_heads, args.linear_num_key_heads
        self.dk, self.dv = args.linear_key_head_dim, args.linear_value_head_dim
        self.key_dim, self.value_dim = self.dk * self.n_k, self.dv * self.n_v
        self.conv_kernel_size = args.linear_conv_kernel_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim
        d = args.hidden_size
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, bias=False,
                                kernel_size=self.conv_kernel_size, groups=self.conv_dim, padding=0)
        self.in_proj_qkv = nn.Linear(d, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(d, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(d, self.n_v, bias=False)
        self.in_proj_a = nn.Linear(d, self.n_v, bias=False)
        self.dt_bias = mx.ones(self.n_v)
        self.A_log = mx.zeros(self.n_v)
        self.norm = RMSNormGated(self.dv, eps=args.rms_norm_eps, activation=args.output_gate_type)
        self.out_proj = nn.Linear(self.value_dim, d, bias=False)

    def __call__(self, x, mask, cache):
        B, S, _ = x.shape
        mixed_qkv = self.in_proj_qkv(x)
        z = self.in_proj_z(x).reshape(B, S, self.n_v, self.dv)
        b = self.in_proj_b(x)
        a = self.in_proj_a(x)
        conv_input = mx.concatenate([cache[0], mixed_qkv], axis=1)
        cache[0] = mx.contiguous(conv_input[:, -(self.conv_kernel_size - 1):, :])
        conv_out = nn.silu(self.conv1d(conv_input))
        q, k, v = mx.split(conv_out, [self.key_dim, 2 * self.key_dim], axis=-1)
        q = q.reshape(B, S, self.n_k, self.dk)
        k = k.reshape(B, S, self.n_k, self.dk)
        v = v.reshape(B, S, self.n_v, self.dv)
        inv_scale = self.dk ** -0.5
        q = (inv_scale ** 2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
        out, state = gated_delta_update(q, k, v, a, b, self.A_log, self.dt_bias, cache[1], mask,
                                        use_kernel=not self.training)
        cache[1] = state
        return self.out_proj(self.norm(out, z).reshape(B, S, -1))


if q4 is not None:
    GatedResidual, GatedDeltaNet = q4.GatedResidual, q4.GatedDeltaNet


class Cache:                            # the two slots and advance() of an ArraysCache
    lengths = None

    def __init__(self, a, b):
        self.cache = [a, b]

    def __getitem__(self, i):
        return self.cache[i]

    def __setitem__(self, i, v):
        self.cache[i] = v

    def advance(self, n):
        pass


def bits(a):
    return np.array(a.view(mx.uint16 if a.dtype in (mx.bfloat16, mx.float16) else mx.uint32))


def same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(bits(a), bits(b))


def perturb(m, scale):
    """Random weights at a layer's scale, quantized as the pack is (4-bit g32 projections)."""
    from mlx.utils import tree_flatten, tree_unflatten
    new = []
    for k, v in tree_flatten(m.parameters()):
        if k.endswith("norm.weight"):
            v = 1 + 0.2 * mx.random.normal(v.shape)
        elif k.endswith("conv1d.weight"):
            v = 0.4 * mx.random.normal(v.shape)
        elif k.endswith(("A_log", "dt_bias")):
            v = mx.random.normal(v.shape)
        else:
            v = scale * mx.random.normal(v.shape)
        new.append((k, v.astype(mx.bfloat16)))
    m.update(tree_unflatten(new))
    nn.quantize(m, group_size=32, bits=4, class_predicate=lambda p, x: isinstance(x, nn.Linear))
    m.eval()                            # as mlx_lm loads it: the delta kernel, not the ops
    mx.eval(m.parameters())


def test_hc(rows):
    bad = {"mixed": 0, "inject": 0, "final": 0, "comb_h": 0, "comb_n": 0}
    hcs = []
    for use in (True, False):
        m = GatedResidual(Args, use_combine=use)
        perturb(m, 0.02)
        if use:
            assert fuse_proj.install_fuse_hc(m) == 1
        assert lg.hc_eligible(m)
        hcs.append(m)
    hc, mixer = hcs
    for r in range(rows):
        hyper = (mx.random.normal((1, 1, 10240)) * float(np.random.uniform(0.1, 30))).astype(mx.bfloat16)
        ref = hc(hyper)
        got = lg.hc_call(hc, hyper)
        bad["mixed"] += not same(got[0], ref[0])
        bad["inject"] += not same(got[2], ref[2])
        bad["final"] += not same(lg.hc_call(mixer, hyper), mixer(hyper))
        # the residual combine in the next half's norm kernel
        x = (mx.random.normal((1, 1, 2560)) * float(np.random.uniform(0.1, 10))).astype(mx.bfloat16)
        h_ref = ref[1] + (x[..., None, :] * ref[2][..., None]).reshape(*x.shape[:-1], -1)
        h, n = lg.hc_pre(hc, None, comb=(ref[1], x, ref[2]))
        bad["comb_h"] += not same(h, h_ref)
        bad["comb_n"] += not same(n, hc.hc_norm(h_ref))
    return bad


def test_gdn(rows):
    bad = {"out": 0, "conv": 0, "state": 0}
    m = GatedDeltaNet(Args)
    perturb(m, 0.02)
    assert lg.gdn_eligible(m), "gdn shapes"
    conv = (mx.random.normal((1, 3, m.conv_dim)) * 0.5).astype(mx.bfloat16)
    state = mx.random.normal((1, m.n_v, m.dv, m.dk)) * 0.05
    for r in range(rows):
        x = (mx.random.normal((1, 1, 2560)) * float(np.random.uniform(0.3, 3))).astype(mx.bfloat16)
        c_ref, c_got = Cache(conv, state), Cache(conv, state)
        y_ref = m(x, None, c_ref)
        y_got = lg.gdn_step(m, x, c_got)
        mx.eval(y_ref, y_got, c_ref.cache, c_got.cache)
        bad["out"] += not same(y_got, y_ref)
        bad["conv"] += not same(c_got[0], c_ref[0])
        bad["state"] += not same(c_got[1], c_ref[1])
        conv, state = c_ref[0], c_ref[1]        # walk the state forward
    return bad


def test_moe(rows):
    bad = {"silu_mul": 0, "end": 0}
    for r in range(rows):
        s = float(np.random.uniform(0.01, 5))
        g = (mx.random.normal((1, 1, 640)) * s).astype(mx.bfloat16)
        u = (mx.random.normal((1, 1, 640)) * s).astype(mx.bfloat16)
        bad["silu_mul"] += not same(lg.silu_mul(g, u), nn.silu(g) * u)
        y = (mx.random.normal((1, 1, 10, 2560)) * s).astype(mx.bfloat16)
        sc = mx.softmax(mx.random.normal((1, 1, 10)) * 2, axis=-1, precise=True)
        sh = (mx.random.normal((1, 1, 2560)) * s).astype(mx.bfloat16)
        sg = (mx.random.normal((1, 1, 1)) * 3).astype(mx.bfloat16)
        ref = (y * sc[..., None]).sum(axis=-2).astype(mx.bfloat16) + mx.sigmoid(sg) * sh
        bad["end"] += not same(lg.moe_end(y, sc, sh, sg), ref)
    return bad


def test_attn(rows):
    """attn_step against qwen4_exp's Attention (Flash-Next's 24 / 2 heads of 256, rope on 64), one row at
    a time on two caches prefilled alike: q / k (the cache's keys) and the output bit-equal."""
    if q4 is None:
        return {"skipped (no qwen4_exp)": 0}
    args = q4.ModelArgs.from_dict({"model_type": "qwen4_exp", "text_config": {}}).text
    m = q4.Attention(args)
    perturb(m, 0.02)
    assert lg.attn_eligible(m)
    rope = q4.RotaryEmbedding(int(args.head_dim * args.partial_rotary_factor), args.rope_theta)
    ca, cb = q4._AttnCache(), q4._AttnCache()
    x = (mx.random.normal((1, 7, 2560)) * 2).astype(mx.bfloat16)
    mx.eval(m(x, rope, "causal", ca, ca.indexer), m(x, rope, "causal", cb, cb.indexer))
    bad = {"out": 0, "keys": 0}
    for r in range(rows):
        x = (mx.random.normal((1, 1, 2560)) * float(np.random.uniform(0.1, 20))).astype(mx.bfloat16)
        ya, yb = m(x, rope, None, ca, ca.indexer), lg.attn_step(m, x, rope, None, cb, cb.indexer)
        mx.eval(ya, yb)
        bad["out"] += not same(ya, yb)
        bad["keys"] += not (same(ca.keys[..., :ca.offset, :], cb.keys[..., :cb.offset, :])
                            and same(ca.values[..., :ca.offset, :], cb.values[..., :cb.offset, :]))
    return bad


def test_model(steps):
    """The wired path, real calls: install()'s hooks on qwen4_exp's own modules. A 4-layer model at
    Flash-Next's shapes (3 delta nets, 1 attention with its QSA indexer; 16 experts, top 10; random 4-bit g32 weights, the
    HCs' inject stacked as the engine does), a 12-token prefill, then `steps` greedy decode rows, each
    through the stock calls and the hooked ones on caches of their own: logits and caches bit-equal."""
    if q4 is None:
        return {"skipped (no qwen4_exp)": 0}
    from mlx.utils import tree_flatten, tree_unflatten
    args = q4.ModelArgs.from_dict({"model_type": "qwen4_exp", "text_config": dict(
        num_hidden_layers=4, num_experts=16, vocab_size=1024, ple_layer_ids=[])})
    model = q4.Model(args)
    new = []
    for k, v in tree_flatten(model.parameters()):
        if k.endswith("norm.weight"):
            v = 1 + 0.2 * mx.random.normal(v.shape)
        elif k.endswith("conv1d.weight"):
            v = 0.4 * mx.random.normal(v.shape)
        elif k.endswith(("A_log", "dt_bias", "embed_tokens.weight")):
            v = mx.random.normal(v.shape)
        else:
            v = 0.02 * mx.random.normal(v.shape)
        new.append((k, v.astype(mx.bfloat16)))
    model.update(tree_unflatten(new))
    nn.quantize(model, group_size=32, bits=4,
                class_predicate=lambda p, m: hasattr(m, "to_quantized") and not p.endswith("mlp.gate"))
    model.eval()
    mx.eval(model.parameters())
    assert fuse_proj.install_fuse_hc(model) == 8
    lg.ON = True
    assert lg.install(model) == 4, "every layer eligible"
    marks = [m for m in model.modules() if m.__dict__.get("_lg")]
    calls = dict.fromkeys(["_layer", "gdn_step", "attn_step", "silu_mul", "moe_end", "hc_call"], 0)
    for name in calls:              # count the hooked paths taken (the hooks look these names up)
        def counted(*a, _f=getattr(lg, name), _n=name, **k):
            calls[_n] += 1
            return _f(*a, **k)
        setattr(lg, name, counted)

    def run(ids, cache, hooked):
        for m in marks:
            m.__dict__["_lg"] = hooked
        out = model(ids, cache)
        mx.eval(out, arrays(cache))
        return out

    def arrays(cache):
        return [v for c in cache for _, v in tree_flatten(c.state) if isinstance(v, mx.array)]

    bad = {"logits": 0, "caches": 0}
    ca, cb = model.make_cache(), model.make_cache()
    ids = mx.array(np.random.randint(0, 1024, (1, 12)))
    la, lb = run(ids, ca, False), run(ids, cb, True)        # a prefill: stock either way
    bad["logits"] += not same(la, lb)
    for _ in range(steps):
        ids = mx.argmax(la[:, -1:], axis=-1)
        la, lb = run(ids, ca, False), run(ids, cb, True)
        bad["logits"] += not same(la, lb)
        xa, xb = arrays(ca), arrays(cb)
        bad["caches"] += len(xa) != len(xb) or not all(same(x, y) for x, y in zip(xa, xb))
    # every decode row: 4 layers, 3 delta nets, 1 attention, 4 shared experts and MoE ends, the final mixer
    want = {"_layer": 4, "gdn_step": 3, "attn_step": 1, "silu_mul": 4, "moe_end": 4, "hc_call": 1}
    bad["hooks"] = sum(calls[k] != steps * n for k, n in want.items())
    return bad


def test_default():
    """MOE_GLUE is on unless it is 0, and only where Metal is (CUDA and CPU builds keep the stock ops)."""
    import subprocess
    code = "import layer_glue, exl3_experts; print(layer_glue.ON, exl3_experts.PREP)"
    want = mx.metal.is_available()
    bad = 0
    for val, on in ((None, True), ("1", True), ("0", False)):
        env = {k: v for k, v in os.environ.items() if k != "MOE_GLUE"}
        if val is not None:
            env["MOE_GLUE"] = val
        out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                             cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).stdout.split()
        bad += out != [str(on), str(on and want)]
    return {"default": bad}


def test_switch():
    """The prefill switch is measured with the prep kernels off (MOE_GLUE must not move it), then PREP is as it was."""
    import tempfile
    from pathlib import Path
    import exl3_experts as ex
    saved = ex.PREP, ex.F16MM, ex.CALIB_DIR, ex._measure, ex._COST
    seen = []
    try:
        ex.PREP, ex.F16MM = True, False
        ex.CALIB_DIR = Path(tempfile.mkdtemp())
        ex._measure = lambda *a: (seen.append(ex.PREP), [1e-6, 1e-6, 1e-4, 1e-4, 1e-6])[1]
        ex.calibrate(256, 128, 48, 16)
        return {"switch": int(seen != [False] or ex.PREP is not True)}
    finally:
        ex.PREP, ex.F16MM, ex.CALIB_DIR, ex._measure, ex._COST = saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=100)
    a = ap.parse_args()
    mx.random.seed(0)
    np.random.seed(0)
    res = {"hc": test_hc(a.rows), "gdn": test_gdn(a.rows), "moe": test_moe(a.rows), "attn": test_attn(a.rows),
           "model": test_model(max(8, a.rows // 4)), "env": test_default(), "calib": test_switch()}
    print(f"mlx {mx.__version__}, modules: {SRC}, {a.rows} rows each: {res}")
    assert all(v == 0 for r in res.values() for v in r.values()), "a kernel differs from the stock lines"
    print("ok: every output bit-equal")


if __name__ == "__main__":
    main()
