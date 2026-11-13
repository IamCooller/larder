#!/usr/bin/env python3
"""GGUF routed experts through the engine (stream_fast's GGUF hook), on a toy packed MoE
(a small packed MoE): a synthetic GGUF of four layers, one ggml type mix each, repacked by
tools/gguf_repack.py beside a trunk without routed experts; against a stock mlx_lm model whose SwitchGLU holds the
reference dequant, per-layer arena, run-ahead, the budget store and its layer-major prefill: a 24-position prefill
(the dense path), three T=1 steps and a T=2 pass (the decode chain), then a 24-position pass with residents.
Logits within bf16's precision of the reference.

    perl -e 'alarm shift; exec @ARGV' 280 python3 tests/test_gguf_engine.py [--hook <dir with a patched stream_fast.py>]
"""
import gc
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools"), str(ROOT / "tests")]
if "--hook" in sys.argv:
    sys.path.insert(0, sys.argv[sys.argv.index("--hook") + 1])

import numpy as np                        # noqa: E402
import mlx.core as mx                     # noqa: E402
import mlx.nn as nn                       # noqa: E402
from mlx.utils import tree_flatten        # noqa: E402
from mlx_lm.models import qwen3_next as q3n               # noqa: E402
from mlx_lm.models.cache import make_prompt_cache         # noqa: E402

import stream_fast                        # noqa: E402  (first: gguf_repack puts the repo root ahead of --hook)
import gguf_experts as ge                 # noqa: E402
import gguf_repack                        # noqa: E402
from test_gguf_repack import write_gguf   # noqa: E402

CFG = dict(
    model_type="qwen3_next", hidden_size=256, num_hidden_layers=4,
    intermediate_size=64, num_attention_heads=2, linear_num_value_heads=2,
    linear_num_key_heads=1, linear_key_head_dim=32, linear_value_head_dim=32,
    linear_conv_kernel_dim=4, num_experts=16, num_experts_per_tok=4,
    decoder_sparse_step=1, shared_expert_intermediate_size=64,
    mlp_only_layers=[], moe_intermediate_size=128, rms_norm_eps=1e-6,
    vocab_size=200, num_key_value_heads=1, rope_theta=1e7,
    partial_rotary_factor=0.25, max_position_embeddings=4096, head_dim=32,
    norm_topk_prob=True, full_attention_interval=2, tie_word_embeddings=False,
)
MIXES = [("Q4_K", "Q4_K", "Q5_1"), ("Q5_K", "Q5_K", "Q8_0"), ("IQ3_XXS", "IQ4_XS", "IQ4_NL"),
         ("IQ2_S", "IQ4_XS", "Q5_0")]
RES = 6
PROJ = ("gate_proj", "up_proj", "down_proj")
# The largest |weight| a type makes per unit of its f16 scale: the toy's scales are 0.02 over it, so every layer's
# weights are alike in size as a real pack's are (the K-quants' 6-bit sub-scales otherwise make theirs 10-100x
# larger, and the toy then turns router near-ties into flips).
REACH = {"Q4_K": 945, "Q5_K": 1953, "Q5_1": 31, "Q5_0": 16, "Q8_0": 127, "Q2_0": 2, "IQ4_NL": 127,
         "IQ4_XS": 3937, "IQ3_XXS": 480, "IQ2_S": 166, "Q6_K": 4064, "Q3_K": 128, "Q2_K": 45, "IQ2_XXS": 166,
         "IQ2_XS": 166, "IQ3_S": 465, "IQ1_M": 17, "IQ1_S": 17, "TQ2_0": 2, "TQ1_0": 1, "Q1_0": 1, "F16": 4,
         "BF16": 4, "F32": 4}
# Two more toys for the 17 types added after the first twelve (the expert width, 128, rules out 256-weight blocks
# for down); MXFP4, NVFP4 and F8_E4M3 take random_experts' own scale ranges.
NEW_MIXES = ([("Q6_K", "Q3_K", "MXFP4"), ("Q2_K", "IQ2_XXS", "Q1_0"), ("IQ3_S", "IQ1_M", "BF16"),
              ("IQ2_XS", "TQ2_0", "F16")],
             [("IQ1_S", "TQ1_0", "NVFP4"), ("F32", "F8_E4M3", "F16"), ("BF16", "F16", "F8_E4M3"),
              ("Q6_K", "Q6_K", "Q8_0")])
_rng = np.random.default_rng(5)
PASSES = [_rng.integers(0, 200, 24).tolist(), [5], [77], [131], [9, 42], _rng.integers(0, 200, 24).tolist()]
BAR = 2 ** -7


def rel(a, b):
    return float(np.linalg.norm(np.asarray(a, np.float64) - b) / np.linalg.norm(b))


def fresh_model(quant_switch=True):
    m = q3n.Model(q3n.ModelArgs.from_dict(CFG))
    base = m.quant_predicate

    def pred(path, mod):
        if not hasattr(mod, "to_quantized") or (".switch_mlp." in path and not quant_switch):
            return False
        return base(path, mod)
    nn.quantize(m, group_size=32, bits=4, class_predicate=pred)
    return m


def build(out):
    """The trunk without routed experts, a two-shard GGUF of them, gguf_repack's experts.bin; returns the
    reference weights {layer: {proj: [E, out, in] float32}}."""
    mx.random.seed(0)
    model = fresh_model()
    mx.eval(model.parameters())
    weights = {k: v for k, v in tree_flatten(model.parameters()) if ".switch_mlp." not in k}
    out.mkdir(parents=True)
    mx.save_safetensors(str(out / "model.safetensors"), weights)
    (out / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {}, "weight_map": {k: "model.safetensors" for k in weights}}))
    (out / "config.json").write_text(json.dumps({**CFG, "quantization": {"group_size": 32, "bits": 4,
                                                                         "mode": "affine"}}))
    E, hid, inter = CFG["num_experts"], CFG["hidden_size"], CFG["moe_intermediate_size"]
    rng = np.random.default_rng(9)
    tensors, ref = [], {}
    for L, types in enumerate(MIXES):
        ref[L] = {}
        for p, t in zip(PROJ, types):
            rows, length = (inter, hid) if p != "down_proj" else (hid, inter)
            raw = ge.random_experts(rng, E, rows, length, t, scale=0.02 / REACH.get(t, 1))
            ref[L][p] = np.stack([ge.ref_dequant(raw[e], t).reshape(rows, length) for e in range(E)])
            tensors.append((f"blk.{L}.ffn_{p.split('_')[0]}_exps.weight", [length, rows, E], ge.TYPES[t][0],
                            raw.tobytes()))
    kv = [("general.architecture", "toy"), ("general.alignment", 32)]
    g1, g2 = out / "g-00001-of-00002.gguf", out / "g-00002-of-00002.gguf"
    write_gguf(g1, kv, tensors[:7])
    write_gguf(g2, kv, tensors[7:])
    gguf_repack.pack([g1, g2], out / "experts.bin", log=lambda *a, **k: None)
    return ref


def reference(out, ref):
    m = fresh_model(quant_switch=False)
    m.load_weights(str(out / "model.safetensors"), strict=False)
    for L, layer in enumerate(m.layers):
        for p in PROJ:
            getattr(layer.mlp.switch_mlp, p).weight = mx.array(ref[L][p])
    mx.eval(m.parameters())
    return m


def engine(out, peak_gb=None, defer=False, lm=False):
    m = fresh_model()
    m.load_weights(str(out / "model.safetensors"), strict=False)
    mx.eval(m.parameters())
    stream_fast._LM_ENV = "1" if lm else None
    stream_fast.install(m, out, resident=RES, defer=defer)
    if peak_gb:
        stream_fast.enable_budget(m, peak_gb)
    return m


def run_passes(model):
    cache, ys = make_prompt_cache(model), []
    for p in PASSES:
        y = model(mx.array(p, dtype=mx.int32)[None], cache=cache)
        mx.eval(y)
        ys.append(np.array(y[0].astype(mx.float32)))
    return ys


CALLS = {"gemv": 0, "dense": 0}


def counted(name, f):
    def g(*a, **kw):
        CALLS[name] += 1
        return f(*a, **kw)
    return g


def check_engine(out, ref_logits, name, **kw):
    model = engine(out, **kw)
    c0 = dict(CALLS)
    got = run_passes(model)
    errs = [rel(g, r) for g, r in zip(got, ref_logits)]
    agree = sum(int((g.argmax(-1) == r.argmax(-1)).sum()) for g, r in zip(got, ref_logits))
    layers = [g._gguf for g in stream_fast.glus_of(model)]
    n = {k: CALLS[k] - c0[k] for k in CALLS}
    print(f"{name}: logits rel err per pass (T " + "/".join(str(len(p)) for p in PASSES) + ") "
          + " ".join(f"{e:.1e}" for e in errs) + f"; argmax equal at {agree}/{sum(map(len, PASSES))}; "
          f"{n['gemv']} decode-chain and {n['dense']} dense calls")
    assert layers == ["/".join(m) for m in MIXES], layers
    assert max(errs) < BAR, errs
    assert n["gemv"] and n["dense"], n
    del model
    gc.collect()
    mx.clear_cache()
    return got


def check_refusal():
    """An index whose layer holds a type the engine does not decode is refused at load in one sentence that names
    the file and the type."""
    class Fd:
        name = "/models/x/experts.bin"
    base = "model.layers.3.mlp.switch_mlp."
    index = {base + p: {"parts": [{"name": "gguf", "ggml_type": t}]}
             for p, t in zip(PROJ, ("Q4_K", "Q4_K", "Q8_K"))}
    try:
        stream_fast._gguf_types({"index": index, "fd": Fd()}, base + "gate_proj")
        raise SystemExit("Q8_K was accepted at load")
    except ValueError as e:
        assert Fd.name in str(e) and "Q8_K" in str(e) and str(e).count(". ") == 0, e
        print(f"refused at load: {e}")
    index[base + "down_proj"]["parts"][0]["ggml_type"] = "Q6_K"
    assert stream_fast._gguf_types({"index": index, "fd": Fd()}, base + "gate_proj") == "Q4_K/Q4_K/Q6_K"


def main():
    check_refusal()
    mx.set_cache_limit(0)
    stream_fast.phys_footprint = lambda: mx.get_active_memory() + mx.get_cache_memory()
    # Both paths in every arm: dense wherever the host has the rows (the toy's own costs would pick otherwise).
    ge.calibrate = lambda *a: None
    ge.dense_pays = lambda u, p: True
    ge.gemv = counted("gemv", ge.gemv)
    ge.dense = counted("dense", ge.dense)
    global MIXES
    tmp = Path(tempfile.mkdtemp(prefix="gguf_"))
    try:
        for i, MIXES in enumerate([MIXES, *NEW_MIXES]):
            out = tmp / f"toy{i}"
            ref = run_passes(reference(out, build(out)))
            gc.collect()
            peak = mx.get_active_memory() / 1e9 + 1.0
            check_engine(out, ref, "per-layer arena")
            check_engine(out, ref, "run-ahead", defer=True)
            tm = check_engine(out, ref, "budget store", peak_gb=peak)
            lm = check_engine(out, ref, "budget store, layer-major prefill", peak_gb=peak, lm=True)
            # Layers whose gate / up / down records differ in size: the layer-major staging gives the same bits.
            assert all(np.array_equal(a.view(np.uint32), b.view(np.uint32)) for a, b in zip(tm, lm)), \
                "layer-major prefill differs from the token-major budget store"
            print("layer-major prefill bit-equal to the token-major budget store")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("all GGUF engine checks passed")


if __name__ == "__main__":
    main()
