#!/usr/bin/env python3
"""A tiny Qwen3.8-shaped (qwen4_exp) checkpoint, built from random weights,
packed the way the engine packs a real one, and decoded through the engine --
sync against --defer from a cleared arena.

The real checkpoint is 111.5 GB; the engine's fit to the ARCHITECTURE
(hyper-connections, PLE layer, sparse-attention indexer, logit-routed MoE,
_AttnCache/_LayerCache) can be checked without it. Everything the engine
touches is exercised: install() over `model.layers.N.mlp.switch_mlp`, the
packed experts, CacheBiasedMoE's logit-routed branch, the layer-wrapper
run-ahead with its snapshots of both cache classes. Only the numbers are
meaningless. Needs mlx_lm with qwen4_exp and a tokenizer to borrow
(--tokenizer-from, e.g. the 80B pack: same 248320 vocab family is not
required, vocab_size follows the borrowed tokenizer).

    python synth_qwen4.py --out /tmp/rpp/tiny_q4 --tokenizer-from <a Qwen3-Next checkpoint directory>
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

try:
    from mlx_lm.models import qwen4_exp
except ImportError:
    print("synth_qwen4: needs mlx_lm with qwen4_exp")
    sys.exit(0)
from mlx.utils import tree_flatten

SEED = 1234     # the reference's n-gram hash seed (mlx-lm PR #1788 head, 2097324)


def build(out, tok_dir, n_layers=8, n_experts=32, top_k=4, hidden=128):
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    tok_cfg = json.loads((Path(tok_dir) / "tokenizer_config.json").read_text())
    vocab = json.loads((Path(tok_dir) / "config.json").read_text())
    vocab = vocab.get("text_config", vocab).get("vocab_size") or 248320
    layer_types = ["linear_attention" if (i + 1) % 4 else "full_attention" for i in range(n_layers)]
    text = dict(model_type="qwen4_exp_text", hidden_size=hidden,
                moe_intermediate_size=64, shared_expert_intermediate_size=64,
                num_hidden_layers=n_layers, num_experts=n_experts, num_experts_per_tok=top_k,
                num_attention_heads=4, num_key_value_heads=2, head_dim=32,
                linear_num_value_heads=4, linear_num_key_heads=2,
                linear_key_head_dim=32, linear_value_head_dim=32,
                vocab_size=vocab, tie_word_embeddings=False, layer_types=layer_types,
                hc_count=4, hc_lowrank=32, indexer_n_heads=2, indexer_kv_heads=1,
                indexer_head_dim=32, indexer_budget=64, indexer_compress_ratio=4,
                ngram_vocab_size_base=4096,
                make_ngram_vocab_size_divisible_by=128, split_ngram_parts=4,
                # 8 n-gram heads x 32 = 256: a head dim the group size divides,
                # so the PLE shards are quantised as in the released checkpoint
                ple_embed_dim=256, ple_layer_ids=[2], heads_per_ngram=4, ngram_size=3,
                eos_token_id=tok_cfg.get("eos_token_id", 248044) if isinstance(tok_cfg.get("eos_token_id"), int) else 248044)
    # The release's n-gram buffers are hashed with seed 1234 while its config
    # names no seed: built with 1234, saved without it, like the real one.
    args = qwen4_exp.ModelArgs.from_dict({"model_type": "qwen4_exp",
                                          "text_config": {**text, "seed": SEED}})
    model = qwen4_exp.Model(args)
    mx.eval(model.parameters())
    # 4-bit g32 like the released conversion; routers 8-bit; PLE shards left
    # unquantised (tiny) -- they are not what this checks.
    def pred(path, m):
        if not hasattr(m, "to_quantized"):
            return False
        if path.endswith("mlp.gate") or path.endswith("shared_expert_gate"):
            return {"group_size": 32, "bits": 8}
        return True
    nn.quantize(model, group_size=32, bits=4, class_predicate=pred)
    weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(out / "model.safetensors"), weights)
    q = {"group_size": 32, "bits": 4, "mode": "affine"}
    cfg = {"model_type": "qwen4_exp", "architectures": ["Qwen4ExpForConditionalGeneration"],
           "text_config": text, "quantization": q}
    for k in weights:
        if (k.endswith("mlp.gate.weight") or k.endswith("shared_expert_gate.weight")):
            cfg["quantization"][k[:-len(".weight")]] = {"group_size": 32, "bits": 8}
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    idx = {"metadata": {"total_size": (out / "model.safetensors").stat().st_size},
           "weight_map": {k: "model.safetensors" for k in weights}}
    (out / "model.safetensors.index.json").write_text(json.dumps(idx))
    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
              "chat_template.jinja", "generation_config.json", "vocab.json", "merges.txt"):
        if (Path(tok_dir) / f).exists():
            shutil.copy(Path(tok_dir) / f, out / f)
    print(f"tiny qwen4_exp: {len(weights)} tensors, "
          f"{(out / 'model.safetensors').stat().st_size / 1e6:.0f} MB", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer-from", required=True)
    args = ap.parse_args()
    out = build(args.out, os.path.expanduser(args.tokenizer_from))
    here = Path(__file__).resolve().parent
    slim = out.parent / (out.name + "-slim")
    if slim.exists():
        shutil.rmtree(slim)
    # The real checkpoint goes through stream_pack (experts.bin + ple.bin +
    # slim dense); the tiny one takes the same road, from the directory.
    subprocess.run([sys.executable, str(here / "tools" / "stream_pack.py"), "--from-dir", str(out),
                    "--out", str(slim)], check=True)
    print(f"packed at {slim}; source kept at {out}", flush=True)


if __name__ == "__main__":
    main()
