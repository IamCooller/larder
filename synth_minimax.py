"""A tiny MiniMax-M2-shaped (minimax_m2) checkpoint from random weights, packed
the way the engine packs a real one. The real MiniMax-M2.5 is 129 GB; the
engine's fit to the ARCHITECTURE -- the MoE block under `block_sparse_moe`,
sigmoid scores with a selection-only correction term, plain KV attention --
is checked without it. Only the numbers are meaningless.
    python synth_minimax.py --out /tmp/rpp/tiny_mm --tokenizer-from <a Qwen3 checkpoint directory>
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

try:
    from mlx_lm.models import minimax
except ImportError:
    print("synth_minimax: needs mlx_lm with minimax")
    sys.exit(0)


def build(out, tok_dir, n_layers=6, n_experts=32, top_k=4, hidden=128):
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    vocab = json.loads((Path(tok_dir) / "config.json").read_text())
    vocab = vocab.get("text_config", vocab).get("vocab_size") or 200064
    cfg = dict(model_type="minimax_m2", hidden_size=hidden, intermediate_size=64,
               num_attention_heads=4, num_key_value_heads=2, head_dim=32,
               max_position_embeddings=4096, num_experts_per_tok=top_k,
               num_local_experts=n_experts, shared_intermediate_size=0,
               num_hidden_layers=n_layers, rms_norm_eps=1e-6, rope_theta=10000.0,
               rotary_dim=16, vocab_size=vocab, tie_word_embeddings=False,
               scoring_func="sigmoid", use_qk_norm=True)
    model = minimax.Model(minimax.ModelArgs.from_dict(cfg))
    mx.eval(model.parameters())
    # A correction term that actually moves selections, as a trained one does.
    for l in model.model.layers:
        l.block_sparse_moe.e_score_correction_bias = mx.random.normal((n_experts,)) * 0.1
    mx.eval(model.parameters())

    def pred(path, m):
        if not hasattr(m, "to_quantized"):
            return False
        if path.endswith("block_sparse_moe.gate"):
            return {"group_size": 64, "bits": 8}
        return True
    nn.quantize(model, group_size=64, bits=4, class_predicate=pred)
    weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(out / "model.safetensors"), weights)
    q = {"group_size": 64, "bits": 4, "mode": "affine"}
    for k in weights:
        if k.endswith("block_sparse_moe.gate.weight"):
            q[k[:-len(".weight")]] = {"group_size": 64, "bits": 8}
    cfg = dict(cfg, architectures=["MiniMaxM2ForCausalLM"], quantization=q)
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    idx = {"metadata": {"total_size": (out / "model.safetensors").stat().st_size},
           "weight_map": {k: "model.safetensors" for k in weights}}
    (out / "model.safetensors.index.json").write_text(json.dumps(idx))
    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
              "chat_template.jinja", "generation_config.json", "vocab.json", "merges.txt"):
        if (Path(tok_dir) / f).exists():
            shutil.copy(Path(tok_dir) / f, out / f)
    print(f"tiny minimax_m2: {len(weights)} tensors, "
          f"{(out / 'model.safetensors').stat().st_size / 1e6:.0f} MB", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer-from", required=True)
    a = ap.parse_args()
    src = build(a.out, Path(a.tokenizer_from).expanduser())
    slim = src.parent / (src.name + "-slim")
    here = Path(__file__).resolve().parent
    subprocess.run([sys.executable, str(here / "tools" / "stream_pack.py"), "--from-dir", str(src),
                    "--out", str(slim)], check=True)
    print(f"packed -> {slim}")


if __name__ == "__main__":
    main()
