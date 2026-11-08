#!/usr/bin/env python3
"""The Qwen3-Next head path of mtp_pack.py + mtp.py on a tiny synthetic head.

No model, no GPU work worth naming: a 4-expert head with the 80B's tensor
names (per-expert experts.N.{gate,up,down}_proj, gated q_proj, q/k norms) is
written as bf16 safetensors, packed to 4 bits by mtp_pack.py, loaded through
mtp.head_args + mtp.load_mtp, and run once. Every key has to land (strict
load_weights) and the output has to be finite. Checks the stacking branch,
the layer-class dispatch and the flat-config path -- the three things that
differ from the Qwen3.6 head.

    python3 tests/test_mtp_next.py
"""
import json, os, subprocess, sys, tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import mtp

H, E, I, NH, KVH, HD = 64, 4, 64, 2, 1, 32
cfg = {
    "model_type": "qwen3_next", "hidden_size": H, "num_hidden_layers": 4,
    "intermediate_size": 64, "num_attention_heads": NH,
    "linear_num_value_heads": 2, "linear_num_key_heads": 1,
    "linear_key_head_dim": 32, "linear_value_head_dim": 32,
    "linear_conv_kernel_dim": 4, "num_experts": E, "num_experts_per_tok": 2,
    "decoder_sparse_step": 1, "shared_expert_intermediate_size": I,
    "mlp_only_layers": [], "moe_intermediate_size": I, "rms_norm_eps": 1e-6,
    "vocab_size": 100, "num_key_value_heads": KVH, "rope_theta": 1e7,
    "partial_rotary_factor": 0.25, "max_position_embeddings": 4096,
    "head_dim": HD, "norm_topk_prob": True, "full_attention_interval": 4,
}
shapes = {
    "mtp.fc.weight": (H, 2 * H), "mtp.pre_fc_norm_embedding.weight": (H,),
    "mtp.pre_fc_norm_hidden.weight": (H,), "mtp.norm.weight": (H,),
    "mtp.layers.0.input_layernorm.weight": (H,),
    "mtp.layers.0.post_attention_layernorm.weight": (H,),
    "mtp.layers.0.self_attn.q_proj.weight": (NH * HD * 2, H),
    "mtp.layers.0.self_attn.k_proj.weight": (KVH * HD, H),
    "mtp.layers.0.self_attn.v_proj.weight": (KVH * HD, H),
    "mtp.layers.0.self_attn.o_proj.weight": (H, NH * HD),
    "mtp.layers.0.self_attn.q_norm.weight": (HD,),
    "mtp.layers.0.self_attn.k_norm.weight": (HD,),
    "mtp.layers.0.mlp.gate.weight": (E, H),
    "mtp.layers.0.mlp.shared_expert.gate_proj.weight": (I, H),
    "mtp.layers.0.mlp.shared_expert.up_proj.weight": (I, H),
    "mtp.layers.0.mlp.shared_expert.down_proj.weight": (H, I),
    "mtp.layers.0.mlp.shared_expert_gate.weight": (1, H),
}
for e in range(E):
    shapes[f"mtp.layers.0.mlp.experts.{e}.gate_proj.weight"] = (I, H)
    shapes[f"mtp.layers.0.mlp.experts.{e}.up_proj.weight"] = (I, H)
    shapes[f"mtp.layers.0.mlp.experts.{e}.down_proj.weight"] = (H, I)

d = Path(tempfile.mkdtemp())
mx.random.seed(0)
w = {k: (mx.random.normal(s) * 0.05).astype(mx.bfloat16) for k, s in shapes.items()}
mx.save_safetensors(str(d / "mtp.safetensors"), w)
out = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "tools" / "mtp_pack.py"),
                      "--src", str(d)],
                     capture_output=True, text=True)
assert out.returncode == 0, out.stdout + out.stderr
assert "experts stacked: 4" in out.stdout, out.stdout
meta = json.loads((d / "mtp-4bit-quant.json").read_text())
assert meta["mtp.layers.0.mlp.gate"]["bits"] == 8 and \
    meta["mtp.layers.0.mlp.switch_mlp.gate_proj"]["bits"] == 4, meta

args = mtp.head_args(cfg)
head = mtp.load_mtp(d, args)                     # strict: every key must land
assert type(head.layers[0]).__name__ == "Qwen3NextDecoderLayer"
x = mx.random.normal((1, 3, H)).astype(mx.float16)
y = head(x, x)
mx.eval(y)
assert y.shape == (1, 3, H) and bool(mx.all(mx.isfinite(y))), y.shape
print(f"ok: qwen3_next head packed ({len(meta)} quantised modules), loaded, "
      f"forward finite {y.shape}")
