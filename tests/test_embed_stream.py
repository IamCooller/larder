#!/usr/bin/env python3
"""embed_tokens rows streamed from disk equal the resident table, and the
engine's greedy tokens are unchanged by streaming them -- the same two
checks test_ple.py runs for the PLE table, here for the model's main
embedding (stream_fast.StreamingEmbedding / install_embed).

A tiny synthetic qwen3_next (MoE + hybrid GDN/attention) checkpoint, built
and quantized in memory then saved to disk -- no real weights, no
tokenizer, a few hundred KB, well under the machine rule's ~200 MB ceiling
for a synthetic tensor test, and nothing outside this repo is touched.

    python3 tests/test_embed_stream.py
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.generate import generate_step
from mlx_lm.models import qwen3_next as q3n

import stream_fast

# Same toy dims fuse_proj.py's own selftest uses for this architecture (a
# proven-working shape, not a new guess): 4 hybrid GDN/attention layers, all
# of them MoE (decoder_sparse_step=1, no mlp_only_layers), so install() finds
# something to stream and install_embed's gate (`if streamed:`) fires.
CFG = dict(
    model_type="qwen3_next", hidden_size=64, num_hidden_layers=4,
    intermediate_size=64, num_attention_heads=2, linear_num_value_heads=2,
    linear_num_key_heads=1, linear_key_head_dim=32, linear_value_head_dim=32,
    linear_conv_kernel_dim=4, num_experts=8, num_experts_per_tok=2,
    decoder_sparse_step=1, shared_expert_intermediate_size=64,
    mlp_only_layers=[], moe_intermediate_size=64, rms_norm_eps=1e-6,
    vocab_size=200, num_key_value_heads=1, rope_theta=1e7,
    partial_rotary_factor=0.25, max_position_embeddings=4096, head_dim=32,
    norm_topk_prob=True, full_attention_interval=2, tie_word_embeddings=False,
)


# A tied checkpoint (LFM2-8B-A1B, Granite-4.0-H-tiny) has no separate lm_head module -- Model.__call__ reaches for
# embed_tokens.as_linear instead, the path install_embed has to refuse.
TIED_CFG = {**CFG, "tie_word_embeddings": True}


def fresh_model(cfg=CFG):
    """A model shell with the checkpoint's exact quantized shapes (weights
    still random -- callers load_weights() over it, the way mlx_lm.load
    quantizes the skeleton before reading real bytes into it)."""
    m = q3n.Model(q3n.ModelArgs.from_dict(cfg))
    pred = m.quant_predicate
    nn.quantize(m, group_size=32, bits=4,
                class_predicate=lambda p, mod: hasattr(mod, "to_quantized") and pred(p, mod))
    return m


def build(out, cfg=CFG):
    mx.random.seed(0)
    model = fresh_model(cfg)
    mx.eval(model.parameters())
    weights = dict(tree_flatten(model.parameters()))
    out.mkdir(parents=True)
    mx.save_safetensors(str(out / "model.safetensors"), weights)
    idx = {"metadata": {"total_size": (out / "model.safetensors").stat().st_size},
           "weight_map": {k: "model.safetensors" for k in weights}}
    (out / "model.safetensors.index.json").write_text(json.dumps(idx))
    (out / "config.json").write_text(json.dumps(
        {**cfg, "quantization": {"group_size": 32, "bits": 4, "mode": "affine"}}))
    print(f"tiny qwen3_next: {len(weights)} tensors, "
          f"{(out / 'model.safetensors').stat().st_size / 1e6:.2f} MB", flush=True)
    return model


def main():
    tmp = Path(tempfile.mkdtemp(prefix="embed_stream_"))
    try:
        out = tmp / "tiny_q3n"
        ref_model = build(out)

        # 1. rows: StreamingEmbedding(disk) == QuantizedEmbedding(resident)
        shard_map = stream_fast.build_shard_map(out)
        st = stream_fast.SafetensorsMap(sorted(set(shard_map.values())))

        stack = stream_fast._layer_stack_of(ref_model)
        emb = stack.embed_tokens
        gid = mx.array(np.random.default_rng(0).integers(0, CFG["vocab_size"], size=(1, 7)),
                        dtype=mx.int32)
        ref_rows = emb(gid)
        mx.eval(ref_rows)

        freed = stream_fast.install_embed(ref_model, st)
        assert freed > 0, "install_embed found nothing to stream"
        got_rows = stack.embed_tokens(gid)          # now routed through StreamingEmbedding
        mx.eval(got_rows)
        diff = float(mx.max(mx.abs(ref_rows - got_rows)))
        assert diff == 0.0 and ref_rows.shape == got_rows.shape and ref_rows.dtype == got_rows.dtype, \
            (diff, ref_rows.shape, got_rows.shape, ref_rows.dtype, got_rows.dtype)
        print(f"1. embed rows: streamed == resident, {gid.size} rows, "
              f"max diff {diff}, {freed} B freed")

        # 2. tokens: a model with embed_tokens streamed through install()
        #    decodes the same greedy tokens as one that never streamed it.
        prompt = mx.array([3, 17, 42, 5, 90, 11], dtype=mx.int32)

        unpacked = fresh_model()
        unpacked.load_weights(str(out / "model.safetensors"))
        mx.eval(unpacked.parameters())
        ref_tokens = [int(t) for t, _ in generate_step(prompt, unpacked, max_tokens=16)]
        del unpacked
        mx.clear_cache()

        streamed_model = fresh_model()
        streamed_model.load_weights(str(out / "model.safetensors"))
        mx.eval(streamed_model.parameters())
        # This toy ships no experts.bin, so the experts come through the
        # safetensors fallback, the same one install_embed relies on.
        n, st2 = stream_fast.install(streamed_model, out, resident=4)
        assert n > 0, "toy checkpoint has no streamed MoE layers"
        stream_fast.install_embed(streamed_model, st2)
        got_tokens = [int(t) for t, _ in generate_step(prompt, streamed_model, max_tokens=16)]

        print(f"2. tokens: unpacked {ref_tokens}\n            streamed {got_tokens}")
        assert ref_tokens == got_tokens, (ref_tokens, got_tokens)

        # 3. a tied checkpoint (tie_word_embeddings=True): install_embed must
        #    refuse rather than stream, because Model.__call__ and spec.py's
        #    Speculator both fall back to embed_tokens.as_linear for the
        #    lm_head projection when there is no separate `lm_head` module,
        #    and nn.QuantizedEmbedding.as_linear reads self['weight'] -- gone
        #    the moment install_embed pops it. Before the fix this raised
        #    KeyError 'weight' on the very next forward.
        tied_out = tmp / "tied_q3n"
        tied_model = build(tied_out, TIED_CFG)
        tied_prompt = mx.array([[3, 17, 42]], dtype=mx.int32)
        ref_tied = tied_model(tied_prompt)
        mx.eval(ref_tied)

        n3, st3 = stream_fast.install(tied_model, tied_out, resident=4)
        assert n3 > 0, "tied toy checkpoint has no streamed MoE layers"
        freed3 = stream_fast.install_embed(tied_model, st3)
        assert freed3 == 0, f"install_embed must refuse a tied checkpoint, freed {freed3}"
        stack3 = stream_fast._layer_stack_of(tied_model)
        assert "weight" in stack3.embed_tokens, "table must stay resident when refused"

        got_tied = tied_model(tied_prompt)          # would KeyError 'weight' pre-fix
        mx.eval(got_tied)
        diff3 = float(mx.max(mx.abs(ref_tied - got_tied)))
        assert diff3 == 0.0, diff3
        print(f"3. tied checkpoint: install_embed refused (freed {freed3}), "
              f"table resident, forward matches (diff {diff3})")

        print("test_embed_stream: ok")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
