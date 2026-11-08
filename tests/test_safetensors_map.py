#!/usr/bin/env python3
"""The engine's own safetensors index (stream_fast.build_shard_map, SafetensorsMap,
find_switch_mlp) reads exactly what mx.load reads, on a tiny synthetic checkpoint:
every tensor's bytes at its indexed range, expert rows through _read_experts, an
index.json checkpoint and a single-file one. find_switch_mlp finds the experts of
mlx_lm's own module trees (mlp.switch_mlp, block_sparse_moe.switch_mlp) and
nothing in a dense one.

With MLX_MOE_DIR pointing at an old mlx-moe checkout, also checks that the index
and shard map equal the ones it built (the engine used it until this file).

    python3 tests/test_safetensors_map.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import numpy as np

import stream_fast


def tiny_checkpoint(out, split):
    """Expert stacks of a 4-bit layer (U32 / BF16), plus every other dtype a
    checkpoint carries. split: two shards and an index.json, else one file."""
    out.mkdir(parents=True)
    w, s, b = mx.quantize(mx.random.normal((6, 32, 64)).astype(mx.bfloat16), group_size=32, bits=4)
    a = {"model.layers.0.mlp.switch_mlp.gate_proj.weight": w,
         "model.layers.0.mlp.switch_mlp.gate_proj.scales": s,
         "model.layers.0.mlp.switch_mlp.gate_proj.biases": b,
         "model.embed_tokens.weight": mx.random.normal((10, 8)).astype(mx.float16)}
    c = {"model.norm.weight": mx.random.normal((8,)),
         "model.ngram.layer_multipliers": mx.array([3, 5, 7], mx.int64),
         "model.mask": mx.array([1, 0, 1], mx.uint8),
         "model.layers.0.mlp.gate.weight": mx.random.normal((6, 64)).astype(mx.bfloat16)}
    if not split:
        mx.save_safetensors(str(out / "model.safetensors"), {**a, **c})
        return {**a, **c}
    mx.save_safetensors(str(out / "model-00001-of-00002.safetensors"), a)
    mx.save_safetensors(str(out / "model-00002-of-00002.safetensors"), c)
    wm = {k: "model-00001-of-00002.safetensors" for k in a}
    wm.update({k: "model-00002-of-00002.safetensors" for k in c})
    (out / "model.safetensors.index.json").write_text(json.dumps({"weight_map": wm}))
    return {**a, **c}


def check_reads(out, ref):
    sm = stream_fast.build_shard_map(out)
    assert set(sm) == set(ref), set(sm) ^ set(ref)
    for k, p in sm.items():
        assert Path(p).parent == out and Path(p).exists(), (k, p)
    st = stream_fast.SafetensorsMap(sorted(set(sm.values())))
    assert all(k in st for k in ref) and "nope" not in st
    for k, want in ref.items():
        path, np_dt, mx_dt, shape, start, length = st._index[k]
        assert list(shape) == list(want.shape) and length == want.nbytes, (k, shape, length)
        raw = os.pread(st._fds[path].fileno(), length, start)
        assert raw == bytes(memoryview(want.view(mx.uint8) if want.dtype == mx.bfloat16 else want)), k
        assert mx_dt == want.dtype, (k, mx_dt, want.dtype)
    # Expert rows the way the engine reads them, BF16 viewed back.
    for part in ("weight", "scales", "biases"):
        key = f"model.layers.0.mlp.switch_mlp.gate_proj.{part}"
        got = stream_fast._read_experts(st, key, [4, 1, 4])
        want = ref[key][mx.array([4, 1, 4])]
        assert got.dtype == want.dtype and mx.array_equal(got, want).item(), key
    st.close()
    return st


def check_find_switch():
    from mlx_lm.models import mixtral, qwen3, qwen3_moe
    moe = qwen3_moe.Model(qwen3_moe.ModelArgs(
        model_type="qwen3_moe", hidden_size=16, num_hidden_layers=2, intermediate_size=32,
        num_attention_heads=2, num_experts=4, num_experts_per_tok=2, decoder_sparse_step=1,
        mlp_only_layers=[1], moe_intermediate_size=8, rms_norm_eps=1e-6, vocab_size=32,
        num_key_value_heads=1, head_dim=8, rope_theta=1e4, tie_word_embeddings=False,
        max_position_embeddings=64, norm_topk_prob=True))
    sw, key = stream_fast.find_switch_mlp(moe.layers[0], 0)
    assert sw is moe.layers[0].mlp.switch_mlp and key == "model.layers.0.mlp.switch_mlp"
    assert stream_fast.find_switch_mlp(moe.layers[1], 1) == (None, None)    # a dense mlp_only layer
    mix = mixtral.Model(mixtral.ModelArgs(
        model_type="mixtral", vocab_size=32, hidden_size=16, intermediate_size=8,
        num_hidden_layers=1, num_attention_heads=2, num_experts_per_tok=2,
        num_key_value_heads=1, num_local_experts=4))
    sw, key = stream_fast.find_switch_mlp(mix.layers[0], 0)
    assert sw is mix.layers[0].block_sparse_moe.switch_mlp
    assert key == "model.layers.0.block_sparse_moe.switch_mlp"
    dense = qwen3.Model(qwen3.ModelArgs(
        model_type="qwen3", hidden_size=16, num_hidden_layers=1, intermediate_size=32,
        num_attention_heads=2, rms_norm_eps=1e-6, vocab_size=32, num_key_value_heads=1,
        max_position_embeddings=64, rope_theta=1e4, head_dim=8, tie_word_embeddings=True))
    assert stream_fast.find_switch_mlp(dense.layers[0], 0) == (None, None)


def check_against_mlx_moe(out, st):
    """Only where an old mlx-moe checkout is named: same shard map, same index."""
    d = os.environ.get("MLX_MOE_DIR")
    if not d:
        print("  (MLX_MOE_DIR unset: the comparison with mlx-moe is skipped)")
        return
    sys.path.insert(0, d)
    from mlx_moe.lazy_experts.loading import SafetensorsMap, _build_shard_map
    old = _build_shard_map(out)
    assert old == stream_fast.build_shard_map(out)
    ost = SafetensorsMap(sorted(set(old.values())))
    new = stream_fast.SafetensorsMap(sorted(set(old.values())))
    assert ost._index == new._index, "index tuples differ"
    print(f"  mlx-moe at {d}: shard map and all {len(new._index)} index tuples equal")


def main():
    mx.random.seed(0)
    with tempfile.TemporaryDirectory() as tmp:
        for split in (True, False):
            out = Path(tmp) / ("split" if split else "single")
            ref = tiny_checkpoint(out, split)
            st = check_reads(out, ref)
            print(f"1. {'index.json, 2 shards' if split else 'one file, no index'}: "
                  f"{len(ref)} tensors read back byte-equal, expert rows equal")
        check_against_mlx_moe(Path(tmp) / "split", st)
    check_find_switch()
    print("2. find_switch_mlp: qwen3_moe mlp.switch_mlp, mixtral block_sparse_moe, dense none")
    print("test_safetensors_map: ok")


if __name__ == "__main__":
    main()
