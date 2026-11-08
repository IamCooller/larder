#!/usr/bin/env python3
"""MOE_KV_HOST on CUDA (kv_host.py): a QSA layer's K/V homed in pinned, GPU-mapped host RAM gives the flag-off
path's outputs bit for bit.

A toy qwen4_exp model with Flash-Next's attention shapes (24 query heads, 2 KV heads x 256, indexer 4 x 128,
budget 2048, blocks of 4), random weights; its attention layer runs on two caches made in one process, the
flag-off class (kv_host.BASE, VRAM as today) and the homed one (kv_host.CLS), with the home line at 3,000
positions, chunks of 1,024 positions and an LRU of one top-k (512 blocks), so a toy context crosses the line in a
pass, spans chunks and evicts every token:

  prefill pieces crossing the line; 48 one-row passes on the picks; a trim and 8 more; a 2-row pass (staged);
  state; a restore that homes again; a trim under the line (picks still homed, then a 2-row pass back to VRAM);
  a refused allocation (stays in VRAM). Every layer output and state bit-equal to the flag-off cache's.

No checkpoint. CUDA only (skips elsewhere); run on the PC:

    MOE_KV_HOST=1 python3 tests/test_kv_host.py
"""
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ["MOE_KV_HOST"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

try:
    from mlx_lm.models import qwen4_exp as q4
except ImportError:
    print("test_kv_host: skipped (mlx_lm without qwen4_exp)")
    sys.exit(0)

import stream_fast as sf

if sf.BACKEND != "cuda":
    print(f"test_kv_host: skipped (backend {sf.BACKEND})")
    sys.exit(0)

HOME = tempfile.mkdtemp(prefix="kv_host_home_")
Path.home = classmethod(lambda cls: Path(HOME))     # the calibration file goes here, not into ~/.cache


def toy_model():
    args = q4.ModelArgs.from_dict({"model_type": "qwen4_exp", "text_config": dict(
        hidden_size=512, num_hidden_layers=2, num_attention_heads=24, num_key_value_heads=2,
        head_dim=256, partial_rotary_factor=0.25, vocab_size=256, num_experts=4,
        num_experts_per_tok=2, moe_intermediate_size=32, shared_expert_intermediate_size=32,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=32,
        linear_value_head_dim=32, hc_count=4, hc_lowrank=16, indexer_n_heads=4,
        indexer_kv_heads=1, indexer_head_dim=128, indexer_budget=2048,
        indexer_compress_ratio=4, ple_layer_ids=[], rope_theta=10_000_000.0,
        layer_types=["linear_attention", "full_attention"])})
    m = q4.Model(args)
    m.set_dtype(mx.bfloat16)
    mx.eval(m.parameters())
    return m


def main():
    m = toy_model()
    assert sf.install_qsa_pool(m) == 1 and sf._QSA_ATTN_ON, "the CUDA QSA attention did not install"
    import kv_host as kh
    assert kh.CLS is not None and q4._AttnCache is kh.CLS, "kv_host did not install"
    attn, rope = m.layers[1].self_attn, m.model.rope
    shape = sf._qsa_shape(attn)
    print(f"  installed: switch {sf._QSA_SWITCH[shape]} keys, line {kh.H} positions")
    kh.H, kh.LOGC, kh.LRU_MULT = 3000, 10, 1
    kh._DUMMY[0] = None
    hidden = attn.q_proj.weight.shape[1]
    mx.random.seed(0)

    def x_of(s):
        return (mx.random.normal((1, s, hidden)) * 0.5).astype(mx.bfloat16)

    def both(on, off, s, what):
        x = x_of(s)
        mask = None if s == 1 else "causal"
        a = attn(x, rope, mask, on, on.indexer)
        b = attn(x, rope, mask, off, off.indexer)
        mx.eval(a, b)
        assert bool(mx.array_equal(a, b)), f"{what}: layer output differs (offset {off.offset})"
        assert on.offset == off.offset, what

    def same_state(on, off, what):
        # Every part this mlx_lm's state has, the offset too where it carries one, the whole buffer where it is that.
        sa, sb = on.state, off.state
        assert len(sa) == len(sb), what
        for x, y in zip(sa, sb):
            if isinstance(x, mx.array):
                assert x.shape == y.shape and bool(mx.array_equal(x, y)), f"{what}: state differs"
            else:
                assert x == y, f"{what}: state differs ({x} against {y})"

    on, off = kh.CLS(), kh.BASE()
    assert type(on.indexer) is type(off.indexer)
    sf._QSA_SWITCH[shape] = 2 ** 62
    for s in (2048, 2000, 52):
        both(on, off, s, f"prefill piece {s}")
    assert on.kv_host, "the cache did not move to the home past the line"
    sf._QSA_SWITCH[shape] = attn.indexer.token_budget + 1
    t0 = time.perf_counter()
    for i in range(48):
        both(on, off, 1, f"pass {i} on the picks")
    t1 = time.perf_counter()
    assert on.lru is not None, "the picks did not go through the LRU"
    on.trim(5), off.trim(5)
    for i in range(8):
        both(on, off, 1, f"pass {i} after a trim")
    both(on, off, 2, "a 2-row pass")
    for i in range(4):
        both(on, off, 1, f"pass {i} after the 2-row pass")
    same_state(on, off, "state")
    rep = kh.report()
    print(f"  homed: 3 prefill pieces, 60 passes on the picks, a trim, a 2-row pass: every output bit-equal; "
          f"LRU {rep['hits']} hits, {rep['misses']} misses ({rep['miss_rate']}), home {rep['home_mb']} MB, "
          f"pinned {rep['pinned_mb']} MB, staged {rep['staged_mb']} MB; 48 picks passes {(t1 - t0) * 1e3:.0f} ms "
          f"with their checks")
    assert rep["hits"] > 0 and rep["misses"] > 0 and rep["hits"] + rep["misses"] == 512 * 60, rep

    r_on, r_off = kh.CLS(), kh.BASE()
    st = off.state
    r_on.state, r_off.state = st, st
    assert not r_on.kv_host
    for i in range(4):
        both(r_on, r_off, 1, f"pass {i} after a restore")
    assert r_on.kv_host, "a restored cache did not home again"
    same_state(r_on, r_off, "state after a restore")
    n = r_off.offset - 2500
    r_on.trim(n), r_off.trim(n)
    both(r_on, r_off, 1, "a pass on the picks under the line")
    assert r_on.kv_host
    both(r_on, r_off, 2, "a 2-row pass under the line")
    assert not r_on.kv_host, "under the line a multi-row pass goes back to VRAM"
    same_state(r_on, r_off, "state back in VRAM")
    print("  restore, homed again, a trim under the line, back to VRAM: every output bit-equal")

    kh.FLOOR, kh._FREE = 1e15, {}
    f_on, f_off = kh.CLS(), kh.BASE()
    for s in (2048, 2000):
        both(f_on, f_off, s, f"refused piece {s}")
    both(f_on, f_off, 1, "a refused cache's pass on the picks")
    assert not f_on.kv_host and f_on.refused
    same_state(f_on, f_off, "refused state")
    print("  a refused allocation stays in VRAM: every output bit-equal")
    print("test_kv_host: ok")


if __name__ == "__main__":
    main()
