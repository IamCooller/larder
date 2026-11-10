#!/usr/bin/env python3
"""MOE_CUDA_EVAL_EVERY, MOE_CUDA_TOKEN_AHEAD, MOE_PIPE_ONE and MOE_PIPE_GUESS_BF16 (stream_fast's CUDA native decode)
give the defaults' tokens: test_store's toy (qwen3_next, 4 MoE layers, 32 experts, top 4, a 6-slot arena) decodes
natively in a child process per setting, cache bias on so each router ranks against the agent's slot table, two
fresh turns of 24 tokens. EVAL_EVERY=3 must also make fewer mx.async_eval calls a token (layers 0-2 in one, the last
layer alone). MOE_PIPE_ONE's kernel skips the wait on layers whose experts the table holds, and the agent still claims
every job: each layer's LRU at the end (order included) must be the defaults'. Skipped without MLX's CUDA backend.

    perl -e 'alarm shift; exec @ARGV' 280 python3 tests/test_native_batch_cuda.py
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))


def child():
    import mlx.core as mx
    from mlx_lm.generate import generate_step
    from mlx_lm.models.cache import make_prompt_cache
    import stream_fast
    import test_store as ts
    calls = {"aeval": 0, "native": 0}
    aeval, native = mx.async_eval, stream_fast.StreamingSwitchGLU._call_native

    def count_aeval(*a):
        calls["aeval"] += 1
        return aeval(*a)

    def count_native(self, *a):
        calls["native"] += 1
        return native(self, *a)
    mx.async_eval = count_aeval
    stream_fast.StreamingSwitchGLU._call_native = count_native
    toks, n = [], 0
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "toy"
        ts.build(out)
        model = ts.stock_model(out)
        stream_fast.install(model, out, resident=ts.RES, cache_bias=0.5, defer=False, native=True)
        assert stream_fast._NATIVE is not None and stream_fast._NATIVE.cuda, "native did not install on CUDA"
        for p in ts.PROMPTS[:2]:
            for t, _ in generate_step(mx.array(p, dtype=mx.int32), model, max_tokens=24,
                                      prompt_cache=make_prompt_cache(model)):
                toks.append(int(t))
                n += 1
        assert stream_fast._NATIVE is not None, "native went off"
        stream_fast.native_sync()
        lru = [list(g._lru.items()) for g in stream_fast._NATIVE.glus]
    print(json.dumps({"tokens": toks, "aeval_a_token": calls["aeval"] / n, "native": calls["native"], "lru": lru}),
          flush=True)


def run(env):
    p = subprocess.run([sys.executable, __file__, "--child"], env={**os.environ, "MOE_NATIVE": "1", **env},
                       capture_output=True, text=True, timeout=240)
    assert p.returncode == 0, f"{env}: child failed\n{p.stdout[-2000:]}\n{p.stderr[-3000:]}"
    return json.loads(p.stdout.strip().splitlines()[-1])


def main():
    import stream_fast
    if stream_fast.BACKEND != "cuda":
        print(f"SKIP: needs MLX's CUDA backend (this is {stream_fast.BACKEND})")
        return
    base = run({})
    assert base["native"] > 0, base
    print(f"defaults: {len(base['tokens'])} tokens, {base['aeval_a_token']:.2f} async_eval a token, "
          f"{base['native']} native layers", flush=True)
    both = {"MOE_CUDA_EVAL_EVERY": "3", "MOE_CUDA_TOKEN_AHEAD": "1"}
    for env in ({"MOE_CUDA_EVAL_EVERY": "3"}, {"MOE_CUDA_TOKEN_AHEAD": "1"}, both,
                {"MOE_PIPE_ONE": "1"}, {"MOE_PIPE_GUESS_BF16": "1"}, {"MOE_PIPE_ONE": "1", "MOE_PIPE_GUESS_BF16": "1"},
                {**both, "MOE_PIPE_ONE": "1", "MOE_PIPE_GUESS_BF16": "1"}):
        r = run(env)
        print(f"{env}: {r['aeval_a_token']:.2f} async_eval a token, {r['native']} native layers", flush=True)
        assert r["tokens"] == base["tokens"], f"{env}: tokens differ from the defaults'"
        assert r["native"] == base["native"], (env, r["native"], base["native"])
        assert r["lru"] == base["lru"], f"{env}: the LRU at the end differs from the defaults'"
        if "MOE_CUDA_EVAL_EVERY" in env:
            assert r["aeval_a_token"] < base["aeval_a_token"] - 0.5, (env, r, base["aeval_a_token"])
    print("ok: same tokens and LRU with MOE_CUDA_EVAL_EVERY, MOE_CUDA_TOKEN_AHEAD, MOE_PIPE_ONE and MOE_PIPE_GUESS_BF16")


if __name__ == "__main__":
    child() if "--child" in sys.argv else main()
