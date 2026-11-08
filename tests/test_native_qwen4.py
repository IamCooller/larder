#!/usr/bin/env python3
"""MOE_NATIVE=1 on a tiny qwen4_exp -- Qwen3.8-Flash-Next's architecture:
hyper-connections, a PLE layer read from ple.bin, the sparse-attention
indexer, logit-routed experts beside a shared one -- packed the way the
engine packs the real checkpoint. Native decode == Python decode, greedy
token ids and logprobs bitwise, over two turns on one cache, in both expert
layouts:

  arena   per-layer arenas of 8 slots (top_k 4);
  store   --mem-peak-gb's shared stores, quotas cut by hand to 0-6 rows a
          layer so the ring halves carry most misses.

Needs mlx_lm with qwen4_exp, moe_native built for this venv
(native/build.sh) and a synth_qwen4.py build:

    python synth_qwen4.py --out /tmp/rpp/tiny_q4 --tokenizer-from <a Qwen3-Next checkpoint directory>
    python tests/test_native_qwen4.py /tmp/rpp/tiny_q4-slim
"""
import gc
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import numpy as np

try:
    from mlx_lm.models import qwen4_exp  # noqa: F401
except ImportError:
    print("test_native_qwen4: skipped (mlx_lm without qwen4_exp)")
    sys.exit(0)
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache

import stream_fast
from moe_stream import StreamingLLM

QUOTA = [0, 2, 3, 6, 1, 4, 2, 0]
TURNS = [("\u041e\u0431\u044a\u044f\u0441\u043d\u0438 \u043a\u043e\u0440\u043e\u0442\u043a\u043e, \u0447\u0442\u043e \u0442\u0430\u043a\u043e\u0435 \u043a\u044d\u0448.", 32), (" And why is it fast? Two sentences.", 24)]


def run(slim, mode, native):
    stream_fast._NATIVE_ENV = native
    peak = stream_fast.phys_footprint() / 1e9 + 4.0 if mode == "store" else None
    llm = StreamingLLM(model=str(slim), resident=8, max_tokens=64, cache_bias=0.0,
                       defer=False, mem_peak_gb=peak)
    model, glus = llm.model, stream_fast.glus_of(llm.model)
    if mode == "store":
        quota = {id(g): QUOTA[i % len(QUOTA)] for i, g in enumerate(glus)}
        llm._budget._quotas = lambda pool: quota
    park0 = stream_fast.TIMERS["native_park"]
    cache, toks, lps = make_prompt_cache(model), [], []
    for text, n in TURNS:
        ids = mx.array(llm.tokenizer.encode(text), dtype=mx.int32)
        for t, lp in generate_step(ids, model, max_tokens=n, prompt_cache=cache):
            toks.append(int(t))
            lps.append(np.array(lp.astype(mx.float32)))
    stream_fast.native_sync()
    ran = stream_fast.TIMERS["native_park"] > park0
    ring = sum(len(g._ring) for g in glus)
    if native:
        assert stream_fast._NATIVE is not None and not stream_fast._NATIVE.agent.error
    del llm, model, glus, cache
    gc.collect()
    mx.clear_cache()
    return toks, lps, ran, ring


def main():
    slim = Path(os.path.expanduser(sys.argv[1]))
    assert (slim / "experts.bin").exists() and (slim / "ple.bin").exists(), \
        f"{slim}: not a synth_qwen4.py -slim pack"
    # Native first: its wrapper must sit inside the budget's (install_native).
    for mode in ("arena", "store"):
        b = run(slim, mode, True)
        a = run(slim, mode, False)
        assert b[2] and not a[2], f"{mode}: the native arm must run native passes, the other none"
        assert a[0] == b[0], f"{mode}: tokens differ\n{a[0]}\n{b[0]}"
        bad = [i for i, (x, y) in enumerate(zip(a[1], b[1])) if not np.array_equal(x, y)]
        assert not bad, f"{mode}: logprobs differ at steps {bad}"
        if mode == "store":
            assert a[3] and a[3] == b[3], f"store: ring members {a[3]} vs {b[3]}"
        print(f"{mode}: {len(a[0])} tokens, logprobs identical, native ran"
              + (f", {b[3]} ring members at the end" if mode == "store" else ""), flush=True)
    print("test_native_qwen4: ok")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("test_native_qwen4: skipped (needs a tiny pack built by synth_qwen4.py: pass its directory as the first argument)")
        sys.exit(0)
    main()
