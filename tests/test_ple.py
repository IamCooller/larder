#!/usr/bin/env python3
"""The streamed PLE rows equal the model's own lookup, and the engine's tokens
equal the model's, on a tiny qwen4_exp checkpoint.

Three checks, all bit-for-bit, none needing the 111 GB:
  1. StreamingPLE(ple.bin) == _ShardedEmbedding(weights) on random row ids.
  2. The engine's sync path on the packed tiny model produces the same greedy
     tokens as mlx_lm on the unpacked one -- the whole streaming stack
     (packed experts, cache bias at lambda 0, PLE rows) against the model as
     shipped.
  3. --defer == sync from a cleared arena (defer_ab --clear-arena).
Needs mlx_lm with qwen4_exp (the PR branch) and a synth_qwen4 build:

    python synth_qwen4.py --out /tmp/rpp/tiny_q4 --tokenizer-from <a Qwen3-Next checkpoint directory>
    python test_ple.py /tmp/rpp/tiny_q4
"""
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import numpy as np

try:
    from mlx_lm.models import qwen4_exp  # noqa: F401
except ImportError:
    print("test_ple: skipped (mlx_lm without qwen4_exp)")
    sys.exit(0)

import mlx_lm
import stream_fast
from moe_stream import StreamingLLM

if len(sys.argv) < 2:
    print("test_ple: skipped (needs a tiny pack built by synth_qwen4.py: pass its directory as the first argument)")
    sys.exit(0)
src = Path(os.path.expanduser(sys.argv[1]))
slim = src.parent / (src.name + "-slim")
assert slim.exists(), f"run synth_qwen4.py first ({slim} missing)"

# 1. rows
model, tok = mlx_lm.load(str(src), lazy=False)
# The checkpoint's n-gram constants are seed 1234, the config names no seed,
# and the model file rebuilds its hash constants from seed 0: the engine hashes
# with the checkpoint's (stream_fast.adopt_ngram_constants). The reference
# gets the same, or check 2 compares against the seed-0 model.
ngram = next(m for _, m in model.named_modules() if "_mults" in m)
seed0 = ngram["_mults"]
assert stream_fast.adopt_ngram_constants(model) >= 1, "the seed-0 constants were not replaced"
assert mx.array_equal(ngram["_mults"], ngram["layer_multipliers"]).item()
assert not mx.array_equal(ngram["_mults"], seed0).item()
stack = stream_fast._layer_stack_of(model)
ple_layer = next(l for l in stack["layers"] if getattr(l, "ple", None) is not None)
sharded = ple_layer.ple.ple_embedding.ngram_embedding
cfg_q = json.loads((slim / "config.json").read_text()).get("quantization", {})
index = json.loads((slim / "ple_index.json").read_text())
fd = os.open(str(slim / "ple.bin"), os.O_RDONLY)
entries = {int(k.rsplit("_", 1)[1]): e for k, e in index.items()}
first_key = next(k for k in index if k.endswith(".shard_0"))
q = cfg_q.get(first_key) or {"group_size": sharded.shard_0.group_size, "bits": sharded.shard_0.bits}
streamer = stream_fast.StreamingPLE(fd, entries, q["group_size"], q["bits"], "affine",
                                    int(sharded.dim), int(sharded.rows))
total_rows = int(sharded.rows) * int(sharded.n_shards)
gid = mx.array(np.random.default_rng(0).integers(0, total_rows, size=(1, 3, 16)), dtype=mx.int64)
a, b = sharded(gid), streamer(gid)
mx.eval(a, b)
diff = float(mx.max(mx.abs(a - b)))
assert diff == 0.0 and a.shape == b.shape, (diff, a.shape, b.shape)
print(f"1. PLE rows: streamed == model, {gid.size} rows, max diff {diff}")

# 1b. the host hash MOE_PLE_AHEAD reads a decode pass's rows by (_ngram_rows) == the model's own row ids, over
#     contexts with and without EOS
import mlx.nn as nn  # noqa: E402


class Grab(nn.Module):
    def __call__(self, g):
        self.__dict__["gid"] = g
        return mx.zeros((*g.shape, 1))


real, grab = ngram.ngram_embedding, Grab()
ngram.ngram_embedding = grab
rng = np.random.default_rng(1)
vocab, eos = model.args.text.vocab_size, ngram.eos_token_id
n_ctx = ngram.ngram_size - 1
for i in range(400):
    hist = rng.integers(0, vocab, size=(1, n_ctx + 1))
    hist[rng.random(hist.shape) < 0.3] = eos
    ngram(mx.array(hist[:, n_ctx:], dtype=mx.int32), mx.array(hist[:, :n_ctx], dtype=mx.int32))
    want = np.asarray(grab.gid).reshape(-1)
    got = stream_fast._ngram_rows(ngram, hist)
    assert np.array_equal(want, got), (i, hist, want, got)
ngram.ngram_embedding = real
print("1b. host n-gram row ids == the model's, 400 contexts (30% EOS)")

# 2. tokens: mlx_lm greedy on the unpacked model vs the SAME generator on the
#    engine's installed model (packed experts, streamed PLE, bias at lambda 0)
#    from the same prompt ids -- the streaming stack alone, no chat-template
#    or stub handling in the comparison.
prompt = "\u041e\u0431\u044a\u044f\u0441\u043d\u0438 \u043a\u043e\u0440\u043e\u0442\u043a\u043e, \u0447\u0442\u043e \u0442\u0430\u043a\u043e\u0435 \u043a\u044d\u0448."
ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True)
ref = [r.token for r in mlx_lm.stream_generate(model, tok, ids, max_tokens=24)]
del model
mx.clear_cache()
stream_fast.WIDE_GROUP = 3        # the tiny union still goes through the grouped wide pass
llm = StreamingLLM(model=str(slim), resident=8, max_tokens=24, cache_bias=0.0, defer=False)
ng = next(m for _, m in llm.model.named_modules() if "_mults" in m)
assert ng["layer_multipliers"].dtype == mx.int64
assert mx.array_equal(ng["_mults"], ng["layer_multipliers"]).item(), "engine hashes with config constants"
assert mx.array_equal(ng["_mults"], ngram["layer_multipliers"]).item()
ours = [r.token for r in mlx_lm.stream_generate(llm.model, llm.tokenizer, ids, max_tokens=24)]
cp = next((i for i, (x, y) in enumerate(zip(ref, ours)) if x != y), min(len(ref), len(ours)))
print(f"2. tokens: model {ref[:6]}... engine {ours[:6]}... common prefix {cp}/{len(ref)}")
assert cp == len(ref), "the streaming stack diverges from the model as shipped"

# 3. defer == sync from a cleared arena
defer_ab = Path(__file__).resolve().parent.parent / "defer_ab.py"
if not defer_ab.exists():
    print("step 3 skipped: needs defer_ab.py from the development tree")
else:
    r = subprocess.run([sys.executable, str(defer_ab),
                        "--model", str(slim), "--resident", "8", "--tokens", "40", "--repeats", "1",
                        "--arms", "sync,sync,defer", "--clear-arena", "--allow-busy", "--tag", "tiny_q4"],
                       capture_output=True, text=True)
    rounds = [l.strip() for l in r.stdout.splitlines() if "tokens in common" in l]
    print("3. " + " | ".join(rounds))
    assert r.returncode == 0 and rounds and all("(identical)" in l for l in rounds), r.stdout[-800:] + r.stderr[-800:]
print("test_ple: ok")
