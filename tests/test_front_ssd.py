#!/usr/bin/env python3
"""The stored front: an exact SSD round trip, checked on the toy engine --
within one engine, between two, and across a process boundary.

_build_front writes straight to SSD (pcache.save_front) instead of keeping a
second, in-RAM copy, and _reuse_front reads it back (pcache.load_front), from
a directory every engine with the same fingerprint shares, so a restarted
server starts from the front its last run built. Things that have to hold
for that to be safe:

  1. array_equal, not just close: every array _build_front just wrote (KV
     keys/values trimmed to offset, GDN conv + recurrent state, plus the
     KVCache offsets) equals what pcache.load_front hands back for the same
     key. Bytes to disk and back, nothing recomputed -- anything less than
     exact is a bug in the round trip, not GPU associativity noise. (That
     noise is real and separate: a build-vs-one-prefill
     comparison carries a ~1e-3 gap that reproduces unchanged on plain
     e4a3758, with none of this file's or moe_stream.py's changes applied --
     not this lever's business.)
  2. A front reload from SSD by a fresh engine with the same fingerprint
     (_reuse_front hits, no _build_front call at all) answers the SAME
     question with the SAME greedy tokens as generation continuing straight
     from the cache _build_front just prefilled in GPU memory.
  3. The same from another process: a child builds its own toy engine, finds
     the front this process wrote and answers with the same tokens.
  4. A front file gone bad is a miss: the request prefills for real, answers
     the same, and leaves a good file behind.

No checkpoint: support.toy_llm is a KB-scale model built from the real
mlx_lm classes (ArraysCache/KVCache, GatedDeltaNet, RoPE) plus the real Qwen
tokenizer, same as selftest() uses. A toy engine is
built without __init__, so it has no fingerprint of its own: the test gives
its engines one by hand (FP). pcache.FRONTS is pointed at a scratch
directory, not the real ~/.cache/moe-ssd-offload/fronts/ the server
writes to.

    python tests/test_front_ssd.py
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

import mlx.core as mx

import pcache
import support
from support import gdn, kv, toy_llm

TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string", "description": "City name"}},
            "required": ["city"]}}},
]

QUESTION = "What is the weather in Paris?"
MESSAGES = [{"role": "user", "content": QUESTION}]
FP = "70ffee0070ffee00"

# Typical sequential SSD read speeds, used only to turn the toy front's byte count into an expected
# cost for a real ~177 MB front, so no real front is read here.
SSD_GBPS_FAST = 5.3
SSD_GBPS_SLOW = 3.5
SMALL_CLIENT_FRONT_MB = 177


def engine(fp=FP):
    llm = toy_llm("slice")
    llm._front_fp = fp
    return llm


def _arrays(cache):
    """Every stored array, KV first then GDN, trimmed to what is held."""
    return [a for pair in kv(cache) for a in pair] + list(gdn(cache))


def _offsets(cache):
    return [c.offset for c in cache if hasattr(c, "offset")]


def check_array_equal():
    """_build_front's cache, byte for byte, against what load_front returns."""
    llm = engine("a" * 16)              # a directory of its own: 2. starts cold
    llm._tools = TOOLS
    ids = llm.tokenizer.encode(llm._raw_render(MESSAGES))

    n = llm._build_front(ids)
    assert n, "the tool block was not recognised as the front of the render"
    front = ids[:n]
    assert pcache.list_fronts(llm._front_root()) == [front]

    built = [mx.array(a) for a in _arrays(llm._cache)]
    mx.eval(built)
    offsets_built = _offsets(llm._cache)

    t0 = time.perf_counter()
    loaded = pcache.load_front(pcache.front_key(front), front, llm._front_root())
    restore_s = time.perf_counter() - t0
    assert loaded is not None, "the front this call just wrote did not load back"

    loaded_arrays = _arrays(loaded)
    assert len(loaded_arrays) == len(built), (len(loaded_arrays), len(built))
    for i, (a, b) in enumerate(zip(built, loaded_arrays)):
        assert a.shape == b.shape and mx.array_equal(a, b), (
            f"array {i} not bit-identical: shapes {a.shape}/{b.shape}")
    assert _offsets(loaded) == offsets_built, (_offsets(loaded), offsets_built)

    front_bytes = sum(a.nbytes for a in built)
    print(f"1. array_equal: {len(built)} arrays, {front_bytes / 1e3:.1f} KB, "
          f"restore {restore_s * 1e3:.3f} ms")
    return front_bytes, restore_s


def check_tokens_disk_vs_memory():
    """A hit loaded from SSD by a fresh engine answers exactly like a front
    just built in RAM -- "front hit from disk" against "front built in
    memory", the two paths production takes on a restart and on the first
    run."""
    memory = engine()
    reply_mem = memory.ask(messages=MESSAGES, tools=TOOLS)
    stats_mem = memory.stats()
    assert stats_mem["cached_tokens"] == 0, (
        "the very first call cannot have hit anything -- it built the front")
    [front] = pcache.list_fronts(memory._front_root())

    disk = engine()                     # nothing carried over but the fingerprint
    reply_disk = disk.ask(messages=MESSAGES, tools=TOOLS)
    stats_disk = disk.stats()
    assert stats_disk["cached_tokens"] == len(front), (
        "the second engine did not actually hit the stored front on disk",
        stats_disk)
    assert reply_mem == reply_disk, (reply_mem, reply_disk)
    print(f"2. tokens identical: {len(front)}-token front, "
          f"{stats_mem['prompt_tokens']} prompt tokens cold / "
          f"{stats_disk['prompt_tokens']} warm")
    return reply_mem, front


def check_other_process(reply, front):
    """A child process with its own toy engine and the same fingerprint:
    the front this process stored is its first request's, same tokens."""
    child = f"""
import json, sys
sys.path.insert(0, {str(REPO)!r})
sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})
from pathlib import Path
import test_front_ssd as t
t.pcache.FRONTS = Path({str(pcache.FRONTS)!r})
llm = t.engine()
out = llm.ask(messages=t.MESSAGES, tools=t.TOOLS)
print(json.dumps({{"reply": out, "cached": llm.stats()["cached_tokens"]}}))
"""
    r = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True,
                       timeout=200)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    got = json.loads(r.stdout.strip().splitlines()[-1])
    assert got["cached"] == len(front), got
    assert got["reply"] == reply, (got["reply"], reply)
    print(f"3. another process: its first request restored the {len(front)}-token "
          f"front this one stored, same tokens")


def check_bad_file(reply, front):
    """A front file cut short is a miss: a real prefill, the same reply, and
    a readable file again after it."""
    llm = engine()
    p = pcache.front_path(llm._front_root(), pcache.front_key(front))
    p.write_bytes(p.read_bytes()[: p.stat().st_size // 2])
    assert pcache.list_fronts(llm._front_root()) == []
    out = llm.ask(messages=MESSAGES, tools=TOOLS)
    assert llm.stats()["cached_tokens"] == 0, llm.stats()
    assert out == reply, (out, reply)
    assert pcache.list_fronts(llm._front_root()) == [front], "the bad file was not replaced"
    print("4. a front file cut short: a miss, the same reply, a good file again")


def main():
    import shutil
    if not support.has_tokenizer():
        print("SKIPPED (no tokenizer: set MOE_TOY_TOKENIZER)")
        return
    pcache.FRONTS = Path(tempfile.mkdtemp(prefix="moe-front-ssd-test-"))
    try:
        front_bytes, restore_s = check_array_equal()
        reply, front = check_tokens_disk_vs_memory()
        check_other_process(reply, front)
        check_bad_file(reply, front)
    finally:
        shutil.rmtree(pcache.FRONTS, ignore_errors=True)

    if front_bytes:
        toy_gbps = (front_bytes / 1e9) / restore_s if restore_s > 0 else float("inf")
        print(f"toy front: {front_bytes / 1e3:.1f} KB restored in "
              f"{restore_s * 1e3:.3f} ms ({toy_gbps:.2f} GB/s effective -- "
              f"dominated by open/parse overhead at this size, not bandwidth)")
    for name, gbps in (("fast SSD", SSD_GBPS_FAST), ("slow SSD", SSD_GBPS_SLOW)):
        # seconds = (MB * 1e6 bytes/MB) / (GB/s * 1e9 bytes/GB); *1e3 for ms.
        ms = SMALL_CLIENT_FRONT_MB / gbps
        print(f"{SMALL_CLIENT_FRONT_MB} MB front at {gbps} GB/s ({name}): "
              f"~{ms:.0f} ms sequential read")

    print("test_front_ssd: ok")


if __name__ == "__main__":
    main()
