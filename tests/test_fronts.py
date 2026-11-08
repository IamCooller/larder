#!/usr/bin/env python3
"""The stored fronts' index, checked without a model: the directory is it.

An application does not send one prompt shape: it may alternate a ~4k-token tool sheet with a
short classifier instruction, so a single stored front is thrown away by every request of the
other kind, and a restarted server must not lose its fronts either.

The real `_build_front` and `_reuse_front` run here, on engines built with
__new__ -- __init__ loads 16 GB of experts and none of it decides whether a
front is reused. The prefill is stood in for: each "front" is a one-position
KVCache whose value marks which front it is, written to and read from real
files under a scratch pcache.FRONTS. Exactness of a real front's bytes is
test_front_ssd.py's job, not this file's.

    python tests/test_fronts.py
"""
import collections
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import mlx.core as mx
from mlx_lm.models.cache import KVCache

import moe_stream
import support
import pcache

FP = "0123456789abcdef"


def tiny_cache(mark):
    c = KVCache()
    c.update_and_fetch(mx.full((1, 1, 1, 4), float(mark)), mx.zeros((1, 1, 1, 4)))
    return [c]


def make(fp=FP):
    """An engine whose prefill is a marked tiny cache, fingerprint `fp`."""
    engine = moe_stream.StreamingLLM.__new__(moe_stream.StreamingLLM)
    engine.model_path = support.MODEL_LABEL
    engine._front_fp = fp
    engine._front_ids = lambda ids: list(ids)

    def prefill(front):
        engine._cache = tiny_cache(front[0])
    engine._prefill = prefill
    return engine


def keep(engine, front):
    assert engine._build_front(front) == len(front)


def mark(engine):
    return int(engine._cache[0].keys[0, 0, 0, 0].item())


def files(fp=FP):
    return sorted(p.name for p in pcache.front_dir(support.MODEL_LABEL, fp).glob("*"))


def check_bookkeeping():
    sheet = list(range(1, 401))         # the tool sheet's ids
    classify = [900, 901, 902]          # a short classifier instruction

    engine = make()
    keep(engine, sheet)
    keep(engine, classify)

    # The alternation a single slot got wrong: both shapes stay warm.
    assert engine._reuse_front(sheet + [7, 8]) == len(sheet)
    assert engine._reuse_front(classify + [7, 8]) == len(classify)
    assert engine._reuse_front(sheet + [9]) == len(sheet), "one alternation and it was gone"
    assert engine._reuse_front(classify + [9]) == len(classify)

    # A shape nobody prefilled is still a miss -- reuse matches on the ids,
    # never merely on having something kept.
    assert engine._reuse_front([500, 501, 502]) == 0

    # The cache handed back belongs to the front that matched, not to whichever
    # was stored last.
    engine._reuse_front(sheet + [1])
    assert mark(engine) == sheet[0], mark(engine)

    # Bounded by count: one shape past the slots evicts the least recently used,
    # so this can never grow into an unbounded pile of kept KV.
    engine = make("1" * 16)
    shapes = [[i * 1000 + n for n in range(3)] for i in range(1, moe_stream.FRONT_SLOTS + 2)]
    for shape in shapes:
        keep(engine, shape)
    assert len(files("1" * 16)) == moe_stream.FRONT_SLOTS, files("1" * 16)
    assert engine._reuse_front(shapes[0]) == 0, "the oldest shape should have gone"
    assert engine._reuse_front(shapes[-1]) == len(shapes[-1])

    # Bounded by tokens too, because a front's cost is its length: the tool sheet
    # is ~72 MB of KV on this model and a short instruction a small fraction.
    engine = make("2" * 16)
    huge = list(range(1, moe_stream.FRONT_TOKEN_BUDGET))
    keep(engine, classify)
    keep(engine, huge)
    assert engine._reuse_front(huge) == len(huge)
    assert engine._reuse_front(classify) == 0, "kept two fronts past the token budget"

    # And the newest is always kept, however large: dropping the front this very
    # request just prefilled would mean prefilling it again next time.
    engine = make("3" * 16)
    keep(engine, list(range(1, moe_stream.FRONT_TOKEN_BUDGET * 2)))
    assert len(files("3" * 16)) == 1, files("3" * 16)

    # LRU, not FIFO: using a front makes it the one that survives.
    engine = make("4" * 16)
    keep(engine, sheet)
    keep(engine, classify)
    engine._reuse_front(sheet)                  # sheet is the most recent again
    # One more shape than the remaining slots, so exactly one entry has to go.
    for i in range(moe_stream.FRONT_SLOTS - 1):
        keep(engine, [50_000 + i])
    assert engine._reuse_front(sheet) == len(sheet), "evicted the front still in use"
    assert engine._reuse_front(classify) == 0

    # One entry per shape: prefilling the same front twice must not spend a
    # slot on a duplicate.
    engine = make("5" * 16)
    keep(engine, sheet)
    keep(engine, sheet)
    assert len(files("5" * 16)) == 1, files("5" * 16)
    assert engine._reuse_front(sheet) == len(sheet)

    # A prefix relationship must not cost the difference: the longer front wins.
    engine = make("6" * 16)
    engine._prefill = lambda front: setattr(engine, "_cache", tiny_cache(len(front)))
    keep(engine, [1, 2])
    keep(engine, [1, 2, 3, 4])
    assert engine._reuse_front([1, 2, 3, 4, 5]) == 4
    assert mark(engine) == 4
    print("bookkeeping: both shapes of an alternation stay warm; a miss is a miss; "
          "count and token bounds, least recently used out; the newest kept; one "
          "file per shape; the longest prefix wins")


def check_shared():
    """A fresh engine with the same fingerprint -- a restarted server -- finds
    what another built and sees its evictions; another fingerprint sees none
    of it. And the same across a process boundary."""
    fp = "7" * 16
    a, b, other = make(fp), make(fp), make("8" * 16)
    sheet = [11, 12, 13, 14]
    keep(a, sheet)
    assert b._reuse_front(sheet + [1]) == len(sheet), "a fresh engine missed a stored front"
    assert mark(b) == 11
    assert other._reuse_front(sheet + [1]) == 0, "a front crossed fingerprints"

    # Across processes: fill the directory in a child, a front this process
    # touched last survives it and the untouched one goes.
    keep(a, [21, 22])                           # the least recently used after the touch
    assert b._reuse_front(sheet) == len(sheet)  # touched: sheet is now the newest
    fillers = [[30_000 + i] for i in range(moe_stream.FRONT_SLOTS - 1)]
    child = f"""
import sys, json
sys.path.insert(0, {str(REPO)!r})
sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})
from pathlib import Path
import test_fronts as t
t.pcache.FRONTS = Path({str(pcache.FRONTS)!r})
e = t.make({fp!r})
for f in {fillers!r}:
    t.keep(e, f)
print(json.dumps([e._reuse_front([11, 12, 13, 14, 1]), e._reuse_front([21, 22, 1])]))
"""
    out = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True,
                         timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    got = json.loads(out.stdout.strip().splitlines()[-1])
    assert got == [len(sheet), 0], got
    assert a._reuse_front([21, 22]) == 0, "the child's eviction did not hold here"
    assert a._reuse_front(sheet) == len(sheet)
    print("shared: a fresh engine hits another's front; another fingerprint does not; "
          "a child process evicts the least recently used and keeps what this one touched")


def check_across_fingerprints():
    """The whole of FRONTS keeps one engine's bound: another fingerprint's
    directory goes whole, least recently used first, an empty one at once;
    this engine's own is never touched by it."""
    for d in pcache.FRONTS.glob("*/fp-*"):
        for p in d.iterdir():
            p.unlink()
        d.rmdir()
    old, mid, now = make("a" * 16), make("b" * 16), make("c" * 16)
    third = moe_stream.FRONT_TOKEN_BUDGET // 3
    keep(old, list(range(1, third)))
    keep(mid, list(range(2, third + 1)))
    empty = pcache.front_dir(support.MODEL_LABEL, "d" * 16)
    empty.mkdir(parents=True)
    keep(now, list(range(3, third + 2)))
    assert not empty.exists(), "an empty fingerprint directory was left behind"
    assert files("a" * 16) and files("b" * 16), "pruned while under the bound"
    keep(now, list(range(4, third + 3)))        # now over: the oldest directory goes
    assert not pcache.front_dir(support.MODEL_LABEL, "a" * 16).exists()
    assert files("b" * 16) and len(files("c" * 16)) == 2
    print("across fingerprints: over one engine's bound the least recently used "
          "directory went whole, an empty one at once, the writer's own kept")


def check_unreadable_and_orphans():
    """A file that does not load as a front is a miss, and goes at the next
    write; so does a temp file whose writer is dead, and only that one."""
    fp = "e" * 16
    engine = make(fp)
    keep(engine, [41, 42])
    d = pcache.front_dir(support.MODEL_LABEL, fp)
    p = pcache.front_path(d, pcache.front_key([41, 42]))
    p.write_bytes(b"not a safetensors file")
    assert engine._reuse_front([41, 42, 1]) == 0, "a corrupt front was used"
    whole = pcache.front_path(d, pcache.front_key([43]))
    keep(engine, [43])
    whole.write_bytes(whole.read_bytes()[: whole.stat().st_size // 2])
    assert engine._reuse_front([43, 1]) == 0, "a file cut short was used"
    dead = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"],
                          capture_output=True, text=True).stdout.strip()
    orphan = d / f"{'f' * 24}-{dead}-1.tmp.safetensors"
    live = d / f"{'f' * 24}-{os.getpid()}-1.tmp.safetensors"
    orphan.write_bytes(b"x")
    live.write_bytes(b"x")
    keep(engine, [44])
    names = files(fp)
    assert p.name not in names and whole.name not in names, names
    assert orphan.name not in names and live.name in names, names
    live.unlink()
    print("unreadable: a corrupt and a cut-short front are misses and go at the next "
          "write; a dead writer's temp file goes, a live one stays")


def check_disk_floor():
    """No front is written that would leave less free disk than the repo's
    own runs insist on (host_stats.MIN_FREE_DISK_GB); the request goes on."""
    fp = "9" * 16
    engine = make(fp)
    real = pcache.shutil.disk_usage
    usage = collections.namedtuple("usage", "total used free")
    floor = pcache.MIN_FREE_DISK_GB * 2**30
    try:
        pcache.shutil.disk_usage = lambda p: usage(0, 0, floor + 10)
        keep(engine, [61, 62])
        assert not files(fp), files(fp)
        assert engine._reuse_front([61, 62]) == 0
        pcache.shutil.disk_usage = lambda p: usage(0, 0, floor + 2**20)
        keep(engine, [61, 62])
        assert engine._reuse_front([61, 62]) == 2
    finally:
        pcache.shutil.disk_usage = real
    print(f"disk floor: not written 10 bytes above host_stats.MIN_FREE_DISK_GB "
          f"({pcache.MIN_FREE_DISK_GB} GB) of free disk, written 1 MB above it")


def check_fingerprint():
    """Settings that can change a front's arrays, the code and the pack name
    different directories; the ones pcache.NOT_IN_FRONT leaves out do not."""
    pack = Path(tempfile.mkdtemp(prefix="moe-model-test-"))
    (pack / "config.json").write_text("{}")
    (pack / "model.safetensors").write_bytes(b"weights")
    model = str(pack)
    base = {"top_k": 0, "resident": 21, "cache_bias": None, "bias_budget": 0.2,
            "prefill_step": None, "mem_peak_gb": 2.0, "stub": "slice",
            "max_tokens": 512, "mtp": None, "draft_window": 0}
    fp = pcache.fingerprint(model, base)
    assert fp == pcache.fingerprint(model, dict(base))
    for k, v in (("cache_bias", 0.5), ("bias_budget", 0.0), ("top_k", 6), ("resident", 32),
                 ("prefill_step", 0), ("mem_peak_gb", None), ("mem_pct", 80),
                 ("reserve_pct", 25)):
        assert pcache.fingerprint(model, {**base, k: v}) != fp, k
    for k, v in (("stub", "keep"), ("max_tokens", 64), ("thinking", True)):
        assert pcache.fingerprint(model, {**base, k: v}) == fp, k
    src = pcache.FRONTS / "mod.py"             # removed with the scratch FRONTS
    saved = pcache.FRONT_SOURCES
    try:
        src.write_text("x = 1\n")
        pcache.FRONT_SOURCES = saved + (str(src),)
        one = pcache.fingerprint(model, base)
        src.write_text("x = 2\n")
        assert pcache.fingerprint(model, base) != one, "an edited module kept its key"
    finally:
        pcache.FRONT_SOURCES = saved
    os.environ["MOE_FUSE_PROJ"] = "1"
    try:
        assert pcache.fingerprint(model, base) != fp, "a MOE_* switch kept its key"
    finally:
        del os.environ["MOE_FUSE_PROJ"]
    assert pcache.front_dir(model, fp).name == f"fp-{fp}"
    shutil.rmtree(pack, ignore_errors=True)
    print(f"fingerprint: {fp}; cache_bias, bias_budget, top_k, resident, prefill_step, "
          f"mem_peak_gb, mem_pct, reserve_pct, an edited module and a MOE_* switch each "
          f"change it; stub, max_tokens and thinking do not")


def main():
    pcache.FRONTS = Path(tempfile.mkdtemp(prefix="moe-fronts-test-"))
    try:
        check_bookkeeping()
        check_shared()
        check_across_fingerprints()
        check_unreadable_and_orphans()
        check_disk_floor()
        check_fingerprint()
    finally:
        shutil.rmtree(pcache.FRONTS, ignore_errors=True)
    print("test_fronts: ok")


if __name__ == "__main__":
    main()
