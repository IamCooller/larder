#!/usr/bin/env python3
"""
Keep the model's own state across sessions instead of rebuilding it every time.

Qwen3.6-35B-A3B is a hybrid: 30 of its 40 blocks are GatedDeltaNet and only 10
are ordinary attention (`config.json: layer_types`). A GatedDeltaNet block holds
a state matrix per head and updates it with the delta rule at every token --
which is a fast-weight memory, in the Schlag/Irie/Schmidhuber sense
(arXiv 2102.11174), already inside the model. Measured from the config:

    32 heads x 128 x 128 x 4 bytes  =  2.10 MB per layer
    x 30 linear-attention layers    =  62.9 MB, and CONSTANT in context length

The 10 attention layers hold an ordinary KV cache, which does grow -- but only
a quarter of the model pays that.

`moe_stream.py` calls `make_prompt_cache` on construction and throws the whole
thing away on `reset()` and at every process start. So the most memory-like
structure in the system is discarded every session and rebuilt by prefilling the
same prefix again.

This file stops that. Run a persistent prefix once -- who the user is, how they
work, what they are building -- snapshot the hybrid cache, and load it at the
start of every session. No training, no gradient, no forgetting, and no expert
reads: the prefix is not re-prefilled, it is restored.

`mlx_lm.models.cache.save_prompt_cache` serialises anything with a `state`
property, which covers the recurrent and conv state as well as KV.

    import pcache
    pcache.prime(llm, [{"role": "system", "content": "..."}], "me")  # once
    pcache.load(llm, "me")                                          # every session

    python3 pcache.py --self-check    # real round trip
"""

import argparse
import errno
import hashlib
import json
import os
import shutil
import sys
import threading
import weakref
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import mlx_lm
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.models import cache as _cache_mod
from mlx_lm.models.cache import make_prompt_cache

import eprofile
from host_stats import MIN_FREE_DISK_GB


def _cache_class(name):
    """The cache class a saved file names, found in mlx_lm's cache module or
    else among the mlx_lm model modules already imported, so a class a model
    file defines for itself no longer misses every stored front (known limit:
    two loaded models defining one name is a KeyError, a miss)."""
    cls = getattr(_cache_mod, name, None)
    if cls is not None:
        return cls
    found = {c for k, m in list(sys.modules.items()) if k.startswith("mlx_lm.models.")
             for c in [getattr(m, name, None)]
             if isinstance(c, type) and issubclass(c, _cache_mod._BaseCache)}
    if len(found) != 1:
        raise KeyError(name)
    return found.pop()


def _from_state(cls, state, meta_state):
    """cls.from_state, except that a model's own class is built by its
    constructor when that takes no arguments: _AttnCache.__init__ makes the
    indexer cache its state setter fills, and from_state's bare __new__
    skips it. The cache module's classes go through from_state as in mlx_lm.
    An ArraysCache slot saved empty (save_prompt_cache below) is None again."""
    obj = None
    if getattr(_cache_mod, cls.__name__, None) is not cls:
        try:
            obj = cls()
        except TypeError:               # needs arguments (_LayerCache)
            pass
    # An mlx_lm without meta_state (from_state(state)) saves "" and reads it back as nothing.
    meta = hasattr(cls, "meta_state")
    if obj is None:
        obj = cls.from_state(state, meta_state) if meta else cls.from_state(state)
    else:
        obj.state = state
        if meta:
            obj.meta_state = meta_state
    if isinstance(obj, _cache_mod.ArraysCache):
        obj.cache = [None if a is not None and a.size == 0 else a for a in obj.cache]
    return obj


def save_prompt_cache(file_name, cache, metadata):
    """mlx_lm.models.cache.save_prompt_cache for both mlx_lm cache APIs, where a
    state leaf that is no array (mx.save_safetensors refuses a None, and an
    mlx_lm without meta_state keeps a KVCache's offset in `state`) is written
    into the metadata under "s:<key>" ("none", "int:<n>") and read back as
    itself."""
    data, scal = {}, {}
    for k, v in tree_flatten([c.state for c in cache]):
        if isinstance(v, mx.array):
            data[k] = v
        elif v is None:
            scal["s:" + k] = "none"
        elif isinstance(v, int):
            scal["s:" + k] = f"int:{int(v)}"
        else:
            raise TypeError(f"prompt cache state leaf {k}: {type(v).__name__}")
    meta = dict(tree_flatten([[getattr(c, "meta_state", "") for c in cache], metadata,
                              [type(c).__name__ for c in cache]]))
    meta.update(scal)
    mx.save_safetensors(file_name, data, meta)


def load_prompt_cache(file_name, return_metadata=False):
    """mlx_lm.models.cache.load_prompt_cache with _cache_class and _from_state, and the state leaves
    save_prompt_cache wrote into the metadata."""
    arrays, meta = mx.load(file_name, return_metadata=True)
    flat = dict(arrays)
    for k, v in meta.items():
        if k.startswith("s:"):
            flat[k[2:]] = None if v == "none" else int(v[4:])
    arrays = tree_unflatten(list(flat.items()))
    info, metadata, classes = tree_unflatten([(k, v) for k, v in meta.items() if not k.startswith("s:")])
    cache = [_from_state(_cache_class(c), s, m) for c, s, m in zip(classes, arrays, info)]
    return (cache, metadata) if return_metadata else cache


# Stored fronts (moe_stream.py's prefilled tool-block/system-prefix caches)
# live here instead of in eprofile.STORE: they are keyed by token ids, not by
# a session tag, and an application's fronts far outlive one profile's habits.
FRONTS = Path.home() / ".cache/moe-ssd-offload/fronts"


def path_for(model_path, tag):
    name = Path(model_path).expanduser().name
    return eprofile.STORE / name / f"pcache-{tag}.safetensors"


def _atomic_save(p, cache, metadata):
    """Write beside the target and rename over it.

    A crash mid-write then leaves the previous file intact instead of a
    truncated one that a loader can only report as "none saved yet" --
    losing a persona, or a front, silently. The temp name keeps the
    .safetensors suffix: mx.save_safetensors appends one when it is missing,
    the same trap np.savez set with .npz.

    A failed write (disk full, ~/.cache read-only) is cleaned up here rather
    than left for the caller: the tmp file is unlinked and the exception
    re-raised, so every caller of this function gets that behaviour for free
    instead of only the ones that remembered to add it.

    The temp name also carries the writer's pid and thread: engines with one
    fingerprint share a front directory, so two writing the same front at
    once must not write into one file, and a writer killed mid-save leaves a
    file prune_fronts can prove orphaned.
    """
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.stem}-{os.getpid()}-{threading.get_ident()}.tmp.safetensors")
    try:
        save_prompt_cache(str(tmp), cache, metadata)
        tmp.replace(p)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def save(llm, tag, model_path=None, extra=None):
    """Snapshot the cache together with what produced it.

    A cache alone is unusable: the engine builds each turn's prompt from the
    tokens it believes are already in the cache (`_seen`) and from the rendered
    history, so all three have to travel together or the next turn is fed
    malformed input. That failure is not loud -- it made the model repeat its
    previous answer forever the last time the prompt increment broke.
    """
    model_path = model_path or llm.model_path
    p = path_for(model_path, tag)
    _atomic_save(p, llm._cache, {
        "tokens": json.dumps(llm._seen),
        "history": json.dumps(llm.history, ensure_ascii=False),
        "ended_cleanly": json.dumps(llm._ended_cleanly),
        # Identity of the packing. Expert ids and cached activations are both
        # meaningless against a different build of the same model.
        "index_sha": eprofile.index_sha(model_path),
        # The chat template renders differently with thinking on, so a cache
        # built one way cannot be continued the other.
        "thinking": json.dumps(bool(llm.thinking)),
        "mlx_lm": getattr(mlx_lm, "__version__", "?"),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **(extra or {}),
    })
    return p


def load(llm, tag, model_path=None, strict=False):
    """Restore a snapshot onto a live engine. Returns its metadata, or None."""
    model_path = model_path or llm.model_path
    p = path_for(model_path, tag)
    if not p.exists():
        return None

    try:
        cache, meta = load_prompt_cache(str(p), return_metadata=True)
    except Exception as e:
        # Same reasoning as eprofile.load: the write is not atomic, this machine
        # kills processes for memory, and a corrupt persona crashed
        # BrainLLM.__init__ where a merely mismatched one degrades politely.
        print(f"  ignoring prompt cache {p.name}: unreadable "
              f"({type(e).__name__})")
        return None
    problems = []
    if not cache:
        print(f"  ignoring prompt cache {p.name}: empty")
        return None
    want = eprofile.index_sha(model_path)
    if meta.get("index_sha") != want:
        problems.append(f"built on packing {meta.get('index_sha')}, this model "
                        f"is {want}")
    if json.loads(meta.get("thinking", "false")) != bool(llm.thinking):
        problems.append(f"built with thinking={meta.get('thinking')}, this "
                        f"engine has {llm.thinking}")
    tokens = json.loads(meta["tokens"])
    # The cache's own length is the ground truth for how many positions it
    # holds. If it disagrees with the token list, the two were not saved
    # together and continuing would feed the model a prompt it never saw.
    held = getattr(cache[0], "offset", None)
    if held is not None and held != len(tokens):
        problems.append(f"holds {held} positions but carries {len(tokens)} "
                        f"tokens")
    if problems:
        msg = f"{p.name}: " + "; ".join(problems)
        if strict:
            raise ValueError(msg)
        print(f"  ignoring prompt cache -- {msg}")
        return None

    llm._cache = cache
    llm._seen = tokens
    llm.history = json.loads(meta["history"])
    llm._ended_cleanly = json.loads(meta.get("ended_cleanly", "false"))
    return meta


# --------------------------------------------------------------------------
# Stored fronts: moe_stream.py's prefilled tool-block/system-prefix caches.
#
# A front is not a conversation, only the cache and the exact ids it was built
# from, so the file is named by those ids and its directory by everything else
# that decides the arrays (`fingerprint`), which lets it outlive the process
# that built it while `load_front` re-checks the ids against the file's own
# metadata so that a hash collision is a miss, never a wrong cache.
# --------------------------------------------------------------------------

# The modules whose code decides a front's arrays: the prefill loop and the
# save (moe_stream, pcache), the MoE forward, its routing and the budget's
# cut (stream_fast, fuse_proj, fused_router), prefill attention (chunkattn)
# and a dense target's multi-row kernels (rowqmm, foldattn), CUDA's GatedDeltaNet
# kernel (gdn_cuda).
FRONT_SOURCES = ("moe_stream.py", "pcache.py", "stream_fast.py", "fuse_proj.py",
                 "fused_router.py", "chunkattn.py", "rowqmm.py", "foldattn.py", "gdn_cuda.py")

# StreamingLLM arguments that cannot change a front's arrays. A front is the
# tool block and the leading system messages prefilled into an empty cache,
# before anything of the turn: the stub and thinking only shape the
# generation prompt after it (whatever they change in the ids is the file's
# key anyway) and max_tokens the decode. A draft model keeps a cache of its own
# (moe_stream._sync_draft_cache), and so does the MTP head (the hcache
# spec.Speculator.generate makes per call). `model` is replaced by the pack's
# own identity. Every other argument, and any added later, is in the
# fingerprint.
NOT_IN_FRONT = {"model", "max_tokens", "thinking", "stub",
                "draft", "draft_tokens", "draft_tau", "draft_window"}


def layout_key(sources, *parts):
    """16 hex chars naming a file in ~/.cache/moe-ssd-offload the way the
    calibration caches there are named (rowqmm/chunkattn/foldattn's
    _layout_path, which inline the same recipe): this device, this MLX, the
    source of every module in `sources` -- an edited module is a new file,
    never a stale one -- and the caller's own parts."""
    key = json.dumps([mx.device_info()["device_name"], mx.__version__,
                      *(hashlib.sha1(Path(s).read_bytes()).hexdigest() for s in sources),
                      *parts], default=str)
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def fingerprint(model_path, settings):
    """What decides a stored front's arrays besides its ids, as 16 hex chars:
    the pack (its config and indexes by hash, its weight files by size and
    mtime, so a rebuilt pack is a new one), the StreamingLLM `settings` not
    in NOT_IN_FRONT, the code (FRONT_SOURCES), the MOE_* switches, mlx_lm's
    version and -- through layout_key -- the device and MLX's. Engines that
    agree on all of it build the same front from the same ids, whatever
    process they run in.

    A front is prefilled with the cache bias off (stream_fast.bias_off), so
    its routing does not follow the building process's arena. At any bias an
    MoE pass's kernel path can still follow what the arena holds (bf16 ties)."""
    path = Path(model_path).expanduser()
    pack = [(n, hashlib.sha1((path / n).read_bytes()).hexdigest())
            for n in ("config.json", "model.safetensors.index.json",
                      "experts_index.json") if (path / n).exists()]
    pack += sorted((p.name, p.stat().st_size, p.stat().st_mtime_ns)
                   for p in path.iterdir() if p.suffix in (".safetensors", ".bin"))
    here = Path(__file__).parent
    return layout_key([here / s for s in FRONT_SOURCES], pack,
                      sorted((k, v) for k, v in settings.items() if k not in NOT_IN_FRONT),
                      sorted((k, v) for k, v in os.environ.items() if k.startswith("MOE_")),
                      getattr(mlx_lm, "__version__", "?"))


def front_dir(model_path, fp):
    """Where every engine with fingerprint `fp` keeps its fronts:
    FRONTS/<model name>/fp-<fp>/. The prefix keeps these apart from the
    <pid>-<id> directories of front_engine_dir."""
    return FRONTS / Path(model_path).expanduser().name / f"fp-{fp}"


def front_key(front):
    return hashlib.sha256(json.dumps(front).encode()).hexdigest()[:24]


def _gone(pid):
    """Whether `pid` no longer runs: a fact os.kill(pid, 0) proves, not an
    age guess, so nothing a live process still uses is swept for being old.
    A pid this process may not signal (another user's) counts as running."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except OSError:
        pass
    return False


def _prune_dead_engines(base):
    """Remove sibling engine directories whose process no longer runs.

    Each is named "<pid>-<hex id(engine)>". A directory this process itself
    made is skipped (its own pid, still very much alive), and so is one
    whose pid this process has no permission to signal -- only a pid that
    `_gone` proves gone gets removed. That is a fact, not an age guess, so a
    long-running server's own fronts are never swept just for being old.
    """
    if not base.is_dir():
        return
    my_pid = os.getpid()
    for child in base.iterdir():
        pid_s = child.name.split("-", 1)[0]
        if pid_s.isdigit() and int(pid_s) != my_pid and _gone(int(pid_s)):
            shutil.rmtree(child, ignore_errors=True)


def front_engine_dir(model_path, engine):
    """A front directory of one engine's own, FRONTS/<model name>/<pid>-<id>/,
    for an engine with no fingerprint, so two engines with different settings
    never overwrite each other's file at the same content hash, removed by
    weakref.finalize (not atexit, since an engine can be replaced long before
    the process exits) with `_prune_dead_engines` as the backstop for a hard
    kill."""
    base = FRONTS / Path(model_path).expanduser().name
    _prune_dead_engines(base)
    d = base / f"{os.getpid()}-{id(engine):x}"
    d.mkdir(parents=True, exist_ok=True)
    weakref.finalize(engine, shutil.rmtree, str(d), ignore_errors=True)
    return d


def front_path(front_dir, key):
    return Path(front_dir) / f"{key}.safetensors"


def save_front(cache, front, front_dir):
    """Serialise one prefilled front (moe_stream._build_front) to SSD and return
    its key, with no clone first because nothing keeps a reference to the arrays
    after the write, and raising OSError(ENOSPC), which _build_front takes as any
    failed write, if the file would leave less free disk than
    host_stats.MIN_FREE_DISK_GB."""
    d = Path(front_dir)
    d.mkdir(parents=True, exist_ok=True)
    free, size = shutil.disk_usage(d).free, sum(c.nbytes for c in cache)
    if free - size < MIN_FREE_DISK_GB * 2**30:
        raise OSError(errno.ENOSPC, f"{free / 2**30:.1f} GB free, and a {size / 1e6:.0f} MB "
                      f"front would leave less than host_stats.MIN_FREE_DISK_GB "
                      f"({MIN_FREE_DISK_GB} GB)")
    key = front_key(front)
    _atomic_save(front_path(d, key), cache, {
        "front_ids": json.dumps(front),
    })
    return key


def _fronts_in(d):
    """(last use, ids, path) of the front files in `d`, the most recently
    used first; ids None for a file that does not load as a front
    (prune_fronts removes those). Only headers are read: mx.load is lazy
    (a 201 MB file loads in 0.06 ms with nothing read), the arrays stay on
    SSD."""
    out = []
    for p in Path(d).glob("*.safetensors"):
        if p.name.endswith(".tmp.safetensors"):
            continue                    # being written, or orphaned
        try:
            used = p.stat().st_mtime_ns
        except OSError:
            continue                    # evicted by another engine meanwhile
        try:
            ids = json.loads(load_prompt_cache(str(p), return_metadata=True)[1]["front_ids"])
        except Exception:
            ids = None
        out.append((used, ids, p))
    return sorted(out, key=lambda e: e[0], reverse=True)


def list_fronts(front_dir):
    """The ids of every front stored in `front_dir`, the most recently used
    first, whichever engine or process wrote it: the directory is the index,
    so what one engine builds or evicts, every engine sharing it sees."""
    return [ids for _, ids, _ in _fronts_in(front_dir) if ids is not None]


def load_front(key, front, front_dir):
    """Restore a front saved by save_front. None on any mismatch or corruption.

    front_ids is re-checked against the file's own metadata, not trusted
    from the key alone: a stale file sitting at a reused key (a
    truncated-hash collision) must come back as a miss and force a real
    prefill, never a wrong cache; a file cut short fails the load itself.
    A hit is touched: file mtimes are the last-use order prune_fronts
    evicts by, for every engine sharing the directory.

    The arrays stay lazy, as they always have (MemBudget evaluates them
    before it measures a pass). Another engine's eviction cannot take them
    away: MLX keeps the file open, and a lazily loaded array still reads
    after its file is unlinked (checked on MLX 0.32).
    """
    p = front_path(front_dir, key)
    try:
        cache, meta = load_prompt_cache(str(p), return_metadata=True)
    except FileNotFoundError:
        return None                     # evicted since it was listed
    except Exception as e:
        print(f"  ignoring front {p.name}: unreadable ({type(e).__name__})")
        return None
    if not cache or json.loads(meta.get("front_ids", "null")) != front:
        return None
    try:
        os.utime(p)
    except OSError:
        pass                            # evicted meanwhile; the arrays are ours
    return cache


def prune_fronts(front_dir, front, slots, tokens):
    """Hold the fronts to `slots` files and `tokens` ids, the most recently used
    first and `front` (just written) kept whatever its size, over the directory
    every engine with this fingerprint shares, then apply the same bound to
    every other fingerprint's directory under FRONTS, whole and least recently
    used first, and remove files that do not load as a front, orphaned temp
    files and dead engines' directories."""
    d = Path(front_dir)
    key = front_key(front)
    n, total = 1, len(front)
    for _, ids, p in _fronts_in(d):
        if p.stem == key:
            continue
        if ids is None or n >= slots or total + len(ids) > tokens:
            p.unlink(missing_ok=True)
        else:
            n, total = n + 1, total + len(ids)
    for p in d.glob("*.tmp.safetensors"):
        parts = p.name.rsplit("-", 2)
        if len(parts) == 3 and parts[1].isdigit() and _gone(int(parts[1])):
            p.unlink(missing_ok=True)
    _prune_dead_engines(d.parent)
    others = []
    for o in FRONTS.glob("*/fp-*"):
        if o == d or not o.is_dir():
            continue
        fronts = _fronts_in(o)
        try:
            used = fronts[0][0] if fronts else o.stat().st_mtime_ns
        except OSError:
            continue
        others.append((used, o, len(fronts), sum(len(ids or ()) for _, ids, _ in fronts)))
    n += sum(e[2] for e in others)
    total += sum(e[3] for e in others)
    for _, o, k, t in sorted(others, key=lambda e: e[0]):
        if k and n <= slots and total <= tokens:
            continue
        shutil.rmtree(o, ignore_errors=True)
        n, total = n - k, total - t


def prime(llm, messages, tag, model_path=None, max_tokens=64, extra=None):
    """Run a prefix once and snapshot the state it leaves behind.

    `messages` is an OpenAI-style list; the model answers it, and the answer is
    part of what gets frozen. That is deliberate -- a prefix ending mid-turn
    leaves the cache in a state the next turn cannot append to cleanly.
    """
    llm.reset()
    reply = llm.ask(messages=list(messages), max_tokens=max_tokens)
    return save(llm, tag, model_path, extra=extra), reply


def _self_check(model_path, tag="selfcheck"):
    """The round trip, across a process boundary, on the real model.

    Two things have to hold and neither is provable in one process: that the
    restored cache actually spares the prefill, and that the conversation
    continues correctly on top of it rather than being silently malformed.
    """
    import subprocess

    here = Path(__file__).parent
    fact = "My name is Anton and I am writing an MoE engine on MLX."
    write = f'''
from moe_stream import StreamingLLM
import pcache, json
llm = StreamingLLM(model={model_path!r}, max_tokens=32)
p, reply = pcache.prime(llm, [
    {{"role": "system", "content": "You are a local assistant. Answer briefly."}},
    {{"role": "user", "content": {fact!r}}}], {tag!r})
print("SAVED", p, round(p.stat().st_size / 1e6, 1), "MB")
print("PRIMED_TOKENS", len(llm._seen))
'''
    read = f'''
from moe_stream import StreamingLLM
import pcache
llm = StreamingLLM(model={model_path!r}, max_tokens=48)
meta = pcache.load(llm, {tag!r})
assert meta is not None, "cache did not load"
print("LOADED", len(llm._seen), "tokens,", len(llm.history), "turns")
out = llm.ask("What is my name and what am I doing?")
s = llm.stats()
print("REPLY", out.strip()[:160].replace(chr(10), " "))
print("STATS", {{k: s[k] for k in ("prompt_tokens", "cached_tokens", "ttft_s")}})
assert s["cached_tokens"] >= 20, ("the restored cache was not used: "
                                  f"cached_tokens={{s['cached_tokens']}}")
assert "Anton" in out, "the primed fact did not survive"
print("SELF-CHECK OK")
'''
    for name, code in (("prime", write), ("restore", read)):
        print(f"===== {name} =====", flush=True)
        r = subprocess.run([sys.executable, "-c", code],
                           cwd=here, capture_output=True, text=True)
        print(r.stdout[-1500:])
        if r.returncode:
            print(r.stderr[-1500:])
            return 1
    return 0


def demo():
    """What can be checked without loading 17 GB: the guards, not the state."""
    class _Stub:
        thinking = False
        model_path = "/models/demo-model"
        _cache, _seen, history, _ended_cleanly = [], [], [], False

    s = _Stub()
    assert load(s, "definitely-not-a-real-tag") is None, \
        "a missing snapshot must be None, not an exception"
    p = path_for(s.model_path, "x")
    assert p.name == "pcache-x.safetensors" and p.parent.name == "demo-model", p
    print("demo ok -- run --self-check for the real round trip")


# Qwen3.5/3.6 answer "who made you" with whatever the training data most often
# said, and a lot of that data was written by other models -- this build has
# introduced itself as trained by Google. Identity is also among the first things
# a 2.30-bit average loses. A system prefix fixes it permanently and, once
# snapshotted, costs no prefill at all.
DEFAULT_SYSTEM = (
    "You are a local assistant running on a Qwen model on the user's laptop. "
    "You were not made by Google or OpenAI. If asked who made you, say: a Qwen "
    "model from Alibaba, running locally. Answer briefly, in the user's language."
)


def _prime_cli(model_path, tag, system, greeting):
    from moe_stream import StreamingLLM

    llm = StreamingLLM(model=model_path, max_tokens=96)
    msgs = [{"role": "system", "content": system}]
    if greeting:
        msgs.append({"role": "user", "content": greeting})
    p, reply = prime(llm, msgs, tag)
    print(f"\nprimed '{tag}': {p}")
    print(f"  {p.stat().st_size / 1e6:.1f} MB, {len(llm._seen)} tokens, "
          f"restored with no prefill")
    print(f"  first reply: {reply.strip()[:200]}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    from defaults import MODEL
    ap.add_argument("--model", default=MODEL, required=MODEL is None, help="checkpoint directory")
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--prime", action="store_true",
                    help="run a system prefix once and snapshot the state")
    ap.add_argument("--tag", default="me")
    ap.add_argument("--system", default=DEFAULT_SYSTEM)
    ap.add_argument("--greeting", default="Introduce yourself in one sentence.",
                    help="a first turn, so the snapshot ends on a clean "
                         "assistant turn the next one can append to")
    a = ap.parse_args()
    if a.self_check:
        raise SystemExit(_self_check(a.model))
    if a.prime:
        raise SystemExit(_prime_cli(a.model, a.tag, a.system, a.greeting))
    raise SystemExit(demo())
