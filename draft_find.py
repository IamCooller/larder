#!/usr/bin/env python3
"""The draft `--draft auto` loads: the smallest model on disk (preferring the target's own quantization) whose tokenizer, vocab_size and size fit the dense target and the memory budget, so `chat.py --model <Qwen3-4B>` speculates with no flag, with the directory scan cached under each directory's mtime in ~/.cache/moe-ssd-offload/draft-scan.json (`python3 draft_find.py <target dir>` prints what auto would pick)."""

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

CACHE = Path.home() / ".cache" / "moe-ssd-offload" / "draft-scan.json"


def model_dirs():
    """Where models live: right under ~/.cache and every snapshot in the Hugging Face
    hub cache, ~/.cache first so a tie in bytes goes to the copy someone put there
    by hand."""
    from huggingface_hub.constants import HF_HUB_CACHE
    home = Path.home() / ".cache"
    own = sorted(d for d in home.iterdir() if d.is_dir()) if home.is_dir() else []
    return own + sorted(Path(HF_HUB_CACHE).glob("models--*/snapshots/*"))


def tokenizer_key(tok):
    """What a token id means, hashed: the model's type, vocabulary and merges
    and the added tokens, out of a parsed tokenizer.json. Merges are written
    as "a b" strings by older `tokenizers` and as [a, b] pairs by newer ones,
    so both become pairs first."""
    m = tok.get("model") or {}
    merges = [x.split(" ", 1) if isinstance(x, str) else list(x)
              for x in m.get("merges") or []]
    added = sorted((t["id"], t["content"]) for t in tok.get("added_tokens") or [])
    blob = json.dumps([m.get("type"), m.get("vocab"), merges, added], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _known(model_type):
    """Whether mlx_lm has a model class for `model_type`, found the way its
    own _get_classes finds it."""
    from mlx_lm.utils import MODEL_REMAPPING
    try:
        return importlib.util.find_spec(
            "mlx_lm.models." + MODEL_REMAPPING.get(model_type, model_type)) is not None
    except (ImportError, ValueError, TypeError):
        return False


def _info(d):
    """What decides whether `d` can draft for a target, or None when mlx_lm
    cannot load it as a model. A snapshot whose blobs never finished
    downloading fails the stat and is None too, and so does any file not
    shaped as expected: these are directories nobody named."""
    try:
        cfg = json.loads((d / "config.json").read_text())
        weights = sum(p.stat().st_size for p in d.glob("model*.safetensors"))
        if (not weights or (d / "experts_index.json").exists()
                or not _known(cfg.get("model_type"))):
            return None
        tok = tokenizer_key(json.loads((d / "tokenizer.json").read_text()))
        q = cfg.get("quantization")
        if q:
            q = dict(q)
            q.setdefault("mode", "affine")  # mlx_lm's default, unwritten by older configs
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None
    return {"vocab": cfg.get("vocab_size"), "bytes": weights,
            "quant": hashlib.sha256(json.dumps(q, sort_keys=True).encode()).hexdigest()[:16],
            "tok": tok}


def _scan(dirs, cache):
    """_info of every directory in `dirs`, the cached one wherever the
    directory's mtime has not moved. The cache is rewritten with exactly
    these directories, so one that is gone drops out of it."""
    try:
        old = json.loads(cache.read_text())
    except (OSError, ValueError):
        old = {}
    if not isinstance(old, dict):
        old = {}
    new = {}
    for d in dirs:
        try:
            mtime = d.stat().st_mtime_ns
        except OSError:
            continue
        hit = old.get(str(d))
        new[str(d)] = hit if hit and hit[0] == mtime else [mtime, _info(d)]
    if new != old:
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_name(f"{cache.name}.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(new))
            os.replace(tmp, cache)          # a reader never sees half a file
        except OSError:
            pass                            # a cache, not a result: rescan next time
    return {d: v[1] for d, v in new.items()}


def find(target, budget=None, dirs=None, cache=CACHE):
    """(path, weight bytes) of the draft for the model at `target`, or None.

    `budget`: bytes the target's and the draft's weights may take together.
    `dirs`: where to look, model_dirs() when None."""
    target = Path(target).expanduser().resolve()
    dirs = model_dirs() if dirs is None else list(dirs)
    info = _scan(dirs + [target], cache)
    t = info.pop(str(target), None)
    if t is None:
        return None
    fits = [(d, i) for d, i in info.items()
            if i and i["tok"] == t["tok"] and i["vocab"] == t["vocab"]
            and i["bytes"] < t["bytes"]
            and (budget is None or t["bytes"] + i["bytes"] <= budget)]
    if not fits:
        return None
    # min() keeps the first of equals, so a tie stays in model_dirs() order.
    d, i = min(fits, key=lambda c: (c[1]["quant"] != t["quant"], c[1]["bytes"]))
    return Path(d), i["bytes"]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    print(find(sys.argv[1]))
