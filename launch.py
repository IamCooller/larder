#!/usr/bin/env python3
"""Run a large language model on this computer with one command (./larder): it measures the machine,
picks and prepares a model, starts the server with settings for this machine and prints the chat
URL."""
import argparse
import collections
import json
import os
import queue
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

import engine_log   # noqa: F401  (sets MLX's CUDA environment before anything imports MLX)
import defaults
import engine_version
import library
import machine
import tuner
import tui

HERE = Path(__file__).resolve().parent
# The pack tools live in tools/; the launcher drives them.
TOOLS = HERE / "tools"
sys.path.insert(0, str(TOOLS))
GB = 1e9
# Free disk to leave after a download: room for the OS and its swap.
DISK_RESERVE = 10 * GB

# Models tested with this engine, where `gated` means the cache bias passed its quality gate on
# the model and `native` means the native reader was tested.
CURATED = [
    {"id": "qwen3.6-35b", "name": "Qwen3.6-35B-A3B", "about": "general chat, code, tool calls",
     "repo": "mlx-community/Qwen3.6-35B-A3B-4bit", "base": "Qwen/Qwen3.6-35B-A3B", "rev": "38740b847e4cb78f352aba30aa41c76e08e6eb46",
     "recipe": "arena", "gated": True, "native": True, "license": "apache-2.0",
     # On CUDA: the same trunk with a Q4_0 GGUF's routed experts and the model's own MTP head (for_backend).
     "cuda": {"id": "qwen3.6-35b-q40",
              "gguf": {"repo": "bartowski/Qwen_Qwen3.6-35B-A3B-GGUF", "file": "Qwen_Qwen3.6-35B-A3B-Q4_0.gguf",
                       "rev": "5c2410d71524f4f72b023ce8daf7a80528226d5f", "size": 20836243072,
                       "sha256": "52312daa5b2190c1f5723d33c3315c01c55af4206f6c6e6eb63f3d8dd52bb85e",
                       "experts_bytes": 18203279360, "types": "Q4_0, Q4_1"},
              "mtp": {"repo": "Qwen/Qwen3.6-35B-A3B", "rev": "345d467b117d1fd5eda9b879a5988de9ae5f47a7",
                      "bytes": 1689281536},
              "notes": {"published": "Q4_0 GGUF experts, cache bias 0.5 and the model's own MTP head, measured "
                                     "on an RTX 5060 Ti (2026-11-11): 166 tok/s against 41 for the 4-bit pack at "
                                     "bias 0; GSM8K 331 against 338 of 400 and HumanEval 149 against 148 of 164, "
                                     "not significantly different",
                        "mlx-community/Qwen3.6-35B-A3B-4bit": "the 4-bit pack the Q4_0 numbers are measured against"}}},
    {"id": "flash-next-exl3", "name": "Qwen3.8-Flash-Next (3-bit EXL3 experts)",
     "about": "the largest; 512 experts a layer",
     "repo": "mlx-community/Qwen3.8-Flash-Next-4bit", "base": "Qwen/Qwen3.8-Flash-Next", "rev": "07b5dc6c54600a359b87f1e53e7adf6351c72a2c",
     "experts_repo": "beamster/Qwen3.8-Flash-Next-Sushi-3bpw",
     "experts_rev": "76bd8f55df2bd4ab6297101c57bf86d450ac0aa3",
     "recipe": "store", "gated": True, "native": True, "hf_layout": True,
     "license": "qwen-community-1.0",
     "notes": {"published": "3-bit EXL3 experts, measured: perplexity +2.45% against 4-bit; GSM8K and "
                            "HumanEval not significantly different",
               "mlx-community/Qwen3.8-Flash-Next-4bit": "the 4-bit reference the 3-bit numbers are measured against"},
     "not_made": ["3-bit plain rounding of the 4-bit experts: perplexity +4.13% (measured); the EXL3 pack "
                  "above is smaller and closer to 4-bit",
                  "3-bit GPTQ: perplexity +1.39% (measured); not made here: it needs a calibration capture "
                  "that does not run unattended within 16 GB"]},
    {"id": "qwen1.5-moe", "name": "Qwen1.5-MoE-A2.7B-Chat", "about": "small MoE, a quick first try",
     "repo": "mlx-community/Qwen1.5-MoE-A2.7B-Chat-4bit", "rev": "cf116003d120c4216cf008eba169f98b95bdf3ee",
     "recipe": "store", "license": "tongyi-qianwen",
     "notes": {"q3": "measured on this model: perplexity +7.6%, 6 of 6 instruction prompts still right"}},
    {"id": "qwen3-8b", "name": "Qwen3-8B", "about": "dense, fits in memory",
     "repo": "mlx-community/Qwen3-8B-4bit", "base": "Qwen/Qwen3-8B", "rev": "545dc4251c05440727734bcd94334791f6ab0192",
     "license": "apache-2.0"},
    {"id": "qwen3-4b", "name": "Qwen3-4B", "about": "dense, small and fast",
     "repo": "mlx-community/Qwen3-4B-4bit", "base": "Qwen/Qwen3-4B", "rev": "4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25",
     "license": "apache-2.0"},
]


def for_backend(c, backend):
    """Curated entry c as it installs on this backend: with its "cuda" overrides on CUDA."""
    o = c.get("cuda")
    return {**c, **o} if o and backend == "cuda" else c


class Unsupported(Exception):
    """A model this engine cannot run, with the reason in words."""


# ---------------------------------------------------------------- what a model is

def _cfg_get(cfg, *names):
    tc = cfg.get("text_config") or {}
    for n in names:
        for c in (cfg, tc):
            if c.get(n) is not None:
                return c[n]
    return None


def inspect_model(cfg, names=None):
    """The facts machine.settings and machine.estimate need, read from config.json (and the
    checkpoint's tensor names when known) by building the model lazily with mlx-lm's own classes, or
    Unsupported with the reason."""
    mt = cfg.get("model_type")
    if not mt:
        raise Unsupported("config.json names no model_type")
    q = cfg.get("quantization")
    qc = cfg.get("quantization_config") or (cfg.get("text_config") or {}).get("quantization_config")
    if qc and not q:
        raise Unsupported(f"quantized for PyTorch ({qc.get('quant_method', '?')}); this engine runs "
                          f"MLX checkpoints: look for an mlx-community version")
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from mlx_lm.utils import _get_classes
    import stream_fast
    try:
        model_cls, args_cls = _get_classes(cfg)
    except ValueError:
        raise Unsupported(f"model type '{mt}' is not in the installed mlx-lm")
    try:
        model = model_cls(args_cls.from_dict(cfg))
    except Exception as e:
        raise Unsupported(f"mlx-lm cannot build this {mt} config ({type(e).__name__}: {e})")
    model.set_dtype(mx.bfloat16)               # the checkpoint's dtype; lazy, nothing computed
    if names is not None:                      # checkpoint keys as module paths (sanitize's prefixes)
        from stream_pack import hf_key, module_key
        names = {module_key(n) for n in names} | {hf_key(n) for n in names} | set(names)
    if q:
        def pred(p, m):
            if p in q:
                return q[p]
            if not hasattr(m, "to_quantized") or m.weight.shape[-1] % q["group_size"]:
                return False
            return names is None or f"{p}.scales" in names or ".switch_mlp." in p
        nn.quantize(model, group_size=q["group_size"], bits=q["bits"], mode=q.get("mode", "affine"),
                    class_predicate=pred)
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear
    moe, switch = 0, None
    for i, layer in enumerate(model.layers):
        sw, _ = stream_fast.find_switch_mlp(layer, i)
        if sw is not None:
            moe += 1
            switch = switch or sw
    params = dict(tree_flatten(model.parameters()))
    nbytes = lambda keys: sum(params[k].nbytes for k in keys)
    total = nbytes(params)
    tied = bool(_cfg_get(cfg, "tie_word_embeddings"))
    embed = nbytes([k for k in params if "embed_tokens." in k])
    ple = nbytes([k for k in params if ".ngram_embedding." in k])
    streamed = switch is not None and isinstance(getattr(switch, "gate_proj", None), QuantizedSwitchLinear)
    f = {"model_type": mt, "layers": len(model.layers), "total_bytes": total, "embed_bytes": embed}
    f.update(bits=q["bits"] if q else 16, group=q["group_size"] if q else 0)
    if not streamed:
        f.update(kind="resident", dense_bytes=total, moe_layers=0, experts=0, top_k=0, expert_bytes=0,
                 read_bytes=total - (0 if tied else embed), hidden=_cfg_get(cfg, "hidden_size"))
        return f
    n_exp = switch.gate_proj.weight.shape[0]
    expert = sum(v.nbytes for _, v in tree_flatten(switch.parameters())) // n_exp
    dense = total - expert * n_exp * moe - ple - (0 if tied else embed)   # the embed table is read by row
    top_k = _cfg_get(cfg, "num_experts_per_tok", "moe_k", "top_k", "num_selected_experts") or 1
    f.update(kind="moe", dense_bytes=dense, read_bytes=dense, moe_layers=moe, experts=n_exp,
             top_k=int(top_k), expert_bytes=expert, ple_bytes=ple,
             hidden=_cfg_get(cfg, "hidden_size"),
             d_ff=switch.gate_proj.weight.shape[1],
             bits=switch.gate_proj.bits, group=switch.gate_proj.group_size)
    return f


def exl3_expert_bytes(f, k_bits):
    """One expert's EXL3 record in bytes: a k-bit trellis for its three projections plus an fp16
    input and output sign vector each."""
    h, d = f["hidden"], f["d_ff"]
    return int(3 * h * d * k_bits / 8 + 6 * (h + d))


# ---------------------------------------------------------------- where models come from

def _hf_json(repo, name, rev):
    from huggingface_hub import hf_hub_download
    return json.loads(Path(hf_hub_download(repo, name, revision=rev)).read_text())


def hf_entry(e):
    """A catalog entry (curated or found) filled from Hugging Face: config, sizes, facts."""
    from huggingface_hub import HfApi
    info = HfApi().model_info(e["repo"], revision=e.get("rev"), files_metadata=True)
    files = {s.rfilename: s.size or 0 for s in info.siblings}
    if any(n.endswith(".gguf") for n in files) and not any(n.endswith(".safetensors") for n in files):
        raise Unsupported("GGUF files only; this engine runs MLX safetensors (look for an mlx-community version)")
    if not any(n.endswith(".safetensors") for n in files):
        raise Unsupported("no safetensors weights in the repo")
    cfg = _hf_json(e["repo"], "config.json", e.get("rev"))
    names = None
    if "model.safetensors.index.json" in files:
        names = set(_hf_json(e["repo"], "model.safetensors.index.json", e.get("rev"))["weight_map"])
    f = inspect_model(cfg, names)
    # The packer holds the shard it reads and at most one earlier one a triple still needs.
    shards = sorted(s for n, s in files.items() if n.endswith(".safetensors"))
    two = sum(shards[-2:])
    size = sum(files.values())
    e = {**e, "facts": f, "rev": e.get("rev") or info.sha}
    e.setdefault("hf_layout", f["model_type"] == "qwen4_exp")     # its model file folds norms in that layout
    if f["kind"] == "moe":
        e.setdefault("recipe", "store")
        if e.get("experts_repo"):               # EXL3: the base without its experts, then the experts
            xinfo = HfApi().model_info(e["experts_repo"], revision=e["experts_rev"], files_metadata=True)
            xfiles = {s.rfilename: s.size or 0 for s in xinfo.siblings
                      if re.match(r"model-exl3-L\d+-\w+\.safetensors$", s.rfilename)}
            xcfg = _hf_json(e["experts_repo"], "config.json", e["experts_rev"])
            eq = xcfg["expert_quant"]
            base = size - f["expert_bytes"] * f["experts"] * f["moe_layers"]
            f["expert_bytes"] = exl3_expert_bytes(f, eq["k"])
            f.update(bits=eq["k"], format="exl3")
            e.update(exl3_files=sorted(xfiles), expert_quant=eq,
                     download=base + sum(xfiles.values()),
                     disk=base + f["expert_bytes"] * f["experts"] * f["moe_layers"],
                     transient=two)
        elif e.get("gguf"):                     # GGUF experts beside the base's trunk, plus the model's MTP head
            g, h = e["gguf"], e["mtp"]
            n = f["experts"] * f["moe_layers"]
            base = size - f["expert_bytes"] * n
            f["expert_bytes"] = g["experts_bytes"] // n
            # Bits a weight with the scales (4.52 for Q4_0 / Q4_1); the pack's expert_quant carries the same number.
            f.update(bits=round(8 * g["experts_bytes"] / (3 * f["hidden"] * f["d_ff"] * n), 2), format="gguf")
            head = int(h["bytes"] * machine.bytes_per_weight(4, 64) / 2)
            e.update(download=base + g["size"] + h["bytes"], disk=base + g["experts_bytes"] + head,
                     transient=max(two, g["size"], h["bytes"]))
        else:
            e.update(download=size, disk=size, transient=two)
    else:
        e.update(download=size, disk=size, transient=0)
    return e


def known_type(mt):
    """Whether the installed mlx-lm has a model file for this config model_type."""
    import importlib.util
    from mlx_lm.utils import MODEL_REMAPPING
    mt = MODEL_REMAPPING.get(mt, mt)
    return bool(mt) and re.fullmatch(r"\w+", mt) is not None and \
        importlib.util.find_spec(f"mlx_lm.models.{mt}") is not None


def hf_search(query, limit=8):
    """Hugging Face models matching `query` that this engine can run: MLX
    safetensors of a model type the installed mlx-lm knows, not gated."""
    from huggingface_hub import HfApi
    found, skipped = [], []
    for m in HfApi().list_models(search=query, filter="mlx", sort="downloads", limit=60,
                                 expand=["config", "gated", "sha", "downloads"]):
        mt = (m.config or {}).get("model_type")
        if m.gated or not known_type(mt):
            skipped.append(m.id)
            continue
        try:
            found.append(hf_entry({"id": m.id, "name": m.id, "repo": m.id, "rev": m.sha,
                                   "about": f"{m.downloads or 0:,} downloads"}))
        except Unsupported as err:
            skipped.append(f"{m.id} ({err})")
        except Exception:
            skipped.append(m.id)
        if len(found) >= limit:
            break
    return found, skipped


HF_LINK = re.compile(r"^(?:https?://)?(?:www\.)?(?:huggingface\.co|hf\.co)/([\w.-]+)/([\w.-]+)(?:[/?#].*)?$")
HF_PAGES = ("datasets", "spaces", "docs", "blog", "papers", "collections", "api", "settings", "models")


def parse_repo(text):
    """owner/name from an owner/name or a Hugging Face link (a /tree/..., /blob/... or /resolve/... page,
    a query string or a fragment cut off), or None for anything else."""
    t = (text or "").strip()
    m = HF_LINK.match(t)
    if m and m[1] not in HF_PAGES:
        return f"{m[1]}/{m[2]}"
    return t if re.fullmatch(r"[\w.-]+/[\w.-]+", t) else None


def search_models(query, limit=8):
    """(entries, skipped) for the one repo an owner/name or a link names, or for a Hugging Face search, with Unsupported giving the reason for a repo it cannot run."""
    repo = parse_repo(query)
    if repo:
        return [hf_entry({"id": repo.replace("/", "--"), "name": repo, "repo": repo})], []
    return hf_search(query, limit)


def local_entry(path):
    """A model folder on this computer, checked the way a download is."""
    p = Path(path).expanduser().resolve()
    if p.is_file() and p.suffix == ".gguf" or any(p.glob("*.gguf")):
        raise Unsupported("GGUF: this engine runs MLX safetensors checkpoints (convert with "
                          "`mlx_lm.convert` or pick an mlx-community version)")
    if not (p / "config.json").exists():
        raise Unsupported(f"{p} has no config.json: not a model folder")
    if not any(p.glob("*.safetensors")):
        raise Unsupported("no .safetensors weights (PyTorch .bin files are not read)")
    import stream_fast
    cfg = json.loads((p / "config.json").read_text())
    f = inspect_model(cfg, set(stream_fast.build_shard_map(p)) | _packed_names(p))
    size = sum(x.stat().st_size for x in p.iterdir() if x.is_file())
    if f["kind"] == "moe" and (p / "experts_index.json").exists():
        idx = json.loads((p / "experts_index.json").read_text())
        stride = sum(v["stride"] for v in idx.values()) / max(1, f["moe_layers"])
        f["expert_bytes"] = int(stride)          # what this pack actually holds
    e = {"id": p.name, "name": p.name, "about": str(p), "path": str(p), "facts": f,
         "recipe": "store", "download": 0, "disk": 0, "transient": 0,
         "size": sum(x.stat().st_size for x in p.iterdir() if x.is_file() or x.is_symlink())}
    if f["kind"] == "moe" and not (p / "experts.bin").exists():
        e.update(needs_pack=True, disk=size, transient=0)      # a packed copy next to it
    return e


def _packed_names(p):
    """Module keys of experts already moved into experts.bin, as weight names."""
    idx = p / "experts_index.json"
    if not idx.exists():
        return set()
    return {f"{k}.{part}" for k in json.loads(idx.read_text()) for part in ("weight", "scales")}


# ---------------------------------------------------------------- variants: other bit widths

# The share of the drive's measured rate, reads and writes together, that quantpack.requant_pack runs at.
MAKE_SHARE = 0.39
BIT_NOTES = {16: "the original weights", 8: "close to the original", 6: "a very small loss",
             5: "a small loss", 4: "usually a small loss", 3: "a noticeable loss",
             2: "a large loss: often breaks instruction following"}
SIBLING = re.compile(r"^(?P<stem>.+)-(?:(?P<bits>[2-8])bit|bf16)$")
_SIBS = {}                                     # repo -> its siblings, once a run


def siblings(e, limit=4):
    """The same model at other bit widths from the same publisher on Hugging Face
    (plain `-Nbit` and `-bf16` repos), as catalog entries."""
    from huggingface_hub import HfApi
    repo = e.get("repo")
    m = repo and SIBLING.match(repo.split("/")[1])
    if not m:
        return []
    if repo in _SIBS:
        return _SIBS[repo]
    owner, out = repo.split("/")[0], []
    _SIBS[repo] = out
    if e.get("experts_repo") or e.get("gguf"):     # EXL3 or GGUF experts: the 4-bit base is a variant too
        out.append(hf_entry({k: e[k] for k in ("repo", "rev", "recipe", "gated", "native") if k in e}
                            | {"id": repo.replace("/", "--"), "name": repo.split("/")[1]}))
    for r in HfApi().list_models(author=owner, search=m["stem"], limit=50, expand=["sha", "gated"]):
        mm = SIBLING.match(r.id.split("/")[1])
        if r.id == repo or r.gated or not mm or mm["stem"] != m["stem"] or len(out) >= limit:
            continue
        try:
            out.append(hf_entry({"id": r.id.replace("/", "--"), "name": r.id.split("/")[1], "repo": r.id,
                                 "rev": r.sha, "recipe": e.get("recipe"), "native": e.get("native")}))
        except Exception:
            pass                               # a sibling that does not inspect is just not listed
    out.sort(key=lambda x: -x["facts"]["bits"])
    return out


def made_entry(e, bits, models):
    """Entry e rounded here to `bits` (plain affine, the published group size): the
    routed experts of a streamed MoE, every quantized tensor of a resident model."""
    f = e["facts"]
    bpw = machine.bytes_per_weight
    ratio = bpw(bits, f["group"]) / bpw(f["bits"], f["group"])
    f2 = dict(f, bits=bits)
    if f["kind"] == "moe":
        moved = f["expert_bytes"] * f["experts"] * f["moe_layers"]
        f2["expert_bytes"] = int(f["expert_bytes"] * ratio)
    else:
        moved = f["total_bytes"]
        f2.update(read_bytes=int(f["read_bytes"] * ratio), dense_bytes=int(f["dense_bytes"] * ratio),
                  total_bytes=int(f["total_bytes"] * ratio))
    src_ready = is_ready(model_dir(e, models)) or (e.get("path") and not e.get("needs_pack"))
    keep = {k: v for k, v in e.items() if k not in ("facts", "path", "needs_pack", "notes", "not_made", "size")}
    return {**keep, "id": f"{e['id']}-q{bits}", "name": f"{e['name']} ({bits}-bit, made here)",
            "facts": f2, "source": e, "make_bits": bits, "gated": False,
            "size": size_of(e) - moved + int(moved * ratio),
            "download": 0 if src_ready else e["download"],
            "disk": int(moved * ratio) + (0 if src_ready else e["disk"]),
            "transient": 0 if src_ready else e["transient"], "make_bytes": int(moved * (1 + ratio))}


def variants(e, models, sibs=None):
    """The ways to run e's model as {"key", "label", "entry", "note"}: as published first (the
    default), the same model at other bits on Hugging Face, and rounded here to fewer bits."""
    copies = library.scan(models)               # folders found elsewhere on this computer
    e = library.adopt(e, models, copies)
    f, notes = e["facts"], e.get("notes", {})
    bits_word = "EXL3 3-bit experts" if f.get("format") == "exl3" else \
        f"GGUF {e['gguf']['types']} experts" if f.get("format") == "gguf" else \
        f"{f['bits']}-bit" if f["bits"] < 16 else "16-bit"
    out = [{"key": "published", "label": f"as published, {bits_word}", "entry": e,
            "note": notes.get("published", "the model as published")}]
    for s in sibs if sibs is not None else siblings(e):
        b = s["facts"]["bits"]
        out.append({"key": s["repo"], "label": f"{b}-bit" if b < 16 else "16-bit",
                    "entry": library.adopt(s, models, copies),
                    "note": notes.get(s["repo"], BIT_NOTES.get(b, "")) + f" | {s['repo']}"})
    if f.get("format") not in ("exl3", "gguf") and f["bits"] < 16 and f.get("group"):
        for b in (4, 3):
            if b < f["bits"]:
                note = notes.get(f"q{b}", BIT_NOTES[b] + f"; rounded from the published {f['bits']}-bit, "
                                 "a little worse than one made from the original weights")
                out.append({"key": f"q{b}", "label": f"{b}-bit, made here", "entry": made_entry(e, b, models),
                            "note": note})
    return out


# ---------------------------------------------------------------- preparing

def model_dir(e, models):
    return Path(e["path"]) if e.get("path") and not e.get("needs_pack") else Path(models) / e["id"]


def is_ready(d):
    return (d / "launcher.json").exists()


def fits(e, models):
    """(bool, words): whether the download fits the disk with room to spare."""
    d = model_dir(e, models)
    if is_ready(d) or (e.get("path") and not e.get("needs_pack")):
        return True, "ready on disk"
    have = shutil.disk_usage(_existing(d)).free
    need = e["disk"] + e["transient"] + DISK_RESERVE
    if need > have:
        return False, f"needs {need / GB:.0f} GB free on disk, {have / GB:.1f} GB free"
    return True, ""


def _existing(p):
    p = Path(p)
    while not p.exists():
        p = p.parent
    return p


def prepare(e, models, py=sys.executable):
    """Download and pack the model (or round a variant made here from its prepared source)
    into its folder and return it, stopping on any error."""
    d = model_dir(e, models)
    if is_ready(d) or (e.get("path") and not e.get("needs_pack")):
        return d
    if e.get("make_bits"):
        src = prepare(e["source"], models, py)
        if not (d / ".made").exists():
            _run([py, str(TOOLS / "quantpack.py"), "--from-pack", str(src), "--out", str(d),
                  "--bits", str(e["make_bits"])])
            (d / ".made").write_text(json.dumps({k: v for k, v in e.items() if k != "source"}))
        return d
    d.mkdir(parents=True, exist_ok=True)
    f = e["facts"]
    if e.get("needs_pack"):
        _run([py, str(TOOLS / "stream_pack.py"), "--from-dir", e["path"], "--out", str(d)])
    elif e.get("gguf"):
        _pack_gguf(e, d, py)
    elif f["kind"] == "moe":
        cmd = [py, str(TOOLS / "stream_pack.py"), "--repo", e["repo"], "--revision", e["rev"], "--out", str(d)]
        cmd += ["--hf-layout"] if e.get("hf_layout") else []
        cmd += ["--skip-experts"] if e.get("experts_repo") else []
        if not (d / ".exl3_state.json").exists():      # EXL3 shards in: the trunk is done, a rerun truncates experts.bin
            _run(cmd)
        if e.get("experts_repo"):
            _pack_exl3(e, d)
    else:
        from huggingface_hub import snapshot_download
        snapshot_download(e["repo"], revision=e["rev"], local_dir=str(d),
                          allow_patterns=["*.safetensors", "*.json", "*.jinja", "*.txt", "*.model", "*.tiktoken"])
    mark_ready(d, e)
    try:
        tuner.record_baseline(d, "written")      # a fresh pack's own rate is the reference for later age checks
    except Exception:
        pass
    return d


def mark_ready(d, e):
    (d / "launcher.json").write_text(json.dumps({k: v for k, v in e.items() if k not in ("path", "source")},
                                                indent=1))


def validate(port, max_tokens=24, warm=False, api_key=None):
    """One short reply (text, "") or (None, the reason it failed): the check of a freshly made variant and,
    marked `warm`, the warm-up every start runs before it says ready, which the server's counters leave out."""
    body = json.dumps({"messages": [{"role": "user", "content": "Say hello in one short sentence."}],
                       "max_tokens": max_tokens}).encode()
    headers = {"Content-Type": "application/json", **({"X-Warmup": "1"} if warm else {}),
               **({"Authorization": "Bearer " + api_key} if api_key else {})}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body, headers)
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            text = json.load(r)["choices"][0]["message"]["content"]
    except (OSError, ValueError, KeyError, IndexError) as err:
        return None, f"no reply ({type(err).__name__})"
    return (text, "") if text and text.strip() else (None, "an empty reply")


def _run(cmd):
    print("  $ " + " ".join(cmd[1:]), flush=True)
    if subprocess.run(cmd, cwd=HERE).returncode != 0:
        raise SystemExit("Preparing the model failed (see above). Run the same command again to resume.")


def _pack_exl3(e, d):
    """Download the EXL3 experts one layer shard at a time, append each to experts.bin with
    repack.pack_shard and delete the shard, resuming from the shards already done."""
    import repack
    from huggingface_hub import hf_hub_download
    state_p = d / ".exl3_state.json"
    st = json.loads(state_p.read_text()) if state_p.exists() else {"done": [], "written": 0, "index": {}}
    tmp = d / "_download_exl3"
    with open(d / "experts.bin", "r+b" if st["done"] else "wb") as dst:
        dst.truncate(st["written"])
        dst.seek(st["written"])
        for i, name in enumerate(e["exl3_files"]):
            if name in st["done"]:
                continue
            p = Path(hf_hub_download(e["experts_repo"], name, revision=e["experts_rev"], local_dir=str(tmp)))
            st["written"] = repack.pack_shard(p, dst, st["written"], st["index"])
            dst.flush()
            os.fsync(dst.fileno())
            p.unlink()
            st["done"].append(name)
            state_p.write_text(json.dumps(st))
            print(f"  experts [{i + 1}/{len(e['exl3_files'])}] {st['written'] / GB:.1f} GB", flush=True)
    (d / "experts_index.json").write_text(json.dumps(st["index"], indent=1))
    cfg = json.loads((d / "config.json").read_text())
    cfg["expert_quant"] = e["expert_quant"]
    (d / "config.json").write_text(json.dumps(cfg, indent=2))
    state_p.unlink()
    shutil.rmtree(tmp, ignore_errors=True)


def _pack_gguf(e, d, py=sys.executable):
    """The base's trunk, the GGUF's routed experts and the model's own 4-bit MTP head into d, each step
    skipped once its last file is in place, so a rerun resumes (the GGUF download resumes too)."""
    from huggingface_hub import hf_hub_download
    g, h = e["gguf"], e["mtp"]
    if not (d / "config.json").exists():       # the trunk is made aside, so it never touches experts.bin
        t = d / "_trunk"
        _run([py, str(TOOLS / "stream_pack.py"), "--repo", e["repo"], "--revision", e["rev"], "--out", str(t),
              "--skip-experts"])
        cfg = json.loads((t / "config.json").read_text())
        cfg["expert_quant"] = {"format": "gguf", "bits": e["facts"]["bits"],
                               "source": f"{g['repo']}/{g['file']}@{g['rev']}"}
        (t / "config.json").unlink()
        for p in t.iterdir():
            if p.name not in ("experts.bin", "experts_index.json"):
                p.replace(d / p.name)
        (d / "config.json").write_text(json.dumps(cfg, indent=2))
        shutil.rmtree(t)
    if not (d / "experts_index.json").exists():    # gguf_repack.py writes the index last
        tmp = d / "_download_gguf"
        src = Path(hf_hub_download(g["repo"], g["file"], revision=g["rev"], local_dir=str(tmp)))
        _run([py, str(TOOLS / "gguf_repack.py"), str(src), "--out", str(d / "experts.bin"),
              "--layers", str(e["facts"]["layers"])])  # an MTP block past the trunk stays out
        shutil.rmtree(tmp)
    head = d / MTP_PACK
    if not head.with_name(head.stem + "-quant.json").exists():   # mtp_pack.py writes it last
        _fetch_mtp(h["repo"], h["rev"], d / "_mtp" / "mtp.safetensors")
        _run([py, str(TOOLS / "mtp_pack.py"), "--src", str(d / "_mtp"), "--out", str(head)])
        shutil.rmtree(d / "_mtp")


def _fetch_mtp(repo, rev, dst, fs=None):
    """The checkpoint's mtp.* tensors into one safetensors file at dst, range-read from its shards
    instead of downloading the whole model."""
    from huggingface_hub import HfFileSystem
    fs = fs or HfFileSystem()
    wm = json.loads(fs.read_text(f"{repo}/model.safetensors.index.json", revision=rev))["weight_map"]
    src = {}
    for sh in sorted({v for k, v in wm.items() if k.startswith("mtp.")}):
        with fs.open(f"{repo}/{sh}", "rb", revision=rev) as f:
            n = int.from_bytes(f.read(8), "little")
            src.update({k: (sh, 8 + n, v) for k, v in json.loads(f.read(n)).items() if k.startswith("mtp.")})
    hdr, off = {}, 0
    for k, (_, _, v) in sorted(src.items()):
        lo, hi = v["data_offsets"]
        hdr[k] = {"dtype": v["dtype"], "shape": v["shape"], "data_offsets": [off, off + hi - lo]}
        off += hi - lo
    blob = json.dumps(hdr).encode()
    blob += b" " * (-len(blob) % 8)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "wb") as out:
        out.write(len(blob).to_bytes(8, "little") + blob)
        for k, (sh, data0, v) in sorted(src.items()):
            lo, hi = v["data_offsets"]
            with fs.open(f"{repo}/{sh}", "rb", revision=rev) as f:
                f.seek(data0 + lo)
                while lo < hi:
                    buf = f.read(min(64 << 20, hi - lo))
                    if not buf:
                        raise OSError(f"{sh}: the download stopped inside {k}")
                    out.write(buf)
                    lo += len(buf)


def native_ready(py=sys.executable):
    """Whether the native reader imports in this Python, building it once if not
    (Apple: native/build.sh, needs the Xcode command line tools; CUDA builds it
    at first use)."""
    test = [py, "-c", "import moe_native"]
    if subprocess.run(test, capture_output=True).returncode == 0:
        return True
    if sys.platform != "darwin" or not shutil.which("xcrun"):
        return False
    print("Building the native expert reader (one time, about a minute)...", flush=True)
    log = machine.CACHE / "native-build.log"
    machine.CACHE.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as f:
        subprocess.run(["bash", str(HERE / "native" / "build.sh"), py], stdout=f, stderr=subprocess.STDOUT)
    ok = subprocess.run(test, capture_output=True).returncode == 0
    print("  built." if ok else f"  not built (log: {log}); running without it.", flush=True)
    return ok


# ---------------------------------------------------------------- the menu

def judge(e, prof, models, disk):
    """Settings, the estimate and whether entry e can run here; "short" is a disk too full for a
    machine that could run it, which is not a model it cannot run."""
    f = e["facts"]
    s = machine.settings(prof, {**f, **e})
    est = machine.estimate(prof, f, s, disk) if s["ok"] else None
    ok, why = fits(e, models)
    short = not ok
    if not s["ok"]:
        ok, why, short = False, s["why"][-1], False
    return {"e": e, "s": s, "est": est, "ok": ok, "why": why, "short": short,
            "mtp": mtp_for(model_dir(e, models), e)}


def table_of(entries, prof, models, disk, sibs=True):
    """[(model entry, [its variants, judged])] for the menu."""
    return [(e, [{**judge(v["entry"], prof, models, disk), **v}
                 for v in variants(e, models, None if sibs else [])]) for e in entries]


def describe(f):
    if f["kind"] == "moe":
        return f"MoE, {f['experts']} experts x {f['moe_layers']} layers"
    return "dense"


def size_of(e):
    """Bytes the variant's own model takes on disk."""
    return e.get("size") or e["disk"]


def status(r, models, disk):
    e = r["e"]
    if r.get("short"):
        return r["why"]
    if not r["ok"]:
        return "can't run: " + r["why"]
    if is_ready(model_dir(e, models)):
        return "ready"
    if e.get("path"):
        where = f"on disk at {library.short_path(e['path'])}"
        return where + "; packs first, no download" if e.get("needs_pack") else where
    if e.get("make_bits") and (model_dir(e, models) / ".made").exists():
        return "made; checked with one reply at its first start"
    if e.get("make_bits"):
        rate = (disk["gbps"] if disk else 1.0) * MAKE_SHARE * GB
        sec = e["make_bytes"] / rate
        t = f"~{max(5, round(sec, -1)):.0f} s to make" if sec < 90 else f"~{round(sec / 60)} min to make"
        return t + (f" after a {e['download'] / GB:.1f} GB download" if e["download"] else "")
    return f"download {e['download'] / GB:.1f} GB"


def disk_path(e, models):
    """Where e's files are on this computer: its folder in the launcher's models folder once
    prepared or made, or the folder it was found in; "" when it is not on disk."""
    d = model_dir(e, models)
    if is_ready(d) or ((d / ".made").exists() and not (d / ".requant_state.json").exists()):
        return str(d)
    return e.get("path") or ""


def machine_json(prof, disk, models):
    """This computer, as another computer's menu needs it."""
    here = machine.volume_of(models)
    gpu = f"{prof.get('gpu')}" + (f", {prof['gpu_mem'] / 2**30:.0f} GB VRAM" if prof["backend"] == "cuda" else "")
    cond = (prof.get("now") or {}).get("conditions") or {}
    return {"chip": prof["chip"] or prof["arch"], "ram_gb": prof["ram"] / 2**30, "gpu": gpu, "os": prof["os"],
            "models_dir": str(models), "free_gb": shutil.disk_usage(_existing(models)).free / GB,
            "ssd_gbps": disk["gbps"] if disk else None,
            "conditions": cond, "conditions_text": machine.conditions_text(cond),
            "others": [(d["mount"], d["free"] / GB) for d in prof["now"]["disks"] if d["dev"] != here]}


def variant_json(r, models, disk):
    """One variant as the JSON listing carries it."""
    e = r["e"]
    path = disk_path(e, models)
    est = r["est"] or {}
    return {"key": r["key"], "label": r["label"], "size": size_of(e), "tps": est.get("tps"),
            "tps_lo": est.get("lo_tps"), "tps_hi": est.get("hi_tps"),
            # The estimate was corrected by this computer's own measured runs of the same recipe.
            "tps_adjusted": bool(est.get("runs")), "need_gb": (r["s"] or {}).get("need_gb"),
            "mtp": bool((r.get("mtp") or {}).get("path")),
            "ok": r["ok"], "why": r["why"], "disk_short": bool(r.get("short")),
            "status": status(r, models, disk), "note": r["note"],
            "ready": is_ready(model_dir(e, models)) or bool(e.get("path") and not e.get("needs_pack")),
            "make_bits": e.get("make_bits", 0), "download": e.get("download", 0), "disk": e.get("disk", 0),
            "expected": e.get("disk", 0), "entry_id": e["id"], "path": path, "own": library.inside(path, models),
            "measured_here": library.lookup(e["facts"])}


def model_json(e, vs, models, disk, tested=False):
    """One model with its judged variants, as the JSON listing carries it."""
    return {"id": e["id"], "name": e["name"], "repo": e.get("repo", ""), "kind": describe(e["facts"]),
            "about": e.get("about", ""), "license": e.get("license", ""), "tested": tested, "unavailable": "",
            "not_made": e.get("not_made", []),
            "variants": [variant_json(r, models, disk) for r in vs]}


def listing(prof, models, disk, fetch=None, say=None):
    """The model list as one JSON object, made on the computer that will run the models."""
    fetch = fetch or hf_entry
    say = say or (lambda *a: None)
    out = []
    for c in CURATED:
        c = for_backend(c, prof["backend"])
        say(f"checking {c['name']}...")
        d = model_dir(c, models)
        try:
            e = {**c, **(load_json(d / "launcher.json") or {})} if is_ready(d) else None
            e = e if e and e.get("facts") else fetch(c)
        except Unsupported as err:
            why = str(err)
            if "not in the installed mlx-lm" in why:
                why += " (requirements.txt installs the mlx-lm that has it: .venv/bin/pip install -r requirements.txt)"
            out.append({"id": c["id"], "name": c["name"], "about": c.get("about", ""), "license": c.get("license", ""),
                        "tested": True, "unavailable": why, "not_made": [], "variants": []})
            continue
        except Exception as err:
            out.append({"id": c["id"], "name": c["name"], "about": c.get("about", ""), "license": c.get("license", ""),
                        "tested": True, "unavailable": f"could not reach Hugging Face ({type(err).__name__})",
                        "not_made": [], "variants": []})
            continue
        say(f"checking {c['name']}'s variants...")
        try:
            vs = table_of([e], prof, models, disk)[0][1]
        except Exception:                  # Hugging Face unreachable: the other bit widths are not offered
            vs = [{**judge(v["entry"], prof, models, disk), **v} for v in variants(e, models, [])]
        out.append(model_json(e, vs, models, disk, tested=True))
    return {"machine": machine_json(prof, disk, models), "engine": engine_version.version(), "models": out}


def show(table, models, disk, start=1):
    tested = {c["id"] for c in CURATED} | {for_backend(c, "cuda")["id"] for c in CURATED}
    for i, (e, vs) in enumerate(table, start):
        print(f" {i:>2}  {e['name']}{' (tested)' if e['id'] in tested else ''} | {describe(e['facts'])} | "
              f"{e.get('about', '')}" + (f" | license {e['license']}" if e.get("license") else ""))
        for j, r in enumerate(vs):
            got = library.lookup(r["e"]["facts"])
            speed = f"{got['tps']:.1f} tok/s here" if got else f"~{r['est']['tps']:.0f} tok/s" if r["est"] else "--"
            print(f"     {i}{'abcdefgh'[j]}  {r['label']:<24} {size_of(r['e']) / GB:6.1f} GB  {speed:>10}  "
                  f"{status(r, models, disk)}\n           {r['note']}")
        for line in e.get("not_made", []):
            print(f"         - {line}")


def header(prof, disk, models):
    ram = prof["ram"] / 2**30
    gpu = f"{prof.get('gpu')}" + (f", {prof['gpu_mem'] / 2**30:.0f} GB VRAM" if prof["backend"] == "cuda" else "")
    print(f"\nThis computer: {prof['chip'] or prof['arch']}, {ram:.0f} GB memory, GPU {gpu}, "
          f"{prof['os']}")
    print(f"Models go to {models}: {shutil.disk_usage(_existing(models)).free / GB:.0f} GB free"
          + (f", the SSD reads {disk['gbps']:.1f} GB/s" if disk else ""))
    others = [d for d in prof["now"]["disks"] if d["dev"] != machine.volume_of(models)]
    if others:
        print("Other disks: " + ", ".join(f"{d['mount']} ({d['free'] / GB:.0f} GB free)" for d in others)
              + "  (use --models-dir to put models there)")


def ask(prompt, default=""):
    try:
        got = input(prompt).strip()
    except EOFError:
        got = ""
    return got or default


# ---------------------------------------------------------------- one command, no questions

def cache_file(name):
    """A file in the launcher's cache directory (where the models and logs live)."""
    return Path(machine.CACHE) / name


def load_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def auto_candidates(prof, models, disk, fetch=None, log=None):
    """[(entry, judged)] for the tested models, read from disk when prepared and
    from Hugging Face otherwise; a model that cannot be inspected is left out."""
    fetch = fetch or hf_entry
    log = log or (lambda *a: None)
    out = []
    for c in CURATED:
        c = for_backend(c, prof["backend"])
        d = model_dir(c, models)
        e = {**c, **(load_json(d / "launcher.json") or {})} if is_ready(d) else None
        if not e or not e.get("facts"):
            try:
                e = fetch(c)
            except Unsupported as err:
                log(f"  {c['name']}: {err}")
                continue
            except Exception as err:
                log(f"  {c['name']}: could not check it now ({type(err).__name__})")
                continue
        e = library.adopt(e, models)
        out.append((e, judge(e, prof, models, disk)))
    return out


def recommend(table, models):
    """The entry with the best estimated speed among those that run, a model
    already on disk first; (entry, judged) or None."""
    best = None
    for e, r in table:
        if not r["ok"]:
            continue
        key = (is_ready(model_dir(e, models)) or bool(e.get("path") and not e.get("needs_pack")),
               (r["est"] or {}).get("tps", 0.0))
        if best is None or key > best[0]:
            best = (key, e, r)
    return (best[1], best[2]) if best else None


def last_entry(models):
    """The model the last run started, or None when it is gone."""
    st = load_json(cache_file("last.json")) or {}
    e = st.get("entry")
    if not e or not e.get("facts"):
        return None
    if e.get("path"):
        return e if Path(e["path"]).expanduser().exists() else None
    return e if model_dir(e, models).exists() else None


def auto_pick(prof, models, disk, fetch=None, log=None):
    """The model for an unattended run: the last one used, else the recommended
    one; returns (entry, variant, tight), tight meaning nothing fits the memory
    free now and the choice describes an idle machine."""
    e = last_entry(models)
    if e:
        r = judge(e, prof, models, disk)
        if r["ok"]:
            return e, (load_json(cache_file("last.json")) or {}).get("variant", "published"), False
        print(f"  The last model ({e['name']}) cannot run now: {r['why']}", flush=True)
    table = auto_candidates(prof, models, disk, fetch=fetch, log=log)
    got = recommend(table, models)
    if got:
        return got[0], "published", False
    idle = {**prof, "now": {**prof.get("now", {}), "avail_gb": None}}
    got = recommend([(e, judge(e, idle, models, disk)) for e, _ in table], models)
    if not got:
        raise SystemExit("No model in the list runs on this computer right now; ./larder --list says why.")
    return got[0], "published", True


def engine_plan(e, prof, disk, tight=False, native=True, d=None):
    """(settings, estimate) for starting entry e on this machine, building the native expert reader
    once on Apple when the model uses it unless native is False, and adding the model's own MTP head
    when one is on disk for it."""
    f = e["facts"]
    native_ok = bool(native and f["kind"] == "moe" and e.get("native") and prof["backend"] == "metal"
                     and native_ready())
    p = idle_profile(prof) if tight else prof
    s = machine.settings(p, {**f, **e, "native_ok": native_ok})
    head = mtp_for(d, e) if d else {"path": "", "note": ""}
    if head.get("path"):
        s["args"] = [*s["args"], "--mtp", head["path"]]
        s["why"].append(head["note"])
    elif head.get("note"):
        s["why"].append(head["note"])
    s["mtp"] = head
    kd = machine.kernel_dir(p)
    if kd and "MLX_PTX_CACHE_DIR" not in os.environ:
        # MLX's default is /tmp/mlx, which a reboot empties: ~90 s of kernel builds again on the PC.
        s["env"]["MLX_PTX_CACHE_DIR"] = str(kd)
    return s, machine.estimate(p, f, s, disk)


# The 4-bit head tools/mtp_pack.py writes beside the downloaded one.
MTP_PACK = "mtp-4bit.safetensors"
_MTP_CACHE = {}      # (pack path, mtime) -> the hidden size its fc weight declares


def _pack_hidden(f):
    """The hidden size a packed head's fc weight declares, from the safetensors header alone."""
    try:
        key = (str(f), os.stat(f).st_mtime_ns)
    except OSError:
        return None
    if key in _MTP_CACHE:
        return _MTP_CACHE[key]
    hidden = None
    try:
        with open(f, "rb") as fh:
            n = int.from_bytes(fh.read(8), "little")
            hdr = json.loads(fh.read(min(n, 4 << 20)).decode("utf-8", "replace"))
        for k in ("mtp.fc.weight", "fc.weight"):
            shape = (hdr.get(k) or {}).get("shape")
            if shape and len(shape) == 2:
                hidden = int(shape[0])
                break
    except (OSError, ValueError, AttributeError):
        pass
    _MTP_CACHE[key] = hidden
    return hidden


def mtp_for(d, e, mod=None):
    """{"path", "note"} for the MTP head this model's own config has a recipe for: the 4-bit pack on
    this computer whose shapes fit it, or what such a head would add and how to make one."""
    import mtp as real
    m = mod or real
    cfg = load_json(Path(str(d)) / "config.json") or {} if d else {}
    try:
        hidden = m.head_args(cfg).hidden_size
    except Exception:
        return {"path": "", "note": ""}
    cands = [Path(str(d)) / MTP_PACK] if d else []
    if m.DEFAULT_MTP:
        cands.append(Path(m.DEFAULT_MTP) / MTP_PACK)
    for f in cands:
        if f.is_file() and f.with_name(f.stem + "-quant.json").is_file() and _pack_hidden(f) == hidden:
            return {"path": str(f), "note": "its MTP head is on this computer and is used: about 25% more "
                                            "tokens a second, the same output"}
    return {"path": "", "note": "its MTP head is not on this computer: tools/mtp_pack.py makes one from the "
                                "downloaded head, about 25% more tokens a second, the same output"}


def note_estimate(e, d, est, s=None):
    """Remember the estimate this run is started on, so its measured speed can be compared with it
    and the next estimate corrected (library.note_estimate); never a reason to fail a start."""
    if not est:
        return
    try:
        recipe = e.get("recipe") or (e.get("facts") or {}).get("recipe") or ""
        library.note_estimate(d, est, recipe, machine.conditions())
    except Exception:
        pass


def live_status(port, timeout=2.0):
    """The running server's /status, or None when it does not answer."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{int(port)}/status", timeout=timeout) as r:
            return json.load(r)
    except (OSError, ValueError, TypeError):
        return None


def speed_ref(e, est, backend):
    """What the Speed line compares with before replies are timed: the estimate, except on CUDA, whose bandwidth
    model misses the resident MTP recipe (~57 against ~147 tok/s measured on the PC): there this computer's own
    earlier replies of the model (library.lookup), else no number."""
    if backend != "cuda":
        return est
    m = library.lookup(e["facts"])
    return ({"tps": m["tps"], "label": f"measured here before, {m['n']} replies"} if m
            else {"tps": None, "label": "not measured yet"})


def speed_text(st, est=None, label=None):
    """One line: the speed the server measured over its replies, beside the estimate in tok/s the
    launcher made before them (or what `label` says it is), and what else is going on."""
    n = (st or {}).get("decode_replies") or 0
    tps = (st or {}).get("decode_tok_s")
    if n and tps:
        text = f"{tps:.1f} tok/s measured over {n} {'reply' if n == 1 else 'replies'}"
    else:
        text = "no reply timed yet"
    if est:
        was = f"~{est:.0f} tok/s {label or 'estimate'}"
        text += f" (the {was})" if n else f"; the {was} for this computer" if not label else f"; {was}"
    elif label and not n:
        text += f"; speed {label} on this computer"
    cond = (st or {}).get("conditions_text")
    return text + (f" · {cond}" if cond else "")


class SpeedWatch(threading.Thread):
    """A line under the banner that the running server keeps honest: the measured speed once it has
    timed replies, the estimate before that."""

    def __init__(self, port, est=None, every=3.0, label=None):
        super().__init__(daemon=True)
        self.port, self.est, self.every, self.label = port, est, every, label
        self._stop = threading.Event()
        self.tty = sys.stdout.isatty()
        self.seen = None

    def run(self):
        while not self._stop.wait(1.0 if self.seen is None else self.every):
            st = live_status(self.port)
            if not st:
                continue
            n = st.get("decode_replies") or 0
            if n < 1 or n == self.seen:
                continue
            self.seen = n
            text = "Speed        " + speed_text(st, self.est, self.label)
            if self.tty:
                print("\r" + fit_line(text), end="", flush=True)
            else:
                print(text, flush=True)

    def stop(self):
        self._stop.set()


def fit_line(text, n=78):
    """text cut or padded to n columns, for a line rewritten in place."""
    return text if len(text) >= n else text.ljust(n)


def idle_profile(prof):
    """The same machine with the memory a start would have on an idle computer."""
    return {**prof, "now": {**prof.get("now", {}), "avail_gb": None}}


def remember(e, variant):
    """Save the model and variant for the next unattended run."""
    machine.CACHE.mkdir(parents=True, exist_ok=True)
    cache_file("last.json").write_text(json.dumps({"variant": variant, "entry": e}, indent=1))


def record_server(e, d, port, proc, log, est=None, mtp=False):
    """Save where the running server is, so `./larder stop` and `./larder status` find it, with the
    estimate and the conditions this run was started on."""
    machine.CACHE.mkdir(parents=True, exist_ok=True)
    cache_file("server.json").write_text(json.dumps({
        "pid": proc.pid, "port": port, "id": e.get("id"), "name": e["name"], "model": str(d),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"), "log": str(log),
        "est": (est or {}).get("tps"), "est_lo": (est or {}).get("lo_tps"),
        "est_hi": (est or {}).get("hi_tps"), "est_runs": (est or {}).get("runs", 0),
        "est_label": (est or {}).get("label"), "mtp": bool(mtp), "cond": machine.conditions()}, indent=1))


def clear_server(pid=None):
    """Forget the saved server when it is this process (or any, with pid None)."""
    st = load_json(cache_file("server.json"))
    if st and (pid is None or st.get("pid") == pid):
        cache_file("server.json").unlink(missing_ok=True)


def pid_alive(pid):
    pid = int(pid or 0)
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def pid_command(pid):
    try:
        r = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return ""
    return r.stdout.strip()


def server_info():
    """(saved state, whether that server is still running) for `./larder status`."""
    st = load_json(cache_file("server.json"))
    if not st or not pid_alive(st.get("pid")):
        return st, False
    cmd = pid_command(st["pid"])
    ours = "server.py" in cmd and (not st.get("model") or st["model"] in cmd)
    return st, ours


def show_status():
    """Print what this launcher started and where, without touching the model."""
    st, ours = server_info()
    if not st:
        print("No model server from ./larder is running.")
        return 0
    if not ours:
        print("No model server from ./larder is running (the last one exited).")
        clear_server()
        return 0
    base = f"http://127.0.0.1:{st['port']}"
    print(f"Running: {st.get('name', 'the model')}")
    print(f"  Web chat     {base}/")
    print(f"  API base     {base}/v1")
    print(f"  Model        {st.get('model')}")
    print(f"  Started      {st.get('started')}   (process {st['pid']}, port {st['port']})")
    live = live_status(st["port"])
    print(f"  Speed        {speed_text(live or {}, st.get('est'), st.get('est_label'))}")
    if st.get("est") and not st.get("est_label") and live and not live.get("decode_replies"):
        print(f"  Estimate     {tui.estimate_text(st['est'], st.get('est_lo'), st.get('est_hi'), st.get('est_runs'))}")
    if st.get("mtp"):
        print("  MTP head     its own head is used, about 25% more tokens a second")
    if live and live.get("arena") and live["arena"].get("slots") is not None:
        print(f"  Arena        {live['arena']['slots']} slots, {LC_GB(live['arena'].get('gb'))} of experts held")
    elif live and live.get("store") and live["store"].get("slots"):
        print(f"  Store        {live['store']['slots']} slots, {LC_GB((live['store'].get('mb') or 0) / 1000)}")
    if live and (live.get("experts") or {}).get("rows_per_token"):
        print(f"  Expert reads {live['experts']['rows_per_token']:.0f} rows a token from the SSD")
    print(f"  Log          {st.get('log')}")
    print("  Stop         ./larder stop")
    return 0


def LC_GB(gb):
    """GB or MB as a short string, "-" when unknown."""
    if gb is None:
        return "-"
    return f"{gb:.2f} GB" if gb >= 0.1 else f"{gb * 1000:.0f} MB"


def stop_server():
    """Stop the saved server with TERM and forget it; never KILL."""
    st, ours = server_info()
    if not st:
        print("No model server from ./larder is running.")
        return 0
    if ours:
        os.kill(int(st["pid"]), signal.SIGTERM)
        print(f"Stopping {st.get('name', 'the model')} (process {st['pid']})...", end=" ", flush=True)
        t0 = time.time()
        while pid_alive(st["pid"]) and time.time() - t0 < 60:
            time.sleep(0.2)
        if pid_alive(st["pid"]):
            print("still running; its log says why (./larder status).")
            return 1
        print("stopped.")
    else:
        print("No model server from ./larder is running (the last one exited).")
    clear_server()
    return 0


# ---------------------------------------------------------------- deleting

def running_models():
    """Real paths of the model folders a server.py on this computer is serving: the one ./larder
    started and any other found by its command line."""
    out = set()
    st, ours = server_info()
    if st and ours and st.get("model"):
        out.add(os.path.realpath(st["model"]))
    try:
        ps = subprocess.run(["ps", "-axo", "command="], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return out
    for line in ps.splitlines():
        if "server.py" not in line or "--model" not in line:
            continue
        try:
            words = shlex.split(line)
        except ValueError:
            continue
        for i, w in enumerate(words):
            got = words[i + 1] if w == "--model" and i + 1 < len(words) else w[8:] if w.startswith("--model=") else ""
            if got:
                out.add(os.path.realpath(os.path.expanduser(got)))
    return out


def delete_folder(path, models, running=None):
    """Delete the model folder at `path` if it is inside the launcher's models folder, returning (True, what was deleted) or (False, why not): never a folder elsewhere, a running model, or one a copy made here links into."""
    p = Path(path)
    if not path or not p.is_dir():
        return False, f"{path or 'that model'} is not on disk"
    if not library.inside(p, models):
        return False, (f"{library.short_path(p)} is outside the launcher's models folder ({models}): the "
                       f"launcher deletes only what it downloaded or made, so delete it there yourself")
    real = p.resolve()
    for r in running_models() if running is None else running:
        if r and (Path(r).resolve() == real or real in Path(r).resolve().parents):
            return False, f"{p.name} is running: stop it first"
    for other in sorted(Path(models).iterdir()):
        if not other.is_dir() or other.resolve() == real:
            continue
        if any(f.is_symlink() and real in f.resolve().parents for f in other.iterdir()):
            return False, f"{other.name} was made from it and links to its files: delete that one first"
    size = tree_bytes(p)
    shutil.rmtree(p)
    return True, f"deleted {p} ({size / GB:.1f} GB)"


def delete_cli(args):
    """--delete ID --variant KEY: the folder the launcher keeps that variant in, deleted after a yes
    (--yes answers it); the exit code says whether it was."""
    models = Path(args.models_dir).expanduser()
    backend = machine.facts().get("backend")
    # A listed id names the entry this backend installs (on CUDA qwen3.6-35b is the folder qwen3.6-35b-q40).
    cur = next((for_backend(c, backend) for c in CURATED if args.delete in (c["id"], for_backend(c, backend)["id"])), None)
    name = cur["id"] if cur else args.delete
    ident = name if args.variant == "published" else \
        f"{name}-{args.variant}" if args.variant in ("q3", "q4") else args.variant.replace("/", "--")
    d = models / ident

    def sure(at):
        return args.yes or ask(f"Delete {args.delete} {args.variant} at {at} ({tree_bytes(at) / GB:.1f} GB)? "
                               "type yes: ").lower() == "yes"
    if d.is_dir() and not sure(d):
        print("Not deleted.")
        return 1
    ok, why = delete_folder(d, models)
    if not ok and not d.is_dir():
        try:                                   # the folder the list found it in, if it is not that one
            at = by_variant(hf_entry(cur), args.variant, models).get("path") if cur else None
        except (Exception, SystemExit):        # not a tested model, or Hugging Face is out of reach
            at = None
        if at and library.inside(at, models):
            if not sure(Path(at)):
                print("Not deleted.")
                return 1
            ok, why = delete_folder(at, models)
        elif at:
            why = (f"{args.delete} {args.variant} is at {library.short_path(at)}, outside the launcher's models "
                   f"folder ({models}): the launcher deletes only what it downloaded or made")
        else:
            why = f"{args.delete} {args.variant} is not in the launcher's models folder ({models})"
    print(why)
    return 0 if ok else 1


def short_banner(name, port, est=None):
    """The whole output of an unattended run: where the chat is and how to stop it."""
    if est and est.get("label"):
        tps = f"~{est['tps']:.0f} tok/s " if est["tps"] else ""
        speed = f"   ({tps}{est['label']})"
    else:
        speed = f"   ({tui.estimate_text(est['tps'], est.get('lo_tps'), est.get('hi_tps'), est.get('runs'))})" if est else ""
    base = f"http://127.0.0.1:{port}"
    print(f"""

  Ready: {name}{speed}

  Chat   {base}/
  API    {base}/v1    (OpenAI-compatible, no key needed)
  Stop   ./larder stop   (or Ctrl+C here)
""", flush=True)


# ---------------------------------------------------------------- serving



def offer_refresh(d, t):
    """Ask once when the pack reads slower than its stored baseline; nothing is rewritten without a yes."""
    if not Path(d, "experts.bin").exists() or not sys.stdin.isatty():
        return
    try:
        a = tuner.pack_age(d)
    except Exception:                            # a side check never blocks a start
        return
    if not a.get("aged") or a.get("asked"):
        return
    gb = a["size_bytes"] / 1e9
    when = " (the baseline came from the first tuner run and may itself be aged)" if a["baseline_source"] == "first_tune" else ""
    print(f"\nThis model's file reads {a['gap_qd8']:.0%} slower than when it was last fresh{when} (noise {a['noise']:.1%}); "
          f"rewriting it with the same bytes restores the speed.\n  Writes {gb:.0f} GB, about {gb / 0.6:.0f} s "
          f"(./larder refresh does it any time).")
    tuner.mark_asked(d)
    if ask("  Rewrite now? [y/N] ").lower().startswith("y"):
        try:
            import pack_refresh
            err = pack_refresh.refresh(d)
        except Exception as e:                   # a failed rewrite never blocks a start
            err = f"{type(e).__name__}: {e}"
        if err:
            print(f"  not refreshed: {err}")


def tune_once(d):
    """The cached first-run tuning of model folder d, measured now when there is none, except in a served CUDA
    process (machine.KEY set), where measuring would import MLX and hold a CUDA context beside the server."""
    t = tuner.load(d)
    if t is None and not machine.KEY and Path(d, "experts.bin").exists():
        print("\nFirst run of this model on this machine: measuring the disk (under 2 minutes, once).", flush=True)
        try:
            t = tuner.tune(d, lp=sys.modules[__name__])
        except Exception as ex:                  # tuning is a convenience: never block a start
            print(f"  tuning skipped: {ex}", flush=True)
    return t


def start_seconds(d, took=None):
    """Seconds this computer's last start of model folder d took to ready, saved first when `took` is given."""
    f = cache_file("starts.json")
    data = load_json(f) or {}
    key = f"{machine.fingerprint()} {d}"
    if took is not None:
        data[key] = round(took, 1)
        machine.CACHE.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(data, indent=1))
    return data.get(key)


def kernel_cache(s):
    """Whether MLX's kernel folder in the plan (MLX_PTX_CACHE_DIR under mlx-kernels/) is empty, so this start builds
    every kernel, None without such a folder; the folder is made and the folders of other keys beside it removed."""
    d = s.get("env", {}).get("MLX_PTX_CACHE_DIR")
    if not d or Path(d).parent.name != "mlx-kernels":
        return None
    d = Path(d)
    d.parent.mkdir(parents=True, exist_ok=True)
    for old in d.parent.iterdir():
        if old != d and old.is_dir():
            shutil.rmtree(old, ignore_errors=True)
    d.mkdir(exist_ok=True)
    return not any(d.iterdir())


def start_note(last, kernels=None):
    """What a start says while the engine loads and warms up; `kernels`: MLX's kernel folder is empty (True), kept
    (False) or not ours (None, Metal)."""
    if last and not kernels:
        return f"  about {last:.0f} s to ready, as the last start here took"
    if kernels and last:
        return ("  first start with this GPU driver and MLX: the engine builds its GPU kernels once and keeps them "
                "for later starts, then answers one short warm-up message")
    if kernels is False:
        return "  loading the model, then one short warm-up message (its GPU kernels are kept from an earlier start)"
    return ("  first start of this model on this computer: the engine measures the GPU and builds its kernels "
            "once, then answers one short warm-up message (later starts are shorter)")


def warm_up(port, api_key=None):
    """One short reply before ready, so the engine's first-use work (kernel builds, its store's fill, the MTP
    rounds' warm-up) is not the user's first message; a failure is said and never stops the start."""
    t = time.time()
    print("\n  warming up: one short reply...", end="", flush=True)
    text, why = validate(port, max_tokens=8, warm=True, api_key=api_key)
    ok = text is not None or why == "an empty reply"     # 8 tokens of thinking leave no answer, and that is fine
    print(f" {time.time() - t:.0f} s" if ok else f" failed ({why})", flush=True)
    return ok


def start_server(d, s, port, log, quiet=False, host="127.0.0.1", api_key=None):
    cmd = [sys.executable, str(HERE / "server.py"), "--model", str(d), "--port", str(port),
           "--host", host] + s["args"]
    if api_key:
        cmd += ["--api-key", api_key]
    t = tune_once(d)
    offer_refresh(d, t)
    # The tuned read-path knobs rank below the user's environment and the plan's own.
    env = {**((t or {}).get("env", {})), **os.environ, **s["env"], "PYTHONUNBUFFERED": "1", "MOE_TUNED": "1"}
    if quiet:
        print(f"\nStarting the model (log: {log})", flush=True)
    else:
        print(f"\nStarting the engine: {' '.join(f'{k}={v}' for k, v in s['env'].items())} "
              f"server.py --model {d} --port {port} --host {host} {' '.join(s['args'])}", flush=True)
        print(f"  (log: {log})", flush=True)
    return subprocess.Popen(cmd, cwd=HERE, env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT)


def wait_ready(proc, port, log, limit=3600, api_key=None):
    t0 = time.time()
    headers = {"Authorization": "Bearer " + api_key} if api_key else {}
    while time.time() - t0 < limit:
        if proc.poll() is not None:
            print("".join(open(log).readlines()[-30:]))
            raise SystemExit(f"The engine stopped while loading (exit {proc.returncode}); log above, full: {log}")
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/models", headers=headers)
            with urllib.request.urlopen(req, timeout=2) as r:
                return json.load(r)["data"][0]["id"]
        except (OSError, ValueError, KeyError, IndexError):
            print(f"\r  loading the model... {time.time() - t0:.0f} s", end="", flush=True)
            time.sleep(1)
    raise SystemExit(f"The engine did not answer in {limit} s; log: {log}")


def banner(name, model_id, port, est, mtp=None):
    base = f"http://127.0.0.1:{port}"
    st = live_status(port) or {}
    est_tps = est.get("tps") if est else None
    speed = speed_text(st, est_tps, (est or {}).get("label"))
    cond = st.get("conditions_text") or machine.conditions_text(machine.conditions())
    print(f"""

  Ready: {name}

  Web chat     {base}/
  API base     {base}/v1        (OpenAI-compatible, no key needed)
  Model id     {model_id}
  Speed        {speed}
""" + (f"  Conditions   {cond}\n" if cond else "")
      + (f"  MTP head     {mtp['note']}\n" if mtp and mtp.get("note") else "") + f"""
  VS Code with Continue: add this to ~/.continue/config.yaml
    name: Local
    version: 0.0.1
    schema: v1
    models:
      - name: {name}
        provider: openai
        model: {model_id}
        apiBase: {base}/v1
        apiKey: none
        roles: [chat, edit, apply]

  Any OpenAI client: base_url="{base}/v1", api_key="none"
""", flush=True)


def terminal_chat(url):
    """Chat with a running server from this terminal, streaming each reply and then its first-token
    time and decode speed."""
    msgs = []
    print("Chat here (/new starts over, Ctrl+C stops a reply, Ctrl+D leaves).")
    while True:
        try:
            text = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text in ("/quit", "/exit"):
            break
        if text == "/new":
            msgs = []
            print("(new conversation)")
            continue
        msgs.append({"role": "user", "content": text})
        body = json.dumps({"messages": msgs, "max_tokens": 2048, "stream": True}).encode()
        req = urllib.request.Request(url + "/v1/chat/completions", body, {"Content-Type": "application/json"})
        t0, times, reply = time.perf_counter(), [], []
        try:
            with urllib.request.urlopen(req, timeout=1800) as r:
                for line in r:
                    line = line.strip()
                    if not line.startswith(b"data: ") or line == b"data: [DONE]":
                        continue
                    c = json.loads(line[6:])["choices"][0]["delta"].get("content")
                    if c:
                        times.append(time.perf_counter() - t0)
                        reply.append(c)
                        print(c, end="", flush=True)
        except KeyboardInterrupt:
            print("\n(stopped)")
        msgs.append({"role": "assistant", "content": "".join(reply)})
        if len(times) > 2:
            tps = (len(times) - 2) / (times[-1] - times[1])
            print(f"\n\n[first token {times[0]:.2f} s | {tps:.1f} tok/s | {len(times)} tokens]")


# ---------------------------------------------------------------- the full-screen menu

def tree_bytes(path):
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.lstat(os.path.join(root, name)).st_size
            except OSError:
                pass
    return total


class TuiBackend(tui.Backend):
    """What the full-screen menu (tui.py) asks of the launcher: the same functions the text menu and
    ./larder --auto use, with the install in a process the menu can stop."""

    def __init__(self, args, models):
        self.args, self.mdir = args, models
        self.prof = self.disk = None
        self.proc = self.child = None
        self.port = None                       # the port the local server answered on
        self.log = None
        self.remote = None                     # the running dict of a model started on another device
        self.rdev = None                       # that device
        self.rproc = None                      # the local ssh carrying the tunnel
        self.proxy = None                      # the local proxy in front of a shared device's server
        self.rlines = collections.deque(maxlen=400)
        self.rqueue = None
        self.local_models = None
        self.rescan_devs = set()               # devices whose next list should look at their disks again

    def info(self):
        self.prof = machine.profile(refresh=self.args.refresh)
        self.disk = machine.ssd(self.prof, self.mdir, refresh=self.args.refresh)
        return machine_json(self.prof, self.disk, self.mdir)

    def _variant(self, r):
        e = r["e"]
        path = disk_path(e, self.mdir)
        est = r["est"] or {}
        return tui.Variant(
            key=r["key"], label=r["label"], size=size_of(e), tps=est.get("tps"), ok=r["ok"], why=r["why"],
            disk_short=bool(r.get("short")), status=status(r, self.mdir, self.disk), note=r["note"],
            ready=is_ready(model_dir(e, self.mdir)) or bool(e.get("path") and not e.get("needs_pack")),
            make_bits=e.get("make_bits", 0), download=e.get("download", 0), disk=e.get("disk", 0),
            expected=e.get("disk", 0), entry_id=e["id"], path=path, own=library.inside(path, self.mdir),
            measured_here=library.lookup(e["facts"]), need_gb=(r["s"] or {}).get("need_gb"),
            tps_lo=est.get("lo_tps"), tps_hi=est.get("hi_tps"), tps_adjusted=bool(est.get("runs")),
            mtp=bool((r.get("mtp") or {}).get("path")), raw=r)

    def _model(self, e, tested, found=False):
        try:
            vs = table_of([e], self.prof, self.mdir, self.disk)[0][1]
        except Exception:                      # Hugging Face unreachable: the other bit widths are not offered
            vs = [{**judge(v["entry"], self.prof, self.mdir, self.disk), **v} for v in variants(e, self.mdir, [])]
        return tui.Model(id=e["id"], name=e["name"], kind=describe(e["facts"]), about=e.get("about", ""),
                         license=e.get("license", ""), tested=tested, variants=[self._variant(r) for r in vs],
                         not_made=e.get("not_made", []), found=found, repo=e.get("repo", ""))

    def rescan(self, dev):
        if dev.kind == "local":
            library.invalidate()
        else:
            self.rescan_devs.add(dev.name)

    def delete(self, dev, model, v):
        """Delete variant v of `model` on dev (the folder its list showed here, or its launcher's
        --delete there); returns what happened, or raises RuntimeError with the reason."""
        if dev.kind == "local":
            ok, msg = delete_folder(v.path, self.mdir)
        else:
            import remote
            code, out, err = remote._run(remote.delete_argv(dev, model.id, v.key), timeout=300)
            text = [x for x in (out + "\n" + err).splitlines() if x.strip()]
            ok, msg = code == 0, text[-1] if text else f"{dev.name} answered with exit {code}"
        if not ok:
            raise RuntimeError(msg)
        return msg

    def models(self, say):
        for c in CURATED:
            c = for_backend(c, (self.prof or {}).get("backend"))
            say(f"checking {c['name']}...")
            d = model_dir(c, self.mdir)
            try:
                e = {**c, **(load_json(d / "launcher.json") or {})} if is_ready(d) else None
                e = e if e and e.get("facts") else hf_entry(c)
            except Unsupported as err:
                why = str(err)
                if "not in the installed mlx-lm" in why:
                    why += " (requirements.txt installs the mlx-lm that has it: .venv/bin/pip install -r requirements.txt)"
                yield tui.Model(c["id"], c["name"], about=c.get("about", ""), license=c.get("license", ""), tested=True,
                                unavailable=why)
                continue
            except Exception as err:
                yield tui.Model(c["id"], c["name"], about=c.get("about", ""), license=c.get("license", ""), tested=True,
                                unavailable=f"could not reach Hugging Face ({type(err).__name__})")
                continue
            say(f"checking {c['name']}'s variants...")
            yield self._model(e, True)

    def search(self, query, say):
        say("reading the repo..." if parse_repo(query) else "searching Hugging Face...")
        try:
            found, skipped = search_models(query)
        except Unsupported as err:
            raise ValueError(str(err))
        except Exception as err:
            raise ValueError(f"{type(err).__name__}: {err}")
        return [self._model(e, False, found=True) for e in found], len(skipped)

    def device_search(self, dev, query, say):
        if dev.kind == "local":
            return self.search(query, say)
        import remote
        say(f"asking {dev.name} to look on Hugging Face (its launcher judges what runs there)...")
        code, out, err = remote._run(remote.search_argv(dev, query), timeout=300)
        if code != 0 and not out.strip():
            last = (err or out).strip().splitlines()
            raise RuntimeError(f"{dev.name}: ssh returned no answer ({last[-1] if last else f'exit {code}'})")
        try:
            data = remote.parse_listing(out)
        except (ValueError, json.JSONDecodeError) as e:
            raise RuntimeError(f"{dev.name}: {e}")
        if data.get("error"):
            raise RuntimeError(data["error"])
        if data.get("machine"):
            dev.info = data["machine"]
        return remote.models_from_listing(data, found=True), int(data.get("skipped") or 0)

    def local(self, path):
        try:
            return self._model(local_entry(path), False, found=True)
        except Unsupported as err:
            raise ValueError(str(err))
        except OSError as err:
            raise ValueError(f"{path}: {err.strerror or err}")

    def last(self):
        e = (load_json(cache_file("last.json")) or {}).get("entry") or {}
        return e.get("id")

    def dest(self, v):
        return str(model_dir(v.raw["e"], self.mdir))

    def progress(self, v):
        e = v.raw["e"]
        return sum(tree_bytes(model_dir(x, self.mdir)) for x in (e, e.get("source")) if x)

    def install(self, v, job):
        e = v.raw["e"]
        job.say(f"preparing {e['name']}")
        fd, path = tempfile.mkstemp(suffix=".json", prefix="moe-entry-")
        with os.fdopen(fd, "w") as f:
            json.dump(e, f)
        cmd = [sys.executable, str(HERE / "launch.py"), "--prepare", path, "--models-dir", str(self.mdir)]
        self.child = subprocess.Popen(cmd, cwd=HERE, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT)
        try:
            while chunk := os.read(self.child.stdout.fileno(), 4096):
                sys.stdout.write(chunk.decode("utf-8", "replace"))
            code = self.child.wait()
        finally:
            os.unlink(path)
        if job.cancelled():
            return
        if code != 0:
            raise RuntimeError(f"the preparing step failed (exit {code}); running the same choice again resumes it")

    def start(self, v, job):
        e = v.raw["e"]
        self.rlines.clear()
        job.say("checking the native expert reader (built once, about a minute, when the model uses it)")
        d = model_dir(e, self.mdir)
        s, est = engine_plan(e, self.prof, self.disk, d=d)
        note_estimate(e, d, est, s)
        port = self.args.port                      # the chat port, so the address stays the same
        self.port = port
        machine.CACHE.mkdir(parents=True, exist_ok=True)
        self.log = machine.CACHE / f"server-{time.strftime('%Y%m%d-%H%M%S')}.log"
        job.say("starting the engine")
        self.proc = start_server(d, s, port, self.log, quiet=True)
        t0 = time.time()
        while True:
            if job.cancelled():
                self.abort()
                return None
            if self.proc.poll() is not None:
                raise RuntimeError(f"the engine stopped while loading (exit {self.proc.returncode}); log: {self.log}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2) as r:
                    model_id = json.load(r)["data"][0]["id"]
                break
            except (OSError, ValueError, KeyError, IndexError):
                job.say(f"loading the model... {time.time() - t0:.0f} s")
                time.sleep(1)
            if time.time() - t0 > 3600:
                raise RuntimeError(f"the engine did not answer in an hour; log: {self.log}")
        job.say("warming up: one short reply, so the first message does not wait for the engine's first-use work")
        validate(port, max_tokens=8, warm=True)
        if e.get("make_bits") and not is_ready(d):          # made here: offered once it has answered
            job.say("checking the copy made here with one short reply")
            text, why = validate(port)
            if text is None:
                self.abort()
                raise RuntimeError(f"the {e['name']} copy failed its check ({why}); it is not offered. Its files: {d}")
            mark_ready(d, e)
        remember(e, v.key)
        record_server(e, d, port, self.proc, self.log, est, bool(s.get("mtp", {}).get("path")))
        base = f"http://127.0.0.1:{port}"
        return {"name": e["name"], "url": base, "api": base + "/v1", "model_id": model_id, "model": str(d),
                "tps": est["tps"] if est else None, "est": est, "need_gb": s.get("need_gb"),
                "mtp": s.get("mtp"), "port": port, "log": str(self.log)}

    def attached(self):
        st, ours = server_info()
        if not (st and ours):
            return None
        base = f"http://127.0.0.1:{st['port']}"
        est = {"tps": st["est"], "lo_tps": st.get("est_lo"), "hi_tps": st.get("est_hi"),
               "runs": st.get("est_runs", 0)} if st.get("est") else None
        return {"name": st.get("name", "the model"), "url": base, "api": base + "/v1", "model_id": "",
                "tps": st.get("est"), "est": est, "model": st.get("model", ""), "log": st.get("log"),
                "port": st["port"], "mtp": {"path": "" if not st.get("mtp") else "on disk",
                                            "note": ""}, "attached": True}

    def live(self):
        """The running server's /status, for the ready screen's measured speed and conditions."""
        port = (self.remote or {}).get("port") or self.port
        return live_status(port) if port else None

    def log_tail(self, n):
        if self.rlines:
            return list(self.rlines)[-n:]
        try:
            with open(self.log, "rb") as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - 8192))
                return [x for x in f.read().decode("utf-8", "replace").splitlines() if x.strip()][-n:]
        except (OSError, TypeError):
            return []

    def abort(self):
        self._stop_remote(wait=30)
        for p in (self.child, self.proc):
            if p and p.poll() is None:
                p.terminate()                  # TERM: the engine releases the GPU itself
                try:
                    p.wait(30)
                except subprocess.TimeoutExpired:
                    p.kill()
        if self.proc:
            clear_server(self.proc.pid)
        self.child = self.proc = None

    def stop(self):
        if self.rproc or self.remote:
            self.stop_device()
        elif self.proc:
            self.abort()
        elif server_info()[1]:                 # a model an earlier run started
            stop_server()

    def shutdown(self):
        self.abort()

    def open_chat(self, url):
        webbrowser.open(url + "/")

    def chat(self, url):
        terminal_chat(url)

    # ---- devices: this computer first, then the presets, ssh hosts and Tailscale peers

    def device_list(self):
        import remote
        if self.prof is None:
            self.info()
        p = self.prof
        st, ours = server_info()
        here = tui.Device(name="this computer", target="", kind="local", source="here",
                          hardware=f"{p['chip'] or p['arch']}, {p['ram'] / 2**30:.0f} GB memory",
                          status="running" if st and ours else "ready",
                          running=(st.get("name") or "a model") if st and ours else "",
                          port=(st.get("port") or 0) if st and ours else 0,
                          detail=f"engine at {HERE}")
        if not self.devices_on():
            return [here]                      # the other computers only on request: `r` then y, or ./larder --devices
        try:
            rows = remote.device_list()
        except Exception as err:
            here.detail += f"; other devices could not be listed ({type(err).__name__}: {err})"
            rows = []
        import devices
        return [here] + rows + remote.shared_devices(devices.beacons())

    def devices_on(self):
        return getattr(self.args, "devices", defaults.DEVICES)

    def device_probe(self, dev):
        import remote
        if dev.kind in ("local", "scan"):      # a host only the network scan found gets no ssh until it is opened
            return dev
        if dev.kind == "shared":
            import share
            if share.token_for(dev.target, dev.port):
                dev.status, dev.detail = "ready", f"paired with {dev.name}"
            else:
                dev.status, dev.detail = "shared", "pair with the code shown on that computer"
            return dev
        try:
            return remote.probe(dev)
        except Exception as err:
            dev.status, dev.detail = "offline", f"could not check it ({type(err).__name__}: {err})"
            return dev

    def pair_fingerprint(self, dev):
        import share
        if not dev.https:
            return ""
        dev.fingerprint = share.fetch_fingerprint(dev.target, dev.port)
        if not dev.fingerprint:
            raise RuntimeError("the device has no TLS certificate to pin")
        return dev.fingerprint

    def sweep_subnet(self):
        import devices
        return devices.subnet()

    def pair(self, dev, code):
        import share
        token, fingerprint = share.pair(dev.target, dev.port, code, https=dev.https,
                                        fingerprint=dev.fingerprint or None)
        share.save_pairing(machine.CACHE, dev.target, dev.port, dev.name, token,
                           https=dev.https, fingerprint=fingerprint)
        dev.status, dev.detail = "ready", f"paired with {dev.name}"
        return dev

    def device_scan(self, say):
        import devices, remote
        say("sweeping this network, a minute at most...")
        result = devices.scan(progress=lambda done, total: say(f"looking at {done}/{total} addresses..."))
        rows = remote.scanned_devices(result)
        rows += remote.shared_devices(devices.beacons())
        say(result.get("error") or f"found {len(rows)} address(es) with a computer on them")
        if not self.devices_on():                 # the list the start left out, asked for by this key
            rows = remote.device_list() + rows
        return rows

    def device_models(self, dev, say):
        if dev.kind == "local":
            yield from self.models(say)
            return
        if dev.kind == "shared":
            yield from self._shared_models(dev, say)
            return
        import remote
        say(f"asking {dev.name} for its model list (its own launcher; no model starts)...")
        argv = remote.list_argv(dev, rescan=dev.name in self.rescan_devs)
        if argv is None:
            raise RuntimeError(remote.no_address(dev))
        self.rescan_devs.discard(dev.name)
        code, out, err = remote._run(argv, timeout=300)
        text = (out or "") + ("\n" + err if err else "")
        if code != 0 and not out.strip():
            last = text.strip().splitlines()[-1] if text.strip() else f"exit {code}"
            raise RuntimeError(f"{dev.name}: ssh returned no list ({last})")
        try:
            data = remote.parse_listing(out)
        except (ValueError, json.JSONDecodeError) as e:
            raise RuntimeError(f"{dev.name}: {e}")
        if data.get("machine"):
            dev.info = data["machine"]
        if data.get("engine"):
            dev.engine = data["engine"]
            dev.engine_state = remote.version_state(dev.engine)
        say(f"{dev.name} listed {len(data['models'])} of the tested models")
        ready_ids = [m["id"] for m in (dev.commands or {}).values() if m.get("id")]
        yield from remote.models_from_listing(data, dev.measured, ready_ids=ready_ids)

    def _shared_models(self, dev, say):
        import remote
        import share
        token = share.token_for(dev.target, dev.port)
        if not token:
            raise RuntimeError(f"{dev.name} is not paired; pair with its code first")
        say(f"asking {dev.name} for its model list...")
        status, data = share.request(dev.target, dev.port, token, "GET", "/models")
        if status != 200:
            raise RuntimeError(data.get("error") or f"{dev.name} refused to list its models")
        if data.get("machine"):
            dev.info = data["machine"]
        if data.get("engine"):
            dev.engine = data["engine"]
            dev.engine_state = remote.version_state(dev.engine)
        say(f"{dev.name} listed {len(data.get('models') or [])} of the tested models")
        yield from remote.models_from_listing(data, dev.measured)

    def update_device(self, dev, job):
        """Update the engine on dev over ssh, then forget its list so it is read fresh."""
        import remote
        ok, msg = remote.update_device(dev, job)
        if not ok:
            raise RuntimeError(msg)
        dev.models, dev.listed_at = [], 0.0
        return msg

    def install_device(self, dev, job):
        """Install the engine on dev over ssh, streaming its progress into the menu's log."""
        import remote
        ok, msg = remote.install_device(dev, job)
        if not ok:
            raise RuntimeError(msg)
        dev.models, dev.listed_at = [], 0.0
        return msg

    def _tail(self, n=8):
        lines = list(self.rlines)[-n:]
        return ("; last lines:\n" + "\n".join(lines)) if lines else ""

    def start_device(self, dev, model, v, job):
        if dev.kind == "local":
            return self.start(v, job)
        if dev.kind == "shared":
            return self._start_shared(dev, model, v, job)
        import remote
        if not dev.target:
            raise RuntimeError(f"{dev.name} has no ssh address")
        key = ""
        if dev.kind == "preset":
            for k, info in dev.commands.items():
                if info.get("id") in (model.id, v.entry_id):
                    key = k
                    break
            if not key:
                known = ", ".join(sorted({i.get("id") for i in dev.commands.values() if i.get("id")}))
                raise RuntimeError(f"{dev.name}'s preset command runs {known or 'no model'}; "
                                   f"its own launcher can start {model.name} there")
            if not v.ready:
                raise RuntimeError(f"{model.name} is not prepared on {dev.name}; prepare it there with "
                                   f"its own launcher, then come back")
        port = self.args.port                      # one chat address for every model and device
        argv, remote_port = remote.start_argv(dev, (model.repo if model.found else model.id) or v.entry_id, port,
                                              key, variant=v.key)
        job.say(f"opening the tunnel to {dev.name}")
        self.rdev, self.remote = dev, None
        self.rlines.clear()
        q = self.rqueue = queue.Queue()
        self.rproc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True, bufsize=1)

        def read():
            for line in self.rproc.stdout:
                q.put(line.rstrip("\n"))
            q.put(None)

        threading.Thread(target=read, daemon=True).start()
        t0 = time.time()
        while True:                            # the remote's own line, never the tunnel's errors
            if job.cancelled():
                self.stop_device()
                return None
            try:
                line = q.get(timeout=0.5)
            except queue.Empty:
                if self.rproc.poll() is not None:
                    raise RuntimeError(f"{dev.name} closed the session before serving" + self._tail())
                if time.time() - t0 > 3600:
                    raise RuntimeError("the model did not become ready in an hour" + self._tail())
                continue
            if line is None:
                raise RuntimeError(f"{dev.name} closed the session before serving" + self._tail())
            if line.strip():
                self.rlines.append(line)
            if remote.serving_line(line, remote_port):
                break
        job.say(f"{dev.name} says it is serving; waiting for the tunnel")
        base = f"http://127.0.0.1:{port}"
        while True:
            if job.cancelled():
                self.stop_device()
                return None
            if self.rproc.poll() is not None:
                raise RuntimeError(f"{dev.name} closed the tunnel" + self._tail())
            try:
                with urllib.request.urlopen(base + "/v1/models", timeout=2) as r:
                    model_id = json.load(r)["data"][0]["id"]
                break
            except (OSError, ValueError, KeyError, IndexError):
                job.say(f"loading on {dev.name}... {time.time() - t0:.0f} s")
                time.sleep(0.5)
            if time.time() - t0 > 7200:
                raise RuntimeError("the tunnel did not open in two hours" + self._tail())
        self.remote = {"name": model.name, "url": base, "api": base + "/v1", "model_id": model_id,
                       "tps": v.tps, "measured": tuple(v.measured), "device": dev.name, "kind": dev.kind,
                       "port": port, "remote_port": remote_port, "log": ""}
        dev.status, dev.running, dev.port = "running", f"{model.name} (from the menu)", remote_port
        return self.remote

    def _start_shared(self, dev, model, v, job):
        """Start a model on a paired shared device and put a key-adding proxy on the chat port."""
        import share
        token = share.token_for(dev.target, dev.port)
        if not token:
            raise RuntimeError(f"{dev.name} is not paired; pair with its code first")
        model_id = (model.repo if model.found else model.id) or v.entry_id
        job.say(f"asking {dev.name} to start {model.name}")
        status, ans = share.request(dev.target, dev.port, token, "POST", "/start",
                                    {"id": model_id, "variant": v.key})
        if status != 200:
            raise RuntimeError(ans.get("error") or f"{dev.name} refused to start the model")
        model_port = int(ans.get("port") or 0)
        job.say("opening the local port to its server")
        self.proxy = share.start_proxy(self.args.port, dev.target, model_port, ans.get("api_key") or "",
                                       https=bool(ans.get("https")), fingerprint=ans.get("fingerprint") or "")
        base = f"http://127.0.0.1:{self.args.port}"
        t0 = time.time()
        while True:
            if job.cancelled():
                self.stop_device()
                return None
            try:
                with urllib.request.urlopen(base + "/v1/models", timeout=2) as r:
                    model_id_out = json.load(r)["data"][0]["id"]
                break
            except (OSError, ValueError, KeyError, IndexError):
                job.say(f"loading on {dev.name}... {time.time() - t0:.0f} s")
                time.sleep(0.5)
            if time.time() - t0 > 7200:
                self.stop_device()
                raise RuntimeError(f"{dev.name}'s server did not answer in two hours")
        self.remote = {"name": model.name, "url": base, "api": base + "/v1", "model_id": model_id_out,
                       "tps": v.tps, "measured": tuple(v.measured), "device": dev.name, "kind": "shared",
                       "port": self.args.port, "remote_port": model_port, "log": ""}
        dev.status, dev.running, dev.port = "running", f"{model.name} (from the menu)", model_port
        return self.remote

    def _stop_remote(self, wait=90):
        """Stop a model started on another device: its launcher when generic, the ssh trap for
        presets, the control call and local proxy for a shared device."""
        import remote
        dev, proc = self.rdev, self.rproc
        if self.remote and dev and dev.kind == "shared":
            self._stop_shared(dev)
        elif self.remote and dev and dev.kind != "preset":
            argv = remote.stop_argv(dev)
            if argv:
                try:
                    remote._run(argv, timeout=60)
                except Exception:
                    pass
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(wait)
            except subprocess.TimeoutExpired:
                return False
        if self.proxy:
            self.proxy.stop()
            self.proxy = None
        self.rproc, self.remote, self.rdev, self.rqueue = None, None, None, None
        return True

    def _stop_shared(self, dev):
        import share
        token = share.token_for(dev.target, dev.port)
        if token:
            try:
                share.request(dev.target, dev.port, token, "POST", "/stop")
            except Exception:
                pass

    def stop_device(self):
        if not (self.rproc or self.remote or self.proxy):
            return
        if not self._stop_remote():
            raise RuntimeError("the remote session did not close; the model may still run there")


def run_menu(args):
    """The full-screen menu; False when this terminal cannot show it (the caller falls back)."""
    if not tui.available() or args.text:
        return False
    try:
        tui.run(TuiBackend(args, Path(args.models_dir).expanduser()))
    except tui.curses.error as err:
        print(f"The full-screen menu can't start in this terminal ({err}); using the text menu.")
        return False
    return True


# ---------------------------------------------------------------- main

def by_variant(e, key, models):
    """Entry e's variant `key`: published, q4 / q3 (made here), or a Hugging Face repo."""
    vs = variants(e, models, [] if key in ("published", "q4", "q3") else None)
    for v in vs:
        if v["key"] == key:
            return v["entry"]
    raise SystemExit(f"--variant {key}: not offered for {e['name']} ({', '.join(v['key'] for v in vs)})")


def pick(args, prof, models, disk):
    """The entry to run: from --model (and --variant), else the menu."""
    if args.model:
        cur = {}
        for c in CURATED:                      # on CUDA a listed id and its CUDA id name the same entry
            cur[c["id"]] = cur[for_backend(c, prof["backend"])["id"]] = for_backend(c, prof["backend"])
        if args.model in cur:
            e = hf_entry(cur[args.model])
        elif Path(args.model).expanduser().exists():
            e = local_entry(args.model)
        elif re.match(r"^[\w.-]+/[\w.-]+$", args.model):
            e = hf_entry({"id": args.model.replace("/", "--"), "name": args.model, "repo": args.model})
        else:
            raise SystemExit(f"--model {args.model}: not a listed id ({', '.join(cur)}), a folder or a "
                             f"Hugging Face repo")
        return by_variant(e, args.variant, models)
    print("\nChecking the models (sizes and speed on this machine)...", flush=True)
    entries = []
    for e in CURATED:
        try:
            entries.append(hf_entry(for_backend(e, prof["backend"])))
        except Unsupported as err:
            print(f"  {e['name']}: {err}")
        except Exception as err:
            print(f"  {e['name']}: could not reach Hugging Face ({type(err).__name__})")
    found = []
    while True:
        print("\nChecking each model's variants...", flush=True)
        table = table_of(entries, prof, models, disk) + table_of(found, prof, models, disk)
        print()
        show(table, models, disk)
        if args.list:
            return None
        print("\n  1 = the first model as published, 1b = its variant b    s  search Hugging Face    "
              "l  a model folder on this computer    q  quit")
        got = ask("\nPick: ").lower()
        if got == "q":
            return None
        if got == "s":
            q = ask("Search for: ")
            print("Searching...", flush=True)
            try:
                found, skipped = search_models(q)
            except Unsupported as err:
                print(f"  Can't use it: {err}")
                continue
            if skipped:
                print(f"  {len(skipped)} results skipped (not MLX safetensors, an unknown model type, or gated)")
            continue
        if got == "l":
            try:
                found = [local_entry(ask("Folder: "))]
            except Unsupported as err:
                print(f"  Can't use it: {err}")
            continue
        m = re.fullmatch(r"(\d+)([a-h]?)", got)
        if m and 1 <= int(m[1]) <= len(table):
            vs = table[int(m[1]) - 1][1]
            j = "abcdefgh".index(m[2]) if m[2] else 0
            if j >= len(vs):
                continue
            r = vs[j]
            if not r["ok"]:
                print(f"  {r['why'].capitalize()}." if r.get("short") else f"  Can't run that one here: {r['why']}")
                continue
            if r["e"].get("make_bits") and ask(
                    f"  This makes a {r['e']['make_bits']}-bit copy: smaller, with a quality loss ({r['note']}). "
                    "The model as published is kept. Make it? [y/N] ", "n").lower() not in ("y", "yes"):
                continue
            return r["e"]


def main():
    ap = argparse.ArgumentParser(description="Run a large language model on this computer.")
    ap.add_argument("--model", help="a listed id, a Hugging Face repo (owner/name) or a model folder")
    ap.add_argument("--search", help="search Hugging Face for runnable models and list them")
    ap.add_argument("--variant", default="published",
                    help="with --model: published (default), q4 / q3 (rounded here to 4 or 3 bits), "
                         "or a Hugging Face repo of the same model at other bits")
    ap.add_argument("--models-dir", default=str(machine.CACHE / "models"),
                    help="where downloaded models go (default: %(default)s)")
    ap.add_argument("--port", type=int, default=8800, help="the chat port, the same for every model (default 8800)")
    ap.add_argument("--host", default="127.0.0.1",
                    help="the address the server binds (default 127.0.0.1; anything else needs --api-key)")
    ap.add_argument("--api-key", default=None,
                    help="require this key as Bearer or x-api-key on every request")
    ap.add_argument("--prepare-only", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--version", action="store_true", help="print the engine's version and protocol, then exit")
    ap.add_argument("--list", action="store_true", help="list the models with their speed here, then exit")
    ap.add_argument("--json", action="store_true",
                    help="with --list: the list as one JSON object, for another computer's menu")
    ap.add_argument("--no-measure", action="store_true",
                    help="with --list --json: use cached measurements, run no GPU or SSD test")
    ap.add_argument("--pick", action="store_true", help="the menu (the default when you run ./larder in a terminal)")
    ap.add_argument("--text", action="store_true", help="the menu as plain text, for terminals without curses")
    ap.add_argument("--prepare", help=argparse.SUPPRESS)           # the menu installs in a process it can stop
    ap.add_argument("--serve", help=argparse.SUPPRESS)             # a CUDA start continues here after exec
    ap.add_argument("--auto", action="store_true", help="run the recommended model with no questions")
    ap.add_argument("--stop", action="store_true", help="stop the server this launcher started")
    ap.add_argument("--status", action="store_true", help="print the server this launcher started")
    ap.add_argument("--yes", action="store_true", help="answer yes to the download question")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--no-chat", action="store_true", help="serve only; Ctrl+C stops")
    ap.add_argument("--dry-run", action="store_true", help="print the engine command and exit")
    ap.add_argument("--refresh", action="store_true", help="measure the machine again")
    ap.add_argument("--rescan", action="store_true", help="look again for model folders outside the models folder")
    ap.add_argument("--devices", action="store_true", default=defaults.DEVICES,
                    help="the menu lists your other computers (ssh config, Tailscale, shared ones) and checks "
                         "each over ssh at start; without it only this one, and `r` asks before it scans the network")
    ap.add_argument("--volumes", action="store_true", default=defaults.VOLUMES,
                    help="also look for model folders at the top of other disks (macOS may ask for access "
                         "to a removable one)")
    ap.add_argument("--delete", metavar="ID", help="delete a model the launcher downloaded or made (with --variant)")
    args = ap.parse_args()
    library.VOLUMES = args.volumes
    if args.version:
        print(engine_version.version_json())
        return
    if args.json:
        args.list = True
    if args.delete:
        raise SystemExit(delete_cli(args))
    if args.stop:
        return stop_server()
    if args.status:
        return show_status()
    if args.prepare:
        prepare(json.loads(Path(args.prepare).read_text()), Path(args.models_dir).expanduser())
        return
    if args.serve:                             # the CUDA plan, served from a process that never imported MLX
        p = json.loads(Path(args.serve).read_text())
        Path(args.serve).unlink()
        machine.KEY[:] = [p["fp"]]             # the machine's key as read before exec: reading it again imports MLX
        return serve(args, p["e"], Path(p["d"]), p["s"], p["est"], p["variant"], p["unattended"], p.get("shown"))
    # ./larder with nothing else opens the full-screen menu, while a script without a terminal and
    # --dry-run get the recommended model with no questions.
    if not (args.model or args.search or args.list or args.pick or args.auto):
        if args.dry_run or not (sys.stdin.isatty() and sys.stdout.isatty()):
            args.auto = True
        elif run_menu(args):
            return
    elif args.pick and run_menu(args):
        return

    # Unattended: no questions, no terminal chat, a short banner only.
    unattended = args.auto and not (args.pick or args.list or args.search)
    if unattended:
        st, ours = server_info()
        if ours and not (args.model or args.dry_run):
            print(f"{st.get('name', 'The model')} is already running.")
            short_banner(st.get("name", "the model"), st["port"])
            if not args.no_browser:
                webbrowser.open(f"http://127.0.0.1:{st['port']}/")
            return
        if st and not ours:
            clear_server()
    if not unattended and not args.json:
        print("Looking at this computer (the first run measures the GPU and the SSD)...", flush=True)
    prof = machine.profile(refresh=args.refresh, measure=not args.no_measure)
    models = Path(args.models_dir).expanduser()
    if args.rescan:
        library.scan(models, refresh=True)
    disk = machine.ssd(prof, models, refresh=args.refresh, measure=not args.no_measure)
    if args.json and not args.search:
        json.dump(listing(prof, models, disk), sys.stdout, indent=1)
        print()
        return
    if not unattended and not args.json:
        header(prof, disk, models)
    if args.search:
        try:
            found, skipped = search_models(args.search)
        except Unsupported as err:
            if not args.json:
                raise SystemExit(f"{args.search}: {err}")
            found, skipped, why = [], [], str(err)
        except Exception as err:
            if not args.json:
                raise SystemExit(f"{args.search}: {type(err).__name__}: {err}")
            found, skipped, why = [], [], f"{type(err).__name__}: {err}"
        else:
            why = ""
        table = table_of(found, prof, models, disk)
        if args.json:               # the same rows as --list --json, for a menu on another computer
            json.dump({"machine": machine_json(prof, disk, models), "engine": engine_version.version(),
                       "skipped": len(skipped), "error": why,
                       "models": [model_json(e, vs, models, disk) for e, vs in table]}, sys.stdout, indent=1)
            print()
            return
        show(table, models, disk)
        print(f"\n({len(skipped)} results skipped: not MLX safetensors, an unknown model type, or gated)")
        return
    variant = args.variant
    tight = False
    if args.model or args.list or args.pick or not args.auto:
        e = pick(args, prof, models, disk)
    else:
        e, variant, tight = auto_pick(prof, models, disk, log=print)
    if e is None:
        return
    r = judge(e, prof, models, disk)
    if not r["ok"]:
        if not (args.dry_run and tight):
            raise SystemExit(f"{e['name']}: {r['why']} (--models-dir puts models on another disk)"
                             if r.get("short") else f"{e['name']} can't run here: {r['why']}")
        print(f"  Nothing fits the memory free right now: {r['why']}")
        print("  This is the command for an idle computer; close some apps to run it now.")
    d = model_dir(e, models)
    if not is_ready(d) and not (e.get("path") and not e.get("needs_pack")):
        if unattended:
            print(f"\nPreparing {e['name']}: {e['download'] / GB:.1f} GB to download, "
                  f"{e['disk'] / GB:.1f} GB on disk" + (f"; {status(r, models, disk)}"
                  if e.get("make_bits") else ""), flush=True)
            if args.dry_run:
                print("(dry run: nothing downloaded)")
            else:
                prepare(e, models)
        else:
            print(f"\n{e['name']}: download {e['download'] / GB:.1f} GB, {e['disk'] / GB:.1f} GB more on disk, "
                  f"to {d}" + (f"; {status(r, models, disk)}" if e.get("make_bits") else ""))
            if not args.yes and ask("Prepare it now? [Y/n] ", "y").lower() not in ("y", "yes"):
                return
            if args.dry_run:
                print("(dry run: nothing downloaded)")
            else:
                prepare(e, models)
    s, est = engine_plan(e, prof, disk, tight, native=not args.dry_run, d=d)
    port = args.port
    if args.prepare_only:
        print(f"{e['name']} is ready.")
        return
    if args.dry_run:
        print("\n" + " ".join([*(f"{k}={v}" for k, v in s["env"].items()), sys.executable, "server.py",
                               "--model", str(d), "--port", str(port), *s["args"]]))
        print("why: " + "; ".join(s["why"]))
        return
    shown = speed_ref(e, est, prof["backend"])
    if prof["backend"] == "cuda" and "mlx.core" in sys.modules:
        # Importing MLX holds a CUDA context (128 MiB of VRAM) for this process's life: serve from a fresh process,
        # with the first run's disk tuning done here, since it reads the machine through MLX.
        tune_once(d)
        plan = cache_file(f"serve-{os.getpid()}.json")
        plan.write_text(json.dumps({"e": e, "d": str(d), "s": s, "est": est, "variant": variant, "unattended": unattended,
                                    "fp": machine.fingerprint(), "shown": shown}))
        sys.stdout.flush()
        os.execv(sys.executable, [sys.executable, str(HERE / "launch.py"), *sys.argv[1:], "--serve", str(plan)])
    serve(args, e, d, s, est, variant, unattended, shown)


def serve(args, e, d, s, est, variant, unattended, shown=None):
    """Start the planned server and stay with it: banner, chat or speed watch, and stop it on exit; `shown` is
    what the Speed line compares with before replies are timed (speed_ref), the estimate when None."""
    port = args.port
    shown = est if shown is None else shown
    note_estimate(e, d, est, s)
    machine.CACHE.mkdir(parents=True, exist_ok=True)
    log = machine.CACHE / f"server-{time.strftime('%Y%m%d-%H%M%S')}.log"
    if server_info()[1]:                       # one model at a time, on the one chat port
        stop_server()
    cold = kernel_cache(s)
    proc = start_server(d, s, port, log, quiet=unattended, host=args.host, api_key=args.api_key)
    t0 = time.time()
    print(start_note(start_seconds(d), cold), flush=True)

    def stop(*_):
        if proc.poll() is None:
            proc.terminate()                  # TERM: the engine releases the GPU itself
            try:
                proc.wait(30)
            except subprocess.TimeoutExpired:
                proc.kill()
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, stop)
    try:
        model_id = wait_ready(proc, port, log, api_key=args.api_key)
        if warm_up(port, args.api_key) and not cold:     # a start that built the kernels is no guide to the next
            start_seconds(d, time.time() - t0)
        if e.get("make_bits") and not is_ready(d):    # made here: offered once it has answered
            text, why = validate(port, api_key=args.api_key)
            if text is None:
                raise SystemExit(f"\nThe {e['name']} copy failed its check ({why}); it is not offered. "
                                 f"Its files: {d}")
            mark_ready(d, e)
            print(f"\n  checked: it answered \"{text.strip()[:60]}\"")
        remember(e, variant)
        record_server(e, d, port, proc, log, shown, bool(s.get("mtp", {}).get("path")))
        if unattended:
            short_banner(e["name"], port, shown)
        else:
            banner(e["name"], model_id, port, shown, s.get("mtp"))
        watch = None
        if unattended or args.no_chat:
            watch = SpeedWatch(port, shown["tps"] if shown else None, label=(shown or {}).get("label"))
            watch.start()
        if not args.no_browser:
            webbrowser.open(f"http://127.0.0.1:{port}/")
        if unattended or args.no_chat:
            pass
        else:
            terminal_chat(f"http://127.0.0.1:{port}")
            print("The server keeps running for the browser and other apps. Ctrl+C stops it.")
        proc.wait()
        if watch:
            watch.stop()
    except KeyboardInterrupt:
        print("\nStopping the engine...")
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(30)
            except subprocess.TimeoutExpired:
                proc.kill()
        clear_server(proc.pid)


if __name__ == "__main__":
    main()
