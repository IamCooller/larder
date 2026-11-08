#!/usr/bin/env python3
"""What the launcher knows about this computer's models beyond its own folder: model folders found elsewhere and matched to the catalog by their config, and the decode speed each model has been measured at here."""
import json
import os
import statistics
import time
from collections import Counter
from pathlib import Path

import defaults
import machine

HOME = Path.home()                       # tests point this at a made-up home
SCAN_TTL = 3600                          # seconds a scan stays good; `r` and --rescan look again
UNFINISHED = (".pack_state.json", ".exl3_state.json", ".requant_state.json")
SKIP_CACHE = ("moe-ssd-offload", "huggingface")
WHERE_ORDER = {"models": 3, "cache": 2, "hf": 1, "volume": 0}      # the copy to prefer when two match
# Other volumes are read only on request (./larder --volumes), since a removable one can raise macOS's access prompt.
VOLUMES = defaults.VOLUMES


# ---------------------------------------------------------------- what a folder is

def signature(path):
    """The model a folder holds, read from its config.json: type, layers, hidden size, experts,
    the experts' bits (the commonest per-module entry, else the model's) and whether they are
    EXL3; None without a readable config."""
    try:
        cfg = json.loads((Path(path) / "config.json").read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(cfg, dict):
        return None
    text = cfg.get("text_config") or {}

    def get(*names):
        for n in names:
            for c in (cfg, text):
                if c.get(n) is not None:
                    return c[n]
        return None

    q = cfg.get("quantization") or text.get("quantization") or {}
    eq = cfg.get("expert_quant") or {}
    per = [v["bits"] for k, v in q.items() if isinstance(v, dict) and "switch_mlp" in k and v.get("bits")]
    if eq:
        bits = eq.get("k") or eq.get("bits")
    elif per:
        bits = Counter(per).most_common(1)[0][0]
    else:
        bits = q.get("bits") or 16
    return {"model_type": cfg.get("model_type") or text.get("model_type"),
            "layers": get("num_hidden_layers", "num_layers", "n_layer"),
            "hidden": get("hidden_size", "d_model", "n_embd"),
            "experts": get("num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts") or 0,
            "bits": bits, "exl3": bool(eq)}


def matches(sig, facts, size=0):
    """Whether a folder with this signature is the model whose catalog facts these are: the same
    type, layers, hidden size and experts, and the same bits (EXL3 only with EXL3)."""
    if not sig or not facts:
        return False
    if facts.get("model_type") != sig["model_type"] or facts.get("layers") != sig["layers"]:
        return False
    if facts.get("hidden") is not None:
        if sig["hidden"] != facts["hidden"]:
            return False
    elif size and facts.get("total_bytes") and abs(size / facts["total_bytes"] - 1) > 0.25:
        return False                     # a dense entry without a hidden size: the weights' size decides
    if facts.get("experts") and sig["experts"] != facts["experts"]:
        return False
    if (facts.get("format") == "exl3") != sig["exl3"]:
        return False
    return sig["bits"] in (None, facts.get("bits"))


# ---------------------------------------------------------------- the scan

def hf_hub():
    return Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME") or HOME / ".cache" / "huggingface") / "hub")


def volumes(models):
    """Mount points of the other disks the launcher knows (machine.disks), not the ones it already reads."""
    here = {machine.volume_of(models), machine.volume_of(HOME)}
    return [d["mount"] for d in machine.disks() if d["dev"] not in here]


def roots(models, others, deep=True):
    """[(where, parent folder)] whose sub-folders may be models: the models folder, ~/.cache, the
    Hugging Face cache (its snapshots too, when `deep`), and the top of each other volume (and its
    models folder)."""
    out = [("models", Path(models)), ("cache", HOME / ".cache")]
    hub = hf_hub()
    if not deep:
        out.append(("hub", hub))
    else:
        try:
            out += [("hf", s) for r in sorted(hub.glob("models--*")) for s in sorted((r / "snapshots").glob("*"))]
        except OSError:
            pass
    for m in others:
        out += [("volume", Path(m)), ("volume", Path(m) / "models")]
    return out


def read_folder(where, p):
    """The facts of one candidate folder, or None for a folder that is unfinished, made by the
    launcher from another (it has its own entry), or holds no weights."""
    sig = signature(p)
    if sig is None or (p / ".made").exists() or any((p / n).exists() for n in UNFINISHED):
        return None
    packed = (p / "experts.bin").is_file() and (p / "experts_index.json").is_file()
    shards = [f for f in os.scandir(p) if f.name.endswith(".safetensors") and f.is_file()]
    if not shards and not packed:
        return None
    try:
        names = set(json.loads((p / "model.safetensors.index.json").read_text())["weight_map"].values())
    except (OSError, ValueError, KeyError, AttributeError):
        names = set()
    if any(not (p / n).exists() for n in names):
        return None                      # a download that stopped half way
    repo = p.parent.parent.name[len("models--"):].replace("--", "/", 1) if where == "hf" else ""
    try:
        made = json.loads((p / "launcher.json").read_text()).get("id", "")
    except (OSError, ValueError, AttributeError):
        made = ""
    size = sum(f.stat().st_size for f in os.scandir(p) if f.is_file())
    return {"path": str(p), "where": where, "repo": repo, "id": made, "sig": sig, "packed": packed, "size": size}


def _walk(parent, skip=()):
    try:
        names = sorted(os.listdir(parent))
    except OSError:
        return
    for n in names:
        if not n.startswith(".") and n not in skip and (Path(parent) / n / "config.json").is_file():
            yield Path(parent) / n


def cache_path():
    return Path(machine.CACHE) / "found.json"


def scan(models, refresh=False):
    """Every complete model folder the roots hold, cached for SCAN_TTL (and until the roots
    change); refresh looks again."""
    models = Path(models).expanduser()
    others = volumes(models) if VOLUMES else []
    key = [[w, str(p)] for w, p in roots(models, others, deep=False)]
    try:
        old = json.loads(cache_path().read_text())
    except (OSError, ValueError):
        old = {}
    if not refresh and old.get("roots") == key and time.time() - old.get("at", 0) < SCAN_TTL:
        return [c for c in old.get("folders", []) if Path(c["path"]).is_dir()]
    out = []
    for w, parent in roots(models, others):
        if w == "hf":
            found = [parent] if (parent / "config.json").is_file() else []
        else:
            found = _walk(parent, SKIP_CACHE if w == "cache" else ())
        for p in found:
            c = read_folder(w, p)
            if c:
                out.append(c)
    try:
        machine.CACHE.mkdir(parents=True, exist_ok=True)
        cache_path().write_text(json.dumps({"at": time.time(), "roots": key, "folders": out}))
    except OSError:
        pass
    return out


def invalidate():
    """Forget the scan, so the next list looks at the disks again."""
    cache_path().unlink(missing_ok=True)


# ---------------------------------------------------------------- matching a catalog entry

def find(copies, e):
    """The folder among `copies` that holds catalog entry e, or None, preferring one that runs as it is (a packed MoE, any dense model), then its own repo's, then the nearest to the launcher's folder."""
    f = e.get("facts") or {}
    best = None
    for c in copies:
        if not matches(c["sig"], f, c["size"]):
            continue
        if f.get("format") == "exl3" and not c["packed"]:
            continue                     # the experts' own repo holds no dense part to run
        rank = (c["packed"] or f.get("kind") != "moe",
                bool(c["repo"] and c["repo"] == e.get("repo")) or (bool(c["id"]) and c["id"] == e.get("id")),
                WHERE_ORDER.get(c["where"], 0))
        if best is None or rank > best[0]:
            best = (rank, c)
    return best[1] if best else None


def adopt(e, models, copies=None):
    """Entry e pointed at the copy of it found elsewhere: it then reads as on disk and starts without
    a download (a raw MoE is packed in place first). e itself when it is already in the launcher's
    folder, already points at a folder, or nothing matches."""
    if e.get("path") or not e.get("facts") or (Path(models) / e["id"] / "launcher.json").exists():
        return e
    c = find(scan(models) if copies is None else copies, e)
    if c is None:
        return e
    if e["facts"].get("kind") == "moe" and not c["packed"]:
        return {**e, "path": c["path"], "needs_pack": True, "download": 0}
    return {**e, "path": c["path"], "size": c["size"], "download": 0, "disk": 0, "transient": 0}


def short_path(path):
    """A path with the home folder written as ~."""
    p, home = str(path), str(HOME)
    return "~" + p[len(home):] if p.startswith(home + os.sep) else p


def inside(path, folder):
    """Whether path is a folder inside `folder` (not the folder itself)."""
    if not path:
        return False
    try:
        p, f = Path(path).resolve(), Path(folder).expanduser().resolve()
    except OSError:
        return False
    return p != f and f in p.parents


# ---------------------------------------------------------------- measured speed

KEEP = 20            # replies the median is taken over
MIN_TOKENS = 64      # a shorter reply says little about the decode speed
RUNS = 20            # runs the per-machine correction is taken over
_FP = []             # this machine's fingerprint, read once


def fingerprint():
    if not _FP:
        _FP.append(machine.fingerprint())
    return _FP[0]


def measured_path():
    return Path(machine.CACHE) / "measured.json"


def load_measured():
    """{machine fingerprint: {model folder: record}} from measured.json, {} when there is none."""
    try:
        data = json.loads(measured_path().read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def estimate_path():
    """Where the estimate the launcher made before each run is kept until the run is measured."""
    return Path(machine.CACHE) / "estimate.json"


def load_notes():
    """{machine fingerprint: {model folder: note}} from estimate.json, {} when there is none."""
    try:
        data = json.loads(estimate_path().read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def note_estimate(model_path, est, recipe="", cond=None, fp=None):
    """Remember the estimate (machine.estimate's dict) the launcher made for the run it is starting
    on model_path, so the run's measured speeds can be compared with it."""
    if not model_path or not est or not est.get("tps"):
        return None
    p = Path(str(model_path))
    note = {"at": round(time.time(), 3), "model": p.name, "recipe": recipe or "",
            "est": round(est["tps"], 2), "raw": round(est.get("raw_tps") or est["tps"], 2),
            "lo": round(est.get("lo_tps") or est["tps"], 2), "hi": round(est.get("hi_tps") or est["tps"], 2),
            "cond": cond or {}}
    data = load_notes()
    data.setdefault(fp or fingerprint(), {})[folder_key(p, signature(p))] = note
    tmp = estimate_path().with_suffix(".tmp")
    machine.CACHE.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, estimate_path())
    return note


def note_for(key, fp=None):
    """The estimate noted for model folder key on this machine, {} when there is none."""
    return (load_notes().get(fp or fingerprint()) or {}).get(key) or {}


def folder_key(path, sig):
    """A model folder's name and, when it has a config, what that config says it is."""
    name = Path(str(path)).name
    if not sig:
        return name
    return "{}@{}/{}/{}/{}/{}{}".format(name, sig["model_type"], sig["layers"], sig["hidden"], sig["experts"],
                                        sig["bits"], "x" if sig["exl3"] else "")


def record(model_path, tokens, decode_s, fp=None, today=None):
    """Note one reply's decode speed, (tokens - 1) / decode_s, in the record of the model folder at model_path (the last KEEP speeds, their median, their count and the date) when it had at least MIN_TOKENS tokens, else None; one run entry holds the estimate its start was made from and the conditions then."""
    if not model_path or not tokens or tokens < MIN_TOKENS or not decode_s or decode_s <= 0:
        return None
    p = Path(str(model_path))
    sig = signature(p)
    key = folder_key(p, sig)
    data = load_measured()
    rec = data.setdefault(fp or fingerprint(), {}).setdefault(key, {
        "name": p.name, "sig": sig, "last": [],
        "size": sum(f.stat().st_size for f in os.scandir(p) if f.is_file()) if p.is_dir() else 0})
    tps = round((tokens - 1) / decode_s, 2)
    rec["last"] = (rec["last"] + [tps])[-KEEP:]
    note = note_for(key, fp)
    runs = rec.get("runs") or []
    if note.get("at") and (not runs or runs[-1].get("at") != note["at"]):
        runs.append({"at": note["at"], "recipe": note.get("recipe", ""), "est": note.get("est"),
                     "raw": note.get("raw"), "tps": tps, "cond": note.get("cond") or {}})
        rec["runs"] = runs[-RUNS:]
    rec.update(tps=round(statistics.median(rec["last"]), 2), n=len(rec["last"]),
               date=today or time.strftime("%Y-%m-%d"))
    if note:
        rec["est"] = note.get("est")
    tmp = measured_path().with_suffix(".tmp")
    machine.CACHE.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, measured_path())
    return rec


def correction(recipe, fp=None, data=None):
    """How this machine's measured runs compare with the raw estimates they were started on, as the
    median ratio over its last RUNS runs of that recipe, bounded to 0.3-1.5x, once there are 2 of
    them; {} when it has too few."""
    recs = ((load_measured() if data is None else data).get(fp or fingerprint()) or {}).values()
    runs = [r for rec in recs for r in rec.get("runs") or []
            if r.get("recipe") == recipe and r.get("tps") and (r.get("raw") or r.get("est"))]
    runs = sorted(runs, key=lambda r: r.get("at") or 0)[-RUNS:]
    if len(runs) < 2:
        return {}
    f = statistics.median([r["tps"] / (r.get("raw") or r["est"]) for r in runs])
    return {"factor": round(min(1.5, max(0.3, f)), 3), "runs": len(runs)}


def lookup(facts, fp=None, data=None):
    """The speed measured on this machine for the model with these catalog facts, whether or not
    its folder is still there: {"tps", "n", "date", "name"} of the newest matching record, or {}."""
    recs = ((load_measured() if data is None else data).get(fp or fingerprint()) or {}).values()
    hits = [r for r in recs if r.get("sig") and r.get("tps") and matches(r["sig"], facts, r.get("size", 0))]
    if not hits:
        return {}
    r = max(hits, key=lambda r: (r.get("date", ""), r.get("n", 0)))
    return {"tps": r["tps"], "n": r["n"], "date": r["date"], "name": r["name"]}
