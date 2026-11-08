#!/usr/bin/env python3
"""First-run autotuner: measure what decides the exact speed settings (about 30 s), cache them per
machine and model, and let the launcher and server start from them. Exact settings only: nothing
here changes a model's output, and the lossy ones (cache bias, bias budget) are never chosen.

    python3 tuner.py --model <model directory> [--refresh] [--json]

Result: ~/.cache/moe-ssd-offload/autotune/<machine key>-<model>-v<TUNE_VERSION>.json
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

CACHE = Path.home() / ".cache" / "moe-ssd-offload" / "autotune"
# Bump when a knob is added, removed or decided differently: older files are then ignored.
TUNE_VERSION = 1
# The first run's time limit in seconds.
BUDGET_S = 120
# The queue depths of the disk curve.
QDS = (1, 2, 4, 8, 16, 24)
# Seconds of reading each curve point gets: a time slice, not a quality threshold.
POINT_S = 1.0
# The reader count the engine runs today (stream_fast._POOL), the one the curve is judged against.
ENGINE_READERS = 8
# Flags the autotuner may cache: exact ones only. Everything else in a launch plan is the
# user's or the launcher's call.
# The Mac footprint guard in GB: the abort line, so a cached ceiling sits one
# margin below it (the engine's own 1 GB margin rule, stream_fast._margin).
FOOTPRINT_GUARD_GB = 11.0
MARGIN_GB = 1.0
EXACT_FLAGS = {"--mem-peak-gb": float, "--mem-peak-max-gb": float, "--prefill-step": int,
               "--keep-convs": int}
# Flags that pin the memory model another way: a cached --mem-peak-gb must not join them.
MEMORY_FLAGS = ("--resident", "--adaptive", "--no-adaptive", "--mem-pct")

# Every exact knob the engine has: (name, flag or env, engine default, where it is read, what
# decides it, how the autotuner treats it).
KNOBS = [
    ("process memory ceiling (store)", "--mem-peak-gb / --mem-peak-max-gb (ceiling = min(live plan, guard - 1 GB))", "none",
     "engine_args.py, stream_fast budget", "RAM or VRAM total, dense part, expert bytes",
     "computed by machine.settings on an idle machine, cached"),
    ("arena slots (Metal)", "--arena-min / --arena-max, --adaptive", "21 / 60",
     "stream_fast.make_governor", "free memory while it runs", "self-sizing at start; not cached"),
    ("memory share / reserve", "--mem-pct, --reserve-pct, --host-gb", "90 / 15 / 0.6",
     "engine_args.py", "RAM", "defaults hold; not cached"),
    ("prefill grouping", "--prefill-step, MOE_LM_PREFILL", "per model",
     "server.py, stream_fast", "model kind (arena recipe: 0)", "recipe value cached"),
    ("conversations kept", "--keep-convs, --keep-gb", "1 / 2.0", "server.py",
     "memory left beside the pool", "recipe value cached (arena: 4)"),
    ("defer mode", "--defer-mode", "auto", "stream_fast", "sync/defer crossover, measured in-process",
     "self-measuring; not cached"),
    ("reader threads", "(none) stream_fast._POOL", "8", "stream_fast.py:2362",
     "disk queue-depth curve", "curve measured; needs a flag to apply (CLOSED: 8 optimal)"),
    ("O_DIRECT reads", "MOE_PIPE_DIRECT (CUDA agent)", "1", "native/moe_pipe_cuda.cpp",
     "direct vs buffered at the record size", "measured; env set only when buffered wins"),
    ("RAM victim tier", "MOE_PIPE_TIER_GB (CUDA agent)", "third of free RAM above 2.5 GB",
     "native/moe_pipe_cuda.py tier_slots", "MemAvailable at start",
     "engine sizes it live at each start; cached value would go stale, so recorded only"),
    ("native agent", "MOE_NATIVE, MOE_NATIVE_MTP", "0", "stream_fast", "backend, built library",
     "recipe value from machine.settings, cached"),
    ("K/V cache on host", "MOE_KV_HOST", "0", "stream_fast.py", "context length against free VRAM",
     "not tuned: depends on the request, not the machine"),
    ("staging readers", "MOE_STAGING_READERS", "default", "native/moe_native.cpp",
     "CLOSED (README 'B' arm 0.999x)", "not tuned"),
    ("stripe sidecar", "MOE_STRIPE", "auto", "stream_fast.py:3028", "a second drive present",
     "self-detecting; not tuned"),
]


def machine_key():
    import machine
    return machine.fingerprint()


def model_bytes(model):
    """The size of the file the disk probe reads, which also tells a changed model from the cached one."""
    f = Path(model).expanduser() / "experts.bin"
    return f.stat().st_size if f.exists() else 0


def cache_path(model, key=None):
    return CACHE / f"{key or machine_key()}-{Path(model).expanduser().resolve().name}-v{TUNE_VERSION}.json"


def load(model, key=None):
    """The cached tuning for this machine and model, or None when absent, stale or unreadable."""
    try:
        t = json.loads(cache_path(model, key).read_text())
    except (OSError, ValueError):
        return None
    return t if t.get("model_bytes") == model_bytes(model) else None


def record_bytes(model):
    """Bytes of one expert in one layer (all its projections), the unit the engine reads."""
    idx = json.loads((Path(model).expanduser() / "experts_index.json").read_text())
    per_layer = {}
    for name, meta in idx.items():
        i = name.split("layers.")[-1].split(".")[0]
        per_layer[i] = per_layer.get(i, 0) + meta["stride"]
    return int(sorted(per_layer.values())[len(per_layer) // 2])


# ------------------------------------------------------------------ decisions (pure)

def readers_from_curve(curve, noise):
    """The smallest queue depth whose rate is within the measurement noise of the best one."""
    peak = max(bw for _, bw in curve)
    return min(qd for qd, bw in curve if bw >= peak * (1 - noise))


def direct_env(direct_gbps, buffered_gbps, noise, direct_ok=True):
    """{"MOE_PIPE_DIRECT": "0"} only when O_DIRECT fails or buffered reads beat it by more than the noise."""
    if not direct_ok or buffered_gbps > direct_gbps * (1 + noise):
        return {"MOE_PIPE_DIRECT": "0"}
    return {}


def exact_args(args):
    """The exact flags of a launch plan as {flag: value}: the lossy and the per-run ones are dropped."""
    out = {}
    for i, a in enumerate(args):
        if a in EXACT_FLAGS and i + 1 < len(args):
            out[a] = EXACT_FLAGS[a](args[i + 1])
    return out


def cap_ceiling(args, backend):
    """On Metal, the growth ceiling held one margin below the footprint guard and dropped if that leaves no room above
    the floor; elsewhere the args as they are, since on CUDA the ceiling is the VRAM store's."""
    k = "--mem-peak-max-gb"
    if backend == "metal" and k in args:
        args[k] = min(args[k], FOOTPRINT_GUARD_GB - MARGIN_GB)
        if args[k] <= args.get("--mem-peak-gb", 0):
            del args[k]
    return args


def recipe_of(model, prof, given=None):
    """arena or store: the argument, else the launcher's note in the model folder, else store on CUDA (the only
    recipe there); None on Metal, where a wrong guess would swap a hand-set adaptive arena for a fixed store."""
    if given:
        return given
    try:
        return json.loads((Path(model).expanduser() / "launcher.json").read_text()).get("recipe")
    except (OSError, ValueError):
        return "store" if prof.get("backend") == "cuda" else None


def tier_gb(avail_bytes, floor=2.5e9):
    """What the CUDA agent's own rule (a third of MemAvailable above its floor) gives right now, for the record."""
    return max(0.0, avail_bytes - floor) / 3 / 1e9


# ------------------------------------------------------------------ measuring

def disk_curve(model, record, t_end):
    """([(qd, GB/s)], noise): random record-sized O_DIRECT reads of experts.bin at each queue depth,
    each point the mean of two runs of POINT_S, and noise the mean spread between the two runs."""
    import autotune
    bin_ = Path(model).expanduser() / "experts.bin"

    def once(qd, n):
        return autotune.probe_bandwidth(None, record, threads=qd, n_reads=n, binf=bin_)

    curve, spread = [], []
    for qd in QDS:
        if time.time() > t_end:
            break
        # A short run first fixes how many reads fill the time slice.
        n = max(16, int(once(qd, max(16, 4 * qd)) * 1e9 * POINT_S / record))
        a, b = once(qd, n), once(qd, n)
        curve.append((qd, round((a + b) / 2, 3)))
        spread.append(abs(a - b) / max(a, b))
    return curve, sum(spread) / len(spread)


def pack_rates(binf, record):
    """(QD1 GB/s, QD8 GB/s): random record-sized direct reads of one file, the autotuner's own prober."""
    import autotune
    return (autotune.probe_bandwidth(None, record, threads=1, n_reads=300, binf=binf),
            autotune.probe_bandwidth(None, record, threads=8, n_reads=1500, binf=binf))


def measure_pack(model, record=None):
    """{"rates": [QD1, QD8] GB/s, "noise"}: the mean of two runs of pack_rates, noise the larger run-to-run spread."""
    real = (Path(model).expanduser() / "experts.bin").resolve()
    runs = [pack_rates(real, record or record_bytes(model)) for _ in range(2)]
    noise = max(abs(runs[0][i] - runs[1][i]) / max(runs[0][i], runs[1][i]) for i in (0, 1))
    return {"rates": [round((runs[0][i] + runs[1][i]) / 2, 3) for i in (0, 1)], "noise": round(noise, 4),
            "measured": time.time(), "runs": runs}


def baseline_path(model):
    """The baseline file of one pack, keyed by the resolved path, size and inode (a rewrite in place keeps all three)."""
    import hashlib
    real = (Path(model).expanduser() / "experts.bin").resolve()
    st = real.stat()
    key = f"{real}|{st.st_size}|{st.st_ino}"
    return CACHE / f"baseline-{hashlib.sha1(key.encode()).hexdigest()[:12]}.json", key


def record_baseline(model, source, record=None):
    """Store the pack's own rate as the reference later checks compare against. source: rewrite, written or first_tune."""
    path, key = baseline_path(model)
    m = measure_pack(model, record)
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"key": key, "base": dict(m, source=source)}, indent=1))
    return m


# Known limit: a fixed day between re-measures (about 4 s each); upgrade to a drive-wear-driven interval if a day proves too slow or too often.
CHECK_EVERY_S = 86400


def aged_verdict(base, cur):
    """Aged when the current rate is below the baseline by more than the larger of the two noises, at QD1 and QD8."""
    noise = max(base["noise"], cur["noise"])
    gap = [1 - cur["rates"][i] / base["rates"][i] for i in (0, 1)]
    return {"aged": all(g > noise for g in gap), "gap_qd1": round(gap[0], 4), "gap_qd8": round(gap[1], 4),
            "noise": round(noise, 4), "rates_gbps": cur["rates"], "baseline_gbps": base["rates"],
            "baseline_source": base.get("source")}


def pack_age(model, force=False, record=None):
    """The pack's verdict against its stored baseline; the first call stores the baseline (first_tune) and says not
    aged. Re-measures at most once a day unless forced; the last verdict is kept in the baseline file."""
    path, key = baseline_path(model)
    try:
        st = json.loads(path.read_text())
    except (OSError, ValueError):
        st = None
    if st is None or st.get("key") != key:
        m = record_baseline(model, "first_tune", record)
        return {"aged": False, "new_baseline": True, "rates_gbps": m["rates"], "baseline_source": "first_tune",
                "noise": m["noise"], "size_bytes": int(key.split("|")[1])}
    last = st.get("check")
    if last and not force and time.time() - last["measured"] < CHECK_EVERY_S:
        return last
    out = aged_verdict(st["base"], measure_pack(model, record))
    out.update(measured=time.time(), size_bytes=int(key.split("|")[1]), asked=bool(last and last.get("asked") and out["aged"]))
    st["check"] = out
    path.write_text(json.dumps(st, indent=1))
    return out


def mark_asked(model):
    path, _ = baseline_path(model)
    st = json.loads(path.read_text())
    st["check"]["asked"] = True
    path.write_text(json.dumps(st, indent=1))


def direct_vs_buffered(model, record):
    """(O_DIRECT GB/s or None if it fails, buffered GB/s) at the engine's reader count, Linux only."""
    import autotune
    bin_ = Path(model).expanduser() / "experts.bin"
    try:
        d = autotune.probe_bandwidth(None, record, threads=ENGINE_READERS, n_reads=600, binf=bin_)
    except OSError:
        d = None
    b = autotune.probe_bandwidth(None, record, threads=ENGINE_READERS, n_reads=600, binf=bin_, direct=False)
    return d, b


def tune(model, refresh=False, lp=None, recipe=None):
    """Measure and cache; return the result dict. `lp`: the launch module when the caller is it."""
    model = str(Path(model).expanduser())
    if not refresh and (t := load(model)):
        return t
    t0 = time.time()
    import machine
    prof = machine.profile()
    key = prof["key"]
    out = {"version": TUNE_VERSION, "machine": key, "chip": prof.get("chip"), "gpu": prof.get("gpu"),
           "backend": prof.get("backend"), "ram_gb": round((prof.get("ram") or 0) / 1e9, 1),
           "model": Path(model).name, "model_bytes": model_bytes(model),
           "measured": time.strftime("%Y-%m-%d %H:%M"), "notes": [], "needs_plumbing": []}
    out["vram_gb"] = round((prof.get("gpu_mem") or 0) / 1e9, 1) if prof.get("backend") == "cuda" else None
    env, args = {}, {}
    if (Path(model) / "experts.bin").exists():
        rec = record_bytes(model)
        curve, noise = disk_curve(model, rec, t0 + BUDGET_S * 0.6)
        best = readers_from_curve(curve, noise)
        bw = dict(curve)
        out["disk"] = {"record_bytes": rec, "curve_gbps": curve, "noise": round(noise, 4),
                       "peak_gbps": max(bw.values()), "readers_best": best, "readers_engine": ENGINE_READERS}
        gain = bw.get(best, 0) / bw[ENGINE_READERS] - 1 if ENGINE_READERS in bw else 0
        if best != ENGINE_READERS and gain > noise:
            out["needs_plumbing"].append(
                f"reader threads: {best} readers read {gain:+.1%} faster than the engine's fixed "
                f"{ENGINE_READERS} (stream_fast._POOL has no flag)")
        try:
            pack_age(model, record=rec)         # the first run stores the pack's baseline
        except Exception as ex:                 # a failed side check never blocks the tuning
            out["notes"].append(f"pack age check skipped: {ex}")
        if sys.platform.startswith("linux") and prof.get("backend") == "cuda":
            d, b = direct_vs_buffered(model, rec)
            out["direct"] = {"direct_gbps": d and round(d, 3), "buffered_gbps": round(b, 3)}
            env.update(direct_env(d or 0, b, noise, direct_ok=d is not None))
            avail = prof["now"].get("avail_gb", 0) * 1e9
            out["tier"] = {"mem_available_gb": round(avail / 1e9, 2), "engine_gb_now": round(tier_gb(avail), 2)}
    else:
        out["notes"].append("no experts.bin: nothing streams from disk, no read-path knob applies")
    # Memory and recipe flags, computed for an idle machine so a busy first run does not freeze a low value.
    try:
        lp = lp or __import__("launch")
        e = lp.local_entry(model)
        e.setdefault("native", True)       # the native reader runs any packed MoE; the engine falls back itself
        e["recipe"] = recipe_of(model, prof, recipe)
        if e["recipe"] is None:
            raise ValueError("memory recipe unknown (arena or store): pass --recipe, or start it once with ./larder")
        s, _ = lp.engine_plan(e, prof, None, tight=True)
        args = cap_ceiling(exact_args(s["args"]), prof.get("backend"))
        env.update({k: v for k, v in s["env"].items() if k.startswith("MOE_NATIVE") or k == "CUDA_HOME"})
    except Exception as ex:                     # a model the launcher cannot plan still gets its disk curve
        out["notes"].append(f"memory plan skipped: {ex}")
    out["args"], out["env"] = args, env
    out["took_s"] = round(time.time() - t0, 1)
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_path(model, key).write_text(json.dumps(out, indent=1))
    return out


# ------------------------------------------------------------------ applying

def opted_in(argv, environ=os.environ):
    """Cached settings apply only to a start the launcher marks (MOE_TUNED=1 or --tuned); a raw command is untouched."""
    return environ.get("MOE_TUNED") == "1" or "--tuned" in argv


def apply_env(model, environ=os.environ):
    """Set the cached environment knobs the user has not set; call before the engine modules import."""
    t = load(model)
    for k, v in (t or {}).get("env", {}).items():
        environ.setdefault(k, str(v))
    return t


def parser_defaults(model, argv):
    """{dest: value} to hand to ArgumentParser.set_defaults: the cached flags, with the memory ceiling left
    out when the command line sets one or pins the memory model another way (a hand-tuned ceiling stays whole)."""
    t = load(model)
    if not t:
        return {}
    own = [a.split("=")[0] for a in argv]
    pinned = any(a in MEMORY_FLAGS + ("--mem-peak-gb", "--mem-peak-max-gb") for a in own)
    out = {}
    for flag, v in t.get("args", {}).items():
        if pinned and flag in ("--mem-peak-gb", "--mem-peak-max-gb"):
            continue
        out[flag.lstrip("-").replace("-", "_")] = v
    return out


def model_from_argv(argv):
    for i, a in enumerate(argv):
        if a == "--model" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--model="):
            return a.split("=", 1)[1]
    return None


def table():
    """The knob table as markdown."""
    rows = ["| knob | flag or env | default | read in | decided by | autotuner |", "|---|---|---|---|---|---|"]
    rows += ["| " + " | ".join(k) + " |" for k in KNOBS]
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--recipe", choices=("arena", "store"), help="memory recipe when the folder has no launcher.json")
    ap.add_argument("--refresh", action="store_true", help="measure again even if cached")
    ap.add_argument("--age", action="store_true", help="only check whether the pack reads slower than fresh data")
    ap.add_argument("--table", action="store_true", help="print the knob table and exit")
    a = ap.parse_args()
    if a.table:
        return print(table())
    if a.age:
        r = pack_age(a.model, force=True)
        r.pop("runs", None)
        return print(json.dumps(r, indent=1))
    t = tune(a.model, a.refresh, recipe=a.recipe)
    print(json.dumps(t, indent=1))
    print(f"\ncached at {cache_path(a.model, t['machine'])} ({t.get('took_s', 'cached')} s)")


if __name__ == "__main__":
    main()
