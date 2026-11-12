#!/usr/bin/env python3
"""This machine as the engine sees it: one profile of facts and measured speeds, cached once and
turned into engine flags and a speed estimate for a model."""
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

CACHE = Path.home() / ".cache" / "moe-ssd-offload"      # rowqmm.LAYOUT_DIR, the engine's cache
GB = 1e9
# The share of RAM a streamed MoE's prefill is held to by default, the same as moe_stream.PREFILL_PCT.
PREFILL_PCT = 35
# Room kept free past what the engine plans, the swap-line margin on Apple and the GPU's free
# floor on CUDA.
GUARD_GB = 1.0
# Apple: GB of swap-line margin that each GB of the engine's footprint costs, because its pages
# push others' into the compressor.
MARGIN_PER_GB = 1.4
# Memory in GB a streamed MoE holds besides its dense part and its expert store: the K/V cache,
# buffers and read rings.
OVERHEAD_GB = {"metal": 1.26, "cuda": 5.23}
# The adaptive arena's floor and ceiling in slots a layer (server.py --arena-min, --arena-max).
ARENA_MIN = 21
ARENA_MAX = 60
# The engine's own memory policy defaults (engine_args.py): the process budget, the machine's reserve
# and the runtime overhead a budget cannot see.
MEM_PCT = 90
RESERVE_PCT = 15
HOST_GB = 0.6
# The MemAvailable the resident tier leaves to other programs (stream_fast.RESIDENT_FLOOR).
RESIDENT_FLOOR_GB = 5.0
# Seconds a conditions reading stays good for, since /status is polled by every open page.
COND_TTL = 5.0
# Other apps using this many CPU cores, or this share of the GPU, are named beside the speed.
# Display only: no engine decision reads these.
SLOW_CORES = 1.0
SLOW_GPU = 25.0
# A Linux thermal zone at or over this many degrees is named as a slowdown, again display only.
SLOW_C = 80.0


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _mlx():
    """(mlx module or None, backend, GPU facts) without failing where MLX is missing."""
    try:
        import mlx.core as mx
    except ImportError:
        return None, "none", {}
    if mx.metal.is_available():
        d = mx.device_info()
        return mx, "metal", {"gpu": d.get("device_name"), "gpu_mem": d.get("memory_size"),
                             "working_set": d.get("max_recommended_working_set_size")}
    cu = getattr(mx, "cuda", None)
    if cu is not None and cu.is_available():
        d = mx.device_info()
        return mx, "cuda", {"gpu": d.get("device_name"), "gpu_mem": d.get("total_memory")}
    return mx, "cpu", {}


def facts():
    """What the machine is: cheap to read, no measuring."""
    mx, backend, gpu = _mlx()
    if sys.platform == "darwin":
        chip = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        ram = int(_run(["sysctl", "-n", "hw.memsize"]) or 0)
        osv = "macOS " + platform.mac_ver()[0]
    else:
        chip = next((ln.split(":", 1)[1].strip() for ln in _cpuinfo() if ln.startswith("model name")), "")
        ram = _meminfo().get("MemTotal", 0)
        osv = f"{platform.system()} {platform.release()}"
    try:
        from importlib.metadata import version
        mlx_v, lm_v = version("mlx"), version("mlx-lm")
    except Exception:
        mlx_v = lm_v = None
    return {"os": osv, "arch": platform.machine(), "chip": chip, "ram": ram,
            "backend": backend, "mlx": mlx_v, "mlx_lm": lm_v, "cpus": os.cpu_count(), **gpu}


def _cpuinfo():
    try:
        return open("/proc/cpuinfo").read().splitlines()
    except OSError:
        return []


def _meminfo():
    try:
        with open("/proc/meminfo") as f:
            return {k: int(v.split()[0]) * 1024 for k, v in (ln.split(":", 1) for ln in f)}
    except OSError:
        return {}


def _key(f):
    k = {x: f.get(x) for x in ("os", "arch", "chip", "ram", "backend", "gpu", "gpu_mem", "mlx")}
    return hashlib.sha1(json.dumps(k, sort_keys=True).encode()).hexdigest()[:12]


KEY = []             # set by a process that must not open the GPU (launch.py --serve): reading facts imports MLX


def fingerprint():
    """The key this machine's profile is cached under: hardware, OS and MLX version."""
    return KEY[0] if KEY else _key(facts())


def profile(refresh=False, measure=True):
    """Facts + measured speeds (cached per key) + live readings under "now"."""
    f = facts()
    path = CACHE / f"machine-{_key(f)}.json"
    p = None
    if not refresh and path.exists():
        try:
            p = json.loads(path.read_text())
        except ValueError:
            p = None
    if p is None:
        p = {**f, "key": _key(f), "measured": time.strftime("%Y-%m-%d %H:%M"), "ssd": {}}
        _save(p)
    if measure and "gpu_bw" not in p:
        p["gpu_bw"] = gpu_bandwidth()
        _save(p)
    p["now"] = now(p)
    return p


def _save(p):
    CACHE.mkdir(parents=True, exist_ok=True)
    keep = {k: v for k, v in p.items() if k != "now"}
    (CACHE / f"machine-{p['key']}.json").write_text(json.dumps(keep, indent=1))


def now(p):
    """Live readings of the GB this process can take before the machine swaps, the free VRAM, the
    disks, and what else changes the speed now."""
    out = {"disks": disks(), "conditions": conditions()}
    if sys.platform == "darwin":
        out["avail_gb"] = _apple_margin_gb()
    else:
        out["avail_gb"] = _meminfo().get("MemAvailable", 0) / GB
    if p.get("backend") == "cuda":
        mx, _, _ = _mlx()
        d = mx.device_info()
        out["vram_free_gb"] = d.get("free_memory", 0) / GB
    return out


def _apple_margin_gb():
    """The margin in GB before macOS starts to swap, from its free-memory level and the compressor's
    pages."""
    try:
        import stream_fast as sf
    except Exception:
        return None
    level = sf._sysctl_int("kern.memorystatus_level")
    total = (sf._sysctl_int("hw.memsize", width=8) or 0) / GB
    vm = sf._vm_counts()
    if level is None or vm is None:
        return None
    return total * level / 100.0 - sf._SWAP_SHARE * vm[0]


# ---------------------------------------------------------------- conditions

_COND = {}                   # {"at": monotonic, "got": dict}: the last reading, shared by a process


def conditions(ttl=COND_TTL):
    """What changes the decode speed right now (power, Low Power Mode, thermal limit, free memory, the
    CPU other apps use, the GPU), read without sudo and reused for `ttl` seconds."""
    t = time.monotonic()
    if _COND and t - _COND.get("at", 0.0) < ttl:
        return dict(_COND["got"])
    got = {"at": round(time.time())}
    try:
        got.update(_darwin_conditions() if sys.platform == "darwin" else _linux_conditions())
    except Exception:
        pass
    _COND.update(at=t, got=got)
    return dict(got)


def _darwin_conditions():
    """The same reading from pmset, vm_stat, ps and ioreg."""
    out = {}
    batt = _run(["pmset", "-g", "batt"])
    out["power"] = "battery" if "Battery Power" in batt else "ac" if "AC Power" in batt else None
    out["low_power"] = any(ln.split()[:1] == ["lowpowermode"] and ln.split()[-1:] == ["1"]
                           for ln in _run(["pmset", "-g"]).splitlines())
    out["thermal"] = _thermal_limit(_run(["pmset", "-g", "therm"]))
    out["mem_free_gb"] = _darwin_free_gb()
    out["cpu_busy"], out["top"] = _cpu_busy()
    out["gpu_busy"] = _gpu_busy()
    return out


def _linux_conditions():
    """The same reading from /sys, /proc/meminfo, ps and nvidia-smi."""
    out = {}
    bats = list(Path("/sys/class/power_supply").glob("BAT*")) if Path("/sys/class/power_supply").is_dir() else []
    status = " ".join(_text(b / "status") for b in bats)
    acs = list(Path("/sys/class/power_supply").glob("A[CD]*")) if Path("/sys/class/power_supply").is_dir() else []
    out["power"] = "battery" if "Discharging" in status else \
        "ac" if "Charging" in status or "Full" in status or any(_text(a / "online") == "1" for a in acs) else None
    out["low_power"] = _text(Path("/sys/firmware/acpi/platform_profile")) == "low-power"
    temps = []
    for z in (Path("/sys/class/thermal").glob("thermal_zone*") if Path("/sys/class/thermal").is_dir() else []):
        try:
            temps.append(int(_text(z / "temp")) / 1000.0)
        except ValueError:
            pass
    out["thermal"] = f"{max(temps):.0f} C, hot" if temps and max(temps) >= SLOW_C else None
    out["mem_free_gb"] = round(_meminfo().get("MemAvailable", 0) / GB, 2) or None
    out["cpu_busy"], out["top"] = _cpu_busy()
    out["gpu_busy"] = _nvidia_busy()
    return out


def _text(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return ""


def _darwin_free_gb():
    """GB macOS counts as free, from vm_stat's free and speculative pages."""
    txt = _run(["vm_stat"])
    page = re.search(r"page size of (\d+) bytes", txt)
    free = re.search(r"Pages free:\s+(\d+)", txt)
    if not (page and free):
        return None
    spec = re.search(r"Pages speculative:\s+(\d+)", txt)
    return round((int(free[1]) + (int(spec[1]) if spec else 0)) * int(page[1]) / GB, 2)


def _thermal_limit(txt):
    """The percent pmset -g therm reports for the CPU, or None while it reports none."""
    vals = {}
    for ln in txt.splitlines():
        parts = ln.replace("=", " ").split()
        if len(parts) >= 2 and parts[-1].isdigit():
            vals[parts[0].lower()] = int(parts[-1])
    lim = min([v for k, v in vals.items() if "limit" in k] or [100])
    return None if lim >= 100 else f"the CPU is held to {lim}%"


def _cpu_busy():
    """(cores, the heaviest three) for the processes outside this process group, so a model this
    launcher started is not counted against the machine."""
    out, mine, rows = _run(["ps", "-axo", "pgid=,pcpu=,comm="]), os.getpgid(0), []
    for ln in out.splitlines():
        parts = ln.split(None, 2)
        if len(parts) < 3:
            continue
        try:
            pgid, cpu = int(parts[0]), float(parts[1])
        except ValueError:
            continue
        if pgid == mine or cpu <= 0:
            continue
        rows.append((cpu, os.path.basename(parts[2]) or parts[2]))
    rows.sort(reverse=True)
    return (round(sum(c for c, _ in rows) / 100.0, 2),
            [{"name": n[:24], "cpu": round(c / 100.0, 2)} for c, n in rows[:3]])


def _gpu_busy():
    """Percent of the GPU the system reports in use by others, from IOAccelerator, or None."""
    txt = _run(["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"])
    vals = [int(v) for v in re.findall(r'"Device Utilization %"\s*=\s*(\d+)', txt)]
    return float(max(vals)) if vals else None


def _nvidia_busy():
    """Percent of the GPU in use, from nvidia-smi, or None without one."""
    txt = _run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"])
    vals = [float(v) for v in re.findall(r"^\s*(\d+(?:\.\d+)?)\s*$", txt, re.M)]
    return max(vals) if vals else None


def conditions_text(c, need_gb=None):
    """The conditions as one line for a person, "on battery · other apps use 2.1 CPU cores (cmux) ·
    1.3 GB free: expect slower", with `need_gb` the memory the speed estimate counted on."""
    c = c or {}
    parts, slow = [], False
    if c.get("power") == "battery":
        parts.append("on battery")
        slow = True
    if c.get("low_power"):
        parts.append("Low Power Mode")
        slow = True
    if c.get("thermal"):
        parts.append(str(c["thermal"]))
        slow = True
    busy = c.get("cpu_busy") or 0.0
    if busy >= SLOW_CORES:
        top = (c.get("top") or [{}])[0].get("name")
        parts.append(f"other apps use {busy:.1f} CPU cores" + (f" ({top})" if top else ""))
        slow = True
    free = c.get("mem_free_gb")
    if free is not None and need_gb is not None and free < need_gb:
        parts.append(f"{free:.1f} GB free")
        slow = True
    gpu = c.get("gpu_busy")
    if gpu is not None and gpu >= SLOW_GPU:
        parts.append(f"other apps use the GPU ({gpu:.0f}%)")
        slow = True
    text = " · ".join(parts)
    return text + (": expect slower" if text and slow else "")


def disks():
    """Volumes a model can go on: mount, free and total bytes (the home volume first)."""
    cands = [Path.home()]
    if sys.platform == "darwin":
        cands += sorted(Path("/Volumes").glob("*"))
    else:
        try:
            for ln in open("/proc/mounts"):
                dev, mnt = ln.split()[:2]
                if dev.startswith("/dev/") and not mnt.startswith(("/boot", "/snap", "/efi")):
                    cands.append(Path(mnt.replace("\\040", " ")))
        except OSError:
            pass
    seen, out = set(), []
    for c in cands:
        try:
            st = os.stat(c)
            u = shutil.disk_usage(c)
        except OSError:
            continue
        if st.st_dev in seen or u.total < 8 * GB or not os.access(c, os.W_OK):
            continue
        seen.add(st.st_dev)
        out.append({"mount": str(c), "free": u.free, "total": u.total, "dev": st.st_dev})
    return out


def volume_of(path):
    """The st_dev of the nearest existing parent: which disk `path` is on."""
    p = Path(path).expanduser()
    while not p.exists():
        p = p.parent
    return os.stat(p).st_dev


def gpu_bandwidth(mb=256):
    """GB/s that a decode-shaped 4-bit matmul reads at, timed on the GPU over a chain of matrices
    bigger than any cache, or None without MLX."""
    mx, backend, _ = _mlx()
    if mx is None or backend == "cpu":
        return None
    k = 8192                                 # square, so one output feeds the next
    w = [mx.quantize(mx.random.normal((k, k)).astype(mx.float16), group_size=64, bits=4)
         for _ in range(max(1, int(mb * 1e6 // (k * k // 2))))]
    mx.eval(w)
    nbytes = sum(a.nbytes for t in w for a in t)
    x = mx.random.normal((1, k)).astype(mx.float16)

    def chain(x, reps):
        for _ in range(reps):
            for q, s, b in w:
                x = mx.quantized_matmul(x, q, s, b, transpose=True, group_size=64, bits=4)
                x = x * (1.0 / 64)           # keep it finite; one tiny op a step
        return x

    mx.eval(chain(x, 1))                     # warm-up: kernels built
    best = 0.0
    for _ in range(3):
        t = time.perf_counter()
        mx.eval(chain(x, 4))
        best = max(best, 4 * nbytes / (time.perf_counter() - t) / GB)
    del w
    mx.clear_cache()
    return round(best, 1)


def ssd(p, path, record=2 << 20, size=1 << 30, refresh=False, measure=True):
    """Read speed and per-read cost of the volume holding `path`, measured once on a scratch file
    and cached, or None when the disk lacks the room or the volume is not cached and `measure` is
    False."""
    vol = str(volume_of(path))
    if not refresh and vol in p.get("ssd", {}):
        return p["ssd"][vol]
    if not measure:
        return None
    d = Path(path).expanduser()
    while not d.exists():
        d = d.parent
    if shutil.disk_usage(d).free < 4 * size:
        return None
    import autotune
    scratch = d / f".moe-ssd-probe-{os.getpid()}"
    try:
        _write_uncached(scratch, size)
        gbps = autotune.probe_bandwidth(None, record, n_reads=size // record, binf=scratch)
        t = time.perf_counter()
        n1 = 24
        autotune.probe_bandwidth(None, record, threads=1, n_reads=n1, binf=scratch)
        per_read = (time.perf_counter() - t) / n1
    finally:
        scratch.unlink(missing_ok=True)
    # One read wave costs its fixed part plus bytes at the drive's rate.
    lat_ms = max(0.0, per_read - record / (gbps * GB)) * 1e3
    out = {"gbps": round(gbps, 2), "lat_ms": round(lat_ms, 3), "mount": str(d),
           "measured": time.strftime("%Y-%m-%d %H:%M")}
    p.setdefault("ssd", {})[vol] = out
    _save(p)
    return out


def _write_uncached(path, size):
    """A scratch file of random bytes written past the page cache, so the reads that follow come
    from the drive."""
    block = os.urandom(8 << 20)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        if sys.platform == "darwin":
            import fcntl
            fcntl.fcntl(fd, 48, 1)           # F_NOCACHE
        for _ in range(size // len(block)):
            os.write(fd, block)
        os.fsync(fd)
        if hasattr(os, "posix_fadvise"):
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


# ---------------------------------------------------------------- settings

def _cuda_home():
    """A CUDA toolkit with headers, for MLX's run-time kernel compiler (nvrtc)."""
    if os.environ.get("CUDA_HOME"):
        return None
    nvcc = shutil.which("nvcc")
    cands = [str(Path(nvcc).resolve().parent.parent)] if nvcc else []
    cands += ["/usr/local/cuda", "/opt/cuda"] + sorted(map(str, Path("/usr/local").glob("cuda-*")))
    return next((c for c in cands if Path(c, "include", "cuda.h").exists()), None)


def arena_slots(p, m):
    """(floor, start) slots stream_fast.make_governor picks at start: the smallest of --arena-max,
    the --mem-pct budget less what the model holds and the swap-line margin less --reserve-pct,
    with the floor at the hard 2*top_k when the budget cannot reach --arena-min."""
    slot_gb = m["expert_bytes"] * m["moe_layers"] / GB
    n_exp = max(1, m.get("experts") or 1)
    layers = max(1, m.get("moe_layers") or 1)
    top_k = m.get("top_k") or 1
    total = (p.get("ram") or 0) / GB
    held = m["dense_bytes"] / GB + OVERHEAD_GB.get(p.get("backend"), OVERHEAD_GB["metal"])
    budget = int(max(0.0, total * MEM_PCT / 100 - held - (n_exp * slot_gb / layers + HOST_GB)) / slot_gb)
    floor = max(2 * top_k, ARENA_MIN if budget >= ARENA_MIN else budget)
    caps = [min(ARENA_MAX, n_exp), budget]
    avail = p.get("now", {}).get("avail_gb")
    if avail is not None:
        caps.append(int(max(0.0, avail - total * RESERVE_PCT / 100) / slot_gb))
    return int(floor), int(max(floor, min(caps)))


def settings(p, m):
    """Engine flags for model facts `m` on profile `p`, a pure function of the two that returns
    {"args", "env", "store_gb", "store_lo_gb", "need_gb", "ok", "why"}."""
    why, args, env = [], [], {}
    backend, now = p.get("backend"), p.get("now", {})
    if backend not in ("metal", "cuda"):         # MLX's CPU device or no MLX: no plan, a plain answer
        why.append("no Apple GPU (Metal) or NVIDIA GPU (CUDA) found here; this engine runs on Apple Silicon "
                   "Macs and on Linux with an NVIDIA GPU")
        return {"args": args, "env": env, "why": why, "store_gb": 0.0, "store_lo_gb": 0.0,
                "need_gb": None, "ok": False, "biased": False}
    dense = m["dense_bytes"] / GB
    over = OVERHEAD_GB.get(backend, OVERHEAD_GB["metal"])
    if backend == "cuda":
        room = (p.get("gpu_mem") or 0) / GB - GUARD_GB       # bounds the whole device's use
        why.append(f"VRAM {(p.get('gpu_mem') or 0) / GB:.1f} GB, {GUARD_GB:g} GB kept free")
    else:
        avail = now.get("avail_gb")
        ws = (p.get("working_set") or p.get("ram") or 0) / GB
        room = ws if avail is None else min(ws, (avail - GUARD_GB) / MARGIN_PER_GB)
        why.append(f"GPU working set {ws:.1f} GB" + ("" if avail is None else
                   f", {avail:.1f} GB free before the machine would swap"))
    # The arena path runs at the server's default cache bias (0.5); elsewhere only a gated model gets it.
    out = {"args": args, "env": env, "why": why, "store_gb": 0.0, "store_lo_gb": 0.0,
           "need_gb": None, "ok": True,
           "biased": bool(m.get("gated")) or (m.get("recipe") == "arena" and backend != "cuda")}
    if m["kind"] == "resident":
        need = m["total_bytes"] / GB + GUARD_GB             # weights + room for the K/V cache
        out.update(ok=need <= room, need_gb=need)
        why.append(f"the whole model stays in memory: {need:.1f} GB needed, {room:.1f} GB usable now"
                   + ("" if out["ok"] else " (closing other apps frees more)"))
        return out

    experts_gb = m["expert_bytes"] * m["experts"] * m["moe_layers"] / GB
    room_left = max(0.0, room - dense - over)
    if backend == "cuda":
        g = math.floor(min(room, dense + experts_gb + over) * 2) / 2
        # The store gives VRAM back to other programs down to no pool at all and takes it again up to g, the fixed plan
        # (stream_fast.VramGovernor).
        lo = math.ceil((dense + over) * 2) / 2
        args += ["--mem-peak-gb", f"{lo:g}", "--mem-peak-max-gb", f"{g:g}"] if lo < g else ["--mem-peak-gb", f"{g:g}"]
        out["store_gb"] = max(0.0, min(experts_gb, g - dense - over))
        ch = _cuda_home()
        if ch:
            env["CUDA_HOME"] = ch
        if m.get("native") and shutil.which("g++"):
            env["MOE_NATIVE"] = "1"
        # Every expert held in the VRAM store and the native agent's tier when both together can.
        ram = (now.get("avail_gb") or 0.0) - HOST_GB - RESIDENT_FLOOR_GB
        if experts_gb <= g - dense + ram:
            env["MOE_RESIDENT"] = "1"
            why.append(f"all {experts_gb:.1f} GB of experts resident across VRAM and a pinned RAM tier ({ram:.1f} GB of RAM free)")
    elif m.get("recipe") == "arena":
        # The adaptive arena sizes itself from free memory while it runs, by the governor's own rule.
        args += ["--prefill-step", "0", "--keep-convs", "4"]
        if m.get("native_ok"):      # 1.25x decode, replies identical at bias 0
            env["MOE_NATIVE"] = "1"
            env["MOE_NATIVE_MTP"] = "1"     # MTP verify native too, 1.13x more, exact
        floor, start = arena_slots(p, m)
        slot_gb = m["expert_bytes"] * m["moe_layers"] / GB
        out["store_gb"] = start * slot_gb
        out["store_lo_gb"] = floor * slot_gb
    else:
        top = min(room, dense + experts_gb + over)
        floor = min(top, (p.get("ram") or 0) / GB * PREFILL_PCT / 100)
        g, n = math.floor(floor * 10) / 10, math.floor(top * 10) / 10
        args += ["--mem-peak-gb", f"{g:g}"]
        if n > g:
            args += ["--mem-peak-max-gb", f"{n:g}"]
        out["store_gb"] = max(0.0, min(experts_gb, n - dense - over))
        out["store_lo_gb"] = max(0.0, min(experts_gb, g - dense - over))
        if m.get("native") and m.get("native_ok"):
            env["MOE_NATIVE"] = "1"
    out["store_lo_gb"] = min(out["store_lo_gb"], out["store_gb"])
    out["need_gb"] = dense + over + out["store_gb"]
    if m.get("gated"):
        args += ["--cache-bias", "0.5", "--bias-budget", "0.2"]
        env["ENGINE_BIAS_GATED"] = "1"      # tells the engine this bias passed its gate, so it does not warn
        why.append("cache bias 0.5: passed its quality gate on this model")
    else:
        why.append("exact routing (cache bias 0): no quality gate run on this model")
    if out["store_gb"] <= 0:
        out["ok"] = False
        why.append(f"its fixed part ({dense:.1f} GB) leaves no memory for experts: {room:.1f} GB usable now"
                   " (closing other apps frees more)")
    return out


# ---------------------------------------------------------------- estimate

# Constants fitted to measured points: the share of the decode kernel's bandwidth a whole token
# gets, and the share of routed experts missing from memory.
ETA = {("metal", "moe"): 0.73, ("metal", "resident"): 0.99,
       ("cuda", "moe"): 0.36, ("cuda", "resident"): 0.77}
MISS = {True: (0.0123, 0.96), False: (0.076, 0.53)}


def bytes_per_weight(bits, group):
    """Affine MLX quantization: the codes plus a bf16 scale and bias per group (GPU time follows
    these bytes)."""
    return bits / 8 + (4 / group if group else 0)


def miss_share(cover, biased):
    if cover >= 1.0:
        return 0.0
    a, b = MISS[bool(biased)]
    return min(1.0, a * max(cover, 1e-4) ** -b)


def _decode(p, m, store_gb, disk, biased):
    """(tokens a second, miss share, GPU ms, SSD ms) with `store_gb` of experts held in memory."""
    bw = (p.get("gpu_bw") or 0) * GB
    routed = m.get("top_k", 0) * m.get("moe_layers", 0) * m.get("expert_bytes", 0)
    eta = ETA.get((p.get("backend"), m["kind"]), ETA[("metal", m["kind"])])
    gpu = (m["read_bytes"] + routed) / (bw * eta)
    read, miss = 0.0, 0.0
    if m["kind"] == "moe":
        held = store_gb * GB / max(1, m["experts"] * m["moe_layers"] * m["expert_bytes"])
        miss = miss_share(held, biased)
        if disk:
            layers = m["moe_layers"] * (1 - (1 - miss) ** m["top_k"])
            read = layers * disk["lat_ms"] / 1e3 + routed * miss / (disk["gbps"] * GB)
    return 1 / (gpu + read), miss, gpu * 1e3, read * 1e3


def correction(recipe):
    """This machine's measured runs against the estimates they were started on, as library's
    per-recipe median, or {} when it has too few runs of that recipe."""
    try:
        import library
        return library.correction(recipe)
    except Exception:
        return {}


def estimate(p, m, s, disk=None, corr=None):
    """Decode tokens a second for model facts `m` under settings `s`, from the GPU's bytes at the
    measured bandwidth plus the missed expert reads at the volume's measured `disk` speed, corrected
    by this machine's own measured runs of the same recipe; returns {"tps", "raw_tps", "lo_tps",
    "hi_tps", "factor", "runs", "gpu_ms", "read_ms", "miss"}."""
    if not (p.get("gpu_bw") or 0):
        return None
    biased = s.get("biased", "--cache-bias" in s.get("args", ()))
    start = s.get("store_gb", 0.0)
    raw, miss, gpu_ms, read_ms = _decode(p, m, start, disk, biased)
    lo = s.get("store_lo_gb")
    lo_raw = raw if lo is None or lo >= start else _decode(p, m, lo, disk, biased)[0]
    c = (correction(m.get("recipe") or "") if corr is None else corr) or {}
    f = c.get("factor") or 1.0
    return {"tps": raw * f, "raw_tps": raw, "lo_tps": lo_raw * f, "hi_tps": raw * f,
            "factor": f, "runs": c.get("runs", 0), "gpu_ms": gpu_ms, "read_ms": read_ms, "miss": miss}


if __name__ == "__main__":
    prof = profile(refresh="--refresh" in sys.argv)
    for d in prof["now"]["disks"]:
        ssd(prof, d["mount"], refresh="--refresh" in sys.argv)
    prof = profile()
    print(json.dumps(prof, indent=1, default=str))
