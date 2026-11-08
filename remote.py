#!/usr/bin/env python3
"""The other computers the launcher menu can run a model on, from a preset script, ~/.ssh/config and Tailscale, with the ssh commands that list and start models there."""
from __future__ import annotations

import glob
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

import engine_version
import tui

HERE = Path(__file__).resolve().parent
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=6"]
SSH_START = ["ssh", "-tt", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=10"]
INSTALL = ("git clone https://github.com/IamCooller/larder ~/larder && "
           "cd ~/larder && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt")
# Name, hardware line, measured speed and running-model name of each preset, and more hosts to leave out, from a private module.
try:
    import remote_presets as _presets
except ImportError:
    _presets = None
HARDWARE = getattr(_presets, "HARDWARE", {})
MEASURED = getattr(_presets, "MEASURED", {})
PRESET_NAME = getattr(_presets, "PRESET_NAME", {})
MODEL_NAME = getattr(_presets, "MODEL_NAME", {})
GIT_HOSTS = ("github.com", "gitlab.com", "bitbucket.org", "dev.azure.com", "visualstudio.com",
             *getattr(_presets, "SKIP_HOSTS", ()))
# An optional preset script beside the engine that describes and starts preset machines; the public tree has none.
PRESET_SCRIPT = getattr(_presets, "PRESET_SCRIPT", "presets.sh")
# The characters an ssh destination may hold: a host, user@host, an IPv4 or [IPv6] address with a zone.
TARGET_RE = re.compile(r"^[A-Za-z0-9._@:%\[\]-]+$")
FLASH_ID = "flash-next-exl3"
GENERIC_DIR = "~/larder"
LIST_PRESET = "cd {dir} && {python} launch.py --list --json"
LIST_GENERIC = "cd ~/larder && ./larder --list --json"


def _run(argv, timeout=20):
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except (OSError, subprocess.SubprocessError) as err:
        return 255, "", str(err)


def _run_bytes(argv, timeout=60):
    """subprocess.run with bytes back, for the binary git bundle."""
    try:
        r = subprocess.run(argv, capture_output=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except (OSError, subprocess.SubprocessError) as err:
        return 255, b"", str(err).encode()


def _run_stdin(argv, data, timeout=60):
    """subprocess.run with bytes on stdin (the update bundle) and text back."""
    try:
        r = subprocess.run(argv, input=data, capture_output=True, timeout=timeout)
        return r.returncode, r.stdout.decode("utf-8", "replace"), r.stderr.decode("utf-8", "replace")
    except (OSError, subprocess.SubprocessError) as err:
        return 255, "", str(err)


def _shell_path(p):
    """A shell word for a path, with a leading ~ expanded rather than quoted."""
    if p.startswith("~/"):
        return '"$HOME/' + p[2:].replace('"', '\\"') + '"'
    return shlex.quote(p)


def safe_target(target):
    """The ssh destination when it is a plain host, user@host or address, else None, since a name like -oProxyCommand=... is an ssh option."""
    if target and TARGET_RE.fullmatch(target) and not re.search(r"(^|@)-", target):
        return target
    return None


BAD_TARGET = "not a plain ssh address (a host, user@host or an IP), so it is left out"


def no_address(dev):
    """Why dev cannot be reached over ssh, in one line."""
    return f"{dev.name}: {BAD_TARGET}" if dev.target else f"{dev.name} has no ssh address"


def _ssh(dev, command, base=SSH, forward=()):
    """The ssh argv for one command on dev, or None when dev has no address or one that is not a plain destination."""
    target = safe_target(dev.target)
    if target is None:
        return None
    return base + list(forward) + [target, command]


# ---------------------------------------------------------------- engine version handshake

def local_version():
    """This computer's own version object, as the device's is compared against."""
    return engine_version.version()


def version_state(remote_v, local_v=None):
    """"ok" when the device's engine speaks this menu's protocol, "older" when it is missing or
    behind, "newer" when the menu must be updated first."""
    local_v = local_v if local_v is not None else local_version()
    rp = (remote_v or {}).get("protocol")
    lp = (local_v or {}).get("protocol")
    if rp is None or rp < lp:
        return "older"
    if rp > lp:
        return "newer"
    return "ok"


def _version_cmd(dev):
    """The device's own launcher printing its version as one JSON line."""
    if dev.kind == "preset":
        return f"cd {dev.dir} && {dev.python} launch.py --version"
    return f"cd {GENERIC_DIR} && ./larder --version"


def version_argv(dev):
    """The ssh call that asks the device for its engine version alone."""
    return _ssh(dev, _version_cmd(dev))


def parse_version(text):
    """The version object from a --version answer, or None when the engine there has none."""
    for line in (text or "").splitlines():
        start = line.find("{")
        end = line.rfind("}")
        if start < 0 or end <= start:
            continue
        try:
            data = json.loads(line[start:end + 1])
        except ValueError:
            continue
        if isinstance(data, dict) and "protocol" in data:
            return data
    return None


# ---------------------------------------------------------------- ssh config

def _blocks(text):
    out, block = [], None
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        key, _, val = line.partition(" ")
        if not val:
            key, _, val = line.partition("=")
        key, val = key.strip().lower(), val.strip().strip('"')
        if key == "host":
            if block:
                out.append(block)
            block = {"aliases": val.split(), "user": "", "hostname": ""}
        elif block is not None and key in ("user", "hostname"):
            block[key] = val
    if block:
        out.append(block)
    return out


def parse_ssh_config(text):
    """Devices for the concrete hosts in an ssh config: one per Host block, its first alias as
    the ssh target, git services and wildcard blocks left out."""
    seen, out = set(), []
    for b in _blocks(text):
        aliases = [a for a in b["aliases"] if a and not re.search(r"[*?!]", a)]
        if not aliases or b["user"] == "git":
            continue
        host = b["hostname"].lower()
        if any(host == g or host.endswith("." + g) for g in GIT_HOSTS):
            continue
        name = aliases[0]
        if name.lower() in seen:
            continue
        seen.add(name.lower())
        names = tuple(dict.fromkeys(a.lower() for a in aliases + ([b["hostname"]] if b["hostname"] else [])))
        seen.update(names)
        user = b["user"] or ""
        target = f"{user}@{b['hostname'] or name}" if user and not b["hostname"] else name
        out.append(tui.Device(name=name, target=name, kind="ssh", source="~/.ssh/config", aliases=names,
                              detail=f"ssh {target}"))
    return out


def ssh_hosts(path=None):
    """parse_ssh_config over ~/.ssh/config, with its Include files read in place."""
    path = Path(path or Path.home() / ".ssh" / "config")
    texts, seen = [], set()

    def read(p):
        p = Path(p).expanduser()
        if p in seen or not p.is_file():
            return
        seen.add(p)
        try:
            text = p.read_text(errors="replace")
        except OSError:
            return
        texts.append(text)
        for line in text.splitlines():
            m = re.match(r"\s*include\s+(.+)", line, re.I)
            if not m:
                continue
            for pat in shlex.split(m.group(1)):
                cand = Path(pat).expanduser()
                if not cand.is_absolute():
                    cand = p.parent / cand
                for f in sorted(glob.glob(str(cand))):
                    read(f)

    read(path)
    return parse_ssh_config("\n".join(texts))


# ---------------------------------------------------------------- tailscale

def tailscale_devices(peers=None):
    """Devices for the peers Tailscale knows, reached at their Tailscale address since a peer picks its own name; online peers first."""
    import devices
    peers = peers if peers is not None else devices.tailscale_peers()
    out = []
    for p in peers:
        name = p.get("name") or p.get("ip") or "tailscale peer"
        out.append(tui.Device(name=name, target=p.get("ip") or name, kind="tailscale", source="tailscale",
                              aliases=tuple(x for x in (name.lower(), (p.get("ip") or "").lower()) if x),
                              hardware=p.get("os", ""),
                              status="checking" if p.get("online") else "offline",
                              detail="tailscale" + ("" if p.get("online") else ", offline")))
    return out


# ---------------------------------------------------------------- presets from the preset script

def parse_presets(text):
    """Devices for the preset script's --json: one per machine, its models keyed by preset name."""
    try:
        data = json.loads(text)
    except ValueError:
        return []
    out, by = [], {}
    for row in data.get("presets", []):
        kind = (row.get("kind") or "").lower()
        if not kind:
            continue
        d = by.get(kind)
        if d is None:
            host = row.get("host", "")
            d = tui.Device(name=PRESET_NAME.get(kind, kind), target=host, kind="preset",
                           preset=kind, source="preset script", dir=row.get("dir", ""),
                           python=row.get("python", ""), hardware=HARDWARE.get(kind, ""),
                           measured=MEASURED.get(kind, ()), status="checking",
                           aliases=tuple(x for x in (kind.lower(), host.lower(), host.split("@")[-1].lower()) if x),
                           detail=f"ssh {host}")
            by[kind] = d
            out.append(d)
        for key, m in (row.get("models") or {}).items():
            d.commands[key] = {"port": int(m.get("port") or 0), "id": m.get("id", ""),
                               "command": m.get("command", "")}
    return out


def presets(run=_run):
    """The presets the script describes, read from the script itself; none when this checkout has no script."""
    code, out, err = run(["bash", str(HERE / PRESET_SCRIPT), "--json"], timeout=10)
    return parse_presets(out) if code == 0 else []


# ---------------------------------------------------------------- the list

def _names(d):
    return {x for x in (d.name.lower(), d.target.lower(), *d.aliases) if x}


def device_list(config=None, peers=None, preset_rows=None):
    """Presets first, then ~/.ssh/config, then Tailscale, one row each."""
    if preset_rows is None:
        preset_rows = presets()
    if config is None:
        config = ssh_hosts()
    if peers is None:
        peers = tailscale_devices()
    out, seen = [], set()
    for d in list(preset_rows) + list(config) + list(peers):
        names = _names(d)
        if names & seen:
            continue
        seen |= names
        out.append(d)
    return out


# ---------------------------------------------------------------- one ssh call per device

PROBE = """d=@DIR@
if [ -f "$d/launch.py" ]; then echo ENGINE; else echo NOENGINE; fi
@VERSION@ 2>/dev/null | sed 's/^/VER /'
if [ "$(uname -s)" = Darwin ]; then
  echo "HW $(sysctl -n machdep.cpu.brand_string), $(( $(sysctl -n hw.memsize) / 1073741824 )) GB"
else
  cpu=$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2- | sed 's/^ *//')
  ram=$(awk '/MemTotal/ {printf "%d", $2/1048576}' /proc/meminfo 2>/dev/null)
  gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)
  echo "HW $cpu, $ram GB RAM, $gpu"
fi
if [ -f @LOCK@ ]; then
  owner=$(cat @LOCK@/owner 2>/dev/null)
  pid=$(echo "$owner" | awk '{print $NF}')
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then echo "RUN $owner"; fi
fi
f=~/.cache/moe-ssd-offload/server.json
if [ -f "$f" ]; then
  pid=$(sed -n 's/.*"pid": *\\([0-9]*\\).*/\\1/p' "$f" | head -1)
  name=$(sed -n 's/.*"name": *"\\([^"]*\\)".*/\\1/p' "$f" | head -1)
  port=$(sed -n 's/.*"port": *\\([0-9]*\\).*/\\1/p' "$f" | head -1)
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then echo "RUN2 $name $port"; fi
fi"""


def probe_argv(dev):
    """The ssh call that fills a device's status, hardware, running model and engine version."""
    script = PROBE.replace("@DIR@", _shell_path(dev.dir or GENERIC_DIR))
    script = script.replace("@VERSION@", _version_cmd(dev))
    script = script.replace("@LOCK@", _shell_path(f"/tmp/{dev.preset}_gpu.lock" if dev.preset else "/tmp/none"))
    return _ssh(dev, script)


def classify(code, out, err):
    """(status, hardware, running name, port, detail, version) for a probe's answer."""
    hardware, running, port = "", "", 0
    engine = False
    version = None
    for line in out.splitlines():
        if line == "ENGINE":
            engine = True
        elif line.startswith("HW "):
            hardware = line[3:].strip().strip(",")
        elif line.startswith("VER "):
            version = parse_version(line[3:])
        elif line.startswith("RUN "):
            parts = line.split()
            model = parts[-2] if len(parts) >= 2 else ""
            running = MODEL_NAME.get(model, model or "a model")
        elif line.startswith("RUN2 "):
            bits = line.split()
            running = " ".join(bits[1:-1]) or "a model"
            port = int(bits[-1]) if bits[-1].isdigit() else 0
    if code == 0:
        if not engine:
            return "no-engine", hardware, "", 0, "", version
        if running:
            return "running", hardware, running, port, "", version
        return "ready", hardware, "", 0, "", version
    text = (err or out).lower()
    if "permission denied" in text or "host key verification failed" in text or "authentication" in text:
        return "no-key", hardware, "", 0, "", version
    if any(w in text for w in ("could not resolve", "connection refused", "timed out", "no route to host",
                               "network is unreachable", "connection closed", "operation timed out")):
        return "offline", hardware, "", 0, "", version
    return "offline", hardware, "", 0, (err or out).strip().splitlines()[-1] if (err or out).strip() else "", version


def apply_probe(dev, code, out, err):
    """Fill a device from a probe's answer, with the fix, install or update line when it needs one."""
    status, hardware, running, port, detail, version = classify(code, out, err)
    if hardware:
        dev.hardware = hardware
    dev.status = status
    dev.running, dev.port = running, port
    dev.engine = version or {}
    dev.engine_state = ""
    target = dev.target or dev.name
    if status in ("ready", "running") and version_state(version) == "older":
        dev.status = "engine-old"
        dev.engine_state = "older"
    elif status in ("ready", "running") and version_state(version) == "newer":
        dev.status = "engine-new"
        dev.engine_state = "newer"
    if dev.status == "no-key":
        dev.detail = f"no ssh key: ssh-copy-id {target}"
    elif dev.status == "no-engine":
        dev.detail = f"engine not installed: {INSTALL}"
    elif dev.status == "engine-old":
        dev.detail = "engine is older: u updates it"
    elif dev.status == "engine-new":
        dev.detail = "update this computer's engine first"
    elif dev.status == "offline":
        dev.detail = detail or "no answer on ssh: check the network or the host"
    elif dev.status == "running":
        dev.detail = f"running: {running}" + (f" (port {port})" if port else "")
    else:
        dev.detail = f"engine at {dev.dir or GENERIC_DIR}"
    return dev


def probe(dev, run=_run):
    argv = probe_argv(dev)
    if argv is None:
        if dev.target:
            dev.status, dev.detail = "offline", BAD_TARGET
        return dev
    code, out, err = run(argv, timeout=25)
    return apply_probe(dev, code, out, err)


def scanned_devices(result):
    """Devices for the /24 sweep's answer, in address order."""
    out = []
    for d in result.get("devices") or []:
        open_ssh = 22 in (d.get("ports") or [])
        out.append(tui.Device(name=d.get("name") or d["ip"], target=d["ip"], kind="scan",
                              source="the network scan", aliases=(d["ip"].lower(), (d.get("name") or "").lower()),
                              status="ssh-open" if open_ssh else "no-ssh",
                              detail=d.get("note") or ("ssh is open" if open_ssh else "no ssh answer on port 22")))
    return out


def shared_devices(rows):
    """Devices for the share beacons, one row each with the control host and port remembered."""
    out = []
    for r in rows or []:
        name = r.get("name") or r.get("host")
        out.append(tui.Device(name=f"{name} (shared)", target=r.get("host", ""), kind="shared",
                              source="share beacon", aliases=(r.get("host", "").lower(),),
                              status="shared", port=int(r.get("port") or 0),
                              https=bool(r.get("https")),
                              detail="pair with the code shown on that computer"))
    return out


# ---------------------------------------------------------------- listing on a device

def list_argv(dev, rescan=False):
    """The device's own launcher, --list --json (--rescan to look at its disks again); presets use
    the python their command uses."""
    if dev.kind == "preset":
        cmd = LIST_PRESET.format(dir=dev.dir, python=dev.python)
    else:
        cmd = LIST_GENERIC
    return _ssh(dev, cmd + (" --rescan" if rescan else ""))


def _launcher(dev):
    """The shell words that run the device's own launcher."""
    return f"cd {dev.dir} && {dev.python} launch.py" if dev.kind == "preset" else "cd ~/larder && ./larder"


def delete_argv(dev, model_id, key):
    """The device's own launcher deleting one variant from its models folder; --yes because the menu
    has already asked."""
    return _ssh(dev, f"{_launcher(dev)} --delete {shlex.quote(model_id)} --variant {shlex.quote(key)} --yes")


def search_argv(dev, query):
    """The device's own launcher searching Hugging Face (or reading the one repo a name or link names)
    and judging the results for its own memory and disk, as JSON."""
    return _ssh(dev, f"{_launcher(dev)} --search {shlex.quote(query)} --json")


def parse_listing(text):
    """The JSON a launcher prints for --list --json."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        tail = text.strip().splitlines()[-1] if text.strip() else "no output"
        raise ValueError(f"the launcher there printed no JSON ({tail})")
    data = json.loads(text[start:end + 1])
    if not isinstance(data.get("models"), list):
        raise ValueError("the launcher there printed JSON without a model list")
    return data


def models_from_listing(data, measured=(), measured_id=FLASH_ID, ready_ids=(), found=False):
    """tui.Model rows from a device's --list --json (or --search --json) answer, with the measured
    speed set apart and the preset's own commands marked ready (they run a prepared cache directly)."""
    out = []
    for row in data.get("models") or []:
        vs = []
        for r in row.get("variants") or []:
            why, status, short = r.get("why", ""), r.get("status", ""), bool(r.get("disk_short"))
            old = re.match(r"needs ([\d.]+) GB free on this disk, ([\d.]+ GB free)$", why)
            if old and not r.get("ok"):          # a launcher from before the disk had its own state
                why, short = f"needs {float(old[1]):.0f} GB free on disk, {old[2]}", True
                status = why
            v = tui.Variant(key=r.get("key", ""), label=r.get("label", ""), size=r.get("size") or 0,
                            tps=r.get("tps"), ok=bool(r.get("ok")), why=why, disk_short=short,
                            status=status, note=r.get("note", ""),
                            ready=bool(r.get("ready")), make_bits=r.get("make_bits") or 0,
                            download=r.get("download") or 0, disk=r.get("disk") or 0,
                            expected=r.get("expected") or r.get("disk") or 0, entry_id=r.get("entry_id", ""),
                            path=r.get("path", ""), own=bool(r.get("own")),
                            measured_here=r.get("measured_here") or {})
            if row.get("id") in ready_ids and v.key == "published":
                v.ready, v.ok, v.why, v.status = True, True, "", "ready"
            if measured and row.get("id") == measured_id and v.key == "published":
                v.measured = tuple(measured)
            vs.append(v)
        out.append(tui.Model(id=row.get("id", ""), name=row.get("name", ""), kind=row.get("kind", ""),
                             about=row.get("about", ""), license=row.get("license", ""),
                             tested=bool(row.get("tested")), variants=vs,
                             not_made=row.get("not_made") or [], unavailable=row.get("unavailable", ""),
                             found=found, repo=row.get("repo", "")))
    return out


# ---------------------------------------------------------------- starting on a device

def preset_script(kind, key, run=_run):
    """The remote script the preset script would run: its lock, its command, its stop trap."""
    code, out, err = run(["bash", str(HERE / PRESET_SCRIPT), "--print", kind, key], timeout=10)
    if code != 0 or not out.strip():
        raise RuntimeError(f"the preset script --print {kind} {key} failed: {(err or out).strip()[:200]}")
    return out


def start_argv(dev, model_id, local_port, key="", run=_run, variant=""):
    """(argv, remote port): ssh with the forward, and the script or command that starts the model (a
    listed id or an owner/name, in `variant` unless it is the one as published)."""
    if safe_target(dev.target) is None:
        raise RuntimeError(no_address(dev))
    if dev.kind == "preset":
        info = dev.commands.get(key) or {}
        remote_port = int(info.get("port") or 0)
        if not remote_port:
            raise RuntimeError(f"{dev.name} has no preset command for {model_id}")
        script = preset_script(dev.preset, key, run=run)
        forward = ["-L", f"{local_port}:127.0.0.1:{remote_port}"]
        return _ssh(dev, script, SSH_START, forward), remote_port
    remote_port = local_port
    command = (f"cd {GENERIC_DIR} && ./larder --auto --model {shlex.quote(model_id)} "
               + (f"--variant {shlex.quote(variant)} " if variant and variant != "published" else "")
               + f"--no-browser --port {remote_port}")
    return _ssh(dev, command, SSH_START, ["-L", f"{local_port}:127.0.0.1:{remote_port}"]), remote_port


def stop_argv(dev):
    """Stop a model that a generic device's launcher started; presets stop through their ssh."""
    if dev.kind == "preset":
        return None
    return _ssh(dev, f"cd {GENERIC_DIR} && ./larder stop")


def serving_line(line, remote_port):
    """Whether a remote line says the server is listening on its port."""
    low = line.lower()
    if "open failed" in low or ("channel " in low and "connect failed" in low):
        return False
    if "the chat is at" in low:
        return False
    port = str(remote_port)
    return (("serving" in low or "ready:" in low or "chat" in low) and "127.0.0.1:" + port in line) or \
        ("serving" in low and f":{port}" in line)


# ---------------------------------------------------------------- updating the engine on a device

UPDATE = r"""d=@DIR@
if [ ! -d "$d/.git" ]; then echo "update refused: no git checkout at $d"; exit 2; fi
if [ -n "$(git -C "$d" status --porcelain --untracked-files=no)" ]; then echo "update refused: local changes in $d"; exit 2; fi
if git -C "$d" merge-base --is-ancestor @SHA@ HEAD 2>/dev/null; then cat > /dev/null; echo "already up to date"; exit 0; fi
if remote=$(git -C "$d" remote get-url origin 2>/dev/null) && [ -n "$remote" ]; then
  if git -C "$d" pull -q --ff-only && git -C "$d" merge-base --is-ancestor @SHA@ HEAD 2>/dev/null; then
    cat > /dev/null; echo "updated from origin"; exit 0
  fi
  echo "the remote did not bring this engine's commit; trying the bundle" 1>&2
fi
t=$(mktemp -d) || exit 1
trap 'rm -rf "$t"' EXIT
cat > "$t/update.bundle"
if git -C "$d" fetch -q "$t/update.bundle" HEAD && git -C "$d" merge --ff-only FETCH_HEAD; then
  echo "updated from the bundle"; exit 0
fi
echo "update refused: the checkout there has diverged from this computer's engine"; exit 3"""


def make_bundle(run=_run_bytes):
    """A git bundle of this checkout's HEAD, as the bytes an offline device fast-forwards from."""
    code, out, err = run(["git", "-C", str(HERE), "bundle", "create", "-", "HEAD"], timeout=120)
    if code != 0:
        raise RuntimeError("could not make a git bundle of this engine: "
                           + (err or b"").decode("utf-8", "replace").strip()[:200])
    return out


def update_argv(dev):
    """The ssh call whose stdin carries this computer's bundle and whose script fast-forwards there."""
    if safe_target(dev.target) is None:
        return None
    script = UPDATE.replace("@DIR@", _shell_path(dev.dir or GENERIC_DIR)).replace("@SHA@", _head_sha())
    return _ssh(dev, script)


def _head_sha():
    """This checkout's HEAD commit, which an updated device must contain."""
    r = subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() or "HEAD"


def update_device(dev, job=None, run=_run_stdin, bundle=None):
    """(ok, what happened): update the engine on dev from its remote, else from a bundle of this
    checkout's HEAD; refuses local changes or a diverged checkout."""
    argv = update_argv(dev)
    if argv is None:
        return False, no_address(dev)
    data = bundle if bundle is not None else make_bundle()
    if job:
        job.say(f"updating the engine on {dev.name}")
    code, out, err = run(argv, data, timeout=300)
    if job and job.cancelled():
        return False, "stopped"
    text = (out or "") + ("\n" + err if err else "")
    lines = [x for x in text.splitlines() if x.strip()]
    said = [x for x in (out or "").splitlines() if x.strip()]
    if code == 0:
        return True, said[-1] if said else f"{dev.name} updated"
    return False, lines[-1] if lines else ((err or out).strip() or f"{dev.name} update failed")


# ---------------------------------------------------------------- installing the engine on a device

def _sh_default(name, text=None):
    """The default of one install.sh variable (RAW_URL / REPO_URL), read from install.sh itself."""
    text = text if text is not None else (HERE / "install.sh").read_text(errors="replace")
    m = re.search(re.escape(name) + r'="\$\{MOE_' + re.escape(name) + r':-([^}]*)\}"', text)
    return m.group(1).strip('"') if m else ""


def install_urls(text=None):
    """(raw_url, repo_url) when install.sh has real public URLs, else (None, None)."""
    raw = _sh_default("RAW_URL", text)
    repo = _sh_default("REPO_URL", text)
    if not raw or not repo or "REPLACE-ME" in raw or "REPLACE-ME" in repo:
        return None, None
    return raw, repo


def _install_script():
    """install.sh's own text, for the device to run against a bundle instead of a URL."""
    return (HERE / "install.sh").read_text()


# The device writes the bundle and install.sh itself, then runs the installer with the bundle as
# its repository URL, so the same platform, venv, pip and native steps run offline.
INSTALL_DRIVER = """set -e
t=$(mktemp -d) || exit 1
trap 'rm -rf "$t"' EXIT
cat > "$t/install.bundle"
cat > "$t/install.sh" <<'__MOE_INSTALL_SCRIPT__'
@INSTALL_SH@
__MOE_INSTALL_SCRIPT__
MOE_REPO_URL="$t/install.bundle" INSTALL_DIR=@DIR@ bash "$t/install.sh"
"""


def install_argv(dev, raw_url=None, run=None):
    """The ssh call that installs the engine there: the public installer when a URL is configured,
    else this checkout's bundle with install.sh's own steps."""
    d = _shell_path(dev.dir or GENERIC_DIR)
    if raw_url:
        return _ssh(dev, f"curl -fsSL {shlex.quote(raw_url.rstrip('/') + '/install.sh')} | INSTALL_DIR={d} bash")
    script = INSTALL_DRIVER.replace("@DIR@", d).replace("@INSTALL_SH@", _install_script())
    return _ssh(dev, script)


def _terminate(proc):
    """TERM to a streamed child, then KILL if it will not go."""
    try:
        proc.terminate()
        proc.wait(15)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except OSError:
            pass
    except OSError:
        pass


def _stream_install(argv, data, job, out, timeout=1800):
    """(ok, message): run argv, pipe data to it, stream its lines to out and stop on cancel."""
    proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    if data is not None:
        try:
            proc.stdin.write(data)
        except (BrokenPipeError, OSError):
            pass
        try:
            proc.stdin.close()
        except OSError:
            pass
    lines = []
    for line in proc.stdout:
        line = line.rstrip("\n")
        if line.strip():
            out(line)
            lines.append(line[:300])
        if job and job.cancelled():
            _terminate(proc)
            break
    code = proc.wait()
    if job and job.cancelled():
        return False, "stopped"
    return code == 0, (lines[-1] if lines else "") or ("" if code == 0 else f"install failed (exit {code})")


def install_device(dev, job=None, bundle=None, out=None, urls=None):
    """(ok, what happened): install the engine on dev, streaming its progress to out (print by
    default), cancellable through job."""
    raw, _ = install_urls() if urls is None else urls
    argv = install_argv(dev, raw)
    if argv is None:
        return False, no_address(dev)
    data = None if raw else (bundle if bundle is not None else make_bundle())
    if job:
        job.say(f"installing the engine on {dev.name}")
    out = out or (lambda line: sys.stdout.write(line + "\n"))
    return _stream_install(argv, data, job, out)