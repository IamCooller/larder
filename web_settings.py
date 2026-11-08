#!/usr/bin/env python3
"""The Settings page's backend: the launcher's model list, its prepare step as a child process, and an option change through /admin/load."""

import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import weakref
from pathlib import Path

import library

HERE = Path(__file__).resolve().parent
GB = 1e9
REFRESH_S = 600          # how old the model list may get before a background rebuild
LOG_LINES = 24           # the child's last lines, for the page's log tail
SAMPLE_S = 0.5           # seconds between two on-disk byte counts
RATE_WINDOW_S = 8.0      # the window a prepare's speed is measured over
CANCEL_WAIT_S = 15       # TERM, then this long for the child to go
CHECK_PROMPT = "Say hello in one short sentence."
CHECK_TOKENS = 24        # the one reply a copy made here must answer before it counts as ready


# ---------------------------------------------------------------- the launcher

class LauncherBackend:
    """The real launcher (launch.py) behind the page, with its own list, flags and prepare child."""

    def __init__(self, models_dir=None):
        import launch
        import machine
        from types import SimpleNamespace
        self.launch = launch
        self.machine = machine
        self.models_dir = Path(models_dir or _models_dir())
        args = SimpleNamespace(port=8800, models_dir=str(self.models_dir), refresh=False)
        self.tui = launch.TuiBackend(args, self.models_dir)
        self.entries = {}
        self._tmp = None

    def info(self):
        return self.tui.info()

    def rescan(self):
        """Look at this computer's disks again for model folders the next list builds from."""
        self.launch.library.invalidate()

    def delete(self, entry, running):
        """(True, what was deleted) or (False, why not) for the folder of this entry; `running` are
        model paths in use, and the launcher adds the ones it finds on the process list."""
        got = {os.path.realpath(r) for r in running if r} | set(self.launch.running_models())
        return self.launch.delete_folder(self.path(entry), self.models_dir, running=got)

    def models(self):
        """The tested models, each variant as the page shows it."""
        self.entries = {}
        return self._rows(self.tui.models(lambda s: None), self.entries)

    def find(self, query):
        """(catalog entries, how many results were skipped) for a Hugging Face search, an owner/name or
        a link; ValueError with the reason when nothing can be used."""
        try:
            found, skipped = self.launch.search_models(query)
        except self.launch.Unsupported as err:
            raise ValueError(str(err))
        except Exception as err:
            raise ValueError(f"{type(err).__name__}: {err}")
        return found, len(skipped)

    def found_rows(self, raw):
        """(rows, entries by (model id, variant key)) for found catalog entries, judged now."""
        if self.tui.prof is None:
            self.tui.info()
        entries = {}
        return self._rows([self.tui._model(e, False, found=True) for e in raw], entries), entries

    def _rows(self, models, entries):
        out = []
        for m in models:
            variants = []
            for v in m.variants:
                e = v.raw["e"]
                entries[(m.id, v.key)] = e
                variants.append({
                    "key": v.key, "label": v.label, "note": v.note,
                    "size_bytes": int(v.size or 0),
                    "download_bytes": int(v.download or 0),
                    "tps": v.tps, "tps_lo": v.tps_lo, "tps_hi": v.tps_hi,
                    "tps_adjusted": bool(v.tps_adjusted), "need_gb": v.need_gb, "mtp": bool(v.mtp),
                    "ok": bool(v.ok), "why": v.why or "",
                    "disk_short": bool(v.disk_short),
                    "status": v.status, "ready": bool(v.ready),
                    "make_bits": int(v.make_bits or 0), "path": self.path(e),
                    "disk_path": v.path, "own": bool(v.own),
                    "memory": self._memory(e),
                })
            out.append({"id": m.id, "name": m.name, "kind": m.kind, "repo": m.repo,
                        "about": m.about, "license": m.license or "",
                        "tested": bool(m.tested), "unavailable": m.unavailable or "",
                        "variants": variants, "not_made": list(m.not_made or [])})
        return out

    def _memory(self, e):
        """The memory flags the launcher would start this entry with, or None."""
        try:
            s = self.machine.settings(self.tui.prof, {**e["facts"], **e})
        except Exception:
            return None
        return _mem_of(s.get("args") or [])

    def entry(self, model_id, variant_key):
        return self.entries.get((model_id, variant_key))

    def path(self, entry):
        return str(self.launch.model_dir(entry, self.models_dir))

    def ready(self, entry):
        d = Path(self.path(entry))
        return self.launch.is_ready(d) or bool(entry.get("path") and not entry.get("needs_pack"))

    def mark_ready(self, entry):
        self.launch.mark_ready(Path(self.path(entry)), entry)

    def plan(self, entry):
        """The engine flags and the estimate for this entry on this machine now."""
        s, est = self.launch.engine_plan(entry, self.tui.prof, self.tui.disk, native=False)
        return {"args": list(s["args"]), "env": dict(s["env"]),
                "tps": (est or {}).get("tps"), "why": list(s["why"]),
                "biased": bool(s.get("biased")), "ok": bool(s.get("ok")),
                "store_gb": s.get("store_gb")}

    def judge(self, entry, freed_gb=0.0):
        """This entry judged now and as a switch, with the running engine's memory given back first."""
        base = self.launch.judge(entry, self.tui.prof, self.models_dir, self.tui.disk)
        tps0 = (base["est"] or {}).get("tps")
        if not freed_gb or freed_gb <= 0:
            return {"ok": bool(base["ok"]), "why": base["why"] or "", "tps": tps0,
                    "disk_short": bool(base.get("short")),
                    "status": self.launch.status(base, self.models_dir, self.tui.disk),
                    "matters": False}
        sw = self.launch.judge(entry, switch_profile(self.tui.prof, freed_gb),
                               self.models_dir, self.tui.disk)
        tps1 = (sw["est"] or {}).get("tps")
        return {"ok": bool(sw["ok"]), "why": sw["why"] or "", "tps": tps1,
                "disk_short": bool(sw.get("short")),
                "status": self.launch.status(sw, self.models_dir, self.tui.disk),
                "matters": bool(sw["ok"] != base["ok"] or tps0 != tps1)}

    def prepare_command(self, entry):
        """The launcher's own prepare child, on a temporary copy of the entry it reads."""
        fd, path = tempfile.mkstemp(suffix=".json", prefix="moe-entry-")
        with os.fdopen(fd, "w") as f:
            json.dump(entry, f)
        self._tmp = path
        return [sys.executable, str(HERE / "launch.py"), "--prepare", path,
                "--models-dir", str(self.models_dir)]

    def prepare_dirs(self, entry):
        return [Path(self.path(x)) for x in (entry, entry.get("source")) if x]

    def prepare_total(self, entry):
        return int(entry.get("disk") or entry.get("size") or 0)


def _models_dir():
    """Where the launcher's models live here: its own setting, then the running server's folder."""
    env = os.environ.get("MOE_MODELS_DIR")
    if env:
        return Path(env).expanduser()
    import machine
    try:
        st = json.loads((Path(machine.CACHE) / "server.json").read_text())
        p = Path(st.get("model") or "")
        if p.is_dir() and (p / "launcher.json").exists():
            return p.parent
    except (OSError, ValueError):
        pass
    return Path(machine.CACHE) / "models"


def _mem_of(args):
    """The (floor, ceiling) of --mem-peak-gb / --mem-peak-max-gb in one flag list."""
    g = n = None
    for i, a in enumerate(args):
        try:
            if a == "--mem-peak-gb":
                g = float(args[i + 1])
            elif a == "--mem-peak-max-gb":
                n = float(args[i + 1])
        except (IndexError, TypeError, ValueError):
            pass
    return {"gb": g, "max_gb": n}


def engine_gb(host):
    """GB the running engine holds that a switch gives back: its MLX active and cached memory."""
    if getattr(host, "llm", None) is None:
        return 0.0
    try:
        import mlx.core as mx
        return max(0.0, (mx.get_active_memory() + mx.get_cache_memory()) / GB)
    except Exception:
        return 0.0


def switch_profile(prof, freed_gb):
    """The profile a switch sees: the running engine's own memory added back to the free reading."""
    if not freed_gb or freed_gb <= 0:
        return prof
    now = dict(prof.get("now") or {})
    for key in ("avail_gb", "vram_free_gb"):
        if now.get(key) is not None:
            now[key] = now[key] + freed_gb
    return {**prof, "now": now}


_CONTEXT = {}


def context_length(path):
    """Tokens the checkpoint at this path was trained to read, from its config.json, or None."""
    key = str(path or "")
    if key not in _CONTEXT:
        value = None
        try:
            cfg = json.loads((Path(key) / "config.json").read_text())
            for c in (cfg, cfg.get("text_config") or {}):
                for name in ("max_position_embeddings", "n_positions", "max_sequence_length", "seq_length"):
                    v = c.get(name)
                    if isinstance(v, int) and not isinstance(v, bool) and 256 <= v <= 10_000_000:
                        value = value or v
        except (OSError, ValueError, AttributeError):
            pass
        _CONTEXT[key] = value
    return _CONTEXT[key]


def config_match(models, entries, sig):
    """(model index, variant index) of the row whose catalog facts match a folder's signature
    (library.signature: type, layers, hidden size, experts, bits, EXL3 or not), or None."""
    if not sig:
        return None
    for mi, m in enumerate(models):
        for vi, v in enumerate(m["variants"]):
            if library.matches(sig, (entries.get((m["id"], v["key"])) or {}).get("facts")):
                return mi, vi
    return None


# ---------------------------------------------------------------- the model list

class Catalog:
    """The launcher's list, built off the request thread and kept until it goes stale."""

    def __init__(self, backend):
        self.backend = backend
        self.lock = threading.Lock()
        self.models, self.machine = [], None
        self.updated, self.building, self.error = 0.0, False, None
        self.entries = {}
        self.found = None                    # the last search: {"query", "raw", "skipped", "models"}
        self.found_entries = {}

    def search(self, query):
        """Look on Hugging Face (words, an owner/name or a link) and keep the results for the page;
        ValueError with the reason when none can be used."""
        raw, skipped = self.backend.find(query)
        rows, entries = self.backend.found_rows(raw)
        with self.lock:
            self.found = {"query": query, "raw": raw, "skipped": skipped, "models": rows}
            self.found_entries = entries
        return rows, skipped

    def clear_found(self):
        with self.lock:
            self.found, self.found_entries = None, {}

    def ensure(self, stale=REFRESH_S):
        """Start a rebuild when there is none running and the list is missing or stale."""
        with self.lock:
            fresh = self.updated and time.time() - self.updated < stale
            if self.building or fresh:
                return
            self.building = True
        threading.Thread(target=self._build, daemon=True).start()

    def _build(self):
        try:
            info = self.backend.info()
            models = list(self.backend.models())
        except Exception as e:
            with self.lock:
                self.error = type(e).__name__ + ": " + str(e)
                self.building, self.updated = False, time.time()
            return
        with self.lock:
            self.models, self.machine = models, info
            self.entries = dict(getattr(self.backend, "entries", {}) or {})
            self.error, self.building, self.updated = None, False, time.time()
            raw = (self.found or {}).get("raw")
        if raw:                              # the results are judged again: a download may have finished
            try:
                rows, entries = self.backend.found_rows(raw)
                with self.lock:
                    if self.found and self.found["raw"] is raw:
                        self.found["models"], self.found_entries = rows, entries
            except Exception:
                pass

    def entry(self, model_id, variant_key):
        with self.lock:
            return self.entries.get((model_id, variant_key)) or self.found_entries.get((model_id, variant_key))

    def variant_for_path(self, path):
        """(((model id, variant key), entry)) for the entry whose own folder is path."""
        want = _real(path)
        if not want:
            return None
        with self.lock:
            items = list(self.entries.items()) + list(self.found_entries.items())
        for (mid, key), e in items:
            try:
                if _real(self.backend.path(e)) == want:
                    return (mid, key), e
            except Exception:
                continue
        return None

    def memory_auto(self, path):
        """The launcher's memory flags for the model at path, or None when it is not listed."""
        got = self.variant_for_path(path)
        if not got:
            return None
        try:
            return _mem_of(self.backend.plan(got[1]).get("args") or [])
        except Exception:
            return None


def _real(path):
    try:
        return os.path.realpath(str(path)) if path else ""
    except OSError:
        return ""


# ---------------------------------------------------------------- one prepare

class Busy(Exception):
    """A prepare is already running."""


class Prepare:
    """One prepare at a time: the launcher's child process, its bytes on disk and its log."""

    def __init__(self):
        # Reentrant: state_dict() reads the samples and the speed under the same lock a caller holds.
        self.lock = threading.RLock()
        self.proc = None
        self.tmp = None
        self.dirs, self.total = [], 0
        self.model = self.variant = self.name = self.label = ""
        self.state, self.error, self.exit = "idle", None, None
        self.started = self.ended = 0.0
        self.log, self.samples = [], []
        self.bytes, self._counted = 0, 0.0

    def start(self, model_id, variant_key, entry, backend):
        with self.lock:
            if self.state == "running":
                raise Busy("already preparing " + (self.name or "a model"))
            self.model, self.variant = model_id, variant_key
            self.name = entry.get("name") or model_id
            self.label = variant_key
            self.dirs = backend.prepare_dirs(entry)
            self.total = backend.prepare_total(entry)
            self.state, self.error, self.exit = "running", None, None
            self.started, self.ended = time.time(), 0.0
            self.log, self.samples = [], [(self.started, 0)]
            self.bytes, self._counted = 0, 0.0
            try:
                cmd = backend.prepare_command(entry)
            except OSError as e:
                self.state, self.error = "failed", "could not start the prepare: " + str(e)
                raise
            self.tmp = getattr(backend, "_tmp", None)
            try:
                # Its own session, so a cancel stops the packer the child started too.
                self.proc = subprocess.Popen(
                    cmd, cwd=HERE, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
            except OSError as e:
                self.state, self.error = "failed", "could not start the prepare: " + str(e)
                self._drop_tmp()
                raise
            threading.Thread(target=self._read, daemon=True).start()
        return self.state_dict()

    def _read(self):
        proc = self.proc
        try:
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    with self.lock:
                        self.log.append(line[:300])
                        del self.log[:-LOG_LINES]
        except (OSError, ValueError):
            pass
        code = proc.wait()
        with self.lock:
            self._drop_tmp()
            if self.state != "running":
                return
            self.ended = time.time()
            self.exit = code
            if code == 0:
                self.state = "done"
                self.bytes = max(self.bytes, self._count(force=True))
            else:
                self.state, self.error = "failed", "the prepare stopped with exit " + str(code)
        if code == 0:
            cat = _catalog()
            if cat is not None:
                cat.ensure(stale=0)

    def _count(self, force=False):
        """Bytes this prepare has left on disk, counted at most every SAMPLE_S."""
        now = time.time()
        if not force and now - self._counted < SAMPLE_S:
            return self.bytes
        self._counted = now
        total = 0
        for d in self.dirs:
            total += _tree_bytes(d)
        self.bytes = total
        self.samples.append((now, total))
        del self.samples[:-int(RATE_WINDOW_S / SAMPLE_S) - 1]
        return total

    def _drop_tmp(self):
        if self.tmp:
            try:
                os.unlink(self.tmp)
            except OSError:
                pass
            self.tmp = None

    def rate(self):
        """(bytes a second, seconds left) over the recent window; (None, None) until it can tell."""
        with self.lock:
            samples = list(self.samples)
        if len(samples) < 2:
            return None, None
        (t0, b0), (t1, b1) = samples[0], samples[-1]
        if t1 - t0 < 0.9 or b1 <= b0:
            return None, None
        r = (b1 - b0) / (t1 - t0)
        left = max(0, self.total - b1) / r if self.total > b1 else 0.0
        return r, left

    def state_dict(self):
        with self.lock:
            if self.state in ("running", "done", "failed", "cancelled") and self.dirs:
                self._count()
            rate, left = self.rate()
            return {
                "state": self.state, "model": self.model, "variant": self.variant,
                "name": self.name, "label": self.label,
                "bytes": int(self.bytes), "total": int(self.total),
                "rate_bps": rate, "eta_s": left,
                "seconds": round((self.ended or time.time()) - self.started, 1)
                if self.started else 0,
                "log": list(self.log), "error": self.error, "exit": self.exit,
            }

    def clear(self, model, variant):
        """Forget a finished prepare of this variant, whose folder is gone."""
        with self.lock:
            if self.state != "running" and (self.model, self.variant) == (model, variant):
                self.state, self.error, self.bytes, self.log, self.samples = "idle", None, 0, [], []

    def cancel(self):
        with self.lock:
            running = self.state == "running" and self.proc is not None
            if running:
                self.state, self.ended = "cancelled", time.time()
                proc = self.proc
        if not running:
            return self.state_dict()
        _terminate(proc)
        with self.lock:
            self._drop_tmp()
        return self.state_dict()


def _terminate(proc):
    """TERM to the child's whole session, then KILL only if it will not go."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except OSError:
        try:
            proc.terminate()
        except OSError:
            return
    try:
        proc.wait(CANCEL_WAIT_S)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except OSError:
            proc.kill()


def _tree_bytes(path):
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.lstat(os.path.join(root, name)).st_size
            except OSError:
                pass
    return total


# ---------------------------------------------------------------- what is running

# The page's own view of the options, seeded from the host and updated on a load.
_SEEN = weakref.WeakKeyDictionary()
_PENDING = {}


def _seen(host):
    """The options the engine is serving with, as the page shows them."""
    got = _SEEN.get(host)
    if got is None:
        kw = dict(getattr(host, "engine_kwargs", {}) or {})
        cb = kw.get("cache_bias")
        got = {
            "memory_gb": getattr(host, "mem_peak_gb", None),
            "memory_max_gb": getattr(host, "mem_peak_max_gb", None),
            # The launcher always starts a model with its own plan, so that is Auto.
            "memory_auto": True,
            # Unset cache bias is the engine's 0.5, except under a memory budget where it is 0.
            "fast": (float(cb) > 0) if cb is not None else getattr(host, "mem_peak_gb", None) is None,
            "thinking": bool(getattr(host, "think_default", False)),
        }
        _SEEN[host] = got
    return got


def settings_of(host):
    return dict(_seen(host))


def _number(value, what, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(what + " must be a number")
    v = float(value)
    if not low < v <= high:
        raise ValueError(what + " must be between " + str(low) + " and " + str(high) + " GB")
    return round(v, 1)


def load_kwargs(host, body, kwargs):
    """The page's options from a /admin/load body, applied to the host and the load's keys; "plan": true adds the launcher's flags for the model, and a key the body names wins over the plan."""
    changes, prev = {}, (getattr(host, "mem_peak_gb", None),
                         getattr(host, "mem_peak_max_gb", None))
    plan = plan_for(body.get("model") or host.model_name) if body.get("plan") else None
    if "memory" in body:
        memo = body["memory"]
        auto = memo is None or memo == "auto"
        if auto:
            got = _mem_of(plan["args"]) if plan and plan.get("args") else None
            if got is None:
                cat = _catalog()
                if cat is not None:
                    cat.ensure(stale=0)
                    got = cat.memory_auto(body.get("model") or host.model_name)
            # A model the launcher does not list keeps whatever the host was started with.
            g, n = (got["gb"], got["max_gb"]) if got else prev
        else:
            g = _number(memo, "the memory limit", 0.5, 512.0)
            n = None
        host.mem_peak_gb, host.mem_peak_max_gb = g, n
        changes.update(memory_gb=g, memory_max_gb=n, memory_auto=auto)
    elif plan is not None:
        got = _mem_of(plan.get("args") or [])
        host.mem_peak_gb, host.mem_peak_max_gb = got["gb"], got["max_gb"]
        changes.update(memory_gb=got["gb"], memory_max_gb=got["max_gb"], memory_auto=True)
    if "fast" in body:
        fast = body["fast"]
        if not isinstance(fast, bool):
            raise ValueError("'fast' must be true or false")
        kwargs["cache_bias"] = 0.5 if fast else 0.0
        changes["fast"] = fast
    elif plan is not None:
        biased = bool(plan.get("biased"))
        kwargs["cache_bias"] = 0.5 if biased else 0.0
        changes["fast"] = biased
        budget = _flag(plan.get("args") or [], "--bias-budget")
        if budget and "bias_budget" not in kwargs:
            # The launcher's budget travels with its cache bias, and a swap can set it.
            kwargs["bias_budget"] = float(budget)
    if "thinking" in body:
        thinking = body["thinking"]
        if not isinstance(thinking, bool):
            raise ValueError("'thinking' must be true or false")
        kwargs["thinking"] = thinking
        changes["thinking"] = thinking
    if changes or plan is not None:
        _PENDING[host] = (changes, prev, plan)
    return kwargs


def plan_for(path):
    """The launcher's flags for the model at path, or None when it is not in its list."""
    cat = _catalog()
    if cat is None:
        return None
    cat.ensure(stale=0)
    got = cat.variant_for_path(path)
    if not got:
        return None
    try:
        return cat.backend.plan(got[1])
    except Exception:
        return None


# Plan flags a swap can set now, and the host settings the others would need at start.
NOW_FLAGS = ("--mem-peak-gb", "--mem-peak-max-gb", "--cache-bias", "--bias-budget")
START_FLAGS = {"--prefill-step": "prefill_step", "--keep-convs": "keep_convs"}


def _flag(args, name):
    """The value after `name` in one flag list, or None."""
    for i, a in enumerate(args):
        if a == name and i + 1 < len(args):
            return args[i + 1]
    return None


def _args_of(host, args):
    """(flags a swap sets now, flags that differ from what the running server was started with)."""
    now, later, i = [], [], 0
    while i < len(args):
        a = args[i]
        pair = [a] if i + 1 >= len(args) or args[i + 1].startswith("--") else [a, args[i + 1]]
        i += len(pair)
        if pair[0] in NOW_FLAGS:
            now.append(" ".join(pair))
            continue
        key = START_FLAGS.get(pair[0], pair[0][2:].replace("-", "_"))
        have = (getattr(host, key, None) if key == "keep_convs"
                else (host.engine_kwargs or {}).get(key))
        if len(pair) > 1 and str(have) == pair[1]:
            continue
        later.append(" ".join(pair))
    return now, later


def plan_note(host, plan):
    """What the page says about the launcher's flags for the model it just loaded."""
    if plan is None:
        return None
    now, later = _args_of(host, list(plan.get("args") or []))
    return {"args": " ".join(list(plan.get("args") or [])), "set_now": now,
            "at_start": later, "why": list(plan.get("why") or [])}


def after_load(host, ok):
    """Record what a finished load applied, undo it on failure, check a copy made here."""
    rec = _PENDING.pop(host, None)
    if not ok:
        if rec:
            host.mem_peak_gb, host.mem_peak_max_gb = rec[1]
        return {}
    out, plan = {}, None
    if rec:
        changes, _, plan = rec
        seen = _seen(host)
        seen.update(changes)
        seen["thinking"] = bool(getattr(host, "think_default", seen.get("thinking", False)))
        out.update(changes)
    out.update(_check_made(host))
    note = plan_note(host, plan)
    if note:
        out["plan"] = note
    return out


def _check_made(host):
    """A copy made here is offered as ready once the engine has answered one reply."""
    cat = _catalog()
    if cat is None:
        return {}
    got = cat.variant_for_path(host.model_name)
    if not got:
        return {}
    backend = _BACKEND
    entry = got[1]
    if not entry.get("make_bits") or backend is None:
        return {}
    try:
        if backend.ready(entry):
            return {}
        text = host.generate([{"role": "user", "content": CHECK_PROMPT}],
                             CHECK_TOKENS, stream=False)
    except Exception:
        return {}
    if not str(text or "").strip():
        return {}
    try:
        backend.mark_ready(entry)
    except OSError:
        return {}
    cat.ensure(stale=0)
    return {"checked": " ".join(str(text).split())[:60]}


# ---------------------------------------------------------------- the endpoints

_LOCK = threading.Lock()
_BACKEND = None
_CATALOG = None
_CATALOG_ERROR = None
_PREPARE = Prepare()


def configure(backend=None):
    """Point the endpoints at the real launcher, or at a stand-in (tests): None restores the real one."""
    global _BACKEND, _CATALOG, _CATALOG_ERROR, _PREPARE
    _PREPARE.cancel()
    with _LOCK:
        if backend is None:
            _BACKEND = _CATALOG = _CATALOG_ERROR = None
        else:
            _BACKEND = backend
            _CATALOG = Catalog(backend)
            _CATALOG_ERROR = None
    _PREPARE = Prepare()


def _catalog():
    """The launcher's list, made on first use; None when the launcher cannot be imported here."""
    global _BACKEND, _CATALOG, _CATALOG_ERROR
    with _LOCK:
        if _CATALOG is not None or _CATALOG_ERROR is not None:
            return _CATALOG
        try:
            _BACKEND = LauncherBackend()
        except Exception as e:
            _CATALOG_ERROR = ("the launcher is not available here ("
                              + type(e).__name__ + ": " + str(e) + ")")
            return None
        _CATALOG = Catalog(_BACKEND)
        return _CATALOG


def _query(h):
    return h.path.split("?", 1)[1] if "?" in h.path else ""


def payload(host):
    """Everything the Settings page draws, as one JSON object."""
    job = _PREPARE.state_dict()
    cat = _catalog()
    if cat is None:
        return {"ok": False, "unavailable": _CATALOG_ERROR, "models": [], "found": None, "machine": None,
                "building": False, "updated": 0.0, "error": None, "job": job,
                "current": _current(host), "settings": settings_of(host)}
    cat.ensure()
    with cat.lock:
        models, machine = cat.models, cat.machine
        entries = {**cat.found_entries, **cat.entries}
        updated, building, error = cat.updated, cat.building, cat.error
        backend = cat.backend
        searched = dict(cat.found) if cat.found else None
    now = _real(host.model_name)
    freed = engine_gb(host)
    out = []
    # The search results are judged and marked like the tested models, and listed apart.
    for found, group in ((False, models), (True, searched["models"] if searched else [])):
        for m in group:
            variants = []
            for v in m["variants"]:
                cur = bool(v.get("path")) and _real(v["path"]) == now
                variants.append({**v, "current": cur, "own_path": "", "after_unload": False})
            out.append({**m, "variants": variants, "found": found})
    if not any(v["current"] for m in out for v in m["variants"]):
        # A copy outside the launcher's folder is matched to its row by its config.
        hit = config_match(out, entries, library.signature(host.model_name or ""))
        if hit:
            v = out[hit[0]]["variants"][hit[1]]
            v["current"], v["own_path"] = True, str(host.model_name)
    current = None
    measured = library.load_measured()
    for m in out:
        for v in m["variants"]:
            facts = (entries.get((m["id"], v["key"])) or {}).get("facts")
            v["measured_here"] = library.lookup(facts, data=measured) if facts else {}
            judged = _judge_for_switch(backend, entries.get((m["id"], v["key"])), freed)
            if judged:
                v.update(ok=bool(judged["ok"]), why=judged.get("why") or "",
                         disk_short=bool(judged.get("disk_short")),
                         tps=judged.get("tps"), after_unload=bool(judged.get("matters")))
                if judged.get("status") is not None:
                    v["status"] = judged["status"]
            if v["current"]:
                # The model that is running here is never one this computer cannot run.
                v.update(ok=True, why="", disk_short=False)
                if v["own_path"] or str(v["status"]).startswith("can't run"):
                    v["status"] = "running now"
                current = {"id": m["id"], "name": m["name"], "variant": v["key"],
                           "label": v["label"], "path": v["own_path"] or v["path"],
                           "own_path": v["own_path"], "memory": v["memory"]}
    st = host.status()
    cur = current or _current(host)
    cur.update(tps=st.get("decode_tok_s"), loaded=bool(st.get("loaded")),
               vision=bool(st.get("vision")))
    return {"ok": True, "unavailable": None, "building": building,
            "updated": round(updated, 1), "error": error, "machine": machine,
            "current": cur, "settings": settings_of(host), "models": [m for m in out if not m["found"]],
            "found": {"query": searched["query"], "skipped": searched["skipped"],
                      "models": [m for m in out if m["found"]]} if searched else None,
            "job": job}


def _judge_for_switch(backend, entry, freed):
    """The backend's judgement with the engine's memory given back, or None when it cannot judge."""
    fn = getattr(backend, "judge", None)
    if fn is None or entry is None:
        return None
    try:
        return fn(entry, freed)
    except Exception:
        return None


def _current(host):
    return {"id": None, "name": Path(str(host.model_name or "")).name or "no model",
            "variant": None, "label": "", "path": str(host.model_name or ""),
            "memory": None, "tps": None, "loaded": False, "vision": False}


def get_models(h):
    """GET /admin/models[?refresh=1]: the launcher's list for this machine."""
    if _catalog() is not None and "refresh=1" in _query(h):
        getattr(_BACKEND, "rescan", lambda: None)()
        _catalog().ensure(stale=0)
    h._json(200, payload(h.host))


def get_prepare(h):
    """GET /admin/prepare: the running or last prepare, with its progress."""
    h._json(200, {"ok": True, "job": _PREPARE.state_dict()})


def post_prepare(h):
    """POST /admin/prepare {model, variant}: start the launcher's prepare as a child."""
    try:
        body = h._read_json()
        if not isinstance(body, dict):
            raise ValueError
    except ValueError:
        h._error(400, "invalid JSON body")
        return
    model = body.get("model")
    variant = body.get("variant") or "published"
    if not isinstance(model, str) or not model:
        h._error(400, "'model' is required")
        return
    cat = _catalog()
    if cat is None:
        h._error(503, _CATALOG_ERROR or "the launcher is not available here")
        return
    entry = cat.entry(model, str(variant))
    if entry is None:
        cat.ensure(stale=0)
        h._error(404, "no model '" + model + "' variant '" + str(variant)
                 + "' in the launcher's list for this computer; it may still be loading")
        return
    try:
        state = _PREPARE.start(model, str(variant), entry, _BACKEND)
    except Busy as e:
        h._error(409, str(e))
        return
    except OSError as e:
        h._error(500, "could not start the prepare: " + str(e))
        return
    h._json(200, {"ok": True, "job": state})


_SEARCH = threading.Lock()


def get_search(h):
    """GET /admin/search?q=...: look on Hugging Face for this computer (words, an owner/name or a
    link), keep the results for the page's list, judged by the launcher here; 422 with the reason when
    nothing can be used."""
    q = (urllib.parse.parse_qs(_query(h)).get("q") or [""])[0].strip()
    if not q:
        h._error(400, "'q' is required: words to search for, an owner/name or a link")
        return
    cat = _catalog()
    if cat is None:
        h._error(503, _CATALOG_ERROR or "the launcher is not available here")
        return
    if not _SEARCH.acquire(blocking=False):
        h._error(409, "a search is already running")
        return
    try:
        rows, skipped = cat.search(q)
    except ValueError as e:
        h._error(422, str(e))
        return
    finally:
        _SEARCH.release()
    h._json(200, {"ok": True, "query": q, "count": len(rows), "skipped": skipped})


def delete_search(h):
    """DELETE /admin/search: forget the results."""
    cat = _catalog()
    if cat is not None:
        cat.clear_found()
    h._json(200, {"ok": True})


def post_delete(h):
    """POST /admin/delete {model, variant}: delete a model folder the launcher downloaded or made;
    409 with the reason for one elsewhere, the model being served or one still downloading."""
    try:
        body = h._read_json()
        if not isinstance(body, dict):
            raise ValueError
    except ValueError:
        h._error(400, "invalid JSON body")
        return
    model, variant = body.get("model"), str(body.get("variant") or "published")
    if not isinstance(model, str) or not model:
        h._error(400, "'model' is required")
        return
    cat = _catalog()
    if cat is None:
        h._error(503, _CATALOG_ERROR or "the launcher is not available here")
        return
    entry = cat.entry(model, variant)
    if entry is None:
        h._error(404, "no model '" + model + "' variant '" + variant + "' in the launcher's list for this computer")
        return
    job = _PREPARE.state_dict()
    if job["state"] == "running" and (job["model"], job["variant"]) == (model, variant):
        h._error(409, "it is still downloading: stop that first")
        return
    ok, message = _BACKEND.delete(entry, [h.host.model_name])
    if not ok:
        h._error(409, message)
        return
    _PREPARE.clear(model, variant)
    cat.ensure(stale=0)
    h._json(200, {"ok": True, "message": message})


def delete_prepare(h):
    """DELETE /admin/prepare: stop the running prepare, TERM first as the menu does."""
    h._json(200, {"ok": True, "job": _PREPARE.cancel()})