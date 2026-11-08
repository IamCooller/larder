#!/usr/bin/env python3
"""The launcher's full-screen menu (open it with ./larder, or try it with made-up computers and models
using `python3 tui.py --demo`)."""
from __future__ import annotations

import collections
import contextlib
import locale
import os
import sys
import textwrap
import threading
import time
from dataclasses import dataclass, field

try:
    import curses
except ImportError:          # Windows: the text menu is used there
    curses = None

GB = 1e9
SPINNER = "|/-\\"


# ---------------------------------------------------------------- what the menu shows

@dataclass
class Variant:
    """One way to run a model: as published, at other bits, or made here."""
    key: str
    label: str
    size: int                       # bytes the variant takes on disk
    tps: float | None               # decode tok/s estimate for this computer
    ok: bool                        # can it run here now
    why: str                        # the reason when not
    status: str                     # "ready", "download 20.4 GB", "can't run: ...", "~5 s to make ..."
    note: str = ""
    ready: bool = False             # already on disk
    make_bits: int = 0              # rounded here to this many bits
    download: int = 0
    disk: int = 0
    expected: int = 0               # bytes the install leaves on disk, for the progress bar
    entry_id: str = ""              # what the launcher remembers as "last used"
    measured: tuple = ()            # (low, high) tok/s measured on that computer, empty when an estimate
    disk_short: bool = False        # it would run, but the disk has too little room: not "can't run"
    path: str = ""                  # where its files are on that computer, empty when not on disk
    own: bool = False               # that folder is inside the launcher's models folder
    measured_here: dict = field(default_factory=dict)   # {"tps", "n", "date"} its replies there ran at
    need_gb: float | None = None    # memory the start needs with the memory it will find free
    tps_lo: float | None = None     # estimate with the engine's own store at its floor
    tps_hi: float | None = None     # estimate with the store at the size it will start at
    tps_adjusted: bool = False      # corrected by this computer's own measured runs
    mtp: bool = False               # its own MTP head is on disk here and is used
    raw: object = None


@dataclass
class Model:
    id: str
    name: str
    kind: str = ""                  # "MoE, 256 experts x 40 layers" / "dense"
    about: str = ""
    license: str = ""
    tested: bool = False
    variants: list = field(default_factory=list)
    not_made: list = field(default_factory=list)
    unavailable: str = ""           # the engine cannot run it at all, in words
    found: bool = False             # from a search or a folder, not the tested list
    repo: str = ""                  # its Hugging Face owner/name, when it has one


@dataclass
class Device:
    """A computer the menu can run a model on: this one, a preset, an ssh host, a peer."""
    name: str
    target: str = ""                # the ssh destination; empty for this computer
    kind: str = "local"             # local preset ssh tailscale scan
    source: str = ""                # where the row came from
    aliases: tuple = ()
    detail: str = ""                # the fix, the install line, or where its engine is
    hardware: str = ""
    status: str = "checking"        # checking ready running no-key no-engine offline ssh-open no-ssh
    running: str = ""               # the model already running there
    port: int = 0
    https: bool = False             # a shared device whose control and model servers speak TLS
    fingerprint: str = ""           # that device's certificate fingerprint, read before its code is asked
    preset: str = ""                # the preset's key
    dir: str = ""                   # its repo folder
    python: str = ""                # the python its preset command uses
    commands: dict = field(default_factory=dict)   # preset key -> {port, id, command}
    measured: tuple = ()            # measured preset speeds there, (low, high)
    info: dict = field(default_factory=dict)       # its --list --json "machine"
    engine: dict = field(default_factory=dict)     # its --version / --list --json "engine"
    engine_state: str = ""          # ok / older / newer, empty until a probe or a list knows
    models: list = field(default_factory=list)     # its cached model list
    listed_at: float = 0.0


class Backend:
    """What the menu needs (launch.TuiBackend); DemoBackend below is a stand-in."""

    def info(self):
        """{"chip", "ram_gb", "gpu", "os", "models_dir", "free_gb", "ssd_gbps", "others"}"""
        raise NotImplementedError

    def attached(self):
        """A model an earlier run left running here, as start_device() returns it, or None."""
        return None

    def live(self):
        """The running server's /status while it serves: the measured speed over its replies, the
        conditions and the engine's own store, or None when it does not answer."""
        return None

    def device_list(self):
        """[Device]: this computer first, then the presets, ssh hosts and Tailscale peers."""
        raise NotImplementedError

    def device_probe(self, dev):
        """Fill dev's status, hardware and running model with one ssh call; returns dev."""
        raise NotImplementedError

    def pair(self, dev, code):
        """Pair with a shared device using its code, store the token, return the device."""
        raise NotImplementedError

    def pair_fingerprint(self, dev):
        """The shared device's certificate fingerprint that pairing will pin, read before its code is asked; "" without TLS."""
        return ""

    def sweep_subnet(self):
        """The network a sweep would look at, as "a.b.c.0/24", or None off a network."""
        return None

    def device_scan(self, say):
        """Look at this network once (devices.py) and return the computers it found, asked for by the user only."""
        raise NotImplementedError

    def device_models(self, dev, say):
        """Yield the models dev's own launcher offers, one at a time; say(text) names each step."""
        raise NotImplementedError

    def update_device(self, dev, job):
        """Update the engine on dev over ssh; returns what happened, or raises RuntimeError."""
        raise NotImplementedError

    def install_device(self, dev, job):
        """Install the engine on dev over ssh, streaming its progress; returns what happened."""
        raise NotImplementedError

    def start_device(self, dev, model, variant, job):
        """Start the model on dev and wait until it answers, returning {"name", "url", "api",
        "model_id", "tps", "device", ...} or None when stopped, with job.say(text) naming each
        phase."""
        raise NotImplementedError

    def rescan(self, dev):
        """Forget what was found on dev's disks, so its next list looks at them again."""

    def delete(self, dev, model, variant):
        """Delete the variant's folder on dev (only inside its launcher's models folder); returns
        what was deleted, or raises RuntimeError with the reason it was not."""
        raise NotImplementedError

    def models(self, say):
        """This computer's tested models, one by one (the local device's list)."""
        raise NotImplementedError

    def search(self, query, say):
        """(models found on Hugging Face, number skipped) for this computer: words to search for, an
        owner/name or a link to one repo; raises ValueError with the reason."""
        raise NotImplementedError

    def device_search(self, dev, query, say):
        """The same for dev, judged by dev's own launcher (its memory and disk); raises RuntimeError
        or ValueError with the reason."""
        if dev.kind == "local":
            return self.search(query, say)
        raise NotImplementedError

    def local(self, path):
        """A Model for a folder on this computer; raises ValueError with the reason."""
        raise NotImplementedError

    def last(self):
        """entry_id of the variant started last time on this computer, or None."""
        return None

    def dest(self, variant):
        """The folder the variant is installed into on this computer."""
        return ""

    def progress(self, variant):
        """Bytes installed so far."""
        return 0

    def install(self, variant, job):
        """Download and prepare the variant on this computer; blocks; raises on failure."""
        raise NotImplementedError

    def log_tail(self, n):
        """The last n lines of what the engine on the other side printed, or this one's log."""
        return []

    def abort(self):
        """Stop an install or a start that is under way (never a model that is already running)."""

    def stop(self):
        """Stop the running model, here or on the other computer; its lock is released there."""

    def open_chat(self, url):
        pass

    def chat(self, url):
        """Chat in the terminal (the menu is suspended meanwhile)."""

    def shutdown(self):
        """Stop whatever this menu started when it closes."""
        self.abort()


# ---------------------------------------------------------------- background work

class Log:
    """What print() and the tools write while the menu owns the screen: lines, with \\r
    (a progress bar redrawing itself) replacing the line being written."""

    def __init__(self, keep=500):
        self.lines = collections.deque(maxlen=keep)
        self.partial = ""
        self.lock = threading.Lock()
        self.encoding = "utf-8"

    def write(self, text):
        if isinstance(text, bytes):
            text = text.decode("utf-8", "replace")
        with self.lock:
            for ch in text:
                if ch == "\n":
                    self.lines.append(self.partial)
                    self.partial = ""
                elif ch == "\r":
                    self.partial = ""
                else:
                    self.partial += ch
        return len(text)

    def flush(self):
        pass

    def isatty(self):
        return False

    def tail(self, n):
        with self.lock:
            out = list(self.lines)[-n:] if n else []
            if self.partial.strip() and n:
                out = (out + [self.partial])[-n:]
            return [x for x in out if x.strip()]


class Job:
    """A function on a thread, with a phase text the screen shows and a way to ask it to stop."""

    def __init__(self, fn, *args):
        self.phase = ""
        self.items = []             # what a producing job has yielded so far
        self.result = None
        self.error = None
        self.done = False
        self.stop_flag = threading.Event()
        self.t0 = time.time()
        self._thread = threading.Thread(target=self._run, args=(fn, args), daemon=True)
        self._thread.start()

    def _run(self, fn, args):
        try:
            self.result = fn(self, *args)
        except (Exception, SystemExit) as err:       # SystemExit: the launcher's way to say "failed"
            self.error = str(err) or type(err).__name__
        finally:
            self.done = True

    def say(self, text):
        self.phase = text

    def cancelled(self):
        return self.stop_flag.is_set()

    def cancel(self):
        self.stop_flag.set()

    def seconds(self):
        return time.time() - self.t0


# ---------------------------------------------------------------- drawing surfaces

STYLES = ("normal", "dim", "title", "accent", "good", "warn", "bad", "bad_dim", "blue", "run",
          "sel", "key", "tab", "topbar", "bar_lo", "bar_mid", "bar_hi")


class TextCanvas:
    """A screen in memory: what App draws, for tests and for reading in a log."""

    def __init__(self, w=100, h=30):
        self.w, self.h = w, h
        self.clear()

    def clear(self):
        self.rows = [[" "] * self.w for _ in range(self.h)]
        self.styles = [["normal"] * self.w for _ in range(self.h)]

    def size(self):
        return self.h, self.w

    def put(self, y, x, text, style="normal"):
        if not 0 <= y < self.h:
            return
        for i, ch in enumerate(text):
            if 0 <= x + i < self.w:
                self.rows[y][x + i] = ch
                self.styles[y][x + i] = style

    def fill(self, y, x, n, style):
        for i in range(n):
            if 0 <= y < self.h and 0 <= x + i < self.w:
                self.styles[y][x + i] = style

    def refresh(self):
        pass

    def line(self, y):
        return "".join(self.rows[y]).rstrip()

    def text(self):
        return "\n".join(self.line(y) for y in range(self.h))

    def has(self, s):
        return s in self.text()

    def style_at(self, y, x):
        return self.styles[y][x]


# ANSI colour numbers, so a test can look at the palette without curses.
BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)


def colors_enabled(has_color, env=None):
    """Whether to use colour: the terminal has it and NO_COLOR is not set."""
    env = os.environ if env is None else env
    return bool(has_color) and "NO_COLOR" not in env


def palette(colors=8, enabled=True):
    """[(style, foreground, background (-1 for the terminal's own), attributes)] for the
    menu's colour pairs: rich colours on 256-colour terminals, the basic eight otherwise."""
    if not enabled:
        return []
    if colors >= 256:
        return [("topbar", 16, 51, "bold"), ("accent", 51, -1, "bold"), ("good", 46, -1, "bold"),
                ("warn", 214, -1, "bold"), ("bad", 196, -1, "bold"), ("bad_dim", 88, -1, "dim"),
                ("blue", 39, -1, "bold"), ("run", 51, -1, "bold"), ("sel", 231, 24, "bold"),
                ("key", 45, -1, "bold"), ("tab", 45, -1, "bold underline"), ("title", 255, -1, "bold")]
    return [("topbar", BLACK, CYAN, "bold"), ("accent", CYAN, -1, "bold"), ("good", GREEN, -1, "bold"),
            ("warn", YELLOW, -1, "bold"), ("bad", RED, -1, "bold"), ("bad_dim", RED, -1, "dim"),
            ("blue", BLUE, -1, "bold"), ("run", CYAN, -1, "bold"), ("sel", BLACK, CYAN, "bold"),
            ("key", CYAN, -1, "bold"), ("tab", CYAN, -1, "bold underline"), ("title", WHITE, -1, "bold")]


def _attrs(extra):
    """curses attributes named in a palette row."""
    out = 0
    for word in extra.split():
        out |= {"bold": curses.A_BOLD, "dim": curses.A_DIM, "underline": curses.A_UNDERLINE}[word]
    return out


class CursesCanvas:
    """The terminal, written a changed row at a time: App clears its buffer every frame and
    refresh() sends only the rows whose text or styles differ from the last frame."""

    def __init__(self, scr):
        self.scr = scr
        self.prev = None
        self.attr = {s: 0 for s in STYLES}
        self.attr.update(dim=curses.A_DIM, title=curses.A_BOLD, sel=curses.A_REVERSE,
                         key=curses.A_BOLD, tab=curses.A_BOLD | curses.A_UNDERLINE, topbar=curses.A_BOLD)
        enabled = colors_enabled(getattr(curses, "has_colors", lambda: False)())
        if enabled:
            curses.start_color()
            try:
                curses.use_default_colors()
            except curses.error:
                pass
            for i, (name, fg, bg, extra) in enumerate(palette(getattr(curses, "COLORS", 8), True), 1):
                try:
                    curses.init_pair(i, fg, bg)
                except curses.error:
                    continue
                self.attr[name] = curses.color_pair(i) | _attrs(extra)
        for name, base in (("bar_lo", "bad"), ("bar_mid", "warn"), ("bar_hi", "good")):
            self.attr[name] = self.attr[base]
        h, w = scr.getmaxyx()
        self._resize(h, w)
        self.scr.erase()

    def _resize(self, h, w):
        self.h, self.w = h, w
        self.rows = [[" "] * w for _ in range(h)]
        self.styles = [["normal"] * w for _ in range(h)]
        self.prev = None

    def size(self):
        return self.scr.getmaxyx()

    def clear(self):
        h, w = self.scr.getmaxyx()
        if (h, w) != (self.h, self.w):
            self._resize(h, w)
            self.scr.erase()
            return
        for row in self.rows:
            row[:] = [" "] * self.w
        for st in self.styles:
            st[:] = ["normal"] * self.w

    def put(self, y, x, text, style="normal"):
        if not 0 <= y < self.h:
            return
        for i, ch in enumerate(text):
            if 0 <= x + i < self.w:
                self.rows[y][x + i] = ch
                self.styles[y][x + i] = style

    def fill(self, y, x, n, style):
        for i in range(n):
            if 0 <= y < self.h and 0 <= x + i < self.w:
                self.styles[y][x + i] = style

    def refresh(self):
        old = self.prev
        for y in range(self.h):
            text = "".join(self.rows[y])
            styles = self.styles[y]
            if old and text == old[0][y] and styles == old[1][y]:
                continue
            room = self.w - (1 if y == self.h - 1 else 0)
            try:
                self.scr.addnstr(y, 0, text, max(0, room), self.attr["normal"])
            except (curses.error, UnicodeError):
                try:
                    self.scr.addnstr(y, 0, text.encode("ascii", "replace").decode(), max(0, room),
                                     self.attr["normal"])
                except curses.error:
                    pass
            x = 0
            while x < self.w:
                style, n = styles[x], 1
                while x + n < self.w and styles[x + n] == style:
                    n += 1
                if style != "normal":
                    try:
                        self.scr.chgat(y, x, n, self.attr.get(style, 0))
                    except curses.error:
                        pass
                x += n
        self.prev = ([row[:] for row in self.rows], [st[:] for st in self.styles])
        self.scr.refresh()


def fit(s, n):
    """`s` cut to `n` columns with a trailing ellipsis when it is longer."""
    if n <= 0:
        return ""
    return s if len(s) <= n else (s[:n - 3] + "..." if n > 3 else s[:n])


def gb(n):
    return f"{n / GB:.1f} GB"


def shown_tps(v):
    """The speed the menu leads with: what that computer measured for it, else the estimate."""
    return (v.measured_here or {}).get("tps") or v.tps


def estimate_text(tps, lo=None, hi=None, runs=0):
    """How a speed estimate reads: one number once this computer's own runs corrected it, a range
    before that, since the engine's store size is known only when it starts."""
    if not tps:
        return "no speed estimate"
    if runs:
        return f"~{tps:.0f} tok/s (adjusted to this computer's measured runs)"
    if lo and hi and lo < hi * 0.85:
        return f"~{lo:.0f}-{hi:.0f} tok/s (estimate for this machine)"
    return f"~{tps:.0f} tok/s (estimate for this machine)"


def measured_here_text(m, where="here"):
    return f"measured {where}: {m['tps']:.1f} tok/s ({m['n']} replies, {m['date']})"


def size_text(b):
    return gb(b) if b >= 0.1 * GB else f"{b / 1e6:.0f} MB"


def wrap(s, n):
    return textwrap.wrap(s, max(10, n)) or [""]


def spin(frozen, t):
    return SPINNER[0] if frozen else SPINNER[int(t * 6) % 4]


def pulse(frozen, t):
    return "●" if frozen or int(t * 2) % 2 == 0 else "○"


def status_style(text):
    t = text.lower()
    if t.startswith("can't"):
        return "bad_dim"
    if t.startswith("needs"):
        return "warn"
    if t.startswith(("ready", "on disk")):
        return "good"
    if t.startswith("download"):
        return "blue"
    if t.startswith("running"):
        return "run"
    if t.startswith("made") or "to make" in t or t.startswith("prepare"):
        return "warn"
    return "accent"


def bar(frac, n, shine_at=None):
    """A text progress bar such as [###...], with a moving = through the filled part while work is
    under way."""
    full = int(round(max(0.0, min(1.0, frac)) * n))
    cells = ["#" if i < full else "." for i in range(n)]
    if shine_at is not None and full:
        cells[shine_at % full] = "="
    return "[" + "".join(cells) + "]"


def speed_bar(cv, y, x, frac, n):
    """A bar whose colour runs red to amber to green with speed; the empty part stays dim."""
    full = int(round(max(0.0, min(1.0, frac)) * n))
    for i in range(n):
        if i < full:
            f = i / max(1, n - 1)
            cv.put(y, x + i, "#", "bar_lo" if f < 0.34 else "bar_mid" if f < 0.67 else "bar_hi")
        else:
            cv.put(y, x + i, ".", "dim")


# ---------------------------------------------------------------- keys

def norm(k):
    """One name for a key whether curses gave a code or a character."""
    if isinstance(k, str):
        return {"\n": "ENTER", "\r": "ENTER", "\x1b": "ESC", "\t": "TAB", "\x7f": "BACKSPACE", "\b": "BACKSPACE",
                "\x03": "CTRL_C"}.get(k, k)
    if curses is None:
        return ""
    names = {curses.KEY_UP: "UP", curses.KEY_DOWN: "DOWN", curses.KEY_LEFT: "LEFT", curses.KEY_RIGHT: "RIGHT",
             curses.KEY_PPAGE: "PGUP", curses.KEY_NPAGE: "PGDN", curses.KEY_HOME: "HOME", curses.KEY_END: "END",
             curses.KEY_ENTER: "ENTER", curses.KEY_BACKSPACE: "BACKSPACE", curses.KEY_BTAB: "BTAB",
             curses.KEY_RESIZE: "RESIZE"}
    return names.get(k, "")


# ---------------------------------------------------------------- the app

class App:
    """All the menu's state: tick() moves background work along, draw(canvas) paints and key(k)
    reacts, returning "quit" or ("chat", url) when the runner has to do something only it can."""

    def __init__(self, backend, log=None, frozen=False):
        self.b = backend
        self.log = log or Log()
        self.frozen = frozen
        self.t0 = 0.0 if frozen else time.time()
        self.mach = None
        self.mode = "boot"            # boot devices models input confirm install start ready stop error
        self.boot_job = Job(self._boot)
        self.boot_skip = False
        self.boot_ready_at = None
        self.running = None
        self.live = None              # the running server's /status, polled while the ready screen shows
        self.live_at = 0.0
        self.dev_rows = []
        self.dev_cursor = 0
        self.dev_top = 0
        self.probe_job = None
        self.probe_more = []          # rows to check once the running check ends
        self.scan_job = None
        self.refresh_job = None       # this computer and the known devices read again (r)
        self.cert_job = None          # a shared device's certificate read before its code is asked
        self.dev = None
        self.models = []              # the chosen device's list
        self.found = []               # from a search or a folder (this computer)
        self.search_note = ""
        self.search_job = None
        self.models_job = None
        self.delete_job = None
        self.cursor = 0
        self.top = 0
        self.open = None              # the model whose variants are shown (a Model)
        self.picked = False
        self.moved = False
        self.recommended = None
        self.sel = None               # (Model, Variant) being confirmed / installed / started
        self.job = None
        self.stop_job = None
        self.run_dev = None           # the device whose model this menu is running
        self.dev_job = None           # an engine update or install on one device
        self.dev_job_dev = None
        self.dev_job_kind = ""
        self.pair_job = None          # pairing with a shared device
        self.progress_t = []          # (time, bytes) samples for the install speed
        self.input = None
        self.flash = ("", "dim", 0.0)
        self.err, self.err_engine = "", False

    # ---- background work

    def now(self):
        return 0.0 if self.frozen else time.time()

    def _boot(self, job):
        job.say("looking at this computer (the first run measures the GPU and the SSD)...")
        self.mach = self.b.info()
        up = self.b.attached()
        if up:
            self.running = up
        job.say("looking for your computers...")
        return self.b.device_list()

    def start_probe(self, rows=None):
        """Check rows (all by default) over ssh, never a host that only the network scan found."""
        rows = [d for d in (self.dev_rows if rows is None else rows) if d.kind != "scan"]
        if self.probe_job is not None:
            self.probe_more += rows
        elif rows:
            self.probe_job = Job(self._probe_all, rows)

    def _probe_all(self, job, rows):
        from concurrent.futures import ThreadPoolExecutor

        def one(d):
            if job.cancelled():
                return
            try:
                self.b.device_probe(d)
            except Exception as err:
                d.status, d.detail = "offline", f"could not check it ({type(err).__name__}: {err})"
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(one, rows))

    def refresh_devices(self):
        """r: read this computer and the known devices again, then ask before sweeping the network."""
        if self.refresh_job or self.probe_job:
            self.say("still checking the devices", "dim", 4)
        else:
            self.refresh_job = Job(lambda job: self.b.device_list())
            self.say("reading this computer and the known devices again...", "dim", 4)
        net = None if self.scan_job else self.b.sweep_subnet()
        if net:
            self.ask(f"Scan 254 addresses on {net} for other computers? y/n: ", self.answer_scan, back="devices")

    def answer_scan(self, text):
        if text.strip().lower() in ("y", "yes"):
            self.start_scan()
        else:
            self.say("no network scan", "dim", 4)

    def start_scan(self):
        if self.scan_job is None:
            self.scan_job = Job(lambda job: self.b.device_scan(job.say))

    def _merge_rows(self, fresh):
        """The refreshed rows, a computer already listed keeping its row and the model list read from it, then the rows the refresh did not list."""
        old, out = list(self.dev_rows), []
        for d in fresh:
            same = next((o for o in old if self._dev_names(o) & self._dev_names(d)), None)
            if same is not None:
                old.remove(same)
                vars(same).update({k: v for k, v in vars(d).items() if k not in ("models", "listed_at")})
                d = same
            out.append(d)
        return out + old

    def load_models(self):
        d = self.dev
        self.models, self.found, self.open = [], [], None
        self.cursor = self.top = 0
        self.picked = self.moved = False

        def work(job):
            for m in self.b.device_models(d, job.say):
                job.items.append(m)
        self.models_job = Job(work)

    def refresh_models(self):
        if self.models_job:
            return
        self.dev.models, self.dev.listed_at = [], 0.0
        self.b.rescan(self.dev)
        self.load_models()

    def enter_device(self):
        d = self.current_device()
        if d is None:
            return
        if d.status in ("no-key", "offline", "no-engine", "no-ssh", "engine-new"):
            self.say(f"{d.name}: {d.detail}", "bad", 12)
            return
        if d.kind == "shared" and d.status != "ready":
            self.start_pair(d)
            return
        self.dev = d
        self.mode = "models"
        if d.models:
            self.models, self.found = list(d.models), []
            self.picked = self.moved = False
            self.open = None
            self.cursor = self.top = 0
            self.pick_start()
        else:
            self.load_models()

    def start_pair(self, d):
        """Read the certificate pairing will pin, so its fingerprint shows beside the question for the code."""
        if self.cert_job is None:
            self.say(f"reading {d.name}'s certificate...", "dim", 30)
            self.cert_job = Job(lambda job: (d, self.b.pair_fingerprint(d)))

    def ask_code(self, d, fingerprint):
        self.flash = ("", "dim", 0.0)
        cert = f" (certificate {fingerprint[:16]}: check that computer shows the same)" if fingerprint else ""
        self.ask(f"Code shown on {d.name}{cert}: ", lambda code: self.do_pair(d, code), back="devices")

    def do_pair(self, d, code):
        self.say(f"pairing with {d.name}...", "dim", 60)
        self.pair_job = Job(lambda job: self.b.pair(d, code))

    def tick(self):
        if self.mode == "boot" and self.boot_job:
            j = self.boot_job
            if j.done:
                if j.error:
                    self.say(f"could not read the device list: {j.error}", "bad", 20)
                else:
                    self.dev_rows = list(j.result or [])
                self.boot_job = None
                self.start_probe()
                self.boot_ready_at = 0.0 if self.frozen else time.time() + 0.8
        if self.mode == "boot" and not self.boot_job and (
                self.boot_skip or self.frozen or time.time() >= (self.boot_ready_at or 0)):
            self.mode = "devices"
        if self.probe_job and self.probe_job.done:
            self.probe_job = None
            rows, self.probe_more = self.probe_more, []
            self.start_probe(rows)
        if self.refresh_job and self.refresh_job.done:
            j, self.refresh_job = self.refresh_job, None
            if j.error:
                self.say(f"could not read the device list: {j.error}", "bad", 12)
            else:
                self.dev_rows = self._merge_rows(j.result or [])
                self.start_probe()
        if self.cert_job and self.cert_job.done:
            j, self.cert_job = self.cert_job, None
            if j.error:
                self.say(f"pairing failed: {j.error}", "bad", 12)
            else:
                self.ask_code(*j.result)
        if self.scan_job and self.scan_job.done:
            j, self.scan_job = self.scan_job, None
            if j.error:
                self.say(f"the network scan failed: {j.error}", "bad", 12)
            else:
                known, added = set(), []
                for d in self.dev_rows:
                    known |= self._dev_names(d)
                for d in j.result or []:
                    names = self._dev_names(d)
                    if names & known:
                        continue
                    known |= names
                    added.append(d)
                self.dev_rows += added
                self.start_probe(added)
                self.say(f"the scan found {len(added)} more", "dim", 8)
        if self.models_job:
            j = self.models_job
            while len(self.models) < len(j.items):
                self.models.append(j.items[len(self.models)])
            if not self.picked and self.models:
                self.pick_start()
            if j.done:
                if j.error:
                    self.say(f"could not read {self.dev.name}'s models: {j.error}", "bad", 20)
                else:
                    if not self.moved:
                        self.pick_start()
                    if self.dev:
                        self.dev.models = list(self.models)
                        self.dev.listed_at = time.time()
                self.models_job = None
        if self.delete_job and self.delete_job.done:
            j, self.delete_job = self.delete_job, None
            if j.error:
                self.say(j.error, "bad", 12)
            else:
                self.say(j.result, "good", 10)
                self.refresh_models()
        if self.search_job and self.search_job.done:
            self.flash = ("", "dim", 0.0)
            if self.search_job.error:
                self.say(f"search failed: {self.search_job.error}", "bad", 12)
            else:
                self.found, skipped = self.search_job.result
                self.search_note = (f"{len(self.found)} found, {skipped} skipped "
                                    f"(not MLX safetensors, an unknown model type, or gated)")
                if self.found:
                    first = self.found[0]
                    self.open = first
                    # one repo (a name or a link): straight to its variants, ready for Enter
                    self.cursor = self.row_of(first, first.variants[0]) if len(self.found) == 1 and first.variants \
                        else self.row_of(first)
            self.search_job = None
        if self.mode == "install" and self.job:
            now = time.time()
            if not self.progress_t or now - self.progress_t[-1][0] >= 0.5:
                self.progress_t.append((now, self.b.progress(self.sel[1])))
                self.progress_t = self.progress_t[-12:]
            if self.job.done:
                if self.job.cancelled():
                    self.mode = "models"
                    self.say("stopped; the same choice resumes where it was", "warn", 8)
                elif self.job.error:
                    self.fail("The install stopped: " + self.job.error)
                else:
                    self.begin_start()
        elif self.mode == "start" and self.job and self.job.done:
            if self.job.cancelled():
                self.mode = "models"
            elif self.job.error:
                self.fail("The model did not start: " + self.job.error, engine=True)
            else:
                self.running = self.job.result
                m, v = self.sel
                v.ready, v.status = True, "ready"
                if self.dev:
                    self.dev.status = "running"
                    self.dev.running = f"{m.name} (from the menu)"
                self.mode = "ready"
        if self.stop_job and self.stop_job.done:
            j, self.stop_job = self.stop_job, None
            if j.error:
                self.say(f"could not stop it: {j.error}", "bad", 12)
                self.mode = "ready"
            else:
                self.running = None
                self.live = None
                if self.dev:
                    self.dev.status, self.dev.running = "ready", ""
                self.mode = "models" if self.dev else "devices"
                self.say("the model is stopped; the other computer released its lock", "dim", 8)
        if self.mode == "ready" and (self.live is None or time.time() - self.live_at >= 1.0):
            self.live_at = time.time()
            try:
                self.live = self.b.live()
            except Exception:
                self.live = None
        if self.dev_job and self.dev_job.done:
            j, self.dev_job = self.dev_job, None
            d, kind, self.dev_job_dev = self.dev_job_dev, self.dev_job_kind, None
            verb = "install" if kind == "install" else "update"
            if j.cancelled():
                self.mode = "devices"
                self.say("stopped", "warn", 6)
            elif j.error:
                self.mode = "devices"
                self.say(f"could not {verb} {d.name}: {j.error}", "bad", 12)
            else:
                self.mode = "devices"
                self.say(j.result or f"{d.name} {verb}ed", "good", 8)
                d.status, d.detail = "checking", ""
                self.reprobe(d)
        if self.pair_job and self.pair_job.done:
            j, self.pair_job = self.pair_job, None
            if j.error:
                self.mode = "devices"
                self.say(f"pairing failed: {j.error}", "bad", 12)
            else:
                self.say(f"paired with {j.result.name}", "good", 8)
                self.mode = "devices"
                self.enter_device()

    @staticmethod
    def _dev_names(d):
        return {x for x in (d.name.lower(), d.target.lower(), *d.aliases) if x}

    def pick_start(self):
        """Put the cursor on the variant used last time (this computer), else the fastest
        published one that runs there, one already on disk first."""
        self.picked = True
        last = self.b.last() if self.dev and self.dev.kind == "local" else None
        best, best_key = None, None
        for m in self.all_models():
            for v in m.variants:
                key = (bool(last) and v.entry_id == last, v.ready, v.tps or 0.0)
                if (v.ok and v.key == "published" or key[0]) and (best_key is None or key > best_key):
                    best, best_key = (m, v), key
        self.recommended = best[1] if best else None
        if best:
            self.open = best[0]
            self.cursor = self.row_of(*best)

    def say(self, text, style="dim", secs=5):
        self.flash = (text, style, time.time() + secs)

    def fail(self, text, engine=False):
        """Show what went wrong with the output that explains it: the engine's log, or the
        install's."""
        self.err, self.err_engine = text, engine
        self.mode = "error"
        self.b.abort()

    # ---- rows of the model list

    def all_models(self):
        return self.models + self.found

    def rows(self):
        out = []
        for m in self.all_models():
            out.append(("model", m))
            if self.open is m:
                out += [("variant", m, v) for v in m.variants]
        return out

    def row_of(self, model, variant=None):
        """Index of a model's row (or one of its variants') in rows(); compared by identity."""
        for i, r in enumerate(self.rows()):
            if r[1] is model and (variant is None) == (r[0] == "model") and (variant is None or r[2] is variant):
                return i
        return 0

    def current(self):
        rows = self.rows()
        if not rows:
            return None
        self.cursor = max(0, min(self.cursor, len(rows) - 1))
        return rows[self.cursor]

    def current_device(self):
        if not self.dev_rows:
            return None
        self.dev_cursor = max(0, min(self.dev_cursor, len(self.dev_rows) - 1))
        return self.dev_rows[self.dev_cursor]

    # ---- keys

    def key(self, k):
        k = norm(k)
        if k == "CTRL_C":
            return "quit"
        if k == "RESIZE":
            return None
        return {"boot": self.key_boot, "devices": self.key_devices, "models": self.key_models,
                "input": self.key_input, "confirm": self.key_confirm, "install": self.key_install,
                "start": self.key_start, "ready": self.key_ready, "stop": self.key_stop,
                "update": self.key_update, "enginstall": self.key_update, "error": self.key_error}[self.mode](k)

    def key_boot(self, k):
        self.boot_skip = True
        return None

    def key_devices(self, k):
        if k in ("q", "ESC"):
            return "quit"
        if k in ("UP", "k"):
            self.dev_cursor = max(0, self.dev_cursor - 1)
        elif k in ("DOWN", "j"):
            self.dev_cursor = min(max(0, len(self.dev_rows) - 1), self.dev_cursor + 1)
        elif k == "PGUP":
            self.dev_cursor = max(0, self.dev_cursor - 8)
        elif k == "PGDN":
            self.dev_cursor = min(max(0, len(self.dev_rows) - 1), self.dev_cursor + 8)
        elif k == "HOME":
            self.dev_cursor = 0
        elif k == "END":
            self.dev_cursor = max(0, len(self.dev_rows) - 1)
        elif k in ("ENTER", "RIGHT"):
            self.enter_device()
        elif k == "r":
            self.refresh_devices()
        elif k == "u" and self.current_device() is not None and self.current_device().status == "engine-old":
            self.start_update()
        elif k == "i" and self.current_device() is not None and self.current_device().status == "no-engine":
            self.start_install()
        elif k == "g" and self.running:
            self.mode = "ready"
        return None

    def key_models(self, k):
        if k == "q":
            return "quit"
        if k in ("ESC", "LEFT"):
            if self.open:
                self.open = None
            else:
                self.mode = "devices"
            return None
        if k == "g" and self.running:
            self.mode = "ready"
            return None
        if k == "r":
            self.refresh_models()
            return None
        rows = self.rows()
        cur = self.current()
        self.moved = self.moved or k in ("UP", "DOWN", "j", "k", "PGUP", "PGDN", "HOME", "END")
        if k in ("UP", "k"):
            self.cursor = max(0, self.cursor - 1)
        elif k in ("DOWN", "j"):
            self.cursor = min(len(rows) - 1, self.cursor + 1)
        elif k == "PGUP":
            self.cursor = max(0, self.cursor - 8)
        elif k == "PGDN":
            self.cursor = min(len(rows) - 1, self.cursor + 8)
        elif k == "HOME":
            self.cursor = 0
        elif k == "END":
            self.cursor = max(0, len(rows) - 1)
        elif k in ("RIGHT", "ENTER") and cur and cur[0] == "model":
            m = cur[1]
            if m.unavailable:
                self.say(f"{m.name} can't run there: {m.unavailable}", "bad", 10)
            elif k == "ENTER" and self.open is m:
                self.open = None
            else:
                self.open = m
                self.cursor = self.row_of(m, m.variants[0]) if m.variants else self.cursor
        elif k == "ENTER" and cur and cur[0] == "variant":
            self.choose_variant(cur[1], cur[2])
        elif k == "d" and cur and cur[0] == "variant":
            self.ask_delete(cur[1], cur[2])
        elif k == "/" and self.dev:
            on = "" if self.dev.kind == "local" else f" for {self.dev.name}"
            self.ask(f"Search Hugging Face{on} (words, owner/name or a link): ", self.do_search)
        elif k == "f" and self.dev and self.dev.kind == "local":
            self.ask("Model folder on this computer: ", self.do_local)
        return None

    def stop_current(self):
        """Stop the model the launcher owns before starting another, and clear the row it ran on."""
        self.b.stop()
        self.running = None
        if self.run_dev:
            self.run_dev.status, self.run_dev.running = "ready", ""
        else:
            for d in self.dev_rows:               # a local model an earlier run left running
                if d.kind == "local":
                    d.status, d.running = "ready", ""
        self.run_dev = None

    def choose_variant(self, m, v):
        """Enter on a variant: stop the running model, start this one there, ask before a
        download, or say why not."""
        if self.running:
            try:
                self.stop_current()
            except RuntimeError as err:
                self.say(f"could not stop the running model: {err}", "bad", 12)
                return
        if v.disk_short:
            self.say(f"{v.why}: free some space, or point --models-dir at a bigger disk", "warn", 10)
        elif not v.ok:
            self.say(f"can't run that one there: {v.why}", "bad", 10)
        elif self.dev.status == "running" and not (
                self.running and self.running.get("device") == self.dev.name):
            self.say(f"{self.dev.name} already runs a model: {self.dev.running}; stop it there first",
                     "warn", 10)
        elif self.dev.kind == "local":
            self.sel = (m, v)
            self.mode = "confirm"
        elif v.ready:
            self.sel = (m, v)
            self.begin_start()             # on the other computer: ready means start, no questions
        elif self.dev.kind == "preset":
            self.say(f"{self.dev.name}'s preset command can't prepare models: prepare {m.name} "
                     f"there with its own launcher first", "bad", 12)
        else:
            self.sel = (m, v)
            self.mode = "confirm"          # its launcher downloads and starts without questions

    def ask_delete(self, m, v):
        """d on a variant: ask before deleting it, or say why it cannot be deleted."""
        where = self.dev.name if self.dev and self.dev.kind != "local" else "this computer"
        r = self.running
        if not v.path:
            self.say(f"{m.name} {v.label} is not on {where}'s disk: nothing to delete", "warn", 8)
        elif not v.own:
            self.say(f"{m.name} {v.label} is at {v.path}, outside the launcher's models folder: the launcher "
                     f"deletes only what it downloaded or made", "warn", 14)
        elif r and (r.get("model") == v.path or (r.get("name") == m.name and r.get("device", "") == (
                self.dev.name if self.dev.kind != "local" else ""))):
            self.say(f"{m.name} is running: press g, then s to stop it first", "warn", 8)
        else:
            self.ask(f"Delete {m.name} {v.label} at {v.path} ({size_text(v.size)})? type yes: ",
                     lambda text: self.do_delete(m, v, text))

    def do_delete(self, m, v, answer):
        if answer.strip().lower() != "yes":
            self.say("not deleted", "dim", 5)
            return
        self.say(f"deleting {v.path}...", "dim", 60)
        self.delete_job = Job(lambda job: self.b.delete(self.dev, m, v))

    def ask(self, prompt, then, back="models"):
        self.input = {"prompt": prompt, "buf": "", "then": then, "back": back}
        self.mode = "input"

    def key_input(self, k):
        inp = self.input
        back = inp.get("back", "models")
        if k == "ESC":
            self.mode = back
        elif k == "ENTER":
            self.mode = back
            if inp["buf"].strip():
                inp["then"](inp["buf"].strip())
        elif k == "BACKSPACE":
            inp["buf"] = inp["buf"][:-1]
        elif len(k) == 1 and k.isprintable():
            inp["buf"] += k
        return None

    def do_search(self, q):
        dev = self.dev
        self.search_job = Job(lambda job: self.b.device_search(dev, q, job.say))
        self.say("searching Hugging Face..." if dev.kind == "local" else
                 f"asking {dev.name} to search Hugging Face...", "dim", 60)

    def do_local(self, path):
        try:
            m = self.b.local(path)
        except ValueError as err:
            self.say(f"can't use it: {err}", "bad", 10)
            return
        m.found = True
        self.found = [m]
        self.open = m
        self.cursor = self.row_of(m)

    def key_confirm(self, k):
        if k in ("ENTER", "y"):
            m, v = self.sel
            if self.dev.kind != "local" or v.ready:
                self.begin_start()
            else:
                self.mode = "install"
                self.progress_t = []
                self.job = Job(lambda job: self.b.install(v, job))
        elif k in ("ESC", "n", "q", "LEFT"):
            self.mode = "models"
        return None

    def begin_start(self):
        m, v = self.sel
        self.mode = "start"
        self.run_dev = self.dev
        self.job = Job(lambda job: self.b.start_device(self.dev, m, v, job))

    def key_install(self, k):
        if k in ("ESC", "q") and self.job:
            self.job.cancel()
            self.b.abort()                # also ends the tool the install runs
        return None

    def key_start(self, k):
        if k in ("ESC", "q") and self.job:
            self.job.cancel()
            self.b.abort()
        return None

    def key_ready(self, k):
        if k == "o":
            self.b.open_chat(self.running["url"])
        elif k == "c":
            return ("chat", self.running["url"])
        elif k == "s":
            self.mode = "stop"
            self.stop_job = Job(lambda job: self.b.stop())
        elif k == "m":
            self.mode = "devices"
            if self.dev:
                self.dev.status = "running"
                self.dev.running = f"{self.running['name']} (from the menu)"
        elif k in ("q", "ESC"):
            return "quit"
        return None

    def key_stop(self, k):
        return None

    def start_update(self):
        d = self.current_device()
        self.dev_job_dev = d
        self.dev_job_kind = "update"
        self.mode = "update"
        self.dev_job = Job(lambda job: self.b.update_device(d, job))

    def start_install(self):
        d = self.current_device()
        self.dev_job_dev = d
        self.dev_job_kind = "install"
        self.mode = "enginstall"
        self.dev_job = Job(lambda job: self.b.install_device(d, job))

    def reprobe(self, d):
        """Ask one device's status again (after an update or install)."""
        self.start_probe([d])

    def key_update(self, k):
        if k in ("ESC", "q") and self.dev_job:
            self.dev_job.cancel()
        return None

    def key_error(self, k):
        self.mode = "models" if self.dev else "devices"
        if self.mode == "models" and self.dev and self.dev.models:
            self.models = list(self.dev.models)
        return None

    # ---- drawing

    def draw(self, cv):
        cv.clear()
        h, w = cv.size()
        if h < 14 or w < 56:
            cv.put(0, 0, "Make the terminal larger (at least 56 x 14).", "warn")
            return
        if self.mode == "boot":
            self.draw_boot(cv, h, w)
            return
        self.topbar(cv, w)
        if self.mode in ("devices", "models", "input"):
            if self.mode == "devices":
                self.draw_devices(cv, h, w)
            else:
                self.draw_models(cv, h, w)
            if self.mode == "input":
                self.footer_input(cv, h, w)
            else:
                self.footer(cv, h, w, self.keys_for_mode())
        elif self.mode == "ready":
            self.draw_ready(cv, h, w)
        else:
            {"confirm": self.draw_confirm, "install": self.draw_install, "start": self.draw_start,
             "stop": self.draw_stop, "update": self.draw_update, "enginstall": self.draw_enginstall,
             "error": self.draw_error}[self.mode](cv, h, w)

    def trail(self):
        parts = ["devices"]
        if self.dev and self.mode not in ("devices",):
            parts.append(self.dev.name)
        if self.mode in ("install", "start"):
            parts.append("starting")
        elif self.mode == "update":
            parts.append("updating the engine")
        elif self.mode == "enginstall":
            parts.append("installing the engine")
        elif self.mode == "ready":
            parts.append("running")
        elif self.mode == "stop":
            parts.append("stopping")
        return " / ".join(parts)

    def topbar(self, cv, w):
        cv.fill(0, 0, w, "topbar")
        cv.put(0, 1, " larder ", "topbar")
        cv.put(0, 9, fit("  " + self.trail(), w - 11), "topbar")
        if self.running:
            right = f"{pulse(self.frozen, self.now())} running: {self.running['name']}  (g) "
            cv.put(0, max(0, w - len(right) - 1), fit(right, w - 1), "topbar")

    def draw_boot(self, cv, h, w):
        t = 0.0 if self.frozen else min(1.0, time.time() - self.t0)
        shown = "larder" if self.frozen else "larder"[:max(0, min(6, int(t * 24)))]
        cv.put(2, 3, "  ".join(shown), "accent")
        cv.put(4, 3, fit("run big language models on any of your computers", w - 6), "dim")
        phase = self.boot_job.phase if self.boot_job else ""
        if phase:
            cv.put(6, 3, fit(f"{spin(self.frozen, self.now())} {phase}", w - 6), "dim")
        cv.put(8, 3, bar(t if not self.frozen else 1.0, 24, 0 if self.frozen else int(t * 40)), "accent")
        cv.put(h - 1, 3, "a key skips the title", "dim")

    def draw_devices(self, cv, h, w):
        cv.put(1, 1, fit("pick the computer, then the model it runs", w - 2), "dim")
        cv.put(2, 0, "-" * w, "dim")
        cv.put(3, 2, f"{'DEVICE':<20}{'HARDWARE':<28}STATUS", "dim")
        if not self.dev_rows:
            if self.boot_job:
                cv.put(5, 2, f"{spin(self.frozen, self.now())} {self.boot_job.phase or 'looking...'}", "dim")
            else:
                cv.put(5, 2, "no computers found", "dim")
            return
        top, bottom = 4, h - 5
        span = max(1, bottom - top)
        self.dev_cursor = max(0, min(self.dev_cursor, len(self.dev_rows) - 1))
        if self.dev_cursor < self.dev_top:
            self.dev_top = self.dev_cursor
        if self.dev_cursor >= self.dev_top + span:
            self.dev_top = self.dev_cursor - span + 1
        for i, d in enumerate(self.dev_rows[self.dev_top:self.dev_top + span]):
            y = top + i
            cur = self.dev_top + i == self.dev_cursor
            status = self.device_status(d)
            left = f" {'>' if cur else ' '} {fit(d.name, 18):<18} {fit(d.hardware or '-', 27):<27} "
            cv.put(y, 0, fit(left + status, w), "title" if d.kind == "local" else "normal")
            cv.put(y, len(left), fit(status, w - len(left)), self.device_style(d))
            if cur:
                cv.fill(y, 0, w, "sel")
        if self.dev_top > 0:
            cv.put(top, w - 2, "^", "dim")
        if self.dev_top + span < len(self.dev_rows):
            cv.put(bottom - 1, w - 2, "v", "dim")
        d = self.current_device()
        cv.put(h - 5, 0, "-" * w, "dim")
        if self.scan_job:
            cv.put(h - 4, 2, fit(f"{spin(self.frozen, self.now())} "
                                 f"{self.scan_job.phase or 'sweeping the network...'}", w - 3), "accent")
        elif d:
            src = f"  ({d.source})" if d.source else ""
            lines = wrap(f"{d.detail or self.device_status(d)}{src}", w - 4)
            for i, line in enumerate(lines[:2]):
                cv.put(h - 4 + i, 2, fit(line, w - 3), "dim" if not d.detail else "normal")
            if d.status in ("ready", "running") or d.kind == "local":
                cv.put(h - 2, 2, "Enter shows the models this computer can run.", "good")
            elif d.status in ("no-key", "offline", "no-engine", "no-ssh"):
                cv.put(h - 2, 2, fit(d.detail, w - 3), "bad" if d.status != "no-engine" else "warn")

    @staticmethod
    def device_status(d):
        if d.status == "running" and d.running:
            return f"running: {d.running}"
        if d.status == "engine-old":
            return "engine is older: u updates it"
        if d.status == "engine-new":
            return "update this computer's engine first"
        return {"checking": "checking...", "ready": "ready", "no-key": "no ssh key",
                "no-engine": "engine not installed", "offline": "offline", "ssh-open": "ssh open",
                "no-ssh": "no ssh"}.get(d.status, d.status)

    @staticmethod
    def device_style(d):
        if d.status == "ready":
            return "good"
        if d.status == "running":
            return "run"
        if d.status == "checking":
            return "dim"
        if d.status in ("no-engine", "engine-old"):
            return "warn"
        if d.status == "engine-new":
            return "accent"
        if d.status in ("no-key", "offline"):
            return "bad"
        if d.status == "no-ssh":
            return "dim"
        return "accent"

    def keys_for_mode(self):
        if self.mode == "devices":
            keys = [("Up/Down", "move"), ("Enter", "models"), ("r", "refresh or scan")]
            if self.current_device() is not None and self.current_device().status == "engine-old":
                keys.append(("u", "update its engine"))
            if self.current_device() is not None and self.current_device().status == "no-engine":
                keys.append(("i", "install the engine"))
            if self.running:
                keys.append(("g", "the running model"))
            keys.append(("q", "quit"))
            return keys
        keys = [("Up/Down", "move"), ("Enter", "choose"), ("r", "refresh"), ("/", "search"), ("d", "delete")]
        if self.dev and self.dev.kind == "local":
            keys.append(("f", "folder"))
        keys += [("Esc", "back"), ("q", "quit")]
        if self.running:
            keys.append(("g", "running"))
        return keys

    def footer(self, cv, h, w, keys):
        text, style, until = self.flash
        if text and time.time() < until:
            cv.put(h - 1, 1, fit(text, w - 2), style)
            return
        x = 1
        for k, what in keys:
            if x + len(k) + len(what) + 2 > w - 1:
                break
            cv.put(h - 1, x, k, "key")
            cv.put(h - 1, x + len(k) + 1, what, "dim")
            x += len(k) + len(what) + 4

    def footer_input(self, cv, h, w):
        prompt, buf = self.input["prompt"], self.input["buf"] + "_"
        if len(prompt) + len(buf) <= w - 2:
            cv.put(h - 1, 1, prompt + buf, "accent")
        else:                       # a long question takes its own lines, the answer goes under them
            rows = wrap(prompt, w - 2)[-3:]
            for i, line in enumerate(rows):
                cv.put(h - 1 - len(rows) + i, 1, line.ljust(w - 2), "accent")
            cv.put(h - 1, 1, fit("> " + buf[-(w - 6):], w - 2), "accent")

    # models screen

    def draw_models(self, cv, h, w):
        d = self.dev
        if d:
            line = d.name + (f"   {d.hardware}" if d.hardware else "") + (f"   {d.detail}" if d.detail else "")
            cv.put(1, 1, fit(line, w - 2), "dim")
            info = d.info or {}
            if info:
                ssd = f", the SSD reads {info['ssd_gbps']:.1f} GB/s" if info.get("ssd_gbps") else ""
                cv.put(2, 1, fit(f"models in {info.get('models_dir', '?')}: "
                                 f"{info.get('free_gb', 0):.0f} GB free{ssd}", w - 2), "dim")
            else:
                cv.put(2, 1, fit("its own list, as that computer computes it; r asks again", w - 2), "dim")
        else:
            cv.put(1, 1, "models", "dim")
        cv.put(3, 0, "-" * w, "dim")
        top, bottom = 4, h - 10
        rows = self.rows()
        if not rows:
            if self.models_job:
                cv.put(top + 1, 2, f"{spin(self.frozen, self.now())} "
                                   f"{self.models_job.phase or 'checking the models...'}", "dim")
            else:
                cv.put(top + 1, 2, "no models to show", "dim")
            return
        cur = self.current()
        fastest = max([shown_tps(v) or 0 for m in self.all_models() for v in m.variants] or [1]) or 1
        span = max(1, bottom - top)
        self.cursor = max(0, min(self.cursor, len(rows) - 1))
        if self.cursor < self.top:
            self.top = self.cursor
        if self.cursor >= self.top + span:
            self.top = self.cursor - span + 1
        for i, row in enumerate(rows[self.top:self.top + span]):
            y = top + i
            cur_row = self.top + i == self.cursor
            if row[0] == "model":
                m = row[1]
                n = self.all_models().index(m) + 1
                mark = ">" if self.open is m else " "
                tag = "  (tested)" if m.tested else ""
                line = f" {mark}{n:>2}  {m.name}{tag}   {m.kind or 'unavailable here'}"
                cv.put(y, 0, fit(line, w), "dim" if m.unavailable else "title")
            else:
                m, v = row[1], row[2]
                idx = "abcdefghij"[m.variants.index(v)]
                left = f"      {idx}  {fit(v.label, 24):<24} {gb(v.size):>9}  "
                cv.put(y, 0, left, "normal")
                x = len(left)
                if v.measured_here:
                    cv.put(y, x, fit(f"{v.measured_here['tps']:.1f} tok/s measured here", 26), "good")
                elif v.measured:
                    cv.put(y, x, fit(f"[{v.measured[0]:.1f}-{v.measured[1]:.1f} tok/s measured]", 26), "good")
                elif v.tps:
                    cv.put(y, x, fit(f"~{v.tps:.0f} tok/s (estimate)", 26), "dim")
                else:
                    cv.put(y, x, fit("--", 26), "dim")
                x += 27
                if w >= 100 and shown_tps(v) and v.ok:
                    speed_bar(cv, y, x, shown_tps(v) / fastest, 8)
                    x += 9
                rec = "  <- start here" if v is self.recommended and v.ok else ""
                cv.put(y, x, fit(v.status + rec, w - x - 1), status_style(v.status))
            if cur_row:
                cv.fill(y, 0, w, "sel")
        if self.top > 0:
            cv.put(top, w - 2, "^", "dim")
        if self.top + span < len(rows):
            cv.put(bottom - 1, w - 2, "v", "dim")
        self.draw_details(cv, h, w, cur)

    def draw_details(self, cv, h, w, cur):
        y = h - 9
        cv.put(y, 0, "-" * w, "dim")
        if not cur:
            return
        lines = []
        if cur[0] == "model":
            m = cur[1]
            lines.append((f"{m.name}{'  (tested with this engine)' if m.tested else ''}", "title"))
            lines.append((" | ".join(x for x in (m.kind, m.about, f"license {m.license}" if m.license else "") if x),
                          "dim"))
            if m.unavailable:
                lines += [(x, "bad_dim") for x in wrap("Can't run there: " + m.unavailable, w - 4)]
            else:
                lines.append(("Enter or Right opens the variants.", "dim"))
        else:
            m, v = cur[1], cur[2]
            lines.append((f"{m.name} | {v.label} | {gb(v.size)} on disk", "title"))
            where = self.dev.name if self.dev else "this computer"
            cond = (self.dev.info.get("conditions_text") if self.dev else None) \
                or (self.mach or {}).get("conditions_text")
            if cond:
                lines.append((cond, "warn" if "expect slower" in cond else "dim"))
            if v.measured:
                lines.append((f"measured on {where}: {v.measured[0]:.1f}-{v.measured[1]:.1f} tok/s at its preset",
                              "good"))
            elif v.tps:
                mtp = ", its MTP head is used: about 25% more" if v.mtp else ""
                lines.append((estimate_text(v.tps, v.tps_lo, v.tps_hi, v.tps_adjusted) + mtp, "dim"))
            else:
                lines.append((f"no speed estimate from {where}", "dim"))
            if v.measured_here:
                lines.append((measured_here_text(v.measured_here, "here" if not self.dev or self.dev.kind == "local"
                                                 else "on " + where), "good"))
            at = f" - {v.path}" if v.path and not v.status.startswith("on disk at") else ""
            lines += [(x, status_style(v.status)) for x in wrap(v.status + at, w - 4)]
            if v.disk_short:
                lines.append(("It would run here; only the disk is short. Free some space or use --models-dir.",
                              "dim"))
            if v.note:
                lines += [(x, "dim") for x in wrap(v.note, w - 4)]
            lines += [(f"- {x}", "dim") for nm in m.not_made for x in wrap(nm, w - 6)]
            if v.ok:
                if self.dev and self.dev.kind != "local":
                    if not v.ready and self.dev.kind == "preset":
                        lines.append((f"{self.dev.name}'s preset command can't prepare models: prepare it there "
                                      f"with its own launcher first.", "warn"))
                    elif v.ready:
                        lines.append((f"Enter starts it on {self.dev.name} and opens its chat here through a tunnel.",
                                      "good"))
                    else:
                        lines.append((f"Enter prepares and starts it on {self.dev.name} (its launcher does both "
                                      f"without questions).", "good"))
                else:
                    lines.append(("Enter " + ("starts it." if v.ready else
                                              "installs it and starts it (you confirm first)."), "good"))
        note = bool(self.search_note and self.found)
        for i, (t, st) in enumerate(lines[:6 if note else 7]):
            cv.put(y + 1 + i, 2, fit(t, w - 3), st)
        if note:
            cv.put(h - 2, 2, fit(self.search_note, w - 3), "dim")

    # install / start / ready

    def draw_confirm(self, cv, h, w):
        m, v = self.sel
        remote = self.dev is not None and self.dev.kind != "local"
        where = self.dev.name if self.dev else "this computer"
        title = "Start" if v.ready else ("Prepare and start on " + where if remote else "Install and start")
        cv.put(3, 2, fit(title, w - 4), "title")
        cv.put(4, 2, fit(f"{m.name} - {v.label}", w - 4), "accent")
        rows = []
        if not v.ready:
            rows.append(("Download", gb(v.download) if v.download else "nothing (made from a model already there)"))
            if remote:
                rows.append(("On disk", f"{gb(v.disk or v.size)} on {where}"))
            else:
                rows.append(("On disk", f"{gb(v.disk or v.size)} in {self.b.dest(v)}"))
        else:
            rows.append(("On disk", f"{gb(v.size)} on {where}" + ("" if remote else f" in {self.b.dest(v)}")
                         + " (already there)"))
        info = (self.dev.info if self.dev else None) or {}
        free = info.get("free_gb") if info else (self.mach or {}).get("free_gb")
        if free is not None:
            rows.append(("Free disk", f"{free:.0f} GB on {where}" if remote else f"{free:.0f} GB"))
        if v.measured:
            rows.append(("Speed", f"{v.measured[0]:.1f}-{v.measured[1]:.1f} tok/s measured on {where}"))
        elif v.tps:
            rows.append(("Speed", estimate_text(v.tps, v.tps_lo, v.tps_hi, v.tps_adjusted)))
        else:
            rows.append(("Speed", "no estimate"))
        if v.measured_here:
            rows.append(("Measured", f"{v.measured_here['tps']:.1f} tok/s on {where} ({v.measured_here['n']} replies, "
                                     f"{v.measured_here['date']})"))
        for i, (a, b) in enumerate(rows):
            cv.put(5 + i, 2, f"{a:<10}", "dim")
            cv.put(5 + i, 13, fit(b, w - 15), "normal")
        if v.make_bits and not v.ready:
            warn = (f"This makes a {v.make_bits}-bit copy: smaller, with a quality loss ({v.note}). "
                    f"The model as published is kept.")
            for i, line in enumerate(wrap(warn, w - 4)):
                cv.put(7 + len(rows) + i, 2, line, "warn")
        if remote and not v.ready:
            note = (f"{where}'s own launcher downloads it and starts it without questions; the chat comes "
                    f"to this computer through a tunnel.")
            for i, line in enumerate(wrap(note, w - 4)):
                cv.put(6 + len(rows) + i, 2, line, "dim")
        cv.put(h - 1, 1, "Enter", "key")
        cv.put(h - 1, 7, "continue", "dim")
        cv.put(h - 1, 18, "Esc", "key")
        cv.put(h - 1, 22, "back", "dim")

    def log_pane(self, cv, y, h, w, lines):
        n = max(0, h - 2 - y)
        for i, line in enumerate(lines[-n:] if n else []):
            cv.put(y + i, 4, fit(line, w - 6), "dim")

    def draw_install(self, cv, h, w):
        m, v = self.sel
        cv.put(5, 2, f"Installing {m.name} ({v.label})", "title")
        cv.put(6, 2, fit(f"into {self.b.dest(v)}", w - 4), "dim")
        have = self.progress_t[-1][1] if self.progress_t else 0
        total = v.expected or v.disk or v.size or 1
        frac = min(0.99, have / total) if have else 0
        rate = 0.0
        if len(self.progress_t) >= 2:
            t0, b0 = self.progress_t[0]
            t1, b1 = self.progress_t[-1]
            rate = (b1 - b0) / (t1 - t0) if t1 > t0 else 0.0
        eta = f"  about {self.fmt_time((total - have) / rate)} left" if rate > 1e5 and have < total else ""
        cv.put(8, 2, f"{bar(frac, max(10, min(40, w - 40)), 0 if self.frozen else int(self.now() * 6))} "
                     f"{frac * 100:3.0f}%  {size_text(have)} of {size_text(total)}"
                     + (f"  {rate / 1e6:.0f} MB/s" if rate > 1e5 else "") + eta, "accent")
        cv.put(9, 2, f"{spin(self.frozen, self.now())} {self.job.phase or 'working...'}  "
                     f"({self.job.seconds():.0f} s)", "dim")
        cv.put(11, 2, "Latest output:", "dim")
        self.log_pane(cv, 12, h, w, self.log.tail(max(1, h - 14)))
        cv.put(h - 1, 1, "Esc", "key")
        cv.put(h - 1, 5, "stop (the next run resumes)", "dim")

    @staticmethod
    def fmt_time(s):
        s = int(s)
        return f"{s // 3600} h {s % 3600 // 60} min" if s >= 3600 else f"{s // 60} min" if s >= 90 else f"{s} s"

    def draw_start(self, cv, h, w):
        m, v = self.sel
        where = self.dev.name if self.dev else "this computer"
        cv.put(5, 2, fit(f"Starting {m.name} on {where}", w - 4), "title")
        cv.put(7, 2, f"{spin(self.frozen, self.now())} {self.job.phase or 'starting the engine...'}", "accent")
        if self.dev and self.dev.kind != "local":
            cv.put(8, 2, fit("the remote's own lines below; the tunnel opens here when it is serving",
                             w - 4), "dim")
        cv.put(10, 2, "The engine's log:", "dim")
        lines = self.b.log_tail(max(1, h - 13)) or self.log.tail(max(1, h - 13))
        self.log_pane(cv, 11, h, w, lines)
        cv.put(h - 1, 1, "Esc", "key")
        cv.put(h - 1, 5, "stop", "dim")

    def speed_line(self, r, live):
        """(what the speed reads now, the estimate it started from, style) for the ready screen."""
        est = r.get("est") or {}
        n = (live or {}).get("decode_replies") or 0
        tps = (live or {}).get("decode_tok_s")
        if n and tps:
            row = f"{tps:.1f} tok/s measured over {n} {'reply' if n == 1 else 'replies'}"
            was = f" (the estimate was ~{est['tps']:.0f})" if est.get("tps") else ""
            return row, was, "good"
        was = estimate_text(est.get("tps"), est.get("lo_tps"), est.get("hi_tps"), est.get("runs")) \
            if est.get("tps") else "no estimate"
        return was, ", no reply timed yet", "normal"

    def draw_ready(self, cv, h, w):
        r = self.running
        live = self.live or {}
        cv.put(3, 2, fit(f"Ready: {r['name']}", w - 4), "good")
        if r.get("device"):
            cv.put(4, 2, fit(f"on {r['device']}; the chat comes to this computer through an ssh tunnel",
                             w - 4), "run")
        cv.put(6, 2, "Chat in the browser", "dim")
        cv.put(6, 24, r["url"] + "/", "title")
        cv.put(7, 2, "API base", "dim")
        cv.put(7, 24, r["api"] + "   (OpenAI-compatible, no key needed)", "normal")
        cv.put(8, 2, "Model id", "dim")
        cv.put(8, 24, fit(r.get("model_id", ""), w - 26), "normal")
        if r.get("measured"):
            cv.put(9, 2, "Speed", "dim")
            cv.put(9, 24, f"{r['measured'][0]:.1f}-{r['measured'][1]:.1f} tok/s measured on "
                          f"{r.get('device', 'it')}", "good")
        else:
            now, was, style = self.speed_line(r, live)
            cv.put(9, 2, "Speed", "dim")
            cv.put(9, 24, fit(now + was, w - 26), style)
        cond = live.get("conditions_text") or (self.mach or {}).get("conditions_text")
        if cond:
            cv.put(10, 2, "Conditions", "dim")
            cv.put(10, 24, fit(cond, w - 26), "warn" if "expect slower" in cond else "dim")
        held = self.engine_line(live)
        if held:
            cv.put(11, 2, "Engine", "dim")
            cv.put(11, 24, fit(held, w - 26), "dim")
        cv.put(12, 2, "Any OpenAI client: base_url=" + f'"{r["api"]}"' + ', api_key="none"', "dim")
        cv.put(13, 2, "The chat address is the same for every model and computer.", "dim")
        x = 1
        left = "leave it running and quit" if r.get("attached") else "stop and quit"
        for k, what in [("o", "open the chat"), ("c", "chat here"), ("s", "stop it"), ("m", "devices"),
                        ("q", left)]:
            cv.put(h - 1, x, k, "key")
            cv.put(h - 1, x + 2, what, "dim")
            x += len(what) + 5

    @staticmethod
    def engine_line(live):
        """What the engine's own store holds and how many expert rows it reads from the SSD."""
        live = live or {}
        arena, store, exp = live.get("arena") or {}, live.get("store") or {}, live.get("experts") or {}
        parts = []
        if arena.get("slots") is not None:
            parts.append(f"arena {arena['slots']} slots, {(arena.get('gb') or 0):.1f} GB")
        elif store.get("slots"):
            parts.append(f"store {store['slots']} slots, {(store.get('mb') or 0) / 1000:.1f} GB")
        if exp.get("rows_per_token"):
            parts.append(f"{exp['rows_per_token']:.0f} expert rows a token from the SSD")
        return " · ".join(parts)

    def draw_stop(self, cv, h, w):
        r = self.running or {}
        where = r.get("device") or "this computer"
        cv.put(5, 2, fit(f"Stopping {r.get('name', 'the model')} on {where}", w - 4), "title")
        cv.put(7, 2, f"{spin(self.frozen, self.now())} the other computer stops it and releases its lock",
               "accent")
        cv.put(h - 1, 1, "wait", "dim")

    def draw_update(self, cv, h, w):
        d = self.dev_job_dev
        cv.put(5, 2, fit(f"Updating the engine on {d.name if d else 'the device'}", w - 4), "title")
        cv.put(7, 2, f"{spin(self.frozen, self.now())} {self.dev_job.phase or 'updating...'}", "accent")
        cv.put(9, 2, "The update runs on the other computer; its list is read again after.", "dim")
        cv.put(h - 1, 1, "Esc", "key")
        cv.put(h - 1, 5, "stop", "dim")

    def draw_enginstall(self, cv, h, w):
        d = self.dev_job_dev
        cv.put(5, 2, fit(f"Installing the engine on {d.name if d else 'the device'}", w - 4), "title")
        cv.put(7, 2, f"{spin(self.frozen, self.now())} {self.dev_job.phase or 'installing...'}", "accent")
        cv.put(9, 2, "The installer's output:", "dim")
        self.log_pane(cv, 10, h, w, self.log.tail(max(1, h - 12)))
        cv.put(h - 1, 1, "Esc", "key")
        cv.put(h - 1, 5, "stop", "dim")

    def draw_error(self, cv, h, w):
        cv.put(5, 2, "That did not work", "bad")
        for i, line in enumerate(wrap(self.err, w - 4)[:4]):
            cv.put(7 + i, 2, line, "normal")
        cv.put(12, 2, "Last lines of output:", "dim")
        n = max(1, h - 15)
        self.log_pane(cv, 13, h, w, self.b.log_tail(n) if self.err_engine else self.log.tail(n))
        cv.put(h - 1, 1, "any key", "key")
        back = f"back to {self.dev.name}'s models" if self.dev else "back to the devices"
        cv.put(h - 1, 9, back, "dim")


# ---------------------------------------------------------------- the runner

def available():
    """Whether a full-screen menu can run here: a terminal on both ends and curses."""
    return bool(curses) and sys.stdin.isatty() and sys.stdout.isatty() and os.environ.get("TERM", "dumb") != "dumb"


def frozen_by_env(env=None):
    env = os.environ if env is None else env
    return bool(env.get("MOE_TUI_FROZEN")) or "NO_COLOR" in env


def run(backend):
    """Open the menu and return when the user leaves it, with anything printed meanwhile going to
    the menu's log pane."""
    locale.setlocale(locale.LC_ALL, "")
    os.environ.setdefault("ESCDELAY", "25")
    log = Log()
    app = App(backend, log, frozen=frozen_by_env())

    def loop(scr):
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        scr.keypad(True)
        scr.timeout(100)
        cv = CursesCanvas(scr)
        while True:
            app.tick()
            app.draw(cv)
            cv.refresh()
            try:
                k = scr.get_wch()
            except curses.error:
                continue
            except KeyboardInterrupt:
                break
            act = app.key(k)
            if act == "quit":
                break
            if isinstance(act, tuple) and act[0] == "chat":
                curses.def_prog_mode()
                curses.endwin()
                with contextlib.redirect_stdout(sys.__stdout__), contextlib.redirect_stderr(sys.__stderr__):
                    print()
                    backend.chat(act[1])
                curses.reset_prog_mode()
                scr.clear()

    try:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            curses.wrapper(loop)
    finally:
        backend.shutdown()
    return app


# ---------------------------------------------------------------- a stand-in for trying the menu

class DemoBackend(Backend):
    """Made-up computers and models for `python3 tui.py --demo` and the tests, where nothing is
    downloaded or run."""

    def __init__(self, delay=0.05):
        self.delay = delay
        self.done = 0
        self.stopped = False
        self.aborted = False
        self.opened = []
        self.started = []             # (device name, model id, variant key)
        self.scanned = 0
        self.probed = []              # names of the devices checked over (pretend) ssh
        self.rescanned = []           # devices whose list was asked to look at their disks again
        self.deleted = []             # (device name, model id, variant key)
        self.searched = []            # (device name, query)
        self.paired = []              # (device name, code tried)

    def info(self):
        time.sleep(self.delay)
        return {"chip": "Apple M4", "ram_gb": 16, "gpu": "Apple M4", "os": "macOS 27.0", "models_dir": "/demo/models",
                "free_gb": 12.6, "ssd_gbps": 6.5, "others": [],
                "conditions_text": "on battery · other apps use 2.1 CPU cores (demo): expect slower"}

    def attached(self):
        return None

    def device_list(self):
        time.sleep(self.delay)
        local = Device("this computer", kind="local", source="here", hardware="Apple M4, 16 GB memory",
                       status="ready", detail="engine at /demo/moe")
        studio = Device("studio", "studio", "preset", "presets", ("studio",), "engine at ~/engine",
                        "Apple M4, 16 GB", "ready", measured=(13.8, 14.2), preset="studio",
                        dir="~/engine", python="~/venv/bin/python",
                        commands={"flash": {"port": 8790, "id": "flash-next-exl3"},
                                  "35b": {"port": 8791, "id": "qwen3.6-35b"}})
        rig = Device("rig", "me@rig", "preset", "presets", ("rig",),
                     "no ssh key: ssh-copy-id me@rig", "RTX 5060 Ti 16 GB, 14 GB RAM", "no-key",
                     measured=(32.0, 34.0), preset="rig", dir="~/engine",
                     commands={"flash": {"port": 8792, "id": "flash-next-exl3"}})
        desk = Device("desk", "desk", "tailscale", "tailscale", ("desk", "100.64.0.2"),
                      "engine at ~/larder", "linux", "ready")
        shed = Device("shed", "shed", "tailscale", "tailscale", ("shed", "100.64.0.9"),
                     "engine at ~/larder", "linux", "ready")
        attic = Device("attic.local", "192.168.1.50", "scan", "~/.ssh/config", ("192.168.1.50",),
                       "no answer on ssh: check the network or the host", "", "offline")
        lamp = Device("lamp (shared)", "192.168.1.9", "shared", "share beacon", ("192.168.1.9",),
                      "pair with the code shown on that computer", "", "shared", port=4321)
        return [local, studio, rig, desk, shed, attic, lamp]

    def device_probe(self, dev):
        time.sleep(self.delay / 5)
        self.probed.append(dev.name)
        if dev.status == "checking":
            dev.status = "ready"
            dev.engine_state = "ok"
        return dev

    def pair(self, dev, code):
        time.sleep(self.delay)
        self.paired.append((dev.name, code))
        if code.replace("-", "").replace(" ", "").upper() != "7K3M9QXDR2HF":
            raise RuntimeError("wrong code")
        dev.status, dev.detail = "ready", f"paired with {dev.name}"
        return dev

    def pair_fingerprint(self, dev):
        return "3f9a1c0b7d2e4a65" + "0" * 48

    def sweep_subnet(self):
        return "192.168.1.0/24"

    def device_scan(self, say):
        self.scanned += 1
        say("sweeping this network, a minute at most...")
        for i in range(0, 254, 64):
            say(f"looking at {i}/254 addresses...")
            time.sleep(self.delay / 4)
        return [Device("studio.local", "192.168.1.14", "scan", "the network scan", ("192.168.1.14",),
                       "ssh is open", "", "ssh-open")]

    def device_models(self, dev, say):
        if dev.kind == "local":
            yield from self.models(say)
            return
        say(f"asking {dev.name} for its model list...")
        time.sleep(self.delay)
        ready = dev.name == "studio"
        measured = dev.measured if dev.kind == "preset" else ()
        flash = Model("flash-next-exl3", "Qwen3.8-Flash-Next (3-bit EXL3 experts)",
                      "MoE, 512 experts x 48 layers", "the largest", "apache-2.0", True)
        pub = Variant("published", "as published, 4-bit", int(50e9), 14 if measured else 9, True, "", "ready",
                      "the model as published", ready=ready, disk=int(50e9), entry_id="flash-next-exl3",
                      measured=measured)
        q3 = Variant("q3", "3-bit, made there", int(40e9), 17, True, "",
                     "~40 min to make after a 50.0 GB download", "a noticeable loss", make_bits=3,
                     download=int(50e9), disk=int(40e9), expected=int(90e9), entry_id="flash-next-exl3-q3")
        flash.variants = [pub, q3]
        yield flash
        q35 = Model("qwen3.6-35b", "Qwen3.6-35B-A3B", "MoE, 256 experts x 40 layers", "general chat",
                    "apache-2.0", True,
                    [Variant("published", "as published, 4-bit", int(20.4e9), 30, True, "", "ready",
                             "the model as published", ready=True, disk=int(20.4e9), entry_id="qwen3.6-35b",
                             path="~/larder/models/qwen3.6-35b", own=True)])
        yield q35
        if dev.kind != "preset":
            small = Model("demo-4b", "Demo-4B", "dense", "small and fast", "apache-2.0", True,
                          [Variant("published", "as published, 4-bit", int(2.3e9), 43, True, "",
                                   "download 2.3 GB", ready=False, download=int(2.3e9), disk=int(2.3e9),
                                   expected=int(2.3e9), entry_id="demo-4b")])
            yield small

    def start_device(self, dev, model, v, job):
        self.started.append((dev.name, model.id, v.key))
        if dev.kind == "local":
            return self.start(v, job)
        for i in range(3):
            job.say(f"waiting for {dev.name} to say it is serving... {i} s")
            if job.cancelled():
                return None
            time.sleep(self.delay)
        dev.status, dev.running = "running", f"{model.name} (from the menu)"
        return {"name": model.name, "url": "http://127.0.0.1:8800", "api": "http://127.0.0.1:8800/v1",
                "model_id": model.id, "tps": v.tps, "measured": tuple(v.measured), "device": dev.name,
                "port": 8800, "remote_port": 8790}

    def update_device(self, dev, job):
        self.updated = getattr(self, "updated", []) + [dev.name]
        for i in range(2):
            job.say(f"pulling the engine on {dev.name}... {i} s")
            if job.cancelled():
                return "stopped"
            time.sleep(self.delay)
        dev.status, dev.engine_state, dev.detail = "ready", "ok", "engine at ~/larder"
        return f"{dev.name}: updated from origin"

    def install_device(self, dev, job):
        self.installed = getattr(self, "installed", []) + [dev.name]
        for i in range(3):
            job.say(f"installing on {dev.name}... {i} s")
            print(f"  installing the engine on {dev.name}: step {i}")
            if job.cancelled():
                return "stopped"
            time.sleep(self.delay)
        dev.status, dev.engine_state, dev.detail = "ready", "ok", "engine at ~/larder"
        return f"{dev.name}: installed ~/larder"

    def models(self, say):
        say("checking Demo-35B...")
        time.sleep(self.delay)
        big = Model("demo-35b", "Demo-35B-A3B", "MoE, 256 experts x 40 layers", "general chat, code",
                    "apache-2.0", True,
                    [Variant("published", "as published, 4-bit", int(20.4e9), 24, True, "", "download 20.4 GB",
                             "the model as published", download=int(20.4e9), disk=int(20.4e9),
                             expected=int(20.4e9), entry_id="demo-35b", tps_lo=14, tps_hi=24, mtp=True),
                     Variant("hf-8", "8-bit", int(37.7e9), 7, False,
                             "needs 58 GB free on disk, 12.6 GB free",
                             "needs 58 GB free on disk, 12.6 GB free", "close to the original", disk_short=True),
                     Variant("hf-16", "16-bit", int(70e9), None, False,
                             "the whole model stays in memory: 71.0 GB needed, 9.2 GB usable now",
                             "can't run: the whole model stays in memory: 71.0 GB needed, 9.2 GB usable now",
                             "the original weights"),
                     Variant("q3", "3-bit, made here", int(16.4e9), 29, True, "",
                             "~3 min to make after a 20.4 GB download",
                             "a noticeable loss; rounded from the published 4-bit", make_bits=3,
                             download=int(20.4e9), disk=int(16.4e9), expected=int(36.8e9),
                             entry_id="demo-35b-q3")],
                    ["3-bit GPTQ: not made here, it needs a calibration run that does not fit 16 GB"])
        yield big
        say("checking Demo-4B...")
        time.sleep(self.delay)
        yield Model("demo-4b", "Demo-4B", "dense", "small and fast", "apache-2.0", True,
                    [Variant("published", "as published, 4-bit", int(2.3e9), 43, True, "", "ready",
                             "the model as published", ready=True, disk=int(2.3e9), expected=int(2.3e9),
                             entry_id="demo-4b", path="/demo/models/demo-4b", own=True)])
        say("checking Demo-EXL3...")
        time.sleep(self.delay)
        yield Model("demo-exl3", "Demo-EXL3 (3-bit experts)", "MoE, 512 experts x 48 layers", "the largest",
                    "apache-2.0", True,
                    [Variant("published", "as published, EXL3 3-bit experts", int(75e9), 11, True, "",
                             "on disk at /home/demo/.cache/demo-exl3", "3-bit EXL3 experts", ready=True,
                             entry_id="demo-exl3", path="/home/demo/.cache/demo-exl3",
                             measured_here={"tps": 13.8, "n": 12, "date": "2026-10-27"}),
                     Variant("hf-4", "4-bit", int(111e9), 7, True, "", "ready", "the 4-bit reference", ready=True,
                             entry_id="demo-exl3-4bit", path="/demo/models/demo-exl3-4bit", own=True),
                     Variant("hf-8", "8-bit", int(190e9), None, False, "needs 200 GB free on disk, 12.6 GB free",
                             "needs 200 GB free on disk, 12.6 GB free", "close to the original",
                             download=int(190e9), disk=int(190e9), disk_short=True)])
        yield Model("demo-flash", "Demo-Flash (3-bit experts)", tested=True,
                    unavailable="model type 'demo_exp' is not in the installed mlx-lm")

    def device_search(self, dev, query, say):
        self.searched.append((dev.name, query))
        if dev.kind == "local":
            return self.search(query, say)
        say(f"asking {dev.name} to look on Hugging Face...")
        time.sleep(self.delay)
        if "nope" in query:
            raise RuntimeError("model type 'demo_exp' is not in the installed mlx-lm")
        name = query.rstrip("/").rsplit("/", 1)[-1]
        # each computer judges for itself: shed has little disk left, the others take it
        short = dev.name == "shed"
        why = "needs 18 GB free on disk, 12.9 GB free"
        v = Variant("published", "as published, 4-bit", int(8e9), 30, not short, why if short else "",
                    why if short else "download 8.0 GB", "the model as published", download=int(8e9),
                    disk=int(8e9), expected=int(8e9), disk_short=short, entry_id=f"mlx-community--{name}")
        return [Model(f"mlx-community--{name}", f"mlx-community/{name}", "MoE, 64 experts x 24 layers",
                      "12,345 downloads", "", False, [v], found=True, repo=f"mlx-community/{name}")], 2

    def search(self, query, say):
        time.sleep(self.delay)
        return [Model(f"found-{query}", f"mlx-community/{query}-4bit", "MoE, 64 experts x 24 layers",
                      "12,345 downloads", "", False,
                      [Variant("published", "as published, 4-bit", int(8e9), 30, True, "", "download 8.0 GB",
                               "the model as published", download=int(8e9), disk=int(8e9), expected=int(8e9))],
                      found=True)], 3

    def local(self, path):
        if "nope" in path:
            raise ValueError(f"{path} has no config.json: not a model folder")
        return Model("local", os.path.basename(path), "dense", path, "", False,
                     [Variant("published", "as it is", int(1e9), 50, True, "", "ready", ready=True)], found=True)

    def last(self):
        return None

    def dest(self, variant):
        return variant.path or "/demo/models/" + variant.entry_id

    def rescan(self, dev):
        self.rescanned.append(dev.name)

    def delete(self, dev, model, variant):
        time.sleep(self.delay)
        self.deleted.append((dev.name, model.id, variant.key))
        return f"deleted {variant.path} ({variant.size / GB:.1f} GB)"

    def progress(self, variant):
        return self.done

    def install(self, variant, job):
        total = variant.expected
        job.say("downloading the weights")
        while self.done < total:
            if job.cancelled():
                return
            self.done = min(total, self.done + total // 40)
            print(f"  fetching shard {self.done * 40 // total}/40 ...", flush=True)
            time.sleep(self.delay)
        self.done = 0

    def start(self, variant, job):
        for i in range(3):
            job.say(f"loading the model... {i} s")
            if job.cancelled():
                return None
            time.sleep(self.delay)
        est = {"tps": variant.tps, "lo_tps": variant.tps_lo or variant.tps,
               "hi_tps": variant.tps_hi or variant.tps, "runs": 1 if variant.tps_adjusted else 0}
        return {"name": variant.label, "url": "http://127.0.0.1:8800", "api": "http://127.0.0.1:8800/v1",
                "model_id": "demo-model", "tps": variant.tps, "est": est, "port": 8800}

    def live(self):
        """A made-up /status as if three replies had been timed on this computer."""
        return {"decode_replies": 3, "decode_tok_s": 12.8,
                "arena": {"slots": 21, "gb": 1.49}, "store": {"mb": None, "slots": None},
                "experts": {"rows": 396, "rows_per_token": 132.0},
                "conditions_text": "on battery · other apps use 2.1 CPU cores (demo): expect slower"}

    def log_tail(self, n):
        return ["loading weights...", "arena ready", "serving on 127.0.0.1:8800"][-n:]

    def abort(self):
        self.aborted = True

    def stop(self):
        self.stopped = True

    def open_chat(self, url):
        self.opened.append(url)

    def chat(self, url):
        print("(demo: the terminal chat would run here)")


if __name__ == "__main__":
    if "--demo" not in sys.argv:
        sys.exit(__doc__)
    run(DemoBackend(delay=0.3))
