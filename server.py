#!/usr/bin/env python3
"""An OpenAI-compatible HTTP front end for the streaming engine, with a chat page and a model-swap route."""

import argparse
import base64
import binascii
import copy
import hmac
import json
import os
import socket
import ssl
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

import queue
from collections import deque
from concurrent.futures import ThreadPoolExecutor

# The launcher's start (MOE_TUNED=1 or --tuned) applies the cached first-run tuning here, before the engine
# modules read the environment; a raw command is untouched and any failure starts the server untuned.
import sys

import engine_log   # noqa: F401  (sets MLX's CUDA environment before anything imports MLX)
try:
    import tuner
    if tuner.opted_in(sys.argv) and tuner.model_from_argv(sys.argv):
        tuner.apply_env(tuner.model_from_argv(sys.argv))
except Exception:
    tuner = None

import mlx.core as mx

from engine_args import add_engine_flags, attach_governor, engine_kwargs, finish, memory_sleep
from engine_log import note
from mlx_lm.models.cache import (BatchKVCache, KVCache, make_prompt_cache,
                                trim_prompt_cache)

import library                      # the speed each model was measured at here
import machine                      # the live conditions the speed depends on
import share                        # the local-only rule this server and the share proxy apply
import stream_fast
from memory_core import clone_cache_keeping_gap
from moe_stream import StreamingLLM
from mtp import DEFAULT_MTP
from sampler import Sampling
import web_settings                     # the Settings page's endpoints and option keys

START_TIME = time.perf_counter()
START_WALL = time.time()
HISTORY_LEN = 200                   # replies GET /admin/history keeps for the Status page's charts
WEB_DIR = Path(__file__).resolve().parent / "web"   # the chat page and its local assets
# Clean addresses for the pages that are more than the chat.
PAGES = {"/settings": "settings.html", "/monitor": "status.html", "/favicon.ico": "favicon-32.png"}
# The GET routes that need the api key; everything else is a page or one of its assets.
API_GET = ("/healthz", "/status", "/admin/history", "/stats", "/v1/models",
           "/admin/models", "/admin/prepare", "/admin/search", "/v1/memory")
STATIC_TYPES = {".html": "text/html; charset=utf-8",
                ".css": "text/css; charset=utf-8",
                ".js": "text/javascript; charset=utf-8",
                ".mjs": "text/javascript; charset=utf-8",
                ".json": "application/json",
                ".svg": "image/svg+xml",
                ".png": "image/png",
                ".ico": "image/x-icon",
                ".woff2": "font/woff2",
                ".txt": "text/plain; charset=utf-8",
                ".md": "text/markdown; charset=utf-8"}
MACHINE_BYTES = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
MAX_BODY = 64 * 1024 * 1024         # a request body, images included
MAX_IMAGE = 10 * 1024 * 1024        # one decoded image
# The first bytes of each image type a data: URL may carry.
IMAGE_MAGIC = {"png": (b"\x89PNG\r\n\x1a\n",), "jpeg": (b"\xff\xd8\xff",),
               "webp": (b"RIFF",), "gif": (b"GIF87a", b"GIF89a")}


def host_name():
    """This computer's name without a .local tail, which the chat page tags its saved chats with."""
    return socket.gethostname().split(".")[0]


_MACHINE = {}


def machine_info():
    """Chip, memory, OS and backend of this computer, read once."""
    if not _MACHINE:
        try:
            import machine
            f = machine.facts()
            gpu = f.get("gpu")
            _MACHINE.update(chip=f.get("chip") or None, ram_gb=round((f.get("ram") or 0) / 2**30, 1) or None,
                            os=f.get("os"), backend=f.get("backend"), cpus=f.get("cpus"),
                            gpu=gpu if gpu and gpu != f.get("chip") else None, mlx=f.get("mlx"))
        except Exception:
            _MACHINE["chip"] = None
    return _MACHINE


class Counters:
    """What GET /status adds up since the server started: one finished reply
    at a time from the worker thread, read by the handler threads."""

    def __init__(self):
        self.lock = threading.Lock()
        self.requests = self.prompt = self.cached = self.completion = 0
        self.ttft_sum, self.ttft_n, self.ttft_last = 0.0, 0, None
        self.decode_tokens, self.decode_s = 0, 0.0
        self.decode_replies = 0
        self.recent = deque(maxlen=HISTORY_LEN)

    def add(self, info):
        """`info` is what the worker filled for one reply (ModelHost._count)."""
        if info.get("warmup"):              # the launcher's warm-up reply is no user's
            return
        n, ttft, dec_s = info.get("completion_tokens") or 0, info.get("ttft_s"), info.get("decode_s") or 0.0
        with self.lock:
            self.requests += 1
            self.prompt += info.get("prompt_tokens") or 0
            self.cached += info.get("cached_tokens") or 0
            self.completion += n
            if ttft is not None:
                self.ttft_sum, self.ttft_n, self.ttft_last = self.ttft_sum + ttft, self.ttft_n + 1, ttft
            if n > 1 and dec_s > 0:         # decode speed weighted by tokens
                self.decode_tokens, self.decode_s = self.decode_tokens + n - 1, self.decode_s + dec_s
                self.decode_replies += 1
            foot = stream_fast.phys_footprint()
            self.recent.append({
                "t": round(time.time(), 1), "ttft_s": None if ttft is None else round(ttft, 3),
                "decode_tok_s": round((n - 1) / dec_s, 2) if n > 1 and dec_s > 0 else None,
                "prompt_tokens": info.get("prompt_tokens") or 0,
                "cached_tokens": info.get("cached_tokens") or 0, "completion_tokens": n,
                "memory_gb": round(foot / 1e9, 2) if foot else None})

    def history(self):
        with self.lock:
            return list(self.recent)

    def report(self):
        with self.lock:
            return {
                "requests": self.requests, "prompt_tokens": self.prompt,
                "cached_tokens": self.cached, "completion_tokens": self.completion,
                "first_token_mean_s": round(self.ttft_sum / self.ttft_n, 2) if self.ttft_n else None,
                "first_token_last_s": None if self.ttft_last is None else round(self.ttft_last, 2),
                "decode_tok_s": round(self.decode_tokens / self.decode_s, 2) if self.decode_s else None,
                "decode_replies": self.decode_replies}


class _Turn:
    """One request in ModelHost._batch_loop: what it asked for, where its
    text goes, and how far its reply has got."""

    def __init__(self, messages, max_tokens, tools, response_format, sampling=None,
                 info=None, thinking=None, think_budget=None):
        self.messages, self.max_tokens = messages, max_tokens
        self.info = info        # dict the caller reads token counts from
        self.thinking = thinking    # None: the engine's own setting
        self.think_budget = think_budget    # thinking tokens before the early-stop text, None: no cap
        self.t0, self.t_first = time.perf_counter(), None
        self.batched = False    # it has been a row of a shared forward
        self.tools, self.response_format = tools, response_format
        self.sampling = sampling    # sampler.Sampling, None = greedy
        self.out = queue.Queue()
        self.limit = self.prompt_len = None
        self.gen = None         # llm.stream, while it decodes alone
        self.seq = None         # prompt + reply ids, once it is a batch row
        self.detok = None
        self.n = 0              # reply tokens so far
        self.gone = False       # nobody reads the reply any more

    def fail(self, e):
        self.out.put(("error", e))

    def pieces(self, client_gone=None):
        # A client that hangs up mid-stream drops this generator: its turn
        # then leaves the batch (or llm.stream) at its next token instead of
        # holding a row to max_tokens while the others wait on it (a whole-reply caller's `client_gone` is polled per piece and once a second).
        try:
            while True:
                try:
                    kind, val = self.out.get(timeout=1 if client_gone else None)
                except queue.Empty:
                    kind = None
                if client_gone and client_gone():
                    return
                if kind is None:
                    continue
                if kind == "error":
                    raise val
                if kind == "end":
                    return
                yield val
        finally:
            self.gone = True


class ModelHost:
    """Holds the single live StreamingLLM and runs every mlx call on its one worker thread."""
    batch = False       # set per load: whether concurrent turns share forwards
    think_default = False   # set per load: the engine's own thinking setting
    think_budget = None     # --thinking-budget: a thinking turn's cap in tokens, None for none
    counters = Counters()   # a host built without __init__ (toy tests) shares this one

    def __init__(self, model, memory=None, sleep_after=0, adaptive=False,
                 reserve_pct=10, arena_min=6, arena_max=0, mem_pct=90,
                 host_gb=0.6, mem_peak_gb=None, reuse=True, keep_convs=1,
                 keep_gb=2.0, prefill_gb=None, mem_peak_max_gb=None, shadow=True,
                 **kwargs):
        # adaptive, reserve_pct, arena_min and arena_max are the host's own, and mem_pct and host_gb are named so _load can pass them on explicitly instead of letting the engine's defaults win.
        self.adaptive = adaptive
        self.reserve_pct = reserve_pct
        self.arena_min = arena_min
        self.arena_max = arena_max
        self.mem_pct = mem_pct
        self.host_gb = host_gb
        # Process policies like mem_pct: a swap keeps them.
        self.mem_peak_gb = mem_peak_gb
        # --mem-peak-max-gb: the store grows toward it while the machine has
        # room (stream_fast.make_store_governor); not a StreamingLLM argument.
        self.mem_peak_max_gb = mem_peak_max_gb
        self.prefill_gb = prefill_gb
        # The engine settings this server started with, which swap() keeps for every later model.
        self.engine_kwargs = dict(kwargs)
        self.reuse = reuse
        self.keep_convs, self.keep_gb = keep_convs, keep_gb
        # The shadow render (StreamingLLM.shadow) prefills the next message's re-render on the worker while nothing else is queued, and _asks, _turns and _queue_s let a request stop it and see how long it waited.
        self.shadow_on = shadow
        self._asks, self._turns, self._queue_s, self._shadow_last = 0, 0, None, None
        self.executor = ThreadPoolExecutor(max_workers=1)
        # Requests for _batch_loop, and whether one is running to take them.
        self._inbox, self._lock, self._looping = queue.Queue(), threading.Lock(), False
        self.llm = None
        self.model_name = None
        self.last_error = None
        self.counters = Counters()
        # Idle consolidation: nobody types /sleep at a server, so after
        # `sleep_after` seconds without a finished turn the pending turns are
        # journalled and the core rebuilt, through the same worker queue.
        self.sleep_after = sleep_after
        self.last_request = time.monotonic()
        # --memory, all of it read and written on the worker thread only: the
        # store for the loaded checkpoint, the KV core held attached (see
        # `resume`), and the user turns said since the last sleep.
        self.memory_tag = memory
        self.store = None
        self.core = None
        self.turns = []
        self.executor.submit(self._load, model, mem_pct=mem_pct,
                             host_gb=host_gb, mem_peak_gb=mem_peak_gb,
                             prefill_gb=prefill_gb, **kwargs).result()

    def _load(self, model, **kwargs):
        """Build a StreamingLLM and drop the old one only once the new one exists, so two models never coexist and a bad path leaves the old one serving."""
        if self.store is not None and self.llm is not None:
            # A core is one checkpoint's, so it is rebuilt while its weights are still loaded and before the new skeleton exists, and a failure here costs the memory, never the swap.
            try:
                note(f"  {self._sleep_now()}")
            except Exception as e:
                print(f"  memory: sleep before the swap failed ({e})")
        if self.llm is not None:
            # No model after this one can take them up: not held beside a
            # second skeleton while it loads.
            self.llm.forget()
        new_llm = StreamingLLM(model=model, **kwargs)
        # Set on the engine rather than passed in: they change no array a
        # front or the budget's calibration is keyed on.
        new_llm.reuse = self.reuse
        new_llm.keep_convs, new_llm.keep_gb = self.keep_convs, self.keep_gb
        if self.adaptive and getattr(new_llm, "_budget", None) is None:
            self._attach_governor(new_llm)
        if self.mem_peak_max_gb and getattr(new_llm, "_budget", None) is not None:
            stream_fast.make_store_governor(new_llm._budget, self.mem_peak_max_gb,
                                            self.reserve_pct)
        if getattr(new_llm, "_budget", None) is not None:
            new_llm._budget.prepare()       # MOE_RESIDENT: the store and the tier filled at load
        old = self.llm
        self.llm = new_llm
        self.think_default = bool(getattr(new_llm, "thinking", False))
        self.model_name = str(model)
        self.last_error = None
        if old is not None:
            del old
            mx.clear_cache()
        if self.memory_tag:
            self._open_memory(model)
        # --memory hands every request its own clone of one core, which a
        # batch row has no place for.
        self.batch = self.store is None and self._can_batch(new_llm)
        # Copied per request: building one reads the whole vocabulary, ~90 ms
        # on a 248k-token tokenizer.
        self._detok = new_llm.tokenizer.detokenizer

    @staticmethod
    def _can_batch(llm):
        """Dense, and every layer a plain KV cache, which is what BatchKVCache holds."""
        return (not stream_fast.glus_of(llm.model)
                and all(type(c) is KVCache for c in make_prompt_cache(llm.model)))

    def _attach_governor(self, llm):
        """Size the arena from the machine's free memory between this server's own floor and ceiling, as chat.py does."""
        attach_governor(llm, self.mem_pct, self.reserve_pct, self.host_gb, self.arena_min, self.arena_max)

    def swap(self, model, **kwargs):
        """Load a new checkpoint with this server's start-up settings and the request's keys on top."""
        # An MTP head or a draft path belongs to one checkpoint, so only the load request can bring one.
        base = {k: v for k, v in self.engine_kwargs.items()
                if k != "mtp" and not (k == "draft" and v not in (None, "auto"))}
        self.executor.submit(self._load, model, mem_pct=self.mem_pct,
                             host_gb=self.host_gb, mem_peak_gb=self.mem_peak_gb,
                             prefill_gb=self.prefill_gb, **{**base, **kwargs}).result()

    # The --memory state and its sleep.

    def _open_memory(self, model):
        """The store for the checkpoint just loaded, and its core."""
        import kvmem            # deferred: --memory is opt-in and kvmem is not
        self.store = kvmem.Store(model, tag=self.memory_tag)
        note(f"  {self.store.summary()}")
        self._attach_core()

    def _attach_core(self):
        """Hold what the store has on disk as one attached copy, and leave the engine holding none."""
        self.core = None
        if self.store.ensure(self.llm):
            self.core = {"cache": self.llm._cache,
                         "seen": list(self.llm._seen),
                         "history": [dict(m) for m in self.llm.history]}
        self.llm.reset()
        # Conversations kept from requests served before there was a core:
        # every request resumes it from here on, and none takes one up.
        self.llm.forget()

    def resume(self):
        """A private clone of the core for one request, or None without --memory."""
        if self.core is None:
            return None
        return dict(self.core, cache=clone_cache_keeping_gap(self.core["cache"]))

    def _journal(self, messages):
        """Keep this request's new user turns, those after the last assistant message, for the next sleep."""
        if self.store is None:
            return
        tail = messages
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "assistant":
                tail = messages[i + 1:]
                break
        self.turns += [m["content"] for m in tail
                       if m.get("role") == "user"
                       and str(m.get("content", "")).strip()]
        self.last_request = time.monotonic()

    def _idle_check(self):
        """Sleep once if turns are pending and the server has been idle long enough, and say whether it slept."""
        if not (self.sleep_after > 0 and self.turns
                and time.monotonic() - self.last_request >= self.sleep_after):
            return False
        note(f"  memory: idle {self.sleep_after}s -> {self.sleep()}", flush=True)
        return True

    def start_arena_watch(self, period=1.0):
        """Once a second, queued on the worker, let the governor give slots back while the server is idle."""
        def loop():
            fut = None
            while True:
                time.sleep(period)
                gov = stream_fast._GOV
                if gov is not None and (fut is None or fut.done()):
                    fut = self.executor.submit(gov.idle_tick)
        if self.adaptive or self.mem_peak_max_gb:
            threading.Thread(target=loop, daemon=True).start()

    def start_idle_sleeper(self, period=30):
        def loop():
            while True:
                time.sleep(period)
                try:
                    self._idle_check()
                except Exception as e:      # a failed rebuild keeps the turns
                    print(f"  memory: idle sleep failed ({e})", flush=True)
        if self.sleep_after > 0 and self.store is not None:
            threading.Thread(target=loop, daemon=True).start()

    def _sleep_now(self):
        """Journal the pending turns, taken off self.turns first so a rebuild that raises cannot journal them twice, and rebuild only if the plan changed."""
        turns, self.turns = self.turns, []
        return memory_sleep(self.store, self.llm, turns)

    def sleep(self):
        """Consolidate on the worker thread, behind any turn in flight, then re-attach the new core."""
        def run():
            if self.llm is None:
                raise RuntimeError(self.last_error or "no model loaded")
            msg = self._sleep_now()
            self._attach_core()      # what is on disk just changed
            return msg
        return self.executor.submit(run).result()

    def generate(self, messages, max_tokens, stream, tools=None,
                 response_format=None, sampling=None, info=None, thinking=None,
                 client_gone=None, think_budget=None):
        """The full text (stream=False) or an iterator of text pieces, every mlx call running on the worker; `info` is filled with the turn's token counts before the reply ends; `client_gone()` turning true stops the turn at its next token; a thinking turn thinks at most the smaller of `think_budget` and --thinking-budget tokens."""
        on = self.think_default if thinking is None else thinking
        think_budget = min(filter(None, (think_budget, self.think_budget)), default=None) if on else None
        if self.batch:
            t = _Turn(messages, max_tokens, tools, response_format, sampling, info,
                      thinking, think_budget)
            with self._lock:
                self._inbox.put(t)
                if not self._looping:
                    self._looping = True
                    self.executor.submit(self._batch_loop)
            return t.pieces(client_gone) if stream else "".join(t.pieces(client_gone))
        t_ask = self._ask()
        if not stream:
            def run():
                self._begin(t_ask)
                if self.llm is None:
                    raise RuntimeError(self.last_error or "no model loaded")
                if client_gone and client_gone():
                    return ""       # it gave up while queued: no prefill for nobody
                self._think(thinking, think_budget)
                parts = []
                gen = self.llm.stream(messages=messages, max_tokens=max_tokens,
                                      resume=self.resume(), tools=tools,
                                      response_format=response_format,
                                      sampling=sampling, keep_on_close=True)
                for piece in gen:
                    if client_gone and client_gone():
                        # Same as produce() below: the half turn is kept for reuse, nothing is counted or journalled.
                        gen.close()
                        return "".join(parts)
                    parts.append(piece)
                text = "".join(parts)
                self._count(info)
                self._journal(messages)
                self._after(messages, text, tools)
                return text
            return self.executor.submit(run).result()

        q = queue.Queue()
        gone = threading.Event()    # the consumer was closed: stop decoding

        def produce():
            try:
                self._begin(t_ask)
                if self.llm is None:
                    raise RuntimeError(self.last_error or "no model loaded")
                if client_gone and client_gone():
                    return
                self._think(thinking, think_budget)
                parts = []
                gen = self.llm.stream(messages=messages,
                                      max_tokens=max_tokens,
                                      resume=self.resume(),
                                      tools=tools,
                                      response_format=response_format,
                                      sampling=sampling,
                                      keep_on_close=True)
                for piece in gen:
                    # A held tool-call reply writes nothing, so only the socket shows that its client left.
                    if gone.is_set() or (client_gone and client_gone()):
                        # Nobody reads the reply, so stop here: the engine keeps the half turn for reuse and nothing is journalled.
                        gen.close()
                        return
                    q.put(("piece", piece))
                    parts.append(piece)
                self._count(info)
                self._journal(messages)
                self._after(messages, "".join(parts), tools)
            except Exception as e:
                q.put(("error", e))
            finally:
                q.put(("end", None))

        fut = self.executor.submit(produce)

        def consume():
            try:
                while True:
                    kind, val = q.get()
                    if kind == "piece":
                        yield val
                    elif kind == "error":
                        raise val     # produce() caught it, so the future itself finished cleanly
                    else:
                        return
            finally:
                gone.set()
        return consume()

    def _think(self, on, budget=None):
        """Set the engine's template flag and thinking budget for the request about to render."""
        if on is None:
            on = self.think_default
        if hasattr(self.llm, "set_thinking"):
            self.llm.set_thinking(on, budget)

    @property
    def vision(self):
        """Whether the loaded engine can look at images."""
        return bool(getattr(self.llm, "supports_images", False))

    def status(self):
        """What GET /status reports."""
        last = self.llm.stats() if self.llm else {}
        foot = stream_fast.phys_footprint()
        store = last.get("store") or {}
        cond = machine.conditions()
        return {
            "model": self.model_name, "loaded": self.llm is not None, "vision": self.vision,
            "uptime_s": round(time.perf_counter() - START_TIME, 1),
            **self.counters.report(),
            "memory_gb": round(foot / 1e9, 2) if foot else None,
            "memory_peak_gb": round(stream_fast.phys_footprint(peak=True) / 1e9, 2) if foot else None,
            "store_mb": store.get("store_mb"),
            # What the engine's own store holds: the arena's slots on the arena path, the store's
            # slots and MB on the budget path, and the expert rows it read from the SSD per token.
            "arena": {"slots": last.get("arena_slots"), "gb": last.get("arena_gb")},
            "store": {"mb": store.get("store_mb"),
                      "slots": (store["pool_slots"] + store["ring_slots"]) if store else None,
                      "budget_gb": store.get("budget_gb")},
            "experts": {"rows": last.get("expert_rows"), "rows_per_token": last.get("expert_rows_per_token")},
            "conditions": cond, "conditions_text": machine.conditions_text(cond),
            "memory_budget_gb": self.mem_peak_gb, "memory_budget_max_gb": self.mem_peak_max_gb,
            "context_length": web_settings.context_length(self.model_name),
            "host": host_name(), "machine": machine_info(), "started_at": round(START_WALL),
        }

    def _count(self, info):
        """The turn the engine just ran, into `info` and the counters (serial path): prompt
        tokens are what it prefilled plus what a kept cache gave, the first token counts
        from the request's arrival."""
        if info is None:
            return
        s = self.llm.stats()
        if "tokens" in s:
            decode_s = (s["tokens"] - 1) / s["decode_tps"] if s.get("decode_tps") else 0.0
            info.update(prompt_tokens=s["prompt_tokens"] + s["cached_tokens"],
                        completion_tokens=s["tokens"], cached_tokens=s["cached_tokens"],
                        decode_s=decode_s)
            if "ttft_s" in s:
                info["ttft_s"] = s["ttft_s"] + (self._queue_s or 0.0)
            self.counters.add(info)
            self._measure(info)

    def _finish(self, t, alone=False):
        """t's reply is complete on the batch path: its counts (from stats() when it ran alone, else from a fresh render of its prompt), then its end."""
        if t.info is not None and not t.gone:
            now = time.perf_counter()
            t.info.update(completion_tokens=t.n,
                          prompt_tokens=t.prompt_len or self._prompt_len(t))
            if t.t_first is not None:
                t.info.update(ttft_s=t.t_first - t.t0, decode_s=now - t.t_first)
            if alone:
                t.info["cached_tokens"] = self.llm.stats().get("cached_tokens", 0)
            self.counters.add(t.info)
            if alone:
                self._measure(t.info)
        t.out.put(("end", None))

    def _measure(self, info):
        """A reply that ran alone, into the model's measured speed (library.record); never a reason to fail it."""
        if info.get("warmup"):
            return
        try:
            library.record(self.model_name, info.get("completion_tokens"), info.get("decode_s"))
        except Exception as e:
            note(f"  measured speed not saved: {e}")

    def _ask(self):
        """A request is coming to the worker: a shadow in progress stops."""
        with self._lock:
            self._asks += 1
        return time.perf_counter()

    def _begin(self, t_ask):
        with self._lock:
            self._asks -= 1
        self._queue_s = round(time.perf_counter() - t_ask, 3)

    def _after(self, messages, text, tools):
        """A turn is over; when its reply went back to the user, queue the shadow of the next request behind whatever is queued."""
        self._turns += 1
        self._shadow_last = None
        if not self.shadow_on or self.store is not None:
            return
        if tools:
            import tools as toolfmt
            content, calls = toolfmt.parse(text, tools)
        else:
            content, calls = text, []
        if not calls:
            self.executor.submit(self._shadow, list(messages) + [
                {"role": "assistant", "content": content or None}], tools, self._turns)

    def _shadow(self, messages, tools, turn):
        """On the worker: the shadow of turn `turn`, unless a request came first."""
        if turn != self._turns or self._asks or self.llm is None:
            return
        try:
            info = self.llm.shadow(messages, tools, stop=lambda: self._asks > 0)
        except Exception as e:      # a background job: the next request is served anyway
            print(f"  shadow failed: {type(e).__name__}: {e}", flush=True)
            return
        if info is None:
            return
        self._shadow_last = info
        stopped = info.get("stopped")
        note(f"  shadow: {info['ids']} ids from {info['from']} in {info['s']:.2f} s, passes "
              f"{info['passes']}"
              + (f", stopped for a request in layer {info['layer']}, {info['kept_ids']} ids "
                 "kept" if stopped else "")
              + (f", kept {info['kept_mb']} MB" if "kept_mb" in info else "")
              + (f", {info['dropped']} older dropped" if info.get("dropped") else ""),
              flush=True)

    # Continuous batching, dense targets only.

    def _batch_loop(self):
        """Every request in flight, on the worker thread, until none is left: a lone request streams through llm.stream and several decode as rows of one forward."""
        solo, rows, batch, pend, waiting, t = None, [], None, None, deque(), None
        try:
            while True:
                while not self._inbox.empty():
                    t = self._inbox.get()
                    t.limit = t.max_tokens or self.llm.max_tokens
                    waiting.append(t)
                active = [solo] if solo else rows
                fits = bool(waiting and active) and self._room(active, batch, waiting[0])
                if pend is not None and (fits or len(rows) == 1):
                    rows, batch, _ = self._hand_out(rows, batch, pend)
                    pend = None
                    continue
                if waiting and not active:
                    solo = waiting.popleft()
                    self._start(solo)
                elif fits:
                    if solo:
                        batch = self._join(None, self._to_row(solo))
                        rows, solo = [solo], None
                    while waiting and self._room(rows, batch, waiting[0]):
                        t = waiting.popleft()
                        cache = self._row(t)
                        if cache is not None:
                            batch = self._join(batch, cache)
                            rows.append(t)
                    continue        # and whoever came in during those prefills
                elif len(rows) == 1:
                    self._to_solo(rows[0], batch)
                    solo, rows, batch = rows[0], [], None
                if solo:
                    if not self._pull(solo):
                        solo = None
                elif rows:
                    rows, batch, pend = self._step(rows, batch, pend)
                elif not waiting:
                    with self._lock:
                        if self._inbox.empty():
                            self._looping = False
                            return
        except BaseException as e:
            with self._lock:
                self._looping = False
                while not self._inbox.empty():
                    waiting.append(self._inbox.get())
            for x in [solo, t] + rows + list(waiting):
                if x is not None:
                    x.fail(e)
            raise

    def _start(self, t):
        """t's turn through llm.stream from its first token: today's path."""
        t.detok = copy.copy(self._detok)
        t.detok.reset()
        self._think(t.thinking, t.think_budget)
        t.gen = self.llm.stream(messages=t.messages, max_tokens=t.limit,
                                tools=t.tools, response_format=t.response_format,
                                sampling=t.sampling)

    def _pull(self, t):
        """One token of t from llm.stream, its text from t's own detokenizer, which has seen the reply from the start; False once t is over."""
        try:
            if t.gone:
                t.gen.close()
            else:
                next(t.gen)
                if not self._emit(t, self.llm._seen[-1]):
                    return True
                for _ in t.gen:     # stream() closes its turn: stats, stub slice
                    pass
                self._finish(t, alone=not t.batched)
        except Exception as e:
            t.fail(e)
        t.gen = None
        return False

    def _emit(self, t, tok):
        """t's text for one more reply token by mlx_lm.stream_generate's rules, and whether the reply is complete."""
        t.n += 1
        if t.t_first is None:
            t.t_first = time.perf_counter()
        stop = tok in self.llm.tokenizer.eos_token_ids
        if not stop:
            t.detok.add_token(tok)
        done = stop or t.n >= t.limit
        if done:
            t.detok.finalize()
        t.out.put(("piece", t.detok.last_segment))
        return done

    def _row(self, t):
        """t's prompt as a new batch row, prefilled by llm.stream up to its last id; None if that failed, the failure being t's alone."""
        if t.gone:
            return None
        llm = self.llm
        t.detok = copy.copy(self._detok)
        t.detok.reset()
        self._think(t.thinking, t.think_budget)
        try:
            for _ in llm.stream(messages=t.messages, max_tokens=t.limit,
                                tools=t.tools, response_format=t.response_format,
                                prefill_only=True):
                pass
        except Exception as e:
            llm.reset()
            t.fail(e)
            return None
        t.seq, cache = list(llm._seen), llm._cache
        t.batched = True
        llm.reset()                 # the batch holds this cache now
        return cache

    def _to_row(self, t):
        """t out of llm.stream as a batch row: its ids and a cache holding all of them but the last, read after the generator closes."""
        llm = self.llm
        t.seq, cache = list(llm._seen), llm._cache
        t.batched = True
        t.gen.close()
        t.gen = None
        keep = min(cache[0].offset, len(t.seq) - 1)
        trim_prompt_cache(cache, cache[0].offset - keep)
        if keep < len(t.seq) - 1:
            llm.model(mx.array(t.seq[keep:-1])[None], cache=cache)
        llm.reset()                 # the batch holds this cache now
        return cache

    def _to_solo(self, t, batch):
        """t is the last row: back to llm.stream, reply where it stopped."""
        t.gen = self.llm.stream(
            reply={"cache": [c.extract(0) for c in batch], "seen": t.seq,
                   "emitted": t.n},
            max_tokens=t.limit - t.n, tools=t.tools,
            response_format=t.response_format, sampling=t.sampling)

    @staticmethod
    def _join(batch, cache):
        """One more row: its cache merged alone, then appended to the batch."""
        one = [BatchKVCache.merge([c]) for c in cache]
        if batch is None:
            return one
        for b, c in zip(batch, one):
            b.extend(c)
        return batch

    def _step(self, rows, batch, pend):
        """Queue one forward for every row on `pend`, the previous forward's ids not yet read back, then hand `pend` out while the new one runs."""
        llm = self.llm
        y = (pend[:, None] if pend is not None
             else mx.array([[t.seq[-1]] for t in rows], mx.uint32))
        logits = llm.model(y, cache=batch)[:, -1, :]
        # generate_step's arithmetic, so a row picks what the solo path would
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        nxt = mx.argmax(logprobs, axis=-1)
        if any(t.sampling for t in rows):
            nxt = mx.concatenate([t.sampling(logprobs[i:i + 1]) if t.sampling else nxt[i:i + 1]
                                  for i, t in enumerate(rows)])
        mx.async_eval(nxt)
        if pend is None:
            return rows, batch, nxt
        rows, batch, keep = self._hand_out(rows, batch, pend)
        if keep is not None:
            # A finished row was one more forward in: its id there is dropped.
            nxt = nxt[mx.array(keep)] if keep else None
        return rows, batch, nxt

    def _hand_out(self, rows, batch, ids):
        """Give every row's next id to its request and drop finished rows from the batch; returns the rows left, the batch, and their indices in `rows`, None when every row stays."""
        keep = []
        for i, tok in enumerate(ids.tolist()):
            t = rows[i]
            t.seq.append(tok)
            if t.gone or self._emit(t, tok):
                self._finish(t)
            else:
                keep.append(i)
        if len(keep) == len(rows):
            return rows, batch, None
        if not keep:
            return [], None, keep
        for c in batch:
            c.filter(keep)
        return [rows[i] for i in keep], batch, keep

    def _room(self, active, batch, t):
        """Whether t can join `active` now: under the kernel's row cap, and with the widened batch's KV fitting --mem-pct beside everything MLX holds."""
        if not self.batch:
            return False
        if t.think_budget or any(a.think_budget for a in active):
            # Known limit: a turn with a thinking budget decodes alone, where llm.stream counts its tokens; per-row forcing in _step if dense batching under a budget matters.
            return False
        llm = self.llm
        cap = getattr(llm.model, "rowqmm_rows", None)
        if cap and len(active) >= cap:
            return False
        if t.prompt_len is None:
            t.prompt_len = self._prompt_len(t)
        lens = [len(llm._seen) if a.gen else len(a.seq) for a in active]
        left = [a.limit - a.n for a in active]
        per_pos = sum((c.keys.nbytes + c.values.nbytes)
                      / (c.keys.shape[0] * c.keys.shape[2])
                      for c in (batch or llm._cache))
        need = ((len(lens) + 1) * per_pos
                * (max(lens + [t.prompt_len]) + max(left + [t.limit])
                   + BatchKVCache.step))
        budget = MACHINE_BYTES * self.mem_pct / 100 - self.host_gb * 1e9
        return mx.get_active_memory() + need <= budget

    def _prompt_len(self, t):
        """The ids t's prompt renders to: what its prefill will write."""
        llm = self.llm
        self._think(t.thinking, t.think_budget)
        held, llm._tools = llm._tools, t.tools      # _raw_render reads them
        try:
            import tools as toolfmt
            return len(toolfmt.encode(llm.tokenizer, llm._raw_render(list(t.messages))))
        except Exception:
            return 0        # llm.stream raises the same, for this request alone
        finally:
            llm._tools = held

    def stats(self):
        if not self.llm:
            return {}
        return dict(self.llm.stats(), queue_s=self._queue_s, shadow=self._shadow_last)


def sse(data):
    return f"data: {json.dumps(data)}\n\n".encode()


IMAGE_TYPES = {"image/png": "png", "image/jpeg": "jpeg", "image/jpg": "jpeg",
               "image/webp": "webp", "image/gif": "gif"}


def check_image(url):
    """Raise ValueError unless `url` is a data: URL of a PNG, JPEG, WebP or GIF within
    MAX_IMAGE bytes (a remote URL is never fetched)."""
    if not isinstance(url, str) or not url.startswith("data:"):
        raise ValueError("images must be sent as data: URLs; the server does not fetch remote URLs")
    head, _, data = url[5:].partition(",")
    fields = head.lower().split(";")
    kind = IMAGE_TYPES.get(fields[0])
    if kind is None:
        raise ValueError(f"image type {fields[0] or '(none)'} is not supported; use PNG, JPEG, WebP or GIF")
    if fields[-1] != "base64":
        raise ValueError("an image data: URL must be base64")
    if len(data) > MAX_IMAGE * 4 // 3 + 4:
        raise ValueError(f"an image is larger than {MAX_IMAGE // 2**20} MB")
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("an image is not valid base64") from None
    if len(raw) > MAX_IMAGE:
        raise ValueError(f"an image is larger than {MAX_IMAGE // 2**20} MB")
    if not (raw.startswith(IMAGE_MAGIC[kind]) and (kind != "webp" or raw[8:12] == b"WEBP")):
        raise ValueError(f"an image is not a valid {kind.upper()} file")


def clean_messages(messages, vision=False):
    """The client's messages as the chat templates read them: string content, `developer` as `system`, and image parts kept only for an engine that can see (ValueError says why); a copy."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("role") == "developer":
            m["role"] = "system"
        parts = m.get("content")
        if isinstance(parts, list):
            parts = [p for p in parts if isinstance(p, dict)]
            if any(p.get("type") == "image_url" for p in parts):
                if not vision:
                    raise ValueError("this model has no vision part; images are not supported")
                keep = []
                for p in parts:
                    if p.get("type") == "image_url":
                        url = p.get("image_url")
                        url = url.get("url") if isinstance(url, dict) else url
                        check_image(url)
                        keep.append({"type": "image_url", "image_url": {"url": url}})
                    elif p.get("type") in (None, "text", "input_text"):
                        keep.append({"type": "text", "text": p.get("text") or ""})
                m["content"] = keep
            else:
                m["content"] = "\n".join(
                    p.get("text") or "" for p in parts
                    if p.get("type") in (None, "text", "input_text"))
        out.append(m)
    return out


class ThinkSplit:
    """A reply that starts inside the think block, cut at </think> while its
    pieces arrive: feed() returns the (reasoning, answer) text to send now,
    flush() the reasoning still held back at the end."""
    END = "</think>"

    def __init__(self):
        self.held, self.done, self.lead = "", False, True   # lead: the newlines after </think>

    def feed(self, piece):
        if self.done:
            return "", self._answer(piece)
        self.held += piece
        i = self.held.find(self.END)
        if i >= 0:
            reasoning, rest = self.held[:i], self.held[i + len(self.END):]
            self.held, self.done = "", True
            return reasoning, self._answer(rest)
        # A tail that could be the start of </think> waits for the next piece.
        k = next((k for k in range(min(len(self.END) - 1, len(self.held)), 0, -1)
                  if self.held.endswith(self.END[:k])), 0)
        out, self.held = self.held[:len(self.held) - k], self.held[len(self.held) - k:]
        return out, ""

    def _answer(self, text):
        if self.lead:
            text = text.lstrip("\n")
            self.lead = not text
        return text

    def flush(self):
        out, self.held = self.held, ""
        return out


def split_think(text):
    """(reasoning, answer) of a whole reply that starts inside the think block;
    a reply cut before </think> is all reasoning."""
    sp = ThinkSplit()
    reasoning, answer = sp.feed(text)
    return (reasoning + sp.flush()).strip(), answer


def with_call_ids(calls, req_id):
    """Ids unique to this reply. tools.parse names a call by its place in the
    reply (call_0_name), the same on every turn, and a client keeps its tool
    results by id across the whole conversation."""
    for i, c in enumerate(calls):
        c["id"] = f"call_{req_id[-8:]}_{i}"
    return calls


def model_sampling(path):
    """The temperature, top_k and top_p a model folder's generation_config.json samples with, empty when it is greedy or absent."""
    try:
        with open(Path(path).expanduser() / "generation_config.json") as f:
            g = json.load(f)
    except (OSError, ValueError):
        return {}
    if not g.get("do_sample"):
        return {}
    return {k: g[k] for k in ("temperature", "top_k", "top_p") if g.get(k) is not None}


class Handler(BaseHTTPRequestHandler):
    host: ModelHost = None  # set by main() before serve_forever
    sampling = False        # --sampling: honour a request's temperature/top_k/top_p/min_p/seed
    default_sampling = {}   # --model-sampling: the model's own sampling for a field a request leaves out
    adaptive_thinking = False   # --adaptive-thinking: an Anthropic request's adaptive thinking turns thinking on
    api_key = None          # when set, every request must present it

    def log_message(self, fmt, *args):
        pass  # chat.py-style terminals don't want per-request noise either

    # Local connections only: no auth, but nothing remote (share.local_refusal, the share proxy's rule too).
    def _local_only(self):
        why = share.local_refusal(self.client_address[0], self.headers)
        if why:
            self._error(403, why)
            return False
        return True

    def _check_key(self):
        """Accept a request only with the api key in Bearer or x-api-key, and tell the page it may
        ask for it."""
        auth = self.headers.get("Authorization")
        if auth and auth.lower().startswith("bearer "):
            got = auth[7:].strip()
        else:
            got = self.headers.get("X-Api-Key", "")
        if not got or not hmac.compare_digest(got, self.api_key):
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.send_header("X-Moe-Api-Key", "required")
            body = json.dumps({"error": {"message": "missing or wrong api key",
                                         "type": "invalid_request_error"}}).encode()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return False
        return True

    def _allowed(self):
        """With an api key, any host with the key; without one, only a local browser."""
        if self.api_key:
            return self._check_key()
        return self._local_only()

    def _page_allowed(self):
        """The page and its assets load without a key when one is set, so the page can ask for it."""
        if self.api_key:
            return True
        return self._local_only()

    def client_gone(self):
        """Whether the client has closed this connection: a non-blocking peek at the socket; behind --tls-cert, or without MSG_DONTWAIT (Windows), it cannot be told and stays False."""
        if isinstance(self.connection, ssl.SSLSocket) or not hasattr(socket, "MSG_DONTWAIT"):
            return False
        try:
            return self.connection.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT) == b""
        except BlockingIOError:
            return False        # still connected, nothing to read
        except OSError:
            return True

    def _json(self, code, body):
        payload = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        try:
            self.end_headers()
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass        # the client hung up (an abandoned turn ends here), nobody to tell

    def _error(self, code, message, **extra):
        """OpenAI's error body, which clients print as `error.message`."""
        self._json(code, dict(error={
            "message": message,
            "type": "invalid_request_error" if code < 500 else "server_error"},
            **extra))

    def _route(self):
        """The path without its query or trailing slash, with /models and /chat/completions reaching the /v1 routes."""
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        return "/v1" + path if path in ("/models", "/chat/completions") else path

    def _page(self):
        self._static("/index.html")

    def _static(self, url_path):
        """One file from web/, with the path made absolute and checked to stay
        inside that directory; no listing, no dot files, known types only."""
        rel = unquote(url_path or "").lstrip("/")
        if not rel:
            rel = "index.html"
        if "\x00" in rel or "\\" in rel or any(part.startswith(".") for part in Path(rel).parts):
            self._error(404, "not found")
            return
        root = WEB_DIR.resolve()
        try:
            target = (root / rel).resolve()
            target.relative_to(root)
        except (OSError, ValueError):
            self._error(404, "not found")
            return
        if target.is_dir():
            target = (target / "index.html").resolve()
            try:
                target.relative_to(root)
            except ValueError:
                self._error(404, "not found")
                return
        if not target.is_file():
            self._error(404, "not found")
            return
        ctype = STATIC_TYPES.get(target.suffix.lower())
        if ctype is None:
            self._error(404, "not found")
            return
        try:
            body = target.read_bytes()
        except OSError:
            self._error(404, "not found")
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        vendor = "vendor" in target.relative_to(root).parts
        self.send_header("Cache-Control", "public, max-age=31536000, immutable" if vendor
                         else "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        if target.suffix.lower() == ".html":
            # The page loads only its own files, talks only to this server, and
            # shows replies as data: say so, so slipped markup can do no more.
            self.send_header("Content-Security-Policy",
                             "default-src 'none'; script-src 'self'; "
                             "style-src 'self' 'unsafe-inline'; connect-src 'self'; "
                             "img-src 'self' data:; base-uri 'none'; form-action 'none'; "
                             "frame-ancestors 'none'; object-src 'none'")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n) if n else b"{}"
        return json.loads(raw or b"{}")

    def do_GET(self):
        path = self._route()
        # The page and its assets load without a key when one is set, so it can ask; the API cannot.
        if path in ("/", "/index.html") or path in PAGES or path not in API_GET:
            if not self._page_allowed():
                return
            if path in ("/", "/index.html"):
                self._page()
            elif path in PAGES:
                self._static("/" + PAGES[path])
            else:
                self._static(path)
            return
        if not self._allowed():
            return
        if path == "/healthz":
            h = self.host
            self._json(200, {"ok": h.llm is not None, "model": h.model_name,
                              "loaded": h.llm is not None, "vision": h.vision})
        elif path == "/status":
            self._json(200, self.host.status())
        elif path == "/admin/history":
            self._json(200, {"replies": self.host.counters.history(), "capacity": HISTORY_LEN})
        elif path == "/stats":
            h = self.host
            body = h.stats()
            body["uptime_s"] = round(time.perf_counter() - START_TIME, 1)
            body["rss_gb"] = round(stream_fast.max_rss() / 1e9, 2)
            self._json(200, body)
        elif path == "/v1/models":
            h = self.host
            now = int(time.time())
            data = []
            if h.model_name:
                data.append({"id": Path(h.model_name).name, "object": "model",
                             "created": now, "owned_by": "local", "vision": h.vision})
            self._json(200, {"object": "list", "data": data})
        elif path == "/admin/models":
            web_settings.get_models(self)
        elif path == "/admin/prepare":
            web_settings.get_prepare(self)
        elif path == "/admin/search":
            web_settings.get_search(self)
        elif path == "/v1/memory" and self.host.store is not None:
            # The memory routes exist only under --memory, so a server started
            # without it answers exactly what it always did: 404.
            body = self.host.store.stats()
            body["attached"] = self.host.core is not None
            body["pending_turns"] = len(self.host.turns)
            self._json(200, body)
        else:
            self._static(path)

    def do_POST(self):
        if not self._allowed():
            return
        path = self._route()
        if path == "/v1/chat/completions":
            self._completions()
        elif path in ("/v1/messages", "/v1/messages/count_tokens"):
            import anthropic_api        # the Messages API route, deferred like kvmem
            anthropic_api.post(self, path)
        elif path == "/admin/load":
            self._load()
        elif path == "/admin/prepare":
            web_settings.post_prepare(self)
        elif path == "/admin/delete":
            web_settings.post_delete(self)
        elif path == "/v1/memory/sleep" and self.host.store is not None:
            self._sleep()
        else:
            self._error(404, "not found")

    def do_DELETE(self):
        if not self._allowed():
            return
        path = self._route()
        if path == "/admin/prepare":
            web_settings.delete_prepare(self)
        elif path == "/admin/search":
            web_settings.delete_search(self)
        else:
            self._error(404, "not found")

    def _sleep(self):
        """Journal this session and rebuild the core if the plan changed, reporting which, behind any turn in flight."""
        try:
            msg = self.host.sleep()
        except Exception as e:
            self._error(503, f"sleep failed: {e}")
            return
        self._json(200, {"ok": True, "message": msg,
                          "memory": self.host.store.stats()})

    def _completions(self):
        try:
            if int(self.headers.get("Content-Length") or 0) > MAX_BODY:
                self._error(413, f"the request is larger than {MAX_BODY // 2**20} MB")
                return
            body = self._read_json()
            if not isinstance(body, dict):
                raise ValueError
        except ValueError:      # json.JSONDecodeError is one
            self._error(400, "invalid JSON body")
            return

        messages = body.get("messages")
        if not messages:
            self._error(400, "'messages' is required")
            return
        if not (isinstance(messages, list)
                and all(isinstance(m, dict) for m in messages)):
            self._error(400, "'messages' must be a list of objects")
            return
        try:
            messages = clean_messages(messages, self.host.vision)
        except ValueError as e:
            self._error(400, str(e))
            return
        # Thinking: `chat_template_kwargs.enable_thinking`, or the same key at
        # the top level; a request that says nothing gets the engine's default.
        kwargs = body.get("chat_template_kwargs")
        if kwargs is not None and not isinstance(kwargs, dict):
            self._error(400, "'chat_template_kwargs' must be an object")
            return
        think = (kwargs or {}).get("enable_thinking")
        if think is None:
            think = body.get("enable_thinking")
        if think is not None and not isinstance(think, bool):
            self._error(400, "'enable_thinking' must be true or false")
            return
        think_on = self.host.think_default if think is None else think
        # OpenAI renamed the field to max_completion_tokens (max_tokens is
        # deprecated there); a client that sends only the new name must not
        # be told to stop at the engine's default.
        max_tokens = body.get("max_tokens")
        if max_tokens is None:
            max_tokens = body.get("max_completion_tokens")
        if max_tokens is not None and not (
                isinstance(max_tokens, int) and not isinstance(max_tokens, bool)
                and max_tokens > 0):
            self._error(400, "'max_tokens' must be a positive integer")
            return
        stream = bool(body.get("stream", False))
        # stream_options.include_usage: OpenAI's opt-in for the closing usage
        # chunk of a stream (editor extensions send it, the chat page too).
        opts = body.get("stream_options")
        usage_chunk = isinstance(opts, dict) and bool(opts.get("include_usage"))
        # Tools are rendered by the checkpoint's template ahead of the system message, so an unchanged list costs a prefill once and a changed one invalidates only that prefix.
        tools = body.get("tools") or None
        if tools is not None and not isinstance(tools, list):
            self._error(400, "'tools' must be a list")
            return
        # response_format is honoured like the tool schemas: a JSON schema proposes verify rows and never masks, so the reply is still the model's greedy text.
        response_format = body.get("response_format") or None
        if response_format is not None and not isinstance(response_format, dict):
            self._error(400, "'response_format' must be an object")
            return
        # tool_choice is accepted and ignored, since refusing a field every OpenAI client sends would only make this less compatible.

        # Sampling knobs are honoured under --sampling and greedy without it.
        sampling = None
        if self.sampling:
            try:
                sampling = Sampling.from_request(
                    {**self.default_sampling, **{k: v for k, v in body.items() if v is not None}})
            except (TypeError, ValueError) as e:
                self._error(400, f"sampling: {e}")
                return

        model_id = body.get("model") or "local"
        created = int(time.time())
        req_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

        if stream:
            self._stream_completion(messages, max_tokens, req_id, created,
                                     model_id, tools, response_format, sampling,
                                     usage_chunk, think, think_on)
        else:
            # X-Warmup: 1 marks the launcher's warm-up reply, which the counters and measured speeds leave out.
            info = {"warmup": True} if self.headers.get("X-Warmup") == "1" else {}
            try:
                text = self.host.generate(messages, max_tokens, stream=False,
                                          tools=tools,
                                          response_format=response_format,
                                          sampling=sampling, info=info, thinking=think,
                                          client_gone=self.client_gone)
            except Exception as e:
                traceback.print_exc()
                self._error(503, f"generation failed: {e}")
                return
            # Without tools the reply is the client's verbatim, so a client that keeps its own tool sheet in the prompt never gets content "" and finish "stop" from a block the parser cannot read.
            reasoning = None
            if think_on:
                reasoning, text = split_think(text)
            if tools:
                import tools as toolfmt      # deferred, like kvmem below
                content, calls = toolfmt.parse(text, tools)
            else:
                content, calls = text, []
            msg = {"role": "assistant", "content": content or None}
            if reasoning:
                msg["reasoning_content"] = reasoning
            if calls:
                msg["tool_calls"] = with_call_ids(calls, req_id)
            out = {
                "id": req_id, "object": "chat.completion", "created": created,
                "model": model_id,
                "choices": [{"index": 0,
                             # The reason a client uses to decide whether to run
                             # the functions and call back, so it has to say
                             # which happened.
                             "finish_reason": self._finish_reason(calls, info, max_tokens),
                             "message": msg}],
            }
            if self._usage(info):
                out["usage"] = self._usage(info)
            store = getattr(self.host, "store", None)
            last = next((m for m in reversed(messages)
                         if m.get("role") == "user"), None)
            if store is not None and last is not None:
                import kvmem            # deferred like _open_memory's import
                q = str(last.get("content", ""))
                if q.rstrip().endswith("?") and kvmem.FIRST_PERSON.search(q):
                    # A question about the user is checked against the memory core and reported, not suppressed (kvmem.ground).
                    ok, why = kvmem.ground(text, store.core_text(), question=q)
                    out["memory"] = {"grounded": ok, "unfounded": why[:5]}
            self._json(200, out)

    def _usage(self, info):
        """OpenAI's `usage` from what the worker counted, or None."""
        p, c = info.get("prompt_tokens"), info.get("completion_tokens")
        if p is None or c is None:
            return None
        return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}

    def _finish_reason(self, calls, info, max_tokens):
        """tool_calls, length (the reply used up its token limit) or stop."""
        if calls:
            return "tool_calls"
        limit = max_tokens or getattr(self.host.llm, "max_tokens", None)
        n = info.get("completion_tokens")
        return "length" if limit and n is not None and n >= limit else "stop"

    def _stream_completion(self, messages, max_tokens, req_id, created,
                            model_id, tools=None, response_format=None, sampling=None,
                            usage_chunk=False, think=None, think_on=False):
        info = {}
        try:
            pieces = self.host.generate(messages, max_tokens, stream=True,
                                        tools=tools,
                                        response_format=response_format,
                                        sampling=sampling, info=info, thinking=think,
                                        client_gone=self.client_gone)
            # Start the generator before the 200 goes out, so a missing model raises while the status can still change.
            first = next(pieces, "")
        except Exception as e:
            traceback.print_exc()
            self._error(503, f"generation failed: {e}")
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def chunk(delta, finish_reason=None):
            return sse({
                "id": req_id, "object": "chat.completion.chunk",
                "created": created, "model": model_id,
                "choices": [{"index": 0, "delta": delta,
                             "finish_reason": finish_reason}],
            })

        try:
            if tools:
                # A tool-enabled request is collected and sent as one delta, since a client cannot act on half a call; known limit: stream the prefix before the first "<tool_call" if an application needs tokens and tools.
                import tools as toolfmt
                whole = first + "".join(pieces)
                reasoning = None
                if think_on:
                    reasoning, whole = split_think(whole)
                content, calls = toolfmt.parse(whole, tools)
                delta = {"role": "assistant", "content": content or None}
                if reasoning:
                    delta["reasoning_content"] = reasoning
                if calls:
                    # A streamed call carries its place in the reply (`index`):
                    # that is how a client assembles the pieces of one.
                    delta["tool_calls"] = [dict(c, index=i) for i, c in
                                           enumerate(with_call_ids(calls, req_id))]
                self.wfile.write(chunk(delta))
                self.wfile.write(chunk(
                    {}, finish_reason=self._finish_reason(calls, info, max_tokens)))
            else:
                # Thinking on: the reply starts inside the think block, and
                # its text before </think> goes out as reasoning_content.
                split = ThinkSplit() if think_on else None

                def emit(piece):
                    if split is None:
                        self.wfile.write(chunk({"content": piece}))
                        return
                    for key, text in zip(("reasoning_content", "content"), split.feed(piece)):
                        if text:
                            self.wfile.write(chunk({key: text}))

                self.wfile.write(chunk({"role": "assistant",
                                        "content": "" if split else first}))
                if split:
                    emit(first)
                self.wfile.flush()
                for piece in pieces:
                    emit(piece)
                    self.wfile.flush()
                rest = split.flush() if split else ""
                if rest:
                    self.wfile.write(chunk({"reasoning_content": rest}))
                self.wfile.write(chunk(
                    {}, finish_reason=self._finish_reason([], info, max_tokens)))
            usage = self._usage(info)
            if usage_chunk and usage:
                # Last chunk, no choices: what OpenAI sends under include_usage.
                self.wfile.write(sse({
                    "id": req_id, "object": "chat.completion.chunk",
                    "created": created, "model": model_id,
                    "choices": [], "usage": usage}))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            # Client hung up mid-stream: closing `pieces` stops the turn at
            # its next token, on a dense target (_Turn.pieces) and a serial one.
            pieces.close()
        except Exception as e:
            # The engine failed after the 200 went out: an error event is the
            # only way left to say so, and OpenAI clients read this shape.
            traceback.print_exc()
            try:
                self.wfile.write(sse({"error": {"message": f"generation failed: {e}",
                                                "type": "server_error"}}))
                self.wfile.flush()
            except OSError:
                pass

    def _load(self):
        try:
            body = self._read_json()
            if not isinstance(body, dict):
                raise ValueError
        except ValueError:
            self._error(400, "invalid JSON body")
            return
        model = body.get("model")
        if not model:
            self._error(400, "'model' is required")
            return
        if not isinstance(model, str):
            self._error(400, "'model' must be a path")
            return
        kwargs = {}
        for key, cast in (("resident", int), ("cache_bias", float)):
            if key in body:
                try:
                    kwargs[key] = cast(body[key])
                except (TypeError, ValueError):
                    self._error(400, f"'{key}' must be a number")
                    return
        if "stub" in body:
            # Validated by StreamingLLM, which raises on an unknown mode; the
            # load path below turns that into a 503 with the reason in it.
            kwargs["stub"] = str(body["stub"])
        if "mtp" in body:
            # The MTP head belongs to one checkpoint, so a swap brings its own: true is the default path, a string that path, false or null ordinary decode.
            m = body["mtp"]
            if not isinstance(m, (bool, str, type(None))):
                self._error(400, "'mtp' must be true, false or a path")
                return
            if m is True and not DEFAULT_MTP:
                self._error(400, "'mtp': true needs a default head and this install has none; pass a path")
                return
            kwargs["mtp"] = DEFAULT_MTP if m is True else (m or None)
        try:
            kwargs = web_settings.load_kwargs(self.host, body, kwargs)
        except ValueError as e:
            self._error(400, str(e))
            return
        try:
            self.host.swap(model, **kwargs)
        except Exception as e:
            web_settings.after_load(self.host, False)
            # A bad load never kills the port: if the old model did not survive either, host.llm is None and generation answers 503 until a good load arrives.
            self.host.last_error = str(e)
            self._error(503, f"load failed, still serving previous model "
                             f"if any: {e}", model=self.host.model_name,
                        loaded=self.host.llm is not None)
            return
        self._json(200, {"ok": True, "model": self.host.model_name,
                         **web_settings.after_load(self.host, True)})


class _Server(ThreadingHTTPServer):
    request_queue_size = 128            # the default 5 resets some of the parallel connections of one page load


def port_error(host, port):
    """None when host:port can be bound now, else a one-line reason, so a busy port shows before the model loads."""
    try:
        with _Server((host, port), BaseHTTPRequestHandler):
            return None
    except OSError as err:
        return (f"cannot listen on {host}:{port} ({err.strerror or err}); another program may hold the port, "
                "pick a free one with --port")


def serve(host_obj, port, host="127.0.0.1", cert=None, key=None):
    Handler.host = host_obj
    httpd = _Server((host, port), Handler)
    if cert and key:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(cert, key)
        httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
    # Known limit: one thread per connection with all generation on the worker, so a bounded pool would matter only past enough idle connections to exhaust threads.
    return httpd


def build_parser():
    ap = argparse.ArgumentParser()
    add_engine_flags(ap)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1",
                    help="the address to bind (default 127.0.0.1; anything else needs --api-key)")
    ap.add_argument("--api-key", default=None,
                    help="require this key as Bearer or x-api-key on every request")
    ap.add_argument("--tls-cert", default=None, help="a PEM certificate for https (with --tls-key)")
    ap.add_argument("--tls-key", default=None, help="the private key for --tls-cert")
    ap.add_argument("--prefill-step", type=int, default=None,
                    help="prefill this many tokens at a time. Default: fronts "
                         "and MTP in one pass, other prompts in the library's "
                         "chunks (2048, 512 with a draft), the memory-bounded "
                         "setting. 0 = one pass while it fits --prefill-gb, "
                         "else the fewest equal passes that do: the soonest "
                         "first token on long messages to a streamed MoE "
                         "(each chunk re-reads its experts)")
    ap.add_argument("--mem-peak-max-gb", type=float, default=None, metavar="N",
                    help="with --mem-peak-gb G: let the expert store grow toward "
                         "N GB while the machine's swap-line margin allows, and "
                         "give it back under memory pressure (off by default)")
    ap.add_argument("--no-reuse", action="store_true",
                    help="prefill every request's whole conversation instead "
                         "of taking up the last request's cache where the new "
                         "one extends it (also under --mem-peak-gb)")
    ap.add_argument("--keep-convs", type=int, default=1, metavar="N",
                    help="keep the caches of the last N conversations for "
                         "reuse, not only the last request's: a title request "
                         "or a subagent in between then leaves the agent's "
                         "next step its cache (default 1)")
    ap.add_argument("--no-shadow", action="store_true",
                    help="do not prefill, while the user reads a reply, the "
                         "history the next message will re-render (a kept "
                         "conversation of its own, within --keep-convs / "
                         "--keep-gb; StreamingLLM.shadow; a request stops it "
                         "and takes over its finished layers)")
    ap.add_argument("--keep-gb", type=float, default=2.0, metavar="G",
                    help="with --keep-convs: the kept conversations hold at "
                         "most G GB together, the one a request extends "
                         "counted at the size it will reach and never dropped "
                         "for its own size; the least recently used go first "
                         "(default 2)")
    ap.add_argument("--prefill-gb", type=float, default=None, metavar="G",
                    help="keep a streamed MoE's prefill under G GB of process "
                         "memory (default: 35%% of the machine, 6.0 GB on 16 "
                         "GB); --prefill-step 0 then means one pass while it "
                         "fits, the widest passes that fit after that")
    ap.add_argument("--sampling", action="store_true",
                     help="honour a request's temperature, top_k, top_p, min_p and seed "
                          "(temperature 0 or absent stays greedy); --mtp then samples by exact "
                          "speculative sampling. Off: every request is greedy, as before")
    ap.add_argument("--model-sampling", action="store_true",
                    help="--sampling, and a request that sets no temperature, top_k or top_p takes "
                         "the model's own from its generation_config.json (Claude Code sends none)")
    ap.add_argument("--adaptive-thinking", action="store_true",
                    help="answer the Anthropic route's adaptive thinking, which Claude Code always "
                         "asks for, with thinking on. Off: only thinking type enabled turns it on")
    ap.add_argument("--thinking-budget", type=int, default=None, metavar="N",
                    help="a thinking turn thinks at most N tokens, then the server writes Qwen's "
                         "early-stop text, closing the think block, and the model answers; an "
                         "Anthropic request's thinking.budget_tokens below N wins (off by default)")
    return ap


def memory_line(args, in_use):
    """One line about memory: the most the engine may take, and the bytes it holds now."""
    if args.mem_peak_gb:
        top = f"{args.mem_peak_gb:g} GB peak"
        if args.mem_peak_max_gb:
            top += f", growing toward {args.mem_peak_max_gb:g} GB while the machine allows"
    else:
        top = f"{MACHINE_BYTES * args.mem_pct / 100e9 - args.host_gb:.1f} GB ({args.mem_pct}% of this machine)"
    return f"memory: budget {top}" + (f", {in_use / 1e9:.1f} GB in use" if in_use else "")


def main():
    ap = build_parser()
    ap.add_argument("--tuned", action="store_true", help="start from the cached first-run tuning (the launcher does)")
    if tuner and tuner.opted_in(sys.argv) and tuner.model_from_argv(sys.argv):
        try:                                 # the cached tuning is a default: any flag given wins
            ap.set_defaults(**tuner.parser_defaults(tuner.model_from_argv(sys.argv), sys.argv))
        except Exception:
            pass
    args = finish(ap.parse_args())
    if args.keep_convs < 1:
        ap.error("--keep-convs must be at least 1 (--no-reuse keeps none)")
    if args.mem_peak_max_gb is not None and not (
            args.mem_peak_gb and args.mem_peak_max_gb > args.mem_peak_gb):
        ap.error("--mem-peak-max-gb N needs --mem-peak-gb G with G < N")
    if args.host not in ("127.0.0.1", "::1", "localhost") and not args.api_key:
        ap.error("--api-key is required when --host binds beyond 127.0.0.1")
    if args.thinking_budget is not None and args.thinking_budget < 1:
        ap.error("--thinking-budget must be at least 1")
    why = port_error(args.host, args.port)
    if why:
        raise SystemExit("server.py: " + why)
    Handler.api_key = args.api_key

    t0 = time.perf_counter()
    host_obj = ModelHost(args.model, memory=args.memory,
                          arena_min=args.arena_min, arena_max=args.arena_max,
                          mem_peak_max_gb=args.mem_peak_max_gb,
                          reuse=not args.no_reuse,
                          keep_convs=args.keep_convs, keep_gb=args.keep_gb,
                          shadow=not args.no_shadow,
                          prefill_step=args.prefill_step,
                          prefill_gb=args.prefill_gb,
                          sleep_after=args.sleep_after,
                          **engine_kwargs(args))
    load_s = time.perf_counter() - t0
    host_obj.start_idle_sleeper()
    host_obj.start_arena_watch()
    Handler.sampling = args.sampling or args.model_sampling
    if args.model_sampling:
        Handler.default_sampling = model_sampling(args.model)
        note(f"sampling: the model's {Handler.default_sampling or 'greedy'} where a request sets none")
    Handler.adaptive_thinking = args.adaptive_thinking
    host_obj.think_budget = args.thinking_budget
    httpd = serve(host_obj, args.port, host=args.host, cert=args.tls_cert, key=args.tls_key)
    reuse = host_obj.reuse
    kept = (f", {args.keep_convs} conversations kept within "
            f"{args.keep_gb:g} GB" if reuse and args.keep_convs > 1
            else "")
    if reuse and host_obj.llm._budget is not None:
        kept += " while the budget has room for them beside its pool"
    note(f"serving {args.model}: "
         f"{'concurrent turns batched' if host_obj.batch else 'one turn at a time'}"
         f"; conversation reuse {'on' if reuse else 'off'}{kept}")
    print(f"model: {Path(args.model).name}, loaded in {load_s:.0f} s")
    print(memory_line(args, stream_fast.phys_footprint()))
    scheme = "https" if args.tls_cert and args.tls_key else "http"
    print(f"serving on {scheme}://127.0.0.1:{args.port} (API base {scheme}://127.0.0.1:{args.port}/v1)")
    print(f"chat in the browser: {scheme}://127.0.0.1:{args.port}/", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        # The exit consolidation guards against any failure, Ctrl-C included, because a memory that can hang or crash a shutdown is worse than one that skips a session.
        if host_obj.store is not None:
            try:
                print("consolidating memory (Ctrl-C to skip)...", flush=True)
                print(f"  {host_obj.sleep()}")
            except BaseException as e:
                print(f"  memory: sleep failed ({e}); the turns since the "
                      f"last /v1/memory/sleep are not in the core")


if __name__ == "__main__":
    main()
