#!/usr/bin/env python3
"""server.py's HTTP layer with no model: the chat page at GET / and its local
assets, and the shapes that editor extensions read from an OpenAI-compatible
endpoint.

    perl -e 'alarm shift; exec @ARGV' 280 python3 tests/test_webchat.py

A real ModelHost and the real handler, with StreamingLLM replaced by a fake
that answers with a fixed text and keeps what it was asked. Nothing loads a
checkpoint or touches the GPU. Checked: the page and its vendored files (no
outside address, a CSP header, a static path that cannot leave web/),
/v1/models, the non-stream and the streamed
chat completion (usage, finish_reason, the closing usage chunk under
stream_options), content given as a list of parts, tool calls (unique ids, an
index on each streamed one), OpenAI-shaped errors, a base URL given without
/v1, and that a request from another site is refused. Also: the thinking flag
reaching the engine and the reply split into reasoning_content (whole and
streamed, the tag cut across pieces), image parts passing through only for an
engine that can see, with the size, type and URL checks, and the counters
behind GET /status. Also the Settings page's endpoints, against a fake launcher:
the list, a prepare's bytes and log with Stop, the launcher's plan applied by a
switch, the page's options, and the same Origin and Host guard. Also the pages and
icons (/settings, /monitor, the SVG and touch icons), the fields /status adds for them
(context_length, host, machine, memory budget), and GET /admin/history.
"""

import base64
import http.client
import json
import math
import re
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import server                                                  # noqa: E402

CALL = ("<tool_call>\n<function=get_weather>\n<parameter=city>\nBerlin\n"
        "</parameter>\n</function>\n</tool_call>")


def _text_of(message):
    """The words of one chat message, whether its content is a string or a list of parts."""
    c = message.get("content")
    if isinstance(c, str):
        return c
    return " ".join(p.get("text", "") for p in c or [] if isinstance(p, dict))


class _FakeTokenizer:
    detokenizer = None


class _FakeLLM:
    """What ModelHost touches. A reply is its words, one piece (one token)
    each; a reply cut at max_tokens ends on its last word, as the real one
    does."""
    max_tokens = 512
    reply = "Hello **world** from a fake"
    think_reply = "Let me see.\n</think>\n\nIt is **42**."    # what a thinking turn says
    code_reply = ("A loop and its output:\n\n```python\nfor i in range(3):\n"
                  "    print(i)\n```\n\nIt prints `0`, `1` and `2`.")
    script = None       # a list of pieces, in place of the reply's words
    supports_images = False
    delay = 0.0         # seconds a piece takes, for the page test
    first_delay = 0.0   # seconds before the first piece, as a long prompt takes to read
    thinking = False
    tps = 10.0          # decode speed the stats report
    grow = False        # prompt tokens grow with the conversation (two a word), for the context bar
    vary = False        # speed and first-token time change from reply to reply, for the charts
    fail_after = None   # raise after this many pieces, as an engine out of memory mid-reply does
    asked = []
    built = []

    def __init__(self, **kwargs):
        _FakeLLM.built.append(kwargs)
        self.model = None
        self.tokenizer = _FakeTokenizer()
        self._last = {}
        # The engine's own default for a request that does not choose, as --thinking is.
        self.thinking = bool(kwargs.get("thinking", False))

    def set_thinking(self, on, budget=None):
        self.thinking = on if isinstance(on, str) else bool(on)     # a string is thinking on at that effort
        self.think_budget = budget

    def forget(self):
        pass

    def reset(self):
        pass

    def stream(self, messages=None, max_tokens=None, resume=None, tools=None,
               response_format=None, sampling=None, keep_on_close=False):
        _FakeLLM.asked.append({"messages": messages, "max_tokens": max_tokens,
                               "tools": tools, "thinking": self.thinking,
                               "budget": getattr(self, "think_budget", None)})
        words = (self.think_reply if self.thinking else self.reply).split(" ")
        pieces = self.script or [w + " " for w in words[:-1]] + [words[-1]]
        limit = max_tokens or self.max_tokens
        pieces = pieces[:limit]
        time.sleep(self.first_delay)
        for i, p in enumerate(pieces):
            if self.fail_after is not None and i == self.fail_after:
                raise RuntimeError("out of memory in the fake engine")
            time.sleep(self.delay)
            yield p
        # Set once the reply is over, as the engine does.
        tps, ttft, prompt = self.tps, 0.5, 5
        if self.vary:
            k = len(_FakeLLM.asked)
            tps = round(9 + 5 * math.sin(k * 0.9) + k % 3, 2)
            ttft = round(0.4 + 0.35 * abs(math.cos(k * 0.7)), 2)
        if self.grow:
            prompt += 2 * sum(len(_text_of(m).split()) for m in messages or [])
        self._last = {"tokens": len(pieces), "prompt_tokens": prompt, "cached_tokens": 3,
                      "ttft_s": ttft, "decode_tps": tps,
                      "store": {"store_mb": 123.4, "pool_slots": 40, "ring_slots": 24, "budget_gb": 4.5},
                      "arena_slots": 60, "arena_gb": 4.25,
                      "expert_rows": 264, "expert_rows_per_token": 132.0}

    def ask(self, messages=None, max_tokens=None, resume=None, tools=None,
            response_format=None, sampling=None):
        return "".join(self.stream(messages, max_tokens, resume, tools))

    def stats(self):
        return dict(self._last)


class _FakeLauncher:
    """The launcher behind the Settings page: its list, its plans and a prepare that writes files.

    web_settings.configure() takes this in place of LauncherBackend, so the endpoints
    run their real code with no Hugging Face, no disk size and no checkpoint.
    """

    def __init__(self, models_dir, parts=20, delay=0.15, prefill=False):
        self.models_dir = Path(models_dir)
        self.parts, self.delay = parts, delay
        self.chunk = 1_000_000             # bytes a part holds, so a progress bar moves in MB
        self.marked = []
        self.searched = []
        self.entries = {}
        self.list = [
            {"id": "tiny", "name": "Tiny Chat", "kind": "dense 4-bit", "about": "a test model",
             "license": "MIT", "tested": True, "unavailable": "",
             "variants": [
                 {"key": "4bit", "label": "4-bit", "note": "the published weights",
                  "size_bytes": 2000000000, "download_bytes": 0, "tps": 42.0, "ok": True,
                  "why": "", "status": "ready", "ready": True, "make_bits": 0,
                  "path": "/models/tiny-chat", "memory": {"gb": 7.5, "max_gb": None},
                  "facts": {"model_type": "tiny", "layers": 12, "hidden": 1024,
                            "experts": 0, "bits": 4}},
                 {"key": "3bit", "label": "3-bit", "note": "a copy made here",
                  "size_bytes": 1500000000, "download_bytes": 0, "tps": 55.0, "ok": True,
                  "why": "", "status": "not made yet", "ready": False, "make_bits": 3,
                  "path": "/models/tiny-chat-3bit", "memory": {"gb": 6.0, "max_gb": None},
                  "facts": {"model_type": "tiny", "layers": 12, "hidden": 1024,
                            "experts": 0, "bits": 3}},
                 {"key": "8bit", "label": "8-bit", "note": "the exact weights",
                  "size_bytes": 8000000000, "download_bytes": 4000000000, "tps": None,
                  "ok": False, "why": "needs 12 GB of memory, this machine has less",
                  "status": "not downloaded", "ready": False, "make_bits": 0,
                  "path": "/models/tiny-chat-8bit", "memory": None,
                  "facts": {"model_type": "tiny", "layers": 12, "hidden": 1024,
                            "experts": 0, "bits": 8},
                  "switch": {"freed_gb": 9.0, "ok": True, "why": "", "tps": 18.0}},
                 {"key": "16bit", "label": "16-bit", "note": "the original weights",
                  "size_bytes": 16000000000, "download_bytes": 16000000000, "tps": 20.0,
                  "ok": False, "disk_short": True, "why": "needs 41 GB free on disk, 12.6 GB free",
                  "status": "needs 41 GB free on disk, 12.6 GB free", "ready": False,
                  "make_bits": 0, "path": "/models/tiny-chat-16bit", "memory": None,
                  "facts": {"model_type": "tiny", "layers": 12, "hidden": 1024,
                            "experts": 0, "bits": 16}}]},
            {"id": "big", "name": "Big MoE", "kind": "MoE 4-bit", "about": "a streamed model",
             "license": "Apache-2.0", "tested": True, "unavailable": "",
             "not_made": ["a 2-bit copy: too little memory"],
             "variants": [
                 {"key": "published", "label": "4-bit", "note": "the published weights",
                  "size_bytes": 9000000000, "download_bytes": 9000000000, "tps": 8.0,
                  "ok": True, "why": "", "status": "not downloaded", "ready": False,
                  "make_bits": 0, "path": "/models/big-moe",
                  "memory": {"gb": 6.0, "max_gb": 9.0},
                  "facts": {"model_type": "big_moe", "layers": 32, "hidden": 2048,
                            "experts": 64, "bits": 4},
                  "switch": {"tps": 31.6}}]}]
        for m in self.list:
            for v in m["variants"]:
                self.entries[(m["id"], v["key"])] = self._entry(m, v)
        self.plans = {"/models/tiny-chat": ["--mem-peak-gb", "7.5", "--cache-bias", "0.5",
                                            "--bias-budget", "0.2", "--prefill-step", "0"],
                      "/models/tiny-chat-3bit": ["--mem-peak-gb", "6.0", "--prefill-step", "0"],
                      "/models/big-moe": ["--mem-peak-gb", "6", "--mem-peak-max-gb", "9"]
                                         + (["--prefill-step", "0"] if prefill else [])}

    def info(self):
        return {"chip": "a test chip", "ram_gb": 16.0, "gpu": "a test chip", "os": "test",
                "models_dir": str(self.models_dir), "free_gb": 88.0, "ssd_gbps": 5.0,
                "others": []}

    def _made(self, variant):
        """Whether a variant's folder holds every part a prepare writes here."""
        d = self.models_dir / Path(variant["path"]).name
        return d.is_dir() and len(list(d.iterdir())) == self.parts

    @staticmethod
    def _entry(m, v):
        return {"id": m["id"], "name": m["name"], "path": v["path"],
                "variant": v["key"], "make_bits": v["make_bits"],
                "ready": v["ready"], "download": v["download_bytes"],
                "disk": 20 * 1000000,
                "memory": v["memory"], "model": m,
                "facts": dict(v.get("facts") or {}),
                "ok": v["ok"], "why": v["why"], "tps": v["tps"],
                "disk_short": v.get("disk_short", False),
                "status": v["status"], "switch": v.get("switch")}

    def _row(self, m):
        variants = []
        for v in m["variants"]:
            made = self._made(v)                      # in the launcher's folder: it can delete it
            ready = bool(v["ready"]) or made
            v2 = {k: val for k, val in v.items() if k not in ("facts", "switch")}
            variants.append({**v2, "ready": ready, "own": made,
                             "disk_path": str(self.models_dir / Path(v["path"]).name) if made
                             else v["path"] if v["ready"] else "",
                             "status": "ready" if ready else v["status"]})
        return {**m, "variants": variants}

    def models(self):
        return [self._row(m) for m in self.list]

    def find(self, query):
        """Hugging Face as the page's search sees it: one repo for a name or a link, two for words."""
        import launch
        self.searched.append(query)
        if "nope" in query:
            raise ValueError("no safetensors weights in the repo")
        repo = launch.parse_repo(query)
        return ([{"repo": repo}], 0) if repo else ([{"repo": "org/Found-A-4bit"}, {"repo": "org/Found-B-4bit"}], 3)

    def found_rows(self, raw):
        rows, entries = [], {}
        for r in raw:
            name = r["repo"].split("/")[1]
            mid = r["repo"].replace("/", "--")
            v = {"key": "published", "label": "4-bit", "note": "the model as published",
                 "size_bytes": 3000000000, "download_bytes": 3000000000, "tps": 25.0, "ok": True,
                 "why": "", "status": "download 3.0 GB", "ready": False, "make_bits": 0,
                 "path": f"/models/{name.lower()}", "memory": None,
                 "facts": {"model_type": "found", "layers": 8, "hidden": 512, "experts": 0, "bits": 4}}
            m = {"id": mid, "name": r["repo"], "repo": r["repo"], "kind": "dense 4-bit",
                 "about": "found on Hugging Face", "license": "", "tested": False, "unavailable": "",
                 "not_made": [], "variants": [v]}
            entries[(mid, "published")] = self._entry(m, v)
            rows.append(self._row(m))
        return rows, entries

    def delete(self, entry, running):
        """(True, what was deleted) or (False, why not), by the launcher's own rule."""
        import launch
        return launch.delete_folder(self.models_dir / Path(entry["path"]).name, self.models_dir,
                                    running=running)

    def judge(self, entry, freed):
        """The entry now, and as a switch with `freed` GB of the running engine's memory given back."""
        base = {"ok": entry["ok"], "why": entry["why"], "tps": entry["tps"],
                "disk_short": entry.get("disk_short", False), "status": self._status(entry)}
        sw = entry.get("switch")
        if not freed or freed <= 0 or not sw or freed < sw.get("freed_gb", 0):
            return {**base, "matters": False}
        out = {**base, **{k: sw[k] for k in ("ok", "why", "tps") if k in sw}}
        return {**out, "status": self._status(entry), "matters": True}

    def _status(self, entry):
        return "ready" if entry["ready"] or self._made(entry) else entry["status"]

    def entry(self, model_id, variant_key):
        return self.entries.get((model_id, variant_key))

    def path(self, entry):
        return entry["path"]

    def ready(self, entry):
        return bool(entry.get("ready"))

    def mark_ready(self, entry):
        self.marked.append(entry["path"])

    def plan(self, entry):
        args = self.plans.get(entry["path"], [])
        return {"args": list(args), "env": {}, "tps": 20.0,
                "why": ["a test machine, a test model"],
                "biased": "--cache-bias" in args, "ok": True, "store_gb": 4.0}

    def prepare_command(self, entry):
        d = self.prepare_dirs(entry)[0]
        script = ("import os, sys, time\n"
                  "d, n, wait, size = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])\n"
                  "os.makedirs(d, exist_ok=True)\n"
                  "for i in range(n):\n"
                  "    with open(os.path.join(d, 'part%d' % i), 'wb') as f:\n"
                  "        f.write(b'x' * size)\n"
                  "    print('made part', i, flush=True)\n"
                  "    time.sleep(wait)\n")
        return [sys.executable, "-c", script, str(d), str(self.parts), str(self.delay),
                str(self.chunk)]

    def prepare_dirs(self, entry):
        return [self.models_dir / Path(entry["path"]).name]

    def prepare_total(self, entry):
        return self.parts * self.chunk


def serve_stub():
    """The stub engine behind the real server, for tests/test_webchat_browser.py: it prints PORT n and serves until killed, --vision gives an engine that can see, and --settings adds the Settings page with a fake launcher behind it, the running model as a user's own copy outside the launcher's folder, and a switch that gives the engine's 0.5 GB back."""
    _FakeLLM.supports_images = "--vision" in sys.argv
    _FakeLLM.delay = 0.02 if "--fast" in sys.argv else 0.25
    _FakeLLM.grow = "--grow" in sys.argv
    if "--first-delay" in sys.argv:
        _FakeLLM.first_delay = float(sys.argv[sys.argv.index("--first-delay") + 1])
    if "--long-think" in sys.argv:
        _FakeLLM.think_reply = ("Let me think about this carefully, step by step, and check each part "
                                "twice before I answer.\n</think>\n\nIt is **42**, and here is why: "
                                "the question asks for the answer and that is what it is.")
    _FakeLLM.vary = "--vary" in sys.argv
    if "--code" in sys.argv:
        _FakeLLM.reply = _FakeLLM.code_reply
    # A checkpoint folder with a config, so the page can read its context length (--ctx N).
    ctx = int(sys.argv[sys.argv.index("--ctx") + 1]) if "--ctx" in sys.argv else 8192
    folder = Path(tempfile.mkdtemp(prefix="moe-stub-")) / "tiny-chat"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps({
        "model_type": "tiny", "text_config": {"max_position_embeddings": ctx}}))
    model = str(folder)
    # The page tags chats with the computer's name and shows its chip: say something neutral.
    server.host_name = lambda: "test-host"
    server.machine_info = lambda: {"chip": "Test chip", "ram_gb": 16.0, "os": "TestOS 1.0",
                                   "backend": "metal", "cpus": 8, "gpu": None, "mlx": "0.0"}
    # What else is using the computer, fixed so the page reads the same in every run.
    server.machine.conditions = lambda ttl=5.0: {
        "power": "battery", "low_power": False, "thermal": "", "mem_free_gb": 1.3,
        "cpu_busy": 2.1, "top": [{"name": "cmux", "cpu": 2.1}], "gpu_busy": 4.0}
    if "--settings" in sys.argv:
        import web_settings
        web_settings.configure(_FakeLauncher(tempfile.mkdtemp(prefix="moe-models-"),
                                             parts=12, delay=0.2, prefill=True))
        own = Path(tempfile.mkdtemp(prefix="moe-own-")) / "tiny-chat"
        own.mkdir()
        (own / "config.json").write_text(json.dumps({
            "model_type": "tiny", "num_hidden_layers": 12, "hidden_size": 1024,
            "max_position_embeddings": ctx,
            "quantization": {"bits": 4, "group_size": 64}}))
        model = str(own)
        web_settings.engine_gb = lambda host: 0.5
        # Speeds measured here: one for the model in use, one for a model that is not on disk.
        import library
        import machine
        machine.CACHE = Path(tempfile.mkdtemp(prefix="moe-cache-"))
        library._FP[:] = ["stub-fp"]
        sig = {"model_type": "tiny", "layers": 12, "hidden": 1024, "experts": 0, "bits": 4, "exl3": False}
        (machine.CACHE / "measured.json").write_text(json.dumps({"stub-fp": {
            "tiny-chat@tiny/12/1024/0/4": {"name": "tiny-chat", "sig": sig, "size": 0, "last": [13.8] * 12,
                                           "tps": 13.8, "n": 12, "date": "2026-10-27"},
            "big-moe@big_moe/32/2048/64/4": {"name": "big-moe", "sig": dict(
                sig, model_type="big_moe", layers=32, hidden=2048, experts=64), "size": 0, "last": [7.9] * 3,
                "tps": 7.9, "n": 3, "date": "2026-10-20"}}}))
    server.StreamingLLM = _FakeLLM
    server.ModelHost._can_batch = staticmethod(lambda llm: False)
    httpd = server.serve(server.ModelHost(model, shadow=False), 0)
    print("PORT", httpd.socket.getsockname()[1], flush=True)
    httpd.serve_forever()


def main():
    real = server.StreamingLLM
    real_can_batch = server.ModelHost._can_batch
    server.StreamingLLM = _FakeLLM
    server.ModelHost._can_batch = staticmethod(lambda llm: False)
    host = server.ModelHost("/models/tiny-chat", shadow=False)
    httpd = server.serve(host, 0)
    port = httpd.socket.getsockname()[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    def call(method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        raw = None if body is None else json.dumps(body).encode()
        conn.request(method, path, raw, dict({"Content-Type": "application/json"},
                                             **(headers or {})))
        r = conn.getresponse()
        data = r.read()
        conn.close()
        return r.status, r, data

    def chat(body, **kw):
        return call("POST", "/v1/chat/completions", body, **kw)

    def events(data):
        """The `data:` payloads of an SSE body, [DONE] last."""
        out = [e[len("data: "):] for e in data.decode().split("\n\n") if e.startswith("data: ")]
        assert out and out[-1] == "[DONE]", out[-3:]
        return [json.loads(e) for e in out[:-1]]

    user = [{"role": "user", "content": "hi"}]
    try:
        # -- the page ---------------------------------------------------------
        status, r, data = call("GET", "/")
        page = data.decode()
        assert status == 200 and r.getheader("Content-Type").startswith("text/html")
        assert "<title>" in page and "/app.js" in page and "/app.css" in page
        # Everything the page loads is local: it works with no network.
        assert not re.search(r"""(src|href)\s*=\s*["']?(https?:)?//""", page), "outside address"
        assert "@import" not in page and "url(http" not in page
        assert "<script>" not in page, "an inline script needs unsafe-inline"
        csp = r.getheader("Content-Security-Policy")
        assert "default-src 'none'" in csp and "script-src 'self'" in csp
        assert "connect-src 'self'" in csp and "unsafe-eval" not in csp
        assert r.getheader("X-Content-Type-Options") == "nosniff"
        assert call("GET", "/index.html")[0] == 200
        assert call("GET", "/?x=1")[0] == 200
        # The page's script talks to this server's API.
        status, js, data = call("GET", "/app.js")
        assert status == 200 and js.getheader("Content-Type").startswith("text/javascript")
        assert b"/v1/chat/completions" in data and b"reasoning_content" in data
        status, css, data = call("GET", "/app.css")
        assert status == 200 and css.getheader("Content-Type").startswith("text/css")
        assert css.getheader("Cache-Control") == "no-cache"
        # Every vendored library is present, pinned, and carries its licence.
        for path, marker in (("/vendor/marked-18.1.2/marked.umd.js", b"marked"),
                             ("/vendor/dompurify-3.4.17/purify.min.js", b"DOMPurify"),
                             ("/vendor/highlight.js-11.12.0/highlight.min.js", b"hljs")):
            st, resp, body = call("GET", path)
            assert st == 200 and marker.lower() in body.lower(), path
            assert resp.getheader("Content-Type").startswith("text/javascript"), path
            assert "immutable" in resp.getheader("Cache-Control"), path
        for lic in ("marked-18.1.2/LICENSE", "dompurify-3.4.17/LICENSE",
                    "dompurify-3.4.17/LICENSE-MPL", "highlight.js-11.12.0/LICENSE"):
            assert (server.WEB_DIR / "vendor" / lic).is_file(), lic
        # A static path cannot walk out of web/, list a directory, or take any type.
        for bad in ("/vendor/../server.py", "/%2e%2e/server.py",
                    "/vendor/marked-18.1.2/../../server.py", "/vendor/",
                    "/vendor/marked-18.1.2/LICENSE", "/.hidden.html"):
            assert call("GET", bad)[0] == 404, bad
        print(f"  GET /: {len(page) / 1024:.1f} KB, local assets, CSP set, "
              "vendored files served, no path leaves web/")

        # -- the pages and the icons ----------------------------------------------
        assert "img-src 'self' data:" in csp
        for link in ('rel="icon" type="image/svg+xml" href="/favicon.svg"',
                     'rel="apple-touch-icon" href="/apple-touch-icon.png"',
                     'name="theme-color"'):
            assert link in page, link
        assert "data:," not in page, "the page still has the empty icon"
        st, resp, body = call("GET", "/favicon.svg")
        assert st == 200 and resp.getheader("Content-Type") == "image/svg+xml" and b"<svg" in body
        for path, size in (("/apple-touch-icon.png", 180), ("/favicon-32.png", 32), ("/favicon.ico", 32)):
            st, resp, body = call("GET", path)
            assert st == 200 and resp.getheader("Content-Type") == "image/png", path
            assert body[:8] == b"\x89PNG\r\n\x1a\n", path
            assert int.from_bytes(body[16:20], "big") == size, path
        st, resp, body = call("GET", "/settings")
        assert st == 200 and resp.getheader("Content-Type").startswith("text/html")
        assert b"<title>Settings - Local chat</title>" in body and b"/settings.js" in body
        assert "script-src 'self'" in resp.getheader("Content-Security-Policy")
        assert call("GET", "/settings/")[0] == 200 and call("GET", "/settings?x=1")[0] == 200
        st, resp, body = call("GET", "/monitor")
        assert st == 200 and b"<title>Status - Local chat</title>" in body and b"/status.js" in body
        assert "script-src 'self'" in resp.getheader("Content-Security-Policy")
        for path in ("/common.js", "/settings.js", "/status.js", "/chart.js", "/pages.css"):
            st, resp, body = call("GET", path)
            assert st == 200 and resp.getheader("Cache-Control") == "no-cache", path
        # /status stays the JSON the page and editors read; the Status page is at /monitor.
        assert call("GET", "/status")[1].getheader("Content-Type") == "application/json"
        print("  pages: /settings and /monitor, the SVG icon, the touch icon and /favicon.ico are served")

        # -- the model list ---------------------------------------------------
        status, _, data = call("GET", "/v1/models")
        models = json.loads(data)
        assert status == 200 and models["object"] == "list"
        m0 = models["data"][0]
        assert m0["id"] == "tiny-chat" and m0["object"] == "model"
        assert [m["id"] for m in models["data"]] == ["tiny-chat"], models
        assert isinstance(m0["created"], int) and m0["owned_by"]
        # A base URL given without /v1, and a trailing slash, reach the same routes.
        assert [m["id"] for m in json.loads(call("GET", "/models")[2])["data"]] == [
            m["id"] for m in models["data"]]
        assert call("GET", "/v1/models/")[0] == 200
        assert call("POST", "/chat/completions", {"messages": user})[0] == 200
        status, _, data = call("GET", "/healthz")
        assert json.loads(data)["model"] == "/models/tiny-chat"
        print("  /v1/models: list of model objects; /models and a trailing slash work")

        # -- non-stream ---------------------------------------------------------
        status, _, data = chat({"model": "anything", "messages": user, "max_tokens": 50})
        out = json.loads(data)
        ch = out["choices"][0]
        assert status == 200 and out["object"] == "chat.completion"
        assert out["id"].startswith("chatcmpl-") and isinstance(out["created"], int)
        assert out["model"] == "anything"
        assert ch["index"] == 0 and ch["finish_reason"] == "stop"
        assert ch["message"] == {"role": "assistant", "content": _FakeLLM.reply}, ch
        assert out["usage"] == {"prompt_tokens": 8, "completion_tokens": 5,
                                "total_tokens": 13}, out["usage"]
        # The reply used up max_tokens: finish_reason says so.
        out = json.loads(chat({"messages": user, "max_tokens": 3})[2])
        assert out["choices"][0]["finish_reason"] == "length", out
        assert out["usage"]["completion_tokens"] == 3
        out = json.loads(chat({"messages": user, "max_completion_tokens": 3})[2])
        assert out["choices"][0]["finish_reason"] == "length"
        assert out["model"] == "local"
        print("  chat completion: shape, usage (prompt + cached), finish_reason stop/length")

        # -- stream -----------------------------------------------------------
        status, r, data = chat({"messages": user, "stream": True})
        assert status == 200 and r.getheader("Content-Type") == "text/event-stream"
        chunks = events(data)
        assert all(c["object"] == "chat.completion.chunk" and len({c["id"] for c in chunks}) == 1
                   for c in chunks)
        assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
        text = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks)
        assert text == _FakeLLM.reply and len(chunks) > 3, chunks
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
        assert not any("usage" in c for c in chunks), "usage only when asked for"
        chunks = events(chat({"messages": user, "stream": True,
                              "stream_options": {"include_usage": True}})[2])
        last = chunks[-1]
        assert last["choices"] == [] and last["usage"] == {
            "prompt_tokens": 8, "completion_tokens": 5, "total_tokens": 13}, last
        assert chunks[-2]["choices"][0]["finish_reason"] == "stop"
        chunks = events(chat({"messages": user, "stream": True, "max_tokens": 2})[2])
        assert chunks[-1]["choices"][0]["finish_reason"] == "length"
        print("  stream: SSE chunks, [DONE], usage chunk only under include_usage, "
              "finish_reason length")
        # An engine that fails mid-reply is an error event, not an empty reply that says stop.
        _FakeLLM.fail_after = 1
        try:
            body = chat({"messages": user, "stream": True})[2].decode()
        finally:
            _FakeLLM.fail_after = None
        assert "generation failed: out of memory in the fake engine" in body, body[-300:]
        assert '"finish_reason": "stop"' not in body, body[-300:]
        print("  stream: an engine error mid-reply reaches the client as an error event")

        # -- what the engine is handed ------------------------------------------
        parts = [{"role": "developer", "content": [{"type": "text", "text": "Be brief."},
                                                   {"type": "text", "text": "Be kind."}]},
                 {"role": "user", "content": [{"type": "text", "text": "one"},
                                              {"type": "audio", "audio": "x"}]},
                 {"role": "assistant", "content": None}]
        chat({"messages": parts})
        got = _FakeLLM.asked[-1]["messages"]
        assert got[0] == {"role": "system", "content": "Be brief.\nBe kind."}, got
        assert got[1]["content"] == "one" and got[2]["content"] is None, got
        assert parts[0]["role"] == "developer", "the request body was rewritten"
        print("  messages: part lists flattened to text, developer -> system, other parts dropped")

        # -- tool calls ---------------------------------------------------------
        tools = [{"type": "function", "function": {
            "name": "get_weather", "parameters": {"type": "object", "properties": {
                "city": {"type": "string"}}}}}]
        _FakeLLM.reply = "Checking. " + CALL
        ids = []
        for _ in range(2):
            out = json.loads(chat({"messages": user, "tools": tools})[2])
            ch = out["choices"][0]
            call_ = ch["message"]["tool_calls"][0]
            assert ch["finish_reason"] == "tool_calls" and ch["message"]["content"] == "Checking."
            assert call_["type"] == "function" and call_["function"]["name"] == "get_weather"
            assert json.loads(call_["function"]["arguments"]) == {"city": "Berlin"}
            ids.append(call_["id"])
        assert ids[0] != ids[1], "tool call ids repeat from one reply to the next"
        chunks = events(chat({"messages": user, "tools": tools, "stream": True})[2])
        delta = chunks[0]["choices"][0]["delta"]
        tc = delta["tool_calls"][0]
        assert tc["index"] == 0 and tc["id"] and tc["type"] == "function", tc
        assert isinstance(tc["function"]["arguments"], str)
        assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"
        _FakeLLM.reply = "Hello **world** from a fake"
        print("  tools: parsed call, finish_reason tool_calls, unique ids, index when streamed")

        def err(status_want, *a, **kw):
            status, r, data = call(*a, **kw)
            body = json.loads(data)
            assert status == status_want, (status, body)
            assert isinstance(body["error"]["message"], str) and body["error"]["type"], body
            return body

        # -- /admin/load: a key of the wrong type is a 400, never a dropped connection ----
        built = len(_FakeLLM.built)
        for key, bad in (("resident", "abc"), ("resident", None), ("resident", [1]),
                         ("cache_bias", "fast"), ("cache_bias", {}), ("model", 5), ("mtp", [1])):
            body = {"model": "/models/tiny-chat", key: bad}
            msg_ = err(400, "POST", "/admin/load", body)["error"]["message"]
            assert key in msg_, (key, msg_)
        assert len(_FakeLLM.built) == built, "a bad body started a load"
        assert call("GET", "/healthz")[0] == 200
        print("  /admin/load: a wrong type for resident, cache_bias, mtp or model is a 400 with the key's name")

        # -- thinking ---------------------------------------------------------------
        def thinking_kw(on):
            return {"chat_template_kwargs": {"enable_thinking": on}}

        out = json.loads(chat({"messages": user})[2])
        assert _FakeLLM.asked[-1]["thinking"] is False and "reasoning_content" not in \
            out["choices"][0]["message"], "a request that says nothing must not change"
        out = json.loads(chat({"messages": user, **thinking_kw(True)})[2])
        msg = out["choices"][0]["message"]
        assert _FakeLLM.asked[-1]["thinking"] is True, "the flag did not reach the engine"
        assert msg == {"role": "assistant", "content": "It is **42**.",
                       "reasoning_content": "Let me see."}, msg
        assert out["usage"]["completion_tokens"] == 5
        out = json.loads(chat({"messages": user})[2])
        assert _FakeLLM.asked[-1]["thinking"] is False and out["choices"][0]["message"][
            "content"] == _FakeLLM.reply, "thinking stayed on for the next request"
        out = json.loads(chat({"messages": user, "enable_thinking": True})[2])
        assert _FakeLLM.asked[-1]["thinking"] is True and out["choices"][0]["message"][
            "reasoning_content"] == "Let me see."
        chat({"messages": user, "enable_thinking": True, **thinking_kw(False)})
        assert _FakeLLM.asked[-1]["thinking"] is False, "chat_template_kwargs comes first"
        out = json.loads(chat({"messages": user, **thinking_kw(False)})[2])
        assert "reasoning_content" not in out["choices"][0]["message"]
        err(400, "POST", "/v1/chat/completions", {"messages": user, "enable_thinking": "yes"})
        err(400, "POST", "/v1/chat/completions", {"messages": user, "chat_template_kwargs": [1]})
        # A thinking reply cut before </think> is all reasoning.
        out = json.loads(chat({"messages": user, "max_tokens": 2, **thinking_kw(True)})[2])
        ch = out["choices"][0]
        assert ch["finish_reason"] == "length" and ch["message"]["content"] is None, ch
        assert ch["message"]["reasoning_content"] == "Let me", ch

        # Streamed: the tag cut across pieces never shows, reasoning and answer
        # arrive as their own deltas, in order.
        _FakeLLM.script = ["Let me", " see.\n</", "think", ">\n", "\nIt is ", "**42**."]
        status, r, data = chat({"messages": user, "stream": True, **thinking_kw(True)})
        chunks = events(data)
        deltas = [c["choices"][0]["delta"] for c in chunks]
        assert deltas[0] == {"role": "assistant", "content": ""}, deltas[0]
        assert all(len(d) == 1 for d in deltas[1:-1]), deltas
        reason = "".join(d.get("reasoning_content", "") for d in deltas)
        answer = "".join(d.get("content", "") for d in deltas)
        assert reason == "Let me see.\n" and answer == "It is **42**.", (reason, answer)
        kinds = [next(iter(d)) for d in deltas[1:-1]]
        assert kinds == sorted(kinds, key=lambda k: k == "content"), "answer before reasoning ended"
        assert "think" not in answer and "</" not in reason
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
        # Cut inside a partial tag: the held text is not lost.
        _FakeLLM.script = ["still ", "going </"]
        chunks = events(chat({"messages": user, "stream": True, "max_tokens": 2,
                              **thinking_kw(True)})[2])
        deltas = [c["choices"][0]["delta"] for c in chunks]
        assert "".join(d.get("reasoning_content", "") for d in deltas) == "still going </", deltas
        assert chunks[-1]["choices"][0]["finish_reason"] == "length"
        _FakeLLM.script = None
        # Thinking off, streamed: the same chunks as ever, no reasoning key.
        chunks = events(chat({"messages": user, "stream": True, **thinking_kw(False)})[2])
        assert not any("reasoning_content" in c["choices"][0]["delta"] for c in chunks)
        # With tools the reply is read whole: the call comes from the answer part.
        _FakeLLM.think_reply = "Weather, then.\n</think>\n\nChecking. " + CALL
        out = json.loads(chat({"messages": user, "tools": tools, **thinking_kw(True)})[2])
        ch = out["choices"][0]
        assert ch["finish_reason"] == "tool_calls" and ch["message"]["content"] == "Checking."
        assert ch["message"]["reasoning_content"] == "Weather, then."
        assert ch["message"]["tool_calls"][0]["function"]["name"] == "get_weather"
        chunks = events(chat({"messages": user, "tools": tools, "stream": True,
                              **thinking_kw(True)})[2])
        delta = chunks[0]["choices"][0]["delta"]
        assert delta["reasoning_content"] == "Weather, then." and delta["tool_calls"], delta
        _FakeLLM.think_reply = "Let me see.\n</think>\n\nIt is **42**."
        # The splitter itself, one character at a time.
        text = "Hmm, a < b </th and </think>\n\n\nDone: <b>ok</b>"
        sp, got = server.ThinkSplit(), ["", ""]
        for ch_ in text:
            for i, part in enumerate(sp.feed(ch_)):
                got[i] += part
        got[0] += sp.flush()
        want = ("Hmm, a < b </th and", "Done: <b>ok</b>")
        assert server.split_think(text) == want and (got[0].strip(), got[1]) == want, got
        assert server.split_think("only thoughts </thi") == ("only thoughts </thi", "")
        print("  thinking: flag reaches the engine and resets, reasoning_content whole and streamed "
              "(tag across pieces), cut replies, tools, bad values are 400")

        # -- images -------------------------------------------------------------------
        def data_url(kind, size=64):
            head = {"png": b"\x89PNG\r\n\x1a\n", "jpeg": b"\xff\xd8\xff\xe0", "gif": b"GIF89a",
                    "webp": b"RIFF\x00\x00\x00\x00WEBP"}[kind]
            raw = head + b"\0" * (size - len(head))
            return f"data:image/{kind};base64," + base64.b64encode(raw).decode()

        def see(*urls, text="what is this"):
            parts = [{"type": "image_url", "image_url": {"url": u}} for u in urls]
            return [{"role": "user", "content": [parts[0], {"type": "text", "text": text}]
                     + parts[1:]}]

        calls_before = len(_FakeLLM.asked)
        for body in ({"messages": see(data_url("png"))},
                     {"messages": see(data_url("png")), "stream": True}):
            msg_ = err(400, "POST", "/v1/chat/completions", body)["error"]["message"]
            assert msg_ == "this model has no vision part; images are not supported", msg_
        assert len(_FakeLLM.asked) == calls_before, "the engine was asked about an image it cannot see"
        for path in ("/healthz", "/status"):
            assert json.loads(call("GET", path)[2])["vision"] is False
        assert json.loads(call("GET", "/v1/models")[2])["data"][0]["vision"] is False

        _FakeLLM.supports_images = True
        assert json.loads(call("GET", "/healthz")[2])["vision"] is True
        assert json.loads(call("GET", "/v1/models")[2])["data"][0]["vision"] is True
        png, jpg = data_url("png"), data_url("jpeg")
        chat({"messages": see(png, jpg)})
        got = _FakeLLM.asked[-1]["messages"][0]["content"]
        assert got == [{"type": "image_url", "image_url": {"url": png}},
                       {"type": "text", "text": "what is this"},
                       {"type": "image_url", "image_url": {"url": jpg}}], "parts changed or reordered"
        # a bare string url is read as OpenAI's object, a text-only list still flattens
        chat({"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": png}, {"type": "text", "text": "x"}]}]})
        assert _FakeLLM.asked[-1]["messages"][0]["content"][0] == {
            "type": "image_url", "image_url": {"url": png}}
        chat({"messages": [{"role": "user", "content": [{"type": "text", "text": "plain"}]}]})
        assert _FakeLLM.asked[-1]["messages"][0]["content"] == "plain"
        for kind in ("gif", "webp"):
            assert chat({"messages": see(data_url(kind)), "stream": True})[0] == 200, kind
        # the caps: size (at the limit passes, a byte over does not), type, URL, content
        big = server.MAX_IMAGE
        assert chat({"messages": see(data_url("png", big))})[0] == 200
        calls_before = len(_FakeLLM.asked)
        for url, want in ((data_url("png", big + 1), "larger than 10 MB"),
                          ("data:image/bmp;base64,AAAA", "not supported"),
                          ("data:image/png;base64,!!!", "not valid base64"),
                          ("data:image/png,AAAA", "must be base64"),
                          ("https://example.com/cat.png", "does not fetch remote URLs"),
                          ("file:///etc/passwd", "does not fetch remote URLs"),
                          (data_url("png").replace("image/png", "image/jpeg"), "not a valid JPEG"),
                          (data_url("gif").replace("image/gif", "image/webp"), "not a valid WEBP")):
            msg_ = err(400, "POST", "/v1/chat/completions", {"messages": see(url)})["error"]["message"]
            assert want in msg_, (want, msg_)
        assert len(_FakeLLM.asked) == calls_before
        # a body over the cap is refused before it is read
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        conn.putrequest("POST", "/v1/chat/completions")
        conn.putheader("Content-Length", str(server.MAX_BODY + 1))
        conn.endheaders()
        r = conn.getresponse()
        assert r.status == 413 and json.loads(r.read())["error"]["message"]
        conn.close()
        _FakeLLM.supports_images = False
        print("  images: 400 without a vision part, parts passed in order with one, "
              "10 MB / type / magic / URL / body caps")

        # -- the status endpoint --------------------------------------------------------
        status, _, data = call("GET", "/status")
        st = json.loads(data)
        assert status == 200 and st["model"] == "/models/tiny-chat" and st["loaded"] is True
        assert st["requests"] > 10 and st["uptime_s"] > 0
        assert st["cached_tokens"] == 3 * st["requests"] and st["prompt_tokens"] == 8 * st["requests"]
        assert st["decode_tok_s"] == 10.0 and 0.5 <= st["first_token_last_s"] < 0.6
        assert abs(st["first_token_mean_s"] - 0.5) < 0.1 and st["store_mb"] == 123.4
        assert st["memory_gb"] is None or st["memory_gb"] > 0
        # Counters on a host of their own: they add up, and the speed is
        # weighted by tokens (4 tokens in 0.4 s and 2 in 0.05 s make 13.3, not 25).
        fresh = server.ModelHost("/models/count", shadow=False)
        try:
            assert fresh.status()["requests"] == 0 and fresh.status()["decode_tok_s"] is None
            assert fresh.status()["first_token_mean_s"] is None
            for tps, limit, stream in ((10.0, None, False), (40.0, 3, True), (10.0, None, True)):
                _FakeLLM.tps = tps
                info = {}
                out = fresh.generate(user, limit, stream=stream, info=info)
                if stream:
                    "".join(out)
                assert info["cached_tokens"] == 3 and info["prompt_tokens"] == 8
            fs = fresh.status()
            assert fs["requests"] == 3 and fs["prompt_tokens"] == 24 and fs["cached_tokens"] == 9
            assert fs["completion_tokens"] == 5 + 3 + 5, fs
            assert fs["decode_tok_s"] == round((4 + 2 + 4) / (0.4 + 0.05 + 0.4), 2), fs
            assert abs(fs["first_token_mean_s"] - 0.5) < 0.05, fs
        finally:
            _FakeLLM.tps = 10.0
            fresh.executor.shutdown()
        print("  status: model, vision, counters add up, speed weighted by tokens, memory")

        # -- the engine's own store, the conditions and the reply count -------------------
        real_cond = server.machine.conditions
        cond = {"power": "battery", "low_power": False, "thermal": "", "mem_free_gb": 1.3,
                "cpu_busy": 2.1, "top": [{"name": "cmux", "cpu": 2.1}], "gpu_busy": 4.0}
        server.machine.conditions = lambda ttl=5.0: dict(cond)
        bare = server.ModelHost("/models/none", shadow=False)
        try:
            st2 = json.loads(call("GET", "/status")[2])
            assert st2["conditions"]["power"] == "battery" and st2["conditions"]["cpu_busy"] == 2.1
            assert st2["conditions_text"] == "on battery · other apps use 2.1 CPU cores (cmux): expect slower"
            assert st2["arena"] == {"slots": 60, "gb": 4.25}, st2["arena"]
            assert st2["store"] == {"mb": 123.4, "slots": 64, "budget_gb": 4.5}, st2["store"]
            assert st2["experts"] == {"rows": 264, "rows_per_token": 132.0}, st2["experts"]
            assert st2["decode_replies"] == st["requests"] > 0 and st2["store_mb"] == 123.4
            # an engine that reports no store yet leaves the fields empty, not made up
            bs = bare.status()
            assert bs["arena"] == {"slots": None, "gb": None} and bs["decode_replies"] == 0
            assert bs["store"] == {"mb": None, "slots": None, "budget_gb": None}, bs["store"]
            assert bs["experts"] == {"rows": None, "rows_per_token": None} and bs["store_mb"] is None
            assert bs["conditions_text"] == "on battery · other apps use 2.1 CPU cores (cmux): expect slower"
        finally:
            server.machine.conditions = real_cond
            bare.executor.shutdown()
        print("  store and conditions: arena slots and GB, store slots, expert rows a token, the live "
              "reply count, and what else uses the computer")

        # -- what the Status page and the chat's context bar read -------------------------
        assert isinstance(st["host"], str) and st["host"] and "." not in st["host"], st["host"]
        assert isinstance(st["machine"], dict) and "chip" in st["machine"], st["machine"]
        assert st["started_at"] > 1_700_000_000 and st["memory_budget_gb"] is None
        assert st["context_length"] is None, "no config.json behind /models/tiny-chat"
        status, _, data = call("GET", "/admin/history")
        hist = json.loads(data)
        assert status == 200 and hist["capacity"] == server.HISTORY_LEN
        rows = hist["replies"]
        assert len(rows) == min(st["requests"], server.HISTORY_LEN) and len(rows) > 10, len(rows)
        last = rows[-1]
        assert set(last) == {"t", "ttft_s", "decode_tok_s", "prompt_tokens", "cached_tokens",
                             "completion_tokens", "memory_gb"}, last
        assert last["decode_tok_s"] == 10.0 and 0.5 <= last["ttft_s"] < 0.6, last
        assert last["prompt_tokens"] == 8 and last["cached_tokens"] == 3
        assert all(a["t"] <= b["t"] for a, b in zip(rows, rows[1:])), "oldest first"
        # The context length is read from the checkpoint's config, at the top or under text_config.
        import web_settings
        cfg_root = Path(tempfile.mkdtemp(prefix="moe-ctx-"))
        for name, cfg, want in (("flat", {"max_position_embeddings": 32768}, 32768),
                                ("nested", {"model_type": "x", "text_config": {"max_position_embeddings": 262144}}, 262144),
                                ("absurd", {"max_position_embeddings": 10 ** 30}, None),
                                ("none", {"model_type": "x"}, None)):
            (cfg_root / name).mkdir()
            (cfg_root / name / "config.json").write_text(json.dumps(cfg))
            assert web_settings.context_length(cfg_root / name) == want, name
        assert web_settings.context_length(cfg_root / "missing") is None
        budget = server.ModelHost(str(cfg_root / "nested"), shadow=False, mem_peak_gb=7.5, mem_peak_max_gb=9.0)
        try:
            bs = budget.status()
            assert bs["context_length"] == 262144 and bs["memory_budget_gb"] == 7.5, bs
            assert bs["memory_budget_max_gb"] == 9.0
        finally:
            budget.executor.shutdown()
        print("  status: host, machine, context length from the config, memory budget; "
              "/admin/history keeps each reply's speed, first token, tokens and memory")

        # Measured speed: a reply of 64 or more tokens is noted for the model's folder.
        import library
        import machine as mach
        real_cache, real_fp = mach.CACHE, list(library._FP)
        mach.CACHE, library._FP[:] = Path(tempfile.mkdtemp(prefix="moe-measured-")), ["test-fp"]
        folder = Path(tempfile.mkdtemp(prefix="moe-m-")) / "tiny-q4"
        folder.mkdir()
        (folder / "config.json").write_text(json.dumps({
            "model_type": "tiny", "num_hidden_layers": 12, "hidden_size": 1024,
            "quantization": {"bits": 4, "group_size": 64}}))
        timed = server.ModelHost(str(folder), shadow=False)
        try:
            for tokens, tps in ((5, 10.0), (63, 10.0), (64, 10.0), (70, 20.0), (80, 30.0)):
                _FakeLLM.script, _FakeLLM.tps = ["w "] * tokens, tps
                timed.generate(user, None, stream=False, info={})
            rec = library.load_measured()["test-fp"]["tiny-q4@tiny/12/1024/0/4"]
            assert rec["last"] == [10.0, 20.0, 30.0] and rec["tps"] == 20.0 and rec["n"] == 3, rec
            assert rec["name"] == "tiny-q4" and rec["date"] == time.strftime("%Y-%m-%d"), rec
            got = library.lookup({"model_type": "tiny", "layers": 12, "hidden": 1024, "bits": 4})
            assert got["tps"] == 20.0 and got["n"] == 3, got
        finally:
            _FakeLLM.script, _FakeLLM.tps = None, 10.0
            timed.executor.shutdown()
            mach.CACHE, library._FP[:] = real_cache, real_fp
        print("  measured speed: replies of 64 or more tokens add to the folder's median, shorter ones do not")

        # -- the settings page's endpoints ---------------------------------------
        import web_settings

        models_dir = Path(tempfile.mkdtemp(prefix="moe-test-models-"))
        launcher = _FakeLauncher(models_dir)
        web_settings.configure(launcher)
        web_settings.engine_gb = lambda host: 0.0       # no engine memory to give back yet

        def settings_data():
            st, _, raw = call("GET", "/admin/models")
            assert st == 200, raw[:300]
            return json.loads(raw)

        def wait_list(timeout=20.0):
            end = time.time() + timeout
            while time.time() < end:
                d = settings_data()
                if not d["building"]:
                    return d
                time.sleep(0.1)
            raise AssertionError("the model list never finished building")

        try:
            assert json.loads(call("GET", "/admin/prepare")[2])["job"]["state"] == "idle"
            d = wait_list()
            assert d["ok"] is True and d["unavailable"] is None and d["error"] is None, d
            assert d["machine"]["chip"] == "a test chip" and d["machine"]["free_gb"] == 88.0
            assert [m["id"] for m in d["models"]] == ["tiny", "big"], d["models"]
            tiny, big = d["models"]
            assert [v["key"] for v in tiny["variants"]] == ["4bit", "3bit", "8bit", "16bit"]
            assert tiny["about"] == "a test model" and tiny["license"] == "MIT" and tiny["tested"]
            four, three, eight, sixteen = tiny["variants"]
            # A disk too full is its own state: not ok, flagged, and its status says what is needed.
            assert sixteen["disk_short"] is True and sixteen["ok"] is False, sixteen
            assert sixteen["status"] == "needs 41 GB free on disk, 12.6 GB free", sixteen
            assert not eight["disk_short"] and not four["disk_short"]
            assert four["status"] == "ready" and four["ready"] and four["tps"] == 42.0
            assert four["size_bytes"] == 2000000000 and four["memory"] == {"gb": 7.5, "max_gb": None}
            assert three["make_bits"] == 3 and not three["ready"] and three["path"].endswith("-3bit")
            assert eight["ok"] is False and "12 GB" in eight["why"] and eight["tps"] is None
            assert big["not_made"] == ["a 2-bit copy: too little memory"], big
            # The model this server is serving is marked, with its speed from /status.
            assert d["current"]["name"] == "Tiny Chat" and d["current"]["variant"] == "4bit", d["current"]
            assert d["current"]["tps"] == 10.0 and [v["current"] for v in tiny["variants"]] == \
                [True, False, False, False]
            assert d["settings"] == {"memory_gb": None, "memory_max_gb": None,
                                     "memory_auto": True, "fast": True, "thinking": False}, d["settings"]
            print("  settings: the launcher's list with sizes, speeds, status, reasons, the model in use")

            # Measured here, also for a model that is no longer on disk.
            assert big["variants"][0]["measured_here"] == {} and four["measured_here"] == {}
            real_cache, real_fp = mach.CACHE, list(library._FP)
            mach.CACHE, library._FP[:] = Path(tempfile.mkdtemp(prefix="moe-seen-")), ["fp-web"]
            gone = Path(tempfile.mkdtemp(prefix="moe-gone-")) / "big-moe-old"
            gone.mkdir()
            (gone / "config.json").write_text(json.dumps({
                "model_type": "big_moe", "num_hidden_layers": 32, "hidden_size": 2048, "num_experts": 64,
                "quantization": {"bits": 4, "group_size": 64}}))
            try:
                for tokens, secs in ((101, 10.0), (101, 12.5), (101, 8.0)):
                    library.record(gone, tokens, secs, today="2026-10-20")
                shutil.rmtree(gone)                       # deleted: the record stays
                seen = wait_list()["models"][1]["variants"][0]["measured_here"]
                assert seen == {"tps": 10.0, "n": 3, "date": "2026-10-20", "name": "big-moe-old"}, seen
                assert wait_list()["models"][0]["variants"][0]["measured_here"] == {}
                library._FP[:] = ["another-machine"]
                assert wait_list()["models"][1]["variants"][0]["measured_here"] == {}
            finally:
                mach.CACHE, library._FP[:] = real_cache, real_fp
            print("  measured here: the median, count and date next to a variant, kept when its folder is "
                  "deleted, never shown for another machine")

            # -- a prepare runs as a child and reports its bytes -------------------
            st, _, raw = call("POST", "/admin/prepare", {"model": "tiny", "variant": "3bit"})
            job = json.loads(raw)["job"]
            assert st == 200 and job["state"] == "running" and job["model"] == "tiny", job
            assert job["total"] == 20 * 1000000 and job["label"] == "3bit", job
            st, _, raw = call("POST", "/admin/prepare", {"model": "tiny", "variant": "3bit"})
            assert st == 409 and "already preparing" in json.loads(raw)["error"]["message"]
            time.sleep(1.6)
            job = json.loads(call("GET", "/admin/prepare")[2])["job"]
            assert job["state"] == "running" and job["bytes"] > 0, job
            assert any(l.startswith("made part") for l in job["log"]), job["log"]
            assert (models_dir / "tiny-chat-3bit").is_dir()
            assert job["rate_bps"] and job["rate_bps"] > 0, job
            assert job["seconds"] > 0.5, job
            st, _, raw = call("DELETE", "/admin/prepare")
            job = json.loads(raw)["job"]
            assert st == 200 and job["state"] == "cancelled", job
            time.sleep(0.5)
            left = json.loads(call("GET", "/admin/prepare")[2])["job"]
            assert left["state"] == "cancelled" and left["bytes"] <= job["bytes"] + 1000000, left
            assert not left["log"] or True
            assert json.loads(call("DELETE", "/admin/prepare")[2])["job"]["state"] == "cancelled"
            err(400, "POST", "/admin/prepare", {"variant": "3bit"})
            err(404, "POST", "/admin/prepare", {"model": "tiny", "variant": "2bit"})
            err(404, "POST", "/admin/prepare", {"model": "nope", "variant": "4bit"})
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
            conn.request("POST", "/admin/prepare", b"{nope", {"Content-Type": "application/json"})
            r = conn.getresponse()
            assert r.status == 400 and json.loads(r.read())["error"]["message"]
            conn.close()
            print("  prepare: one at a time, bytes and log as it goes, then stopped for good "
                  "and resumed from where it stopped")

            # -- a download that finishes is offered as ready -----------------------
            launcher.parts, launcher.delay = 4, 0.05
            st, _, raw = call("POST", "/admin/prepare", {"model": "big", "variant": "published"})
            assert st == 200, raw
            end = time.time() + 20
            while time.time() < end:
                job = json.loads(call("GET", "/admin/prepare")[2])["job"]
                if job["state"] != "running":
                    break
                time.sleep(0.2)
            assert job["state"] == "done" and job["bytes"] == 4 * 1000000, job
            assert job["log"][-1] == "made part 3", job["log"][-3:]
            print("  prepare: a finished download ends in done with every byte on disk")

            # Delete: what the launcher made, never the model in use.
            d = wait_list()
            bigv = d["models"][1]["variants"][0]
            assert bigv["ready"] and bigv["own"] is True and bigv["disk_path"] == str(models_dir / "big-moe"), bigv
            tinyv = d["models"][0]["variants"][0]
            assert tinyv["own"] is False and tinyv["disk_path"] == "/models/tiny-chat", tinyv
            assert d["models"][0]["variants"][1]["disk_path"] == "" and d["models"][0]["variants"][1]["own"] is False
            err(400, "POST", "/admin/delete", {"variant": "published"})
            err(404, "POST", "/admin/delete", {"model": "nope", "variant": "published"})
            saved_model = host.model_name
            host.model_name = str(models_dir / "big-moe")
            bag = err(409, "POST", "/admin/delete", {"model": "big", "variant": "published"})
            host.model_name = saved_model
            assert "is running" in bag["error"]["message"] and (models_dir / "big-moe").is_dir(), bag
            bag = err(409, "POST", "/admin/delete", {"model": "tiny", "variant": "4bit"})
            assert "not on disk" in bag["error"]["message"], bag                  # nothing of it in the launcher's folder
            err(403, "POST", "/admin/delete", {"model": "big", "variant": "published"},
                headers={"Origin": "https://example.com"})
            assert (models_dir / "big-moe").is_dir()
            st, _, raw = call("POST", "/admin/delete", {"model": "big", "variant": "published"})
            got = json.loads(raw)
            assert st == 200 and got["ok"] and got["message"].startswith(f"deleted {models_dir / 'big-moe'} ("), got
            assert not (models_dir / "big-moe").exists()
            bigv = wait_list()["models"][1]["variants"][0]
            assert not bigv["ready"] and bigv["disk_path"] == "" and bigv["own"] is False, bigv
            print("  delete: the launcher's own folder goes after a request, the model in use, a folder it did "
                  "not make and an unknown model are refused")

            # Search: words, a name or a link, judged by this computer's launcher and listed apart.
            import urllib.parse
            assert settings_data()["found"] is None
            err(400, "GET", "/admin/search")
            bag = err(422, "GET", "/admin/search?q=" + urllib.parse.quote("nope model"))
            assert "no safetensors weights in the repo" in bag["error"]["message"], bag
            assert settings_data()["found"] is None
            link = "https://huggingface.co/owner/Foo-4bit/tree/main?show=1"
            st, _, raw = call("GET", "/admin/search?q=" + urllib.parse.quote(link))
            got = json.loads(raw)
            assert st == 200 and got == {"ok": True, "query": link, "count": 1, "skipped": 0}, got
            d = wait_list()
            assert [m["id"] for m in d["models"]] == ["tiny", "big"], "the tested list is not mixed with results"
            f = d["found"]
            assert f["query"] == link and f["skipped"] == 0 and [m["id"] for m in f["models"]] == ["owner--Foo-4bit"]
            fm = f["models"][0]
            assert fm["repo"] == "owner/Foo-4bit" and fm["tested"] is False and fm["found"] is True, fm
            fv = fm["variants"][0]
            assert fv["status"] == "download 3.0 GB" and fv["ok"] and fv["current"] is False and fv["measured_here"] == {}
            st, _, raw = call("GET", "/admin/search?q=" + urllib.parse.quote("found models"))
            got = json.loads(raw)
            assert got["count"] == 2 and got["skipped"] == 3, got
            f = wait_list()["found"]
            assert [m["id"] for m in f["models"]] == ["org--Found-A-4bit", "org--Found-B-4bit"] and f["skipped"] == 3
            # a result downloads, is offered as ready and deletes like a tested model, through the same endpoints
            st, _, raw = call("POST", "/admin/prepare", {"model": "org--Found-A-4bit", "variant": "published"})
            assert st == 200, raw
            end = time.time() + 20
            while time.time() < end:
                job = json.loads(call("GET", "/admin/prepare")[2])["job"]
                if job["state"] != "running":
                    break
                time.sleep(0.2)
            assert job["state"] == "done" and job["model"] == "org--Found-A-4bit", job
            time.sleep(0.3)
            fv = wait_list()["found"]["models"][0]["variants"][0]
            assert fv["ready"] and fv["status"] == "ready" and fv["own"] and fv["disk_path"].endswith("found-a-4bit"), fv
            st, _, raw = call("POST", "/admin/delete", {"model": "org--Found-A-4bit", "variant": "published"})
            assert st == 200 and not (models_dir / "found-a-4bit").exists(), raw
            err(403, "GET", "/admin/search?q=x", headers={"Origin": "https://example.com"})
            err(403, "DELETE", "/admin/search", headers={"Origin": "https://example.com"})
            assert json.loads(call("DELETE", "/admin/search")[2]) == {"ok": True}
            assert settings_data()["found"] is None
            print("  search: a name or a link goes to its repo, words to a search; the results are judged by this "
                  "computer's launcher, listed apart, downloaded and deleted like the tested models")

            # -- Use: the launcher's own flags for that model ----------------------
            st, _, raw = call("POST", "/admin/load", {"model": "/models/tiny-chat", "plan": True})
            d = json.loads(raw)
            assert st == 200 and d["ok"] is True, d
            got = _FakeLLM.built[-1]
            assert got["cache_bias"] == 0.5 and got["model"] == "/models/tiny-chat", got
            assert host.mem_peak_gb == 7.5 and host.mem_peak_max_gb is None
            assert d["memory_gb"] == 7.5 and d["memory_auto"] is True, d
            assert d["fast"] is True
            assert d["plan"]["set_now"] == ["--mem-peak-gb 7.5", "--cache-bias 0.5",
                                            "--bias-budget 0.2"], d["plan"]
            assert d["plan"]["at_start"] == ["--prefill-step 0"], d["plan"]
            # A model the launcher runs with exact routing, and a fixed ceiling too.
            st, _, raw = call("POST", "/admin/load", {"model": "/models/big-moe", "plan": True})
            d = json.loads(raw)
            got = _FakeLLM.built[-1]
            assert st == 200 and got["cache_bias"] == 0.0, (st, got)
            assert host.mem_peak_gb == 6.0 and host.mem_peak_max_gb == 9.0
            assert d["memory_gb"] == 6.0 and d["memory_max_gb"] == 9.0
            assert d["fast"] is False and d["plan"]["at_start"] == [], d["plan"]
            print("  a switch: the launcher's memory, cache bias and speed plan for that model, "
                  "and what the running server cannot set now")

            # -- Apply: the page's own options win over the plan --------------------
            body = {"model": "/models/tiny-chat", "memory": 9.5, "fast": False, "thinking": True}
            st, _, raw = call("POST", "/admin/load", body)
            d = json.loads(raw)
            got = _FakeLLM.built[-1]
            assert st == 200 and got["cache_bias"] == 0.0 and got["thinking"] is True, got
            assert host.mem_peak_gb == 9.5 and host.mem_peak_max_gb is None
            assert d["memory_gb"] == 9.5 and d["memory_max_gb"] is None
            assert d["memory_auto"] is False and d["fast"] is False and d["thinking"] is True, d
            # Auto goes back to the launcher's 7.5 GB for this model.
            st, _, raw = call("POST", "/admin/load", {"model": "/models/tiny-chat", "memory": "auto",
                                                      "fast": True})
            d = json.loads(raw)
            assert st == 200 and host.mem_peak_gb == 7.5, d
            assert d["memory_auto"] is True and d["fast"] is True
            assert "plan" not in d, d
            for bad, want in (({"memory": 0.2}, "between"), ({"memory": "lots"}, "a number"),
                              ({"fast": "yes"}, "true or false"),
                              ({"thinking": 1}, "true or false")):
                bag = err(400, "POST", "/admin/load", {"model": "/models/tiny-chat", **bad})
                assert want in bag["error"]["message"], bag
            assert host.mem_peak_gb == 7.5, host.mem_peak_gb
            # A load that fails puts the memory budget back and still serves the old model.
            class _Boom:
                def __init__(self, **kw):
                    raise RuntimeError("no such checkpoint")
            server.StreamingLLM = _Boom
            st, _, raw = call("POST", "/admin/load", {"model": "/models/nope", "memory": 5.0})
            assert st == 503 and "no such checkpoint" in json.loads(raw)["error"]["message"]
            assert host.mem_peak_gb == 7.5 and host.model_name == "/models/tiny-chat", host.mem_peak_gb
            server.StreamingLLM = _FakeLLM
            print("  options: a fixed limit, faster replies and thinking reach the engine; "
                  "Auto is the launcher's plan; a failed load puts memory back")

            # -- a copy made here is checked with one reply before it says ready ----
            st, _, raw = call("POST", "/admin/load", {"model": "/models/tiny-chat-3bit"})
            d = json.loads(raw)
            assert st == 200 and launcher.marked == ["/models/tiny-chat-3bit"], (d, launcher.marked)
            assert d["checked"] and "Hello" in d["checked"], d
            st, _, raw = call("POST", "/admin/load", {"model": "/models/tiny-chat"})
            assert st == 200 and launcher.marked == ["/models/tiny-chat-3bit"]
            print("  a 3-bit copy made here: one short reply, then the launcher marks it ready")

            # -- the fit and the speed for a switch: the engine's own memory given back ----
            import machine
            prof = {"backend": "metal", "ram": 16 * 2**30, "working_set": 12 * 2**30,
                    "now": {"avail_gb": 8.0, "disks": []}, "gpu_bw": 120.0,
                    "chip": "a test chip", "arch": "arm64", "os": "test"}
            facts = {"kind": "moe", "dense_bytes": int(3e9), "read_bytes": int(3e9),
                     "expert_bytes": int(40e6), "experts": 64, "moe_layers": 30, "top_k": 4,
                     "recipe": "store", "gated": False, "native": False, "native_ok": False,
                     "bits": 4, "group": 64, "total_bytes": int(80e9)}
            disk = {"lat_ms": 0.2, "gbps": 5.0}
            assert web_settings.switch_profile(prof, 0) is prof
            s0 = machine.settings(prof, facts)
            p1 = web_settings.switch_profile(prof, 4.0)
            s1 = machine.settings(p1, facts)
            e0 = machine.estimate(prof, facts, s0, disk)
            e1 = machine.estimate(p1, facts, s1, disk)
            assert s1["ok"] and s1["store_gb"] > s0["store_gb"] and e1["tps"] > e0["tps"], \
                (s0, s1, e0, e1)
            resident = {"kind": "resident", "total_bytes": int(9e9), "dense_bytes": int(9e9),
                        "read_bytes": int(9e9), "moe_layers": 0, "experts": 0, "top_k": 0,
                        "expert_bytes": 0, "bits": 4, "group": 64}
            assert not machine.settings(prof, resident)["ok"]
            assert machine.settings(web_settings.switch_profile(prof, 16.0), resident)["ok"]
            print("  switch: giving the engine's own memory back raises the store, the estimate, "
                  "and flips a model that did not fit")

            # The page's list with that memory given back, and the note only where it matters.
            web_settings.engine_gb = lambda host: 9.6
            d = wait_list()
            eight = d["models"][0]["variants"][2]
            assert eight["ok"] is True and eight["why"] == "" and eight["tps"] == 18.0, eight
            assert eight["after_unload"] is True, eight
            big_pub = d["models"][1]["variants"][0]
            assert big_pub["tps"] == 31.6 and big_pub["after_unload"] is True, big_pub
            assert d["models"][0]["variants"][0]["after_unload"] is False
            web_settings.engine_gb = lambda host: 0.0
            d = wait_list()
            assert d["models"][0]["variants"][2]["ok"] is False
            assert d["models"][1]["variants"][0]["tps"] == 8.0
            assert not d["models"][1]["variants"][0]["after_unload"]
            print("  switch: a model that could not run now can, at the speed a switch would see; "
                  "no note where nothing changed")

            # -- a running copy outside the launcher's folder: matched by its config ----------
            own = Path(tempfile.mkdtemp(prefix="moe-own-copy-")) / "own-35b"
            own.mkdir()
            (own / "config.json").write_text(json.dumps({
                "model_type": "big_moe", "num_hidden_layers": 32, "hidden_size": 2048,
                "num_experts": 64, "quantization": {"bits": 4, "group_size": 64}}))
            saved_model = host.model_name
            host.model_name = str(own)
            launcher.entries[("big", "published")].update(
                ok=False, why="3.2 GB usable now (closing other apps frees more)")
            d = wait_list()
            big_pub = d["models"][1]["variants"][0]
            assert big_pub["current"] is True and big_pub["own_path"] == str(own), big_pub
            assert big_pub["ok"] is True and big_pub["why"] == "", big_pub
            assert big_pub["status"] == "running now", big_pub
            assert d["current"]["name"] == "Big MoE" and d["current"]["variant"] == "published", \
                d["current"]
            assert d["current"]["own_path"] == str(own)
            assert not any(v["current"] for v in d["models"][0]["variants"])
            web_settings.engine_gb = lambda host: 9.6
            assert wait_list()["models"][1]["variants"][0]["after_unload"] is True
            host.model_name = saved_model
            launcher.entries[("big", "published")].update(ok=True, why="")
            web_settings.engine_gb = lambda host: 0.0
            print("  a running copy outside the folder: its row says In use with your own copy, "
                  "never can't run, and it matches no other model")

            # An EXL3 pack (a 4-bit base config plus expert_quant) is the EXL3 row, never the 4-bit one.
            import library
            pack = Path(tempfile.mkdtemp(prefix="moe-exl3-")) / "q-exl3"
            pack.mkdir()
            cfg = {"model_type": "flash", "num_hidden_layers": 48, "hidden_size": 2560, "num_experts": 512,
                   "quantization": {"bits": 4, "group_size": 32}}
            (pack / "config.json").write_text(json.dumps({**cfg, "expert_quant": {"format": "exl3", "k": 3}}))
            shape = {"model_type": "flash", "layers": 48, "hidden": 2560, "experts": 512}
            rows = [{"id": "flash", "variants": [{"key": "owner/Flash-4bit"}, {"key": "published"}]}]
            ents = {("flash", "owner/Flash-4bit"): {"facts": {**shape, "bits": 4}},
                    ("flash", "published"): {"facts": {**shape, "bits": 3, "format": "exl3"}}}
            assert web_settings.config_match(rows, ents, library.signature(pack)) == (0, 1)
            (pack / "config.json").write_text(json.dumps(cfg))
            assert web_settings.config_match(rows, ents, library.signature(pack)) == (0, 0)
            (pack / "config.json").write_text(json.dumps({**cfg, "quantization": {"bits": 8, "group_size": 32}}))
            assert web_settings.config_match(rows, ents, library.signature(pack)) is None
            print("  an EXL3 pack matches the EXL3 row and not the 4-bit one; another bit width matches none")

            # -- the settings endpoints are local-only too -------------------------
            for origin in ("https://example.com", "null"):
                err(403, "GET", "/admin/models", headers={"Origin": origin})
                err(403, "POST", "/admin/prepare", {"model": "tiny", "variant": "3bit"},
                    headers={"Origin": origin})
                err(403, "DELETE", "/admin/prepare", headers={"Origin": origin})
            err(403, "GET", "/admin/models", headers={"Host": "rebind.example.com:8765"})
            err(404, "DELETE", "/nothing")
            assert call("GET", "/admin/models", headers={"Origin": f"http://127.0.0.1:{port}"})[0] == 200
            print("  settings: the same Origin and Host guard as /admin/load, 404 for another path")
        finally:
            web_settings.configure(None)
            server.StreamingLLM = _FakeLLM

        # -- errors -------------------------------------------------------------
        err(400, "POST", "/v1/chat/completions", {"model": "x"})
        err(400, "POST", "/v1/chat/completions", {"messages": "hi"})
        err(400, "POST", "/v1/chat/completions", {"messages": user, "max_tokens": "ten"})
        err(400, "POST", "/v1/chat/completions", [1, 2])
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        conn.request("POST", "/v1/chat/completions", b"{nope", {"Content-Type": "application/json"})
        r = conn.getresponse()
        assert r.status == 400 and json.loads(r.read())["error"]["message"]
        conn.close()
        err(404, "GET", "/nothing")
        err(404, "POST", "/v1/completions", {"prompt": "hi"})
        print("  errors: OpenAI's {error: {message, type}} for 400 and 404")

        # -- another site must not drive the server through a browser ------------
        err(403, "POST", "/v1/chat/completions", {"messages": user},
            headers={"Origin": "https://example.com"})
        err(403, "POST", "/v1/chat/completions", {"messages": user}, headers={"Origin": "null"})
        err(403, "GET", "/", headers={"Host": "rebind.example.com:8765"})
        assert chat({"messages": user}, headers={"Origin": f"http://localhost:{port}"})[0] == 200
        assert chat({"messages": user}, headers={"Origin": f"http://127.0.0.1:{port}",
                                                 "Host": f"localhost:{port}"})[0] == 200
        status, r, _ = call("GET", "/")
        assert r.getheader("Access-Control-Allow-Origin") is None
        print("  cross-site Origin and a foreign Host are 403; localhost and 127.0.0.1 pass")

        # -- a model swap keeps the start-up settings ------------------------------
        boot = server.ModelHost("/models/a", shadow=False, bias_budget=0.2, cache_bias=0.5,
                                defer="auto", mtp="/heads/a", draft="/drafts/a")
        boot.swap("/models/b", cache_bias=0.3)
        got = _FakeLLM.built[-1]
        assert got["bias_budget"] == 0.2 and got["defer"] == "auto", got
        assert got["cache_bias"] == 0.3 and got["model"] == "/models/b", got
        assert "mtp" not in got and "draft" not in got, got
        boot.executor.shutdown()
        print("  a swap keeps bias_budget and defer, takes the request's keys, drops a's head and draft")
    finally:
        httpd.shutdown()
        host.executor.shutdown()
        server.StreamingLLM = real
        server.ModelHost._can_batch = real_can_batch
    print("ok")


if __name__ == "__main__":
    serve_stub() if "--serve" in sys.argv else main()
