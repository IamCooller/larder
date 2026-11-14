#!/usr/bin/env python3
"""On a KB-scale toy model (no weights, no GPU): StreamingLLM.warm_tiles runs one prompt pass of each MLX CUDA qmm tile
height on a throwaway cache and keeps nothing, ModelHost.warm_tiles runs it on CUDA only and never fails the warm-up,
the server runs it on an `X-Warmup: tiles` request's messages, before its reply, and on no other request, and the
launcher asks for it on a model's first start on a kernel folder only (launch.tiles_marker).

    perl -e 'alarm shift; exec @ARGV' 280 python3 tests/test_warm_tiles.py
"""
import json
import sys
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import mlx.core as mx                                           # noqa: E402
from mlx_lm.models.cache import make_prompt_cache               # noqa: E402

import moe_stream                                               # noqa: E402
import server                                                   # noqa: E402
import stream_fast                                              # noqa: E402
from support import kv, toy_model                               # noqa: E402


MSGS = [{"role": "user", "content": "Say hello in one short sentence."}]


def tile(rows):
    """MLX 0.32.4's CUDA rule (quantized.cpp, qmm_sm80.cu): qmv under 8 rows, else a qmm tile of this height."""
    return None if rows < 8 else max(16, min(64, 1 << (rows - 1).bit_length()))


class Budget:
    """MemBudget's pass record alone: tick sets `_pass` as a pass begins and close() folds it at the next one's entry."""

    def __init__(self):
        self._pass, self.folded = "warm-up", []


class Watch:
    """The toy model, with the width of every pass it runs recorded and an optional failure on pass `fail`."""

    def __init__(self, model, fail=None):
        self.m, self.widths, self.fail, self.budget = model, [], fail, None

    def __call__(self, x, cache=None):
        self.widths.append(x.shape[1])
        if self.budget is not None:
            self.budget.folded.append(self.budget._pass)
            self.budget._pass = x.shape[1]
        if len(self.widths) == self.fail:
            raise RuntimeError("pass failed")
        return self.m(x, cache=cache)

    def __getattr__(self, name):
        return getattr(self.m, name)


def engine(fail=None):
    """A StreamingLLM holding one finished 25-id turn and one kept conversation, as after the warm-up reply."""
    llm = moe_stream.StreamingLLM.__new__(moe_stream.StreamingLLM)
    llm.model = Watch(toy_model(n_layers=4), fail)
    llm.prefill_step = None
    llm._seen = list(range(5, 30))
    llm._cache = make_prompt_cache(llm.model)
    llm.model(mx.array(llm._seen)[None], cache=llm._cache)
    mx.eval([c.state for c in llm._cache])
    llm._convs, llm._pieces, llm._parked, llm._dropped = ({"seen": [1]},), (25,), 0, 0
    llm.model.widths.clear()
    return llm


def check_engine():
    llm = engine()
    held = (llm._cache, llm._convs, llm._pieces, llm._parked, llm._dropped)
    before = [(mx.array(k), mx.array(v)) for k, v in kv(llm._cache)]
    llm.warm_tiles()
    got = llm.model.widths
    assert got == list(moe_stream.TILE_ROWS), got
    assert sorted(tile(n) for n in got) == [16, 32, 64], [tile(n) for n in got]
    assert (llm._cache, llm._convs, llm._pieces, llm._parked, llm._dropped) == held
    after = kv(llm._cache)
    assert all(mx.array_equal(a, c).item() and mx.array_equal(b, d).item()
               for (a, b), (c, d) in zip(before, after)), "the live cache changed"
    assert llm._seen == list(range(5, 30))

    failing = engine(fail=2)
    held = (failing._cache, failing._convs, failing._pieces)
    try:
        failing.warm_tiles()
        raise AssertionError("a failed pass must raise")
    except RuntimeError:
        pass
    assert (failing._cache, failing._convs, failing._pieces) == held, "a failed pass left its cache behind"

    budgeted = engine()
    budgeted._budget = budgeted.model.budget = Budget()
    budgeted.warm_tiles()
    b = budgeted._budget
    assert b.folded == ["warm-up", None, None] and b._pass is None, (b.folded, b._pass)

    fresh = engine()
    fresh._seen, rendered = [], []
    fresh.tokenizer = Tokenizer()
    fresh._raw_render = lambda hist: rendered.append(hist) or "seven"
    fresh.warm_tiles(MSGS)
    assert rendered == [MSGS] and fresh.model.widths == list(moe_stream.TILE_ROWS), (rendered, fresh.model.widths)

    empty = engine()
    empty._seen = []
    empty.warm_tiles()
    assert empty.model.widths == []
    print(f"engine: passes {got} hit tiles {sorted(tile(n) for n in got)}; the live cache, kept conversations and "
          "turn counters unchanged, also after a failed pass; the budget folds the warm-up's pass and none of these; "
          "a fresh engine renders the warm-up's messages; no ids, no pass")


class Tokenizer:
    """tools.encode's one call: a 7-id render."""

    def encode(self, text):
        return list(range(3, 10))


class FakeLLM:
    def __init__(self, fail=False):
        self.calls, self.fail = 0, fail

    def warm_tiles(self, messages):
        assert messages == MSGS
        self.calls += 1
        if self.fail:
            raise RuntimeError("no room")


def model_host(llm):
    h = server.ModelHost.__new__(server.ModelHost)
    h.executor, h.llm = ThreadPoolExecutor(max_workers=1), llm
    return h


def check_host():
    real = stream_fast.BACKEND
    try:
        for backend, want in (("metal", 0), ("cpu", 0), ("cuda", 1)):
            stream_fast.BACKEND = backend
            h = model_host(FakeLLM())
            h.warm_tiles(MSGS)
            assert h.llm.calls == want, (backend, h.llm.calls)
            h.executor.shutdown()
        h = model_host(FakeLLM(fail=True))
        h.warm_tiles(MSGS)                      # said, never raised
        assert h.llm.calls == 1 and h.executor.submit(int).result(5) == 0
        h.executor.shutdown()
        h = model_host(None)
        h.warm_tiles(MSGS)
    finally:
        stream_fast.BACKEND = real
    print("host: runs on CUDA only, a failure is said and the worker goes on, no model is a no-op")


class FakeHost:
    vision, think_default, llm, store = False, False, None, None

    def __init__(self):
        self.order = []

    def generate(self, messages, max_tokens, stream, info=None, **kw):
        self.order.append("reply")
        return "Hello."

    def warm_tiles(self, messages):
        assert messages == MSGS
        self.order.append("tiles")


def check_handler():
    host = FakeHost()
    httpd = server.serve(host, 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    port = httpd.socket.getsockname()[1]
    body = json.dumps({"messages": MSGS, "max_tokens": 8}).encode()
    try:
        for warm, want in ((None, ["reply"]), ("1", ["reply", "reply"]), ("tiles", ["reply", "reply", "tiles", "reply"]),
                           (None, ["reply", "reply", "tiles", "reply", "reply"])):
            headers = {"Content-Type": "application/json", **({"X-Warmup": warm} if warm else {})}
            req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body, headers)
            with urllib.request.urlopen(req, timeout=30) as r:
                assert json.load(r)["choices"][0]["message"]["content"] == "Hello."
            assert host.order == want, (warm, host.order)
    finally:
        httpd.shutdown()
    print("handler: X-Warmup: tiles runs the tile passes on its messages before its reply; X-Warmup: 1 and a plain "
          "request do not")


def check_launcher():
    """A model's first warm-up on a kernel folder asks for the passes and leaves its marker; later ones and Metal do not."""
    import tempfile
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import launch
    got = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            got.append(self.headers.get("X-Warmup"))
            body = json.dumps({"choices": [{"message": {"content": "Hello."}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass
    tmp = Path(tempfile.mkdtemp())
    kd = tmp / "mlx-kernels" / "0.32.4-abc"
    plan = {"env": {"MLX_PTX_CACHE_DIR": str(kd)}}
    big, small = launch.tiles_marker(plan, tmp / "models" / "qwen36-35b-q40"), launch.tiles_marker(plan, tmp / "qwen3-4b")
    assert big == kd / "tiles-qwen36-35b-q40" and small == kd / "tiles-qwen3-4b", (big, small)
    assert launch.tiles_marker({"env": {}}, tmp / "qwen3-4b") is None                      # Metal: no kernel folder
    assert launch.tiles_marker({"env": {"MLX_PTX_CACHE_DIR": str(tmp / "mine")}}, tmp) is None   # the user's own
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    try:
        assert launch.warm_request(port, small) == (True, "") and small.exists()       # first start: passes, marker
        assert launch.warm_request(port, small)[0]                                      # second: plain
        assert launch.warm_request(port, big)[0] and big.exists()                       # another model: its own
        assert launch.warm_request(port, None)[0]                                       # Metal: plain
        assert not launch.warm_request(port + 1, launch.tiles_marker(plan, tmp / "x"))[0]
        assert not (kd / "tiles-x").exists()                                            # no reply, no marker
    finally:
        srv.shutdown()
        srv.server_close()
    assert got == ["tiles", "1", "tiles", "1"], got
    print("launcher: the first warm-up of each model on a kernel folder asks for the passes and writes its marker; "
          "later warm-ups, Metal and a failed reply leave none")


if __name__ == "__main__":
    check_engine()
    check_host()
    check_handler()
    check_launcher()
    print("ok")
