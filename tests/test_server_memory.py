#!/usr/bin/env python3
"""server.py's --memory plumbing, its routes and the completion fields, with no model and no GPU.

    perl -e 'alarm shift; exec @ARGV' 280 python3 tests/test_server_memory.py

Four ways --memory can be wrong with nothing raising: a request that does not render as a
continuation of the core, a core that reaches the engine before the reset or not as a copy, a sleep
that rebuilds when the plan did not change, and routes that exist without --memory.
"""

import http.client
import inspect
import json
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

import mlx.core as mx                                          # noqa: E402

from moe_stream import StreamingLLM                            # noqa: E402
from sampler import Sampling                                   # noqa: E402
from server import Handler, ModelHost, serve                   # noqa: E402


class _FakeLLM:
    """Everything ModelHost touches, recording what it was handed."""
    reply = "Biscuit"

    def __init__(self):
        self.given, self.resets = [], 0

    def reset(self):
        self.resets += 1

    def forget(self):
        pass

    def ask(self, **kw):
        return "".join(self.stream(**kw))     # as StreamingLLM.ask does

    def stream(self, messages=None, max_tokens=None, resume=None, tools=None,
               response_format=None, sampling=None, keep_on_close=False):
        self.given.append(resume)
        self.max_tokens, self.response_format = max_tokens, response_format
        self.sampling = sampling
        half = len(self.reply) // 2
        yield self.reply[:half]
        yield self.reply[half:]

    def stats(self):
        return {}


class _FakeStore:
    """A store whose plan changes only when a turn says something rare."""

    def __init__(self):
        self.sessions, self.rebuilds, self.sha = [], 0, "a"

    def append_session(self, turns):
        self.sessions.append(list(turns))
        if any("rare" in x for x in turns):
            self.sha = f"sha{len(self.sessions)}"

    def plan(self):
        return {"sha": self.sha}

    def rebuild(self, llm):
        self.rebuilds += 1

    def attach(self, llm):
        return False

    def ensure(self, llm):
        return self.attach(llm)

    def stats(self):
        return {"plan_sha": self.sha, "sessions_total": len(self.sessions)}

    def summary(self):
        return "fake store"


def check_rendering():
    """A core ends with an assistant ack whose end token is not in the cache, so the next user
    turn must open with `between`."""
    def raw_render(hist):
        out = "<|im_start|>system\nYou are helpful.<|im_end|>\n"
        for m in hist:
            out += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        return out + "<|im_start|>assistant\n<think>\n\n</think>\n\n"

    llm = StreamingLLM.__new__(StreamingLLM)      # no weights, no arena
    llm._raw_render = raw_render
    llm.tokenizer = None                          # no added tokens, so _next_chunk escapes nothing
    llm._tpl = llm._learn_template()
    llm._ended_cleanly = False
    q = {"role": "user", "content": "What is my cat called?"}
    assert llm._continue_chunk([q]) == llm._next_chunk(q["content"], False)[0]
    t = llm._tpl
    many = [q, {"role": "assistant", "content": "Biscuit."},
            {"role": "user", "content": "And how old is he?"}]
    assert llm._continue_chunk(many) == (
        t["between"] + many[0]["content"] + t["after_user"]
        + many[1]["content"] + t["between"] + many[2]["content"] + t["tail"])
    print(f"  render: one turn is {llm._continue_chunk([q])[:24]!r}..., "
          f"byte-identical to _next_chunk's")


def check_install_after_reset():
    """The install sits after the reset, asserted on the source because only a model could run it."""
    src = inspect.getsource(StreamingLLM.stream)
    assert src.index("self.reset()") < src.index('resume["cache"]'), src


def check_toy_engine():
    """The whole resume path on a toy engine; only the generation loop is faked."""
    import memory_core as MC
    from support import toy_llm

    try:
        toy = toy_llm("slice", max_tokens=4)
    except (OSError, ValueError) as e:
        print(f"  (toy engine skipped, no tokenizer to load: {e})")
        return
    enc = lambda s: toy.tokenizer.encode(s, add_special_tokens=False)    # noqa: E731
    ids = (MC.turn_ids(toy, "My name is Sam.", True) + enc(MC.ACK)
           + MC.turn_ids(toy, "My cat is called Biscuit.", False)
           + enc(MC.ACK))
    cache, _ = MC.prefill_chunked(toy.model, ids, 512, zero=False)
    fed = []

    def fake_generate(model, tokenizer, prompt, prompt_cache=None, **kw):
        fed.append(list(map(int, prompt)))
        model(prompt[None], cache=prompt_cache)       # the prefill
        model(mx.array([[13]]), cache=prompt_cache)   # and one decode step
        yield types.SimpleNamespace(token=13, text="Biscuit", finish_reason="stop")

    import spec_draft
    real_generate = spec_draft.plain
    spec_draft.plain = fake_generate
    try:
        ask = [{"role": "user", "content": "What is my cat called?"}]
        core = {"cache": cache, "seen": list(ids),
                "history": [{"role": "user", "content": "..."},
                            {"role": "assistant", "content": MC.ACK}]}
        toy.ask(messages=ask, resume=dict(core, cache=MC.clone_cache_keeping_gap(cache)))
        st = toy.stats()
        assert st["cached_tokens"] == len(ids), (st, len(ids))
        assert toy._seen[:len(ids)] == ids
        assert fed[-1] == enc(toy._tpl["between"] + ask[0]["content"] + toy._tpl["tail"])
        kv = [i for i, c in enumerate(cache) if hasattr(c, "offset")]
        assert all(cache[i].offset == len(ids) for i in kv), "core written"
        toy.ask(messages=ask)          # and the same call with no core
        assert toy.stats()["cached_tokens"] == 0
    finally:
        spec_draft.plain = real_generate
    print(f"  toy engine: {st['prompt_tokens']} new / {st['cached_tokens']} cached behind the core, "
          f"{toy.stats()['prompt_tokens']} new / 0 cached without it")


def build_host():
    host = ModelHost.__new__(ModelHost)           # nothing to load
    host.executor = ThreadPoolExecutor(max_workers=1)
    host.llm, host.model_name, host.last_error = _FakeLLM(), "toy", None
    host.memory_tag, host.store, host.core, host.turns = None, None, None, []
    host._lock, host.shadow_on = threading.Lock(), False
    host._asks, host._turns, host._queue_s, host._shadow_last = 0, 0, None, None
    return host


def check_without_memory(host, msgs):
    """Without --memory nothing happens."""
    assert host.resume() is None
    assert host.generate(msgs, 8, stream=False) == "Biscuit"
    assert "".join(host.generate(msgs, 8, stream=True)) == "Biscuit"
    assert host.llm.given == [None, None], host.llm.given
    assert host.turns == []
    print("  no --memory: store None, resume None on both request paths, nothing journalled")


def check_core_is_cloned(host, msgs):
    """The core is cloned, never lent."""
    import memory_core as MC
    core = MC.GapKVCache(0)
    MC.set_kv(core, mx.zeros((1, 2, 3, 4)), mx.arange(24, dtype=mx.float32).reshape(1, 2, 3, 4))
    mx.eval(core.state)
    host.store = _FakeStore()
    host.core = {"cache": [core], "seen": [7, 8, 9],
                 "history": [{"role": "user", "content": "..."},
                             {"role": "assistant", "content": MC.ACK}]}
    a, b = host.resume(), host.resume()
    assert a["cache"][0] is not core and b["cache"][0] is not a["cache"][0]
    assert a["seen"] == [7, 8, 9] and a["history"][1]["content"] == MC.ACK
    assert bool(mx.all(a["cache"][0].state[1] == core.state[1]))

    host.llm.given.clear()
    host.generate(msgs, 8, stream=False)
    "".join(host.generate(msgs, 8, stream=True))
    given = host.llm.given
    assert len(given) == 2 and all(g is not None for g in given)
    assert given[0]["cache"][0] is not core
    assert given[1]["cache"][0] is not given[0]["cache"][0]
    print("  every request is handed its own clone of the core, the stored one is never lent out")


def check_journal_and_sleep(host):
    """The journal keeps the new turn only, and a sleep rebuilds only when the plan changed."""
    assert host.turns == ["u2", "u2"], host.turns
    host.turns = ["u2"]
    store = host.store
    assert "core unchanged" in host.sleep()
    assert store.sessions == [["u2"]] and store.rebuilds == 0
    host.turns = ["my cat is rare"]
    assert host.sleep() == "fake store"
    assert store.rebuilds == 1 and host.turns == []
    assert host.sleep() == "nothing said this session"
    assert store.rebuilds == 1
    host.sleep_after = 600
    host.turns = ["my cat is rare"]
    host.last_request = time.monotonic() - 10
    assert host._idle_check() is False and store.rebuilds == 1
    host.last_request = time.monotonic() - 601
    assert host._idle_check() is True and store.rebuilds == 2 and host.turns == []
    assert host._idle_check() is False
    host.sleep_after = 0
    print("  idle sleep: fires once after the silence, never on an empty session")
    print(f"  sleep: {len(store.sessions)} sessions journalled, "
          f"{store.rebuilds} rebuild, only the one that changed the plan")


def check_routes(host):
    """A server with memory and one without differ by a 404."""
    for tag in (None, "t"):
        host.store = _FakeStore() if tag else None
        httpd = serve(host, 0)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            conn = http.client.HTTPConnection("127.0.0.1", httpd.socket.getsockname()[1], timeout=10)
            codes = []
            for method, path in (("GET", "/v1/memory"), ("POST", "/v1/memory/sleep")):
                conn.request(method, path, b"{}", {"Content-Type": "application/json"})
                r = conn.getresponse()
                r.read()
                codes.append(r.status)
            conn.close()
        finally:
            httpd.shutdown()
        assert codes == ([200, 200] if tag else [404, 404]), (tag, codes)
        print(f"  routes with{'' if tag else 'out'} --memory: "
              f"/v1/memory {codes[0]}, /v1/memory/sleep {codes[1]}")


def check_completions(host):
    """The reply is the client's verbatim unless the request offered tools, and the token limit is
    read under either of its names."""
    host.store = None
    call = ("<tool_call>\n<function=get_weather>\n<parameter=city>\nBerlin\n"
            "</parameter>\n</function>\n</tool_call>")
    host.llm.reply = call
    httpd = serve(host, 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        def completion(body):
            conn = http.client.HTTPConnection("127.0.0.1", httpd.socket.getsockname()[1], timeout=10)
            conn.request("POST", "/v1/chat/completions", json.dumps(body).encode(),
                         {"Content-Type": "application/json"})
            r = conn.getresponse()
            out = json.loads(r.read())
            conn.close()
            return out["choices"][0] if r.status == 200 else r.status
        user = [{"role": "user", "content": "weather?"}]
        ch = completion({"messages": user, "max_completion_tokens": 33})
        assert ch["finish_reason"] == "stop" and ch["message"]["content"] == call, ch
        assert "tool_calls" not in ch["message"]
        assert host.llm.max_tokens == 33, host.llm.max_tokens
        ch = completion({"messages": user, "max_tokens": 44, "tools": [
            {"type": "function", "function": {"name": "get_weather", "parameters": {
                "type": "object", "properties": {"city": {"type": "string"}}}}}]})
        assert ch["finish_reason"] == "tool_calls" and ch["message"]["content"] is None, ch
        assert ch["message"]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert host.llm.max_tokens == 44
        fmt = {"type": "json_schema", "json_schema": {"name": "w", "schema": {
            "type": "object", "properties": {"city": {"type": "string"}}}}}
        ch = completion({"messages": user, "response_format": fmt})
        assert ch["message"]["content"] == call, ch     # proposes, never edits
        assert host.llm.response_format == fmt, host.llm.response_format
        knobs = {"messages": user, "temperature": 0.7, "top_p": 0.9, "top_k": 20, "seed": 4}
        completion(knobs)
        assert host.llm.sampling is None, "sampling fields honoured without --sampling"
        Handler.sampling = True
        completion(knobs)
        assert host.llm.sampling == Sampling(0.7, 20, 0.9, 0.0, 4), host.llm.sampling
        completion(dict(knobs, temperature=0))
        assert host.llm.sampling is None, host.llm.sampling
        assert completion(dict(knobs, top_p=2)) == 400
    finally:
        Handler.sampling = False
        httpd.shutdown()
    host.llm.reply = "Biscuit"
    print("  completions: no tools -> text verbatim (tag kept), tools -> parsed; max_completion_tokens "
          "honoured; response_format reaches the engine; sampling fields only under --sampling, "
          "temperature 0 greedy, out of range 400")


def main():
    mx.set_default_device(mx.cpu)       # the measurement queue owns the GPU
    check_rendering()
    check_install_after_reset()
    check_toy_engine()
    host = build_host()
    msgs = [{"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"}]
    check_without_memory(host, msgs)
    check_core_is_cloned(host, msgs)
    check_journal_and_sleep(host)
    check_routes(host)
    check_completions(host)
    host.executor.shutdown()
    print("ok")


if __name__ == "__main__":
    main()
