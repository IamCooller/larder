#!/usr/bin/env python3
"""The thinking budget (sampler.ThinkBudget, StreamingLLM._think_cap, server.py --thinking-budget and the
Messages route's thinking.budget_tokens) on scripted samplers and the toy engine. No checkpoint.

    perl -e 'alarm shift; exec @ARGV' 280 python3 tests/test_think_budget.py

Checked: the wrapper hands back its inner sampler's ids before the budget and after the forced ids, forces
Qwen's early-stop ids at the budget, and forces nothing when </think> came first (also as the last thinking
token, the one read with a wait); on the toy engine, a reply with no budget or one it never reaches is the
reply of today, a budget of N is N thinking ids then the early-stop ids then the answer (greedy and sampled,
with an MTP head loaded, with tools), and a reply that closes its think block before N is unchanged; through
server.py, both routes streamed and whole split the reply into thinking (the early-stop phrase included) and
answer, and budget_tokens and --thinking-budget take the smaller one.
"""

import http.client
import json
import sys
import threading
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mlx.core as mx                                              # noqa: E402

import sampler                                                     # noqa: E402
import server                                                      # noqa: E402
import spec                                                        # noqa: E402
import support                                                     # noqa: E402
from sampler import Sampling, ThinkBudget                          # noqa: E402

N, M = 6, 48        # the budget, and the reply's max_tokens: room for the early-stop ids and an answer
ASK = [{"role": "user", "content": "Say the word apple."}]
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Weather in a city",
                                           "parameters": {"type": "object",
                                                          "properties": {"city": {"type": "string"}}}}}]


class Script:
    """A sampler that returns its ids in order, one a call."""
    seed = None

    def __init__(self, ids):
        self.ids, self.i = list(ids), 0

    def __call__(self, lp):
        self.i += 1
        self.last = mx.array([self.ids[self.i - 1]], mx.uint32)
        return self.last


class CloseAt:
    """Greedy, except that call `at` writes </think>."""
    seed = None

    def __init__(self, at, close):
        self.at, self.close, self.k = at, close, 0

    def __call__(self, lp):
        self.k += 1
        if self.k - 1 == self.at:
            return mx.full(lp.shape[:-1], self.close, mx.uint32)
        return mx.argmax(lp, axis=-1)


def test_wrapper():
    lp = mx.zeros((1, 16))
    forced, close = [100, 101, 102], 9

    def run(inner, n):
        tb = ThinkBudget(inner, 4, forced, close)
        return [tb(lp).item() for _ in range(n)], tb
    got, tb = run(Script(range(1, 20)), 12)
    assert got == [1, 2, 3, 4, 100, 101, 102, 5, 6, 7, 8, 9] and tb.cut, got
    for at in (1, 3):           # 3: the last thinking id, read with the one wait
        ids = [1, 2, 3, 4, 5, 6, 7, 8]
        ids[at] = close
        got, tb = run(Script(ids), 8)
        assert got == ids and not tb.cut, (at, got)
    peak = mx.zeros((1, 16)).at[0, 7].add(5.0)
    tb = ThinkBudget(None, 2, forced, close)
    assert [tb(peak).item() for _ in range(7)] == [7, 7, 100, 101, 102, 7, 7]
    inner = Script(range(1, 20))
    tb = ThinkBudget(inner, 1, forced, close)
    assert tb(lp) is inner.last and [tb(lp).item() for _ in range(3)] == forced and tb(lp) is inner.last
    print("  wrapper: inner's own ids, then the early-stop ids at the budget, then inner's again; "
          "none forced after a </think> at 1 or at budget-1; argmax with no inner")


def reply(llm, budget=None, sampling=None, tools=None, max_tokens=M):
    """The engine's reply to ASK with thinking on: (text, ids, stats)."""
    llm.reset()
    llm.set_thinking(True, budget)
    text = "".join(llm.stream(messages=ASK, max_tokens=max_tokens, sampling=sampling, tools=tools))
    st = llm.stats()
    return text, llm._seen[len(llm._seen) - st["tokens"]:], st


def test_engine():
    llm = support.toy_llm("slice", max_tokens=M)
    tok = llm.tokenizer
    stop = tok.encode(sampler.EARLY_STOP, add_special_tokens=False)
    close = tok.encode("</think>", add_special_tokens=False)
    assert len(close) == 1 and stop[-2] == close[0], (stop, close)
    close = close[0]

    text0, ids0, st0 = reply(llm)
    assert len(ids0) == M and close not in ids0 and "think_cut" not in st0, (len(ids0), st0)
    text, ids, st = reply(llm, budget=M + 10)
    assert (text, ids) == (text0, ids0) and st["think_cut"] is False
    llm.set_thinking(False, N)      # thinking off: no budget, whatever was asked
    assert llm._think_cap(None) is None
    print(f"  engine, no budget or one never reached: the reply of today ({M} ids)")

    text, ids, st = reply(llm, budget=N)
    assert ids[:N] == ids0[:N] and ids[N:N + len(stop)] == stop and len(ids) == M, ids
    assert st["think_cut"] is True, st
    reasoning, answer = server.split_think(text)
    assert reasoning.endswith("based on the thinking directly now.") and \
        "Considering the limited time" in reasoning and answer == text.split("</think>", 1)[1].lstrip("\n")
    print(f"  engine, budget {N}: {N} thinking ids, the {len(stop)} early-stop ids, "
          f"{M - N - len(stop)} answer ids")

    S = Sampling(0.8, top_k=20, seed=5)
    _, s0, _ = reply(llm, sampling=S)
    _, s1, st = reply(llm, budget=N, sampling=S)
    assert s1[:N] == s0[:N] and s1[N:N + len(stop)] == stop and st["think_cut"], (s0, s1)
    print(f"  engine, sampled: the same {N} draws, then the early-stop ids")

    for at in (2, N - 1):
        _, a, _ = reply(llm, sampling=CloseAt(at, close))
        _, b, st = reply(llm, budget=N, sampling=CloseAt(at, close))
        assert a == b and a[at] == close and st["think_cut"] is False, (at, a, b)
    print(f"  engine, </think> at 2 and at {N - 1}: the reply is unchanged, nothing forced")

    from mtp import MTPHead
    head = MTPHead(llm.model.args)
    mx.eval(head.parameters())
    llm._spec = spec.Speculator(llm.model, head, gamma=2)
    llm.decode_policy_free = types.SimpleNamespace(pick=lambda: "mtp", record=lambda *x: None)
    try:
        st = reply(llm)[2]
        assert st.get("mtp_rounds", 0) > 0, st
        _, m1, st = reply(llm, budget=N)
        assert m1 == ids and "mtp_rounds" not in st, st
    finally:
        llm._spec, llm.decode_policy_free = None, None
    st = reply(llm, tools=TOOLS)[2]
    assert "forced_only" in st, st
    _, t1, st = reply(llm, budget=N, tools=TOOLS)
    assert t1[N:N + len(stop)] == stop and "forced_only" not in st, st
    print("  engine, an MTP head or tools loaded: the budget turn decodes plain, forced the same way")
    return llm, text0, text


def test_routes(llm, text0, text):
    real = server.StreamingLLM
    server.StreamingLLM = lambda model, **kw: llm
    try:
        host = server.ModelHost(support.MODEL_LABEL, shadow=False, reuse=False)
    finally:
        server.StreamingLLM = real
    httpd = server.serve(host, 0)
    port = httpd.socket.getsockname()[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    def call(path, body):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
        conn.request("POST", path, json.dumps(body).encode(), {"Content-Type": "application/json"})
        r = conn.getresponse()
        data = r.read().decode()
        conn.close()
        assert r.status == 200, (r.status, data[:300])
        return data

    def sse(data):
        return [json.loads(line[6:]) for line in data.split("\n")
                if line.startswith("data: ") and line != "data: [DONE]"]

    def openai(stream):
        body = {"messages": ASK, "max_tokens": M, "stream": stream,
                "chat_template_kwargs": {"enable_thinking": True}}
        data = call("/v1/chat/completions", body)
        if not stream:
            m = json.loads(data)["choices"][0]["message"]
            return m.get("reasoning_content") or "", m["content"] or ""
        deltas = [c["choices"][0]["delta"] for c in sse(data) if c.get("choices")]
        return ("".join(d.get("reasoning_content") or "" for d in deltas).strip(),
                "".join(d.get("content") or "" for d in deltas))

    def anthropic(stream, thinking):
        body = {"model": "x", "messages": ASK, "max_tokens": M, "stream": stream, "thinking": thinking}
        data = call("/v1/messages", body)
        if not stream:
            blocks = json.loads(data)["content"]
            return ("".join(b["thinking"] for b in blocks if b["type"] == "thinking"),
                    "".join(b["text"] for b in blocks if b["type"] == "text"))
        deltas = [e["delta"] for e in sse(data) if e["type"] == "content_block_delta"]
        return ("".join(d.get("thinking", "") for d in deltas).strip(),
                "".join(d.get("text", "") for d in deltas))

    want0, want = server.split_think(text0), server.split_think(text)
    try:
        for stream in (False, True):
            how = "streamed" if stream else "whole"
            assert anthropic(stream, {"type": "enabled"}) == want0, how
            assert openai(stream) == want0, how
            host.think_budget = N
            try:
                assert openai(stream) == want, how
                assert anthropic(stream, {"type": "enabled", "budget_tokens": N + 20}) == want, how
            finally:
                host.think_budget = None
            assert anthropic(stream, {"type": "enabled", "budget_tokens": N}) == want, how
        print(f"  routes: chat completions and messages, streamed and whole: no budget is today's split, budget {N} "
              f"puts the early-stop phrase in the thinking; --thinking-budget {N} beats budget_tokens {N + 20}")
    finally:
        httpd.shutdown()


def test_batch():
    """server.py's batch loop (dense targets), on the CPU as test_batching: a turn with a budget decodes
    alone through llm.stream, forced there, and the two behind it still share forwards."""
    import spec_draft
    import test_batching as tb
    from mlx_lm.utils import load_tokenizer
    from test_reuse import cpu
    cpu()
    gen = sys.modules["mlx_lm.generate"]
    spec_draft.generation_stream = gen.generation_stream
    spec_draft.wired_limit = gen.wired_limit
    tok = load_tokenizer(support.tokenizer_dir())
    target, _ = tb.toys(max([tok.vocab_size] + list(tok.get_added_vocab().values())) + 1)
    ref = tb.engine(target, tok)
    ref.set_thinking(True, N)
    want = [ref.ask(messages=tb.MSGS[0], max_tokens=M)]
    ref.set_thinking(True)
    want += [ref.ask(messages=tb.MSGS[i], max_tokens=M) for i in (1, 2)]
    h, seen = tb.host(tb.engine(target, tok))
    gate = threading.Event()
    h.executor.submit(gate.wait)
    try:
        pieces = [h.generate(tb.MSGS[i], M, stream=True, thinking=True, think_budget=N if i == 0 else None)
                  for i in range(3)]
    finally:
        gate.set()
    got = ["".join(x) for x in pieces]
    assert got == want and "Considering the limited time" in got[0] and seen["width"] == 2, (got, want, seen)
    print("  batch: the budget turn alone and forced, the two behind it rows of one forward, "
          "every reply its solo one")


def main():
    mx.random.seed(1234)
    test_wrapper()
    if not support.has_tokenizer():
        print("engine and routes: SKIPPED (no tokenizer: set MOE_TOY_TOKENIZER)")
        print("ok")
        return
    test_routes(*test_engine())
    test_batch()        # last: it moves the process to the CPU
    print("ok")


if __name__ == "__main__":
    main()
