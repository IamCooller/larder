#!/usr/bin/env python3
"""A client that hangs up on a streamed serial (MoE) turn stops it.

ModelHost.generate(stream=True) used to decode the whole reply to EOS or
max_tokens after the consumer was closed, and the single worker stayed busy
for the next request. Now produce() stops at its next token, closes
llm.stream(), journals nothing and does not count the turn.

The same for a non-streaming request (generate(stream=False,
client_gone=Handler.client_gone), the OpenAI and Messages routes): a closed
client socket stops the turn at its next token, serial or batch, and a request
that was already abandoned in the queue is not run at all.

On the toy engines (CPU, reuse on, greedy, served serially): the hybrid
(support.toy_llm), a dense llama, and the dense one with a draft (the
spec arm):
  1. read 2 pieces, close: the worker is free within a few tokens' time and
     far fewer than max_tokens were decoded;
  2. nothing journaled, _after not called for the abandoned turn;
  3. the same request resent (what an agent does after Esc) is a reuse hit:
     only a tail is prefilled, and its reply equals a fresh host's;
  4. the non-streaming twin of 1-3 over a real socket pair, a request
     abandoned while queued, a connected client's reply equal to the
     engine's own ask(), the batch path on the dense toy, and both routes
     (/v1/chat/completions, /v1/messages) over a real connection.

    python3 tests/test_stop_abandoned.py
"""

import json
import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import moe_stream                                              # noqa: E402
import server                                                  # noqa: E402
import support
from support import toy_llm                                    # noqa: E402

MAX = 200
TOKEN_S = 0.05
MSGS = [{"role": "user", "content": "Name three rivers."}]


def host(llm):
    llm.reuse = True
    h = server.ModelHost.__new__(server.ModelHost)
    h.executor = ThreadPoolExecutor(max_workers=1)
    h.llm, h.store, h.batch = llm, None, False
    h._lock, h._asks, h._turns = threading.Lock(), 0, 0
    h.shadow_on, h._shadow_last = False, None
    h.resume = lambda: None
    h.journaled, h.finished = [], []
    h._journal = lambda messages: h.journaled.append(messages)
    after = h._after
    h._after = lambda *a: (h.finished.append(a), after(*a))
    # One slow token at a time, counted, so a stop is visible.
    h.decoded, stream = [0], llm.stream

    def slow(*a, **kw):
        for piece in stream(*a, **kw):
            h.decoded[0] += 1
            time.sleep(TOKEN_S)
            yield piece
    llm.stream = slow
    return h


def reply(h):
    return "".join(h.generate(MSGS, 12, stream=True))


def check(name, make):
    h = host(make())
    pieces = h.generate(MSGS, MAX, stream=True)
    got = [next(pieces), next(pieces)]
    t0 = time.perf_counter()
    pieces.close()
    # The worker is free: a second generate() runs to its end.
    second = reply(h)
    wait = time.perf_counter() - t0
    fed, cached = h.llm.stats()["prompt_tokens"], h.llm.stats()["cached_tokens"]
    assert h.decoded[0] < 40, f"{name}: abandoned turn decoded {h.decoded[0]} of {MAX}"
    assert wait < 2.0, f"{name}: second request waited {wait:.2f} s"
    assert len(h.journaled) == 1 and len(h.finished) == 1, \
        f"{name}: only the second turn may be journaled and finished"
    f = host(make())
    fresh = reply(f)
    full = f.llm.stats()["prompt_tokens"]
    assert second == fresh, (name, second, fresh)
    assert cached > 0 and fed < full, f"{name}: resend fed {fed} of {full}, cached {cached}"
    print(f"{name}: abandoned after {len(got)} pieces, decoded {h.decoded[0] - 12} of "
          f"{MAX}; resend done in {wait:.2f} s, fed {fed} of {full} ids (cached {cached}), "
          "same reply as a fresh host; 1 journal entry")
    h.executor.shutdown()
    f.executor.shutdown()


def client():
    """(the client's end, a Handler whose connection is the server's end)."""
    a, b = socket.socketpair()
    conn = server.Handler.__new__(server.Handler)
    conn.connection = b
    return a, conn


def call(h, a_conn, n, info=None):
    """generate(stream=False) on a thread with a_conn's client_gone; returns (thread, out)."""
    out = []
    t = threading.Thread(target=lambda: out.append(
        h.generate(MSGS, n, stream=False, info=info, client_gone=a_conn.client_gone)))
    t.start()
    return t, out


def check_plain(name, make):
    h = host(make())
    a, conn = client()
    assert not conn.client_gone(), "an idle connected client reads as gone"
    # Abandoned while queued behind another job: no engine call at all.
    a2, conn2 = client()
    gate = threading.Event()
    h.executor.submit(gate.wait)
    t, out = call(h, conn2, MAX)
    a2.close()
    gate.set()
    t.join(10)
    assert out == [""] and h.decoded[0] == 0 and h._asks == 0, (name, out, h.decoded, h._asks)
    # Abandoned mid-decode.
    info = {}
    t, out = call(h, conn, MAX, info)
    while h.decoded[0] < 2:
        time.sleep(0.01)
    t0 = time.perf_counter()
    a.close()
    t.join(10)
    freed = time.perf_counter() - t0
    assert not t.is_alive() and freed < 1.0, f"{name}: worker freed after {freed:.2f} s"
    assert h.decoded[0] < 40, f"{name}: abandoned turn decoded {h.decoded[0]} of {MAX}"
    assert h.journaled == [] and h.finished == [] and info == {}, \
        f"{name}: the abandoned turn was journaled, finished or counted"
    stopped = h.decoded[0]
    # The resend, connected: a reuse hit that equals a fresh host's reply.
    b, conn3 = client()
    second = h.generate(MSGS, 12, stream=False, client_gone=conn3.client_gone)
    fed, cached = h.llm.stats()["prompt_tokens"], h.llm.stats()["cached_tokens"]
    assert len(h.journaled) == 1 and len(h.finished) == 1
    f = host(make())
    fresh = f.generate(MSGS, 12, stream=False, client_gone=conn3.client_gone)
    full = f.llm.stats()["prompt_tokens"]
    assert second == fresh == make().ask(messages=MSGS, max_tokens=12), (name, second, fresh)
    assert cached > 0 and fed < full, f"{name}: resend fed {fed} of {full}, cached {cached}"
    print(f"{name} (non-streaming): hung up after {stopped} tokens of {MAX}, worker free in "
          f"{freed:.2f} s, nothing journaled or counted; a client gone while queued ran "
          f"nothing; resend fed {fed} of {full} ids (cached {cached}), reply equals a fresh "
          "host's and the engine's own ask()")
    for x in (h, f):
        x.executor.shutdown()


def check_http(make):
    """Both non-streaming routes over a real connection: a client that closes stops the turn, a connected one gets its 200."""
    want = make().ask(messages=MSGS, max_tokens=6)
    for route, body, text in (
            ("/v1/chat/completions", {"messages": MSGS, "max_tokens": MAX},
             lambda r: r["choices"][0]["message"]["content"]),
            ("/v1/messages", {"model": "x", "messages": MSGS, "max_tokens": MAX},
             lambda r: r["content"][0]["text"])):
        h = host(make())
        h._measure = lambda info: None
        httpd = server.serve(h, 0)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        port = httpd.socket.getsockname()[1]

        def post(max_tokens):
            raw = json.dumps(dict(body, max_tokens=max_tokens)).encode()
            s = socket.create_connection(("127.0.0.1", port))
            s.sendall((f"POST {route} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
                       f"Content-Type: application/json\r\nContent-Length: {len(raw)}\r\n\r\n"
                       ).encode() + raw)
            return s
        s = post(MAX)
        while h.decoded[0] < 3:
            time.sleep(0.01)
        s.close()
        h.executor.submit(int).result(10)       # the worker is free again
        stopped = h.decoded[0]
        assert stopped < 40 and h.journaled == [] and h.finished == [], (route, stopped)
        s = post(6)
        reply = b""
        while chunk := s.recv(65536):
            reply += chunk
        head, _, payload = reply.partition(b"\r\n\r\n")
        assert head.startswith(b"HTTP/1.0 200") and text(json.loads(payload)) == want, (route, reply)
        print(f"{route} (non-streaming, real socket): closed after 3 tokens of {MAX}, worker "
              f"stopped at {stopped}, nothing journaled; a connected client got its 200 with "
              "the engine's own reply")
        httpd.shutdown()
        h.executor.shutdown()


def check_http_held(make):
    """Both routes streaming with tools listed hold the reply and write nothing while decoding, so a client that closes is seen on the socket alone."""
    schema = {"type": "object", "properties": {"path": {"type": "string"}}}
    for route, body in (
            ("/v1/chat/completions", {"messages": MSGS, "max_tokens": MAX, "stream": True, "tools": [
                {"type": "function", "function": {"name": "read", "parameters": schema}}]}),
            ("/v1/messages", {"model": "x", "messages": MSGS, "max_tokens": MAX, "stream": True,
                              "tools": [{"name": "read", "input_schema": schema}]})):
        h = host(make())
        h._measure = lambda info: None
        httpd = server.serve(h, 0)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        port = httpd.socket.getsockname()[1]
        raw = json.dumps(body).encode()
        s = socket.create_connection(("127.0.0.1", port))
        s.sendall((f"POST {route} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
                   f"Content-Type: application/json\r\nContent-Length: {len(raw)}\r\n\r\n").encode() + raw)
        while h.decoded[0] < 3:
            time.sleep(0.01)
        s.close()
        h.executor.submit(int).result(10)
        stopped = h.decoded[0]
        assert stopped < 40 and h.journaled == [] and h.finished == [], (route, stopped)
        print(f"{route} (streaming, tools listed, real socket): closed after 3 tokens of {MAX}, "
              f"worker stopped at {stopped}, nothing journaled")
        httpd.shutdown()
        h.executor.shutdown()


def check_batch(name, target, tok, draft=None):
    import test_batching as tb
    want = tb.engine(target, tok, draft).ask(messages=MSGS, max_tokens=12)
    h, _ = tb.host(tb.engine(target, tok, draft))
    assert h.batch, name
    made, emit = [0], h._emit

    def slow(t, id_):
        made[0] += 1
        time.sleep(TOKEN_S)
        return emit(t, id_)
    h._emit = slow
    a, conn = client()
    t, out = call(h, conn, MAX)
    while made[0] < 2:
        time.sleep(0.01)
    t0 = time.perf_counter()
    a.close()
    t.join(10)
    freed = time.perf_counter() - t0
    assert not t.is_alive() and freed < 1.5, f"{name}: returned after {freed:.2f} s"
    b, conn2 = client()
    second = h.generate(MSGS, 12, stream=False, client_gone=conn2.client_gone)
    assert second == want, (name, second, want)
    assert made[0] - 12 < 20, f"{name}: abandoned turn decoded {made[0] - 12} of {MAX}"
    print(f"{name} (non-streaming, batch path): hung up after 2 tokens, returned in "
          f"{freed:.2f} s, decoded {made[0] - 12} of {MAX}; the next reply equals the solo one")
    h.executor.shutdown()


def main():
    if not support.has_tokenizer():
        print("SKIPPED (no tokenizer: set MOE_TOY_TOKENIZER)")
        return
    import test_batching as tb
    from mlx_lm.utils import load_tokenizer
    from test_reuse import cpu
    cpu()
    gen = sys.modules["mlx_lm.generate"]
    tb.spec_draft.generation_stream = gen.generation_stream
    tb.spec_draft.wired_limit = gen.wired_limit
    tok = load_tokenizer(support.tokenizer_dir())
    vocab = max([tok.vocab_size] + list(tok.get_added_vocab().values())) + 1
    target, draft = tb.toys(vocab)
    check("hybrid", lambda: toy_llm("slice", max_tokens=MAX))
    check("dense", lambda: tb.engine(target, tok))
    check("dense + draft", lambda: tb.engine(target, tok, draft))
    check_plain("hybrid", lambda: toy_llm("slice", max_tokens=MAX))
    check_plain("dense", lambda: tb.engine(target, tok))
    check_plain("dense + draft", lambda: tb.engine(target, tok, draft))
    check_batch("dense", target, tok)
    check_http(lambda: toy_llm("slice", max_tokens=MAX))
    check_http_held(lambda: toy_llm("slice", max_tokens=MAX))
    print("ok")


if __name__ == "__main__":
    main()
