#!/usr/bin/env python3
"""Conversation reuse (StreamingLLM.reuse, on in server.py): an agent sends
the whole conversation every step, and only what it added is prefilled.

On the toy engines, fp32: support.toy_llm's hybrid (GatedDeltaNet + KV
layers, the real Qwen3.6 tokenizer and chat template) and test_batching's
dense llama pair.

  1. A 5-step agent loop -- system + tools + a query, then per step the
     model's reply and a tool result of real text -- gives the same tokens
     with reuse as without, on the GPU and on the CPU. Each step feeds only
     what the previous request does not hold: its cached ids are exactly the
     previous request's whole cache where the render repeats it, else its
     newest mark the render still agrees with (dense: the common prefix).
  2. Exactness, on the CPU: after every step the cache equals one prefill of
     the ids it claims to hold. Not on the GPU, where a one-position decode
     step differs from the same positions inside a prefill pass by up to
     1.8e-2 on this toy with no reuse at all (CPU: 4e-6) -- that noise would
     hide a wrong restore.
  3. Tool calls the template renders back token for token (Qwen3.6 keeps an
     empty think block on every turn after the last query) are taken up
     whole: prompt, reply and end token. A new user query re-renders the
     loop's turns without their think blocks: taken up at the mark of the
     turn that answered the last query. A request that extends nothing goes
     to the stored front, and the old conversation is let go.
  4. The MTP arm (a scripted speculator: accepts, rejections, a replay)
     leaves a cache that is taken up the same way.
  5. Bounded: one conversation, at most two marks, and a mark holds the
     recurrent states only -- never a copy of the KV.
  6. Dense: the KV cut back at exactly the common prefix; and server
     batching with reuse on still gives every request its solo reply.
  7. Under a peak budget (--mem-peak-gb) a conversation is kept only while
     the budget's room beside the pool covers it: with none, nothing kept
     and no mark; with room, taken up as without a budget from the second
     turn; dropped when the room runs out. Replies unchanged throughout.
  8. Two conversations kept (server.py --keep-convs 2), CPU: two agents
     alternating, and an agent with a title request after its first step
     and again later. Each request takes up its own conversation where 1
     and 3 above take one up (the whole previous turn, or its mark), the
     other stays kept, and the replies are those of one kept conversation,
     where every switch goes back to the stored front.
  9. Evicted by count and by bytes (--keep-gb), least recently used first;
     the newest stays whatever its size. A call that keeps nothing drops
     only the conversation whose cache it moves on.
 10. A released auto draft takes its half out of every kept conversation.
 11. Batching with three kept: a newcomer takes up its own conversation and
     leaves the other; the batch keeps nothing.

    python3 tests/test_reuse.py
"""
import contextlib
import faulthandler
import sys
import tempfile
import threading
import types
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, make_prompt_cache

import moe_stream
import pcache
import spec
import spec_draft
import stream_fast
import tools as toolfmt
import support
from support import ScriptedSpeculator, gdn, kv, toy_llm

TOOLS = [
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a file of the repository.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file", "description": "Write a file of the repository.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "run_tests", "description": "Run a test file and report.",
        "parameters": {"type": "object", "properties": {
            "target": {"type": "string"}}, "required": ["target"]}}},
]
SYSTEM = ("You are a coding agent in a Python repository. Read, change and "
          "test the code with the tools; keep changes small.")
QUERY = "Find where prefilled fronts are saved and add a test for it."
SRC = (REPO / "pcache.py").read_text()
RESULTS = [SRC[k * 1500:(k + 1) * 1500] for k in range(8)]     # ~400 tokens each
CALLS = [
    "<tool_call>\n<function=read_file>\n<parameter=path>\npcache.py\n"
    "</parameter>\n</function>\n</tool_call>",
    "The save is in save_front.\n\n<tool_call>\n<function=read_file>\n"
    "<parameter=path>\ntests/test_fronts.py\n</parameter>\n</function>\n</tool_call>",
    "<tool_call>\n<function=write_file>\n<parameter=path>\ntests/test_save.py\n"
    "</parameter>\n<parameter=content>\nimport pcache\n\n\ndef test_dir():\n"
    "    assert pcache.FRONTS.name == \"fronts\"\n</parameter>\n</function>\n</tool_call>",
    "<tool_call>\n<function=run_tests>\n<parameter=target>\ntests/test_save.py\n"
    "</parameter>\n</function>\n</tool_call>",
    "<tool_call>\n<function=read_file>\n<parameter=path>\nmoe_stream.py\n"
    "</parameter>\n</function>\n</tool_call>",
]
TOL = 1e-4      # CPU fp32: decode against one prefill pass is <= 4e-6 here
PATTERN = [True, True, False, False, False, False, False]   # accepts, then a replay


def max_abs(a, b):
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))


def fresh_gap(llm):
    """The kept cache against one prefill of the ids it holds: worst |diff|."""
    fresh = make_prompt_cache(llm.model)
    llm.model(mx.array(llm._seen)[None], cache=fresh)
    mx.eval([c.state for c in fresh])
    worst = 0.0
    for (a, b), (c, d) in zip(kv(llm._cache), kv(fresh)):
        assert a.shape == c.shape, (a.shape, c.shape)
        worst = max(worst, max_abs(a, c), max_abs(b, d))
    for a, b in zip(gdn(llm._cache), gdn(fresh)):
        worst = max(worst, max_abs(a, b))
    return worst


def common(a, b):
    m = 0
    while m < min(len(a), len(b)) and a[m] == b[m]:
        m += 1
    return m


def step_msgs(reply, k):
    """What the client sends back: the server's parse of the reply (content
    and tool_calls, server.py's shape) and a tool result."""
    content, calls = toolfmt.parse(reply, TOOLS)
    turn = {"role": "assistant", "content": content or None}
    if calls:
        turn["tool_calls"] = calls
    return [turn, {"role": "tool", "tool_call_id": calls[0]["id"] if calls else f"c{k}",
                   "content": RESULTS[k]}]


def agent_loop(llm, steps=5, max_tokens=10, check=None):
    """Steps of an agent conversation; per step (reply ids, stats), and the
    messages after the last one."""
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": QUERY}]
    out = []
    for k in range(steps):
        prev = llm._convs[-1] if llm._convs else None
        reply = llm.ask(messages=msgs, tools=TOOLS, max_tokens=max_tokens)
        st = llm.stats()
        out.append((llm._seen[len(llm._seen) - st["tokens"]:], st))
        if check:
            check(llm, prev, st, k)
        msgs = msgs + step_msgs(reply, k)
    return out, msgs


def took_up(llm, prev, st, dense=False):
    """Where the turn took the kept cache up, asserted exactly: `seen`'s end
    when the render repeats all of it, else (hybrid) the newest mark the
    render still agrees with, else (dense) the common prefix itself."""
    n = st["prompt_tokens"] + st["cached_tokens"]
    m = min(common(prev["seen"], llm._seen[:n]), n - 1)
    if m == len(prev["seen"]) or dense:
        want = m
    else:
        want = max(k for k, _, _ in prev["marks"] if k <= m)
    assert st["cached_tokens"] == want, (st["cached_tokens"], want, m, len(prev["seen"]))
    return "whole" if want == len(prev["seen"]) else "mark"


def scripted(replies, eos):
    """spec_draft.plain's contract with a scripted reply: the prompt
    prefilled, then every token fed before it is yielded (generate_step's
    order), so the cache holds the reply and its end token."""
    it = iter(replies)

    def gen(model, tokenizer, prompt, max_tokens=256, prompt_cache=None, **kw):
        ids = next(it)
        model(prompt[None], cache=prompt_cache)
        for j, t in enumerate(ids):
            model(mx.array([[t]]), cache=prompt_cache)
            mx.eval([c.state for c in prompt_cache])
            yield types.SimpleNamespace(
                token=t, text="" if t in eos else tokenizer.decode([t]),
                finish_reason="stop" if j == len(ids) - 1 else None, from_draft=False)
    return gen


class AlwaysMTP:
    """The MTP engine's decode policy, pinned to the MTP round."""
    def pick(self):
        return "mtp"

    def record(self, *a):
        pass


def hybrid_loop(where, exact):
    on, off = toy_llm("slice", max_tokens=10), toy_llm("slice", max_tokens=10)
    on.reuse = True
    ways, worst = [], 0.0

    def check(llm, prev, st, k):
        nonlocal worst
        if k:
            ways.append(took_up(llm, prev, st))
        worst = max(worst, fresh_gap(llm))
    got, _ = agent_loop(on, check=check)
    ref, _ = agent_loop(off)
    assert [g for g, _ in got] == [g for g, _ in ref], f"reuse changed a reply ({where})"
    if exact:
        assert worst < TOL, worst
    fed = [st["prompt_tokens"] for _, st in got]
    full = [st["prompt_tokens"] + st["cached_tokens"] for _, st in ref]
    print(f"1. hybrid, greedy, {where}: same tokens with and without reuse over 5 "
          f"steps; fed {fed} of {full} ids, taken up at {', '.join(ways)}; cache vs "
          f"one prefill {worst:.2g}{'' if exact else ' (GPU one-position noise)'}")
    return [g for g, _ in ref]


def check_calls_and_queries():
    llm = toy_llm("slice", max_tokens=64)
    llm.reuse = True
    eos = set(llm.tokenizer.eos_token_ids)
    end = llm.tokenizer.encode("<|im_end|>", add_special_tokens=False)
    replies = [llm.tokenizer.encode(c, add_special_tokens=False) + end for c in CALLS]
    real = spec_draft.plain
    spec_draft.plain = scripted(replies + replies, eos)
    worst = 0.0

    def check(llm, prev, st, k):
        nonlocal worst
        if k:
            assert took_up(llm, prev, st) == "whole", "a call was not taken up whole"
        worst = max(worst, fresh_gap(llm))
    try:
        out, msgs = agent_loop(llm, check=check)
        assert worst < TOL, worst
        print(f"3. tool calls: steps 2-5 each took the previous prompt, reply and "
              f"end token up whole, fed {[st['prompt_tokens'] for _, st in out]}; "
              f"cache vs one prefill {worst:.2g}")

        assert len(llm._convs) == 1, len(llm._convs)
        conv = llm._convs[0]
        q = [n for n, _, query in conv["marks"] if query]
        assert len(q) == 1 and len(conv["marks"]) == 2, conv["marks"]
        assert conv["cache"] is llm._cache, "a second copy of the conversation is kept"
        state = sum(a.nbytes for c in llm._cache if isinstance(c, ArraysCache)
                    for a in c.cache if a is not None)
        held = sum(a.nbytes for _, h, _ in conv["marks"] if h
                   for s in h if isinstance(s, list) for a in s if a is not None)
        assert held == 2 * state, (held, state)
        assert not llm._marks, "a dropped mark is still held by the engine"
        print(f"5. bounded after 5 steps: one conversation, 2 marks (the query's and "
              f"the newest) holding {held / 1e3:.1f} KB, the recurrent state twice "
              f"({state / 1e3:.1f} KB), no KV copy")
        llm.ask(messages=msgs + [{"role": "user", "content": "Now run all the tests."}],
                tools=TOOLS, max_tokens=64)
        st = llm.stats()
        assert st["cached_tokens"] == q[0], (st["cached_tokens"], q)
        gap = fresh_gap(llm)
        assert gap < TOL, gap
        old = llm._convs[-1]["cache"]
        other = [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": "Explain what MoE routing is."}]
        llm.ask(messages=other, tools=TOOLS, max_tokens=64)
        st = llm.stats()
        ids = llm._seen[:st["prompt_tokens"] + st["cached_tokens"]]
        front = llm._front_ids(ids)
        assert front and st["cached_tokens"] == len(front), (st["cached_tokens"], len(front))
        assert len(llm._convs) == 1 and llm._convs[0]["cache"] is llm._cache \
            and old is not llm._cache, "the old conversation was kept"
        print(f"   a new query: taken up at the query's mark ({q[0]} ids, cache vs "
              f"one prefill {gap:.2g}); a request extending nothing: the "
              f"{len(front)}-id front, the old conversation let go")
    finally:
        spec_draft.plain = real


def check_mtp(ref):
    """The MTP round through the same loop, reuse on and off."""
    outs, worst = [], 0.0
    for reuse in (True, False):
        llm = toy_llm("slice", max_tokens=10)
        llm.reuse = reuse
        llm.decode_policy_free = AlwaysMTP()
        vocab = llm.model.args.vocab_size
        llm._spec = ScriptedSpeculator(llm.model, ref[0], PATTERN, vocab=vocab)

        def check(llm, prev, st, k):
            nonlocal worst
            assert "mtp_rounds" in st, "the MTP round did not run"
            if reuse:
                if k:
                    took_up(llm, prev, st)
                worst = max(worst, fresh_gap(llm))
            if k + 1 < len(ref):
                llm._spec = ScriptedSpeculator(llm.model, ref[k + 1], PATTERN, vocab=vocab)
        out, _ = agent_loop(llm, check=check)
        outs.append([g for g, _ in out])
    assert outs[0] == outs[1], "reuse changed an MTP reply"
    assert worst < TOL, worst
    same = sum(a == b for a, b in zip(outs[0], ref))
    print(f"4. MTP round (scripted accepts, rejections, a replay): same tokens with "
          f"and without reuse, cache vs one prefill {worst:.2g}; {same}/{len(ref)} "
          f"replies also plain decode's")


def check_dense(exact):
    import test_batching as tb
    from mlx_lm.utils import load_tokenizer
    tok = load_tokenizer(support.tokenizer_dir())
    vocab = max([tok.vocab_size] + list(tok.get_added_vocab().values())) + 1
    target, _ = tb.toys(vocab)
    on, off = tb.engine(target, tok), tb.engine(target, tok)
    on.reuse = True
    worst = 0.0

    def check(llm, prev, st, k):
        nonlocal worst
        if k:
            took_up(llm, prev, st, dense=True)
        worst = max(worst, fresh_gap(llm))
    got, _ = agent_loop(on, max_tokens=12, check=check)
    ref, _ = agent_loop(off, max_tokens=12)
    assert [g for g, _ in got] == [g for g, _ in ref], "reuse changed a dense reply"
    if exact:
        assert worst < TOL, worst
    print(f"6. dense, {'CPU' if exact else 'GPU'}: same tokens with and without reuse, "
          f"the KV cut back at the common prefix, fed "
          f"{[st['prompt_tokens'] for _, st in got]}; cache vs one prefill {worst:.2g}")
    if not exact:
        return

    # batching with reuse on: an extending request alone, then four at once
    h, seen = tb.host(tb.engine(target, tok))
    h.llm.reuse = True
    msgs = [{"role": "user", "content": "Name three rivers."}]
    first = h.generate(msgs, 9, stream=False)
    more = msgs + [{"role": "assistant", "content": first},
                   {"role": "user", "content": "And three lakes."}]
    solo = tb.engine(target, tok)
    solo.ask(messages=msgs, max_tokens=9)
    want = solo.ask(messages=more, max_tokens=9)
    assert h.generate(more, 9, stream=False) == want
    assert h.llm.stats()["cached_tokens"] > 0, h.llm.stats()
    refs = [solo.ask(messages=m, max_tokens=n) for m, (_, n) in zip(tb.MSGS, tb.ASKS)]
    assert tb.at_once(h, range(4), False) == refs
    assert seen["width"] == 4, seen
    h.executor.shutdown()
    print("   batching with reuse on: an extending request took the last one's "
          "cache up, its reply the solo one; four at once, each its solo reply")


def fake_budget(peak, room_ok=None):
    """MemBudget as _keep_room reads it: the room beside the pool is `peak`
    less what MLX holds, a mark 1 MB. `room_ok`: how the last process of the
    setup ended its last turn (the calibration's, _KEEP_FIRST)."""
    return types.SimpleNamespace(reset=lambda: None, peak=peak, report=lambda: {},
                                 _need=lambda n, fresh, prefill: 0.0, min_w=128,
                                 stores={}, cache=0.0, _host_top=0.0, ring_bytes=0.0,
                                 state=1e6, room_ok=room_ok, _save_calibration=lambda: None)


def check_budget(ref):
    # MOE_KEEP_FIRST=0 first (main's rule before it), then as by default below.
    kf, stream_fast._KEEP_FIRST = stream_fast._KEEP_FIRST, False
    try:
        check_budget_first(ref)
    finally:
        stream_fast._KEEP_FIRST = kf


def check_budget_first(ref):
    # No room (a 2 GB client): nothing kept, no pass split at a mark, step 2
    # takes up only the stored front.
    llm = toy_llm("slice", max_tokens=10)
    llm.reuse = True
    llm._budget = fake_budget(-1.0)
    out, _ = agent_loop(llm, steps=2)
    front = llm._front_ids(llm._seen)
    assert not llm._convs and not llm._room_ok, "a conversation was kept with no room"
    assert front and out[1][1]["cached_tokens"] == len(front), (out[1][1], len(front))
    # The front's own prefill on step 1, nothing on step 2: no split at a mark.
    assert [st["prefill_pieces"] for _, st in out] == [[len(front)], []], out
    assert all(st["keep_room_mb"] < 0 for _, st in out), out
    assert [g for g, _ in out] == ref[:2], "the budget changed a reply"
    # Room: the first turn of a process has no measurement yet and keeps
    # nothing; from then on kept and taken up as without a budget. The room
    # runs out after step 3: step 4 still takes step 3 up and keeps nothing,
    # step 5 goes to the front. Every reply the one without reuse.
    llm = toy_llm("slice", max_tokens=10)
    llm.reuse = True
    llm._budget = fake_budget(1e15)
    ways = []

    def check(llm, prev, st, k):
        if k in (2, 3):
            ways.append(took_up(llm, prev, st))
        if k == 2:
            llm._budget.peak = -1.0
    out, _ = agent_loop(llm, check=check)
    cached = [st["cached_tokens"] for _, st in out]
    assert [g for g, _ in out] == ref, "the budget changed a reply"
    assert cached[1] == cached[4] == len(front), (cached, len(front))
    assert not llm._convs and not llm._room_ok
    print(f"7. under --mem-peak-gb: with no room nothing kept and no mark, step 2 took "
          f"only the stored front ({len(front)} ids); with room kept from the turn after "
          f"the first (taken up at {', '.join(ways)}), dropped when the room ran out; "
          f"cached {cached}, every reply the one without reuse")
    # MOE_KEEP_FIRST (the default): a first turn goes by the last process's last turn, so after
    # one that ended without room (a 2 GB client) there is no mark, no split at one and the
    # verdict stays, while with room the first turn is kept too, step 2 takes it up, and a turn
    # that ends without room says so for the next process.
    stream_fast._KEEP_FIRST = True
    try:
        for last in (None, False):
            llm = toy_llm("slice", max_tokens=10)
            llm.reuse = True
            llm._budget = fake_budget(-1.0, room_ok=last)
            out, _ = agent_loop(llm, steps=2)
            assert not llm._convs and not llm._room_ok, "kept with no room"
            assert [st["prefill_pieces"] for _, st in out] == [[len(front)], []], out
            assert [g for g, _ in out] == ref[:2], "the budget changed a reply"
            assert llm._budget.room_ok is False, llm._budget.room_ok
        llm = toy_llm("slice", max_tokens=10)
        llm.reuse = True
        llm._budget = fake_budget(1e15, room_ok=True)
        ways = []
        out, _ = agent_loop(llm, steps=3, check=lambda llm, prev, st, k:
                            k and ways.append(took_up(llm, prev, st)))
        cached = [st["cached_tokens"] for _, st in out]
        assert [g for g, _ in out] == ref[:3], "keeping the first turn changed a reply"
        assert len(ways) == 2 and cached[1] > len(front), (ways, cached, len(front))
        assert llm._budget.room_ok is True
    finally:
        stream_fast._KEEP_FIRST = False
    print(f"   MOE_KEEP_FIRST: no room as above; with room the first turn kept, step 2 "
          f"taken up at {ways[0]} (cached {cached}), every reply the one without reuse")


QUERY_B = "Check how a damaged front file is handled and say what you would change."
TITLE = ("Write a title of at most six words for the conversation the user "
         "starts. Reply with the title only.")


def same(convs, want):
    """The kept list is exactly these conversations, by identity."""
    return len(convs) == len(want) and all(c is w for c, w in zip(convs, want))


def plan_loop(llm, plan, max_tokens=10, check=None):
    """Requests of several conversations in the order `plan` names them: A
    is agent_loop's conversation, B a second agent on the same system prompt
    and tools (a subagent), T a title request -- its own system prompt and
    A's query, no tools, the same every time. Per request (reply ids,
    stats); `check` also gets the conversation the name's last request left
    (None before one) and the kept list before this request."""
    msgs = {"A": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": QUERY}],
            "B": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": QUERY_B}],
            "T": [{"role": "system", "content": TITLE}, {"role": "user", "content": QUERY}]}
    left, out = {}, []
    for k, name in enumerate(plan):
        before = llm._convs
        reply = llm.ask(messages=msgs[name], tools=None if name == "T" else TOOLS,
                        max_tokens=max_tokens)
        st = llm.stats()
        out.append((llm._seen[len(llm._seen) - st["tokens"]:], st))
        if check:
            check(llm, left.get(name), before, st)
        left[name] = llm._convs[-1]
        if name != "T":
            msgs[name] = msgs[name] + step_msgs(reply, k)
    return out


def check_several():
    """Two agents alternating, and an agent with a title request now and
    then: with two kept, each request takes up its own conversation and the
    other stays kept. The replies are those of one kept conversation, where
    every switch goes back to the stored front."""
    for plan in ("ABABAB", "ATAATA"):
        one, two = toy_llm("slice", max_tokens=10), toy_llm("slice", max_tokens=10)
        one.reuse = two.reuse = True
        two.keep_convs = 2
        ways, worst = [], 0.0

        def check(llm, prev, before, st):
            nonlocal worst
            worst = max(worst, fresh_gap(llm))
            if prev is not None:
                ways.append(took_up(llm, prev, st))
            assert same(llm._convs[:-1], [c for c in before if c is not prev]), \
                "not its own taken up, or another one dropped"
        got = plan_loop(two, plan, check=check)
        ref = plan_loop(one, plan)
        assert [g for g, _ in got] == [g for g, _ in ref], f"two kept changed a reply ({plan})"
        assert worst < TOL, worst
        fed = [(st["prompt_tokens"], r["prompt_tokens"]) for (_, st), (_, r) in zip(got, ref)]
        print(f"8. {plan}, two kept: each request took its own conversation up "
              f"({', '.join(ways)}), the other kept; replies those of one kept; fed "
              f"{' '.join(f'{a}/{b}' for a, b in fed)} (two/one kept); cache vs one "
              f"prefill {worst:.2g}")


def check_evict():
    """Bounded by count and by bytes, least recently used first, the newest
    never; the bytes count the conversation a request is about to grow, at
    the KV its prompt will hold. A call that keeps nothing drops only the
    conversation whose cache it moves on, and the others are taken up as
    they were."""
    llm = toy_llm("slice", max_tokens=6)
    llm.reuse, llm.keep_convs = True, 2

    def ask(query):
        llm.ask(messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": query}], tools=TOOLS)
        return llm._convs[-1]
    ask(QUERY)
    b, c = ask(QUERY_B), ask("List the test files.")
    assert same(llm._convs, [b, c]), "count: not the least recently used one out"
    size = llm._conv_bytes
    # b and c fit g together, not with the next request's KV beside them
    # (counted at c's KV a position): the older goes before its prefill.
    g = size(b) + size(c)
    llm.keep_convs, llm.keep_gb = 4, g / 1e9
    d = ask("Which file holds the arena?")
    st = llm.stats()
    assert same(llm._convs, [c, d]), "bytes: not the least recently used one out"
    assert st["kept_convs"] == 2 and st["kept_mb"] == [
        round(size(c) / 1e6, 1), round(size(d) / 1e6, 1)], st
    llm.keep_gb = 0
    e = ask("Where is the router?")
    assert same(llm._convs, [e]), "the newest has to stay, whatever its size"
    assert size(c) + size(d) <= g, (size(c), size(d), g)
    print(f"9. evicted by count (2 kept, the oldest of 3 out) and by bytes "
          f"({size(b) / 1e3:.0f} + {size(c) / 1e3:.0f} KB held fit {g / 1e3:.0f} "
          f"KB, not with the next request's KV to come: the older out, "
          f"{size(c) / 1e3:.0f} + {size(d) / 1e3:.0f} KB kept); at 0 GB the "
          f"newest still kept ({size(e) / 1e3:.0f} KB)")

    llm.keep_convs, llm.keep_gb = 3, 2.0
    f = ask(QUERY_B)
    assert same(llm._convs, [e, f])
    llm.ask(message="And the tests?")        # moves f's cache on, keeps nothing
    assert same(llm._convs, [e]), "a call that keeps nothing dropped the wrong ones"
    want = max(n for n, _, _ in e["marks"])
    ask("Where is the router?")             # e's request again: at its newest mark
    assert llm.stats()["cached_tokens"] == want, (llm.stats()["cached_tokens"], want)
    gap = fresh_gap(llm)
    assert gap < TOL, gap
    print(f"   a call that keeps nothing dropped only the conversation it "
          f"continued; the other was taken up at its mark ({want} ids), cache "
          f"vs one prefill {gap:.2g}")


def check_dense_several():
    """Dense, CPU: a released draft takes its half out of every kept
    conversation; batch newcomers take their own kept conversation up and
    leave the others kept."""
    import test_batching as tb
    from mlx_lm.utils import load_tokenizer
    gen = sys.modules["mlx_lm.generate"]
    spec_draft.generation_stream = gen.generation_stream
    spec_draft.wired_limit = gen.wired_limit
    tok = load_tokenizer(support.tokenizer_dir())
    vocab = max([tok.vocab_size] + list(tok.get_added_vocab().values())) + 1
    target, draft = tb.toys(vocab)
    solo = tb.engine(target, tok)
    a1 = [{"role": "user", "content": "Name three rivers."}]
    b1 = [{"role": "user", "content": "Count to five."}]
    c1 = [{"role": "user", "content": "What is MoE routing?"}]

    def more(msgs, reply, ask):
        return msgs + [{"role": "assistant", "content": reply},
                       {"role": "user", "content": ask}]

    llm = tb.engine(target, tok, draft, auto=True)
    llm.reuse, llm.keep_convs = True, 3
    ra, rb = (llm.ask(messages=m, max_tokens=9) for m in (a1, b1))
    assert all(c["draft"][0] is not None for c in llm._convs)
    llm.decode_policy_free = types.SimpleNamespace(
        outclassed=lambda arm: True, ms={"draft": [9.0], "plain": [5.0]})
    llm._drop_losing_draft()
    assert llm._draft_model is None
    assert all(c["draft"] == (None, ()) for c in llm._convs), "a draft half was kept"
    a2 = more(a1, ra, "And three lakes.")
    assert llm.ask(messages=a2, max_tokens=9) == solo.ask(messages=a2, max_tokens=9)
    assert llm.stats()["cached_tokens"] > 0, llm.stats()
    print("10. dense, a draft released: no kept conversation holds its half; one "
          "taken up after it, its reply the solo one")

    h, seen = tb.host(tb.engine(target, tok))
    h.llm.reuse, h.llm.keep_convs = True, 3
    ra, rb = (h.generate(m, 9, stream=False) for m in (a1, b1))
    _, b = h.llm._convs
    a2, b2 = more(a1, ra, "And three lakes."), more(b1, rb, "And up to ten.")
    gate = threading.Event()
    h.executor.submit(gate.wait)
    try:                # c1 starts alone, a2 comes in as a newcomer
        pieces = [h.generate(m, 9, stream=True) for m in (c1, a2)]
    finally:
        gate.set()
    got = ["".join(p) for p in pieces]
    assert seen["row"] == 1 and seen["width"] == 2, seen
    assert got == [solo.ask(messages=m, max_tokens=9) for m in (c1, a2)]
    assert same(h.llm._convs, [b]), "the newcomer left its own, or dropped another"
    assert h.generate(b2, 9, stream=False) == solo.ask(messages=b2, max_tokens=9)
    assert h.llm.stats()["cached_tokens"] > 0 and all(c is not b for c in h.llm._convs)
    h.executor.shutdown()
    print("11. batching, three kept: a newcomer took its own conversation up and "
          "left the other; the batch kept nothing; the other taken up later; "
          "every reply the solo one")


def cpu():
    """Everything from here on the CPU, mlx_lm's generation stream included
    (it is made on the GPU at import) and its wired limit off (a Metal-only
    setting that reads a key the CPU's device_info has not got)."""
    gen = sys.modules["mlx_lm.generate"]
    mx.set_default_device(mx.cpu)
    gen.generation_stream = mx.new_thread_local_stream(mx.cpu)
    gen.wired_limit = lambda *a, **kw: contextlib.nullcontext()


def main():
    import shutil
    faulthandler.dump_traceback_later(270, exit=True)
    if not support.has_tokenizer():
        print("SKIPPED (no tokenizer: set MOE_TOY_TOKENIZER)")
        return
    pcache.FRONTS = Path(tempfile.mkdtemp(prefix="moe-reuse-test-"))
    # The grammar's forced rows would have the decode arm picked by timing,
    # differently in the two engines of a comparison: no forcer, a fixed arm.
    real = moe_stream.forced_rows.make
    moe_stream.forced_rows.make = lambda *a, **kw: None
    try:
        hybrid_loop("GPU", exact=False)
        check_dense(exact=False)
        cpu()
        ref = hybrid_loop("CPU", exact=True)
        check_calls_and_queries()
        check_mtp(ref)
        check_dense(exact=True)
        check_budget(ref)
        check_several()
        check_evict()
        check_dense_several()
    finally:
        moe_stream.forced_rows.make = real
        shutil.rmtree(pcache.FRONTS, ignore_errors=True)
    print("test_reuse: ok")


if __name__ == "__main__":
    main()
