#!/usr/bin/env python3
"""Tools end to end: the model is offered functions, calls one, gets a result.

The one number worth watching is the second turn's `cached`. Qwen3.5 renders
the tool block ahead of even the system message, so an unchanged tool list is
prefilled once and copied after that -- which is what makes a large tool list
affordable per request rather than per turn. The last check is the other half
of the same fact: a `--memory` request is a continuation and has no front, so
the block has to be put back by hand or the model never sees the functions.

Whether the model calls the function at all is not fully repeatable, and a
failure here is not by itself a regression: an earlier probe ran the same
first turn six times in one process and got a call 2 of 3 times with the tool
block cached and 2 of 3 without it -- the misses were an image placeholder and
a malformed <tool_call>. Routing is biased towards whichever experts are
resident, so the arena's state going into a turn is part of the input. Re-run
before concluding anything; the cached/prompt numbers below are repeatable and
are what this file is really for.

  python3 tests/test_tools_e2e.py CHECKPOINT_DIR   # or MOE_TEST_MODEL; skips without one
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tools as toolfmt
from moe_stream import StreamingLLM
import support

TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string", "description": "City name"},
            "days": {"type": "integer", "description": "Forecast days"}},
            "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "send_email",
        "description": "Send an email.",
        "parameters": {"type": "object", "properties": {
            "to": {"type": "string"}, "subject": {"type": "string"},
            "body": {"type": "string"}}, "required": ["to", "body"]}}},
]


def main():
    model = support.model_dir("test_tools_e2e")
    if model is None:
        return
    llm = StreamingLLM(model=model, max_tokens=200)

    t = time.perf_counter()
    out = llm.ask(messages=[{"role": "user",
                             "content": "What is the weather in Berlin?"}],
                  tools=TOOLS)
    first_s = time.perf_counter() - t
    s1 = llm.stats()
    content, calls = toolfmt.parse(out, TOOLS)
    print(f"turn 1: {s1['prompt_tokens']} prompt / {s1['cached_tokens']} cached"
          f" | {first_s:.2f}s | ttft {s1['ttft_s']}s")
    print(f"  content: {content[:80]!r}")
    print(f"  calls  : {calls}")
    assert calls, f"the model was offered tools and called none: {out[:300]!r}"
    assert calls[0]["function"]["name"] == "get_weather", calls
    args = json.loads(calls[0]["function"]["arguments"])
    assert "berlin" in str(args.get("city", "")).lower(), args

    # Hand the result back the way an application would, and check the model
    # answers from it. Same tools, so the block must come out of the cache.
    convo = [{"role": "user", "content": "What is the weather in Berlin?"},
             {"role": "assistant", "content": content or "",
              "tool_calls": calls},
             {"role": "tool", "tool_call_id": calls[0]["id"],
              "content": json.dumps({"temp_c": 7, "sky": "rain"})}]
    t = time.perf_counter()
    out2 = llm.ask(messages=convo, tools=TOOLS)
    second_s = time.perf_counter() - t
    s2 = llm.stats()
    print(f"turn 2: {s2['prompt_tokens']} prompt / {s2['cached_tokens']} cached"
          f" | {second_s:.2f}s | ttft {s2['ttft_s']}s")
    print(f"  reply: {out2.strip()[:120]!r}")
    assert "7" in out2 or "rain" in out2.lower() or "\u0434\u043e\u0436\u0434" in out2.lower(), \
        f"the tool result did not reach the answer: {out2[:200]!r}"

    # The block is prefilled once and copied after that (moe_stream's
    # _build_front), so turn 2 must not pay for it again. Before that existed
    # the reset emptied _seen and the id diff started from zero: measured 0 of
    # 430 tokens cached and 6.07 s to first token on exactly this exchange.
    # `tool_tokens` counts only the JSON of the list -- the rendered block also
    # carries the template's own instructions, so it is a floor, not the size.
    tool_tokens = len(llm.tokenizer.encode(
        json.dumps(TOOLS, ensure_ascii=False)))
    assert s2["cached_tokens"] >= tool_tokens, (
        f"turn 2 cached {s2['cached_tokens']} tokens against a tool list of "
        f"at least {tool_tokens}: the block is being re-prefilled every "
        f"request ({s2['prompt_tokens']} prompt, ttft {s2['ttft_s']}s)")
    print(f"  tool block >= {tool_tokens} tokens, {s2['cached_tokens']} cached "
          f"on turn 2, ttft {s1['ttft_s']}s -> {s2['ttft_s']}s")

    # No tools offered: the prompt must be exactly what it always was, and a
    # conversation that shares no prefix must still be correct -- the cache is
    # kept across `messages` calls, so this is the case that would break if the
    # id diff ever trusted more than it matched.
    plain = llm.ask(messages=[{"role": "user", "content": "Say hi."}])
    assert plain.strip(), "a tool-free request came back empty"
    assert "<tool_call>" not in plain
    assert llm.stats()["cached_tokens"] == 0, (
        "a request with no tools reused the tool block's cache: "
        f"{llm.stats()['cached_tokens']} tokens cached")
    print(f"  no-tools turn still fine: {plain.strip()[:50]!r}")

    # A third turn after an unrelated conversation must still be correct.
    out3 = llm.ask(messages=convo + [
        {"role": "assistant", "content": out2},
        {"role": "user", "content": "And should I take an umbrella?"}],
        tools=TOOLS)
    assert out3.strip(), "third turn came back empty"
    print(f"  turn 3 after an unrelated conversation: {out3.strip()[:60]!r}")

    # Tools behind a memory core (server.py --memory). A core is a cache the
    # request CONTINUES instead of re-rendering, and the template writes the
    # tool block at the very front of a render -- so the marker split in
    # _continue_chunk used to drop it and the model was never shown the
    # functions at all: finish_reason "stop" and an invented forecast, where
    # the same request without --memory answered with a call. One turn is
    # prefilled here to stand in for kvmem's core; what matters is that the
    # request arrives as a continuation of a cache, which is the shape that
    # broke.
    llm.ask(messages=[{"role": "user", "content": "My name is Ilya."}])
    core = {"cache": llm._cache, "seen": list(llm._seen),
            "history": [dict(m) for m in llm.history]}
    out4 = llm.ask(messages=[{"role": "user",
                              "content": "What is the weather in Berlin?"}],
                   tools=TOOLS, resume=core)
    s4 = llm.stats()
    _, calls4 = toolfmt.parse(out4, TOOLS)
    print(f"  behind a core: {s4['prompt_tokens']} prompt / "
          f"{s4['cached_tokens']} cached | calls {calls4}")
    assert calls4, ("behind a memory core the model was offered tools and "
                    f"called none: {out4[:300]!r}")
    assert calls4[0]["function"]["name"] == "get_weather", calls4
    assert "berlin" in str(json.loads(
        calls4[0]["function"]["arguments"]).get("city", "")).lower(), calls4

    print("tools end to end: called, answered from the result, block cached, "
          "and still offered behind a memory core")


if __name__ == "__main__":
    main()
