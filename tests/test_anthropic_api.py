#!/usr/bin/env python3
"""server.py's Anthropic Messages API with no model: /v1/messages and /v1/messages/count_tokens through the real handler.

    perl -e 'alarm shift; exec @ARGV' 280 python3 tests/test_anthropic_api.py

The stub engine of tests/test_webchat.py sits behind the real ModelHost and server. Checked: a whole reply
and its usage, the exact stream events (and pings while the prompt is read), a system prompt as a string or
blocks, a tool_use turn and its tool_result follow-up, thinking on and off (whole and streamed), stop
sequences, count_tokens through the model's own template, an image block, tool_choice, sampling, the error
shapes, the ignored headers and the Origin guard, and that the reply text equals the OpenAI route's for the
same conversation.
"""

import base64
import http.client
import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import anthropic_api                                               # noqa: E402
import server                                                      # noqa: E402
import test_webchat as stub                                        # noqa: E402

FakeLLM = stub._FakeLLM
REPLY = FakeLLM.reply
CALL = stub.CALL


class Tok:
    """A tokenizer with a chat template that shows its inputs: one token per word of the rendering."""
    detokenizer = None
    calls = []

    def apply_chat_template(self, messages, tools=None, enable_thinking=False,
                            add_generation_prompt=True, tokenize=False):
        Tok.calls.append({"messages": messages, "tools": tools, "enable_thinking": enable_thinking})
        text = " ".join(f"<{m['role']}> {m.get('content') or ''}" for m in messages)
        return text + (" TOOLS " + json.dumps(tools) if tools else "") + " <assistant>"

    def encode(self, text):
        return text.split()


def main():
    real = server.StreamingLLM
    real_can_batch = server.ModelHost._can_batch
    real_tok = stub._FakeTokenizer
    server.StreamingLLM = FakeLLM
    stub._FakeTokenizer = Tok
    server.ModelHost._can_batch = staticmethod(lambda llm: False)
    host = server.ModelHost("/models/tiny-chat", shadow=False)
    httpd = server.serve(host, 0)
    port = httpd.socket.getsockname()[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    def call(path, body=None, headers=None, method="POST", raw=None):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        payload = raw if raw is not None else json.dumps(body).encode()
        conn.request(method, path, payload, dict({"Content-Type": "application/json"}, **(headers or {})))
        r = conn.getresponse()
        data = r.read()
        conn.close()
        return r.status, r, data

    def msg(body, **kw):
        status, r, data = call("/v1/messages", dict({"model": "claude-x", "max_tokens": 100}, **body), **kw)
        return status, json.loads(data)

    def chat(body):
        return json.loads(call("/v1/chat/completions", body)[2])

    def events(body):
        """The events of a streamed reply, each checked to carry its own name as `type`."""
        status, r, data = call("/v1/messages", dict({"model": "claude-x", "max_tokens": 100,
                                                     "stream": True}, **body))
        assert status == 200 and r.getheader("Content-Type") == "text/event-stream", (status, data[:200])
        out = []
        for chunk in data.decode().split("\n\n"):
            if chunk:
                head, payload = chunk.split("\n")
                assert head.startswith("event: ") and payload.startswith("data: "), chunk
                ev = json.loads(payload[6:])
                assert ev["type"] == head[7:], chunk
                out.append(ev)
        return out

    def err(status_want, kind, body, path="/v1/messages", **kw):
        status, _, data = call(path, body, **kw)
        got = json.loads(data)
        assert status == status_want, (status, got)
        assert got["type"] == "error" and got["error"]["type"] == kind \
            and isinstance(got["error"]["message"], str), got
        return got["error"]["message"]

    def names(evs):
        return [e["type"] for e in evs]

    hi = [{"role": "user", "content": "hi"}]
    weather = [{"name": "get_weather", "description": "Weather in a city",
                "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}}]
    weather_oai = [{"type": "function", "function": {
        "name": "get_weather", "description": "Weather in a city",
        "parameters": weather[0]["input_schema"]}}]
    try:
        # a whole reply
        status, out = msg({"messages": hi, "system": "Be brief."})
        assert status == 200, out
        assert out["id"].startswith("msg_") and out["type"] == "message" and out["role"] == "assistant"
        assert out["model"] == "claude-x" and out["content"] == [{"type": "text", "text": REPLY}], out
        assert out["stop_reason"] == "end_turn" and out["stop_sequence"] is None
        # 8 prompt tokens, 3 of them from the kept cache, are 5 new and 3 read.
        assert out["usage"] == {"input_tokens": 5, "output_tokens": 5,
                                "cache_creation_input_tokens": 0, "cache_read_input_tokens": 3}, out["usage"]
        sent = FakeLLM.asked[-1]
        assert sent["messages"] == [{"role": "system", "content": "Be brief."},
                                    {"role": "user", "content": "hi"}], sent
        assert sent["max_tokens"] == 100 and sent["tools"] is None and sent["thinking"] is False
        # The same conversation through the OpenAI route: the engine is asked the same, the text is equal.
        oai = chat({"messages": [{"role": "system", "content": "Be brief."}] + hi, "max_tokens": 100})
        assert FakeLLM.asked[-1] == sent, (FakeLLM.asked[-1], sent)
        assert oai["choices"][0]["message"]["content"] == out["content"][0]["text"]
        assert oai["usage"]["completion_tokens"] == out["usage"]["output_tokens"]
        assert msg({"messages": hi})[1]["id"] != out["id"]
        print("  whole reply: shape, usage with cache_read, engine request and text equal to the OpenAI route")

        # system and content forms
        msg({"messages": [{"role": "user", "content": [{"type": "text", "text": "a"},
                                                       {"type": "text", "text": "b"}]}],
             "system": [{"type": "text", "text": "One.", "cache_control": {"type": "ephemeral"}},
                        {"type": "text", "text": "Two."}]})
        assert FakeLLM.asked[-1]["messages"] == [{"role": "system", "content": "One.\nTwo."},
                                                 {"role": "user", "content": "a\nb"}]
        msg({"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"},
                          {"role": "system", "content": "Reminder."}, {"role": "user", "content": "q2"}]})
        assert [m["role"] for m in FakeLLM.asked[-1]["messages"]] == ["user", "assistant", "user", "user"]
        assert FakeLLM.asked[-1]["messages"][2]["content"] == "<system-reminder>\nReminder.\n</system-reminder>"
        # Claude Code's environment block comes as a system entry after the task, before any reply.
        msg({"system": "Rules.", "messages": [{"role": "user", "content": "task"},
                                              {"role": "system", "content": "Env."}]})
        assert FakeLLM.asked[-1]["messages"] == [{"role": "system", "content": "Rules.\n\nEnv."},
                                                 {"role": "user", "content": "task"}]
        assert anthropic_api.to_tools({"tools": weather + [dict(weather[0], name="Placeholder", defer_loading=True)]}) \
            == anthropic_api.to_tools({"tools": weather})
        print("  system as a string or blocks, text blocks joined, a system entry before the first reply joins the system "
              "prompt and a later one is a marked user turn, deferred tools left out")

        # limits and stop sequences
        status, out = msg({"messages": hi, "max_tokens": 3})
        assert out["stop_reason"] == "max_tokens" and out["usage"]["output_tokens"] == 3, out
        status, out = msg({"messages": hi, "stop_sequences": ["wor"]})
        assert out["content"] == [{"type": "text", "text": "Hello **"}], out
        assert out["stop_reason"] == "stop_sequence" and out["stop_sequence"] == "wor", out
        status, out = msg({"messages": hi, "stop_sequences": ["ld** fr"]})     # across two pieces
        assert out["content"][0]["text"] == "Hello **wor" and out["stop_sequence"] == "ld** fr", out
        status, out = msg({"messages": hi, "stop_sequences": ["never"]})
        assert out["content"][0]["text"] == REPLY and out["stop_reason"] == "end_turn"
        FakeLLM.delay = 0.05        # slow enough that the turn is still running when the stop sequence is found
        try:
            evs = events({"messages": hi, "stop_sequences": ["wor"]})
        finally:
            FakeLLM.delay = 0.0
        assert "".join(e["delta"]["text"] for e in evs if e["type"] == "content_block_delta") == "Hello **"
        assert evs[-2]["delta"] == {"stop_reason": "stop_sequence", "stop_sequence": "wor"}, evs[-2]
        assert evs[-2]["usage"]["output_tokens"] == 2, "the turn was read past the stop sequence"
        print("  max_tokens, stop_sequences (cut inside and across pieces, stream ends early)")

        # the stream
        evs = events({"messages": hi})
        assert names(evs) == ["message_start", "content_block_start"] + ["content_block_delta"] * 5 + [
            "content_block_stop", "message_delta", "message_stop"], names(evs)
        assert evs[0]["message"] == {
            "id": evs[0]["message"]["id"], "type": "message", "role": "assistant", "model": "claude-x",
            "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 0,
                      "cache_read_input_tokens": 0}}, evs[0]
        assert evs[0]["message"]["id"].startswith("msg_")
        assert evs[1] == {"type": "content_block_start", "index": 0,
                          "content_block": {"type": "text", "text": ""}}
        assert all(e["index"] == 0 and e["delta"]["type"] == "text_delta" for e in evs[2:7])
        assert "".join(e["delta"]["text"] for e in evs[2:7]) == REPLY
        assert evs[7] == {"type": "content_block_stop", "index": 0}
        assert evs[8]["delta"] == {"stop_reason": "end_turn", "stop_sequence": None}
        assert evs[8]["usage"] == {"input_tokens": 5, "output_tokens": 5,
                                   "cache_creation_input_tokens": 0, "cache_read_input_tokens": 3}
        assert evs[9] == {"type": "message_stop"}
        assert events({"messages": hi, "max_tokens": 2})[-2]["delta"]["stop_reason"] == "max_tokens"
        # While the prompt is read nothing arrives but pings, and they come before the first block.
        old_ping, old_delay = anthropic_api.PING_S, FakeLLM.delay
        anthropic_api.PING_S, FakeLLM.delay = 0.05, 0.3
        try:
            evs = events({"messages": hi, "max_tokens": 2})
        finally:
            anthropic_api.PING_S, FakeLLM.delay = old_ping, old_delay
        kinds = names(evs)
        assert kinds[0] == "message_start" and kinds[1] == "ping" and kinds.count("ping") >= 3, kinds
        assert [k for k in kinds if k != "ping"] == ["message_start", "content_block_start"] + [
            "content_block_delta"] * 2 + ["content_block_stop", "message_delta", "message_stop"]
        print("  stream: exact event order, usage in message_delta, pings before the first block")

        # tools
        FakeLLM.reply = "Checking. " + CALL
        status, out = msg({"messages": hi, "tools": weather})
        assert FakeLLM.asked[-1]["tools"] == weather_oai, FakeLLM.asked[-1]["tools"]
        text, use = out["content"]
        assert text == {"type": "text", "text": "Checking."}, out
        assert use["type"] == "tool_use" and use["id"].startswith("toolu_") and use["name"] == "get_weather"
        assert use["input"] == {"city": "Berlin"} and out["stop_reason"] == "tool_use", out
        oai = chat({"messages": hi, "tools": weather_oai})["choices"][0]["message"]
        assert oai["content"] == text["text"]
        assert json.loads(oai["tool_calls"][0]["function"]["arguments"]) == use["input"]
        assert msg({"messages": hi, "tools": weather})[1]["content"][1]["id"] != use["id"]
        evs = events({"messages": hi, "tools": weather})
        assert names(evs) == ["message_start", "content_block_start", "content_block_delta",
                              "content_block_stop", "content_block_start", "content_block_delta",
                              "content_block_stop", "message_delta", "message_stop"], names(evs)
        assert evs[1]["content_block"] == {"type": "text", "text": ""} and evs[2]["delta"]["text"] == "Checking."
        start = evs[4]
        assert start["index"] == 1 and start["content_block"]["type"] == "tool_use"
        assert start["content_block"]["name"] == "get_weather" and start["content_block"]["input"] == {}
        assert evs[5]["delta"]["type"] == "input_json_delta" and evs[5]["index"] == 1
        assert json.loads(evs[5]["delta"]["partial_json"]) == {"city": "Berlin"}
        assert evs[7]["delta"]["stop_reason"] == "tool_use"
        # A schema type is applied to what the model wrote.
        FakeLLM.reply = CALL.replace("Berlin", "3").replace("city", "days")
        schema = [{"name": "get_weather", "input_schema": {"type": "object",
                                                           "properties": {"days": {"type": "integer"}}}}]
        assert msg({"messages": hi, "tools": schema})[1]["content"][0]["input"] == {"days": 3}
        FakeLLM.reply = "Checking. " + CALL

        # The follow-up: assistant tool_use and the user's tool_result become the engine's tool turn.
        use_id = use["id"]
        convo = [{"role": "user", "content": "weather?"},
                 {"role": "assistant", "content": [
                     {"type": "thinking", "thinking": "hmm", "signature": "s"},
                     {"type": "text", "text": "Checking."},
                     {"type": "tool_use", "id": use_id, "name": "get_weather", "input": {"city": "Berlin"}}]},
                 {"role": "user", "content": [
                     {"type": "tool_result", "tool_use_id": use_id, "content": "Sunny"}]}]
        FakeLLM.reply = "It is sunny."
        status, out = msg({"messages": convo, "tools": weather})
        assert out["content"] == [{"type": "text", "text": "It is sunny."}] and out["stop_reason"] == "end_turn"
        got = FakeLLM.asked[-1]["messages"]
        assert got == [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": "Checking.", "reasoning_content": "hmm", "tool_calls": [
                {"id": use_id, "type": "function",
                 "function": {"name": "get_weather", "arguments": {"city": "Berlin"}}}]},
            {"role": "tool", "tool_call_id": use_id, "name": "get_weather", "content": "Sunny"}], got
        # Block content, a note after the result, two results in a row, and tool_choice.
        convo[2]["content"] = [
            {"type": "tool_result", "tool_use_id": use_id, "is_error": True,
             "content": [{"type": "text", "text": "line 1"}, {"type": "text", "text": "line 2"}]},
            {"type": "tool_result", "tool_use_id": "other", "content": "second"},
            {"type": "text", "text": "and then?"}]
        msg({"messages": convo, "tools": weather})
        got = FakeLLM.asked[-1]["messages"][2:]
        assert got == [{"role": "tool", "tool_call_id": use_id, "name": "get_weather", "content": "line 1\nline 2"},
                       {"role": "tool", "tool_call_id": "other", "name": None, "content": "second"},
                       {"role": "user", "content": "and then?"}], got
        for choice, listed in (({"type": "auto"}, True), ({"type": "any"}, True),
                               ({"type": "tool", "name": "get_weather"}, True), ({"type": "none"}, False)):
            msg({"messages": hi, "tools": weather, "tool_choice": choice})
            assert (FakeLLM.asked[-1]["tools"] is not None) == listed, choice
        # With the tools left out the history still renders: its arguments are a mapping already.
        msg({"messages": convo, "tools": weather, "tool_choice": {"type": "none"}})
        assert FakeLLM.asked[-1]["messages"][1]["tool_calls"][0]["function"]["arguments"] == {"city": "Berlin"}
        FakeLLM.reply = REPLY
        print("  tools: tool_use block and stream, schema types, tool_result follow-up, blocks, tool_choice")

        # thinking
        for thinking in (None, {"type": "disabled"}, {"type": "adaptive"}):
            body = {"messages": hi} if thinking is None else {"messages": hi, "thinking": thinking}
            out = msg(body)[1]
            assert FakeLLM.asked[-1]["thinking"] is False and FakeLLM.asked[-1]["budget"] is None, thinking
            assert out["content"] == [{"type": "text", "text": REPLY}], out
        status, out = msg({"messages": hi, "thinking": {"type": "enabled", "budget_tokens": 2000}})
        assert FakeLLM.asked[-1]["thinking"] is True and FakeLLM.asked[-1]["budget"] == 2000
        # budget_tokens and --thinking-budget: the smaller one caps a thinking turn, on both routes.
        host.think_budget = 500
        try:
            for thinking, want in (({"type": "enabled", "budget_tokens": 2000}, 500), ({"type": "enabled"}, 500),
                                   ({"type": "enabled", "budget_tokens": 100}, 100), ({"type": "disabled"}, None)):
                msg({"messages": hi, "thinking": thinking})
                assert FakeLLM.asked[-1]["budget"] == want, (thinking, FakeLLM.asked[-1])
            for think, want in ((True, 500), (False, None)):
                chat({"messages": hi, "chat_template_kwargs": {"enable_thinking": think}})
                assert FakeLLM.asked[-1]["budget"] == want, (think, FakeLLM.asked[-1])
        finally:
            host.think_budget = None
        for bad in (0, -5, "2000", True, 1.5):
            assert "budget_tokens" in err(400, "invalid_request_error", {
                "model": "m", "max_tokens": 5, "messages": hi,
                "thinking": {"type": "enabled", "budget_tokens": bad}}), bad
        assert out["content"] == [{"type": "thinking", "thinking": "Let me see.", "signature": ""},
                                  {"type": "text", "text": "It is **42**."}], out
        # --adaptive-thinking turns Claude Code's adaptive request on, while disabled stays off.
        server.Handler.adaptive_thinking = True
        try:
            msg({"messages": hi, "thinking": {"type": "adaptive"}})
            assert FakeLLM.asked[-1]["thinking"] is True
            msg({"messages": hi, "thinking": {"type": "disabled"}})
            assert FakeLLM.asked[-1]["thinking"] is False
            # Claude Code's effort reaches the template as reasoning_effort, high and max as xhigh.
            for effort, want in (("low", "low"), ("medium", "medium"), ("high", "xhigh"), ("max", "xhigh"), ("odd", True)):
                msg({"messages": hi, "thinking": {"type": "adaptive"}, "output_config": {"effort": effort}})
                assert FakeLLM.asked[-1]["thinking"] == want, (effort, FakeLLM.asked[-1]["thinking"])
            msg({"messages": hi, "thinking": {"type": "disabled"}, "output_config": {"effort": "low"}})
            assert FakeLLM.asked[-1]["thinking"] is False
        finally:
            server.Handler.adaptive_thinking = False
        oai = chat({"messages": hi, "chat_template_kwargs": {"enable_thinking": True}})["choices"][0]["message"]
        assert oai["reasoning_content"] == out["content"][0]["thinking"] and oai["content"] == out["content"][1]["text"]
        out = msg({"messages": hi, "max_tokens": 2, "thinking": {"type": "enabled"}})[1]
        assert out["content"] == [{"type": "thinking", "thinking": "Let me", "signature": ""}], out
        assert out["stop_reason"] == "max_tokens"
        FakeLLM.script = ["Let me", " see.\n</", "think", ">\n", "\nIt is ", "**42**."]
        evs = events({"messages": hi, "thinking": {"type": "enabled"}})
        FakeLLM.script = None
        starts = [e["content_block"] for e in evs if e["type"] == "content_block_start"]
        assert starts == [{"type": "thinking", "thinking": "", "signature": ""}, {"type": "text", "text": ""}], starts
        deltas = [e["delta"] for e in evs if e["type"] == "content_block_delta"]
        assert [d["type"] for d in deltas] == ["thinking_delta"] * 2 + ["text_delta"] * 2, deltas
        assert "".join(d["thinking"] for d in deltas[:2]) == "Let me see.\n"
        assert "".join(d["text"] for d in deltas[2:]) == "It is **42**."
        order = [e["type"] + str(e.get("index", "")) for e in evs if e["type"].startswith("content_block")]
        assert order[:4] == ["content_block_start0", "content_block_delta0", "content_block_delta0",
                             "content_block_stop0"] and order[4] == "content_block_start1", order
        # Thinking and a tool call: the call comes from the answer part.
        FakeLLM.think_reply = "Weather, then.\n</think>\n\nChecking. " + CALL
        out = msg({"messages": hi, "tools": weather, "thinking": {"type": "enabled"}})[1]
        assert [b["type"] for b in out["content"]] == ["thinking", "text", "tool_use"], out
        assert out["content"][0]["thinking"] == "Weather, then." and out["stop_reason"] == "tool_use"
        FakeLLM.think_reply = "Let me see.\n</think>\n\nIt is **42**."
        print("  thinking: on only for type enabled, its budget_tokens or --thinking-budget (the smaller), "
              "thinking block then text, whole and streamed, with tools")

        # count_tokens
        before = len(FakeLLM.asked)
        want_msgs = [{"role": "system", "content": "Be brief."}] + [
            {"role": "user", "content": "hi"}]
        status, _, data = call("/v1/messages/count_tokens", {
            "model": "claude-x", "system": "Be brief.", "messages": hi, "tools": weather})
        assert status == 200
        want = Tok().encode(Tok().apply_chat_template(want_msgs, tools=weather_oai))
        assert json.loads(data) == {"input_tokens": len(want)}, (data, len(want))
        sent = Tok.calls[-1]
        assert sent["messages"] == want_msgs and sent["tools"] == weather_oai and sent["enable_thinking"] is False
        assert len(FakeLLM.asked) == before, "counting started a turn"
        n_plain = json.loads(call("/v1/messages/count_tokens", {"messages": hi})[2])["input_tokens"]
        assert n_plain == 3 and n_plain < len(want), n_plain
        call("/v1/messages/count_tokens", {"messages": hi, "thinking": {"type": "enabled"}})
        assert Tok.calls[-1]["enable_thinking"] is True
        assert json.loads(call("/v1/messages/count_tokens?beta=true", {"messages": hi})[2]) == {"input_tokens": 3}
        err(400, "invalid_request_error", {"system": "x"}, path="/v1/messages/count_tokens")
        print("  count_tokens: the template's own count over system, messages and tools, no turn started")

        # images
        png = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\0" * 56).decode()
        img = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": png}}
        see = [{"role": "user", "content": [img, {"type": "text", "text": "what is this"}]}]
        calls_before = len(FakeLLM.asked)
        assert err(400, "invalid_request_error", {"model": "m", "max_tokens": 5, "messages": see}) == \
            "this model has no vision part; images are not supported"
        assert len(FakeLLM.asked) == calls_before
        FakeLLM.supports_images = True
        try:
            assert msg({"messages": see})[0] == 200
            assert FakeLLM.asked[-1]["messages"][0]["content"] == [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + png}},
                {"type": "text", "text": "what is this"}]
            # An image inside a tool_result follows the tool message as a user turn.
            convo = [{"role": "user", "content": "look"},
                     {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "read", "input": {}}]},
                     {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1",
                                                   "content": [{"type": "text", "text": "file.png"}, img]}]}]
            assert msg({"messages": convo})[0] == 200
            got = FakeLLM.asked[-1]["messages"]
            assert got[2] == {"role": "tool", "tool_call_id": "t1", "name": "read", "content": "file.png"}
            assert got[3]["content"][0]["image_url"]["url"].startswith("data:image/png;base64,")
            bad = {"type": "image", "source": {"type": "url", "url": "https://example.com/cat.png"}}
            assert "does not fetch remote URLs" in err(400, "invalid_request_error", {
                "model": "m", "max_tokens": 5, "messages": [{"role": "user", "content": [bad]}]})
        finally:
            FakeLLM.supports_images = False
        print("  images: 400 without a vision part, base64 blocks become image parts, tool_result images, remote url refused")

        # sampling
        seen, orig_stream = [], FakeLLM.stream     # the stub's ask() drops sampling, so these read the stream

        def spy(self, *a, sampling=None, **kw):
            seen.append(sampling)
            return orig_stream(self, *a, sampling=sampling, **kw)

        FakeLLM.stream = spy
        try:
            events({"messages": hi, "temperature": 0.7, "top_p": 0.9, "top_k": 40})
            assert seen[-1] is None, "sampling counts only under --sampling"
            server.Handler.sampling = True
            events({"messages": hi, "temperature": 0.7, "top_p": 0.9, "top_k": 40})
            s = seen[-1]
            assert (s.temperature, s.top_p, s.top_k) == (0.7, 0.9, 40), s
            events({"messages": hi, "temperature": 0})
            assert seen[-1] is None
            assert "sampling" in err(400, "invalid_request_error", {
                "model": "m", "max_tokens": 5, "messages": hi, "temperature": 0.5, "top_p": 5})
            # --model-sampling: a request that sets none takes the model's, and its own fields still win.
            server.Handler.default_sampling = {"temperature": 1.0, "top_k": 20, "top_p": 0.95}
            events({"messages": hi})
            s = seen[-1]
            assert (s.temperature, s.top_p, s.top_k) == (1.0, 0.95, 20), s
            events({"messages": hi, "temperature": 0})
            assert seen[-1] is None
        finally:
            server.Handler.sampling = False
            server.Handler.default_sampling = {}
            FakeLLM.stream = orig_stream
        print("  sampling: temperature, top_p, top_k reach the engine only under --sampling")

        # errors
        ok = {"model": "m", "max_tokens": 5, "messages": hi}
        cases = [
            (dict(ok, messages=[]), "messages"), ({k: v for k, v in ok.items() if k != "messages"}, "messages"),
            ({k: v for k, v in ok.items() if k != "max_tokens"}, "max_tokens"),
            (dict(ok, max_tokens=0), "max_tokens"), (dict(ok, max_tokens="5"), "max_tokens"),
            (dict(ok, messages=[{"role": "tool", "content": "x"}]), "role"),
            (dict(ok, messages=[{"role": "user", "content": 5}]), "content"),
            (dict(ok, messages=[{"role": "user", "content": [{"type": "document"}]}]), "document"),
            (dict(ok, messages=[{"role": "assistant", "content": [{"type": "image"}]}]), "image"),
            (dict(ok, system=5), "system"), (dict(ok, tools="x"), "tools"),
            (dict(ok, tool_choice={"type": "sometimes"}), "tool_choice"),
            (dict(ok, stop_sequences="x"), "stop_sequences"), (dict(ok, thinking="on"), "thinking"),
        ]
        for body, word in cases:
            assert word in err(400, "invalid_request_error", body), body
        assert "invalid JSON" in err(400, "invalid_request_error", None, raw=b"{nope")
        assert "JSON object" in err(400, "invalid_request_error", [1])
        assert err(400, "invalid_request_error", dict(ok, stream=True, max_tokens=-1))
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        conn.putrequest("POST", "/v1/messages")
        conn.putheader("Content-Length", str(server.MAX_BODY + 1))
        conn.endheaders()
        r = conn.getresponse()
        body413 = json.loads(r.read())
        conn.close()
        assert r.status == 413 and body413["error"]["type"] == "request_too_large", body413
        # The engine failing: a whole reply is a 500, a streamed one an error event after the 200.
        good_stream = FakeLLM.stream

        def boom(self, *a, **kw):
            raise RuntimeError("boom")
            yield

        FakeLLM.stream = boom
        try:
            assert err(500, "api_error", ok) == "generation failed: boom"
            # ModelHost's serial stream path swallows a failing turn, so the failure is raised from generate()'s iterator here.
            host.generate = lambda *a, **kw: boom(None)
            evs = events({"messages": hi})
            del host.generate
            assert names(evs) == ["message_start", "error"], names(evs)
            assert evs[1]["error"] == {"type": "api_error", "message": "generation failed: boom"}, evs[1]
        finally:
            FakeLLM.stream = good_stream
        loaded, host.llm = host.llm, None
        try:
            err(503, "api_error", ok)
            err(503, "api_error", dict(ok, stream=True))
            err(503, "api_error", ok, path="/v1/messages/count_tokens")
        finally:
            host.llm = loaded
        assert msg({"messages": hi})[0] == 200
        print("  errors: 400 per bad field, 413, 500 and a stream error event, 503 with no model, Anthropic shape")

        # headers and the guard
        extra = {"anthropic-version": "2023-06-01", "anthropic-beta": "tools-2024-04-04,x",
                 "x-api-key": "anything", "Authorization": "Bearer anything", "User-Agent": "claude-cli"}
        assert msg({"messages": hi}, headers=extra)[0] == 200
        assert msg({"messages": hi, "metadata": {"user_id": "u"}, "service_tier": "auto"})[0] == 200
        for origin in ("https://example.com", "null"):
            status, _, data = call("/v1/messages", ok, headers={"Origin": origin})
            assert status == 403, (origin, status)
        status, _, data = call("/v1/messages", ok, headers={"Host": "evil.example"})
        assert status == 403
        assert call("/v1/messages", ok, headers={"Origin": f"http://127.0.0.1:{port}"})[0] == 200
        assert call("/v1/messages", ok, headers={"Origin": f"http://localhost:{port}"})[0] == 200
        assert call("/v1/messages", ok)[0] == 200, "a client with no Origin must pass"
        print("  headers: anthropic-version / -beta and any key ignored, the Origin and Host guard applies, no Origin passes")
    finally:
        httpd.shutdown()
        host.executor.shutdown()
        server.StreamingLLM = real
        stub._FakeTokenizer = real_tok
        server.ModelHost._can_batch = real_can_batch
    print("ok")


if __name__ == "__main__":
    main()
