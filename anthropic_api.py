#!/usr/bin/env python3
"""Anthropic's Messages API on server.py's engine path: POST /v1/messages (whole or streamed) and POST /v1/messages/count_tokens."""

import json
import queue
import sys
import threading
import uuid
from types import SimpleNamespace

import tools as toolfmt

PING_S = 5.0    # silence before a ping, under the 10 s Claude Code's byte watchdog counts on
# Claude Code's output_config.effort as the reasoning_effort Qwen's templates take (xhigh, medium, low), mapped as Qwen's own Anthropic endpoint does.
EFFORT = {"low": "low", "medium": "medium", "high": "xhigh", "xhigh": "xhigh", "max": "xhigh"}
STOP_REASONS = {"tool_calls": "tool_use", "length": "max_tokens", "stop": "end_turn"}
ERROR_KINDS = {400: "invalid_request_error", 403: "permission_error", 404: "not_found_error",
               413: "request_too_large", 429: "rate_limit_error"}


class ApiError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status, self.message = status, message


def error_body(status, message):
    return {"type": "error", "error": {"type": ERROR_KINDS.get(status, "api_error"),
                                       "message": message}}


def event(kind, data):
    return f"event: {kind}\ndata: {json.dumps({'type': kind, **data})}\n\n".encode()


# the request, as the OpenAI route would have built it

def blocks_of(content, where):
    """A message's content as a list of block dicts, a string being one text block."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list) and all(isinstance(b, dict) for b in content):
        return content
    raise ApiError(400, f"{where}: content must be a string or a list of blocks")


def image_part(block, where):
    src = block.get("source")
    if not isinstance(src, dict):
        raise ApiError(400, f"{where}: an image block needs a source")
    url = src.get("url") if src.get("type") == "url" else \
        f"data:{src.get('media_type')};base64,{src.get('data')}"
    return {"type": "image_url", "image_url": {"url": url}}


def system_text(system):
    if system is None or isinstance(system, str):
        return system
    if isinstance(system, list) and all(isinstance(b, dict) for b in system):
        return "\n".join(b.get("text") or "" for b in system if b.get("type", "text") == "text")
    raise ApiError(400, "system: must be a string or a list of text blocks")


def assistant_message(blocks, names, where):
    text, calls, reasoning = [], [], []
    for b in blocks:
        kind = b.get("type")
        if kind == "text":
            text.append(b.get("text") or "")
        elif kind == "tool_use":
            names[b.get("id")] = b.get("name")
            # The arguments stay a mapping, so a template reads them even when this request lists no tools.
            calls.append({"id": b.get("id"), "type": "function", "function": {
                "name": b.get("name"), "arguments": b.get("input") or {}}})
        elif kind == "thinking":
            reasoning.append(b.get("thinking") or "")
        elif kind != "redacted_thinking":
            raise ApiError(400, f"{where}: content block type '{kind}' is not supported")
    msg = {"role": "assistant", "content": "\n".join(text)}
    if reasoning:
        # Templates that keep earlier reasoning (Qwen's preserve_thinking) render it again, so the turn reads as it was generated.
        msg["reasoning_content"] = "\n".join(reasoning)
    if calls:
        msg["tool_calls"] = calls
    return msg


def user_messages(blocks, names, where):
    """A user turn as OpenAI messages: each tool_result becomes a `tool` message, in order, with the other blocks around it as user turns."""
    out, parts = [], []

    def flush():
        if parts:
            out.append({"role": "user", "content": list(parts)})
            parts.clear()

    for b in blocks:
        kind = b.get("type")
        if kind == "text":
            parts.append({"type": "text", "text": b.get("text") or ""})
        elif kind == "image":
            parts.append(image_part(b, where))
        elif kind == "tool_result":
            flush()
            inner = blocks_of(b.get("content") or "", where)
            out.append({"role": "tool", "tool_call_id": b.get("tool_use_id"),
                        "name": names.get(b.get("tool_use_id")),
                        "content": "\n".join(i.get("text") or "" for i in inner
                                              if i.get("type") == "text")})
            parts.extend(image_part(i, where) for i in inner if i.get("type") == "image")
        else:
            raise ApiError(400, f"{where}: content block type '{kind}' is not supported")
    flush()
    return out


def to_messages(body):
    """The system prompt and messages as the OpenAI route's `messages`."""
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ApiError(400, "messages: Field required")
    out, names = [], {}
    system = system_text(body.get("system"))
    replied = False
    for i, m in enumerate(messages):
        where = f"messages.{i}"
        if not isinstance(m, dict) or m.get("role") not in ("user", "assistant", "system"):
            raise ApiError(400, f"{where}.role: must be 'user' or 'assistant'")
        blocks = blocks_of(m.get("content"), where)
        if m["role"] == "assistant":
            replied = True
            out.append(assistant_message(blocks, names, where))
        elif m["role"] == "system" and not replied and all(b.get("type") == "text" for b in blocks):
            # Chat templates take one leading system turn, so a system entry before the first reply (Claude Code's environment block) joins it.
            system = "\n\n".join(t for t in [system, *(b.get("text") or "" for b in blocks)] if t)
        elif m["role"] == "system":
            # A later one is a user turn marked the way Claude Code marks its own reminders.
            out.extend(user_messages([{"type": "text", "text": f"<system-reminder>\n{b.get('text') or ''}\n</system-reminder>"}
                                      if b.get("type") == "text" else b for b in blocks], names, where))
        else:
            out.extend(user_messages(blocks, names, where))
    return ([{"role": "system", "content": system}] if system else []) + out


def to_tools(body):
    """The tool list in OpenAI's shape, or None; tool_choice none leaves it out, and any / tool are read as auto."""
    tools, choice = body.get("tools"), body.get("tool_choice")
    if tools is not None and not (isinstance(tools, list) and all(isinstance(t, dict) for t in tools)):
        raise ApiError(400, "tools: must be a list of objects")
    if choice is not None and not (isinstance(choice, dict)
                                   and choice.get("type") in ("auto", "any", "tool", "none")):
        raise ApiError(400, "tool_choice: type must be auto, any, tool or none")
    if not tools or (choice or {}).get("type") == "none":
        return None
    out = [{"type": "function", "function": {
        "name": t["name"], "description": t.get("description") or "", "parameters": t["input_schema"]}}
        for t in tools if t.get("name") and isinstance(t.get("input_schema"), dict) and not t.get("defer_loading")]
    return out or None


def parse(h, srv, body, counting=False):
    """The request as the namespace both routes use, or ApiError."""
    if not isinstance(body, dict):
        raise ApiError(400, "the body must be a JSON object")
    messages = to_messages(body)
    try:
        messages = srv.clean_messages(messages, h.host.vision)
    except ValueError as e:
        raise ApiError(400, str(e))
    max_tokens = body.get("max_tokens")
    if not counting and not (isinstance(max_tokens, int) and not isinstance(max_tokens, bool)
                             and max_tokens > 0):
        raise ApiError(400, "max_tokens: must be a positive integer")
    stops = body.get("stop_sequences") or []
    if not (isinstance(stops, list) and all(isinstance(s, str) for s in stops)):
        raise ApiError(400, "stop_sequences: must be a list of strings")
    thinking = body.get("thinking")
    if thinking is not None and not isinstance(thinking, dict):
        raise ApiError(400, "thinking: must be an object")
    budget = thinking.get("budget_tokens") if thinking and thinking.get("type") == "enabled" else None
    if budget is not None and not (isinstance(budget, int) and not isinstance(budget, bool) and budget > 0):
        raise ApiError(400, "thinking.budget_tokens: must be a positive integer")
    sampling = None
    if h.sampling:
        fields = {**h.default_sampling,
                  **{k: body[k] for k in ("temperature", "top_p", "top_k") if body.get(k) is not None}}
        try:
            sampling = srv.Sampling.from_request(fields)
        except (TypeError, ValueError) as e:
            raise ApiError(400, f"sampling: {e}")
    return SimpleNamespace(
        messages=messages, tools=to_tools(body), max_tokens=max_tokens,
        stops=[s for s in stops if s], sampling=sampling, stream=bool(body.get("stream")),
        # Adaptive thinking is on only under --adaptive-thinking, and every other type stays off; on, it carries the client's effort as the template's reasoning_effort.
        think=bool(thinking) and (thinking.get("type") == "enabled"
                                  or (thinking.get("type") == "adaptive" and h.adaptive_thinking))
        and (EFFORT.get((body.get("output_config") or {}).get("effort")) or True),
        budget=budget, model=body.get("model") or "local", id="msg_" + uuid.uuid4().hex[:24])


# the reply, as Anthropic's blocks

class StopCut:
    """Cuts the answer at the first stop sequence, holding back a tail that may still grow into one."""

    def __init__(self, stops):
        self.stops, self.held, self.hit = stops, "", None

    def feed(self, piece):
        text = self.held + piece
        found = [(text.find(s), s) for s in self.stops if s in text]
        if found:
            i, self.hit = min(found, key=lambda f: f[0])
            self.held = ""
            return text[:i]
        keep = max((k for s in self.stops for k in range(1, len(s)) if text.endswith(s[:k])),
                   default=0)
        self.held = text[len(text) - keep:]
        return text[:len(text) - keep]

    def flush(self):
        out, self.held = self.held, ""
        return out


class Reply:
    """A reply's text pieces as ("thinking", text) and ("text", text) items while they arrive, then one ("tool_use", block) per call the parser finds in the whole answer."""

    def __init__(self, srv, req):
        self.req = req
        self.think = srv.ThinkSplit() if req.think else None
        self.cut = StopCut(req.stops)
        self.answer = []    # kept whole when tools are listed, since a client cannot act on half a call

    def feed(self, piece):
        reasoning, text = self.think.feed(piece) if self.think else ("", piece)
        return self._items(reasoning, self.cut.feed(text))

    def finish(self):
        items = self._items(self.think.flush() if self.think else "", self.cut.flush())
        if self.req.tools:
            text, calls = toolfmt.parse("".join(self.answer), self.req.tools)
            if text:
                items.append(("text", text))
            items += [("tool_use", tool_use(c, self.req.id, i)) for i, c in enumerate(calls)]
        return items

    def _items(self, reasoning, text):
        items = [("thinking", reasoning)] if reasoning else []
        if text and self.req.tools:
            self.answer.append(text)
        elif text:
            items.append(("text", text))
        return items


def tool_use(call, msg_id, i):
    fn = call["function"]
    try:
        args = json.loads(fn["arguments"])
    except ValueError:
        args = fn["arguments"]
    if not isinstance(args, dict):
        args = {"input": args}
    return {"type": "tool_use", "id": f"toolu_{msg_id[4:]}_{i}", "name": fn["name"], "input": args}


def new_block(kind, text=""):
    block = {"type": kind, kind: text}
    if kind == "thinking":
        block["signature"] = ""
    return block


def fold(items):
    """The items of a whole reply as content blocks: neighbours of one kind merge, reasoning is trimmed as the OpenAI route trims it."""
    blocks = []
    for kind, payload in items:
        if kind == "tool_use":
            blocks.append(payload)
        elif blocks and blocks[-1]["type"] == kind:
            blocks[-1][kind] += payload
        else:
            blocks.append(new_block(kind, payload))
    for b in blocks:
        if b["type"] == "thinking":
            b["thinking"] = b["thinking"].strip()
    return [b for b in blocks if b["type"] != "thinking" or b["thinking"]] or [new_block("text")]


def usage_of(info, pieces):
    """Anthropic's usage: input_tokens leave out what a kept cache supplied, which is counted apart."""
    cached = info.get("cached_tokens") or 0
    done = info.get("completion_tokens")
    return {"input_tokens": max((info.get("prompt_tokens") or 0) - cached, 0),
            "output_tokens": pieces if done is None else done,   # a reply cut early has no engine count
            "cache_creation_input_tokens": 0, "cache_read_input_tokens": cached}


def stop_of(h, reply, items, info, req):
    """(stop_reason, stop_sequence) of a finished reply."""
    if reply.cut.hit:
        return "stop_sequence", reply.cut.hit
    calls = [p for k, p in items if k == "tool_use"]
    return STOP_REASONS[h._finish_reason(calls, info, req.max_tokens)], None


def start(h, req, info, streaming):
    """The engine's reply to this request as an iterator of pieces: many when streaming, else the whole text as one."""
    if h.host.llm is None:
        raise ApiError(503, h.host.last_error or "no model is loaded")
    try:
        out = h.host.generate(req.messages, req.max_tokens, stream=streaming, tools=req.tools,
                              sampling=req.sampling, info=info, thinking=req.think,
                              client_gone=h.client_gone, think_budget=req.budget)
    except Exception as e:
        raise ApiError(500, f"generation failed: {e}")
    return out if streaming else [out]


# the two answers

def respond(h, srv, req):
    """A whole reply as one message object; stop sequences read the stream so they can end the turn early."""
    info, items, pieces = {}, [], 0
    reply = Reply(srv, req)
    try:
        source = start(h, req, info, streaming=bool(req.stops))
        for piece in source:
            pieces += 1
            items += reply.feed(piece)
            if reply.cut.hit:
                break
        if hasattr(source, "close"):
            source.close()
        items += reply.finish()
    except ApiError:
        raise
    except Exception as e:
        raise ApiError(500, f"generation failed: {e}")
    reason, sequence = stop_of(h, reply, items, info, req)
    h._json(200, {"id": req.id, "type": "message", "role": "assistant", "model": req.model,
                  "content": fold(items), "stop_reason": reason, "stop_sequence": sequence,
                  "usage": usage_of(info, pieces)})


class Blocks:
    """Writes content blocks as Anthropic's block events, opening a block when the kind changes."""

    def __init__(self, write):
        self.write, self.index, self.kind = write, -1, None

    def add(self, kind, payload):
        if kind == "tool_use":
            self.open(dict(payload, input={}))
            self.write("content_block_delta", {"index": self.index, "delta": {
                "type": "input_json_delta", "partial_json": json.dumps(payload["input"])}})
            self.close()
            return
        if self.kind != kind:
            self.open(new_block(kind))
        self.write("content_block_delta", {"index": self.index, "delta": {
            "type": kind + "_delta", kind: payload}})

    def open(self, block):
        self.close()
        self.index += 1
        self.kind = block["type"]
        self.write("content_block_start", {"index": self.index, "content_block": block})

    def close(self):
        if self.kind is not None:
            self.write("content_block_stop", {"index": self.index})
            self.kind = None


def pump(pieces, q, stop):
    """Moves a generator's pieces into a queue so the handler can ping while the prompt is read; it closes the generator itself, since only the thread running it may."""
    try:
        for piece in pieces:
            q.put(("piece", piece))
            if stop.is_set():
                break
        q.put(("end", None))
    except Exception as e:
        q.put(("error", e))
    finally:
        pieces.close()


def stream(h, srv, req):
    """A reply as server-sent events, the headers and message_start going out before the prompt is read."""
    info = {}
    source = start(h, req, info, streaming=True)
    h.send_response(200)
    h.send_header("Content-Type", "text/event-stream")
    h.send_header("Cache-Control", "no-cache")
    h.end_headers()

    def write(kind, data):
        h.wfile.write(event(kind, data))

    q, stop, reply, items, pieces = queue.Queue(), threading.Event(), Reply(srv, req), [], 0
    blocks = Blocks(write)
    try:
        write("message_start", {"message": {
            "id": req.id, "type": "message", "role": "assistant", "model": req.model,
            "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": usage_of({}, 0)}})
        h.wfile.flush()
        threading.Thread(target=pump, args=(source, q, stop), daemon=True).start()
        while True:
            try:
                kind, val = q.get(timeout=PING_S)
            except queue.Empty:
                write("ping", {})
                h.wfile.flush()
                continue
            if kind == "error":
                raise val
            if kind == "end":
                break
            pieces += 1
            new = reply.feed(val)
            items += new
            for item in new:
                blocks.add(*item)
            h.wfile.flush()
            if reply.cut.hit:
                break
        new = reply.finish()
        items += new
        for item in new:
            blocks.add(*item)
        if blocks.index < 0:
            blocks.open(new_block("text"))      # a reply with nothing in it still has one block
        blocks.close()
        reason, sequence = stop_of(h, reply, items, info, req)
        write("message_delta", {"delta": {"stop_reason": reason, "stop_sequence": sequence},
                                "usage": usage_of(info, pieces)})
        write("message_stop", {})
        h.wfile.flush()
    except (BrokenPipeError, ConnectionResetError):
        pass    # the client hung up; the pump closes the turn at its next piece
    except Exception as e:
        try:    # the 200 is out, so an error event is the only way left to say so
            h.wfile.write(event("error", error_body(500, f"generation failed: {e}")))
            h.wfile.flush()
        except OSError:
            pass
    finally:
        stop.set()


def count_tokens(h, req):
    """The prompt's length in tokens: the model's own chat template over system, messages and tools, then its tokenizer."""
    if h.host.llm is None:
        raise ApiError(503, h.host.last_error or "no model is loaded")
    tok = h.host.llm.tokenizer
    kw = {"add_generation_prompt": True, "tokenize": False}
    messages = toolfmt.escape(req.messages, tok)
    if req.tools:
        kw["tools"] = toolfmt.escape(req.tools, tok)
        messages = toolfmt.for_template(messages)
    try:
        try:
            effort = {"reasoning_effort": req.think} if isinstance(req.think, str) else {}
            text = tok.apply_chat_template(messages, enable_thinking=bool(req.think), **effort, **kw)
        except TypeError:
            text = tok.apply_chat_template(messages, **kw)
        return len(toolfmt.encode(tok, text))
    except Exception as e:
        raise ApiError(400, f"the chat template refused this conversation: {e}")


def post(h, path):
    """Handles POST /v1/messages and /v1/messages/count_tokens for server.py's Handler `h`; anthropic-version, anthropic-beta and any key header are ignored."""
    srv = sys.modules[type(h).__module__]   # server.py, whether it runs as __main__ or is imported
    try:
        if int(h.headers.get("Content-Length") or 0) > srv.MAX_BODY:
            raise ApiError(413, f"the request is larger than {srv.MAX_BODY // 2**20} MB")
        try:
            body = h._read_json()
        except ValueError:
            raise ApiError(400, "invalid JSON body")
        counting = path.endswith("/count_tokens")
        req = parse(h, srv, body, counting)
        if counting:
            h._json(200, {"input_tokens": count_tokens(h, req)})
        elif req.stream:
            stream(h, srv, req)
        else:
            respond(h, srv, req)
    except ApiError as e:
        h._json(e.status, error_body(e.status, e.message))
