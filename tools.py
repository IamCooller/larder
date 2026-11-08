#!/usr/bin/env python3
"""Function calling: render the tool block, read back what the model emits.

Qwen3.5 does not emit JSON for a call. Its template asks for an XML-ish form:

    <tool_call>
    <function=get_weather>
    <parameter=city>
    Berlin
    </parameter>
    </function>
    </tool_call>

so the reply has to be parsed rather than json.loads'd. Values are raw text and
may span lines, which is why the closing tag ends a parameter, not a newline.

Types come from the tool's own JSON schema. The wire format carries none, so a
parameter declared integer arrives as the string "3" and an application that
trusts the schema would break on it; `coerce` is where that is repaired, and it
leaves anything it cannot convert as the string the model actually produced.

Why this file is small: the tool text itself is built by the checkpoint's own
chat template (`apply_chat_template(tools=...)`), which puts the block at the
very front of the prompt -- ahead of the system message. That position is what
makes tools cheap to reuse: a stable list is the same leading ids every
request, and `StreamingLLM._build_front` keeps a prefilled cache of exactly
those to copy from. Nothing here needs to cache anything itself.

Client text that spells a control token (a file quoting <|im_end|>) goes through
the template as a placeholder (`escape`) and into the ids as plain text (`encode`),
so only the template's own structure and the model's own turns hold control tokens.
"""
import json
import re

START = re.compile(r"<tool_call>|<function=")
FUNC = re.compile(r"\s*<function=([^>\s]+)>\s*")
PARAM = re.compile(r"\s*<parameter=([^>\s]+)>\n?")
NEXT = re.compile(r"\s*<parameter=([^>\s]+)>")
CLOSE = re.compile(r"</parameter>")
# What may follow a call's last value: </function> then the block's end or another call, or the block's end, or the reply's end.
STOP = re.compile(r"\s*(?:(</function>)\s*(?:</tool_call>|<tool_call>|<function=|$)|</tool_call>|$)")
FUNC_END = re.compile(r"</function>\s*(?:</tool_call>|<tool_call>|<function=|$)")
BLOCK_END = re.compile(r"\s*</tool_call>")
JSON_BLOCK = re.compile(r"\s*(.*?)\s*</tool_call>", re.S)


def _coerce(value, spec):
    """One argument, converted to what the tool's schema declares."""
    kind = (spec or {}).get("type")
    try:
        if kind == "integer":
            return int(value.strip())
        if kind == "number":
            return float(value.strip())
        if kind == "boolean":
            return value.strip().lower() in ("true", "1", "yes")
        if kind in ("array", "object"):
            return json.loads(value)
    except (ValueError, TypeError, json.JSONDecodeError):
        # The model wrote something the schema did not describe. Handing the
        # application the raw string beats raising: it can see what came back
        # and decide, where an exception here would lose the reply entirely.
        return value
    return value


def _fn(tools, name):
    for t in tools or ():
        fn = t.get("function", t)
        if fn.get("name") == name:
            return fn


def _extra_tail(text, p):
    """Whether text from p is only whole parameters up to the call's close, each read to its first </parameter> as Qwen's own parser reads them, with no stray tag left after."""
    q = p
    while (m := NEXT.match(text, q)):
        end = text.find("</parameter>", m.end())
        if end < 0:
            return False
        q = end + len("</parameter>")
    return bool(STOP.match(text, q)) and text.count("</parameter>", p) == text.count("<parameter=", p)


def _value_end(text, i, names, missing):
    """(where the value that starts at i ends, where the call resumes), or None.

    A `</parameter>` ends a value where one of `names` (None: any) follows or the call closes whole, so a value quoting these tags stays whole.
    """
    # Ceiling: a model that leaves out a required parameter before a second well-formed call is read as one call.
    spare = None
    for m in CLOSE.finditer(text, i):
        nxt = NEXT.match(text, m.end())
        # An unlisted name ends the value too once nothing required is missing and only whole parameters follow, as when grep gets an extra path.
        if nxt and (names is None or nxt.group(1) in names or (not missing and _extra_tail(text, m.end()))):
            return m.start(), m.end()
        stop = STOP.match(text, m.end())
        if stop and not missing and text.count("</parameter>", m.end()) <= text.count("<parameter=", m.end()):
            return m.start(), m.end()
        if stop and spare is None and stop.group().strip():
            spare = m.start(), m.end()      # the call is closed but not whole: the reading from before the schema
    if spare:
        return spare
    # A last value that lost its own </parameter> runs to the </function> that ends the call.
    end = FUNC_END.search(text, i)
    return (end.start(), end.start()) if end else None


def _function(text, p, tools):
    """(name, args, end, closed) of the <function=...> block at p, or None when it is not a whole call."""
    m = FUNC.match(text, p)
    if not m:
        return None
    name = m.group(1)
    schema = (_fn(tools, name) or {}).get("parameters") or {}
    props = schema.get("properties") or {}
    args, p = {}, m.end()
    while (m := PARAM.match(text, p)):
        missing = set(schema.get("required") or ()) - set(args) - {m.group(1)}
        # Only a text value can quote a parameter tag, so after a number, a flag or a name the tool does not list any parameter may follow.
        spec = props.get(m.group(1))
        free = not props or spec is None or spec.get("type") in ("integer", "number", "boolean")
        found = _value_end(text, m.end(), None if free else set(props), missing)
        if not found:
            return None
        value = text[m.end():found[0]]
        args[m.group(1)] = _coerce(value[:-1] if value.endswith("\n") else value, props.get(m.group(1)))
        p = found[1]
    stop = STOP.match(text, p)
    if not stop:
        return None
    return name, args, stop.end(1) if stop.group(1) else p, bool(stop.group(1))


def _block(text, p, tools):
    """([(name, arguments)], end) of the call block whose <tool_call> ends at p, or None."""
    calls = []
    while (got := _function(text, p, tools)):
        calls.append((got[0], json.dumps(got[1])))
        p = got[2]
    if calls:
        close = BLOCK_END.match(text, p)
        return calls, close.end() if close else p
    # Dense Qwen3's template asks for the call as JSON inside the same
    # tags: {"name": ..., "arguments": {...}}. Without this a dense
    # target's tool calls reached the client as empty content.
    m = JSON_BLOCK.match(text, p)
    try:
        obj = json.loads(m.group(1)) if m else None
    except json.JSONDecodeError:
        return None
    if isinstance(obj, dict) and isinstance(obj.get("name"), str):
        args = obj.get("arguments", {})
        return [(obj["name"], args if isinstance(args, str) else json.dumps(args, ensure_ascii=False))], m.end()


def parse(text, tools=None):
    """(clean_text, calls) -- calls are OpenAI-shaped tool_call dicts.

    The template allows reasoning BEFORE a call and forbids it after, so
    whatever precedes the first block is real assistant text and is returned
    as such rather than thrown away. A block that is not a whole call stays in
    the text. A call cut short of its closing tags counts when its parameters
    are complete, and a bare <function=NAME> block counts when it is closed
    and NAME is one of the listed tools.
    """
    calls, kept, last, pos = [], [], 0, 0
    while (m := START.search(text, pos)):
        pos = m.end()
        if m.group() == "<tool_call>":
            got = _block(text, pos, tools)
        else:
            f = _function(text, m.start(), tools)
            got = ([(f[0], json.dumps(f[1]))], f[2]) if f and f[3] and _fn(tools, f[0]) else None
        if got:
            kept.append(text[last:m.start()])
            calls += [{"id": f"call_{len(kept) - 1}_{name}", "type": "function",
                       "function": {"name": name, "arguments": args}} for name, args in got[0]]
            last = pos = got[1]
    kept.append(text[last:])
    return "".join(kept).strip(), calls


def for_template(messages):
    """History as the chat template wants it, not as OpenAI sends it.

    OpenAI carries `tool_calls[].function.arguments` as a JSON STRING, and
    Qwen's template does `tool_call.arguments|items` -- a mapping operation.
    Feeding it the string raises "Can only get item pairs from a mapping" and
    the whole turn is lost, which is what happened the first time an
    application handed a tool result back.

    A copy, not an edit in place: the caller's list is its own conversation
    state, and an engine that rewrote it would hand back something that no
    longer round-trips through its own API.
    """
    out = []
    for m in messages:
        calls = m.get("tool_calls") if isinstance(m, dict) else None
        if not calls:
            out.append(m)
            continue
        fixed = []
        for c in calls:
            fn = dict(c.get("function", c))
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    fn["arguments"] = json.loads(args)
                except json.JSONDecodeError:
                    # Not JSON. One name/value pair keeps the call visible to
                    # the model rather than dropping the turn.
                    fn["arguments"] = {"input": args}
            fixed.append({**c, "function": fn} if "function" in c else fn)
        out.append({**m, "tool_calls": fixed})
    return out


# An added-token string held as its id between two Unicode noncharacters, reserved for internal use and absent from real text.
_HOLD = re.compile("\ufdd0(\\d+)\ufdd1")


def _added(tok):
    """The tokenizer's added-token strings and ids, or {} when `encode` could not rebuild its split (no fast backend, or a token that strips or normalizes)."""
    dec = getattr(tok, "added_tokens_decoder", None)
    if getattr(tok, "backend_tokenizer", None) is None or not dec or any(
            t.lstrip or t.rstrip or t.single_word or t.normalized for t in dec.values()):
        return {}
    return {t.content: i for i, t in dec.items()}


def _alternation(added):
    return "|".join(map(re.escape, sorted(added, key=len, reverse=True)))


def escape(x, tok):
    """Messages, tools or a string with every added-token string in client text held as a placeholder, an assistant turn kept as the model wrote it."""
    added = _added(tok)
    if not added:
        return x
    pat = re.compile(_alternation(added))

    def walk(v):
        if isinstance(v, str):
            return pat.sub(lambda m: f"\ufdd0{added[m[0]]}\ufdd1", v)
        if isinstance(v, dict):
            return v if v.get("role") == "assistant" else {k: walk(w) for k, w in v.items()}
        if isinstance(v, (list, tuple)):
            return [walk(w) for w in v]
        return v
    return walk(x)


def _plain(back, s):
    """`s` through the normalizer, pre-tokenizer and model alone, so a control-token string in it is ordinary text."""
    if back.normalizer is not None:
        s = back.normalizer.normalize_str(s)
    pieces = back.pre_tokenizer.pre_tokenize_str(s) if back.pre_tokenizer is not None else [(s, None)]
    return [t.id for p, _ in pieces for t in back.model.tokenize(p)]


def encode(tok, text, add_special_tokens=True):
    """The ids of a render made from `escape`d messages: `tok.encode(text)`, with each placeholder's string as plain text."""
    added = _added(tok) if "\ufdd0" in text else None
    if not added:
        return tok.encode(text) if add_special_tokens else tok.encode(text, add_special_tokens=False)
    names = {i: s for s, i in added.items()}
    back = tok.backend_tokenizer
    # Known limit: assumes the post-processor adds at most a leading BOS, as every chat checkpoint here does; an appended EOS would need backend_tokenizer.post_process.
    ids = list(tok.encode("", add_special_tokens=add_special_tokens))
    # A fast tokenizer cuts at added tokens first and encodes each piece between them alone, so these are its own ids wherever no placeholder sits.
    for k, piece in enumerate(re.split(f"({_alternation(added)})", text)):
        if k % 2:
            ids.append(added[piece])
        elif "\ufdd0" in piece:
            ids += _plain(back, _HOLD.sub(lambda m: names.get(int(m[1]), m[0]), piece))
        elif piece:
            ids += tok.encode(piece, add_special_tokens=False)
    return ids


def demo():
    tools = [{"type": "function", "function": {
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string"},
            "days": {"type": "integer"},
            "metric": {"type": "boolean"},
            "tags": {"type": "array"}}}}}]

    text = ("Let me look that up.\n"
            "<tool_call>\n<function=get_weather>\n"
            "<parameter=city>\nBerlin\n</parameter>\n"
            "<parameter=days>\n3\n</parameter>\n"
            "<parameter=metric>\ntrue\n</parameter>\n"
            "<parameter=tags>\n[\"a\", \"b\"]\n</parameter>\n"
            "</function>\n</tool_call>")
    clean, calls = parse(text, tools)
    assert clean == "Let me look that up.", repr(clean)
    assert len(calls) == 1
    args = json.loads(calls[0]["function"]["arguments"])
    assert args == {"city": "Berlin", "days": 3, "metric": True,
                    "tags": ["a", "b"]}, args

    # A value spanning lines: the closing tag ends it, not the newline.
    multi = ("<tool_call>\n<function=write>\n<parameter=body>\n"
             "line one\nline two\n</parameter>\n</function>\n</tool_call>")
    _, c = parse(multi)
    assert json.loads(c[0]["function"]["arguments"])["body"] == \
        "line one\nline two"

    # Two calls in one reply, and one call carrying no parameters at all.
    two = ("<tool_call>\n<function=a>\n<parameter=x>\n1\n</parameter>\n"
           "</function>\n</tool_call>\n"
           "<tool_call>\n<function=b>\n</function>\n</tool_call>")
    _, c = parse(two)
    assert [x["function"]["name"] for x in c] == ["a", "b"]
    assert json.loads(c[1]["function"]["arguments"]) == {}

    # A value quoting the closing tags, as the agent bench's edit of forced_rows.py did, stays one call.
    quoted = "a = 1  # nothing but </tool_call>, or the\nb = '</parameter>'\nc = '</function>'"
    clean, c = parse("<tool_call>\n<function=Edit>\n<parameter=old_string>\n" + quoted +
                     "\n</parameter>\n<parameter=new_string>\nx\n</parameter>\n</function>\n</tool_call>")
    assert clean == "" and len(c) == 1, (clean, c)
    assert json.loads(c[0]["function"]["arguments"]) == {"old_string": quoted, "new_string": "x"}

    # An integer the model wrote as prose stays a string rather than raising.
    _, c = parse("<tool_call>\n<function=get_weather>\n<parameter=days>\n"
                 "three\n</parameter>\n</function>\n</tool_call>", tools)
    assert json.loads(c[0]["function"]["arguments"])["days"] == "three"

    # Dense Qwen3's JSON call, and its text before the call kept.
    clean, c = parse('I will check.\n<tool_call>\n{"name": "get_weather", '
                     '"arguments": {"city": "Zürich", "days": 2}}\n</tool_call>')
    assert clean == "I will check." and c[0]["function"]["name"] == "get_weather"
    assert json.loads(c[0]["function"]["arguments"]) == {"city": "Zürich", "days": 2}
    # A block that is neither form stays text, not guessed at.
    assert parse("<tool_call>\nnot a call\n</tool_call>")[1] == []

    # OpenAI's JSON-string arguments become a mapping for the template, and
    # the caller's own list is not touched.
    convo = [{"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "type": "function", "function": {
            "name": "get_weather", "arguments": '{"city": "Berlin"}'}}]}]
    ready = for_template(convo)
    assert ready[0]["tool_calls"][0]["function"]["arguments"] == \
        {"city": "Berlin"}
    assert convo[0]["tool_calls"][0]["function"]["arguments"] == \
        '{"city": "Berlin"}', "the caller's messages were rewritten"
    # Arguments that are not JSON keep the turn instead of losing it.
    bad = for_template([{"role": "assistant", "tool_calls": [
        {"function": {"name": "f", "arguments": "not json"}}]}])
    assert bad[0]["tool_calls"][0]["function"]["arguments"] == \
        {"input": "not json"}
    # A history with no calls passes through untouched.
    plainmsgs = [{"role": "user", "content": "hi"}]
    assert for_template(plainmsgs) == plainmsgs

    # Ordinary replies are untouched, including ones that merely mention tools.
    plain = "I cannot call anything here, but <tools> is just a word."
    assert parse(plain) == (plain, [])

    print("tools: parse, multi-line values, several calls, type coercion, "
          "template shaping and plain text all hold")


if __name__ == "__main__":
    demo()
