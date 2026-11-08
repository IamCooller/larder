#!/usr/bin/env python3
"""No-regression check: record greedy replies for a fixed prompt set, then compare another
build's replies field by field, byte for byte, and print the first difference.

    python3 tools/golden_check.py record --url http://127.0.0.1:8790/v1 --out before.json
    python3 tools/golden_check.py check  before.json --url http://127.0.0.1:8790/v1
    python3 tools/golden_check.py diff   before.json after.json

`record` sends each request once (temperature 0, no streaming) and saves the request and the reply:
content, reasoning, tool calls (without their random ids), finish reason and token usage.
`check` replays the saved requests against another endpoint and compares; `diff` compares two files.
Exit code 0 when every field of every request is identical, 2 at the first difference, 1 on an error.
The prompt set is built in (plain chat, code, a two-turn chat, a system message, a tool call);
--prompts takes a JSON list of {"name", "messages", "tools"} to use your own instead.
Run it with the same model, flags and a warm or cold cache on both sides: a reply that depends on
timing (a speculative arm chosen by speed, for instance) can differ without a regression.
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

FIELDS = ("content", "reasoning_content", "tool_calls", "finish_reason", "usage")
TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "The current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}},
                   "required": ["city"]}}}]
PROMPTS = [
    ("en", [{"role": "user", "content": "Explain in simple words why the sky is blue, and give two "
                                        "everyday examples of light scattering."}], None),
    ("de", [{"role": "user", "content": "Erklaere in zwei Saetzen, warum der Himmel blau ist."}], None),
    ("code", [{"role": "user", "content": "Write a Python function that checks whether a string is a "
                                          "palindrome, ignoring case and punctuation, with three asserts."}], None),
    ("turn1", [{"role": "user", "content": "A train travels 60 km in 45 minutes. How fast is it in km/h? "
                                           "Give the number."}], None),
    ("turn2", None, None),
    ("system", [{"role": "system", "content": "You answer in one short sentence and never use lists."},
                {"role": "user", "content": "What is the capital of Australia?"}], None),
    ("tool", [{"role": "user", "content": "What is the weather in Berlin? Use the tool."}], TOOLS),
]
FOLLOW_UP = "And how far would it go in 2 hours at that speed?"


def post(url, model, messages, tools, max_tokens, timeout):
    body = {"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens, "stream": False}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def reply_of(doc):
    """The compared fields of one completion, tool calls without the server's per-reply id."""
    choice = (doc.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    calls = [{"type": c.get("type"), "function": {"name": (c.get("function") or {}).get("name"),
                                                  "arguments": (c.get("function") or {}).get("arguments")}}
             for c in message.get("tool_calls") or []]
    return {"content": message.get("content"), "reasoning_content": message.get("reasoning_content"),
            "tool_calls": calls, "finish_reason": choice.get("finish_reason"), "usage": doc.get("usage")}


def record(url, model, cases, max_tokens, timeout):
    out, last = [], None
    for case in cases:
        name, messages, tools = case["name"], case["messages"], case.get("tools")
        if messages is None:
            messages = [{"role": "user", "content": PROMPTS[3][1][0]["content"]},
                        {"role": "assistant", "content": last or ""},
                        {"role": "user", "content": FOLLOW_UP}]
        reply = reply_of(post(url, model, messages, tools, max_tokens, timeout))
        last = reply["content"]
        out.append({"name": name, "messages": messages, "tools": tools, "reply": reply})
        print(f"  {name}: {len(last or '')} chars", file=sys.stderr)
    return {"max_tokens": max_tokens, "cases": out}


def first_difference(a, b):
    """None when the two records are identical, else a one-line description of the first difference."""
    for x, y in zip(a["cases"], b["cases"]):
        for field in FIELDS:
            u, v = x["reply"].get(field), y["reply"].get(field)
            if u == v:
                continue
            su, sv = json.dumps(u, ensure_ascii=False), json.dumps(v, ensure_ascii=False)
            at = next((i for i, (p, q) in enumerate(zip(su, sv)) if p != q), min(len(su), len(sv)))
            return (f"{x['name']}.{field}: first difference at char {at}\n"
                    f"  A: ...{su[max(0, at - 30):at + 40]}\n  B: ...{sv[max(0, at - 30):at + 40]}")
    if len(a["cases"]) != len(b["cases"]):
        return f"different number of requests: {len(a['cases'])} and {len(b['cases'])}"
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("record", "check"):
        p = sub.add_parser(name)
        p.add_argument("--url", required=True, help="base URL of the endpoint, e.g. http://127.0.0.1:8790/v1")
        p.add_argument("--model", default="model")
        p.add_argument("--max-tokens", type=int, default=128)
        p.add_argument("--timeout", type=float, default=1800)
    sub.choices["record"].add_argument("--out", required=True)
    sub.choices["record"].add_argument("--prompts", help="JSON list of {name, messages, tools} instead of the built-in set")
    sub.choices["check"].add_argument("golden", help="a file written by record")
    d = sub.add_parser("diff")
    d.add_argument("a")
    d.add_argument("b")
    a = ap.parse_args()
    if a.cmd == "record":
        cases = json.loads(Path(a.prompts).read_text()) if a.prompts else [
            {"name": n, "messages": m, "tools": t} for n, m, t in PROMPTS]
        Path(a.out).write_text(json.dumps(record(a.url, a.model, cases, a.max_tokens, a.timeout),
                                          indent=1, ensure_ascii=False))
        print("saved", a.out)
        return 0
    if a.cmd == "check":
        golden = json.loads(Path(a.golden).read_text())
        cases = [{"name": c["name"], "messages": c["messages"], "tools": c["tools"]} for c in golden["cases"]]
        # Replay the saved requests verbatim, including the second turn built from the recorded reply.
        other = {"cases": [{"name": c["name"], "reply": reply_of(post(a.url, a.model, c["messages"], c["tools"],
                                                                      golden["max_tokens"], a.timeout))}
                           for c in cases]}
        left, right = golden, other
    else:
        left, right = (json.loads(Path(p).read_text()) for p in (a.a, a.b))
    diff = first_difference(left, right)
    if diff:
        print("DIFFERENT " + diff)
        return 2
    print(f"IDENTICAL: {len(left['cases'])} requests, every field equal")
    return 0


if __name__ == "__main__":
    sys.exit(main())
