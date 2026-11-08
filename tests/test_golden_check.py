#!/usr/bin/env python3
"""tools/golden_check.py on a stub OpenAI-compatible server: no model, a few seconds.

    python3 tests/test_golden_check.py

Records a fixed prompt set, checks it against an identical endpoint (the random tool-call ids must
not matter) and against one that changes a single character of a tool call, which must be reported.
"""
import contextlib
import hashlib
import io
import json
import sys
import tempfile
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import golden_check as gc


def serve(city="Berlin", extra=""):
    """A stub endpoint whose reply depends only on the messages; tool calls carry a fresh random id."""
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert body["temperature"] == 0
            digest = hashlib.sha256(json.dumps(body["messages"]).encode()).hexdigest()[:12]
            message = {"role": "assistant", "content": "reply " + digest + extra}
            if body.get("tools"):
                message["content"] = None
                message["tool_calls"] = [{"id": "call_" + uuid.uuid4().hex, "type": "function", "function": {
                    "name": "get_weather", "arguments": json.dumps({"city": city})}}]
            out = json.dumps({"choices": [{"message": message, "finish_reason": "stop"}],
                              "usage": {"prompt_tokens": 5, "completion_tokens": 3}}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

    s = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s, "http://127.0.0.1:%d/v1" % s.server_address[1]


def run(*argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
        sys.argv = ["golden_check.py", *argv]
        code = gc.main()
    return code, out.getvalue()


def main():
    a, a_url = serve()
    b, b_url = serve()
    c, c_url = serve(city="Berlim")
    d, d_url = serve(extra="!")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            gold = str(Path(tmp) / "gold.json")
            code, text = run("record", "--url", a_url, "--out", gold)
            saved = json.loads(Path(gold).read_text())
            assert code == 0 and [x["name"] for x in saved["cases"]][-1] == "tool" and len(saved["cases"]) == 7
            assert "id" not in json.dumps(saved["cases"][-1]["reply"]["tool_calls"])
            turn2 = saved["cases"][4]["messages"]
            assert [m["role"] for m in turn2] == ["user", "assistant", "user"] and turn2[1]["content"].startswith("reply ")
            code, text = run("check", gold, "--url", b_url)
            assert code == 0 and text.startswith("IDENTICAL: 7 requests"), text
            code, text = run("check", gold, "--url", c_url)
            assert code == 2 and text.startswith("DIFFERENT tool.tool_calls: first difference at char "), text
            assert "Berlim" in text and "Berlin" in text, text
            # The first difference in request order is reported: every plain reply here differs, so "en" comes first.
            code, text = run("check", gold, "--url", d_url)
            assert code == 2 and text.startswith("DIFFERENT en.content"), text
            other = str(Path(tmp) / "other.json")
            run("record", "--url", c_url, "--out", other)
            code, text = run("diff", gold, other)
            assert code == 2 and "tool.tool_calls" in text, text
            code, text = run("diff", gold, gold)
            assert code == 0, text
    finally:
        for s in (a, b, c, d):
            s.shutdown()
    print("test_golden_check: ok")


if __name__ == "__main__":
    main()
