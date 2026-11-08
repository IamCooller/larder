#!/usr/bin/env python3
"""The HTTP server on a real model: every route answers, and a streamed reply arrives in pieces.

    python3 tests/test_server_demo.py CHECKPOINT_DIR

Loads the checkpoint (the first argument or $MOE_TEST_MODEL) and skips without one. The routes themselves are covered without a model by tests/test_webchat.py.
"""

import http.client
import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import server                                                  # noqa: E402
import support                                                 # noqa: E402


def main():
    model = support.model_dir("test_server_demo")
    if model is None:
        return
    host = server.ModelHost(model, max_tokens=12)
    httpd = server.serve(host, 0)
    port = httpd.socket.getsockname()[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=120)

        conn.request("GET", "/healthz")
        health = json.loads(conn.getresponse().read())
        assert health["ok"] and health["loaded"], health
        print("  /healthz ok:", health)

        conn.request("GET", "/v1/models")
        ids = [m["id"] for m in json.loads(conn.getresponse().read())["data"]]
        assert any(Path(model).name in i for i in ids), ids
        print("  /v1/models ok:", ids)

        body = json.dumps({"messages": [{"role": "user", "content": "Say the word apple and nothing else."}],
                           "max_tokens": 12}).encode()
        conn.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
        completion = json.loads(conn.getresponse().read())
        text = completion["choices"][0]["message"]["content"]
        assert text.strip(), completion
        print("  non-streaming ok:", text.strip()[:50])

        body = json.dumps({"messages": [{"role": "user", "content": "Say the word banana and nothing else."}],
                           "max_tokens": 12, "stream": True}).encode()
        conn.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
        r = conn.getresponse()
        assert r.status == 200, r.status
        events = [e for e in r.read().decode().split("\n\n") if e.startswith("data: ")]
        # More than one event before [DONE]: the reply arrived as pieces, not one blob.
        assert len(events) > 2, f"too few SSE events: {len(events)}"
        assert events[-1] == "data: [DONE]", events[-1]
        chunks = [json.loads(e[len("data: "):]) for e in events[:-1]]
        streamed = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
        assert streamed.strip(), streamed
        print(f"  streaming ok: {len(events) - 1} chunks, {streamed.strip()[:50]!r}")
        conn.close()
    finally:
        httpd.shutdown()
    print("ok")


if __name__ == "__main__":
    main()
