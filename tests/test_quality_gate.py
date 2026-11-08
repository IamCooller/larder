#!/usr/bin/env python3
"""tools/quality_gate.py on a stub OpenAI-compatible server and tiny datasets: no model, a few seconds.

    python3 tests/test_quality_gate.py

Checks the exact McNemar numbers against hand-computed cases, the scoring of both benchmarks,
and the verdict of an end-to-end run, including the --arm and --compare route.
"""
import contextlib
import io
import json
import re
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import quality_gate as qg


def serve(wrong_gsm, wrong_he):
    """A stub endpoint that answers the toy items right except the listed indices."""
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert body["temperature"] == 0 and body["stream"] is False
            text = body["messages"][-1]["content"]
            if "Solve the problem" in text:
                i = int(re.search(r"Q(\d+):", text).group(1))
                reply = "Adding gives\n#### %d" % (2 * i + 1 + (i in wrong_gsm))
            else:
                i = int(re.search(r"def f(\d+)", text).group(1))
                reply = "```python\ndef f%d():\n    return %d\n```" % (i, -1 if i in wrong_he else i)
            out = json.dumps({"choices": [{"message": {"content": reply}}]}).encode()
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
        sys.argv = ["quality_gate.py", *argv]
        code = qg.main()
    return code, out.getvalue()


def main():
    # Hand-computed: 7 against 1 is 2 * (C(8,0) + C(8,1)) / 2^8 = 18/256; 9 against 0 is 2 / 2^9; 3 against 3 is 1.
    assert qg.mcnemar_exact(7, 1) == 18 / 256 == 0.0703125
    assert qg.mcnemar_exact(1, 7) == 0.0703125
    assert qg.mcnemar_exact(9, 0) == 2 / 512 == 0.00390625
    assert qg.mcnemar_exact(3, 3) == 1.0 and qg.mcnemar_exact(0, 0) == 1.0
    # Newcombe method 10, hand-checked against the published worked example (CAMIS, "Confidence intervals
    # for paired proportions", 20/15/6/5 of 46, hybrid score 0.0026 to 0.3699): the first arm wins 15 pairs.
    d, lo, hi = qg.paired_diff_ci([20, 6, 15, 5])
    assert abs(d - 9 / 46) < 1e-12 and round(lo, 4) == 0.0026 and round(hi, 4) == 0.3699, (d, lo, hi)
    # Swapping the arms mirrors the interval; no discordant pairs gives a symmetric interval around 0.
    d2, lo2, hi2 = qg.paired_diff_ci([20, 15, 6, 5])
    assert abs(lo2 + hi) < 1e-12 and abs(hi2 + lo) < 1e-12 and abs(d2 + d) < 1e-12
    d3, lo3, hi3 = qg.paired_diff_ci([30, 0, 0, 20])
    assert d3 == 0 and abs(lo3 + hi3) < 1e-12 and hi3 < 0.1
    assert qg.table([1, 1, 0, 0, 1], [1, 0, 1, 0, 0]) == [1, 2, 1, 1]
    assert qg.final_number("so 1,234 apples\n#### 1,234") == 1234.0 and qg.final_number("it is 7.") == 7.0
    assert qg.final_number("no digits") is None

    with tempfile.TemporaryDirectory() as tmp:
        gsm, he = Path(tmp) / "gsm.jsonl", Path(tmp) / "he.jsonl"
        gsm.write_text("\n".join(json.dumps({"question": "Q%d: what is %d plus %d?" % (i, i, i + 1),
                                             "answer": "work\n#### %d" % (2 * i + 1)}) for i in range(10)))
        he.write_text("\n".join(json.dumps({
            "task_id": "T/%d" % i, "prompt": "def f%d():\n    \"\"\"Return %d.\"\"\"\n" % (i, i),
            "entry_point": "f%d" % i, "test": "def check(c):\n    assert c() == %d\n" % i}) for i in range(4)))
        base, b_url = serve(set(), set())
        cand, c_url = serve({0}, {3})
        bad, d_url = serve(set(range(9)), set())
        try:
            # Candidate loses 1 + 1 items: 2 discordant against 0, exact p 0.5, not significant.
            code, text = run("--baseline", b_url, "--candidate", c_url, "--gsm8k", str(gsm), "--humaneval", str(he),
                             "--json", str(Path(tmp) / "o.json"))
            rows = {r["set"]: r for r in json.loads((Path(tmp) / "o.json").read_text())["rows"]}
            assert code == 0 and "verdict: PASS" in text, text
            assert (rows["gsm8k"]["only_baseline"], rows["gsm8k"]["only_candidate"], rows["gsm8k"]["p_exact"]) == (1, 0, 1.0)
            assert rows["pooled"]["n"] == 14 and rows["pooled"]["only_baseline"] == 2 and rows["pooled"]["p_exact"] == 0.5
            # The same run with a margin: 2 lost items of 14 is not inside 3 points (lower bound far below -0.03),
            # but a wide margin of 0.5 passes; the report names the rule.
            code, text = run("--baseline", b_url, "--candidate", c_url, "--gsm8k", str(gsm), "--humaneval", str(he),
                             "--margin", "0.03")
            assert code == 2 and "non-inferiority" in text and "not above -0.03" in text, text
            code, text = run("--baseline", b_url, "--candidate", c_url, "--gsm8k", str(gsm), "--humaneval", str(he),
                             "--margin", "0.5")
            assert code == 0 and "above -0.5" in text, text
            assert abs(rows["pooled"]["diff"] + 2 / 14) < 1e-12 and rows["pooled"]["ci_low"] < -0.03 < rows["pooled"]["ci_high"]
            # Candidate wrong on 9 of 10 GSM8K items: 9 against 0, p 0.0039, FAIL, exit code 2.
            code, text = run("--baseline", b_url, "--candidate", d_url, "--gsm8k", str(gsm))
            assert code == 2 and "verdict: FAIL" in text and "0.0039" in text, text
            # --n takes the first items only; two arms saved apart and paired later give the same table.
            code, text = run("--baseline", b_url, "--candidate", d_url, "--gsm8k", str(gsm), "--n", "3")
            assert code == 0 and "PASS" in text and "0.2500" in text, text   # 3 against 0: 2 / 2^3, too few to tell
            run("--baseline", b_url, "--gsm8k", str(gsm), "--arm", str(Path(tmp) / "a.json"))
            run("--baseline", d_url, "--gsm8k", str(gsm), "--arm", str(Path(tmp) / "b.json"))
            code, text = run("--compare", str(Path(tmp) / "a.json"), str(Path(tmp) / "b.json"))
            assert code == 2 and "0.0039" in text, text
        finally:
            for s in (base, cand, bad):
                s.shutdown()
    print("test_quality_gate: ok")


if __name__ == "__main__":
    main()
