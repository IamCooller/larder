#!/usr/bin/env python3
"""Paired quality gate: GSM8K and HumanEval through two OpenAI-compatible endpoints, greedy, scored per
item, reported as a paired table with the exact McNemar p-value.

The verdict is PASS when the difference is not significant at 0.05 (pooled over the benchmarks run),
FAIL otherwise. Both endpoints answer the same items, so only the items where exactly one of them is
right matter; that is what McNemar's test counts. "Not significant" does not bound the drop the test
could miss, so each row also shows the paired accuracy difference (candidate minus baseline) with a
95% confidence interval. With --margin M the verdict is a non-inferiority check instead: PASS only
when the interval's lower bound is above -M (for example --margin 0.03).

    python3 tools/quality_gate.py --baseline http://127.0.0.1:8790/v1 --candidate http://127.0.0.1:8791/v1 \
        --gsm8k data/gsm8k_test.jsonl --humaneval data/HumanEval.jsonl.gz --n 100

Datasets are local files and nothing is downloaded here.
  GSM8K:     test.jsonl of github.com/openai/grade-school-math (grade_school_math/data/test.jsonl),
             one JSON object per line with "question" and "answer" (the answer ends in "#### <number>").
  HumanEval: HumanEval.jsonl.gz of github.com/openai/human-eval (data/HumanEval.jsonl.gz), with
             "task_id", "prompt", "entry_point" and "test".
HumanEval executes model-written code in a child python process with a 10 s limit and no sandbox: run
it in a container or a throwaway VM.

--n takes the first N items of each file. If the two servers do not fit in memory together, run each
arm alone with --arm OUT.json (pass that arm's URL as --baseline), then pair the two files with --compare.
"""
import argparse
import gzip
import json
import math
import re
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

GSM_PROMPT = ("Solve the problem step by step, then give the final answer as a number "
              "on the last line in the form: #### <number>\n\n")
HE_PROMPT = ("Complete the following Python function. Reply with the complete function, "
             "signature included, in one ```python code block and nothing else.\n\n"
             "```python\n{prompt}```")
MAX_TOKENS = {"gsm8k": 512, "humaneval": 500}
ALPHA = 0.05


def read_rows(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def final_number(text):
    """The number after the last '####', else the last number in the text."""
    found = re.findall(r"####\s*\$?\s*(-?[\d,]*\.?\d+)", text)
    if not found:
        found = re.findall(r"-?\$?\s*(\d[\d,]*\.?\d*)", text)
    if not found:
        return None
    try:
        return float(found[-1].replace(",", "").rstrip("."))
    except ValueError:
        return None


def extract_code(reply, prompt, entry_point):
    """The first code block that defines entry_point, with the problem's imports in front."""
    header = "\n".join(l for l in prompt.splitlines() if l.startswith(("import ", "from ")))
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", reply, re.S)
    for block in blocks:
        if re.search(rf"def\s+{re.escape(entry_point)}\s*\(", block):
            return header + "\n" + block
    if blocks:
        return prompt + blocks[0]
    return prompt + reply


def run_tests(code, test, entry_point, timeout=10.0):
    source = code + "\n\n" + test + f"\n\ncheck({entry_point})\n"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "t.py"
        path.write_text(source)
        try:
            run = subprocess.run([sys.executable, str(path)], cwd=tmp, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False
        return run.returncode == 0


def items(bench, path, n):
    """[(prompt text, scorer(reply) -> bool)] for the first n rows of a dataset file."""
    rows = read_rows(path)[:n]
    if bench == "gsm8k":
        out = []
        for r in rows:
            want = final_number(r["answer"])
            out.append((GSM_PROMPT + r["question"],
                        lambda reply, want=want: (lambda got: got is not None and abs(got - want) < 1e-6)(
                            final_number(reply))))
        return out
    return [(HE_PROMPT.format(prompt=r["prompt"]),
             lambda reply, r=r: run_tests(extract_code(reply, r["prompt"], r["entry_point"]),
                                          r["test"], r["entry_point"])) for r in rows]


def ask(base, model, text, max_tokens, no_thinking, timeout):
    """One greedy chat completion; the reply text."""
    body = {"model": model, "messages": [{"role": "user", "content": text}],
            "temperature": 0, "max_tokens": max_tokens, "stream": False}
    if no_thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        message = json.load(r)["choices"][0]["message"]
    return message.get("content") or ""


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p: the binomial test of b against b + c at 1/2."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


def wilson(x, n, z=1.959963984540054):
    """Wilson score interval for x of n."""
    p = x / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - half) / (1 + z * z / n), (centre + half) / (1 + z * z / n)


def paired_diff_ci(t):
    """Candidate minus baseline accuracy and its 95% CI from [both, only baseline, only candidate, neither].

    Newcombe (1998), Stat. Med. 17:2635-2650, method 10: Wilson intervals for the two correlated
    proportions combined by the hybrid score rule with the continuity-corrected phi.
    """
    a, c, b, d = t   # a both, b only candidate, c only baseline, d neither
    n = a + b + c + d
    if n == 0:
        return 0.0, -1.0, 1.0
    p1, p2 = (a + b) / n, (a + c) / n   # candidate, baseline
    l1, u1 = wilson(a + b, n)
    l2, u2 = wilson(a + c, n)
    den = (a + b) * (c + d) * (a + c) * (b + d)
    phi = 0.0
    if den:
        num = a * d - b * c
        num = num - n / 2 if num > n / 2 else (0 if num >= 0 else num)
        phi = num / math.sqrt(den)
    lo = p1 - p2 - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2))
    hi = p1 - p2 + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2))
    return p1 - p2, lo, hi


def table(base_hits, cand_hits):
    """[both right, only baseline right, only candidate right, neither] over the paired items."""
    both = sum(a and b for a, b in zip(base_hits, cand_hits))
    only_a = sum(a and not b for a, b in zip(base_hits, cand_hits))
    only_b = sum(b and not a for a, b in zip(base_hits, cand_hits))
    return [both, only_a, only_b, len(base_hits) - both - only_a - only_b]


def line(name, t):
    both, only_a, only_b, neither = t
    n = sum(t)
    p = mcnemar_exact(only_a, only_b)
    row = {"set": name, "n": n, "baseline": both + only_a, "candidate": both + only_b, "both": both,
            "only_baseline": only_a, "only_candidate": only_b, "neither": neither, "p_exact": p}
    row["diff"], row["ci_low"], row["ci_high"] = paired_diff_ci(t)
    return row


def collect(base, model, bench, path, n, no_thinking, timeout):
    """Replies are scored as they arrive: [bool] for one endpoint."""
    hits = []
    for i, (text, score) in enumerate(items(bench, path, n)):
        hits.append(bool(score(ask(base, model, text, MAX_TOKENS[bench], no_thinking, timeout))))
        print(f"  {bench} {i + 1}: {sum(hits)}/{len(hits)}", end="\r", file=sys.stderr)
    print(file=sys.stderr)
    return hits


def report(runs, margin=None):
    """runs: {bench: (baseline hits, candidate hits)} -> (rows, verdict)."""
    rows, pooled = [], [0, 0, 0, 0]
    for bench, (a, b) in runs.items():
        t = table(a, b)
        pooled = [x + y for x, y in zip(pooled, t)]
        rows.append(line(bench, t))
    if len(rows) > 1:
        rows.append(line("pooled", pooled))
    if margin is None:
        verdict = "PASS" if rows[-1]["p_exact"] >= ALPHA else "FAIL"
    else:
        verdict = "PASS" if rows[-1]["ci_low"] > -margin else "FAIL"
    return rows, verdict


def print_report(rows, verdict, margin=None):
    print(f"{'set':10} {'n':>4} {'base':>5} {'cand':>5} {'both':>5} {'only-base':>9} {'only-cand':>9} {'neither':>7} "
          f"{'exact p':>8} {'cand-base':>9}  95% CI (Newcombe 10)")
    for r in rows:
        print(f"{r['set']:10} {r['n']:>4} {r['baseline']:>5} {r['candidate']:>5} {r['both']:>5} "
              f"{r['only_baseline']:>9} {r['only_candidate']:>9} {r['neither']:>7} {r['p_exact']:>8.4f} "
              f"{r['diff']:>+9.3f}  [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]")
    if margin is None:
        word = "not significant" if verdict == "PASS" else "significant"
        print(f"verdict: {verdict} (pooled difference {word} at {ALPHA})")
    else:
        print(f"verdict: {verdict} (non-inferiority: pooled CI lower bound {rows[-1]['ci_low']:+.3f} "
              f"{'above' if verdict == 'PASS' else 'not above'} -{margin})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", help="base URL of the baseline endpoint, e.g. http://127.0.0.1:8790/v1")
    ap.add_argument("--candidate", help="base URL of the candidate endpoint")
    ap.add_argument("--gsm8k", help="path of GSM8K test.jsonl")
    ap.add_argument("--humaneval", help="path of HumanEval.jsonl or .jsonl.gz")
    ap.add_argument("--n", type=int, default=100, help="first N items of each file (default 100)")
    ap.add_argument("--model", default="model", help="model name sent in the request")
    ap.add_argument("--no-thinking", action="store_true", help="send enable_thinking=false in chat_template_kwargs")
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--arm", metavar="OUT.json", help="ask only --baseline and save its per-item hits to OUT.json")
    ap.add_argument("--compare", nargs=2, metavar=("BASE.json", "CAND.json"), help="pair two files saved with --arm")
    ap.add_argument("--margin", type=float, help="non-inferiority margin, e.g. 0.03: PASS only if the pooled CI lower bound is above -margin")
    ap.add_argument("--json", help="write the table and verdict here")
    a = ap.parse_args()
    benches = [(b, p) for b, p in (("gsm8k", a.gsm8k), ("humaneval", a.humaneval)) if p]
    if a.compare:
        A, B = (json.loads(Path(p).read_text()) for p in a.compare)
        runs = {b: (A[b], B[b]) for b in A if b in B}
    else:
        if not benches or not a.baseline:
            ap.error("give --baseline and at least one of --gsm8k, --humaneval (or --compare)")
        if a.arm:
            hits = {b: collect(a.baseline, a.model, b, p, a.n, a.no_thinking, a.timeout) for b, p in benches}
            Path(a.arm).write_text(json.dumps(hits))
            print("saved", a.arm)
            return 0
        if not a.candidate:
            ap.error("--candidate is needed (or use --arm / --compare)")
        runs = {b: (collect(a.baseline, a.model, b, p, a.n, a.no_thinking, a.timeout),
                    collect(a.candidate, a.model, b, p, a.n, a.no_thinking, a.timeout)) for b, p in benches}
    rows, verdict = report(runs, a.margin)
    print_report(rows, verdict, a.margin)
    if a.json:
        Path(a.json).write_text(json.dumps({"rows": rows, "verdict": verdict, "margin": a.margin}, indent=1))
    return 0 if verdict == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
