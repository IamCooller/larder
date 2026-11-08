#!/usr/bin/env python3
"""A restarted server starts where the last one stopped: its prefilled front
comes back from SSD and, under --mem-peak-gb, its budget plans without the
probe. The real StreamingLLM, each run in a process of its own, on a tiny
synthetic MoE pack (synth_minimax.py: random weights, the real Qwen tokenizer
and chat template, so tools render at the front the way a coding agent's do).

  1. --mem-peak-gb: the first process builds the front and probes; the
     second one's first request restores that front (cached == the front),
     answers with the same tokens, and runs no probe.
  2. Default mode: the same, for the front.
  3. Another setting (cache_bias) is another directory: cold again.

A server with a 5.5k-token front once rebuilt it after every start: 79-123 s under
--mem-peak-gb 2 on a 16 GB Mac.

    MOE_TOY_TOKENIZER=CHECKPOINT_DIR python3 tests/test_persist.py   # skips without a tokenizer
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
QWEN = Path(os.environ.get("MOE_TOY_TOKENIZER") or "/nonexistent")      # skips without a tokenizer

RUN = r"""
import json, sys
from pathlib import Path
repo, slim, fronts, calib, kw = sys.argv[1:6]
sys.path.insert(0, repo)
import pcache, rowqmm
pcache.FRONTS, rowqmm.LAYOUT_DIR = Path(fronts), Path(calib)
from moe_stream import StreamingLLM
TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}]
MSG = [{"role": "system", "content": "You are terse."},
       {"role": "user", "content": "What is the weather in Paris?"}]
llm = StreamingLLM(model=slim, max_tokens=12, defer=False, **json.loads(kw))
reply = llm.ask(messages=MSG, tools=TOOLS)
s = llm.stats()
b = llm._budget
print(json.dumps({"reply": reply, "cached": s["cached_tokens"], "prompt": s["prompt_tokens"],
                  "probes": None if b is None else b.probes,
                  "dir": str(llm._front_root())}))
"""


def run(slim, fronts, calib, **kw):
    r = subprocess.run([sys.executable, "-c", RUN, str(REPO), str(slim), str(fronts),
                        str(calib), json.dumps(kw)],
                       capture_output=True, text=True, timeout=200)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def main():
    if not (QWEN / "tokenizer.json").exists():
        print(f"test_persist: skipped (no Qwen tokenizer at {QWEN})")
        return
    tmp = Path(tempfile.mkdtemp(prefix="persist_"))
    try:
        subprocess.run([sys.executable, str(REPO / "synth_minimax.py"), "--out",
                        str(tmp / "tiny_mm"), "--tokenizer-from", str(QWEN)],
                       check=True, capture_output=True, timeout=200)
        slim, fronts, calib = tmp / "tiny_mm-slim", tmp / "fronts", tmp / "calib"

        budget = dict(cache_bias=0.0, mem_peak_gb=4.0)
        first, again = run(slim, fronts, calib, **budget), run(slim, fronts, calib, **budget)
        front = first["prompt"] - again["prompt"]
        assert first["cached"] == 0 and first["probes"] == 1, first
        assert again["cached"] == front > 0, (first, again)
        assert again["reply"] == first["reply"] and again["probes"] == 0, (first, again)
        assert again["dir"] == first["dir"]
        assert len(list(calib.glob("membudget-*.json"))) == 1
        print(f"1. --mem-peak-gb: restart restored the {front}-token front, same reply, "
              f"no probe (first run: built it and probed)")

        plain, again = run(slim, fronts, calib, cache_bias=0.0), run(slim, fronts, calib, cache_bias=0.0)
        assert plain["dir"] != first["dir"], "a budget-mode front reached default mode"
        assert plain["cached"] == 0 and again["cached"] == front, (plain, again)
        assert again["reply"] == plain["reply"] and plain["probes"] is None
        print(f"2. default mode: its own directory; restart restored the {front}-token "
              f"front, same reply")

        other = run(slim, fronts, calib, cache_bias=0.5)
        assert other["dir"] not in (first["dir"], plain["dir"]) and other["cached"] == 0, other
        print("3. cache_bias 0.5: another directory, built cold")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("test_persist: ok")


if __name__ == "__main__":
    main()
