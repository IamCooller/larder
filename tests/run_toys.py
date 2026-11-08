#!/usr/bin/env python3
"""Run every toy test (no weights) alone in a child process and print one line each.

    python3 tests/run_toys.py              # every test listed in tests/toys.txt
    python3 tests/run_toys.py store tui    # only the tests whose file name contains one of the words

A test that prints "skipped" or "SKIP" in its last line counts as skipped, not passed; it needs something
this machine lacks (a tokenizer, a CUDA device, a built extension), named in that line. Exit code 0 when
nothing failed. Tests that need a checkpoint are not here: tests/README.md lists them.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOY_TESTS = [line.strip() for line in (ROOT / "tests" / "toys.txt").read_text().splitlines() if line.strip()]


def run(rel, timeout):
    t0 = time.time()
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    try:
        p = subprocess.run([sys.executable, rel], cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "FAILED", time.time() - t0, f"timed out after {timeout:.0f} s"
    lines = (p.stdout + p.stderr).strip().splitlines()
    last = lines[-1] if lines else ""
    if p.returncode != 0:
        return "FAILED", time.time() - t0, last
    skipped = "skipped" in last.lower() or last.startswith("SKIP")
    return ("skipped" if skipped else "ok"), time.time() - t0, last


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("words", nargs="*")
    ap.add_argument("--timeout", type=float, default=900, help="seconds per test (default 900)")
    a = ap.parse_args()
    chosen = [t for t in TOY_TESTS if not a.words or any(w in t for w in a.words)]
    counts = {"ok": 0, "skipped": 0, "FAILED": 0}
    for rel in chosen:
        status, secs, last = run(rel, a.timeout)
        counts[status] += 1
        print(f"{status:8} {secs:6.1f}s  {rel}  {last[:100]}", flush=True)
    print(f"{counts['ok']} ok, {counts['skipped']} skipped, {counts['FAILED']} failed of {len(chosen)}")
    return 1 if counts["FAILED"] else 0


if __name__ == "__main__":
    sys.exit(main())
