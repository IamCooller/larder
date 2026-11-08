#!/usr/bin/env python3
"""stream_fast._lm_whole: the old rule unchanged when the reader is not queued (res None), the corrected union cost
pre + max(win - pre, u_read * B / rate_u) on queued CUDA reads picks union at the off arm's 1,000 / 2,000 tails and
whole at a GPU-bound front, and a mutant that drops the residents' share or adds the full window back is caught.
The numbers are modelled on a measured layer-major read (B = 60.87 GB / 48 layers), not read from a file.

    perl -e 'alarm shift; exec @ARGV' 280 python3 tests/test_lm_rule.py
"""
import inspect
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import stream_fast as sf

B = 1.27e9
# name: (win, pre, routed share, residents' share, rate_u, rate_w, whole by the old rule, whole by the corrected one)
CASES = {
    "tail1000": (0.18, 0.07, 0.74, 0.16, 4.0e9, 4.0e9, True, False),
    "tail2000": (0.18, 0.07, 0.85, 0.15, 4.0e9, 4.0e9, True, False),
    "gpu front": (0.60, 0.20, 0.95, 0.15, 4.0e9, 4.0e9, True, True),
}


def call(fn, case, res):
    """fn's choice for layer 1 with the measures of `case`, res None for the old rule."""
    win, pre, u, _, ru, rw, _, _ = case
    rp = sf._LMRP
    keep = dict(rp), sf._RP_MODE, sf._layer_bytes
    rp.update(rate_u=ru, rate_w=rw, win=[win], pre=[pre], T=1000, prev=None, prev_pre=None, u0={1000: u})
    sf._RP_MODE, sf._layer_bytes = "auto", lambda glu: B
    try:
        return fn(None, 1, u, res)
    finally:
        rp.clear()
        rp.update(keep[0])
        sf._RP_MODE, sf._layer_bytes = keep[1], keep[2]


def mutant(old, new):
    src = inspect.getsource(sf._lm_whole)
    assert old in src, old
    ns = {}
    exec(src.replace(old, new), ns)
    # Run it in stream_fast's globals so the patched _layer_bytes and _RP_MODE apply.
    return types.FunctionType(ns["_lm_whole"].__code__, sf.__dict__)


def old_hand_cases():
    """Without a residents' share the choice is the old formula: whole iff max(win, B / rate_w) <= win + u * B / rate_u."""
    rp = sf._LMRP
    keep = dict(rp), sf._RP_MODE, sf._layer_bytes
    sf._RP_MODE, sf._layer_bytes = "auto", lambda glu: 1e9
    try:
        # B / rate = 1 s on both drives.
        for win, u, want in ((0.1, 0.0, False), (0.5, 0.5, True), (0.1, 0.5, False), (0.1, 0.95, True),
                             (2.0, 0.0, True), (0.0, 1.0, True)):
            rp.update(rate_u=1e9, rate_w=1e9, win=[win], T=1, prev=None, u0={})
            assert sf._lm_whole(None, 1, u) is want, (win, u, want)
        rp.update(rate_u=None)
        assert sf._lm_whole(None, 1, 0.5) is False
    finally:
        rp.clear()
        rp.update(keep[0])
        sf._RP_MODE, sf._layer_bytes = keep[1], keep[2]
    print("   old rule: hand cases unchanged")


def main():
    old_hand_cases()
    for name, c in CASES.items():
        old, new = call(sf._lm_whole, c, None), call(sf._lm_whole, c, c[3])
        print(f"   {name}: whole by the old rule {old}, by the corrected one {new}")
        assert (old, new) == (c[6], c[7]), name
    mutants = {
        "drops the residents' share": mutant("last_u * (1 - res)", "last_u"),
        "adds the full window back": mutant("pre + max(win - pre, ", "win + max(0.0, "),
    }
    for name, fn in mutants.items():
        bad = [k for k in ("tail1000", "tail2000") if call(fn, CASES[k], CASES[k][3]) is not CASES[k][7]]
        assert bad, f"mutant not caught: {name}"
        print(f"   mutant caught: {name} (wrong at {', '.join(bad)})")
    print("ok")


if __name__ == "__main__":
    main()
