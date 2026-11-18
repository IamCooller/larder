#!/usr/bin/env python3
"""MOE_WIRED_HOLD with no model: on by default, the wired limit held from load, dropped by the idle tick at pressure 2,
re-armed at the next request; with MOE_WIRED_HOLD=0, mlx_lm's window a reply as before.

    perl -e 'alarm shift; exec @ARGV' 280 python3 tests/test_wired_hold.py
"""
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx                                   # noqa: E402
import mlx.nn as nn                                     # noqa: E402
from mlx_lm.generate import generation_stream, wired_limit  # noqa: E402

import stream_fast as sf                                # noqa: E402
from server import ModelHost                            # noqa: E402

MAX = mx.device_info()["max_recommended_working_set_size"]
MACHINE = {"kern.memorystatus_level": 60, "kern.memorystatus_vm_pressure_level": 1, "hw.memsize": 17179869184}


class _Glu:
    def __init__(self, cap):
        self._cap, self._projs, self._slot_of = cap, [], object()

    def resize(self, n):
        self._cap = n


def limit():
    """MLX's wired limit now."""
    old = mx.set_wired_limit(0)
    mx.set_wired_limit(old)
    return old


def reply(model):
    """A reply's wired window as spec.py and spec_draft.py open it: the limit inside, then after."""
    with wired_limit(model, [generation_stream]):
        inside = limit()
    return inside, limit()


def session(hold):
    """Load, a reply, an idle tick at pressure 2, a request at pressure 2, then one at pressure 1: the limit at each."""
    sf._WIRED_HOLD, sf._WIRED = hold, None
    mx.set_wired_limit(0)
    MACHINE["kern.memorystatus_vm_pressure_level"] = 1
    sf._GOV = sf.ArenaGovernor([_Glu(60) for _ in range(4)], 6, 96, slot_gb=0.063, top_k=8)
    host = ModelHost.__new__(ModelHost)
    host._lock, host._asks = threading.Lock(), 0
    model = nn.Linear(4, 4)
    seen = {}
    sf.wired_arm()                                      # what ModelHost._load calls after the governor
    seen["load"] = limit()
    seen["reply"] = reply(model)
    sf._GOV.idle_tick()
    seen["idle_quiet"] = limit()
    MACHINE["kern.memorystatus_vm_pressure_level"] = 2
    sf._GOV.idle_tick()
    seen["idle_press2"], seen["cap_press2"] = limit(), sf._GOV.cap
    host._begin(time.perf_counter())
    seen["request_press2"] = limit()
    seen["reply_press2"] = reply(model)
    MACHINE["kern.memorystatus_vm_pressure_level"] = 1
    host._begin(time.perf_counter())
    seen["request_press1"] = limit()
    seen["reply_press1"] = reply(model)
    return seen


def main():
    if "MOE_WIRED_HOLD" not in os.environ:
        assert sf._WIRED_HOLD, "MOE_WIRED_HOLD must be on by default"
    real = sf._sysctl_int, sf._vm_counts, sf._WIRED_HOLD, sf._GOV
    sf._sysctl_int = lambda name, width=4: MACHINE.get(name)
    sf._vm_counts = lambda: (0.5, 0)
    try:
        on = session(True)
        assert on["load"] == MAX, on
        assert on["reply"] == (MAX, MAX), f"the held window must stay open after the reply: {on}"
        assert on["idle_quiet"] == MAX, f"a quiet idle tick must keep the hold: {on}"
        assert on["idle_press2"] == 0, f"pressure 2 must drop the hold: {on}"
        assert on["cap_press2"] == 30, f"pressure 2 still halves the arena: {on}"
        assert on["request_press2"] == 0, f"no re-arm while the pressure is still 2: {on}"
        assert on["reply_press2"] == (MAX, 0), f"a dropped hold leaves today's window: {on}"
        assert on["request_press1"] == MAX, f"the next request at pressure 1 re-arms: {on}"
        assert on["reply_press1"] == (MAX, MAX), on

        off = session(False)
        assert off["load"] == 0 and off["idle_quiet"] == 0 and off["idle_press2"] == 0, off
        assert off["request_press2"] == 0 and off["request_press1"] == 0, off
        for k in ("reply", "reply_press2", "reply_press1"):
            assert off[k] == (MAX, 0), f"flag off: mlx_lm's window a reply, as today: {off}"
        assert off["cap_press2"] == 30, off

        # The budget store (--mem-peak-gb) and a server without the arena governor keep today's windows with the flag on.
        sf._WIRED_HOLD, sf._WIRED = True, None
        mx.set_wired_limit(0)
        sf._GOV = None
        sf.wired_arm()
        assert limit() == 0, "no arena governor: no hold"
        sf._GOV = g = sf.ArenaGovernor([_Glu(60)], 6, 96, slot_gb=0.063, top_k=8)
        g.store = object()
        sf.wired_arm()
        assert limit() == 0, "a store governor: no hold"
    finally:
        sf._sysctl_int, sf._vm_counts, sf._WIRED_HOLD, sf._GOV = real
        sf._WIRED = None
        mx.set_wired_limit(0)
    print(f"test_wired_hold: ok (held {MAX / 1e9:.2f} GB, dropped at pressure 2, re-armed at the next request)")


if __name__ == "__main__":
    main()
