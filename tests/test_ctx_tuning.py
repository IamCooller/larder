#!/usr/bin/env python3
"""The online choosers under a growing context, no weights.

1. ArenaGovernor: a smaller size measured in a shorter context's log2 bucket does not settle the arena down,
   while a smaller size measured in the same bucket still does (the 10% rule); a settle made in an earlier bucket
   is undone at the next bucket, and one made in the same bucket holds.
2. ForcedPolicy: with each turn's context set, the faster arm wins a session whose context grows, where the old
   pooled spread made it alternate; with one context it decides exactly as before.
"""
import os
import random
import statistics
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import stream_fast as sf
import moe_stream as ms


class _Glu:
    def __init__(self, cap):
        self._cap, self._projs, self._slot_of = cap, [], object()

    def resize(self, n):
        self._cap = n


def _run(cap, rates, ctx_then, ctx_now, settled=False):
    """A governor at `cap` slots with `rates` on record from ctx_then, run 4 speed windows at 17 tok/s at ctx_now."""
    clock = types.SimpleNamespace(t=0.0)
    sf._sysctl_int = lambda name, width=4: {"hw.memsize": 17179869184, "kern.memorystatus_level": 50,
                                           "kern.memorystatus_vm_pressure_level": 1}.get(name)
    sf._vm_counts = lambda: (0.0, 0)
    sf.time = types.SimpleNamespace(perf_counter=lambda: clock.t, sleep=lambda s: None)
    g = sf.ArenaGovernor([_Glu(cap)], 21, 60, slot_gb=0.0629, top_k=8, period=64, step=16)
    g._held_gb = lambda: 3.6
    g._ctx_b = ctx_then.bit_length()
    g._rate = dict(rates)
    if settled:
        g._roof, g._settled = cap, True
    g._skip = False
    g.ctx = ctx_now
    for _ in range(64 * 4):
        clock.t += 1 / 17.0
        g.tick(False, 1)
    return g


def _settles(ctx_then, ctx_now):
    """Rate 20 tok/s on record at 21 slots from ctx_then; 60 slots now runs 17 tok/s at ctx_now."""
    return _run(60, {21: 20.0, 60: 0.0}, ctx_then, ctx_now)._settled


def _unsettles(ctx_then, ctx_now):
    """Settled at 21 slots in ctx_then's bucket, 60 having measured slower there; whether ctx_now re-probes upward."""
    g = _run(21, {21: 20.0, 60: 15.0}, ctx_then, ctx_now, settled=True)
    return not g._settled and g._roof == g.cap_max and g.cap > 21


def _session(pol, grow, seed=0):
    rng = random.Random(seed)
    picks = []
    for turn in range(40):
        k = 5 + (75 * turn / 39 if grow else 0)
        pol.ctx = int(k * 1000)
        arm = pol.pick()
        cost = {"mtp": 22.0 + 0.48 * k, "plain": 22.0 + 0.58 * k}[arm]
        pol.record(arm, cost * rng.gauss(1.0, 0.02))
        picks.append(arm)
    return picks


def main():
    real = sf._sysctl_int, sf._vm_counts, sf.time
    try:
        assert not _settles(20000, 70000), "a rate from a shorter context's bucket settled the arena"
        assert _settles(40000, 60000), "the 10% rule no longer settles within one bucket"
        assert _unsettles(30000, 40000), "a settle from a shorter context's bucket held after the bucket changed"
        assert not _unsettles(40000, 60000), "a settle within one bucket was undone"
    finally:
        sf._sysctl_int, sf._vm_counts, sf.time = real
    ms.note = lambda *a, **k: None

    picks = _session(ms.ForcedPolicy(("mtp", "plain")), grow=True)
    # From 20k on (turn 8) MTP is more than 2% faster in this model.
    share = sum(p == "mtp" for p in picks[8:]) / len(picks[8:])
    assert share >= 0.9, f"MTP on {share:.0%} of turns after 20k"

    # One context: the line is never fitted, so means and spread are the plain ones.
    pol = ms.ForcedPolicy(("mtp", "plain"))
    _session(pol, grow=False)
    means, spread = pol._stats()
    assert means == {a: statistics.fmean(pol.ms[a]) for a in pol.arms}
    assert spread == statistics.fmean([statistics.variance(pol.ms[a]) for a in pol.arms]) ** 0.5
    print("test_ctx_tuning: ok")


if __name__ == "__main__":
    main()
