#!/usr/bin/env python3
"""wait_for hands the gather the slot table the agent has just written, even
when the GPU read that table earlier in the same command buffer (no model).

Each round is one command buffer: a kernel reads the whole table (its lines
go into the GPU's caches), signal_after hands K ids the table does not hold
to the agent, the agent claims slots for them (host stores into the table)
and reads their rows into the arena, and take(wait_for(table), ids) must give
those slots and gather those rows. Stage 2 as merged (wait_for an alias of
the table, plain loads) fails it: on an M4 Mac (2026-10-10) all 3000 rounds
read the old -1s, and 0 of 3000 once the first read is left out (the
engine's single-row pass reads each table once per command buffer, which is
why stage 2's replies were identical). With the coherent copy: 0 and 0.

    python3 tests/test_native_coherent.py [dir holding moe_native]
"""
import os
import sys
import tempfile
from pathlib import Path

if len(sys.argv) > 1:
    sys.path.insert(0, sys.argv[1])
else:
    sys.path.append(str(Path(__file__).resolve().parents[1] / "native" / "build"))

import numpy as np
import mlx.core as mx
try:
    import moe_native as nv
except ImportError:
    print("test_native_coherent: skipped (needs moe_native built by native/build.sh)")
    sys.exit(0)

N_EXP, CAP, K, ROW, ROUNDS = 256, 16, 8, 64, 3000


def main():
    rng = np.random.default_rng(0)
    rows = rng.integers(0, 256, (N_EXP, ROW), dtype=np.uint8)
    with tempfile.TemporaryFile() as f:
        f.write(rows.tobytes())
        f.flush()
        table = mx.full((N_EXP,), -1, dtype=mx.int32)
        arena = mx.zeros((CAP, ROW), dtype=mx.uint8)
        mx.eval(table, arena)
        ev = nv.SharedEvent()
        agent = nv.Agent(ev, f.fileno(), 1, 2)
        agent.set_layer(0, table, [(0, ROW, [arena])])
        agent.set_state(0, [])
        host = np.array(table, copy=False)        # the agent's table, in place
        v, stale, bad_rows = 1, 0, 0
        for r in range(ROUNDS):
            absent = np.flatnonzero(host < 0)
            want = np.sort(rng.choice(absent, K, replace=False)).astype(np.int32)
            ids = mx.array(want)
            seen = mx.sum(table)                  # the GPU reads the table first
            sig = nv.signal_after(ids + seen * 0, ev, v, agent, 0)
            slots = mx.take(nv.wait_for(table, sig, ev, v + 1), ids)
            got = mx.take(arena, mx.maximum(slots, 0), axis=0)
            mx.eval(slots, got)
            assert agent.drain(5.0) and not agent.error, agent.error
            v += 2
            if not np.array_equal(np.array(slots), host[want]):
                stale += 1
            elif not np.array_equal(np.array(got), rows[want]):
                bad_rows += 1
        print(f"{ROUNDS} rounds: {stale} saw a stale slot table, {bad_rows} stale arena rows")
        assert stale == 0 and bad_rows == 0
    print("ok")


if __name__ == "__main__":
    main()
