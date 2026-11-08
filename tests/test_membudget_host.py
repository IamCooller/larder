#!/usr/bin/env python3
"""MemBudget's saved host term: a short process never lowers a longer one's,
and a file without host_n is replaced by the first process. No model, no GPU."""
import json
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import stream_fast as sf

MB = 10**6


def make(calib):
    proj = types.SimpleNamespace(bits=4, group_size=64, mode="affine",
                                 _packed={"n_experts": 8, "stride": 64, "parts": []})
    glu = types.SimpleNamespace(_projs=[proj])
    return sf.MemBudget([glu], 6, 0.5, 2, 1, 1, None, None, calib=calib)


def run(calib, passes, host_mb):
    """A process of `passes` measured passes, its host reading host_mb."""
    b = make(calib)
    b.rate = b.state = 1.0
    sf.phys_footprint = lambda peak=False: host_mb * MB
    sf.mx.get_active_memory = lambda: 0
    sf.mx.get_cache_memory = lambda: 0
    sf.mx.get_peak_memory = lambda: 0
    for _ in range(passes):
        b._ticks += 1
        b._measure()
    b._save_calibration()
    return b


def saved(calib):
    return json.loads(open(calib).read())


with tempfile.TemporaryDirectory() as d:
    calib = os.path.join(d, "m.json")
    run(calib, 40, 700)
    assert saved(calib)["host_mb"] == 700 and saved(calib)["host_n"] == 32
    # A short process with a high host sizes its own room at it, but the
    # file keeps the long one's value.
    b = run(calib, 3, 2100)
    assert b._host_now() == 2100 * MB and saved(calib)["host_mb"] == 700
    run(calib, 40, 2100)
    run(calib, 3, 300)
    assert saved(calib)["host_mb"] == 2100, saved(calib)
    # A long process replaces it, up or down.
    run(calib, 40, 800)
    assert saved(calib)["host_mb"] == 800 and saved(calib)["host_n"] == 32
    # A file without host_n (today's inflated ones) is replaced at once.
    old = saved(calib)
    old["host_mb"] = 2111
    del old["host_n"]
    open(calib, "w").write(json.dumps(old))
    run(calib, 1, 650)
    assert saved(calib)["host_mb"] == 650, saved(calib)
print("ok")
