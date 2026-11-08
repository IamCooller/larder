#!/usr/bin/env python3
"""CUDA's elastic memory (the launcher's --mem-peak-max-gb on CUDA) without a model or a GPU: stream_fast's
VramGovernor and TierGovernor on fake readings and a fake clock.

  1. The store on VRAM: it starts at the room the device leaves (with only the desktop on it, the fixed plan less
     the desktop, never more than the plan); another process's VRAM shrinks it at the next guard tick, eating into
     half the reserve halves it (macOS's warning), idle_tick shrinks it and rebuilds the store at once; with the VRAM
     given back the roof lifts and the speed lane grows it back half the way a window, never past the plan;
     set_elastic_limits moves the ceiling and the next guard tick follows it.
  2. The victim tier: other programs taking RAM shrink it at the next reading by a third of it (its install rule
     read live), a fall under the jitter does not move it, idle readings never grow it, it grows back only after a
     whole quiet window and half the way at a time, never past its install size, a disk swap rise halves it, and
     its own process's growth moves it only where MemAvailable would fall under the floor.

    perl -e 'alarm shift; exec @ARGV' 280 python3 tests/test_cuda_adaptive.py
"""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import stream_fast as sf

GB = 1e9


class _Budget:
    """MemBudget as the governor sees it."""

    def __init__(self, glus, peak):
        self.all, self.peak, self.resized, self.fits, self.stores = glus, peak, False, 0, {}

    def fit(self, rebuild):
        assert rebuild is False
        self.fits += 1


def check_store():
    total, slot = 17.10, 0.1189         # the PC's RTX 5060 Ti; K4: one 2.48 MB expert in each of 48 layers
    base, plan = 8.0, 16.0              # --mem-peak-gb (no pool) and --mem-peak-max-gb (today's fixed plan)
    state = {"others": 0.40}            # the desktop
    box, clock = [None], [0.0]
    proj = types.SimpleNamespace(_packed={"stride": int(slot * GB / 48)})
    glus = [types.SimpleNamespace(_slot_of=object(), _projs=[proj], _want=10) for _ in range(48)]
    slot = sum(p._packed["stride"] for g in glus for p in g._projs) / GB

    def info():
        own = box[0].store.peak / GB if box[0] else base
        return {"total_memory": total * GB, "free_memory": (total - state["others"] - own) * GB}
    real = sf.BACKEND, sf.mx, sf.vram_others, sf.time, sf._NATIVE, sf._GOV, sf._VRAM_OWN
    sf.BACKEND = "cuda"
    sf.mx = types.SimpleNamespace(device_info=info)
    sf.vram_others = lambda: state["others"] * GB
    sf.time = types.SimpleNamespace(perf_counter=lambda: clock[0], time=lambda: clock[0])
    sf._NATIVE = None
    try:
        b = _Budget(glus, base * GB)
        gov = box[0] = sf.make_store_governor(b, plan)
        assert isinstance(gov, sf.VramGovernor) and sf._VRAM_OWN
        top = int((total - 0.40 - (total - plan) - base) / slot)
        assert gov.cap == top and gov.cap_max == int((plan - base) / slot), (gov.cap, top)
        assert b.peak / GB + 0.40 <= plan + 1e-9 and not b.resized
        print(f"1a. start: {b.peak / GB:.2f} GB ({gov.cap} of {gov.cap_max} steps): the plan's {plan:g} GB "
              f"less the desktop's 0.40, reserve {gov._reserve_gb():.2f} GB (what the plan leaves free)")

        def tokens(k, s=0.1):
            for _ in range(k):
                clock[0] += s
                for _ in glus:
                    gov.tick(False, 1)
        tokens(128)
        assert gov.cap == top and not b.resized
        state["others"] = 0.90          # a browser takes 0.5 GB: the ceiling binds, no pressure
        tokens(2)
        assert gov.cap == top - 4 and b.resized, gov.cap
        b.resized = False
        state["others"] = 1.90          # a game's first GB: under half the reserve left, halved
        tokens(2)
        half = (top - 4) // 2
        assert gov.cap == half and b.resized, gov.cap
        state["others"] = 4.90          # it holds 4.5 GB: the ceiling again
        tokens(2)
        low = int((total - 4.90 - base - (total - plan)) / slot)
        assert gov.cap == low, (gov.cap, low)
        assert b.peak / GB + 4.90 <= plan + 1e-9
        print(f"1b. others 0.4 -> 0.9 GB: {top} -> {top - 4} steps; 1.9: halved to {half}; "
              f"4.9: {low} (peak {b.peak / GB:.2f} GB), each at the next guard tick")
        # Idle: the store moves at once (MemBudget.fit), never under the base.
        b.resized = False
        state["others"] = 9.5
        gov.idle_tick()
        assert gov.cap == 0 and b.peak == base * GB and b.fits == 1, gov.cap
        print("1c. idle, others 9.5 GB: to the base (no pool) and rebuilt at once")
        # The VRAM back: the roof lifts, and with it the dip the game left in the reserve across a prefill; the speed
        # lane grows it half the way a window.
        gov._swings.append(4.5)
        state["others"] = 0.40
        caps = []
        for _ in range(12):
            tokens(64)
            caps.append(gov.cap)
        assert caps[0] == 0 and 0 < caps[1] < caps[-1] <= top, caps
        assert all(x <= y for x, y in zip(caps, caps[1:])) and caps[-1] == top, caps
        assert b.peak / GB + 0.40 <= plan + 1e-9
        print(f"1d. others back to 0.40 GB: the store a window at a time {caps} (64 tokens each), "
              f"never past the plan")
        # set_elastic_limits: the one place the ceiling moves; the lanes follow it down at once.
        sf.set_elastic_limits(store_max_gb=12.0)
        tokens(2)
        assert gov.cap == gov.cap_max == int((12.0 - base) / slot) and b.peak / GB <= 12.0 + 1e-9, gov.cap
        print(f"1e. set_elastic_limits(store_max_gb=12): {gov.cap} steps at the next guard tick")
        # No NVML: nothing says when to give back, so the store keeps the fixed plan, never the launcher's floor.
        sf.vram_others = lambda: None
        b2 = _Budget(glus, base * GB)
        b2.stores = {0: object()}
        assert sf.make_store_governor(b2, plan) is None
        assert abs(b2.peak / GB - plan) < 1e-9 and b2.resized, b2.peak
        print(f"1f. no NVML: no governor, the store fixed at the plan's {plan:g} GB")
    finally:
        sf.BACKEND, sf.mx, sf.vram_others, sf.time, sf._NATIVE, sf._GOV, sf._VRAM_OWN = real


def check_tier():
    slot = 2.48e6
    n_max = int((9.0e9 - 2.5e9) / 3 // slot)         # installed at MemAvailable 9.0 GB
    state = {"avail": 9.0e9 - n_max * slot, "mine": 0, "swap": 0}
    tg = sf.TierGovernor(n_max, n_max, slot, 2.5e9, read=lambda: (state["avail"], state["mine"], state["swap"]))
    assert tg.target(state["avail"]) == n_max
    # This process's own growth since the install is not another program's: 2 GB of it leaves the tier as it is.
    state["avail"] -= 2.0e9
    state["mine"] = 2.0e9
    assert tg.tick(idle=True) is None and tg.target(state["avail"], state["mine"]) == n_max

    def tokens(k):
        out = None
        for _ in range(k):
            w = tg.tick(1)
            if w is not None:
                state["avail"] -= (w - tg.n) * slot
                tg.n = w
                out = w
        return out
    assert tokens(100) is None
    state["avail"] -= 0.3e9                          # a third is 0.1 GB: jitter
    assert tokens(4) is None and tg.n == n_max
    state["avail"] -= 2.7e9                          # 3 GB in all: a GB back at the next reading
    tokens(2)
    assert abs(tg.n - int((9.0e9 - 3.0e9 - 2.5e9) / 3 // slot)) <= 1, tg.n
    shrunk = tg.n
    print(f"2a. MemAvailable -3 GB: the tier {n_max * slot / 1e9:.2f} -> {tg.n * slot / 1e9:.2f} GB at the next "
          f"reading (a third of the fall); -0.3 GB alone did not move it")
    state["avail"] += 3.0e9                          # the 3 GB back
    for _ in range(5):
        w = tg.tick(idle=True)
        assert w is None, w
    assert tg.n == shrunk
    tokens(63)
    assert tg.n == shrunk                            # not before a whole window
    steps = []
    for _ in range(6):
        tokens(64)
        steps.append(tg.n)
    assert shrunk < steps[0] < n_max and steps[-1] == n_max and steps == sorted(steps), steps
    print(f"2b. back: idle readings never grow it; then half the way a 64-token window: "
          f"{[round(s * slot / 1e9, 2) for s in steps]} GB, up to its install size")
    state["swap"] += 100
    tokens(2)
    assert tg.n == n_max // 2, tg.n
    print(f"2c. disk swap grows: halved to {tg.n * slot / 1e9:.2f} GB")
    # Its own growth still never runs MemAvailable under the floor: 4 GB more of it, and the tier gives way.
    state["mine"] += 4.0e9
    state["avail"] -= 4.0e9
    tokens(2)
    assert state["avail"] >= 2.5e9 - slot and tg.n < n_max // 2, (state["avail"], tg.n)
    print(f"2d. 4 GB more of its own: MemAvailable kept at {state['avail'] / 1e9:.2f} GB (floor 2.5), "
          f"the tier {tg.n * slot / 1e9:.2f} GB; 2 GB of its own growth at idle did not move it")


def main():
    check_store()
    check_tier()
    print("test_cuda_adaptive: ok")


if __name__ == "__main__":
    main()
