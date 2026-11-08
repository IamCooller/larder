#!/usr/bin/env python3
"""The elastic store (server.py --mem-peak-max-gb, stream_fast.make_store_governor)
without a model: test_store's toy packed MoE and fake memory signals.

  1. cache_bias 0: greedy tokens with the store resized mid-generation by the
     governor (its _resize raising the budget's peak to the max, then dropping
     it to the base, every few decode passes) equal stock mlx_lm's, sync and
     run-ahead. The store was rebuilt bigger and smaller each time: a pure cache.
  2. The governor on a store (fake MemBudget, fake sysctl, fake clock): the
     --reserve-pct share is its reserve's floor (its own prefill dip reads ~0); it starts
     at the base; on a quiet machine it grows only from the speed lane, half the
     room first, then to the max; at 0.3 s a token (over the arena's fixed 0.25 s
     cap) it still grows; a falling margin shrinks it within one guard tick, a
     pressure warning halves it, a swapout steps it down under a roof; idle_tick
     shrinks it and rebuilds the store at once; it never goes under the base. An
     idle reading before the model's first pass is no reference for the first
     prefill's dip, and a store one step under its max takes that last step.
  3. server.py refuses --mem-peak-max-gb without a smaller --mem-peak-gb (before
     any load).
  4. A store rebuilt in place of a dropped one is built only once the footprint
     is back to what its pool was sized from (MemBudget._settle), with a bound.
  5. MOE_STORE_GROW: the store built at its max at the first decode pass after
     the prefill, tokens equal stock's (2g: grow_now's rule).

    perl -e 'alarm shift; exec @ARGV' 280 python3 tests/test_store_resize.py
"""
import os
import subprocess
import sys
import tempfile
import shutil
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import mlx.core as mx
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache

import stream_fast as sf
import test_store as ts

N_TOK = 40


def resized_run(out, defer):
    """Greedy tokens of PROMPTS[0] with the store grown and shrunk by the
    governor every few decode passes; the pools seen."""
    model, bud = ts.engine(out, peak_gb=ts.budget_gb(), defer=defer)
    unit = sum(p._packed["stride"] for g in bud.all for p in g._projs)
    real, n, pools, gov = bud.tick, [0], [], [None]

    def tick(pos, fresh, caches):
        if pos * bud.top_k < sf.SORT_MIN and bud.rate is not None:
            n[0] += 1
            if gov[0] is None:
                # The base: ~4 slots a layer beside the standing room
                # (test_store's turn 2), so a bigger peak really grows it.
                sf._eval_caches(caches)
                bud.copy = sf._copy(caches)
                held, host, _ = bud._measure()
                bud.peak = (held + host + bud.cache + bud.ring_bytes
                            + bud._need(bud.min_w - 1, False, False) + 4 * unit)
                gov[0] = sf.make_store_governor(bud, (bud.peak + 12 * unit) / 1e9)
                sf._GOV = None          # resizes come from here, not the machine
            elif n[0] % 8 == 2:
                gov[0]._resize(gov[0].cap_max, 50, 1)
            elif n[0] % 8 == 6:
                gov[0]._resize(0, 50, 1)
        r = real(pos, fresh, caches)
        if gov[0] is not None:
            pools.append(sum(st.pool for st in bud.stores.values()))
        return r
    bud.tick = tick
    toks = [int(t) for t, _ in generate_step(
        mx.array(ts.PROMPTS[0], dtype=mx.int32), model, max_tokens=N_TOK,
        prompt_cache=make_prompt_cache(model))]
    g = gov[0]
    assert g is not None and g.cap_max == 12 and len(g.history) >= 6, g and g.history
    rebuilds = bud.rebuilds
    del model, bud, tick, real, gov, g
    ts.settle()
    return toks, pools, rebuilds


def check_tokens(out):
    ref = [int(t) for t, _ in generate_step(
        mx.array(ts.PROMPTS[0], dtype=mx.int32), ts.stock_model(out), max_tokens=N_TOK,
        prompt_cache=make_prompt_cache(ts.stock_model(out)))]
    ts.settle()
    for defer in (False, True):
        toks, pools, rebuilds = resized_run(out, defer)
        name = "run-ahead" if defer else "sync"
        assert toks == ref, (name, toks, ref)
        moves = [b - a for a, b in zip(pools, pools[1:]) if b != a]
        ups, downs = sum(m > 0 for m in moves), sum(m < 0 for m in moves)
        assert ups >= 2 and downs >= 2 and rebuilds >= 6, (
            name, pools, rebuilds)
        print(f"1. {name}, cache_bias 0: {len(toks)} tokens == stock with the store "
              f"grown {ups}x and shrunk {downs}x mid-generation ({rebuilds} rebuilds, "
              f"pool {min(pools)}..{max(pools)} slots)")


class _Store:
    """MemBudget as the governor sees it."""
    def __init__(self, peak):
        self.peak, self.resized, self.fits = peak, False, 0

    def fit(self, rebuild):
        assert rebuild is False
        self.fits += 1


def check_governor():
    memsize = 16 * 2**30
    total = memsize / 1e9
    slot = 0.147                    # Flash-Next: one 3.07 MB expert in each of 48 layers
    # "free": the swap-line margin with the store at its base; the store's
    # own steps come off it, as on a real machine.
    state = {"free": 6.0, "press": 1, "comp": 1.0, "swap": 0}
    box = [None]
    real = sf._sysctl_int, sf._vm_counts, sf.time
    clock = [0.0]

    def sysctl(name, width=4):
        if name == "kern.memorystatus_level":   # margin = level x total / 100 - 2/3 x comp
            margin = state["free"] - box[0].cap * slot
            return (margin + sf._SWAP_SHARE * state["comp"]) / total * 100
        return {"hw.memsize": memsize,
                "kern.memorystatus_vm_pressure_level": state["press"]}.get(name)
    sf._sysctl_int = sysctl
    sf._vm_counts = lambda: (state["comp"], state["swap"])
    sf.time = types.SimpleNamespace(perf_counter=lambda: clock[0])
    hold0 = sf._HOLD
    try:
        glus = [types.SimpleNamespace(_slot_of=object(), _projs=[], _want=10)
                for _ in range(48)]
        st = _Store(6e9)
        gov = box[0] = sf.ArenaGovernor(glus, 0, 13, reserve_pct=15, slot_gb=slot, store=st)
        assert gov.cap == 0 and gov.cap_min == 0
        # A store's own prefill dip reads ~0: the --reserve-pct share stays the floor.
        gov._swings.append(0.0)
        assert abs(gov._reserve_gb() - total * 0.15) < 1e-9
        gov._swings.append(4.0)
        assert gov._reserve_gb() == 4.0
        gov._swings.clear()
        # R1 (MOE_PF_R1): a prefill is planned at the peak the governor would grow the store to
        # now -- the margin over its reserve, under its roof and max -- and never under the base.
        r1, g0 = sf._R1, sf._GOV
        try:
            sf._GOV, sf._R1 = gov, False
            assert sf.MemBudget._prefill_peak(st) == st.peak
            sf._R1 = True
            want = lambda free: st.peak + min(13, int((free - total * 0.15) / slot)) * slot * 1e9
            for free in (6.0, 3.0, 2.0):
                state["free"] = free
                assert abs(sf.MemBudget._prefill_peak(st) - max(st.peak, want(free))) < 1, free
            state["free"], gov._roof = 6.0, 5
            assert abs(sf.MemBudget._prefill_peak(st) - (st.peak + 5 * slot * 1e9)) < 1
            gov._roof = gov.cap_max
            sf._GOV = None
            assert sf.MemBudget._prefill_peak(st) == st.peak
        finally:
            sf._R1, sf._GOV = r1, g0
            state["free"] = 6.0

        def tokens(k, s=0.3):
            for _ in range(k):
                clock[0] += s
                for _ in glus:
                    gov.tick(False, 1)
        tokens(64)          # the first speed window measures, nothing moves
        assert gov.cap == 0 and st.peak == 6e9 and not st.resized
        tokens(64)
        # room (6.0 - reserve 2.58) / 0.147 = 23 steps; the first growth takes half
        assert gov.cap == 11 and st.resized, gov.cap
        assert abs(st.peak - (6e9 + 11 * slot * 1e9)) < 1
        st.resized = False
        tokens(128)         # a window to settle at the new size, then the next step
        assert gov.cap == 13 and st.resized, gov.cap
        assert abs(st.peak - (6e9 + 13 * slot * 1e9)) < 1
        print(f"2a. quiet machine, 0.3 s a token: the base for 128 tokens, then 11, "
              f"then 13 steps at 256 (peak {st.peak / 1e9:.2f} GB), from the speed lane")
        st.resized = False
        tokens(192, s=0.45)  # code after prose: slower windows at 13 than at 11 and 0
        assert gov.cap == 13 and not st.resized and not gov._settled, gov.cap
        print("2a. windows 1/3 slower at the max (another workload) do not send it back")
        # A hog: the margin falls, the ceiling binds at the next guard tick.
        st.resized = False
        state["free"] = 5.0          # room 16 steps: stays
        tokens(2)
        assert gov.cap == 13
        state["free"] = 3.4          # room 5 steps
        tokens(2)
        assert gov.cap == 5 and st.resized, gov.cap
        state["press"] = 2
        tokens(2)
        assert gov.cap == 2, gov.cap
        state["press"] = 1
        state["swap"] += 10
        tokens(2)
        assert gov.cap == 0 and st.peak == 6e9, gov.cap
        print(f"2b. the margin falls to 3.4 GB: 13 -> 5 steps in one guard tick; "
              f"pressure 2: 2; a swapout: 0, the base ({st.peak / 1e9:.0f} GB)")
        # Idle: a grown store shrinks and is rebuilt at once.
        st2 = _Store(6e9)
        gov2 = box[0] = sf.ArenaGovernor(glus, 0, 13, reserve_pct=15, slot_gb=slot, store=st2)
        gov2.cap, st2.peak = 13, 6e9 + 13 * slot * 1e9
        state["free"] = 3.4
        gov2.idle_tick()
        assert gov2.cap == 5 and st2.fits == 1 and not st2.resized, (gov2.cap, st2.fits)
        state["free"] = 1.0
        gov2.idle_tick()
        assert gov2.cap == 0 and st2.peak == 6e9 and st2.fits == 2, gov2.cap
        print(f"2c. idle: 13 -> 5 -> 0 steps as the margin falls, the store rebuilt "
              f"at once each time ({st2.fits} fits); never under the base")
        # 2d. Idle ticks before the first pass read a high level (no store yet, the skeleton not
        # in it) that is no reference for the first prefill's dip, and "half the room first"
        # must not land one small step short of the last.
        st3 = _Store(6e9)
        gov3 = box[0] = sf.ArenaGovernor(glus, 0, 13, reserve_pct=15, slot_gb=slot, store=st3)
        state.update(free=11.9, press=1)
        gov3.idle_tick()
        state["free"] = 6.0

        def ticks(k, sort=False):
            for _ in range(k):
                clock[0] += 0.3
                for _ in glus:
                    gov3.tick(sort, 1)
        ticks(1, sort=True)          # the first prefill
        state["free"] = 6.18         # room 24 steps: half is 12
        ticks(128)
        assert abs(gov3._reserve_gb() - total * 0.15) < 1e-9, gov3._reserve_gb()
        assert gov3.cap == 12, gov3.cap
        ticks(128)
        assert gov3.cap == 13 and abs(st3.peak - (6e9 + 13 * slot * 1e9)) < 1, gov3.cap
        state["free"] = 5.0
        ticks(1, sort=True)          # a later prefill's own dip still counts
        state["free"] = 6.18
        ticks(2)
        assert len(gov3._swings) == 1 and gov3._swings[0] > 1.0, list(gov3._swings)
        print(f"2d. an idle 11.9 GB reading before the first pass is no reference (reserve "
              f"{total * 0.15:.2f} GB, the floor); 12 steps, then the last 147 MB one: 13; "
              f"a later prefill's {gov3._swings[0]:.2f} GB dip is measured")
        # 2e-2f: MOE_STORE_HOLD, its own tests.
        sf._HOLD = True
        # 2e. A growth at a reply's last token must not raise the next prefill's peak into a
        # false dip, and a prefill-learned reserve must not shrink a grown store, since only
        # the floor shrinks a store and the reserve bounds growth.
        def fresh():
            box[0] = sf.ArenaGovernor(glus, 0, 13, reserve_pct=15, slot_gb=slot,
                                      store=_Store(6e9))
            return box[0]

        def ticks4(k, sort=False):
            for _ in range(k):
                clock[0] += 0.3
                for _ in glus:
                    box[0].tick(sort, 1)
        gov4 = fresh()
        state.update(free=8.0, press=1)
        ticks4(1, sort=True)         # the first prefill, unmeasured
        ticks4(128)                  # grows at its last token: room 36 steps, half 18
        assert gov4.cap == 13, gov4.cap
        ticks4(1, sort=True)         # the next prefill, at the new peak
        ticks4(2)
        assert list(gov4._swings) == [0.0], list(gov4._swings)
        state["free"] = 1.5
        ticks4(1, sort=True)         # a prefill that read 6.5 GB under its quiet reading
        state["free"] = 7.5          # margin 5.6 GB at 13 steps, under the 6.5 GB reserve
        ticks4(64)
        gov4.idle_tick()
        assert gov4.cap == 13 and abs(gov4._reserve_gb() - 6.5) < 1e-9, gov4.cap
        state["free"] = 3.4          # the machine lost room: the floor binds
        ticks4(2)
        assert gov4.cap == 5, gov4.cap
        state["free"] = 8.0          # back, but the reserve still bounds growth: 10 steps
        ticks4(512)
        assert 5 < gov4.cap <= 10, gov4.cap
        print(f"2e. MOE_STORE_HOLD: a growth at a reply's end is no dip of the next prefill; a 6.5 GB "
              f"reserve holds 13 steps at a 5.6 GB margin (decode and idle), the floor takes 13 -> 5 "
              f"at 3.4 GB, and growth stops at the reserve's {gov4.cap} steps")
        # 2f. After a reply macOS can let go of the server's GPU wiring so the idle margin reads
        # high, and the next prefill, wiring it back, must not read as a dip.
        gov5 = fresh()
        state["free"] = 8.0
        ticks4(1, sort=True)
        ticks4(128)
        state["free"] = 15.0
        gov5.idle_tick()
        state["free"] = 8.0
        ticks4(1, sort=True)
        ticks4(2)
        assert gov5.cap == 13 and list(gov5._swings) == [0.0], (gov5.cap, list(gov5._swings))
        print("2f. MOE_STORE_HOLD: an idle reading 7 GB over the reply's (its GPU wiring let go) is no "
              "reference: the next prefill's dip reads 0")
        # 2g. MOE_STORE_GROW: grow_now, at the first decode pass after a prefill, takes the speed
        # lane's step at once (half the room first, this prefill's dip in the reserve), and
        # nothing under pressure, after swapouts or at the max.
        gov6 = fresh()
        state.update(free=8.0, press=1)
        ticks4(1, sort=True)         # a fresh server's first prefill, unmeasured
        assert gov6.grow_now() and gov6.cap == 13 and gov6.store.resized, gov6.cap
        assert abs(gov6.store.peak - (6e9 + 13 * slot * 1e9)) < 1
        assert not gov6.grow_now(), "grew past the max"
        gov7 = fresh()
        state["free"] = 6.18         # room 24 steps: half is 12
        ticks4(2)
        ticks4(1, sort=True)
        assert gov7.grow_now() and gov7.cap == 12, gov7.cap
        gov8 = fresh()
        ticks4(2)
        state["free"] = 4.0          # the prefill reads 2.18 GB under its quiet reading
        ticks4(1, sort=True)
        state["free"] = 6.18         # a dip under the 2.58 GB floor: the floor binds, 12 steps
        assert gov8.grow_now() and gov8.cap == 12, gov8.cap
        gov9 = fresh()
        ticks4(2)
        state["free"] = 2.0          # a 4.18 GB dip: room 13 steps, half 6
        ticks4(1, sort=True)
        state["free"] = 6.18
        assert gov9.grow_now() and gov9.cap == 6, gov9.cap
        for change in ({"press": 2}, {"swap": state["swap"] + 5}):
            gov10 = fresh()
            ticks4(2)
            ticks4(1, sort=True)
            state.update(change)
            assert not gov10.grow_now() and gov10.cap == 0, (change, gov10.cap)
            state["press"] = 1
        print(f"2g. MOE_STORE_GROW: after a prefill the store grows at once to 13 / 12 steps "
              f"(8.0 / 6.18 GB free), to 6 behind a 4.18 GB prefill dip; not under pressure, "
              f"after swapouts or at the max")
    finally:
        sf._HOLD = hold0
        sf._sysctl_int, sf._vm_counts, sf.time = real


def check_settle(out):
    """4. A store built in place of a dropped one waits until the footprint is back
    to what the pool was sized from (macOS takes a freed Metal buffer's pages
    back shortly after MLX drops it): with the
    old store's bytes held in the footprint for 5 reads the new one is built after
    them; held for good, the wait gives up at its 0.5 s bound."""
    model, bud = ts.engine(out, peak_gb=ts.budget_gb())
    list(generate_step(mx.array(ts.PROMPTS[0], dtype=mx.int32), model, max_tokens=2,
                       prompt_cache=make_prompt_cache(model)))      # a measured rate
    unit = sum(p._packed["stride"] for g in bud.all for p in g._projs)
    held, host, _ = bud._measure()
    base = held + host + bud.cache + bud.ring_bytes + bud._need(bud.min_w - 1, False, False)
    bud.peak = base + 2 * unit
    bud.fit(rebuild=True)                   # a store whose size follows the peak
    real_fp, real_settle, real_store = sf.phys_footprint, bud._settle, sf.ExpertStore.__init__
    st = {"lag": 0, "left": 0, "reads": 0, "seen": []}

    def fp():
        st["reads"] += 1
        st["left"] -= 1
        held = st["lag"] if st["left"] >= 0 else 0
        return mx.get_active_memory() + mx.get_cache_memory() + bud._host_top + held

    def store_init(self, *a):
        st["seen"].append(st["left"])           # reads of the held bytes still to come
        real_store(self, *a)
    sf.phys_footprint, sf.ExpertStore.__init__ = fp, store_init
    try:
        for hold, bound in ((5, False), (10**9, True)):
            dropped = sum(s.nbytes for s in bud.stores.values())
            st.update(lag=dropped, left=0, seen=[])
            bud._settle = lambda host: (st.update(left=hold), real_settle(host))
            bud.peak = base + (4 if bound else 6) * unit
            t = time.perf_counter()
            bud.fit(rebuild=True)
            took = time.perf_counter() - t
            assert st["seen"] and (st["seen"][0] < 0) != bound, st
            n = len(bud.stores)
            assert (0.5 * n <= took < 0.5 * n + 0.5) if bound else took < 0.4, (took, n)
        print(f"4. a rebuild built its store only once the dropped one's bytes left the "
              f"footprint; held for good, each of its {len(bud.stores)} stores went ahead "
              f"after the 0.5 s bound ({took:.2f} s)")
    finally:
        sf.phys_footprint, sf.ExpertStore.__init__ = real_fp, real_store
        del model, bud
        ts.settle()


def check_grow(out):
    """5. MOE_STORE_GROW on the toy: with a roomy machine the store governor builds the
    pool at its max at the first decode pass after the prompt's prefill (off: it stays at
    the base for these 40 tokens), and greedy tokens equal stock's."""
    model = ts.stock_model(out)
    ref = [int(t) for t, _ in generate_step(mx.array(ts.PROMPTS[0], dtype=mx.int32), model,
                                           max_tokens=N_TOK, prompt_cache=make_prompt_cache(model))]
    del model
    ts.settle()
    real = sf._sysctl_int, sf._vm_counts, sf._GROW
    sf._sysctl_int = lambda name, width=4: {
        "kern.memorystatus_level": 90, "kern.memorystatus_vm_pressure_level": 1,
        "hw.memsize": 16 * 2**30}.get(name)
    sf._vm_counts = lambda: (0.0, 0)
    first = {}
    try:
        for grow in (False, True):
            sf._GROW = grow
            model, bud = ts.engine(out, peak_gb=ts.budget_gb())
            unit = sum(p._packed["stride"] for g in bud.all for p in g._projs)
            real_tick, pools, gov = bud.tick, [], [None]

            def tick(pos, fresh, caches):
                if gov[0] is None and pos * bud.top_k < sf.SORT_MIN and bud.rate is not None:
                    # The base as check_tokens', the governor in place before the first decode pass.
                    sf._eval_caches(caches)
                    bud.copy = sf._copy(caches)
                    held, host, _ = bud._measure()
                    bud.peak = (held + host + bud.cache + bud.ring_bytes
                                + bud._need(bud.min_w - 1, False, False) + 4 * unit)
                    gov[0] = sf.make_store_governor(bud, (bud.peak + 12 * unit) / 1e9)
                r = real_tick(pos, fresh, caches)
                if gov[0] is not None:
                    pools.append(sum(st.pool for st in bud.stores.values()))
                return r
            bud.tick = tick
            toks = [int(t) for t, _ in generate_step(
                mx.array(ts.PROMPTS[0], dtype=mx.int32), model, max_tokens=N_TOK,
                prompt_cache=make_prompt_cache(model))]
            assert toks == ref, (grow, toks, ref)
            assert gov[0].cap == (12 if grow else 0) and len(set(pools)) == 1, (grow, gov[0].cap, pools)
            first[grow] = pools[0]
            sf._GOV = None
            del model, bud, tick, real_tick, gov
            ts.settle()
    finally:
        sf._sysctl_int, sf._vm_counts, sf._GROW = real
        sf._GOV = None
    assert first[True] > first[False], first
    print(f"5. MOE_STORE_GROW: the pool built at {first[True]} slots at the first decode pass "
          f"(off: {first[False]} for all {N_TOK} tokens); tokens == stock")


def check_flag():
    r = subprocess.run([sys.executable, str(ROOT / "server.py"), "--model", "/nonexistent",
                        "--mem-peak-max-gb", "8"], capture_output=True, text=True,
                       timeout=60, cwd=ROOT)
    assert r.returncode == 2 and "--mem-peak-max-gb N needs" in r.stderr, r.stderr[-300:]
    print("3. server.py: --mem-peak-max-gb without --mem-peak-gb is refused before a load")


def main():
    mx.set_cache_limit(0)
    sf.phys_footprint = lambda: mx.get_active_memory() + mx.get_cache_memory()
    check_governor()
    tmp = Path(tempfile.mkdtemp(prefix="store_resize_"))
    try:
        out = tmp / "tiny_q3n"
        ts.build(out)
        check_tokens(out)
        check_settle(out)
        check_grow(out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    check_flag()
    print("test_store_resize: ok")


if __name__ == "__main__":
    main()
