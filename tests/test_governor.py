#!/usr/bin/env python3
"""The governor decides sizes; this checks the decisions, not the model.

Runs without weights and without a GPU, so it can be run on any machine at any
time. The bug it exists to catch is the one that shipped: a floor below top_k,
where every layer falls out of the arena and the machine that was "protected"
runs at a tenth of its speed.
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import stream_fast as sf


class _Glu:
    def __init__(self, cap, built=True):
        self._cap, self._projs = cap, []
        # _init_arena sets this (mx.full(...)); resize() clears it back to
        # None. `built=True` is every existing test's assumption -- they
        # already size get_active_memory as held + cap*slot_gb, which is
        # only true once the arena exists.
        self._slot_of = object() if built else None

    def resize(self, n):
        self._cap = n


class _ArenaGlu:
    """Models the ONE invariant the real _init_arena()/_claim() enforce that
    _Glu above does not: the iov list is built once, sized to _cap AT THAT
    MOMENT, and stays that size until resize() drops it for a lazy rebuild
    (see stream_fast.py's _init_arena/_claim/resize). A caller that pokes
    `_cap` directly instead of going through resize() leaves iov stale --
    claim() then indexes a slot the NEW (larger) cap allows into a list still
    sized to the OLD one.
    """
    def __init__(self, cap):
        self._cap = cap
        self._iov_len = None      # None until claim() builds it, like _ARENAS
        self._projs = []
        self._slot_of = None

    def resize(self, n):
        if n == self._cap:
            return
        self._cap = n
        self._iov_len = None      # _ARENAS.pop(): rebuilt lazily
        self._slot_of = None

    def claim(self, n_ids):
        """Stand-in for _claim(): lazily builds the arena at the CURRENT
        cap, then assigns slots 0..n_ids-1 and indexes iov with each one."""
        if self._iov_len is None:
            self._iov_len = self._cap
            self._slot_of = object()
        if n_ids > self._cap:
            raise ValueError("more distinct experts than the arena holds")
        for slot in range(n_ids):
            if slot >= self._iov_len:
                raise IndexError("list index out of range")


class _Clock:
    """Wall clock that advances as fast as the synthetic model would run.

    `stall` injects a gap the size of a turn boundary every `stall_every` guard
    ticks, which is what a real session does: prefill plus the wait for the user
    lands inside a speed window and has nothing to do with arena size.
    """
    def __init__(self, curve, gov_box, stall=0.0, stall_every=0):
        self.t, self.curve, self.box = 0.0, curve, gov_box
        self.stall, self.stall_every, self.n = stall, stall_every, 0
        self._last_toks = 0

    def perf_counter(self):
        g = self.box[0]
        if g is not None:
            # Advance by the tokens actually banked since the last guard
            # tick (g._toks), not a fixed guard_period -- with T=1 every
            # tick that's the same number, so every existing curve-based
            # test is unaffected; a T>1 test needs wall time to scale with
            # the real tokens a guard tick now represents.
            gap = g._toks - self._last_toks
            self._last_toks = g._toks
            self.t += gap / self.curve(g.cap)
            self.n += 1
            if self.stall_every and self.n % self.stall_every == 0:
                self.t += self.stall
        return self.t


def _gov(top_k=8, cap=21, level=60, press=1, curve=None, memsize=17179869184,
         stall=0.0, stall_every=0, cap_max=96, slot_gb=0.063,
         mem_pct=90, n_exp=0, host_gb=0.6):
    box = [None]
    # A list so a test can move the machine underneath a running governor.
    # "vm": (compressor GB, swapouts in pages) as _vm_counts returns it;
    # None is a machine without those counters (the level alone).
    state = {"level": level, "press": press, "vm": None}
    sf._sysctl_int = lambda name, width=4: {
        "kern.memorystatus_level": state["level"],
        "kern.memorystatus_vm_pressure_level": state["press"],
        "hw.memsize": memsize,
    }.get(name)
    sf._vm_counts = lambda: state["vm"]
    if curve is not None:
        sf.time = _Clock(curve, box, stall, stall_every)
    g = sf.ArenaGovernor([_Glu(cap) for _ in range(40)], 6, cap_max,
                         slot_gb=slot_gb, top_k=top_k, period=64, step=16,
                         mem_pct=mem_pct, n_exp=n_exp, host_gb=host_gb)
    box[0] = g
    g._state = state
    return g


def _run(g, windows):
    for _ in range(windows):
        for _ in range(g.period * len(g.glus)):
            g.tick()


def _climbs_past_half(g, windows=12):
    """After the first measured window the arena may close half the remaining
    gap to the memory ceiling each window, instead of parking at half of it
    for good (the 80B parked at 61 of a 122 ceiling and lost 1.4x)."""
    _run(g, windows)
    ceil = g._ceiling(g._avail(60))
    assert g.cap > ceil // 2 + g.step, f"parked at {g.cap} of a {ceil} ceiling"
    assert g.cap <= min(ceil, g.cap_max), f"cap {g.cap} over ceiling {ceil}"


def main():
    real_sysctl, real_time, real_vm = sf._sysctl_int, sf.time, sf._vm_counts
    sf._vm_counts = lambda: None        # never this machine's real counters

    # 1. The floor is two tokens' worth of experts, whatever the caller asked
    #    for. Exactly top_k is the worst arena there is: it holds one token's
    #    experts and evicts all of them for the next, paying the bookkeeping
    #    and the reads for nothing. Measured on a busy machine at 8 slots
    #    against top_k 8: 7.4 tok/s against 27 with room to keep a working set.
    g = _gov(top_k=8)
    assert g.cap_min == 16, g.cap_min
    g = _gov(top_k=12)
    assert g.cap_min == 24, g.cap_min

    # 2. Memory-critical shrinks to the floor and NOT below it. This is the
    #    2026-09-04 regression: cap 6 against top_k 8 took the assistant from
    #    27 tok/s to 3, because no layer can take an arena that cannot hold the
    #    experts it selected.
    g = _gov(top_k=8, cap=21, press=4, curve=lambda c: 25.0)
    _run(g, 3)
    assert g.cap == 16, f"critical floor is {g.cap}, must not go under 2 x top_k"
    assert all(x._cap == 16 for x in g.glus)

    # 3. On a quiet machine it grows to what memory allows and stops there.
    #    More resident experts is faster, measured and monotone, so the target
    #    is the ceiling -- not some interior point a noisy window liked.
    g = _gov(top_k=8, cap=21, press=1, level=60, curve=lambda c: 20.0 + c * 0.05)
    _run(g, 30)
    ceil = g._ceiling(g._avail(60))
    assert g.cap >= min(ceil, g.cap_max) - g.step, \
        f"stopped at {g.cap} with room to {min(ceil, g.cap_max)}"
    # 3b. The 80B shape: 85 MB slots, cap_max 512, a ceiling of ~120 well
    #     under cap_max, no pressure ever met. The first-step-only clamp lets
    #     it climb past half instead of parking there.
    _climbs_past_half(_gov(top_k=10, cap=21, press=1, level=60,
                           curve=lambda c: 20.0 + c * 0.05, cap_max=512,
                           slot_gb=0.0849))

    # 4. One bad window does not park it below where it started. This is the
    #    2026-09-04 hill-climb failure: a single 21.5 tok/s sample at 53 slots
    #    beat a three-point sweep that said 64 and 96 were faster, and the
    #    controller sat at 37 for the rest of the session.
    dip = {53: 21.5}
    g = _gov(top_k=8, cap=21, press=1, level=60,
             curve=lambda c: dip.get(c, 20.0 + c * 0.05))
    _run(g, 30)
    assert g.cap > 53, f"one bad sample at 53 parked it at {g.cap}"

    # 5. But a size that is genuinely much worse does cap it, so a real
    #    page-cache collapse at the top end would not be ignored.
    g = _gov(top_k=8, cap=21, press=1, level=60,
             curve=lambda c: 25.0 if c <= 53 else 5.0)
    _run(g, 30)
    assert g.cap <= 53, f"grew to {g.cap} through a 5x slowdown"

    # 4b. And it gets to the top in ONE move, not fifteen. Each resize empties
    #     the arena, so a staircase spends the session re-warming: measured
    #     2026-09-04, five steps held 26 tok/s internally while the per-turn
    #     figure the user sees fell from 26.3 to 20.6.
    g = _gov(top_k=8, cap=21, press=1, level=60, curve=lambda c: 20.0 + c * 0.05)
    _run(g, 30)
    assert len(g.history) <= 3, \
        f"took {len(g.history)} resizes to settle, each one empties the arena"

    # 5a. A turn boundary inside the window must not be read as the arena
    #     being slow. This is the 2026-09-04 failure exactly: prefill plus the
    #     wait for the user landed in the window, 53 slots measured 19% below
    #     37, and the arena walked backwards while the static sweep said the
    #     opposite.
    g = _gov(top_k=8, cap=21, press=1, level=60,
             curve=lambda c: 20.0 + c * 0.05, stall=3.0, stall_every=25)
    _run(g, 30)
    assert g.cap > 53, f"a turn boundary in the window walked it back to {g.cap}"

    # 5b. The memory ceiling still binds: a nearly full machine cannot be talked
    #    into a big arena by a curve that says bigger is better.
    g = _gov(top_k=8, cap=21, press=1, level=8,
             curve=lambda c: 10.0 + c * 0.5)
    _run(g, 20)
    ceil = g._ceiling(g._avail(8))
    assert g.cap <= max(g.cap_min, ceil), f"cap {g.cap} over ceiling {ceil}"

    # 5c. Growing must not start a cycle. The arena's own allocation drops
    #     kern.memorystatus_level, which reads exactly like somebody else
    #     taking memory -- so a controller that shrinks on pressure and then
    #     grows back into the same wall oscillates forever. Measured
    #     2026-09-04 before the fix: 21 -> 143 -> 8 -> 141 -> 70 -> 146 -> 8,
    #     four cycles in six turns, 15-21 tok/s where a fixed arena gave 27.
    #     The machine here reports a level that falls as the arena grows,
    #     exactly as a real one does.
    g = _gov(top_k=8, cap=21, press=1, level=60, curve=lambda c: 25.0)
    st = g._state
    total, slot = g.total_gb, g.slot_gb
    real_free = total * 0.60 + 21 * slot        # free with the arena given back

    def follow():
        # level as the machine would report it, given what we currently hold
        free = max(0.4, real_free - g.cap * slot)
        st["level"] = int(100 * free / total)
        st["press"] = 4 if free < 0.8 else (2 if free < 1.6 else 1)

    for _ in range(30 * g.period * len(g.glus)):
        follow()
        g.tick()
    caps = [h["new_cap"] for h in g.history]
    assert len(caps) <= 6, f"{len(caps)} resizes: {caps}"
    assert g.cap > g.cap_min, f"settled on the floor ({g.cap}) after oscillating"
    follow()
    assert st["press"] == 1, f"settled while the machine is still at pressure {st['press']}"

    # 5d. A roof set under pressure that was never ours must lift, since other applications can
    #     keep a machine at pressure for reasons our arena cannot fix and without a release the
    #     arena sits at the floor for the rest of the session.
    g = _gov(top_k=8, cap=64, press=2, level=30, curve=lambda c: 25.0)
    _run(g, 2)
    assert g._roof < g.cap_max, "no roof was set under pressure"
    g._state["press"] = 1                       # the other application quit
    _run(g, 12)
    assert g._roof == g.cap_max, "roof never lifted after the pressure passed"
    assert g.cap > g.cap_min, f"still at {g.cap} on a machine that recovered"

    # 5e. A quiet machine whose free memory jitters must not be answered with a resize, since
    #     every one of them empties the arena and costs about 30 tokens of re-warming.
    g = _gov(top_k=8, cap=64, press=1, level=60, curve=lambda c: 25.0)
    _run(g, 20)
    settled, jitter = g.cap, 0
    base = g._state["level"]
    for i in range(20 * g.guard_period * len(g.glus)):
        # +/- 2 points of free memory, which is ordinary background noise
        g._state["level"] = base + (2 if (i // 80) % 2 else -2)
        g.tick()
    moves = [h for h in g.history if h["cap"] != h["new_cap"]]
    assert len(moves) <= 2, f"{len(moves)} resizes chasing memory jitter"

    # 5f. Once a bigger arena has measured worse, stop probing upward. Every
    #     probe rebuilds the arena and costs ~30 tokens of re-warming, and a
    #     120-token turn cannot absorb that repeatedly. Measured 2026-09-04
    #     with an unconditional roof rise: rates 85 -> 27.86 and 101 -> 25.02
    #     and it kept walking back up, 29.6 tok/s falling to 24.1.
    g = _gov(top_k=8, cap=64, press=1, level=60,
             curve=lambda c: 28.0 if c <= 85 else 24.0)
    _run(g, 40)
    settled = g.cap
    before = len(g.history)
    _run(g, 40)
    assert len(g.history) == before, (
        f"{len(g.history) - before} more resizes after settling at {settled}")

    # 5g. When a bigger size measures worse, go back to the size that measured
    #     BEST, not halfway to it. The rates are already in hand; halving lands
    #     on a size nobody measured. Observed 2026-09-04: 88 -> 27.76 tok/s,
    #     104 -> 24.90, settled on 96 at 24.1.
    peak = 80
    g = _gov(top_k=8, cap=64, press=1, level=60,
             curve=lambda c: 28.0 if c <= peak else 22.0)
    _run(g, 40)
    assert g.cap in g._rate, f"settled on {g.cap}, which was never measured"
    assert g._rate[g.cap] == max(g._rate.values()), (
        f"settled on {g.cap} at {g._rate[g.cap]:.1f} with "
        f"{max(g._rate, key=g._rate.get)} measured faster")

    # 6. Giving memory back is immediate. The user launches something big
    #    mid-answer; the arena must come down within a couple of tokens, not at
    #    the end of the 64-token speed window. This is the whole point of the
    #    guard lane.
    g = _gov(top_k=8, cap=64, press=1, level=60, curve=lambda c: 25.0)
    _run(g, 1)                                  # settle at 64
    before = g.cap
    g._state["level"] = 5                       # something ate the machine
    for _ in range(g.guard_period * len(g.glus)):
        g.tick()
    assert g.cap < before, f"still at {g.cap} a guard window after memory vanished"
    assert g.cap >= g.cap_min
    tokens_taken = g.guard_period
    assert tokens_taken < g.period, "guard lane is no faster than the speed lane"

    # 7. And it does NOT grow on the guard lane: growing without a measurement
    #    is how the old controller ended up chasing free memory.
    g = _gov(top_k=8, cap=21, press=1, level=90, curve=lambda c: 25.0)
    for _ in range(g.guard_period * len(g.glus) * 4):
        g.tick()
    assert g.cap == 21, f"grew to {g.cap} on the guard lane, with no rate measured"

    # 8. The process memory budget (D1/D2) is a second, independent ceiling.
    #    Pure functions first -- no machine, no governor, just the formula.
    assert sf.arena_floor(8) == 32, "4x top_k is the soak-proven default"
    assert sf.arena_floor(4) == 16
    assert sf.budget_slots(10.0, 2.0, 0.1, 0.5) == 75, \
        "(budget - held - headroom) / slot, floored"
    assert sf.budget_slots(1.0, 2.0, 0.1, 0.5) == 0, \
        "held + headroom already over budget must floor at 0, not go negative"

    # Regression: at the shipped default (mem_pct=90) the budget must not be
    # what binds on an ordinary machine -- every check above holds only
    # because of this.
    g = _gov(top_k=8, cap=64, level=60, slot_gb=0.063)
    assert g._budget_ceiling() > g._ceiling(g._avail(60)), \
        "default --mem-pct 90 must not be the binding ceiling"

    # Now make the budget the binding one and watch it behave like the
    # machine ceiling: shrinks live as held_gb grows (a KV cache filling up),
    # recovers when it shrinks back (e.g. /reset), and the machine's own
    # ceiling (level 90, plenty free) is NOT what is doing the limiting.
    real_active = sf.mx.get_active_memory
    try:
        g = _gov(top_k=4, cap=26, level=90, memsize=17179869184,
                 slot_gb=0.117, cap_max=512, mem_pct=30, n_exp=60)
        held = [2.0]  # everything mx holds outside the arena, GB
        # get_active_memory always reports held + the CURRENT arena, exactly
        # the invariant _held_gb() relies on: shrinking the arena must not
        # look like held_gb itself changed.
        sf.mx.get_active_memory = lambda: (held[0] + g.cap * g.slot_gb) * 1e9
        bceil = sf.budget_slots(g.budget_gb, held[0], g.slot_gb, g.headroom_gb)
        assert bceil < g._ceiling(g._avail(90)), \
            "test is not exercising the budget path"
        _run(g, 2)
        assert g.cap <= bceil, f"cap {g.cap} over the budget ceiling {bceil}"

        # KV growth: held_gb up 0.8 GB, arena must come down within one guard
        # window -- the same speed the machine ceiling already gives memory
        # back at (test 6), not stuck at the size the session started with.
        before = g.cap
        held[0] += 0.8
        for _ in range(g.guard_period * len(g.glus)):
            g.tick()
        assert g.cap < before, f"still at {g.cap} a guard window after held_gb grew"

        # And it recovers once the KV shrinks back.
        held[0] -= 0.8
        _run(g, 4)
        assert g.cap >= before - g.step, f"did not grow back from {g.cap}"
    finally:
        sf.mx.get_active_memory = real_active

    # 9. Floor lowering under a tight budget (D3): arena_floor(top_k) is the
    #    default, but a budget too small to afford it lowers the floor rather
    #    than pretending the size still fits -- down to, never below, 2*top_k.
    try:
        top_k = 8
        sf._sysctl_int = lambda name, width=4: {
            "hw.memsize": 2_000_000_000,  # ~2 GB machine, deliberately tiny
        }.get(name)
        g = sf.make_governor([_Glu(0) for _ in range(8)], 64, top_k, 90, 15, 0.6)
        floor = sf.arena_floor(top_k)
        assert g is not None
        assert g.cap_min < floor, f"floor {g.cap_min} was not lowered below {floor}"
        assert g.cap_min >= 2 * top_k, f"floor {g.cap_min} went under 2*top_k"

        # A budget that CAN afford the default floor leaves it alone.
        sf._sysctl_int = lambda name, width=4: {
            "hw.memsize": 200_000_000_000,  # ~200 GB, plenty
        }.get(name)
        g2 = sf.make_governor([_Glu(0) for _ in range(8)], 64, top_k, 90, 15, 0.6)
        assert g2.cap_min == floor, f"floor {g2.cap_min} changed with room to spare"
    finally:
        sf._sysctl_int = real_sysctl
        sf._GOV = None

    # 10. held_gb must not go negative before the arena exists (D2 fix): the
    #    arena is built lazily on a layer's first decode (_init_arena), so
    #    at construction time -- and on the very first tick -- nothing has
    #    been allocated for it yet. This is the "mlx holds -0.07 GB" banner
    #    bug: cap*slot_gb was subtracted from active memory that never
    #    included it, inflating budget_slots along with it (48 slots read
    #    out for a 5.15 GB budget).
    g = _gov(top_k=8, cap=21)
    for glu in g.glus:
        glu._slot_of = None                  # arena not yet built anywhere
    active = sf.mx.get_active_memory() / 1e9
    assert g._held_gb() == active, \
        f"held_gb {g._held_gb():.3f} subtracted an arena that doesn't exist"
    assert g._held_gb() >= 0, f"held_gb {g._held_gb():.3f} went negative"
    for glu in g.glus:
        glu._slot_of = object()              # every layer has built its arena
    assert g._held_gb() == active - g.cap * g.slot_gb, \
        "held_gb should subtract the arena once every layer has one"

    # 11. tick(T=3) must bank tokens three times as fast, pass for pass, as
    #    the default T=1 -- the governor's own clock has to track positions
    #    processed, not invocations, or a batched draft-verification pass
    #    (T=k+1 in one forward) starves the guard/speed lanes the same way
    #    plain call-counting did (measured: 70-100 passes for 200 output
    #    tokens under --draft k=2, the period=64 speed lane fired once).
    g_t1, g_t3 = _gov(top_k=8, cap=21), _gov(top_k=8, cap=21)
    span = len(g_t1.glus)
    for _ in range(5):
        for _ in range(span):
            g_t1.tick()
        for _ in range(span):
            g_t3.tick(False, 3)
    assert g_t1._toks == 5, g_t1._toks
    assert g_t3._toks == 15, g_t3._toks
    assert g_t3._toks == 3 * g_t1._toks, \
        f"T=3 banked {g_t3._toks} against T=1's {g_t1._toks}, not 3x"

    # 12. The first speed-tick must actually grow the arena once it has a
    #    real measurement -- the G5 regression: under --mem-pct 90 (a
    #    132-slot budget) the arena sat at its 21-slot start for 120
    #    tokens, because the speed lane (period=64) was gated on tick()
    #    CALLS, and a batched draft pass ticks once per LAYER per PASS
    #    regardless of how many positions T it covers. With T banked
    #    properly, two speed windows' worth of real tokens is enough --
    #    one to clear the post-construction `_skip` (a resize costs ~30
    #    tokens of re-warming, so the very first window never acts on its
    #    own measurement), the next to actually grow -- no matter how few
    #    passes it took to bank them.
    g = _gov(top_k=8, cap=21, press=1, level=60, curve=lambda c: 20.0 + c * 0.05,
             cap_max=512, slot_gb=0.0849)
    span, draft_T, toks = len(g.glus), 3, 0
    while toks < 3 * g.period:          # comfortably past the 2nd window
        for _ in range(span):
            g.tick(False, draft_T)
        toks += draft_T
    assert g.cap > 21, f"two speed windows under a batched pass left it at {g.cap}"
    ceil = g._ceiling(g._avail(60))
    assert g.cap <= max(g.cap_min, ceil // 2 + g.step), \
        f"grew to {g.cap}, past the first-step half-of-ceiling clamp"

    # 13. make_governor must START the arena at the ceiling the budget and the machine already
    #    allow, not at the floor, because the arena is built lazily on the first decode
    #    (_init_arena, sized off each glu's own _cap) so a bigger STARTING _cap costs no resize.
    try:
        top_k = 8

        # 13a. A generous budget and a quiet machine: start right at cap_max.
        sf._sysctl_int = lambda name, width=4: {
            "hw.memsize": 200_000_000_000,        # 200 GB, plenty
            "kern.memorystatus_level": 90,         # mostly free
            "kern.memorystatus_vm_pressure_level": 1,
        }.get(name)
        glus_a = [_Glu(21, built=False) for _ in range(8)]
        g = sf.make_governor(glus_a, 0, top_k, 90, 15, 0.6, cap_max=96)
        assert g.cap == 96, f"generous budget started at {g.cap}, not cap_max 96"
        assert all(x._cap == 96 for x in glus_a), "glus not raised to the new start"

        # 13b. A tight process-memory budget: start at what the budget
        #    allows, below cap_max -- not at the floor, and not waiting for
        #    the speed lane to grow into it one window at a time.
        sf._sysctl_int = lambda name, width=4: {
            "hw.memsize": 20_000_000_000,          # 20 GB
            "kern.memorystatus_level": 70,
            "kern.memorystatus_vm_pressure_level": 1,
        }.get(name)
        glus_b = [_Glu(21, built=False) for _ in range(8)]
        g = sf.make_governor(glus_b, 0, top_k, 50, 15, 0.6, cap_max=300)
        bslots = sf.budget_slots(g.budget_gb, g._held_gb(), g.slot_gb, g.headroom_gb)
        assert g.cap == bslots, \
            f"tight budget started at {g.cap}, not the {bslots}-slot budget ceiling"
        assert g.cap < g.cap_max, "test is not exercising a tight budget"

        # 13c. A budget too tight even for the default floor (D3, see #9)
        #    still must not start BELOW its own (lowered) floor.
        sf._sysctl_int = lambda name, width=4: {
            "hw.memsize": 2_000_000_000,           # ~2 GB, deliberately tiny
            "kern.memorystatus_level": 90,
            "kern.memorystatus_vm_pressure_level": 1,
        }.get(name)
        glus_c = [_Glu(21, built=False) for _ in range(8)]
        g = sf.make_governor(glus_c, 0, top_k, 50, 15, 0.6, cap_max=64)
        assert g._floor < sf.arena_floor(top_k), "test is not exercising a lowered floor"
        assert g._floor >= 2 * top_k, f"floor {g._floor} went under 2*top_k"
        assert g.cap == g._floor, f"cap {g.cap} started below its own floor {g._floor}"

        # 13d. The first guard tick must not shrink what was just budgeted,
        #    with no real memory pressure. tick()'s "half the ceiling until
        #    a speed window has measured something" caution (the
        #    `self._roof >= self.cap_max` block) exists for a GROWTH
        #    attempt after a real shrink; it must not fire against a start
        #    that was already derived from that same ceiling. `level` here
        #    tracks the arena's own footprint the way a real sysctl read
        #    would once the arena materializes on the first decode, so this
        #    reproduces the regression exactly: without the fix the live
        #    ceiling (220 slots) gets halved to 110, well under the 188-slot
        #    start, and the guard tick shrinks it right back down.
        total_gb, slot_gb, avail0 = 20.0, 0.05, 20.0 * 0.70
        box = {}

        def _sysctl(name, width=4):
            if name == "hw.memsize":
                return int(total_gb * 1e9)
            if name == "kern.memorystatus_level":
                cap = box["g"].cap if "g" in box else 0
                return int(100 * (avail0 - cap * slot_gb) / total_gb)
            if name == "kern.memorystatus_vm_pressure_level":
                return 1
            return None
        sf._sysctl_int = _sysctl
        glus_d = [_Glu(21, built=False) for _ in range(8)]
        g = sf.make_governor(glus_d, 0, top_k, 50, 15, 0.6, cap_max=300)
        box["g"] = g
        before = g.cap
        assert before > g._ceiling(avail0) // 2, \
            "test parameters do not actually risk the halving bug"
        span = len(g.glus)
        for _ in range(2 * span):            # exactly one guard evaluation
            g.tick()
        assert g.cap == before, \
            f"first guard tick shrank {before} -> {g.cap} with no memory pressure"
        assert not g.history, "first guard tick resized the arena with no pressure"
    finally:
        sf._sysctl_int = real_sysctl
        sf._GOV = None

    # 14. Raising the starting cap must go through resize(), not a direct `_cap = n`, because a
    #    layer whose arena was already built elsewhere keeps its stale iov list and _claim()
    #    would index past its end, which _ArenaGlu models with an iov list that only resize()
    #    invalidates (an invariant _Glu, used by every other test here, does not have).
    sf._sysctl_int = lambda name, width=4: {
        "hw.memsize": 200_000_000_000,          # generous: start should hit cap_max
        "kern.memorystatus_level": 90,
        "kern.memorystatus_vm_pressure_level": 1,
    }.get(name)
    try:
        top_k, cap_min = 8, 21
        glus_e = [_ArenaGlu(cap_min) for _ in range(4)]
        for glu in glus_e:
            glu.claim(cap_min)      # arena already built HERE, iov sized cap_min
        g = sf.make_governor(glus_e, 0, top_k, 90, 15, 0.6, cap_max=60)
        assert g.cap == 60, f"expected the generous-budget start (60), got {g.cap}"
        assert g.cap > cap_min, "test is not exercising a raise past the stale iov"
        for glu in glus_e:
            # Must not raise: resize() has to have dropped the stale iov so
            # claim() rebuilds it at the new cap, the way _init_arena would.
            glu.claim(40)            # cap_min(21) < 40 <= new cap(60)
    finally:
        sf._sysctl_int = real_sysctl
        sf._GOV = None

    # 15. Between requests nothing calls tick(): idle_tick() (server.py's
    #     arena watch) gives slots back when the machine runs under the
    #     reserve or macOS warns, keeps them on a quiet machine, never grows.
    g = _gov(cap=64, level=60)
    g._floor = 21
    g.idle_tick()
    assert g.cap == 64, f"idle tick moved a quiet machine's arena: {g.cap}"
    g._state["level"] = 10          # 1.7 GB available + the arena's own 4 GB
    ceil = g._ceiling(g._avail(10))
    g.idle_tick()
    assert 21 <= g.cap == max(21, ceil) < 64, f"idle shrink to {g.cap}, ceiling {ceil}"
    shrunk = g.cap
    g._state["level"] = 90
    g.idle_tick()
    assert g.cap == shrunk, "idle tick must not grow"
    g._state.update(level=60, press=2)
    g.idle_tick()
    assert g.cap == max(21, shrunk // 2), f"warning: {g.cap}"
    g._state["press"] = 4
    g.idle_tick()
    assert g.cap == 21, f"critical: {g.cap}"

    # 16. The compressor counts against the room: kern.memorystatus_level counts non-compressed
    #     memory only, so the same level must hold a big arena over an empty compressor and
    #     shrink it over a full one.
    g = _gov(cap=60, level=40, curve=lambda c: 25.0)
    g._floor = 21
    g._state["vm"] = (0.5, 1000)
    _run(g, 1)
    assert g.cap == 60, f"a roomy machine shrank the arena to {g.cap}"
    g._state["vm"] = (7.5, 1000)
    ceil = g._ceiling(g._avail(40, 7.5))
    for _ in range(g.guard_period * len(g.glus)):
        g.tick()
    assert 21 <= g.cap == max(21, ceil) < 60, f"full compressor: {g.cap}, ceiling {ceil}"

    # 16b. With speeds on record the squeeze still binds, without waiting for
    #      a pressure warning. The growth clamp used to lift a ceiling under
    #      the current size back to it whenever a rate was on record, so free
    #      memory alone could never shrink a measured arena.
    g = _gov(cap=64, level=60, cap_max=64, curve=lambda c: 25.0)
    _run(g, 4)
    assert g._rate and g.cap == 64, "test needs rates on record at 64"
    g._state["level"] = 10
    ceil = g._ceiling(g._avail(10))
    for _ in range(g.guard_period * len(g.glus)):
        g.tick()
    assert g.cap == max(g.cap_min, ceil) < 64, f"measured arena held {g.cap}, ceiling {ceil}"

    # 17. Swap reflex: swapouts seen at a guard tick take one step off and
    #     set a roof there, whatever the other signals say; no new swapouts,
    #     no further shrink; quiet windows bring it back.
    g = _gov(cap=60, level=60, curve=lambda c: 25.0)
    g._floor = 21
    g._state["vm"] = (0.5, 1000)
    _run(g, 1)
    assert g.cap == 60
    g._state["vm"] = (0.5, 1010)            # ten pages written to swap
    for _ in range(g.guard_period * len(g.glus)):
        g.tick()
    assert g.cap == 60 - g.step, f"swap reflex: {g.cap}"
    assert g._roof == g.cap, f"no roof at {g.cap}: {g._roof}"
    for _ in range(4 * g.guard_period * len(g.glus)):
        g.tick()
    assert g.cap == 60 - g.step, f"shrank again with no new swapouts: {g.cap}"
    _run(g, 12)
    assert g.cap >= 60, f"never grew back after the swap stopped: {g.cap}"

    # 18. The reserve is measured, not assumed: a prefill that drags the margin 3 GB under its
    #     quiet reading makes the reserve 3 GB from the next decode tick on, since the margin
    #     counts two thirds of the compressor (_SWAP_SHARE).
    g = _gov(cap=40, level=60, curve=lambda c: 25.0)
    g._state["vm"] = (2.0, 1000)
    _run(g, 1)
    assert abs(g._reserve_gb() - g.total_gb * 0.15) < 1e-9, "prior is --reserve-pct"
    g._state["vm"] = (6.5, 1000)            # the prefill's dip
    for _ in range(len(g.glus)):
        g.tick(do_sort=True)
    g._state["vm"] = (2.0, 1000)            # gone again once it decodes
    for _ in range(g.guard_period * len(g.glus)):
        g.tick()
    assert abs(g._reserve_gb() - 3.0) < 1e-6, f"reserve {g._reserve_gb()}"

    # 19. Between requests the swap reflex holds too.
    g = _gov(cap=60, level=60)
    g._floor = 21
    g._state["vm"] = (0.5, 1000)
    g.idle_tick()
    assert g.cap == 60, f"idle tick moved a quiet machine's arena: {g.cap}"
    g._state["vm"] = (0.5, 1100)
    g.idle_tick()
    assert g.cap == 60 - g.step, f"idle swap reflex: {g.cap}"

    # 20. The start reads the same margin: a roomy level over a full compressor starts the
    #     arena lower.
    try:
        sf._sysctl_int = lambda name, width=4: {
            "hw.memsize": 17179869184, "kern.memorystatus_level": 60,
            "kern.memorystatus_vm_pressure_level": 1}.get(name)
        starts = []
        for comp in (0.0, 6.0):
            sf._vm_counts = lambda comp=comp: (comp, 0)
            starts.append(sf.make_governor([_Glu(21, built=False) for _ in range(8)], 0, 8,
                                           90, 15, 0.6, cap_min=21, cap_max=300).cap)
        assert 21 <= starts[1] < starts[0], f"start ignores the compressor: {starts}"
    finally:
        sf._GOV = None

    sf._sysctl_int, sf.time, sf._vm_counts = real_sysctl, real_time, real_vm
    print("governor: floor, critical shrink, grows to the memory ceiling, "
          "survives one bad window, caps on a real collapse, does not "
          "oscillate against its own allocation, instant give-back, "
          "no blind growth, memory budget binds and recovers live, "
          "budget can lower the floor but never past 2x top_k, "
          "held_gb never negative pre-arena, tick(T) banks tokens not "
          "calls, first speed-tick grows under a batched pass, "
          "make_governor starts at the budget/machine ceiling not the "
          "floor, the first guard tick does not shrink it back, and "
          "raising the start goes through resize() so a stale iov never "
          "outlives it, idle ticks only shrink, the compressor counts "
          "against the room (start included), swapouts take a step off and "
          "set a roof, the reserve is the prefill's measured dip -- all hold")


if __name__ == "__main__":
    main()
