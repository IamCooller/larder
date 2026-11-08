#!/usr/bin/env python3
"""One slot per expert, whoever asks.

Past `cap` distinct ids in one _claim call the LRU head is an entry that same
call just inserted, so two experts got one slot and the second read silently
overwrote the first. It showed up as non-reproducible generation: per-token byte
traces for arena 21 / width 24 matched their own repeat in 2 of 173 tokens.

    python3 test_claim_slots.py
"""
import os
import sys
from collections import OrderedDict
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import stream_fast
from stream_fast import StreamingSwitchGLU, _ArenaKey


def claim(cap, ids, count):
    """Run the real _claim against a stub arena and return (slots, dropped)."""
    key = _ArenaKey()
    stream_fast._ARENAS[key] = []          # no projections -> no read jobs
    glu = SimpleNamespace(_akey=key, _lru=OrderedDict(), _cap=cap,
                          _slot_np=None, _bias_hook=None,
                          _seen=None, _want=0, _budget=None)
    glu._claimed = StreamingSwitchGLU._claimed.__get__(glu, SimpleNamespace)
    before = stream_fast.STATS.get("warm_dropped", 0)
    slots, _ = StreamingSwitchGLU._claim(glu, ids, count=count)
    return slots, stream_fast.STATS.get("warm_dropped", 0) - before


def main():
    # Speculative path: more ids than slots is normal for a wide prefetch, and
    # the ones that do not fit must be dropped, not aliased onto live slots.
    for cap, n in ((21, 24), (12, 16), (4, 9)):
        slots, dropped = claim(cap, list(range(n)), count=False)
        kept = [int(s) for s in slots if s >= 0]
        assert len(kept) == len(set(kept)), f"cap {cap}, {n} ids: aliased {kept}"
        assert len(kept) <= cap, f"cap {cap}: claimed {len(kept)} slots"
        assert dropped == n - cap, f"cap {cap}, {n} ids: dropped {dropped}"

    # Whatever fits keeps every slot distinct, and nothing is dropped.
    for cap, n in ((21, 8), (21, 21), (12, 12), (4, 1)):
        slots, dropped = claim(cap, list(range(n)), count=False)
        kept = [int(s) for s in slots]
        assert min(kept) >= 0 and len(set(kept)) == n, f"cap {cap}, {n} ids: {kept}"
        assert dropped == 0, f"cap {cap}, {n} ids: dropped {dropped}"

    # Demand path cannot drop anything -- the layer needs every expert it asks
    # for -- so an arena smaller than top_k has to fail loudly.
    try:
        claim(4, list(range(9)), count=True)
    except ValueError as e:
        assert "top_k exceeds the arena" in str(e), e
    else:
        raise AssertionError("demand path accepted more ids than slots")

    # The callers dedupe before calling -- the parameter is named unique_ids --
    # so duplicates inside one call are out of contract and not checked here.

    # Under run-ahead a prefetch may NOT publish what it claims: the layer's
    # own gather has been submitted and its residency flag has not been checked
    # yet, so the slot table that gather read must not move under it. warm()
    # parks the newcomers -- out of _lru and out of _slot_np -- and _commit,
    # which every claiming path reaches through _await_pending, publishes them.
    key = _ArenaKey()
    stream_fast._ARENAS[key] = []              # no projections -> no read jobs
    glu = SimpleNamespace(_akey=key, _lru=OrderedDict(), _cap=8,
                          _slot_np=np.full(16, -1, dtype=np.int32),
                          _bias_hook=None, _seen=None, _want=0,
                          _pending=[], _defer=[], _projs=[], _budget=None,
                          _ring={}, _store=None)
    glu._resident_ok = lambda: True
    for name in ("_claim", "_claimed", "_await_pending", "_commit", "warm"):
        setattr(glu, name, getattr(StreamingSwitchGLU, name).__get__(glu, SimpleNamespace))
    warm = glu.warm
    saved = stream_fast._DEFER
    stream_fast._DEFER = True
    try:
        warm([1, 2, 3])
        assert list(glu._lru) == [], list(glu._lru)
        assert (glu._slot_np < 0).all(), glu._slot_np
        assert sorted(e for e, _ in glu._defer) == [1, 2, 3], glu._defer
        assert len({sl for _, sl in glu._defer}) == 3, glu._defer
        glu._commit()
        assert sorted(glu._lru) == [1, 2, 3], list(glu._lru)
        assert [int(glu._slot_np[e]) for e in (1, 2, 3)] == [glu._lru[e] for e in (1, 2, 3)]
        # A hit adds nothing to park and publishes nothing new.
        warm([2])
        assert glu._defer == [] and sorted(glu._lru) == [1, 2, 3], (glu._defer, list(glu._lru))
    finally:
        stream_fast._DEFER = saved

    store_claims()
    print("ok -- no expert shares a slot with another, at any width; "
          "a prefetch under run-ahead publishes nothing before its commit; "
          "budget mode: quota first, then the ring half, and a ring slot taken "
          "over is unpublished from the layer that held it")


def store_claims():
    """Budget mode's placement (_place_store) on a stub store: a pool of 4
    (quotas 0-1 and 2-3) and two ring halves of 4 (slots 4-7, 8-11)."""
    st = SimpleNamespace(pool=4, per_half=4, owner=[None] * 12, next=[0, 0], dirty=False)
    st.ring_slot = stream_fast.ExpertStore.ring_slot.__get__(st)

    def layer(half, quota):
        key = _ArenaKey()
        stream_fast._ARENAS[key] = []          # no projections -> no read jobs
        g = SimpleNamespace(_akey=key, _lru=OrderedDict(), _ring={}, _store=st,
                            _half=half, _qslots=quota, _cap=len(quota) + 4,
                            _slot_np=np.full(64, -1, dtype=np.int32), _budget=True,
                            _bias_hook=None, _seen=None, _want=0,
                            _ghost_last=np.full(64, -1, dtype=np.int64),
                            _ghost_hist=np.zeros(64, dtype=np.int64), _ghost_clock=0)
        for name in ("_claim", "_place_store", "_claimed", "_ghost"):
            setattr(g, name, getattr(StreamingSwitchGLU, name).__get__(g, SimpleNamespace))
        return g

    a, c = layer(0, [0, 1]), layer(0, [])       # same ring half
    slots, _ = a._claim(np.arange(6))
    assert slots.tolist()[:2] == [0, 1] and sorted(slots.tolist()[2:]) == [4, 5, 6, 7], slots
    assert all(a._slot_np[e] == slots[e] for e in range(6))
    # c has no quota: its misses take the next slots of half 0, a's 2 and 3
    got, _ = c._claim(np.array([10, 11]))
    assert got.tolist() == [4, 5], got
    assert a._slot_np[2] == a._slot_np[3] == -1 and 2 not in a._ring and 3 not in a._ring
    # a's own ring hits are kept while its quota LRU makes room
    slots, _ = a._claim(np.array([0, 4, 5]))
    assert slots.tolist() == [0, 6, 7], slots
    slots, _ = a._claim(np.array([2, 3]))       # evicts 1, then 0, from the quota
    assert sorted(slots.tolist()) == [0, 1] and a._slot_np[0] == a._slot_np[1] == -1
    # wider than quota + ring half: a demand raises, a prediction is trimmed
    try:
        a._claim(np.arange(20, 27))
    except ValueError as e:
        assert "rounds path" in str(e), e
    else:
        raise AssertionError("budget mode claimed more experts than it holds")
    slots, _ = a._claim(np.arange(30, 38), count=False)
    assert (slots >= 0).sum() == 6 and len(set(slots[slots >= 0].tolist())) == 6, slots


if __name__ == "__main__":
    main()
