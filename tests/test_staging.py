#!/usr/bin/env python3
"""MOE_STAGING (staging rows in the native agent)
on test_native's budget toy, every layer 4-bit so the
eight MoE layers share one store, quotas cut by hand (equal pool in every
arm):

  1. The Python path, MOE_STAGING=dry (guesses handed over, rows charged,
     nothing read) and MOE_STAGING=1 -- also with the read probes
     MOE_STAGING_NOCACHE=1 and MOE_STAGING_READERS=pool, and with guess reads
     failing and slowed (MOE_STAGING_FAULT) -- give the same tokens, bitwise
     the same logprobs, and end with the same quota LRU, `_ring`, `_seen`,
     ghost LRU and ring halves, at cache_bias 0 and 0.5, across Python passes
     between the native ones (each a hand-back) and a store rebuild.
     Staging promoted guessed misses and moved rows back; the faults show as
     failed reads read again, records dropped and partial promotions.
     MOE_CPU_GUESS=1 (the agent ranks the guesses on the CPU) and =check (it
     compares its guesses with the GPU's) give the same too.
  2. The same with a sidecar mirror (guesses read on both drives).
  3. The store governor's resize mid-decode (MemBudget.tick: the agent
     hands back while the last pass may still gather): same tokens.
  4. unstage() on random logical -> physical maps (set_perm), cycles
     included: every logical row's bytes end at its own row.
  5. Where staging cannot work it is off at install: two expert shapes (two
     stores) and no budget store -- no rows allocated, no guess wired, the
     same tokens.
  6. An older moe_native (no set_staging, no `guess=`, five-argument
     set_layer): the flag off works, and asked for, staging turns itself off.

    python3 tests/test_staging.py      # needs moe_native from native/build.sh
"""
import os
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import mlx.core as mx
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache

import stream_fast
import test_native as tn
import test_store as ts

ts.TWO_BIT = ()             # one expert shape: one store (section 5 puts it back)

ENV = ("MOE_STAGING_NOCACHE", "MOE_STAGING_READERS", "MOE_STAGING_FAULT")
# The read probes and the faults (the agent reads them at set_staging).
PROBES = {"1": {}, "1+nocache": {"MOE_STAGING_NOCACHE": "1"},
          "1+nocache+pool": {"MOE_STAGING_NOCACHE": "1", "MOE_STAGING_READERS": "pool"},
          "1+fault": {"MOE_STAGING_FAULT": "3,2000"},
          "1+pool+fault": {"MOE_STAGING_READERS": "pool", "MOE_STAGING_FAULT": "3,2000"},
          "1+cpu": {}, "1+cpucheck": {}}
CPU = {"1+cpu": "1", "1+cpucheck": "check"}     # MOE_CPU_GUESS, read at the call
NAMES = ("quota LRU", "_ring", "_seen", "ghost hist", "ghost last", "ghost clock")


def set_stage(stage):
    for k in ENV:
        os.environ.pop(k, None)
    os.environ.update(PROBES.get(stage, {}))
    stream_fast._CPU_GUESS = CPU.get(stage, "0")
    s = "1" if stage in PROBES else stage
    stream_fast._STAGE = s
    stream_fast._STAGING_ROWS = 16 if s in ("1", "dry") else 0


def arm(out, native, stage, bias, run=tn.run_budget):
    set_stage(stage)
    s0 = dict(stream_fast.STATS)
    r = run(out, native, bias)
    return r, {k[6:] if k.startswith("stage_") else k: stream_fast.STATS[k] - s0[k]
               for k in s0 if k.startswith(("stage_", "cg_")) and k != "cg_max_us"}


def same_run(a, b, tag, state=True):
    assert a[0] == b[0], f"{tag}: tokens differ\n{a[0]}\n{b[0]}"
    bad = [i for i, (x, y) in enumerate(zip(a[1], b[1])) if not tn.same(x, y)]
    assert not bad, f"{tag}: logprobs differ at steps {bad}"
    if state:
        for l, (sa, sb) in enumerate(zip(a[2], b[2])):
            for name, x, y in zip(NAMES, sa, sb):
                assert tn.same(x, y), f"{tag}: layer {l} {name}: {x} != {y}"
        assert tn.same(a[3], b[3]), f"{tag}: ring halves {a[3]} != {b[3]}"


def check_arms(out, sidecar=None):
    stages = ("dry", *PROBES) if sidecar is None else ("1", "1+nocache+pool")
    total = {}
    for bias in (0.0, 0.5):
        # A native arm first: the native stack wrapper must go inside the
        # budget's (install_native), and both wrap the class.
        runs = {stages[0]: arm(out, True, stages[0], bias)}
        a, _ = arm(out, False, "0", bias)
        for s in stages[1:]:
            runs[s] = arm(out, True, s, bias)
        for s, (b, st) in runs.items():
            tag = f"{'sidecar, ' if sidecar else ''}bias {bias}, MOE_STAGING={s}"
            assert b[4], f"{tag}: no native pass ran"
            same_run(a, b, tag)
            if s != "dry":
                assert st["prom"] and st["unstaged"], f"{tag}: nothing promoted or moved: {st}"
            if "fault" in s:
                assert st["bad"] and st["drop"] and st["part"] and st["wrong"], f"{tag}: {st}"
            if s in CPU:
                assert st["cg_n"] and not st["cg_skip"], f"{tag}: no guess ranked on the CPU: {st}"
                assert (st["cg_cmp"] == st["cg_n"]) == (s == "1+cpucheck"), f"{tag}: {st}"
            for k, v in st.items():
                total[k] = total.get(k, 0) + v
            print(f"  {tag}: {len(a[0])} tokens identical to the Python path; {st}")
    return total


def run_resize(out, native, bias):
    """run_budget's turns with the store governor's resize raised every 4th
    decode token: the next pass hands the agent's tables back and refits."""
    model = ts.stock_model(out, tn.CFG8)
    stream_fast.install(model, out, resident=ts.RES,
                        cache_bias=bias, defer=False, native=native)
    bud = stream_fast.enable_budget(model, ts.budget_gb())
    glus = stream_fast.glus_of(model)
    quota = {id(g): q for g, q in zip(glus, tn.QUOTA)}
    bud._quotas = lambda pool: quota
    cache, toks, lps = make_prompt_cache(model), [], []
    for p, n in zip(ts.PROMPTS, ts.GEN):
        for t, lp in generate_step(mx.array(p, dtype=mx.int32), model,
                                   max_tokens=n, prompt_cache=cache):
            toks.append(int(t))
            lps.append(np.array(lp.astype(mx.float32)))
            if len(toks) % 4 == 0:
                bud.resized = True          # make_store_governor moved the peak
    stream_fast.native_sync()
    if native:
        assert not stream_fast._NATIVE.agent.error
    del model, bud, glus, cache
    ts.settle()
    return toks, lps


def check_unstage(out):
    """set_perm + unstage on the agent of a staging run, over its store."""
    set_stage("1")
    model = ts.stock_model(out, tn.CFG8)
    stream_fast.install(model, out, resident=ts.RES, cache_bias=0.0,
                        defer=False, native=True)
    bud = stream_fast.enable_budget(model, ts.budget_gb())
    glus = stream_fast.glus_of(model)
    quota = {id(g): q for g, q in zip(glus, tn.QUOTA)}
    bud._quotas = lambda pool: quota
    for _ in generate_step(mx.array(ts.PROMPTS[0], dtype=mx.int32), model, max_tokens=4,
                           prompt_cache=make_prompt_cache(model)):
        pass
    nat = stream_fast._NATIVE
    nat.to_python()                         # idle, rows home
    st, agent = glus[0]._store, nat.agent
    slots, n = st.slots, st.slots + stream_fast._STAGING_ROWS
    rows = [v.reshape(v.shape[0], -1).view(np.uint8) for vs in st.views for v in vs]
    assert all(r.shape[0] == n for r in rows)
    rng = np.random.default_rng(0)
    cycles = copies = 0
    for trial in range(40):
        agent.set_staging(slots, n - slots)
        # Trial 0: one cycle through every logical row, no free row in it.
        perm = (np.roll(np.arange(slots), 1) if trial == 0
                else rng.permutation(n)[:slots]).tolist()
        for r in rows:
            r[:] = 0xEE
            for x, p in enumerate(perm):
                r[p, :4] = np.frombuffer(np.int32(x).tobytes(), np.uint8)
                r[p, 4:] = x % 251
        seen, c = set(), 0                  # cycles of the map, for the record
        for x in range(slots):
            if x in seen:
                continue
            y = x
            while y < slots and y not in seen:
                seen.add(y)
                y = perm[y]
            c += y == x and perm[x] != x
        cycles += c
        agent.set_perm(perm)
        copies += agent.unstage()
        for r in rows:
            for x in range(slots):
                assert int(np.frombuffer(r[x, :4].tobytes(), np.int32)[0]) == x and \
                    (r[x, 4:] == x % 251).all(), f"trial {trial}: logical row {x} not home"
    print(f"4. unstage: 40 maps over {slots} logical + {n - slots} staging rows "
          f"({cycles} closed cycles), {copies} row copies, every row home")
    del model, bud, glus
    ts.settle()


def old_module(real):
    """moe_native as before staging: no set_staging, no `guess=`, set_layer
    with five arguments."""
    class OldAgent:
        def __init__(self, *a, **k):
            self._a = real.Agent(*a, **k)

        def __getattr__(self, name):
            if name in ("set_staging", "unstage", "stage_stats", "set_perm"):
                raise AttributeError(name)
            return getattr(self._a, name)

        def set_layer(self, i, slot_of, projs, quota=(), ring=-1):
            return self._a.set_layer(i, slot_of, projs, list(quota), ring)

    m = types.ModuleType("moe_native")
    for name in dir(real):
        if not name.startswith("__"):
            setattr(m, name, getattr(real, name))
    m.Agent = OldAgent

    def signal_after(x, ev, v, agent, layer, *, stream=None):
        return real.signal_after(x, ev, v, None if agent is None else agent._a, layer,
                                 stream=stream)
    m.signal_after = signal_after
    return m


def main():
    sys.path.append(str(Path(__file__).resolve().parent.parent / "native" / "build"))
    try:
        import moe_native  # noqa: F401
    except ImportError:
        print("test_staging: skipped (needs moe_native built by native/build.sh)")
        return
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "toy8"
        ts.build(out, tn.CFG8)
        print("1. budget toy, one store:")
        total = check_arms(out)
        print(f"   totals over the arms: {total}")

        import test_stripe
        meta = test_stripe.mirror(out, Path(d) / "drive", 1.0)
        os.environ["MOE_STRIPE"] = str(meta)
        print("2. with a sidecar mirror:")
        check_arms(out, sidecar=meta)
        os.environ.pop("MOE_STRIPE", None)

        for bias in (0.0, 0.5):
            set_stage("0")
            a = run_resize(out, False, bias)
            for s in ("1", "1+nocache+pool"):
                set_stage(s)
                s0 = stream_fast.STATS["stage_unstaged"]
                b = run_resize(out, True, bias)
                same_run(a, b, f"resize, bias {bias}, MOE_STAGING={s}", state=False)
                assert stream_fast.STATS["stage_unstaged"] > s0, "no hand-back moved a row"
        print(f"3. a store resize every 4th decode token: {len(a[0])} tokens identical, "
              f"bias 0 and 0.5, threads and pool readers")

        check_unstage(out)

        # 5. Two expert shapes, and no budget store: off at install.
        built, wired = [], []
        real_init, real_guess = stream_fast.ExpertStore.__init__, stream_fast._stage_guess

        def init(st, *a, **k):
            real_init(st, *a, **k)
            built.append(st.bufs[0][0].shape[0] - st.slots)

        def guess(*a, **k):
            wired.append(1)
            return real_guess(*a, **k)
        stream_fast.ExpertStore.__init__, stream_fast._stage_guess = init, guess
        try:
            ts.TWO_BIT = (1, 2)
            out2 = Path(d) / "toy8two"
            ts.build(out2, tn.CFG8)
            a, _ = arm(out2, False, "0", 0.5)
            built.clear()
            b, st = arm(out2, True, "1", 0.5)
            same_run(a, b, "two stores, MOE_STAGING=1")
            assert stream_fast._STAGE == "0" and built and not any(built) and not wired, \
                (stream_fast._STAGE, built, wired)
            assert not any(st.values()), st
            ts.TWO_BIT = ()
            out4 = Path(d) / "toy4"
            ts.build(out4)
            a, _ = arm(out4, False, "0", 0.5, run=tn.run)
            b, st = arm(out4, True, "1", 0.5, run=tn.run)
            assert a[0] == b[0] and not wired and not any(st.values()), (wired, st)
        finally:
            stream_fast.ExpertStore.__init__, stream_fast._stage_guess = real_init, real_guess
            ts.TWO_BIT = ()
        print("5. two expert shapes and no budget store: staging off at install, no rows, "
              "no guesses, the same tokens")

        # 6. An older moe_native.
        import moe_native as real
        sys.modules["moe_native"] = old_module(real)
        try:
            a, _ = arm(out, False, "0", 0.5)
            for s in ("0", "1"):
                b, st = arm(out, True, s, 0.5)
                same_run(a, b, f"old moe_native, MOE_STAGING={s}")
                assert b[4] and stream_fast._STAGE == "0" and not any(st.values()), st
        finally:
            sys.modules["moe_native"] = real
        print("6. an older moe_native: native passes with the flag off, and asked for, "
              "staging turns itself off; the same tokens")
    print("ok")


if __name__ == "__main__":
    main()
