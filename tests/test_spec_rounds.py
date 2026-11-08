#!/usr/bin/env python3
"""The speculator's round contract on a toy model, no checkpoint.

    python3 tests/test_spec_rounds.py

What is asserted is the thing a conversation depends on and a benchmark never
notices: whatever the draft head guesses, the cache ends up holding exactly
the tokens that were emitted, and it holds them as a plain prefill would.

Greedy equivalence is REPORTED, not asserted -- what is asserted is that every
position where the two differ is a near-tie. A multi-position pass does not
reduce in the same order as the same positions one at a time, so the toy
diverges from greedy at gaps of ~7e-4 in float32, which is the same effect
seen on the full model (about one position in sixty). Asserting bit-exactness
here would be asserting something that is not true of the hardware.
"""

import sys
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

import spec
from support import check_cache, greedy, spec_run, tie_gap

VERBOSE = True


def test_rounds():
    from support import toy_model
    vocab = 128
    m = toy_model(seed=0, vocab=vocab)
    ids = [3, 9, 27, 81, 5, 11, 23, 47]
    ref = list(greedy(m, ids, 24))
    ties = []

    # The rebuild at every accept point of a 5-row pass, n = k + a = 1..4: any
    # pending length and any accepted prefix. The cache must be a prefill of n rows.
    sp, rows = spec.Speculator(m, None), ref[:5]
    for n in range(1, 5):
        cache = make_prompt_cache(m)
        sp.core(mx.array([ids]), cache=cache)
        snap, gdn = spec._snapshot(cache), sp._gdn_of(cache)
        assert gdn, "the toy's GatedDeltaNet caches were not paired"
        _, tape = sp._taped(mx.array([rows]), cache, gdn)
        sp._rebuild(cache, snap, n, gdn, tape)
        worst = check_cache(m, cache, ids + rows[:n], f"rebuild at {n}/5")
    assert not any(n in g.__dict__ for _, g, ns in gdn for n in ns), "a tape was left on"
    if VERBOSE:
        print(f"  rebuild at 1..4 of 5  cache exact, max|cache-prefill| {worst:.2g}")

    # Rebuild on, then off: off is the drop-and-repeat / replay path that
    # models it cannot rebuild still take.
    for rb, name, pattern in [(rb, *c) for rb in (True, False) for c in (
                              ("all accept", [True]),
                              ("all reject", [False]),
                              ("alternating", [True, False]),
                              ("4 rejects -> replay", [False] * 5 + [True]))]:
        name = f"{name}{'' if rb else ' (no rebuild)'}"
        got, st, cache = spec_run(m, ref, pattern, ids, 16, vocab=vocab, rebuild=rb)
        worst = check_cache(m, cache, ids + got, name)
        # Only the FIRST divergence is comparable: after it the two runs are
        # continuing different prefixes, so a later mismatch says nothing.
        d = next((i for i, (a, b) in enumerate(zip(got, ref)) if a != b), None)
        gap = tie_gap(m, ids + got[:d]) if d is not None else None
        if d is not None:
            ties.append(gap)
            assert gap < 1e-2, (f"{name}: token {d} differs from greedy at a "
                                f"top-2 gap of {gap:.3g} -- not a near-tie")
        if VERBOSE:
            print(f"  {name:22s} {len(got)} tokens, {st.rounds} rounds, "
                  f"acceptance {st.acceptance:.2f}, {st.per_round:.2f}/round, "
                  f"rebuilds {st.rebuilds}, replays {st.replays}, greedy agrees to "
                  f"{d if d is not None else 16}/16"
                  + (f" (tie {gap:.1e})" if d is not None else "")
                  + f", max|cache-prefill| {worst:.2g}")
        if pattern[0] is False:
            assert (st.rebuilds if rb else st.replays) > 0, f"{name}: the branch was never reached"

    # EOS inside a round stops there, wherever in the round it lands. With
    # every round accepted, index 0 is the prefill's token and each round then
    # emits an odd index (the verified token) and an even one (the second half
    # of the accepted pair), so both halves have to be covered -- and the id
    # has to be one that does not occur earlier, or the reply stops before it.
    ks = [k for k in range(1, 12) if ref[k] not in ref[:k]][:5]
    assert any(k % 2 for k in ks) and any(k % 2 == 0 for k in ks), ks
    for k in ks:
        got, _, cache = spec_run(m, ref, [True], ids, 16, eos=(ref[k],),
                                 vocab=vocab)
        assert got[-1] == ref[k] and len(got) == k + 1, (k, got, ref[:k + 1])
        check_cache(m, cache, ids + got, f"eos at {k}")
    if VERBOSE:
        print(f"  eos at {','.join(map(str, ks)):14s} stops ON the eos token, "
              f"cache exact")

    # max_tokens is a hard cap even though an accepted round emits two tokens.
    for n in (1, 2, 3, 7):
        got, _, cache = spec_run(m, ref, [True], ids, n, vocab=vocab)
        assert len(got) == n, (n, got)
        check_cache(m, cache, ids + got, f"max_tokens {n}")
    if VERBOSE:
        print("  max_tokens 1,2,3,7     exact, never overshoots a round")

    # ---- gamma > 1: a round drafts gamma tokens and verifies gamma+1 -------
    # The patterns are chosen for the number of positions the round has to roll
    # back, which is the thing gamma adds and the thing that can corrupt a
    # cache: 0 (everything verified, nothing to undo), 1, and gamma.
    for g in (2, 3):
        cases = [("all accept", [True], g),
                 ("last rejected", [True] * (g - 1) + [False], g - 1),
                 ("all rejected", [False], 0)]
        for rb, (name, pattern, want_a) in [(rb, c) for rb in (True, False) for c in cases]:
            rolled = g - want_a          # speculative positions to undo
            name = f"{name}{'' if rb else ' (no rebuild)'}"
            # 12 tokens, not 16: this toy leaves greedy at token 13 on a 7e-4
            # tie, and past that point the script's "accept" is proposing a
            # token the target is no longer about to produce, which would read
            # as a rejection the script did not ask for.
            got, st, cache = spec_run(m, ref, pattern, ids, 12, vocab=vocab,
                                      gamma=g, rebuild=rb)
            worst = check_cache(m, cache, ids + got, f"g{g} {name}")
            d = next((i for i, (a, b) in enumerate(zip(got, ref)) if a != b),
                     None)
            assert st.rounds and len(st.seen) == want_a + (want_a < g), \
                (name, st.seen, st.hit)
            if d is None:            # the script means what it says
                assert st.hit[:want_a] == st.seen[:want_a], (name, st.seen,
                                                             st.hit)
            if want_a < g:
                assert st.hit[want_a] == 0, (name, st.seen, st.hit)
            gap = tie_gap(m, ids + got[:d]) if d is not None else None
            if d is not None:
                ties.append(gap)
                assert gap < 1e-2, (f"g{g} {name}: token {d} differs from "
                                    f"greedy at a top-2 gap of {gap:.3g}")
            if VERBOSE:
                print(f"  g={g} {name:28s} {len(got)} tokens, {st.rounds} "
                      f"rounds, {st.per_round:.2f}/round, {rolled} position(s) "
                      f"rolled back, rebuilds {st.rebuilds}, replays {st.replays}, "
                      f"greedy agrees to "
                      f"{d if d is not None else 12}/12"
                      + (f" (tie {gap:.1e})" if d is not None else "")
                      + f", max|cache-prefill| {worst:.2g}")

        # eos and max_tokens against this gamma's OWN all-accept run rather
        # than against greedy: what is under test is that a round stops on the
        # token it is told to stop on, wherever in the round it falls, and a
        # near-tie divergence from greedy would only obscure that.
        base, _, _ = spec_run(m, ref, [True], ids, 16, vocab=vocab, gamma=g)
        # One eos position per offset within a round, so every place a round
        # can end is covered. Index 0 is the prefill's token; after it each
        # accepted round emits g+1. The id has to be one that does not occur
        # earlier or the reply stops before reaching it.
        ks = []
        for r in range(g + 1):
            fits = [k for k in range(1, len(base))
                    if k % (g + 1) == r and base[k] not in base[:k]]
            assert fits, (g, r, base)
            ks.append(fits[0])
        for k in ks:
            got, _, cache = spec_run(m, ref, [True], ids, 16, eos=(base[k],),
                                     vocab=vocab, gamma=g)
            assert got == base[:k + 1], (g, k, got, base[:k + 1])
            check_cache(m, cache, ids + got, f"g{g} eos at {k}")
        for n in range(1, 2 * g + 3):
            got, _, cache = spec_run(m, ref, [True], ids, n, vocab=vocab,
                                     gamma=g)
            assert got == base[:n], (g, n, got)
            check_cache(m, cache, ids + got, f"g{g} max_tokens {n}")
        if VERBOSE:
            print(f"  g={g} eos at {','.join(map(str, ks))} stops ON the eos "
                  f"token; max_tokens 1..{2 * g + 2} exact; cache exact")

    print(f"  {len(ties)} greedy divergences, worst top-2 gap "
          f"{max(ties, default=0.0):.3g}  OK")


if __name__ == "__main__":
    test_rounds()
    print("test_spec_rounds: ok")