#!/usr/bin/env python3
"""`mtp=` through the whole engine, on the toy model and a scripted draft.

    python3 tests/test_mtp_seam.py

The real head is never opened: what is checked here is the seam between
spec.py and moe_stream.py -- that the reply is the accepted tokens and nothing
else, that the cache and `_seen` still agree afterwards, and that the turn
after it is still an increment rather than a re-render. Whether the head
guesses well needs the checkpoint and a benchmark, not this file.
"""

import sys
from pathlib import Path

from mlx_lm.models.cache import KVCache

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

import spec
from support import ScriptedSpeculator, has_tokenizer, toy_llm

Q = ["Say the word apple.", "Now say the word banana."]


def no_speculator(*a, **kw):
    raise AssertionError("a Speculator was constructed with mtp off")


def test_seam():
    if not has_tokenizer():
        print("test_mtp_seam: SKIPPED (no tokenizer)")
        return

    # --- flag off: the speculator is not merely unused, it is never built ---
    # An import-time mistake could leave `_spec` set on every engine and cost
    # the default path a whole extra code path without changing a single
    # printed number. So the class itself is booby-trapped for the turn.
    off = toy_llm("slice")
    assert off._spec is None, "the flag is off and a speculator exists"
    real, spec.Speculator = spec.Speculator, no_speculator
    try:
        off.ask(Q[0])
    finally:
        spec.Speculator = real
    assert "mtp_acceptance" not in off.stats(), off.stats()

    # --- a head that is not there falls back, and says so ------------------
    assert off._load_speculator("/nonexistent/mtp-head") is None, \
        "a missing head has to fall back, not raise and not pretend"

    # --- flag on: same tokens, same cache, and a second turn that continues -
    # 24 tokens, not the toy default of 6: the pattern below needs a run of
    # four rejections to reach spec.generate's replay branch, and that branch
    # is where the cache is written twice for the same positions.
    on = toy_llm("slice", max_tokens=24)
    ref = toy_llm("slice", max_tokens=24)
    ref.ask(Q[0])
    refgen = ref._seen[len(ref._seen) - ref.stats()["tokens"]:]
    on._spec = ScriptedSpeculator(
        on.model, refgen,
        [True, True, False, False, False, False, False], vocab=128)
    text = on.ask(Q[0])
    st = on.stats()
    assert st["mtp_rounds"] > 0 and 0.0 <= st["mtp_acceptance"] <= 1.0, st
    assert 1.0 <= st["mtp_tokens_per_round"] <= 2.0, st
    gen = on._seen[len(on._seen) - st["tokens"]:]
    assert text == on.tokenizer.decode(
        [t for t in gen if t not in on.tokenizer.eos_token_ids]), (
        f"the streamed text is not the accepted tokens\n"
        f"  streamed {text!r}\n  decoded  "
        f"{on.tokenizer.decode(gen)!r}")
    # The stub slice runs at the end of the turn and asserts this itself, so
    # reaching here already proves it; asserted again because a future slice
    # that stops asserting must not take the check away with it.
    for c in on._cache:
        if isinstance(c, KVCache):
            assert c.offset - c.gap == len(on._seen), (c.offset, c.gap,
                                                       len(on._seen))
    on.ask(Q[1])
    assert on.stats()["cached_tokens"] > 0, \
        "the turn after a speculative reply re-rendered instead of continuing"
    for c in on._cache:
        if isinstance(c, KVCache):
            assert c.offset - c.gap == len(on._seen)

    # --- gamma 2: a round that emits three tokens still leaves the cache -----
    # exact and the reply still equals the accepted tokens. The rollback
    # arithmetic is tests/test_spec_rounds.py's; what this adds is the seam --
    # `_seen`, the stub slice and the streamed text all see a round that is
    # wider than two.
    g2 = toy_llm("slice", max_tokens=24)
    g2._spec = ScriptedSpeculator(
        g2.model, refgen, [True, True, False], vocab=128, gamma=2)
    text2 = g2.ask(Q[0])
    st2 = g2.stats()
    gen2 = g2._seen[len(g2._seen) - st2["tokens"]:]
    assert text2 == g2.tokenizer.decode(
        [t for t in gen2 if t not in g2.tokenizer.eos_token_ids]), text2
    assert 1.0 <= st2["mtp_tokens_per_round"] <= 3.0, st2
    assert len(st2["mtp_depth"]) == 2, st2
    for c in g2._cache:
        if isinstance(c, KVCache):
            assert c.offset - c.gap == len(g2._seen), (c.offset, c.gap)
    g2.ask(Q[1])
    assert g2.stats()["cached_tokens"] > 0
    print(f"  engine g=2: reply {len(gen2)} tokens, "
          f"{st2['mtp_tokens_per_round']}/round, per-draft acceptance "
          f"{st2['mtp_depth']}, turn 2 cached {g2.stats()['cached_tokens']}")
    # Reported, not asserted, and for the reason test_spec_rounds.py spells
    # out: a multi-position pass does not reduce in the same order as the same
    # positions one at a time, so a near-tie can flip and every token after it
    # is then a different continuation. The number is here so a real
    # regression -- disagreement from the first token -- is visible.
    same = next((i for i, (a, b) in enumerate(zip(gen, refgen)) if a != b),
                min(len(gen), len(refgen)))
    print(f"  engine: reply {len(gen)} tokens, acceptance {st['mtp_acceptance']}, "
          f"{st['mtp_tokens_per_round']}/round, agrees with the ordinary path "
          f"to {same}/{len(refgen)}, turn 2 cached {on.stats()['cached_tokens']}  OK")


if __name__ == "__main__":
    test_seam()
    print("test_mtp_seam: ok")