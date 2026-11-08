#!/usr/bin/env python3
"""The four think-stub modes, on the toy engine and the real tokenizer.

    python3 tests/test_stub_modes.py

keep leaves one stub per turn in `_seen` and in the cache. slice removes those
rows from the KV and nothing else, and every turn still ends with offset - gap
== len(_seen). refeed's cache equals a fresh prefill of the same tokens and is
a prefix of the official rendering. none stops the prompt at the assistant
header. The messages= path is checked for all four.

Not on the CPU device, though it would be the polite thing: mlx_lm's
stream_generate opens `wired_limit`, which reads
`max_recommended_working_set_size` out of mx.device_info() and that key only
exists on Metal. The toy is 64 MB of weights and a handful of forward passes,
so the GPU barely notices.
"""

import sys
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.cache import KVCache, make_prompt_cache

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

import moe_stream
from support import find, gdn, has_tokenizer, kv, max_abs, toy_llm


def test_modes():
    if not has_tokenizer():
        print("test_stub_modes: SKIPPED (no tokenizer)")
        return
    Q = ["Say the word apple.", "Now say the word banana."]

    # --- keep: one stub per turn, in the tokens and in the cache ----------
    keep = toy_llm("keep")
    L = len(keep._stub_ids)
    assert L, "this tokenizer's generation prompt has no think stub to test"
    for q in Q:
        keep.ask(q)
    assert len(find(keep._seen, keep._stub_ids)) == 2, (
        f"keep must leave one stub per turn, found "
        f"{find(keep._seen, keep._stub_ids)}")
    assert keep._gap == 0 and keep.stats()["kv_gap"] == 0
    for c in keep._cache:
        if isinstance(c, KVCache):
            assert c.offset == len(keep._seen), (c.offset, len(keep._seen))

    # --- slice: the stub's rows leave the KV, nothing else moves ----------
    # Same seed, same weights, greedy sampling: turn one is token-for-token
    # identical to the keep engine, so the sliced cache can be compared
    # against the keep cache with exactly those rows cut out. That is a
    # stronger statement than any shape assertion -- it says the slice removed
    # the stub and NOTHING ELSE.
    one, sl = toy_llm("keep"), toy_llm("slice")
    one.ask(Q[0])
    sl.ask(Q[0])
    at = find(one._seen, one._stub_ids)
    assert len(at) == 1, at
    at = at[0]
    assert sl._seen == one._seen[:at] + one._seen[at + L:], "slice cut the wrong ids"
    assert sl._gap == L
    for (k, v), c_keep, c_sl in zip(kv(sl._cache),
                                    [c for c in one._cache if isinstance(c, KVCache)],
                                    [c for c in sl._cache if isinstance(c, KVCache)]):
        n = c_keep.offset
        kk, vv = c_keep.keys[..., :n, :], c_keep.values[..., :n, :]
        want_k = mx.concatenate([kk[..., :at, :], kk[..., at + L:, :]], axis=2)
        want_v = mx.concatenate([vv[..., :at, :], vv[..., at + L:, :]], axis=2)
        assert max_abs(k, want_k) == 0.0 and max_abs(v, want_v) == 0.0, \
            "the sliced keys are not the keep keys minus the stub rows"
        # RoPE offset untouched, storage one stub shorter: the next turn's
        # keys land at offset - gap, which is exactly len(_seen).
        assert c_sl.offset == n, (c_sl.offset, n)
        assert c_sl.offset - c_sl.gap == len(sl._seen)

    sl.ask(Q[1])
    assert not find(sl._seen, sl._stub_ids), "slice left a stub in _seen"
    assert sl._gap == 2 * L, sl._gap
    for c in sl._cache:
        if isinstance(c, KVCache):
            assert c.offset - c.gap == len(sl._seen), (c.offset, c.gap,
                                                       len(sl._seen))
            assert c.offset == len(sl._seen) + 2 * L
    sl.ask(Q[0])          # a third turn on the gapped cache must still run

    # --- refeed: the cache IS the official rendering ----------------------
    rf = toy_llm("refeed")
    for q in Q:
        rf.ask(q)
    assert not find(rf._seen, rf._stub_ids), "refeed left a stub in _seen"
    assert rf._gap == 0
    for c in rf._cache:
        if isinstance(c, KVCache):
            assert c.offset == len(rf._seen), (c.offset, len(rf._seen))
    # A cache built turn by turn, with a stub fed and then undone twice, must
    # equal one prefill of the tokens it claims to hold -- for the attention
    # keys and for the GatedDeltaNet states both. The tolerance is for the
    # arithmetic: forty positions in one matmul do not reduce in the same order
    # as four calls of ten.
    fresh = make_prompt_cache(rf.model)
    mx.eval(rf.model(mx.array(rf._seen)[None], cache=fresh),
            [c.state for c in fresh])
    worst = 0.0
    for (a, b), (c, d) in zip(kv(rf._cache), kv(fresh)):
        worst = max(worst, max_abs(a, c), max_abs(b, d))
    for a, b in zip(gdn(rf._cache), gdn(fresh)):
        worst = max(worst, max_abs(a, b))
    assert worst < 1e-3, f"refeed cache differs from a fresh prefill by {worst}"
    # ... and those tokens are a prefix of the official rendering. Rendered
    # with a NEXT user turn appended, because that is the only way the two
    # turns already taken become PAST turns: Qwen's template keeps the think
    # block on the LAST assistant message and strips it from every earlier
    # one, which is the asymmetry this whole flag exists for. The prefix stops
    # at the reply -- the `<|im_end|>` closing it arrives at the head of the
    # next turn's separator, which is where the engine has always fed it.
    official = rf._raw_render(rf.history + [{"role": "user", "content": "Q3"}])
    got = rf.tokenizer.decode(rf._seen)
    assert " thinking" not in got, got[:200]
    assert official.startswith(got), (
        f"refeed did not reproduce the official rendering\n"
        f"  got      {got[-120:]!r}\n  official {official[-120:]!r}")

    # --- none: the generation prompt stops at the assistant header --------
    no = toy_llm("none")
    no.ask(Q[0])
    hdr = no.tokenizer.encode(no._tpl["after_user"], add_special_tokens=False)
    fed = no._seen[:no.stats()["prompt_tokens"]]
    assert fed[-len(hdr):] == hdr, "none must end the prompt at assistant\\n"
    assert not find(no._seen, no._stub_ids)
    assert isinstance(no.stats()["opened_think"], bool)

    # --- the messages= path, for all four ---------------------------------
    # A full render is the one route that does not go through _next_chunk, and
    # every mode has to leave `_seen` equal to what the cache holds -- storage,
    # not RoPE -- or the turn after it is built against tokens the model never
    # saw. That is the invariant the whole engine rests on.
    for mode in moe_stream.StreamingLLM.STUB_MODES:
        m = toy_llm(mode)
        m.ask(messages=[{"role": "system", "content": "Be brief."},
                        {"role": "user", "content": Q[0]}])
        m.ask(Q[1])
        for c in m._cache:
            if isinstance(c, KVCache):
                assert c.offset - getattr(c, "gap", 0) == len(m._seen), (
                    mode, c.offset, getattr(c, "gap", 0), len(m._seen))
        want = 2 if mode == "keep" else 0
        assert len(find(m._seen, m._stub_ids)) == want, (mode, m._seen[-24:])

    print(f"  stub {keep._stub_ids} ({keep._stub!r}), refeed max|cache - "
          f"fresh prefill| = {worst:.3g}, opened_think none/keep = "
          f"{no.stats()['opened_think']}/{keep.stats()['opened_think']}  OK")


if __name__ == "__main__":
    test_modes()
    print("test_stub_modes: ok")