#!/usr/bin/env python3
"""window_cache.py: the draft-only sink+window KV cache.

    python3 tests/test_window_cache.py          # toy only, no checkpoint
    python3 tests/test_window_cache.py --real   # + Qwen3-0.6B-4bit

Four checks, all but the last on spec_draft._toy (a random llama, fp32, no
weights on disk):

 1. exact    -- until the first compaction a WindowKVCache IS a KVCache:
                greedy decode through both, logits bit-identical, and after
                200 tokens storage is still bounded by sink+window+window/4.
 2. rewind   -- trim(k) then re-feed the same k tokens reproduces the pre-trim
                logits bit for bit, ACROSS a compaction. This is what
                spec_draft._rewind does every rejected round, so it is the one
                that would corrupt a conversation if it were only approximate.
 3. mask     -- make_mask(3) against a mask built by an explicit loop.
 4. engine   -- the wiring: two turns through moe_stream.StreamingLLM with a
                windowed draft must reply EXACTLY what the same engine replies
                with no draft at all. Speculative decode emits the target's
                argmax whatever the draft proposes, so a window can only cost
                acceptance -- if the reply moves, the cache is corrupting
                _sync_draft_cache's trim/refeed, not merely drafting worse.
 5. real     -- Qwen3-0.6B-4bit, 64 greedy tokens after a ~1500-token context,
                window 512: reports how many agree with the full-cache run
                (they need not all agree -- a smaller window is a different
                draft distribution, which is the alpha side of the trade) and
                the stored length. Skipped unless --real: it loads a real
                checkpoint.
"""

import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import spec_draft                                              # noqa: E402
import support                                                 # noqa: E402
import window_cache                                            # noqa: E402
from window_cache import WindowKVCache, to_window              # noqa: E402
from mlx_lm.models.cache import make_prompt_cache              # noqa: E402

DRAFT = "mlx-community/Qwen3-0.6B-4bit"


def _decode(model, cache, prompt, n):
    """Greedy decode, one row at a time, returning every logits row.

    Deliberately not mlx_lm.generate_step: the point here is to control the
    shape of every pass (prefill in one block, then single rows) so that the
    two runs make the same compaction decisions at the same moments.
    """
    logits = model(mx.array(prompt)[None], cache=cache)[0, -1]
    out, tok = [logits], int(mx.argmax(logits))
    for _ in range(n - 1):
        logits = model(mx.array([tok])[None], cache=cache)[0, -1]
        mx.eval(logits)
        out.append(logits)
        tok = int(mx.argmax(logits))
    return out


def test_exact():
    """A window cache is a plain cache until the first row is dropped."""
    mx.random.seed(0)
    model = spec_draft._toy(200, 32, 2)
    prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    sink, window, n = 8, 32, 200
    cap, slack = sink + window, window // 4

    full = make_prompt_cache(model)
    win = to_window(make_prompt_cache(model), sink=sink, window=window)
    assert isinstance(win[0], WindowKVCache) and win[0].sink == sink

    a = _decode(model, full, prompt, n)
    b = _decode(model, win, prompt, n)

    # Step i (0-based) appends one row onto a cache holding len(prompt)+i of
    # them (the prefill is step 0's own append), so the first drop happens at
    # the step where that count first exceeds cap+slack.
    exact_steps = cap + slack - len(prompt) + 1
    assert 0 < exact_steps < n, exact_steps
    for i in range(exact_steps):
        assert mx.array_equal(a[i], b[i]), f"logits differ at step {i} < first drop"
    print(f"  exact: {exact_steps} steps bit-identical to a full cache "
          f"(sink {sink} + window {window} + slack {slack} = "
          f"{cap + slack} rows), then the first drop")

    assert full[0].offset == win[0].offset == len(prompt) + n - 1, win[0].offset
    assert win[0].stored <= cap + slack, win[0].stored
    assert win[0].stored > cap - 1, win[0].stored
    assert full[0].size() > 3 * win[0].stored, (full[0].size(), win[0].stored)
    # The point of the whole file: rows read per step stopped growing. (Not
    # `nbytes`, which reports the ALLOCATION -- both caches sit in the same
    # 256-row buffer KVCache.step rounds up to; what attention reads is the
    # slice update_and_fetch hands back, which is `stored` rows wide.)
    print(f"  bounded: after {n} tokens offset {win[0].offset} (absolute, for "
          f"RoPE) but only {win[0].stored} rows stored vs {full[0].size()} "
          f"full -- {win[0].stored / full[0].size():.2f}x the bytes read")


def test_rewind():
    """trim(k) + re-feed == never trimmed, across a compaction.

    spec_draft._rewind trims the draft cache by the rows a rejected round
    over-fed, and the next round feeds those positions again. If that were
    only approximately reversible the draft would drift away from the tokens
    the target actually committed, silently, and only acceptance would show it.
    """
    mx.random.seed(0)
    model = spec_draft._toy(200, 32, 2)
    prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    sink, window, n, k = 8, 32, 120, 5

    win = to_window(make_prompt_cache(model), sink=sink, window=window)
    rows = _decode(model, win, prompt, n)
    assert win[0].stored < win[0].offset, "no compaction happened, test is void"
    ref, off, stored = rows[-1], win[0].offset, win[0].stored
    tail = [int(mx.argmax(r)) for r in rows[-k - 1:-1]]

    trimmed = [c.trim(k) for c in win]
    assert trimmed == [k] * len(win), trimmed
    assert win[0].offset == off - k and win[0].stored == stored - k

    for t in tail:
        logits = model(mx.array([t])[None], cache=win)[0, -1]
    mx.eval(logits)
    assert win[0].offset == off and win[0].stored == stored
    assert mx.array_equal(logits, ref), "refeed after trim changed the logits"
    print(f"  rewind: trim({k}) + refeed reproduces the pre-trim logits bit "
          f"for bit at offset {off} / {stored} stored rows (compaction "
          f"dropped {off - stored})")

    # A trim deeper than what is stored must report the shortfall rather than
    # pretend: moe_stream._sync_draft_cache rebuilds the cache on that signal.
    assert win[0].trim(10 ** 6) == stored
    print(f"  rewind: an over-deep trim returns {stored}, not the "
          f"{10 ** 6} asked -- the caller's rebuild signal")


def test_mask():
    """make_mask(N) against the same thing spelled out as a loop."""
    c = WindowKVCache(window=32, sink=8)
    assert c.make_mask(1) is None, "N=1 needs no mask: everything stored is past"

    B, H, D = 1, 2, 4
    for fill in (5, 8 + 32 + 8):           # before, and at, the drop threshold
        c = WindowKVCache(window=32, sink=8)
        c.update_and_fetch(mx.zeros((B, H, fill, D)), mx.zeros((B, H, fill, D)))
        N = 3
        s = c._stored_after(N)             # old rows that survive this pass
        got = c.make_mask(N)
        want = [[True] * s + [j <= i for j in range(N)] for i in range(N)]
        assert got.shape == (N, s + N), (got.shape, s, N)
        assert got.tolist() == want, f"fill={fill}\n{got.tolist()}\n{want}"
        # ... and the mask must still be the right WIDTH once the pass runs.
        keys, _ = c.update_and_fetch(mx.zeros((B, H, N, D)),
                                     mx.zeros((B, H, N, D)))
        assert keys.shape[2] == s + N, (keys.shape, s, N)
        print(f"  mask: N=3 over {s} stored rows matches the loop-built mask "
              f"(fill {fill} -> {c.stored} stored, offset {c.offset})")

    try:
        WindowKVCache(window=32).make_mask(3, window_size=16)
    except ValueError:
        print("  mask: a model-side sliding window is refused, not guessed")
    else:
        raise AssertionError("window_size should have been refused")


def test_engine():
    """moe_stream, two turns, windowed draft vs no draft.

    Toy weights around the checkpoint's real tokenizer -- the same harness
    spec_draft.engine_selftest uses, and for the same reason: it is the only
    thing that exercises _new_draft_cache, _sync_draft_cache's token-id diff
    (which trims the draft cache and re-feeds the difference across a turn
    boundary) and the flat target+draft cache list, all against a cache whose
    storage no longer matches its offset.
    """
    if not support.has_tokenizer():
        print("engine: SKIPPED (no tokenizer: set MOE_TOY_TOKENIZER)")
        return
    from mlx_lm.utils import load_tokenizer
    import moe_stream

    tok = load_tokenizer(support.tokenizer_dir())
    vocab = max([tok.vocab_size] + list(tok.get_added_vocab().values())) + 1
    target, draft, _ = spec_draft._drafts(vocab, 32, 3, 2)

    def build(window, tau):
        llm = moe_stream.StreamingLLM.__new__(moe_stream.StreamingLLM)
        llm.model, llm.tokenizer = target, tok
        llm.max_tokens, llm.prefill_step, llm.resident = 16, None, 0
        llm.thinking, llm.profile = False, None
        llm.model_path = support.MODEL_LABEL
        llm._setup_chat("keep")
        llm._draft_model = draft if window is not None else None
        llm._draft_seen = []
        llm.draft_tokens, llm.draft_tau = 4, tau
        llm.draft_window = window or 0
        llm.reset()
        return llm

    # One engine per setting, every question on it: a multi-turn conversation
    # is what puts _sync_draft_cache's trim/refeed on the path at all, and it
    # has to run past sink+window (64+16=80 rows here) for a single row to be
    # dropped -- a draft cache that never compacted proves nothing.
    qs = ["Say the word apple.", "Now say the word banana.",
          "And the word cherry.", "Now the word plum.", "Now the word pear."]
    base = build(None, None)
    ref = [base.ask(q) for q in qs]
    for window, tau in ((0, None), (16, None), (16, 0.0), (16, 2.0)):
        llm = build(window, tau)
        reps = [llm.ask(q) for q in qs]
        assert reps == ref, (f"draft_window={window} draft_tau={tau} changed "
                             f"the reply\n  baseline: {ref}\n  windowed: {reps}")
        c = llm._draft_cache[0]
        kind = type(c).__name__
        if window:
            assert c.stored < c.offset, (c.stored, c.offset, "never compacted")
        print(f"  engine: draft_window={window} draft_tau={tau} -> same reply, "
              f"draft cache {kind} offset {c.offset} stored "
              f"{getattr(c, 'stored', c.offset)}")
    assert isinstance(build(16, None)._draft_cache[0], WindowKVCache)


def test_real():
    """Qwen3-0.6B-4bit: how much of a 64-token greedy run survives window 512.

    Reported, not asserted. A window is a different draft distribution, so
    tokens are allowed to differ; what the engine cares about is the drop in
    acceptance, and this is the cheap proxy for it.
    """
    import mlx_lm
    model, tokenizer = mlx_lm.load(DRAFT)
    text = ("The following is a transcript of a technical discussion about "
            "memory bandwidth, speculative decoding and KV caches. ") * 40
    prompt = tokenizer.encode(text)[:1500]
    n, window = 64, 512

    full = make_prompt_cache(model)
    win = to_window(make_prompt_cache(model), window=window)
    a = [int(mx.argmax(r)) for r in _decode(model, full, prompt, n)]
    b = [int(mx.argmax(r)) for r in _decode(model, win, prompt, n)]
    agree = sum(x == y for x, y in zip(a, b))
    prefix = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), n)
    print(f"  real: {agree}/{n} tokens agree with the full-cache run "
          f"({prefix} before the first divergence); stored "
          f"{win[0].stored} rows vs {full[0].size()} full, offset "
          f"{win[0].offset}, {win[0].stored / full[0].size():.3f}x the rows "
          f"read per step")


if __name__ == "__main__":
    test_exact()
    test_rewind()
    test_mask()
    test_engine()
    if "--real" in sys.argv:
        test_real()
    else:
        print("  real: skipped (pass --real; loads Qwen3-0.6B-4bit)")
    print("test_window_cache OK")
