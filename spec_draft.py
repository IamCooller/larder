#!/usr/bin/env python3
"""Adaptive draft-chain length for the external-draft speculative path: a drop-in
replacement for mlx_lm's draft-model stream that decides the chain length token by token
from the draft's own confidence."""

import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.generate import (GenerationResponse, generate_step,
                             generation_stream, wired_limit)
from mlx_lm.models import cache
from mlx_lm.tokenizer_utils import TokenizerWrapper

import window_cache

# tau is a gap in probabilities, so it lies in [0, 1] with 0 never stopping a
# chain early and anything above 1 always stopping it.
DEFAULT_TAU = 0.9
DEFAULT_K_MAX = 6

# Rows in one verify pass, root included, for a target that names no ceiling.
MAX_VERIFY_ROWS = 24

# Round bookkeeping, module-level so a harness can read the per-turn difference:
# `steps` counts the draft forwards run and `drafted` the rows that reached the
# verify pass.
STATS = {"rounds": 0, "steps": 0, "drafted": 0, "accepted": 0,
         "forced": 0, "forced_accepted": 0}


def _gap(logprobs_row):
    """top1-top2 gap in probabilities, the softmax of the raw logits (exp of a
    log-softmax is the same vector). mx.topk over the whole vocabulary rather than a sort: 2 rows of a
    150k-wide distribution, once per draft step."""
    two = mx.topk(mx.exp(logprobs_row), 2)
    return mx.max(two) - mx.min(two)


def adaptive_generate_step(prompt, model, draft_model, *,
                           draft_tau=DEFAULT_TAU,
                           num_draft_tokens=DEFAULT_K_MAX,
                           max_tokens=256, sampler=None,
                           logits_processors=None, prompt_cache=None,
                           prefill_step_size=512, draft_window=0,
                           forcer=None):
    """speculative_generate_step with the chain length gated on the draft's margin,
    yielding the same (token, logprobs, from_draft) triples."""
    if draft_tau is not None and draft_tau < 0:
        raise ValueError(f"draft_tau must be >= 0, got {draft_tau}")
    # The width of rowqmm's qmm8 kernel when the target has it installed.
    max_rows = getattr(model, "rowqmm_rows", MAX_VERIFY_ROWS)
    if max_tokens < 0:
        # mlx_lm documents -1 as infinite but its own path breaks on it, so refuse
        # rather than inherit the bug.
        raise ValueError("max_tokens must be >= 0 for speculative decode")

    y = prompt.astype(mx.uint32)
    if prompt_cache is None:
        model_cache = cache.make_prompt_cache(model)
        draft_cache = cache.make_prompt_cache(draft_model)
        if draft_window:
            window_cache.to_window(draft_cache, window=draft_window)
    else:
        model_cache = prompt_cache[: len(model.layers)]
        draft_cache = prompt_cache[len(model.layers):]
    if not cache.can_trim_prompt_cache(model_cache):
        types = {type(c).__name__ for c in model_cache if not c.is_trimmable()}
        raise ValueError(f"Speculative decoding requires a trimmable prompt "
                         f"cache (got {types}).")

    sampler = sampler or (lambda x: mx.argmax(x, axis=-1))
    # Token history for logits processors, kept as a python list until one exists so the
    # default path allocates nothing.
    hist = prompt.tolist() if logits_processors else None

    def _step(m, c, ids, n_predict=1, chain=()):
        """`chain` lists the ids fed after hist's last entry, one per row beyond the first,
        so each row is scored against the history mlx_lm's own step would use."""
        with mx.stream(generation_stream):
            logits = m(ids[None], cache=c)
            logits = logits[:, -n_predict:, :].squeeze(0)     # (n_predict, V)
            if logits_processors:
                rows = []
                for i in range(n_predict):
                    toks = mx.array(hist + list(chain[:i]), mx.uint32)
                    row = logits[i:i + 1]           # keep the batch dim: some
                    for processor in logits_processors:  # processors (e.g.
                        row = processor(toks, row)  # repetition_penalty) index
                    rows.append(row)                # logits[:, tokens]
                logits = mx.concatenate(rows, axis=0)
            logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            return sampler(logprobs), logprobs

    def _prefill(m, c, ids):
        while ids.size > 1:
            n = min(prefill_step_size, ids.size - 1)
            m(ids[:n][None], cache=c)
            mx.eval([x.state for x in c])
            ids = ids[n:]
            mx.clear_cache()
        return ids

    with mx.stream(generation_stream):
        draft_y = _prefill(draft_model, draft_cache, y)
        y = _prefill(model, model_cache, y)

    # Trims owed to the caches for the round in flight, in one place so the finally
    # below cannot rewind twice.
    pending = None

    def _rewind():
        nonlocal pending
        if pending is None:
            return
        trim_model, trim_draft = pending
        pending = None
        if trim_model:
            cache.trim_prompt_cache(model_cache, trim_model)
        if trim_draft:
            cache.trim_prompt_cache(draft_cache, trim_draft)

    ntoks = 0
    try:
        while ntoks < max_tokens:
            # Same budget rule as mlx_lm's `min(max_tokens - ntoks, k)`: the
            # round still yields one target token past the chain, so this only
            # keeps the last round from drafting rows nobody can accept.
            room = max_tokens - ntoks
            # Chain rows of this pass: forced and drafted, interleaved, the
            # whole pass capped at max_rows.
            cap = min(room, max_rows - 1)
            forced = forcer.propose()[:cap] if forcer is not None else []
            c_arrs = [mx.array([t], mx.uint32) for t in forced]
            kinds = [True] * len(forced)     # True = a forced row
            # Inside a call a fork of the forcer walks this round's own chain, so a drafted token
            # that closes a string gets the span the grammar implies after it.
            spec = forcer.fork() if forcer is not None and forcer.active else None
            for t in forced:
                spec.feed(t)

            # Draft: step j feeds the previous token and produces the next position,
            # whose gap the gate reads after the forward that computed it.
            steps = fed = 0
            dy = mx.concatenate([draft_y] + c_arrs) if c_arrs else draft_y
            fed_next = len(c_arrs)
            while steps < num_draft_tokens and len(c_arrs) < cap:
                tok, lp = _step(draft_model, draft_cache, dy)
                steps += 1
                # Chain rows the draft itself consumed, never the last drafted token and
                # the span after it, which is where the +-1 of mlx_lm's rewind comes
                # from.
                fed += fed_next
                if draft_tau is None:
                    mx.async_eval(tok)
                else:
                    gap = _gap(lp[0])
                    mx.eval(tok, gap)
                    if float(gap) <= draft_tau:
                        break
                c_arrs.append(tok)
                kinds.append(False)
                span = []
                if spec is not None and spec.active:
                    spec.feed(tok.item())
                    span = spec.propose()[:cap - len(c_arrs)]
                    for t in span:
                        spec.feed(t)
                s_arrs = [mx.array([t], mx.uint32) for t in span]
                c_arrs += s_arrs
                kinds += [True] * len(span)
                dy = mx.concatenate([tok] + s_arrs) if s_arrs else tok
                fed_next = 1 + len(span)

            # Verify: the chain comes back to the host in one transfer, before the pass
            # only when a logits processor needs it.
            chain = (mx.concatenate(c_arrs).tolist()
                     if c_arrs and logits_processors else [])
            y_in = mx.concatenate([y] + c_arrs) if c_arrs else y
            tokens, logprobs = _step(model, model_cache, y_in,
                                     len(c_arrs) + 1, chain=chain)
            mx.eval(tokens)
            toks = tokens.tolist()
            if len(chain) < len(c_arrs):
                chain = mx.concatenate(c_arrs).tolist()

            n = 0
            while n < len(chain) and toks[n] == chain[n]:
                n += 1

            # Counted before the yields: a consumer that stops at EOS closes
            # the generator mid-round, and nothing after a yield runs then.
            # max_tokens can cut the accepted rows short (F7 below).
            acc = min(n, max_tokens - ntoks)
            STATS["rounds"] += 1
            STATS["steps"] += steps
            STATS["drafted"] += kinds.count(False)
            STATS["forced"] += kinds.count(True)
            STATS["forced_accepted"] += kinds[:acc].count(True)
            STATS["accepted"] += kinds[:acc].count(False)

            # The target ate y and every drafted row, of which only the accepted ones
            # are real, while the draft falls behind rather than over-running when fed <
            # n.

            # `pending` is refreshed before every yield, so a consumer that stops
            # mid-chain leaves the caches trimmed to exactly what it saw.
            for j in range(n):
                pending = (len(chain) - (j + 1), max(fed - (j + 1), 0))
                ntoks += 1
                if hist is not None:
                    hist.append(chain[j])
                if forcer is not None:
                    forcer.feed(chain[j])
                # from_draft marks the draft's rows only, not forced ones
                yield chain[j], logprobs[j], not kinds[j]
                if ntoks == max_tokens:
                    # max_tokens can cut this loop short at exactly j+1 rows, folded
                    # into `n` so the bookkeeping agrees with what was emitted.
                    n = j + 1
                    break
            if ntoks < max_tokens:
                pending = (len(chain) - n, max(fed - n, 0))
                ntoks += 1
                if hist is not None:
                    hist.append(toks[n])
                if forcer is not None:
                    forcer.feed(toks[n])
                yield toks[n], logprobs[n], False

            y = mx.array([toks[n]], mx.uint32)
            # What the draft still has to swallow before it can propose again:
            # the accepted rows it never saw, then the token the target just
            # committed. `steps == 0` means budget was 0 and draft_y itself was
            # never fed, so it stays at the front.
            carry = ([draft_y] if steps == 0 else []) + c_arrs[min(fed, n):n]
            draft_y = mx.concatenate(carry + [y]) if carry else y
            _rewind()
            if ntoks >= max_tokens:
                break
    finally:
        _rewind()


def stream(model, tokenizer, prompt, max_tokens=256, draft_model=None,
           draft_tau=DEFAULT_TAU, **kwargs):
    """mlx_lm.stream_generate's signature and response stream over adaptive_generate_step,
    reproduced because stream_generate takes no override."""
    if draft_model is None:
        raise ValueError("spec_draft.stream needs a draft_model; without one "
                         "there is no chain to adapt (use mlx_lm directly)")
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)
    if not isinstance(prompt, mx.array):
        prompt = mx.array(prompt)
    yield from _responses(model, tokenizer, prompt, max_tokens, adaptive_generate_step(
        prompt, model, draft_model, draft_tau=draft_tau,
        max_tokens=max_tokens, **kwargs))


def first_early(prompt, model, *, max_tokens=256, stop=(), sampler=None,
                logits_processors=None, prompt_cache=None, **kwargs):
    """generate_step's (token, logprobs) with the first token out before the forward that
    feeds it back in."""
    if prompt_cache is None:
        prompt_cache = cache.make_prompt_cache(model, max_kv_size=kwargs.get("max_kv_size"))
    sampler = sampler or (lambda x: mx.argmax(x, axis=-1))
    got, hist = [], []

    def grab(logprobs):
        got.append((sampler(logprobs), logprobs))
        return got[-1][0]

    def keep(toks, logits):
        hist.append(toks)
        return logits
    kw = dict(kwargs, prompt_cache=prompt_cache)
    for _ in generate_step(prompt, model, max_tokens=0, sampler=grab,
                           logits_processors=logits_processors and [keep, *logits_processors],
                           **kw):
        pass
    if max_tokens == 0:
        return
    y, logprobs = got[0]
    # The processors go on from the history the first generate_step gave them
    # (mlx_lm 0.31.3: the prompt's last id; 0.32: the whole prompt).
    procs = [lambda t, lg, p=p: p(mx.concatenate([hist[0], t]), lg)
             for p in logits_processors or ()] or None
    rest = dict(kw, sampler=sampler, logits_processors=procs)
    if max_tokens == 1 or y.item() in stop:
        for _ in generate_step(y, model, max_tokens=0, **rest):
            pass
        yield y.item(), logprobs.squeeze(0)
        return
    yield y.item(), logprobs.squeeze(0)
    yield from generate_step(y, model, max_tokens=max_tokens - 1, **rest)


def plain(model, tokenizer, prompt, max_tokens=256, **kwargs):
    """mlx_lm.stream_generate without a draft, over first_early: the same
    stream, its first token one forward sooner."""
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)
    if not isinstance(prompt, mx.array):
        prompt = mx.array(prompt)
    steps = first_early(prompt, model, max_tokens=max_tokens,
                        stop=tokenizer.eos_token_ids, **kwargs)
    yield from _responses(model, tokenizer, prompt, max_tokens,
                          ((t, lp, False) for t, lp in steps))


def _responses(model, tokenizer, prompt, max_tokens, token_generator):
    """stream_generate's loop over (token, logprobs, from_draft), field for
    field (see stream), on what mlx_lm's own module holds at the call (a
    CPU run swaps its stream and wired limit, tests/test_reuse.py)."""
    detokenizer = tokenizer.detokenizer
    gen = sys.modules["mlx_lm.generate"]   # the package's `generate` is a function
    with gen.wired_limit(model, [gen.generation_stream]):
        tic = time.perf_counter()
        for n, (token, logprobs, from_draft) in enumerate(token_generator):
            if n == 0:
                prompt_time = time.perf_counter() - tic
                prompt_tps = prompt.size / prompt_time
                tic = time.perf_counter()
            if token in tokenizer.eos_token_ids:
                break
            detokenizer.add_token(token)
            if (n + 1) == max_tokens:
                break
            yield GenerationResponse(
                text=detokenizer.last_segment, token=token, logprobs=logprobs,
                from_draft=from_draft, prompt_tokens=prompt.size,
                prompt_tps=prompt_tps, generation_tokens=n + 1,
                generation_tps=(n + 1) / (time.perf_counter() - tic),
                peak_memory=mx.get_peak_memory() / 1e9, finish_reason=None)

        detokenizer.finalize()
        yield GenerationResponse(
            text=detokenizer.last_segment, token=token, logprobs=logprobs,
            from_draft=from_draft, prompt_tokens=prompt.size,
            prompt_tps=prompt_tps, generation_tokens=n + 1,
            generation_tps=(n + 1) / (time.perf_counter() - tic),
            peak_memory=mx.get_peak_memory() / 1e9,
            finish_reason="stop" if token in tokenizer.eos_token_ids else "length")


# ---------------------------------------------------------------- selftest
def _toy(vocab, hidden, layers, heads=4, inter=64):
    from mlx_lm.models.llama import Model, ModelArgs
    m = Model(ModelArgs(model_type="llama", hidden_size=hidden,
                        num_hidden_layers=layers, intermediate_size=inter,
                        num_attention_heads=heads, rms_norm_eps=1e-5,
                        vocab_size=vocab, head_dim=hidden // heads,
                        num_key_value_heads=heads))
    mx.eval(m.parameters())
    return m


def _drain(gen):
    """Consume a (token, logprobs, from_draft) generator to exhaustion, which is what makes
    its finally-block rewind run."""
    return [(int(t), bool(fd)) for t, _, fd in gen]


def _drafts(V, hidden, layers_t, layers_d):
    """A target and two drafts sharing its vocabulary: one copied from the target so every
    chain is accepted, one noised so most are rejected."""
    from mlx.utils import tree_flatten, tree_unflatten

    mx.random.seed(0)
    target = _toy(V, hidden, layers_t)
    out = [target]
    for noise in (0.0, 0.1):
        mx.random.seed(0)
        d = _toy(V, hidden, layers_d)
        tp = dict(tree_flatten(target.parameters()))
        dp = dict(tree_flatten(d.parameters()))
        d.update(tree_unflatten(
            [(k, tp[k] + noise * mx.random.normal(tp[k].shape)) for k in dp
             if k in tp and tp[k].shape == dp[k].shape]))
        mx.eval(d.parameters())
        out.append(d)
    return out


def selftest():
    """Two toy llamas sharing a vocabulary, no checkpoint: at tau=0 the stream and cache
    offsets must equal mlx_lm's, and at tau above any gap plain greedy."""
    from mlx_lm.generate import generate_step, speculative_generate_step
    from mlx_lm.models.cache import make_prompt_cache

    V, N, K = 200, 24, 4
    target, exact, noisy = _drafts(V, 32, 3, 2)
    prompt = mx.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    def caches(d):
        tc, dc = make_prompt_cache(target), make_prompt_cache(d)
        return tc, dc, list(tc) + list(dc)

    # --- the gap itself: hand-checked against the softmax top-2 formula
    lg = mx.array([5.0, 4.0, 1.0, 0.0])
    lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
    pr = mx.exp(lp)
    assert abs(float(_gap(lp)) - float(pr[0] - pr[1])) < 1e-6, float(_gap(lp))

    # 1. tau=0: the gate can never fire, so every round drafts k_max and this file has
    # to BE speculative_generate_step.
    for name, d in (("accept-all", exact), ("reject-most", noisy)):
        before = dict(STATS)
        tc_a, dc_a, flat_a = caches(d)
        mine = _drain(adaptive_generate_step(
            prompt, target, d, draft_tau=0.0, num_draft_tokens=K,
            max_tokens=N, prompt_cache=flat_a))
        st = {k: STATS[k] - before[k] for k in STATS}
        tc_b, dc_b, flat_b = caches(d)
        ref = _drain(speculative_generate_step(
            prompt, target, d, num_draft_tokens=K, max_tokens=N,
            prompt_cache=flat_b))
        assert mine == ref, f"{name}: tau=0 != mlx_lm k={K}\n{mine}\n{ref}"
        assert (tc_a[0].offset, dc_a[0].offset) == \
               (tc_b[0].offset, dc_b[0].offset), \
            f"{name}: cache offsets {(tc_a[0].offset, dc_a[0].offset)} != " \
            f"{(tc_b[0].offset, dc_b[0].offset)}"
        assert st["drafted"] == st["steps"], st   # gate never fired at tau=0
        print(f"  tau=0 [{name}] == mlx_lm num_draft_tokens={K}: {N} tokens, "
              f"{st['accepted']}/{st['drafted']} drafted rows accepted over "
              f"{st['rounds']} rounds, cache offsets target {tc_a[0].offset} "
              f"draft {dc_a[0].offset}  OK")

    # 2. tau above any reachable gap: k=0 every round and the stream is plain greedy.
    tc_c, dc_c, flat_c = caches(exact)
    never = _drain(adaptive_generate_step(
        prompt, target, exact, draft_tau=2.0, num_draft_tokens=K,
        max_tokens=N, prompt_cache=flat_c))
    tc_d, _, flat_d = caches(exact)
    ref1 = _drain(speculative_generate_step(
        prompt, target, exact, num_draft_tokens=1, max_tokens=N,
        prompt_cache=flat_d))
    gc = make_prompt_cache(target)
    greedy = []
    for t, _ in generate_step(prompt, target, max_tokens=N, prompt_cache=gc):
        greedy.append(int(t))
        if len(greedy) == N:
            break
    assert [t for t, _ in never] == [t for t, _ in ref1], \
        f"tau=inf != mlx_lm k=1\n{never}\n{ref1}"
    assert [t for t, _ in never] == greedy, "tau=inf != greedy"
    assert not any(fd for _, fd in never), "tau=inf still proposed draft rows"
    print(f"  tau=2.0 == mlx_lm num_draft_tokens=1 == greedy: {N} tokens, "
          f"0 from draft  OK")

    # 3. a round that ends on a target token leaves it out of the target cache, the
    # convention moe_stream heals.
    assert tc_c[0].offset == len(prompt) + N - 1, tc_c[0].offset
    assert gc[0].offset == len(prompt) + N, gc[0].offset
    # Never AHEAD of the emission: rows nobody emitted are what corrupts the
    # next turn's prefill, and a mid-chain stop is the case that risks it.
    assert tc_d[0].offset <= len(prompt) + N, tc_d[0].offset
    # The draft cache may lag arbitrarily (a k=0 round never feeds it; the
    # backlog rides in draft_y and goes in as one forward when the gate next
    # opens), but must not run ahead either.
    assert dc_c[0].offset <= len(prompt) + N, dc_c[0].offset
    print(f"  cache convention: a round ending on a target token leaves the "
          f"target at {tc_c[0].offset} = len(prompt)+{N}-1, one BEHIND the "
          f"emission; generate_step leaves it at {gc[0].offset}. mlx_lm's own "
          f"skew, matched not fixed  OK")

    # 4. a middling tau varies k without changing the reply, and STATS sees the tokens
    # the gate threw away.
    before = dict(STATS)
    tc_e, dc_e, flat_e = caches(exact)
    mid = _drain(adaptive_generate_step(
        prompt, target, exact, draft_tau=0.05, num_draft_tokens=K,
        max_tokens=N, prompt_cache=flat_e))
    st = {k: STATS[k] - before[k] for k in STATS}
    assert [t for t, _ in mid] == greedy, "tau=0.05 changed the emitted tokens"
    assert st["rounds"] == sum(1 for _, fd in mid if not fd), st
    assert 0 < st["drafted"] < st["steps"], st
    print(f"  tau=0.05: same {N} tokens, {st['rounds']} rounds, "
          f"{st['drafted']} rows drafted in {st['steps']} draft steps "
          f"({st['steps'] - st['drafted']} thrown away by the gate), "
          f"{st['accepted']} accepted  OK")

    # --- 5. first_early: generate_step's tokens, logprobs and caches, its
    #        first token out before the forward that feeds it (the cache one
    #        behind at that yield, level at every later one), and a token the
    #        consumer stops at fed before it comes out, as generate_step does.
    def run(gen, stop_after=None):
        out = []
        for t, lp in gen:
            out.append((int(t), lp, gc_[0].offset))
            if len(out) == stop_after:
                break
        return out
    # Reads the history (its length and last id), so a processor that saw
    # only the tokens after the first would change the logprobs.
    procs = [lambda toks, lg: lg - 5.0 * (mx.arange(V) == toks.size % V)
             - 1e9 * (mx.arange(V) == toks[-1])]
    for kw in ({}, {"logits_processors": procs}):
        gc_ = make_prompt_cache(target)
        ref = run(generate_step(prompt, target, max_tokens=N, prompt_cache=gc_, **kw))
        ref_c = [mx.array(a) for a in gc_[0].state]
        gc_ = make_prompt_cache(target)
        new = run(first_early(prompt, target, max_tokens=N, prompt_cache=gc_, **kw))
        assert [t for t, _, _ in new] == [t for t, _, _ in ref], (new, ref)
        assert all(mx.array_equal(a[1], b[1]).item() for a, b in zip(new, ref))
        assert [o for _, _, o in new] == [len(prompt)] + [o for _, _, o in ref[1:]]
        assert ref[0][2] == len(prompt) + 1
        assert all(mx.array_equal(a, b).item() for a, b in zip(gc_[0].state, ref_c))
        y0 = ref[0][0]
        for case in ({"max_tokens": 1}, {"max_tokens": N, "stop": {y0}}):
            gc_ = make_prompt_cache(target)
            got = run(first_early(prompt, target, prompt_cache=gc_, **case, **kw),
                      stop_after=1)
            assert [t for t, _, _ in got] == [y0], got
            assert got[0][2] == len(prompt) + 1, (case, got[0][2])
    print(f"  first_early == generate_step: {N} tokens, logprobs and caches equal; "
          f"the first out before its forward; a stop token fed first  OK")

    print("selftest OK")


def engine_selftest():
    """The moe_stream call site end to end on toy weights and the real tokenizer, checked
    against a no-draft baseline as well."""
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.utils import load_tokenizer

    import os

    import moe_stream   # deferred: moe_stream imports this module at load

    toy_checkpoint = os.environ.get("MOE_TOY_TOKENIZER") or moe_stream.DEFAULT_MODEL
    if not toy_checkpoint:
        print("engine_selftest skipped: set MOE_TOY_TOKENIZER to a checkpoint directory")
        return
    tok = load_tokenizer(Path(toy_checkpoint))
    vocab = max([tok.vocab_size] + list(tok.get_added_vocab().values())) + 1
    target, draft, _ = _drafts(vocab, 32, 3, 2)

    def build(stub, with_draft, tau):
        llm = moe_stream.StreamingLLM.__new__(moe_stream.StreamingLLM)
        llm.model, llm.tokenizer = target, tok
        llm.max_tokens, llm.prefill_step, llm.resident = 16, None, 0
        llm.thinking, llm.profile = False, None
        llm.model_path = toy_checkpoint
        llm._setup_chat(stub)
        llm._draft_model = draft if with_draft else None
        llm._draft_seen = []
        llm._draft_cache = make_prompt_cache(draft) if with_draft else None
        llm.draft_tokens, llm.draft_tau = 4, tau
        llm.reset()
        return llm

    qs = ["Say the word apple.", "Now say the word banana."]
    for stub in ("slice", "keep"):
        ref = [build(stub, False, None).ask(q) for q in qs]
        for tau in (None, 0.0, 2.0):
            reps = [build(stub, True, tau).ask(q) for q in qs]
            assert reps == ref, (
                f"stub={stub!r} draft_tau={tau}: turn-2 reply differs from "
                f"the no-draft baseline\n  baseline: {ref}\n  draft:    {reps}")
        print(f"  stub={stub!r}: no-draft baseline == draft_tau "
              f"None/0.0/2.0, both turns  OK")

    llm = moe_stream.StreamingLLM.__new__(moe_stream.StreamingLLM)
    llm.model, llm.tokenizer = target, tok
    llm.max_tokens, llm.prefill_step, llm.resident = 16, None, 0
    llm.thinking, llm.profile = False, None
    llm.model_path = toy_checkpoint
    llm._setup_chat("keep")
    llm._draft_model, llm._draft_seen = draft, []
    llm._draft_cache = make_prompt_cache(draft)
    llm.draft_tokens = 4

    out = {}
    for tau in (None, 0.0, 2.0):
        llm.draft_tau = tau
        llm.reset()
        reps, sts = [], []
        for q in qs:
            reps.append(llm.ask(q))
            sts.append(dict(llm.stats()))
        out[tau] = (reps, sts)
        print(f"  draft_tau={tau}: accept "
              f"{[s.get('draft_acceptance') for s in sts]}  mean k "
              f"{[s.get('draft_mean_k') for s in sts]}  rounds "
              f"{[s.get('draft_rounds') for s in sts]}  wasted steps "
              f"{[s.get('draft_steps_wasted') for s in sts]}  cached "
              f"{[s['cached_tokens'] for s in sts]}")

    assert out[None][0] == out[0.0][0] == out[2.0][0], \
        f"replies differ across tau:\n{out[None][0]}\n{out[0.0][0]}\n{out[2.0][0]}"
    # tau=0 must reproduce the fixed chain exactly, acceptance included.
    assert [s["draft_acceptance"] for s in out[None][1]] == \
           [s["draft_acceptance"] for s in out[0.0][1]], "tau=0 != fixed k"
    # mean k lands just under the cap, not on it: the last round of a turn is
    # clipped by max_tokens the same way mlx_lm clips num_draft_tokens.
    assert 3.0 <= out[0.0][1][0]["draft_mean_k"] <= 4.0, out[0.0][1]
    assert out[2.0][1][0]["draft_mean_k"] == 0.0, out[2.0][1]
    # Every k=0 round still pays one draft forward to read the margin.
    assert out[2.0][1][0]["draft_steps_wasted"] == out[2.0][1][0]["tokens"], \
        out[2.0][1]
    # The adaptive keys appear only when the gate is on; stats() with
    # draft_tau=None is the dict it has always been.
    assert "draft_tau" not in out[None][1][0]
    assert all(s["cached_tokens"] > 0 for _, sts in out.values()
               for s in sts[1:]), "turn 2 did not reuse the KV cache"
    print("engine_selftest OK")


if __name__ == "__main__":
    import sys
    if "--engine-selftest" in sys.argv:
        engine_selftest()
    elif "--selftest" in sys.argv:
        selftest()
    else:
        print(__doc__)
