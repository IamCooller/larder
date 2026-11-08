#!/usr/bin/env python3
"""Toy fixtures shared by the tests: a KB-scale engine and a scripted drafter.

`toy_llm` wraps `toy_model` in a real StreamingLLM, built from the
same mlx_lm classes the big model uses, so the caches, the RoPE call and the
GatedDeltaNet blocks under test are the real ones and no weights are opened.
Only a tokenizer is read, from $MOE_TOY_TOKENIZER when set and from
moe_stream.DEFAULT_MODEL otherwise; `has_tokenizer` lets a test skip instead.

`ScriptedSpeculator` is spec.Speculator with a script in place of a head, so
the accept, reject and replay branches run on demand.
"""

import os
import sys
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.cache import KVCache, make_prompt_cache

import moe_stream
import spec

TOKENIZER_ENV = "MOE_TOY_TOKENIZER"

# A checkpoint path used only as a name for the stored fronts; nothing is read from it.
MODEL_LABEL = "/models/demo-model"


def tokenizer_dir():
    """The directory that holds the toy tests' tokenizer.json, or None when no checkpoint is named."""
    path = os.environ.get(TOKENIZER_ENV) or moe_stream.DEFAULT_MODEL
    return Path(path) if path else None


def model_dir(test):
    """The checkpoint a model test runs on: its first argument or $MOE_TEST_MODEL, or None after one line saying what is missing."""
    path = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("MOE_TEST_MODEL", "")).expanduser()
    if (path / "config.json").exists():
        return str(path)
    print(f"{test}: skipped (needs a checkpoint directory: pass it as the first argument or set MOE_TEST_MODEL)")
    return None


def has_tokenizer():
    """Whether tokenizer_dir() has the tokenizer the toy engine reads."""
    path = tokenizer_dir()
    return path is not None and (path / "tokenizer.json").exists()


def max_abs(a, b):
    """The largest absolute difference between two arrays, as a float."""
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())


def toy_model(seed=0, n_layers=8, vocab=128):
    """A KB-scale Qwen3_5 from the real classes: GatedDeltaNet and attention blocks, a dense MLP, no checkpoint.

    The head dimension is 32 because the GatedDeltaNet Metal kernel declares float state[Dk/32] per thread and a smaller one does not compile.
    """
    from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs
    mx.random.seed(seed)
    args = TextModelArgs(
        model_type="qwen3_5", hidden_size=32, intermediate_size=64,
        num_hidden_layers=n_layers, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=vocab, head_dim=32,
        linear_num_value_heads=4, linear_num_key_heads=2,
        linear_key_head_dim=32, linear_value_head_dim=32,
        linear_conv_kernel_dim=4, full_attention_interval=4,
        tie_word_embeddings=False, num_experts=0)
    m = TextModel(args)
    mx.eval(m.parameters())
    m.eval()
    return m


def toy_llm(stub, max_tokens=6, seed=0):
    """An engine around a KB-scale model and the real Qwen tokenizer.

    The tokenizer decides where the template's think stub lands in the token
    stream, which is the whole subject of the stub-mode tests; the weights are
    random and tiny.
    """
    from mlx_lm.utils import load_tokenizer
    path = tokenizer_dir()
    if path is None:
        raise OSError(f"no tokenizer to load: set {TOKENIZER_ENV} to a checkpoint directory")
    tok = load_tokenizer(path)
    vocab = max([tok.vocab_size] + list(tok.get_added_vocab().values())) + 1
    llm = moe_stream.StreamingLLM.__new__(moe_stream.StreamingLLM)
    llm.model, llm.tokenizer = toy_model(seed=seed, vocab=vocab), tok
    llm.max_tokens, llm.prefill_step, llm.resident = max_tokens, None, 0
    llm.thinking, llm.profile, llm.model_path = False, None, str(path)
    llm._setup_chat(stub)
    return llm


def find(seq, sub):
    """Every index at which `sub` occurs in `seq`."""
    if not sub:
        return []
    return [i for i in range(len(seq) - len(sub) + 1)
            if seq[i:i + len(sub)] == sub]


def kv(cache):
    """(keys, values) of every attention cache, trimmed to what is STORED.

    `offset - gap`, not `offset`: a sliced cache keeps RoPE's inflated number
    and stores fewer positions than that, which is the whole trick.
    """
    return [(c.keys[..., :c.offset - getattr(c, "gap", 0), :],
             c.values[..., :c.offset - getattr(c, "gap", 0), :])
            for c in cache if isinstance(c, KVCache)]


def gdn(cache):
    """The GatedDeltaNet state arrays of a hybrid cache."""
    return [a for c in cache if not isinstance(c, KVCache)
            for a in c.cache if a is not None]


def greedy(model, ids, max_tokens, eos=()):
    """Plain greedy decoding, the reference the speculator must reproduce."""
    lm = getattr(model, "language_model", model)
    cache = make_prompt_cache(model)
    h = lm.model(mx.array([ids]), cache=cache)[:, -1:]
    for _ in range(max_tokens):
        tok = int(mx.argmax(lm.lm_head(h)[0, -1]))
        yield tok
        if tok in eos:
            return
        h = lm.model(mx.array([[tok]]), cache=cache)


class ScriptedSpeculator(spec.Speculator):
    """A speculator whose draft is a script, not a head.

    `ref` is the token sequence plain greedy produces, so "accept" means
    proposing what the target is about to produce anyway and "reject" means
    proposing anything else; either way the emitted sequence must come out the
    same. A script reaches the replay branch, which a real head's acceptance
    rate makes a one-in-ten-thousand event.
    """

    def __init__(self, model, ref, pattern, vocab, gamma=1):
        super().__init__(model, head=None, gamma=gamma)
        self.ref, self.pattern, self.vocab, self.i = ref, pattern, vocab, 0

    def _draft(self, pairs, hcache, st):
        # Draft j of the round is checked against the token at index
        # st.tokens + j: st.tokens is how many reference tokens have been
        # emitted so far. One pattern entry is consumed per DRAFT, not per
        # round, so at gamma=1 the script means exactly what it always did and
        # at gamma>1 a round's accepted prefix can be set to any length.
        out = []
        for j in range(self.gamma):
            n = st.tokens + j
            want = self.ref[n] if n < len(self.ref) else 0
            hit = self.pattern[self.i % len(self.pattern)]
            self.i += 1
            out.append(want if hit else (want + 1) % self.vocab)
        return out


def spec_run(model, ref, pattern, ids, n, eos=(), vocab=128, gamma=1,
             rebuild=True):
    """One scripted generation over a fresh cache; returns tokens, stats, cache."""
    sp = ScriptedSpeculator(model, ref, pattern, vocab, gamma=gamma)
    sp.rebuild = rebuild
    core, widths = sp.core, []

    def watched(x, cache=None):
        widths.append(x.shape[-1])
        return core(x, cache=cache)
    sp.core = watched
    cache = make_prompt_cache(model)
    st = spec.Stats()
    got = list(sp.generate(ids, n, eos, st, cache))
    # The kernel cliff, asserted rather than reasoned about: gather_qmm is
    # linear in T and quantized_matmul is flat only to T=4, so a verification
    # pass wider than max_window+1 costs more than the tokens it buys. widths[0]
    # is the prefill, which is as wide as the prompt and not a decode pass.
    assert max(widths[1:], default=0) <= spec.Speculator.max_window + 1, widths
    # With the rebuild a rejection never widens the next pass: 1 + gamma rows.
    assert not rebuild or (max(widths[1:], default=0) <= 1 + gamma
                           and not st.replays), widths
    return got, st, cache


# How far a cache built in chunks may sit from one built in a single pass,
# with no speculation involved at all. Measured on this toy: feeding 24 tokens
# as 8+16 or 8+4+4+4+4 agrees to 7e-10, but ANY chunking that ends with a
# single trailing token -- 23+1, the shape every decode step has -- lands at
# 4.7e-3, because GatedDeltaNet takes a different kernel at S=1. So the floor
# is a property of the model, not of the rollback. The failure this is meant
# to catch is a duplicated or dropped position, and that measured 2.0 before
# _restore stopped aliasing its snapshot: two orders of magnitude clear.
CACHE_TOL = 2e-2


def check_cache(model, cache, ids, why):
    """The cache must BE a prefill of `ids`, not merely the right length.

    The length alone would pass a cache that fed a token twice and dropped
    another, and the replay branch is exactly where that can happen.
    """
    for c in cache:
        if isinstance(c, KVCache):
            stored = c.offset - getattr(c, "gap", 0)
            assert stored == len(ids), (
                f"{why}: cache holds {stored} positions against {len(ids)} tokens")
    fresh = make_prompt_cache(model)
    mx.eval(model(mx.array([ids]), cache=fresh), [c.state for c in fresh])
    worst = 0.0
    for a, b in zip(cache, fresh):
        if isinstance(a, KVCache):
            n = a.offset - getattr(a, "gap", 0)
            worst = max(worst, max_abs(a.keys[..., :n, :], b.keys[..., :n, :]),
                        max_abs(a.values[..., :n, :], b.values[..., :n, :]))
        else:
            for x, y in zip(a.cache, b.cache):
                if x is not None:
                    worst = max(worst, max_abs(x, y))
    assert worst < CACHE_TOL, (f"{why}: the cache is not a prefill of the tokens "
                               f"it claims to hold, off by {worst}")
    return worst


def tie_gap(model, ids):
    """Top-2 logit gap for the token that follows `ids`, from a fresh prefill."""
    out = model(mx.array([ids]), cache=make_prompt_cache(model))[0, -1]
    s = mx.sort(out)
    return float(s[-1] - s[-2])