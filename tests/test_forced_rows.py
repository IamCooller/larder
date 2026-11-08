#!/usr/bin/env python3
"""Grammar-forced verify rows (forced_rows.py, spec_draft's `forcer=`), toy.

    python3 tests/test_forced_rows.py

1. proposers -- a fake one that is always right, always wrong, or random --
   through adaptive_generate_step, gated and ungated, on an exact and a noisy
   draft, with spans both opening the chain and after drafted tokens: the
   stream is plain greedy token for token, the proposer is fed exactly the
   emitted tokens, and both caches hold what a fresh prefill of the emitted
   sequence gives (the draft caught up over every forced span).
2. forcer=None, draft_tau=None is mlx_lm's fixed chain, flags and cache
   offsets included -- the path tool requests now take.
3. a target marked with rowqmm_rows never sees a wider verify pass.
   only mlx_lm's json_tools call format gets a forcer.
4. the real grammar where the Qwen3 tokenizer exists, on the built-in
   grammars and on xgrammar where it is installed: over a canonical
   tool call every proposal is a prefix of what follows, a fork walks on
   without touching its parent, and a reply that leaves the schema switches
   the forcer off; the same for a response_format schema over a whole JSON
   reply, which ends by proposing the end of turn.
5. response_format: which ones get a grammar; through the engine it reaches
   the forcer and the replies stay the no-draft ones, turn two included;
   and the real reply-mode Forcer (both engines, on a toy vocabulary)
   through spec_draft is plain greedy with both caches synced, whether its
   grammar spells the reply or leaves it halfway.
6. the spec.py verify pass (the 35B's: GatedDeltaNet state rolled back by
   snapshot), on the hybrid toy with a dense and with an MoE MLP: forced
   rows with no drafter (ForcedSpeculator) and after or instead of a
   scripted head's drafts. The cache is a prefill of exactly the emitted
   tokens, no pass is wider than max_window + 1, the forcer is fed exactly
   the emitted tokens, and the reply is the no-forcer one up to a near-tie
   (a multi-row pass reduces in another order; tests/test_spec_rounds.py's rule).
   Through the engine the same, turn two included, on either arm of a
   pinned ForcedPolicy (the plain arm exactly the plain engine); the default
   policy explores both, and on scripted turn timings it settles on the
   faster arm, re-probes the slower and follows a swap; verify_mass keeps
   the forced-only pass off. ForcedPolicy generalised past two arms: the
   same warm-up/converge/re-probe on three synthetic arms, and an
   MTP-loaded engine's own two policies (a turn with a forcer choosing among
   plain/forced_only/mtp, one without among plain/mtp) kept separate.
7. the real qwen3_coder XML grammar (the 35B's tokenizer, both engines):
   every proposal over a canonical call with a multi-line value is a prefix
   of what follows, parameters out of declared order switch it off, only the
   built-in grammars also propose a string enum's value, a number's line end
   and free text behind both (a value off the list does not switch them off),
   recorded calls over a long tool list and a json_object reply are covered,
   and with torch and xgrammar blocked a tool request still gets its forcer
   and torch stays out.
8. the lean-host torch shadow (server.py/chat.py): _xgrammar() leaves it be
   when xgrammar is genuinely absent (forced so even on a machine where it
   is really installed), and lifts it -- letting a stub package shaped like
   xgrammar's own `import torch` succeed -- when it is there; and a stub
   whose chain imports torch fine and then hits a different missing
   dependency doesn't re-shadow the now-real torch, which used to turn the
   next lift's re-registration into a RuntimeError instead of a plain
   "xgrammar not usable". Subprocesses, since they mutate sys.modules in
   ways no other test here should see.
"""

import contextlib
import copy
import random
import sys
import types
from pathlib import Path

import mlx.core as mx
from mlx_lm.generate import generate_step, speculative_generate_step
from mlx_lm.models.cache import KVCache, make_prompt_cache

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import forced_rows                                             # noqa: E402
from memory_core import kv_arrays                              # noqa: E402
import support                                                 # noqa: E402
from support import (ScriptedSpeculator, check_cache, greedy, tie_gap,
                     toy_llm)                                  # noqa: E402
import spec_draft                                              # noqa: E402

V, N = 200, 40
PROMPT = mx.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


class Fake:
    """Proposes 0-7 tokens: the greedy reference's next ones ("right"), each
    shifted by one ("wrong"), or noise ("random"). Forks walk the round's own
    chain, drafted tokens included, and count the spans they propose."""

    active = True

    def __init__(self, ref, mode, seed):
        self.ref, self.mode, self.seen = ref, mode, 0
        self.rng = random.Random(seed)
        self.forked, self.spans = False, [0]

    def fork(self):
        f = copy.copy(self)
        f.forked = True
        return f

    def feed(self, tok):
        if not self.forked:
            assert tok == self.ref[self.seen], (self.seen, tok)
        self.seen += 1

    def propose(self):
        p = self._propose()
        if self.forked and p:
            self.spans[0] += 1
        return p

    def _propose(self):
        r = self.rng.randint(0, 7)
        nxt = self.ref[self.seen:self.seen + r]
        if self.mode == "right":
            return nxt
        if self.mode == "wrong":
            return [(t + 1) % V for t in nxt]
        return [self.rng.randrange(V) for _ in range(r)]


def _toys():
    """spec_draft's toy target and drafts, residual branches scaled up so the
    next token depends on the context: plain random toys repeat their last
    token forever, and a draft that skipped or doubled a row would never show
    in a cache that holds one token."""
    ms = spec_draft._drafts(V, 32, 3, 2)
    for m in ms:
        for layer in m.model.layers:
            layer.self_attn.o_proj.weight = 4 * layer.self_attn.o_proj.weight
            layer.mlp.down_proj.weight = 4 * layer.mlp.down_proj.weight
        mx.eval(m.parameters())
    return ms


def _greedy(target, n):
    out = []
    for t, _ in generate_step(PROMPT, target, max_tokens=n,
                              prompt_cache=make_prompt_cache(target)):
        out.append(int(t))
    return out


def _synced(model, c, seq):
    """The cache holds a fresh prefill of seq[:offset]. A batched and a
    row-by-row forward differ by ~1e-3 here; a skipped or wrong row by ~1."""
    off = c[0].offset
    fresh = make_prompt_cache(model)
    model(mx.array(seq[:off])[None], cache=fresh)
    for a, b in zip(c, fresh):
        pairs = zip(kv_arrays(a), kv_arrays(b)) if isinstance(a, KVCache) else zip(a.state, b.state)
        for x, y in pairs:
            if not mx.allclose(x, y, atol=2e-2).item():
                return False
    return True


def test_proposers():
    target, exact, noisy = _toys()
    ref = _greedy(target, N)
    seq = PROMPT.tolist() + ref
    for mode in ("right", "wrong", "random"):
        for dname, d in (("exact", exact), ("noisy", noisy)):
            for tau in (None, 0.05):
                for n in (N, 7):
                    tc, dc = make_prompt_cache(target), make_prompt_cache(d)
                    fake = Fake(ref, mode, seed=n)
                    before = dict(spec_draft.STATS)
                    out = spec_draft._drain(spec_draft.adaptive_generate_step(
                        PROMPT, target, d, draft_tau=tau, num_draft_tokens=2,
                        max_tokens=n, prompt_cache=tc + dc, forcer=fake))
                    st = {k: spec_draft.STATS[k] - before[k]
                          for k in spec_draft.STATS}
                    label = f"{mode}/{dname}/tau={tau}/n={n}"
                    assert [t for t, _ in out] == ref[:n], label
                    assert fake.seen == n, label
                    assert tc[0].offset <= len(PROMPT) + n, label
                    assert dc[0].offset <= len(PROMPT) + n, label
                    assert _synced(target, tc, seq), f"{label}: target cache"
                    assert _synced(d, dc, seq), f"{label}: draft cache"
                    if mode == "wrong":
                        assert st["forced_accepted"] == 0, (label, st)
                    if mode == "right" and n == N:
                        assert st["forced_accepted"] > 0, (label, st)
                        if tau is None:     # spans after drafted tokens too
                            assert fake.spans[0] > 0, label
                        assert st["forced_accepted"] <= st["forced"], (label, st)
                    print(f"  {label:26s} == greedy, forced "
                          f"{st['forced_accepted']}/{st['forced']}, draft "
                          f"{st['accepted']}/{st['drafted']}, "
                          f"{st['rounds']} rounds, {fake.spans[0]} mid-chain "
                          f"spans, caches synced  OK")


def test_fixed_chain():
    target, exact, noisy = _toys()
    for dname, d in (("exact", exact), ("noisy", noisy)):
        ta, da = make_prompt_cache(target), make_prompt_cache(d)
        mine = spec_draft._drain(spec_draft.adaptive_generate_step(
            PROMPT, target, d, draft_tau=None, num_draft_tokens=2,
            max_tokens=N, prompt_cache=ta + da))
        tb, db = make_prompt_cache(target), make_prompt_cache(d)
        theirs = spec_draft._drain(speculative_generate_step(
            PROMPT, target, d, num_draft_tokens=2, max_tokens=N,
            prompt_cache=tb + db))
        assert mine == theirs, dname
        assert (ta[0].offset, da[0].offset) == (tb[0].offset, db[0].offset)
        print(f"  draft_tau=None [{dname}] == mlx_lm k=2, flags and cache "
              f"offsets  OK")


class Rows:
    """A target marked the way rowqmm.install marks one, logging the rows of
    every forward (after the one-token prompt, every forward is a verify)."""

    rowqmm_rows = 4

    def __init__(self, m):
        self.m, self.layers, self.rows = m, m.layers, []

    def __call__(self, x, cache=None):
        self.rows.append(x.shape[1])
        return self.m(x, cache=cache)


def test_row_cap():
    target, exact, _ = _toys()
    ref = _greedy(target, N)
    seq = PROMPT.tolist() + ref
    for tau in (None, 0.05):
        marked = Rows(target)
        tc, dc = make_prompt_cache(target), make_prompt_cache(exact)
        out = spec_draft._drain(spec_draft.adaptive_generate_step(
            PROMPT, marked, exact, draft_tau=tau, num_draft_tokens=2,
            max_tokens=N, prompt_cache=tc + dc, forcer=Fake(ref, "right", 1)))
        assert [t for t, _ in out] == ref, tau
        assert max(marked.rows[1:]) == Rows.rowqmm_rows, marked.rows
        assert _synced(target, tc, seq) and _synced(exact, dc, seq), tau
        print(f"  rowqmm_rows={Rows.rowqmm_rows} [tau={tau}]: {len(marked.rows) - 1} "
              f"passes, none over {max(marked.rows[1:])} rows, == greedy  OK")


def test_layout_cache():
    """rowqmm.install calibrates once per machine and model, then reuses the
    saved layout, and marks the model only when qmm8 got a shape."""
    import tempfile

    import mlx.nn as nn
    import rowqmm

    calls = []
    real = rowqmm.calibrate
    rowqmm.LAYOUT_DIR = Path(tempfile.mkdtemp())
    rowqmm.calibrate = lambda m, **k: calls.append(1) or {(8, 64, 128): rowqmm.MM8_CANDS[0]}
    try:
        for _ in range(2):
            m = spec_draft._toy(64, 64, 1, inter=128)
            nn.quantize(m, group_size=64, bits=4)
            layout = rowqmm.install(m)
            assert layout == {(8, 64, 128): rowqmm.MM8_CANDS[0]}, layout
            assert m.rowqmm_rows == rowqmm.ROWS
        assert len(calls) == 1, "calibrated twice"
        rowqmm.calibrate = lambda m, **k: {(5, 64, 128): 2}
        rowqmm.LAYOUT_DIR = Path(tempfile.mkdtemp())
        m = spec_draft._toy(64, 64, 1, inter=128)
        nn.quantize(m, group_size=64, bits=4)
        rowqmm.install(m)
        assert not hasattr(m, "rowqmm_rows"), "marked without qmm8"
    finally:
        rowqmm.calibrate = real
        rowqmm.uninstall(None)
    print("  rowqmm layout: calibrated once, reused from disk; mark only "
          "with qmm8  OK")


def test_gate():
    """Only the call formats with a grammar get a forcer: json_tools and
    qwen3_coder. Any other parser, or none, gets nothing."""
    import types

    def parse_tool_call(text, tools=None):
        pass
    parse_tool_call.__module__ = "mlx_lm.tool_parsers.glm47"
    tok = types.SimpleNamespace(tool_parser=parse_tool_call,
                                tool_call_start_tokens=(1,),
                                tool_call_end_tokens=(2,))
    assert forced_rows.make(tok, [{"name": "f"}]) is None
    assert forced_rows.make(types.SimpleNamespace(), [{"name": "f"}]) is None
    print("  gate: no forcer for a tool format without a grammar  OK")


def test_response_format_gate():
    """Which response_format gets a grammar: a json_schema one its schema
    (any object without one), json_object any object, the rest none. Tools
    come first when a request carries both."""
    import types
    schema = {"type": "object", "properties": {"city": {"type": "string"}}}
    js = {"type": "json_schema", "json_schema": {"name": "w", "schema": schema}}
    assert forced_rows._schema(js) == schema
    assert forced_rows._schema({"type": "json_schema",
                                "json_schema": {"name": "w"}}) == {"type": "object"}
    assert forced_rows._schema({"type": "json_object"}) == {"type": "object"}
    for fmt in (None, {"type": "text"}, "json_object", {}):
        assert forced_rows._schema(fmt) is None, fmt
        assert forced_rows.make(types.SimpleNamespace(eos_token_id=2), None,
                                fmt) is None, fmt
    xml_tools = types.SimpleNamespace(tool_parser=None, eos_token_id=2)
    assert forced_rows.make(xml_tools, [{"name": "f"}], js) is None
    print("  response_format: json_schema/json_object get a grammar, text "
          "and junk none, tools first  OK")


def test_engine_response_format():
    """server.py hands response_format to StreamingLLM.stream, which hands it
    to forced_rows.make. A toy target and draft around the checkpoint's
    tokenizer (spec_draft.engine_selftest's pair), a Fake forcer proposing
    the no-draft reply: both turns come back as the no-draft ones, so the
    forced rows left the engine's caches where a plain turn would."""
    import moe_stream
    from mlx_lm.utils import load_tokenizer

    path = support.tokenizer_dir()
    if not support.has_tokenizer():
        print("  engine response_format: SKIPPED (no tokenizer: set MOE_TOY_TOKENIZER)")
        return
    tok = load_tokenizer(path)
    vocab = max([tok.vocab_size] + list(tok.get_added_vocab().values())) + 1
    target, draft, _ = spec_draft._drafts(vocab, 32, 3, 2)

    def build(with_draft):
        llm = moe_stream.StreamingLLM.__new__(moe_stream.StreamingLLM)
        llm.model, llm.tokenizer = target, tok
        llm.max_tokens, llm.prefill_step, llm.resident = 16, None, 0
        llm.thinking, llm.profile = False, None
        llm.model_path = support.MODEL_LABEL
        llm._setup_chat("keep")
        llm._draft_model = draft if with_draft else None
        llm._draft_seen = []
        llm._draft_cache = make_prompt_cache(draft) if with_draft else None
        llm.draft_tokens, llm.draft_tau = 4, None
        llm.reset()
        return llm

    ask = [{"role": "user", "content": "The weather in Paris, JSON."}]
    base = build(False)
    ref = [base.ask(messages=ask)]
    ids = base._seen[len(base._seen) - base.stats()["tokens"]:]
    ref.append(base.ask("And tomorrow?"))

    fmt = {"type": "json_object"}
    given, real = [], moe_stream.forced_rows.make
    moe_stream.forced_rows.make = lambda tk, tools, rf=None: (
        given.append(rf) or (Fake(ids, "right", 0) if rf else None))
    try:
        llm = build(True)
        got = [llm.ask(messages=ask, response_format=fmt)]
        st = llm.stats()
        got.append(llm.ask("And tomorrow?"))
    finally:
        moe_stream.forced_rows.make = real
    assert given == [fmt, None], given
    assert got == ref, (got, ref)
    assert st["forced_accepted"] > 0, st
    print(f"  engine response_format: reaches forced_rows.make, forced "
          f"{st['forced_accepted']}/{st['forced_rows']}, both turns == "
          f"no-draft  OK")


def _engines():
    """Which grammar engines to run: the built-in ones always, xgrammar
    where it is installed."""
    try:
        import xgrammar  # noqa: F401
        return (True, False)
    except ImportError:
        return (True,)


@contextlib.contextmanager
def _built_in(on):
    """make() on the built-in grammars even where xgrammar is installed."""
    real = forced_rows._xgrammar
    if on:
        forced_rows._xgrammar = lambda: False
    try:
        yield "built-in" if on else "xgrammar"
    finally:
        forced_rows._xgrammar = real


def _walk(f, ids):
    """A forced-only pass over a known reply: every proposal is a prefix of
    what follows, the proposed rows and the target's next token are fed.
    The number of proposed tokens."""
    i, forced = 0, 0
    while i < len(ids):
        p = f.propose()
        assert p == ids[i:i + len(p)], (i, p, ids[i:i + len(p)])
        for t in ids[i:i + len(p) + 1]:
            f.feed(t)
        forced += len(p)
        i += len(p) + 1
    return forced


def _tokenizer(*dirs):
    """The first of these checkpoint dirs with a tokenizer (never weights)."""
    from mlx_lm.utils import load_tokenizer
    d = next((d for d in dirs if (d / "tokenizer.json").exists()), None)
    return d and load_tokenizer(d)


QWEN3 = [Path.home() / ".cache" / "qwen3-4b", *sorted(
    (Path.home() / ".cache" / "huggingface" / "hub"
     / "models--mlx-community--Qwen3-4B-4bit" / "snapshots").glob("*"))]
QWEN36 = [p for p in [support.tokenizer_dir()] if p]


def test_real_reply_grammar():
    """The real Forcer in reply mode (no open token) through spec_draft. The
    vocabulary is one distinct character per toy id, so a grammar can spell
    the greedy reply exactly: every row it proposes is right. A second
    grammar leaves the reply at token 12: the target's token there is
    rejected, the forcer goes quiet for good, and the output never moves.
    On xgrammar where it is installed and on the built-in grammars."""
    target, exact, noisy = _toys()
    ref = _greedy(target, N)
    seq = PROMPT.tolist() + ref
    char = lambda t: chr(0x4E00 + t)                       # noqa: E731
    encode = lambda s, add_special_tokens=False: [        # noqa: E731
        ord(c) - 0x4E00 for c in s]
    end = next(t for t in range(V) if t not in ref)       # the end of turn
    leaves = ref[:12] + [(ref[12] + 1) % V] + ref[13:]
    for lite in _engines():
        if lite:
            vocab = types.SimpleNamespace(convert_ids_to_tokens=char)
            compile_ = lambda text: forced_rows._Lite((text.encode(),), vocab)  # noqa: E731
        else:
            import xgrammar as xgr
            compiler = xgr.GrammarCompiler(xgr.TokenizerInfo(
                [char(t) for t in range(V)], stop_token_ids=[end]))
            compile_ = lambda text: compiler.compile_grammar(f'root ::= "{text}"')  # noqa: E731
        _reply_grammar_run(compile_, "built-in" if lite else "xgrammar", ref,
                           seq, leaves, char, encode, end, target, exact, noisy)


def _reply_grammar_run(compile_, engine, ref, seq, leaves, char, encode, end,
                       target, exact, noisy):
    for name, spelled in (("spells it", ref), ("leaves at 12", leaves)):
        grammar = compile_("".join(map(char, spelled)))
        for dname, d in (("exact", exact), ("noisy", noisy)):
            for tau in (None, 0.05):
                f = forced_rows.Forcer(grammar, encode, None, end)
                assert f.active
                tc, dc = make_prompt_cache(target), make_prompt_cache(d)
                before = dict(spec_draft.STATS)
                out = spec_draft._drain(spec_draft.adaptive_generate_step(
                    PROMPT, target, d, draft_tau=tau, num_draft_tokens=2,
                    max_tokens=N, prompt_cache=tc + dc, forcer=f))
                st = {k: spec_draft.STATS[k] - before[k] for k in spec_draft.STATS}
                label = f"{engine}/{name}/{dname}/tau={tau}"
                assert [t for t, _ in out] == ref, label
                assert _synced(target, tc, seq), f"{label}: target cache"
                assert _synced(d, dc, seq), f"{label}: draft cache"
                assert st["forced_accepted"] >= 12 - 1, (label, st)
                if spelled is ref:
                    assert st["forced_accepted"] == st["forced"], (label, st)
                else:
                    assert not f.active, label
                print(f"  real reply grammar {label:37s} == greedy, forced "
                      f"{st['forced_accepted']}/{st['forced']}, "
                      f"{st['rounds']} rounds, caches synced  OK")


def test_real_grammar():
    tok = _tokenizer(*QWEN3)
    if tok is None:
        print("  real grammar: SKIPPED (no Qwen3 tokenizer)")
        return
    from tool_fixture import TOOLS
    for lite in _engines():
        with _built_in(lite) as engine:
            _real_grammar(tok, TOOLS, engine)


def _real_grammar(tok, TOOLS, engine):
    call = ('<tool_call>\n{"name": "get_weather", "arguments": '
            '{"location": "Paris, France", "unit": "celsius"}}\n</tool_call>')
    ids = tok.encode(call, add_special_tokens=False)
    f = forced_rows.make(tok, TOOLS)
    assert f is not None
    forced = _walk(f, ids)
    assert f.propose() == [], "still forcing after </tool_call>"
    print(f"  real grammar [{engine}]: {forced}/{len(ids)} tokens of a canonical "
          f"call forced, every proposal a prefix of the reply  OK")

    # leaving the schema: the forcer goes quiet until the next <tool_call>
    f.feed(f.open_id)
    first = f.propose()
    assert first, "nothing forced right after <tool_call>"
    g = f.fork()                       # a fork walks on without touching f
    for t in ids[1:6]:
        g.feed(t)
    assert f.propose() == first and g.propose() != first
    for t in tok.encode("\nhello", add_special_tokens=False):
        f.feed(t)
    assert f.propose() == [], "still forcing after the model left the schema"
    print(f"  real grammar [{engine}]: a reply outside the schema switches it off  OK")

    # response_format: the whole reply is the schema's JSON, then end of turn
    fmt = {"type": "json_schema",
           "json_schema": {"name": "w", "schema": TOOLS[0]["function"]["parameters"]}}
    reply = '{"location": "Paris, France", "unit": "celsius"}'
    ids = tok.encode(reply, add_special_tokens=False) + [tok.eos_token_id]
    f = forced_rows.make(tok, None, fmt)
    assert f is not None and f.active
    i, forced, last = 0, 0, []
    while i < len(ids):
        p = f.propose()
        assert p == ids[i:i + len(p)], (i, p, ids[i:i + len(p)])
        for t in ids[i:i + len(p) + 1]:
            f.feed(t)
        forced += len(p)
        last = p or last
        i += len(p) + 1
    assert last[-1] == tok.eos_token_id, "the end of turn was never proposed"
    print(f"  real grammar [{engine}]: {forced}/{len(ids)} tokens of a "
          f"response_format reply forced, ending on the end of turn  OK")
    f = forced_rows.make(tok, None, fmt)
    for t in tok.encode("```json\n{", add_special_tokens=False):
        f.feed(t)
    assert not f.active and f.propose() == [], "a fenced reply kept it on"
    if engine == "xgrammar":        # the built-in grammars take any value there
        bad = {"type": "json_schema", "json_schema": {"schema": {"type": "nope"}}}
        assert forced_rows.make(tok, None, bad) is None
    print(f"  real grammar [{engine}]: a fenced reply switches it off"
          + (", a schema xgrammar cannot compile gets no forcer" if engine == "xgrammar"
             else "") + "  OK")


class Loose(Fake):
    """A Fake that follows whatever is emitted instead of asserting it is the
    reference: a spec.py run may leave the reference at a near-tie, and its
    proposals are then simply wrong. Records what it was fed; with
    `forks_only` it proposes nothing at a round's start, only after drafts."""

    def __init__(self, ref, mode, seed, forks_only=False):
        super().__init__(ref, mode, seed)
        self.fed, self.forks_only = [], forks_only

    def propose(self):
        return [] if self.forks_only and not self.forked else super().propose()

    def feed(self, tok):
        if not self.forked:
            self.fed.append(tok)
        self.seen += 1


def _hybrids():
    """state_memory's toy (Qwen3_5, GatedDeltaNet blocks, dense MLP) and the
    same with a routed MoE MLP -- the 35B's two ingredients, KB-scale."""
    from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs
    from support import toy_model
    mx.random.seed(0)
    moe = TextModel(TextModelArgs(
        model_type="qwen3_5", hidden_size=32, intermediate_size=64,
        num_hidden_layers=8, num_attention_heads=4, num_key_value_heads=2,
        vocab_size=128, head_dim=32, linear_num_value_heads=4,
        linear_num_key_heads=2, linear_key_head_dim=32,
        linear_value_head_dim=32, linear_conv_kernel_dim=4,
        full_attention_interval=4, tie_word_embeddings=False,
        num_experts=8, num_experts_per_tok=2, moe_intermediate_size=32,
        shared_expert_intermediate_size=32))
    mx.eval(moe.parameters())
    moe.eval()
    return (("dense", toy_model(seed=0)), ("moe", moe))


def _tie_ok(model, ids, got, ref, label):
    """got == ref, or they part where the target's top two are a near-tie."""
    d = next((i for i, (a, b) in enumerate(zip(got, ref)) if a != b), None)
    if d is not None:
        gap = tie_gap(model, ids + got[:d])
        assert gap < 1e-2, f"{label}: token {d} differs at a top-2 gap of {gap:.3g}"
    return d


def _spec_run(sp, ids, n, forcer):
    """One spec.py generation over a fresh cache, every pass width logged."""
    import spec
    core, widths = sp.core, []

    def watched(x, cache=None):
        widths.append(x.shape[-1])
        return core(x, cache=cache)
    sp.core, sp.forcer = watched, forcer
    cache, st = make_prompt_cache(sp.model), spec.Stats()
    got = list(sp.generate(ids, n, (), st, cache))
    assert max(widths[1:], default=0) <= spec.Speculator.max_window + 1, widths
    return got, st, cache


def test_spec_forced():
    """ForcedSpeculator, the 35B's pass when it has no drafter: fake forcers
    right, wrong and random, both toys, a long reply and one cut at 7."""
    import spec
    ids = [3, 9, 27, 81, 5, 11, 23, 47]
    n = 24
    for name, m in _hybrids():
        ref, _, _ = _spec_run(spec.ForcedSpeculator(m), ids, n, None)
        assert ref == list(greedy(m, ids, n)), f"{name}: no forcer != greedy"
        for mode in ("right", "wrong", "random"):
            for cut in (n, 7):
                label = f"{name}/{mode}/n={cut}"
                fake = Loose(ref, mode, seed=cut)
                got, st, cache = _spec_run(spec.ForcedSpeculator(m), ids, cut, fake)
                assert len(got) == cut and fake.fed == got, label
                check_cache(m, cache, ids + got, label)
                d = _tie_ok(m, ids, got, ref, label)
                assert st.forced_accepted <= st.forced and not st.seen, (label, st)
                if mode == "wrong":
                    assert st.forced_accepted == 0, (label, st)
                if mode == "right" and cut == n:
                    assert st.forced_accepted > 0, (label, st)
                print(f"  spec forced-only {label:22s} forced {st.forced_accepted}/"
                      f"{st.forced}, {st.rounds} rounds, {st.replays} replays, "
                      f"cache exact, no-forcer agrees to "
                      f"{d if d is not None else cut}/{cut}  OK")


def test_spec_forced_with_drafts():
    """A scripted head (spec.ScriptedSpeculator) and a forcer in one round:
    forced spans instead of the head's drafts and after them. seen/hit keep
    counting the head's rows only."""
    import spec
    ids = [3, 9, 27, 81, 5, 11, 23, 47]
    n = 24
    for name, m in _hybrids():
        ref = list(greedy(m, ids, n))
        for g, pattern in ((1, [True, False]), (2, [True, True, False])):
            for mode, after in (("right", False), ("right", True), ("wrong", False)):
                label = f"{name}/g={g}/{mode}" + ("/after" if after else "")
                fake = Loose(ref, mode, seed=g, forks_only=after)
                sp = ScriptedSpeculator(m, ref, pattern, vocab=128, gamma=g)
                got, st, cache = _spec_run(sp, ids, n, fake)
                assert fake.fed == got, label
                check_cache(m, cache, ids + got, label)
                d = _tie_ok(m, ids, got, ref, label)
                assert sum(st.seen) <= sp.i, (label, st)     # head rows only
                if mode == "right":
                    assert st.forced_accepted > 0, (label, st)
                    assert fake.spans[0] > 0 or not after, (label, st)
                else:
                    assert st.forced_accepted == 0, (label, st)
                print(f"  spec head+forced  {label:22s} head {sum(st.hit)}/"
                      f"{sum(st.seen)}, forced {st.forced_accepted}/{st.forced}, "
                      f"{fake.spans[0]} spans after drafts, cache exact, greedy "
                      f"agrees to {d if d is not None else n}/{n}  OK")
    # A real (random) MTP head: its drafts stay on the GPU until a forced
    # span after them needs them on the host.
    from mtp import MTPHead
    name, m = _hybrids()[0]
    head = MTPHead(m.args)
    mx.eval(head.parameters())
    head.eval()
    ref = list(greedy(m, ids, n))
    fake = Loose(ref, "right", 0, forks_only=True)
    got, st, cache = _spec_run(spec.Speculator(m, head, gamma=2), ids, n, fake)
    assert fake.fed == got and fake.spans[0] > 0, st
    check_cache(m, cache, ids + got, "mtp head")
    d = _tie_ok(m, ids, got, ref, "mtp head")
    print(f"  spec head+forced  {name}/random MTP head g=2  forced "
          f"{st.forced_accepted}/{st.forced}, {fake.spans[0]} spans after "
          f"drafts, cache exact, greedy agrees to {d if d is not None else n}/{n}  OK")


class Pinned:
    """A moe_stream.ForcedPolicy that always takes one arm."""

    def __init__(self, arm):
        self.arm, self.turns = arm, 0

    def pick(self):
        return self.arm

    def record(self, arm, ms):
        assert arm == self.arm and ms > 0, (arm, ms)
        self.turns += 1


def test_forced_policy():
    """ForcedPolicy on scripted turn timings: both arms explored first, then
    the faster one runs and the slower is re-probed -- rarely when the gap is
    many spreads wide, every other turn when it is within one -- and a swap of
    the two speeds (an arena that grew) is followed."""
    import moe_stream

    def run(pol, turns, ms_of, seed):
        rng, arms = random.Random(seed), []
        for _ in range(turns):
            arm = pol.pick()
            arms.append(arm)
            pol.record(arm, ms_of(arm) * (1 + 0.05 * rng.uniform(-1, 1)))
        return arms

    for fast in (True, False):
        pol = moe_stream.ForcedPolicy()
        arms = run(pol, 60, lambda a: 40.0 if a == fast else 55.0, 0)
        assert sorted(arms[:4]) == [False, False, True, True], arms[:4]
        tail = arms[4:]
        slow = [i for i, a in enumerate(tail) if a != fast]
        assert pol.best == fast and 0 < len(slow) <= len(tail) // 8, arms
        gaps = [b - a for a, b in zip(slow, slow[1:])]
        assert all(g <= pol.EVERY_MAX + 1 for g in gaps), gaps
        print(f"  policy: fast arm {'forced' if fast else 'plain'} wins, "
              f"slower probed {len(slow)}x in {len(tail)} turns  OK")

    pol = moe_stream.ForcedPolicy()
    arms = run(pol, 40, lambda a: 50.0 if a else 51.0, 1)
    probes = sum(a != pol.best for a in arms[4:])
    assert probes >= len(arms[4:]) // 3, arms
    print(f"  policy: a gap within the spread is re-probed {probes}x in "
          f"{len(arms) - 4} turns  OK")

    pol = moe_stream.ForcedPolicy()
    run(pol, 30, lambda a: 40.0 if not a else 55.0, 2)
    arms = run(pol, 80, lambda a: 40.0 if a else 55.0, 3)
    assert pol.best is True and arms[-10:].count(True) >= 8, arms
    print(f"  policy: speeds swapped, forced-only took over "
          f"{pol.history[-1][0] - 30} turns later, {arms[-20:].count(True)}/20 "
          f"at the end  OK")


def test_engine_forced_only():
    """moe_stream with no drafter and a tool request: with the policy pinned
    to either arm the replies are the plain engine's up to a near-tie, turn two
    continues the cache, stats carry forced_only and no mtp_*, and forced_*
    only on the forced arm; the default policy explores the forced arm first,
    then the plain one; with an MTP speculator the forcer rides its rounds."""
    import moe_stream
    if not support.has_tokenizer():
        print("  engine forced-only: SKIPPED (no tokenizer)")
        return
    tools = [{"type": "function", "function": {
        "name": "get_weather", "parameters": {"type": "object", "properties": {
            "city": {"type": "string"}}, "required": ["city"]}}}]
    ask = [{"role": "user", "content": "The weather in Berlin?"}]
    real, made = moe_stream.forced_rows.make, []
    moe_stream.forced_rows.make = lambda tk, tl, rf=None: None
    try:
        base = toy_llm("slice", max_tokens=24)
        base.ask(messages=ask, tools=tools)
        ref = base._seen[len(base._seen) - base.stats()["tokens"]:]
        prompt = base._seen[:len(base._seen) - len(ref)]
        base.ask("And tomorrow?")
        ref2 = base._seen[len(base._seen) - base.stats()["tokens"]:]
        prompt2 = base._seen[:len(base._seen) - len(ref2)]

        # turn 2 continues with the same tools, so it gets a forcer too --
        # proposing turn 1's reply, which the target rejects
        moe_stream.forced_rows.make = lambda tk, tl, rf=None: (
            made.append(tl) or (Loose(ref, "right", 0) if tl else None))
        for arm, with_mtp in ((True, False), (False, False), (None, True)):
            llm = toy_llm("slice", max_tokens=24)
            if arm is not None:
                llm.forced_policy = Pinned(arm)
            if with_mtp:
                llm._spec = ScriptedSpeculator(llm.model, ref, [True, False],
                                                    vocab=128)
            llm.ask(messages=ask, tools=tools)
            st = llm.stats()
            got = llm._seen[len(llm._seen) - st["tokens"]:]
            d = _tie_ok(llm.model, prompt, got, ref, f"engine mtp={with_mtp}")
            assert st.get("forced_only") == arm, st
            assert ("forced_rows" in st) == (arm is not False), st
            assert arm is False or st["forced_accepted"] > 0, st
            assert ("mtp_rounds" in st) == with_mtp, st
            llm.ask("And tomorrow?")
            st2 = llm.stats()
            assert st2["cached_tokens"] > 0, st2
            assert ("forced_rows" in st2) == (arm is not False), st2
            assert with_mtp or llm.forced_policy.turns == 2
            got2 = llm._seen[len(llm._seen) - st2["tokens"]:]
            assert arm is not False or (got, got2) == (ref, ref2), \
                "the plain arm is not the plain engine"
            d2 = (_tie_ok(llm.model, prompt2, got2, ref2, f"turn 2 mtp={with_mtp}")
                  if d is None else "-")
            for c in llm._cache:
                if isinstance(c, moe_stream.KVCache):
                    assert c.offset - c.gap == len(llm._seen)
            print(f"  engine arm={arm} mtp={with_mtp}: forced "
                  f"{st.get('forced_accepted', 0)}/"
                  f"{st.get('forced_rows', 0)}, agrees with the plain engine to "
                  f"{d if d is not None else len(ref)}/{len(ref)}, turn 2 "
                  f"{d2 if d2 is not None else len(ref2)}/{len(ref2)} with "
                  f"{st2['cached_tokens']} cached  OK")
        llm = toy_llm("slice", max_tokens=24)
        llm.ask(messages=ask, tools=tools)
        llm.ask("And tomorrow?")
        pol = llm.forced_policy
        assert [a for a, v in pol.ms.items() if v] == [True, False], pol.ms
        assert llm.stats()["forced_only"] is False, llm.stats()
        print(f"  engine default policy: turn 1 forced-only "
              f"({pol.ms[True][0]:.1f} ms/token), turn 2 plain "
              f"({pol.ms[False][0]:.1f} ms/token)  OK")
    finally:
        moe_stream.forced_rows.make = real


def test_decode_policy_n_arm():
    """ForcedPolicy past two arms: moe_stream's MTP-loaded engines run one
    with ("mtp", "forced_only", "plain") for a turn that has a forcer, one
    with ("mtp", "plain") for a turn that does not. Warm-up round-robins
    every arm two turns each in that order -- a fresh policy's first turns
    are its first arm, which is why "mtp" leads both tuples: a cold policy
    starts out behaving exactly like the unconditional MTP path it replaces.
    Past warm-up the fastest of the three converges, and of the two slower
    ones the closer converges to the winner is re-probed more often -- its
    own z is smaller, so z*z reaches EVERY_MAX later."""
    import moe_stream

    def run(pol, turns, ms_of, seed):
        rng, arms = random.Random(seed), []
        for _ in range(turns):
            arm = pol.pick()
            arms.append(arm)
            pol.record(arm, ms_of(arm) * (1 + 0.05 * rng.uniform(-1, 1)))
        return arms

    pol = moe_stream.ForcedPolicy(("mtp", "forced_only", "plain"))
    means = {"mtp": 40.0, "forced_only": 45.0, "plain": 60.0}
    arms = run(pol, 6, lambda a: means[a], 1)
    assert arms == ["mtp", "forced_only", "plain"] * 2, arms
    arms += run(pol, 150, lambda a: means[a], 1)
    assert pol.best == "mtp", pol.best
    tail = arms[6:]
    close, far = tail.count("forced_only"), tail.count("plain")
    assert close > far >= 1, (close, far)
    print(f"  3-arm policy: warm-up {arms[:6]}, winner {pol.best!r}, "
          f"re-probed the closer arm {close}x vs the farther {far}x in "
          f"{len(tail)} turns  OK")

    # With arms close enough together that a `due` probe can land right after another one, the
    # self.last[best] == self.turn guard in pick() keeps two slower arms from filling every turn
    # and shutting the best arm out of its own samples for good.
    rng = random.Random(7)
    pol = moe_stream.ForcedPolicy(("mtp", "plain", "forced_only"))
    means = {"mtp": 55.9, "plain": 61.0, "forced_only": 62.2}
    arms = []
    for _ in range(300):
        arm = pol.pick()
        arms.append(arm)
        arms_ms = max(rng.gauss(means[arm], 10.0), 1.0)
        pol.record(arm, arms_ms)
    frac = arms[6:].count("mtp") / len(arms[6:])
    assert frac >= 0.4, (frac, arms[-30:])
    print(f"  3-arm policy, means within a spread: 'mtp' (truth: fastest) "
          f"took {frac:.0%} of {len(arms) - 6} post-warm-up turns, tail "
          f"{arms[-10:]}  OK")


def test_engine_decode_policy():
    """An MTP-loaded engine keeps one N-arm policy per turn kind rather than
    one for both, since a turn with a forcer and one without measured
    different winners on the 35B (forced_only has nothing to do without a
    forcer). Every pinned arm -- plain, forced_only, mtp -- is the plain
    engine's own greedy reply up to a near-tie, "decode_arm" is the key
    stats() carries (not "forced_only", that stays a no-MTP-engine thing),
    and only the "mtp" arm gets mtp_* keys: the bug this guards is `sp` (a
    ForcedSpeculator, not self._spec) still being truthy on the forced_only
    arm of an MTP-loaded engine, which could mislabel that round as the
    head's own. A forcer-less turn gets its own policy, with no forced_only
    arm to pick since there is nothing for it to verify."""
    import moe_stream
    if not support.has_tokenizer():
        print("  engine decode policy: SKIPPED (no tokenizer)")
        return
    tools = [{"type": "function", "function": {
        "name": "get_weather", "parameters": {"type": "object", "properties": {
            "city": {"type": "string"}}, "required": ["city"]}}}]
    ask = [{"role": "user", "content": "The weather in Berlin?"}]
    base = toy_llm("slice", max_tokens=16)
    base.ask(messages=ask, tools=tools)
    ref = base._seen[len(base._seen) - base.stats()["tokens"]:]
    prompt = base._seen[:len(base._seen) - len(ref)]
    real = moe_stream.forced_rows.make
    moe_stream.forced_rows.make = lambda tk, tl, rf=None: (
        Loose(ref, "right", 0) if tl else None)
    try:
        for arm in ("mtp", "forced_only", "plain"):
            llm = toy_llm("slice", max_tokens=16)
            llm._spec = ScriptedSpeculator(llm.model, ref, [True, False],
                                                vocab=128)
            llm.decode_policy_call = Pinned(arm)
            llm.ask(messages=ask, tools=tools)
            st = llm.stats()
            got = llm._seen[len(llm._seen) - st["tokens"]:]
            d = _tie_ok(llm.model, prompt, got, ref, f"decode arm={arm}")
            assert st.get("decode_arm") == arm, st
            assert "forced_only" not in st, st
            assert ("mtp_rounds" in st) == (arm == "mtp"), (arm, st)
            assert ("forced_rows" in st) == (arm != "plain"), (arm, st)
            assert arm == "plain" or st["forced_accepted"] > 0, st
            for c in llm._cache:
                if isinstance(c, moe_stream.KVCache):
                    assert c.offset - c.gap == len(llm._seen)
            print(f"  decode arm={arm}: agrees with the plain engine to "
                  f"{d if d is not None else len(ref)}/{len(ref)}, "
                  f"mtp_rounds {'in' if 'mtp_rounds' in st else 'not in'} "
                  f"stats as expected  OK")

        # Same engine, a fresh render with no tools this time: a genuinely
        # forcer-less turn picks from its own policy (plain/mtp only -- there
        # is nothing for forced_only to verify), kept separate from the one
        # above.
        llm = toy_llm("slice", max_tokens=16)
        llm._spec = ScriptedSpeculator(llm.model, ref, [True, False],
                                            vocab=128)
        llm.ask(messages=ask, tools=tools)
        assert llm.decode_policy_call is not None
        assert llm.decode_policy_free is None, "no forcer-less turn ran yet"
        llm.ask(messages=[{"role": "user", "content": "The capital of France?"}])
        st2 = llm.stats()
        assert st2.get("decode_arm") in ("mtp", "plain"), st2
        assert llm.decode_policy_free is not None
        assert llm.decode_policy_free is not llm.decode_policy_call
        assert llm.decode_policy_free.arms == ("mtp", "plain")
        assert "forced_only" not in st2 and "forced_rows" not in st2, st2
        print(f"  engine decode policy: forcer-less turn arm="
              f"{st2['decode_arm']!r}, kept on its own policy  OK")
    finally:
        moe_stream.forced_rows.make = real


def test_real_xml_grammar():
    tok = _tokenizer(*QWEN36)
    if tok is None:
        print("  real XML grammar: SKIPPED (no tokenizer)")
        return
    from tool_fixture import TOOLS
    assert tok.tool_parser.__module__.endswith(".qwen3_coder")
    for lite in _engines():
        with _built_in(lite) as engine:
            _real_xml_grammar(tok, TOOLS, engine)


def _real_xml_grammar(tok, TOOLS, engine):
    f = forced_rows.make(tok, TOOLS)
    assert f is not None
    body = ("\n<function=create_calendar_event>\n<parameter=title>\nLaunch\n"
            "</parameter>\n<parameter=date>\n2026-12-01\n</parameter>\n"
            "<parameter=start_time>\n10:00\n</parameter>\n"
            "<parameter=attendees>\n[\"a@b.c\",\n </div>\n</p \"d@e.f\"]\n"
            "</parameter>\n</function>\n")
    ids = ([f.open_id] + tok.encode(body, add_special_tokens=False)
           + [f.close_id])
    forced = _walk(f, ids)
    assert not f.active and f.propose() == []
    print(f"  real XML grammar [{engine}]: {forced}/{len(ids)} tokens of a call "
          f"with a multi-line value forced, every proposal a prefix  OK")
    swapped = ("\n<function=create_calendar_event>\n<parameter=date>\n")
    f.feed(f.open_id)
    for t in tok.encode(swapped, add_special_tokens=False):
        f.feed(t)
    assert not f.active and f.propose() == []
    print(f"  real XML grammar [{engine}]: parameters out of declared order "
          f"switch it off  OK")


def test_built_in_values():
    """What the built-in grammars know beyond xgrammar's, on the real
    tokenizers: a string enum's value once its first letters pick it and a
    number's line end in the XML call, and free text behind both, so a value
    off the list only stops the proposals until its parameter closes (in
    JSON too, where xgrammar's grammar would switch off), plus two recorded
    calls over the tool-list fixture and a json_object reply of nested
    values ending on the end of turn."""
    from tool_fixture import TOOLS
    xml, js = _tokenizer(*QWEN36), _tokenizer(*QWEN3)
    if xml is None or js is None:
        print("  built-in values: SKIPPED (no Qwen3 or Qwen3.6 tokenizer)")
        return
    text = lambda tok, ids: tok.decode([t for t in ids if t != tok.eos_token_id])  # noqa: E731
    with _built_in(True):
        def after(tok, tools, s, fmt=None):
            f = forced_rows.make(tok, tools, fmt)
            for t in ([] if fmt else [f.open_id]) + tok.encode(s, add_special_tokens=False):
                f.feed(t)
            return f, f.propose()

        head = "\n<function=get_weather>\n<parameter=location>\nParis\n</parameter>\n"
        f, p = after(xml, TOOLS, head + "<parameter=unit>\nc")
        assert text(xml, p[:-1]) == "elsius\n</parameter>\n</function>\n" \
            and p[-1] == f.close_id, text(xml, p)
        f, p = after(xml, TOOLS, head + "<parameter=unit>\nk")
        assert p == [] and f.active
        for t in xml.encode("elvin\n</parameter>\n", add_special_tokens=False):
            f.feed(t)
        assert text(xml, f.propose()[:-1]) == "</function>\n", "lost after kelvin"
        _, p = after(xml, TOOLS, "\n<function=calculate>\n<parameter=a>\n45\n")
        assert text(xml, p).startswith("</parameter>\n<parameter=b>"), text(xml, p)
        f, p = after(js, TOOLS, '\n{"name": "get_weather", "arguments": '
                     '{"location": "Paris", "unit": "kelvin"')
        assert text(js, p[:-1]) == "}}\n" and p[-1] == f.close_id, text(js, p)
        print("  built-in values: enum completed from its first letter, a "
              "number's line end, a value off the list skipped  OK")

        reply = '{"city": "Paris", "days": [1, 2.5, {"a": null}], "ok": true}'
        ids = js.encode(reply, add_special_tokens=False) + [js.eos_token_id]
        f = forced_rows.make(js, None, {"type": "json_object"})
        assert _walk(f, ids[:-1]) == 0 and f.propose() == [js.eos_token_id]
        print("  built-in values: a json_object reply of nested values, the end "
              "of turn proposed after its last brace  OK")


def test_torch_unblock():
    """_xgrammar() has to leave a real "xgrammar not installed" alone and, on an install where xgrammar's own __init__ chain does `import torch` unconditionally, lift the lean-host shadow so that succeeds, using a stub package for the matcher.py behaviour in a subprocess (so the shadowed torch and the stub never leak into later tests) and `sys.modules['xgrammar'] = None` to force the "genuinely missing" branch even where xgrammar is installed, so it passes the same way everywhere."""
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        pkg = Path(tmp) / "xgrammar"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("import torch\n")
        script = f"""
import sys
sys.path.insert(0, {str(ROOT)!r})
sys.modules['torch'] = None
sys.modules['xgrammar'] = None
import forced_rows

assert forced_rows._xgrammar() is False, "xgrammar truly missing must return False"
assert sys.modules['torch'] is None, "shadow must survive a genuinely-missing xgrammar"

del sys.modules['xgrammar']
sys.path.insert(0, {tmp!r})
forced_rows._notice[0] = False
assert forced_rows._xgrammar() is True, "stub xgrammar (needs torch) must import once unblocked"
assert sys.modules['torch'] is not None, "torch must be the real module now"
print("subprocess checks passed")
"""
        r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "subprocess checks passed" in r.stdout, r.stdout + r.stderr
    print("  torch unblock: shadow kept when xgrammar is missing (forced so "
          "even where it's really installed), lifted when it isn't  OK")


def test_torch_unblock_partial_import():
    """A regression case for _xgrammar(): xgrammar's own __init__ chain can
    import torch for real (matcher.py) and *then* hit a different missing
    dependency further down (pydantic, tvm_ffi, ...) -- that ImportError
    still lands in _xgrammar()'s except clause, but torch by then is a real,
    fully-loaded module, not the thing that's missing. Re-shadowing it there
    doesn't undo the load (the C extension stays resident), so the next lift
    used to re-run torch/__init__.py's registration over an already-
    registered extension and raise RuntimeError ("Only a single
    TORCH_LIBRARY..."), not ImportError -- past _xgrammar() and past
    make() (moe_stream.py's forcer_ok path), failing the request outright.
    Two calls, both required to return False with nothing raised.
    """
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        pkg = Path(tmp) / "xgrammar"
        pkg.mkdir()
        (pkg / "__init__.py").write_text(
            "import torch\nimport a_dependency_xgrammar_does_not_ship\n")
        script = f"""
import sys
sys.path.insert(0, {str(ROOT)!r})
sys.path.insert(0, {tmp!r})
sys.modules['torch'] = None
import forced_rows

assert forced_rows._xgrammar() is False, "call 1: torch loads, then the real ImportError -- False"
assert forced_rows._xgrammar() is False, "call 2 must also return False, not raise RuntimeError"
print("subprocess checks passed")
"""
        r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "subprocess checks passed" in r.stdout, r.stdout + r.stderr
    print("  torch unblock: a partial import (torch loads, then something "
          "else is missing) doesn't re-shadow a real torch  OK")



def test_matcher_error_switches_off():
    """A matcher that raises (the built-in one is Python) turns the forcer off
    for the call instead of failing the request: feed, propose, fork and the
    matcher's own construction are all covered."""
    class Boom:
        def accept_token(self, t): raise RecursionError("deep")
        def is_terminated(self): raise IndexError("x")
        def find_jump_forward_string(self): raise IndexError("x")
        def fork(self): raise ValueError("fork")
    f = forced_rows.Forcer.__new__(forced_rows.Forcer)
    f._matcher, f._encode, f.open_id, f.close_id = (lambda: Boom()), None, 7, 8
    f._m = None
    f.feed(7)
    assert f.active
    assert not f.fork().active and f.active
    assert f.propose() == [] and not f.active
    f.feed(7)
    f.feed(5)
    assert not f.active
    def fails():
        raise RecursionError("expand")
    f._matcher = fails
    f.feed(7)
    assert not f.active
    print("  matcher errors switch the forcer off, the request goes on  OK")

if __name__ == "__main__":
    test_fixed_chain()
    test_proposers()
    test_row_cap()
    test_layout_cache()
    test_gate()
    test_response_format_gate()
    test_engine_response_format()
    test_real_reply_grammar()
    test_real_grammar()
    test_spec_forced()
    test_spec_forced_with_drafts()
    test_forced_policy()
    test_engine_forced_only()
    test_decode_policy_n_arm()
    test_engine_decode_policy()
    test_real_xml_grammar()
    test_built_in_values()
    test_torch_unblock()
    test_torch_unblock_partial_import()
    test_matcher_error_switches_off()
    try:
        import test_forced_rows_extra
    except ImportError:
        pass
    else:
        test_forced_rows_extra.main()
    print("test_forced_rows OK")
