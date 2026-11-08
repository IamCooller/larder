#!/usr/bin/env python3
"""server.py's batch loop (ModelHost._batch_loop) on a toy dense target.

    python3 tests/test_batching.py

A toy llama pair (spec_draft's, residual branches scaled so the next token
depends on the context) around the checkpoint's real tokenizer, driven
through ModelHost.generate -- the handler's own entry, no HTTP. Six
configurations of the solo path: plain decode, a draft chain,
a draft found on disk (draft="auto": an arm of the decode policy), a forcer
with no draft (the ForcedPolicy arm), and a forcer with a draft. In each:

1. one request alone is the engine's own llm.ask reply, llm.stream ran its
   turn to the end (stats() counts the reply), with its draft where there
   is one;
2. four requests at once, max_tokens 9..30: each reply is its solo greedy
   one, they shared forwards (width 4), the three behind the first came in
   as rows together -- prefilled and nothing more (llm.stream's
   prefill_only) -- before a step ran, the steps ran one ahead (a forward
   queued on the ids of the one before, rows finishing in between), and the
   last row went back to llm.stream;
3. a request joining mid-way: the first is held at its 7th token until the
   second is queued, then leaves llm.stream (speculating, where the
   configuration does) for the batch, the second comes in as a row and
   leaves early, and the first finishes back in llm.stream -- both replies
   still the solo ones;
4. the width caps: a kernel covering 2 rows never sees a third, and a memory
   budget with no room leaves every request to run alone;
5. hanging up: a client that drops its stream at its 3rd token takes its row
   out of the batch at the next step (no 4th token is made for it) while
   the other three replies stay the solo ones, and one that hangs up on a
   turn running alone ends it there, the next request's reply still its
   solo one.

With the forcer, each request's forcer is an oracle for its own solo reply
that asserts it is fed exactly that reply, in order: a turn handed back to
llm.stream mid-reply has to arrive with its grammar where the reply is.
With no draft, ForcedPolicy has sampled the turn that ran alone and none of
the turns in 2 and 3, each of which shared forwards; with the auto draft,
the decode policy likewise.

And the gate: a hybrid target (GatedDeltaNet state, what the MoE checkpoints
are) is not batchable, so it keeps the serial path.
"""

import copy
import faulthandler
import queue
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import moe_stream                                              # noqa: E402
import support                                                 # noqa: E402
import server                                                  # noqa: E402
import spec_draft                                              # noqa: E402

ASKS = [("Say the word apple.", 23), ("Name three rivers.", 9),
        ("Count to five.", 17), ("What is MoE routing?", 30)]
MSGS = [[{"role": "user", "content": q}] for q, _ in ASKS]


def toys(vocab):
    """Target and draft. Plain random toys repeat one token forever, which
    would hide a cache that is off by a position; sharper attention and
    louder residual branches make nearly every reply token a new one, and a
    prompt with one token dropped or doubled a different reply."""
    target, draft, _ = spec_draft._drafts(vocab, 64, 3, 2)
    for m in (target, draft):
        for layer in m.model.layers:
            a = layer.self_attn
            a.q_proj.weight = 3 * a.q_proj.weight
            a.k_proj.weight = 3 * a.k_proj.weight
            a.o_proj.weight = 4 * a.o_proj.weight
            layer.mlp.down_proj.weight = 4 * layer.mlp.down_proj.weight
        mx.eval(m.parameters())
    return target, draft


def engine(target, tok, draft=None, auto=False):
    """spec_draft.engine_selftest's toy engine, at the server's stub mode;
    `auto` makes its draft an arm of the decode policy (draft="auto")."""
    llm = moe_stream.StreamingLLM.__new__(moe_stream.StreamingLLM)
    llm.model, llm.tokenizer = target, tok
    llm.max_tokens, llm.prefill_step, llm.resident = 16, None, 0
    llm.thinking, llm.profile = False, None
    llm.model_path = support.MODEL_LABEL
    llm._setup_chat("slice")
    llm._draft_model, llm._draft_seen = draft, []
    llm._draft_cache = make_prompt_cache(draft) if draft else None
    llm.draft_tokens, llm.draft_tau = 4, None
    llm._draft_auto = auto
    llm.reset()
    return llm


def host(llm):
    """A ModelHost around a built engine, the fields generate() reads."""
    h = server.ModelHost.__new__(server.ModelHost)
    h.executor = ThreadPoolExecutor(max_workers=1)
    h.llm, h.store, h.mem_pct, h.host_gb = llm, None, 90, 0.6
    h._inbox, h._lock, h._looping = queue.Queue(), threading.Lock(), False
    h.batch = server.ModelHost._can_batch(llm)
    h._detok = llm.tokenizer.detokenizer
    # The widest step, the ways into and out of the batch, the steps that
    # handed out the ids of a forward queued before them, and their order.
    seen = {"width": 0, "to_row": 0, "row": 0, "to_solo": 0, "ahead": 0,
            "log": []}
    step, row, to_row, to_solo = h._step, h._row, h._to_row, h._to_solo

    def _step(rows, batch, pend):
        seen["width"] = max(seen["width"], len(rows))
        seen["ahead"] += pend is not None
        seen["log"].append("step")
        return step(rows, batch, pend)

    def _row(t):
        seen["row"] += 1
        seen["log"].append("row")
        return row(t)

    def _to_row(t):
        seen["to_row"] += 1
        seen["log"].append("to_row")
        return to_row(t)

    def _to_solo(t, batch):
        seen["to_solo"] += 1
        seen["log"].append("to_solo")
        return to_solo(t, batch)

    h._step, h._row, h._to_row, h._to_solo = _step, _row, _to_row, _to_solo
    return h, seen


def clear(seen):
    for k in seen:
        seen[k] = [] if k == "log" else 0


class Oracle:
    """A forcer proposing a request's solo reply, asserting it is fed exactly
    that reply: every token, in order, from the first."""
    active = True

    def __init__(self, ref):
        self.ref, self.seen, self.forked = ref, 0, False

    def fork(self):
        f = copy.copy(self)
        f.forked = True
        return f

    def feed(self, tok):
        if not self.forked:
            assert tok == self.ref[self.seen], (self.seen, tok, self.ref)
        self.seen += 1

    def propose(self):
        return self.ref[self.seen:self.seen + 3]


def fmt(i):
    return {"type": "json_object", "i": i}


def at_once(h, idx, forced, infos=None):
    """Requests queued while the worker is busy, so the loop sees them all
    in its first look; their replies. `infos` are the dicts generate() fills
    with each request's token counts."""
    gate = threading.Event()
    h.executor.submit(gate.wait)
    try:
        pieces = [h.generate(MSGS[i], ASKS[i][1], stream=True,
                             response_format=fmt(i) if forced else None,
                             info=infos[n] if infos else None)
                  for n, i in enumerate(idx)]
    finally:
        gate.set()      # a parked worker would keep the process from exiting
    return ["".join(p) for p in pieces]


def held(h, at, n):
    """h._emit parked once the request asking MSGS[at] has had n tokens, until
    the returned `go` is set; `reached` says it got there."""
    reached, go, emit = threading.Event(), threading.Event(), h._emit

    def held_emit(t, tok):
        done = emit(t, tok)
        if t.messages is MSGS[at] and t.n == n and not reached.is_set():
            reached.set()
            go.wait()
        return done

    h._emit = held_emit
    return reached, go, emit


def run(name, target, tok, refs, draft=None, forced=False, auto=False):
    h, seen = host(engine(target, tok, draft, auto))
    assert h.batch, "a dense toy with plain KV caches has to batch"
    rf = (lambda i: fmt(i)) if forced else (lambda i: None)
    want = [r[0] for r in refs]

    # 1. alone: today's path, turn closed by llm.stream itself
    info = {}
    assert h.generate(MSGS[0], ASKS[0][1], stream=False,
                      response_format=rf(0), info=info) == refs[0][0]
    st = h.llm.stats()
    assert st["tokens"] == len(refs[0][1]), st
    # what the API reports as `usage`, counted by the worker as the reply ends
    assert info["completion_tokens"] == st["tokens"], (info, st)
    assert info["prompt_tokens"] == st["prompt_tokens"] + st["cached_tokens"], (info, st)
    # a turn that ran alone has the engine's cached count, and its timings
    assert info["cached_tokens"] == st["cached_tokens"] and info["ttft_s"] > 0, info
    assert info["decode_s"] > 0, info
    done = h.counters.requests
    assert ("draft_acceptance" in st) == (draft is not None), st
    assert seen["log"] == [], seen

    # 2. four at once: the first starts alone, the other three come in
    # together behind it, and the steps run one ahead
    infos = [{} for _ in range(4)]
    got = at_once(h, range(4), forced, infos)
    assert got == want, (name, got, want)
    # rows that shared forwards have no stats() of their own: the counts come
    # from the request itself
    assert [x["completion_tokens"] for x in infos] == [len(r[1]) for r in refs], infos
    assert all(x["prompt_tokens"] > 0 and x["ttft_s"] > 0 for x in infos), infos
    assert all("cached_tokens" not in x for x in infos), "a shared forward has no cache count"
    assert h.counters.requests == done + 4, "every finished reply is counted once"
    assert seen["width"] == 4 and seen["to_solo"] >= 1, seen
    assert seen["log"][:5] == ["to_row", "row", "row", "row", "step"], seen["log"]
    assert seen["ahead"] == seen["log"].count("step") - 1, seen

    # 3. a join mid-way and an early leave
    clear(seen)
    reached, go, emit = held(h, 3, 7)
    try:
        long = h.generate(MSGS[3], ASKS[3][1], stream=True,
                          response_format=rf(3))
        assert reached.wait(60), "the first request never reached its 7th token"
        short = h.generate(MSGS[1], ASKS[1][1], stream=True,
                           response_format=rf(1))
    finally:
        go.set()        # same: never leave the worker parked on a failure
    assert "".join(short) == refs[1][0]
    assert "".join(long) == refs[3][0]
    h._emit = emit
    # the long one into the batch mid-reply, the short one in, and the long
    # one back to llm.stream once the short one has left
    assert seen["width"] == 2 and seen["to_row"] == 1 and seen["row"] == 1, seen
    assert seen["to_solo"] == 1, seen
    for pol in (h.llm.forced_policy, h.llm.decode_policy_free,
                h.llm.decode_policy_call):
        assert pol is None or pol.turn == 1, "a batched turn became a sample"

    # 4. the caps: two rows of kernel, then no memory at all
    target.rowqmm_rows = 2
    clear(seen)
    try:
        assert at_once(h, range(4), forced) == want
    finally:
        del target.rowqmm_rows
    assert seen["width"] == 2, seen
    h.mem_pct = 0
    clear(seen)
    assert at_once(h, range(4), forced) == want
    assert seen["log"] == [], seen
    h.mem_pct = 90

    # 5. hanging up: a row at its 3rd token, then a turn running alone
    clear(seen)
    made = {}
    reached, go, emit = held(h, 3, 3)

    def counted(t, tok):
        made[id(t.messages)] = t.n + 1
        return held_emit(t, tok)
    held_emit, h._emit = h._emit, counted
    gate = threading.Event()
    h.executor.submit(gate.wait)
    try:
        pieces = [h.generate(MSGS[i], ASKS[i][1], stream=True,
                             response_format=rf(i)) for i in range(4)]
    finally:
        gate.set()
    try:
        assert reached.wait(60), "the row never reached its 3rd token"
        assert [next(pieces[3]) for _ in range(3)]
        pieces[3].close()
    finally:
        go.set()
    assert ["".join(p) for p in pieces[:3]] == want[:3]
    h._emit = emit
    assert made[id(MSGS[3])] == 3, made
    assert seen["width"] == 4, seen

    reached, go, emit = held(h, 3, 2)
    try:
        p = h.generate(MSGS[3], ASKS[3][1], stream=True, response_format=rf(3))
        assert reached.wait(60), "the turn never reached its 2nd token"
        next(p)
        p.close()
    finally:
        go.set()
    h._emit = emit
    assert h.generate(MSGS[2], ASKS[2][1], stream=False,
                      response_format=rf(2)) == refs[2][0]
    h.executor.shutdown()
    print(f"  {name}: alone, 4 at once, a join mid-way, both caps, hang-ups -- "
          f"every reply the solo one  OK", flush=True)


def main():
    from mlx_lm.utils import load_tokenizer

    # A stall anywhere -- the loop, a worker parked on a failed case -- dumps
    # every thread's stack and exits rather than blocking whoever ran it.
    faulthandler.dump_traceback_later(300, exit=True)

    path = support.tokenizer_dir()
    if not support.has_tokenizer():
        print("SKIPPED (no tokenizer: set MOE_TOY_TOKENIZER)")
        return
    # On the CPU, where "the solo reply" can be asserted token for token. On
    # the GPU this fp32 toy moves by up to ~1e-2 between a one-row and a
    # two-row forward of the same ids, and reply 3 has a 0.0053 logit gap at
    # its 14th token: which forward a step shares with whom flips it, the
    # way a bf16 tie flips on a real model (a real-model tie check).
    from test_reuse import cpu
    cpu()
    gen = sys.modules["mlx_lm.generate"]
    spec_draft.generation_stream = gen.generation_stream
    spec_draft.wired_limit = gen.wired_limit
    tok = load_tokenizer(path)
    vocab = max([tok.vocab_size] + list(tok.get_added_vocab().values())) + 1
    target, draft = toys(vocab)

    # The solo greedy replies: the engine's own plain turn, one at a time.
    ref = engine(target, tok)
    refs = []
    for m, (_, k) in zip(MSGS, ASKS):
        text = ref.ask(messages=m, max_tokens=k)
        ids = ref._seen[len(ref._seen) - ref.stats()["tokens"]:]
        refs.append((text, ids))
    assert all(len(set(ids)) > len(ids) // 2 for _, ids in refs), \
        "the toy repeats itself; a misaligned cache would not show"

    run("plain", target, tok, refs)
    run("draft chain", target, tok, refs, draft=draft)
    run("auto draft", target, tok, refs, draft=draft, auto=True)
    real = moe_stream.forced_rows.make
    moe_stream.forced_rows.make = lambda tk, tools, rf=None: (
        Oracle(refs[rf["i"]][1]) if rf else None)
    try:
        run("forcer, no draft", target, tok, refs, forced=True)
        run("forcer + draft", target, tok, refs, draft=draft, forced=True)
    finally:
        moe_stream.forced_rows.make = real

    from support import toy_model
    assert not server.ModelHost._can_batch(engine(toy_model(vocab=vocab), tok))
    print("  a hybrid (GatedDeltaNet) target is not batchable: serial  OK",
          flush=True)
    print("ok")


if __name__ == "__main__":
    main()
