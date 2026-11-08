#!/usr/bin/env python3
"""Shadow render (StreamingLLM.shadow, server.py's ModelHost._after/_shadow):
after a reply that goes back to the user, the history the next user message
re-renders is prefilled into a kept conversation of its own.

CPU, fp32, support.toy_llm's hybrid (GatedDeltaNet + KV layers, the real
Qwen3.6 tokenizer and template), its layers through stream_fast's prefill
wrapper as a streamed MoE's are, so a pass can be stopped.

  1. A 4-step agent loop (3 tool calls, then a final answer), then the shadow:
     branched at the mark after the task, its ids the render of the loop with
     the think stubs gone, up to the next user turn's header; kept as the
     newest conversation beside the live one.
  2. The user's next message takes the shadow up whole and feeds only itself;
     its reply equals the one without a shadow (taken up at the task's mark),
     and the cache equals one prefill of the ids it holds.
  3. Stopped: in its first layer nothing is kept and the next message is as
     without it; later, every layer before the one it stopped in is kept with
     that layer's input (start or middle of a layer, one pass or the second
     of three), and the next message runs the rest of that pass before its
     own ids: the cache then equals the complete shadow's bit for bit, the
     reply is the same, the cache equals one prefill.
  4. No room beside the live conversation (keep_convs 1, a tiny keep_gb) or a
     budget: no shadow. A plain chat turn has one: its reply loses its stub.
  5. server.py: a reply with a tool call queues no shadow, one without does;
     a request submitted meanwhile makes the queued shadow do nothing and
     stops one in progress.
  6. On a streamed MoE (GPU, test_layer_prefill's packed toy) a pass stopped
     at a layer start, after a dense half or between wide groups, its next
     layer read ahead, leaves no prefill state behind; taken up from the layer
     it stopped in (stream_fast.prefill_skip), or run again when that is the
     first, it gives the uninterrupted pass's logits, cache arrays and arenas
     bit for bit. chunkattn's query blocks check the stop between blocks.

    python3 tests/test_shadow.py
"""
import faulthandler
import sys
import tempfile
import types
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

import mlx.core as mx

import moe_stream
import pcache
import spec_draft
import stream_fast
import support
from support import toy_llm
from test_layer_prefill import arrays, same
from test_reuse import (CALLS, QUERY, SYSTEM, TOL, TOOLS, cpu, fresh_gap, scripted,
                        step_msgs)

FINAL = "Done: tests/test_save.py checks the fronts directory and it passes."
NEXT = "Thanks. Now make the test also check that a damaged front is ignored."


def engine(keep_convs=4):
    llm = toy_llm("slice", max_tokens=12)
    llm.reuse, llm.keep_convs = True, keep_convs
    stream_fast.install_prefill_sync(llm.model)
    return llm


def loop(llm):
    """3 tool-call steps and a final answer, scripted; the messages a client
    sends next, without its new user turn."""
    tok = llm.tokenizer
    end = tok.encode("<|im_end|>", add_special_tokens=False)
    replies = [tok.encode(c, add_special_tokens=False) + end for c in CALLS[:3] + [FINAL]]
    real, spec_draft.plain = spec_draft.plain, scripted(replies, set(tok.eos_token_ids))
    try:
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": QUERY}]
        for k in range(3):
            msgs = msgs + step_msgs(llm.ask(messages=msgs, tools=TOOLS, max_tokens=64), k)
        text = llm.ask(messages=msgs, tools=TOOLS, max_tokens=64)
    finally:
        spec_draft.plain = real
    return msgs + [{"role": "assistant", "content": text}]


def ask_next(llm, msgs):
    llm.ask(messages=msgs + [{"role": "user", "content": NEXT}], tools=TOOLS)
    st = llm.stats()
    return llm._seen[len(llm._seen) - st["tokens"]:], st


def check_shadow():
    ref = engine()
    msgs = loop(ref)
    q = next(n for n, _, query in ref._convs[-1]["marks"] if query)
    want, st0 = ask_next(ref, msgs)
    assert st0["cached_tokens"] == q, (st0["cached_tokens"], q)

    llm = engine()
    assert loop(llm) == msgs
    live = llm._convs[-1]
    info = llm.shadow(msgs, TOOLS)
    assert info and not info.get("stopped"), info
    sh = llm._convs[-1]
    assert len(llm._convs) == 2 and llm._convs[0] is live and llm._cache is live["cache"]
    assert info["from"] == q and len(sh["seen"]) == q + info["ids"], (info, q)
    a = llm.tokenizer.encode(llm._raw_render(msgs + [{"role": "user", "content": NEXT}]))
    assert a[:len(sh["seen"])] == sh["seen"], "the shadow's ids are not the next render's"
    stub = llm.tokenizer.encode("<think>\n\n</think>\n\n", add_special_tokens=False)
    assert not any(sh["seen"][i:i + len(stub)] == stub for i in range(len(sh["seen"]))), \
        "a think stub left in the shadow"
    assert live["seen"][:q] == sh["seen"][:q] and live["seen"][q] != sh["seen"][q]
    print(f"1. shadow after a 4-step loop: {info['ids']} ids from the task's mark ({q}) of "
          f"{len(live['seen'])} held, passes {info['passes']}; kept beside the live one "
          f"{info['kept_mb']} MB")
    full = arrays(sh["cache"])
    mx.eval(full)

    got, st = ask_next(llm, msgs)
    assert st["cached_tokens"] == len(sh["seen"]), (st["cached_tokens"], len(sh["seen"]))
    assert got == want, f"the reply moved: {got} vs {want}"
    gap = fresh_gap(llm)
    assert gap < TOL, gap
    print(f"2. the next message took the shadow up whole: fed {st['prompt_tokens']} ids "
          f"(without it {st0['prompt_tokens']}), same reply, cache vs one prefill {gap:.2g}")
    return msgs, want, full


def check_stopped(msgs, want, full):
    # The 1st check is shadow()'s own, before its pass; then each of the toy's
    # 8 layers checks at its start, before a GatedDeltaNet scan and after its
    # dense half.
    for at, step in ((2, None), (3, None), (6, None), (12, None), (13, None),
                     (24, 512), (30, 512)):
        llm = engine()
        llm.prefill_step = step
        loop(llm)
        before = list(llm._convs)
        q = next(n for n, _, query in before[-1]["marks"] if query)
        calls = [0]

        def stop():
            calls[0] += 1
            return calls[0] >= at
        info = llm.shadow(msgs, TOOLS, stop=stop)
        assert info["stopped"], info
        assert llm._convs[:len(before)] == tuple(before) and llm._cache is before[-1]["cache"]
        assert stream_fast.PREFILL_STOP is None and stream_fast._AHEAD is None
        assert stream_fast._LAYER_SETS == [None, None] and not stream_fast._IN_PREFILL
        assert stream_fast._PREFILL_AT is None and stream_fast.PREFILL_SKIP is None
        k, done, taken = info["layer"], sum(info["passes"]), []
        if not k and not done:
            assert list(llm._convs) == before, "a shadow stopped in its first layer was kept"
        else:
            sh = llm._convs[-1]
            n = sh["resume"][1].shape[1] if k else 0
            assert len(sh["seen"]) == q + done + n == q + info["kept_ids"], (info, q)
            assert (sh["resume"] or (0,))[0] == k
            assert info["kept_mb"][-1] * 1e6 >= (sh["resume"][1].nbytes if k else 0)
            real = llm._resume

            def spy(*a):
                real(*a)
                taken.append(arrays(llm._cache))
                mx.eval(taken[-1])
            llm._resume = spy
        got, st = ask_next(llm, msgs)
        assert got == want, f"the reply moved: {got} vs {want}"
        if not k and not done:
            assert st["cached_tokens"] == q, (st["cached_tokens"], q)
            print(f"   stopped at check {at}: layer 0, nothing kept; the next message from "
                  f"the task's mark, same reply")
            continue
        assert st["cached_tokens"] == q + info["kept_ids"], (st["cached_tokens"], info)
        assert (st["resumed"] or {}).get("layer", 0) == k and len(taken) == bool(k)
        if step is None:
            assert same(taken[0], full), "the pass taken up differs from the complete shadow"
        gap = fresh_gap(llm)
        assert gap < TOL, gap
        print(f"   stopped at check {at}{f' (passes of {step})' if step else ''}: layer {k}, "
              f"{info['kept_ids']} of {info['ids']} ids kept; the next message ran the rest "
              f"of the pass ({st['resumed']}) and fed {st['prompt_tokens']}, same reply"
              + (", cache == the complete shadow's" if step is None else "")
              + f", vs one prefill {gap:.2g}")
    print("3. a stopped shadow: in its first layer nothing kept; later its finished layers "
          "and passes are, and the next message takes them up")


def check_none(msgs):
    for why, setup in [("keep_convs 1", lambda e: setattr(e, "keep_convs", 1)),
                       ("keep_gb 1 KB", lambda e: setattr(e, "keep_gb", 1e-6)),
                       ("a budget", lambda e: setattr(e, "_budget", object()))]:
        llm = engine()
        loop(llm)
        before = list(llm._convs)
        setup(llm)
        assert llm.shadow(msgs, TOOLS) is None, why
        assert list(llm._convs) == before, why
    plain = engine()
    plain.ask(messages=[{"role": "user", "content": "Say hi."}])
    assert plain.shadow([{"role": "user", "content": "Say hi."},
                         {"role": "assistant", "content": plain.history[-1]["content"]}]) \
        is not None
    print("4. no shadow without room beside the live conversation (keep_convs 1, keep_gb) "
          "or under a budget; a plain chat turn without tools has one (its reply's stub)")


def check_moe_stop():
    """A streamed MoE pass (test_layer_prefill's packed toy, whole layers read
    ahead, groups of 8 experts) stopped at a layer start, after a dense half or
    mid-layer with its read ahead in flight: the prefill state is clean, the
    layer it stopped in as at its start; taken up from that layer (run again
    when it is the first) it gives the uninterrupted pass's logits, cache
    arrays and arenas bit for bit. Then chunkattn's blocks: a stop between
    two."""
    import numpy as np
    import chunkattn
    from mlx_lm.models.cache import make_prompt_cache
    from test_layer_prefill import build, stock

    def run(out, stop_at):
        model = stock(out)
        stream_fast.install(model, out, resident=6, cache_bias=0.0)
        cache, rng = make_prompt_cache(model), np.random.default_rng(3)
        for n in (40, 1, 1, 1):
            mx.eval(model(mx.array(rng.integers(0, 200, n))[None], cache=cache))
        ids = mx.array(rng.integers(0, 200, 24))[None]
        k = None
        if stop_at:
            calls = [0]

            def stop():
                calls[0] += 1
                return calls[0] >= stop_at
            stream_fast.PREFILL_STOP = stop
            try:
                model(ids, cache=cache)
                raise AssertionError("the pass was not stopped")
            except stream_fast.PrefillStopped as e:
                k, h = e.args
            finally:
                stream_fast.PREFILL_STOP = stream_fast._PREFILL_AT = None
            assert stream_fast._AHEAD is None and stream_fast._LAYER_SETS == [None, None]
            assert not stream_fast._IN_PREFILL and not stream_fast._WIDE_SCRATCH
        if k:
            with stream_fast.prefill_skip(cache, k, h):
                y = model(ids, cache=cache)
        else:
            y = model(ids, cache=cache)
        mx.eval(y)
        assert stream_fast.PREFILL_SKIP is None
        arena = [(list(g._lru.items()), g._slot_np.copy()) for g in stream_fast.glus_of(model)]
        return y, arrays(cache), arena, k

    seen = set()
    stream_fast.WIDE_GROUP = 8
    try:
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "q3n"
            build(out)
            for mode, stops in (("whole", (1, 2, 3, 6, 8, 9, 12, 15, 20)), ("auto", (4, 10))):
                stream_fast._RP_MODE = mode
                ref = run(out, 0)
                for at in stops:
                    y, cache, arena, k = run(out, at)
                    seen.add(k)
                    assert mx.array_equal(y, ref[0]).item() and same(cache, ref[1]), (mode, at)
                    for (lru, slots), (lru0, slots0) in zip(arena, ref[2]):
                        assert lru == lru0 and np.array_equal(slots, slots0), (mode, at)
    finally:
        stream_fast._RP_MODE, stream_fast.WIDE_GROUP = "auto", 128
    assert 0 in seen and len(seen) >= 3, seen

    q = mx.random.normal((1, 4, 24, 32)).astype(mx.bfloat16)
    kv = mx.random.normal((1, 2, 40, 32)).astype(mx.bfloat16)
    calls = [0]
    stream_fast.PREFILL_STOP = lambda: calls.__setitem__(0, calls[0] + 1) or calls[0] >= 2
    try:
        chunkattn.chunked(q, kv, kv, 0.1, 8, 5)
        raise AssertionError("chunked() did not stop")
    except stream_fast.PrefillStopped as e:
        assert e.args == (0, None) and calls[0] == 2
    finally:
        stream_fast.PREFILL_STOP = None
    print(f"6. a streamed MoE pass stopped at layer starts, after dense halves and between wide "
          f"groups (the next layer read ahead whole or not), in layers {sorted(seen)}: prefill "
          f"state clean; taken up from that layer == the uninterrupted pass's logits, cache "
          f"arrays and arenas; chunkattn stops between query blocks")


def check_server():
    import server
    host = server.ModelHost.__new__(server.ModelHost)
    host.shadow_on, host.store = True, None
    host._asks, host._turns, host._queue_s, host._shadow_last = 0, 0, None, None
    import threading
    host._lock = threading.Lock()
    queued, ran = [], []
    host.executor = types.SimpleNamespace(submit=lambda fn, *a: queued.append((fn, a)))

    class Fake:
        def shadow(self, messages, tools, stop=None):
            ran.append((messages[-1], stop()))
            return {"ids": 5, "from": 1, "s": 0.0, "passes": [5], "kept_mb": [1.0],
                    "dropped": 0}
    host.llm = Fake()
    call = CALLS[0]
    host._after([{"role": "user", "content": QUERY}], call, TOOLS)
    assert not queued, "a reply with a tool call queued a shadow"
    host._after([{"role": "user", "content": QUERY}], FINAL, TOOLS)
    assert len(queued) == 1
    fn, a = queued.pop()
    t = host._ask()                 # a request comes before the shadow runs
    fn(*a)
    assert not ran, "the shadow ran ahead of a waiting request"
    host._begin(t)
    host._after([{"role": "user", "content": QUERY}], FINAL, None)
    fn, a = queued.pop()
    fn(*a)
    assert ran == [({"role": "assistant", "content": FINAL}, False)], ran
    host._after([{"role": "user", "content": QUERY}], FINAL, None)
    fn, a = queued.pop()
    host._after([{"role": "user", "content": QUERY}], FINAL, None)     # a later turn
    fn(*a)
    assert len(ran) == 1, "an overtaken shadow ran"
    print("5. server: a tool call queues no shadow, a final answer does; a waiting request "
          "or a later turn makes it do nothing")


def main():
    import shutil
    faulthandler.dump_traceback_later(270, exit=True)
    if not support.has_tokenizer():
        print("SKIPPED (no tokenizer: set MOE_TOY_TOKENIZER)")
        return
    pcache.FRONTS = Path(tempfile.mkdtemp(prefix="moe-shadow-test-"))
    real = moe_stream.forced_rows.make
    moe_stream.forced_rows.make = lambda *a, **kw: None
    try:
        check_moe_stop()
        cpu()
        msgs, want, full = check_shadow()
        check_stopped(msgs, want, full)
        check_none(msgs)
        check_server()
    finally:
        moe_stream.forced_rows.make = real
        shutil.rmtree(pcache.FRONTS, ignore_errors=True)
    print("test_shadow: ok")


if __name__ == "__main__":
    main()
