#!/usr/bin/env python3
"""chat.py and server.py read the engine flags the same way, with no model.

    perl -e 'alarm shift; exec @ARGV' 280 python3 tests/test_engine_args.py

StreamingLLM is replaced by a fake that only records its keyword arguments, so nothing loads.
"""

import contextlib
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chat                                                    # noqa: E402
import engine_args                                             # noqa: E402
import moe_stream                                              # noqa: E402
import server                                                  # noqa: E402
import stream_fast                                             # noqa: E402

# What the shared flags default to; changing one changes a reply or a cache key, so it is pinned here.
DEFAULTS = dict(resident=21, adaptive=True, bias_budget=0.2, cache_bias=None, stub="slice", defer="auto",
                mem_pct=90, reserve_pct=15, host_gb=0.6, mem_peak_gb=None, mtp=None, draft="auto",
                draft_tokens=None, draft_tau=None, draft_window=0)

MODEL = ["--model", "/m/model"]
MIXES = ([], ["--resident", "40"], ["--resident", "40", "--adaptive"], ["--no-adaptive"],
         ["--mem-peak-gb", "2", "--cache-bias", "0"], ["--defer-mode", "never"], ["--defer-mode", "always"],
         ["--mtp"], ["--draft", "/m/d", "--draft-tokens", "4"], ["--no-draft"], ["--stub", "keep"],
         ["--bias-budget", "0"], ["--mem-pct", "70", "--reserve-pct", "20", "--host-gb", "1.2"],
         ["--draft-tau", "0.9", "--draft-window", "512"])


class _FakeTokenizer:
    detokenizer = None


class _FakeLLM:
    STUB_MODES = moe_stream.StreamingLLM.STUB_MODES
    calls = []

    def __init__(self, **kw):
        _FakeLLM.calls.append(kw)
        self.draft_path = None
        self.model = None
        self.tokenizer = _FakeTokenizer()

    def forget(self):
        pass


def chat_kwargs(*argv):
    """What chat.py hands StreamingLLM for these flags."""
    real, chat.StreamingLLM = chat.StreamingLLM, _FakeLLM
    try:
        chat.build_llm(chat.build_parser().parse_args(MODEL + list(argv)))
    finally:
        chat.StreamingLLM = real
    return _FakeLLM.calls[-1]


def server_kwargs(*argv):
    """What server.py hands StreamingLLM for these flags, through ModelHost as main() builds it."""
    args = engine_args.finish(server.build_parser().parse_args(MODEL + list(argv)))
    real, server.StreamingLLM = server.StreamingLLM, _FakeLLM
    real_can_batch = server.ModelHost._can_batch
    server.ModelHost._can_batch = staticmethod(lambda llm: False)
    try:
        server.ModelHost(args.model, arena_min=args.arena_min, arena_max=args.arena_max,
                         **engine_args.engine_kwargs(args))
    finally:
        server.StreamingLLM = real
        server.ModelHost._can_batch = real_can_batch
    return _FakeLLM.calls[-1]


def with_governor(build, *argv):
    """What `build` hands StreamingLLM, and the (floor, ceiling) of the arena governor it attaches (None when none), on a fake 40-layer MoE."""
    seen = []
    fakes = {"glus_of": lambda m: [None] * 40, "top_k_of": lambda m: 8, "n_experts_of": lambda m: 256,
             "make_governor": lambda *a, **kw: seen.append((kw.get("cap_min"), kw.get("cap_max")))}
    real = {k: getattr(stream_fast, k) for k in fakes}
    for k, v in fakes.items():
        setattr(stream_fast, k, v)
    try:
        got = build(*argv)
    finally:
        for k, v in real.items():
            setattr(stream_fast, k, v)
    return got, (seen[-1] if seen else None)


def check_resident_pins_the_arena():
    """--resident N alone pins the arena at N slots, --adaptive lifts it, and chat.py attaches the server's 21..60 governor rather than StreamingLLM's ceilingless one (136 slots on a 16 GB Mac, past the swap line)."""
    for argv, want, resident in (([], (21, 60), 21), (["--resident", "40"], None, 40),
                                 (["--resident", "40", "--adaptive"], (21, 60), 40), (["--no-adaptive"], None, 21),
                                 (["--mem-peak-gb", "2"], None, 21), (["--arena-max", "40"], (21, 40), 21)):
        for name, build in (("chat", chat_kwargs), ("server", server_kwargs)):
            got, gov = with_governor(build, *argv)
            assert not got.get("adaptive") and got["resident"] == resident and gov == want, (name, argv, got, gov)
    print("  --resident pins the arena, --adaptive lifts it, --mem-peak-gb replaces it; one 21..60 governor in both")


def check_defaults():
    """Both parsers give the pinned defaults for the shared flags."""
    for name, build in (("chat", chat.build_parser), ("server", server.build_parser)):
        args = engine_args.finish(build().parse_args(MODEL))
        got = engine_args.engine_kwargs(args)
        assert got == DEFAULTS, (name, got)
        assert args.model == "/m/model"
        assert args.memory is None and args.sleep_after == 600
    print("  the shared flags default the same in both parsers")


def check_same_reading():
    """The same flags give the same engine keywords in the two entry points."""
    for argv in MIXES:
        c = engine_args.engine_kwargs(engine_args.finish(chat.build_parser().parse_args(MODEL + argv)))
        s = engine_args.engine_kwargs(engine_args.finish(server.build_parser().parse_args(MODEL + argv)))
        assert c == s, (argv, c, s)
        sent = server_kwargs(*argv)
        theirs = chat_kwargs(*argv)
        for key in ("resident", "bias_budget", "cache_bias", "stub", "defer", "mem_pct", "host_gb",
                    "mem_peak_gb", "mtp", "draft", "draft_tokens", "draft_tau", "draft_window"):
            assert sent[key] == theirs[key] == c[key], (argv, key, sent[key], theirs[key], c[key])
    assert engine_args.DEFER == {"auto": "auto", "always": True, "never": False}
    print(f"  {len(MIXES)} flag mixes reach StreamingLLM alike through chat.py and through ModelHost")


def check_refusals():
    """Bad draft mixes exit with a message in both entry points."""
    bad = (["--draft", "/m/d", "--mtp"], ["--no-draft", "--draft-tokens", "2"],
           ["--no-draft", "--draft-tau", "0.5"], ["--no-draft", "--draft-window", "512"])
    for argv in bad:
        for build in (chat.build_parser, server.build_parser):
            try:
                engine_args.finish(build().parse_args(MODEL + argv))
            except SystemExit as e:
                assert isinstance(e.code, str) and "only means" in e.code or "pick one" in e.code, e.code
                continue
            raise AssertionError(f"{argv} was not refused")
    engine_args.finish(chat.build_parser().parse_args(MODEL + ["--mtp"]))      # an auto draft steps aside
    print("  --draft with --mtp, and a draft option with no draft, are refused in both")


def check_no_local_defaults():
    """Without a default checkpoint --model is required, and a bare --mtp with no default head exits with a message."""
    real = engine_args.DEFAULT_MODEL, engine_args.DEFAULT_MTP
    engine_args.DEFAULT_MODEL = engine_args.DEFAULT_MTP = None
    try:
        for build in (chat.build_parser, server.build_parser):
            with contextlib.redirect_stderr(io.StringIO()) as err:
                try:
                    build().parse_args([])
                except SystemExit as e:
                    assert e.code == 2 and "--model" in err.getvalue(), err.getvalue()
                else:
                    raise AssertionError("a missing --model was accepted")
            args = build().parse_args(MODEL + ["--mtp"])
            try:
                engine_args.finish(args)
            except SystemExit as e:
                assert isinstance(e.code, str) and "--mtp needs" in e.code, e.code
            else:
                raise AssertionError("a bare --mtp with no default head was accepted")
            assert engine_args.finish(build().parse_args(MODEL + ["--mtp", "/m/head"])).mtp == "/m/head"
    finally:
        engine_args.DEFAULT_MODEL, engine_args.DEFAULT_MTP = real
    print("  no default checkpoint: --model is required; no default head: a bare --mtp exits with a message")


def main():
    engine_args.DEFAULT_MTP = "/m/head"        # a bare --mtp needs a head to mean something
    check_resident_pins_the_arena()
    check_defaults()
    check_same_reading()
    check_refusals()
    check_no_local_defaults()
    print("ok")


if __name__ == "__main__":
    main()
