#!/usr/bin/env python3
"""`--draft auto` (draft_find.py and the auto-draft arm in moe_stream), toy.

    python3 tests/test_draft_auto.py

1. discovery on a fake models tree -- directories right under ~/.cache and
   Hugging Face snapshots of symlinks into blobs, no real weights: the
   smallest model with the target's tokenizer and quantization wins; the
   same tokenizer re-saved by another `tokenizers` (merges as pairs, other
   flags) counts as the same; a tokenizer with the same vocab_size and one
   other added token, a model as big as the target, a streamed pack, a
   model_type mlx_lm lacks and a pair over the memory budget do not; another
   quantization is taken only when nothing quantized like the target is
   there; a tie in bytes goes to ~/.cache.
2. the scan cache: a second start reads no tokenizer, a directory whose mtime
   moved is read again alone, a directory that is gone leaves the cache.
3. what the engine asks for: chat.py's flags (auto by default, --no-draft and
   --draft none off, a path as given, k=2 only for auto), and
   StreamingLLM.__init__ with the model load stubbed out -- a dense target
   looks and loads what it found, a streamed one does not look, an explicit
   draft is loaded without looking, a find that comes back empty or a draft
   that does not load leaves an engine with no draft.
4. the policy on a fake clock, through the engine on toy weights and the
   real tokenizer: a draft that costs more than it saves is switched off
   after the warm-up and one that pays is kept, an explicit draft runs every
   turn, and every turn's reply is the no-draft engine's token for token.
"""

import faulthandler
import json
import os
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

faulthandler.dump_traceback_later(270, exit=True)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import draft_find                                              # noqa: E402
import support                                                 # noqa: E402

Q4 = {"group_size": 64, "bits": 4}


def tokenizer(added="<|im_end|>", pairs=False, trim=False):
    """A tokenizer.json shaped like Qwen3's: BPE vocabulary, merges, added
    tokens. `pairs`/`trim` are what another `tokenizers` writes for the
    same tokenizer."""
    vocab = {t: i for i, t in enumerate(["a", "b", "c", "ab", "abc", "Ġ"])}
    merges = ["a b", "ab c"]
    return {"version": "1.0",
            "added_tokens": [{"id": 6, "content": "<|endoftext|>", "special": True},
                             {"id": 7, "content": added, "special": True}],
            "pre_tokenizer": {"type": "ByteLevel", "trim_offsets": trim},
            "model": {"type": "BPE", "vocab": vocab,
                      "merges": [m.split(" ") for m in merges] if pairs else merges}}


def model(d, nbytes, quant=Q4, tok=None, model_type="qwen3", vocab=8, pack=False):
    """A model directory draft_find can judge: config, tokenizer, weights of
    `nbytes` bytes (zeros -- only their size is read)."""
    d.mkdir(parents=True)
    cfg = {"model_type": model_type, "vocab_size": vocab}
    if quant:
        cfg["quantization"] = quant
    (d / "config.json").write_text(json.dumps(cfg))
    (d / "tokenizer.json").write_text(json.dumps(tok or tokenizer()))
    (d / "model.safetensors").write_bytes(bytes(nbytes))
    if pack:
        (d / "experts_index.json").write_text("{}")
    return d


def snapshot(hub, repo, nbytes, **kw):
    """A hub snapshot the way huggingface_hub lays one out: the files are
    symlinks into blobs/."""
    src = model(hub / "src" / repo, nbytes, **kw)
    snap = hub / f"models--{repo}" / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    blobs = hub / f"models--{repo}" / "blobs"
    blobs.mkdir()
    for f in src.iterdir():
        shutil.move(str(f), blobs / f.name)
        (snap / f.name).symlink_to(Path("..") / ".." / "blobs" / f.name)
    return snap


@contextmanager
def home(tmp):
    """Path.home() and the hub cache under `tmp`, as model_dirs() reads them."""
    import huggingface_hub.constants as hc
    saved = os.environ.get("HOME"), hc.HF_HUB_CACHE
    os.environ["HOME"] = str(tmp)
    hc.HF_HUB_CACHE = str(tmp / ".cache" / "huggingface" / "hub")
    try:
        yield tmp / ".cache", Path(hc.HF_HUB_CACHE)
    finally:
        os.environ["HOME"] = saved[0]
        hc.HF_HUB_CACHE = saved[1]


def test_discovery():
    tmp = Path(tempfile.mkdtemp())
    cache = tmp / "scan.json"
    try:
        with home(tmp) as (dot, hub):
            target = model(dot / "qwen3-4b", 1000)
            model(dot / "qwen3-draft", 100)
            model(dot / "qwen3-draft17", 300)
            model(dot / "qwen3-draft-2bit", 60, quant={"group_size": 64, "bits": 2},
                  tok=tokenizer(pairs=True, trim=True))
            model(dot / "qwen25-05b", 50, tok=tokenizer(added="<|im_end2|>"))
            model(dot / "qwen3-4b-copy", 1000)
            model(dot / "packed-moe", 10, pack=True)
            model(dot / "asr", 10, model_type="qwen3_asr_nope")
            (dot / "uv").mkdir()                     # not a model at all
            # files not shaped as expected: skipped, never a crash
            model(dot / "odd-config", 10)
            (dot / "odd-config" / "config.json").write_text("[]")
            bad = tokenizer()
            del bad["added_tokens"][0]["content"]
            model(dot / "odd-tokenizer", 10, tok=bad)
            snapshot(hub, "mlx-community--Qwen3-0.6B-4bit", 100)

            def pick(**kw):
                got = draft_find.find(target, cache=cache, **kw)
                return got and (got[0].name if got[0].name != "abc123"
                                else got[0].parent.parent.name, got[1])

            assert pick() == ("qwen3-draft", 100), pick()
            print(f"  smallest same tokenizer + quantization: {pick()}; not the "
                  f"2-bit copy (60), other added token (50), pack, unknown type, "
                  f"malformed config or tokenizer  OK")
            assert pick(budget=1000 + 59) is None
            assert pick(budget=1000 + 99) == ("qwen3-draft-2bit", 60), pick(budget=1099)
            assert pick(budget=1000 + 100) == ("qwen3-draft", 100)
            print("  memory budget: a pair over it is no candidate (the 2-bit copy "
                  "when only it fits, none when nothing does)  OK")

            shutil.rmtree(dot / "qwen3-draft")
            assert pick() == ("models--mlx-community--Qwen3-0.6B-4bit", 100), pick()
            print(f"  a hub snapshot of symlinks is found: {pick()}  OK")
            shutil.rmtree(hub / "models--mlx-community--Qwen3-0.6B-4bit")
            assert pick() == ("qwen3-draft17", 300), pick()
            shutil.rmtree(dot / "qwen3-draft17")
            assert pick() == ("qwen3-draft-2bit", 60), pick()
            print("  other quantization only when none like the target's; its "
                  "tokenizer re-saved with merges as pairs still matches  OK")
            shutil.rmtree(dot / "qwen3-draft-2bit")
            assert pick() is None, pick()
            print("  same vocab_size, other added token: not a draft  OK")

            # a tie in bytes: ~/.cache before the hub
            model(dot / "zz-draft", 100)
            snapshot(hub, "aa--draft", 100)
            assert pick() == ("zz-draft", 100), pick()
            print("  tie in bytes: ~/.cache before the hub cache  OK")
            assert draft_find.find(dot / "uv", cache=cache) is None
    finally:
        shutil.rmtree(tmp)


def test_scan_cache():
    tmp = Path(tempfile.mkdtemp())
    cache = tmp / "scan.json"
    calls = []
    real = draft_find._info
    draft_find._info = lambda d: calls.append(d.name) or real(d)
    try:
        with home(tmp) as (dot, hub):
            target = model(dot / "t", 1000)
            model(dot / "d1", 100)
            model(dot / "d2", 200)
            assert draft_find.find(target, cache=cache)[0].name == "d1"
            first = len(calls)
            assert first >= 3, calls
            calls.clear()
            t0 = time.perf_counter()
            assert draft_find.find(target, cache=cache)[0].name == "d1"
            warm = time.perf_counter() - t0
            assert calls == [], calls
            print(f"  scan cache: {first} directories read cold, 0 warm "
                  f"({warm * 1e3:.1f} ms)  OK")
            st = (dot / "d2").stat()
            os.utime(dot / "d2", ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
            assert draft_find.find(target, cache=cache)[0].name == "d1"
            assert calls == ["d2"], calls
            shutil.rmtree(dot / "d1")
            calls.clear()
            assert draft_find.find(target, cache=cache)[0].name == "d2"
            assert str(dot / "d1") not in json.loads(cache.read_text())
            print("  scan cache: a moved mtime re-reads that directory alone, a "
                  "removed one leaves the cache  OK")
    finally:
        draft_find._info = real
        shutil.rmtree(tmp)


# ------------------------------------------------------------------ engine

def test_chat_flags():
    sys.modules.setdefault("sklearn", None)
    import chat
    seen = []
    real = chat.StreamingLLM

    class Recorder:
        STUB_MODES = real.STUB_MODES        # build_parser reads it
        model = None                        # no model, so chat.py attaches no arena governor

        def __init__(self, **kw):
            seen.append(kw)

    chat.StreamingLLM = Recorder
    try:
        for argv, want in (([], ("auto", None)), (["--no-draft"], (None, None)),
                           (["--draft", "none"], (None, None)),
                           (["--draft", "/m/d"], ("/m/d", None)),
                           (["--draft-tokens", "4"], ("auto", 4)),
                           (["--mtp", "/m/head"], ("auto", None))):
            chat.build_llm(chat.build_parser().parse_args(["--model", "/m/model"] + argv))
            got = (seen[-1]["draft"], seen[-1]["draft_tokens"])
            assert got == want, (argv, got)
        for argv in (["--no-draft", "--draft-tokens", "2"],
                     ["--draft", "/m/d", "--mtp", "/m/head"]):
            try:
                chat.build_llm(chat.build_parser().parse_args(["--model", "/m/model"] + argv))
            except SystemExit:
                continue
            raise AssertionError(f"{argv} was not refused")
    finally:
        chat.StreamingLLM = real
    print("  chat.py: --draft auto by default, --no-draft/--draft none off, a "
          "path as given, auto beside --mtp, k left to the engine  OK")


@contextmanager
def stubbed_engine(streamed, found, load_fails=False):
    """StreamingLLM.__init__ with everything that loads or calibrates a real
    model stubbed out: the target is a toy, `streamed` says whether install()
    found experts to stream, draft_find.find returns `found`, and the draft
    load records what it was asked for instead of loading."""
    import mlx_lm
    import moe_stream
    import chunkattn
    import foldattn
    import rowqmm
    tok = mlx_lm.utils.load_tokenizer(support.tokenizer_dir())
    log = {"find": [], "load": []}

    def load_draft(self, path, draft_path, window):
        log["load"].append(Path(draft_path).name)
        if load_fails:
            raise OSError("truncated safetensors")
        self._draft_model = support.toy_model(seed=1)
        self.draft_path = str(draft_path)
        return 8

    saved = (moe_stream.load_slim_aware, moe_stream.install,
             moe_stream.stream_fast.install_embed, rowqmm.install,
             foldattn.install, chunkattn.install, draft_find.find,
             moe_stream.StreamingLLM._load_draft)
    moe_stream.load_slim_aware = lambda p, s: (support.toy_model(seed=0), tok, None)
    moe_stream.install = lambda *a, **k: (3 if streamed else 0, None)
    moe_stream.stream_fast.install_embed = lambda m, st: None
    rowqmm.install = lambda m: {}
    foldattn.install = lambda m: None
    chunkattn.install = lambda m: None
    def find(p, budget=None):
        log["find"].append(budget)
        if found == "raise":
            raise KeyError("content")
        return found
    draft_find.find = find
    moe_stream.StreamingLLM._load_draft = load_draft
    try:
        yield moe_stream.StreamingLLM, log
    finally:
        (moe_stream.load_slim_aware, moe_stream.install,
         moe_stream.stream_fast.install_embed, rowqmm.install,
         foldattn.install, chunkattn.install, draft_find.find,
         moe_stream.StreamingLLM._load_draft) = saved


def test_engine_init():
    import moe_stream
    if not support.has_tokenizer():
        print("  engine init: SKIPPED (no tokenizer)")
        return
    tmp = Path(tempfile.mkdtemp())
    try:
        target = model(tmp / "target", 1000)
        drafted = model(tmp / "explicit-draft", 100)
        found = (tmp / "found-draft", 100)
        cases = (  # draft, streamed, find returns, load fails -> want
            ("auto", False, found, False, ["budget"], ["found-draft"], True, 2),
            ("auto", True, found, False, [], [], False, 2),
            (None, False, found, False, [], [], False, 3),
            (str(drafted), False, found, False, [], ["explicit-draft"], False, 3),
            ("auto", False, None, False, ["budget"], [], False, 2),
            ("auto", False, found, True, ["budget"], ["found-draft"], False, 2),
            ("auto", False, "raise", False, ["budget"], [], False, 2))
        for draft, streamed, ret, fails, want_find, want_load, auto, k in cases:
            with stubbed_engine(streamed, ret, fails) as (cls, log):
                llm = cls(model=str(target), draft=draft, adaptive=False)
            label = (draft and Path(draft).name, streamed, ret and str(ret)[-24:], fails)
            assert ["budget" for b in log["find"] if b and b > 0] == want_find, (label, log)
            assert log["load"] == want_load, (label, log)
            assert llm._draft_auto is auto and llm.draft_tokens == k, (label, llm.draft_tokens)
            has = bool(want_load) and not fails
            assert (llm._draft_model is not None) == has, label
            assert (llm.draft_path is not None) == has, label
        print("  engine init: dense + auto looks and loads at k=2, streamed does "
              "not look, --no-draft neither, an explicit draft loads without "
              "looking at k=3, nothing found, a failed load or a failed scan "
              "-> no draft  OK")
    finally:
        shutil.rmtree(tmp)


class Clock:
    """Time as the engine reads it: advanced only by model calls, each
    costing what its model is charged, and by a nanosecond per read so two
    reads never tie."""

    def __init__(self):
        self.t = 0.0

    def perf_counter(self):
        self.t += 1e-9
        return self.t

    def charge(self, model, ms):
        cls, clock = type(model), self

        def __call__(self, *a, **k):
            clock.t += ms / 1e3
            return cls.__call__(self, *a, **k)
        model.__class__ = type(cls.__name__, (cls,), {"__call__": __call__})


def engine(target, draft, tok, auto):
    """spec_draft.engine_selftest's toy engine; `auto` makes its draft an arm."""
    import moe_stream
    from mlx_lm.models.cache import make_prompt_cache
    llm = moe_stream.StreamingLLM.__new__(moe_stream.StreamingLLM)
    llm.model, llm.tokenizer = target, tok
    llm.max_tokens, llm.prefill_step, llm.resident = 16, None, 0
    llm.thinking, llm.profile = False, None
    llm.model_path = support.MODEL_LABEL
    llm._setup_chat("slice")
    llm._draft_model, llm._draft_seen = draft, []
    llm._draft_cache = make_prompt_cache(draft) if draft is not None else None
    llm.draft_tokens, llm.draft_tau = 2, None
    llm._draft_auto = auto
    llm.reset()
    return llm


QS = ["Say the word apple.", "Name three rivers.", "Count to five.",
      "What is MoE routing?", "Once more.", "And a fourth river?",
      "Why is the sky blue?", "Write hello world in C.", "Thank you!", "One more."]


def test_policy():
    import moe_stream
    import spec_draft
    from mlx_lm.utils import load_tokenizer
    if not support.has_tokenizer():
        print("  auto-draft policy: SKIPPED (no tokenizer)")
        return
    tok = load_tokenizer(support.tokenizer_dir())
    vocab = max([tok.vocab_size] + list(tok.get_added_vocab().values())) + 1
    # The draft is the target's own weights: every chain is accepted, so a
    # cheap draft surely pays, and the cost alone decides.
    target, draft, _ = spec_draft._drafts(vocab, 32, 3, 3)
    ref = engine(target, None, tok, False)
    want = [ref.ask(q) for q in QS]

    real_time = moe_stream.time
    try:
        for name, draft_ms, best in (("slower", 40.0, "plain"), ("faster", 0.1, "draft")):
            clock = Clock()
            moe_stream.time = clock
            target, draft, _ = spec_draft._drafts(vocab, 32, 3, 3)
            clock.charge(target, 10.0)
            clock.charge(draft, draft_ms)
            llm = engine(target, draft, tok, True)
            got, arms = [], []
            for q in QS:
                got.append(llm.ask(q))
                st = llm.stats()
                # A released draft leaves no policy: the turn is plain decode.
                arms.append(st.get("decode_arm", "plain"))
                assert ("draft_acceptance" in st) == (arms[-1] == "draft"), st
            assert got == want, f"{name}: a reply differs from the no-draft engine"
            assert arms[:4] == ["draft", "plain"] * 2, arms
            assert arms[4:] == [best] * len(arms[4:]), arms
            if best == "plain":
                # Outclassed on the fake clock (no spread): freed after warm-up.
                assert llm._draft_model is None and not llm._draft_auto
                assert llm.decode_policy_free is None
            else:
                assert llm._draft_model is not None and llm.decode_policy_free.best == "draft"
            print(f"  policy, draft {name} on the fake clock: arms {arms}, "
                  f"draft {'released' if llm._draft_model is None else 'kept'}, "
                  f"replies == no-draft engine  OK")

        clock = Clock()
        moe_stream.time = clock
        target, draft, _ = spec_draft._drafts(vocab, 32, 3, 3)
        clock.charge(target, 10.0)
        clock.charge(draft, 40.0)
        llm = engine(target, draft, tok, False)
        for q, w in zip(QS[:4], want):
            assert llm.ask(q) == w
            st = llm.stats()
            assert "decode_arm" not in st and "draft_acceptance" in st, st
        assert llm.decode_policy_free is None
        print("  explicit draft: every turn drafts, no policy, even when slower  OK")
    finally:
        moe_stream.time = real_time


if __name__ == "__main__":
    test_discovery()
    test_scan_cache()
    test_chat_flags()
    test_engine_init()
    test_policy()
    print("ok")
