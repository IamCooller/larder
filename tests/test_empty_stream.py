#!/usr/bin/env python3
"""A turn whose decode yields no token still ends the turn.

The epilogue reads the last response to say how the turn ended, and a generator
that yields nothing never binds one; the turn must still come back empty, keep
its stats and leave the engine usable.

    python3 tests/test_empty_stream.py
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

import spec_draft                                              # noqa: E402
from support import has_tokenizer, toy_llm                     # noqa: E402

ASK = "Say the word apple."


def test_empty():
    if not has_tokenizer():
        print("  empty stream: SKIPPED (no tokenizer)")
        return
    llm = toy_llm("slice", max_tokens=8)
    real = spec_draft.plain

    def nothing(*a, **kw):
        return iter(())

    spec_draft.plain = nothing
    try:
        out = llm.ask(ASK)
    finally:
        spec_draft.plain = real
    st = llm.stats()
    assert out == "", repr(out)
    assert st == {}, st
    assert llm._ended_cleanly is False, llm._ended_cleanly
    assert llm.history[-1] == {"role": "assistant", "content": ""}, llm.history
    print(f"  empty stream: reply '', no stats, ended_cleanly "
          f"{llm._ended_cleanly}  OK")

    full = llm.ask(ASK)
    assert full and llm.stats()["tokens"] > 0, llm.stats()
    print(f"  the next turn decodes: {len(full)} characters, "
          f"{llm.stats()['tokens']} tokens  OK")


if __name__ == "__main__":
    test_empty()
    print("test_empty_stream: ok")