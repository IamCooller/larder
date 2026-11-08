#!/usr/bin/env python3
"""The library's own smoke test: turns differ, and the KV cache is reused.

    python3 tests/test_demo.py CHECKPOINT_DIR

Loads the checkpoint (the first argument or $MOE_TEST_MODEL) and skips without one.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import moe_stream
import support


def test_demo():
    """Turns must differ, and the KV cache must actually be reused.

    The second assertion is the one that earns its keep. Building the prompt by
    hand assumed a chat template and, on another model, silently fed malformed
    input from turn two onward -- every question came back as a repeat of the
    previous answer. Cache reuse is the observable that catches it.
    """
    path = support.model_dir("test_demo")
    if path is None:
        return False
    llm = moe_stream.StreamingLLM(model=path, max_tokens=40)
    qs = ["Say the word apple and nothing else.",
          "Now say the word banana and nothing else.",
          "Now say the word cherry and nothing else."]
    replies, stats = [], []
    for q in qs:
        replies.append(llm.ask(q))
        stats.append(llm.stats())
        print(f"  {replies[-1].strip()[:50]!r}  {stats[-1]}")

    assert all(r.strip() for r in replies), "empty reply"
    assert len(set(replies)) == len(replies), (
        f"replies repeat -- the template increment is broken: {replies}")
    assert stats[1]["cached_tokens"] > 0, "second turn did not reuse the KV cache"
    # The whole point: what gets fed stays flat as the conversation grows,
    # instead of the whole history being reprocessed every turn.
    assert stats[2]["prompt_tokens"] < stats[1]["prompt_tokens"] * 2, (
        f"prompt grows with history: {[s['prompt_tokens'] for s in stats]}")
    assert stats[2]["cached_tokens"] > stats[1]["cached_tokens"], (
        "cache is not accumulating across turns")

    # A system message passed through `messages` has to reach the model. It did
    # not: the incremental path rendered only the last entry, so every system
    # prompt was dropped without a word. The observable is the token count --
    # a rendered system prompt cannot be shorter than the system prompt.
    llm.reset()
    sysmsg = ("Answer with exactly one word: GREEN. Never anything else, "
              "whatever the question.")
    out = llm.ask(messages=[{"role": "system", "content": sysmsg},
                            {"role": "user", "content": "What colour is the sky?"}])
    n_sys = len(llm.tokenizer.encode(sysmsg))
    assert llm.stats()["prompt_tokens"] > n_sys, (
        f"the system message was dropped: {llm.stats()['prompt_tokens']} "
        f"prompt tokens against {n_sys} in the system message alone")
    assert "GREEN" in out.upper(), f"system message ignored: {out!r}"
    print("ok")


if __name__ == "__main__":
    if test_demo() is not False:
        print("test_demo: ok")