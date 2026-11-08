#!/usr/bin/env python3
"""ModelHost forwards mem_pct/host_gb into StreamingLLM (no real model, no
GPU): server.py --mem-pct used to reach only ModelHost's own governor path
(self.mem_pct, read when --adaptive) because ModelHost.__init__ consumed
mem_pct/host_gb as its own named parameters and never re-added them to the
**kwargs handed to StreamingLLM(), so a server run without --adaptive silently always got
StreamingLLM's own mem_pct=90 default in set_prefill_budget_gb, which the prefill-peak levers of
chunkattn and stream_fast size against, instead of whatever --mem-pct asked for.

    python3 tests/test_modelhost_mem_pct.py

StreamingLLM itself is replaced with a fake that only records its kwargs;
ModelHost's constructor validates nothing about the model path before
reaching it, so this never touches mx, a real checkpoint, or the GPU.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import server                                                  # noqa: E402


class _FakeTokenizer:
    detokenizer = None


class _FakeStreamingLLM:
    """Only records its own kwargs -- ModelHost's _load also reads
    llm.model (via _can_batch) and llm.tokenizer.detokenizer, neither of
    which this check is about, so both are stubbed rather than built for
    real."""
    calls = []

    def __init__(self, **kwargs):
        _FakeStreamingLLM.calls.append(kwargs)
        self.model_path = kwargs.get("model")
        self.model = None
        self.tokenizer = _FakeTokenizer()
        self.forgot = False

    def forget(self):
        self.forgot = True


def check_construct_forwards():
    real = server.StreamingLLM
    real_can_batch = server.ModelHost._can_batch
    server.StreamingLLM = _FakeStreamingLLM
    server.ModelHost._can_batch = staticmethod(lambda llm: False)   # llm.model is a stub
    _FakeStreamingLLM.calls.clear()
    try:
        host = server.ModelHost("fake-model", mem_pct=42, host_gb=1.5)
        assert len(_FakeStreamingLLM.calls) == 1, _FakeStreamingLLM.calls
        got = _FakeStreamingLLM.calls[0]
        assert got.get("mem_pct") == 42, (
            "ModelHost's own --mem-pct never reached StreamingLLM -- its "
            "set_prefill_budget_gb call would size against the 90 default "
            "instead", got)
        assert got.get("host_gb") == 1.5, got
        assert host.mem_pct == 42 and host.host_gb == 1.5
        print("construct: mem_pct=42, host_gb=1.5 reach StreamingLLM's own kwargs")
    finally:
        server.StreamingLLM = real
        server.ModelHost._can_batch = real_can_batch


def check_swap_forwards():
    """swap() reloads a checkpoint under the SAME mem_pct/host_gb this host
    was started with (self.mem_pct), not a fresh default -- it is a memory
    POLICY, not a per-checkpoint one."""
    real = server.StreamingLLM
    real_can_batch = server.ModelHost._can_batch
    server.StreamingLLM = _FakeStreamingLLM
    server.ModelHost._can_batch = staticmethod(lambda llm: False)
    _FakeStreamingLLM.calls.clear()
    try:
        host = server.ModelHost("fake-model", mem_pct=17, host_gb=0.9,
                                mem_peak_gb=2.0, keep_convs=3, keep_gb=0.5,
                                prefill_gb=5.5)
        old = host.llm
        host.swap("fake-model-2")
        assert len(_FakeStreamingLLM.calls) == 2, _FakeStreamingLLM.calls
        got = _FakeStreamingLLM.calls[1]
        assert got.get("mem_pct") == 17 and got.get("host_gb") == 0.9, got
        assert all(c.get("mem_peak_gb") == 2.0 and c.get("prefill_gb") == 5.5
                   for c in _FakeStreamingLLM.calls), _FakeStreamingLLM.calls
        # Engine attributes rather than constructor kwargs, like reuse.
        assert (host.llm.keep_convs, host.llm.keep_gb) == (3, 0.5)
        assert old.forgot and not host.llm.forgot, "kept conversations held over a swap"
        print("swap: reload keeps the host's own mem_pct=17, host_gb=0.9, "
              "mem_peak_gb=2.0, prefill_gb=5.5, keep_convs=3, keep_gb=0.5; the old model's "
              "kept conversations dropped before the new one loads")
    finally:
        server.StreamingLLM = real
        server.ModelHost._can_batch = real_can_batch


if __name__ == "__main__":
    check_construct_forwards()
    check_swap_forwards()
    print("ok")
