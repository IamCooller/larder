#!/usr/bin/env python3
"""The default start-up output of server.py and chat.py, with no model.

    perl -e 'alarm shift; exec @ARGV' 280 python3 tests/test_quiet_start.py

Checked: the engine's notes print only under --verbose; the cache-bias warning prints for a hand-typed
--mem-peak-gb with --cache-bias and not when the launcher says the bias passed its gate; the f16 GEMM
line is not printed on CUDA; and server.py's default output is the model, the memory, where the API and
the chat are, and any warning, with StreamingLLM replaced by a fake.
"""

import contextlib
import io
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chat                                                    # noqa: E402
import engine_args                                             # noqa: E402
import engine_log                                              # noqa: E402
import exl3_experts                                            # noqa: E402
import machine                                                 # noqa: E402
import moe_stream                                              # noqa: E402
import server                                                  # noqa: E402


def captured(fn, *args, **kwargs):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        try:
            fn(*args, **kwargs)
        except BaseException as e:      # noqa: BLE001 - the caller asserts on the text, not the exit
            out.write(f"[{type(e).__name__}]\n")
    return out.getvalue()


def check_notes_follow_verbose():
    """note() prints only while verbose is on, and --verbose is what turns it on."""
    try:
        engine_log.set_verbose(True)
        assert "hello" in captured(engine_log.note, "hello")
        engine_log.set_verbose(False)
        assert captured(engine_log.note, "hello") == ""
        for build in (chat.build_parser, server.build_parser):
            engine_args.finish(build().parse_args(["--model", "/m/model"]))
            assert engine_log.verbose is False
            engine_args.finish(build().parse_args(["--model", "/m/model", "--verbose"]))
            assert engine_log.verbose is True
        print("  notes are silent by default and come back with --verbose, in both entry points")
    finally:
        engine_log.set_verbose(True)


def check_bias_warning():
    """A hand-typed --mem-peak-gb with --cache-bias warns; the launcher's gated setup does not."""
    def build(**kw):
        return captured(moe_stream.StreamingLLM, model="/nonexistent/model", **kw)   # raises before any load

    key = "ENGINE_BIAS_GATED"
    before = os.environ.pop(key, None)
    try:
        assert "WARNING: --mem-peak-gb with --cache-bias 0.5" in build(mem_peak_gb=2.0, cache_bias=0.5)
        assert "WARNING" not in build(mem_peak_gb=2.0, cache_bias=0.0)
        assert "WARNING" not in build(mem_peak_gb=2.0)
        os.environ[key] = "1"
        assert "WARNING" not in build(mem_peak_gb=2.0, cache_bias=0.5)
    finally:
        os.environ.pop(key, None)
        if before is not None:
            os.environ[key] = before
    gated = dict(kind="moe", dense_bytes=2.7e9, read_bytes=2.7e9, moe_layers=48, experts=512, top_k=10,
                 expert_bytes=1862400, recipe="store", gated=True, native=False, total_bytes=50e9)
    prof = {"backend": "metal", "ram": 16 * 2**30, "gpu_mem": 16e9, "working_set": 12.7e9, "gpu_bw": 94.9,
            "now": {"avail_gb": 14.0}}
    s = machine.settings(prof, gated)
    assert "--cache-bias" in s["args"] and s["env"][key] == "1", s
    s = machine.settings(prof, dict(gated, gated=False))
    assert "--cache-bias" not in s["args"] and key not in s["env"], s
    print("  the cache-bias warning shows for a hand-typed setup and not for the launcher's gated one")


def check_f16_line_not_on_cuda():
    """On CUDA the Metal f16 GEMM does not apply, so its check says nothing; elsewhere it still does."""
    real = exl3_experts._CUDA
    engine_log.set_verbose(True)
    try:
        exl3_experts._CUDA = True
        exl3_experts._CHECKS.pop("f16mm", None)
        text = captured(exl3_experts._f16mm_ok)
        assert "f16" not in text and exl3_experts._CHECKS["f16mm"] is False, text
        exl3_experts._CUDA = False
        assert "exl3: some lever off: why" in captured(
            exl3_experts._checked, "t1", "some lever", "ref", lambda: (False, "why"))
        assert captured(exl3_experts._checked, "t2", "some lever", "ref", lambda: (False, "why"),
                        applies=False) == ""
    finally:
        exl3_experts._CUDA = real
        exl3_experts._CHECKS.pop("f16mm", None)
    print("  the f16 gate/up GEMM line is not printed on CUDA")


class _FakeLLM:
    """What ModelHost and main() touch, printing the engine's lines as the real one does."""
    STUB_MODES = moe_stream.StreamingLLM.STUB_MODES
    max_tokens = 512
    supports_images = False
    _budget = None

    def __init__(self, **kw):
        engine_log.note("  adaptive arena: governing 40 layers")
        engine_log.note("  memory budget 90% = 14.40 GB | mlx holds 1.20 GB | headroom 4.10 GB")
        print("  WARNING: a real warning from the engine")
        self.model = None
        self.tokenizer = type("T", (), {"detokenizer": None})()

    def forget(self):
        pass


class _Stop:
    socket = None

    def serve_forever(self):
        raise KeyboardInterrupt


def run_server(*argv):
    real = (server.StreamingLLM, server.serve, server.ModelHost._can_batch, sys.argv, server.port_error)
    server.StreamingLLM = _FakeLLM
    server.port_error = lambda host, port: None
    server.serve = lambda host_obj, port, host="127.0.0.1", cert=None, key=None: _Stop()
    server.ModelHost._can_batch = staticmethod(lambda llm: False)
    sys.argv = ["server.py", "--model", "/models/tiny-chat", "--port", "8123", *argv]
    try:
        return captured(server.main)
    finally:
        server.StreamingLLM, server.serve, server.ModelHost._can_batch, sys.argv, server.port_error = real
        engine_log.set_verbose(True)


def check_busy_port():
    """A port another program holds stops server.py with one line naming it, before any model loads."""
    import socket
    held = socket.socket()
    held.bind(("127.0.0.1", 0))
    held.listen()
    port = held.getsockname()[1]
    loaded = []
    real = (server.StreamingLLM, sys.argv)
    server.StreamingLLM = lambda **kw: loaded.append(kw)
    sys.argv = ["server.py", "--model", "/models/tiny-chat", "--port", str(port)]
    try:
        server.main()
    except SystemExit as stop:
        why = str(stop.code)
    else:
        why = ""
    finally:
        server.StreamingLLM, sys.argv = real
        held.close()
    assert why.startswith(f"server.py: cannot listen on 127.0.0.1:{port} (") and "\n" not in why, why
    assert not loaded, "the model must not load before the port is known to be free"
    print("  a busy port: " + why)


def check_server_output():
    """Default: model, memory, serving, chat, and the engine's real warning. --verbose adds the notes."""
    quiet = run_server("--mem-peak-gb", "2")
    lines = quiet.rstrip().split("\n")
    assert lines[0] == "  WARNING: a real warning from the engine", quiet
    assert lines[1].startswith("model: tiny-chat, loaded in ") and lines[1].endswith(" s"), quiet
    assert lines[2].startswith("memory: budget 2 GB peak"), quiet
    assert lines[3] == "serving on http://127.0.0.1:8123 (API base http://127.0.0.1:8123/v1)", quiet
    assert lines[4] == "chat in the browser: http://127.0.0.1:8123/", quiet
    assert len(lines) == 5, quiet
    adaptive = run_server()
    assert "memory: budget " in adaptive and "% of this machine)" in adaptive, adaptive
    loud = run_server("--mem-peak-gb", "2", "--verbose")
    for want in ("adaptive arena: governing 40 layers", "memory budget 90% = 14.40 GB",
                 "serving /models/tiny-chat: one turn at a time; conversation reuse on"):
        assert want in loud, loud
    print("  server.py default output:")
    for ln in lines:
        print("    " + ln)


def main():
    check_notes_follow_verbose()
    check_bias_warning()
    check_f16_line_not_on_cuda()
    check_server_output()
    check_busy_port()
    print("ok")


if __name__ == "__main__":
    main()
