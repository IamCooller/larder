"""Toy tests for tuner.py: decisions, the cache round trip and the disk probe on a scratch file. No model."""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import tuner


def test_decisions():
    curve = [(1, 1.0), (2, 2.0), (4, 3.9), (8, 4.0), (16, 4.05)]
    assert tuner.readers_from_curve(curve, 0.02) == 8       # 3.9 < 4.05 * 0.98, 4.0 is not
    assert tuner.readers_from_curve(curve, 0.05) == 4
    assert tuner.readers_from_curve(curve, 0.0) == 16
    assert tuner.direct_env(4.0, 3.0, 0.02) == {}
    assert tuner.direct_env(3.0, 4.0, 0.02) == {"MOE_PIPE_DIRECT": "0"}
    assert tuner.direct_env(0, 4.0, 0.02, direct_ok=False) == {"MOE_PIPE_DIRECT": "0"}
    assert tuner.exact_args(["--prefill-step", "0", "--keep-convs", "4", "--cache-bias", "0.5",
                             "--mem-peak-gb", "6", "--mtp", "x"]) == {
        "--prefill-step": 0, "--keep-convs": 4, "--mem-peak-gb": 6.0}
    assert tuner.opted_in(["--tuned"], {}) and tuner.opted_in([], {"MOE_TUNED": "1"})
    assert not tuner.opted_in(["x"], {})
    assert tuner.exact_args(["--mem-peak-max-gb", "8"]) == {"--mem-peak-max-gb": 8.0}
    assert tuner.cap_ceiling({"--mem-peak-gb": 6.0, "--mem-peak-max-gb": 12.7}, "metal") == {
        "--mem-peak-gb": 6.0, "--mem-peak-max-gb": 10.0}
    assert tuner.cap_ceiling({"--mem-peak-gb": 6.0, "--mem-peak-max-gb": 8.0}, "metal")["--mem-peak-max-gb"] == 8.0
    assert tuner.cap_ceiling({"--mem-peak-gb": 10.5, "--mem-peak-max-gb": 12.0}, "metal") == {"--mem-peak-gb": 10.5}
    # On CUDA the ceiling is the VRAM store's, which the host footprint guard does not bound.
    assert tuner.cap_ceiling({"--mem-peak-gb": 7.0, "--mem-peak-max-gb": 16.0}, "cuda") == {
        "--mem-peak-gb": 7.0, "--mem-peak-max-gb": 16.0}
    assert tuner.recipe_of("/nonexistent", {"backend": "metal"}) is None
    assert tuner.recipe_of("/nonexistent", {"backend": "cuda"}) == "store"
    assert tuner.recipe_of("/nonexistent", {"backend": "metal"}, "arena") == "arena"
    assert abs(tuner.tier_gb(8.5e9) - 2.0) < 1e-9


def test_cache_round_trip_and_defaults():
    with tempfile.TemporaryDirectory() as d:
        tuner.CACHE = Path(d) / "at"
        tuner.machine_key = lambda: "k"
        model = Path(d) / "m"
        model.mkdir()
        (model / "experts.bin").write_bytes(b"x" * 4096)
        assert tuner.load(str(model)) is None
        t = {"version": 1, "model_bytes": 4096, "env": {"MOE_PIPE_DIRECT": "0"},
             "args": {"--mem-peak-gb": 6.0, "--mem-peak-max-gb": 8.0, "--keep-convs": 4}}
        tuner.CACHE.mkdir()
        tuner.cache_path(str(model)).write_text(json.dumps(t))
        assert tuner.load(str(model))["args"]["--keep-convs"] == 4
        (model / "experts.bin").write_bytes(b"x" * 8192)       # a changed model drops the cache
        assert tuner.load(str(model)) is None
        (model / "experts.bin").write_bytes(b"x" * 4096)
        argv = ["server.py", "--model", str(model)]
        assert tuner.parser_defaults(str(model), argv) == {
            "mem_peak_gb": 6.0, "mem_peak_max_gb": 8.0, "keep_convs": 4}
        # a hand-set ceiling, or another memory model, keeps the cached ceiling out
        assert "mem_peak_max_gb" not in tuner.parser_defaults(str(model), argv + ["--mem-peak-gb", "16"])
        assert "mem_peak_gb" not in tuner.parser_defaults(str(model), argv + ["--resident", "21"])
        env = {"MOE_PIPE_DIRECT": "1"}
        tuner.apply_env(str(model), env)
        assert env["MOE_PIPE_DIRECT"] == "1"                     # the user's value wins
        env = {}
        tuner.apply_env(str(model), env)
        assert env == {"MOE_PIPE_DIRECT": "0"}


def test_disk_curve_on_scratch_file():
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "experts.bin").write_bytes(os.urandom(64 << 20))
        tuner.POINT_S = 0.05
        curve, noise = tuner.disk_curve(d, 1 << 20, time.time() + 30)
        assert [q for q, _ in curve] == list(tuner.QDS) and all(bw > 0 for _, bw in curve)
        assert 0 <= noise <= 1


def test_record_bytes():
    with tempfile.TemporaryDirectory() as d:
        idx = {f"model.layers.{i}.mlp.{p}": {"stride": s, "n_experts": 4}
               for i in range(3) for p, s in (("gate", 100), ("up", 100), ("down", 50))}
        (Path(d) / "experts_index.json").write_text(json.dumps(idx))
        assert tuner.record_bytes(d) == 250


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f()
            print("ok", n)
