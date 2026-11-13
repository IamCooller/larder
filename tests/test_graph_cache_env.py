#!/usr/bin/env python3
"""engine_log sets MLX's CUDA graph cache variables on Linux only, keeps a user's own values, and server.py, chat.py
and launch.py import it before anything imports MLX. No model, no GPU.

    perl -e 'alarm shift; exec @ARGV' 280 python3 tests/test_graph_cache_env.py
"""
import ast
import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
KEYS = ("MLX_CUDA_GRAPH_CACHE_SIZE", "MLX_ENABLE_CACHE_THRASHING_CHECK")


def load(platform, env):
    saved, real = {k: os.environ.pop(k, None) for k in KEYS}, sys.platform
    os.environ.update(env)
    sys.platform = platform
    try:
        sys.modules.pop("engine_log", None)
        m = importlib.import_module("engine_log")
        return m, {k: os.environ.get(k) for k in KEYS}
    finally:
        sys.platform = real
        for k in KEYS:
            os.environ.pop(k, None)
            if saved[k] is not None:
                os.environ[k] = saved[k]


def first_imports(path):
    """Top-level module names a file imports, in order."""
    out = []
    for node in ast.parse(path.read_text()).body:
        for n in ([node] + [b for b in getattr(node, "body", []) if isinstance(b, (ast.Import, ast.ImportFrom))]):
            if isinstance(n, ast.Import):
                out += [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.module:
                out.append(n.module)
    return out


def main():
    m, env = load("linux", {})
    assert env == {"MLX_CUDA_GRAPH_CACHE_SIZE": m.GRAPH_CACHE, "MLX_ENABLE_CACHE_THRASHING_CHECK": "0"}, env
    assert int(m.GRAPH_CACHE) > 0
    _, env = load("darwin", {})
    assert env == {k: None for k in KEYS}, env
    _, env = load("linux", {"MLX_CUDA_GRAPH_CACHE_SIZE": "400", "MLX_ENABLE_CACHE_THRASHING_CHECK": "1"})
    assert env == {"MLX_CUDA_GRAPH_CACHE_SIZE": "400", "MLX_ENABLE_CACHE_THRASHING_CHECK": "1"}, env
    srv = first_imports(ROOT / "server.py")
    assert srv.index("engine_log") < min(i for i, n in enumerate(srv) if n.split(".")[0] in ("mlx", "tuner")), srv
    chat = [n for n in first_imports(ROOT / "chat.py") if n.split(".")[0] not in sys.stdlib_module_names]
    assert chat[0] == "engine_args", chat
    ea = [n for n in first_imports(ROOT / "engine_args.py") if n.split(".")[0] not in sys.stdlib_module_names]
    assert ea[0] == "engine_log", ea
    la = [n for n in first_imports(ROOT / "launch.py") if n.split(".")[0] not in sys.stdlib_module_names]
    assert la[0] == "engine_log", la
    print("graph cache env: ok")


if __name__ == "__main__":
    main()
