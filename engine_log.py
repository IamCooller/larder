"""The engine's start-up and per-request notes, printed unless an entry point switched them off."""

import os
import sys

# On Linux MLX's CUDA graph cache is capped, since by default it keeps a graph for every new prefill shape in driver memory.
GRAPH_CACHE = "16"
if sys.platform.startswith("linux"):
    os.environ.setdefault("MLX_CUDA_GRAPH_CACHE_SIZE", GRAPH_CACHE)
    os.environ.setdefault("MLX_ENABLE_CACHE_THRASHING_CHECK", "0")

verbose = True      # library users and tests see every note; chat.py and server.py turn them off without --verbose


def set_verbose(on):
    global verbose
    verbose = bool(on)


def note(*args, **kwargs):
    """print() for what only someone debugging the engine wants to read."""
    if verbose:
        print(*args, **kwargs)
