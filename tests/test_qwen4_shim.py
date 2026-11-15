#!/usr/bin/env python3
"""The vendored qwen4_exp: mlx_lm.models.qwen4_exp imports with whichever mlx-lm is installed, and on
Linux (PyPI's mlx-lm) gated_delta_update takes the arguments gdn_cuda expects.

    perl -e 'alarm shift; exec @ARGV' 280 python3 tests/test_qwen4_shim.py
"""
import importlib
import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import engine_log   # noqa: F401  (installs the finder)
import mlx_lm.models.gated_delta as gd

m = importlib.import_module("mlx_lm.models.qwen4_exp")
assert hasattr(m, "Model") and hasattr(m, "ModelArgs"), "qwen4_exp has no Model"
if sys.platform.startswith("linux"):
    import gdn_cuda
    assert list(inspect.signature(gd.gated_delta_update).parameters) == gdn_cuda._PARAMS, \
        "gated_delta_update differs from gdn_cuda._PARAMS"
print("ok", m.__file__)
