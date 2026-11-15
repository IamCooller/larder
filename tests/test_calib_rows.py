#!/usr/bin/env python3
"""rowqmm's saved layout records which row counts were calibrated, so a start asks only for the rows it needs.

    python3 tests/test_calib_rows.py
"""

import json
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import rowqmm                                                  # noqa: E402
import launch                                                  # noqa: E402


def test_old_file_holds_all_rows():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "old.json"
        p.write_text(json.dumps({"1,2560,4096": "rb1", "8,2560,4096": "rb2"}))
        layout, rows = rowqmm._load_layout(p)
        assert rows == set(rowqmm.ALL_ROWS), rows
        assert layout == {(1, 2560, 4096): 1, (8, 2560, 4096): 2}, layout


def test_new_file_keeps_its_rows_and_merges():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "new.json"
        rowqmm._save_layout(p, {(1, 2560, 4096): 2}, {1})
        layout, rows = rowqmm._load_layout(p)
        assert rows == {1} and layout == {(1, 2560, 4096): 2}
        need = sorted(set(rowqmm.ALL_ROWS) - rows)
        assert need == [2, 3, 4, 5, 6, 7, 8], need
        layout[(2, 2560, 4096)] = 1
        rowqmm._save_layout(p, layout, rows | {2})
        assert rowqmm._load_layout(p)[1] == {1, 2}


def test_bad_label_drops_the_file():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "bad.json"
        p.write_text(json.dumps({"1,1,1": "nope", "rows": "1"}))
        assert rowqmm._load_layout(p) is None


def test_launcher_shows_the_phase():
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "s.log"
        log.write_text("x\n  measuring this Mac's kernels for this model, once (about 90 s, T=[1])\n")
        assert "measuring this Mac's kernels" in launch.loading_text(log, time.time())
        log.write_text(log.read_text() + "  kernels measured in 80 s, saved to y\n")
        assert "measuring" not in launch.loading_text(log, time.time())
        assert launch.loading_text(Path(d) / "none.log", time.time()).startswith("loading the model")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
