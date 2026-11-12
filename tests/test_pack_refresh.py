"""Toy test for pack_refresh.rewrite: identical bytes after a rewrite, and a resume after a stop."""
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pack_refresh
import tuner


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


with tempfile.TemporaryDirectory() as d:
    f, prog = Path(d, "experts.bin"), Path(d, "prog.json")
    f.write_bytes(os.urandom(5 * (1 << 20) + 1234))      # an unaligned tail on purpose
    h = sha(f)
    ch = 1 << 20
    # Other jobs writing to this disk would trip the free-space rule, which this test does not cover.
    pack_refresh.free_jitter = lambda p: (0, 0)
    assert pack_refresh.rewrite(f, prog, chunk=ch, log=lambda *_: None) == "done" and sha(f) == h and not prog.exists()
    # A stop after two chunks: progress says 2 MB done, and the next run starts there.
    prog.write_text(json.dumps({"path": str(f), "size": f.stat().st_size, "done": 2 * ch}))
    seen = []
    orig = os.preadv
    os.preadv = lambda fd, bufs, off: (seen.append(off), orig(fd, bufs, off))[1]
    try:
        assert pack_refresh.rewrite(f, prog, chunk=ch, log=lambda *_: None) == "done"
    finally:
        os.preadv = orig
    assert min(seen) == 2 * ch and sha(f) == h
    # Baseline rule: slower beyond the larger noise at both rates is aged, within it is not.
    base = {"rates": [3.0, 5.0], "noise": 0.03}
    assert tuner.aged_verdict(base, {"rates": [2.5, 4.4], "noise": 0.02})["aged"]
    assert not tuner.aged_verdict(base, {"rates": [2.95, 4.9], "noise": 0.02})["aged"]
    assert not tuner.aged_verdict(base, {"rates": [2.5, 4.9], "noise": 0.02})["aged"]
    # Stored baseline on a toy pack: the first call stores it and says not aged; a forced check compares to it.
    tuner.CACHE = Path(d, "cache")
    m = Path(d, "model")
    m.mkdir()
    (m / "experts.bin").write_bytes(os.urandom(4 << 20))
    real_rates = tuner.pack_rates
    tuner.pack_rates = lambda binf, record: (3.0, 5.0)
    try:
        r = tuner.pack_age(m, record=4096)
        assert r["aged"] is False and r["baseline_source"] == "first_tune"
        tuner.pack_rates = lambda binf, record: (2.0, 3.0)
        assert tuner.pack_age(m, force=True, record=4096)["aged"] is True
        tuner.pack_rates = lambda binf, record: (3.0, 5.0)
        assert tuner.pack_age(m, record=4096)["aged"] is True        # inside the day: the stored verdict
        assert tuner.pack_age(m, force=True, record=4096)["aged"] is False
    finally:
        tuner.pack_rates = real_rates
print("ok")
