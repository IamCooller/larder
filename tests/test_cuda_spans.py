#!/usr/bin/env python3
"""MOE_CUDA_SPANS (cuda_stage.read_span): a record whose size and offset are not page
multiples, read as its page-aligned span through the second fd, lands at cuda_stage.lead() in its room with the
bytes a buffered pread gives; so do the page-cache fallback (a second fd that fails), a record off the 16-byte
grid (never tried direct) and the file's last record, whose span runs past the end. Then the engine's two users
with the copy kernel emulated in numpy, flag on and off: a Staging wave of two chunks (decode, a narrow tail) and
two LMStage groups (LMStage.read, and jobs() run one by one as _LMReader's reader queue runs them) with a skipped
expert and a resident copied slot to slot leave every slot with the bytes pread gives. Linux opens the second fd
with O_DIRECT on a file in TMPDIR (put TMPDIR on the drive: tmpfs refuses O_DIRECT and the test fails), macOS
opens it with F_NOCACHE, which checks the span arithmetic only. PACK=<model dir> also reads 128 random records
of its experts.bin (read only) both ways.

    perl -e 'alarm shift; exec @ARGV' 280 python3 tests/test_cuda_spans.py
    PACK=<a packed model directory> python tests/test_cuda_spans.py      # on an NVIDIA machine
"""
import importlib
import json
import mmap
import os
import random
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["MOE_CUDA_SPANS"] = "1"

import mlx.core as mx
import numpy as np

import cuda_stage as cs

STRIDES = (825_600, 620_800, 2_476_800, 1_000_016, 409_600, 4_112)    # K4, K3, a K4 expert, odd, a page multiple, a page and a bit
HEADERS = (0, 3 * cs.PAGE, 1_234_560, 1_000)                         # page, page, 16-byte grid, off the grid
PARTS = (798_720, 25_856, 1_024)                                     # a K4 record in three parts, as a projection's tensors are


def second_fd(path):
    """The fd read_span reads spans through: O_DIRECT on Linux, F_NOCACHE on macOS."""
    if hasattr(os, "O_DIRECT"):
        try:
            return os.open(path, os.O_RDONLY | os.O_DIRECT)
        except OSError as e:
            raise SystemExit(f"test_cuda_spans: O_DIRECT refused in {path.parent} ({e}); set TMPDIR to the drive")
    import fcntl
    d = os.open(path, os.O_RDONLY)
    fcntl.fcntl(d, fcntl.F_NOCACHE, 1)
    return d


def check(fd, d, off, n):
    """One record both ways, into a page-aligned room filled with a stale pattern first, returning whether the span read served it."""
    m = mmap.mmap(-1, cs.room(n))
    buf = np.frombuffer(m, dtype=np.uint8)
    buf[:] = 0xA5
    direct = cs.read_span((fd, buf, off, n, d))
    at = cs.lead(d, off)
    want = os.pread(fd, n, off)
    assert len(want) == n, (off, n)
    assert bytes(buf[at:at + n]) == want, f"bytes differ: stride {n} at {off} (lead {at}, direct {direct})"
    del buf
    m.close()
    return direct


def toy():
    size = 12 * 2**20 + 2_320              # on the 16-byte grid, not a page multiple: the last span runs past the end
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "experts.bin"
        path.write_bytes(np.random.default_rng(0).integers(0, 256, size, dtype=np.uint8).tobytes())
        fd, d = os.open(path, os.O_RDONLY), second_fd(path)
        bad = os.open(path, os.O_RDONLY)
        os.close(bad)                      # a closed fd: every span read through it fails
        n_direct = n_all = 0
        try:
            for n in STRIDES:
                assert cs.room(n) % cs.PAGE == 0 and cs.room(n) >= n + cs.PAGE - 1, n
                offs = [h + e * n for h in HEADERS for e in range(4) if h + (e + 1) * n <= size] + [size - n]
                for off in offs:
                    grid = off % 16 == 0
                    assert check(fd, d, off, n) == grid, f"stride {n} at {off}: direct {not grid}"
                    assert not check(fd, bad, off, n)
                    assert not check(fd, None, off, n) and cs.lead(None, off) == 0
                    n_direct += grid
                    n_all += 1
        finally:
            os.close(fd)
            os.close(d)
    print(f"   toy: {n_all} records ({n_direct} on the 16-byte grid read as spans), each also through a failing "
          f"fd and buffered, equal to pread ({'O_DIRECT' if hasattr(os, 'O_DIRECT') else 'F_NOCACHE'}, "
          f"strides {list(STRIDES)})")


class Kernel:
    """cuda_stage._SRC in numpy: each table row (dst part, dst slot, src in 16 B, 16 B units, src slot) copies from `src` or from another slot of the part."""
    src = parts = None

    def __call__(self, inputs, **_):
        for dp, sl, s16, units, ss in np.array(inputs[-1]).reshape(-1, 5).tolist():
            part = self.parts[dp]
            part[sl] = self.src[16 * s16:16 * (s16 + units)] if ss < 0 else part[ss]
        return [mx.array([1], dtype=mx.int32)]


def plain(job):
    fd, iov, off, n = job
    assert os.preadv(fd, iov, off) == n


def engine(spans):
    """Staging.wave and LMStage under the flag (or without it) with the emulated kernel, checked slot by slot against pread."""
    os.environ["MOE_CUDA_SPANS"] = "1" if spans else "0"
    m = importlib.reload(cs)
    stride, n_exp, hdr, slots = sum(PARTS), 12, 8_208, 12
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "experts.bin"
        path.write_bytes(np.random.default_rng(2).integers(0, 256, hdr + 3 * n_exp * stride, dtype=np.uint8).tobytes())
        fd = os.open(path, os.O_RDONLY)
        offs = [hdr + j * n_exp * stride for j in range(3)]

        def want(j, e):
            return os.pread(fd, stride, offs[j] + e * stride)

        def row(j, s):
            return np.concatenate([p[s] for p in parts[j]]).tobytes()

        kern, st = Kernel(), m.Staging(plain)
        st.dma = False                      # the emulated kernel even where libcuda would take the copy engine
        st._kernel = lambda n: kern
        if spans and not hasattr(os, "O_DIRECT"):
            d = second_fd(path)
            st._direct_fd = lambda f: d
        st._room(stride)
        parts = [[np.zeros((slots, b), dtype=np.uint8) for b in PARTS] for _ in range(3)]
        ar = [(PARTS, [[p[s] for p in parts[j]] for s in range(slots)], fd, offs[j], stride) for j in range(3)]
        kern.src, kern.parts = st.view, [p for ps in parts for p in ps]
        picks = [(j, s, (5 * j + 7 * s) % n_exp) for j in range(3) for s in range(slots)]
        outside = np.zeros(stride, dtype=np.uint8)
        jobs = [(fd, ar[j][1][s], offs[j] + e * stride, stride) for j, s, e in picks]
        jobs.append((fd, [outside], offs[1] + 3 * stride, stride))     # not a slot of the arena: read straight in
        with ThreadPoolExecutor(8) as pool:
            st.wave(jobs, ar, pool)
        assert all(row(j, s) == want(j, e) for j, s, e in picks), "wave"
        assert outside.tobytes() == want(1, 3)

        parts = [[np.zeros((slots, b), dtype=np.uint8) for b in PARTS] for _ in range(3)]
        kern.parts = [p for ps in parts for p in ps]
        store = SimpleNamespace(per_half=4, slot_bytes=3 * stride, bufs=[PARTS] * 3, views=parts)
        lm = m.LMStage(st, store)
        kern.src = lm.view
        glu = SimpleNamespace(_projs=[SimpleNamespace(_packed={"stride": stride, "fd": fd, "offset": offs[j]})
                                      for j in range(3)])
        ids, copies, skip = np.array([1, 4, 6, 9]), {2: 11}, {1}
        for j in range(3):                  # slot 11 holds expert 6 from before the pass
            at = 0
            for p in parts[j]:
                p[11] = np.frombuffer(want(j, 6), dtype=np.uint8)[at:at + p.shape[1]]
                at += p.shape[1]
        assert lm.read(glu, ids, 0, copies, skip) == 6 * stride
        for job in lm.jobs(glu, ids, 4, copies, skip):
            lm.one(job)
        for r0 in (0, 4):
            lm.take(r0)
            lm.give(r0)
            for j in range(3):
                for i, e in enumerate(ids.tolist()):
                    assert row(j, r0 + i) == (bytes(stride) if i in skip else want(j, e)), ("group", r0, j, i)
        lm.close()
        os.close(fd)
    n = 3 * slots + 2 * 6
    assert st.stats == {"span_direct": n if spans else 0, "span_buffered": 0}, st.stats
    print(f"   engine, MOE_CUDA_SPANS={int(spans)}: a {len(jobs)}-record wave and two layer-major groups land the "
          f"bytes pread gives in every slot; {st.stats}")


def pack(model):
    """128 random records of the pack's projections both ways, all of them span reads since the pack's records sit on the 16-byte grid."""
    idx = json.loads((model / "experts_index.json").read_text())
    path = model / "experts.bin"
    fd, d = os.open(path, os.O_RDONLY), second_fd(path)
    rnd = random.Random(0)
    names = rnd.sample(sorted(idx), min(8, len(idx)))
    strides, n = set(), 0
    try:
        for name in names:
            meta = idx[name]
            strides.add(meta["stride"])
            for e in rnd.sample(range(meta["n_experts"]), min(16, meta["n_experts"])):
                assert check(fd, d, meta["offset"] + e * meta["stride"], meta["stride"]), (name, e, "not a span read")
                n += 1
    finally:
        os.close(fd)
        os.close(d)
    print(f"   pack {model.name}: {n} span reads of {len(names)} projections equal to pread "
          f"(strides {sorted(strides)})")


def main():
    assert cs.SPANS
    toy()
    if os.environ.get("PACK"):
        pack(Path(os.path.expanduser(os.environ["PACK"])))
    engine(True)
    engine(False)
    print("test_cuda_spans: ok")


if __name__ == "__main__":
    main()
