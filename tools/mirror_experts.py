#!/usr/bin/env python3
"""Copy part of a pack's experts to a second drive, so expert reads run on both
devices at once (stream_fast._Stripe).

    python3 mirror_experts.py --model ~/models/<name> --drive "/Volumes/<drive>"
    python3 mirror_experts.py --model ~/models/<name> --drive "/Volumes/<drive>" --remove

1. Measures both devices with random reads of single expert records under
   F_NOCACHE, at the engine's queue depths, each alone and both at once (the
   drive's side on a scratch copy of the pack's first records).
2. Picks the experts to mirror by a hash of (layer, expert): a fraction of the
   pack, by default twice the drive's share of the combined read rate. A read
   wave has only ~15 records, so the engine needs about that many mirrored to
   find the drive's share in most waves.
3. Copies their records to <drive>/<pack name>/experts.ext.bin with F_NOCACHE
   reads and writes, at most --max-mbps, stopping when the machine starts to
   swap; reads the copy back and checks it; measures both devices again on
   the real sidecar.
4. Writes experts.ext.json last: the engine ignores a sidecar without it.

experts.bin stays whole, so the engine reads any record from it when the
drive is missing or fails. MOE_STRIPE=0 turns the sidecar off.
"""
import argparse
import fcntl
import hashlib
import json
import os
import random
import threading
import time
import zlib
from pathlib import Path

F_NOCACHE = 48
SIDECAR, META = "experts.ext.bin", "experts.ext.json"
INT_READERS = 8             # stream_fast._POOL's default (--threads)


def nocache_fd(path, flags=os.O_RDONLY):
    fd = os.open(path, flags, 0o644)
    fcntl.fcntl(fd, F_NOCACHE, 1)
    return fd


def all_records(index):
    """(group, expert, experts.bin offset, bytes) for every record, file order.
    A group is one layer's experts (gate, up and down share it)."""
    for key, e in sorted(index.items(), key=lambda kv: kv[1]["offset"]):
        group = key.rsplit(".", 1)[0]
        for x in range(e["n_experts"]):
            yield group, x, e["offset"] + x * e["stride"], e["stride"]


def picked(group, expert, fraction):
    return zlib.crc32(f"{group}:{expert}".encode()) < fraction * 2 ** 32


def bench(targets, seconds):
    """Random whole-record reads on every (fd, [(offset, bytes)], threads)
    target at once for `seconds`; GB/s per target."""
    got = [0] * len(targets)
    lock = threading.Lock()
    end = time.perf_counter() + seconds

    def work(k, fd, recs, seed):
        rng, n = random.Random(seed), 0
        buf = memoryview(bytearray(max(b for _, b in recs)))
        while time.perf_counter() < end:
            off, size = recs[rng.randrange(len(recs))]
            n += os.preadv(fd, [buf[:size]], off)
        with lock:
            got[k] += n

    t0 = time.perf_counter()
    ths = [threading.Thread(target=work, args=(k, fd, recs, k * 1000 + i))
           for k, (fd, recs, qd) in enumerate(targets) for i in range(qd)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    dt = time.perf_counter() - t0
    return [round(g / dt / 1e9, 3) for g in got]


class Guard:
    """Stop the copy when the machine starts to swap."""

    def __init__(self, max_swap_mb, min_margin_gb):
        import stream_fast as sf            # the engine's own counters
        self.sf, self.max_swap, self.min_margin = sf, max_swap_mb, min_margin_gb
        self.total = sf._sysctl_int("hw.memsize", width=8) / 1e9
        self.page = sf._sysctl_int("hw.pagesize", width=8) or 16384
        self.swap0 = sf._vm_counts()[1]
        self.margin_low = float("inf")

    def check(self):
        comp, swapouts = self.sf._vm_counts()
        margin = self.total * self.sf._sysctl_int("kern.memorystatus_level") / 100 \
            - self.sf._SWAP_SHARE * comp
        self.margin_low = min(self.margin_low, margin)
        swap_mb = (swapouts - self.swap0) * self.page / 1e6
        if swap_mb > self.max_swap:
            raise SystemExit(f"stopped: swapouts +{swap_mb:.0f} MB")
        if margin < self.min_margin:
            raise SystemExit(f"stopped: swap-line margin {margin:.2f} GB")
        return swap_mb


def copy(src, dst, recs, max_mbps, guard, chunk=64 << 20):
    """Records into dst back to back; crc32 per chunk, for the read-back."""
    buf = memoryview(bytearray(chunk + max(n for _, n in recs)))
    crcs, pos, t0, i = [], 0, time.perf_counter(), 0
    while i < len(recs):
        n = 0
        while i < len(recs) and n < chunk:
            off, size = recs[i]
            if os.preadv(src, [buf[n:n + size]], off) != size:
                raise SystemExit(f"short read from experts.bin at {off}")
            n, i = n + size, i + 1
        if os.pwrite(dst, buf[:n], pos) != n:
            raise SystemExit("short write to the sidecar")
        crcs.append((n, zlib.crc32(buf[:n])))
        pos += n
        guard.check()
        ahead = pos / (max_mbps * 1e6) - (time.perf_counter() - t0)
        if ahead > 0:
            time.sleep(ahead)
    return crcs, pos, time.perf_counter() - t0


def verify(fd, crcs):
    pos = 0
    for n, crc in crcs:
        data = os.pread(fd, n, pos)
        if len(data) != n or zlib.crc32(data) != crc:
            raise SystemExit(f"read-back differs at sidecar offset {pos}")
        pos += n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="pack directory (experts.bin + experts_index.json)")
    ap.add_argument("--drive", required=True, help="the second drive's mount point")
    ap.add_argument("--fraction", type=float, default=None,
                    help="share of experts to mirror (default: twice the drive's measured share)")
    ap.add_argument("--seconds", type=float, default=5.0, help="per bench point")
    ap.add_argument("--max-mbps", type=float, default=300.0, help="copy rate cap")
    ap.add_argument("--max-swap-mb", type=float, default=40.0)
    ap.add_argument("--min-margin-gb", type=float, default=1.0)
    ap.add_argument("--remove", action="store_true", help="delete this pack's sidecar")
    a = ap.parse_args(argv)

    model = Path(a.model).expanduser().resolve()
    out = Path(a.drive) / model.name
    if a.remove:
        for f in (META, SIDECAR):           # the json first: never a json without its file
            (out / f).unlink(missing_ok=True)
        print(f"removed {out}")
        return
    index_b = (model / "experts_index.json").read_bytes()
    index = json.loads(index_b)
    out.mkdir(parents=True, exist_ok=True)
    (out / META).unlink(missing_ok=True)    # an old json must not describe a new copy
    src = nocache_fd(model / "experts.bin")
    recs = [(off, n) for _, _, off, n in all_records(index)]
    guard = Guard(a.max_swap_mb, a.min_margin_gb)

    # 1. Both devices, the drive on a scratch copy of the first records.
    scratch = out / "bench.tmp"
    head = recs[:max(1, (1 << 30) // recs[0][1])]
    dst = nocache_fd(scratch, os.O_RDWR | os.O_CREAT | os.O_TRUNC)
    crcs, size, secs = copy(src, dst, head, a.max_mbps, guard)
    write_gbs = size / secs / 1e9
    ext_recs = []
    pos = 0
    for _, n in head:
        ext_recs.append((pos, n))
        pos += n
    pre = {"write_gbs": round(write_gbs, 3)}
    for qd in (2, 4, 8):
        pre[f"ext_qd{qd}"] = bench([(dst, ext_recs, qd)], a.seconds)[0]
    best = max(pre[f"ext_qd{q}"] for q in (2, 4, 8))
    ext_qd = next(q for q in (2, 4, 8) if pre[f"ext_qd{q}"] >= 0.98 * best)
    pre["int"] = bench([(src, recs, INT_READERS)], a.seconds)[0]
    pre["both"] = bench([(src, recs, INT_READERS), (dst, ext_recs, ext_qd)], a.seconds)
    os.close(dst)
    scratch.unlink()
    bi, be = pre["both"]
    share = be / (bi + be)
    frac = min(1.0, 2 * share) if a.fraction is None else a.fraction
    print(f"bench: write {write_gbs:.2f} GB/s; drive alone {best:.3f} GB/s (QD {ext_qd}), "
          f"internal alone {pre['int']:.3f}; both at once {bi:.3f} + {be:.3f}; "
          f"drive share {share:.3f} -> mirror {frac:.3f} of the experts", flush=True)

    # 2-3. The copy, its check, and both devices on the real sidecar.
    mine = [(off, n) for g, x, off, n in all_records(index) if picked(g, x, frac)]
    need = sum(n for _, n in mine)
    free = os.statvfs(out)
    if need > free.f_bavail * free.f_frsize:
        raise SystemExit(f"{need / 1e9:.1f} GB needed on {a.drive}")
    print(f"copying {len(mine)} records, {need / 1e9:.1f} GB, at most {a.max_mbps:.0f} MB/s",
          flush=True)
    dst = nocache_fd(out / SIDECAR, os.O_RDWR | os.O_CREAT | os.O_TRUNC)
    try:
        crcs, size, secs = copy(src, dst, mine, a.max_mbps, guard)
        print(f"copied in {secs:.0f} s ({size / secs / 1e9:.2f} GB/s); reading it back",
              flush=True)
        verify(dst, crcs)
    except BaseException:
        os.close(dst)
        (out / SIDECAR).unlink(missing_ok=True)
        raise
    records, pos = [], 0
    for off, n in mine:
        records.append([off, pos, n])
        pos += n
    bi, be = bench([(src, recs, INT_READERS), (dst, [(e, n) for _, e, n in records], ext_qd)],
                   a.seconds)
    os.close(dst)
    os.close(src)
    print(f"on the sidecar, both at once: {bi:.3f} + {be:.3f} GB/s", flush=True)

    # 4. The json last, atomically.
    meta = {"file": SIDECAR,
            "pack": {"dir": str(model), "bin_bytes": os.path.getsize(model / "experts.bin"),
                     "index_sha1": hashlib.sha1(index_b).hexdigest()},
            "made": time.strftime("%Y-%m-%d %H:%M:%S"),
            "fraction": frac, "share": round(share, 4),
            "bw_int": bi, "bw_ext": be, "int_readers": INT_READERS, "ext_readers": ext_qd,
            "bench_before_copy": pre, "copy_s": round(secs, 1),
            "swapouts_mb": round(guard.check(), 1), "margin_low_gb": round(guard.margin_low, 2),
            "ext_bytes": pos, "records": records}
    tmp = out / (META + ".tmp")
    tmp.write_text(json.dumps(meta))
    tmp.rename(out / META)
    print(f"wrote {out / META}: {len(records)} records, {pos / 1e9:.1f} GB; "
          f"swapouts +{meta['swapouts_mb']} MB, margin >= {meta['margin_low_gb']} GB")


if __name__ == "__main__":
    main()
