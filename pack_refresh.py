#!/usr/bin/env python3
"""Rewrite a model's experts.bin in place with its own bytes: old flash data reads slower than fresh.

    ./larder refresh [model dir or name]
"""
import json
import mmap
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

CH, PAGE = 64 << 20, 4096


def real_pack(model):
    """The file behind experts.bin, symlinks resolved once (a pack can link to another model's file)."""
    return (Path(model).expanduser() / "experts.bin").resolve()


def open_by(path):
    """PIDs that hold the file open: a server.py with the pack loaded."""
    path = str(path)
    pids = set()
    if Path("/proc").is_dir():
        for d in Path("/proc").glob("[0-9]*"):
            try:
                if any(os.path.realpath(f) == path for f in d.glob("fd/*")) or path in (d / "maps").read_text():
                    pids.add(d.name)
            except OSError:
                pass
        pids.discard(str(os.getpid()))
        return sorted(pids)
    r = subprocess.run(["lsof", "-t", path], capture_output=True, text=True)
    return sorted(set(r.stdout.split()) - {str(os.getpid())})


def free_jitter(path, n=5):
    """(free bytes now, how much free space wobbles on its own over a second): the rule's noise floor."""
    xs = []
    for _ in range(n):
        xs.append(shutil.disk_usage(path.parent).free)
        time.sleep(0.25)
    return xs[-1], max(xs) - min(xs)


def open_direct(path, flags):
    """O_DIRECT where the file system takes it, F_NOCACHE on macOS, plain otherwise."""
    if sys.platform == "darwin":
        import fcntl
        fd = os.open(path, flags)
        fcntl.fcntl(fd, 48, 1)
        return fd
    try:
        return os.open(path, flags | getattr(os, "O_DIRECT", 0))
    except OSError:
        return os.open(path, flags)


def rewrite(path, progress, chunk=CH, log=print):
    """Rewrite `path` chunk by chunk; return "done" or the reason it stopped. Every chunk is read, written back at the
    same offset, fsynced, read again and compared, so a stop at any point leaves the content intact. Resumes from
    `progress` (a json file with the offset done) when the file is the same size."""
    path = Path(path)
    size = path.stat().st_size
    try:
        st = json.loads(Path(progress).read_text())
        off = st["done"] if st["size"] == size and st["path"] == str(path) else 0
    except (OSError, ValueError, KeyError):
        off = 0
    if pids := open_by(path):
        return f"in use by pid {', '.join(pids)}: stop the server first"
    free0, jitter = free_jitter(path)
    # A snapshot holding the old extents makes every chunk eat free space; the allowance is a few chunks plus the wobble seen.
    allow = 8 * chunk + jitter
    rd, wr, buf = open_direct(path, os.O_RDONLY), open_direct(path, os.O_WRONLY), os.open(path, os.O_RDWR)
    a, b = mmap.mmap(-1, chunk), mmap.mmap(-1, chunk)
    why, t0 = "done", time.time()
    try:
        while off < size:
            if shutil.disk_usage(path.parent).free < free0 - allow:
                why = f"free space fell {(free0 - shutil.disk_usage(path.parent).free) / 1e9:.2f} GB (snapshot holding old data?)"
                break
            n = min(chunk, size - off)
            na = n // PAGE * PAGE
            if na and os.preadv(rd, [memoryview(a)[:na]], off) != na:
                why = f"short read at {off}"
                break
            tail = os.pread(buf, n - na, off + na) if n > na else b""
            if na and os.pwritev(wr, [memoryview(a)[:na]], off) != na:
                why = f"short write at {off}"
                break
            if tail and os.pwrite(buf, tail, off + na) != len(tail):
                why = f"short tail write at {off + na}"
                break
            os.fsync(wr)
            os.fsync(buf)
            if na and (os.preadv(rd, [memoryview(b)[:na]], off) != na or a[:na] != b[:na]):
                why = f"read-back differs at {off}"
                break
            if tail and os.pread(buf, len(tail), off + na) != tail:
                why = f"tail read-back differs at {off + na}"
                break
            off += n
            Path(progress).write_text(json.dumps({"path": str(path), "size": size, "done": off}))
            if (off // chunk) % 16 == 0:
                log(f"  {off / 1e9:.1f} of {size / 1e9:.1f} GB, {time.time() - t0:.0f} s")
    finally:
        for fd in (rd, wr, buf):
            os.close(fd)
    if why == "done":
        Path(progress).unlink(missing_ok=True)
    return why


def resolve_model(arg):
    p = Path(arg).expanduser()
    if not (p / "experts.bin").exists():
        import machine
        p = Path(machine.CACHE) / arg
    return p if (p / "experts.bin").exists() else None


def progress_file(pack):
    import tuner
    tuner.CACHE.mkdir(parents=True, exist_ok=True)
    return tuner.CACHE / f"refresh-{pack.name}-{pack.stat().st_size}.json"


def refresh(model):
    """Check, rewrite, check again. Returns an error text or None. Used by ./larder refresh and by the launcher's question."""
    import tuner
    pack = real_pack(model)
    if open_by(pack):
        return "a server has this pack open: stop it first (./larder stop)"
    r = tuner.pack_age(model, force=True)
    print(f"before: {r['rates_gbps']} GB/s (QD1, QD8) against baseline {r.get('baseline_gbps', 'none yet')}; aged: {r['aged']}")
    print(f"rewriting {pack} ({pack.stat().st_size / 1e9:.1f} GB, about {pack.stat().st_size / 0.6e9:.0f} s)", flush=True)
    why = rewrite(pack, progress_file(pack))
    if why != "done":
        return f"stopped: {why} (run again to resume)"
    m = tuner.record_baseline(model, "rewrite")
    print(f"after: {m['rates']} GB/s, stored as the new baseline")


def main():
    if sys.argv[1:2] in (["-h"], ["--help"]):
        print(__doc__)
        return
    if len(sys.argv) < 2:
        sys.exit("usage: ./larder refresh <model dir or name>")
    model = resolve_model(sys.argv[1])
    if not model:
        sys.exit(f"no experts.bin in {sys.argv[1]}")
    if err := refresh(model):
        sys.exit(err)


if __name__ == "__main__":
    main()
