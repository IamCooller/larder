#!/usr/bin/env python3
"""
Lay each expert's weight+scales+biases out contiguously, so one read gets all three.

In the safetensors layout the three arrays of one expert live in
three different tensors at unrelated offsets, so fetching an expert costs three
preads. At top_k 4 across 81 projections that is ~972 syscalls per token, and
syscall count, not bandwidth, is what the profile shows as dominant.

Written back-to-back, an expert becomes a single byte range. os.preadv scatters
one contiguous read into several destination buffers, so the reader hands it the
weight, scales and bias views and gets all three filled by one syscall: 3x fewer.

Only routed experts (`.switch_mlp.`) move here; everything else stays in the
original safetensors and is loaded normally. EXL3 trellis banks
(exl3_experts.py) are records of trellis, suh and svh the same way, and so are
exllamav3's own packs (turboderp's), whose experts are tensors of their own.

Usage:
    python3 repack.py --src ~/models/<name>
"""

import argparse
import fcntl
import json
import os
import re
import struct
import time
from pathlib import Path

import numpy as np

EXPERT_MARK = ".switch_mlp."
# A record's parts in order, keyed by the first: MLX affine, or EXL3.
PARTS = {"weight": ("weight", "scales", "biases"), "trellis": ("trellis", "suh", "svh")}
# Experts read and written at a time: one projection's chunk is ~20 MB on Flash-Next.
CHUNK = 32


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return hdr, 8 + n


def _nocache(f):
    """Past the page cache where the OS allows it (macOS): a pack is tens of GB read once."""
    if hasattr(fcntl, "F_NOCACHE"):
        fcntl.fcntl(f.fileno(), fcntl.F_NOCACHE, 1)
    return f


def pack_shard(shard, dst, written, index):
    """Append one shard's routed-expert banks to the open experts.bin `dst` as
    records and their entries to `index`; returns the bytes written so far.
    launch.py calls it shard by shard as it downloads (an EXL3 pack)."""
    hdr, data_off = read_header(shard)
    bases = sorted({k.rsplit(".", 1)[0] for k in hdr
                    if k.rsplit(".", 1)[-1] in PARTS and EXPERT_MARK in k})
    if not bases:
        return written
    with _nocache(open(shard, "rb")) as fsrc:
        for base in bases:
            names = next(v for k, v in PARTS.items() if f"{base}.{k}" in hdr)
            metas = {p: hdr[f"{base}.{p}"] for p in names}
            n_exp = metas[names[0]]["shape"][0]
            rows = {}
            for p, m in metas.items():
                lo, hi = m["data_offsets"]
                rows[p] = (data_off + lo, (hi - lo) // n_exp)

            stride = sum(rb for _, rb in rows.values())
            index[base] = {
                "offset": written,
                "stride": stride,
                "n_experts": n_exp,
                "parts": [
                    {"name": p,
                     "bytes": rows[p][1],
                     "shape": list(metas[p]["shape"][1:]),
                     "dtype": metas[p]["dtype"]}
                    for p in names
                ],
            }

            # One record per expert: its parts back to back, CHUNK experts a write.
            for e0 in range(0, n_exp, CHUNK):
                n = min(CHUNK, n_exp - e0)
                cols = []
                for start, rb in rows.values():
                    b = os.pread(fsrc.fileno(), n * rb, start + e0 * rb)
                    assert len(b) == n * rb, f"short read in {Path(shard).name}"
                    cols.append(np.frombuffer(b, np.uint8).reshape(n, rb))
                dst.write(np.concatenate(cols, axis=1).tobytes())
                written += n * stride
    print(f"  {Path(shard).name}: {len(bases)} projections packed", flush=True)
    return written


# exllamav3's layout: `<layer>.mlp.experts.<E>.<proj>.trellis` (and .suh, .svh, .mul1), any shard.
PER_EXPERT = re.compile(r"(.*\.mlp)\.experts\.(\d+)\.(\w+)\.trellis")


def pack_per_expert(shards, dst, written, index):
    """exllamav3's per-expert EXL3 tensors (the MTP head's skipped) as pack_shard's records, keyed
    `<layer>.mlp.switch_mlp.<proj>`; returns the bytes written so far, unchanged for any other layout."""
    where = {}
    for shard in shards:
        hdr, off = read_header(shard)
        where.update((k, (shard, off, m)) for k, m in hdr.items() if k != "__metadata__")
    banks = {}
    for k in where:
        g = PER_EXPERT.fullmatch(k)
        if g and not k.startswith("mtp."):
            banks.setdefault((g[1], g[3]), set()).add(int(g[2]))
    fds = {}
    try:
        for (pre, proj), es in sorted(banks.items()):
            n_exp = len(es)
            assert es == set(range(n_exp)), f"{pre} {proj}: experts not 0..{n_exp - 1}"
            parts = [[where[f"{pre}.experts.{e}.{proj}.{p}"] for p in PARTS["trellis"]] for e in range(n_exp)]
            metas = [m for _, _, m in parts[0]]
            sizes = [m["data_offsets"][1] - m["data_offsets"][0] for m in metas]
            index[f"{pre}.switch_mlp.{proj}"] = {
                "offset": written,
                "stride": sum(sizes),
                "n_experts": n_exp,
                # exllamav3 stores the trellis as I16; the engine reads the same bits as U16.
                "parts": [{"name": p, "bytes": b, "shape": m["shape"],
                           "dtype": "U16" if m["dtype"] == "I16" else m["dtype"]}
                          for p, b, m in zip(PARTS["trellis"], sizes, metas)],
            }
            for e0 in range(0, n_exp, CHUNK):
                buf = bytearray()
                for e in range(e0, min(e0 + CHUNK, n_exp)):
                    for (shard, off, m), m0 in zip(parts[e], metas):
                        assert (m["shape"], m["dtype"]) == (m0["shape"], m0["dtype"]), (pre, proj, e)
                        if shard not in fds:
                            fds[shard] = _nocache(open(shard, "rb"))
                        lo, hi = m["data_offsets"]
                        b = os.pread(fds[shard].fileno(), hi - lo, off + lo)
                        assert len(b) == hi - lo, f"short read in {Path(shard).name}"
                        buf += b
                dst.write(buf)
                written += len(buf)
            print(f"  {pre}.{proj}: {n_exp} experts packed", flush=True)
    finally:
        for f in fds.values():
            f.close()
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default=None, help="defaults to <src>/experts.bin")
    args = ap.parse_args()

    src = Path(args.src).expanduser()
    out_bin = Path(args.out).expanduser() if args.out else src / "experts.bin"
    out_idx = out_bin.with_name("experts_index.json")

    shards = sorted(src.glob("*.safetensors"))
    index, written = {}, 0
    t0 = time.time()

    with _nocache(open(out_bin, "wb")) as dst:
        written = pack_per_expert(shards, dst, written, index)
        if not index:
            for shard in shards:
                written = pack_shard(shard, dst, written, index)

    out_idx.write_text(json.dumps(index, indent=1))
    print(f"\n{len(index)} projections -> {written/1e9:.2f} GB in {time.time()-t0:.0f}s")
    print(f"wrote {out_bin.name} and {out_idx.name}")
    print("one preadv per expert instead of three")


if __name__ == "__main__":
    main()
