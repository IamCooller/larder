#!/usr/bin/env python3
"""GGUF routed experts -> our experts.bin + experts_index.json (gguf_experts.py decodes them).

Each expert of a GGUF expert tensor is one contiguous slice (ne0 x ne1 weights in ggml blocks); it becomes one
record of one part, "gguf" (uint8 [ne1, row bytes], its ggml type in the part), padded to a 4 KiB stride. The
trunk stays the engine's own pack; only `blk.N.ffn_{gate,up,down}_exps` move here. Streaming: CHUNK experts in
memory at a time.

    python3 tools/gguf_repack.py <shard>.gguf ... --out <dir>/experts.bin

Header layout: llama.cpp's gguf-py (gguf_reader.py, constants.py), commit b1c902f0.
"""
import argparse
import fcntl
import json
import os
import re
import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gguf_ref import HALVES, TYPES, valid      # noqa: E402,F401

# ggml type id -> (name, elements a block, bytes a block): every type of gguf-py constants.py GGML_QUANT_SIZES
# (llama.cpp dd6b945b), decoded or not, so any file's header parses.
GGML = {0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20), 6: ("Q5_0", 32, 22),
        7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36), 10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110),
        12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
        16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98), 19: ("IQ1_S", 256, 50),
        20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110), 22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136),
        24: ("I8", 1, 1), 25: ("I16", 1, 2), 26: ("I32", 1, 4), 27: ("I64", 1, 8), 28: ("F64", 1, 8),
        29: ("IQ1_M", 256, 56), 30: ("BF16", 1, 2), 34: ("TQ1_0", 256, 54), 35: ("TQ2_0", 256, 66),
        39: ("MXFP4", 32, 17), 40: ("NVFP4", 64, 36), 41: ("Q1_0", 128, 18), 42: ("Q2_0", 64, 18),
        43: ("F8_E4M3", 1, 1)}

# GGUF value types: 0 u8 .. 12 f64, 8 string, 9 array.
_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}

EXPS = re.compile(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight")
PROJ = ("gate", "up", "down")
ALIGN = 4096        # every record starts 4 KiB aligned (O_DIRECT reads), a pad part filling the stride
CHUNK = 32          # experts copied at a time: ~30-55 MB on Flash-Next


def _u(f, fmt):
    n = struct.calcsize(fmt)
    b = f.read(n)
    assert len(b) == n, "short GGUF header"
    return struct.unpack(fmt, b)[0]


def _str(f):
    return f.read(_u(f, "<Q")).decode("utf-8", "replace")


def _val(f, t, keep=64):
    if t in _SCALAR:
        return _u(f, _SCALAR[t])
    if t == 8:
        return _str(f)
    assert t == 9, f"GGUF value type {t}"
    et, n = _u(f, "<I"), _u(f, "<Q")
    if et in _SCALAR and n > keep:          # long numeric arrays: skipped, only their length kept
        f.read(n * struct.calcsize(_SCALAR[et]))
        return {"array_len": n}
    vals = [_val(f, et) for _ in range(n)]
    return vals if n <= keep else {"array_len": n, "head": vals[:4]}


def read_header(f):
    """(metadata, tensors, data_start) of the GGUF file object f at offset 0: tensors as dicts of name,
    ne (ggml order, innermost first), type, offset (absolute) and nbytes."""
    assert f.read(4) == b"GGUF", "not a GGUF file"
    version = _u(f, "<I")
    assert version in (2, 3), f"GGUF version {version}"
    nt, nkv = _u(f, "<Q"), _u(f, "<Q")
    meta = {"version": version}
    for _ in range(nkv):
        k = _str(f)
        meta[k] = _val(f, _u(f, "<I"))
    tensors = []
    for _ in range(nt):
        name = _str(f)
        ne = [_u(f, "<Q") for _ in range(_u(f, "<I"))]
        t, off = _u(f, "<I"), _u(f, "<Q")
        tensors.append({"name": name, "ne": ne, "type": t, "offset": off})
    align = meta.get("general.alignment", 32)
    pos = f.tell()
    data = pos + (-pos) % align
    for t in tensors:
        t["offset"] += data
        if t["type"] not in GGML:
            t["nbytes"] = None      # an unknown type: its size is unknown too; plan() refuses it if it is an expert
            continue
        _, be, bb = GGML[t["type"]]
        n = 1
        for d in t["ne"]:
            n *= d
        assert n % be == 0, (t["name"], t["ne"])
        t["nbytes"] = n // be * bb
    return meta, tensors, data


def plan(shards, layers=None):
    """[(index key, shard, tensor)] of every routed-expert tensor of layers below `layers` (all when None), by
    layer, then gate, up, down."""
    found = {}
    for sh in shards:
        with open(sh, "rb") as f:
            _, tensors, _ = read_header(f)
        for t in tensors:
            m = EXPS.fullmatch(t["name"])
            if m and (layers is None or int(m.group(1)) < layers):
                found[int(m.group(1)), PROJ.index(m.group(2))] = (sh, t)
    for L in sorted({L for L, _ in found}):
        assert all((L, p) in found for p in range(3)), f"layer {L}: gate, up and down not all present"
    for sh, t in found.values():
        name = GGML[t["type"]][0] if t["type"] in GGML else f"ggml type id {t['type']}"
        if name not in TYPES:
            raise ValueError(f"{Path(sh).name}: {t['name']} is {name}, which the engine does not decode "
                             f"(it decodes {', '.join(TYPES)})")
    return [(f"model.layers.{L}.mlp.switch_mlp.{PROJ[p]}_proj", *found[L, p]) for L, p in sorted(found)]


def _finite(rec, t):
    """Every block's scales (and float values) finite, the check llama.cpp runs on load (ggml_validate_row_data)."""
    assert valid(rec.reshape(-1, TYPES[t][2]), t), f"{t}: a non-finite scale"


def _drop(fd, off, n):
    """Out of the page cache, so the read-back comes from the drive (Linux; macOS has F_NOCACHE)."""
    if hasattr(os, "posix_fadvise"):
        os.posix_fadvise(fd, off, n, os.POSIX_FADV_DONTNEED)


def pack(shards, out_bin, log=print, layers=None, drop_src=True):
    """Write out_bin and its experts_index.json beside it; returns the index. Each chunk of records is read
    back from the drive and compared byte for byte with the GGUF's expert slices."""
    out_bin = Path(out_bin)
    index, written, t0 = {}, 0, time.time()
    todo = plan(shards, layers)
    with open(out_bin, "wb") as dst, open(out_bin, "rb") as back:
        for f in (dst, back):
            if hasattr(fcntl, "F_NOCACHE"):
                fcntl.fcntl(f.fileno(), fcntl.F_NOCACHE, 1)
        for i, (key, shard, t) in enumerate(todo):
            name = GGML[t["type"]][0]
            ne0, ne1, n_exp = t["ne"]
            nbytes = t["nbytes"] // n_exp
            stride = nbytes + (-nbytes) % ALIGN
            parts = [{"name": "gguf", "bytes": nbytes, "shape": [ne1, nbytes // ne1], "dtype": "U8",
                      "ggml_type": name, "ne": [ne0, ne1]}]
            if stride > nbytes:
                parts.append({"name": "pad", "bytes": stride - nbytes, "shape": [stride - nbytes], "dtype": "U8"})
            index[key] = {"offset": written, "stride": stride, "n_experts": n_exp, "parts": parts}
            with open(shard, "rb") as src:
                for e0 in range(0, n_exp, CHUNK):
                    n = min(CHUNK, n_exp - e0)
                    b = os.pread(src.fileno(), n * nbytes, t["offset"] + e0 * nbytes)
                    assert len(b) == n * nbytes, f"short read in {Path(shard).name}"
                    if drop_src:
                        _drop(src.fileno(), t["offset"] + e0 * nbytes, n * nbytes)     # others' cached pages stay
                    rec = np.zeros((n, stride), np.uint8)
                    rec[:, :nbytes] = np.frombuffer(b, np.uint8).reshape(n, nbytes)
                    _finite(rec[:, :nbytes], name)
                    dst.write(rec.data)
                    dst.flush()
                    os.fsync(dst.fileno())
                    _drop(dst.fileno(), written, n * stride)
                    got = os.pread(back.fileno(), n * stride, written)
                    assert got == rec.tobytes(), f"{key}: experts {e0}..{e0 + n - 1} read back differently"
                    written += n * stride
            log(f"  [{i + 1}/{len(todo)}] {key}: {name} {n_exp} x {stride:,} B, {written / 1e9:.2f} GB, "
                f"{written / 1e9 / max(time.time() - t0, 1e-9):.2f} GB/s")
    out_bin.with_name("experts_index.json").write_text(json.dumps(index, indent=1))
    return index


def main():
    ap = argparse.ArgumentParser(description="GGUF routed experts -> experts.bin + experts_index.json")
    ap.add_argument("shards", nargs="+", help="the GGUF shard files (all of a split, or those with the experts)")
    ap.add_argument("--out", required=True, help="the experts.bin to write; the index goes beside it")
    ap.add_argument("--layers", type=int, default=None,
                    help="only blk.N below this (an MTP layer past the trunk stays out)")
    ap.add_argument("--keep-src-cache", action="store_true",
                    help="leave the shards' pages cached (someone else's file that they read too)")
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    index = pack([Path(s).expanduser() for s in a.shards], out, log=lambda s: print(s, flush=True),
                 layers=a.layers, drop_src=not a.keep_src_cache)
    types = sorted({p["parts"][0]["ggml_type"] for p in index.values()})
    print(f"{len(index)} projections, types {types}, {out.stat().st_size / 1e9:.2f} GB, every record read back equal")


if __name__ == "__main__":
    main()
