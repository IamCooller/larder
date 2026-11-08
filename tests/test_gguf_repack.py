#!/usr/bin/env python3
"""tools/gguf_repack.py on a synthetic two-shard GGUF written here (GGUF v3, 32-byte alignment): two MoE layers,
one 4 KiB-aligned (Q4_K / Q4_K / Q5_1) and one not (IQ3_XXS / IQ3_XXS / Q5_0, padded), with non-expert tensors
between. Every record must be its expert's bytes, 4 KiB aligned, the pad zero, the index right; a missing
projection and a non-finite scale must be refused.

    perl -e 'alarm shift; exec @ARGV' 280 python3 tests/test_gguf_repack.py
"""
import json
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import gguf_ref as ge      # noqa: E402
import gguf_repack             # noqa: E402

HID, INTER, E = 512, 64, 6
LAYERS = [("Q4_K", "Q4_K", "Q5_1"), ("IQ3_XXS", "IQ3_XXS", "Q5_0")]


def _s(x):
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


def write_gguf(path, kv, tensors):
    """A GGUF v3 file: kv [(key, str or uint32)], tensors [(name, ne, type id, raw bytes)]."""
    h = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv))
    for k, v in kv:
        h += _s(k) + (struct.pack("<I", 8) + _s(v) if isinstance(v, str) else struct.pack("<II", 4, v))
    data, off = b"", 0
    for name, ne, t, raw in tensors:
        h += _s(name) + struct.pack("<I", len(ne)) + b"".join(struct.pack("<Q", d) for d in ne)
        h += struct.pack("<IQ", t, off)
        raw += b"\0" * ((-len(raw)) % 32)
        data += raw
        off += len(raw)
    path.write_bytes(h + b"\0" * ((-len(h)) % 32) + data)


def blocks(rng, t, n_weights):
    """Random blocks of type t with finite f16 scales."""
    _, be, bb = ge.TYPES[t]
    b = rng.integers(0, 256, (n_weights // be, bb), dtype=np.uint8)
    for at in gguf_repack.HALVES[t]:
        b[:, at:at + 2] = (rng.standard_normal((len(b), 1)) * 0.01).astype(np.float16).view(np.uint8)
    return b.tobytes()


def expert_tensors(rng):
    out = []
    for L, ts in enumerate(LAYERS):
        for p, t in zip(("gate", "up", "down"), ts):
            ne = [HID, INTER, E] if p != "down" else [INTER, HID, E]
            out.append((f"blk.{L}.ffn_{p}_exps.weight", ne, ge.TYPES[t][0], blocks(rng, t, HID * INTER * E)))
    return out


def main():
    rng = np.random.default_rng(1)
    exps = expert_tensors(rng)
    other = [("token_embd.weight", [16, 4], 0, rng.standard_normal(64).astype(np.float32).tobytes()),
             ("blk.0.ffn_gate_shexp.weight", [HID, INTER], 12, blocks(rng, "Q4_K", HID * INTER))]
    kv = [("general.architecture", "qwen4exp"), ("general.alignment", 32)]
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        s1, s2 = d / "m-00001-of-00002.gguf", d / "m-00002-of-00002.gguf"
        write_gguf(s1, kv, [other[0]] + exps[:4])
        write_gguf(s2, kv, [other[1]] + exps[4:])
        index = gguf_repack.pack([s1, s2], d / "experts.bin", log=lambda *a, **k: None)
        idx = json.loads((d / "experts_index.json").read_text())
        assert idx == index and len(idx) == 6, sorted(idx)
        binb = (d / "experts.bin").read_bytes()
        for name, ne, tid, raw in exps:
            L, p = int(name.split(".")[1]), name.split("_")[1]
            m = idx[f"model.layers.{L}.mlp.switch_mlp.{p}_proj"]
            part = m["parts"][0]
            nb = len(raw) // E
            assert part["ggml_type"] == ge.NAME_OF[tid] and part["bytes"] == nb and part["ne"] == ne[:2], m
            assert m["stride"] % 4096 == 0 and m["offset"] % 4096 == 0 and m["n_experts"] == E, m
            assert sum(q["bytes"] for q in m["parts"]) == m["stride"], m
            for e in range(E):
                rec = binb[m["offset"] + e * m["stride"]:m["offset"] + (e + 1) * m["stride"]]
                assert rec[:nb] == raw[e * nb:(e + 1) * nb], (name, e)
                assert rec[nb:] == b"\0" * (m["stride"] - nb), (name, e)
            print(f"{name}: {part['ggml_type']} {E} records of {nb:,} B at stride {m['stride']:,}, equal")
        assert len(binb) == sum(m["stride"] * E for m in idx.values())

        # A layer without its down projection, and a NaN scale, are refused.
        write_gguf(s2, kv, [other[1]] + exps[4:5])
        try:
            gguf_repack.plan([s1, s2])
            raise SystemExit("a missing projection was accepted")
        except AssertionError as e:
            print(f"refused: {e}")
        bad = bytearray(exps[0][3])
        bad[0:2] = np.array([np.nan], np.float16).tobytes()
        write_gguf(s1, kv, [(exps[0][0], exps[0][1], exps[0][2], bytes(bad))] + exps[1:6])
        try:
            gguf_repack.pack([s1], d / "bad.bin", log=lambda *a, **k: None)
            raise SystemExit("a NaN scale was accepted")
        except AssertionError as e:
            print(f"refused: {e}")
    print("all GGUF repack checks passed")


if __name__ == "__main__":
    main()
