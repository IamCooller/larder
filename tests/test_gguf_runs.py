#!/usr/bin/env python3
"""gguf_experts' run kernels (CUDA only) against the decode chain, bit for bit, h (gate/up) and the output (down):

  1. random Q4_0 / Q4_1 experts at the 35B's shapes (2048, 512) in its two mixes and a Q4_1 gate/up one, a K-quant mix
     at Flash-Next's (2560, 640), and (1440, 480) whose rows do not fill the last threadgroup; seven mixes at the
     35B's shapes covering the 17 types added after the first twelve;
  2. sorted rows in runs of T = 1..16 rows a slot, mixed runs of 1 to 20, and unsorted rows;
  3. x in bf16 (and f16, f32 on the 35B's mix), out in bf16 and f32, at the default tile and two others;
  4. gemv() itself takes the run kernels there, after its own check passed.

    python3 tests/test_gguf_runs.py      (on the PC; prints SKIP where MLX has no CUDA)
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mlx.core as mx       # noqa: E402

import gguf_experts as ge   # noqa: E402

E = 20
MIXES = [(("Q4_0", "Q4_0", "Q4_0"), 2048, 512), (("Q4_0", "Q4_0", "Q4_1"), 2048, 512),
         (("Q4_1", "Q4_1", "Q4_0"), 2048, 512), (("Q4_K", "Q4_K", "Q5_1"), 2560, 640),
         (("Q4_0", "Q8_0", "Q4_1"), 1440, 480)]
# The 17 types added after the first twelve, at the 35B's shapes, the 35B UD-Q4_K_XL's mix first.
MIXES += [(t, 2048, 512) for t in (("Q4_K", "Q4_K", "Q6_K"), ("Q2_K", "IQ2_XXS", "Q3_K"), ("IQ3_S", "IQ1_M", "MXFP4"),
                                   ("IQ2_XS", "IQ1_S", "BF16"), ("TQ1_0", "TQ2_0", "NVFP4"),
                                   ("Q1_0", "F8_E4M3", "F16"), ("F32", "Q6_K", "Q6_K"))]
TILES = ([1, 1, 3, 4], [4, 2, 16, 8])


def layouts(rng):
    """(name, slots): T rows a slot for T = 1..16, mixed sorted runs, and unsorted rows."""
    out = [(f"T{t}", np.repeat(rng.permutation(E)[:12], t)) for t in range(1, 17)]
    lens = rng.integers(1, 21, 40)
    out.append(("mixed", np.repeat(np.sort(rng.integers(0, E, len(lens))), lens)))
    out.append(("unsorted", rng.integers(0, E, 70)))
    return [(name, mx.array(s.astype(np.uint32))) for name, s in out]


def check(rng, types, hid, inter, x_dtypes, tiles):
    raw = [ge.random_experts(rng, E, r, c, t) for (r, c), t in zip(((inter, hid), (inter, hid), (hid, inter)), types)]
    w = tuple(mx.array(b) for b in raw)
    n = 0
    for name, slots in layouts(rng):
        x32 = mx.array(rng.standard_normal((slots.size, hid)).astype(np.float32))
        for xdt in x_dtypes:
            x = x32.astype(xdt)
            for odt in (mx.bfloat16, mx.float32):
                ge.RUNS = False
                h0, y0 = ge._chain(x, w, slots, 1, types, odt)
                assert ge._same_bits(y0, ge.gemv(x, w, slots, 1, types, odt)), "gemv() without RUNS is not the chain"
                ge.RUNS = True
                for tile in tiles:
                    ge.RUN_TILE[:] = tile
                    h1, y1 = ge._gemv_runs(x, w, slots, types, odt)
                    what = f"{'/'.join(types)} {hid}/{inter} {name} x {xdt} out {odt} tile {tile}"
                    assert ge._same_bits(h0, h1), f"h differs: {what}"
                    assert ge._same_bits(y0, y1), f"out differs: {what}"
                    n += 1
    return n


def main():
    if not ge._CUDA:
        print("SKIP: the run kernels are CUDA only (MLX here has no CUDA device)")
        return
    print(f"MLX {mx.__version__}, {mx.default_device()}, default tile {ge.RUN_TILE}")
    rng = np.random.default_rng(11)
    default = list(ge.RUN_TILE)
    try:
        for i, (types, hid, inter) in enumerate(MIXES):
            first = i == 0
            n = check(rng, types, hid, inter, (mx.bfloat16, mx.float16, mx.float32) if first else (mx.bfloat16,),
                      [default, *TILES] if i < 2 else [default])
            print(f"1-3. {'/'.join(types)} at {hid}/{inter}: h and out bit-equal to the chain in {n} cases")
    finally:
        ge.RUN_TILE[:] = default
    for types, hid, inter in MIXES[:2]:
        w = tuple(mx.array(ge.random_experts(rng, E, r, c, t))
                  for (r, c), t in zip(((inter, hid), (inter, hid), (hid, inter)), types))
        slots = mx.array(np.repeat(np.arange(E), 7).astype(np.uint32))
        x = mx.array(rng.standard_normal((slots.size, hid)).astype(np.float32)).astype(mx.bfloat16)
        y = ge.gemv(x, w, slots, 1, types, mx.bfloat16)
        assert ge._RUNS_OK.get((hid, inter, types, str(x.dtype), str(mx.bfloat16))), "gemv() did not take the run kernels"
        assert ge._same_bits(y, ge._chain(x, w, slots, 1, types, mx.bfloat16)[1])
    print("4. gemv() takes the run kernels at one slot a row, bit-equal to the chain")
    print("all GGUF run-kernel checks passed")


if __name__ == "__main__":
    main()
