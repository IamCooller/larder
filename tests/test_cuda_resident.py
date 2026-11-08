#!/usr/bin/env python3
"""MOE_RESIDENT without a model or a GPU.

  1. The agent's tier filled by hand (native/moe_pipe_cuda.cpp built for this machine with test_pipe_staging_emu's
     libcuda stand-in): tier_fill hands out each free slot once and 0 for an expert held or a full tier, tier_find
     gives the same addresses, and a decode job then copies those experts from the tier (tier hits, the file's bytes
     in their slots), the store's victims going into the slots the hits freed;
  2. a prefill group's rows (_group_rows): held quota rows kept, holes of a dense span on slot 0, ring rows and absent
     experts listed as the ones to fetch;
  3. the resident tier's rule (TierGovernor at RESIDENT_FLOOR, all of the room): full while MemAvailable is above the
     floor, giving back everything another program takes below it at the next reading;
  4. the launcher sets MOE_RESIDENT on CUDA for a model whose experts fit VRAM and free RAM (the 35B) and not for one
     far larger (Flash-Next), nor with little free RAM;
  5. MOE_DENSE_VRAM's copy (dense_to_device): every bit kept (-0.0 and NaN payloads too), the experts and a streamed
     embedding table left alone, a tied array copied once.

    perl -e 'alarm shift; exec @ARGV' 280 python3 tests/test_cuda_resident.py
"""
import ctypes
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))


def tier_child(d):
    import test_pipe_staging_emu as emu
    lib = ctypes.CDLL(str(d / "libagent.so"))
    cu = ctypes.CDLL("libcuda.so.1")
    v, I = ctypes.c_void_p, ctypes.c_int
    lib.mpc_error.restype = ctypes.c_char_p
    lib.mpc_block.restype = v
    lib.mpc_agent_new.restype = v
    lib.mpc_agent_new.argtypes = [I, I, I]
    lib.mpc_agent_tier.argtypes = [v, ctypes.c_uint64, I, I]
    lib.mpc_agent_drain.argtypes = [v, ctypes.c_double]
    lib.mpc_agent_arm.argtypes = [v]
    lib.mpc_agent_stats.argtypes = [v, v]
    lib.mpc_set_layer.argtypes = [v, I, v, I, I, I, v, v, v, v, v, v, I, I]
    lib.mpc_set_ring.argtypes = [v, I, I, I, v, I]
    lib.mpc_tier_fill.argtypes = [v, I, I, I, ctypes.POINTER(ctypes.c_uint64)]
    lib.mpc_tier_find.argtypes = [v, I, v, I, v]
    cu.emu_ld.restype = ctypes.c_uint
    cu.emu_ld.argtypes = [v]
    cu.emu_st.argtypes = [v, ctypes.c_uint]
    assert lib.mpc_init() == 0, lib.mpc_error()
    data = np.random.default_rng(5).integers(0, 256, emu.L * emu.E * sum(emu.STRIDE), dtype=np.uint8)
    (d / "experts.bin").write_bytes(data.tobytes())
    fd = os.open(d / "experts.bin", os.O_RDONLY)
    r = emu.Run(lib, cu, fd, data, True, 0, "")         # a store layout, its tier of 6 slots, staging off

    def fill(layer, e):
        host = ctypes.c_uint64()
        assert lib.mpc_tier_fill(r.a, layer, emu.E, e, ctypes.byref(host)) == 0, lib.mpc_error()
        return host.value

    filled = {}
    for e in range(6):
        a = fill(0, e)
        assert a and a not in filled.values(), f"expert {e}: no fresh slot ({a})"
        filled[e] = a
        for p in range(3):          # the expert's records back to back, as the agent saves an eviction
            at = r.off(0, p) + e * emu.STRIDE[p]
            ctypes.memmove(a + sum(emu.STRIDE[:p]), data[at:].ctypes.data, emu.STRIDE[p])
    assert fill(0, 0) == 0 and fill(0, 7) == 0, "a held expert or a full tier must give 0"
    ids = np.arange(8, dtype=np.int32)
    out = np.zeros(8, dtype=np.uint64)
    assert lib.mpc_tier_find(r.a, 0, ids.ctypes.data, 8, out.ctypes.data) == 0, lib.mpc_error()
    assert out.tolist() == [filled[e] for e in range(6)] + [0, 0], out
    st = (ctypes.c_uint64 * 9)()
    lib.mpc_agent_stats(r.a, st)
    lib.mpc_agent_arm(r.a)
    r.job(0, np.array([0, 1, 2, 3], np.uint32), None)       # the quota takes 3, the ring the 4th: all from the tier
    r.job(0, np.array([4, 5, 6, 7], np.uint32), None)       # 4, 5 from the tier, 6, 7 from the file, victims put back
    assert lib.mpc_agent_drain(r.a, 10.0)
    lib.mpc_agent_stats(r.a, st)
    hits, puts = st[6], st[7]
    assert hits == 6, f"tier hits {hits}, want 6"
    assert puts >= 1, f"no victim went into the freed tier slots ({puts})"
    print(f"  1. tier filled by hand: 6 slots once each, held and full give 0, find agrees; a job's hits {hits}, "
          f"victims put {puts}")


def check_tier():
    import test_pipe_staging_emu as emu
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        emu.build(d)
        env = dict(os.environ, RESIDENT_EMU_DIR=t, MOE_PIPE_CHUNK="512")
        env["DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"] = t
        rc = subprocess.run([sys.executable, __file__], env=env, timeout=200).returncode
        assert rc == 0, "the tier child failed"


def check_rows():
    import stream_fast as sf
    slot = np.full(16, -1, dtype=np.int32)
    slot[[3, 5, 6, 9]] = [0, 2, 40, 1]              # 6 sits in a ring row (pool 32), 4 and 8 are held nowhere
    ids = np.array([3, 4, 6, 8, 9])
    want = np.arange(3, 10)                          # dense span 3..9, holes 5 and 7
    sl, miss = sf._group_rows(want, ids, slot, 32)
    assert sl.tolist() == [0, -1, 0, 40, 0, -1, 1], sl
    assert want[miss].tolist() == [4, 6, 8], want[miss]
    sl, miss = sf._group_rows(ids, ids, slot, 32)    # not dense: the ids themselves
    assert sl.tolist() == [0, -1, 40, -1, 1] and ids[miss].tolist() == [4, 6, 8]
    print("  2. group rows: held quota rows kept, holes on slot 0, ring and absent experts fetched")


def check_rule():
    import stream_fast as sf
    GB = 1e9
    state = {"avail": 6.0 * GB}
    g = sf.TierGovernor(10, 10, 0.5 * GB, sf.RESIDENT_FLOOR, read=lambda: (state["avail"], 0, 0), div=1)
    assert g.target(6.0 * GB) == 10                  # 6 + 5 - 5 = 6 GB of room: capped at its 10 slots
    assert g.tick(idle=True) is None
    state["avail"] = 3.0 * GB                        # another program took 3 GB
    want = g.tick(idle=True)
    assert want == 6, want                           # 3 + 5 - 5 = 3 GB: 6 slots, MemAvailable back at the floor
    old = sf.TierGovernor(10, 10, 0.5 * GB, 2.5e9, read=lambda: (state["avail"], 0, 0))
    assert old.target(6.0 * GB) == 5                 # the install rule, a third, unchanged
    print(f"  3. resident tier rule: full above the {sf.RESIDENT_FLOOR / GB:g} GB floor, 3 GB taken -> 10 to 6 slots")


def check_launcher():
    import machine

    def prof(avail):
        return {"backend": "cuda", "ram": 15 * 2**30, "gpu_mem": 17.1e9, "working_set": 15e9, "gpu_bw": 448,
                "now": {"avail_gb": avail}}
    q35 = {"kind": "moe", "dense_bytes": 1.39e9, "read_bytes": 1.39e9, "moe_layers": 40, "experts": 256,
           "top_k": 8, "expert_bytes": 1835008, "recipe": "store", "gated": False, "native": True,
           "total_bytes": 20e9}
    fn = {**q35, "moe_layers": 48, "experts": 512, "expert_bytes": 2480000, "dense_bytes": 5.5e9}
    real = machine.shutil.which
    machine.shutil.which = lambda name: "/usr/bin/" + name
    try:
        on = machine.settings(prof(12.3), q35)
        assert on["env"].get("MOE_RESIDENT") == "1", on
        assert on["args"][:4] == ["--mem-peak-gb", "7", "--mem-peak-max-gb", "16"], on["args"]
        assert "MOE_RESIDENT" not in machine.settings(prof(12.3), fn)["env"]
        assert "MOE_RESIDENT" not in machine.settings(prof(4.0), q35)["env"]
    finally:
        machine.shutil.which = real
    print("  4. launcher: MOE_RESIDENT for the 35B on CUDA with 12.3 GB free, not for Flash-Next or with 4 GB free")


def check_dense_copy():
    import mlx.core as mx
    import mlx.nn as nn
    import stream_fast

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(16, 64)
            self.proj = nn.Linear(64, 64)
            self.switch_mlp = nn.Linear(64, 64)
            self.norm = nn.RMSNorm(8)

    b = Block()
    nn.quantize(b, group_size=32, bits=4, class_predicate=lambda p, m: p == "proj")
    raw = np.array([0x8000, 0x7FC1, 0xFFC3, 0x7F80, 0x3F80, 0x0001, 0xBF80, 0x0000], dtype=np.uint16)
    b.norm.weight = mx.array(raw).view(mx.bfloat16)      # -0.0, NaN payloads, inf, a subnormal
    b.proj.bias = b.norm.weight                           # tied
    mx.eval(b.parameters())
    before = {k: np.array(v.view(mx.uint8)) for k, v in nn.utils.tree_flatten(b.parameters())}
    keep = {k: v for k, v in nn.utils.tree_flatten(b.parameters()) if k.startswith(("switch_mlp", "embed_tokens"))}
    moved = stream_fast.dense_to_device(b)
    after = dict(nn.utils.tree_flatten(b.parameters()))
    assert all(np.array_equal(before[k], np.array(v.view(mx.uint8))) for k, v in after.items()), "bits moved"
    assert all(after[k] is v for k, v in keep.items()), "experts or the streamed table copied"
    assert b.norm.weight is b.proj.bias and b.norm.weight is not keep.get("norm.weight")
    assert moved == sum(v.nbytes for k, v in after.items() if k.startswith("proj")), moved
    print(f"  5. dense copy: {moved} bytes, every bit kept, experts and the streamed table left, the tied array once")


def main():
    if os.environ.get("RESIDENT_EMU_DIR"):
        tier_child(Path(os.environ["RESIDENT_EMU_DIR"]))
        return
    check_tier()
    check_rows()
    check_rule()
    check_launcher()
    check_dense_copy()
    print("test_cuda_resident: ok")


if __name__ == "__main__":
    main()
