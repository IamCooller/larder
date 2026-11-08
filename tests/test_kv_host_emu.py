#!/usr/bin/env python3
"""kv_host (MOE_KV_HOST) without a GPU: the homed cache class run against the flag-off cache, its five CUDA
kernels replaced by NumPy transliterations of their index arithmetic, line for line, over fake pinned memory.

  1. mlx_lm's KVCache capacity rule (kv_host._grown) over random passes and trims;
  2. the class below its line is the flag-off cache: same arrays, offsets, state, nbytes, trims;
  3. above it, with chunks of 64 positions and an LRU of 2 top-ks: a prefill crossing the line (the VRAM buffer
     becomes the pass's scratch), staged passes, 67 one-row passes on the picks (hits, misses, evictions, the
     newest row inside a picked block), trims, a multi-row pass, state, a restore, a trim under the line, rows
     under 16 bytes and a refused allocation: every gathered byte, view, capacity, whole buffer (stale rows past
     the offset too) and state equal to the flag-off path's, the LRU's maps consistent and every resident block
     equal to its rows after every pass; 2 and 3 on mlx_lm's two-part state and on the PC's three-part one;
  4. the budget's accounting: attn_bytes' scratch term, MemBudget.close net of the LRU, _copy and the pass
     evaluation never staging;
  5. the kernel text compiles as C++ inside a stand-in of MLX's CUDA wrapper.

Laptop (Metal or CPU, a few seconds): perl -e 'alarm shift; exec @ARGV' 280 python tests/test_kv_host_emu.py
"""
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mlx.core as mx
from mlx_lm.models.cache import KVCache

import kv_host as kh

try:
    from mlx_lm.models import qwen4_exp as q4
except ImportError:
    print("test_kv_host_emu: skipped (mlx_lm without qwen4_exp)")
    sys.exit(0)

HK, D, CR, NSEL = 2, 32, 4, 16
DT = mx.float16


# ------------------------------------------------------------------------- fake pinned memory
class FakeCu:
    """cuMemHostAlloc's stand-in: chunks of numpy bytes, host and device address the same number."""

    def __init__(self):
        self.mem, self.next, self.refuse = {}, 1 << 40, False

    def alloc(self, nbytes):
        if self.refuse:
            return None
        a = self.next
        self.next += nbytes + 4096
        self.mem[a] = np.zeros(nbytes, np.uint8)
        return a, a


CU = FakeCu()


def row(tbl, pos, kv, h, d, hk, logc):
    """kvh_row: the uint16 view of a chunk and the element offset of row pos."""
    i = pos & ((1 << logc) - 1)
    off = ((((kv * hk + h) << logc) + i) * d) * 2
    return CU.mem[int(tbl[pos >> logc])].view(np.uint16), off // 2


def u16(a):
    return np.array(a).view(np.uint16)


def f16(a):
    return mx.array(np.ascontiguousarray(a).view(np.float16))


def inplace(a):
    """A writable view of an evaluated MLX array, how the kernels' const casts write their inputs."""
    mx.eval(a)
    return np.array(a, copy=False)


def emu_put(inputs, template, **_):
    k, v, tbl, p0, dep = inputs
    t = dict(template)
    tb, base = np.array(tbl), p0.shape[0] - 1
    for kv, src in ((0, u16(k)), (1, u16(v))):
        for h in range(t["HK"]):
            for r in range(k.shape[2]):
                buf, o = row(tb, base + r, kv, h, t["D"], t["HK"], t["LOGC"])
                buf[o:o + t["D"]] = src[0, h, r]
    return [mx.array([1], dtype=mx.int32)]


def emu_stage(inputs, template, **_):
    tbl, npa, cp, dep = inputs
    t = dict(template)
    tb, prev, cap = np.array(tbl), npa.shape[0] - 1, cp.shape[0] - 1
    out = [np.zeros((1, t["HK"], cap, t["D"]), np.uint16) for _ in range(2)]
    for kv in range(2):
        for h in range(t["HK"]):
            for r in range(cap):
                if r < prev:
                    buf, o = row(tb, r, kv, h, t["D"], t["HK"], t["LOGC"])
                    out[kv][0, h, r] = buf[o:o + t["D"]]
    return [f16(out[0]), f16(out[1]), mx.array([1], dtype=mx.int32)]


def emu_plan(inputs, template, **_):
    tbi, som, bom, stp, ref, meta, ldep = inputs
    t = dict(template)
    nsel, ns, nt = t["NSEL"], t["NS"], t["NT"]
    so, bo, st, rf, mt = (inplace(a) for a in (som, bom, stp, ref, meta))
    tb = np.array(tbi).astype(np.int64)
    tt = int(mt[1]) + 1
    j = np.arange(nt)
    b = np.full(nt, -1)
    b[:nsel] = tb
    s = np.full(nt, -1)
    s[:nsel] = so[tb]
    hit = s >= 0
    st[s[hit]] = tt
    rf[s[hit]] = 1
    miss = (j < nsel) & (s < 0)
    rank, M = np.cumsum(miss) - miss, int(miss.sum())
    hand, found, scanned = int(mt[0]), 0, 0
    vict = np.zeros(nsel, np.int64)
    while found < M and scanned < 2 * ns + nt:
        x = (hand + scanned + j) % ns
        prot = st[x] == tt
        el = ~prot & (rf[x] == 0)
        e, E = np.cumsum(el) - el, int(el.sum())
        take = min(E, M - found)
        cut = nt - 1
        last = el & (take == M - found) & (e == take - 1)
        if last.any():
            cut = int(j[last][0])
        pick = el & (e < take)
        vict[found + e[pick]] = x[pick]
        st[x[pick]] = tt
        clear = (j <= cut) & ~prot & ~el
        rf[x[clear]] = 0
        found += take
        scanned += cut + 1
    sl = np.zeros(nsel, np.int32)
    for jj in range(nsel):
        if s[jj] >= 0:
            sl[jj] = s[jj]
        elif rank[jj] < found:
            x = int(vict[rank[jj]])
            old = int(bo[x])
            if old >= 0:
                so[old] = -1
            bo[x], so[b[jj]], rf[x] = b[jj], x, 0
            sl[jj] = -2 - x
        else:
            sl[jj] = -1
    mt[0], mt[1] = (hand + scanned) % ns, tt
    mt[2] += nsel - M
    mt[3] += M
    return [mx.array(sl)]


def emu_gather(inputs, template, **_):
    pool, tbl, tbi, sli, kn, vn, npa, dep = inputs
    t = dict(template)
    d, cr, nsel, hk, logc = t["D"], t["CR"], t["NSEL"], t["HK"], t["LOGC"]
    P = inplace(pool).view(np.uint16)
    tbt, tb, sl = np.array(tbl), np.array(tbi).astype(np.int64), np.array(sli)
    n = npa.shape[0]
    own0 = n // cr * cr
    m = nsel * cr + n - own0
    new = (u16(kn), u16(vn))
    out = [np.zeros((1, hk, m, d), np.uint16) for _ in range(2)]
    for kv in range(2):
        for h in range(hk):
            for r in range(m + 1):
                slot = vic = -1
                if r == m:
                    pos = n - 1
                elif r >= nsel * cr:
                    pos = own0 + r - nsel * cr
                else:
                    jj = r // cr
                    pos = int(tb[jj]) * cr + r % cr
                    c = int(sl[jj])
                    slot = c if c >= 0 else -1
                    vic = -2 - c if c <= -2 else -1
                if pos == n - 1:
                    x = new[kv][0, h, 0].copy()
                elif slot >= 0:
                    x = P[slot, kv, h, r % cr].copy()
                else:
                    buf, o = row(tbt, pos, kv, h, d, hk, logc)
                    x = buf[o:o + d].copy()
                if r == m:
                    buf, o = row(tbt, pos, kv, h, d, hk, logc)
                    buf[o:o + d] = x
                else:
                    out[kv][0, h, r] = x
                if vic >= 0:
                    P[vic, kv, h, r % cr] = x
    return [f16(out[0]), f16(out[1]), mx.array([1], dtype=mx.int32)]


def emu_inval(inputs, template, **_):
    som, bom, ref, lim, ldep = inputs
    so, bo, rf = (inplace(a) for a in (som, bom, ref))
    L = lim.shape[0] - 1
    for x in range(dict(template)["NS"]):
        b = int(bo[x])
        if b >= 0 and b >= L:
            so[b], bo[x], rf[x] = -1, -1, 0
    return [mx.array([1], dtype=mx.int32)]


EMU = {"put": emu_put, "stage": emu_stage, "plan": emu_plan, "gather": emu_gather, "inval": emu_inval}


# --------------------------------------------------------------------------------- references
def ref_gather(k, v, tb, n):
    """_QSA_GATHER's rows over the flag-off views k, v [1, HK, n, D]: the picked blocks in order, then the trailing
    partial block."""
    own0 = n // CR * CR
    pos = [int(b) * CR + i for b in np.array(tb) for i in range(CR)] + list(range(own0, n))
    return [u16(a)[:, :, pos] for a in (k, v)]


def rand_kv(rng, s):
    return [f16(rng.integers(0, 0x7BFF, (1, HK, s, D), dtype=np.uint16)) for _ in range(2)]


def same(a, b, what):
    assert a.shape == b.shape, (what, a.shape, b.shape)
    assert np.array_equal(u16(a), u16(b)), what


def same_state(on, off, what):
    sa, sb = on.state, off.state
    assert len(sa) == len(sb), (what, len(sa), len(sb))
    for x, y in zip(sa, sb):
        if isinstance(x, mx.array):
            assert x.shape == y.shape and bool(mx.array_equal(x, y)), f"{what}: state differs"
        else:
            assert x == y, (what, x, y)


class Three(q4._AttnCache):
    """_AttnCache over an mlx_lm whose KVCache.state is the whole buffer and the offset, as the PC's
    (tests/test_kv_compat.py)."""

    @property
    def state(self):
        return self.keys, self.values, self.offset, self.indexer.state

    @state.setter
    def state(self, v):
        self.keys, self.values, self.offset, self.indexer.state = v


def check_lru(c, ref):
    """The maps agree with each other, and every resident block holds its rows."""
    lru = c.lru
    if lru is None:
        return
    so, bo = np.array(lru.so), np.array(lru.bo)
    res = np.flatnonzero(bo >= 0)
    assert (so[bo[res]] == res).all(), "slot_of and block_of disagree"
    assert (so >= 0).sum() == res.size, "a block held twice or a stale slot_of"
    P = u16(lru.pool)
    K, V = u16(ref.keys), u16(ref.values)
    for x in res:
        b = int(bo[x])
        assert b * CR + CR <= ref.offset, f"slot {x} holds block {b} past the rows ({ref.offset})"
        for kv, R in ((0, K), (1, V)):
            assert np.array_equal(P[x, kv], R[0, :, b * CR:b * CR + CR]), f"slot {x} block {b} is stale"


# ---------------------------------------------------------------------------------- sections
def cap_rule(rng):
    for _ in range(40):
        c, cap, off = KVCache(), None, 0
        for _ in range(12):
            if off and rng.random() < 0.25:
                c.trim(int(rng.integers(1, off + 1)))
                off = c.offset
                continue
            s = int(rng.choice([1, 1, 3, 100, 255, 256, 300, 700]))
            want = kh._grown(cap, off, s, c.step)
            c.update_and_fetch(*[mx.zeros((1, 1, s, 8)) for _ in range(2)])
            assert c.keys.shape[2] == want, (cap, off, s, c.keys.shape[2], want)
            cap, off = want, c.offset
    print("  1. capacity rule: mlx_lm's KVCache over 40 random histories")


def vram_mode(cls, base, rng):
    kh.H = 1 << 62
    on, off = cls(), base()
    for step in range(30):
        if step % 7 == 6:
            n = int(rng.integers(1, on.offset + 1))
            assert on.trim(n) == off.trim(n)
        s = int(rng.choice([1, 2, 50, 300]))
        k, v = rand_kv(rng, s)
        a, b = on.update_and_fetch(k, v), off.update_and_fetch(k, v)
        for x, y in zip(a, b):
            same(x, y, f"vram step {step}")
        assert on.offset == off.offset and on.nbytes == off.nbytes and not on.kv_host
        same(on.keys, off.keys, "keys")
        same_state(on, off, f"vram state {step}")
    c2, d2 = cls(), base()
    c2.state, d2.state = off.state, off.state
    same_state(c2, d2, "state round trip")
    assert not c2.empty() and cls().empty() == base().empty()
    print("  2. under the line: arrays, offsets, nbytes, trims and state as the flag-off cache's")


def homed(cls, base, rng):
    kh.H, kh.LOGC, kh.LRU_MULT, kh.NSEL, kh.CR = 300, 6, 2, NSEL, CR
    kh._DUMMY[0] = None
    kh.take_added()
    on, off = cls(), base()
    counts = {"decode": 0, "newest": 0, "trim": 0}

    def multi(s, what):
        k, v = rand_kv(rng, s)
        a, b = on.update_and_fetch(k, v), off.update_and_fetch(k, v)
        for x, y in zip(a, b):
            same(x, y, what)
        assert on.offset == off.offset
        if on.kv_host and on.scratch is not None:
            assert on.scratch.keys.shape[2] == off.keys.shape[2] == on.cap, "capacity"
        return a

    def decode(prev_picks):
        n = off.offset + 1
        nb = n // CR
        keep = {b for b in prev_picks if b < nb and rng.random() < 0.6}
        if n % CR == 0 and rng.random() < 0.5:
            keep.add(nb - 1)            # the block the new row completes
            counts["newest"] += 1
        rest = np.setdiff1d(np.arange(nb), sorted(keep))
        picks = np.sort(np.concatenate([sorted(keep), rng.choice(rest, NSEL - len(keep), replace=False)])
                        ).astype(np.uint32)
        assert np.unique(picks).size == NSEL
        k, v = rand_kv(rng, 1)
        tb = mx.array(picks)
        g = on.decode(k, v, tb, CR)
        rk, rv = off.update_and_fetch(k, v)
        assert g is not None, "a homed cache declined a pass on the picks"
        want = ref_gather(rk, rv, picks, n)
        for x, y, w in zip(g, want, "KV"):
            assert np.array_equal(u16(x), y), f"gathered {w} differs at {n}"
        assert on.offset == off.offset and on.cap == off.keys.shape[2]
        check_lru(on, off)
        counts["decode"] += 1
        return list(picks)

    for s in (128, 128):
        multi(s, "prefill under the line")
    assert not on.kv_host
    multi(100, "prefill crossing the line")
    assert on.kv_host and on.scratch is not None, "the crossing pass's VRAM buffer is its scratch"
    assert on.lru is not None and kh.take_added() == on.lru.nbytes, "the LRU comes with the home, counted once"
    multi(64, "the scratch reused")
    kh.drop_live()
    multi(70, "a staged pass")
    other_on, other_off = cls(), base()
    for s in (200, 150):
        k, v = rand_kv(rng, s)
        for x, y in zip(other_on.update_and_fetch(k, v), other_off.update_and_fetch(k, v)):
            same(x, y, "second layer")
    assert on.scratch is None and other_on.scratch is not None, "one scratch alive at a time"
    picks = []
    for _ in range(40):
        picks = decode(picks)
    for n in (7, 3):
        assert on.trim(n) == off.trim(n)
        counts["trim"] += 1
        same_state(on, off, "state after a trim")       # the trimmed rows still in the buffer
        for _ in range(10):
            picks = decode(picks)
    multi(3, "a multi-row pass after the picks")
    same_state(on, off, "state after a staged pass over trimmed rows")
    picks = decode(picks)
    same_state(on, off, "state")
    same(on.keys, off.keys, "the whole buffer read from the home")
    same(on.values, off.values, "the whole buffer read from the home")
    rep = kh.report()
    assert rep["hits"] > 0 and rep["misses"] > 0 and rep["hits"] + rep["misses"] == NSEL * counts["decode"], rep
    assert rep["homed_layers"] == 2 and rep["lru_blocks"] == 2 * NSEL, rep

    # A restore puts the cache back in VRAM, and the next pass on the picks homes it again.
    r_on, r_off = cls(), base()
    r_on.state, r_off.state = off.state, off.state
    assert not r_on.kv_host
    on, off = r_on, r_off
    picks = []
    for _ in range(6):
        picks = decode(picks)
    assert on.kv_host
    same_state(on, off, "state after a restore homed again")

    # A trim under the line takes the cache back to VRAM at its next pass.
    n = off.offset - 200
    on.trim(n), off.trim(n)
    multi(1, "a pass under the line after a trim")
    assert not on.kv_host and on.offset == off.offset == 201
    same(on.keys, off.keys, "back in VRAM")
    same_state(on, off, "state back in VRAM")

    # Rows under 16 bytes (the kernels copy whole 16-byte words) stay in VRAM.
    s_on, s_off = cls(), base()
    for s in (250, 100):
        z = mx.zeros((1, 1, s, 4), dtype=DT)
        for x, y in zip(s_on.update_and_fetch(z, z), s_off.update_and_fetch(z, z)):
            same(x, y, "8-byte rows")
    assert not s_on.kv_host

    # Pinned RAM refused (and no freed chunk to reuse): the cache stays in VRAM, the same bytes, said once.
    CU.refuse = True
    kh._FREE.clear()
    f_on, f_off = cls(), base()
    on, off = f_on, f_off
    for s in (250, 100, 1):
        multi(s, "refused")
    assert not f_on.kv_host and f_on.refused and kh.STATS["refused"] >= 1
    CU.refuse = False
    print(f"  3. homed, {len(off.state)}-part state: {counts['decode']} passes on the picks ({rep['hits']} hits, "
          f"{rep['misses']} misses in the first cache), {counts['newest']} with the newest row in a picked block, "
          f"{counts['trim']} trims, staged and scratch passes, state, restore, a trim under the line, 8-byte rows, a "
          f"refusal: every byte, buffer and state the flag-off path's, the LRU consistent after every pass")


def budget(cls, base, rng):
    import stream_fast as sf
    kh.H, kh.ROW_BYTES = 1000, 2 * HK * D * 2
    pf = sf._PF_ATTN
    try:
        sf._PF_ATTN = (10.0, 2.0, 100.0, 5.0, 64)
        sf.kv_host, flag = kh, sf._KV_HOST
        sf._KV_HOST = False
        offs = [sf.attn_bytes(t, n) for t, n in ((128, 500), (128, 5000), (0, 5000))]
        sf._KV_HOST = True
        ons = [sf.attn_bytes(t, n) for t, n in ((128, 500), (128, 5000), (0, 5000))]
        assert ons[0] == offs[0] and ons[2] == offs[2] == 0.0, (ons, offs)
        assert ons[1] == offs[1] + 2 * kh.ROW_BYTES * 5000, (ons, offs)

        # MemBudget.close: a decode pass that allocated an LRU measures what it grew without it.
        def bud():
            b = object.__new__(sf.MemBudget)
            b.__dict__.update(_pass=(1, False, False, None, None, 0.0, False, 1000, 64), _top=5e8, _have=0,
                              _held0=1e8, _foot0=0, rate=1e3, _mine=(0.0, 0), _keep_w=0, keep=0.0, _calib=None,
                              _saved=None, state=0.0, _ticks=1, _floor_n=0, _host_top=0, _host_floor=0,
                              _lm_seen={}, room_ok=None, layer_major=False, _lm_rows={})
            return b
        mx.reset_peak_memory()
        kh._ADDED[0] = 0
        a = bud()
        a.close()
        kh._ADDED[0] = int(1.5e8)
        b = bud()
        b.close()
        assert abs((a.keep - b.keep) - 1.5e8) < 1, (a.keep, b.keep)
        assert kh._ADDED[0] == 0

        # A homed cache: _copy sees its VRAM only, and forcing a pass never stages it.
        kh.H, kh.LOGC = 300, 6
        c = cls()
        for s in (200, 200):
            c.update_and_fetch(*rand_kv(rng, s))
        picks = mx.array(np.arange(NSEL, dtype=np.uint32))
        c.decode(*rand_kv(rng, 1), picks, CR)
        assert c.kv_host and sf._copy([c]) == c.lru.nbytes + c.step * c.lru.nbytes / c.offset
        staged = kh.STATS["staged"]
        sf._eval_caches([c])
        mx.eval(sf.cache_state(c))
        assert kh.STATS["staged"] == staged, "forcing a pass staged the home"
    finally:
        sf._PF_ATTN, sf._KV_HOST = pf, flag
    print("  4. budget: attn_bytes adds 2 rows a key past the line only, close() leaves the LRU out of a pass, "
          "_copy and the pass evaluation see VRAM only")


def syntax():
    cxx = shutil.which("clang++") or shutil.which("g++")
    if not cxx:
        return "skipped: no C++ compiler"
    ctype = {"tbl": "uint64_t", "p0": "uint8_t", "np": "uint8_t", "cp": "uint8_t", "lim": "uint8_t",
             "dep": "int32_t", "ldep": "int32_t", "tb": "uint32_t", "som": "int32_t", "bom": "int32_t",
             "stp": "int32_t", "ref": "int32_t", "meta": "int32_t", "sl": "int32_t", "done": "int32_t"}
    vals = {"T": "__nv_bfloat16", "D": 256, "HK": 2, "LOGC": 13, "CR": 4, "NSEL": 512, "NS": 2048, "NT": 512}
    hdr = kh._HDR[kh._HDR.index("template <typename T"):]     # kvh_ld's PTX is stubbed below
    tu = ["#include <array>\n#include <cstdint>\n#include <algorithm>\nusing std::min;\n"
          "struct uint4 { unsigned x, y, z, w; };\nstatic uint4 make_uint4(unsigned a, unsigned b, unsigned c, "
          "unsigned d) { return {a, b, c, d}; }\nstruct dim3 { unsigned x, y, z; };\n"
          "static dim3 threadIdx, blockIdx, blockDim, gridDim;\nstruct __nv_bfloat16 { unsigned short bits; };\n"
          "using Shape = std::array<int32_t, 8>;\nusing Strides = std::array<int64_t, 8>;\n#define __global__\n"
          "#define __device__\n#define __forceinline__ inline\n#define __grid_constant__\n#define __shared__ static\n"
          "static void __syncthreads() {}\nstatic unsigned __ballot_sync(unsigned, bool p) { return p; }\n"
          "static int __popc(unsigned x) { return __builtin_popcount(x); }\n"
          "static inline uint4 kvh_ld(const uint4* p) { return *p; }\n", hdr]
    for name, (ins, outs, names, body) in kh.SPECS.items():
        ps = []
        for x in ins:
            ps.append(f"const {ctype.get(x, '__nv_bfloat16')}* {x}")
            ps += [f"const __grid_constant__ {t} {x}_{s}" for s, t in (("shape", "Shape"), ("strides", "Strides"))
                   if f"{x}_{s}" in body]
        ps += [f"{ctype.get(x, '__nv_bfloat16')}* {x}" for x in outs]
        consts = "".join(f"  using {k} = {vals[k]};\n" if k == "T" else f"  constexpr int {k} = {vals[k]};\n"
                         for k in names)
        tu.append(f"__global__ void kvh_{name}_({', '.join(ps)}) {{\n{consts}{body}}}\n")
    with tempfile.NamedTemporaryFile("w", suffix=".cpp", delete=False) as f:
        f.write("".join(tu))
    r = subprocess.run([cxx, "-std=c++17", "-fsyntax-only", "-Wall", "-Wno-unused-variable", f.name],
                       capture_output=True, text=True)
    os.unlink(f.name)
    assert r.returncode == 0 and not r.stderr.strip(), f"kernel text ({cxx}):\n{r.stderr}"
    return "ok"


def main():
    rng = np.random.default_rng(0)
    kh._CU = CU
    kh._avail = lambda: None
    kh._kern = lambda name: (lambda **kw: EMU[name](**kw))
    cap_rule(rng)
    for base in (q4._AttnCache, Three):
        cls = kh.make_class(base)
        vram_mode(cls, base, rng)
        homed(cls, base, rng)
    base = q4._AttnCache
    budget(kh.make_class(base), base, rng)
    print(f"  5. kernel text as C++: {syntax()}")
    print("test_kv_host_emu: ok")


if __name__ == "__main__":
    main()
