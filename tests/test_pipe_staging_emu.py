#!/usr/bin/env python3
"""MOE_PIPE_STAGING and MOE_PIPE_PIECES without a GPU: native/moe_pipe_cuda.cpp built for this machine (O_DIRECT off on macOS, libcuda a
stand-in that hands out host memory as both the host and the device pointer) and driven through its C ABI by this file
playing the GPU: it posts each layer's job and the guess beside it, waits for the claim, runs the wait kernel's plan
(a tier save first, then each copy once its record's flag is up) and checks every gathered row against the file.

  1. bytes: staging off, on at cap 2 and cap 1, with guess reads slowed (MOE_PIPE_STAGING_FAULT) so claims meet
     them in flight and unstarted, with every third guess read failing, and both: every row a job gathers holds its
     expert's bytes from experts.bin, on a per-layer arena and on a shared store with ring halves, the tier on;
  2. no store pollution: slot tables, LRU order, ring halves, seen counts, claim logs and the tier's hits and puts end
     the same in every arm as with staging off;
     with the guess two layers ahead (MOE_PIPE_GUESS_D=2), and with records read in 4 KiB pieces (MOE_PIPE_PIECES,
     io_uring on Linux, one piece after another elsewhere), staging off and on;
  3. scheduling: guess records were read and copied into slots, some promoted in flight and some dropped unstarted
     when slowed, failed ones read again on demand, never more than the cap in flight, none started while a job's
     own records were read;
  4. post_guess's kernel text (native/moe_pipe_cuda.py) run as C++ with its warp's 32 lanes as threads meeting at
     each shuffle: the entry holds the job, k and the top k in descending order, a tie to the larger index;
  5. MOE_PIPE_PIECES' wait kernel (_WAIT_P's copy blocks) run as C++, 31 blocks of 8 threads with a barrier and shared
     words per block, pieces landing in a random order: every entry gets its bytes only after its piece lands, every
     saved entry's old bytes go out first, at 1, 3 and 8 slices an entry (no wait, a slice off by 16 bytes or a save
     from the wrong place each fail it).

Laptop or the PC, no GPU, a few seconds: perl -e 'alarm shift; exec @ARGV' 280 python3 tests/test_pipe_staging_emu.py
"""
import ctypes
import glob
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "native" / "moe_pipe_cuda.cpp"

# moe_pipe_cuda.cpp's block words.
POSTED, DONE, SENT, CLAIMED, RING, READY, GUESS, PLAN = 0, 1, 5, 7, 64, 6144, 7168, 8192
L, E, K = 4, 24, 4                              # layers, experts, top-k
PARTS = [[4096, 1536], [9216], [2560]]          # each projection's parts (row bytes); its stride is their sum
STRIDE = [sum(p) for p in PARTS]
ARENA, QUOTA, HALF = 6, 3, 4                    # arena rows; a store's quota a layer and rows a ring half
TOKENS, GPU_S = 10, 0.002                       # tokens a run, and the GPU's work between two jobs

FAKE_CUDA = r"""
#include <stdlib.h>
typedef unsigned long long CUdeviceptr;
static int ctx;
int cuInit(unsigned f) { return 0; }
int cuDeviceGet(int* d, int o) { *d = 0; return 0; }
int cuCtxGetCurrent(void** c) { *c = &ctx; return 0; }
int cuCtxSetCurrent(void* c) { return 0; }
int cuDevicePrimaryCtxRetain(void** c, int d) { *c = &ctx; return 0; }
int cuMemHostAlloc(void** p, size_t n, unsigned f) { return posix_memalign(p, 4096, n ? n : 1) ? 2 : 0; }
int cuMemFreeHost(void* p) { free(p); return 0; }
int cuMemHostGetDevicePointer_v2(CUdeviceptr* d, void* p, unsigned f) { *d = (CUdeviceptr)p; return 0; }
unsigned emu_ld(const unsigned* p) { return __atomic_load_n(p, __ATOMIC_SEQ_CST); }
void emu_st(unsigned* p, unsigned v) { __atomic_store_n(p, v, __ATOMIC_SEQ_CST); }
"""

# The kernel's stand-ins: a warp of 32 threads that all meet at each shuffle, each lane posting its value and
# taking its partner's.
WARP = r"""
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <thread>
#include <vector>
struct dim3 { unsigned x, y, z; };
static thread_local dim3 threadIdx;
static std::mutex mu;
static std::condition_variable cv;
static int arrived = 0, gen = 0;
static double box[32];
static void meet() {
  std::unique_lock<std::mutex> lk(mu);
  int g = gen;
  if (++arrived == 32) { arrived = 0; gen++; cv.notify_all(); }
  else cv.wait(lk, [&] { return gen != g; });
}
template <class T> T __shfl_xor_sync(unsigned, T v, int m) {
  box[threadIdx.x] = double(v);
  meet();
  T r = T(box[threadIdx.x ^ m]);
  meet();
  return r;
}
static void __threadfence_system() {}
static void kernel(const uint32_t* tick, const float* lg, const uint64_t* blk, uint32_t* tock) {
  constexpr int NE = @NE@, KT = @KT@, NPL = @NPL@;
@SRC@
}
int main() {
  std::vector<float> lg(NE_);
  for (auto& x : lg) if (scanf("%f", &x) != 1) return 2;
  std::vector<uint32_t> mem(16384, 0);
  uint64_t blk = reinterpret_cast<uint64_t>(mem.data());
  uint32_t tick = @V@, tock = 0;
  std::vector<std::thread> lanes;
  for (unsigned l = 0; l < 32; l++)
    lanes.emplace_back([&, l] { threadIdx = {l, 0, 0}; kernel(&tick, lg.data(), &blk, &tock); });
  for (auto& t : lanes) t.join();
  const uint32_t* G = mem.data() + 7168 + 32 * (tick % 8);
  printf("%u %u", tock, G[0]);
  for (uint32_t j = 0; j < G[1] + 1; j++) printf(" %u", G[1 + j]);
  printf("\n");
}
"""


def compile_cxx(args, out):
    """The compiler on args, and on macOS again against each older SDK when the default one's stubs do not link."""
    base = [os.environ.get("CXX", "c++"), "-O2", *args, "-o", str(out)]
    tries = [[]]
    if sys.platform == "darwin":
        tries += [["-isysroot", s] for s in sorted(glob.glob("/Library/Developer/CommandLineTools/SDKs/MacOSX*.*.sdk"),
                                                    reverse=True)]
    err = ""
    for t in tries:
        r = subprocess.run(base[:2] + t + base[2:], capture_output=True, text=True)
        if r.returncode == 0:
            return
        err = r.stderr[-1500:]
    raise RuntimeError(f"compile failed: {' '.join(base)}\n{err}")


def build(d):
    """The agent and the libcuda stand-in in d."""
    (d / "fake.c").write_text(FAKE_CUDA)
    compile_cxx(["-x", "c", "-shared", "-fPIC", str(d / "fake.c")], d / "libcuda.so.1")
    extra = ["-DO_DIRECT=0"] if sys.platform == "darwin" else ["-ldl", "-lpthread"]
    compile_cxx(["-std=c++17", "-shared", "-fPIC", str(SRC), *extra], d / "libagent.so")


# ------------------------------------------------------------------------- the GPU stand-in
class Run:
    """One agent over one layout, its jobs played as the GPU plays them."""

    def __init__(self, lib, cu, fd, data, store, cap, fault, dist=1, pieces=False):
        self.lib, self.cu, self.data, self.store = lib, cu, data, store
        if fault:
            os.environ["MOE_PIPE_STAGING_FAULT"] = fault
        else:
            os.environ.pop("MOE_PIPE_STAGING_FAULT", None)
        if pieces:
            os.environ.update(MOE_PIPE_PIECES="1", MOE_PIPE_PIECE="4096")
        else:
            os.environ.pop("MOE_PIPE_PIECES", None)
        self.ready_max = 0          # the most ready words a job's plan used
        self.base = lib.mpc_block()
        self.blk = np.ctypeslib.as_array((ctypes.c_uint32 * 16384).from_address(self.base))
        self.a = lib.mpc_agent_new(fd, L, 3)
        assert self.a, lib.mpc_error()
        assert lib.mpc_agent_tier(self.a, sum(STRIDE), 6, 0) == 6, lib.mpc_error()      # one chunk, the slots it holds
        if cap:
            assert lib.mpc_agent_staging(self.a, cap, dist) == 0, lib.mpc_error()
        rows = QUOTA * L + 2 * HALF if store else ARENA
        self.slot = [np.full(E, -1, dtype=np.int32) for _ in range(L)]
        shared = [[np.zeros((rows, r), np.uint8) for r in parts] for parts in PARTS] if store else None
        self.mem = [shared or [[np.zeros((rows, r), np.uint8) for r in parts] for parts in PARTS] for _ in range(L)]
        for i in range(L):
            off = np.array([self.off(i, p) for p in range(3)], dtype=np.int64)
            stride = np.array(STRIDE, dtype=np.int64)
            nparts = np.array([len(p) for p in PARTS], dtype=np.int32)
            addr = np.array([a.ctypes.data for parts in self.mem[i] for a in parts], dtype=np.uint64)
            row = np.array([r for parts in PARTS for r in parts], dtype=np.uint64)
            q = np.arange(QUOTA * i, QUOTA * (i + 1), dtype=np.int32) if store else np.zeros(0, np.int32)
            rc = lib.mpc_set_layer(self.a, i, self.slot[i].ctypes.data, E, rows, 3, off.ctypes.data, stride.ctypes.data,
                                   nparts.ctypes.data, addr.ctypes.data, row.ctypes.data, q.ctypes.data, q.size,
                                   i % 2 if store else -1)
            assert rc == 0, lib.mpc_error()
        if store:
            for r in range(2):
                own = np.full(2 * HALF, -1, dtype=np.int32)
                assert lib.mpc_set_ring(self.a, r, QUOTA * L + HALF * r, 0, own.ctypes.data, HALF) == 0, lib.mpc_error()

    @staticmethod
    def off(i, p):
        return E * sum(STRIDE) * i + E * sum(STRIDE[:p])

    def ld(self, w):
        return self.cu.emu_ld(ctypes.c_void_p(self.base + 4 * w))

    def st(self, w, v):
        self.cu.emu_st(ctypes.c_void_p(self.base + 4 * w), v)

    def wait(self, w, v):
        t = time.monotonic()
        while self.ld(w) < v:
            assert time.monotonic() - t < 10, f"word {w} never reached {v}"

    def job(self, layer, ids, guess):
        v = int(self.blk[POSTED]) + 1
        e = RING + 128 * (v % 8)
        self.blk[e + 1], self.blk[e + 2] = layer, len(ids)
        self.blk[e + 3:e + 3 + len(ids)] = ids
        self.st(e, v)
        self.st(POSTED, v)
        self.st(SENT, v)
        if guess is not None:
            g = GUESS + 32 * (v % 8)
            self.blk[g + 1] = len(guess)
            self.blk[g + 2:g + 2 + len(guess)] = guess
            self.st(g, v)
        self.wait(CLAIMED, v)
        assert self.ld(PLAN) == v, "the plan is not this job's"
        self.ready_max = max(self.ready_max, int(self.blk[PLAN + 2]))
        for c in range(int(self.blk[PLAN + 1])):
            w = self.blk[PLAN + 4 + 8 * c:PLAN + 12 + 8 * c].astype(np.uint64)
            dst, src, n, rec, sv = w[0] | w[1] << 32, w[2] | w[3] << 32, int(w[4]), int(w[5]), w[6] | w[7] << 32
            if sv:
                ctypes.memmove(int(sv), int(dst), n)
            self.wait(READY + rec, v)
            ctypes.memmove(int(dst), int(src), n)
        for x in ids:
            s = self.slot[layer][x]
            assert s >= 0, f"layer {layer} expert {x} has no slot after its claim"
            for p, parts in enumerate(PARTS):
                at = self.off(layer, p) + x * STRIDE[p]
                got = np.concatenate([a[s] for a in self.mem[layer][p]])
                assert np.array_equal(got, self.data[at:at + STRIDE[p]]), \
                    f"layer {layer} expert {x} projection {p}: slot {s} holds other bytes"
        time.sleep(GPU_S)

    def state(self):
        lib, a, out = self.lib, self.a, {}
        for i in range(L):
            buf, n = np.zeros(2 * (QUOTA * L + 2 * HALF), np.int32), ctypes.c_int()
            assert lib.mpc_lru(a, i, buf.ctypes.data, buf.size // 2, ctypes.byref(n)) == 0, lib.mpc_error()
            seen = np.zeros(E, np.uint32)
            assert lib.mpc_take_seen(a, i, seen.ctypes.data, E) == 0
            ni, nl = ctypes.c_int(), ctypes.c_int()
            assert lib.mpc_take_log(a, i, None, None, ctypes.byref(ni), ctypes.byref(nl)) == 0
            ids, lens = np.zeros(ni.value, np.int32), np.zeros(nl.value, np.int32)
            assert lib.mpc_take_log(a, i, ids.ctypes.data, lens.ctypes.data, ctypes.byref(ni), ctypes.byref(nl)) == 0
            out[i] = (buf[:2 * n.value].tolist(), seen.tolist(), ids.tolist(), lens.tolist(), self.slot[i].tolist())
        if self.store:
            for r in range(2):
                nx, own, n = ctypes.c_int(), np.zeros(2 * HALF, np.int32), ctypes.c_int()
                assert lib.mpc_ring_state(a, r, ctypes.byref(nx), own.ctypes.data, HALF, ctypes.byref(n)) == 0
                out[f"ring {r}"] = (nx.value, own.tolist())
        st = (ctypes.c_uint64 * 9)()
        lib.mpc_agent_stats(a, st)
        out["tier hits, puts"] = (st[6], st[7])
        return out


def sequence(rng):
    """Every token's ids per layer, skewed so the arena both hits and misses, and each layer's guess of the next:
    two of its experts and two others."""
    pop = [rng.permutation(E) for _ in range(L)]
    p = 1.0 / np.arange(1, E + 1) ** 0.9
    p /= p.sum()
    toks = [[pop[l][rng.choice(E, K, replace=False, p=p)] for l in range(L)] for _ in range(TOKENS)]
    return toks, guesses_of(rng, toks, 1)


def guesses_of(rng, toks, dist):
    """Each layer's guess of layer + dist: two of its experts and two others."""
    guesses = []
    for t in toks:
        g = []
        for l in range(L - dist):
            right = rng.choice(t[l + dist], 2, replace=False)
            other = rng.choice(np.setdiff1d(np.arange(E), right), 2, replace=False)
            g.append(rng.permutation(np.concatenate([right, other])))
        guesses.append(g + [None] * dist)
    return guesses


def child(d):
    lib = ctypes.CDLL(str(d / "libagent.so"))
    cu = ctypes.CDLL("libcuda.so.1")           # the stand-in, found through the library path
    lib.mpc_error.restype = ctypes.c_char_p
    lib.mpc_block.restype = ctypes.c_void_p
    lib.mpc_agent_new.restype = ctypes.c_void_p
    lib.mpc_agent_new.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.mpc_agent_tier.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int, ctypes.c_int]
    lib.mpc_agent_staging.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.mpc_agent_drain.argtypes = [ctypes.c_void_p, ctypes.c_double]
    for f in ("mpc_agent_close", "mpc_agent_arm"):
        getattr(lib, f).argtypes = [ctypes.c_void_p]
    lib.mpc_agent_error.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
    lib.mpc_agent_stats.argtypes = lib.mpc_agent_stage_stats.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    v = ctypes.c_void_p
    lib.mpc_set_layer.argtypes = [v, ctypes.c_int, v, ctypes.c_int, ctypes.c_int, ctypes.c_int, v, v, v, v, v, v,
                                  ctypes.c_int, ctypes.c_int]
    lib.mpc_set_ring.argtypes = [v, ctypes.c_int, ctypes.c_int, ctypes.c_int, v, ctypes.c_int]
    lib.mpc_lru.argtypes = [v, ctypes.c_int, v, ctypes.c_int, v]
    lib.mpc_take_seen.argtypes = [v, ctypes.c_int, v, ctypes.c_int]
    lib.mpc_take_log.argtypes = [v, ctypes.c_int, v, v, v, v]
    lib.mpc_ring_state.argtypes = [v, ctypes.c_int, v, v, ctypes.c_int, v]
    cu.emu_ld.restype = ctypes.c_uint
    cu.emu_ld.argtypes = [v]
    cu.emu_st.argtypes = [v, ctypes.c_uint]
    assert lib.mpc_init() == 0, lib.mpc_error()

    rng = np.random.default_rng(3)
    data = rng.integers(0, 256, L * E * sum(STRIDE), dtype=np.uint8)
    path = d / "experts.bin"
    data.tofile(path)
    fd = os.open(path, os.O_RDONLY)
    toks, guesses = sequence(rng)
    guesses2 = guesses_of(np.random.default_rng(4), toks, 2)
    # (name, guess threads, fault, guess distance, pieces)
    arms = [("off", 0, ""), ("cap 2", 2, ""), ("cap 1", 1, ""), ("slow", 2, "0,4000"), ("fail", 2, "3,0"),
            ("slow + fail", 1, "2,3000"), ("d 2", 2, "", 2), ("d 2 slow", 2, "0,4000", 2), ("pieces", 0, "", 1, True),
            ("pieces + cap 2", 2, "", 1, True), ("pieces + d 2 slow", 2, "0,3000", 2, True)]
    for store in (False, True):
        layout = "store" if store else "arena"
        ref = None
        for name, cap, fault, *rest in arms:
            dist, pieces = (list(rest) + [1, False][len(rest):])[:2]
            gs = guesses2 if dist == 2 else guesses
            r = Run(lib, cu, fd, data, store, cap, fault, dist, pieces)
            for t in range(TOKENS):
                lib.mpc_agent_arm(r.a)
                for layer in range(L):
                    r.job(layer, toks[t][layer].astype(np.uint32), None if gs[t][layer] is None
                          else gs[t][layer].astype(np.uint32))
                assert lib.mpc_agent_drain(r.a, 10.0), f"{layout} {name}: the agent did not go idle"
            buf = ctypes.create_string_buffer(512)
            lib.mpc_agent_error(r.a, buf, 512)
            assert not buf.value, f"{layout} {name}: agent error {buf.value!r}"
            st = (ctypes.c_uint64 * 11)()
            lib.mpc_agent_stage_stats(r.a, st)
            g, rec, prom, part, wrong, wait, drop, _, bad, peak, ahead = st
            state = r.state()
            lib.mpc_agent_close(r.a)
            if ref is None:
                ref = state
                print(f"  {layout} {name}: {L * TOKENS} jobs, every gathered row right; tier hits / puts "
                      f"{state['tier hits, puts']}")
                continue
            for k in ref:
                assert state[k] == ref[k], f"{layout} {name}: {k} differs from staging off\n{state[k]}\n{ref[k]}"
            if pieces:
                assert r.ready_max > 3 * K, f"{layout} {name}: at most {r.ready_max} ready words a job, no record in pieces"
            if not cap:
                print(f"  {layout} {name}: rows right, tables and tier as off; up to {r.ready_max} ready words a job")
                continue
            assert rec > 0 and prom > 0, f"{layout} {name}: {rec} guess records read, {prom} used"
            assert peak <= cap, f"{layout} {name}: {peak} guess reads in flight, cap {cap}"
            assert ahead == 0, f"{layout} {name}: {ahead} guess reads started during a job's own reads"
            if fault.endswith(",4000") or fault.endswith(",3000"):
                assert part > 0 and drop > 0 and wait > 0, \
                    f"{layout} {name}: slowed, {part} promoted in flight, {drop} dropped, {wait} ns waited"
            if fault.startswith(("3,", "2,")):
                assert bad > 0, f"{layout} {name}: no guess read failed"
            print(f"  {layout} {name}: rows right, tables and tier as off; guessed {g} experts, read {rec} records, "
                  f"used {prom} ({part} in flight), unused {wrong}, dropped {drop}, failed {bad}, "
                  f"waited {wait / 1e6:.1f} ms, in flight at most {peak}")
    os.close(fd)
    print("  1-3. ok")


# ------------------------------------------------------------------------- 4. the guess kernel
def kernel_check(d):
    src = re.search(r'_GUESS = r"""(.*?)"""', (ROOT / "native" / "moe_pipe_cuda.py").read_text(), re.S).group(1)
    rng = np.random.default_rng(5)
    cases = [(512, 10, 13, rng.standard_normal(512).astype(np.float32)),
             (60, 4, 8, np.round(rng.standard_normal(60), 1).astype(np.float32))]    # rounded: ties
    for ne, kt, v, lg in cases:
        text = (WARP.replace("@SRC@", src).replace("@NE@", str(ne)).replace("@KT@", str(kt))
                .replace("@NPL@", str((ne + 31) // 32)).replace("@V@", str(v)).replace("NE_", str(ne)))
        (d / "warp.cpp").write_text(text)
        compile_cxx(["-std=c++17", str(d / "warp.cpp"), "-lpthread"], d / "warp")
        out = subprocess.run([str(d / "warp")], input=" ".join(f"{x:.9g}" for x in lg), capture_output=True,
                             text=True, timeout=60)
        tock, tag, n, *ids = map(int, out.stdout.split())
        want = sorted(range(ne), key=lambda e: (-lg[e], -e))[:kt]
        assert (tock, tag, n) == (v, v, kt) and ids == want, f"guess kernel, {ne} logits: {ids} != {want}"
    print("  4. the guess kernel's text: the top k, descending, ties to the larger index, entry v % 8")


# The pieces wait kernel's copy blocks (moe_pipe_cuda.py's _WAIT_P after the leader) as C++: 31 blocks of 8 threads, each a thread, with
# a barrier and shared memory per block, and a stand-in leader that lands each piece's bytes in staging in a random order and only then
# republishes its word.
COPY = r"""
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <random>
#include <thread>
#include <vector>
struct dim3 { unsigned x, y, z; };
static thread_local dim3 threadIdx = {0, 0, 0}, blockIdx = {0, 0, 0};
static const dim3 blockDim = {8, 1, 1}, gridDim = {32, 1, 1};
struct Bar { std::mutex m; std::condition_variable cv; unsigned arrived = 0, gen = 0; };
static Bar BAR[32];
static void __syncthreads() {
  Bar& b = BAR[blockIdx.x];
  std::unique_lock<std::mutex> lk(b.m);
  unsigned g = b.gen;
  if (++b.arrived == blockDim.x) { b.arrived = 0; b.gen++; b.cv.notify_all(); }
  else b.cv.wait(lk, [&] { return b.gen != g; });
}
#define __threadfence() std::atomic_thread_fence(std::memory_order_seq_cst)
using std::min;
static unsigned mpc_ld(const unsigned* p) { return __atomic_load_n(p, __ATOMIC_ACQUIRE); }
static void mpc_copy(char* d, const char* s, unsigned n) {
  const volatile char* vs = s;
  for (unsigned i = threadIdx.x; i < n; i += blockDim.x) d[i] = vs[i];
}
static void atomicMin(unsigned* p, unsigned x) {
  unsigned o = __atomic_load_n(p, __ATOMIC_RELAXED);
  while (x < o && !__atomic_compare_exchange_n(p, &o, x, false, __ATOMIC_SEQ_CST, __ATOMIC_RELAXED)) {}
}
struct Blk {         // one block: the kernel's constants and shared words as members, its body as run()
  static constexpr unsigned SL = @SL@;
@HOIST@
  void run(const unsigned* P, unsigned* D, unsigned v) {
@SRC@
  }
};
static Blk BLK[32];
static void block(const unsigned* P, unsigned* D, unsigned v) { BLK[blockIdx.x].run(P, D, v); }
int main() {
  std::mt19937 rng(@SEED@);
  const unsigned v = 7, npieces = 9;
  std::vector<unsigned> cut = {0};          // piece edges in staging, 4 KiB to 40 KiB apart
  for (unsigned k = 0; k < npieces; k++) cut.push_back(cut.back() + 4096 * (1 + rng() % 10));
  const size_t total = cut.back();
  std::vector<char> fin(total), stage(total), dst(total), old(total), sv(total, 0);
  for (auto& c : fin) c = char(rng());
  for (auto& c : stage) c = char(rng());    // what staging holds before a piece lands
  for (auto& c : dst) c = char(rng());
  old = dst;
  std::vector<unsigned> P(16384, 0), D(2048, 0);
  std::vector<uint64_t> doff;               // each entry's place in dst, the entries' order shuffled
  std::vector<std::array<size_t, 4>> ents;  // src offset, bytes, piece, saved
  for (unsigned k = 0; k < npieces; k++)
    for (size_t o = cut[k]; o < cut[k + 1];) {
      size_t n = std::min<size_t>(cut[k + 1] - o, rng() % 3 == 0 ? 1 + rng() % 9000 : 16 * (1 + rng() % 2048));
      ents.push_back({o, n, k, size_t(rng() % 3 == 0)});
      o += n;
    }
  std::vector<size_t> at(ents.size());
  std::vector<size_t> order(ents.size());
  for (size_t i = 0; i < order.size(); i++) order[i] = i;
  std::shuffle(order.begin(), order.end(), rng);
  size_t put = 0;
  for (size_t i : order) { at[i] = put; put += ents[i][1]; }
  const unsigned n = ents.size();
  P[8193] = n;
  for (unsigned c = 0; c < n; c++) {
    uint64_t d = uint64_t(uintptr_t(dst.data() + at[c])), s = uint64_t(uintptr_t(stage.data() + ents[c][0]));
    uint64_t w = ents[c][3] ? uint64_t(uintptr_t(sv.data() + at[c])) : 0;
    unsigned* E = P.data() + 8196 + 8 * c;
    E[0] = unsigned(d); E[1] = unsigned(d >> 32); E[2] = unsigned(s); E[3] = unsigned(s >> 32);
    E[4] = unsigned(ents[c][1]); E[5] = unsigned(ents[c][2]); E[6] = unsigned(w); E[7] = unsigned(w >> 32);
  }
  D[1] = 1;
  __atomic_store_n(&D[0], v, __ATOMIC_RELEASE);     // the leader: job v, state copy
  std::vector<std::thread> blocks;
  for (unsigned b = 1; b < gridDim.x; b++)
    for (unsigned t = 0; t < blockDim.x; t++)
      blocks.emplace_back([&, b, t] { blockIdx = {b, 0, 0}; threadIdx = {t, 0, 0}; block(P.data(), D.data(), v); });
  std::vector<unsigned> land(npieces);
  for (unsigned k = 0; k < npieces; k++) land[k] = k;
  std::shuffle(land.begin(), land.end(), rng);
  for (unsigned k : land) {
    std::this_thread::sleep_for(std::chrono::microseconds(300));
    memcpy(stage.data() + cut[k], fin.data() + cut[k], cut[k + 1] - cut[k]);
    __atomic_store_n(&D[128 + k], v, __ATOMIC_RELEASE);
  }
  for (auto& t : blocks) t.join();
  for (unsigned c = 0; c < n; c++) {
    if (memcmp(dst.data() + at[c], fin.data() + ents[c][0], ents[c][1])) { printf("entry %u: wrong bytes\n", c); return 1; }
    if (ents[c][3] && memcmp(sv.data() + at[c], old.data() + at[c], ents[c][1])) { printf("entry %u: save wrong\n", c); return 1; }
  }
  printf("ok %u entries, %u pieces\n", n, npieces);
}
"""


def copy_check(d):
    src = re.search(r'_WAIT_P = _WAIT_LEAD\.replace\(.*?\) \+ r"""(.*?)"""', (ROOT / "native" / "moe_pipe_cuda.py").read_text(),
                    re.S).group(1)
    # The kernel's constexpr and __shared__ lines become the block's members, the rest its body.
    keep, hoist = [], []
    for ln in src.splitlines():
        s = ln.strip()
        if s.startswith("constexpr"):
            hoist.append("  static " + s)
        elif s.startswith("__shared__"):
            hoist.append("  " + s[len("__shared__"):].strip())
        else:
            keep.append(ln)
    assert hoist, "the copy blocks' shared words moved"
    for sl, seed in ((8, 1), (3, 2), (1, 3), (8, 4)):
        (d / "copy.cpp").write_text(COPY.replace("@SRC@", "\n".join(keep)).replace("@HOIST@", "\n".join(hoist))
                                    .replace("@SL@", str(sl)).replace("@SEED@", str(seed)))
        compile_cxx(["-std=c++17", str(d / "copy.cpp"), "-lpthread"], d / "copy")
        out = subprocess.run([str(d / "copy")], capture_output=True, text=True, timeout=60)
        assert out.returncode == 0 and out.stdout.startswith("ok"), f"pieces copy, {sl} slices: {out.stdout} {out.stderr}"
    print(f"  5. the pieces wait kernel's copy blocks: every entry's bytes once their piece lands, saves first, "
          f"1 / 3 / 8 slices ({out.stdout.strip()[3:]})")


def main():
    if os.environ.get("EMU_DIR"):
        child(Path(os.environ["EMU_DIR"]))
        return
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        build(d)
        env = dict(os.environ, EMU_DIR=t, MOE_PIPE_CHUNK="512")
        env["DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"] = t
        r = subprocess.run([sys.executable, __file__], env=env, timeout=240)
        assert r.returncode == 0, "the agent's run failed"
        kernel_check(d)
        copy_check(d)
    print("test_pipe_staging_emu: ok")


if __name__ == "__main__":
    main()
