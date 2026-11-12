"""moe_native's API on MLX's CUDA backend: MOE_NATIVE=1 under CUDA (stream_fast._Native).

The agent is native/moe_pipe_cuda.cpp, a C ABI library g++ builds here at first use (native/build/, keyed on the
source's hash; no MLX headers, no nanobind). This module adds the two handshake kernels, both
mx.fast.cuda_kernel (public API). Per MoE layer of a one-position decode pass:

  tick = signal(ids, layer)          the GPU writes the layer's ids (uint32, the router's) into the mapped block's
                                     ring and posts the job
  rows = wait_rows(tick, ids, table) the GPU waits (ld.acquire.sys spins) for the agent's claim, then its blocks
                                     copy each missed record from mapped staging into its slot as the agent flags
                                     it landed; the gather rows table[ids] (uint32) come out. Every spin ends at the
                                     host's release, another wait's give-up, MOE_NATIVE_WAIT_S of %globaltimer, and
                                     with MOE_PIPE_BEAT_MS that long after the agent's heartbeat last moved
  committed()                        after the layer's mx.async_eval: drain() and pending() count it

With staging (MOE_PIPE_STAGING=1, stream_fast), between the two: post_guess(tick, logits, k) writes the top-k of layer
L + D's gate on this layer's input beside job `tick` (MOE_PIPE_GUESS_D, 1), and the agent reads those L + D misses into
guess records once this job's own records are in (Agent.set_staging, Agent.stage_stats). With MOE_PIPE_HITS_FIRST=1,
resident(ids, table) before the signal gives the held experts' rows, whose chain runs inside the park.

The read path: the agent reads through an O_DIRECT fd (MOE_PIPE_DIRECT=0: buffered); the wait
kernel copies each part in 128 KB chunks, a block each (MOE_PIPE_CHUNK, bytes, 0 whole parts); the store's evictions
go into a victim tier in pinned host memory that the wait kernel copies misses from (MOE_PIPE_TIER_GB: its size, 0
none; unset, tier_slots sizes it from MemAvailable at install; under the store governor it follows that rule while it
runs, stream_fast.TierGovernor). MOE_PIPE_PIECES (on by default since ea5b476a, =0 turns it off): records land in 128 KB
pieces (MOE_PIPE_PIECE) on io_uring with a ready word each, and the wait kernel copies a chunk in MOE_PIPE_SLICES slices
on as many blocks.

Process-wide, as moe_native's stop words: one block, one device job counter, one bound, one live agent.
A KILLed process whose kernel still runs a few ms later leaves the GPU at 100% until another CUDA context is made,
which the guard process below does, and MOE_PIPE_BEAT_MS=2 (opt-in) is the other fix: the wait gives up 2 ms after
the agent's heartbeat stops, so it ends before the driver's window closes, but a live process whose beat thread
stalls that long fails that pass.
"""
import atexit
import ctypes
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import mlx.core as mx

HERE = Path(__file__).resolve().parent
SRC = HERE / "moe_pipe_cuda.cpp"
BLOCK_WORDS = 65536 // 4                    # moe_pipe_cuda.cpp's BLOCK_BYTES
POSTED, DONE, RELEASE, GAVE, SENT, CLAIMED, LOG, GUESS = 0, 1, 2, 3, 5, 7, 2048, 7168
BLOCKS = 32                                 # the wait kernel's blocks: a leader and 31 that copy
_LOG = os.environ.get("MOE_PIPE_LOG", "0") == "1"     # the waits' GPU times (wait_log); not for speed runs
# MOE_PIPE_PIECES (on by default since 2026-11-06, =0 turns it off): a record lands in pieces with a ready word each and the wait
# kernel copies each plan entry in MOE_PIPE_SLICES slices; the default is written into the environment because the agent reads it too.
_PIECES = os.environ.setdefault("MOE_PIPE_PIECES", "1") == "1"
_SLICES = int(os.environ.get("MOE_PIPE_SLICES", "8"))


def build():
    """The library for this source, built once by g++ (atomic rename, so two processes may race)."""
    tag = hashlib.sha256(SRC.read_bytes()).hexdigest()[:12]
    lib = HERE / "build" / f"libmoe_pipe_cuda-{tag}.so"
    if not lib.exists():
        lib.parent.mkdir(exist_ok=True)
        tmp = lib.with_name(f"{lib.name}.{os.getpid()}.tmp")
        cmd = [os.environ.get("CXX", "g++"), "-O2", "-std=c++17", "-shared", "-fPIC", str(SRC), "-o", str(tmp),
               "-ldl", "-lpthread"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            raise ImportError(f"moe_pipe_cuda: {' '.join(cmd)} failed:\n{r.stderr[-2000:]}")
        os.replace(tmp, lib)
    return lib


_P, _I, _U64 = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64
_IP, _I64P, _U64P = ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_uint64)
_lib = ctypes.CDLL(str(build()))
for _name, _res, _args in [
        ("mpc_error", ctypes.c_char_p, []), ("mpc_now_ns", _U64, []), ("mpc_init", _I, []),
        ("mpc_block", _P, []), ("mpc_block_dev", _U64, []), ("mpc_release", None, []), ("mpc_pending", _I, []),
        ("mpc_agent_new", _P, [_I, _I, _I]), ("mpc_agent_close", None, [_P]),
        ("mpc_agent_drain", _I, [_P, ctypes.c_double]), ("mpc_agent_arm", None, [_P]),
        ("mpc_agent_error", _I, [_P, ctypes.c_char_p, _I]), ("mpc_agent_stats", None, [_P, _U64P]),
        ("mpc_agent_direct", _I, [_P]), ("mpc_agent_tier", _I, [_P, _U64, _I, _I]),
        ("mpc_agent_staging", _I, [_P, _I, _I]), ("mpc_agent_stage_stats", None, [_P, _U64P]),
        ("mpc_tier_fill", _I, [_P, _I, _I, _I, _U64P]), ("mpc_tier_find", _I, [_P, _I, _IP, _I, _U64P]),
        ("mpc_set_layer", _I, [_P, _I, _P, _I, _I, _I, _I64P, _I64P, _IP, _U64P, _U64P, _IP, _I, _I]),
        ("mpc_clear_layer", _I, [_P, _I]), ("mpc_set_state", _I, [_P, _I, _IP, _I, _IP, _I]),
        ("mpc_lru", _I, [_P, _I, _IP, _I, _IP]), ("mpc_take_seen", _I, [_P, _I, _P, _I]),
        ("mpc_take_log", _I, [_P, _I, _P, _P, _IP, _IP]), ("mpc_set_ring", _I, [_P, _I, _I, _I, _IP, _I]),
        ("mpc_ring_state", _I, [_P, _I, _IP, _IP, _I, _IP])]:
    _f = getattr(_lib, _name)
    _f.restype, _f.argtypes = _res, _args


def _ok(rc):
    if rc != 0:
        raise RuntimeError(f"moe_pipe_cuda: {_lib.mpc_error().decode()}")


_cu = ctypes.CDLL("libcuda.so.1")
_cu.cuMemAdvise.argtypes = [ctypes.c_uint64, ctypes.c_size_t, ctypes.c_int, ctypes.c_int]
_ADVISE = os.environ.get("MOE_PIPE_ADVISE", "1") == "1"


def _advise_host(a):
    """A slot table the agent writes every layer and the GPU reads: its pages stay on the host and the GPU reads
    them in place (cuMemAdvise preferred location CPU, accessed by the GPU, as cuda_stage's staging), instead of
    migrating to the host at each claim and back at each read. MOE_PIPE_ADVISE=0: left as MLX made it."""
    if not _ADVISE:
        return
    p, n = a.ctypes.data, a.nbytes
    rc = (_cu.cuMemAdvise(p, n, 3, -1), _cu.cuMemAdvise(p, n, 5, 0))
    if any(rc):
        print(f"  moe_pipe_cuda: cuMemAdvise of a slot table failed {rc}", flush=True)


def _ints(pairs):
    a = np.ascontiguousarray(np.asarray(list(pairs), dtype=np.int32).reshape(-1))
    return a, a.ctypes.data_as(_IP), a.size // 2


# The block, MLX's context current on this thread first (the agent and its readers take it over).
mx.eval(mx.zeros((1,)))
_ok(_lib.mpc_init())
_blk = np.ctypeslib.as_array((ctypes.c_uint32 * BLOCK_WORDS).from_address(_lib.mpc_block()))
_WAIT_S = float(os.environ.get("MOE_NATIVE_WAIT_S", "5"))
_BEAT_MS = float(os.environ.get("MOE_PIPE_BEAT_MS", "0"))      # 0: no heartbeat (the guard instead)
if not (_WAIT_S > 0 and _BEAT_MS >= 0):
    raise ValueError("MOE_NATIVE_WAIT_S must be > 0 and MOE_PIPE_BEAT_MS >= 0")
# Arrays made once and kept: never a fresh host array a call (a freed one can be a device-wide cudaFree).
_ADDR = mx.array([_lib.mpc_block_dev()], dtype=mx.uint64)
_BOUND = mx.array([int(_WAIT_S * 1e9), int(_BEAT_MS * 1e6)], dtype=mx.uint64)
# The job counter (word 0), only the GPU moves it; word 1 counts MOE_PIPE_ONE's blocks out of a kernel.
_CTR = mx.array(np.array([int(_blk[POSTED]), 0, 0, 0], dtype=np.uint32))
# The wait's device words: job, state, records republished, heartbeat (MOE_PIPE_PIECES: pieces republished at 128 on).
_SCR = mx.zeros((128 + 1024 if _PIECES else 72,), dtype=mx.uint32)
mx.eval(_ADDR, _BOUND, _CTR, _SCR)
_LAY = {}

_HDR = r"""
__device__ __forceinline__ unsigned long long mpc_gt() {
  unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}
__device__ __forceinline__ unsigned mpc_ld(const unsigned* p) {
  unsigned v; asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v;
}
__device__ __forceinline__ uint4 mpc_ld4(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.cv.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
// One block copies `bytes` from s to d (either may be mapped host memory), 16 bytes a load where aligned.
__device__ void mpc_copy(char* d, const char* s, unsigned bytes) {
  if (((reinterpret_cast<unsigned long long>(d) | reinterpret_cast<unsigned long long>(s) | bytes) & 15) == 0) {
    const uint4* s4 = reinterpret_cast<const uint4*>(s);
    uint4* d4 = reinterpret_cast<uint4*>(d);
    const unsigned m = bytes >> 4, st = blockDim.x;
    unsigned i = threadIdx.x;
    for (; i + 3 * st < m; i += 4 * st) {
      const uint4 x0 = mpc_ld4(s4 + i), x1 = mpc_ld4(s4 + i + st), x2 = mpc_ld4(s4 + i + 2 * st),
                  x3 = mpc_ld4(s4 + i + 3 * st);
      d4[i] = x0;
      d4[i + st] = x1;
      d4[i + 2 * st] = x2;
      d4[i + 3 * st] = x3;
    }
    for (; i < m; i += st) d4[i] = mpc_ld4(s4 + i);
  } else {
    const volatile char* vs = s;
    for (unsigned i = threadIdx.x; i < bytes; i += blockDim.x) d[i] = vs[i];
  }
}
// Spin until *w >= v: 1. 0 at the host's release or another wait's give-up, or (this wait gives up: word 3 = v)
// past the bound b since t0 or, while an agent is live (word 6), hb after its heartbeat (word 4) last moved.
__device__ unsigned mpc_spin(const unsigned* P, const unsigned* w, unsigned v, unsigned long long t0,
                             unsigned long long b, unsigned long long hb, unsigned nap) {
  unsigned beat = mpc_ld(P + 4);
  unsigned long long tb = mpc_gt();
  for (unsigned round = 1;; ++round) {
    if (mpc_ld(w) >= v) return 1u;
    if (round & 7u) continue;     // the stop words every 8th look: one host read a look while it waits
    if (mpc_ld(P + 2) != 0u || mpc_ld(P + 3) != 0u) return 0u;
    const unsigned long long t = mpc_gt();
    const unsigned b2 = mpc_ld(P + 4);
    if (b2 != beat) {
      const unsigned gap = static_cast<unsigned>((t - tb) / 1000);   // word 12: the longest the GPU saw (us)
      if (gap > mpc_ld(P + 12)) reinterpret_cast<volatile unsigned*>(const_cast<unsigned*>(P))[12] = gap;
      beat = b2;
      tb = t;
    }
    if (t - t0 > b || (hb && mpc_ld(P + 6) != 0u && t - tb > hb)) {
      reinterpret_cast<volatile unsigned*>(const_cast<unsigned*>(P))[3] = v;
      __threadfence_system();
      return 0u;
    }
    if (nap) __nanosleep(nap);
  }
}
"""

# Job v = ++counter: its ids into ring entry v % 8 (128 words: v, layer, n, ids; 126-127 the post time), then
# posted = v, each behind a system fence. The output is v, the wait's input, so MLX orders the two.
_SIG = r"""
  if (threadIdx.x == 0) {
    unsigned* c = const_cast<unsigned*>(ctr);
    const unsigned v = c[0] + 1;
    c[0] = v;
    unsigned* P = reinterpret_cast<unsigned*>(blk[0]);
    unsigned* e = P + 64 + 128 * (v % 8);
    for (int j = 0; j < NK; ++j) e[3 + j] = static_cast<unsigned>(ids[j]);
    e[1] = static_cast<unsigned>(lay[0]);
    e[2] = NK;
    if (LOG) {
      const unsigned long long t = mpc_gt();
      e[126] = static_cast<unsigned>(t);
      e[127] = static_cast<unsigned>(t >> 32);
    }
    __threadfence_system();
    reinterpret_cast<volatile unsigned*>(e)[0] = v;
    __threadfence_system();
    reinterpret_cast<volatile unsigned*>(P)[0] = v;
    __threadfence_system();
    tick[0] = v;
  }
"""

# Only block 0's first warp (the leader) reads the mapped block: lane 0 waits for the claim (word 7 >= v) and the
# plan being job v's own (word 8192), then each lane polls its records' flags (word 6144 + record) and republishes
# each one that reached v into device memory (scr), where the other blocks wait for it: host memory polled by one
# warp, not by every block (32 blocks polling it put 50-300 us between the agent's writes and the GPU seeing them).
# Block c >= 1 copies plan entries c - 1, c - 1 + 31, ...: when the entry has a save address, the row's old bytes
# (the evicted expert's part) to its tier slot first; then, once its record is republished, from mapped staging
# or the tier to the slot's row, 16 bytes a load where aligned. scr[0] = v once job v's state scr[1] is set: 1 copy,
# 2 do not (a release, a give-up, or a plan that is not v's: the agent's error; that pass fails at its token
# boundary). Then block 0 writes the gather rows from the slot table the claim wrote; the gather is the next
# kernel, after every block's copies.
_WAIT_LEAD = r"""
  const unsigned* P = reinterpret_cast<const unsigned*>(blk[0]);
  unsigned* D = const_cast<unsigned*>(scr);
  const unsigned v = tick[0];
  const unsigned long long t0 = mpc_gt(), b = bound[0], hb = bound[1];
  unsigned long long tc = 0, tf = 0;
  if (blockIdx.x == 0) {
    if (threadIdx.x < 32) {
      const unsigned lane = threadIdx.x;
      unsigned ok = 0, nrec = 0;
      if (lane == 0) {
        ok = mpc_spin(P, P + 7, v, t0, b, hb, 0u) && mpc_ld(P + 8192) == v;
        tc = mpc_gt();
      }
      ok = __shfl_sync(0xffffffffu, ok, 0);
      if (ok) nrec = mpc_ld(P + 8194);
      if (lane == 0) {
        D[1] = ok ? 1u : 2u;
        __threadfence();
        reinterpret_cast<volatile unsigned*>(D)[0] = v;
      }
      if (ok) {
        unsigned mine = 0;          // bit k: record lane + 32 k seen
        for (unsigned k = 0; lane + 32 * k < nrec; ++k) mine |= 1u << k;
        for (unsigned round = 0;; ++round) {
          for (unsigned k = 0; lane + 32 * k < nrec; ++k) {
            const unsigned r = lane + 32 * k;
            if ((mine >> k & 1u) && mpc_ld(P + 6144 + r) >= v) {
              reinterpret_cast<volatile unsigned*>(D)[2 + r] = v;
              mine &= ~(1u << k);
            }
          }
          const unsigned full = nrec > lane ? (nrec > lane + 32 ? 3u : 1u) : 0u;
          if (LOG && !tf && __any_sync(0xffffffffu, mine != full)) tf = mpc_gt();
          if (__all_sync(0xffffffffu, mine == 0u)) break;
          unsigned stop = 0;
          if (lane == 0 && (round & 7u) == 7u) {
            const unsigned long long t = mpc_gt();
            stop = mpc_ld(P + 2) != 0u || mpc_ld(P + 3) != 0u || t - t0 > b;
            if (!stop && hb && mpc_ld(P + 6) != 0u) {
              // the heartbeat: no move for hb ns ends this wait as the bound does
              const unsigned bt = mpc_ld(P + 4);
              if (bt != D[64] && D[67] == v) {
                const unsigned gap = static_cast<unsigned>(
                    (t - ((static_cast<unsigned long long>(D[66]) << 32) | D[65])) / 1000);
                if (gap > mpc_ld(P + 12)) reinterpret_cast<volatile unsigned*>(const_cast<unsigned*>(P))[12] = gap;
              }
              if (bt != D[64] || D[67] != v) {
                D[64] = bt;
                D[65] = static_cast<unsigned>(t);
                D[66] = static_cast<unsigned>(t >> 32);
                D[67] = v;
              } else {
                stop = t - ((static_cast<unsigned long long>(D[66]) << 32) | D[65]) > hb;
              }
            }
            if (stop && mpc_ld(P + 2) == 0u && mpc_ld(P + 3) == 0u) {
              reinterpret_cast<volatile unsigned*>(const_cast<unsigned*>(P))[3] = v;
              __threadfence_system();
            }
          }
          if (__shfl_sync(0xffffffffu, stop, 0)) {
            if (lane == 0) {
              __threadfence();
              reinterpret_cast<volatile unsigned*>(D)[1] = 2u;
            }
            break;
          }
        }
      }
    }
    __syncthreads();
    if (LOG && threadIdx.x == 0) {
      volatile unsigned long long* L =
          reinterpret_cast<volatile unsigned long long*>(const_cast<unsigned*>(P) + 2048) + 8 * (v % 128);
      const volatile unsigned* e = P + 64 + 128 * (v % 8);
      L[1] = (static_cast<unsigned long long>(e[127]) << 32) | e[126];
      L[2] = t0;
      L[3] = mpc_gt();
      L[4] = tc;
      L[5] = 0;
      L[6] = tf;
      L[7] = 0;
      L[0] = v;
    }
    if (threadIdx.x < NK) {
      const volatile int* T = reinterpret_cast<const volatile int*>(tab);
      const int s = T[ids[threadIdx.x]];
      rhs[threadIdx.x] = s < 0 ? 0u : static_cast<unsigned>(s);
    }
    return;
  }
"""
_WAIT_COPY = r"""  // thread 0 reads the host's words, the block takes them from shared memory
  __shared__ unsigned go, n, sh[8];
  if (threadIdx.x == 0) {
    const volatile unsigned* VD = D;
    while (VD[0] < v) {}
    __threadfence();
    go = VD[1] == 1u;
    if (go) n = mpc_ld(P + 8193);
  }
  __syncthreads();
  if (!go) return;
  for (unsigned c = blockIdx.x - 1; c < n; c += gridDim.x - 1) {
    __syncthreads();
    if (threadIdx.x == 0) {
      const volatile unsigned* E = P + 8196 + 8 * c;
      for (int j = 0; j < 8; ++j) sh[j] = E[j];
    }
    __syncthreads();
    char* d = reinterpret_cast<char*>((static_cast<unsigned long long>(sh[1]) << 32) | sh[0]);
    char* sv = reinterpret_cast<char*>((static_cast<unsigned long long>(sh[7]) << 32) | sh[6]);
    if (sv) {
      // the evicted expert's part out to its tier slot first, while the reads are still in flight
      mpc_copy(sv, d, sh[4]);
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      const volatile unsigned* VD = D;
      while (VD[2 + sh[5]] < v && VD[1] == 1u) {}
      __threadfence();
      go = VD[2 + sh[5]] >= v;
    }
    __syncthreads();
    if (!go) break;
    mpc_copy(d, reinterpret_cast<const char*>((static_cast<unsigned long long>(sh[3]) << 32) | sh[2]), sh[4]);
  }
"""
_WAIT = _WAIT_LEAD + _WAIT_COPY

# MOE_PIPE_PIECES: the leader as above over the job's pieces (word 8194 counts ready words), republished at scr[128 + piece]. Block c >= 1
# owns units c - 1, c - 1 + 31, ..., unit u slice u % SL of plan entry u / SL, so an entry's SL slices sit on SL blocks side by side. A
# block first saves every unit's old bytes to the tier (while the reads are in flight) and notes its piece, then copies, each round the
# first of its units whose piece is republished, so no unit waits behind one whose piece lands later (an entry's words read by 8 threads).
_WAIT_P = _WAIT_LEAD.replace("D)[2 + r] = v;", "D)[128 + r] = v;") + r"""
  constexpr unsigned UM = (1020u * SL + 30u) / 31u + 1u;
  __shared__ unsigned go, n, nu, rem, pick, sh[8], pc[UM];
  __shared__ unsigned char left[UM];
  const volatile unsigned* VD = D;
  if (threadIdx.x == 0) {
    while (VD[0] < v) {}
    __threadfence();
    go = VD[1] == 1u;
    if (go) n = mpc_ld(P + 8193);
  }
  __syncthreads();
  if (!go) return;
  const unsigned g1 = gridDim.x - 1, b1 = blockIdx.x - 1;
  if (threadIdx.x == 0) nu = rem = n * SL > b1 ? (n * SL - b1 + g1 - 1) / g1 : 0u;
  for (unsigned j = 0;; ++j) {
    __syncthreads();
    if (j >= nu) break;
    if (threadIdx.x < 8)
      sh[threadIdx.x] = reinterpret_cast<const volatile unsigned*>(P + 8196 + 8 * ((b1 + j * g1) / SL))[threadIdx.x];
    __syncthreads();
    const unsigned len = sh[4], per = ((len + SL - 1) / SL + 15u) & ~15u, lo = min(len, (b1 + j * g1) % SL * per),
                   cnt = min(len - lo, per);
    char* sv = reinterpret_cast<char*>((static_cast<unsigned long long>(sh[7]) << 32) | sh[6]);
    if (threadIdx.x == 0) {
      pc[j] = sh[5];
      left[j] = 1;
    }
    if (sv && cnt) mpc_copy(sv + lo, reinterpret_cast<char*>((static_cast<unsigned long long>(sh[1]) << 32) | sh[0]) + lo, cnt);
  }
  for (;;) {
    __syncthreads();
    if (threadIdx.x == 0) {
      pick = UM;
      go = VD[1] == 1u;
    }
    __syncthreads();
    if (rem == 0u) break;
    for (unsigned j = threadIdx.x; j < nu; j += blockDim.x)
      if (left[j] && VD[128 + pc[j]] >= v) atomicMin(&pick, j);
    __syncthreads();
    const unsigned j = pick;
    if (j == UM) {
      if (!go) break;
      continue;
    }
    if (threadIdx.x < 8)
      sh[threadIdx.x] = reinterpret_cast<const volatile unsigned*>(P + 8196 + 8 * ((b1 + j * g1) / SL))[threadIdx.x];
    __syncthreads();
    __threadfence();
    const unsigned len = sh[4], per = ((len + SL - 1) / SL + 15u) & ~15u, lo = min(len, (b1 + j * g1) % SL * per),
                   cnt = min(len - lo, per);
    if (cnt)
      mpc_copy(reinterpret_cast<char*>((static_cast<unsigned long long>(sh[1]) << 32) | sh[0]) + lo,
               reinterpret_cast<const char*>((static_cast<unsigned long long>(sh[3]) << 32) | sh[2]) + lo, cnt);
    if (threadIdx.x == 0) {
      left[j] = 0;
      rem--;
    }
  }
"""

# One warp writes the KT largest of NE logits, a tie to the larger index and never a NaN (KT rounds of route_cuda's warp
# argmax, lane l holding experts l, l + 32, ...), into guess entry v % 8 at word 7168 + 32 * (v % 8) as [v, n, ids] with
# v last behind a system fence.
_GUESS = r"""
  const int lane = threadIdx.x;
  const unsigned v = tick[0];
  float g[NPL];
  for (int i = 0; i < NPL; ++i) {
    const int e = lane + 32 * i;
    g[i] = e < NE ? static_cast<float>(lg[e]) : -INFINITY;
  }
  unsigned* G = reinterpret_cast<unsigned*>(blk[0]) + 7168 + 32 * (v % 8);
  for (int j = 0; j < KT; ++j) {
    float bv = -INFINITY;
    int bi = -1;
    for (int i = 0; i < NPL; ++i) {
      if (g[i] >= bv) { bv = g[i]; bi = lane + 32 * i; }
    }
    for (int m = 16; m > 0; m >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, bv, m);
      const int oi = __shfl_xor_sync(0xffffffffu, bi, m);
      if (ov > bv || (ov == bv && oi > bi)) { bv = ov; bi = oi; }
    }
    if (lane == 0) G[2 + j] = static_cast<unsigned>(bi);
    for (int i = 0; i < NPL; ++i) {
      if (lane + 32 * i == bi) g[i] = -INFINITY;
    }
  }
  if (lane == 0) {
    G[1] = KT;
    __threadfence_system();
    reinterpret_cast<volatile unsigned*>(G)[0] = v;
    __threadfence_system();
    tock[0] = v;
  }
"""

# MOE_PIPE_ONE: signal, post_guess and wait_rows (pieces mode) as one kernel a layer. Every block reads the job number v = ctr[0] + 1
# at its start and the last block to leave raises ctr[0] to v (ctr[1] counts the blocks out), since the next layer's kernel starts only
# after this one has ended. Block 0's first warp reads table[ids] first: when every id is held, the agent's claim of the job will move
# no slot (a hit keeps its slot, claim() / claim_store()), so the job is posted and the rows go out without waiting for the claim; the
# agent still claims every job in order, so the LRU, the quota rows, the ring, the tier and the logs see the same jobs as before. To
# keep the job and guess rings (8 entries) from being overrun, a job is posted only once the agent has answered job v - LAG. A job
# with a miss waits as wait_rows does.
_ONE_HEAD = r"""
  const unsigned* P = reinterpret_cast<const unsigned*>(blk[0]);
  unsigned* D = const_cast<unsigned*>(scr);
  unsigned* C = const_cast<unsigned*>(ctr);
  const unsigned v = C[0] + 1;
  const unsigned long long t0 = mpc_gt(), b = bound[0], hb = bound[1];
  unsigned long long tc = 0, tf = 0;
  __shared__ unsigned all_s;
  __shared__ int held[32];
  if (blockIdx.x == 0) {
    if (threadIdx.x < 32) {
      const unsigned lane = threadIdx.x;
      int s = 0;
      if (lane < NK) s = reinterpret_cast<const volatile int*>(tab)[ids[lane]];
      held[lane] = s;
      const bool all = __all_sync(0xffffffffu, lane >= NK || s >= 0);
      unsigned gp[GK > 0 ? GK : 1];
      /*G*/if (GK > 0) {
        // _GUESS's picks: the GK largest of GNE logits, a tie to the larger index
        float g[GNPL];
        for (int i = 0; i < GNPL; ++i) {
          const int e = lane + 32 * i;
          g[i] = e < GNE ? static_cast<float>(lg[e]) : -INFINITY;
        }
        for (int j = 0; j < GK; ++j) {
          float bv = -INFINITY;
          int bi = -1;
          for (int i = 0; i < GNPL; ++i) {
            if (g[i] >= bv) { bv = g[i]; bi = lane + 32 * i; }
          }
          for (int m = 16; m > 0; m >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, m);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, m);
            if (ov > bv || (ov == bv && oi > bi)) { bv = ov; bi = oi; }
          }
          gp[j] = static_cast<unsigned>(bi);
          for (int i = 0; i < GNPL; ++i) {
            if (lane + 32 * i == bi) g[i] = -INFINITY;
          }
        }
      }/*G*/
      unsigned go = 1;
      if (lane == 0) {
        // `done` as last read (scr[70], it only grows) spares the host read while it is recent enough
        volatile unsigned* VD = D;
        if (v > LAG && VD[70] + LAG < v) {
          go = mpc_spin(P, P + 1, v - LAG, t0, b, hb, 0u);
          VD[70] = mpc_ld(P + 1);
        }
        unsigned* Pw = const_cast<unsigned*>(P);
        unsigned* e = Pw + 64 + 128 * (v % 8);
        unsigned* G = Pw + 7168 + 32 * (v % 8);
        for (int j = 0; j < NK; ++j) e[3 + j] = static_cast<unsigned>(ids[j]);
        e[1] = static_cast<unsigned>(lay[0]);
        e[2] = NK;
        if (LOG) {
          e[126] = static_cast<unsigned>(t0);
          e[127] = static_cast<unsigned>(t0 >> 32);
        }
        if (GK > 0) {
          for (int j = 0; j < GK; ++j) G[2 + j] = gp[j];
          G[1] = GK;
        }
        // the entries, then their job words, then `posted`: the agent reads them in that order
        __threadfence_system();
        reinterpret_cast<volatile unsigned*>(e)[0] = v;
        if (GK > 0) reinterpret_cast<volatile unsigned*>(G)[0] = v;
        __threadfence_system();
        reinterpret_cast<volatile unsigned*>(Pw)[0] = v;
        all_s = (all || !go) ? 1u : 0u;
        if (all_s) {
          // no claim to wait for: the copy blocks leave at once
          D[1] = 2u;
          __threadfence();
          reinterpret_cast<volatile unsigned*>(D)[0] = v;
        }
      }
      if (!__shfl_sync(0xffffffffu, all_s, 0)) {
"""
# then _WAIT_LEAD's warp-0 body (from `unsigned ok = 0` to its records loop), closed below
_ONE_MID = r"""
      }
    }
    __syncthreads();
    if (LOG && threadIdx.x == 0) {
      volatile unsigned long long* L =
          reinterpret_cast<volatile unsigned long long*>(const_cast<unsigned*>(P) + 2048) + 8 * (v % 128);
      L[1] = t0;
      L[2] = t0;
      L[3] = mpc_gt();
      L[4] = tc;
      L[5] = 0;
      L[6] = tf;
      L[7] = all_s;
      L[0] = v;
    }
    if (threadIdx.x < NK) {
      int s = held[threadIdx.x];
      if (!all_s) s = reinterpret_cast<const volatile int*>(tab)[ids[threadIdx.x]];
      rhs[threadIdx.x] = s < 0 ? 0u : static_cast<unsigned>(s);
    }
    /*SORT*/
  } else do {
"""
# MOE_CUDA_ROWS: the rows of a T-row job sorted by slot (stable): srt the slots, src each one's row of x (id / KR), inv each
# id's place in srt, so the experts' run kernels read each slot's weights once for every row that wants it.
_ONE_SORT = r"""
    if (threadIdx.x < NK) held[threadIdx.x] = static_cast<int>(rhs[threadIdx.x]);
    __syncthreads();
    if (threadIdx.x < NK) {
      const int me = held[threadIdx.x];
      unsigned rank = 0;
      for (int j = 0; j < NK; ++j) rank += (held[j] < me) || (held[j] == me && j < static_cast<int>(threadIdx.x));
      srt[rank] = static_cast<unsigned>(me);
      src[rank] = threadIdx.x / KR;
      inv[threadIdx.x] = rank;
    }
"""
# then _WAIT_P's copy part with its `return` as `break`, then every block leaves through the counter
_ONE_TAIL = r"""
  } while (0);
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    if (atomicAdd(C + 1, 1u) == gridDim.x - 1) {
      C[1] = 0u;
      __threadfence();
      reinterpret_cast<volatile unsigned*>(C)[0] = v;
    }
  }
"""
LAG = 4


def _one_src(guess, sort=False):
    """The kernel's source, without the guess's lines when it takes no logits, with the rows sorted by slot under `sort`."""
    head = _ONE_HEAD
    if not guess:
        a, z = head.index("/*G*/"), head.rindex("/*G*/")
        head = head[:a] + head[z + 5:]
    lead = _WAIT_LEAD.replace("D)[2 + r] = v;", "D)[128 + r] = v;")
    a, z = lead.index("      unsigned ok = 0, nrec = 0;"), lead.index("    __syncthreads();\n    if (LOG")
    body = lead[a:z].rstrip()
    assert body.endswith("}\n    }"), body[-40:]
    body = body[:body.rindex("}")]          # the leader's `if (threadIdx.x < 32) {` closes in _ONE_MID
    copy = _WAIT_P[len(lead):]
    assert _WAIT_P.startswith(lead) and copy.count("if (!go) return;") == 1
    mid = _ONE_MID.replace("/*SORT*/", _ONE_SORT if sort else "")
    return head + body + mid + copy.replace("if (!go) return;", "if (!go) break;") + _ONE_TAIL


def _lay(layer):
    lay = _LAY.get(layer)
    if lay is None:
        lay = _LAY[layer] = mx.array([layer], dtype=mx.int32)
        mx.eval(lay)
    return lay


def sync_rows(ids, layer, table, lg=None, k=0, sort=False):
    """MOE_PIPE_ONE: signal(ids, layer), post_guess(tick, lg, k) when lg is given and wait_rows(tick, ids, table) in one
    kernel; a job whose ids the table all holds does not wait for its claim. -> the gather rows (uint32, ids' shape), and
    under `sort` (ids [..., rows, KR]) also the rows sorted by slot: (rows, sorted slots, their row of x, each id's place)."""
    assert ids.size <= 32 and _PIECES
    guess = lg is not None
    if guess:
        assert k <= 30 and lg.shape[-1] <= 1024 and lg.dtype in (mx.float32, mx.bfloat16)
    ne = lg.shape[-1] if guess else 32
    name = ("mpc_one_g" if guess else "mpc_one") + ("_s" if sort else "")
    ins = ["ids", "lay", "ctr", "blk", "tab", "bound", "scr"] + (["lg"] if guess else [])
    outs = ["rhs"] + (["srt", "src", "inv"] if sort else [])
    if name not in _kern:
        _kern[name] = mx.fast.cuda_kernel(name=name, input_names=ins, output_names=outs, source=_one_src(guess, sort),
                                          header=_HDR)
    arrs = [ids, _lay(layer), _CTR, _ADDR, table, _BOUND, _SCR] + ([lg] if guess else [])
    n = ids.size
    got = _kern[name](
        inputs=arrs, template=[("NK", n), ("LOG", _LOG), ("SL", _SLICES), ("GK", k if guess else 0),
                               ("GNE", ne), ("GNPL", (ne + 31) // 32), ("LAG", LAG)] + ([("KR", ids.shape[-1])] if sort else []),
        grid=(256 * BLOCKS, 1, 1), threadgroup=(256, 1, 1), output_shapes=[ids.shape] + [(n,)] * (len(outs) - 1),
        output_dtypes=[mx.uint32] * len(outs))
    return got if sort else got[0]


# MOE_PIPE_HITS_FIRST: the slot of each id from the table as it stands before the job is posted (volatile loads, as the wait's
# rows), 0xFFFFFFFF for an id not held; the claim keeps a held id's slot, so these rows are the wait's rows for the hits.
_RES = r"""
  if (threadIdx.x < NK) {
    const volatile int* T = reinterpret_cast<const volatile int*>(tab);
    const int s = T[ids[threadIdx.x]];
    rows[threadIdx.x] = s < 0 ? 0xFFFFFFFFu : static_cast<unsigned>(s);
  }
"""
SKIP = 0xFFFFFFFF

_kern = {}


def _kernel(name, src, ins, outs):
    if name not in _kern:
        _kern[name] = mx.fast.cuda_kernel(name=name, input_names=ins, output_names=outs, source=src, header=_HDR)
    return _kern[name]


def signal(ids, layer):
    """Post job (layer, ids) for the agent once the GPU gets here; -> the job number, a (1,) uint32 array."""
    return _kernel("mpc_signal", _SIG, ["ids", "lay", "ctr", "blk"], ["tick"])(
        inputs=[ids, _lay(layer), _CTR, _ADDR], template=[("NK", ids.size), ("LOG", _LOG)], grid=(1, 1, 1),
        threadgroup=(1, 1, 1), output_shapes=[(1,)], output_dtypes=[mx.uint32])[0]


def wait_rows(tick, ids, table):
    """table[ids] as uint32 gather rows (ids' shape), once the agent has claimed job `tick` and its records are
    copied into their slots."""
    assert ids.size <= 256
    if _PIECES:
        return _kernel("mpc_wait_p", _WAIT_P, ["tick", "ids", "tab", "blk", "bound", "scr"], ["rhs"])(
            inputs=[tick, ids, table, _ADDR, _BOUND, _SCR], template=[("NK", ids.size), ("LOG", _LOG), ("SL", _SLICES)],
            grid=(256 * BLOCKS, 1, 1), threadgroup=(256, 1, 1), output_shapes=[ids.shape],
            output_dtypes=[mx.uint32])[0]
    return _kernel("mpc_wait", _WAIT, ["tick", "ids", "tab", "blk", "bound", "scr"], ["rhs"])(
        inputs=[tick, ids, table, _ADDR, _BOUND, _SCR], template=[("NK", ids.size), ("LOG", _LOG)],
        grid=(256 * BLOCKS, 1, 1), threadgroup=(256, 1, 1), output_shapes=[ids.shape],
        output_dtypes=[mx.uint32])[0]


def resident(ids, table):
    """MOE_PIPE_HITS_FIRST: table[ids] as uint32 rows with SKIP for an id the table does not hold, read before the job of
    these ids is posted (the caller orders it by mx.depends)."""
    assert ids.size <= 256
    return _kernel("mpc_resident", _RES, ["ids", "tab"], ["rows"])(
        inputs=[ids, table], template=[("NK", ids.size)], grid=(256, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[ids.shape], output_dtypes=[mx.uint32])[0]


def post_guess(tick, lg, k):
    """The top k of the next layer's gate logits `lg` (E of them, f32) written beside job `tick` for the agent's staging;
    -> a (1,) uint32 array the wait must depend on, so the kernel runs between the job's post and its wait."""
    ne = lg.shape[-1]
    assert k <= 30 and ne <= 1024 and lg.dtype in (mx.float32, mx.bfloat16)
    return _kernel("mpc_guess", _GUESS, ["tick", "lg", "blk"], ["tock"])(
        inputs=[tick, lg, _ADDR], template=[("NE", ne), ("KT", k), ("NPL", (ne + 31) // 32)], grid=(32, 1, 1),
        threadgroup=(32, 1, 1), output_shapes=[(1,)], output_dtypes=[mx.uint32])[0]


# MOE_SPEC_AHEAD: a speculative round's argmax rows and draft ids written into the mapped block, so the host reads its
# verdict with no sync while it builds the next round: words VERDICT + 1 on the ids, VERDICT the round's number, last.
VERDICT = 1280
_VCTR = mx.zeros((1,), dtype=mx.uint32)
mx.eval(_VCTR)
_VSENT = [0]
_VER = r"""
  if (threadIdx.x == 0) {
    unsigned* c = const_cast<unsigned*>(ctr);
    const unsigned v = c[0] + 1;
    c[0] = v;
    unsigned* P = reinterpret_cast<unsigned*>(blk[0]) + 1280;
    for (int j = 0; j < NA; ++j) P[1 + j] = static_cast<unsigned>(am[j]);
    for (int j = 0; j < ND; ++j) P[1 + NA + j] = static_cast<unsigned>(dr[j]);
    __threadfence_system();
    reinterpret_cast<volatile unsigned*>(P)[0] = v;
    __threadfence_system();
    seq[0] = v;
  }
"""


def verdict(am, drafts):
    """The round's argmax rows `am` and draft ids into the mapped block once the GPU has them; -> (an array to hand to
    the GPU with them, the round's number for peek)."""
    assert am.size + drafts.size <= 32
    ins = [am, drafts if drafts.size else am, _VCTR, _ADDR]
    out = _kernel("mpc_verdict", _VER, ["am", "dr", "ctr", "blk"], ["seq"])(
        inputs=ins, template=[("NA", am.size), ("ND", drafts.size)], grid=(1, 1, 1), threadgroup=(1, 1, 1),
        output_shapes=[(1,)], output_dtypes=[mx.uint32])[0]
    _VSENT[0] += 1
    return out, _VSENT[0]


def peek(n, count):
    """Round n's `count` ids (argmax rows, then drafts) if its verdict is in the block, else None; never waits."""
    if int(_blk[VERDICT]) != n:
        return None
    return _blk[VERDICT + 1:VERDICT + 1 + count].tolist()


def committed():
    """One more job is on its way to the GPU (its layer was mx.async_eval'd)."""
    _blk[SENT] = _blk[SENT] + 1


_warm = set()


def warm(k):
    """Both kernels for k ids compiled and run once, before any agent exists (install), since loading a kernel while
    a wait is pending holds the host until the GPU drains; `done` is raised ahead so the job answers itself."""
    if k in _warm:
        return
    if Agent.live:
        raise RuntimeError("moe_pipe_cuda.warm: an agent is live")
    _blk[DONE] = _blk[CLAIMED] = max(int(_blk[DONE]), int(_blk[POSTED]) + 1)
    ids = mx.zeros((1, 1, k), dtype=mx.uint32)
    tab = mx.zeros((max(k, 1),), dtype=mx.int32)
    mx.eval(ids, tab)
    mx.eval(wait_rows(signal(ids, 0), ids, tab))
    _blk[SENT] = _blk[POSTED]
    _warm.add(k)


def warm_resident(k):
    """resident() for k ids compiled and run once before any agent exists, as warm()."""
    if ("res", k) in _warm:
        return
    if Agent.live:
        raise RuntimeError("moe_pipe_cuda.warm_resident: an agent is live")
    ids, tab = mx.zeros((1, 1, k), dtype=mx.uint32), mx.full((max(k, 1),), -1, dtype=mx.int32)
    mx.eval(ids, tab)
    mx.eval(resident(ids, tab))
    _warm.add(("res", k))


def warm_guess(ne, k, dtype=mx.float32):
    """post_guess for ne logits and k picks compiled and run once before any agent exists, as warm(); it writes entry 0
    with job 0, which no agent ever takes."""
    if (ne, k, dtype) in _warm:
        return
    if Agent.live:
        raise RuntimeError("moe_pipe_cuda.warm_guess: an agent is live")
    tick, lg = mx.zeros((1,), dtype=mx.uint32), mx.zeros((1, 1, ne), dtype=dtype)
    mx.eval(tick, lg)
    mx.eval(post_guess(tick, lg, k))
    _warm.add((ne, k, dtype))


def warm_one(k, ne=0, dtype=mx.float32, rows=1, sort=False):
    """sync_rows for rows x k ids (and a guess of k of ne logits when ne; sorted under `sort`) compiled and run once
    before any agent exists, as warm(): `done` is raised ahead and the table holds every id, so the job answers itself."""
    key = ("one", k, ne, dtype, rows, sort)
    if key in _warm:
        return
    if Agent.live:
        raise RuntimeError("moe_pipe_cuda.warm_one: an agent is live")
    _blk[DONE] = _blk[CLAIMED] = max(int(_blk[DONE]), int(_blk[POSTED]) + 1)
    ids = mx.zeros((1, rows, k), dtype=mx.uint32)
    tab = mx.zeros((max(k, 1),), dtype=mx.int32)
    lg = mx.zeros((1, 1, ne), dtype=dtype) if ne else None
    mx.eval(ids, tab, *([lg] if ne else []))
    mx.eval(sync_rows(ids, 0, tab, lg, k, sort))
    _blk[SENT] = _blk[POSTED]
    _warm.add(key)


def beat_gaps():
    """(the heartbeat's longest gap the GPU saw, the beat thread's longest gap between bumps, in us; the beat
    thread's gaps over 1 ms) since the process started."""
    return int(_blk[12]), int(_blk[13]), int(_blk[14])


def now_ns():
    return _lib.mpc_now_ns()


def release():
    """Every GPU wait of this process falls through, now and later."""
    _lib.mpc_release()


def stopped():
    """(released, the job a wait gave up on or 0)."""
    return int(_blk[RELEASE]), int(_blk[GAVE])


def pending():
    """A committed job the agent has not answered (a KILL now would leave its wait spinning)."""
    return bool(_lib.mpc_pending())


TIER_FLOOR = 2.5e9      # the guards' MemAvailable bar


def tier_slots(slot_bytes, experts):
    """The victim tier's size in experts, read at install: a third of the free RAM (MemAvailable) above the guards'
    floor, capped at the model's experts and leaving the rest for the process's own growth and other apps since
    pinned memory cannot be paged out; MOE_PIPE_TIER_GB sets the size instead (within the floor's room, 0 for no
    tier). -> (slots, MemAvailable B)."""
    with open("/proc/meminfo") as f:
        avail = next(int(ln.split()[1]) * 1024 for ln in f if ln.startswith("MemAvailable:"))
    room = max(0.0, avail - TIER_FLOOR)
    gb = os.environ.get("MOE_PIPE_TIER_GB")
    want = room / 3 if gb is None else min(float(gb) * 1e9, room)
    return max(0, min(int(want // slot_bytes), experts)), avail


def wait_bound():
    """(bound in ns, ns a second, seconds): moe_native's (spin rounds, rounds a second, seconds). With
    MOE_PIPE_BEAT_MS a wait also gives up that long after the agent's heartbeat stopped."""
    return int(_WAIT_S * 1e9), 1e9, _WAIT_S


def wait_log():
    """MOE_PIPE_LOG=1: the last 128 waits as rows (job, posted, wait entered, the leader done (every record
    republished), the leader saw the claim, 0, saw a first record land, 0; 0: did not) in GPU ns, oldest first,
    and the agent's word for each: misses | tier hits << 8 | experts put in the tier << 16."""
    a = np.ctypeslib.as_array((ctypes.c_uint64 * (128 * 8)).from_address(_lib.mpc_block() + LOG * 4))
    a = a.reshape(128, 8).copy()
    a = a[a[:, 0] > 0]
    a = a[np.argsort(a[:, 0])]
    return a, _blk[1536 + a[:, 0] % 128].copy()


def agent_log():
    """The agent's last 128 jobs as rows (job, noticed, claimed, records landed, answered), host ns
    (CLOCK_MONOTONIC; 0: that step did not run), oldest first."""
    a = np.ctypeslib.as_array((ctypes.c_uint64 * (128 * 5)).from_address(_lib.mpc_block() + 4096 * 4))
    a = a.reshape(128, 5).copy()
    a = a[a[:, 0] > 0]
    return a[np.argsort(a[:, 0])]


class SharedEvent:
    """moe_native's SharedEvent: here the block's `claimed` and `done` words (tests raise them by hand)."""

    @property
    def value(self):
        return int(_blk[DONE])

    def signal(self, v):
        if v > _blk[CLAIMED]:
            _blk[CLAIMED] = v
        if v > _blk[DONE]:
            _blk[DONE] = v


class Agent:
    """moe_native's Agent: the claims, the reads and the slot tables of the layers handed to it.

    set_layer takes the layer's slot table as its numpy view and each projection's parts as the arena's iovecs
    (iov[slot] = the parts' numpy views), so no MLX array is touched from here."""

    live = 0

    def __init__(self, ev, fd, layers, readers=8):
        self.h = _lib.mpc_agent_new(fd, layers, readers)
        if not self.h:
            raise RuntimeError(f"moe_pipe_cuda: {_lib.mpc_error().decode()}")
        Agent.live += 1
        self.closed = False
        _guard()
        self.held, self.n_exp, self.room, self.ring_n = {}, {}, {}, {}

    def close(self):
        """Stop the threads; .error and .drain() still answer (moe_native's close)."""
        if not self.closed:
            _lib.mpc_agent_close(self.h)
            Agent.live -= 1
            self.closed = True      # the C++ object is never freed; one per model install

    def drain(self, timeout_s=30.0):
        return bool(_lib.mpc_agent_drain(self.h, timeout_s))

    def arm(self):
        _lib.mpc_agent_arm(self.h)

    @property
    def error(self):
        buf = ctypes.create_string_buffer(4096)
        _lib.mpc_agent_error(self.h, buf, len(buf))
        return buf.value.decode()

    def stats(self):
        """moe_native's six (layers, hits, misses, miss layers, park ns, read ns), then the read path's (tier
        hits, experts put in the tier, records read buffered after O_DIRECT failed), since the last call."""
        out = (ctypes.c_uint64 * 9)()
        _lib.mpc_agent_stats(self.h, out)
        return tuple(out)

    @property
    def direct(self):
        """Reads go through an O_DIRECT fd (MOE_PIPE_DIRECT=0, or a file system without it: buffered)."""
        return bool(_lib.mpc_agent_direct(self.h))

    def set_tier(self, slot_bytes, n, per=0):
        """The victim tier: n slots of slot_bytes in pinned host memory, in chunks of `per` slots (0: one chunk, or the chunks
        it has), on an idle agent; called again it resizes, keeping the kept chunks' experts. -> the slots it holds."""
        held = _lib.mpc_agent_tier(self.h, slot_bytes, n, per)
        _ok(0 if held >= 0 else held)
        return held

    def tier_fill(self, layer, n_exp, e):
        """Resident mode: a free tier slot for expert e of `layer`, put as an eviction is, on an idle agent; -> its host
        address for the caller to read the expert's records into (projections back to back), 0 when none is free or e
        is held already."""
        host = ctypes.c_uint64()
        _ok(_lib.mpc_tier_fill(self.h, layer, n_exp, e, ctypes.byref(host)))
        return host.value

    def tier_find(self, layer, ids):
        """The host address of each expert's tier slot (uint64, 0 where not held), on an idle agent."""
        ids = np.ascontiguousarray(ids, dtype=np.int32)
        out = np.zeros(ids.size, dtype=np.uint64)
        _ok(_lib.mpc_tier_find(self.h, layer, ids.ctypes.data_as(_IP), ids.size, out.ctypes.data_as(_U64P)))
        return out

    def set_staging(self, cap, dist=1):
        """Staging, once and before any layer is set: post_guess's guesses, each of the layer `dist` past the job's,
        read into guess records by `cap` threads, so at most cap guess records are in flight."""
        _ok(_lib.mpc_agent_staging(self.h, cap, dist))

    def stage_stats(self):
        """moe_native's nine staging counters (guessed experts, guess records read, used by a claim, of those in flight
        then, read and not used, ns waited on promoted records, dropped unstarted, 0, failed), then the most guess
        reads in flight at once and those started during a job's own reads, since the last call."""
        out = (ctypes.c_uint64 * 11)()
        _lib.mpc_agent_stage_stats(self.h, out)
        return tuple(out)

    def set_layer(self, i, slot_of, projs, quota=(), ring=-1):
        assert slot_of.dtype == np.int32 and slot_of.flags.c_contiguous
        off = np.array([p[0] for p in projs], dtype=np.int64)
        stride = np.array([p[1] for p in projs], dtype=np.int64)
        nparts = np.array([len(p[2][0]) for p in projs], dtype=np.int32)
        addr = np.array([v.ctypes.data for p in projs for v in p[2][0]], dtype=np.uint64)
        row = np.array([v.nbytes for p in projs for v in p[2][0]], dtype=np.uint64)
        cap = len(projs[0][2])
        q = np.ascontiguousarray(np.asarray(list(quota), dtype=np.int32))
        _ok(_lib.mpc_set_layer(self.h, i, slot_of.ctypes.data, slot_of.size, cap, len(projs),
                               off.ctypes.data_as(_I64P), stride.ctypes.data_as(_I64P),
                               nparts.ctypes.data_as(_IP), addr.ctypes.data_as(_U64P), row.ctypes.data_as(_U64P),
                               q.ctypes.data_as(_IP), q.size, ring))
        _advise_host(slot_of)
        self.held[i] = (slot_of, projs)
        self.n_exp[i], self.room[i] = slot_of.size, cap

    def clear_layer(self, i):
        _ok(_lib.mpc_clear_layer(self.h, i))
        self.held.pop(i, None)

    def set_state(self, i, lru, ring=()):
        a, pa, na = _ints(lru)
        b, pb, nb = _ints(ring)
        _ok(_lib.mpc_set_state(self.h, i, pa, na, pb, nb))

    def lru(self, i):
        out, n = np.zeros(2 * self.room[i], dtype=np.int32), ctypes.c_int()
        _ok(_lib.mpc_lru(self.h, i, out.ctypes.data_as(_IP), self.room[i], ctypes.byref(n)))
        return [tuple(p) for p in out[:2 * n.value].reshape(-1, 2).tolist()]

    def take_seen(self, i):
        out = np.zeros(self.n_exp[i], dtype=np.uint32)
        _ok(_lib.mpc_take_seen(self.h, i, out.ctypes.data, out.size))
        return out.tobytes()

    def take_log(self, i):
        ni, nl = ctypes.c_int(), ctypes.c_int()
        _ok(_lib.mpc_take_log(self.h, i, None, None, ctypes.byref(ni), ctypes.byref(nl)))
        ids, lens = np.zeros(ni.value, dtype=np.int32), np.zeros(nl.value, dtype=np.int32)
        _ok(_lib.mpc_take_log(self.h, i, ids.ctypes.data, lens.ctypes.data, ctypes.byref(ni), ctypes.byref(nl)))
        return ids.tobytes(), lens.tobytes()

    def set_ring(self, r, base, next, owner):
        a, pa, na = _ints(owner)
        _ok(_lib.mpc_set_ring(self.h, r, base, next, pa, na))
        self.ring_n[r] = na

    def ring_state(self, r):
        nxt, n = ctypes.c_int(), ctypes.c_int()
        out = np.zeros(2 * self.ring_n[r], dtype=np.int32)
        _ok(_lib.mpc_ring_state(self.h, r, ctypes.byref(nxt), out.ctypes.data_as(_IP), self.ring_n[r],
                                ctypes.byref(n)))
        return nxt.value, [tuple(p) for p in out[:2 * n.value].reshape(-1, 2).tolist()]


# The guard is a process with no CUDA context of its own that outlives this one: a KILLed engine with a kernel still
# running leaves the GPU at 100% until another CUDA context is made, so when this process goes without saying bye (a
# KILL, a crash, a signal) and NVML reads the GPU busy for 2.1 s (longer than its counter's tail after any decode),
# the guard makes one and leaves (MOE_PIPE_GUARD=0: none).
_GUARD_SRC = r"""
import ctypes, signal, sys, time
for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
    signal.signal(s, signal.SIG_IGN)
if b"bye" in sys.stdin.buffer.read():
    sys.exit(0)
busy = True
try:
    nv = ctypes.CDLL("libnvidia-ml.so.1")
    h, u = ctypes.c_void_p(), (ctypes.c_uint * 2)()
    assert nv.nvmlInit_v2() == 0 and nv.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(h)) == 0
    seen = []
    for _ in range(7):
        time.sleep(0.3)
        seen.append(nv.nvmlDeviceGetUtilizationRates(h, u) == 0 and u[0] >= 50)
    busy = all(seen)
    nv.nvmlShutdown()
except Exception:
    pass
if busy:
    cu = ctypes.CDLL("libcuda.so.1")
    d, c = ctypes.c_int(), ctypes.c_void_p()
    if (cu.cuInit(0) == 0 and cu.cuDeviceGet(ctypes.byref(d), 0) == 0
            and cu.cuDevicePrimaryCtxRetain(ctypes.byref(c), d) == 0 and cu.cuCtxSetCurrent(c) == 0):
        cu.cuCtxSynchronize()
        cu.cuDevicePrimaryCtxRelease_v2(d)
"""
_GUARD = None


def _guard():
    global _GUARD
    if _GUARD is not None or os.environ.get("MOE_PIPE_GUARD", "1") == "0":
        return
    _GUARD = subprocess.Popen([sys.executable, "-I", "-c", _GUARD_SRC, "moe_pipe_cuda_guard"],
                              stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                              start_new_session=True)     # holds no pipe of this process open after it
    atexit.register(_bye)


def _bye():
    try:
        _GUARD.stdin.write(b"bye")
        _GUARD.stdin.close()
    except (OSError, ValueError):
        pass
