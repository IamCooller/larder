"""MOE_KV_HOST=1 (CUDA only, off by default): from `H` positions on, a QSA attention cache keeps its K and V in
pinned, GPU-mapped host RAM with a VRAM LRU of picked blocks, attending to the flag-off path's bytes in its order
while the card's bytes go to the expert store."""
import ctypes
import os
import weakref
from types import SimpleNamespace

import numpy as np
import mlx.core as mx
from mlx_lm.models.cache import KVCache

from engine_log import note

ON = os.environ.get("MOE_KV_HOST") == "1"
LRU_MULT = int(os.environ.get("MOE_KV_LRU") or 4)  # LRU blocks a layer, in multiples of QSA's picked blocks
LOGC = 12           # positions a chunk of the home, as a power of two (4,096: 8.4 MB a layer on Flash-Next)
MAXC = 128          # chunks a layer at most (524,288 positions)
# MemAvailable left beside the home: the PC guard's 2,560 MiB (2.68 GB) plus ~120 MB for the process's own growth.
FLOOR = float(os.environ.get("MOE_KV_HOST_FLOOR_GB") or 2.8) * 1e9
H = 1 << 62         # positions from which a cache moves to the home, set by install()
ROW_BYTES = 0       # one position's K and V in one layer, set by install()
NSEL = CR = None    # QSA's picked blocks and block size, set by install()
CLS = BASE = None   # the cache class install() put in qwen4_exp's place, and the one it replaced
STATS = {"staged": 0, "put": 0, "refused": 0}

# ------------------------------------------------------------------------------------------- kernels
# Host loads bypass the GPU caches (as moe_pipe_cuda's), so a row rewritten after a trim is never served stale.
_HDR = r"""
__device__ __forceinline__ uint4 kvh_ld(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.cv.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}

// Row pos of K (kv 0) or V (kv 1), KV head h, in the home: chunk pos >> LOGC of the table, laid out [K/V][head][C][D].
template <typename T, int D, int HK, int LOGC>
__device__ __forceinline__ uint4* kvh_row(const uint64_t* tbl, int64_t pos, int kv, int h) {
  const int64_t i = pos & ((int64_t(1) << LOGC) - 1);
  return reinterpret_cast<uint4*>(tbl[pos >> LOGC] + uint64_t(((((int64_t(kv) * HK + h) << LOGC) + i) * D) * sizeof(T)));
}

// Exclusive rank of f among the block's threads, and the block's count of f in *total; every thread calls it.
__device__ __forceinline__ int kvh_scan(bool f, int* wsum, int* total) {
  const unsigned m = __ballot_sync(0xffffffffu, f);
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  if (lane == 0) {
    wsum[w] = __popc(m);
  }
  __syncthreads();
  int off = 0, tot = 0;
  for (int i = 0; i < int(blockDim.x >> 5); ++i) {
    const int c = wsum[i];
    off += i < w ? c : 0;
    tot += c;
  }
  __syncthreads();
  *total = tot;
  return off + __popc(m & ((1u << lane) - 1u));
}
"""

# Rows p0 .. p0 + S - 1 of K and V views [1, HK, S, D] into the home, one warp a row (p0 is p0's shape minus one).
_PUT = r"""
  const int lane = threadIdx.x & 31;
  const int64_t row = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
  const int h = blockIdx.y, kv = blockIdx.z;
  if (blockIdx.x == 0 && h == 0 && kv == 0 && threadIdx.x == 0) {
    done[0] = 1;
  }
  if (row >= k_shape[2]) {
    return;
  }
  const T* src = kv ? v + h * v_strides[1] + row * v_strides[2] : k + h * k_strides[1] + row * k_strides[2];
  const int64_t s3 = kv ? v_strides[3] : k_strides[3];
  uint4* dst = kvh_row<T, D, HK, LOGC>(tbl, int64_t(p0_shape[0]) - 1 + row, kv, h);
  constexpr int VEC = 16 / sizeof(T);
  if (s3 == 1 && (reinterpret_cast<uintptr_t>(src) & 15) == 0) {
    for (int c = lane; c < D / VEC; c += 32) {
      dst[c] = reinterpret_cast<const uint4*>(src)[c];
    }
  } else {
    T* d = reinterpret_cast<T*>(dst);
    for (int c = lane; c < D; c += 32) {
      d[c] = src[c * s3];
    }
  }
"""

# K and V [1, HK, cap, D] of the home's rows 0 .. prev - 1 and zeros after, as mlx_lm's buffer of that capacity holds.
_STAGE = r"""
  const int lane = threadIdx.x & 31;
  const int64_t row = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
  const int h = blockIdx.y, kv = blockIdx.z;
  const int64_t prev = int64_t(np_shape[0]) - 1, cap = int64_t(cp_shape[0]) - 1;
  if (blockIdx.x == 0 && h == 0 && kv == 0 && threadIdx.x == 0) {
    done[0] = 1;
  }
  if (row >= cap) {
    return;
  }
  constexpr int VEC = 16 / sizeof(T);
  uint4* dst = reinterpret_cast<uint4*>((kv ? vs : ks) + (int64_t(h) * cap + row) * D);
  if (row < prev) {
    const uint4* s = kvh_row<T, D, HK, LOGC>(tbl, row, kv, h);
    for (int c = lane; c < D / VEC; c += 32) {
      dst[c] = kvh_ld(s + c);
    }
  } else {
    for (int c = lane; c < D / VEC; c += 32) {
      dst[c] = make_uint4(0, 0, 0, 0);
    }
  }
"""

# The LRU's plan for one token's picks in one block of NT threads, misses taking victims by CLOCK from the hand but
# never a slot this token uses, written to sl as the hit's slot, -2 - victim, or -1 for a miss without one.
_PLAN = r"""
  __shared__ int wsum[32];
  __shared__ int vict[NSEL];
  __shared__ int cut;
  int* so = const_cast<int*>(som);
  int* bo = const_cast<int*>(bom);
  int* st = const_cast<int*>(stp);
  int* rf = const_cast<int*>(ref);
  int* mt = const_cast<int*>(meta);
  const int j = threadIdx.x;
  const int t = mt[1] + 1;
  const int b = j < NSEL ? int(tb[j]) : -1;
  const int s = j < NSEL ? so[b] : -1;
  if (s >= 0) {
    st[s] = t;
    rf[s] = 1;
  }
  __syncthreads();
  int M;
  const int rank = kvh_scan(j < NSEL && s < 0, wsum, &M);
  const int hand = mt[0];
  int found = 0, scanned = 0;
  while (found < M && scanned < 2 * NS + NT) {
    const int x = (hand + scanned + j) % NS;
    const bool prot = st[x] == t;
    const bool el = !prot && rf[x] == 0;
    int E;
    const int e = kvh_scan(el, wsum, &E);
    const int take = min(E, M - found);
    if (j == 0) {
      cut = NT - 1;
    }
    __syncthreads();
    if (el && take == M - found && e == take - 1) {
      cut = j;
    }
    if (el && e < take) {
      vict[found + e] = x;
      st[x] = t;
    }
    __syncthreads();
    if (j <= cut && !prot && !el) {
      rf[x] = 0;
    }
    found += take;
    scanned += cut + 1;
  }
  __syncthreads();
  if (j < NSEL) {
    if (s >= 0) {
      sl[j] = s;
    } else if (rank < found) {
      const int x = vict[rank];
      const int old = bo[x];
      if (old >= 0) {
        so[old] = -1;
      }
      bo[x] = b;
      so[b] = x;
      rf[x] = 0;
      sl[j] = -2 - x;
    } else {
      sl[j] = -1;
    }
  }
  if (j == 0) {
    mt[0] = (hand + scanned) % NS;
    mt[1] = t;
    mt[2] += NSEL - M;
    mt[3] += M;
  }
"""

# _qsa_gather's rows over the home, one warp a row: picked block rows from the LRU on a hit, through the mapping on a
# miss (also written to its victim row), the trailing partial block through the mapping, and the newest position
# (n - 1) from the new row kn/vn wherever it falls; row m writes the new row into the home.
_GATHER = r"""
  const int lane = threadIdx.x & 31;
  const int row = int((int64_t(blockIdx.x) * blockDim.x + threadIdx.x) >> 5);
  const int h = blockIdx.y, kv = blockIdx.z;
  const int64_t n = np_shape[0];
  const int64_t own0 = n / CR * CR;
  const int m = NSEL * CR + int(n - own0);
  if (blockIdx.x == 0 && h == 0 && kv == 0 && threadIdx.x == 0) {
    done[0] = 1;
  }
  if (row > m) {
    return;
  }
  constexpr int VEC = 16 / sizeof(T);
  int64_t pos;
  int slot = -1, vic = -1;
  if (row == m) {
    pos = n - 1;
  } else if (row >= NSEL * CR) {
    pos = own0 + row - NSEL * CR;
  } else {
    const int jj = row / CR;
    pos = int64_t(tb[jj]) * CR + row % CR;
    const int c = sl[jj];
    slot = c >= 0 ? c : -1;
    vic = c <= -2 ? -2 - c : -1;
  }
  uint4* dst = row == m ? kvh_row<T, D, HK, LOGC>(tbl, pos, kv, h)
                        : reinterpret_cast<uint4*>((kv ? vg : kg) + (int64_t(h) * m + row) * D);
  uint4* f = vic < 0 ? nullptr
                     : reinterpret_cast<uint4*>(const_cast<T*>(pool) + (((int64_t(vic) * 2 + kv) * HK + h) * CR + row % CR) * D);
  if (pos == n - 1) {
    const T* src = kv ? vn + h * vn_strides[1] : kn + h * kn_strides[1];
    const int64_t s3 = kv ? vn_strides[3] : kn_strides[3];
    if (s3 == 1 && (reinterpret_cast<uintptr_t>(src) & 15) == 0) {
      for (int c = lane; c < D / VEC; c += 32) {
        const uint4 x = reinterpret_cast<const uint4*>(src)[c];
        dst[c] = x;
        if (f) {
          f[c] = x;
        }
      }
    } else {
      for (int c = lane; c < D; c += 32) {
        const T x = src[c * s3];
        reinterpret_cast<T*>(dst)[c] = x;
        if (f) {
          reinterpret_cast<T*>(f)[c] = x;
        }
      }
    }
  } else if (slot >= 0) {
    const uint4* s = reinterpret_cast<const uint4*>(pool + (((int64_t(slot) * 2 + kv) * HK + h) * CR + row % CR) * D);
    for (int c = lane; c < D / VEC; c += 32) {
      dst[c] = s[c];
    }
  } else {
    const uint4* s = kvh_row<T, D, HK, LOGC>(tbl, pos, kv, h);
    for (int c = lane; c < D / VEC; c += 32) {
      const uint4 x = kvh_ld(s + c);
      dst[c] = x;
      if (f) {
        f[c] = x;
      }
    }
  }
"""

# Slots holding a block at or past lim (lim's shape minus one) lose it: a trim rewrites those positions.
_INVAL = r"""
  const int x = int(blockIdx.x * blockDim.x + threadIdx.x);
  if (x == 0) {
    done[0] = 1;
  }
  if (x >= NS) {
    return;
  }
  int* so = const_cast<int*>(som);
  int* bo = const_cast<int*>(bom);
  int* rf = const_cast<int*>(ref);
  const int b = bo[x];
  if (b >= 0 && b >= int(lim_shape[0]) - 1) {
    so[b] = -1;
    bo[x] = -1;
    rf[x] = 0;
  }
"""

# name: (inputs, outputs, template names, source); `dep` and `ldep` inputs only order the launch after the last one
# that touched the home or the LRU, which MLX cannot see since those are written in place.
SPECS = {
    "put": (["k", "v", "tbl", "p0", "dep"], ["done"], ["T", "D", "HK", "LOGC"], _PUT),
    "stage": (["tbl", "np", "cp", "dep"], ["ks", "vs", "done"], ["T", "D", "HK", "LOGC"], _STAGE),
    "plan": (["tb", "som", "bom", "stp", "ref", "meta", "ldep"], ["sl"], ["NSEL", "NS", "NT"], _PLAN),
    "gather": (["pool", "tbl", "tb", "sl", "kn", "vn", "np", "dep"], ["kg", "vg", "done"],
               ["T", "D", "CR", "NSEL", "HK", "LOGC"], _GATHER),
    "inval": (["som", "bom", "ref", "lim", "ldep"], ["done"], ["NS"], _INVAL),
}
_K = {}


def _kern(name):
    k = _K.get(name)
    if k is None:
        ins, outs, _, src = SPECS[name]
        k = _K[name] = mx.fast.cuda_kernel(name=f"kvh_{name}", input_names=ins, output_names=outs, source=src,
                                           header=_HDR, ensure_row_contiguous=False)
    return k


# --------------------------------------------------------------------------------- pinned host memory
class _Cuda:
    """The driver calls the home needs, through ctypes as native/moe_pipe_cuda and the mapped S2 reader do."""

    def __init__(self):
        cu = self.cu = ctypes.CDLL("libcuda.so.1")
        P, U, I, V = ctypes.POINTER, ctypes.c_uint, ctypes.c_int, ctypes.c_void_p
        cu.cuInit.argtypes = [U]
        cu.cuCtxGetCurrent.argtypes = [P(V)]
        cu.cuCtxSetCurrent.argtypes = [V]
        cu.cuDeviceGet.argtypes = [P(I), I]
        cu.cuDevicePrimaryCtxRetain.argtypes = [P(V), I]
        cu.cuMemHostAlloc.argtypes = [P(V), ctypes.c_size_t, U]
        cu.cuMemHostGetDevicePointer_v2.argtypes = [P(ctypes.c_uint64), V, U]
        cu.cuMemFreeHost.argtypes = [V]
        if cu.cuInit(0) != 0:
            raise OSError("cuInit failed")
        self.ctx = None

    def _current(self):
        """MLX's primary context current on this thread, which a thread that never made a runtime call lacks."""
        cu, ctx = self.cu, ctypes.c_void_p()
        if cu.cuCtxGetCurrent(ctypes.byref(ctx)) == 0 and ctx.value:
            return True
        if self.ctx is None:
            dev, c = ctypes.c_int(), ctypes.c_void_p()
            if cu.cuDeviceGet(ctypes.byref(dev), 0) or cu.cuDevicePrimaryCtxRetain(ctypes.byref(c), dev):
                return False
            self.ctx = c
        return cu.cuCtxSetCurrent(self.ctx) == 0

    def alloc(self, nbytes):
        """(host, device) addresses of nbytes of pinned host memory mapped for the GPU, or None."""
        if not self._current():
            return None
        h, d = ctypes.c_void_p(), ctypes.c_uint64()
        if self.cu.cuMemHostAlloc(ctypes.byref(h), nbytes, 0x01 | 0x02) != 0:     # PORTABLE | DEVICEMAP
            return None
        if self.cu.cuMemHostGetDevicePointer_v2(ctypes.byref(d), h, 0) != 0:
            self.cu.cuMemFreeHost(h)
            return None
        return h.value, d.value


_CU = None
_FREE = {}          # chunk bytes -> [(host, device, the dep of its last user)]
PINNED = [0]        # bytes of pinned host memory this process holds for homes


def _avail():
    """MemAvailable in bytes, None where /proc/meminfo does not exist."""
    try:
        with open("/proc/meminfo") as f:
            for ln in f:
                if ln.startswith("MemAvailable:"):
                    return int(ln.split()[1]) * 1024
    except OSError:
        return None
    return None


def _take(nbytes):
    """A chunk of nbytes: a freed one first, its last user's kernels done before anyone writes it again, else a new
    pinned one while MemAvailable stays over FLOOR; None when refused."""
    free = _FREE.get(nbytes)
    if free:
        hp, dp, dep = free.pop()
        if dep is not None:
            mx.eval(dep)
        return hp, dp
    a = _avail()
    if a is not None and a - nbytes < FLOOR:
        return None
    got = _CU.alloc(nbytes) if _CU is not None else None
    if got is not None:
        PINNED[0] += nbytes
    return got


def _give(nbytes, chunks, dep):
    # Known limit: chunks are reused, never freed, so pinned RAM stays at its high-water mark; free the list when RAM runs short if that bites.
    _FREE.setdefault(nbytes, []).extend((hp, dp, dep) for hp, dp in chunks)


class _Home:
    """One cache's K and V in pinned chunks of 1 << LOGC positions, with the chunks' device addresses as a table."""

    def __init__(self, hk, d, dtype):
        self.hk, self.d, self.dtype = hk, d, dtype
        self.cb = (2 * hk * d * dtype.size) << LOGC
        self.chunks, self.tbl, self.dep = [], None, None

    def grow(self, n):
        """Room for n positions, False when a chunk is refused or n is past MAXC chunks."""
        need = (n + (1 << LOGC) - 1) >> LOGC
        if need > MAXC:
            return False
        have = len(self.chunks)
        while len(self.chunks) < need:
            c = _take(self.cb)
            if c is None:
                break
            self.chunks.append(c)
        if len(self.chunks) > have:
            ptr = [dp for _, dp in self.chunks] + [0] * (MAXC - len(self.chunks))
            self.tbl = mx.array(np.array(ptr, dtype=np.uint64))
        return len(self.chunks) >= need

    def free(self):
        if self.chunks:
            _give(self.cb, self.chunks, self.dep)
            self.chunks = []


_DUMMY = [None]
_ZERO = [None]
_ADDED = [0]                        # VRAM bytes allocated to stand (LRUs), for MemBudget.close
_LIVE = [None]                      # the cache whose scratch is alive
_HOMED = weakref.WeakSet()          # caches with a home
_NOTED = set()


def _len(n):
    """A device array whose shape is (n,): how a kernel gets a count that changes every call without a recompile or
    a fresh host array (native/moe_pipe_cuda: a freed one can cost a device-wide cudaFree)."""
    if _DUMMY[0] is None:
        _DUMMY[0] = mx.zeros(((MAXC << LOGC) + 512,), dtype=mx.uint8)     # a capacity runs up to 255 past the rows
        mx.eval(_DUMMY[0])
    return _DUMMY[0][:n]


def _zero():
    if _ZERO[0] is None:
        _ZERO[0] = mx.zeros((1,), dtype=mx.int32)
        mx.eval(_ZERO[0])
    return _ZERO[0]


def _once(key, msg):
    if key not in _NOTED:
        _NOTED.add(key)
        note(msg, flush=True)


def _grown(cap, prev, s, step):
    """The capacity mlx_lm's KVCache.update_and_fetch leaves for s rows at prev on a buffer of cap (None: none yet)."""
    if cap is not None and prev + s <= cap:
        return cap
    add = (step + s - 1) // step * step
    return add if cap is None else (prev if prev % step else cap) + add


def lru_slots(nsel):
    """LRU blocks a layer: LRU_MULT top-ks, at least one plan block's threads."""
    return max(LRU_MULT * nsel, (nsel + 31) // 32 * 32)


def drop_live():
    """MemBudget's plan and tick: the last layer's scratch goes before a pass is sized."""
    c = _LIVE[0]() if _LIVE[0] is not None else None
    if c is not None:
        c._drop_scratch()
    _LIVE[0] = None


def take_added():
    """VRAM bytes allocated to stand since the last call, which the pass that made them did not need."""
    n, _ADDED[0] = _ADDED[0], 0
    return n


def scratch_bytes(keys):
    """A prefill pass's charge for its staged layer over `keys` keys: the scratch and the copy its growth makes."""
    return 2.0 * ROW_BYTES * keys if keys >= H else 0.0


def pass_arrays(c):
    """What forcing a pass needs of a homed cache: its last kernel, its scratch and its indexer keys, where its
    `state` would stage every row back to the GPU."""
    out = [c.home.dep] if c.home.dep is not None else []
    if c.scratch is not None:
        out += [c.scratch.keys, c.scratch.values]
    ix = getattr(c, "indexer", None)
    if ix is not None and ix.keys is not None:
        out.append(ix.keys)
    return out


def report():
    """The turn's figures: the homed caches, the LRU's hits and misses since the last report, the bytes pinned."""
    cs = [c for c in list(_HOMED) if c.home is not None]
    lru = [c.lru for c in cs if c.lru is not None]
    hits = misses = 0
    if lru:
        mx.synchronize()
        for x in lru:
            h, m = (x.meta + 0).tolist()[2:4]
            hits += h - x.seen[0]
            misses += m - x.seen[1]
            x.seen = (h, m)
    st = dict(STATS)
    STATS.update(staged=0, put=0)
    return {"from_positions": H if H < (1 << 62) else None, "homed_layers": len(cs),
            "lru_blocks": max((x.slots for x in lru), default=0),
            "lru_mb": round(sum(x.nbytes for x in lru) / 1e6, 1), "hits": hits, "misses": misses,
            "miss_rate": round(misses / (hits + misses), 4) if hits + misses else None,
            "home_mb": round(sum(len(c.home.chunks) * c.home.cb for c in cs) / 1e6, 1),
            "pinned_mb": round(PINNED[0] / 1e6, 1), "staged_mb": round(st["staged"] / 1e6, 1),
            "put_mb": round(st["put"] / 1e6, 1), "refused": st["refused"]}


# --------------------------------------------------------------------------------------- the cache
def make_class(base):
    """qwen4_exp's _AttnCache with the home: the flag-off cache's own lines (mlx_lm's KVCache) below H positions,
    and above it the same views and bytes from the home."""

    class _AttnCache(base):
        __doc__ = base.__doc__
        _k = _v = None
        _off = 0
        home = scratch = lru = cap = lo = fin = None
        hi = 0              # rows of mlx_lm's buffer written (stale ones past the offset too), zeros after
        refused = False

        # Storage behind mlx_lm's attributes: in VRAM as the flag-off cache, or the home staged on a read.
        @property
        def keys(self):
            return self._k if self.home is None else self._stage(self.cap, self.hi)[0]

        @keys.setter
        def keys(self, a):
            self._leave()
            self._k = a

        @property
        def values(self):
            return self._v if self.home is None else self._stage(self.cap, self.hi)[1]

        @values.setter
        def values(self, a):
            self._leave()
            self._v = a

        @property
        def offset(self):
            return self._off

        @offset.setter
        def offset(self, n):
            # A position given back is rewritten later, so the LRU forgets its blocks before the next pass on the picks.
            if n < self._off:
                self.lo = n if self.lo is None else min(self.lo, n)
                self._drop_scratch()
            self._off = n

        @property
        def kv_host(self):
            return self.home is not None

        @property
        def nbytes(self):
            if self.home is None:
                return super().nbytes
            sc = self.scratch
            return ((0 if sc is None else sc.nbytes)
                    + (0 if self.lru is None else self.lru.nbytes))

        @property
        def state(self):
            if self.home is None:
                return super().state
            # The base class's own getter over the staged buffer, so the parts are this mlx_lm's (with or without the offset).
            tmp = base.__new__(base)
            tmp.keys, tmp.values = self._stage(self.cap, self.hi)
            tmp.offset, tmp.indexer = self._off, self.indexer
            return base.state.fget(tmp)

        @state.setter
        def state(self, v):
            base.state.fset(self, v)

        def empty(self):
            return self.home is None and self._k is None

        def update_and_fetch(self, keys, values):
            s = keys.shape[2]
            if self.home is None:
                if self._off + s < H or not self._enter(keys, s):
                    return super().update_and_fetch(keys, values)
                if self._k is not None:
                    # The flag-off buffer itself is this pass's scratch.
                    self._scratch(self._k, self._v, self._off)
                    self._k = self._v = None
            elif self._off + s < H or not self._room(self._off + s):
                # Trimmed under the line (a pass under the switch would stage every row a token) or out of RAM.
                self._to_vram()
                return super().update_and_fetch(keys, values)
            prev, sc, cap = self._off, self.scratch, self.cap
            # mlx_lm's growth cuts the buffer at prev when prev is off a step, dropping the rows written past it.
            cut = cap is None or (prev + s > cap and prev % self.step)
            if sc is None or sc.offset != prev:
                sc = self._scratch(*self._stage(_grown(cap, prev, s, self.step), prev if cut else self.hi), prev)
            self._put(keys, values, prev)
            out = sc.update_and_fetch(keys, values)
            self.cap, self._off = sc.keys.shape[2], sc.offset
            self.hi = prev + s if cut else max(self.hi, prev + s)
            return out

        def decode(self, k, v, tb, cr):
            """attn_cuda's one-row pass on the sorted picks tb: (kg, vg) as _qsa_gather gives over what
            update_and_fetch(k, v) returns, the row appended to the home; None leaves the pass to the flag-off lines."""
            if self.home is None:
                if self._off + 1 < H or not self._enter(k, 1):
                    return None
                self._k = self._v = None
            elif not self._room(self._off + 1):
                self._to_vram()
                return None
            self._drop_scratch()
            lru = self.lru
            if lru is None or lru.nsel != tb.size:
                lru = self._new_lru(k, tb.size, cr)
            if self.lo is not None:
                self._inval(self.lo // cr)
                self.lo = None
            hk, d = k.shape[1], k.shape[3]
            n = self._off + 1
            sl = _kern("plan")(
                inputs=[tb, lru.so, lru.bo, lru.st, lru.rf, lru.meta, lru.dep],
                template=[("NSEL", tb.size), ("NS", lru.slots), ("NT", lru.nt)],
                grid=(lru.nt, 1, 1), threadgroup=(lru.nt, 1, 1),
                output_shapes=[(tb.size,)], output_dtypes=[mx.int32])[0]
            m = tb.size * cr + n % cr
            kg, vg, done = _kern("gather")(
                inputs=[lru.pool, self.home.tbl, tb, sl, k, v, _len(n), self._dep()],
                template=[("T", k.dtype), ("D", d), ("CR", cr), ("NSEL", tb.size), ("HK", hk), ("LOGC", LOGC)],
                grid=(32 * (m + 1), hk, 2), threadgroup=(256, 1, 1),
                output_shapes=[(1, hk, m, d), (1, hk, m, d), (1,)], output_dtypes=[k.dtype, k.dtype, mx.int32])
            self.home.dep = lru.dep = done
            cap = self.cap
            cut = cap is None or (n > cap and self._off % self.step)
            self.cap = _grown(cap, self._off, 1, self.step)
            self.hi = n if cut else max(self.hi, n)
            self._off = n
            return kg, vg

        # ------------------------------------------------------------------------ the home's moves
        def _enter(self, keys, s):
            """Into the home for s more rows, the rows held so far copied there; False keeps this cache in VRAM."""
            # Known limit: rows of whole 16-byte words only (the kernels' copies), Flash-Next's are 512 B.
            if self.refused or keys.shape[3] * keys.dtype.size % 16:
                return False
            cap = None if self._k is None else self._k.shape[2]
            home = _Home(keys.shape[1], keys.shape[3], keys.dtype)
            if not home.grow(max(self._off + s, cap or 0)):
                home.free()
                self.refused = True
                STATS["refused"] += 1
                _once("refused", f"  kv host: pinned RAM refused at {self._off + s} positions (MemAvailable "
                                 f"{(_avail() or 0) / 1e9:.2f} GB, floor {FLOOR / 1e9:.1f} GB, {PINNED[0] / 1e9:.2f} "
                                 f"GB pinned); that conversation's K/V stays in VRAM")
                return False
            self.home, self.lo, self.cap, self.hi = home, None, cap, cap or 0
            self.fin = weakref.finalize(self, home.free)
            _HOMED.add(self)
            if NSEL is not None:
                # Made with the home, inside the pass that crosses the line, so the store sized after it counts it.
                self._new_lru(keys, NSEL, CR)
            if cap:
                self._put(self._k, self._v, 0)      # the whole buffer, so a state staged back is mlx_lm's byte for byte
            return True

        def _room(self, n):
            if self.home.grow(n):
                return True
            STATS["refused"] += 1
            _once("refused", f"  kv host: pinned RAM refused at {n} positions (MemAvailable "
                             f"{(_avail() or 0) / 1e9:.2f} GB, floor {FLOOR / 1e9:.1f} GB); back to VRAM")
            self.refused = True
            return False

        def _to_vram(self):
            """Back to the flag-off cache: the rows staged into mlx_lm's buffer of the same capacity."""
            sc = self.scratch
            if sc is not None and sc.offset == self._off:
                ks, vs = sc.keys, sc.values
            elif self.cap is not None:
                ks, vs = self._stage(self.cap, self.hi)
            else:
                ks = vs = None
            if ks is not None:
                mx.eval(ks, vs)
            self._leave()
            self._k, self._v = ks, vs

        def _leave(self):
            if self.home is None:
                return
            self._drop_scratch()
            if self.fin is not None:
                self.fin.detach()
            self.home.free()
            self.home = self.lru = self.fin = None
            _HOMED.discard(self)

        def _scratch(self, ks, vs, n):
            # One layer's scratch alive at a time: layers run one after another, so the one before is done with its own.
            drop_live()
            sc = KVCache()
            sc.keys, sc.values, sc.offset = ks, vs, n
            self.scratch = sc
            _LIVE[0] = weakref.ref(self)
            return sc

        def _drop_scratch(self):
            self.scratch = None
            if _LIVE[0] is not None and _LIVE[0]() is self:
                _LIVE[0] = None

        def _dep(self):
            return self.home.dep if self.home.dep is not None else _zero()

        def _put(self, k, v, p0):
            hk, s, d = k.shape[1], k.shape[2], k.shape[3]
            self.home.dep = _kern("put")(
                inputs=[k, v, self.home.tbl, _len(p0 + 1), self._dep()],
                template=[("T", k.dtype), ("D", d), ("HK", hk), ("LOGC", LOGC)],
                grid=(32 * s, hk, 2), threadgroup=(256, 1, 1),
                output_shapes=[(1,)], output_dtypes=[mx.int32])[0]
            STATS["put"] += s * 2 * hk * d * k.dtype.size

        def _stage(self, cap, prev):
            """K and V [1, hk, cap, d] holding the home's rows 0 .. prev - 1 and zeros after (lazy)."""
            hm = self.home
            ks, vs, done = _kern("stage")(
                inputs=[hm.tbl, _len(prev + 1), _len(cap + 1), self._dep()],
                template=[("T", hm.dtype), ("D", hm.d), ("HK", hm.hk), ("LOGC", LOGC)],
                grid=(32 * cap, hm.hk, 2), threadgroup=(256, 1, 1),
                output_shapes=[(1, hm.hk, cap, hm.d)] * 2 + [(1,)],
                output_dtypes=[hm.dtype, hm.dtype, mx.int32])
            hm.dep = done
            STATS["staged"] += prev * 2 * hm.hk * hm.d * hm.dtype.size
            return ks, vs

        def _new_lru(self, k, nsel, cr):
            ns = lru_slots(nsel)
            lru = SimpleNamespace(
                slots=ns, nsel=nsel, nt=(nsel + 31) // 32 * 32, dep=_zero(), seen=(0, 0),
                pool=mx.zeros((ns, 2, k.shape[1], cr, k.shape[3]), dtype=k.dtype),
                so=mx.full(((MAXC << LOGC) // cr,), -1, dtype=mx.int32),
                bo=mx.full((ns,), -1, dtype=mx.int32), st=mx.full((ns,), -1, dtype=mx.int32),
                rf=mx.zeros((ns,), dtype=mx.int32), meta=mx.zeros((4,), dtype=mx.int32))
            arrs = [lru.pool, lru.so, lru.bo, lru.st, lru.rf, lru.meta]
            mx.eval(arrs)
            lru.nbytes = sum(a.nbytes for a in arrs)
            _ADDED[0] += lru.nbytes
            self.lru = lru
            return lru

        def _inval(self, lim):
            lru = self.lru
            if lru is None:
                return
            lru.dep = _kern("inval")(
                inputs=[lru.so, lru.bo, lru.rf, _len(lim + 1), lru.dep], template=[("NS", lru.slots)],
                grid=(-(-lru.slots // 256) * 256, 1, 1), threadgroup=(256, 1, 1),
                output_shapes=[(1,)], output_dtypes=[mx.int32])[0]

    _AttnCache.__qualname__ = base.__qualname__
    return _AttnCache


def install(q4, switch, nsel, cr, hk, d, dtype):
    """Put the homed cache in qwen4_exp's place for every cache made from now on, homed from max(switch, twice the
    LRU's own positions): under that the VRAM freed would not pay for the LRU. Returns whether it is on."""
    global _CU, H, ROW_BYTES, CLS, BASE, NSEL, CR
    if CLS is not None:
        return True
    if nsel > 1024:
        note(f"  kv host: off ({nsel} picked blocks, the plan kernel takes 1024)", flush=True)
        return False
    try:
        _CU = _Cuda()
    except (OSError, AttributeError) as e:
        note(f"  kv host: off (libcuda: {e})", flush=True)
        return False
    BASE = q4._AttnCache
    CLS = q4._AttnCache = make_class(BASE)
    ROW_BYTES, NSEL, CR = 2 * hk * d * dtype.size, nsel, cr
    H = max(int(switch), 2 * lru_slots(nsel) * cr)
    if os.environ.get("MOE_KV_HOST_FROM"):
        # An exactness screen's line: short requests homed too, a pass under the switch then staging every row.
        H = int(os.environ["MOE_KV_HOST_FROM"])
    note(f"  kv host: QSA K/V in pinned host RAM from {H} positions, an LRU of {lru_slots(nsel)} blocks a layer "
         f"in VRAM (MOE_KV_HOST=1, MOE_KV_LRU={LRU_MULT})", flush=True)
    return True
