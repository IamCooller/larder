#!/usr/bin/env python3
"""Faster true streaming: a persistent map plus byte-range expert reads, exact because a miss still reads from disk and --lru only caches hot rows."""

import argparse
import atexit
import contextlib
import hashlib
import json
import os
import struct
import sys
import threading
import time
import weakref
from collections import Counter, OrderedDict, deque
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import ctypes

import mlx.core as mx
import mlx.nn as nn
import mlx_lm
import numpy as np
from mlx_lm.models.cache import ArraysCache, KVCache, make_prompt_cache

import ane_experts
import ane_overlap
import ane_split
import exl3_experts
import gguf_experts
import fuse_proj
import fused_router
import layer_glue
import route_k
from host_stats import Telemetry, free_disk_gb, idle_baseline, swap_used_mb
from engine_log import note

RESULTS = Path(__file__).parent / "results"
MODEL = "lmstudio-community/ERNIE-4.5-21B-A3B-MLX-8bit"

STATS = {"reads": 0, "hits": 0, "rows_read": 0, "index_reuse": 0,
         "defer_rollbacks": 0, "defer_wide": 0, "wide_split": 0, "defer_tokens": 0, "defer_regen": 0,
         "defer_now_misses": 0, "defer_now_hits": 0, "repair_half": 0,
         # DeferPolicy's sync-path signal: layers a token served from outside the arena (a layer_arena claim with jobs, or the wide/split path when the union did not fit), with defer_rollbacks + defer_wide the same miss one layer late.
         "miss_layers": 0,
         "look_issued": 0,
         "rows_warm": 0, "rows_block": 0, "need_block": 0,
         # Sidecar read counters (_Stripe): bytes each device read in read waves, and reads that failed there and went to experts.bin.
         "stripe_int_b": 0, "stripe_ext_b": 0, "stripe_fell": 0,
         # Layer-major prefill (_LMReader): bytes its reader thread read, layers read whole ahead or their union after routing, and rows copied from the store's residents instead of read.
         "lm_read_b": 0, "lm_whole": 0, "lm_union": 0, "lm_copied": 0,
         # Plain-path prefill (_whole_layer): layers that read their union then, layers found read whole ahead, SSD bytes, rows copied from the arena, and the rp_b share sent to a sidecar.
         "rp_union": 0, "rp_whole": 0, "rp_b": 0, "rp_copied": 0, "rp_ext_b": 0,
         # Staging rows (MOE_STAGING=1, _Native): guessed experts queued, their records read, guessed misses promoted (and those read on demand), guesses unclaimed, records dropped unstarted, and rows copied back at a hand-back.
         "stage_guess": 0, "stage_rec": 0, "stage_prom": 0, "stage_part": 0,
         "stage_wrong": 0, "stage_drop": 0, "stage_unstaged": 0, "stage_bad": 0,
         # The CUDA agent's read path: misses copied from the victim tier, evicted experts put there, and records read buffered after an O_DIRECT failure.
         "tier_hits": 0, "tier_puts": 0, "dio_fell": 0,
         # MOE_STORE_KEEP: quota members published again after a prefill released the store (MemBudget._put_back), the members it held at the release and its pool's slots.
         "store_kept": 0, "store_held": 0, "store_slots": 0, "store_seeded": 0,
         # MOE_NATIVE_MTP: passes of 2+ positions through the native agent.
         "native_rows": 0,
         # MOE_CPU_GUESS (_Native): guesses the agent ranked on a CPU core, the most one took (us), inputs it could not take, and under =check those compared with the GPU's guess, equal in order, equal as sets only.
         "cg_n": 0, "cg_max_us": 0, "cg_skip": 0, "cg_cmp": 0, "cg_same": 0, "cg_set": 0,
         # MOE_PLE_AHEAD: native decode passes whose PLE rows were read before layer 0, and those that first read the n-gram context off the GPU (_ple_decode_ahead).
         "ple_next": 0, "ple_ctx_sync": 0,
         # MOE_RESIDENT: a prefill group's experts copied from the tier and bytes read from experts.bin into ring rows (_take_group).
         "resident_tier_rows": 0, "resident_ssd_b": 0, "resident_ring": 0, "resident_absent": 0}
# Phase timers: I/O was 0.086 s of the ~1.2 s a token took at 499 MB a token, so the rest is what needs finding.
TIMERS = {"decode": 0.0, "fetch": 0.0, "gemm": 0.0, "sync": 0.0, "numpy": 0.0,
          "defer_build": 0.0, "defer_submit": 0.0,
          "wait_pred": 0.0, "look": 0.0,
          # MOE_NATIVE=1: the agent's time with the GPU parked on a layer (claim plus reads), and the reads alone.
          "native_park": 0.0, "native_read": 0.0,
          # With a sidecar: each device's time per read wave, summed.
          "stripe_int": 0.0, "stripe_ext": 0.0,
          # Layer-major prefill: the reader thread's reads, the model thread's waits for them, and PLE rows read in the model's call or on the thread.
          "lm_read": 0.0, "lm_wait": 0.0, "ple": 0.0, "ple_ahead": 0.0,
          # Staging rows: claims waiting for a guess still in flight, and the hand-back moving the store back to its layout.
          "stage_wait": 0.0, "stage_unstage": 0.0,
          # MOE_CPU_GUESS: the agent's time ranking guesses.
          "cpu_guess": 0.0}

# Memoising the top_k remap by the indices array's identity drops 81 scheduler syncs a token to 27, with the array kept alive so its id cannot be recycled.
_IDX_MEMO: OrderedDict = OrderedDict()
_IDX_MEMO_MAX = 8


def _decode_indices(indices, also=None, memo=True, dev=True):
    # memo=False is for callers that need what they put in `also` forced, since a memo hit skips the mx.eval, and dev=False skips an MLX array for positions nothing reads as one (MOE_CUDA_SLOTS).
    key = id(indices)
    hit = _IDX_MEMO.get(key) if memo else None
    if hit is not None:
        STATS["index_reuse"] += 1
        if dev and hit[1][1] is None:
            hit = _IDX_MEMO[key] = (hit[0], (hit[1][0], mx.array(hit[1][2]), hit[1][2]))
        return hit[1]
    # Split the two costs apart: mx.eval forces the pending graph (real compute) and the numpy bookkeeping around it is ours.
    t = time.perf_counter()
    if also is None:
        mx.eval(indices)
    else:
        mx.eval(indices, also)
    TIMERS["sync"] += time.perf_counter() - t
    t = time.perf_counter()
    # Reshape on the numpy side, since indices.reshape(-1) would build a new lazy node after the mx.eval and force a second sync (26.9 us against 1.5 us, ~1.0 ms per 40-layer pass).
    flat = np.asarray(indices).reshape(-1)
    unique_ids = np.unique(flat)
    remap = np.empty(int(unique_ids[-1]) + 1, dtype=np.int32)
    remap[unique_ids] = np.arange(len(unique_ids), dtype=np.int32)
    # pos stays numpy too, so the resident path's translation into an arena slot avoids a round trip through MLX.
    pos = remap[flat].reshape(indices.shape)
    decoded = (unique_ids, mx.array(pos) if dev else None, pos)
    if memo:
        _IDX_MEMO[key] = (indices, decoded)
        if len(_IDX_MEMO) > _IDX_MEMO_MAX:
            _IDX_MEMO.popitem(last=False)
    TIMERS["numpy"] += time.perf_counter() - t
    return decoded


# --threads sizes the pool: 8 was optimal when reads were the only work in flight, and lookahead now shares it with the speculative reads of layer L+d.
PREFILL_CHUNK = 16

_LIBC = None


def _backend():
    """'metal' (Apple), 'cuda' (Linux + NVIDIA through MLX's CUDA backend) or 'cpu'."""
    if mx.metal.is_available():
        return "metal"
    cu = getattr(mx, "cuda", None)
    return "cuda" if cu is not None and cu.is_available() else "cpu"


# The probes below (sysctl, host_statistics64, proc_pid_rusage) answer from /proc on Linux, so the guards and budgets above them run unchanged.
BACKEND = _backend()
_LINUX = sys.platform.startswith("linux")

# MOE_CUDA_GDN (on by default under CUDA, never on Metal or CPU): mlx_lm's GatedDeltaNet recurrence as a CUDA kernel instead of a per-position ops loop (gdn_cuda.py, imported only when on).
if BACKEND == "cuda" and os.environ.get("MOE_CUDA_GDN", "1") == "1":
    import gdn_cuda
    gdn_cuda.install()

# MOE_CPU_KERNELS (on by default on MLX's CPU device, never on Metal or CUDA): cpu_ext's threaded AVX-512 kernels for the quantized matmuls, the expert gather, attention and the GatedDeltaNet scan (cpu_kernels.py).
if BACKEND == "cpu" and os.environ.get("MOE_CPU_KERNELS", "1") == "1":
    import cpu_kernels
    cpu_kernels.install()

# MOE_CUDA_SLOTS (on by default under CUDA, never on Metal or CPU): the gather's slot rows and the bias mask read from the host's slot table (_slot_of) instead of fresh arrays every layer, with route_cuda.py's mask kernel bit-identical to the ~16 MLX ops.
_CUDA_SLOTS = BACKEND == "cuda" and os.environ.get("MOE_CUDA_SLOTS", "1") == "1"
# MOE_CUDA_COMBINE (CUDA only, on unless =0): mlx_lm's probability-routed MoE block's decode row as two kernels, router and combine (route_cuda.install_tail).
_CUDA_COMBINE = BACKEND == "cuda" and os.environ.get("MOE_CUDA_COMBINE", "1") == "1"
if _CUDA_SLOTS or _CUDA_COMBINE:
    import route_cuda

# MOE_RESIDENT=1 (CUDA under a budget store, off by default; the launcher sets it where the experts fit VRAM + RAM): the store lives in device memory, is filled at load with the native agent's tier holding the rest, and a prefill reads its experts from both instead of emptying the store.
_RESIDENT = _RESIDENT_ASKED = BACKEND == "cuda" and os.environ.get("MOE_RESIDENT") == "1"
# The resident tier fills RAM down to this MemAvailable and gives back all others take below it: twice the tier's own floor (moe_pipe_cuda.TIER_FLOOR, the guards' bar), so a 3 GB burst leaves the guards' 1.5 GB.
RESIDENT_FLOOR = 5.0e9
_RESIDENT_LOG = os.environ.get("MOE_RESIDENT_LOG") == "1"     # a line a pass (debug)
# MOE_RESIDENT_INPLACE (on unless =0): a prefill's GGUF chain groups read the store in place instead of a take of their rows.
_INPLACE = os.environ.get("MOE_RESIDENT_INPLACE", "1") != "0"
# MOE_RESIDENT_OVERLAP (on unless =0): a GGUF prefill's next unit of expert rows has its misses copied into the ring half the running unit does not read, with no sync a unit.
_OVERLAP = os.environ.get("MOE_RESIDENT_OVERLAP", "1") != "0"
# MOE_RESIDENT_ONECALL (on unless =0): a GGUF prefill layer's chain rows of quota experts go in one call and its misses in rounds through the ring, instead of ~20 groups.
_ONECALL = os.environ.get("MOE_RESIDENT_ONECALL", "1") != "0"
# The fresh prefill the resident store leaves room for beside it, in min_w passes (MLX's sorted-kernel line, 128 positions on the 35B), since at two the 144-token bench prompt was cut in two pieces.
RESIDENT_ROOM = 4
# MOE_EMBED_VRAM (on under MOE_RESIDENT unless =0): the embedding table stays in device memory instead of a row read from the checkpoint and dequantized every token.
EMBED_VRAM = _RESIDENT and os.environ.get("MOE_EMBED_VRAM", "1") == "1"
# MOE_DENSE_VRAM=1 (under MOE_RESIDENT, off by default: its store shrinks and the reads it adds ate the gain): the model's own weights (not the experts) bit-copied into device memory at install, since MLX CUDA keeps loaded arrays in managed memory, which the driver moves to host RAM beside a full expert store.
_DENSE_VRAM = _RESIDENT and os.environ.get("MOE_DENSE_VRAM") == "1"
# MOE_FUSE_PROJ=1 (CUDA only, off by default: no gain measured on the resident 35B): projections that read the same row in one matvec (fuse_proj.py).
_FUSE_PROJ = BACKEND == "cuda" and os.environ.get("MOE_FUSE_PROJ") == "1"

# MOE_KV_HOST=1 (CUDA only, off by default): the QSA layers' K/V in pinned host RAM past a length, their VRAM going to the expert store (kv_host.py, imported only when on).
_KV_HOST = BACKEND == "cuda" and os.environ.get("MOE_KV_HOST") == "1"
if _KV_HOST:
    import kv_host


def _meminfo():
    with open("/proc/meminfo") as f:
        return {k: int(v.split()[0]) * 1024 for k, v in (ln.split(":", 1) for ln in f)}


def _linux_sysctl(name):
    """The sysctls this file reads from /proc/meminfo, where memorystatus_level is MemAvailable's share and the compressor term of the swap line is 0 (_vm_counts)."""
    if name == "hw.pagesize":
        return os.sysconf("SC_PAGE_SIZE")
    m = _meminfo()
    if name == "hw.memsize":
        return m["MemTotal"]
    if name == "kern.memorystatus_level":
        return int(100 * m["MemAvailable"] / m["MemTotal"])
    if name == "kern.memorystatus_vm_pressure_level":
        # Always "normal": the level, the swap line and the swap guard stand for macOS's pressure warning, with PSI the upgrade.
        return 1
    return None


def _disk_swap_pages():
    """Pages on disk swap now (zram left out: compressed RAM, not a disk)."""
    with open("/proc/swaps") as f:
        kb = sum(int(r.split()[3]) for r in f.readlines()[1:] if not r.startswith("/dev/zram"))
    return kb * 1024 // os.sysconf("SC_PAGE_SIZE")


def max_rss():
    """ru_maxrss in bytes (macOS reports bytes, Linux KB)."""
    import resource
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r * 1024 if _LINUX else r


def _libc():
    global _LIBC
    if _LIBC is None:
        # `import ctypes.util` here would bind a local name `ctypes` and shadow the module-level import for the rest of this function.
        from ctypes.util import find_library
        _LIBC = ctypes.CDLL(find_library("c"))
    return _LIBC


def _sysctl_int(name, width=4):
    """Read an integer sysctl at ~1 us a call, with `width` 8 for 64-bit ones because a 32-bit read of hw.memsize silently returned 0 and shrank the arena to its floor every time."""
    if _LINUX:
        return _linux_sysctl(name)
    val = ctypes.c_uint64(0) if width == 8 else ctypes.c_int(0)
    sz = ctypes.c_size_t(width)
    if _libc().sysctlbyname(name.encode(), ctypes.byref(val), ctypes.byref(sz),
                            None, 0) != 0:
        return None
    return val.value


_VM64 = _PAGE = None
# The swap line: macOS swaps once non-compressed memory (kern.memorystatus_level) falls under about two thirds of what the compressor holds, a 40% share of the two together.
_SWAP_SHARE = 2 / 3


def swap_margin_gb(level, comp_gb, total_gb):
    """GB the machine can still give before the swap line: available memory less _SWAP_SHARE of the compressor."""
    return total_gb * level / 100.0 - _SWAP_SHARE * comp_gb


def _vm_counts():
    """(GB the compressor occupies, swapouts in pages) from host_statistics64, the counters vm_stat prints, or on Linux (0, pages on disk swap now) a level the guards only read growth of."""
    global _VM64, _PAGE
    if _LINUX:
        return 0.0, _disk_swap_pages()
    if _VM64 is None:
        _PAGE = _sysctl_int("hw.pagesize", width=8) or 16384
        u32, u64 = ctypes.c_uint32, ctypes.c_uint64
        # struct vm_statistics64, mach/vm_statistics.h; HOST_VM_INFO64 = 4
        _VM64 = type("VM64", (ctypes.Structure,), {"_fields_": [
            ("free", u32), ("active", u32), ("inactive", u32), ("wired", u32),
            ("zero_fill", u64), ("reactivations", u64), ("pageins", u64),
            ("pageouts", u64), ("faults", u64), ("cow_faults", u64), ("lookups", u64),
            ("hits", u64), ("purges", u64), ("purgeable", u32), ("speculative", u32),
            ("decompressions", u64), ("compressions", u64), ("swapins", u64),
            ("swapouts", u64), ("comp_pages", u32), ("throttled", u32),
            ("external", u32), ("internal", u32), ("comp_stored", u64)]})
        _libc().mach_host_self.restype = ctypes.c_uint32
    s = _VM64()
    n = ctypes.c_uint32(ctypes.sizeof(_VM64) // 4)
    try:
        if _libc().host_statistics64(_libc().mach_host_self(), 4, ctypes.byref(s),
                                     ctypes.byref(n)) != 0:
            return None
    except AttributeError:          # not macOS
        return None
    return s.comp_pages * _PAGE / 1e9, s.swapouts


_RUSAGE = None
# CUDA's peak is the most this process saw when it read the footprint, not a driver counter.
_FOOT_TOP = [0]
# Set by the VRAM store governor (make_store_governor on CUDA): the footprint is this process's share of the device, other processes' memory (NVML) left to the governor.
_VRAM_OWN = False
_NVML = None            # (library, device handle) once loaded, False where NVML is missing
_OTHERS = [0]           # the last good reading of vram_others


class _NvProc(ctypes.Structure):
    # nvmlProcessInfo_t: pid, usedGpuMemory, gpuInstanceId, computeInstanceId.
    _fields_ = [("pid", ctypes.c_uint), ("used", ctypes.c_ulonglong),
                ("gi", ctypes.c_uint), ("ci", ctypes.c_uint)]


def vram_others():
    """Bytes other processes hold on GPU 0 (NVML's per-process use, compute and graphics, ~75 us), or None without NVML."""
    global _NVML
    if _NVML is None:
        _NVML = False
        try:
            nv = ctypes.CDLL("libnvidia-ml.so.1")
            h = ctypes.c_void_p()
            if nv.nvmlInit_v2() == 0 and nv.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(h)) == 0:
                _NVML = (nv, h)
        except OSError:
            pass
    if not _NVML:
        return None
    nv, h = _NVML
    # GPU 0 by NVML's index, which ignores CUDA_VISIBLE_DEVICES; a multi-GPU machine needs the device's UUID.
    used, me = {}, os.getpid()
    for fn in (nv.nvmlDeviceGetComputeRunningProcesses_v3, nv.nvmlDeviceGetGraphicsRunningProcesses_v3):
        n, buf = ctypes.c_uint(256), (_NvProc * 256)()
        if fn(h, ctypes.byref(n), buf) != 0:
            return _OTHERS[0]
        for p in buf[:n.value]:
            if p.pid != me and p.used < 1 << 63:        # ULLONG_MAX: not available
                used[p.pid] = max(used.get(p.pid, 0), p.used)
    _OTHERS[0] = sum(used.values())
    return _OTHERS[0]


def phys_footprint(peak=False):
    """This process's ri_phys_footprint in bytes, Metal buffers included, or with `peak` the kernel's lifetime maximum, and on CUDA the GPU's memory in use from cudaMemGetInfo."""
    global _RUSAGE
    if BACKEND == "cuda":
        # Other processes first and the device after, so one that allocates between the two reads is not counted as ours.
        others = vram_others() if _VRAM_OWN else 0
        d = mx.device_info()
        now = d["total_memory"] - d["free_memory"] - others
        _FOOT_TOP[0] = max(_FOOT_TOP[0], now)
        return _FOOT_TOP[0] if peak else now
    if _LINUX:
        with open("/proc/self/status") as f:
            st = dict(ln.split(":", 1) for ln in f)
        return int(st["VmHWM" if peak else "VmRSS"].split()[0]) * 1024
    if _RUSAGE is None:
        _libc().proc_pid_rusage.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_void_p)
        _RUSAGE = (ctypes.c_uint64 * 40)()          # rusage_info_v4 is 304 bytes
    if _libc().proc_pid_rusage(os.getpid(), 4, _RUSAGE) != 0:
        return 0
    # ri_phys_footprint is the 10th u64 of the rusage info and ri_lifetime_max_phys_footprint the 31st, after the uuid, the times, the wakeups and the pageins (the rusage_info_v4 layout).
    return int(_RUSAGE[30 if peak else 9])


def arena_floor(top_k):
    """Soak-proven default floor for the arena in slots: 4x top_k, which is 32 on the 35B model at top_k 8 and the smallest arena that passed a 40x400 soak with --bias-budget 0.2, while a memory budget (D3) may still lower it to the hard 2x top_k minimum ArenaGovernor._floor enforces."""
    return 4 * top_k


def budget_slots(budget_gb, held_gb, slot_gb, headroom_gb):
    """Pure function (D2) giving the slots a process-memory budget still allows, from `held_gb` (everything MLX holds outside the arena) and `headroom_gb` (room for what a point-in-time reading cannot see), both off mx.get_active_memory since ru_maxrss does not see Metal buffers at all."""
    return int(max(0.0, budget_gb - held_gb - headroom_gb) / max(slot_gb, 1e-9))


# Ceiling for prefill's own transient (chunkattn's query-chunked attention), derived at moe_stream.py's install call site like ArenaGovernor.budget_gb but without a governor, since the fixed arena path never builds one and that is where the prefill peak happens.
_PREFILL_BUDGET_GB = None
_PREFILL_HOST_GB = 0.0
# Elements one prompt position's MoE output holds (top_k rows of the hidden size), from moe_row_elems() at the install call site, which prefill_cap_bytes() turns into chunkattn's ceiling, with None leaving the raw budget room as the only cap.
_PREFILL_ROW_ELEMS = None
# The MemBudget measuring the pass that runs now (--mem-peak-gb), else None: chunkattn caps attention at that pass's own MoE rows and brackets a prefill out of the measurement.
_MEASURING = None
# Bytes one layer's prefill attention materialises per (query row x key), per query row and per key, measured at install by chunkattn.prefill_attn_slope and None where it was not, so a plan sized without them is the plan that let a 64k prompt's scores outgrow the budget.
_PF_ATTN = None
# The narrowest query chunk a layer-major pass is cut to when the keys leave no room for LM_W: 128 rows is the narrowest width whose chunked attention reproduced a stock causal call bit for bit at chunkattn's calibration shapes, the narrower kernels summing a row in another order.
ATTN_CHUNK_MIN = 128
# The calibration file's schema: 2 wrote a prefill's growth without its attention (the plan adds that back), 3 keeps the fresh passes beside all of them for the line's slope, so a file of an engine that fitted the slope off a single narrow width is not read.
CALIB_VERSION = 3


# What a prefill pass of n positions grows MLX by (fixed + rate x n plus the KV caches' copy), with the fixed part the two whole-layer read-ahead sets and each pass on an empty cache measured by note_prefill and the line refitted.
PREFILL_ROWS = 4.0
_PREFILL_FIXED = 0.0
_PREFILL_SEEN = {}      # width of a measured pass -> the most MLX grew


def set_prefill_budget_gb(gb, host_gb=0.0, row_elems=None, fixed=0.0):
    global _PREFILL_BUDGET_GB, _PREFILL_HOST_GB, _PREFILL_ROW_ELEMS, _PREFILL_FIXED
    _PREFILL_BUDGET_GB = gb
    _PREFILL_HOST_GB = host_gb
    _PREFILL_ROW_ELEMS = row_elems
    _PREFILL_FIXED = fixed


def set_prefill_attention(attn):
    """(sdpa_per_row_key, indexer_per_row_key, per_row, per_key, sdpa_rows) bytes of one layer's prefill attention, from chunkattn.prefill_attn_slope at install."""
    global _PF_ATTN
    _PF_ATTN = attn


# MOE_LM_CHUNK and MOE_PIECE force a layer-major chunk width and piece size (stream_fast.set_lm_chunk), the golden harness's exactness check that a cut piece gives the whole pass's logits.
_LM_CHUNK = int(os.environ.get("MOE_LM_CHUNK") or 0)
_PIECE = int(os.environ.get("MOE_PIECE") or 0)


def set_lm_chunk(rows=0, piece=0):
    """Force the layer-major chunk width and the piece a prefill is cut to (rows or piece 0 = planned), for the exactness check a run outside the engine makes between two passes of one cache."""
    global _LM_CHUNK, _PIECE
    _LM_CHUNK, _PIECE = int(rows), int(piece)


def attn_bytes(t, n):
    """Bytes one layer's attention holds for `t` query rows over `n` keys: the SDPA score matrix at the width chunkattn cuts it to, plus the indexer's own scores, which grow with the full chunk and alone tripped the guard at 64k; 0 where the backend fuses and nothing was measured."""
    if _KV_HOST and t > 0:
        # A homed layer's K/V staged into VRAM for the pass, where the flag-off path holds every layer's (kv_host).
        return _attn_bytes(t, n) + kv_host.scratch_bytes(n)
    return _attn_bytes(t, n)


def _attn_bytes(t, n):
    if not _PF_ATTN or t <= 0 or n <= 0:
        return 0.0
    sdpa, ix, per_row, flat, rows = _PF_ATTN
    srows = min(t, rows) if rows else t
    return srows * sdpa * n + t * (ix * n + per_row) + flat * n


def prefill_room():
    """What the prefill budget leaves MLX right now, bytes; None without one."""
    if _PREFILL_BUDGET_GB is None:
        return None
    return _PREFILL_BUDGET_GB * 1e9 - _PREFILL_HOST_GB * 1e9 - mx.get_active_memory()


def prefill_need():
    """(fixed, rate) bytes, the slope between the narrowest and the widest pass when they are 25% apart and over the sets otherwise, with the fixed part raised until the line covers every pass measured (12.3k ids grew 2.66 GB and 16.4k 3.09 GB on the 35B)."""
    prior = PREFILL_ROWS * (_PREFILL_ROW_ELEMS or 0) * 2
    if not _PREFILL_SEEN:
        return _PREFILL_FIXED, prior
    seen = _PREFILL_SEEN
    lo, hi = min(seen), max(seen)
    rate = ((seen[hi] - seen[lo]) / (hi - lo) if hi >= 1.25 * lo
            else (seen[hi] - _PREFILL_FIXED) / hi)
    rate = max(rate, prior / 2)
    return max([_PREFILL_FIXED] + [g - rate * n for n, g in seen.items()]), rate


def prefill_fit(caches):
    """Positions a prefill pass on `caches` can take within the budget now, or None without a budget or MoE rows, with attention's own transient left out since chunkattn caps that at half the same room (prefill_cap_bytes)."""
    room = prefill_room()
    if room is None or not _PREFILL_ROW_ELEMS:
        return None
    _eval_caches(caches)
    fixed, rate = prefill_need()
    return max(0, int((room - fixed - _copy(caches)) / rate))


def note_prefill(n, grew):
    """A pass of n positions on an empty cache grew MLX by `grew` bytes at its peak."""
    _PREFILL_SEEN[n] = max(_PREFILL_SEEN.get(n, 0.0), grew)


def whole_layer_bytes(model):
    """The two whole-layer sets a prefill's read-ahead holds (_layer_set): twice the widest layer's experts."""
    return 2 * max((sum(p._packed["stride"] for p in g._projs)
                    * g._projs[0]._packed["n_experts"]
                    for g in glus_of(model)
                    if all(p._packed for p in g._projs)), default=0)


def park_arenas(model):
    """Free every per-layer arena's buffers for a long prefill while keeping the LRU and the slot table, so the prefill routes as it would have and decode goes on from the same residents (_init_arena reads the same experts back into the same slots), and return the bytes freed."""
    freed, a0 = 0, mx.get_active_memory()
    if _NATIVE is not None:
        _NATIVE.release()
    for g in glus_of(model):
        ar = _ARENAS.get(g._akey)
        if ar is None or g._budget is not None:
            continue
        g._await_pending()
        freed += sum(b.nbytes for bufs, *_ in ar for b in bufs)
        del _ARENAS[g._akey], ar
    if freed:
        # A buffer the last decode's command buffer still held is released by its completion handler a moment after, so the room is read once it is (a second at most).
        mx.synchronize()
        t = time.perf_counter()
        while mx.get_active_memory() > a0 - freed and time.perf_counter() - t < 1.0:
            time.sleep(0.002)
        mx.clear_cache()
    return freed


def unpark_arenas(model):
    for g in glus_of(model):
        if PREFILL_STOP is not None and PREFILL_STOP():
            return      # a request waits: the rest come back at their first claim
        if g._akey not in _ARENAS and g._slot_np is not None and g._budget is None:
            g._init_arena()


def prefill_cap_bytes(n_keys, itemsize):
    """Room left for chunkattn's own transient now: the configured budget minus host_gb and whatever MLX already holds (skeleton, arena, KV and whatever of this pass landed), capped at the MoE output a prefill of all `n_keys` positions holds, since a score chunk never outgrows what the same layer's wide pass pays anyway (181 MB on a 5.5k-token front where the old ceiling reached 503 MB), and None leaves callers unbounded as before."""
    ceiling = (None if _PREFILL_ROW_ELEMS is None
               else n_keys * _PREFILL_ROW_ELEMS * itemsize)
    if _PREFILL_BUDGET_GB is None:
        return ceiling
    # Half the room: a chunk's freed pages stay in the footprint while the next chunk is made, so a 64k prompt needs the budget above its MLX active size.
    room = max(0.0, prefill_room()) / 2
    return room if ceiling is None else min(room, ceiling)


def moe_row_elems(model):
    """top_k x hidden, the elements one position's MoE output holds (hidden read off the packed down projection's rows), or 0 when the model has no packed experts so a dense verify target stays on one stock pass."""
    glus = glus_of(model)
    k = top_k_of(model)
    if not glus or not k:
        return 0
    hidden = max((next(p for p in g._projs[2]._packed["parts"]
                       if p["name"] in ("weight", "svh", "gguf"))["shape"][0]
                  for g in glus if g._projs[2]._packed), default=0)
    return k * hidden


class ArenaGovernor:
    """Grows and shrinks the expert arenas with the machine's free memory, since the arena is a pure cache whose slots only turn hits into reads, and a resize is rate-limited because it costs ~30 tokens of re-warming."""

    def __init__(self, glus, cap_min=6, cap_max=512,
                 period=64, reserve_pct=15, slot_gb=None, top_k=0, step=16,
                 guard_period=2, mem_pct=90, n_exp=0, host_gb=0.6, store=None):
        self.glus, self.cap_max = glus, cap_max
        # --mem-peak-max-gb (make_store_governor): what moves is a MemBudget's peak, not per-layer arenas, so `cap` counts slot_gb units of store above the budget from 0 and the arena floors do not apply.
        self.store = store
        self._base = None if store is None else store.peak
        self.period = period
        # Floor is 2*top_k, not top_k: an arena sized to exactly one token's experts evicts all of them for the next one, which thrashed 27 tok/s down to 3 under pressure.
        self._floor = max(cap_min, 2 * (top_k or 0))
        self._warned_floor = False
        # Absolute targeting (total_gb/slot_gb), not a bare level reading, since growing the arena lowers the level itself and the controller would shrink straight back.
        self.total_gb = (_sysctl_int("hw.memsize", width=8) or 0) / 1e9
        self.slot_gb = slot_gb or self._measure_slot()
        # How much of the machine to leave alone even when nothing else wants it, since taking everything invites the compressor and the swap writer and a swapped-in hit is slower than the pread it replaced.
        self.reserve_pct = reserve_pct
        # The second ceiling: the reserve above keeps the machine out of trouble and this one keeps the process (skeleton, arena, KV and host_gb) under mem_pct of it, with tick() taking the smaller.
        self.mem_pct, self.host_gb = mem_pct, host_gb
        self.budget_gb = self.total_gb * mem_pct / 100.0
        # Headroom for what a live reading cannot see: one wide prefill read spans nearly a whole layer's experts, divided across the layers because _batch_wide's transient is one layer at a time, plus host_gb for the runtime that mx.get_active_memory never counts; a model that does not state its expert count guards here once rather than in every caller.
        self.n_exp = n_exp or 0
        self.headroom_gb = (self.n_exp * self.slot_gb / len(glus) if glus else 0.0) + host_gb
        self.cap = glus[0]._cap if glus and store is None else 0
        self._paces = deque(maxlen=16)  # store mode: seconds a token, recent guard ticks
        self.calls = 0
        self.history = []
        # Memory sets the ceiling and measured speed picks the operating point under it, since targeting free memory alone grew to whatever fitted and shrank when anything else wanted memory without ever checking the move made the model faster.
        self.step = step
        self._t = None          # wall clock at the previous guard tick
        self._el = 0.0          # decode seconds accumulated in this window
        self._n = 0             # tokens behind that time
        self._toks = 0          # decode positions actually ticked (D2: this
                                 # is tick()'s own clock, in tokens rather than calls (see tick()).
        self._guard_at = 0      # self._toks at the last guard-lane firing
        self._speed_at = 0      # self._toks at the last speed-lane firing
        self._rate = {}         # cap -> best tokens/s ever measured there
        # Budget ceiling from the last ORDINARY (non-prefill) tick, since prefill's wide pass allocates a per-layer transient outside the arena (measured 0.0-0.85 GB) that would clamp the arena on a number already gone.
        self._bceil_cache = None
        self._dir = 1           # which way the last accepted step went
        self._skip = True       # a resize costs ~30 tokens of re-warming
        self._prev = None       # the cap we stepped away from
        self._park = None       # memory level we settled at, None while probing
        self._roof = cap_max    # largest size that has proved sustainable
        self._roof_avail = None # memory that was free when the roof was set
        self._calm = 0          # consecutive quiet windows since the roof was set
        self._settled = False   # a size above this one measured worse
        # Two lanes, because the two jobs have opposite deadlines: giving memory back has to be immediate (two sysctl reads at ~1 us against a ~38 ms token) while finding the fastest arena is the opposite, each probe costing ~30 tokens of re-warming, so the guard lane runs every `guard_period` tokens and may only SHRINK and the speed lane runs every `period` tokens and is the only thing allowed to grow.
        self.guard_period = guard_period
        # Swap safety: kern.memorystatus_level alone overstated the room by several GB because it counts non-compressed memory only and macOS swaps once that falls under two thirds of what the compressor occupies, which _margin() takes off, while how far this process itself drags the machine toward the line is measured from its own quiet windows against the lowest reading inside each prefill and that is the reserve.
        self._swap0 = None      # swapouts (pages) at the previous guard/idle tick
        self._m_quiet = None    # margin at the last quiet tick (decode or idle)
        self._m_low = None      # lowest margin seen in the prefill since then
        self._swings = deque(maxlen=8)  # own falls of the margin, last 8 prefills
        self._why = {}          # the last guard tick's ceiling, for the history

    @property
    def cap_min(self):
        """Smallest arena that is still a cache rather than dead weight, taken from what the layers have actually demanded (the decode path uses the arena only while a token's distinct experts fit in it, so anything under that sends every layer to the wide read while still holding the slots), or for a store its --mem-peak-gb with cap 0."""
        if self.store is not None:
            return 0
        seen = max((getattr(g, "_want", 0) for g in self.glus), default=0)
        return max(self._floor, seen)

    def _measure_slot(self):
        """GB held by one resident expert across every layer."""
        total = 0
        for g in self.glus:
            for p in g._projs:
                if p._packed is not None:
                    total += p._packed["stride"]
        return total / 1e9 or 0.05

    def _margin(self, level, comp_gb):
        """GB the machine can still give before it starts swapping (swap_margin_gb)."""
        return swap_margin_gb(level, comp_gb, self.total_gb)

    def _avail(self, level, comp_gb=0.0):
        """The margin (_margin) if we handed the whole arena back, the quantity to reason with because it does not move when WE grow while the level itself does."""
        return self._margin(level, comp_gb) + self.cap * self.slot_gb

    def _reserve_gb(self, floor=False):
        """Room to keep above the swap line: the furthest this process's own last prefills pulled the margin under its quiet reading (up to 4.5 GB on a short prompt at 60 slots, ~0.1 on a long one whose prefill parks the arena), and the --reserve-pct share of the machine until one has been measured, which a store keeps as a floor since its prefill releases the pool and its own dip reads ~0."""
        prior = self.total_gb * self.reserve_pct / 100.0
        if self.store is not None:
            return prior if floor else max([prior, *self._swings])
        if self._swings:
            return max(self._swings)
        return prior

    def _ceiling(self, avail_gb, floor=False):
        """Most slots the machine can hold without us becoming the problem."""
        return int(max(0.0, avail_gb - self._reserve_gb(floor)) / max(self.slot_gb, 1e-9))

    def _ref(self, level, comp):
        """What a prefill's dip is measured on, where _HOLD, a store, keeps the margin with its own steps handed back (_avail)."""
        if _HOLD and self.store is not None:
            return self._avail(level, comp)
        return self._margin(level, comp)

    def _peek(self):
        """level, pressure, compressor GB and the swapout counter (None without one), read now."""
        level = _sysctl_int("kern.memorystatus_level")
        press = _sysctl_int("kern.memorystatus_vm_pressure_level")
        vm = _vm_counts()
        return level, press, (0.0 if vm is None else vm[0]), (None if vm is None else vm[1])

    def _read(self):
        """level, pressure, compressor GB, and pages swapped out since the last read (guard or idle tick)."""
        level, press, comp, sw = self._peek()
        swapped = 0
        if sw is not None:
            swapped = sw - self._swap0 if self._swap0 is not None else 0
            self._swap0 = sw
        return level, press, comp, swapped

    def _quiet(self, level, comp):
        """A quiet reading, decode or idle, closes the prefill before it, since a store's readings count from the model's first pass (before it the store is not built and the level has not taken in the skeleton, so the margin held 11.9 GB for 1.5 s after a 3.2 GB load and then fell ~5 GB in the first pass, and an idle tick there made a 5.2-5.8 GB dip that held the store back for 8 prefills in 5 of 10 runs)."""
        if self.store is not None and not self.calls:
            return
        m = self._ref(level, comp)
        if self._m_low is not None:
            self._swings.append(max(0.0, self._m_quiet - self._m_low))
            self._m_low = None
        self._m_quiet = m

    def _held_gb(self):
        """Everything MLX holds outside the arena (D2): active_memory() minus the arena's own cap*slot_gb share once the arena actually exists (_init_arena is lazy, on a layer's first decode), since subtracting before that went negative and inflated budget_slots."""
        active_gb = mx.get_active_memory() / 1e9
        if self.glus and not all(g._slot_of is not None for g in self.glus):
            return active_gb
        return active_gb - self.cap * self.slot_gb

    def _budget_ceiling(self, do_sort=False):
        """Most slots the process memory budget (--mem-pct) still allows, recomputed live, and on a prefill tick from the value cached at the last ordinary decode tick since the wide read's transient is not growth."""
        if do_sort and self._bceil_cache is not None:
            return self._bceil_cache
        val = budget_slots(self.budget_gb, self._held_gb(), self.slot_gb,
                           self.headroom_gb)
        self._bceil_cache = val
        return val

    def _damp(self, ceiling):
        """A growth toward `ceiling`: half of it while no speed is on record, then half the way."""
        return max(self.cap, ceiling // 2 if not self._rate else self.cap + (ceiling - self.cap) // 2)

    def _halved(self, avail_gb):
        """idle_tick's answer to a pressure warning: half the slots."""
        return max(self.cap_min, self.cap // 2)

    def _roof_lifts(self, avail_gb):
        """A roof set under pressure lifts only when the machine genuinely has more to give, 15% more than when we hit it, which our own shrinking cannot produce, with abs() since the margin under the compressor can be negative."""
        return self._roof_avail is not None and avail_gb > abs(self._roof_avail) * 1.15

    def _set_roof(self, cap, avail_gb):
        """Remember that we were too big and at what available memory, so the controller does not grow straight back into the same wall."""
        self._roof = max(self.cap_min, min(self._roof, cap))
        self._roof_avail = avail_gb
        self._park, self._calm = None, 0

    def tick(self, do_sort=False, T=1):
        # `do_sort` marks a tick taken during prefill's wide batched pass (see _budget_ceiling), always False where a pass is a single decode step.
        # One tick per MoE layer per token, with T positions banked once per pass on its last layer (`self.calls % span == 0`), since counting calls instead of T measured 70-100 passes for 200 output tokens under --draft k=2 and left the arena at its 21-slot start against a 48-slot budget (27 tok/s where a pinned 31-slot arena gave 60.7), and a prefill pass (do_sort) is not banked at all because one call can cover a whole prompt.
        self.calls += 1
        if not self.glus:
            return
        if do_sort and self._m_quiet is not None:
            # Inside a prefill, every layer: its lowest margin (see _quiet), two reads of a few us against a ~0.1 s layer.
            level, _, comp, _ = self._peek()
            if level is not None:
                m = self._ref(level, comp)
                self._m_low = m if self._m_low is None else min(self._m_low, m)
        span = len(self.glus)
        if self.calls % span:
            return
        if not do_sort:
            self._toks += T
        if self._toks - self._guard_at < self.guard_period:
            return
        gap, self._guard_at = self._toks - self._guard_at, self._toks
        level, press, comp, swapped = self._read()
        if level is None:
            return
        if not do_sort:
            self._quiet(level, comp)
        avail_gb = self._avail(level, comp)
        if self._roof_lifts(avail_gb):
            self._roof, self._roof_avail = self.cap_max, None
            self._settled = False
        ceiling = self._ceiling(avail_gb)
        self._why = {"ceil": ceiling, "rates": len(self._rate), "roof": self._roof}
        # Free memory counts the page cache the arena cannot actually take, so jumping straight to the ceiling hits pressure and bounces back to about half, and a half-step until a roof is learned keeps that clamp from taxing the 80B 2x (parked at 61/24 tok/s against the 112/34.5 the machine could hold).
        # Growth only: a ceiling under the current size is a real squeeze and binds as it is, since the clamp applied to it lifted it back to the current size once any rate was on record and halved it when none was (60 -> 25 slots, ceiling 51).
        if self._roof >= self.cap_max and ceiling > self.cap:
            ceiling = self._damp(ceiling)
        if _HOLD and self.store is not None:
            # A store grows only into the room its last prefills left (the reserve) and gives back only what the machine lacks now (the floor), so a reserve read off its own prefills never shrinks it between requests.
            ceiling = max(ceiling, min(self.cap, self._ceiling(avail_gb, floor=True)))

        # Guard-lane clock only: a window spans turn boundaries, and counting prefill or user think-time as decode walked the arena backwards (53 slots read 19% slower than 37).
        now = time.perf_counter()
        if self._t is not None:
            dt = now - self._t
            if self.store is None:
                ok = dt < 0.25 * gap
            else:
                # A 0.25 s bound taken on the 35B would drop Flash-Next's 0.1-0.3 s tokens, so the store's lane drops a gap only when it is 4x the recent median.
                ok = bool(self._paces) and dt / gap < 4 * float(np.median(self._paces))
                self._paces.append(dt / gap)
            if ok:
                # A gap this long is not decoding, since 250 ms a token would be four times worse than the worst arena, so the sample is dropped rather than the window, with `gap` the tokens actually elapsed since the last guard tick rather than a fixed guard_period.
                self._el += dt
                self._n += gap
        self._t = now

        # The speed lane is a multiple of the guard lane in tokens rather than calls, so it fires once `period` tokens have elapsed rather than on an exact multiple, the same slack the guard lane accepts.
        slow = self._toks - self._speed_at >= self.period
        rate = None
        if slow:
            self._speed_at = self._toks
            # Half a window of usable samples or the number means nothing.
            if self._n >= self.period // 2:
                rate = self._n / max(self._el, 1e-9)
            self._el, self._n = 0.0, 0

        # A roof must not be permanent: on a machine busy for reasons nothing to do with us we shrink, the pressure never lifts because it was never ours and avail_gb never rises, which locked a 16 GB machine at the floor for a whole session at 7.4 tok/s, so quiet ticks release it as freed memory does.
        if press and press > 1 or swapped:
            self._calm = 0
        elif slow and self._roof < self.cap_max and not self._settled:
            # One step per quiet window and never straight back to the top, since releasing the roof outright threw the arena back into the wall it had just left (21->171->85->164->158->121->103, six resizes, 19-25 tok/s).
            self._calm += 1
            if self._calm >= 4:
                self._roof = min(self.cap_max, self._roof + self.step)
                self._calm = 0

        if press and press >= 4:
            # Critical: macOS is about to start killing processes, so drop to the floor in one step rather than easing down over several ticks.
            want = self.cap_min
            self._set_roof(want, avail_gb)
        elif press and press > 1:
            # Warning: stop being the problem now, argue later.
            want = min(self.cap, max(self.cap_min, self.cap // 2))
            self._set_roof(want, avail_gb)
            if want == self.cap_min and not self._warned_floor:
                # Pinned at the floor by pressure this process did not create, since the arena is already as small as it goes (16 slots at 12 tok/s, once read as unexplained in the status line), so say why once.
                self._warned_floor = True
                print(f"  arena pinned at its floor ({self.cap_min} slots) by system "
                      f"memory pressure (level {press}, {avail_gb:.1f} GB available): "
                      f"other processes hold the memory. Close them, or pin the arena "
                      f"with --no-adaptive --resident N", flush=True)
        elif swapped:
            # Reflex: the machine is writing to swap right now, whatever the signals above said, so one step down and a roof there.
            want = max(self.cap_min, self.cap - self.step)
            self._set_roof(want, avail_gb)
        elif not slow or self._skip or rate is None:
            # Guard lane, or the first window at a new size: nothing here knows anything about speed, the only thing that can move the arena is the ceiling clamp below and that can only shrink it, since growing needs a measurement this lane has none of.
            if slow:
                self._skip = False
            want = self.cap
        else:
            # Bigger is monotonically faster here (21->64->96 slots measured 21.6->25.9->27.0 tok/s), so jump straight to the roof rather than stepping, since every resize costs ~30 tokens of re-warming and stepping paid that repeatedly for no benefit.
            self._rate[self.cap] = max(self._rate.get(self.cap, 0.0), rate)
            want = self._roof
            # A jump that measures >10% below the best smaller size on record (5x the window noise) means this size crowds the page cache the streaming path reads through, so fall back to the best MEASURED size rather than halfway to it (halving landed on 96 at 24.1 tok/s, worse than sizes on record) and stop probing upward.
            # Not for a store, which crowds no page cache it reads through (6 < 7 < 8 GB on both workloads), while its windows span requests whose speeds differ by more than the 10% (Flash-Next prose ~35% faster than code at any size), so a code window at 4 steps against a prose one at 1 settled the store at 1.
            small = [c for c in self._rate if c < self.cap]
            best_below = max((self._rate[c] for c in small), default=0.0)
            if small and rate < best_below * 0.90 and self.store is None:
                good = max(small, key=lambda c: self._rate[c])
                self._roof = max(self.cap_min, good)
                want = self._roof
                self._settled = True
            want = max(self.cap_min, min(self.cap_max, self._roof, want))

        # D1: the process memory budget is a second, independent ceiling (min(machine, budget)) and it does not get the page-cache damping above, since free memory counts the page cache as available when the arena cannot have it while the budget ceiling is a direct read of what the process itself already holds.
        clamped = min(want, max(self.cap_min, ceiling),
                     max(self.cap_min, self._budget_ceiling(do_sort)))
        squeezed = clamped < want or (clamped < self.cap and not slow)
        want = clamped
        # resize() empties the arena (_lru.clear(), _ARENAS.pop()), so a move costs ~30 tokens of re-warming whatever its size, and the guard lane's free-memory ceiling jitters by a slot or two constantly, so without a filter the arena is wiped every couple of tokens and never warms at all (5.2 tok/s where the same arena held still gives 27).
        # Ignore anything not worth 250 MB, unless the machine is actually under pressure, where giving memory back beats keeping a cache warm.
        quiet = (not press or press <= 1) and not swapped
        # A store's step to its max is its last one rather than jitter, since filtered a store that went half the room first landed one 147 MB step short (12 of 13 on Flash-Next) and never took it.
        last = self.store is not None and want == self.cap_max
        if quiet and abs(want - self.cap) * self.slot_gb < 0.25 and not last:
            want = self.cap
        if squeezed and want != self.cap:
            # Somebody else took memory, so whatever optimum we had was measured on a machine that no longer exists, and the rates above the ceiling are forgotten and re-probed rather than sending us straight back up as soon as it lifts.
            # Only on a real move, since a dip the jitter filter just ignored used to wipe the table too and the next tick, finding it empty, read the arena as unmeasured and halved its ceiling (60 -> 23 slots with the machine at 5 GB of margin).
            self._park = None
            self._rate = {c: r for c, r in self._rate.items() if c <= clamped}
        self._resize(want, level, press, rate, swapped)

    def _resize(self, want, level, press, rate=None, swapped=0):
        if want == self.cap:
            return
        if self.store is not None:
            # Only the peak moves here, often mid-pass, and the store is rebuilt at the stack's next pass boundary (MemBudget.tick) where its measurements are taken.
            self.store.peak = self._base + want * self.slot_gb * 1e9
            self.store.resized = True
        else:
            for g in self.glus:
                g.resize(want)
            mx.clear_cache()
        self.history.append({"tok": self._toks, "t": round(time.perf_counter(), 2),
                             "level": level, "press": press, "swapped": swapped,
                             "margin": None if self._m_quiet is None else round(self._m_quiet, 2),
                             "reserve": round(self._reserve_gb(), 2),
                             "rate": None if rate is None else round(rate, 2),
                             "cap": self.cap, "new_cap": want, **self._why})
        if len(self.history) > 64:
            del self.history[:-64]
        self._prev, self.cap, self._skip = self.cap, want, True

    def idle_tick(self):
        """Idle tick, on the model's thread while it is not decoding: the guard lane's shrink rules and swap reflex without growth, and a reading that is a quiet window for the reserve."""
        level, press, comp, swapped = self._read()
        if level is None or not self.glus:
            return
        if not (_HOLD and self.store is not None):
            # _HOLD: not a store's, since macOS lets go of an idle process's GPU wiring 1-2 s after a reply and the next prefill wiring it back read as a 4.6-7.2 GB dip (store_hold).
            self._quiet(level, comp)
        avail_gb = self._avail(level, comp)
        if press and press >= 4:
            want = self.cap_min
        elif press and press > 1:
            want = self._halved(avail_gb)
        else:
            want = max(self.cap_min, min(self.cap, self._ceiling(
                avail_gb, floor=_HOLD and self.store is not None)))
            if swapped:
                want = max(self.cap_min, min(want, self.cap - self.step))
            elif (self.cap - want) * self.slot_gb < 0.25:     # tick()'s jitter filter
                want = self.cap
        if want < self.cap:
            self._set_roof(want, avail_gb)
            self._resize(want, level, press, swapped=swapped)
            if self.store is not None:
                # No pass is coming to apply it: give the bytes back now.
                self.store.fit(rebuild=False)
                self.store.resized = False

    def grow_now(self):
        """_GROW, a store's first decode pass after a prefill: the speed lane's growth (roof, first-step damping, margin over the reserve, budget, jitter filter) taken now, since the prefill left the pool empty and a bigger one costs no re-warming; True when the peak moved."""
        if self.store is None or self.cap >= min(self._roof, self.cap_max):
            return False
        level, press, comp, sw = self._peek()
        # Read without _read(), so swapouts since the last guard tick stay that tick's to act on and only stop growth here.
        if level is None or (press and press > 1) or (
                sw is not None and self._swap0 is not None and sw > self._swap0):
            return False
        reserve = self._reserve_gb()
        if self._m_quiet is not None and self._m_low is not None:
            # This prefill's own dip, which the next guard tick folds into the reserve.
            reserve = max(reserve, self._m_quiet - self._m_low)
        avail = self._avail(level, comp)
        ceiling = int(max(0.0, avail - reserve) / max(self.slot_gb, 1e-9))
        if self._roof >= self.cap_max and ceiling > self.cap:
            ceiling = self._damp(ceiling)
        want = min(self._roof, self.cap_max, ceiling, self._budget_ceiling())
        if want <= self.cap or ((want - self.cap) * self.slot_gb < 0.25 and want != self.cap_max):
            return False
        self._why = {"ceil": ceiling, "rates": len(self._rate), "roof": self._roof, "early": True}
        self._resize(want, level, press)
        return True


class VramGovernor(ArenaGovernor):
    """CUDA's store governor (--mem-peak-max-gb): ArenaGovernor's store lanes on the device's memory, the room being what other processes (NVML) leave above the store's base and the pressure their memory eating into half the reserve."""

    def __init__(self, budget, slot_gb, cap_max, max_gb):
        super().__init__(budget.all, 0, cap_max, slot_gb=slot_gb, store=budget)
        self.total_gb = mx.device_info()["total_memory"] / 1e9
        # The reserve's prior: what the launcher's fixed plan (max_gb) leaves free on the device, so the store never takes more than that plan, raised by the prefills' measured dips (_quiet).
        self._prior = max(0.0, self.total_gb - max_gb)
        self._others = 0.0

    def _peek(self):
        o = vram_others()
        if o is not None:
            self._others = o / 1e9
        free = mx.device_info()["free_memory"] / 1e9
        planned = self._base / 1e9 + self.cap * self.slot_gb
        # Warning, as macOS's: the others and the peak planned here leave under half the reserve for whoever allocates next.
        press = 2 if self._others + planned > self.total_gb - self._prior / 2 else 1
        return 100.0 * free / self.total_gb, press, 0.0, None

    def _avail(self, level, comp=0.0):
        """Room above the store's base that the other processes leave on the device."""
        return self.total_gb - self._others - self._base / 1e9

    def _reserve_gb(self, floor=False):
        return self._prior if floor else max([self._prior, *self._swings])

    def _damp(self, ceiling):
        # Half the way, but never a step the jitter filter would drop, so it ends at the room itself.
        gap = ceiling - self.cap
        step = (gap + 1) // 2
        return self.cap + (gap if step * self.slot_gb < 0.25 else step)

    def _halved(self, avail_gb):
        # Half, or less where the room the others leave says so, as the guard lane's clamp does, since an idle server reads once a second.
        return max(self.cap_min, min(self.cap // 2, self._ceiling(avail_gb, floor=True)))

    def _roof_lifts(self, avail_gb):
        # The room is read exactly here (no page cache or compressor in it), so other processes giving back more than the jitter lifts it.
        lifts = self._roof_avail is not None and avail_gb >= self._roof_avail + 0.25
        if lifts:
            # And the dips their taking left in the reserve go with it: p_on2 kept a 4.0 GB swing and regrew 11.45 -> 12.40 GB of 15.6.
            self._swings.clear()
        return lifts

    def _budget_ceiling(self, do_sort=False):
        return self.cap_max         # the device is the budget here, MemBudget's peak

    def _resize(self, want, level, press, rate=None, swapped=0):
        old = self.cap
        super()._resize(want, level, press, rate, swapped)
        if self.cap != old:
            print(f"  elastic store (VRAM): peak {(self._base / 1e9 + old * self.slot_gb):.2f} -> "
                  f"{self.store.peak / 1e9:.2f} GB, other processes {self._others:.2f} GB, device free "
                  f"{self.total_gb * level / 100:.2f} GB{', pressure' if press and press > 1 else ''} "
                  f"(t {time.time():.2f})", flush=True)

    def idle_tick(self):
        super().idle_tick()
        if _NATIVE is not None and _NATIVE.tier_gov is not None:
            _NATIVE.tier_tick(idle=True)


def _rss_anon():
    with open("/proc/self/status") as f:
        return next(int(ln.split()[1]) * 1024 for ln in f if ln.startswith("RssAnon:"))


class TierGovernor:
    """CUDA's pinned victim tier under the store governor: its install rule (moe_pipe_cuda.tier_slots, a third of MemAvailable above its floor) read live on what other programs leave (this process's own growth since the tier was made added back) as if the tier were handed back, and never past the floor itself; quick to shrink (any guard or idle reading past the 0.25 GB jitter) and slow to grow (half the way to the smallest reading of a whole `period`-token decode window), halved when disk swap grows: the store governor's rules on host RAM."""

    def __init__(self, n, n_max, slot, floor, period=64, guard=2, read=None, own0=0, div=3):
        self.n, self.n_max, self.slot, self.floor = n, n_max, slot, floor
        # The install rule takes 1 / div of what is above the floor (MOE_RESIDENT: all of it, above RESIDENT_FLOOR).
        self.div = div
        self.period, self.guard = period, guard
        # (MemAvailable, this process's anonymous growth since own0, pages on disk swap).
        self.read = read or (lambda: (_meminfo()["MemAvailable"], max(0, _rss_anon() - own0), _disk_swap_pages()))
        self._toks = self._at = self._grow_at = 0
        self._low = None        # the smallest target read since the growth window began
        self._swap0 = None
        self.avail = None       # MemAvailable at the last reading

    def target(self, avail, mine=0):
        """Slots the install rule gives with `avail` bytes of MemAvailable, `mine` of this process's growth and this tier's own bytes back, and at most what keeps MemAvailable over the floor."""
        back = avail + self.n * self.slot - self.floor
        return max(0, min(self.n_max, int((back + mine) / self.div // self.slot), int(back // self.slot)))

    def tick(self, T=1, idle=False):
        """A token boundary of T positions, or an idle second: the size the tier moves to now, else None."""
        if not idle:
            self._toks += T
            if self._toks - self._at < self.guard:
                return None
            self._at = self._toks
        self.avail, mine, swap = self.read()
        top = self.target(self.avail, mine)
        swapped = self._swap0 is not None and swap > self._swap0
        self._swap0 = swap
        self._low = top if self._low is None else min(self._low, top)
        want = self.n
        if swapped or (top < self.n and ((self.n - top) * self.slot >= 0.25e9 or top == 0)):
            want = min(top, self.n // 2) if swapped else top
            self._low, self._grow_at = None, self._toks
        elif not idle and self._toks - self._grow_at >= self.period:
            low, self._low, self._grow_at = self._low, None, self._toks
            if low > self.n:
                want = self.n + (low - self.n + 1) // 2
                if (want - self.n) * self.slot < 0.25e9:
                    # Half the way is jitter, so the whole way or nothing, the step to the top never being jitter.
                    want = low if (low - self.n) * self.slot >= 0.25e9 or low == self.n_max else self.n
        return None if want == self.n else want


_GOV = None


def make_governor(glus, n_exp, top_k, mem_pct, reserve_pct, host_gb,
                  cap_min=None, cap_max=None):
    """The one ArenaGovernor constructor for chat.py, moe_stream.py, server.py and this file's CLI, so --adaptive picks the same floor and ceiling everywhere."""
    if not glus or not top_k:
        note("  adaptive arena: no streaming layers or unknown top_k, "
              "staying at the fixed size", flush=True)
        return None
    cap_max = cap_max or n_exp or 512
    floor = arena_floor(top_k) if cap_min is None else cap_min
    gov = ArenaGovernor(glus, floor, cap_max, reserve_pct=reserve_pct,
                        top_k=top_k, mem_pct=mem_pct, n_exp=n_exp,
                        host_gb=host_gb)
    bslots = gov._budget_ceiling()
    if bslots < floor:
        hard_min = 2 * top_k
        gov._floor = max(hard_min, bslots)
        print(f"  memory budget: {mem_pct}% of {gov.total_gb:.1f} GB allows "
              f"~{bslots} slots, below the floor of {floor} "
              f"({'this deployment' if cap_min is not None else 'arena_floor, the 35B soak'}); "
              f"floor lowered to {gov._floor}; long-dialogue quality is "
              f"unverified at this size on this model", flush=True)

    # Start at the ceiling the budget and the machine already allow rather than the floor, since the arena is built lazily on each layer's first decode (_init_arena, sized off the glu's own _cap) and raising the starting _cap costs nothing, unlike growing later, which resize()s a live arena and pays ~30 tokens of re-warming, while a floor start grew one 64-token window at a time and spent a 300-token reply at 21-40 slots (22.2 tok/s against 72-80 pinned at 60).
    level, _, comp, _ = gov._read()
    bound = "unchanged, no memory signal"
    if level is not None:
        # avail read straight from the sysctl rather than through _avail()'s "+ self.cap*slot_gb" add-back, which assumes the arena is resident and could be handed back, false before the first decode has run, as _held_gb()'s own guard already allows for.
        # The compressor comes off as in _avail(), since the level alone overstated the room by several GB.
        sys_ceiling = gov._ceiling(gov._margin(level, comp))
        raw = min(cap_max, bslots, sys_ceiling)
        start_cap = max(gov._floor, raw)
        bound = ("floor" if start_cap > raw else
                 "budget ceiling" if raw == bslots else
                 "machine reserve" if raw == sys_ceiling else "ceiling")
        if start_cap != gov.cap:
            # resize(), not a direct `g._cap = start_cap`: a direct assignment leaves _ARENAS[akey]'s iov list sized to whatever cap was live when it was built, so the next _claim() sees the key already there, skips rebuilding and indexes iov past it (IndexError, reproduced 2026-09-11 at --mem-pct 90), while resize() is a no-op beyond the attribute set when nothing was built yet and costs the same as before.
            for g in glus:
                g.resize(start_cap)
            gov.cap = start_cap
            # Seed the rate table so tick()'s half-the-ceiling caution does not read this fresh, budget-justified start as an untested guess and clamp it back on the very first guard tick, while a real shrink still re-filters _rate and drops this seed when it no longer fits.
            gov._rate[start_cap] = 0.0
    note(f"  memory budget {mem_pct}% = {gov.budget_gb:.2f} GB | mlx holds "
          f"{gov._held_gb():.2f} GB | headroom {gov.headroom_gb:.2f} GB -> "
          f"budget allows {bslots} slots | arena floor {gov.cap_min} slots "
          f"(top_k {top_k}), ceiling {cap_max}, starting at {gov.cap} "
          f"({bound}) | machine: {gov.total_gb * (level or 0) / 100:.1f} GB available "
          f"less {comp:.1f} GB compressor, reserve {gov._reserve_gb():.1f} GB "
          f"({reserve_pct}% until a prefill is measured)", flush=True)
    global _GOV
    _GOV = gov
    return gov


def make_store_governor(budget, max_gb, reserve_pct=15):
    """--mem-peak-max-gb: the budget's store grows toward `max_gb` while the swap-line margin allows and gives the bytes back under pressure, quick to shrink and slow to grow, steering MemBudget through ArenaGovernor's own lanes and margin; on CUDA the margin is the device's (VramGovernor) and the victim tier follows host RAM (TierGovernor)."""
    global _GOV
    slot_gb = sum(p._packed["stride"] for g in budget.all for p in g._projs) / 1e9
    # The 1e-9: a max exactly N steps up read as N - 1 when the float division landed a hair under N (test_store_resize, 2 runs in 11).
    cap_max = max(0, int((max_gb * 1e9 - budget.peak) / 1e9 / slot_gb + 1e-9))
    if BACKEND == "cuda":
        return _make_vram_governor(budget, max_gb, slot_gb, cap_max)
    gov = ArenaGovernor(budget.all, 0, cap_max, reserve_pct=reserve_pct,
                        slot_gb=slot_gb, store=budget)
    _GOV = gov
    note(f"  elastic store: {budget.peak / 1e9:.2f} -> up to {max_gb:.2f} GB, "
          f"{cap_max} steps of {slot_gb * 1e3:.0f} MB (one expert a layer), "
          f"grown while the swap-line margin stays over the reserve "
          f"({gov._reserve_gb():.1f} GB until a prefill is measured)", flush=True)
    return gov


def _make_vram_governor(budget, max_gb, slot_gb, cap_max):
    """make_store_governor on CUDA: a VramGovernor started at the room the device has now (with nothing else on it, the launcher's fixed plan) and the native agent's tier put on a TierGovernor."""
    global _GOV, _VRAM_OWN
    if vram_others() is None:
        # Without NVML nothing tells us when to give back, so the store keeps the fixed plan (the ceiling), as before the governor.
        budget.peak = max(budget.peak, max_gb * 1e9)
        budget.resized = bool(budget.stores)
        print("  elastic store: NVML (libnvidia-ml.so.1) not found, so other processes' VRAM is unknown; "
              f"the store stays fixed at --mem-peak-max-gb {max_gb:g}", flush=True)
        return None
    gov = VramGovernor(budget, slot_gb, cap_max, max_gb)
    _VRAM_OWN = True
    budget._host_top = 0            # its host term is this process's own from here on
    level, _, comp, _ = gov._peek()
    start = max(0, min(cap_max, gov._ceiling(gov._avail(level, comp), floor=True)))
    gov.cap, gov._rate[start] = start, 0.0
    budget.peak = gov._base + start * slot_gb * 1e9
    budget.resized = bool(budget.stores)
    _GOV = gov
    tier = ""
    nat = _NATIVE
    if nat is not None and nat.cuda and nat.tier_n:
        import moe_pipe_cuda as nv
        nat.drain()
        mx.synchronize()
        # Chunks of ~256 MB, so a shrink frees the chunks above the new size and keeps the experts below it.
        n = nat.agent.set_tier(nat.tier_slot, nat.tier_n, max(1, int(256e6 // nat.tier_slot)))
        nat.tier_gov = TierGovernor(n, nat.tier_n, nat.tier_slot, nv.TIER_FLOOR, own0=nat.tier_rss0)
        tier = (f"; victim tier {n * nat.tier_slot / 1e9:.2f} GB, given back as other programs take RAM "
                f"(a third of what they take) and before MemAvailable falls under {nv.TIER_FLOOR / 1e9:g} GB")
    note(f"  elastic store (VRAM): {gov._base / 1e9:.2f} -> up to {max_gb:.2f} GB, starting at "
         f"{budget.peak / 1e9:.2f} GB, {cap_max} steps of {slot_gb * 1e3:.0f} MB; device {gov.total_gb:.2f} GB, "
         f"other processes {gov._others:.2f} GB, reserve {gov._reserve_gb():.2f} GB{tier}", flush=True)
    return gov


def set_elastic_limits(store_max_gb=None, tier_max_gb=None):
    """The one place CUDA's elastic store and tier ceilings move (a step that fills them at load, say): the store's peak and the pinned tier's GB, which the governors still give back under pressure and grow toward when the room is there."""
    gov, nat = _GOV, _NATIVE
    if store_max_gb is not None and isinstance(gov, VramGovernor):
        cap = max(0, int((store_max_gb * 1e9 - gov._base) / 1e9 / gov.slot_gb))
        # A roof set under pressure stays; one at the old ceiling follows the new one.
        gov._roof = cap if gov._roof >= gov.cap_max else min(gov._roof, cap)
        gov.cap_max = cap
    if tier_max_gb is not None and nat is not None and nat.tier_gov is not None:
        nat.tier_gov.n_max = int(tier_max_gb * 1e9 // nat.tier_slot)


# ---------------------------------------------------------------------------
# Memory-budget mode (--mem-peak-gb): every per-layer arena costs a slot in each of the 40 layers at once (62.9 MB a slot on Qwen3.6) and even --resident 0 keeps 8 rows a layer in _dest (503 MB no setting gives back), so the layers of one packed shape share ONE tensor per projection part instead, a pool cut into per-layer quotas and a small ring for what a quota cannot take, and nothing of it runs unless enable_budget() was called.

# MOE_STORE_KEEP (on by default): the decode store survives a turn, since a store that fits beside a pass's need is not rebuilt for it, a prefill's release is undone for the quota members whose rows the prefill did not write, and on CUDA a native route keeps its kernel on an emptied LRU.
_KEEP = os.environ.get("MOE_STORE_KEEP", "1") != "0"
# MOE_STORE_SEED=1 (an arm, off): MemBudget._seed after every prefill.
# MOE_STORE_HOLD (on by default): the store governor keeps the store's size across requests, since an idle reading (macOS un-wires an idle server) made the next turn's pass a 7.7 GB dip and shrank the store 19 -> 0 steps at every turn.
_SEED = os.environ.get("MOE_STORE_SEED", "0") == "1"
_HOLD = os.environ.get("MOE_STORE_HOLD", "1") != "0"
# MOE_PF_R1 and MOE_PF_R3 (on by default): R1 cuts a prefill under a store governor at the peak the governor would grow the store to now (MemBudget._prefill_peak) rather than at the store's own, and R3 keeps a cut pass's earlier pieces' evaluated outputs instead of the last layer's hidden state behind them (enable_budget's stack_call).
_R1 = os.environ.get("MOE_PF_R1", "1") != "0"
_R3 = os.environ.get("MOE_PF_R3", "1") != "0"
# MOE_STORE_GROW (on by default; =0 turns it off): the store governor grows the store at the first decode pass after a prefill (ArenaGovernor.grow_now), when the pool is empty anyway, instead of at its speed lane's second 64-token window.
_GROW = os.environ.get("MOE_STORE_GROW", "1") != "0"
# MOE_POOL_TRIM (CUDA, on by default): the budget trims MLX's device pool before a reading would raise its host term and budgets a prefill on the device's growth, since the pool's freed reserve reads as used memory and grows past MLX's bytes in a pass.
_POOL_TRIM = BACKEND == "cuda" and os.environ.get("MOE_POOL_TRIM", "1") != "0"
# MOE_KEEP_FIRST (on by default): under the budget a server's first turn is kept for reuse when the last turn of this setup ended with room (MemBudget.room_ok).
_KEEP_FIRST = os.environ.get("MOE_KEEP_FIRST", "1") != "0"


def _shape_key(glu):
    """Layers whose experts fit one tensor: gather_qmm infers the in-dim from w.shape[-1] * 32 / bits, so bits, group size and every part's shape must agree."""
    return tuple((p.bits, p.group_size, p.mode,
                  tuple((part["dtype"], tuple(part["shape"]))
                        for part in p._packed["parts"]))
                 for p in glu._projs)


class ExpertStore:
    """The experts of every layer of one shape in shared tensors, one per projection part as [slots, *part], with slots [0, pool) cut into the layers' quotas and the ring above, gathered over the whole tensor by global slot number since gather_qmm depends on neither E nor which slot holds an expert."""

    def __init__(self, glus, pool, halves, per_half):
        self.glus, self.pool = glus, pool
        self.halves, self.per_half = halves, per_half
        self.slots = pool + halves * per_half
        self.slot_bytes = sum(p._packed["stride"] for p in glus[0]._projs)
        # MOE_STAGING: the native agent's staging rows sit past the logical rows, so nothing in Python addresses them.
        self.nbytes = (self.slots + _STAGING_ROWS) * self.slot_bytes
        mx.clear_cache()        # a store being replaced has just been dropped
        # CUDA: a numpy view moves its buffer to managed memory and leaves the device copy in MLX's pool until a clear_cache, so each part is trimmed at once, largest first (build_extra).
        made = {}
        # `swing`: how far the driver's own memory beside MLX's moved while this build ran (CUDA: +15-19 MB a managed buffer, -110-140 MB at times).
        outside = lambda: phys_footprint() - mx.get_active_memory() - mx.get_cache_memory()
        x0 = outside() if BACKEND == "cuda" else 0
        lo = hi = 0
        for j, i in self._order(glus[0]):
            part = glus[0]._projs[j]._packed["parts"][i]
            b, v = (_alloc_dev_rows if _RESIDENT else _alloc_rows)(part, self.slots + _STAGING_ROWS)
            made[j, i] = (v, b)
            if BACKEND == "cuda":
                x = outside() - x0 - (0 if _RESIDENT else b.nbytes)    # the freed device copy is b.nbytes, untrimmed
                lo, hi = min(lo, x), max(hi, x)
                mx.clear_cache()
        self.swing = hi - lo
        self.bufs, self.views, self.iov = [], [], []
        for j, p in enumerate(glus[0]._projs):
            views = [made[j, i][0] for i in range(len(p._packed["parts"]))]
            self.bufs.append(tuple(made[j, i][1] for i in range(len(views))))
            self.views.append(views)
            self.iov.append([[v[s] for v in views] for s in range(self.slots)])
        # Ring bookkeeping: which (layer, expert) holds a ring slot and each half's next slot, while quota slots need neither since only their layer ever writes them.
        self.owner = [None] * self.slots
        self.next = [0] * halves
        self.dirty = False      # anything published since the last release
        # _KEEP: each layer's quota members at the last release that had any, and the rows written since (a prefill's scratch), see MemBudget._put_back.
        self.kept, self.written = None, None

    @staticmethod
    def _part_bytes(p, i):
        part = p._packed["parts"][i]
        return int(np.prod(part["shape"])) * np.dtype(_ST_NP[part["dtype"]]).itemsize

    @staticmethod
    def _order(glu):
        """(projection, part) in build order: largest first."""
        return sorted(((j, i) for j, p in enumerate(glu._projs) for i in range(len(p._packed["parts"]))),
                      key=lambda ji: -ExpertStore._part_bytes(glu._projs[ji[0]], ji[1]))

    @staticmethod
    def build_extra(glu):
        """Bytes a slot of `glu`'s shape holds beyond the store while it is built, the part being moved twice on CUDA, 0 elsewhere (and under MOE_RESIDENT, whose store stays in device memory)."""
        if BACKEND != "cuda" or _RESIDENT:
            return 0
        sizes = [ExpertStore._part_bytes(glu._projs[j], i) for j, i in ExpertStore._order(glu)]
        total, run, top = sum(sizes), 0, 0
        for s in sizes:
            run += s
            top = max(top, run + s)
        return max(0, top - total)

    @staticmethod
    def gather_slots(glu):
        """The most slots a store of `glu`'s shape may hold, since MLX CUDA's qmm kernels offset a gathered expert in 32 bits."""
        if BACKEND != "cuda":
            return 1 << 62
        per = max(int(np.prod(part["shape"])) * (np.dtype(_ST_NP[part["dtype"]]).itemsize * 8 // p.bits
                                                if part["name"] == "weight" else 1)
                  for p in glu._projs for part in p._packed["parts"])
        return (1 << 32) // per

    def half_of(self, sl):
        """The ring half slot `sl` is in; None for a quota slot."""
        return None if sl < self.pool else (sl - self.pool) // self.per_half

    def ring_slot(self, glu, eid, keep, half=None):
        """The next slot of ring half `half` (`glu`'s own by default), skipping `keep`: the previous occupant's gather is done because its layer is at least two layers back and a lookahead writes a half neither this layer's nor the next one's (StreamingSwitchGLU._call_rounds)."""
        h, n = glu._half if half is None else half, self.per_half
        base = self.pool + h * n
        for _ in range(n):
            sl = base + self.next[h]
            self.next[h] = (self.next[h] + 1) % n
            if sl not in keep:
                break
        else:
            raise RuntimeError("ring half full of this call's own hits")
        old = self.owner[sl]
        if old is not None:
            g, e = old
            if g._ring.pop(e, None) is not None and g._slot_np is not None:
                g._slot_np[e] = -1
                if g is not glu and g._bias_hook is not None:
                    g._bias_hook.refresh()
        self.owner[sl] = (glu, eid)
        glu._ring[eid] = sl
        self.dirty = True
        return sl

    def scratch(self, n, base=0):
        """MOE_RESIDENT: n ring rows from `base` on as a prefill group's scratch (base + n <= the ring), their members unpublished first as ring_slot evicts them."""
        rows = list(range(self.pool + base, self.pool + base + n))
        for sl in rows:
            old, self.owner[sl] = self.owner[sl], None
            if old is not None:
                g, e = old
                if g._ring.pop(e, None) is not None and g._slot_np is not None:
                    g._slot_np[e] = -1
                    if g._bias_hook is not None:
                        g._bias_hook.refresh()
        return rows

    def release(self):
        """Unpublish every quota and ring member of every layer of this shape, so the whole tensor can be a wide pass's scratch, safe because a wide layer's routing eval has forced every gather below it."""
        if not self.dirty:
            return
        if _KEEP:
            self.kept = [(g, list(g._lru.items())) for g in self.glus if g._lru]
            self.written = np.zeros(self.slots, dtype=bool)
        for g in self.glus:
            g._await_pending()
            g._lru.clear()
            g._ring.clear()
            if g._slot_np is not None:
                g._slot_np[:] = -1
            g._arena_gen += 1
            g.__dict__.pop("_log", None)
            if g._bias_hook is not None:
                g._bias_hook.refresh()
        self.owner = [None] * self.slots
        self.next = [0] * self.halves
        self.dirty = False


class MemBudget:
    """Sizes the stores' pools from a peak memory budget as budget minus host, held, ring and the coming pass's need, so a prefill time-shares its bytes between the ring and the pool and a decode keeps the pool."""

    def __init__(self, glus, peak_gb, cache_gb, top_k, t_max, depth, row_elems,
                 make_cache, calib=None):
        self.all = glus
        self.peak, self.cache = peak_gb * 1e9, cache_gb * 1e9
        self.top_k = top_k
        self.n_exp = glus[0]._projs[0]._packed["n_experts"]
        # Consecutive layers may not share a half: run-ahead queues layer L+1 while L's check claims, and a lookahead of depth d writes L+d's half while L's gather is queued, so it needs 1+d.
        self.halves = 1 + max(1, depth)
        # One pass of the widest decode step claims at most top_k*t_max experts, which the ring half holds even when the layer's quota is 0.
        self.per_half = top_k * t_max
        self.groups = {}
        for i, g in enumerate(glus):
            n = g._projs[0]._packed["n_experts"]
            g._budget, g._half, g._skey = self, i % self.halves, _shape_key(g)
            g._qslots = []
            # Ghost LRU: the last pass each expert was wanted in and how many hits a quota of q slots would have had (hist[d] = accesses at stack distance d), ids only.
            g._ghost_last = np.full(n, -1, dtype=np.int64)
            g._ghost_hist = np.zeros(n, dtype=np.int64)
            g._ghost_clock = 0
            self.groups.setdefault(g._skey, []).append(g)
        self.slot_bytes = {k: sum(p._packed["stride"] for p in gs[0]._projs)
                           for k, gs in self.groups.items()}
        self.ring_bytes = sum((self.halves * self.per_half + _STAGING_ROWS) * b
                              for b in self.slot_bytes.values())
        # The pool slots each shape's gathers can address (ExpertStore.gather_slots), and the share of a store build held beyond the store (ExpertStore.build_extra, CUDA only).
        self.slot_cap = {k: ExpertStore.gather_slots(gs[0]) - self.halves * self.per_half - _STAGING_ROWS
                         for k, gs in self.groups.items()}
        self.build_frac = max(ExpertStore.build_extra(gs[0]) / self.slot_bytes[k]
                              for k, gs in self.groups.items())
        self.build_swing = 0            # the widest ExpertStore.swing seen; the first build is the ring's
        self.stores = {}
        # The narrowest pass that is a prefill whatever the cache holds, since no piece of a cut prefill is narrower and every narrower pass keeps the pool, while an MLX whose two kernels never part has no such width and nothing is cut.
        line = max(SORT_MIN, _rhs_rows(self.n_exp))
        self.min_w = line if line == float("inf") else -(-int(line) // top_k)
        # What a need is predicted from (see above), None until measured, with `copy` read at every pass's entry from the conversation.
        self.rate = self.state = None
        self.keep, self.copy = 0.0, 0.0
        # The widest pass behind `keep`, and this conversation's own most and widest, which reset starts over.
        self._keep_w, self._mine = 0, (0.0, 0)
        # The pass being measured: positions, empty cache, prefill, the need it was sized for, weak references to its recurrent caches, its copy, what the stack held when it began, the host and the stores' bytes then, and the peak kept.
        self._pass = None
        self._held0 = self._host = self._have = self._top = self._foot0 = 0
        # The host: the file's floor with the passes behind it and this process's own top and passes, the floor sizing the room until this process has run as many passes.
        self._host_top = self._ticks = 0
        self._host_floor = self._floor_n = 0
        self.prefill = False
        self._make_cache = make_cache
        # chunkattn's ceiling per key (moe_row_elems, prefill_cap_bytes).
        self._row_elems = row_elems
        self.rebuilds = self.probes = 0
        # MOE_RESIDENT: whether the one fill (prepare) has run, what it placed, and this process's and this setup's most RssAnon (calibration).
        self.filled, self.resident = False, None
        self.intact = False     # the fill's store is still the store (no rebuild since)
        self._rss_top = self._rss_saved = 0
        # Set when the store governor (make_store_governor) moved the peak, so the next pass rebuilds the store at the new size, while a lower peak alone shrinks it only once a pass would not fit.
        self.resized = False
        self.pieces = []        # widths of the last prefill's passes on one cache
        # Layer-major prefill (lm_supported, enable_budget): its passes' growth beside their state by width, of every pass (lm_seen) and of the fresh ones only (lm_fresh), which are the ones a room is compared to, with lm_ok for a stack call that may run it and lm_pass for the pass that does.
        self.layer_major = self.lm_ok = self.lm_pass = False
        self._lm_seen, self._lm_fresh, self._lm_rows, self._rows_top = {}, {}, {}, 0
        # The width this pass is cut to (LM_W unless the keys leave no room for a chunk that wide) and the narrowest it may go, since a chunk's attention grows with the keys and a fixed LM_W paid 6.4 GB for scores at 64k.
        self.lm_w, self.min_chunk = LM_W, ATTN_CHUNK_MIN
        self.last = {}
        self.grew = self.grew_n = self.grew_keys = self.grew_w = self.grew_attn = 0
        self._short = False
        self._calib, self._saved = calib, None
        # Whether the last turn of this setup ended with room for a kept conversation (_KEEP_FIRST), None until known.
        self.room_ok = None
        try:
            saved = json.loads(Path(calib).read_text())
            if int(saved.get("v", 0)) != CALIB_VERSION:
                raise ValueError(f"calibration schema {saved.get('v')}")
            self.room_ok = saved.get("room_ok")
            self._rss_saved = float(saved.get("rss_mb", 0)) * 1e6
            self.rate, self.state, self.keep, self._keep_w, self._host_floor = (
                float(saved["rate"]), float(saved["state"]), float(saved["keep"]),
                int(saved["keep_w"]), float(saved["host_mb"]) * 1e6)
            self._floor_n = int(saved.get("host_n", 0))
            # On CUDA, widths under min_w cannot come from a pass this budget made (its probe runs at min_w), so they are another engine's or a toy's, while Metal keeps every width as df27abb4 did.
            lo = self.min_w if BACKEND == "cuda" else 0
            self._lm_seen = {int(k): float(v) for k, v in saved.get("lm_seen", {}).items()
                             if int(k) >= lo}
            self._lm_fresh = {int(k): float(v) for k, v in saved.get("lm_fresh", {}).items()
                              if int(k) >= lo}
            self._lm_rows = {int(k): float(v) for k, v in saved.get("lm_rows", {}).items()
                             if int(k) >= lo}
            self._saved = self._calibration()
        except (TypeError, OSError, ValueError, KeyError):
            pass

    def _calibration(self):
        # The host in whole MB, rounded up, since it creeps by KBs and a write per KB would be a write per pass, with its passes rounded down to a power of two for the same reason.
        mine = self._ticks >= self._floor_n
        host = self._host_top if mine else self._host_floor
        d = {"v": CALIB_VERSION,
             "rate": self.rate, "state": self.state, "keep": self.keep,
             "keep_w": self._keep_w, "host_mb": -(-int(host) // 10**6),
             "host_n": 1 << max(self._ticks, self._floor_n).bit_length() >> 1}
        if self._lm_seen:
            d["lm_seen"] = {str(k): v for k, v in sorted(self._lm_seen.items())}
            d["lm_fresh"] = {str(k): v for k, v in sorted(self._lm_fresh.items())}
            d["lm_rows"] = {str(k): v for k, v in sorted(self._lm_rows.items())}
        if self.room_ok is not None:
            d["room_ok"] = self.room_ok
        if _RESIDENT:
            d["rss_mb"] = -(-int(max(self._rss_top, self._rss_saved)) // 10**6)
        return d

    def _host_now(self):
        # A short process keeps the longer one's host as a floor.
        if self._ticks >= self._floor_n:
            return self._host_top
        return max(self._host_top, self._host_floor)

    def _save_calibration(self):
        """Write what this process measured for the next one of this setup, atomically and only once it has changed, since a file without a rate and a state would only mean the probe again."""
        now = self._calibration()
        if self._calib is None or now == self._saved or None in (self.rate, self.state):
            return
        self._saved = now
        p = Path(self._calib)
        tmp = p.with_name(f"{p.name}.{os.getpid()}.tmp")
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(now))
            tmp.replace(p)
        except OSError as e:
            tmp.unlink(missing_ok=True)
            self._calib = None
            print(f"  memory budget: could not save its calibration to {p}: {e}",
                  flush=True)

    def _measure(self):
        active = mx.get_active_memory()
        have = sum(st.nbytes for st in self.stores.values())
        # The host is budgeted at the most it has been, not at this reading, since it moves by tens of MB between and within identical runs and a room sized off a low reading ended 10 MB over a 2 GB budget.
        mlx = active
        # CUDA's footprint (cudaMemGetInfo) lets go at once while MLX's freed bytes stay its cache, so there the peak under-read the host by 0.3 GB after a long pass.
        if self.layer_major and BACKEND != "cuda":
            # The footprint keeps a big pass's freed pages for a while, so right after one it is MLX's peak there and not what MLX holds now, and the host is read against the most MLX held since the last pass began.
            mlx = max(active, self._top, mx.get_peak_memory())
        host = phys_footprint() - mlx - mx.get_cache_memory()
        if _POOL_TRIM and host > self._host_top:
            # The pool's freed reserve reads as other programs' memory (host 1.30 -> 7.03 GB after a 20k pass), so it is read again with the pool given back.
            mx.clear_cache()
            host = phys_footprint() - mlx - mx.get_cache_memory()
        self._host_top = max(self._host_top, host)
        return active - have, self._host_now(), have

    def _dev(self):
        """_POOL_TRIM, in a prefill pass: MLX's bytes as if the device's growth since the pass began were all its own."""
        if not self._foot0:
            return 0
        return phys_footprint() - self._foot0 + self._held0 + self._have

    def _is_prefill(self, n, fresh):
        pairs = n * self.top_k
        return pairs >= SORT_MIN and (fresh or _takes_rhs(pairs, self.n_exp))

    def _need(self, n, fresh, prefill, copy=None, keys=None):
        """What a pass of `n` positions is predicted to grow MLX by, None while there is nothing to predict it from, with a prefill's attention for `t` rows over `keys` (the pass's last key) added since the scores are the one term that grows with the context."""
        if self.rate is None or (fresh and self.state is None):
            return None
        fixed = self.state if fresh else (self.copy if copy is None else copy)
        keys = n if keys is None else keys
        if prefill and self._lm_now():
            a, b = self._lm_line()
            if n < LM_W:
                # Upper bound from each width measured at least as wide, since a narrower width bounds nothing (scaled off 205's, a reuse turn's 1995 ids planned 399 MB and grew 737).
                a = max(0.0, min([a] + [g - b * w for w, g in self._lm_seen.items()
                                        if w >= n]))
            return (a + b * n + fixed + self._keep_extra()
                    + attn_bytes(min(n, self.lm_w), keys))
        if prefill:
            # Unchunked, every one of the pass's own rows attends at once.
            return (self.rate * n + fixed + attn_bytes(n, keys))
        return self.rate * n + fixed + self.keep

    def _prefill_peak(self):
        """R1 (_R1): the peak a prefill is planned at, the room the store governor would give the store now, since the prefill releases the store anyway."""
        gov = _GOV
        if not _R1 or gov is None or gov.store is not self:
            return self.peak
        level, _, comp, _ = gov._peek()
        if level is None:
            return self.peak
        slots = min(gov.cap_max, gov._roof, gov._ceiling(gov._avail(level, comp)))
        return max(self.peak, gov._base + slots * gov.slot_gb * 1e9)

    def _keep_extra(self):
        """E1 (exl3_experts.KEEP_MB): what MLX's cache may hold through a layer-major pass on EXL3 experts beyond the `cache` the room already leaves it."""
        if not any(g._exl3 is not None for g in self.all):
            return 0.0
        return max(0.0, exl3_experts.KEEP_MB * 1e6 - self.cache)

    def _lm_now(self):
        return self.layer_major and self.lm_ok and bool(self._lm_seen) and bool(self._lm_rows)

    def _lm_pair(self):
        """Whether the line has two widths a factor of two apart, which is what cancels its fixed part (_lm_line)."""
        seen = self._lm_fresh or self._lm_seen
        # A probe that recorded no layer-major width (a toy model) has no line for a second width to fix.
        return not seen or any(2 * w <= max(seen) for w in seen)

    def _lm_line(self):
        """(fixed, per position) of a layer-major prefill's growth beside its state: a pass holds every position's hidden state, expert rows, output and KV, so its growth is a fixed part plus a position's share times the positions, and two widths whose ratio is at least two cancel that fixed part, since the widest pass alone divides it by its own width and a 205-position probe then read two to four times the slope it measured. Under one chunk the fixed part itself still grows with the width (a golden run's 265-position pass grew 98.7 MB and its 530 one 179.6, so the pair reads 0.30 MB a position where the 8k pass grew 0.14), and there the plan takes the smaller of that pair and the rows a position's expert output holds (four of them, PREFILL_ROWS), which both machines' long passes measured at 0.13 to 0.15."""
        if BACKEND != "cuda":
            # df27abb4's line on Metal: the widest pass's rows a position, raised by the secant of the two widest long passes, since the two-width rule below was fitted for CUDA.
            seen = self._lm_seen
            rate = self._lm_rows[max(self._lm_rows)]
            wide = [w for w in seen if w >= 2 * LM_W]
            if wide:
                hi = max(wide)
                lo = max((w for w in wide if 1.25 * w <= hi), default=None)
                if lo:
                    rate = max(rate, (seen[hi] - seen[lo]) / (hi - lo))
            a = max(g - rate * w for w, g in seen.items())
            top = max(seen)
            if top < LM_W:
                a = max(a, (seen[top] - rate * top) * LM_W / top)
            return a, rate
        seen = self._lm_fresh or self._lm_seen
        hi = max(seen)
        rate = seen[hi] / hi
        # The pair is the widest pass and the widest at most half as wide at any width, since two long passes a few hundred positions apart divide their run to run spread by that gap (the PC's 32,558 and 32,767 grew 6,024 and 5,714 MB, a slope of 0).
        lo = max((w for w in seen if 2 * w <= hi), default=None)
        # A pair whose wider pass grew no more measures noise, so the widest pass's own growth a position stands there, where a slope of 0 cut every piece to min_chunk.
        if lo is not None and seen[hi] > seen[lo]:
            rate = (seen[hi] - seen[lo]) / (hi - lo)
        if hi < LM_W and self._row_elems:
            rate = min(rate, PREFILL_ROWS * self._row_elems * 2)
        # The widest fixed part any measured pass implies, so the line covers every pass behind it and not only the two the slope came from.
        return max(0.0, max(g - rate * w for w, g in seen.items())), rate

    def lm_rows(self):
        """_lm_layer_call, at a layer's first combine: every row the layer holds (its input, MoE input and rows, output, the chunks' kept state, the new KV) is alive here (MemBudget.close's rows figure, beside the footprint's)."""
        self._rows_top = max(self._rows_top, mx.get_active_memory(), self._dev())

    def plan(self, n, fresh, caches):
        """Positions of the first piece of a pass of `n`: all of them unless it is a prefill whose need does not fit even at pool 0, then as many as fit at the widest query chunk the room allows, with the rest planning again at its own entry and today's LM_W chunk and piece kept wherever they already fit."""
        if _KV_HOST:
            kv_host.drop_live()         # the last pass's staged layer is not this one's
        self.prepare()
        self.lm_w = max(self.min_chunk, _LM_CHUNK) if _LM_CHUNK else LM_W
        if n <= self.min_w or not self._is_prefill(n, fresh):
            return n
        if _PIECE:
            # The golden harness's exactness check: a piece and chunk width forced from outside, to show a cut piece gives the whole pass's logits.
            return min(n, max(self.min_w, _PIECE))
        _eval_caches(caches)
        copy = _copy(caches)
        keys = _keys(caches) + n
        need = self._need(n, fresh, True, copy, keys)
        if need is None:
            return None
        held, host, have = self._measure()
        # MOE_RESIDENT: the store stays through the prefill (no release), so the pass plans in the room beside all of it.
        room = (self.peak - self.cache - host - held - have if _RESIDENT
                else self._prefill_peak() - self.cache - host - held - self.ring_bytes)
        if need <= room:
            return n
        fixed = self.state if fresh else copy
        if BACKEND != "cuda":
            # df27abb4's piece rule on Metal: one chunk width, as many whole LM_W chunks as fit, else equal pieces, since the width ladder below was fitted for CUDA.
            lm = self._lm_now()
            a, rate = self._lm_line() if lm else (0.0, self.rate)
            fits = int((room - fixed - a - (self._keep_extra() if lm else 0.0)) / rate) if rate > 0 else 0
            if lm and fits >= LM_W:
                # A layer-major piece: as many whole LM_W chunks as fit, the rest starting on the 2048-token chunk grid of the passes it replaces so its arithmetic is theirs however the budget cuts it.
                return min(n, fits // LM_W * LM_W)
            k = -(-n // max(fits, 1))
            # A narrower piece would keep the pool, so min_w stands even where it does not fit and _fit says the budget is not met.
            return max(self.min_w, -(-n // k))
        sdpa, ix, per_row, flat = (_PF_ATTN or (0.0, 0.0, 0.0, 0.0))[:4]
        if self._lm_now():
            a, rate = self._lm_line()
            a += self._keep_extra()
            if fresh:
                # The line is measured apart from the state a fresh pass allocates, which the room does not hold yet.
                a += self.state or 0.0
            if _PF_ATTN is None:
                # A CUDA load whose attention probe failed plans with one chunk
                # width, as many whole LM_W chunks as fit, as Metal does above.
                fits = int((room - a - (0.0 if fresh else copy)) / rate) if rate > 0 else 0
                if fits >= LM_W:
                    return min(n, fits // LM_W * LM_W)
                k = -(-n // max(fits, 1))
                return max(self.min_w, -(-n // k))
            # A chunk's scores grow with the keys, so every width is tried: the widest that takes the whole pass in one piece wins, else the one that takes the fewest pieces, since a piece reads every expert again and a narrower chunk only divides the scores.
            widths = [max(self.min_chunk, w) for w in (LM_W, 1024, 512, 256, self.min_chunk)]
            fits = dict((w, int((room - a - attn_bytes(min(n, w), keys)) / rate)
                         if rate > 0 else 0) for w in widths)
            counts = dict((w, -(-n // fits[w]) if fits[w] > 0 else n) for w in widths)
            w = next((x for x in widths if counts[x] <= 1), None)
            if w is not None:
                self.lm_w = w
                return n
            w = min(widths, key=lambda x: (counts[x], -x))
            self.lm_w = w
            if fits[w] >= w:
                # Equal pieces at that width, since the room a pass leaves is the room its next one plans in and a need cut to the last position leaves a hair at the end.
                piece = min(fits[w], -(-n // counts[w]))
                return min(n, -(-piece // w) * w)
            return max(self.min_chunk, min(n, fits[w])) if fits[w] else self.min_chunk
        # Unchunked, the piece's own rows attend at once, its SDPA cut only by
        # chunkattn's own width, so the fit solves for the rows whose own
        # growth and indexer scores both land in the room.
        rows = (_PF_ATTN[4] if _PF_ATTN else 0) or n
        cost = self.rate + ix * keys + per_row + (sdpa * keys if rows >= n else 0.0)
        fits = int((room - fixed - flat * keys) / cost) if cost > 0 else 0
        k = -(-n // max(fits, 1))
        # A narrower piece would keep the pool, so min_w stands even where it does not fit and _fit says the budget is not met.
        return max(self.min_w, -(-n // k))

    def tick(self, n, fresh, caches):
        """Once per forward pass of the layer stack, before any layer runs (the stack's call, after close() and plan()), since a refit clears quotas every router ranks against and at the first streaming layer instead it landed between that layer's router and its gather (defer + cache_bias 0.5 repaired against an emptied hold and raised), sizing the pool for this pass of `n` positions and then measuring it."""
        global _MEASURING
        if _KV_HOST:
            kv_host.drop_live()
        _eval_caches(caches)
        self._ticks += 1
        after = self.prefill
        self.prefill = self._is_prefill(n, fresh)
        self.lm_pass = self.prefill and self.layer_major and self.lm_ok
        if not fresh:
            self.copy = _copy(caches)
        # The pass's own rows plus what the cache already holds: its attention reads both.
        keys = _keys(caches) + n
        need = self._need(n, fresh, self.prefill, keys=keys)
        if self.prefill:
            # The pass releases the pool (it is _call_grouped's scratch), so a smaller store costs nothing here and gives up just this need.
            self.pieces = self.pieces + [n] if after and not fresh else [n]
            if not after:
                for g in self.all:
                    g._pc = None        # the prompt's routing counts start again (_call_grouped)
            self.resized = False        # sized from the new peak here and after
            if _RESIDENT and self.filled:
                mx.synchronize()        # the ring rows become the groups' scratch (ExpertStore.scratch)
                if not _device_room(need):
                    self._fit(False, need)
            else:
                self._fit(False, need)
        elif after and _RESIDENT and self.filled:
            # MOE_RESIDENT: the prefill left the store as it was, and it stays unless the device itself has no room for this pass.
            if not _device_room(need):
                self.fit(rebuild=False)
        elif after and _GROW and _GOV is not None and _GOV.store is self and _GOV.grow_now():
            # The governor grew the store as the prefill ended, so the pool the prefill emptied is built at its new size now, as the resized branch below does mid-decode.
            self.resized = False
            if _NATIVE is not None:
                _NATIVE.release()
            mx.synchronize()
            self.fit(rebuild=True)
            if _SEED:
                STATS["store_seeded"] += self._seed()
        elif after:
            # The prefill is over and left the pool empty, so back to the standing room with _KEEP's quota members whose rows it did not write coming back first and the store then keeping its size while it fits.
            kept = self._put_back() if _KEEP else 0
            if kept:
                STATS["store_kept"] += kept
                self._fit(False, self._need(self.min_w - 1, False, False))
            else:
                self.fit(rebuild=True)
            if _SEED:
                STATS["store_seeded"] += self._seed()
        elif self.resized:
            # The store governor moved the peak, so the pool is rebuilt at its new size empty and refilled by misses, with the native agent drained and its LRU back first since a store kept at its size is released in place.
            self.resized = False
            if _NATIVE is not None:
                _NATIVE.release()
            mx.synchronize()            # no gather of the last pass still reads it
            self.fit(rebuild=True)
        else:
            held, host, have = self._measure()
            if not self.stores or (
                    need is not None and any(st.pool for st in self.stores.values())
                    and held + have + host + self.cache + need > self.peak
                    # MOE_RESIDENT: past the estimate, the store stays unless the device itself has no room for this pass.
                    and not (_RESIDENT and self.filled and _device_room(need))):
                self.fit(rebuild=False)
        if _RESIDENT_LOG:
            print(f"  resident pass {n} prefill {self.prefill} pool {sum(st.pool for st in self.stores.values())} "
                  f"rebuilds {self.rebuilds} tier {STATS['resident_tier_rows']} ssd {STATS['resident_ssd_b'] / 1e9:.2f} GB "
                  f"ring {STATS['resident_ring']} absent {STATS['resident_absent']}", flush=True)
        # Weak: held here, a conversation's cache outlived the reset that dropped it until the next request's first pass.
        rec = ([weakref.ref(c) for c in caches if isinstance(c, ArraysCache)]
               if fresh and caches else None)
        self._pass = (n, fresh, self.prefill, need, rec, 0.0 if fresh else self.copy,
                      self.lm_pass, keys, self.lm_w)
        self._held0, self._host, self._have = self._measure()
        if _POOL_TRIM and self.prefill:
            mx.clear_cache()
            self._foot0 = phys_footprint()
        self._top = self._rows_top = 0
        _MEASURING = self
        mx.reset_peak_memory()

    def close(self):
        """Fold the pass that ran last into the prediction: how far MLX grew past what the stack held when it began, stores left out, called at the next pass's entry so what the caller did in between counts with it."""
        if self._pass is None:
            return
        n, fresh, prefill, _, rec, copy, lm, keys, w = self._pass
        self._pass = None
        if _RESIDENT:
            self._rss_top = max(self._rss_top, _rss_anon())
        # MOE_ANE_EXPERTS: its surfaces live only inside a prefill pass, so they stay in the line's fixed part and out of a position's rate, which sizes decode's pool (0 when off).
        side = ane_experts.take_bytes()
        # The pass just folded in, for the plan-vs-measurement probe and report().
        self.grew = self.grew_n = self.grew_keys = self.grew_w = self.grew_attn = 0
        # reset_peak_memory() zeroes the peak and the next allocation sets it to the active bytes then, so it can read under what is held.
        top = max(self._top, mx.get_peak_memory(), self._dev())
        self._foot0 = 0
        grew = max(0.0, top - self._have - self._held0)
        if _KV_HOST:
            # An LRU the pass allocated stands from now on and is in `held` at the next reading, so no pass needs it.
            grew = max(0.0, grew - kv_host.take_added())
        fixed = copy
        rec = None if rec is None else [r() for r in rec]
        if rec is not None and None not in rec:
            # Read off the cache, not fitted, since spread over a 128-position probe Qwen3.6's 62.9 MB of it would read ~5x its ~126 KB a position, and a model without one leaves 0.
            fixed = float(sum(c.nbytes for c in rec))
            self.state = max(self.state or 0.0, fixed)
        if prefill:
            # Attention's scores are the one part of the pass that grows with the keys at every width, so they come out before the line is fitted and the plan adds them back at its own key count.
            attn = attn_bytes(min(n, w) if lm else n, keys)
            self.grew, self.grew_n, self.grew_keys = grew, n, keys
            self.grew_w, self.grew_attn = w, attn
            self.rate = max(self.rate or 0.0, (grew - fixed - attn - side) / n)
            if lm and self._rows_top:
                self._lm_seen[n] = max(self._lm_seen.get(n, 0.0), grew - fixed - attn)
                if fresh:
                    # A piece after the first grows from the copy of what is held and not from the state, so only a fresh pass measures what a room is compared to.
                    self._lm_fresh[n] = max(self._lm_fresh.get(n, 0.0), grew - fixed - attn)
                rows = self._rows_top - self._have - self._held0 - (fixed if fresh else 0.0) - side
                # Net of the same scores `grew` is net of, since MLX's active bytes at the combine hold them too.
                self._lm_rows[n] = max(self._lm_rows.get(n, 0.0), max(0.0, rows - attn) / n)
        elif self.rate is not None:
            k = max(0.0, grew - fixed - self.rate * n)
            self._mine = (max(self._mine[0], k), max(self._mine[1], n))
            if self._mine[1] >= self._keep_w:
                # This conversation has run a pass as wide as any behind the keep it was handed, so its own measurement replaces it.
                self.keep, self._keep_w = self._mine
            else:
                self.keep = max(self.keep, k)
        self._save_calibration()

    def reset(self):
        """A new conversation (StreamingLLM.reset) measures its keep again, the one held sizing the standing room until the conversation has run a pass that keeps the pool as wide as the widest behind it, with the rate kept since a prefill's growth a position falls with its width."""
        self._mine = (0.0, 0)

    def room(self):
        """What the budget leaves free now, the host as this pass's tick read it, which chunkattn caps a pass that keeps the pool at."""
        return self.peak - self.cache - self._host - mx.get_active_memory()

    def attention(self, start):
        """chunkattn's bracket around one attention call with its inputs already evaluated: the peak so far is kept and the call's own transient is capped instead (a prefill's at its own MoE rows, a pass that keeps the pool at the room left)."""
        if start:
            self._top = max(self._top, mx.get_peak_memory())
        else:
            # eval returns before the call's intermediates are released (a stock SDPA's 2.9 MB score buffers still read as active until a synchronize), so a reset with them still counted keeps them in the next reading.
            mx.synchronize()
            mx.reset_peak_memory()

    def fit(self, rebuild):
        """The pool beside the standing room: the widest pass that keeps the pool (min_w - 1 positions) fits next to it."""
        self._fit(rebuild, self._standing())

    def _standing(self):
        """The need the pool leaves room for: the widest pass that keeps the pool, and under MOE_RESIDENT, whose prefills
        run beside the store, also a fresh prefill of RESIDENT_ROOM x min_w positions, so a short prompt is one pass as
        without it."""
        need = self._need(self.min_w - 1, False, False)
        if _RESIDENT and need is not None:
            w = RESIDENT_ROOM * self.min_w
            pf = self._need(w, True, True, keys=w)
            need = need if pf is None else max(need, pf)
        return need

    def _put_back(self):
        """_KEEP, after a prefill released the stores: each layer's quota members from the release whose rows no pass wrote since (ExpertStore.kept, ExpertStore.written) are published again in LRU order, with the quotas cut again from the ghost LRU at the store's size (_quotas), since left as the first turn's few ghosts cut them the later turns read 10-25% more."""
        stores = {k: st for k, st in self.stores.items() if st.kept}
        if not stores:
            return 0
        quota = self._quotas(sum(st.pool * self.slot_bytes[k] for k, st in self.stores.items()))
        n = 0
        for key, st in stores.items():
            kept, written = st.kept, st.written
            st.kept = st.written = None
            STATS["store_held"] += sum(len(items) for _, items in kept)
            STATS["store_slots"] += st.pool
            gs = self.groups[key]
            if sum(quota[id(g)] for g in gs) > st.pool:
                quota.update({id(g): len(g._qslots) for g in gs})
            surv = {id(g): [] for g in gs}
            for g, items in kept:
                if g._store is st and g._slot_np is not None and not g._lru:
                    q = set(g._qslots)
                    surv[id(g)] = [(e, sl) for e, sl in items if sl in q and not written[sl]]
            rows, free = {}, set(range(st.pool))
            for g in gs:
                mine = [sl for _, sl in reversed(surv[id(g)])]
                mine += sorted(set(g._qslots) - set(mine))
                rows[id(g)] = mine[:quota[id(g)]]
                free -= set(rows[id(g)])
            free = sorted(free)
            for g in gs:
                r = rows[id(g)]
                k = quota[id(g)] - len(r)
                g._qslots, free = sorted(r + free[:k]), free[k:]
                g._cap = len(g._qslots) + st.per_half
                q = set(g._qslots)
                for e, sl in surv[id(g)]:
                    if sl in q:
                        g._lru[e] = sl
                        g._slot_np[e] = sl
                g._arena_gen += 1
                if g._bias_hook is not None:
                    g._bias_hook.refresh()
                n += len(g._lru)
                st.dirty = st.dirty or bool(g._lru)
        return n

    def _seed(self):
        """MOE_STORE_SEED (an arm, off by default): after a prefill, each layer's empty quota rows take the experts the prefill routed most (StreamingSwitchGLU._pc), read now and returned as a count."""
        mx.synchronize()            # the prefill's last gathers read store rows
        n = 0
        for g in self.all:
            pc = g.__dict__.get("_pc")
            if pc is None:
                continue
            if g._akey not in _ARENAS:
                g._init_arena()     # the store was rebuilt: its arena comes at the first claim
            st = g._store
            free = sorted(set(g._qslots) - set(g._lru.values()))
            top = [int(e) for e in np.argsort(-pc, kind="stable")[:len(free) + len(g._lru)]
                   if pc[e] > 0 and int(e) not in g._lru and g._slot_np[e] < 0][:len(free)]
            if not top:
                continue
            pairs = list(zip(top, free))
            for e, sl in reversed(pairs):
                g._lru[e] = sl
                g._slot_np[e] = sl
            jobs = [(fd, iov[sl], off + e * stride, stride)
                    for (_, iov, fd, off, stride) in _ARENAS[g._akey] for e, sl in pairs]
            _read_wave(jobs, g)
            STATS["rows_warm"] += len(jobs)
            st.dirty = True
            g._arena_gen += 1
            if g._bias_hook is not None:
                g._bias_hook.refresh()
            n += len(pairs)
        return n

    def prepare(self):
        """MOE_RESIDENT, once a pass has been measured: the stores built at the standing room and filled (fill), at load
        (server.py, after the store governor) or else at the first pass's plan, so a first long prompt does not size them."""
        if not _RESIDENT or self.filled or self.rate is None:
            return
        self.fit(rebuild=bool(self.stores))
        if any(st.pool for st in self.stores.values()):
            self.fill()

    def fill(self):
        """MOE_RESIDENT, once: each layer's empty quota rows take the experts neither the store nor the native agent's tier
        holds, the most asked-for by this process first, and the tier the rest as far as its RAM rule allows, read from
        experts.bin (the store's rows through the staging copy, the tier's straight into its pinned slots)."""
        self.filled = True
        t0 = time.perf_counter()
        self.intact = True
        nat = _NATIVE
        if nat is not None:
            nat.release()               # the tables are Python's while rows move, and the agent is idle
        mx.synchronize()
        tier = nat is not None and nat.cuda and self._size_tier(nat)
        dfd, n_vram, n_tier, b_vram, b_tier, total = {}, 0, 0, 0, 0, 0
        for g in self.all:
            if g._akey not in _ARENAS:
                g._init_arena()
            ar = _ARENAS[g._akey]
            n = g._projs[0]._packed["n_experts"]
            total += n
            hot = np.zeros(n) if g._seen is None else g._seen[:n].copy()
            if g.__dict__.get("_pc") is not None:
                hot += g._pc[:n]
            in_tier = (nat.agent.tier_find(g._native_idx, np.arange(n)) != 0) if tier else np.zeros(n, bool)
            held = set(g._lru) | set(g._ring)
            cand = [int(e) for e in np.argsort(-hot, kind="stable") if int(e) not in held and not in_tier[e]]
            used = set(g._lru.values())
            free = [sl for sl in g._qslots if sl not in used]
            put = list(zip(cand, free))
            if put:
                _read_wave([(fd, iov[sl], off + e * stride, stride) for (_, iov, fd, off, stride) in ar
                            for e, sl in put], g)
                for e, sl in reversed(put):     # the hottest last, so the LRU evicts it last
                    g._lru[e] = sl
                    g._slot_np[e] = sl
                g._store.dirty = True
                n_vram += len(put)
                b_vram += len(put) * g._store.slot_bytes
            got = []
            for e in cand[len(free):] if tier else ():
                a = nat.agent.tier_fill(g._native_idx, n, e)
                if not a:
                    break
                got.append((e, a))
            if got:
                jobs, at = [], 0
                for (_, _, fd, off, stride) in ar:
                    if fd not in dfd:
                        dfd[fd] = _direct_fd(fd)
                    d = dfd[fd] if (off % 4096 == 0 and stride % 4096 == 0 and nat.tier_slot % 4096 == 0) else None
                    for e, a in sorted(got):
                        jobs.append((fd, d, [(ctypes.c_uint8 * stride).from_address(a + at)], off + e * stride, stride))
                    at += stride
                list(_POOL.map(_fill_read, jobs))
                n_tier += len(got)
                b_tier += sum(s for (_, _, _, _, s) in ar) * len(got)
            g._arena_gen += 1
            if g._bias_hook is not None:
                g._bias_hook.refresh()
        for d in dfd.values():
            if d is not None:
                os.close(d)
        s = time.perf_counter() - t0
        self.resident = {"experts": total, "vram": n_vram, "tier": n_tier, "vram_gb": round(b_vram / 1e9, 2),
                         "tier_gb": round(b_tier / 1e9, 2), "fill_s": round(s, 1)}
        note(f"  resident: {n_vram} experts in VRAM ({b_vram / 1e9:.2f} GB) and {n_tier} in the tier "
             f"({b_tier / 1e9:.2f} GB) of {total}, {total - n_vram - n_tier} left to the SSD; read in {s:.1f} s "
             f"({(b_vram + b_tier) / max(s, 1e-9) / 1e9:.1f} GB/s)", flush=True)

    def _size_tier(self, nat):
        """MOE_RESIDENT: the native agent's tier sized for the experts the pools leave out plus a ring of puts, within
        RESIDENT_FLOOR of MemAvailable, and put on the store governor's TierGovernor at that rule; -> whether there is one."""
        slot = nat.tier_slot or max(sum(p._packed["stride"] for p in g._projs) for g in nat.glus)
        out = sum(g._projs[0]._packed["n_experts"] - len(g._qslots) for g in self.all) + self.halves * self.per_half
        now = nat.tier_gov.n if nat.tier_gov is not None else nat.tier_n
        # Room left for this process's own growth to the most RssAnon this setup has measured, since filled at load it grew 0.9 -> 2.05 GB in its first request and the kernel moved 470 MB to zram.
        grow = max(0.0, self._rss_saved - _rss_anon())
        room = _meminfo()["MemAvailable"] + now * slot - RESIDENT_FLOOR - grow
        n = max(0, min(out, int(room // slot)))
        if n == 0 and not nat.tier_slot:
            return False
        # Chunks of ~256 MB under the store governor, as _make_vram_governor makes them, so a shrink keeps the experts below the cut.
        gov_on = isinstance(_GOV, VramGovernor)
        n = nat.agent.set_tier(slot, n, max(1, int(256e6 // slot)) if gov_on else 0)
        nat.tier_n, nat.tier_slot = n, slot
        if not nat.tier_rss0:
            nat.tier_rss0 = _rss_anon()
        gov = nat.tier_gov
        if gov is None and gov_on:
            gov = nat.tier_gov = TierGovernor(n, n, slot, RESIDENT_FLOOR, own0=nat.tier_rss0, div=1)
        elif gov is not None:
            gov.n = gov.n_max = n
            gov.floor, gov.div = RESIDENT_FLOOR, 1
        return n > 0

    def _fit(self, rebuild, need):
        """Pool target from the budget now, leaving `need` (None: not known yet, pool 0), with `rebuild` saying the pool is empty so any size change is free and otherwise only a store over its target rebuilt since a peak budget cannot wait for a better moment."""
        held, host, have = self._measure()
        room = self.peak - self.cache - host - held
        # A build holds its own extra (build_frac) and the driver's swing beside it before the next pass needs anything, so both must fit the room.
        pool = 0.0 if need is None else max(0.0, min(
            room - need, (room - self.build_swing) / (1.0 + self.build_frac)) - self.ring_bytes)
        # R1: a prefill planned over this peak (_prefill_peak) is meant to run over it.
        if (need is not None and room - need < self.ring_bytes and not self._short
                and not (self.prefill and self._prefill_peak() > self.peak)):
            # Said once: the ring is what decode cannot do without, so the budget is then simply not met and the shortfall is reported rather than read off a footprint graph later.
            self._short = True
            print(f"  memory budget {self.peak / 1e9:.2f} GB not met: host "
                  f"{host / 1e6:.0f} MB + held {held / 1e6:.0f} MB + the next "
                  f"pass's need {need / 1e6:.0f} MB leave "
                  f"{(room - need) / 1e6:.0f} MB, the ring alone needs "
                  f"{self.ring_bytes / 1e6:.0f} MB; pool 0", flush=True)
        quota = self._quotas(pool)
        # _KEEP: a store that fits beside `need` as it is stays, and the pool above budgets a build's transient (CUDA) that no build pays here, where that transient alone rebuilt it twice a turn.
        fits = (_KEEP and not rebuild and need is not None
                and room - need >= sum(st.nbytes for st in self.stores.values()))

        def due(key, st):
            size = sum(quota[id(g)] for g in self.groups[key])
            return st is None or (size < st.pool and not fits) or (rebuild and size != st.pool)
        drop = [k for k, s in self.stores.items() if BACKEND == "cuda" and due(k, s)]
        if drop:
            # CUDA: every store due for a rebuild goes before any is built, since a program taking the freed VRAM made the small store's rebuild fail first and kept the big one held.
            mx.synchronize()
            for k in drop:
                for g in self.groups[k]:
                    g.clear_arena()
                del self.stores[k]
            self._settle(host)
        for key, gs in self.groups.items():
            size = sum(quota[id(g)] for g in gs)
            st = self.stores.get(key)
            if due(key, st):
                if st is not None:
                    # The last pass may still be on the GPU holding the old tensors, and they have to be gone before the new ones land or the peak is both.
                    mx.synchronize()
                    for g in gs:
                        g.clear_arena()
                    del self.stores[key], st
                    self._settle(host)
                self.stores[key] = st = ExpertStore(gs, size, self.halves, self.per_half)
                self.intact = False     # MOE_RESIDENT: a rebuild after the fill leaves it empty
                self.rebuilds += 1
                self.build_swing = max(self.build_swing, st.swing)
            elif not rebuild:
                continue
            st.release()
            st.kept = st.written = None     # the quotas move: nothing to put back
            off = 0
            for g in gs:
                g._qslots = list(range(off, off + quota[id(g)]))
                off += quota[id(g)]
                if g._store is st:
                    g._cap = len(g._qslots) + st.per_half
        self.last = {"room_mb": room / 1e6, "pool_target_mb": pool / 1e6,
                     "pool_mb": sum(st.pool * st.slot_bytes
                                    for st in self.stores.values()) / 1e6,
                     "host_mb": host / 1e6, "held_mb": held / 1e6}
        # chunkattn's cap is the MoE output the same pass holds (prefill_cap_bytes's ceiling, T*k*hidden) rather than the live room, which reads <= 0 where the budget is tight and leaves attention unbounded (a toy's first 1000-token prefill ran stock SDPA where the ceiling makes it 2 chunks of 512 rows).
        set_prefill_budget_gb(None, row_elems=self._row_elems)

    def _settle(self, host):
        """Before a store is built in place of a dropped one, wait until the footprint is what the pool was sized from, since MLX lets a buffer go at once but macOS takes its pages back ~0.1 s later and a big pass's freed pages linger the same way (Flash-Next at the elastic store's 7.92 GB max: the old 3.98 GB store still counted 0.1 s after MLX dropped it, the new one's first tensors landed on it and the footprint ended 1.0 GB over)."""
        mx.clear_cache()
        # A 0.5 s bound, so a host term read too low (the footprint never comes down to it) costs a wait, not a hang.
        end = time.perf_counter() + 0.5
        while (phys_footprint() > mx.get_active_memory() + mx.get_cache_memory() + host
               and time.perf_counter() < end):
            time.sleep(0.001)

    def _quotas(self, pool):
        """Slots per layer for `pool` bytes, greedy on the ghost LRU's cumulative hits one extension at a time, since a slot of layer L costs its expert's bytes and each hit saves the same bytes so bytes saved per pool byte is simply hits per slot and a token's k experts arriving together make a quota under k gain nothing alone."""
        gs = self.all
        cost = [self.slot_bytes[g._skey] for g in gs]
        n = [len(g._ghost_hist) for g in gs]
        cap = dict(self.slot_cap)       # slots each store can still take
        q = [0] * len(gs)
        left = pool
        if any(g._ghost_hist.any() for g in gs):
            cum = [np.concatenate(([0], np.cumsum(g._ghost_hist))) for g in gs]

            def best(i):
                m = min(n[i] - q[i], int(left // cost[i]), cap[gs[i]._skey])
                if m < 1:
                    return 0.0, 0
                gain = (cum[i][q[i] + 1:q[i] + m + 1] - cum[i][q[i]]) / np.arange(1, m + 1)
                j = int(np.argmax(gain))
                return float(gain[j]), j + 1

            ext = [best(i) for i in range(len(gs))]
            while True:
                i = max(range(len(gs)), key=lambda i: ext[i][0])
                if ext[i][0] <= 0.0:
                    break
                q[i] += ext[i][1]
                left -= ext[i][1] * cost[i]
                cap[gs[i]._skey] -= ext[i][1]
                # Only the layer that grew, and any whose best step no longer fits, can have a different best step now.
                ext = [best(j) if (j == i or ext[j][1] * cost[j] > left
                                   or ext[j][1] > cap[gs[j]._skey]) else ext[j]
                       for j in range(len(gs))]
        moved = True
        while moved:
            moved = False
            for i in range(len(gs)):
                if q[i] < n[i] and left >= cost[i] and cap[gs[i]._skey] > 0:
                    q[i] += 1
                    left -= cost[i]
                    cap[gs[i]._skey] -= 1
                    moved = True
        return {id(g): k for g, k in zip(gs, q)}

    def store_for(self, glu):
        if glu._skey not in self.stores:
            self.fit(rebuild=False)
        return self.stores[glu._skey]

    def carries(self, n):
        """Whether every layer's quota can hold a pass of `n` positions' experts from one pass to the next, the residency a run-ahead gather bets on, which the ring cannot give where a quota is smaller than a half (2 GB with pool 0: 364 ms a token and 40 rollbacks against 104 layer by layer)."""
        return all(len(g._qslots) >= self.top_k * n for g in self.all)

    def report(self):
        return {"budget_gb": self.peak / 1e9,
                "store_mb": round(sum(st.nbytes for st in self.stores.values()) / 1e6, 1),
                "pool_slots": sum(st.pool for st in self.stores.values()),
                "ring_slots": sum(st.slots - st.pool for st in self.stores.values()),
                "prefill_kb_per_pos": None if self.rate is None else round(self.rate / 1e3, 1),
                "state_mb": None if self.state is None else round(self.state / 1e6, 1),
                "keep_mb": round(self.keep / 1e6, 1),
                "last_prefill": self.pieces,
                "layer_major": self.layer_major,
                "lm_chunk": self.lm_w,
                "attn_b_per_row_key": None if not _PF_ATTN else round(_PF_ATTN[0], 1),
                "last_pass_mb": {"grew": round(self.grew / 1e6, 1),
                                 "attn": round(self.grew_attn / 1e6, 1),
                                 "rows": self.grew_n, "keys": self.grew_keys,
                                 "chunk": self.grew_w},
                "probes": self.probes,
                "rebuilds": self.rebuilds,
                **({"kv_host": kv_host.report()} if _KV_HOST else {}),
                **({"resident": self.resident} if self.resident else {}),
                "lm_line_mb": (None if not self._lm_now() else
                               [round(a / 1e6, 1) for a in self._lm_line()]),
                "lm_seen_mb": {str(k): round(v / 1e6, 1)
                               for k, v in sorted(self._lm_seen.items())},
                "lm_fresh_mb": {str(k): round(v / 1e6, 1)
                                for k, v in sorted(self._lm_fresh.items())},
                **{k: round(v, 1) for k, v in self.last.items()}}


def enable_budget(model, peak_gb, cache_gb=0.0, t_max=1, calib=None):
    """Put `model`'s streaming layers on shared stores sized from a peak budget (MemBudget) and return it, or None on a model with nothing streamed, with `calib` the file it starts from and keeps its measurements in."""
    glus = glus_of(model)
    if not glus:
        note("  memory budget: no streaming layers, nothing to size", flush=True)
        return None
    k = top_k_of(model)
    if not k:
        raise ValueError("--mem-peak-gb needs the model's top_k to size the ring")
    if not _rhs_line()[2]:
        # Budget mode's prefill groups are slices of the store, so their kernel is stock's only where the line is known (_call_grouped).
        raise RuntimeError(f"--mem-peak-gb: MLX {mx.__version__}'s gather_qmm "
                           f"kernel switch is not max(floor, k * experts) any "
                           f"more (stream_fast._rhs_line); the prefill groups "
                           f"cannot be kept on stock's kernel")
    if not phys_footprint():
        # The host term would read 0 and the pool come out too big, silently.
        raise RuntimeError("--mem-peak-gb: proc_pid_rusage gave no "
                           "phys_footprint, so the budget cannot be measured")
    stack = _layer_stack_of(model)
    if stack is None:
        raise ValueError("--mem-peak-gb: no module owning a `layers` list, so "
                         "no pass boundary to size the store at")
    global _GOV
    _GOV = None             # the budget sizes the store; no governor on top
    at = {id(g): i for i, g in enumerate(glus)}
    depth = max((at[id(g._next_glu)] - i for i, g in enumerate(glus)
                 if g._look_m and id(g._next_glu) in at), default=0)
    # Whatever install() already allocated goes: an arena (the Ling bias term builds one, as does check_equivalence's two-row pass through layer 0), _dest rows, and a wide scratch left by any earlier engine in this process.
    for g in glus:
        g.clear_arena()
        for p in g._projs:
            p._bufs.clear()
    _WIDE_SCRATCH.clear()
    if _STAGING_ROWS:
        if _NATIVE is None or _NATIVE.stack is not stack:
            _stage_off("no native agent on this model")
        else:
            # Each layer guesses the next one's experts, in the agent's order.
            mlp_of = {id(_glu_of(l)): _moe_of(l) for l in stack.layers if _moe_of(l) is not None}
            for g, nxt in zip(_NATIVE.glus, _NATIVE.glus[1:]):
                g._stage_next = _stage_guess(mlp_of[id(nxt)], k)
            note(f"  staging rows: {_STAGING_ROWS} (MOE_STAGING={_STAGE}), each layer "
                  f"guesses the next one's experts", flush=True)
    b = MemBudget(glus, peak_gb, cache_gb, k, t_max, depth, moe_row_elems(model),
                  lambda: make_prompt_cache(model), calib)
    # The pass boundary: the layer stack's own call, wrapped once per class and routed by a per-instance mark (install_defer's pattern), so a server swap keeps each model's budget apart.
    cls = type(stack)
    if not getattr(cls, "_budget_wrapped", False):
        orig = cls.__call__

        def stack_call(self, inputs, *args, **kwargs):
            global _MEASURING
            bud = self.__dict__.get("_mem_budget")
            if bud is None:
                return orig(self, inputs, *args, **kwargs)
            emb = kwargs.get("input_embeddings")
            x = inputs if emb is None else emb
            n = x.size if x.ndim < 3 else x.size // x.shape[-1]
            caches = next((a for a in list(args) + list(kwargs.values())
                           if isinstance(a, list)), None) or []
            # Only one sequence's tokens (or embeddings) on a cache are cut, since without one a piece would not attend to the positions before it, and any other array argument keeps the pass whole.
            whole = not caches or x.ndim < 2 or x.shape[0] != 1 or any(
                isinstance(v, mx.array) for v in list(args)
                + [v for key, v in kwargs.items() if key != "input_embeddings"])
            ys, a = [], 0
            bud.lm_ok = not whole
            try:
                while a < n:
                    # KV caches carry the position; the recurrent ones do not.
                    fresh = next((c.offset for c in caches if hasattr(c, "offset")), 0) == 0
                    bud.close()
                    w = n if whole else bud.plan(n - a, fresh, caches)
                    if w is None:
                        # Nothing to predict this prefill from yet (the process's first), so its first min_w positions run once more on a cache of their own and are measured, a throwaway pass that leaves a prefill which then fits running as one pass exactly as stock's.
                        _probe_pass(bud, inputs, emb, a, bud.min_w, self, orig, args, kwargs)
                        if BACKEND == "cuda" and not bud._lm_pair():
                            # A second width, since one pass's growth a position divides the pass's fixed part by its own width, and the two cancel it (_lm_line).
                            _probe_pass(bud, inputs, emb, a,
                                        min(n - a, PROBE_MULT * bud.min_w),
                                        self, orig, args, kwargs)
                        w = bud.plan(n - a, fresh, caches)
                    # Cut here at the stack, so every caller's prefill (a front, a turn, MTP's) is the passes a caller-chunked one would be, and the rest plans again at its own entry.
                    part, kw = (inputs, {}) if w == n else _cut(inputs, emb, a, a + w)
                    bud.tick(w, fresh, caches)
                    ys.append(orig(self, part, *args, **{**kwargs, **kw}))
                    a += w
                    if _R3 and a < n:
                        # R3: a lazy output held the last layer's hidden state ([w, 4 x 2560] on Flash-Next, ~20 KB a position) until the concatenation below, and evaluated it holds its own rows only (5 KB).
                        mx.eval(ys[-1])
            finally:
                # chunkattn reads it only inside a pass, since held past it, it kept a swapped-out engine (and its model) alive and a swapped-in model's attention read the old budget.
                _MEASURING = None
                _lm_close()
            return ys[0] if len(ys) == 1 else mx.concatenate(ys, axis=1)

        cls.__call__ = stack_call
        cls._budget_wrapped = True
    stack.__dict__["_mem_budget"] = b
    # MOE_RESIDENT keeps the store through a prefill, and a layer-major pass reads into the whole store.
    b.layer_major = lm_supported(model) and not _RESIDENT
    note(f"  memory budget: peak {peak_gb:.2f} GB | shared stores for "
          f"{len(b.groups)} expert shape(s) | ring {b.halves}x{b.per_half} "
          f"slots each = {b.ring_bytes / 1e6:.1f} MB | prefills of {b.min_w}+ "
          f"positions cut to fit, "
          + ("measured before: no probe" if b.rate is not None and b.state is not None
             else "the first one probed")
          + (f"; prefill layer-major, {LM_W}-position chunks, experts read once a pass"
             if b.layer_major else ""), flush=True)
    return b


def _probe_pass(bud, inputs, emb, a, m, stack, orig, args, kwargs):
    """One throwaway pass of `m` positions from `a` on a cache of its own, measured by the budget, since a prefill the plan cannot predict yet needs a width and its growth."""
    bud.probes += 1
    head, kw = _cut(inputs, emb, a, a + m)
    cache = bud._make_cache()
    bud.tick(m, True, cache)
    orig(stack, head, cache=cache, **kw)
    bud.close()


def _copy(caches):
    """What the KV caches among `caches` add to a pass beyond its positions' own bytes (MemBudget's copy), since mlx_lm grows a KV cache a step at a time by concatenating onto a copy that keeps the old buffer alive, so a pass can hold the largest one twice plus a step of every one (a toy decode crossing a step grew 1.58 MB where the largest cache alone was 0.79 MB)."""
    kv = [c for c in caches if hasattr(c, "offset") and c.nbytes]
    # One step's bytes: a cache shorter than a step holds exactly one, since per used position the full step read 6x too big on a toy.
    return float(max((c.nbytes for c in kv), default=0)
                 + sum(c.step * c.nbytes / max(c.offset, c.step, 1)
                       for c in kv if getattr(c, "step", 0)))


def _eval_caches(caches):
    """Evaluate every array `caches` hold, a no-op once they are, since a front restored from SSD is lazy until then (MemBudget's `held`)."""
    mx.eval([a for c in caches
             for a in (c.cache if isinstance(c, ArraysCache)
                       else kv_host.pass_arrays(c) if getattr(c, "kv_host", False)
                       else (getattr(c, "keys", None), getattr(c, "values", None)))
             if a is not None])


def cache_state(c):
    """What a prefill evaluates of cache c to force its pass: its state, or a homed cache's own arrays, whose state would stage every row back to the GPU (kv_host)."""
    return kv_host.pass_arrays(c) if getattr(c, "kv_host", False) else c.state


def _keys(caches):
    """Positions `caches` already hold, the key count a pass's attention reads past its own rows (a recurrent cache has no offset and adds none)."""
    return max((c.offset for c in caches if hasattr(c, "offset")), default=0)


def trim_kv(cache):
    """Trim each KV cache of a freshly prefilled `cache` to its length one at a time, since mlx_lm's save_prompt_cache copies every slice at once (1.31 MB of copies for 1.23 MB of KV, ~113 MB on a 5.5k-token front) while trimmed first it copies nothing and the same values keep the same replies."""
    for c in cache:
        if (isinstance(c, KVCache) and not getattr(c, "gap", 0) and not getattr(c, "kv_host", False)
                and c.keys is not None and c.offset < c.keys.shape[2]):
            c.keys, c.values = (mx.contiguous(a[..., :c.offset, :])
                                for a in (c.keys, c.values))
            mx.eval(c.keys, c.values)


def _cut(inputs, emb, a, b):
    """Positions [a, b) of a layer stack's input: the token ids, and the embeddings a VL model passes in their place as call kwargs."""
    return (None if inputs is None else inputs[:, a:b],
            {} if emb is None else {"input_embeddings": emb[:, a:b]})


def _module_items(mod):
    """The child modules of `mod`, one level down with containers unwrapped, since nn.Module is itself a dict and a child module must be recognised BEFORE the container test or it is walked as a dict of its own attributes."""
    children = getattr(mod, "children", None)
    if not callable(children):
        return
    for c in children().values():
        if callable(getattr(c, "children", None)):
            yield c
        elif isinstance(c, (list, tuple)):
            yield from (x for x in c if callable(getattr(x, "children", None)))
        elif isinstance(c, dict):
            yield from (x for x in c.values() if callable(getattr(x, "children", None)))


_MOE_ATTRS = ("mlp", "block_sparse_moe")


def _moe_attr(layer):
    """Name of the layer's MoE block (`mlp` on Qwen, DeepSeek, GLM and Ling, `block_sparse_moe` on MiniMax and Mixtral), None on a dense layer, looking through a CacheBiasedMoE wrapper that keeps the block under `.block`."""
    for a in _MOE_ATTRS:
        m = getattr(layer, a, None)
        if hasattr(getattr(m, "block", m), "switch_mlp"):
            return a
    return None


def _moe_of(layer):
    a = _moe_attr(layer)
    return getattr(layer, a) if a else None


def glus_of(model):
    """Every streaming GLU in the model, whatever the mlp is wrapped in, walking the tree explicitly because the layer list may nest under language_model.model.layers and `layer.mlp.switch_mlp` is only the shape at cache_bias 0."""
    out, seen = [], set()

    def walk(mod, depth=0):
        if depth > 12 or id(mod) in seen:
            return
        seen.add(id(mod))
        layers = getattr(mod, "layers", None)
        if isinstance(layers, (list, tuple)):
            for layer in layers:
                mlp = _moe_of(layer)
                mlp = getattr(mlp, "block", mlp)      # unwrap CacheBiasedMoE
                glu = getattr(mlp, "switch_mlp", None)
                if isinstance(glu, StreamingSwitchGLU) and id(glu) not in seen:
                    seen.add(id(glu))
                    out.append(glu)
        for c in _module_items(mod):
            walk(c, depth + 1)

    walk(model)
    return out


def n_experts_of(model):
    """Experts per MoE layer, the largest arena that means anything since holding every expert is the end of streaming, or None when the model does not say rather than a number that happens to fit one architecture."""
    # From the packing metadata, not the GLU's own _n_exp, which stays 0 until the first decode builds the arena and would read as unknown at construction time, as eprofile.py takes the same number from the same place.
    for g in glus_of(model):
        for p in getattr(g, "_projs", ()):
            n = (p._packed or {}).get("n_experts") if p._packed else None
            if n:
                return int(n)
    return None


def top_k_of(model):
    """How many experts one token activates per layer, or None, an estimate only since the governor's real floor is the high-water mark the layers report."""
    seen = set()

    def walk(m, depth=0):
        if id(m) in seen or depth > 6:
            return None
        seen.add(id(m))
        inner = getattr(m, "block", m)
        if isinstance(getattr(inner, "switch_mlp", None), StreamingSwitchGLU):
            for owner in (m, inner):
                k = (getattr(owner, "k", None) or getattr(owner, "top_k", None)
                     or getattr(owner, "num_experts_per_tok", None))
                if k:
                    return int(k)
        for c in _module_items(m):
            k = walk(c, depth + 1)
            if k:
                return k
        return None

    return walk(model)


# Arenas live outside the module tree, keyed by an _ArenaKey, because held as an attribute they were walked by nn.Module.parameters(), counted as weights (a 20475 MB report for a 0.82 GB model) and wired to their limit by mlx_lm, while this byte cache written by preadv is never a parameter.
class _ArenaKey:
    """Identity token for one layer's arena, keyed from a module-level weak table rather than id(self), which CPython reuses across short-lived models and would let two share a dead arena's buffers across incompatible quantizations."""


_ARENAS: "weakref.WeakKeyDictionary[_ArenaKey, list]" = weakref.WeakKeyDictionary()

# Timers incremented inside another timer's window, so they must not be added to the total a second time (sync/numpy live inside decode, wait_pred inside fetch).
_NESTED_TIMERS = ("sync", "numpy", "wait_pred")

# Cache-conditional routing pulls the selection toward the resident set, so a strong bias at 5.9% coverage can become "only ever use these 20", which the count of distinct experts actually used shows and six prompts would not, keyed by _ArenaKey so the entries die with their layer.
_DEFER = False


# Verify the re-run of a repaired layer (its flag is read back and a slot-table or LRU drift raises), on by default, with MOE_DEFER_CHECK=0 and MOE_CLAIMED_WAIT=0 the measured 1.05x path that trades the drift net for the speed.
_DEFER_CHECK = os.environ.get("MOE_DEFER_CHECK", "1") == "1"
# Widest pass (positions) the run-ahead takes: a speculative round verifies k+gamma positions in one pass, and the deferred GLU path exists only while the GLU does not sort (indices.size < 64), so install_defer sets this to 63 // top_k.
_DEFER_MAX_T = 1
# Wait for the GPU on the re-run of a repaired layer before rebuilding the layer after it: pure host scheduling with the check off, not a lever on its own (0.997x), but 1.05x bit-exact together with MOE_DEFER_CHECK=0.
CLAIMED_WAIT = os.environ.get("MOE_CLAIMED_WAIT", "1") == "1"


# MOE_NATIVE=1: one-position decode through the native expert agent (native/moe_native.cpp, and native/moe_pipe_cuda on CUDA), off by default, with CPU turning the request down at install.
_NATIVE_ASKED = os.environ.get("MOE_NATIVE", "0") == "1"
_NATIVE_ENV = _NATIVE_ASKED and BACKEND in ("metal", "cuda")
_NATIVE = None          # the _Native of the installed model, when on
# MOE_STAGING (native passes on one budget store): "1" reads each layer's guess of the next layer's experts into 16 staging rows the agent promotes when they are claimed, "dry" hands the guesses over and charges the rows but reads none, and it is on by default under MOE_NATIVE=1.
_STAGE = os.environ.get("MOE_STAGING", "1") if _NATIVE_ENV and BACKEND == "metal" else "0"
_STAGING_ROWS = 16 if _STAGE in ("1", "dry") else 0
# MOE_PIPE_STAGING (CUDA native passes, on by default since 2026-11-06, =0 turns it off): each layer posts the next layer's gate top-k on its own MoE input beside its job, and the agent reads the guessed misses into host guess records, MOE_PIPE_STAGING_CAP of them in flight (default 2).
_PIPE_STAGING = _NATIVE_ENV and BACKEND == "cuda" and os.environ.get("MOE_PIPE_STAGING", "1") == "1"
# MOE_PIPE_GUESS_D (staging, default 1): the guess posted beside layer L's job names layer L + D's experts (L + D's gate on L's MoE input).
_PIPE_GUESS_D = int(os.environ.get("MOE_PIPE_GUESS_D", "1"))
# MOE_PIPE_HITS_FIRST=1 (CUDA native passes on an EXL3 pack, off by default): the layer's held experts run between its job's post and its wait, the missed ones after it, each slot's output the same bits.
_PIPE_HITS = _NATIVE_ENV and BACKEND == "cuda" and os.environ.get("MOE_PIPE_HITS_FIRST", "0") == "1"
_HITS_SKIP = None        # its rows' "not held" mark as a uint32 array, made once at install
# MOE_CPU_GUESS (staging rows, on by default, 0 off): the agent ranks each next layer's 8-bit router on this layer's MoE input on a CPU core instead of the GPU's guess kernels, and "check" keeps the GPU's guess and has the agent compare its own with it.
_CPU_GUESS = os.environ.get("MOE_CPU_GUESS", "1")
# MOE_NATIVE_MTP=1 (Metal, no staging rows): a pass of T positions, an MTP verify pass among them, goes through the agent too when every layer's slots and one agent job (64 ids, moe_native.cpp's push) hold its ids; off keeps every pass wider than one on the Python path.
_NATIVE_ROWS = os.environ.get("MOE_NATIVE_MTP", "0") == "1"
_NATIVE_MISSING_SAID = False


def native_default(defer=False, arena=True):
    """True when a run on the adaptive arena (no --mem-peak-gb) on Metal takes the native agent and its T-row MTP verify without MOE_NATIVE set, as the launcher's arena recipe does, else None: an explicit MOE_NATIVE or MOE_NATIVE_MTP wins, and a missing or stale moe_native or a path install() refuses beside it keeps Python."""
    global _NATIVE_ROWS, _NATIVE_MISSING_SAID
    _NATIVE_ROWS = os.environ.get("MOE_NATIVE_MTP", "0") == "1"    # each model load decides again
    if (not arena or "MOE_NATIVE" in os.environ or BACKEND != "metal" or defer is True
            or fused_router.on()):
        return None
    sys.path.append(str(Path(__file__).parent / "native" / "build"))
    try:
        import moe_native
    except ImportError:
        if not _NATIVE_MISSING_SAID:       # said once: without it the SSD reads run about 2x slower
            _NATIVE_MISSING_SAID = True
            print(f"  the native reader is not built for this Python, so expert reads take the slower Python path; "
                  f"build it with: native/build.sh {sys.executable}", file=sys.stderr, flush=True)
        return None
    if not hasattr(moe_native, "release"):
        return None
    _NATIVE_ROWS = os.environ.get("MOE_NATIVE_MTP", "1") == "1"
    return True


class _Native:
    """The native agent behind one model's T=1 decode passes (and T-row ones under MOE_NATIVE_MTP), where every MoE layer hands its router ids to a host thread on the GPU's own handshake (signal_after), the thread claims slots with this file's LRU, reads the misses into the arena and lets the gather run (wait_for), every other pass stays on the Python path, and the LRU is held either by Python's `_lru` or by the agent's table with the arenas' lifecycle going through release()."""

    def __init__(self, nv, model, glus, readers):
        fds = {p._packed["fd"] for g in glus for p in g._projs}
        assert len(fds) == 1, "native: one experts.bin per model"
        self.nv, self.glus = nv, glus
        self.stack = _layer_stack_of(model)
        self.ev = nv.SharedEvent()
        # CUDA: the handshake kernels run once before the agent exists, and the first decode pass after install runs on the Python path, so no kernel loads for the first time while a wait is pending.
        self.cuda = BACKEND == "cuda"
        self.warm, self.holds, self.hold_s = not self.cuda, 0, 0.0
        if self.cuda and top_k_of(model):
            nv.warm(top_k_of(model))
        self.pipe_stage = _PIPE_STAGING
        if self.pipe_stage:
            # Every layer but the last D guesses layer L + D's experts, the post kernel warmed before the agent exists as warm()'s are.
            mlp_of = {id(_glu_of(l)): _moe_of(l) for l in self.stack.layers if _moe_of(l) is not None}
            for g, nxt in zip(glus, glus[_PIPE_GUESS_D:]):
                m = mlp_of[id(nxt)]
                m = m.block if isinstance(m, CacheBiasedMoE) else m
                g._guess_next = _pipe_guess(m.gate, m.top_k)
                nv.warm_guess(m.gate.weight.shape[0], m.top_k)
        if _PIPE_HITS and top_k_of(model):
            # The held experts' chain needs the EXL3 kernels' skipped slots; another pack keeps one chain after the wait.
            for g in glus:
                g._hits_first = g._exl3 is not None
            nv.warm_resident(top_k_of(model))
            global _HITS_SKIP
            _HITS_SKIP = mx.array(nv.SKIP, dtype=mx.uint32)
            mx.eval(_HITS_SKIP)
            note(f"  hits first: {sum(g._hits_first for g in glus)} of {len(glus)} layers (MOE_PIPE_HITS_FIRST=1)", flush=True)
        self.agent = nv.Agent(self.ev, fds.pop(), len(glus), readers)
        # The victim tier (CUDA): its slots, a slot's bytes, and the governor make_store_governor puts on it.
        self.tier_n = self.tier_slot = self.tier_rss0 = 0
        self.tier_gov = None
        if self.cuda:
            # The read path: O_DIRECT reads, and the store's evictions kept in a tier of host RAM sized now.
            slot = max(sum(p._packed["stride"] for p in g._projs) for g in glus)
            n, avail = nv.tier_slots(slot, sum(g._projs[0]._packed["n_experts"] for g in glus))
            if n:
                self.agent.set_tier(slot, n)
                self.tier_n, self.tier_slot, self.tier_rss0 = n, slot, _rss_anon()
            note(f"  native reads: O_DIRECT {'on' if self.agent.direct else 'off'}, victim tier "
                  f"{n * slot / 1e9:.2f} GB ({n} experts; MemAvailable {avail / 1e9:.2f} GB)", flush=True)
            if self.pipe_stage:
                cap = int(os.environ.get("MOE_PIPE_STAGING_CAP", "2"))
                self.agent.set_staging(cap, _PIPE_GUESS_D)
                note(f"  staging: each layer guesses layer L + {_PIPE_GUESS_D}'s experts, {cap} guess records in flight "
                      f"(MOE_PIPE_STAGING=1)", flush=True)
        self.stripe = _STRIPE is not None and hasattr(self.agent, "set_mirror")
        if self.stripe:
            st = _STRIPE
            self.agent.set_mirror(st.fd, [(m, e, n) for m, (e, n) in st.at.items()],
                                  st.bw("int"), st.bw("ext"), st.readers)
        elif _STRIPE is not None:
            print("  stripe: this moe_native has no set_mirror; native passes read "
                  "experts.bin only", flush=True)
        self.gens = [None] * len(glus)  # arena generation (+ quota) each layer was handed
        self.rings = []                 # (store, half) per agent ring, as last handed over
        self.owner = "py"
        self.active = False
        self.v = 1                      # the GPU raises the word to v, the agent to v + 1
        self.k = top_k_of(model)
        self.stage = _STAGE == "1"      # staging rows (set_staging at begin)
        self.cpu = False                # MOE_CPU_GUESS's routers handed over (_cpu_routers)
        for i, g in enumerate(glus):
            g._native_idx = i

    def ready(self):
        return all(g._akey in _ARENAS and g._slot_np is not None for g in self.glus)

    def fits(self, T):
        """A pass of T positions can go native: one, or under MOE_NATIVE_MTP T rows whose T x top_k ids one job carries and every layer's slots hold (the Python path's own arena test), with no staging rows."""
        # Metal without staging only (the 35B's arena); T rows on CUDA or with staging guesses need the multi-row branch's warm-up and rank-major guesses.
        return T == 1 or (_NATIVE_ROWS and not self.cuda and not self.stage and T * self.k <= 64
                          and all(g._cap >= T * self.k for g in self.glus))

    @staticmethod
    def _gen(g):
        # A store layer's quota is a row range (MemBudget._fit), kept with the generation so a range moved without a release is handed over too.
        q = g._qslots if g._store is not None else ()
        return (g._arena_gen, len(q), q[0] if q else -1)

    def begin(self, T=1):
        """Token boundary of a native pass of T positions with the token before it done, False sending the pass to Python (an arena missing, or just resized)."""
        self.drain()
        if _GOV is not None:
            for _ in self.glus:          # what the sync path ticks per layer
                _GOV.tick(False, T)
        if self.tier_gov is not None:
            self.tier_tick(T)
        if not self.ready():
            return False
        rings = []
        for g in self.glus:
            if g._store is not None and (g._store, g._half) not in rings:
                rings.append((g._store, g._half))
        for i, g in enumerate(self.glus):
            gen = self._gen(g)
            if self.gens[i] != gen:
                assert self.owner == "py", "native: an arena changed under the agent"
                if self.cuda:
                    # The slot table and each part's rows as the numpy views (managed memory) the agent writes and copies into.
                    projs = [(off, stride, iov) for _, iov, _, off, stride in _ARENAS[g._akey]]
                    slot = g._slot_np
                else:
                    projs = [(off, stride, list(bufs)) for bufs, _, _, off, stride in _ARENAS[g._akey]]
                    mx.eval([b for _, _, bufs in projs for b in bufs])   # the bf16 views are lazy
                    slot = g._slot_of
                if g._store is None:
                    self.agent.set_layer(i, slot, projs)
                else:
                    # phys only with staging: an older moe_native takes five.
                    extra = () if g._phys_of is None else (g._phys_of,)
                    self.agent.set_layer(i, slot, projs, list(g._qslots),
                                         rings.index((g._store, g._half)), *extra)
                self.gens[i] = gen
        if self.owner == "py":
            for i, g in enumerate(self.glus):
                self.agent.set_state(i, [(s, e) for e, s in g._lru.items()],
                                     [(s, e) for e, s in g._ring.items()])
            at = {id(g): i for i, g in enumerate(self.glus)}
            for r, (st, h) in enumerate(rings):
                base = st.pool + h * st.per_half
                self.agent.set_ring(r, base, st.next[h], [
                    (-1, -1) if o is None else (at.get(id(o[0]), -1), o[1])
                    for o in st.owner[base:base + st.per_half]])
            self.rings = rings
            if self.stage:
                stores = {id(g._store) for g in self.glus}
                if len(stores) != 1 or None in {g._store for g in self.glus}:
                    # install() and enable_budget() decide and this is the guard: the agent stops promoting and the layers stop guessing.
                    note("  staging rows: off, the layers are not on one budget store",
                          flush=True)
                    self.stage = False
                    self.agent.set_staging(0, 0)
                    for g in self.glus:
                        g._stage_next = None
                else:
                    self.agent.set_staging(self.glus[0]._store.slots, _STAGING_ROWS)
                    if _CPU_GUESS != "0" and not self.cpu:
                        self._cpu_routers()
            self.owner = "agent"
        for g in self.glus:
            g._want = max(g._want, self.k)
            if g._bias_hook is not None:
                g._bias_hook.refresh()
        self._stats()
        if self.cuda:
            self.agent.arm()        # it polls the GPU's word: spin for this pass's jobs
        return True

    def _cpu_routers(self):
        """MOE_CPU_GUESS: every next layer's 8-bit affine router handed to the agent once, a layer whose router it cannot take keeping the GPU's guess."""
        self.cpu = True
        if not hasattr(self.agent, "set_router"):
            note(f"  staging guess: this moe_native has no CPU guess (native/build.sh {sys.executable})", flush=True)
            return
        mlp_of = {id(_glu_of(l)): _moe_of(l) for l in self.stack.layers if _moe_of(l) is not None}
        n = 0
        for i, g in enumerate(self.glus[1:], 1):
            mlp = mlp_of.get(id(g))
            gate = getattr(mlp.block if isinstance(mlp, CacheBiasedMoE) else mlp, "gate", None)
            if (getattr(gate, "bits", None) != 8 or getattr(gate, "mode", "affine") != "affine"
                    or "bias" in gate or self.glus[i - 1]._stage_next is None):
                continue
            mx.eval(gate.weight, gate.scales, gate.biases)
            self.agent.set_router(i, gate.weight, gate.scales, gate.biases, gate.group_size, self.k)
            self.glus[i - 1].__dict__["_stage_cpu"] = mlp     # not a child module
            n += 1
        note(f"  staging guess: {n} of {len(self.glus) - 1} layers ranked on a CPU core "
             f"(MOE_CPU_GUESS={_CPU_GUESS})", flush=True)

    def to_python(self):
        """The LRU and the demand counters back to Python (what _claimed keeps on the sync path)."""
        if self.owner != "agent":
            return
        self.drain()
        if self.stage:
            t = time.perf_counter()
            # The agent is idle, but the last pass's last gathers may still read store rows on the GPU (the next pass is built before the last one ends), so they finish before rows move.
            mx.synchronize()
            self.agent.unstage()
            TIMERS["stage_unstage"] += time.perf_counter() - t
        for i, g in enumerate(self.glus):
            if self.gens[i] is None:
                continue
            g._lru.clear()
            g._lru.update((e, s) for s, e in self.agent.lru(i))
            seen = np.frombuffer(self.agent.take_seen(i), dtype=np.uint32)
            if g._seen is not None:
                g._seen += seen
            if g._store is not None:
                ids, lens = (np.frombuffer(b, dtype=np.int32) for b in self.agent.take_log(i))
                a = 0
                for n in lens.tolist():
                    g._ghost(ids[a:a + n])
                    a += n
                g._store.dirty = True       # the agent published; release must unpublish
                g._ring.clear()
        # The ring halves back, and each layer's `_ring` holds the members whose row still maps to them (ExpertStore.ring_slot's pop).
        for r, (st, h) in enumerate(self.rings):
            base = st.pool + h * st.per_half
            st.next[h], owners = self.agent.ring_state(r)
            for j, (l, e) in enumerate(owners):
                g = self.glus[l] if l >= 0 else None
                st.owner[base + j] = None if g is None else (g, e)
                if g is not None and self.gens[l] is not None and g._slot_np is not None \
                        and g._slot_np[e] == base + j:
                    g._ring[e] = base + j
        # Dropped here: held past the hand-back, a store the budget then rebuilt stayed alive beside its successor at +2.2 GB with pool 0.
        self.rings = []
        self._stats()
        self.owner = "py"

    def release(self):
        """Before an arena is built, dropped or parked: the agent lets go of every arena, so the Python handles are the last ones."""
        self.to_python()
        for i, gen in enumerate(self.gens):
            if gen is not None:
                self.agent.clear_layer(i)
                self.gens[i] = None

    def drain(self):
        if not self.agent.drain(30.0):
            # A job the GPU never reached or a read that hangs lets every wait through (a raised word can be lowered again by a later signal on the GPU, the stop word cannot) before it stops.
            self.nv.release()
            self.fail("the agent did not go idle in 30 s")
        if self.agent.error:
            self.fail(self.agent.error)
        # Checked at every token boundary: the token before was computed (stack_call evaluated it) but not yet handed out.
        released, gave_up = self.nv.stopped()
        if gave_up:
            self.fail(f"a GPU wait gave up after {self.nv.wait_bound()[2]:g} s unanswered "
                      f"(value {gave_up}), so that pass gathered stale slots")
        if released:
            self.fail("the GPU waits were released (an exit signal or hook)")

    def fail(self, why):
        """The agent answered every job so the GPU is free but the arena may hold wrong bytes: drop every arena and decode on the Python path."""
        global _NATIVE
        _NATIVE = None
        for g in self.glus:
            g.clear_arena()
        raise RuntimeError(f"native agent: {why}; decoding on the Python path from here on")

    def _stats(self):
        st = self.agent.stats()
        n, hits, miss, miss_layers, park, read = st[:6]
        if len(st) > 6:             # CUDA's read path
            STATS["tier_hits"] += st[6]
            STATS["tier_puts"] += st[7]
            STATS["dio_fell"] += st[8]
        STATS["reads"] += n
        STATS["hits"] += 3 * hits
        STATS["rows_read"] += 3 * miss
        STATS["rows_block"] += 3 * miss
        STATS["need_block"] += 3 * (hits + miss)
        STATS["miss_layers"] += miss_layers
        TIMERS["native_park"] += park / 1e9
        TIMERS["native_read"] += read / 1e9
        if self.stripe:
            ib, eb, ins, ens, fell = self.agent.mirror_stats()
            STATS["stripe_int_b"] += ib
            STATS["stripe_ext_b"] += eb
            STATS["stripe_fell"] += fell
            TIMERS["stripe_int"] += ins / 1e9
            TIMERS["stripe_ext"] += ens / 1e9
            if fell and _STRIPE is not None:
                _STRIPE.dead = True         # the agent said so on stderr
        if self.stage or self.pipe_stage:
            g, rec, prom, part, wrong, wait, drop, back, bad = self.agent.stage_stats()[:9]
            for k, v in (("guess", g), ("rec", rec), ("prom", prom), ("part", part),
                         ("wrong", wrong), ("drop", drop), ("unstaged", back), ("bad", bad)):
                STATS["stage_" + k] += v
            TIMERS["stage_wait"] += wait / 1e9
        if self.cpu and hasattr(self.agent, "cpu_guess_stats"):
            n, ns, top, skip, cmp, same, sets = self.agent.cpu_guess_stats()
            for k, v in (("n", n), ("skip", skip), ("cmp", cmp), ("same", same), ("set", sets)):
                STATS["cg_" + k] += v
            STATS["cg_max_us"] = max(STATS["cg_max_us"], top // 1000)
            TIMERS["cpu_guess"] += ns / 1e9

    def tier_tick(self, T=1, idle=False):
        """TierGovernor's reading at a token boundary (the agent drained) or an idle second, the tier resized when it moves."""
        want = self.tier_gov.tick(T, idle)
        if want is None:
            return
        if idle:
            self.drain()
        mx.synchronize()            # no copy of the last pass reads or writes the tier any more
        t, old = time.perf_counter(), self.tier_gov.n
        self.tier_gov.n = self.agent.set_tier(self.tier_slot, want)
        if self.tier_gov.n < old and self.glus[0]._budget is not None:
            self.glus[0]._budget.intact = False     # MOE_RESIDENT: the dropped experts are no longer few
        print(f"  victim tier: {old * self.tier_slot / 1e9:.2f} -> {self.tier_gov.n * self.tier_slot / 1e9:.2f} GB "
              f"in {time.perf_counter() - t:.2f} s, MemAvailable {self.tier_gov.avail / 1e9:.2f} GB "
              f"(t {time.time():.2f})", flush=True)


def native_sync():
    """Hand the LRU and the demand counter `_seen` back to Python."""
    if _NATIVE is not None:
        _NATIVE.to_python()


def _native_exit():
    """The process is going, so every native GPU wait falls through and this thread's GPU work finishes, and nothing spins on a word the agent will not raise once it is gone."""
    nv = sys.modules.get("moe_native") or sys.modules.get("moe_pipe_cuda")
    if nv is None:
        return
    nv.release()
    try:
        mx.synchronize()
    except Exception:
        pass                    # exiting anyway; the waits are released


def _hook_native_exit():
    """Exits Python sees: a normal one (also after a KeyboardInterrupt) and os._exit, while SIGTERM and SIGHUP are moe_native's own handler and SIGKILL only the GPU wait's own bound."""
    global _EXIT_HOOKED
    if _EXIT_HOOKED:
        return
    atexit.register(_native_exit)
    real_exit = os._exit

    def _exit(code):
        _native_exit()
        real_exit(code)
    # Wrapped here, not in each guard, since a guard that binds os._exit before this runs keeps the plain one and the bound still holds.
    os._exit = _exit
    _EXIT_HOOKED = True


_EXIT_HOOKED = False


def install_native(model, glus, readers):
    """The token boundary for native passes, on the class owning the layer list and routed by the instance as install_defer is."""
    global _NATIVE
    if BACKEND == "cuda":
        # MLX's CUDA backend: a ctypes library g++ builds at first use.
        sys.path.append(str(Path(__file__).parent / "native"))
        import moe_pipe_cuda as moe_native
    else:
        # The one native/build.sh put into this venv first, then a manual cmake build in native/build (native/README.md).
        sys.path.append(str(Path(__file__).parent / "native" / "build"))
        try:
            import moe_native
        except ImportError as e:
            raise ImportError(f"MOE_NATIVE=1 needs the extension built for this venv: "
                              f"native/build.sh {sys.executable} ({e})") from e
    if not hasattr(moe_native, "release"):
        # Its waits spin with no bound: killed mid-pass, it hangs the GPU.
        raise ImportError(f"{moe_native.__file__} predates the bounded GPU wait; "
                          f"rebuild it: native/build.sh {sys.executable}")
    _hook_native_exit()
    scls = type(_layer_stack_of(model))
    if getattr(scls, "_budget_wrapped", False) and not getattr(scls, "_native_wrapped", False):
        # The budget's pass boundary (a refit, a release of every quota) has to run outside before a native pass takes the slot tables over, so install() wraps native first and enable_budget after.
        raise RuntimeError("MOE_NATIVE=1: the memory budget already wraps this layer "
                           "stack; the native wrapper must go inside it")
    _NATIVE = _Native(moe_native, model, glus, readers)
    cap, rate, budget = moe_native.wait_bound()
    note(f"  native: a GPU wait gives up after {budget:g} s "
          + ("(%globaltimer)" if BACKEND == "cuda" else
             f"({cap} spin rounds at {rate / 1e6:.1f} M/s)"), flush=True)
    if getattr(scls, "_native_wrapped", False):
        return
    sorig = scls.__call__

    def stack_call(self, inputs, *args, **kwargs):
        nat = _NATIVE
        if nat is None or nat.stack is not self:
            return sorig(self, inputs, *args, **kwargs)
        T = inputs.shape[1] if inputs.ndim == 2 and inputs.shape[0] == 1 else 0
        if T and kwargs.get("input_embeddings") is None and nat.ready() and nat.fits(T):
            if not nat.warm:
                nat.warm = True         # CUDA: this pass on the Python path loads every kernel
                nat.to_python()
                return sorig(self, inputs, *args, **kwargs)
            # The token before must be done, since its claims are what this one's cache bias ranks against and the agent's table is read here.
            mx.eval(inputs)
            if nat.begin(T):
                STATS["native_rows"] += T > 1
                nat.active = True
                ple, ok = None, False
                try:
                    if T == 1:
                        ple = _ple_decode_ahead(self, inputs, args[0] if args else kwargs.get("cache"))
                    out = sorig(self, inputs, *args, **kwargs)
                    ok = True
                    return out
                finally:
                    nat.active = False
                    _ple_decode_done(self, ple, ok)
        nat.to_python()
        return sorig(self, inputs, *args, **kwargs)

    scls.__call__ = stack_call
    scls._native_wrapped = True


def set_defer_check(on):
    global _DEFER_CHECK
    _DEFER_CHECK = bool(on)


# True only while a deferred forward runs: the GLU takes its unchecked GPU path on THIS flag and never on _DEFER alone, since with the loop not engaged (a call shape it declines) the same path ran all 40 layers without a check at 49 tok/s of garbage.
_DEFER_ACTIVE = False
_DEFER_REC = []
# mx.async_eval hands a layer's work to the GPU early so the SSD and the GPU overlap, and a backward pass turns it off because MLX forbids it inside a graph transformation.
NO_ASYNC = False
# Set by a profiling script to count the distinct experts the model actually uses.
_TRACK_USED = False
# Set at install so the tuner can report diversity as a fraction of the total.
_EXPERTS_PER_LAYER: list = []
_USED: "weakref.WeakKeyDictionary[_ArenaKey, set]" = weakref.WeakKeyDictionary()

_POOL = ThreadPoolExecutor(max_workers=8)


def set_threads(n):
    global _POOL
    if n != _POOL._max_workers:
        _POOL.shutdown(wait=True)
        _POOL = ThreadPoolExecutor(max_workers=n)

_ST_NP = {"U8": np.uint8, "U16": np.uint16, "U32": np.uint32, "I32": np.int32,
          "F16": np.float16, "F32": np.float32, "BF16": np.uint16}
_ST_MX = {"U8": mx.uint8, "U16": mx.uint16, "U32": mx.uint32, "I32": mx.int32,
          "F16": mx.float16, "F32": mx.float32, "BF16": mx.bfloat16}

_MX_OF = {np.dtype("uint32"): mx.uint32, np.dtype("uint16"): mx.uint16,
          np.dtype("uint8"): mx.uint8, np.dtype("float16"): mx.float16,
          np.dtype("float32"): mx.float32, np.dtype("int32"): mx.int32}


def build_shard_map(model_path):
    """{tensor name: shard path} of a checkpoint, from model.safetensors.index.json's weight_map when there is one and else every *.safetensors file's own header."""
    p = Path(model_path)
    idx = p / "model.safetensors.index.json"
    if idx.exists():
        return {k: str(p / f) for k, f in json.loads(idx.read_text())["weight_map"].items()}
    out = {}
    for f in sorted(p.glob("*.safetensors")):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            out.update((k, str(f)) for k in json.loads(fh.read(n)) if k != "__metadata__")
    return out


class SafetensorsMap:
    """Where each tensor of some safetensors files lives, for byte-range reads: an 8-byte little-endian header length, a JSON header of dtype, shape and [begin, end) offsets past the header, then the bytes, with each _index entry the path, both dtypes, the shape, the file offset and the length."""
    # BF16 travels as uint16 and is viewed as bfloat16 once in MLX.
    _NP = {**_ST_NP, "I8": np.int8, "I16": np.int16, "I64": np.int64, "U64": np.uint64,
           "F64": np.float64, "BOOL": np.bool_}
    _MX = {**_ST_MX, "I8": mx.int8, "I16": mx.int16, "I64": mx.int64, "U64": mx.uint64,
           "F64": mx.float64, "BOOL": mx.bool_}

    def __init__(self, paths):
        self._fds, self._index, self.packed = {}, {}, None
        for path in paths:
            if path in self._fds:
                continue
            f = open(path, "rb")
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
            self._fds[path] = f
            for name, m in header.items():
                if name == "__metadata__":
                    continue
                lo, hi = m["data_offsets"]
                self._index[name] = (path, self._NP.get(m["dtype"]), self._MX.get(m["dtype"]),
                                     m["shape"], 8 + n + lo, hi - lo)

    def __contains__(self, name):
        return name in self._index

    def close(self):
        for f in self._fds.values():
            f.close()


def find_switch_mlp(layer, i):
    """(routed-experts module, its checkpoint key) of decoder layer i, or (None, None) for a layer without one, since mlx_lm puts the stacked experts at `mlp.switch_mlp` or `block_sparse_moe.switch_mlp` and install() resolves the key against the packed index."""
    for block in ("mlp", "block_sparse_moe"):
        switch = getattr(getattr(layer, block, None), "switch_mlp", None)
        if switch is not None:
            return switch, f"model.layers.{i}.{block}.switch_mlp"
    return None, None


def _read_experts(st, key, ids):
    """Read the router-selected expert rows with one os.pread per row, so the SSD's queue depth is used and every miss is read rather than dropped."""
    path, np_dt, mx_dt, shape, start, length = st._index[key]
    row_bytes = length // shape[0]
    row_shape = tuple(shape[1:])
    fd = st._fds[path].fileno()

    out = np.empty((len(ids),) + row_shape, dtype=np_dt)
    misses = [(i, int(eid)) for i, eid in enumerate(ids)]

    def fill(job):
        i, eid = job
        buf = os.pread(fd, row_bytes, start + eid * row_bytes)
        out[i] = np.frombuffer(buf, dtype=np_dt).reshape(row_shape)

    if len(misses) > 1:
        list(_POOL.map(fill, misses))
    elif misses:
        fill(misses[0])
    STATS["rows_read"] += len(misses)
    arr = mx.array(out)
    return arr.view(mx_dt) if mx_dt == mx.bfloat16 else arr


# Scratch for the wide (prefill) read path, since per-layer mx.zeros + mx.eval cost ~69 ms per 880 MB set (~3.3 s over 48 layers of a 2k-token prefill) for bytes preadv overwrote a moment later, with two sets alternating so layer L+1 reads into one while layer L's gather_qmm may still be queued.
_WIDE_SCRATCH = {}
# Experts per group in the wide pass (StreamingSwitchGLU._call_grouped): the scratch is one group's rows rather than the union's, since 441 experts of Qwen3.8 in one piece are 1.35 GB and a 2k-token prefill then no longer fits beside a 48-slot arena on a 16 GB machine, and a dense span may hold up to 2x this many rows (_batch_wide).
WIDE_GROUP = 128


# Selections at or above which the GLU sorts its rows (mlx_lm's SwitchGLU rule): below it a pass takes the deferred or narrow paths and at or above it the wide paths, a module constant so a test on a tiny model can lower it and exercise the wide paths.
SORT_MIN = 64
# From this many (row, expert) pairs a deferred pass is sorted by slot and gathered grouped (sorted_indices=True), since gather_qmm reads an expert once per pair (a 2-position verify pass costs 1.6x a token's experts although its union is 1.34x) and each slot read once is bit-identical output, never slower in the 35B A/B (1.07x code, 1.11x prose).
DEFER_SORT_MIN = 16


_RHS_LINE = None


def _rhs_line():
    """(floor, per_row, ok): a sorted gather_qmm of B one-row pairs over a tensor of E experts runs MLX's sorted-rhs kernel from B = max(floor, per_row * E), found here once per process (~0.1 s) by fitting at E 1 and 256 and checking at E 8 and 16, since MLX 0.32 hard-codes 16 and 4 and a retune would silently move every group off stock's kernel."""
    global _RHS_LINE
    if _RHS_LINE is None:
        _RHS_LINE = _probe_rhs_line()
        # The probe's buffers (46 MB peak) would otherwise sit in MLX's cache up to its limit, which idle footprint counts (50 MB at 0.05 GB), so they are synchronized and then cleared.
        mx.synchronize()
        mx.clear_cache()
    return _RHS_LINE


def _probe_rhs_line():
    rng = np.random.default_rng(0)
    w = mx.quantize(mx.array(rng.standard_normal((256, 128, 256)).astype(np.float32) * 0.05),
                    group_size=32, bits=4)
    x = mx.array(rng.standard_normal((4096, 1, 256)).astype(np.float32)).astype(mx.bfloat16)

    def parts(E, B):
        r = mx.array((np.arange(B) * E // B).astype(np.uint32))
        y = [mx.gather_qmm(x[:B], *(a[:E] for a in w), rhs_indices=r, transpose=True,
                           group_size=32, bits=4, sorted_indices=s) for s in (True, False)]
        return not mx.array_equal(*y).item()

    def first(E):
        lo, hi = 0, 1
        while not parts(E, hi):
            lo, hi = hi, 2 * hi
            if hi > len(x):
                return None
        while hi - lo > 1:
            mid = (lo + hi) // 2
            lo, hi = (lo, mid) if parts(E, mid) else (mid, hi)
        return hi

    floor, big = first(1), first(256)
    if floor is None and big is None:
        # The kernels never part (MLX on CUDA) so any line is exact, but an infinite one left the budget no prefill width to cut at and the pool no standing room (0.43 tok/s), so MLX Metal's own constants stand in (GatherQMM::eval_gpu's 16 and 4).
        return 16, 4, True
    if floor is None or big is None:
        return float("inf"), 0, False
    per_row = big // 256
    line = lambda E: max(floor, per_row * E)
    ok = big == line(256) and all(parts(E, line(E)) and not parts(E, line(E) - 1)
                                  for E in (8, 16))
    return floor, per_row, ok


def _rhs_rows(n_rows):
    """Pairs from which a sorted call over `n_rows` experts takes the sorted-rhs kernel (_rhs_line)."""
    floor, per_row, _ = _rhs_line()
    return max(floor, per_row * n_rows)


def _takes_rhs(pairs, n_rows):
    """Whether a sorted gather_qmm of `pairs` one-row pairs over a tensor of `n_rows` experts runs MLX's sorted-rhs kernel rather than the per-row qmv (_rhs_line), which a gather over a slice or a shared store only matches to mlx_lm's SwitchGLU if the caller makes it."""
    return pairs >= _rhs_rows(n_rows)


def _wide_scratch(role, parts, rows):
    # One set per projection ROLE (gate, up, down), shared across layers, which is safe because layer L+1 decodes its routing before it fetches and that eval completes layer L's gather, while within a layer the three gathers queue after all three reads, so gate and up on one buffer had up's bytes under gate's gather and every prefill went wrong from the first token (test_ple 0/24).
    cur = _WIDE_SCRATCH.get(role)
    key = tuple((p["dtype"], tuple(p["shape"])) for p in parts)
    if cur is None or cur[0] != key or cur[1] < rows:
        bufs, views = [], []
        mx.clear_cache()          # growing under a full arena: give MLX's freed buffers back first
        for part in parts:
            b, v = _alloc_rows(part, rows)
            bufs.append(b)
            views.append(v)
        cur = (key, rows, bufs, views)
        _WIDE_SCRATCH[role] = cur
    _, cap, bufs, views = cur
    out_b, out_v = [], []
    for b, v in zip(bufs, views):
        out_b.append(b[:rows] if rows < cap else b)
        out_v.append(v[:rows])
    return out_b, out_v


# Prefill read-ahead: a layer whose grouped path runs reads the NEXT layer's whole expert region into the other of two sets on the pool while its own gathers and the next attention run, since a prefill pass's layer selects nearly all of its experts and reading them on the model thread after the routing sync left the GPU idle (~4 s a pass on the 35B).
_IN_PREFILL = False
_LAYER_SETS = [None, None]   # (shape key, {role: (bufs, views)}) per parity
_AHEAD = None                # (glu, parity, futures, issued at, bytes) of the read in flight


def _dense(lo, hi, n):
    """_batch_wide reads the span lo..hi whole, holes too, rather than the n wanted rows (see there)."""
    return hi - lo + 1 <= 2 * n and 3 * (hi - lo + 1) < 900


def _from_whole(bufs, ids):
    """_batch_wide's buffers for these experts, cut from a whole-layer set with the same rows in the same order, so the gather_qmm after it is the same call."""
    lo, hi = int(ids[0]), int(ids[-1])
    if _dense(lo, hi, len(ids)):
        return tuple(b[lo:hi + 1] for b in bufs), lo
    idx = mx.array(ids)
    return tuple(mx.take(b, idx, axis=0) for b in bufs), None


def _layer_set(i, glu):
    """Set i, sized for every expert of glu's three projections."""
    key = tuple((p["dtype"], tuple(p["shape"])) for g in glu._projs for p in g._packed["parts"])
    if _LAYER_SETS[i] is None or _LAYER_SETS[i][0] != key:
        _LAYER_SETS[i] = None
        mx.clear_cache()
        roles = {}
        for g in glu._projs:
            bufs, views = [], []
            for part in g._packed["parts"]:
                b, v = _alloc_rows(part, g._packed["n_experts"])
                bufs.append(b)
                views.append(v)
            roles[g._wide_role] = (bufs, views)
        _LAYER_SETS[i] = (key, roles)
    return _LAYER_SETS[i][1]


# Read path per layer: a layer either finds its whole set read ahead or reads now only the rows it gathers into the same set at their own index, the choice taken from this process's measurements (a read ahead costs max(window, B / rate_w) and a union read window + bytes / rate_u).
_RP_MODE = os.environ.get("MOE_READPATH", "auto")
_RP = {"rate_u": None, "rate_w": None, "t": None, "next": None, "win": [], "T": 0,
       "prev": None}


def _ema(k, v):
    _RP[k] = v if _RP[k] is None else _RP[k] + 0.3 * (v - _RP[k])


def _layer_bytes(glu):
    return sum(g._packed["stride"] for g in glu._projs) * glu._projs[0]._packed["n_experts"]


def _arena_of(glu):
    """glu's arena when its rows may be copied from it (the plain path's, not parked, no native agent owning it), else None."""
    if _NATIVE is not None or glu._budget is not None or glu._slot_np is None:
        return None
    return _ARENAS.get(glu._akey)


def _ssd_rows(glu, ids):
    """How many of glu's rows `ids` the SSD has to give (_read_layer)."""
    return len(ids) if _arena_of(glu) is None else int((glu._slot_np[ids] < 0).sum())


def _copy_rows(views, eids, srcs):
    for e, src in zip(eids, srcs):
        for v, s in zip(views, src):
            v[e] = s


def _read_layer(glu, i, ids=None):
    """Reads of glu's expert rows `ids` (sorted, every expert by default) into set i, each at its own index, on the pool: a row the layer's arena holds is copied from it, read-only and with its LRU untouched, and the rest are preadv'd PREFILL_CHUNK consecutive experts at a time."""
    roles, futs, nb = _layer_set(i, glu), [], 0
    n = glu._projs[0]._packed["n_experts"]
    ids = np.arange(n) if ids is None else np.asarray(ids)
    ar = _arena_of(glu)
    if ar is not None:
        glu._await_pending()            # lookahead reads into the arena land first
        sl = glu._slot_np[ids]
        held = sl >= 0
        for j, g in enumerate(glu._projs):
            futs.append(_POOL.submit(_copy_rows, roles[g._wide_role][1], ids[held],
                                     [ar[j][1][s] for s in sl[held]]))
        STATS["rp_copied"] += int(held.sum())
        ids = ids[~held]
    runs = np.split(ids, np.flatnonzero(np.diff(ids) != 1) + 1) if len(ids) else []
    jobs = []
    for g in glu._projs:
        pk, views = g._packed, roles[g._wide_role][1]
        for run in runs:
            for c0 in range(0, len(run), PREFILL_CHUNK):
                e0, e1 = int(run[c0]), int(run[min(c0 + PREFILL_CHUNK, len(run)) - 1]) + 1
                iov = [v[e] for e in range(e0, e1) for v in views]
                jobs.append((pk["fd"], iov, pk["offset"] + e0 * pk["stride"],
                             (e1 - e0) * pk["stride"]))
                nb += (e1 - e0) * pk["stride"]
    s = _STRIPE
    if s is None or s.dead:
        futs += [_POOL.submit(_do_read, j) for j in jobs]
        return futs, nb
    # A chunk the sidecar holds whole may be read from the other drive at the same time, split by the rates both drives gave together at install, since a layer is one long wave rather than a decode wave's ~15 records.
    own, ext, _, eb = s.split(jobs, s.bw_install)
    futs += [_POOL.submit(_do_read, j) for j in own]
    futs += [s.pool.submit(s.read, j, eoff) for j, eoff in ext]
    STATS["rp_ext_b"] += eb
    return futs, nb


def _ahead_pays(glu, nxt, ids):
    """Read nxt whole ahead (True), or its union after its routing (False)."""
    if _RP_MODE != "auto":
        return _RP_MODE == "whole"
    rp = _RP
    if rp["rate_u"] is None:            # nothing read yet: a union read measures it
        return False
    if rp["win"]:
        win = sum(rp["win"]) / len(rp["win"])
    elif rp["prev"]:
        # The first layer of a pass guesses its window from the last pass's, scaled to this width, and the measured one takes over at the second layer.
        win = rp["prev"][1] * rp["T"] / max(rp["prev"][0], 1)
    else:
        win = 0.0
    # SSD bytes either way (arena rows are copied), with the union predicted by this layer's share of rows the SSD gave.
    n_nxt, B = nxt._projs[0]._packed["n_experts"], _layer_bytes(nxt)
    whole = B * _ssd_rows(nxt, np.arange(n_nxt)) / n_nxt
    union = B * _ssd_rows(glu, ids) / glu._projs[0]._packed["n_experts"]
    return max(win, whole / (rp["rate_w"] or rp["rate_u"])) < win + union / rp["rate_u"]


def _whole_layer(glu, ids, T):
    """This layer's set: the whole layer from the read-ahead, else the rows `ids` read now, then the next layer's whole read started into the other set or not (_ahead_pays), with both sets free here since _prefill_layer_call forced the previous layer and this layer's routing sync has run."""
    global _AHEAD
    rp, t = _RP, time.perf_counter()
    if glu is rp["next"]:
        rp["win"].append(t - rp["t"])
    else:                               # a new pass
        if rp["win"]:
            rp["prev"] = (rp["T"], sum(rp["win"]) / len(rp["win"]))
        rp["win"] = []
    rp["T"] = T
    ahead, _AHEAD = _AHEAD, None
    if ahead is not None:
        for f in ahead[2]:
            f.result()
    if ahead is not None and ahead[0] is glu:
        i = ahead[1]
        STATS["rp_whole"] += 1
        if ahead[4] and time.perf_counter() - t > 1e-3:   # it held the layer up: its own rate
            _ema("rate_w", ahead[4] / (time.perf_counter() - ahead[3]))
    else:
        i = 0
        futs, nb = _read_layer(glu, i, None if _RP_MODE == "whole" else ids)
        for f in futs:
            f.result()
        STATS["rp_union"] += 1
        STATS["rp_b"] += nb
        if nb:
            _ema("rate_u", nb / (time.perf_counter() - t))
    nxt = glu.__dict__.get("_wide_next")
    if nxt is not None and _ahead_pays(glu, nxt, ids):
        futs, nb = _read_layer(nxt, 1 - i)
        _AHEAD = (nxt, 1 - i, futs, time.perf_counter(), nb)
        STATS["rp_b"] += nb
    rp["t"], rp["next"] = time.perf_counter(), nxt
    return {r: b for r, (b, _) in _LAYER_SETS[i][1].items()}


def _drop_layer_sets():
    global _AHEAD
    if _AHEAD is not None:
        for f in _AHEAD[2]:
            f.result()
        _AHEAD = None
    _LAYER_SETS[:] = [None, None]


# A background prefill gives way to a request: while it runs PREFILL_STOP is a callable and the pass stops at the next check (a layer's start, a GatedDeltaNet's scan, the end of its dense half, a full-attention query block or a wide group), with the read ahead cancelled, the sets dropped and the layer it stopped in put back as it was at its start (_PREFILL_AT), so PrefillStopped carries its index and input and a later pass of the same ids can take it up (PREFILL_SKIP).
PREFILL_STOP = None
_PREFILL_AT = None      # (index, input, [(cache, offset or states)]) of the layer running

# (k, h): a pass taking up a stopped one, where its layers before k return h, the input layer k had, and touch nothing.
PREFILL_SKIP = None


class PrefillStopped(Exception):
    """args (k, h): the layer the pass stopped in and its input, or (0, None) when no layer of it can be taken up."""


def prefill_stoppable(model):
    """Whether every layer of model's prefill passes checks PREFILL_STOP (install_prefill_sync's layers, a streamed MoE's)."""
    stack = _layer_stack_of(model)
    return stack is not None and all(l.__dict__.get("_prefill_on") for l in stack["layers"])


def _prefill_stop():
    global _AHEAD, _PREFILL_AT
    if PREFILL_STOP is None or not PREFILL_STOP():
        return
    if _AHEAD is not None:
        for f in _AHEAD[2]:
            f.cancel()
        concurrent.futures.wait(_AHEAD[2])
        _AHEAD = None
    _LAYER_SETS[:] = [None, None]
    _WIDE_SCRATCH.clear()
    at, _PREFILL_AT = _PREFILL_AT, None
    if at is None:
        raise PrefillStopped(0, None)
    k, h, held = at
    for c, s in held:
        if isinstance(c, ArraysCache):
            c.cache = s
        else:
            c.offset = s
            if getattr(c, "indexer", None) is not None:
                c.indexer.trim(s)       # qwen4_exp: its indexer keys grew with the KV
    raise PrefillStopped(k, h)


@contextlib.contextmanager
def prefill_skip(cache, k, h):
    """A model call inside takes up a pass stopped in layer k (PrefillStopped's (k, h)) on the same ids: layers before k return h and touch nothing, layer k on runs on it, and the KV caches before k are put back where the pass found them meanwhile since the model builds its masks from its caches before any layer runs."""
    global PREFILL_SKIP
    back = [(c, c.offset) for c in cache[:k] if isinstance(c, KVCache)]
    for c, o in back:
        c.offset = o - h.shape[1]
    PREFILL_SKIP = (k, h)
    try:
        yield
    finally:
        PREFILL_SKIP = None
        for c, o in back:
            c.offset = o


def _layer_at(layer, args, kwargs):
    """_PREFILL_AT for a layer starting: a KV cache only grows so its offset, a recurrent state is replaced so the arrays by reference, and None for any other cache."""
    cs = _caches_in(args, kwargs)
    if not all(isinstance(c, (KVCache, ArraysCache)) for c in cs):
        return None
    return (layer.__dict__["_prefill_idx"], args[0],
            [(c, list(c.cache) if isinstance(c, ArraysCache) else c.offset) for c in cs])


class StreamingSwitchLinear(nn.Module):
    """Reads router-selected experts straight from a persistent mmap, where a missing expert is READ rather than zeroed so the output is bit-identical to the resident model and the shard is mapped once so pages survive between layers and tokens."""

    def __init__(self, st_map, key_prefix, group_size, bits, mode, *, slots):
        super().__init__()
        self._st = st_map
        self._kp = key_prefix
        self.group_size, self.bits, self.mode = group_size, bits, mode
        # Destination buffers are allocated once, because mx.array on a cold buffer runs at 0.9 GB/s and the same call on a warm one at 55.5 GB/s.
        self._slots = slots
        self._bufs: dict[tuple, tuple] = {}
        packed = getattr(st_map, "packed", None)
        self._packed = None
        if packed and key_prefix in packed["index"]:
            meta = dict(packed["index"][key_prefix])
            meta["fd"] = packed["fd"].fileno()
            self._packed = meta
        self.freeze()

    def prepare(self, unique_ids, pos, sorted_indices):
        """Claim destinations and hand back the reads WITHOUT running them, which is what lets StreamingSwitchGLU issue gate, up and down as one wave (random 1.47 MB reads reach 4.37 GB/s at queue depth 1 and 7.08 GB/s at depth 8, while three sequential waves of top_k=4 never exceed depth 4)."""
        n = len(unique_ids)
        if self._packed is not None and n > self._slots:
            # Prefill: one big sequential read with nothing to parallelise, whose positions the caller already sorted by expert (_gather_sort), so `pos` and `unique_ids[pos]` are monotone and gather_qmm may take its sorted path as mlx_lm's SwitchGLU does (the unsorted kernel measured 197 ms per projection against 31 ms sorted on a 2028-position prefill).
            (bufs, lo) = self._batch_wide(unique_ids)
            if lo is not None:
                return bufs, mx.array(unique_ids[pos] - lo), sorted_indices, []
            return bufs, mx.array(pos), sorted_indices, []

        if _WIDE_SCRATCH:
            _WIDE_SCRATCH.clear()          # prefill is over; decode wants the memory
        if self._packed is not None:
            pk = self._packed
            fd, base, stride = pk["fd"], pk["offset"], pk["stride"]
            views, bufs = [], []
            for part in pk["parts"]:
                buf, view = self._dest(part["name"], n)
                views.append(view)
                bufs.append(buf)
            jobs = [(fd, [v[i] for v in views], base + int(e) * stride, stride)
                    for i, e in enumerate(unique_ids)]
            STATS["rows_read"] += n
            return tuple(bufs), mx.array(pos), sorted_indices, jobs

        # No experts.bin: fall back to the safetensors path, read inline.
        out = tuple(_read_experts(self._st, f"{self._kp}.{s}", unique_ids)
                    for s in ("weight", "scales", "biases"))
        STATS["rows_read"] += n
        return out, mx.array(pos), sorted_indices, []

    def _dest(self, suffix, n):
        """A buffer for n rows of this projection, made once and kept."""
        got = self._bufs.get((suffix, n))
        if got is None:
            if self._packed is not None:
                part = next(p for p in self._packed["parts"] if p["name"] == suffix)
                got = _alloc_rows(part, n)
            else:
                _, np_dt, mx_dt, shape, _, _ = self._st._index[f"{self._kp}.{suffix}"]
                b = mx.zeros((n,) + tuple(shape[1:]), dtype=_MX_OF[np.dtype(np_dt)])
                mx.eval(b)
                view = np.array(b, copy=False)
                got = (b.view(mx_dt) if mx_dt == mx.bfloat16 else b, view)
            self._bufs[(suffix, n)] = got
        return got


    def _batch_wide(self, unique_ids, scratch=None):
        """Many experts at once (prefill) straight out of experts.bin: when the requested ids cover a dense span it is one SEQUENTIAL read with 3 iovecs an expert (prefill sees nearly every expert of every layer, 64 x 27 x 4.42 MB = 7.64 GB, where one read an expert is 5184 syscalls of random I/O), and the buffers are temporary because kept they would be 7.6 GB."""
        pk = self._packed
        fd, base, stride = pk["fd"], pk["offset"], pk["stride"]
        lo, hi = int(unique_ids[0]), int(unique_ids[-1])   # _decode_indices sorts
        span, n = hi - lo + 1, len(unique_ids)
        # IOV_MAX is 1024 and an expert takes 3 iovecs, so the span stays well under it, its holes costing scratch (up to 2x the rows), since reading only the n wanted rows is not exact: gather_qmm picks its kernel from rows per expert in the buffer.
        dense = _dense(lo, hi, n)

        rows = span if dense else n
        if scratch is None:
            bufs, views = _wide_scratch(getattr(self, "_wide_role", id(self)), pk["parts"], rows)
        else:
            # Budget mode: the first rows of this projection's shared store.
            bufs, views = [b[:rows] for b in scratch[0]], [v[:rows] for v in scratch[1]]

        if dense:
            # One preadv per chunk instead of one for the whole span: the same sequential access, with IOV_MAX comfortable and no 200-iovec array in the kernel.
            for c0 in range(0, span, PREFILL_CHUNK):
                c1 = min(c0 + PREFILL_CHUNK, span)
                iov = [v[i] for i in range(c0, c1) for v in views]
                _do_read((fd, iov, base + (lo + c0) * stride, (c1 - c0) * stride))
            STATS["rows_read"] += span
            return tuple(bufs), lo
        _run_all(_do_read, [(fd, [v[i] for v in views], base + int(e) * stride, stride)
                            for i, e in enumerate(unique_ids)])
        STATS["rows_read"] += n
        return tuple(bufs), None


def _direct_fd(fd):
    """An O_DIRECT fd on fd's file, or None where the system has none."""
    try:
        return os.open(os.readlink(f"/proc/self/fd/{fd}"), os.O_RDONLY | os.O_DIRECT)
    except (OSError, AttributeError):
        return None


def _fill_read(job):
    """MemBudget.fill's tier read (fd, O_DIRECT fd or None, buffers, offset, bytes): through O_DIRECT where it is given, else
    through the page cache with the range dropped after, since route B's buffered reads grew zram +543 MB in 8 s."""
    fd, d, bufs, off, n = job
    if d is not None:
        try:
            if os.preadv(d, bufs, off) == n:
                return
        except OSError:
            pass
    got = os.preadv(fd, bufs, off)
    assert got == n, f"short read {got} != {n} at {off}"
    os.posix_fadvise(fd, off, n, os.POSIX_FADV_DONTNEED)


def _device_room(need):
    """MOE_RESIDENT's test before a store would shrink (a rebuild empties it): whether the device's free memory and MLX's
    cache hold `need` bytes (None: unknown, so yes), since on the budget's own estimate the store shrank after a resident
    8k prompt and decode read its experts again."""
    if need is None:
        return True
    free = mx.device_info()["free_memory"] + mx.get_cache_memory()
    if free < need:
        print(f"  resident: {free / 1e9:.2f} GB of the device free for a pass of {need / 1e9:.2f} GB, the store shrinks",
              flush=True)
    return free >= need


def _group_rows(want, ids, slot_np, pool):
    """MOE_RESIDENT's group rows: the store slot of each row in `want` (holes of a dense span, which no gather reads,
    take slot 0) and the positions of the ids among them that the store's quota rows [0, pool) do not hold."""
    sl = slot_np[want].astype(np.int32)
    asked = np.isin(want, ids)
    held = (sl >= 0) & (sl < pool)
    sl[~asked] = 0
    return sl, np.flatnonzero(asked & ~held)


def _do_read(job):
    """pread one (fd, iovecs, offset, bytes) job and insist on all of it: a short read leaves the tail of whatever held that spot before."""
    fd, iov, off, n = job
    got = os.preadv(fd, iov, off)
    assert got == n, f"short read {got} != {n} at {off}"


def _run_all(fn, items):
    """Run fn over items on the pool, or here when there is only one."""
    if len(items) > 1:
        list(_POOL.map(fn, items))
    elif items:
        fn(items[0])


def _alloc_rows(part, rows):
    """A warm [rows, *part] buffer and its numpy view; bfloat16 is held as uint16 and reinterpreted by the view."""
    np_dt, mx_dt = _ST_NP[part["dtype"]], _ST_MX[part["dtype"]]
    b = mx.zeros((rows,) + tuple(part["shape"]), dtype=_MX_OF[np.dtype(np_dt)])
    mx.eval(b)
    return b.view(mx_dt) if mx_dt == mx.bfloat16 else b, np.array(b, copy=False)


_ADDR_KERNEL = None


def _alloc_dev_rows(part, rows):
    """_alloc_rows for MOE_RESIDENT's store: the buffer stays in device memory and the "view" is a numpy array over its
    device address that only lends addresses and shapes (the copies into it run on the GPU), since a numpy view moves a
    CUDA buffer to managed memory through a second copy, a third of the store's VRAM held while it is built."""
    global _ADDR_KERNEL
    np_dt, mx_dt = _ST_NP[part["dtype"]], _ST_MX[part["dtype"]]
    shape = (rows,) + tuple(part["shape"])
    b = mx.zeros(shape, dtype=_MX_OF[np.dtype(np_dt)])
    if _ADDR_KERNEL is None:
        _ADDR_KERNEL = mx.fast.cuda_kernel(
            name="nn_addr_of", input_names=["a"], output_names=["p"],
            source="if (threadIdx.x == 0) p[0] = reinterpret_cast<unsigned long long>(&a[0]);")
    p = _ADDR_KERNEL(inputs=[b], grid=(1, 1, 1), threadgroup=(1, 1, 1), output_shapes=[(1,)],
                     output_dtypes=[mx.uint64])[0]
    mx.eval(b, p)
    nbytes = int(np.prod(shape)) * np.dtype(np_dt).itemsize
    v = np.ctypeslib.as_array((ctypes.c_uint8 * nbytes).from_address(int(p.item()))).view(np_dt).reshape(shape)
    return b.view(mx_dt) if mx_dt == mx.bfloat16 else b, v


# A second read device (mirror_experts.py): a sidecar on another drive holds a copy of some experts.bin records, and a read wave sends a mirrored record there while that drive would finish its share before the internal SSD finishes the rest.
_STRIPE = None
# Each device's bytes and seconds decay by this per wave it reads in, so the split follows what the device delivers now, about a token's worth of waves.
_STRIPE_DECAY = 0.98
# Where the default search looks for a sidecar (a test points it elsewhere).
_VOLUMES = Path("/Volumes")


class _Stripe:
    def __init__(self, meta_p, model_path, packed):
        meta = json.loads(Path(meta_p).read_text())
        main_fd = packed["fd"].fileno()
        pk = meta["pack"]
        sha = hashlib.sha1((Path(model_path) / "experts_index.json").read_bytes()).hexdigest()
        if pk["bin_bytes"] != os.fstat(main_fd).st_size or pk["index_sha1"] != sha:
            raise ValueError("made for another pack")
        self.path = Path(meta_p).with_name(meta["file"])
        self.fd = os.open(self.path, os.O_RDONLY)
        try:
            if os.fstat(self.fd).st_size != meta["ext_bytes"]:
                raise ValueError("sidecar size differs from its json")
            # A sidecar that is not a copy would be wrong weights, silently.
            for m, e, n in (meta["records"][0], meta["records"][-1]):
                if os.pread(self.fd, n, e) != os.pread(main_fd, n, m):
                    raise ValueError(f"record at {m} differs from experts.bin")
        except BaseException:
            os.close(self.fd)
            raise
        self.main_fd = main_fd
        self.at = {m: (e, n) for m, e, n in meta["records"]}
        self.readers = int(meta["ext_readers"])
        self.pool = ThreadPoolExecutor(self.readers)
        # [bytes, seconds], seeded with the install measurement (both at once) as one millisecond's worth.
        self.ema = {d: [meta[f"bw_{d}"] * 1e6, 1e-3] for d in ("int", "ext")}
        self.bw_install = (meta["bw_int"], meta["bw_ext"])
        self.dead = False

    def bw(self, dev):
        b, t = self.ema[dev]
        return b / t

    def note(self, dev, nbytes, secs):
        e = self.ema[dev]
        e[0] = _STRIPE_DECAY * e[0] + nbytes
        e[1] = _STRIPE_DECAY * e[1] + secs
        STATS[f"stripe_{dev}_b"] += nbytes
        TIMERS[f"stripe_{dev}"] += secs

    def ext_off(self, job):
        """The sidecar offset of this job's bytes, one record or a run of one tensor's consecutive records, if it holds them all, else None, since the sidecar keeps experts.bin's record order."""
        if self.dead or job[0] != self.main_fd:
            return None
        hit = self.at.get(job[2])
        if hit is None:
            return None
        e, n = hit
        if job[3] == n:
            return e
        last = self.at.get(job[2] + job[3] - n)
        return e if job[3] % n == 0 and last == (e + job[3] - n, n) else None

    def split(self, jobs, bw=None):
        """(jobs for experts.bin, [(job, sidecar offset)], bytes of each): unmirrored records first, then each mirrored one to whichever device would finish sooner with it by the rates `bw` (int, ext) or else the waves' own, and whole records because splitting the boundary record measured no faster."""
        own, cand, ib = [], [], 0
        for j in jobs:
            eoff = self.ext_off(j)
            if eoff is not None:
                cand.append((j, eoff))
            else:
                own.append(j)
                ib += j[3]
        ext, eb = [], 0
        bi, be = bw or (self.bw("int"), self.bw("ext"))
        for j, eoff in cand:
            n = j[3]
            if (eb + n) / be < (ib + n) / bi:
                ext.append((j, eoff))
                eb += n
            else:
                own.append(j)
                ib += n
        return own, ext, ib, eb

    def read(self, job, eoff):
        """One record from the sidecar, or from experts.bin if that fails (the drive unplugged or unreadable) with the iovecs then overwritten whole."""
        _, iov, _, n = job
        try:
            got = os.preadv(self.fd, iov, eoff)
        except OSError as e:
            got = e
        if got != n:
            STATS["stripe_fell"] += 1
            if not self.dead:
                self.dead = True
                print(f"  stripe: sidecar read failed ({got!r}); experts.bin from here on",
                      flush=True)
            _do_read(job)
        return time.perf_counter()

    def close(self):
        self.pool.shutdown(wait=True)
        os.close(self.fd)


def open_stripe(model_path, packed):
    """The sidecar mirror_experts.py made for this pack, if there is one and it matches, else None and the exact old read path."""
    global _STRIPE
    if _STRIPE is not None:         # a model installed before this one
        _STRIPE.close()
        _STRIPE = None
    want = os.environ.get("MOE_STRIPE", "auto")
    if not packed or want == "0":
        return None
    name = Path(model_path).resolve().name
    found = ([Path(want)] if want != "auto"
             else sorted(_VOLUMES.glob(f"*/{name}/experts.ext.json")))
    for p in found:
        try:
            s = _Stripe(p, model_path, packed)
        except (OSError, ValueError, KeyError, IndexError) as e:
            print(f"  stripe: {p} not used ({e})", flush=True)
            continue
        # A sidecar found on experts.bin's own volume adds readers, not a drive, and one named by MOE_STRIPE is used as is.
        if want == "auto" and os.fstat(s.fd).st_dev == os.fstat(s.main_fd).st_dev:
            s.close()
            note(f"  stripe: {p} not used (on experts.bin's drive)", flush=True)
            continue
        _STRIPE = s
        note(f"  stripe: {len(s.at)} records ({os.fstat(s.fd).st_size / 1e9:.1f} GB) "
              f"mirrored on {s.path.parent}, {s.readers} readers; "
              f"{s.bw('int') / 1e9:.2f} + {s.bw('ext') / 1e9:.2f} GB/s", flush=True)
        return s
    return None


# MOE_CUDA_STAGE (on by default under CUDA, never on Metal or CPU): decode's blocking waves and the layer-major prefill land in host staging read with O_DIRECT and a kernel copies the rows into their slots (cuda_stage.py, imported only when on).
_CUDA_STAGE_ON = (BACKEND == "cuda"
                  and os.environ.get("MOE_CUDA_STAGE", "1") == "1")
_CUDA_STAGE = None


def _cuda_stage():
    global _CUDA_STAGE
    if _CUDA_STAGE is None:
        import cuda_stage
        _CUDA_STAGE = cuda_stage.Staging(lambda job: _do_read(job))
    return _CUDA_STAGE


def _read_wave(jobs, glu=None, buffered=False):
    """Run these _do_read jobs and wait for all of them, with a sidecar's share going to its own readers at the same time and `glu` the layer whose arena slots the jobs fill (`buffered`: through the page cache)."""
    s = _STRIPE
    if glu is not None and _CUDA_STAGE_ON and s is None and jobs:
        _cuda_stage().wave(jobs, _ARENAS[glu._akey], _POOL, by_stride=_RESIDENT, buffered=buffered)
        return
    if s is None:
        if len(jobs) > 1:
            list(_POOL.map(_do_read, jobs))
        elif jobs:
            _do_read(jobs[0])
        return
    own, ext, ib, eb = s.split(jobs)
    t0 = time.perf_counter()
    futs = [s.pool.submit(s.read, j, eoff) for j, eoff in ext]
    if len(own) > 1:
        list(_POOL.map(_do_read, own))
    elif own:
        _do_read(own[0])
    ti = time.perf_counter() - t0
    ends = [f.result() for f in futs]
    if own:
        s.note("int", ib, ti)
    if ext:
        s.note("ext", eb, max(ends) - t0)


# Bounded-loss cache bias: the biased top-k is kept only while the TRUE router mass it selects is within `_BIAS_BUDGET` (relative) of the honest top-k's mass, else that layer routes honestly and pays its misses, with 0 off.
_BIAS_BUDGET = 0.0


def set_bias_budget(tau):
    global _BIAS_BUDGET
    _BIAS_BUDGET = float(tau)

# MOE_ROUTE_KERNEL: the last router input and its fp32 cast, for the staging guess that ranks the next layer's router on the same array.
_XF = [None, None]
# The shared expert of a T=1 native pass computed while the agent answers its handshake, the same ops on the same input ordered by mx.depends between signal_after and wait_for, on by default with MOE_SHARED_EARLY=0 off.
_SHARED_EARLY = os.environ.get("MOE_SHARED_EARLY", "1") == "1"


class CacheBiasedMoE(nn.Module):
    """Nudge the router toward experts the arena already holds, biasing the RANKING only (rank = softmax(z) + lambda * delta * m, m marking resident experts and delta this token's spread) while the weights still come from the untouched softmax, which is why this wraps the whole block rather than the gate and why check_equivalence asserts lambda=0 reproduces the original block exactly."""

    def __init__(self, block, lam):
        super().__init__()
        if getattr(block, "sharding_group", None) is not None:
            raise ValueError("cache bias does not handle a sharded MoE block")
        self.block = block
        self.top_k = block.top_k
        self._lam = lam
        self._buf = None

    def ranking(self, x, gates=None):
        """The array the selection is made from, also what the lookahead ranks, since it has to see the same bias or its recall would drop, with `gates` letting the caller pass a softmax it has already computed and so save a second router run a layer."""
        b = self.block
        if gates is None:
            gates = mx.softmax(b.gate(x), axis=-1, precise=True)
        # Under --defer a repaired layer is recomputed after its misses were claimed and this bias would see them as resident, ranking a different top-k than the pass being repaired (a 0.16% drift of the layer output and a divergence at token 5), so the hold is the LRU as the router saw it the first time.
        keys = self._keys()
        if self._lam <= 0.0 or not keys:
            return gates
        spread = (mx.max(gates, axis=-1, keepdims=True)
                  - mx.min(gates, axis=-1, keepdims=True))
        return gates + (self._lam * spread) * self._mask(keys, gates.shape[-1])

    def _keys(self):
        """The residency the router ranks against: the host LRU of the layer, the pre-claim view a repaired layer is held to, or on a native pass the agent's slot table."""
        if _NATIVE is not None and _NATIVE.active:
            glu = self.block.switch_mlp
            s = glu._slot_np
            return np.flatnonzero((s >= 0) if glu._store is None
                                  else (s >= 0) & (s < glu._store.pool)).tolist()
        hold = getattr(self.block.switch_mlp, "_defer_hold", None)
        if hold is not None:
            return hold.keys
        return list(getattr(self.block.switch_mlp, "_lru", ()))

    def _mask(self, keys, n):
        """m, the 0/1 indicator of `keys` over n experts, a fresh 1 KB array rather than a written-in-place view since the lookahead calls the NEXT layer's ranking and one buffer would be touched twice a token."""
        if self._buf is None or self._buf.shape[0] != n:
            self._buf = np.zeros(n, dtype=np.float32)
        self._buf[:] = 0.0
        self._buf[keys] = 1.0
        return mx.array(self._buf)

    def _fused_route(self, gates, glu, k):
        """P3: bias, top-k, budget, slot lookup and residency flag in ONE metal kernel (fused_router.py), returning the selection or None when the arm is off or the shape is not one it serves, with its ok/rhs parked on the GLU for `_call_defer` and dropped unread whenever that path is not the one taken."""
        cuda = _CUDA_SLOTS and route_cuda.fits(gates, k)
        if (not (fused_router.on() or cuda) or self._lam <= 0.0
                or gates.shape[-2] > fused_router.MAX_T
                or getattr(glu, "_slot_of", None) is None):
            return None
        if cuda:
            return self._cuda_route(gates, glu, k)
        keys = self._keys()
        if not keys:
            return None
        inds, ok, rhs = fused_router.route(
            gates, self._mask(keys, gates.shape[-1]), glu._slot_of,
            self._lam, _BIAS_BUDGET, k)
        glu.__dict__["_fused_res"] = (inds, ok, rhs)
        return inds

    def _cuda_route(self, gates, glu, k):
        """MOE_CUDA_SLOTS: the same route as one CUDA kernel, with the mask read from the slot table itself unless a repair holds the router to an older view."""
        lim = None
        if _CUDA_SLOTS and getattr(glu, "_defer_hold", None) is None:
            # On a native pass the table is the agent's LRU and Python's is stale, which sent the whole turn to the ops once emptied by a release.
            if not glu._lru and not (_KEEP and _NATIVE is not None and _NATIVE.active):
                return None
            lim = glu._store.pool if glu._store is not None else 2 ** 31 - 1
            mask = None
        else:
            keys = self._keys()
            if not keys:
                return None
            mask = self._mask(keys, gates.shape[-1])
        out = route_cuda.route(gates, mask, glu._slot_of, self._lam, _BIAS_BUDGET, k, lim=lim,
                               slots=_DEFER_ACTIVE)
        if _DEFER_ACTIVE:
            glu.__dict__["_fused_res"] = out
        return out[0]

    # Three routing conventions share this wrapper, told apart by the block: the probability conventions (softmax over all experts, top-k of those probabilities, renormalised only where norm_topk_prob says so, which Qwen2-MoE never does for want of the flag) and Qwen3.8's qwen4_exp (top-k of raw fp32 logits, softmax over the k selected), with the bias added to the array the block ranks on and the weights always the block's own, keyed on the block since hasattr(norm_topk_prob) alone put Qwen2-MoE in the Qwen3.8 branch and inflated raw top-4-of-60 probabilities to 1 (a 3.52e-01 divergence at lambda 0, check_equivalence).
    _PROB_ROUTED_BLOCKS = ("Qwen2MoeSparseMoeBlock",)

    @property
    def _logit_routed(self):
        if type(self.block).__name__ in self._PROB_ROUTED_BLOCKS:
            return False
        return not hasattr(self.block, "norm_topk_prob")

    def _select(self, x, gates):
        """(rank, the top-k of it): the biased ranking, or the honest top-k where the bias costs more true mass than the budget."""
        k = self.top_k
        rank = self.ranking(x, gates) if self._lam > 0.0 else gates
        inds = mx.argpartition(rank, kth=-k, axis=-1)[..., -k:]
        if _BIAS_BUDGET > 0.0 and self._lam > 0.0:
            # Honest top-k beside the biased one, falling back per token when the bias costs more true mass than the budget, as a lazy `where` with no sync.
            honest = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
            m_h = mx.take_along_axis(gates, honest, axis=-1).sum(axis=-1, keepdims=True)
            m_b = mx.take_along_axis(gates, inds, axis=-1).sum(axis=-1, keepdims=True)
            inds = mx.where(m_h - m_b > _BIAS_BUDGET * m_h, honest, inds)
        return rank, inds

    def guess(self, x):
        """This layer's selection on another layer's input, best first: MOE_STAGING's guess of its experts from the layer below's MoE input, with the mask as this layer's residency."""
        lg = self.block.gate(x.astype(mx.float32) if self._logit_routed else x)
        gates = mx.softmax(lg, axis=-1, precise=True)
        keys = self._keys()
        if self._lam > 0.0 and keys and gates.shape[-2] <= fused_router.MAX_T:
            # The same selection in one launch instead of ~18 small ones (fused_router.py, index for index), ascending by rank.
            inds = fused_router.route(gates, self._mask(keys, gates.shape[-1]),
                                      self.block.switch_mlp._slot_of, self._lam,
                                      _BIAS_BUDGET, self.top_k)[0]
            return inds[..., ::-1]
        return _best_first(*self._select(x, gates))

    def _route_args(self, n):
        """(mask, lam, tau) for fused_router.route_logits, where no keys means no bias and no budget as ranking() and _select() decide it."""
        keys = self._keys() if self._lam > 0.0 else None
        if not keys:
            return fused_router.zeros_mask(n), 0.0, 0.0
        return self._mask(keys, n), self._lam, _BIAS_BUDGET

    def guess_cat(self, x, prev):
        """MOE_ROUTE_KERNEL: guess()'s selection in one launch, written after `prev` as _call_native sends them, or None where guess() would not take the fused route."""
        if not (self._lam > 0.0 and self._logit_routed and x.shape[-2] == 1):
            return None
        mask, lam, tau = self._route_args(self.block.gate.weight.shape[0])
        if lam == 0.0:
            return None
        xf = _XF[1] if _XF[0] is x else x.astype(mx.float32)
        return fused_router.route_logits(self.block.gate(xf), mask, lam, tau, self.top_k,
                                         prev=prev)

    def _route_kernel(self, x):
        """MOE_ROUTE_KERNEL, one logit-routed decode row: the gate matvec, then softmax, bias, top-k, budget and the selected softmax in one launch (fused_router.route_logits, bit-equal to the ops below), with the fp32 input kept for the staging guess and the int32 ids for the native handshake."""
        xf = x.astype(mx.float32)
        _XF[0], _XF[1] = x, xf
        logits = self.block.gate(xf)
        mask, lam, tau = self._route_args(logits.shape[-1])
        inds, scores, ids = fused_router.route_logits(logits, mask, lam, tau, self.top_k)
        self.block.switch_mlp.__dict__["_ids32"] = (inds, ids)
        if (k2 := self.block.__dict__.get("_k2")) is not None:
            scores = route_k.weights(logits, inds, k2)
        return inds, scores

    def __call__(self, x):
        b = self.block
        k = self.top_k
        if (fused_router.route_on() and self._logit_routed and x.ndim == 3
                and x.shape[0] == 1 and x.shape[1] == 1 and _NATIVE is not None
                and _NATIVE.active):
            inds, scores = self._route_kernel(x)
            if b.__dict__.get("_lg"):           # MOE_GLUE: the block's end in one kernel
                return layer_glue.moe_tail(self, x, inds, scores)
            early = self._mark_shared_early(x)
            y = (b.switch_mlp(x, inds) * scores[..., None]).sum(axis=-2).astype(x.dtype)
            if getattr(b, "shared_expert", None) is not None:
                y = y + self._shared_out(x, early)
            return y
        if (fused_router.route_on() and not self._logit_routed and x.ndim == 3
                and x.shape[0] == 1 and x.shape[1] <= fused_router.MAX_T and self.top_k <= 32
                and (fused_router.route_sync() or (_NATIVE is not None and _NATIVE.active))):
            # The probability-routed block: the same ops after the gate in one launch a row, for decode and MTP verify passes (fused_router.route_probs), on the sync path only with MOE_ROUTE_KERNEL=1.
            lg = b.gate(x)
            if lg.dtype == mx.bfloat16 and lg.shape[-1] % 32 == 0:
                mask, lam, tau = self._route_args(lg.shape[-1])
                inds, scores, ids = fused_router.route_probs(
                    lg, mask, lam, tau, k, getattr(b, "norm_topk_prob", False))
                if _NATIVE is not None and _NATIVE.active and x.shape[1] == 1:
                    b.switch_mlp.__dict__["_ids32"] = (inds, ids)
                early = self._mark_shared_early(x)
                y = (b.switch_mlp(x, inds) * scores[..., None]).sum(axis=-2)
                if getattr(b, "shared_expert", None) is not None:
                    y = y + self._shared_out(x, early)
                return y
        if (b.__dict__.get("_lgp") and self._logit_routed and "_lm" in b.switch_mlp.__dict__
                and layer_glue.rows_ok(x) and not fused_router.on()
                and getattr(b, "shared_expert", None) is not None):
            # MOE_GLUE_PREFILL, a layer-major pass: the lines below in route_logits' kernel and the block's end over the chunk's rows.
            logits = b.gate(x.astype(mx.float32))
            if layer_glue.route_rows_ok(logits, k):
                inds, scores = layer_glue.route_rows(logits, *self._route_args(logits.shape[-1]), k)
                if (k2 := b.__dict__.get("_k2")) is not None:
                    scores = route_k.weights(logits, inds, k2)
                return layer_glue.moe_end_rows(b.switch_mlp(x, inds), scores, x, b)
        if self._logit_routed:
            # Rank on probabilities here too, since the block ranks raw logits and softmax is monotone (an identical selection at lambda 0) while with a bias lambda means the same share of the probability spread instead of lambda x a logit spread of 5-15, which forced the residents in and broke the first Qwen3.8 answers.
            logits = b.gate(x.astype(mx.float32))
            gates = mx.softmax(logits, axis=-1, precise=True)
        else:
            logits = None
            gates = mx.softmax(b.gate(x), axis=-1, precise=True)
        if (fused := self._fused_route(gates, b.switch_mlp, k)) is not None:
            inds = fused
        else:
            inds = self._select(x, gates)[1]
        if self._logit_routed:
            # The block's weights: softmax over the k SELECTED logits, or over the top k2's mass under MOE_ROUTE_K.
            scores = route_k.weights(logits, inds, b.__dict__.get("_k2") or k)
        else:
            scores = mx.take_along_axis(gates, inds, axis=-1)
            # Absent on Qwen2-MoE, which never renormalises.
            if getattr(b, "norm_topk_prob", False):
                scores = scores / scores.sum(axis=-1, keepdims=True)
        early = self._mark_shared_early(x)
        y = (b.switch_mlp(x, inds) * scores[..., None]).sum(axis=-2)
        if self._logit_routed:
            y = y.astype(x.dtype)
        if getattr(b, "shared_expert", None) is not None:
            y = y + self._shared_out(x, early)
        return y

    def _shared(self, x):
        b = self.block
        return mx.sigmoid(b.shared_expert_gate(x)) * b.shared_expert(x)

    def _mark_shared_early(self, x):
        """MOE_SHARED_EARLY: hand the shared expert to this native pass (one row, or T under MOE_NATIVE_MTP), which builds it between its handshake and its wait (_call_native)."""
        if not (_SHARED_EARLY and _NATIVE is not None and _NATIVE.active and x.ndim == 3
                and x.shape[0] == 1 and getattr(self.block, "shared_expert", None) is not None):
            return False
        self.block.switch_mlp.__dict__["_shared_early"] = self._shared
        return True

    def _shared_out(self, x, early):
        sh = self.block.switch_mlp.__dict__.pop("_shared_out", None) if early else None
        return self._shared(x) if sh is None else sh


class CacheBiasedExpertBias:
    """Cache-conditional routing for bailing_moe (Ling), which already separates selection from weighting via `e_score_correction_bias` in group_expert_select, so this rewrites that bias array in place instead of duplicating routing logic the way CacheBiasedMoE has to for Qwen3.5/3.6, scaling by 1/top_k so lambda is comparable in spirit to the Qwen path."""

    def __init__(self, owner, attr, glu, lam, n_experts, top_k):
        self.glu, self.lam = glu, lam
        self.scale = lam / max(top_k, 1)
        self.base = None
        b = getattr(owner, attr, None)
        if b is not None:
            self.base = np.array(b.astype(mx.float32))
        self.arr = mx.zeros((n_experts,), dtype=mx.float32)
        mx.eval(self.arr)
        self.view = np.array(self.arr, copy=False)
        setattr(owner, attr, self.arr)
        glu._bias_hook = self
        self.refresh()

    def refresh(self):
        """Rewrite the mask from the arena. Called wherever slots move."""
        self.view[:] = 0.0 if self.base is None else self.base
        if self.lam > 0.0 and self.glu._slot_np is not None:
            self.view[self.glu._slot_np >= 0] += self.scale


def check_equivalence(block, wrapper, dim):
    """At lambda 0 the wrapper must be the block. Catches the duplication drifting."""
    lam, wrapper._lam = wrapper._lam, 0.0
    x = mx.random.normal((1, 2, dim))
    a, c = block(x), wrapper(x)
    mx.eval(a, c)
    wrapper._lam = lam
    gap = float(mx.max(mx.abs(a - c)).item())
    if gap > 1e-4:
        raise AssertionError(
            f"cache-bias wrapper diverges from the MoE block by {gap:.2e} at "
            f"lambda 0: mlx-lm's block has changed, update CacheBiasedMoE")


def _wide_out(out, n_rows, y):
    """The wide pass's unsorted output buffer, made on the first write in the dtype gather_qmm actually returned."""
    if out is None:
        out = mx.zeros((n_rows,) + y.shape[1:], dtype=y.dtype)
    return out


class StreamingSwitchGLU(nn.Module):
    """Whole MoE block, so all three projections read in ONE wave: SwitchGLU calls them as separate modules one after another, each issuing top_k reads and waiting (three waves of depth 4 where the measured optimum is 8), and nothing forces that order since all three are indexed by the SAME experts."""

    def __init__(self, st, key_base, proto, resident=0, *, slots):
        super().__init__()
        self.activation = proto.activation
        self._projs = []
        for name in ("gate_proj", "up_proj", "down_proj"):
            orig = getattr(proto, name)
            m = StreamingSwitchLinear(st, f"{key_base}.{name}", orig.group_size,
                                      orig.bits, orig.mode, slots=slots)
            m._wide_role = name          # its own prefill scratch; see _wide_scratch
            setattr(self, name, m)
            self._projs.append(m)
        # Depth lookahead, wired by install(): the NEXT layer's router and its streaming block, see warm().
        self._next_gate = None
        self._next_glu = None
        # MOE_STAGING, wired by install(): the next layer's guess on this layer's input (_stage_guess) and the GPU's expert to physical row table the agent keeps beside _slot_of (_init_arena).
        self._stage_next = None
        self._phys_of = None
        # MOE_PIPE_STAGING, wired by _Native: the next layer's gate logits on this layer's input and its top-k (_pipe_guess).
        self._guess_next = None
        # MOE_PIPE_HITS_FIRST, wired by _Native: this layer's held experts run before its wait.
        self._hits_first = False
        self._look_m = 0
        # High-water mark of distinct experts one token has asked this layer for, written by _claim and read by the governor as its floor.
        self._want = 0
        self._pending = []
        # Slots a lookahead read is filling, as (expert, slot), kept out of _lru and _slot_of until _commit publishes them.
        self._defer = []
        # One arena and one LRU for the whole layer (see _init_arena), kept in _ARENAS rather than on self so nn.Module.parameters() cannot see it.
        self._cap = resident
        self._lru = OrderedDict()
        self._akey = _ArenaKey()
        # Budget mode only (enable_budget): the MemBudget, the shared store this layer's slots live in, and the ring slots it holds (eid to slot), with _lru then holding its quota members.
        self._budget = self._store = None
        self._ring = {}
        # _slot_of maps expert id to arena slot (-1 = absent) for the GPU, which cannot read the host's OrderedDict.
        self._slot_of = self._slot_np = self._slot_u32 = None
        # _seen counts the experts the decode path asked for, for eprofile.py; _warm is a profile _init_arena fills the arena from.
        self._seen = None
        self._warm = None
        self._n_exp = 0
        # Set by CacheBiasedExpertBias on architectures whose gate already takes a selection-only bias term.
        self._bias_hook = None
        # EXL3 trellis records (install, from config.json's expert_quant): the codeword window, None for MLX affine ones.
        self._exl3 = None
        # GGUF block records (install, from experts_index.json's parts): "gate/up/down" ggml types, None otherwise.
        self._gguf = None
        self.freeze()

    @property
    def _own_kernels(self):
        """Experts computed by our own kernels (EXL3 or GGUF), which take rhs unpadded and give only the last output."""
        return self._exl3 is not None or self._gguf is not None

    def _experts(self, x, bufs, rhs, sort=False, host=None):
        """gate, up, the activation and down over `bufs` (one tuple of parts a projection) at rows `rhs`: the list of the three outputs, where an EXL3 or GGUF pack gives only the last."""
        if self._exl3 is not None:
            return [None, None, exl3_experts.experts(x, bufs, rhs, self._exl3, host)]
        if self._gguf is not None:
            return [None, None, gguf_experts.experts(x, bufs, rhs, self._gguf, host)]
        outs = []
        for p, b in zip(self._projs, bufs):
            src = x if len(outs) < 2 else self.activation(outs[1], outs[0])
            outs.append(mx.gather_qmm(src, *b, rhs_indices=rhs, transpose=True,
                                      group_size=p.group_size, bits=p.bits,
                                      mode=p.mode, sorted_indices=sort))
        return outs

    def _init_arena(self):
        """One slot space and one LRU for the layer rather than one per projection, since the three are always asked for the SAME experts and three LRUs meant 3240 claims a run where 1080 suffice (48 ms a token against the SSD's ~6), with the iovec lists built once per slot too."""
        # The arena is built on the first decode right after prefill allocated ~1 GB of transient buffers MLX still holds in its allocator cache, and without this the two stack (4.48 GB peak measured) though neither alone approaches it.
        if _NATIVE is not None:
            _NATIVE.release()
        mx.clear_cache()
        if self._budget is not None:
            # Budget mode: the shared store's tensors and iovecs and this layer's own offsets into experts.bin, so every reader of _ARENAS (claims, the deferred gather, the profile warm-up) sees the same (bufs, iov, fd, offset, stride) with global slots.
            st = self._store = self._budget.store_for(self)
            self._cap = len(self._qslots) + st.per_half
            ar = [(st.bufs[j], st.iov[j], p._packed["fd"], p._packed["offset"],
                   p._packed["stride"]) for j, p in enumerate(self._projs)]
            room = self._qslots
        else:
            cap, ar = self._cap, []
            for p in self._projs:
                pk = p._packed
                bufs, views = [], []
                for part in pk["parts"]:
                    b, v = _alloc_rows(part, cap)
                    bufs.append(b)
                    views.append(v)
                iov = [[v[sl] for v in views] for sl in range(cap)]
                ar.append((tuple(bufs), iov, pk["fd"], pk["offset"], pk["stride"]))
            room = range(cap)
        _ARENAS[self._akey] = ar

        # Sync-free lookup table: the host's OrderedDict cannot be read from a Metal kernel, so residency is mirrored into a GPU array whose numpy view writes it in place as the arena buffers are.
        self._n_exp = self._projs[0]._packed["n_experts"]
        self._slot_of = mx.full((self._n_exp,), -1, dtype=mx.int32)
        mx.eval(self._slot_of)
        self._slot_np = np.array(self._slot_of, copy=False)
        if _CUDA_SLOTS:
            # The same buffer as uint32 rows for the gather (MOE_CUDA_SLOTS).
            self._slot_u32 = self._slot_of.view(mx.uint32)
            mx.eval(self._slot_u32)
        if _STAGE == "1" and self._budget is not None:
            self._phys_of = mx.full((self._n_exp,), -1, dtype=mx.int32)
            mx.eval(self._phys_of)
        self._arena_gen += 1
        for eid, sl in self._lru.items():
            self._slot_np[int(eid)] = sl
        if self._bias_hook is not None:
            self._bias_hook.refresh()
        # _seen is the whole process's demand, so a rebuilt arena keeps it.
        if self._seen is None or self._seen.shape[0] != self._n_exp:
            self._seen = np.zeros(self._n_exp, dtype=np.float64)
        if self._lru and self._budget is None:
            # Parked for a long prefill (park_arenas): the same experts back into the same slots, so decode goes on as if it never left.
            list(_POOL.map(_do_read, [(fd, iov[sl], off + e * stride, stride)
                                      for e, sl in self._lru.items()
                                      for (_, iov, fd, off, stride) in ar]))
        elif self._warm is not None and not _RESIDENT:     # MOE_RESIDENT's store is filled by MemBudget.fill, on the GPU
            if self._warm.shape[0] < self._n_exp:
                raise ValueError(
                    f"profile covers {self._warm.shape[0]} experts, this layer "
                    f"has {self._n_exp}: wrong packing?")
            order = np.argsort(-self._warm[:self._n_exp])[:len(room)]
            jobs = []
            for sl, e in zip(room, (int(x) for x in order
                                    if self._warm[int(x)] > 0.0)):
                self._lru[e] = sl
                self._slot_np[e] = sl
                jobs += [(fd, iov[sl], off + e * stride, stride)
                         for (_, iov, fd, off, stride) in _ARENAS[self._akey]]
            if jobs:
                list(_POOL.map(_do_read, jobs))
                STATS["rows_warm"] += len(jobs)
                if self._store is not None:
                    self._store.dirty = True

    def _claim(self, unique_ids, count=True, half=None, busy=()):
        """Assign slots for the whole layer and return (slots, read jobs), with `half` and `busy` for budget mode's pipelined rounds (_place_store)."""
        if self._akey not in _ARENAS:
            self._init_arena()
        if self._budget is not None:
            return self._claimed(unique_ids, *self._place_store(unique_ids, count, half, busy),
                                 count)
        lru, cap = self._lru, self._cap
        slots = np.empty(len(unique_ids), dtype=np.int32)
        misses = []
        for i, eid in enumerate(unique_ids):
            eid = int(eid)
            sl = lru.get(eid)
            if sl is None:
                misses.append((i, eid))
            else:
                lru.move_to_end(eid)
                slots[i] = sl
        # Past `cap` distinct ids in one call the LRU head is an entry this same loop just inserted, so two ids land on one slot and the matmul uses whichever arrived, which fails silently: per-token byte traces for arena 21 / width 24 and arena 12 / width 16 matched their own repeat in 2 of 173 and 1 of 200 tokens.
        hits = len(unique_ids) - len(misses)
        room = cap - hits
        if len(misses) > room:
            if count:
                raise ValueError(
                    f"demand path needs {len(unique_ids)} distinct experts in a "
                    f"{cap}-slot arena: top_k exceeds the arena and cannot be "
                    f"served without corrupting the layer")
            # Speculative path: a prediction that does not fit is dropped, which is what a prediction is for and keeps warm() from evicting the slots the blocking path is about to read.
            STATS["warm_dropped"] = STATS.get("warm_dropped", 0) + len(misses) - room
            for i, _ in misses[room:]:
                slots[i] = -1
            misses = misses[:room]
        for i, eid in misses:
            if len(lru) < cap:
                sl = len(lru)
            else:
                old_eid, sl = lru.popitem(last=False)
                if self._slot_np is not None:
                    self._slot_np[int(old_eid)] = -1
            lru[eid] = sl
            slots[i] = sl
            if self._slot_np is not None:
                self._slot_np[int(eid)] = sl
        return self._claimed(unique_ids, slots, misses, count)

    def _place_store(self, unique_ids, count, half=None, busy=()):
        """_claim's placement in budget mode: hits in the quota LRU or the ring, misses into the quota (its LRU victim once full) and what the quota cannot take into the ring half, with nothing this call hits evicted, a demand beyond quota + ring half raising, and `busy` the slots a round still on the GPU reads (_call_rounds)."""
        st, lru, ring, q = self._store, self._lru, self._ring, self._qslots
        slots = np.empty(len(unique_ids), dtype=np.int32)
        misses, keep = [], set()
        for i, eid in enumerate(unique_ids):
            eid = int(eid)
            sl = lru.get(eid)
            if sl is not None:
                lru.move_to_end(eid)
            else:
                sl = ring.get(eid)
                if sl is None:
                    misses.append((i, eid))
                    continue
                keep.add(sl)
            slots[i] = sl
        hits = len(unique_ids) - len(misses)
        half = self._half if half is None else half
        room = self._cap - hits - sum(st.half_of(s) in (None, half) for s in busy)
        if len(misses) > room:
            if count:
                raise ValueError(
                    f"demand path needs {len(unique_ids)} distinct experts, the "
                    f"layer holds {self._cap} (quota {len(q)} + ring half "
                    f"{st.per_half}): the caller must take the rounds path")
            STATS["warm_dropped"] = STATS.get("warm_dropped", 0) + len(misses) - room
            for i, _ in misses[room:]:
                slots[i] = -1
            misses = misses[:room]
        quota_room = len(q) - (hits - len(keep)) - sum(s < st.pool for s in busy)
        keep.update(busy)
        free = None
        for n, (i, eid) in enumerate(misses):
            if n >= quota_room:
                sl = st.ring_slot(self, eid, keep, half)
            elif len(lru) < len(q):
                # The lowest quota row no member holds (the native agent's rule), since a quota put back after a prefill has holes.
                if free is None:
                    used = set(lru.values())
                    free = iter([s for s in q if s not in used])
                sl = lru[eid] = next(free)
            else:
                # The least recently used member the GPU is done with.
                old_eid = next(e for e, s in lru.items() if s not in keep)
                sl = lru.pop(old_eid)
                self._slot_np[old_eid] = -1
                lru[eid] = sl
            slots[i] = sl
            self._slot_np[eid] = sl
        if misses:
            st.dirty = True
        if count:
            self._ghost(unique_ids)
        return slots, misses

    def _ghost(self, unique_ids):
        """Ghost LRU: the stack distance of every expert seen before, this pass's own peers included since they need slots at the same time, also replayed for the native agent's claims (_Native.to_python)."""
        last = self._ghost_last
        ids = np.asarray(unique_ids, dtype=np.int64)
        t = last[ids]
        t = t[t >= 0]
        if t.size:
            d = (last[None, :] >= t[:, None]).sum(axis=1) - 1
            self._ghost_hist += np.bincount(d, minlength=len(last))[:len(last)]
        self._ghost_clock += 1
        last[ids] = self._ghost_clock

    def _claimed(self, unique_ids, slots, misses, count):
        """The part of a claim every placement shares: bias mask, demand counters, and the read jobs for the misses."""
        if self._bias_hook is not None and misses:
            self._bias_hook.refresh()
        if count:
            # Demand from the ordinary path, counted for free since the host already has the ids.
            ids64 = np.asarray(unique_ids, dtype=np.int64)
            if self._seen is not None:
                self._seen[ids64] += 1.0
        if count:
            # Only the blocking path may claim a hit, since warm() fills the very slots the blocking call then reads, and counting both credited the same byte twice (60.4% where the truth was ~37%).
            STATS["hits"] += 3 * (len(unique_ids) - len(misses))
        jobs = [(fd, iov[int(slots[i])], off + eid * stride, stride)
                for (_, iov, fd, off, stride) in _ARENAS[self._akey]
                for i, eid in misses]
        STATS["rows_read"] += len(jobs)
        return slots, jobs

    def warm(self, ids):
        """Start reading these experts into the arena without waiting, called by the PREVIOUS layer so the reads run while the GPU is busy with that layer's gather_qmm, which is safe because this arena's last gather_qmm was forced by an mx.eval one or more layers ago."""
        if not self._resident_ok():
            return
        if self._defer:
            # A parked newcomer may still be written into its slot and _claim may hand that slot to another expert, so this waits while _defer is not empty.
            self._await_pending()
        new = [int(e) for e in ids if int(e) not in self._lru and int(e) not in self._ring]
        _, jobs = self._claim(ids, count=False)
        if _DEFER and new and self._slot_np is not None:
            # Run-ahead: this layer's own gather has not been CHECKED yet and the check reads the slot table the gather used, so the newcomers are parked out of _lru and _slot_of, absent to the GPU, _claim and the cache bias, until _commit publishes them from _await_pending before this layer's router, and the slot they are written into maps from no expert meanwhile.
            park = [(e, int(self._slot_np[e])) for e in new if self._slot_np[e] >= 0]
            for e, _ in park:
                if self._lru.pop(e, None) is None:
                    del self._ring[e]
                self._slot_np[e] = -1
            self._defer += park
        STATS["rows_warm"] += len(jobs)
        self._pending += [_POOL.submit(_do_read, j) for j in jobs]

    def _resident_ok(self):
        return self._cap and all(p._packed is not None for p in self._projs)

    def resize(self, cap):
        """Change slot count, with in-flight reads targeting the OLD views and therefore landing before the arena is dropped."""
        if cap == self._cap:
            return
        self._await_pending()
        # Never below top_k: the arena path is taken only while len(unique_ids) <= _cap and a layer always wants top_k experts, so the governor's floor of 6 against top_k 8 sent EVERY layer to _batch_wide on a shrink under memory pressure (~12.7 ms a layer, ~500 ms a pass).
        cap = max(cap, self._projs[0]._slots if self._projs else 1)
        if cap == self._cap:
            return
        self._cap = cap
        # The prefetch width has to come down with the arena, since warm() claims look_m ids at once and a shrink to the floor of 6 against a look_m of 8 leaves the speculative path asking for more slots than exist, so it is held one under the arena because the demand path needs room too.
        if self._look_m > cap - 1:
            self._look_m = max(0, cap - 1)
        self.clear_arena()

    def clear_arena(self):
        """Forget every slot: the arena is rebuilt empty, lazily, on first use, with in-flight reads targeting the old views and therefore landing first."""
        if _NATIVE is not None:
            _NATIVE.release()
        self._await_pending()
        _ARENAS.pop(self._akey, None)
        self._lru.clear()
        self._ring.clear()
        self.__dict__.pop("_log", None)
        self._slot_of = self._slot_np = self._phys_of = self._slot_u32 = None
        self._store = None
        self._arena_gen += 1
        if self._bias_hook is not None:
            # The correction term still carried the old arena's mask, so the first token after a resize routed on residency that no longer existed (test_model: sync != sync from a cleared arena).
            self._bias_hook.refresh()

    def _await_pending(self):
        if self._pending:
            t = time.perf_counter()
            for f in self._pending:
                f.result()
            self._pending = []
            # If the overlap works this is ~0, since the reads finished while the GPU was busy, and whatever shows up here is overlap that did NOT happen.
            TIMERS["wait_pred"] += time.perf_counter() - t
        if self._defer:
            self._commit()

    def _commit(self):
        """Publish the slots a claim parked (the LATE residency marking, once every caller has drained _pending and their bytes are in the arena), and called with the arena gone the publication is simply forgotten, which is right since resize clears _lru too."""
        defer, self._defer = self._defer, []
        if self._slot_np is None:
            return
        st = self._store
        for e, sl in defer:
            if st is not None and sl >= st.pool:
                # A parked ring slot goes back to the ring unless another layer took the slot meanwhile, which the half spacing rules out (ExpertStore.ring_slot).
                o = st.owner[sl]
                if o is None or o[0] is not self or o[1] != e:
                    continue
                self._ring[e] = sl
            else:
                self._lru[e] = sl
            self._slot_np[e] = sl
        if self._bias_hook is not None:
            self._bias_hook.refresh()

    _defer_wide_once = False
    # Bumped whenever the arena and its slot table are dropped or rebuilt (resize, _init_arena): a deferred gather records the generation it read, and a different one at check time means the check has to treat the gather as a miss whatever the flag said.
    _arena_gen = 0
    # The residency the router saw before the claim, while a repaired layer is being recomputed (see CacheBiasedMoE.ranking and _defer_release), a plain object rather than a tuple because nn.Module.__setattr__ routes tuples into the module's own dict and tries to delattr a missing instance attribute.
    _defer_hold = None

    def _call_split(self, x, unique_ids, pos, rows, out, n_rows):
        """Wide pass whose union exceeds the arena: resident rows gathered from the arena buffers and the others through the grouped read, the same arithmetic with two gathers instead of one, and nothing claimed since a pass this wide is speculation or prefill."""
        slots = self._slot_np[unique_ids]
        res_u = slots >= 0
        row_res = res_u[pos]
        ir = np.nonzero(row_res)[0]
        im = np.nonzero(~row_res)[0]
        STATS["wide_split"] += 1
        if len(ir):
            rhs = mx.array(slots[pos[ir]])
            prep = [a[0] for a in _ARENAS[self._akey]]
            dst = rows[mx.array(ir)]
            src0 = x[dst // (n_rows // x.shape[0])]     # flat row -> position
            y = self._experts(src0, prep, rhs, host=slots[pos[ir]])[2]
            out = _wide_out(out, n_rows, y)
            out[dst] = y
        if len(im):
            miss_ids = unique_ids[~res_u]
            remap = np.cumsum(~res_u) - 1
            out = self._call_grouped(x, miss_ids, remap[pos[im]],
                                     rows[mx.array(im)], out, n_rows)
        return out

    def _call_grouped(self, x, unique_ids, pos, rows, out, n_rows, store=None):
        """The wide pass in groups of WIDE_GROUP experts so the scratch is one group's rows whatever the union is: `pos` sorted by expert makes a group a contiguous range, `rows[i]` is the flat (position, slot) row of entry i so no sorted copy of x, no concatenation and no unsort is needed, and a group's output is evaluated before the next group's read lands in the same scratch; in budget mode the released store is the scratch and a group is half its ring, since groups that grew with the pool put the transient past the need that sized it (toy: groups 8 -> 72, the next prefill grew 11.3 MB against a 5.6 MB reserve)."""
        k = n_rows // x.shape[0]
        whole = None
        if store is not None and _KEEP:
            # The prompt's routing counts (MemBudget._seed): pos indexes unique_ids.
            pc = self.__dict__.get("_pc")
            if pc is None:
                pc = self._pc = np.zeros(self._projs[0]._packed["n_experts"], dtype=np.int64)
            pc[np.asarray(unique_ids)] += np.bincount(np.asarray(pos).ravel(), minlength=len(unique_ids))
        if store is None:
            group, sort = WIDE_GROUP, True
            if _IN_PREFILL:
                whole = _whole_layer(self, unique_ids, x.shape[0])
        elif id(store) in _LM_READ:
            return self._call_ahead(x, unique_ids, pos, rows, n_rows, store,
                                    _LM_READ[id(store)])
        elif _RESIDENT and self._gguf is not None and (_OVERLAP or _ONECALL):
            return self._call_resident(x, unique_ids, pos, rows, out, n_rows, store)
        else:
            group = (store.slots - store.pool) // 2
            sort = _takes_rhs(len(pos), self._projs[0]._packed["n_experts"])
        n = len(unique_ids)
        cuts = np.searchsorted(pos, np.arange(group, n, group))
        bounds = [0, *cuts.tolist(), len(pos)]
        for gi, (r0, r1) in enumerate(zip(bounds, bounds[1:])):
            if whole is not None:
                _prefill_stop()
            a = gi * group
            ids = unique_ids[a:a + group]
            pg = pos[r0:r1] - a
            dst = rows[r0:r1]
            src0, pad, bufs, at = x[dst // k], 0, [], None
            if store is not None and _RESIDENT:
                # A GGUF group on the decode chain reads its rows from the store in place (each row's expert by its store slot, the same bytes), so only the dense path's groups are taken.
                inplace = _INPLACE and self._gguf is not None and not gguf_experts.dense_pays(len(np.unique(pg)), r1 - r0)
                bufs, lo, at = self._take_group(ids, store, copy=not inplace)
            for j, p in enumerate(self._projs if not bufs else ()):
                if whole is not None:
                    b, lo = _from_whole(whole[p._wide_role], ids)
                else:
                    b, lo = p._batch_wide(
                        ids, None if store is None else (store.bufs[j], store.views[j]))
                    if store is not None and store.written is not None:
                        store.written[:b[0].shape[0]] = True
                bufs.append(b)
            r = ids[pg] - lo if lo is not None else pg
            if at is not None:
                r = at[r]
            if sort and store is not None and not self._own_kernels:
                pad = _rhs_rows(bufs[0][0].shape[0]) - (r1 - r0)
            if pad > 0:
                src0 = mx.concatenate(
                    [src0, mx.zeros((pad,) + src0.shape[1:], dtype=src0.dtype)])
                r = np.concatenate([r, np.full(pad, r[-1], dtype=r.dtype)])
            rhs = mx.array(r)
            outs = self._experts(src0, bufs, rhs, sort, r)
            y = outs[2][:r1 - r0] if pad > 0 else outs[2]
            out = _wide_out(out, n_rows, y)
            out[dst] = y
            if r1 < len(pos) or (at is not None and (at >= store.pool).any()):
                # An in-place group that read ring rows is evaluated even when last, so no later copy lands under it.
                mx.eval(out)
        return out

    def _take_group(self, ids, store, copy=True, ring=None):
        """MOE_RESIDENT, _call_grouped's group without the release: the rows _batch_wide would read (the span lo..hi when
        dense, else the ids) gathered from the store, the ids it does not hold first copied into ring rows (_fetch, or
        `ring`, the rows _call_resident already copied them into); -> (each projection's parts, lo or None, None), the
        same rows as _batch_wide's, or with copy=False (the store's own parts, lo or None, each of those rows' store slot)."""
        lo, hi = int(ids[0]), int(ids[-1])
        dense = _dense(lo, hi, len(ids))
        want = np.arange(lo, hi + 1) if dense else np.asarray(ids)
        sl, miss = _group_rows(want, ids, self._slot_np, store.pool)
        if len(miss):
            sl[miss] = self._fetch(want[miss], store) if ring is None else ring
        if not copy:
            # The kernels read the ring rows in place, and _call_grouped evaluates such a group before any later copy lands there.
            return [tuple(b) for b in store.bufs], (lo if dense else None), sl
        if self._own_kernels:
            # Our kernels read only the rows they are given, so the buffers take one shape (the ring's rows) and MLX reuses them, where a fresh size a group grew the device's pool.
            sl = np.concatenate([sl, np.zeros(store.slots - store.pool - len(sl), np.int32)])
        idx = mx.array(sl)
        bufs = [tuple(mx.take(t, idx, axis=0) for t in b) for b in store.bufs]
        if len(miss):
            mx.eval(bufs)       # the ring rows are the next group's scratch
        return bufs, (lo if dense else None), None

    def _fetch(self, e, store, base=0):
        """MOE_RESIDENT: experts `e`, which the store's quota does not hold, copied into the ring rows from `base` on from
        the native agent's tier or read there from experts.bin; -> those rows."""
        s = self._slot_np[e]
        STATS["resident_ring"] += int((s >= store.pool).sum())
        STATS["resident_absent"] += int((s < 0).sum())
        rows = store.scratch(len(e), base)
        nat = _NATIVE
        at = nat.agent.tier_find(self._native_idx, e) if nat is not None and nat.tier_slot else np.zeros(len(e), np.uint64)
        ar = _ARENAS[self._akey]
        copies, jobs, off_p = [], [], 0
        for j, (_, iov, fd, off, stride) in enumerate(ar):
            for k, (eid, sl_k) in enumerate(zip(e.tolist(), rows)):
                if at[k]:
                    src = int(at[k]) + off_p
                    for v in iov[sl_k]:
                        copies.append((v.ctypes.data, src, v.nbytes))
                        src += v.nbytes
                else:
                    jobs.append((fd, iov[sl_k], off + eid * stride, stride))
            off_p += stride
        if copies:
            _cuda_stage().copy_in(copies)
            STATS["resident_tier_rows"] += int((at != 0).sum())
        if jobs:
            # While the fill's store stands, through the page cache: the few experts neither store holds are asked for by every piece of a long prompt, and read once (an emptied store's thousands would flood it: zram +536 MB in keep_p).
            _read_wave(jobs, self, buffered=bool(getattr(self._budget, "intact", False)))
            STATS["resident_ssd_b"] += sum(j[3] for j in jobs)
        return rows

    def _call_resident(self, x, unique_ids, pos, rows, out, n_rows, store):
        """_call_grouped's MOE_RESIDENT path for a GGUF pack under MOE_RESIDENT_OVERLAP or _ONECALL: its groups as units,
        ONECALL putting the chain groups' rows of quota experts in one unit and their misses in rounds of a ring half (the
        whole ring without OVERLAP), OVERLAP launching each unit without a sync and copying the next one's misses into the
        other ring half meanwhile; every row goes through the same kernel on the same bytes as in its group."""
        k = n_rows // x.shape[0]
        half = (store.slots - store.pool) // 2
        cuts = np.searchsorted(pos, np.arange(half, len(unique_ids), half))
        bounds = [0, *cuts.tolist(), len(pos)]
        slot = self._slot_np[unique_ids]
        inq = (slot >= 0) & (slot < store.pool)
        # A unit: its rows (indices into pos), its misses (indices into unique_ids, ascending), and a taken group's first index or None on the chain.
        units, hit, cmiss = [], [], []
        for gi, (r0, r1) in enumerate(zip(bounds, bounds[1:])):
            a = gi * half
            m = np.flatnonzero(~inq[a:a + half]) + a
            if not _INPLACE or gguf_experts.dense_pays(len(np.unique(pos[r0:r1])), r1 - r0):
                units.append((np.arange(r0, r1), m, a))
            elif _ONECALL:
                hit.append(np.arange(r0, r1)[inq[pos[r0:r1]]])
                cmiss.append(m)
            else:
                units.append((np.arange(r0, r1), m, None))
        if cmiss:
            hit, cmiss = np.concatenate(hit), np.concatenate(cmiss)
            per = half if _OVERLAP else 2 * half
            rounds = [(np.flatnonzero(np.isin(pos, cmiss[i:i + per])), cmiss[i:i + per], None)
                      for i in range(0, len(cmiss), per)]
            # Under OVERLAP the quota rows are cut into as many pieces as there are rounds, so each round's copy runs under a piece.
            hits = [(h, cmiss[:0], None) for h in np.array_split(hit, max(1, len(rounds) if _OVERLAP else 1)) if len(h)]
            units += [u for j in range(max(len(hits), len(rounds))) for u in hits[j:j + 1] + rounds[j:j + 1]]
        base, n_m, cut, slots = [], 0, [0], []
        for r, m, a in units:
            base.append(n_m % 2 * half if _OVERLAP else 0)
            n_m += len(m) > 0
            cut.append(cut[-1] + len(r))
            if a is None:
                # Each chain row's store slot: its quota slot, or for a miss the ring row _fetch gives it (pool + base + its place in the unit's misses).
                p = pos[r]
                s = slot[p].astype(np.int64)
                o = ~inq[p]
                s[o] = store.pool + base[-1] + np.searchsorted(m, p[o])
                slots.append(s)
        # Every unit's rows and chain slots in one host array a layer, sliced on the device: an array the host makes a unit cost ~0.2 ms of GPU each.
        dev = mx.array(np.concatenate([u[0] for u in units] + slots).astype(np.uint32))
        dst_all = rows[dev[:cut[-1]]]
        src_all = x[dst_all // k]
        c0 = cut[-1]
        own = [tuple(b) for b in store.bufs]
        busy = [None, None]     # OVERLAP: per ring half, the output of the last launched unit that reads it

        def fetch(i):
            m = units[i][1]
            if not len(m):
                return None
            h = base[i] // half
            if busy[h] is not None:
                mx.eval(busy[h])
                busy[h] = None
            return self._fetch(unique_ids[m], store, base[i])

        ring, reads = fetch(0) if units else None, None
        for i, (r, m, a) in enumerate(units):
            dst, src0 = dst_all[cut[i]:cut[i + 1]], src_all[cut[i]:cut[i + 1]]
            if a is not None:
                ids = unique_ids[a:a + half]
                pg = pos[r] - a
                bufs, lo, _ = self._take_group(ids, store, ring=ring)
                h = ids[pg] - lo if lo is not None else pg
                y, reads = self._experts(src0, bufs, mx.array(h), False, h)[2], None
            else:
                y = self._experts(src0, own, dev[c0:c0 + len(r)])[2]
                c0 += len(r)
                reads = base[i] // half if len(m) else None
            out = _wide_out(out, n_rows, y)
            out[dst] = y
            if _OVERLAP:
                mx.async_eval(out)
                if reads is not None:
                    busy[reads] = y
                if i + 1 < len(units):
                    ring = fetch(i + 1)
            elif i + 1 < len(units):
                mx.eval(out)
                ring = fetch(i + 1)
        if reads is not None or any(b is not None for b in busy):
            # A unit that read ring rows is evaluated even when last, so no later copy lands under it.
            mx.eval(out)
        return out

    def _call_ahead(self, x, unique_ids, pos, rows, n_rows, store, reader):
        """_call_grouped's store path in a layer-major prefill: the reader thread reads each group of experts into a region of the released store (whole ranges of `reader.g` experts ahead of the GPU, or once this layer has routed its union, g experts a region), with the same gathers on the same rows and the same kernel, and every group evaluated before its region goes back to the reader."""
        k = n_rows // x.shape[0]
        n_exp = self._projs[0]._packed["n_experts"]
        sort = _takes_rhs(len(pos), n_exp)
        eids = unique_ids[pos]
        out = None
        groups = reader.post(self, unique_ids)
        ane = self._ane_experts(x, groups, eids, store) if ane_experts.ON else None
        for gi, ids in enumerate(groups):
            r0 = reader.take(self, gi)
            # eids is sorted: a group's rows are one run of it.
            i0 = int(np.searchsorted(eids, ids[0]))
            i1 = int(np.searchsorted(eids, ids[-1], "right"))
            if i1 > i0 and ane is not None:
                # MOE_ANE_EXPERTS: the group's largest experts on the ANE a group ahead, the rest here (ane_experts.Layer).
                r = np.searchsorted(ids, eids[i0:i1]).astype(eids.dtype)
                bufs = [tuple(t[r0:r0 + len(ids)] for t in b) for b in store.bufs]
                out = ane.group(out, n_rows, np.asarray(rows[i0:i1]) // k, r, rows[i0:i1], bufs)
            elif i1 > i0:
                r = np.searchsorted(ids, eids[i0:i1]).astype(eids.dtype)
                dst = rows[i0:i1]
                src0 = x[dst // k]
                pad = _rhs_rows(len(ids)) - (i1 - i0) if sort and not self._own_kernels else 0
                if pad > 0:
                    src0 = mx.concatenate(
                        [src0, mx.zeros((pad,) + src0.shape[1:], dtype=src0.dtype)])
                    r = np.concatenate([r, np.full(pad, r[-1], dtype=r.dtype)])
                rhs = mx.array(r)
                bufs = [tuple(t[r0:r0 + len(ids)] for t in b) for b in store.bufs]
                y = self._experts(src0, bufs, rhs, sort, r)[2]
                y = y[:i1 - i0] if pad > 0 else y
                out = _wide_out(out, n_rows, y)
                out[dst] = y
                mx.eval(out)
            reader.give(r0)
        if ane is not None:
            out = ane.drain(out, n_rows)
        return out

    def _ane_experts(self, x, groups, eids, store):
        """MOE_ANE_EXPERTS: this layer's ane_experts.Layer for an EXL3 pack, else None (ane_experts.engaged: Metal, worth it)."""
        if self._exl3 is None:
            return None
        cnts = []
        for ids in groups:
            i0, i1 = np.searchsorted(eids, ids[0]), np.searchsorted(eids, ids[-1], "right")
            cnts.append(np.bincount(np.searchsorted(ids, eids[i0:i1]), minlength=len(ids)))
        ps = ane_experts.engaged(self._exl3, x.shape[-1], store.bufs[0][0].shape[2] * 16, cnts)
        return None if ps is None else ane_experts.Layer(ps, x.reshape(-1, x.shape[-1]), x.dtype)

    def _call_rounds(self, x, unique_ids, pos):
        """Budget mode: a pass that is not a prefill (MemBudget.tick) whose union is wider than the layer's quota + ring half, or that is sorted, goes in rounds of what fits, each round gathering only its own rows unsorted into their rows of one flat output (gathering every row each round did the whole pass's GEMMs once per round, ~7-13 for a 100-token suffix), and off run-ahead the rounds are pipelined and alternate between the layer's ring half and the half of the layer below it, since serial 667 rounds of a ~3.5 ms read then a ~1.4 ms wait took 2.3 s of reads and 1.2 s of GPU never at the same time."""
        held = np.fromiter((int(e) in self._lru or int(e) in self._ring
                            for e in unique_ids), dtype=bool, count=len(unique_ids))
        order = np.concatenate((np.nonzero(held)[0], np.nonzero(~held)[0]))
        # One row of x per position (x unsorted, [..., 1, 1, H]) or per pair (the sorted copy, [pairs, 1, H]).
        flat = pos.reshape(-1)
        xr = x.reshape(-1, 1, x.shape[-1])
        per = flat.size // xr.shape[0]
        st = self._store
        halves = ((self._half,) if _DEFER_ACTIVE
                  else (self._half, (self._half - 1) % st.halves))
        # The quota is not double-buffered, so with a pool the rounds alternate quota + half and half alone, while at pool 0 every round is a full half as serial.
        y, r0, r, busy, gpu = None, 0, 0, (), None
        while True:
            h = halves[r % len(halves)]
            n = self._cap - sum(st.half_of(s) in (None, h) for s in busy)
            if n < 1:
                # The round on the GPU holds all the room this one has (its hits in this half and the whole quota), so wait for it.
                mx.eval(gpu)
                n, busy = self._cap, ()
            take = order[r0:r0 + n]
            slots, jobs = self._claim(unique_ids[take], half=h, busy=busy)
            STATS["rows_block"] += len(jobs)
            # CUDA: through the staging copy as decode's waves do (cuda_stage.py), since a host write moved the store's managed pages to RAM (-2.7 GB of MemAvailable over a 32-id turn).
            _read_wave(jobs, self)
            at = np.full(len(unique_ids), -1, dtype=np.int32)
            at[take] = slots
            row = at[flat]
            sel = np.nonzero(row >= 0)[0]
            dst = mx.array(sel)
            src0, rhs = xr[dst // per], mx.array(row[sel])
            o = self._experts(src0, [a[0] for a in _ARENAS[self._akey]], rhs, host=row[sel])[2]
            y = _wide_out(y, flat.size, o)
            y[dst] = o
            r0, r = r0 + len(take), r + 1
            if r0 == len(order):
                return y.reshape(pos.shape + y.shape[1:])
            if len(halves) == 1:
                mx.eval(y)
                continue
            mx.async_eval(y, o)
            if gpu is not None:
                mx.eval(gpu)
            gpu, busy = o, set(slots.tolist())

    def _call_defer(self, x, idx, indices, top=None, ok=None, rhs=None):
        """One layer with the residency decision taken on the GPU and checked one layer later: the same lookup as the sync path with no masking, so a non-resident row reads slot 0 and produces garbage that `ok` reports, nothing here blocks or touches the LRU until _defer_resolve runs with the indices on the host, and `ok`/`rhs` arrive given when the fused router kernel already took the same lookup behind the same _await_pending."""
        if self._akey not in _ARENAS:
            self._init_arena()
        # Whatever the depth prefetch started for this layer must have landed and been published before its slot table is read, which the layer boundary normally already did.
        self._await_pending()
        t = time.perf_counter()
        if rhs is None:
            slots = mx.take(self._slot_of, idx)
            ok = mx.all(slots >= 0)
            rhs = mx.maximum(slots, 0)
        prep = [a[0] for a in _ARENAS[self._akey]]
        # Budget mode never enters the sorted-rhs kernel, since stock runs a pass this narrow unsorted per row and a small store would otherwise cross _takes_rhs's line (4 pairs per slot), while the per-layer arena keeps the rule its replies were measured on.
        grouped = (indices.size >= DEFER_SORT_MIN
                   and not (self._budget is not None
                            and _takes_rhs(indices.size, prep[0][0].shape[0])))
        if grouped:
            # Pairs sorted by slot on the GPU (argsort, no sync), through the same _gather_sort/_scatter_unsort the wide path uses from SORT_MIN.
            from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort
            x, rhs, inv_order = _gather_sort(x, rhs)
        y = self._experts(x, prep, rhs, grouped)[2]
        if grouped:
            y = _scatter_unsort(y, inv_order, indices.shape)
        y = y.squeeze(-2)
        TIMERS["gemm"] += time.perf_counter() - t
        # The residency the router has just ranked against, kept with the record since a repair recomputes the layer later, possibly after a resize emptied the LRU, and its router must see exactly this.
        _DEFER_REC.append((self, idx, ok, indices, self._arena_gen,
                           list(self._lru), top))
        return y

    def _call_native(self, x, indices):
        """A native pass (_Native; T rows under MOE_NATIVE_MTP): the router's ids go to the host thread on the GPU's handshake and the gather waits for the slots the thread fills, so the host never blocks."""
        nat = _NATIVE
        if nat.cuda:
            # MLX CUDA (native/moe_pipe_cuda): the signal kernel posts the router's ids and the wait kernel hands back the gather rows once the agent has claimed and copied them in.
            nv, ids = nat.nv, indices.astype(mx.uint32)
            pre = None
            if self._hits_first:
                # MOE_PIPE_HITS_FIRST: the held experts' rows read before the post, so the claim cannot have touched them.
                pre = nv.resident(ids, self._slot_of)
                ids = mx.depends([ids], [pre])[0]
            dep = tick = nv.signal(ids, self._native_idx)
            deps = []
            if self._guess_next is not None:
                # MOE_PIPE_STAGING: the next layer's guess runs after this job's post, inside its park, and goes to the agent beside the job.
                lg, k = self._guess_next(mx.depends([x], [tick])[0])
                deps.append(nv.post_guess(tick, lg, k))
            early = self.__dict__.pop("_shared_early", None)
            if early is not None:
                self.__dict__["_shared_out"] = sh = early(mx.depends([x], [tick])[0])
                deps.append(sh)
            bufs = [a[0] for a in _ARENAS[self._akey]]
            if pre is not None:
                # The held experts' chain inside the park, the missed ones' after the wait, and each slot's output taken from its own chain.
                yh = exl3_experts.experts(mx.expand_dims(mx.depends([x], [tick])[0], (-2, -3)), bufs, pre, self._exl3,
                                          skip=True)
                deps.append(yh)
            if deps:
                dep = mx.depends([tick], deps)[0]
            rhs = nv.wait_rows(dep, ids, self._slot_of)
            if pre is None:
                y = self._experts(mx.expand_dims(x, (-2, -3)), bufs, rhs)[2].squeeze(-2)
            else:
                held = pre != _HITS_SKIP
                ym = exl3_experts.experts(mx.expand_dims(x, (-2, -3)), bufs, mx.where(held, _HITS_SKIP, rhs),
                                          self._exl3, skip=True)
                y = mx.where(held[..., None, None], yh, ym).squeeze(-2)
            t = time.perf_counter()
            mx.async_eval(y)
            nv.committed()
            t = time.perf_counter() - t
            if t > 1e-3:
                nat.holds += 1
                nat.hold_s += t
            return y
        v, nat.v = nat.v, nat.v + 2
        held = self.__dict__.pop("_ids32", None)
        fused = held is not None and held[0] is indices     # MOE_ROUTE_KERNEL's ids
        ids, kw = (held[1] if fused else indices.astype(mx.int32)), {}
        cpu = self.__dict__.get("_stage_cpu") if self._stage_next is not None else None
        if cpu is not None:
            # MOE_CPU_GUESS: the agent ranks the next layer's router on x once the GPU is let go.
            lam = getattr(cpu, "_lam", 0.0)
            kw = {"h": x, "lam": lam, "tau": _BIAS_BUDGET if lam > 0.0 else 0.0}
        if self._stage_next is not None and (cpu is None or _CPU_GUESS == "check"):
            # MOE_STAGING: the next layer's guess rides on this handshake, which an older moe_native does not take.
            owner = getattr(self._stage_next, "__self__", None)
            both = owner.guess_cat(x, ids) if fused and isinstance(owner, CacheBiasedMoE) else None
            if both is not None:
                kw["guess"] = both.shape[-1] - ids.shape[-1]
                ids = both
            else:
                guess = self._stage_next(x).astype(mx.int32)
                ids, kw["guess"] = mx.concatenate([ids, guess], axis=-1), guess.shape[-1]
        sig = nat.nv.signal_after(ids, nat.ev, v, nat.agent, self._native_idx, **kw)
        table = self._phys_of if nat.stage else self._slot_of
        dep, early = sig, self.__dict__.pop("_shared_early", None)
        if early is not None:
            # MOE_SHARED_EARLY: the block's shared expert between the handshake and the wait (mx.depends aliases, no copy), so the GPU computes it while the agent claims and reads.
            self.__dict__["_shared_out"] = sh = early(mx.depends([x], [sig])[0])
            dep = mx.depends([sig], [sh])[0]
        rhs = mx.take(nat.nv.wait_for(table, dep, nat.ev, v + 1), indices)
        x = mx.expand_dims(x, (-2, -3))
        y = self._experts(x, [a[0] for a in _ARENAS[self._akey]], rhs)[2].squeeze(-2)
        if not NO_ASYNC:
            mx.async_eval(y)
        return y

    def __call__(self, x, indices):
        from mlx_lm.models.switch_layers import _scatter_unsort

        lm = self.__dict__.get("_lm")
        if lm is not None:
            # A layer-major prefill's routing or combine phase (_lm_layer_call).
            return lm(x, indices)
        if _NATIVE is not None and _NATIVE.active:
            return self._call_native(x, indices)
        # P3: the fused router kernel left this layer's residency decision here, taken only for the very indices it was computed from, so a stale one is dropped and the ordinary take/all/maximum runs.
        fused = self.__dict__.pop("_fused_res", None)
        if fused is not None and fused[0] is not indices:
            fused = None
        x_in = x
        x = mx.expand_dims(x, (-2, -3))
        do_sort = indices.size >= SORT_MIN
        idx, inv_order = indices, None
        if do_sort:
            # mlx_lm's _gather_sort, keeping `order` and the unsorted rows, so the wide paths gather each group's rows through them and the sorted copy of x (T*k rows) is never evaluated there.
            flat = indices.flatten()
            order = mx.argsort(flat)
            inv_order = mx.argsort(order)
            x_rows = x.flatten(0, -3)
            x, idx = x_rows[order // indices.shape[-1]], flat[order]

        bud = self._budget
        if bud is not None and self._akey not in _ARENAS:
            # Budget mode: the stack's tick sized the stores before this pass's first router, and _cap is the quota plus ring half.
            self._init_arena()

        # The lookahead logits are built BEFORE the decode sync so that one mx.eval forces both, since doing it after cost a second GPU round trip a layer (0.52 s per 40 tokens, 14% of the run).
        top = None
        if self._next_glu is not None and self._look_m and not do_sort:
            g = self._next_gate(x_in).astype(mx.float32)
            top = mx.argpartition(-g, kth=self._look_m - 1,
                                  axis=-1)[..., :self._look_m]

        # Run-ahead: decide residency on the GPU and hand the routing to the layer loop, which checks it one layer late, an empty arena included, since inside the loop a layer may be running one ahead on a pass a miss will roll back and the ordinary path CLAIMS what no rollback undoes, while the deferred path claims only in _defer_resolve after the check.
        if _DEFER_ACTIVE and not do_sort and self._resident_ok():
            if self._defer_wide_once:
                self._defer_wide_once = False
            else:
                # `top` rides along because the host cannot read a prediction without a sync and the only sync this path pays is the check one layer later, where the prefetch issues from, and `idx is indices` here (not do_sort) makes the fused pair the right one for this gather.
                return self._call_defer(x, idx, indices, top,
                                        *(fused[1:] if fused is not None else (None, None)))

        if _GOV is not None:
            # Positions this pass actually covers (1 for a decode step, k+1 for a batched draft-verification pass), so the governor's clock advances in tokens rather than calls, with `indices.shape[-1]` top_k whatever leading dims it carries.
            _GOV.tick(do_sort, indices.size // indices.shape[-1])

        t = time.perf_counter()
        also = [top] if top is not None else None
        unique_ids, _, pos = _decode_indices(idx, also, memo=True, dev=not _CUDA_SLOTS)
        TIMERS["decode"] += time.perf_counter() - t
        STATS["reads"] += 1
        # What one step actually asks this layer for, observed rather than read out of a config, which keeps the governor's floor right on an architecture nobody has looked at and under MTP, since below it a step's experts cannot fit and the cache is dead weight that still holds its slots.
        # One position only, since `not do_sort` is NOT that test (do_sort is indices.size >= 64, so a prompt of two to seven positions passes it with a whole prompt's experts, 16 distinct against a top_k of 8), and a step that wants more than the arena holds is not a failure either because __call__ takes the wide path.
        if indices.size == indices.shape[-1] and len(unique_ids) > self._want:
            self._want = len(unique_ids)
        if _TRACK_USED:
            _USED.setdefault(self._akey, set()).update(int(e) for e in unique_ids)

        t = time.perf_counter()
        # Whatever the previous layer speculatively started must land before the arena is read or its slots reassigned.
        self._await_pending()
        layer_arena = self._resident_ok() and len(unique_ids) <= self._cap
        if bud is not None and (do_sort or not layer_arena):
            # Budget mode never allocates outside the store: a prefill (the tick's call, or wide enough for stock's sorted-rhs kernel) releases it and reads through it in groups, while anything else (a forced-rows or verify pass, a short turn suffix, a union wider than quota plus ring half) goes in rounds of what the quota and the ring hold and keeps the pool.
            STATS["need_block"] += 3 * len(unique_ids)
            STATS["miss_layers"] += 1
            # MOE_RESIDENT: every sorted pass, a prompt's short tail too, since the rounds' claims evict quota members the tier never gets (an 8k turn then read 35 GB).
            if do_sort and (bud.prefill or _takes_rhs(indices.size, bud.n_exp) or _RESIDENT):
                if not _RESIDENT:
                    self._store.release()   # MOE_RESIDENT's groups gather from the store instead (_take_group)
                y = self._call_grouped(x_rows, unique_ids, pos, order, None,
                                       indices.size, self._store)
                TIMERS["fetch"] += time.perf_counter() - t
                return mx.unflatten(y, 0, indices.shape).squeeze(-2)
            # A pass this narrow: the sorted copy of x and the unsort are small.
            y = self._call_rounds(x, unique_ids, pos)
            if do_sort:
                y = _scatter_unsort(y, inv_order, indices.shape)
            y = y.squeeze(-2)
            TIMERS["fetch"] += time.perf_counter() - t
            if not NO_ASYNC:
                mx.async_eval(y)
            return y
        if not layer_arena and do_sort and self._projs[0]._packed is not None:
            # The union does not fit the arena, so serve what IS resident from it and read only the rest, since the old wide path re-read the whole union from SSD, residents included (1183 rows a token against 498 on a small arena).
            # Not a prefill's union wider than a group, where the read-ahead has the whole layer and the residents' rows (thousands at 16k) went through the unsorted per-row kernel re-reading each expert's weights, while the grouped path gathers them sorted as mlx_lm's SwitchGLU does.
            split = (self._slot_np is not None
                     and not (_IN_PREFILL and len(unique_ids) > WIDE_GROUP)
                     and bool((self._slot_np[unique_ids] >= 0).any()))
            if split and self._akey not in _ARENAS:
                self._init_arena()      # parked (park_arenas): its residents back first
            if split or len(unique_ids) > WIDE_GROUP:
                STATS["need_block"] += 3 * len(unique_ids)
                # The union did not fit the arena at all, unconditionally a miss for DeferPolicy's signal, the same as defer_wide under the run-ahead path.
                STATS["miss_layers"] += 1
                # Rows land in `out` already unsorted: no concatenation and no unsort gather, each group T*k rows wide, made by the callee on its first write in gather_qmm's own output dtype (bf16 x with f16/f32 scales comes out f32).
                y = (self._call_split if split else self._call_grouped)(
                    x_rows, unique_ids, pos, order, None, indices.size)
                TIMERS["fetch"] += time.perf_counter() - t
                return mx.unflatten(y, 0, indices.shape).squeeze(-2)
        if layer_arena:
            slots, jobs = self._claim(unique_ids)
            prep = [(a[0], None, False, None) for a in _ARENAS[self._akey]]
            if _CUDA_SLOTS and not do_sort:
                # The claimed slots straight from the table the claim has just written (managed memory, read by the GPU in place) as uint32 rows, with no host-made array and no cast.
                rhs = mx.take(self._slot_u32, idx)
            else:
                rhs = mx.array(slots[pos])
            if jobs:
                STATS["miss_layers"] += 1
        else:
            prep = [p.prepare(unique_ids, pos, do_sort) for p in self._projs]
            jobs = [j for pr in prep for j in pr[3]]
            # Not layer_arena and not the split/grouped branch above: a step wider than the arena but still under WIDE_GROUP (or with no packed rows) served entirely off the arena, also a miss.
            STATS["miss_layers"] += 1
        STATS["rows_block"] += len(jobs)
        STATS["need_block"] += 3 * len(unique_ids)
        if layer_arena:
            _read_wave(jobs, self)
        else:
            _run_all(_do_read, jobs)
        TIMERS["fetch"] += time.perf_counter() - t

        if top is not None:
            t = time.perf_counter()
            pred = np.unique(np.asarray(top).reshape(-1))
            STATS["look_issued"] += len(pred)
            self._next_glu.warm(pred)
            TIMERS["look"] += time.perf_counter() - t

        t = time.perf_counter()
        if layer_arena:
            outs = self._experts(x, [pr[0] for pr in prep], rhs,
                                 host=slots[pos] if do_sort else None)
        elif self._own_kernels:
            # prepare() hands the three projections the same rows.
            outs = self._experts(x, [pr[0] for pr in prep], prep[0][1])
        else:
            outs = []
            for p, (bufs, r, srt, _) in zip(self._projs, prep):
                src = x if len(outs) < 2 else self.activation(outs[1], outs[0])
                outs.append(mx.gather_qmm(src, *bufs, rhs_indices=r, transpose=True,
                                          group_size=p.group_size, bits=p.bits,
                                          mode=p.mode, sorted_indices=srt))
        y = outs[2]
        TIMERS["gemm"] += time.perf_counter() - t

        if do_sort:
            y = _scatter_unsort(y, inv_order, indices.shape)
        y = y.squeeze(-2)

        # Reads are issued and left in flight, then async_eval hands this layer's gather_qmm to the GPU, so the SSD and the GPU work at once.
        if not NO_ASYNC:
            mx.async_eval(y)
        return y


def set_top_k(model, k):
    """Override the router's expert count across architectures (mlp.k on ERNIE, mlp.top_k on Qwen3 and Ling), which is the cheapest speed and quality dial since bytes a token scale linearly with it."""
    n = 0
    for layer in model.layers:
        mlp = _moe_of(layer)
        for name in ("k", "top_k", "num_experts_per_tok"):
            if hasattr(mlp, name):
                setattr(mlp, name, k)
                n += 1
                break
    return n


class _VocabView:
    """id -> token string straight from the Rust tokenizer for the streaming detokenizer, since mlx_lm's BPEStreamingDetokenizer copies the whole vocabulary into a Python list every turn (on a 248,077-token vocabulary a 24 MB temporary dict, ~100 ms and +20 MB held for the turn)."""

    def __init__(self, rust):
        self._t = rust
        self._n = rust.get_vocab_size(with_added_tokens=True)

    def __len__(self):
        return self._n

    def __getitem__(self, i):
        return self._t.id_to_token(i)


def _lean_detokenizer(tokenizer):
    """Swap mlx_lm's BPE streaming detokenizer for one that reads tokens from the Rust core instead of a per-turn Python copy, leaving any other detokenizer class as it is."""
    from mlx_lm.tokenizer_utils import BPEStreamingDetokenizer

    class LeanBPEDetokenizer(BPEStreamingDetokenizer):
        def __init__(self, tok):
            self.clean_spaces = tok.clean_up_tokenization_spaces
            self.tokenmap = _VocabView(tok.backend_tokenizer)
            self.reset()
            self.make_byte_decoder()

    if (getattr(tokenizer, "_detokenizer_class", None) is BPEStreamingDetokenizer
            and hasattr(tokenizer, "backend_tokenizer")):
        tokenizer._detokenizer_class = LeanBPEDetokenizer
    return tokenizer


def load_slim_aware(model_path, name):
    """Load a checkpoint whose expert tensors may live only in experts.bin, dropping load_weights' strict check because install() replaces every QuantizedSwitchLinear with a StreamingSwitchLinear before anything is evaluated, so the uninitialised parameters are discarded before they could be read."""
    idx = Path(model_path) / "model.safetensors.index.json"
    slim = ((Path(model_path) / "experts.bin").exists() and idx.exists()
            and not any(".switch_mlp." in k
                        for k in json.loads(idx.read_text())["weight_map"]))
    if not slim:
        model, tokenizer = mlx_lm.load(name, lazy=True)
        return model, _lean_detokenizer(tokenizer), False
    from mlx_lm.utils import load_model, load_tokenizer
    model, _ = load_model(Path(model_path), lazy=True, strict=False)
    return model, _lean_detokenizer(load_tokenizer(Path(model_path))), True


class StreamingPLE:
    """Rows of an n-gram embedding table read from ple.bin as they are asked for, since the table is 32 GB in 128 shards while a token touches only ~16 rows of ~100 bytes and its row ids depend only on token ids."""

    def __init__(self, fd, entries, group_size, bits, mode, dim, rows):
        # entries: shard index -> {offset, stride, parts}
        self.fd, self.entries = fd, entries
        self.group_size, self.bits, self.mode = group_size, bits, mode
        self.dim, self.rows = dim, rows
        e0 = next(iter(entries.values()))
        self.stride = e0["stride"]
        self.parts = e0["parts"]           # weight, scales, biases: bytes, shape, dtype
        assert [p["name"] for p in self.parts] == ["weight", "scales", "biases"], self.parts
        # grab: a list while _lm_start hashes a pass's ids (row ids kept, nothing read), ahead: (sorted ids, their rows, done) read on a thread for the pass that runs now (read_ahead), and next: one decode pass's rows being read (read_next).
        self.grab = self.ahead = self.next = None

    def _jobs(self, ids, buf):
        shard_of, row_of = ids // self.rows, ids % self.rows
        return [(self.fd, [memoryview(buf[i])],
                 self.entries[int(shard_of[i])]["offset"] + int(row_of[i]) * self.stride,
                 self.stride) for i in range(len(ids))]

    def _read(self, ids, buf):
        """Row ids[i] into buf[i], each row once, since n-gram rows repeat on 27-31% of a prompt's positions."""
        _run_all(_do_read, self._jobs(ids, buf))

    def read_next(self, flat):
        """MOE_PLE_AHEAD: start reading the rows of the next call's host row ids `flat` (its order) on the pool, which that call takes without looking at its own lazy ids."""
        ids, inv = np.unique(flat, return_inverse=True)
        rows = np.empty((len(ids), self.stride), dtype=np.uint8)
        self.next = (flat.size, rows, inv.reshape(-1), [_POOL.submit(_do_read, j) for j in self._jobs(ids, rows)])

    def read_ahead(self, gid):
        """Start reading the distinct rows of `gid` (a whole prefill pass's) on a thread, with __call__ taking its rows from there and a read that fails kept and raised by __call__ since the rows it left unread cannot be served."""
        ids = np.unique(np.asarray(gid).reshape(-1).astype(np.int64))
        buf = np.empty((len(ids), self.stride), dtype=np.uint8)
        done, err = threading.Event(), []

        def run():
            t = time.perf_counter()
            try:
                self._read(ids, buf)
            except BaseException as e:
                err.append(e)
            finally:
                TIMERS["ple_ahead"] += time.perf_counter() - t
                done.set()
        threading.Thread(target=run, daemon=True).start()
        self.ahead = (ids, buf, done, err)

    def __call__(self, gid):
        if self.grab is not None:
            self.grab.append(gid)
            return mx.zeros((*gid.shape, self.dim), dtype=mx.float32)
        nxt, self.next = self.next, None
        buf = None
        if nxt is not None and nxt[0] == gid.size:
            # MOE_PLE_AHEAD: the rows of the host's row ids, read since before layer 0, so gid stays lazy.
            t = time.perf_counter()
            n, rows, inv, futs = nxt
            for f in futs:
                f.result()
            buf = rows[inv]
            STATS["ple_next"] += 1
        else:
            flat = np.asarray(gid).reshape(-1).astype(np.int64)   # the model syncs here too
            n = flat.size
            t = time.perf_counter()
            if self.ahead is not None:
                ids, rows, done, err = self.ahead
                done.wait()
                if err:
                    raise err[0]
                j = np.minimum(np.searchsorted(ids, flat), len(ids) - 1)
                if len(ids) and (ids[j] == flat).all():
                    buf = rows[j]
            if buf is None:
                ids, inv = np.unique(flat, return_inverse=True)
                rows = np.empty((len(ids), self.stride), dtype=np.uint8)
                self._read(ids, rows)
                buf = rows[inv.reshape(-1)]
        TIMERS["ple"] += time.perf_counter() - t
        cols, out = 0, []
        for part in self.parts:
            b = part["bytes"]
            arr = buf[:, cols:cols + b].copy()
            cols += b
            np_dt, mx_dt = _ST_NP[part["dtype"]], _ST_MX[part["dtype"]]
            a = mx.array(arr.view(np_dt).reshape(n, *part["shape"]))
            out.append(a.view(mx_dt) if mx_dt == mx.bfloat16 else a)
        w, sc, bi = out
        emb = mx.dequantize(w, scales=sc, biases=bi, group_size=self.group_size,
                            bits=self.bits, mode=self.mode)
        return emb.astype(mx.float32).reshape(*gid.shape, self.dim)


def _ngram_rows(ng, hist):
    """NGramEmbedding.__call__'s row ids for the last position of `hist` (its n-gram context, then the pass's one id; int64 [1, ctx + 1]) on the host, where numpy's int64 wraps and its % takes the divisor's sign as MLX's do."""
    c = ng.__dict__.get("_np_hash")
    if c is None:
        c = ng.__dict__["_np_hash"] = tuple(np.asarray(ng[k]) for k in ("_mults", "_sizes", "_offsets"))
    mults, sizes, offs = c
    h = hist[0].astype(np.int64)
    pos = np.arange(h.size)
    eos = ng.eos_token_id
    # _shift_right: no shift crosses an EOS before the position.
    prev = np.concatenate([[-1], np.maximum.accumulate(np.where(h == eos, pos, -1))[:-1]])
    seg = pos - (prev + 1)
    shifted = [h] + [np.where((seg >= s) & (pos >= s), h[np.maximum(pos - s, 0)], eos)
                     for s in range(1, ng.ngram_size)]
    out = []
    for n in range(2, ng.ngram_size + 1):
        lo, hi = (n - 2) * ng.heads_per_ngram, (n - 1) * ng.heads_per_ngram
        mixed = shifted[0] * mults[0]
        for p in range(1, n):
            mixed = mixed ^ (shifted[p] * mults[p])
        out.append(mixed[-1] % sizes[lo:hi] + offs[lo:hi])
    return np.concatenate(out)


# MOE_PLE_AHEAD (on by default, 0 off): a native decode pass reads its PLE rows before layer 0 instead of on demand.
_PLE_AHEAD = os.environ.get("MOE_PLE_AHEAD", "1") != "0"


def _ple_decode_ahead(stack, inputs, cache):
    """MOE_PLE_AHEAD, before a native one-position pass is built: hash its n-gram rows on the host and start their reads, so the PLE layer takes them without waiting for layer 0 on the GPU, with the context known on the host from the pass before while the cache slot holds the same array; what _ple_decode_done needs, or None."""
    pls = getattr(stack, "ple_layers", None)
    if not (_PLE_AHEAD and pls) or cache is None:
        return None
    ngs = [stack.layers[i].ple.ple_embedding for i in pls]
    streamers = [ng.ngram_embedding.__dict__.get("_ple") for ng in ngs]
    pc = cache[pls[0]]
    if None in streamers or pc is None or pc.lengths is not None:
        return None
    ctx = stack.args.ngram_size - 1
    prev = pc[3]
    known = stack.__dict__.get("_ple_ctx")
    if prev is None:
        prev_np = np.full((1, ctx), ngs[0].eos_token_id, dtype=np.int64)
    elif known is not None and known[0] is prev:
        prev_np = known[1]
    else:
        prev_np = np.asarray(prev)      # after a prefill or a restored cache: the GPU is idle here
        STATS["ple_ctx_sync"] += 1
    hist = np.concatenate([prev_np.astype(np.int64), np.asarray(inputs).astype(np.int64)], axis=1)
    for ng, st in zip(ngs, streamers):
        st.read_next(_ngram_rows(ng, hist))
    return pc, prev, hist[:, -ctx:], streamers


def _ple_decode_done(stack, state, ok):
    """After the pass: rows it never took go, and if it ran, the context the model just put in the cache slot is kept with its host copy and evaluated behind the token."""
    if state is None:
        return
    pc, prev, ctx, streamers = state
    for st in streamers:
        st.next = None
    new = pc[3]
    if ok and new is not None and new is not prev:
        mx.async_eval(new)
        stack.__dict__["_ple_ctx"] = (new, ctx)


# qwen4_exp's NGramEmbedding (the PLE) keeps the checkpoint's hash constants in public attributes and hashes with private copies rebuilt from config["seed"], which the file we run defaults to 0 while the release's buffers are seed 1234 and its config names no seed, so every n-gram row id was wrong while the text stayed coherent, with the checkpoint's buffers the truth.
_NGRAM_CONSTS = (("layer_multipliers", "_mults"), ("ngram_heads_vocab_sizes", "_sizes"),
                 ("ngram_heads_offsets", "_offsets"))


def adopt_ngram_constants(model):
    """Hash with the checkpoint's n-gram constants rather than the config-rebuilt ones, returning how many arrays were replaced."""
    n = 0
    for name, mod in model.named_modules():
        if not all(k in mod for pair in _NGRAM_CONSTS for k in pair):
            continue
        for pub, priv in _NGRAM_CONSTS:
            ck = mod[pub]
            # They sit in parameters(), and an astype there would turn them into floats and the hash into noise.
            if ck.dtype != mx.int64:
                raise ValueError(f"{name}.{pub} is {ck.dtype}, not int64: the n-gram "
                                 f"hash constants were cast")
            if not mx.array_equal(ck, mod[priv]).item():
                print(f"  ple: {name}.{priv} rebuilt from the config differs from the "
                      f"checkpoint's {pub}; hashing with the checkpoint's", flush=True)
                # Evaluated here, since a lazy copy is bound to this thread's stream and another thread reading it raises.
                mod[priv] = mx.array(ck)
                mx.eval(mod[priv])
                n += 1
    return n


def install_ple(model, model_path, cfg_q):
    """Swap every sharded n-gram embedding for a StreamingPLE, if ple.bin exists."""
    adopt_ngram_constants(model)
    idx_p, bin_p = Path(model_path) / "ple_index.json", Path(model_path) / "ple.bin"
    if not (idx_p.exists() and bin_p.exists()):
        return 0
    index = json.loads(idx_p.read_text())
    fd = os.open(str(bin_p), os.O_RDONLY)
    # index keys: <prefix>.layers.N.<...>.ngram_embedding.shard_K
    by_suffix = {}
    for k, e in index.items():
        i = k.find(".layers.")
        by_suffix[k[i + 1:] if i >= 0 else k] = (k, e)
    done = 0
    seen = set()

    def walk(mod, path, depth=0):
        nonlocal done
        if depth > 12 or id(mod) in seen or not callable(getattr(mod, "children", None)):
            return
        seen.add(id(mod))
        if "shard_0" in mod and hasattr(mod, "n_shards") and hasattr(mod, "rows"):
            # Same normalisation as the index keys, from ".layers." onward, so the walk's root prefix does not have to agree with the checkpoint's.
            j = path.find(".layers.")
            suffix = path[j + 1:] if j >= 0 else path.lstrip(".")
            entries, q = {}, None
            for k in range(int(mod.n_shards)):
                hit = by_suffix.get(f"{suffix}.shard_{k}")
                if hit is None:
                    raise ValueError(f"ple.bin has no shard {k} for {suffix}")
                full, e = hit
                entries[k] = e
                q = q or cfg_q.get(full)
            first = mod["shard_0"]
            group_size = (q or {}).get("group_size", getattr(first, "group_size", 32))
            bits = (q or {}).get("bits", getattr(first, "bits", 4))
            mode = (q or {}).get("mode", getattr(first, "mode", "affine"))
            streamer = StreamingPLE(fd, entries, group_size, bits, mode, int(mod.dim), int(mod.rows))
            for k in range(int(mod.n_shards)):
                mod.pop(f"shard_{k}", None)     # never materialise the random init
            cls = type(mod)
            if not getattr(cls, "_ple_wrapped", False):
                orig = cls.__call__

                def call(self, gid):
                    f = self.__dict__.get("_ple")
                    return f(gid) if f is not None else orig(self, gid)
                cls.__call__ = call
                cls._ple_wrapped = True
            mod.__dict__["_ple"] = streamer
            done += 1
            return
        for name, c in mod.children().items():
            items = [(name, c)] if callable(getattr(c, "children", None)) else (
                [(f"{name}.{i}", x) for i, x in enumerate(c)] if isinstance(c, (list, tuple)) else
                [(f"{name}.{k}", x) for k, x in c.items()] if isinstance(c, dict) else [])
            for sub, x in items:
                walk(x, f"{path}.{sub}", depth + 1)

    walk(model, "")
    if done:
        note(f"  ple.bin: {done} n-gram embedding table(s) streamed by row, "
              f"{sum(e['n_experts'] for e in index.values()) / 1e6:.0f}M rows", flush=True)
    return done


# qwen4_exp's QSA indexer re-pools every key of the context on every call (8.4 ms of a 24k decode token for its 12 layers), so its cache keeps the pooled blocks and a call pools only those completed since, bit-identical to stock's (tests/test_qsa_pool.py), with any other write of `keys` dropping the pool.
_QSA_ENV = os.environ.get("MOE_QSA_POOL", "1")
# The stock source the pool replays (QSAIndexer.__call__, _IndexerCache and _AttnCache.trim of mlx_lm 0.32.0 plus the qwen4_exp PR at 2196836), and any other version stays stock.
_QSA_SRC = "bb46c0e28a06"
_QSA_STOCK = None       # (QSAIndexer.__call__, _IndexerCache) once installed
_QSA_SELECT = None      # the pooled indexer's lines up to the top-k once installed, for tests and benches
# MOE_QSA_STEP (Metal and CUDA, on): the raw indexer keys in a buffer grown 256 positions at a time, where stock copies the whole array each token (1.037x at 64k on a 16 GB Mac).
_QSA_STEP = BACKEND in ("metal", "cuda") and os.environ.get("MOE_QSA_STEP", "1") == "1"
# On CUDA the step also keeps the pooled blocks in an fp32 buffer, where the pool is otherwise concatenated in bf16 and cast to fp32 every token.
_QSA_POOL32 = _QSA_STEP and BACKEND == "cuda"
# MOE_QSA_ATTN (Metal and CUDA, on): a one-row decode pass attends to QSA's picks only, on Metal bit-equal to the masked SDPA over the whole context (_qsa_sdpa, 1.026x at 64k), on CUDA the picks gathered for MLX's own SDPA (_qsa_gather) where the install's measurement says it is faster.
_QSA_ATTN = BACKEND in ("metal", "cuda") and os.environ.get("MOE_QSA_ATTN", "1") == "1"
# The stock source the compact attention replays (qwen4_exp's Attention.__call__).
_QSA_ATTN_SRC = "128d02967ca5"
# MLX's two-pass vector SDPA, the kernel _qsa_sdpa replays: its header's sha1 and the MLX versions whose host dispatch was read from source.
_SDPA_HDR = "b9a50bf47996"
_SDPA_MLX = ("0.32.2",)


def _sdpa_parts(n_heads, n_kv_heads, n_keys):
    """How many partitions MLX 0.32.2's vector SDPA splits a one-query decode over n_keys keys into (sdpa_vector_2pass, scaled_dot_product_attention.cpp), or None where it runs its one-pass kernel."""
    devc = mx.device_info()["architecture"][-1]
    if not ((devc in "ds" and n_keys >= 1024) or (n_kv_heads < n_heads and n_keys >= 4096)):
        return None
    sims = n_heads // n_kv_heads
    if devc == "s":
        parts = 64
        if n_keys > 1024 and sims > 4:
            parts = (128 if n_keys <= 8192 else 256 if n_keys <= 32768
                     else 512 if n_keys <= 65536 else 1024)
    elif devc == "d":
        parts = 128
        if sims <= 2 and n_keys > 8192:
            parts = 256
        elif sims >= 6 and n_keys >= 16384:
            parts = 512 if n_keys < 65536 else 1024
    else:
        parts = 64 if sims >= 4 else 32
    return parts


# sdpa_vector_2pass_1's arithmetic over the picks only: stock walks partition b's keys b, b + P, ... in order and skips masked ones untouched, so the picked keys of b in ascending order give the same float ops.
_QSA_ACC = """
      const device T* kp = keys + i * kss;
      const device T* vp = values + i * vss;
      U score = 0;
      for (int e = 0; e < qk_per_thread; e++) {
        score += qv[e] * kp[e];
      }
      score = simd_sum(score);
      U new_max = max(max_score, score);
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);
      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;
      for (int e = 0; e < v_per_thread; e++) {
        o[e] = o[e] * factor + exp_score * vp[e];
      }
"""
_QSA_PASS1 = """
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;
  typedef float U;

  thread U qv[qk_per_thread];
  thread U o[v_per_thread] = {0};

  const uint simd_lid = thread_index_in_simdgroup;
  const int kv_head_idx = threadgroup_position_in_grid.x;
  const int block_idx = threadgroup_position_in_grid.z;
  const int q_head_idx = GQA * kv_head_idx + int(thread_position_in_threadgroup.y);
  const int own0 = prm[0];
  const int kv_len = prm[1];
  const int64_t kss = kc_strides[2];
  const int64_t vss = vc_strides[2];

  const device T* queries = qr + q_head_idx * qr_strides[1] + simd_lid * qk_per_thread;
  const device T* keys = kc + kv_head_idx * kc_strides[1] + simd_lid * qk_per_thread;
  const device T* values = vc + kv_head_idx * vc_strides[1] + simd_lid * v_per_thread;

  for (int i = 0; i < qk_per_thread; i++) {
    qv[i] = static_cast<U>(scl[0]) * queries[i];
  }

  U max_score = Limits<U>::finite_min;
  U sum_exp_score = 0;

  const int j = block_idx % CR;
  for (int c0 = 0; c0 < NSEL + 31; c0 += 32) {
    const int c = c0 + int(simd_lid);
    const int t = c < NSEL ? int(tb[c]) : 0;
    const int p = t * CR + j;
    const bool hit = c < NSEL && (p % P) == block_idx;
    uint bits = uint(static_cast<simd_vote::vote_t>(simd_ballot(hit)));
    while (bits) {
      const int i = simd_shuffle(p, ushort(ctz(bits)));
      bits &= bits - 1;
{ACC}    }
  }
  for (int i = own0; i < kv_len; i++) {
    if (i % P != block_idx) {
      continue;
    }
{ACC}  }

  const int row = q_head_idx * P + block_idx;
  if (simd_lid == 0) {
    sums[row] = sum_exp_score;
    maxs[row] = max_score;
  }
  for (int e = 0; e < v_per_thread; e++) {
    part[row * V + simd_lid * v_per_thread + e] = static_cast<T>(o[e]);
  }
"""

# sdpa_vector_2pass_2 as it is (one batch row, one query).
_QSA_PASS2 = """
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int elem_per_thread = D / BD;
  typedef float U;

  thread U o[elem_per_thread] = {0};
  threadgroup U outputs[BN * BD];

  const uint simd_gid = simdgroup_index_in_threadgroup;
  const uint simd_lid = thread_index_in_simdgroup;
  const int q_offset = threadgroup_position_in_grid.x;
  const device T* partials = part + q_offset * P * D + simd_gid * D + simd_lid * elem_per_thread;
  const device float* sp = sums + q_offset * P;
  const device float* mp = maxs + q_offset * P;
  device T* op = out + q_offset * D + simd_gid * elem_per_thread;

  U sum_exp_score = 0.0;
  U max_score = Limits<U>::finite_min;
  for (int b = 0; b < P / BN; ++b) {
    max_score = max(max_score, mp[simd_lid + BN * b]);
  }
  max_score = simd_max(max_score);
  for (int b = 0; b < P / BN; ++b) {
    U factor = fast::exp(mp[simd_lid + BN * b] - max_score);
    sum_exp_score += factor * sp[simd_lid + BN * b];
  }
  sum_exp_score = simd_sum(sum_exp_score);
  for (int b = 0; b < P / BN; ++b) {
    U factor = fast::exp(mp[simd_gid] - max_score);
    for (int i = 0; i < elem_per_thread; i++) {
      o[i] += factor * static_cast<U>(partials[i]);
    }
    mp += BN;
    sp += BN;
    partials += BN * D;
  }
  for (int i = 0; i < elem_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid]);
    o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (simd_lid == 0) {
    for (int i = 0; i < elem_per_thread; i++) {
      op[i] = static_cast<T>(o[i]);
    }
  }
"""
_QSA_KERNELS = {}


def _qsa_kernel(name):
    k = _QSA_KERNELS.get(name)
    if k is None:
        if name == "pass2":
            k = mx.fast.metal_kernel(name="qsa_sdpa_2", input_names=["part", "sums", "maxs"],
                                     output_names=["out"], source=_QSA_PASS2)
        else:
            k = mx.fast.metal_kernel(
                name="qsa_sdpa_1", input_names=["qr", "kc", "vc", "tb", "prm", "scl"],
                output_names=["part", "sums", "maxs"],
                source=_QSA_PASS1.replace("{ACC}", _QSA_ACC), ensure_row_contiguous=False)
        _QSA_KERNELS[name] = k
    return k


def _qsa_sdpa(q, k, v, tb, own0, kv_len, scale, cr, parts):
    """mx.fast.scaled_dot_product_attention(q, k, v, scale, mask) of one query whose boolean mask keeps the QSA blocks `tb` and the positions own0 .. kv_len - 1, bit for bit where MLX runs its two-pass kernel over `parts` partitions (_sdpa_parts)."""
    H, D, V, Hk = q.shape[1], q.shape[3], v.shape[3], k.shape[1]
    G = H // Hk
    part, sums, maxs = _qsa_kernel("pass1")(
        inputs=[q, k, v, tb, mx.array([own0, kv_len], mx.int32), mx.array([scale], mx.float32)],
        template=[("T", q.dtype), ("D", D), ("V", V), ("P", parts), ("CR", cr), ("GQA", G),
                  ("NSEL", tb.size)],
        grid=(32 * Hk, G, parts), threadgroup=(32, G, 1),
        output_shapes=[(H, parts, V), (H, parts), (H, parts)],
        output_dtypes=[q.dtype, mx.float32, mx.float32])
    return _qsa_kernel("pass2")(
        inputs=[part, sums, maxs], template=[("T", q.dtype), ("D", V), ("P", parts)],
        grid=(1024 * H, 1, 1), threadgroup=(1024, 1, 1),
        output_shapes=[(1, H, 1, V)], output_dtypes=[q.dtype])[0]


def _qsa_pool(ix, rope, raw, b0, b1):
    """Blocks [b0, b1) of the raw indexer keys, pooled by QSAIndexer's lines."""
    from mlx_lm.models.qwen4_exp import _rope_partial
    cr = ix.compress_ratio
    p = raw[:, b0 * cr:b1 * cr].reshape(raw.shape[0], b1 - b0, cr, ix.head_dim)
    p = ix.k_layernorm(p.astype(mx.float32).mean(axis=2).astype(raw.dtype))
    cos_k, sin_k = rope((mx.arange(b0, b1) * cr)[None, :])
    return _rope_partial(p, cos_k, sin_k)


def install_qsa_pool(model):
    """Keep the QSA indexer's pooled keys between calls for every cache made from now on, returning the indexers covered (0 without any, with MOE_QSA_POOL=0, or when mlx_lm's indexer is not the source replayed)."""
    global _QSA_STOCK, _QSA_SELECT
    q4 = sys.modules.get("mlx_lm.models.qwen4_exp")
    n = 0 if q4 is None else sum(isinstance(m, q4.QSAIndexer) for m in model.modules())
    if not n:
        return 0
    if _QSA_ENV == "0":
        note("  qsa indexer: stock (MOE_QSA_POOL=0)", flush=True)
        return 0
    if _QSA_STOCK is None:
        import inspect
        import math
        src = (inspect.getsource(q4.QSAIndexer.__call__) + inspect.getsource(q4._IndexerCache)
               + inspect.getsource(q4._AttnCache.trim))
        if hashlib.sha1(src.encode()).hexdigest()[:12] != _QSA_SRC:
            note("  qsa indexer: stock (mlx_lm's QSAIndexer is not the one the pool "
                  "replays)", flush=True)
            return 0
        stock, base = q4.QSAIndexer.__call__, q4._IndexerCache

        class PooledIndexerCache(base):
            """_IndexerCache that also holds `pool` (compress_ratio and the pooled blocks 0..m-1 of its keys, m at most the complete blocks), with MOE_QSA_STEP's keys a fresh view `buf[:, :n]` each read since a kept view would pin the buffer, and _QSA_POOL32's pool likewise a view `pbuf[:, :pm]`."""
            buf = None
            n = 0
            pbuf = None
            pm = 0
            pcr = 0

            @property
            def pool(self):
                if self.pbuf is not None:
                    return (self.pcr, self.pbuf[:, :self.pm])
                return self.__dict__.get("_pool")

            @pool.setter
            def pool(self, v):
                self.__dict__["_pool"] = v
                self.pbuf = None

            def pool32(self, cr, have, new):
                """The pooled blocks 0..have-1 extended by `new` in the fp32 step buffer, returned as a view."""
                m, buf = have + new.shape[1], self.pbuf
                if not have or buf is None or m > buf.shape[1]:
                    grown = mx.zeros((new.shape[0], (m + 64) // 64 * 64, new.shape[2]), mx.float32)
                    if have:
                        grown[:, :have] = buf[:, :have]
                    buf = grown
                buf[:, have:m] = new.astype(mx.float32)
                self.pool = None
                self.pbuf, self.pm, self.pcr = buf, m, cr
                return buf[:, :m]

            @property
            def keys(self):
                if self.buf is not None:
                    return self.buf[:, :self.n]
                return self.__dict__.get("_keys")

            @keys.setter
            def keys(self, v):
                # Any write but update's and trim's below starts the pool over.
                self.__dict__["_keys"] = v
                self.buf = None
                self.pool = None

            def update(self, k):
                pool, pbuf = self.pool, self.pbuf
                if not _QSA_STEP:
                    out = base.update(self, k)
                    # An append leaves every complete block.
                    if pbuf is not None:
                        self.pbuf = pbuf
                    else:
                        self.pool = pool
                    return out
                old, buf, S = self.keys, self.buf, k.shape[1]
                n = 0 if old is None else old.shape[1]
                if (buf is None or n + S > buf.shape[1] or buf.dtype != k.dtype
                        or buf.shape[0] != k.shape[0] or buf.shape[2] != k.shape[2]):
                    buf = mx.zeros((k.shape[0], (n + S + 255) // 256 * 256, k.shape[2]), k.dtype)
                    if n:
                        buf[:, :n] = old
                buf[:, n:n + S] = k
                self.buf, self.n = buf, n + S
                return self.keys

            def trim(self, size):
                pool, pbuf = self.pool, self.pbuf
                if self.buf is not None:
                    self.n = len(range(self.n)[:size])      # keys[:, :size]
                else:
                    base.trim(self, size)
                if pbuf is not None:
                    self.pbuf, self.pm = pbuf, min(pool[1].shape[1], max(size, 0) // pool[0])
                elif pool is not None:
                    self.pool = (pool[0], pool[1][:, :max(size, 0) // pool[0]])

        def select(self, x, rope, cache, offset):
            """call's lines up to the top-k: None when every key fits the budget, else (top, picked, n_complete, n_blocks, kv_len)."""
            B, S, _ = x.shape
            qk = self.index_qk_proj(x)
            split = self.n_heads * self.head_dim
            q = qk[..., :split].reshape(B, S, self.n_heads, self.head_dim)
            raw_k = cache.update(qk[..., split:].reshape(B, S, self.head_dim))
            kv_len = raw_k.shape[1]
            if kv_len <= self.token_budget:
                return None
            cr = self.compress_ratio
            n_blocks = kv_len // cr
            pool = cache.pool
            pooled = (pool[1] if pool is not None and pool[0] == cr
                      and pool[1].shape[0] == B and pool[1].shape[1] <= n_blocks else None)
            have = 0 if pooled is None else pooled.shape[1]
            if have < n_blocks:
                new = _qsa_pool(self, rope, raw_k, have, n_blocks)
                if _QSA_POOL32:
                    pooled = cache.pool32(cr, have, new)
                else:
                    pooled = new if pooled is None else mx.concatenate([pooled, new], axis=1)
                    cache.pool = (cr, pooled)

            # From here on QSAIndexer.__call__'s lines, without left padding.
            q_pos = q4._positions(offset, S)
            cos_q, sin_q = rope(q_pos)
            q = self.q_layernorm(q)
            q = q4._rope_partial(q, cos_q[:, :, None, :], sin_q[:, :, None, :])
            scores = mx.einsum("bshd,bnd->bsnh", q.astype(mx.float32),
                               pooled.astype(mx.float32))
            scores = mx.maximum(scores, 0).sum(axis=-1) / math.sqrt(self.head_dim)
            n_complete = mx.maximum(q_pos + 1, 0) // cr
            visible = mx.arange(n_blocks)[None, None, :] < n_complete[..., None]
            scores = mx.where(visible, scores, -mx.inf)
            k = min(self.block_topk, n_blocks)
            top = mx.argpartition(-scores, k - 1, axis=-1)[..., :k]
            picked = mx.take_along_axis(visible, top, axis=-1)
            return top, picked, n_complete, n_blocks, kv_len

        def call(self, x, rope, cache, offset, left_padding=None):
            if (left_padding is not None or not isinstance(offset, int)
                    or not isinstance(cache, PooledIndexerCache)):
                return stock(self, x, rope, cache, offset, left_padding)
            sel = select(self, x, rope, cache, offset)
            if sel is None:
                return None
            top, picked, n_complete, n_blocks, kv_len = sel
            B, S, _ = x.shape
            cr = self.compress_ratio
            keep_block = mx.put_along_axis(
                mx.zeros((B, S, n_blocks + 1), dtype=mx.bool_),
                mx.where(picked, top, n_blocks), mx.array(True), axis=-1)[..., :n_blocks]
            keep = mx.repeat(keep_block, cr, axis=-1)
            rest = kv_len - n_blocks * cr
            if rest:
                keep = mx.concatenate([keep, mx.zeros((B, S, rest), dtype=mx.bool_)], axis=-1)
            own_start = n_complete * cr
            tokens = mx.arange(kv_len)
            kv_pos = mx.arange(kv_len - S, kv_len)
            own = (tokens[None, None, :] >= own_start[..., None]) & (
                tokens[None, None, :] <= kv_pos[None, :, None])
            return (keep | own)[:, None]

        _QSA_STOCK, _QSA_SELECT = (stock, base), select
        q4.QSAIndexer.__call__ = call
        # _AttnCache and _BatchAttnCache make their indexer cache by this name.
        q4._IndexerCache = PooledIndexerCache
        if _QSA_ATTN:
            _install_qsa_attn(q4, select, PooledIndexerCache)
    picks = ""
    if _QSA_ATTN_ON and BACKEND == "cuda":
        for shape in {_qsa_shape(m) for m in model.modules() if isinstance(m, q4.Attention)}:
            if shape not in _QSA_SWITCH:
                try:
                    _QSA_SWITCH[shape] = _qsa_calibrate(shape)
                except Exception as e:
                    # A probe that cannot run leaves this shape on stock's attention rather than stopping the load.
                    _QSA_SWITCH[shape] = 2 ** 62
                    note(f"  qsa attention: not measured ({type(e).__name__}: {e}); stock's", flush=True)
            sw = _QSA_SWITCH[shape]
            picks += f" from {sw} keys" if sw < 2 ** 62 else " never (stock faster)"
    note(f"  qsa indexer: pooled keys kept between calls ({n} layers; "
          f"MOE_QSA_POOL=0 for stock){'; raw keys in a step buffer' if _QSA_STEP else ''}"
          f"{'; pooled keys in fp32' if _QSA_POOL32 else ''}"
          f"{'; decode attends to the picks only' + picks if _QSA_ATTN_ON else ''}", flush=True)
    if _KV_HOST and _QSA_ATTN_ON:
        mods = [m for m in model.modules() if isinstance(m, q4.Attention)]
        sw = max(_QSA_SWITCH[_qsa_shape(m)] for m in mods)
        a = max(mods, key=lambda m: m.n_kv_heads * m.head_dim)
        if sw < 2 ** 62:
            kv_host.install(q4, sw, a.indexer.block_topk, a.indexer.compress_ratio, a.n_kv_heads, a.head_dim,
                            a.k_norm.weight.dtype)
        else:
            note("  kv host: off (decode never attends to the picks here)", flush=True)
    elif os.environ.get("MOE_KV_HOST") == "1" and BACKEND != "cuda":
        note("  kv host: off (MOE_KV_HOST is CUDA only)", flush=True)
    return n


_QSA_ATTN_ON = False


def _install_qsa_attn(q4, select, pooled_cache):
    """qwen4_exp's Attention.__call__ with one change: a one-row decode pass whose QSA picks MLX would attend to with its two-pass kernel runs _qsa_sdpa over the picks instead of building the mask (bit-identical, tests/test_qsa_attn.py), on CUDA MLX's SDPA over the gathered picks from the measured length on (tests/test_qsa_cuda.py), and any other pass is stock's."""
    global _QSA_ATTN_ON
    import inspect
    stock = q4.Attention.__call__
    attn_cache = q4._AttnCache          # the class itself, which MOE_KV_HOST later replaces by a subclass
    src = hashlib.sha1(inspect.getsource(stock).encode()).hexdigest()[:12]
    if BACKEND == "cuda":
        if src != _QSA_ATTN_SRC:
            note(f"  qsa attention: stock (Attention {src}: not the one replayed)", flush=True)
            return
    else:
        hdr = os.path.join(os.path.dirname(mx.__file__), "include", "mlx", "backend", "metal",
                           "kernels", "sdpa_vector.h")
        try:
            with open(hdr, "rb") as f:
                hsum = hashlib.sha1(f.read()).hexdigest()[:12]
        except OSError:
            hsum = None
        if src != _QSA_ATTN_SRC or hsum != _SDPA_HDR or mx.__version__ not in _SDPA_MLX:
            note(f"  qsa attention: stock (Attention {src}, sdpa_vector.h {hsum}, "
                  f"MLX {mx.__version__}: not the ones replayed)", flush=True)
            return

    def proj(self, x, rope, offset):
        """Attention.__call__'s q, gate, k and v lines (before the cache)."""
        B, S, _ = x.shape
        q, gate = mx.split(self.q_proj(x).reshape(B, S, self.n_heads, -1), 2, axis=-1)
        gate = gate.reshape(B, S, -1)
        q = self.q_norm(q).transpose(0, 2, 1, 3)
        k = self.k_norm(self.k_proj(x).reshape(B, S, self.n_kv_heads, -1)).transpose(
            0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, S, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        cos, sin = rope(q4._positions(offset, S))
        cos, sin = cos[:, None], sin[:, None]
        return q4._rope_partial(q, cos, sin), gate, q4._rope_partial(k, cos, sin), v

    def decode_parts(self, x, mask, cache, idx_cache):
        """The two-pass SDPA's partitions for a one-row decode pass whose mask is QSA's top-k blocks and the tail after them, else None."""
        B, S, _ = x.shape
        ix = self.indexer
        if not (S == 1 and B == 1 and mask is None and isinstance(idx_cache, pooled_cache)
                and isinstance(cache, q4._AttnCache) and isinstance(cache.offset, int)):
            return None
        keys = idx_cache.keys
        kv_len = cache.offset + 1
        # The indexer's keys follow the KV, so every block is visible to this row, and the mask keeps the top-k blocks and the tail after them.
        if not (keys is not None and keys.shape[0] == 1 and keys.shape[1] + 1 == kv_len
                and kv_len > ix.token_budget):
            return None
        parts = _sdpa_parts(self.n_heads, self.n_kv_heads, kv_len)
        return None if parts is None or parts % ix.compress_ratio else parts

    def attn(self, x, rope, mask, cache, idx_cache):
        B, S, _ = x.shape
        parts = decode_parts(self, x, mask, cache, idx_cache)
        if parts is None:
            return stock(self, x, rope, mask, cache, idx_cache)
        ix = self.indexer
        offset = cache.offset
        kv_len = offset + 1
        top = select(ix, x, rope, idx_cache, offset)[0]
        # From here on Attention.__call__'s lines, the mask's SDPA replaced.
        q, gate, k, v = proj(self, x, rope, offset)
        k, v = cache.update_and_fetch(k, v)
        cr = ix.compress_ratio
        tb = mx.sort(top.reshape(-1).astype(mx.int32))
        out = _qsa_sdpa(q, k, v, tb, kv_len // cr * cr, kv_len, self.scale, cr, parts)
        out = out.transpose(0, 2, 1, 3).reshape(B, S, -1)
        return self.o_proj(out * mx.sigmoid(gate))

    def attn_cuda(self, x, rope, mask, cache, idx_cache):
        B, S, _ = x.shape
        ix = self.indexer
        if not (S == 1 and B == 1 and mask is None and isinstance(idx_cache, pooled_cache)
                and isinstance(cache, attn_cache) and isinstance(cache.offset, int)):
            return stock(self, x, rope, mask, cache, idx_cache)
        keys = idx_cache.keys
        offset = cache.offset
        kv_len = offset + 1
        # As decode_parts: every block is visible to this row, and from the measured length on the picks beat the mask.
        if not (keys is not None and keys.shape[0] == 1 and keys.shape[1] + 1 == kv_len
                and kv_len > ix.token_budget
                and kv_len >= _QSA_SWITCH.get(_qsa_shape(self), float("inf"))):
            return stock(self, x, rope, mask, cache, idx_cache)
        top = select(ix, x, rope, idx_cache, offset)[0]
        # From here on Attention.__call__'s lines, the masked SDPA over every key replaced by MLX's own SDPA over the picks.
        q, gate, k, v = proj(self, x, rope, offset)
        if _KV_HOST and hasattr(cache, "decode"):
            # The same rows from the cache's home and LRU (kv_host), or the lines below while it is in VRAM.
            tb = mx.sort(top.reshape(-1))
            g = cache.decode(k, v, tb, ix.compress_ratio)
            if g is None:
                k, v = cache.update_and_fetch(k, v)
                g = _qsa_gather(k, v, tb, ix.compress_ratio)
            kg, vg = g
        else:
            k, v = cache.update_and_fetch(k, v)
            kg, vg = _qsa_gather(k, v, mx.sort(top.reshape(-1)), ix.compress_ratio)
        out = mx.fast.scaled_dot_product_attention(q, kg, vg, scale=self.scale)
        out = out.transpose(0, 2, 1, 3).reshape(B, S, -1)
        return self.o_proj(out * mx.sigmoid(gate))

    q4.Attention.__call__ = attn_cuda if BACKEND == "cuda" else attn
    _QSA_ATTN_ON = True


# One warp copies one picked row (key or value, one KV head) out of the cache in place, in position order: the blocks tb, then the trailing partial block.
_QSA_GATHER = """
  const int lane = threadIdx.x & 31;
  const int row = int((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
  const int h = blockIdx.y;
  const int n = kc_shape[2];
  const int own0 = n / CR * CR;
  const int m = NSEL * CR + n - own0;
  if (row >= m) {
    return;
  }
  const int64_t pos = row < NSEL * CR ? int64_t(tb[row / CR]) * CR + row % CR
                                      : int64_t(own0 + row - NSEL * CR);
  const bool isv = blockIdx.z == 1;
  const T* src = isv ? vc + h * vc_strides[1] + pos * vc_strides[2]
                     : kc + h * kc_strides[1] + pos * kc_strides[2];
  const int64_t s3 = isv ? vc_strides[3] : kc_strides[3];
  T* dst = (isv ? vg : kg) + (int64_t(h) * m + row) * D;
  constexpr int VEC = 16 / sizeof(T);
  if (D % VEC == 0 && s3 == 1 && (reinterpret_cast<uintptr_t>(src) & 15) == 0) {
    for (int c = lane; c < D / VEC; c += 32) {
      reinterpret_cast<uint4*>(dst)[c] = reinterpret_cast<const uint4*>(src)[c];
    }
  } else {
    for (int c = lane; c < D; c += 32) {
      dst[c] = src[c * s3];
    }
  }
"""


def _qsa_gather(k, v, tb, cr):
    """K and V [1, Hk, m, D] of the sorted picked blocks tb and the trailing partial block, copied out of the cache views k and v by one kernel, since MLX's take returns rows its SDPA fallback copies once per query head."""
    kern = _QSA_KERNELS.get("gather_cuda")
    if kern is None:
        kern = _QSA_KERNELS["gather_cuda"] = mx.fast.cuda_kernel(
            name="qsa_gather", input_names=["kc", "vc", "tb"], output_names=["kg", "vg"],
            source=_QSA_GATHER, ensure_row_contiguous=False)
    _, hk, n, d = k.shape
    m = tb.size * cr + n % cr
    return kern(inputs=[k, v, tb], template=[("T", k.dtype), ("D", d), ("CR", cr), ("NSEL", tb.size)],
                grid=(32 * m, hk, 2), threadgroup=(256, 1, 1),
                output_shapes=[(1, hk, m, d)] * 2, output_dtypes=[k.dtype] * 2)


# CUDA: (heads, KV heads, head dim, block, picked blocks, budget, dtype) -> the shortest context whose one-row decode attends to the picks, from _qsa_calibrate.
_QSA_SWITCH = {}
_QSA_CALIB_V = 1


def _qsa_shape(attn):
    ix = attn.indexer
    return (attn.n_heads, attn.n_kv_heads, attn.head_dim, ix.compress_ratio, ix.block_topk,
            ix.token_budget, str(attn.k_norm.weight.dtype))


def _qsa_time(shape, n, reps=7, layers=4, calls=12):
    """Milliseconds of `calls` one-row decode attentions over n keys, stock's (the indexer's mask from the picks, then the masked SDPA) and the picks' (sort, _qsa_gather, SDPA), cycling over `layers` layers' own random keys."""
    h, hk, d, cr, topk, _, dt = shape
    dtype = getattr(mx, dt.split(".")[-1])
    cap = -(-n // 256) * 256
    nb = n // cr
    data = []
    for i in range(layers):
        # Own keys, so the engine's global random state is left as it was.
        ks = mx.random.split(mx.random.key(n * layers + i), 4)
        q = mx.random.normal((1, h, 1, d), key=ks[0]).astype(dtype)
        kb = mx.random.normal((1, hk, cap, d), key=ks[1]).astype(dtype)
        vb = mx.random.normal((1, hk, cap, d), key=ks[2]).astype(dtype)
        top = mx.random.permutation(nb, key=ks[3])[:topk].astype(mx.uint32).reshape(1, 1, topk)
        mx.eval(q, kb, vb, top)
        data.append((q, kb, vb, top))
    scale = d ** -0.5

    def stock():
        out = []
        for c in range(calls):
            q, kb, vb, top = data[c % layers]
            # QSAIndexer's mask from the picks of a one-row decode pass (every block visible).
            keep = mx.repeat(mx.put_along_axis(mx.zeros((1, 1, nb + 1), dtype=mx.bool_), top,
                                               mx.array(True), axis=-1)[..., :nb], cr, axis=-1)
            if n % cr:
                keep = mx.concatenate([keep, mx.zeros((1, 1, n % cr), dtype=mx.bool_)], axis=-1)
            mask = (keep | (mx.arange(n) >= nb * cr)[None, None])[:, None]
            out.append(mx.fast.scaled_dot_product_attention(q, kb[..., :n, :], vb[..., :n, :],
                                                            scale=scale, mask=mask))
        return out

    def picks():
        out = []
        for c in range(calls):
            q, kb, vb, top = data[c % layers]
            kg, vg = _qsa_gather(kb[..., :n, :], vb[..., :n, :], mx.sort(top.reshape(-1)), cr)
            out.append(mx.fast.scaled_dot_product_attention(q, kg, vg, scale=scale))
        return out

    res = []
    for fn in (stock, picks):
        for _ in range(2):
            mx.eval(fn())
        t = []
        for _ in range(reps):
            t0 = time.perf_counter()
            mx.eval(fn())
            t.append((time.perf_counter() - t0) * 1e3)
        res.append(sorted(t)[reps // 2])
    del data
    mx.clear_cache()
    return res


def _qsa_calibrate(shape, top_n=32768):
    """The shortest context from which the picks beat stock's masked SDPA at this shape on this device: both timed at doubling lengths from the budget until the picks win twice running (or top_n), the crossing interpolated, and kept in ~/.cache/moe-ssd-offload."""
    key = json.dumps([_QSA_CALIB_V, mx.device_info().get("device_name", ""), mx.__version__,
                      hashlib.sha1(_QSA_GATHER.encode()).hexdigest()[:12], list(shape)])
    path = Path.home() / ".cache" / "moe-ssd-offload" / \
        f"qsaattn-{hashlib.sha1(key.encode()).hexdigest()[:16]}.json"
    try:
        return int(json.loads(path.read_text())["switch"])
    except (OSError, ValueError, KeyError, TypeError):
        pass
    budget = shape[5]
    pts, wins, n = [], [], budget
    while n <= top_n and wins[-2:] != [True, True]:
        s, p = _qsa_time(shape, n + shape[3] - 1)     # a trailing partial block, as most decode steps have
        pts.append((n, s, p))
        wins.append(p < s)
        n *= 2
    if not wins[-1]:
        switch = 2 ** 62        # stock up to top_n: never the picks
    else:
        # The start of the last winning run, moved back to where the two times cross.
        i = len(wins) - 1
        while i and wins[i - 1]:
            i -= 1
        if i == 0:
            switch = budget + 1
        else:
            (na, sa, pa), (nw, sw, pw) = pts[i - 1], pts[i]
            switch = int(na + (nw - na) * (pa - sa) / ((sw - pw) + (pa - sa)))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"switch": switch, "points": pts, "key": key}))
        tmp.replace(path)
    except OSError as e:
        print(f"  qsa attention: could not save its calibration to {path}: {e}", flush=True)
    return switch


class StreamingEmbedding:
    """embed_tokens rows read from the checkpoint's own safetensors shards as a forward asks for them, dequantized as nn.QuantizedEmbedding does and read through the SafetensorsMap install() already opened for expert rows."""

    def __init__(self, st, key, group_size, bits, mode, has_biases):
        self.st, self.key = st, key
        self.group_size, self.bits, self.mode = group_size, bits, mode
        self.has_biases = has_biases

    def __call__(self, gid):
        ids = np.asarray(gid).reshape(-1)  # the model syncs here too (StreamingPLE)
        # Read each DISTINCT id once, as prepare()'s unique_ids does for expert rows, since a prompt repeats far more than it varies and dequantize is per-row, which is bit-identical to dequantizing every position and faster even on all-distinct ids because pread over np.unique's sorted ids has better locality (5622 positions 526 -> 82 ms, 6000 distinct 370 -> 185 ms, one decode row 1.4 -> 0.5 ms).
        u, inv = np.unique(ids, return_inverse=True)
        w = _read_experts(self.st, f"{self.key}.weight", u)
        sc = _read_experts(self.st, f"{self.key}.scales", u)
        bi = _read_experts(self.st, f"{self.key}.biases", u) if self.has_biases else None
        # No astype here, since nn.QuantizedEmbedding.__call__ returns whatever dtype mx.dequantize gives back (typically fp16/bf16) with no cast, which is what the bit-identity check compares against.
        emb = mx.dequantize(w, scales=sc, biases=bi, group_size=self.group_size,
                            bits=self.bits, mode=self.mode)
        emb = emb[mx.array(inv)]
        return emb.reshape(*gid.shape, emb.shape[-1])


def install_embed(model, st):
    """Swap a streamed pack's embed_tokens for a StreamingEmbedding and free its resident weight/scales/biases, only for a quantized table whose bytes `st` can still reach and never for a tied checkpoint, whose Model.__call__ needs embed_tokens.as_linear for the lm_head projection."""
    lm = getattr(model, "language_model", model)
    if "lm_head" not in lm:
        return 0
    stack = _layer_stack_of(model)
    emb = getattr(stack, "embed_tokens", None) if stack is not None else None
    if emb is None or "weight" not in emb or not hasattr(emb, "group_size"):
        return 0
    hits = [k[:-len(".weight")] for k in st._index
            if k == "embed_tokens.weight" or k.endswith(".embed_tokens.weight")]
    if len(hits) != 1:
        return 0
    key = hits[0]
    has_biases = f"{key}.biases" in st
    freed = sum(int(emb[p].nbytes) for p in ("weight", "scales", "biases") if p in emb)
    streamer = StreamingEmbedding(st, key, emb.group_size, emb.bits, emb.mode, has_biases)
    cls = type(emb)
    if not getattr(cls, "_embed_wrapped", False):
        orig = cls.__call__

        def call(self, gid):
            f = self.__dict__.get("_stream")
            return f(gid) if f is not None else orig(self, gid)
        cls.__call__ = call
        cls._embed_wrapped = True
    for p in ("weight", "scales", "biases"):
        emb.pop(p, None)     # last reference, frees the already-loaded table
    emb.__dict__["_stream"] = streamer
    # Per-row read on Qwen3.6: 1024 B weight (2048 dims x 4 bit) + 64 B scales + 64 B biases (32 groups of 64, fp16) = 1152 B a token.
    note(f"  embed_tokens: streamed by row from '{key}', "
          f"{freed / 1e6:.0f}MB freed", flush=True)
    return freed


def load_packed(model_path):
    """Index written by repack.py, if the experts have been laid out contiguously."""
    idx = Path(model_path) / "experts_index.json"
    binf = Path(model_path) / "experts.bin"
    if not (idx.exists() and binf.exists()):
        return None
    return {"index": json.loads(idx.read_text()),
            "fd": open(binf, "rb")}


def set_cache_bias(model, lam):
    """Change lambda on a live model and return how many layers were touched, since rebuilding the model per lambda starts each instance with a cold arena and only layers wrapped at install time can be set."""
    n = 0
    for layer in model.layers:
        mlp = _moe_of(layer)
        if isinstance(mlp, CacheBiasedMoE):
            mlp._lam = lam
            n += 1
    return n


@contextlib.contextmanager
def bias_off(model):
    """Route with the model's own router (cache bias 0) inside the block, so a front built under the bias does not carry the routing of whatever the building process's arena held, covering both the CacheBiasedMoE wrapper and the correction-term hook."""
    wraps, hooks = [], []
    if model is not None:
        wraps = [m for m in map(_moe_of, model.layers) if isinstance(m, CacheBiasedMoE)]
        hooks = [g._bias_hook for g in glus_of(model) if getattr(g, "_bias_hook", None)]
    lams = [w._lam for w in wraps]
    hook_lams = [h.lam for h in hooks]
    for w in wraps:
        w._lam = 0.0
    for h in hooks:
        h.lam = 0.0
        h.refresh()
    try:
        yield
    finally:
        for w, lam in zip(wraps, lams):
            w._lam = lam
        for h, lam in zip(hooks, hook_lams):
            h.lam = lam
            h.refresh()


def _ranker_of(mlp):
    """A callable giving an array whose largest entries are the preferred experts, since architectures disagree on what the gate returns (Qwen3.5/3.6's mlp.gate(x) is the raw logits, bailing_moe's already returns (inds, scores) with its logits one level down at gate.gate_proj) and the prediction may over-predict but must not under-predict."""
    if isinstance(mlp, CacheBiasedMoE):
        return mlp.ranking
    gate = getattr(mlp, "gate", None)
    inner = getattr(gate, "gate_proj", None)
    if inner is None:
        return gate
    bias = getattr(gate, "expert_bias", None)
    if bias is None:
        return inner
    return lambda x: mx.sigmoid(inner(x).astype(mx.float32)) + bias


def _best_first(rank, inds):
    """`inds` ordered by `rank`, highest first."""
    order = mx.argsort(-mx.take_along_axis(rank, inds, axis=-1), axis=-1)
    return mx.take_along_axis(inds, order, axis=-1)


def _stage_off(why):
    """MOE_STAGING asked for where it cannot work: no rows charged or allocated, no guesses computed, the agent never told."""
    global _STAGE, _STAGING_ROWS
    if _STAGE != "0":
        note(f"  staging rows: off, {why}", flush=True)
    _STAGE, _STAGING_ROWS = "0", 0
    if _NATIVE is not None:
        _NATIVE.stage = False
        for g in _NATIVE.glus:
            g._stage_next = None


def _stage_guess(mlp, k):
    """Layer L's selection rule as a function of layer L-1's MoE input, best first: MOE_STAGING's guess."""
    if isinstance(mlp, CacheBiasedMoE):
        return mlp.guess
    gate = mlp.gate

    def guess(x):
        lg = gate(x)
        return _best_first(lg, mx.argpartition(lg, kth=-k, axis=-1)[..., -k:])
    return guess


def _pipe_guess(gate, k):
    """MOE_PIPE_STAGING's guess of a layer's experts from the layer below's MoE input: its gate's f32 logits, as the block's own router takes them, and its top-k."""
    def guess(x):
        lg = gate(x.astype(mx.float32))
        return (lg if lg.dtype == mx.float32 else lg.astype(mx.float32)), k
    return guess


def _glu_of(layer):
    """The streaming block, whether or not the MoE block is wrapped for bias."""
    mlp = _moe_of(layer)
    return (mlp.block if isinstance(mlp, CacheBiasedMoE) else mlp).switch_mlp


def _defer_resolve(glu, idx, ok, indices, claimed, gen, keys, top=None):
    """The check plus the depth prefetch it pays the sync for: `top` is layer L's prediction of what layer L+look_depth will want, forced by the same submit as L's routing so reading it costs no round trip, and it is issued AFTER the check's own blocking reads so a miss never waits behind a speculation, with look_depth at least 2 under run-ahead and a re-run (`claimed`) predicting nothing."""
    missed = _defer_check(glu, idx, ok, indices, claimed, gen, keys)
    if top is not None and not claimed:
        t = time.perf_counter()
        pred = np.unique(np.asarray(top).reshape(-1))
        STATS["look_issued"] += len(pred)
        glu._next_glu.warm(pred)
        TIMERS["look"] += time.perf_counter() - t
    return missed


def _defer_check(glu, idx, ok, indices, claimed, gen, keys):
    """Learn what the GPU decided a layer ago, true meaning the layer was wrong, with the eval waiting for this layer's routing while the GPU is busy on the next one and `claimed` marking the re-run of a repaired layer whose slots were assigned already."""
    if claimed and not _DEFER_CHECK and not CLAIMED_WAIT:
        # Same early return as below, without standing still for the GPU first.
        return False
    t = time.perf_counter()
    mx.eval(idx, ok)
    TIMERS["sync"] += time.perf_counter() - t
    if claimed and not _DEFER_CHECK:
        # The re-run of a repaired layer cannot miss unless the slot table and the LRU have drifted apart, which the acceptance runs check with _DEFER_CHECK on, while in production the wait for its flag (~0.6 ms x rollbacks) is a round trip the next layer's flag pays anyway.
        return False
    STATS["reads"] += 1
    hit = bool(ok.item())
    if gen != glu._arena_gen:
        # The arena was resized or rebuilt after this gather read its slot table, so whatever the flag says, the slots it used no longer exist and the LRU is empty: a miss, claimed afresh.
        STATS["defer_regen"] += 1
        hit, claimed = False, False
    if claimed:
        if not hit:
            ids = [int(e) for e in np.unique(np.asarray(idx).reshape(-1))]
            slots_gpu = np.asarray(mx.take(glu._slot_of, mx.array(ids, dtype=mx.int32))).tolist()
            raise RuntimeError(
                "defer: a repaired layer still misses; the GPU slot table and "
                f"the host LRU disagree. experts {ids}; in host LRU "
                f"{[e in glu._lru for e in ids]}; slot_np {[int(glu._slot_np[e]) for e in ids]}; "
                f"slot_of now {slots_gpu}; gen {gen}/{glu._arena_gen}; lru size {len(glu._lru)}/{glu._cap}")
        return False
    unique_ids, _, _ = _decode_indices(idx, None, memo=False)
    if indices.size == indices.shape[-1] and len(unique_ids) > glu._want:
        glu._want = len(unique_ids)
    if _TRACK_USED:
        _USED.setdefault(glu._akey, set()).update(int(e) for e in unique_ids)
    glu._await_pending()
    if len(unique_ids) > glu._cap or not glu._resident_ok():
        # The step is wider than the arena (a governor shrink, or a prompt-shaped step), so the speculative gather cannot have been right and the ordinary path serves this with a wide read on the re-run.
        glu._defer_wide_once = True
        STATS["defer_wide"] += 1
        return True
    if hit:
        slots, jobs = glu._claim(unique_ids)
        if jobs:
            raise RuntimeError("defer: the GPU saw every expert resident and the "
                               "host LRU did not: _slot_of and _lru have "
                               "drifted apart")
        return False
    # A miss: the layer is recomputed after this claim and its router must see the residency it saw the first time, so keys are taken before the claim and the bias hook is kept from refreshing until the loop lifts the hold.
    hook = glu._bias_hook
    glu._defer_hold = _Hold(keys, hook)
    glu._bias_hook = None
    try:
        slots, jobs = glu._claim(unique_ids)
    finally:
        glu._bias_hook = hook
    if not jobs:
        raise RuntimeError("defer: the GPU saw a miss the host LRU does not: "
                           "_slot_of and _lru have drifted apart")
    t = time.perf_counter()
    STATS["rows_block"] += len(jobs)
    STATS["need_block"] += 3 * len(unique_ids)
    _read_wave(jobs, glu)
    TIMERS["fetch"] += time.perf_counter() - t
    return True


class _Hold:
    __slots__ = ("keys", "hook")

    def __init__(self, keys, hook):
        self.keys, self.hook = keys, hook


def _defer_release(glu):
    """Lift the pre-claim residency view once the repaired layer has run."""
    hold = getattr(glu, "_defer_hold", None)
    if hold is None:
        return
    glu._defer_hold = None
    if hold.hook is not None:
        hold.hook.refresh()


def _layer_stack_of(model):
    """The module that OWNS the decoder layer list, the one whose __call__ runs the loop, found by walking because the outer Model classes forward `layers` to the inner stack and a Qwen3.5 VL checkpoint nests it two levels down."""
    seen = set()

    def walk(mod, depth=0):
        if depth > 12 or id(mod) in seen or not callable(getattr(mod, "children", None)):
            return None
        seen.add(id(mod))
        if "layers" in mod and isinstance(mod["layers"], list) and mod["layers"]:
            return mod
        for c in _module_items(mod):
            found = walk(c, depth + 1)
            if found is not None:
                return found
        return None

    return walk(model)


def _caches_in(args, kwargs):
    """The cache objects a layer was handed whatever position they sit at, skipping one that is an attribute of another cache in the same call (qwen4_exp's attention cache and its `.indexer`) since the trim of the cache holding it covers it."""
    from mlx_lm.models.cache import _BaseCache
    found = [a for a in list(args) + list(kwargs.values()) if isinstance(a, _BaseCache)]
    owned = {id(v) for c in found for v in vars(c).values() if isinstance(v, _BaseCache)}
    return [c for c in found if id(c) not in owned]


def _snapshot_caches(caches):
    snaps = []
    for c in caches:
        if isinstance(c, ArraysCache):
            snaps.append(list(c.cache))
        elif c.is_trimmable():
            snaps.append(None)                  # give the position back
        else:
            snaps.append(("state", c.state))    # references, not copies
    return snaps


def _restore_caches(caches, snaps, n=1):
    for c, s in zip(caches, snaps):
        if isinstance(s, list):
            # A COPY: the layer about to re-run writes into c.cache in place (cache[0] = ...) and the snapshot has to survive that as the pending layer's fallback if its re-run misses again, since aliasing it made the second rollback restore the post-run state (the re-run's input differed, |x| 844 vs 840, two of eight experts changed).
            c.cache = list(s)
        elif s is None:
            got = c.trim(n)
            assert got == n, f"{type(c).__name__} gave back {got} of {n} positions"
        else:
            c.state = s[1]


class _DeferState:
    """Per-forward bookkeeping of the run-ahead, shared by the layer wrappers."""
    __slots__ = ("pending", "ticks")

    def __init__(self):
        self.pending = None     # (layer, args, kwargs, snaps, rec) awaiting its check
        self.ticks = 0


_DEFER_STATE = _DeferState()


class _Probe:
    """One A/B decision inside DeferPolicy: WINDOW tokens in whichever mode runs, then WINDOW in the other, both as a rate so a dropped sample stays comparable."""
    __slots__ = ("mode_a", "m_probe", "phase", "elapsed", "n", "rate_a")

    def __init__(self, mode_a, m_probe):
        self.mode_a = mode_a      # the mode already running when the probe opened
        self.m_probe = m_probe    # the EMA level this decision resolves
        self.phase = 0            # 0: timing mode_a, 1: timing the other mode
        self.elapsed = 0.0
        self.n = 0
        self.rate_a = None


class DeferPolicy:
    """Picks sync vs. defer per token from one signal on both paths, MoE layers a token served from outside the arena, normalized by T for a batched pass, with lo/hi bracketing the crossing in misses where defer is trusted below lo and sync above hi."""
    ALPHA = 1.0 / 16     # EMA smoothing on the per-token miss count, ~16-token memory
    WINDOW = 64          # positions per half of a probe (accumulated across
                         # calls, since a draft pass of T counts T at once like T single-position steps, on ArenaGovernor's own period.
    BAND = 1.4           # re-probe only once the unknown zone widens past BAND**2
    NOISE = 0.03         # a probe within 3% of even changes nothing
    COOLDOWN = 64        # tokens held off after an arena resize or a fresh
                         # decision, long enough for the arena's own re-warm (~30 tokens, ArenaGovernor.resize) or the probe's aftermath to stop looking like new signal.
    GAP = 4              # a pass over GAP x the recent pace is a pause, not
                         # decode: the two modes measured at most ~3x apart (2 GB: 309 vs 105 ms a token) while a turn boundary is 10-50x a decode step.
    GAP_CAP = 10.0       # seconds: longer is never one decode-shaped pass,
                         # whatever the pace says, since the slowest measured, five positions run ahead at 2 GB, took 0.7 s.

    def __init__(self):
        self.m = 0.0
        # Seeded at 1, not 0: defer's only cost is c per rollback and its only saving B a token, both positive, so below one rollback a token it cannot lose whatever c turns out to be here (every c measured is 0.6-1.9 ms and every B 6-7 ms on Qwen1.5, larger on bigger models).
        self.lo, self.hi = 1.0, float("inf")
        self.probe = None
        self.history = []      # (token, mode, m, lo, hi) on every mode change
        self._t = None         # last step()'s wall clock, for this step's dt
        self._paces = deque(maxlen=16)   # seconds per position of the last
                                         # passes, ALPHA's ~16-token memory
        self._n_tok = 0
        self._cap_seen = None  # last _GOV.cap seen, to catch a resize
        self._cooldown = 0

    def step(self, misses, T=1):
        """Once per decode-shaped forward (stack_call), a T=1 step or one batched draft-verification pass, with `misses` a raw per-pass count divided by T to land in a token's unit before the shared EMA."""
        self._n_tok += T
        now = time.perf_counter()
        dt = None if self._t is None else now - self._t
        self._t = now
        self.m += self.ALPHA * (misses / T - self.m)
        if _GOV is not None:
            if _GOV.cap != self._cap_seen:
                self._cap_seen = _GOV.cap
                self._cooldown = self.COOLDOWN
                if self.probe is not None:
                    # An open probe times WINDOW tokens per phase (128 total) against a governor that resizes every period=64, so a resize mid-window makes both phases meaningless and the probe reverts to the mode running before it opened, dropping the bracket it never touched.
                    self._apply(self.probe.mode_a)
                    self.probe = None
            elif getattr(_GOV, "_skip", False):
                self._cooldown = max(self._cooldown, 1)
        self._cooldown = max(0, self._cooldown - T)
        # A pass far slower than the recent pace is a turn boundary or a pause rather than a decode step, so the SAMPLE is dropped rather than the window, as ArenaGovernor.tick's guard lane does, with the pace the median seconds per position of the last passes and no fixed 0.25 s threshold, which dropped every pass slower than that (at 2 GB the forced passes take 0.3-0.7 s).
        if dt is not None:
            pace = dt / T
            quiet = (len(self._paces) > 0 and dt <= self.GAP_CAP
                     and pace <= self.GAP * float(np.median(self._paces)))
            self._paces.append(pace)
            if quiet and self.probe is not None:
                self._probe_step(dt, T)
        if self.probe is not None:
            return
        if self.m <= self.lo:
            want = True
        elif self.m >= self.hi:
            want = False
        elif self._cooldown == 0 and self.hi / max(self.lo, 1.0) > self.BAND * self.BAND:
            want = self._start_probe()
        else:
            want = _DEFER          # unknown, narrow zone: hold (hysteresis)
        self._apply(want)

    def _start_probe(self):
        self.probe = _Probe(_DEFER, self.m)
        return _DEFER               # phase 0 measures whatever is running now

    def _probe_step(self, dt, T=1):
        p = self.probe
        p.elapsed += dt
        p.n += T          # accumulated positions, not calls: a draft batch counts like single steps
        if p.n < self.WINDOW:
            return
        rate = p.n / max(p.elapsed, 1e-9)
        if p.phase == 0:
            p.rate_a, p.phase = rate, 1
            p.elapsed, p.n = 0.0, 0
            self._apply(not p.mode_a)     # flip for phase 1
            return
        self._resolve_probe(p, rate)

    def _resolve_probe(self, p, rate_b):
        self.probe = None
        if _GOV is not None:
            # The reverse alias: phase 1 forces the SLOWER of the two modes for a whole 64-token window regardless of the arena, and the governor's own speed lane is exactly such a window, so a window that landed on phase 1 records a rate that is low because of the probe, and tick's rate < best_below * 0.90 backoff cannot tell the two apart; clearing _rate is the same move tick() makes on its own ceiling drop, forgetting what is not trustworthy so the next quiet window remeasures.
            _GOV._rate.clear()
        ratio = rate_b / max(p.rate_a, 1e-9)     # > 1: the OTHER mode is faster
        if abs(ratio - 1.0) <= self.NOISE:
            winner = p.mode_a       # a draw: not worth moving, bracket unchanged
        else:
            winner = (not p.mode_a) if ratio > 1.0 else p.mode_a
            if winner:                          # defer measured faster at m_probe
                self.lo = max(self.lo, p.m_probe)
            else:                                # sync measured faster at m_probe
                self.hi = min(self.hi, p.m_probe)
            if self.lo >= self.hi:
                # The two measured points crossed, so gain(m) is not the clean monotone line the model above assumes at this resolution, and the bracket is wiped rather than acted on.
                self.lo, self.hi = 1.0, float("inf")
        self._cooldown = self.COOLDOWN
        self._apply(winner)

    def _apply(self, want):
        global _DEFER
        if want != _DEFER:
            _DEFER = want
            self.history.append((self._n_tok, want, round(self.m, 3),
                                 round(self.lo, 3),
                                 None if self.hi == float("inf") else round(self.hi, 3)))


_AUTO = None      # DeferPolicy instance while install(defer="auto"); else None


DEFER_FULL_REPAIR = False       # A/B switch: back to repairing the whole layer

_HALF_LAYERS = ("Qwen3NextDecoderLayer", "DecoderLayer", "Qwen2MoeDecoderLayer")


def _halves(layer):
    """(dense, moe) for a layer whose two halves can be run apart, else None: the attention half writes the cache alone, so a repair re-runs the MoE half and leaves the cache as it is."""
    if DEFER_FULL_REPAIR or type(layer).__name__ not in _HALF_LAYERS:
        return None
    # Qwen2MoeDecoderLayer has no `is_linear` attribute at all (only the Next/3.6 linear-attention layers do), so getattr's default takes the ordinary self_attn branch, the only one this class has.
    attn = getattr(layer, "linear_attn" if getattr(layer, "is_linear", False)
                   else "self_attn", None)
    n1 = getattr(layer, "input_layernorm", None)
    n2 = getattr(layer, "post_attention_layernorm", None)
    mlp = getattr(layer, "mlp", None)
    if attn is None or n1 is None or n2 is None or mlp is None:
        return None
    return (lambda *a, **k: a[0] + attn(n1(a[0]), *a[1:], **k),
            lambda h: h + mlp(n2(h)))


def _run_layer(orig, layer, args, kwargs, halves=None, h=None):
    """One layer through the model's own code, returning (output, rec, h), as its two halves with `halves` so a repair can re-run the MoE half on the dense half's `h`."""
    del _DEFER_REC[:]
    if halves is None:
        out = orig(layer, *args, **kwargs)
    else:
        if h is None:
            h = halves[0](*args, **kwargs)
        out = halves[1](h)
    rec = _DEFER_REC.pop() if _DEFER_REC else None
    assert not _DEFER_REC, "defer: more than one streaming GLU in a layer"
    return out, rec, h


def _defer_submit(out, rec):
    """Hand the layer to the GPU together with everything its check reads: the routing, the residency flag and the lookahead's prediction."""
    if rec is None:
        mx.async_eval(out)
    elif rec[6] is None:
        mx.async_eval(out, rec[1], rec[2])
    else:
        mx.async_eval(out, rec[1], rec[2], rec[6])


def _defer_layer_call(orig, layer, args, kwargs):
    """A decoder layer under run-ahead: layer L is built and submitted before L-1's flag is read, and a miss rolls the cache back, redoes L-1's MoE half and recomputes L on the corrected input, with the last layer checking itself."""
    st = _DEFER_STATE
    h = args[0]
    T = h.shape[1]
    if T > _DEFER_MAX_T:
        return orig(layer, *args, **kwargs)
    g = layer.__dict__.get("_defer_glu")
    if g is not None and (g._pending or g._defer):
        # The depth prefetch for THIS layer, issued two layers back, lands and publishes its slots here before the block's router runs, the same order the ordinary path gives them, so the cache bias and the gather see the same residency in both.
        g._await_pending()
    caches = _caches_in(args, kwargs)
    halves = _halves(layer) if g is not None else None
    # A snapshot is only worth taking for a cache that can still go back: this layer re-runs in full only when the layer behind it is pending, and its own repair needs one only when that repair is the whole layer.
    snaps = (_snapshot_caches(caches)
             if st.pending is not None or (g is not None and halves is None)
             else None)
    # Host time split: building the layer's graph versus handing it to the GPU (kernel encoding), where the residual over the byte floor lives.
    t0 = time.perf_counter()
    out, rec, h_dense = _run_layer(orig, layer, args, kwargs, halves)
    t1 = time.perf_counter()
    if not NO_ASYNC:
        _defer_submit(out, rec)
    t2 = time.perf_counter()
    TIMERS["defer_build"] += t1 - t0
    TIMERS["defer_submit"] += t2 - t1

    prev = st.pending
    st.pending = None
    if prev is not None:
        p_layer, p_args, p_kwargs, p_snaps, p_rec, p_halves, p_h = prev
        p_missed = _defer_resolve(p_rec[0], p_rec[1], p_rec[2], p_rec[3], False,
                                  p_rec[4], p_rec[5], p_rec[6])
        if p_missed:
            STATS["defer_rollbacks"] += 1
            # Undo both layers, redo the missed one on its saved input with the arena repaired, then this one on the corrected output, while the GPU finishes the garbage it was handed.
            _restore_caches(caches, snaps, T)
            if p_halves is not None:
                # Only the MoE half of the pending layer saw a missing expert, and since its dense half ran on an already checked input and is the only half that wrote the cache, the cache stays as it is and the saved h goes straight back into the MoE.
                STATS["repair_half"] += 1
                p_out, p_rec2, _ = _run_layer(orig, p_layer, p_args, p_kwargs,
                                              p_halves, p_h)
            else:
                _restore_caches(_caches_in(p_args, p_kwargs), p_snaps, T)
                p_out, p_rec2, _ = _run_layer(orig, p_layer, p_args, p_kwargs)
            _defer_release(p_rec[0])
            if p_rec2 is not None:
                if not NO_ASYNC:
                    _defer_submit(p_out, p_rec2)
                _defer_resolve(p_rec2[0], p_rec2[1], p_rec2[2], p_rec2[3], True,
                               p_rec2[4], p_rec2[5])
            args = (p_out,) + tuple(args[1:])
            out, rec, h_dense = _run_layer(orig, layer, args, kwargs, halves)
            if not NO_ASYNC:
                _defer_submit(out, rec)
    if rec is not None:
        st.ticks += 1
        if getattr(layer, "_defer_last", False):
            # No layer behind this one to hide the round trip, and the model's tail must not see a pass a miss would roll back.
            missed = _defer_resolve(rec[0], rec[1], rec[2], rec[3], False,
                                    rec[4], rec[5], rec[6])
            if missed:
                STATS["defer_rollbacks"] += 1
                STATS["defer_now_misses"] += 1
                # Same repair as above, on this layer's own check: its input is already correct, so with halves only the MoE half redoes.
                if halves is None:
                    _restore_caches(caches, snaps, T)
                else:
                    STATS["repair_half"] += 1
                out, rec2, h_dense = _run_layer(orig, layer, args, kwargs,
                                                halves, h_dense)
                _defer_release(rec[0])
                if rec2 is not None:
                    if not NO_ASYNC:
                        _defer_submit(out, rec2)
                    _defer_resolve(rec2[0], rec2[1], rec2[2], rec2[3], True,
                                   rec2[4], rec2[5])
            else:
                STATS["defer_now_hits"] += 1
        else:
            st.pending = (layer, args, kwargs, snaps, rec, halves, h_dense)
    return out


def install_defer(model):
    """Run-ahead decode for any model whose decoder is a list of layers: two wrappers marked per instance, the build/submit/check order on the layer class and the token boundary on the module owning the list."""
    stack = _layer_stack_of(model)
    if stack is None:
        raise ValueError("defer: no module owning a `layers` list was found, "
                         "so there is no decoder loop to attach to")
    layers = stack["layers"]
    ours = {id(g) for g in glus_of(model)}
    streaming = []
    for l in layers:
        mlp = _moe_of(l)
        mlp = getattr(mlp, "block", mlp)
        if id(getattr(mlp, "switch_mlp", None)) in ours:
            streaming.append(l)
            # The arena the layer boundary has to publish into before the block's router reads residency off it.
            l.__dict__["_defer_glu"] = mlp.switch_mlp
    if not streaming:
        raise ValueError("defer: no decoder layer holds a streaming GLU")

    lcls = type(layers[0])
    if not getattr(lcls, "_defer_wrapped", False):
        lorig = lcls.__call__

        def layer_call(self, *args, **kwargs):
            if not _DEFER_ACTIVE or not self.__dict__.get("_defer_on"):
                return lorig(self, *args, **kwargs)
            return _defer_layer_call(lorig, self, args, kwargs)

        lcls.__call__ = layer_call
        lcls._defer_wrapped = True
    for i, l in enumerate(layers):
        l.__dict__["_defer_on"] = True
        l.__dict__["_defer_last"] = False
        l.__dict__["_defer_idx"] = i
    streaming[-1].__dict__["_defer_last"] = True
    _set_widest_unsorted(model)

    scls = type(stack)
    if not getattr(scls, "_defer_wrapped", False):
        sorig = scls.__call__

        def stack_call(self, inputs, *args, **kwargs):
            global _DEFER_ACTIVE
            # DeferPolicy's token boundary: a single-position decode step or one batched draft-verification pass, since a prefill means something different per layer and is not one of the two things being chosen between, read here as offset==0 on the first cache that keeps one (a hybrid's linear-attention caches have none, and reading layer 0's called every pass there a prefill) and read before the call, since either branch below may flip _DEFER through _AUTO before it returns.
            T = inputs.shape[-1]
            cache_list = next((a for a in list(args) + list(kwargs.values())
                               if isinstance(a, list) and a and a[0] is not None),
                              None)
            kv = next((c for c in cache_list or () if hasattr(c, "offset")), None)
            prefill = kv is None or kv.offset == 0
            probe_call = (_AUTO is not None and T <= _DEFER_MAX_T and not prefill
                         and kwargs.get("input_embeddings") is None)
            # Auto under --mem-peak-gb: the stack's tick has just sized the quotas, and a pass they cannot carry would only roll back.
            bud = self.__dict__.get("_mem_budget")
            if (not _DEFER or not self.__dict__.get("_defer_on") or _DEFER_ACTIVE
                    or T > _DEFER_MAX_T
                    or kwargs.get("input_embeddings") is not None
                    or cache_list is None
                    or (_AUTO is not None and bud is not None and not bud.carries(T))):
                m0 = STATS["miss_layers"]
                out = sorig(self, inputs, *args, **kwargs)
                if probe_call:
                    _AUTO.step(STATS["miss_layers"] - m0, T)
                return out
            st = _DEFER_STATE
            st.pending, st.ticks = None, 0
            STATS["defer_tokens"] += 1
            _DEFER_ACTIVE = True
            r0 = STATS["defer_rollbacks"] + STATS["defer_wide"]
            try:
                out = sorig(self, inputs, *args, **kwargs)
            finally:
                _DEFER_ACTIVE = False
                del _DEFER_REC[:]
                assert st.pending is None, "defer: a layer's check was never run"
            # The governor's ticks (one per streaming layer, as on the sync path) at the token boundary, so a resize lands where nothing is in flight, since mid-token it cost 17 rollbacks a token, with T positions to a pass so a batched draft verification banks k+1 tokens.
            if _GOV is not None:
                for _ in range(st.ticks):
                    _GOV.tick(False, T)
            if probe_call:
                _AUTO.step(STATS["defer_rollbacks"] + STATS["defer_wide"] - r0, T)
            return out

        scls.__call__ = stack_call
        scls._defer_wrapped = True
    stack.__dict__["_defer_on"] = True
    note(f"  defer: run-ahead decode over {len(layers)} layers, "
          f"{len(streaming)} of them streaming"
          + (", policy auto (sync/defer chosen from measured miss rate)"
             if _AUTO is not None else ""), flush=True)
    if fused_router.on():
        note("  fused router (P3): on, one metal kernel per MoE layer", flush=True)
    return stack


def _set_widest_unsorted(model):
    """_DEFER_MAX_T: the widest pass whose selections stay under SORT_MIN, i.e. the widest decode pass, a speculative round's verify, since anything wider is a prefill."""
    global _DEFER_MAX_T
    k = top_k_of(model)
    _DEFER_MAX_T = max(1, (SORT_MIN - 1) // k) if k else 1


def _recurrent_states(args, kwargs):
    return [a for c in _caches_in(args, kwargs) if isinstance(c, ArraysCache)
            for a in c.cache if a is not None]


# ---------------------------------------------------------------------------
# Layer-major prefill (budget mode): a layer runs its dense part (attention or GatedDeltaNet with its caches, PLE, norms) in LM_W-position chunks, routes each chunk, multiplies every expert once over all the pass's rows (read ahead of the GPU into the released store, _LMReader) and combines chunk by chunk, so a long prompt costs a few passes rather than one a layer's per-position memory (a prefill wider than a few hundred positions reads every expert of every layer anyway, 75.5 GB and ~22 s from an internal SSD on Flash-Next).

# mlx_lm's prefill_step_size: the chunks are the passes this one replaces.
LM_W = 2048
# The width of the cold probe's second pass as a multiple of the narrowest one: two widths cancel the line's fixed part exactly, and two is the least ratio whose difference still stands clear of a footprint reading's few MB (MemBudget._lm_line).
PROBE_MULT = 2
_LM_READ = {}           # id(store) -> its _LMReader, for the pass that runs now
# MOE_LM_PREFILL: unset = on for Flash-Next's layers (qwen4_exp) under a budget store, 1 = also Qwen3-Next/3.6 (_halves), 0 = off.
_LM_ENV = os.environ.get("MOE_LM_PREFILL")
# CUDA: a union read keeps the whole path's groups (its ranges of g that hold a routed expert) and reads only the routed experts of each, so the others' slots are never indexed, since on CUDA groups of g routed experts gave other bits than the ranges while on Metal both groupings are exact (test_union_lm).
_LM_RANGES = BACKEND == "cuda"
# MOE_LM_PIPE=0: each group is one read wave that both drives finish before the next group starts.
_LM_PIPE = os.environ.get("MOE_LM_PIPE", "1") != "0"
# MOE_LM_COPY_WHOLE=1: a layer read whole copies the store's residents slot to slot as a union read does, instead of reading them again.
_LM_COPY_WHOLE = os.environ.get("MOE_LM_COPY_WHOLE", "0") == "1"


class _LMReader:
    """The expert reads of one layer-major pass: every MoE layer from `glus[0]` on, on a thread of its own into G-slot regions of the released store, whole ahead of the GPU where a wide prompt routes to every expert and by union once a short one has routed, with free regions taken last in, first out."""

    def __init__(self, store, glus, res=None):
        self.store, self.g, self.glus = store, store.per_half, glus
        n = store.slots // self.g
        self.res = res or {}
        held = np.zeros(n, dtype=np.int64)
        for m in self.res.values():
            for sl in m.values():
                if sl < n * self.g:
                    held[sl // self.g] += 1
        self.free = deque(int(r) * self.g for r in np.argsort(-held, kind="stable"))
        assert len(self.free) >= 2, "layer-major prefill: a store of under two groups"
        self.written = np.zeros(n, dtype=bool)
        self.plans, self.unions, self.done = {}, {}, {}
        self.last_u, self.last_res, self.err, self.stop = None, None, None, False
        self.routed = None      # the reader thread's union, under _LM_RANGES
        lk = threading.RLock()
        self.cv, self.work = threading.Condition(lk), threading.Condition(lk)
        # MOE_CUDA_STAGE: the groups go through host staging and are copied into their store rows on the GPU at take() (cuda_stage.LMStage).
        self.stage = None
        if _CUDA_STAGE_ON and (_STRIPE is None or _STRIPE.dead):
            import cuda_stage
            self.stage = cuda_stage.LMStage(_cuda_stage(), store)
        # The pipeline: records queued (key, job, sidecar offset), records a group still waits for, a layer's [groups left, bytes, whole].
        self.q, self.left, self.lay = deque(), {}, {}
        # Busy clocks [reads in flight, last tick, seconds] of each drive and of either, plus bytes landed and the last group's marks per drive.
        self.busy = {d: [0, 0.0, 0.0] for d in ("int", "ext", "any")}
        self.got, self.mark = {"int": 0, "ext": 0}, {"int": (0, 0.0), "ext": (0, 0.0)}
        self.t_read = self.t_lay = 0.0
        self.pullers = []
        if _LM_PIPE and (self.stage is None or self.stage.queued):
            s = _STRIPE
            n_ext = s.readers if s is not None and not s.dead else 0
            self.pullers = [threading.Thread(target=self._pull, args=(ext,), daemon=True)
                            for ext in [False] * _POOL._max_workers + [True] * n_ext]
            for p in self.pullers:
                p.start()
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _region(self):
        with self.cv:
            while not self.free and not self.stop:
                self.cv.wait()
            if self.stop:
                return None
            r0 = self.free.pop()
            self.written[r0 // self.g] = True
            w = getattr(self.store, "written", None)
            if w is not None:
                w[r0:r0 + self.g] = True
            return r0

    def _resident(self, glu, e):
        """The slot still holding glu's expert e from before the pass, or None."""
        sl = self.res.get(id(glu), {}).get(e)
        if sl is None or (sl < len(self.written) * self.g and self.written[sl // self.g]):
            return None
        return sl

    def _run(self):
        try:
            for i, glu in enumerate(self.glus):
                n_exp = glu._projs[0]._packed["n_experts"]
                # Queued on CUDA (MOE_CUDA_SPANS=1) the union cost is the corrected one, with the share of the routed bytes a union copies.
                queued = self.stage is not None and self.stage.queued
                whole = _lm_whole(glu, i, self.last_u, (self.last_res or 0.0) if queued else None)
                routed = None
                if whole:
                    groups = [np.arange(a, min(a + self.g, n_exp)) for a in range(0, n_exp, self.g)]
                else:
                    with self.cv:
                        while id(glu) not in self.unions and not self.stop:
                            self.cv.wait()
                        if self.stop:
                            return
                        ids = self.unions[id(glu)]
                    if _LM_RANGES:
                        routed = ids
                        groups = [np.arange(a, min(a + self.g, n_exp))
                                  for a in np.unique(ids // self.g) * self.g]
                    else:
                        groups = [ids[k:k + self.g] for k in range(0, len(ids), self.g)]
                with self.cv:
                    self.plans[id(glu)] = groups
                    self.lay[id(glu)] = [len(groups), 0, whole]
                    self.cv.notify_all()
                STATS["lm_whole" if whole else "lm_union"] += 1
                nb, secs = 0, 0.0
                for k, ids in enumerate(groups):
                    r0 = self._region()
                    if r0 is None:
                        return
                    t = time.perf_counter()
                    self.routed = routed
                    b, jobs = self._read(glu, ids, r0, whole)
                    STATS["lm_read_b"] += b
                    if jobs is not None:
                        self._put((id(glu), k), r0, jobs)
                        continue
                    dt = time.perf_counter() - t
                    TIMERS["lm_read"] += dt
                    nb, secs = nb + b, secs + dt
                    with self.cv:
                        self.done[(id(glu), k)] = r0
                        self.cv.notify_all()
                if nb and secs > 0:
                    _lm_ema("rate_w" if whole else "rate_u", nb / secs)
        except BaseException as e:      # handed to the model thread by take()
            with self.cv:
                self.err = e
                self.cv.notify_all()

    def _read(self, glu, ids, r0, whole):
        """glu's experts `ids` (sorted) into slots r0.. in order but those not in self.routed, returning the bytes asked of the drives and the reads left to the pipeline."""
        a, n, routed = int(ids[0]), len(ids), self.routed
        skip = set() if routed is None else set(np.flatnonzero(~np.isin(ids, routed)).tolist())
        copies = {} if whole and not _LM_COPY_WHOLE else {
            i: sl for i, e in enumerate(ids.tolist())
            if i not in skip and (sl := self._resident(glu, e)) is not None}
        if self.stage is not None:
            # The residents are copied slot to slot on the GPU at take(), since the slot's region is written later in the stream as on the host.
            STATS["lm_copied"] += len(copies)
            if self.pullers:
                jobs = self.stage.jobs(glu, ids, r0, copies, skip)
                return sum(j[3] for j in jobs), jobs
            return self.stage.read(glu, ids, r0, copies, skip), None
        if whole and not copies and (_STRIPE is None or _STRIPE.dead):
            got = 0
            for j, p in enumerate(glu._projs):
                pk = p._packed
                iov = [v for s in range(r0, r0 + n) for v in self.store.iov[j][s]]
                _do_read((pk["fd"], iov, pk["offset"] + a * pk["stride"], n * pk["stride"]))
                got += n * pk["stride"]
            return got, None
        jobs = []
        for i, e in enumerate(ids.tolist()):
            if i in skip:
                continue
            sl = copies.get(i)
            if sl is not None:
                for vs in self.store.views:
                    for v in vs:
                        v[r0 + i] = v[sl]
                STATS["lm_copied"] += 1
                continue
            jobs += [(p._packed["fd"], self.store.iov[j][r0 + i],
                      p._packed["offset"] + e * p._packed["stride"], p._packed["stride"])
                     for j, p in enumerate(glu._projs)]
        b = sum(j[3] for j in jobs)
        if self.pullers:
            return b, jobs
        # With a sidecar both drives read at once, record by record, split by the rates its waves measure (_Stripe.split), as a decode wave is.
        _read_wave(jobs)
        return b, None

    def _put(self, key, r0, jobs):
        """Group `key`'s reads into region r0, queued behind every earlier group's and landing at once with none."""
        s = _STRIPE
        items = [(key, j, None if s is None else s.ext_off(j)) for j in jobs]
        with self.cv:
            self.left[key] = [len(jobs), r0, sum(j[3] for j in jobs)]
            self.q.extend(items)
            if not jobs:
                self._land(key)
            self.work.notify_all()

    def _pull(self, ext):
        """One reader of a drive (the sidecar if `ext`: only the records it holds) until the pass ends: the list's first record it can read."""
        dev, s = ("ext" if ext else "int"), _STRIPE
        try:
            while True:
                with self.cv:
                    while True:
                        if self.stop or self.err is not None:
                            return
                        i = next((i for i, it in enumerate(self.q) if not ext or it[2] is not None), None)
                        if i is not None:
                            break
                        self.work.wait()
                    key, job, eoff = self.q[i]
                    del self.q[i]
                    self._busy(dev, 1)
                if ext:
                    s.read(job, eoff)
                elif self.stage is not None:
                    self.stage.one(job)
                else:
                    _do_read(job)
                with self.cv:
                    self._busy(dev, -1)
                    self.got[dev] += job[3]
                    g = self.left[key]
                    g[0] -= 1
                    if not g[0]:
                        self._land(key)
        except BaseException as e:      # handed to the model thread by take()
            with self.cv:
                self.err = e
                self.cv.notify_all()
                self.work.notify_all()

    def _busy(self, dev, step=0):
        """Tick the busy clocks of drive `dev` and of either drive, `step` +1 starting a read and -1 ending one."""
        now = time.perf_counter()
        for d in (dev, "any"):
            b = self.busy[d]
            if b[0]:
                b[2] += now - b[1]
            b[0] += step
            b[1] = now

    def _land(self, key):
        """Group `key` is in its region under the lock, and the drives' bytes and busy seconds since the last group go to the stripe's rates and a layer's last group gives its read rate."""
        _, r0, nb = self.left.pop(key)
        self.done[key] = r0
        s = _STRIPE
        for d in ("int", "ext"):
            self._busy(d)
            b, t = self.got[d], self.busy[d][2]
            if b > self.mark[d][0] and s is not None:
                s.note(d, b - self.mark[d][0], t - self.mark[d][1])
            self.mark[d] = (b, t)
        t = self.busy["any"][2]
        TIMERS["lm_read"] += t - self.t_read
        self.t_read = t
        lay = self.lay[key[0]]
        lay[0] -= 1
        lay[1] += nb
        if not lay[0]:
            if lay[1] and t > self.t_lay:
                _lm_ema("rate_w" if lay[2] else "rate_u", lay[1] / (t - self.t_lay))
            self.t_lay = t
        self.cv.notify_all()

    def post(self, glu, ids):
        """glu has routed to `ids` (sorted): its groups, once the thread has planned them."""
        t = time.perf_counter()
        with self.cv:
            self.last_u = len(ids) / glu._projs[0]._packed["n_experts"]
            if self.stage is not None and self.stage.queued:
                self.last_res = sum(self._resident(glu, e) is not None for e in ids.tolist()) / max(len(ids), 1)
                if _lm_count("pre") > 1:
                    _LMRP["pre"].append(t - _LMRP["t0"])
            if glu is self.glus[0]:
                u0, T = _LMRP["u0"], _LMRP["T"]
                u0[T] = max(u0.get(T, 0.0), self.last_u)
            self.unions[id(glu)] = ids
            self.cv.notify_all()
            while id(glu) not in self.plans:
                if self.err is not None:
                    raise RuntimeError("layer-major prefill: an expert read failed") from self.err
                self.cv.wait()
            del self.unions[id(glu)]
            groups = self.plans.pop(id(glu))
        TIMERS["lm_wait"] += time.perf_counter() - t
        return groups

    def take(self, glu, k):
        t = time.perf_counter()
        with self.cv:
            while (id(glu), k) not in self.done:
                if self.err is not None:
                    raise RuntimeError("layer-major prefill: an expert read failed") from self.err
                self.cv.wait()
            r0 = self.done.pop((id(glu), k))
        TIMERS["lm_wait"] += time.perf_counter() - t
        if self.stage is not None:
            self.stage.take(r0)
        return r0

    def give(self, r0):
        if self.stage is not None:
            self.stage.give(r0)
        with self.cv:
            self.free.append(r0)
            self.cv.notify_all()

    def close(self):
        with self.cv:
            self.stop = True
            self.cv.notify_all()
            self.work.notify_all()
        if self.stage is not None:
            self.stage.close()
        self.t.join()
        # A record still in flight lands before the store is used again.
        for p in self.pullers:
            p.join()


# Read path per layer of a layer-major pass, from what this process measured (the plain path's rule, _ahead_pays): a layer read whole ahead costs max(window, B / rate_w) and its union read after its routing window + union / rate_u, with B the layer's bytes, the union predicted by the last layer that routed and the window the model thread's time a layer spends outside its waits for reads.
_LMRP = {"rate_u": None, "rate_w": None, "win": [], "T": 0, "prev": None, "u0": {},
         "n": {}, "pre": [], "prev_pre": None, "t0": 0.0}


def _lm_count(k):
    """This process's samples of measure k so far, this one included, with each measure's first sample dropped since it pays first-use costs and a path not taken would never be measured again."""
    n = _LMRP["n"][k] = _LMRP["n"].get(k, 0) + 1
    return n


def _lm_ema(k, v):
    # The first sample only starts the estimate: the second replaces it.
    n = _lm_count(k)
    _LMRP[k] = v if n <= 2 else _LMRP[k] + 0.3 * (v - _LMRP[k])


def _lm_whole(glu, i, last_u, res=None):
    """Whether layer i is read whole; `res`, the residents' share of the routed bytes, switches on the corrected union cost (queued CUDA reads), None keeps the old one."""
    if _RP_MODE != "auto":
        return _RP_MODE == "whole"
    rp = _LMRP
    if i == 0:
        # One entry per width seen, a list over a server's life, copied in one step since a post() may add to it meanwhile.
        last_u = max((u for t, u in list(rp["u0"].items()) if t <= rp["T"]), default=None)
    if last_u is None or rp["rate_u"] is None:
        return False
    win = rp["win"][:]
    if win:
        win = sum(win) / len(win)
    elif rp["prev"]:
        # Until this pass's first layer is done, the last pass's window scaled to this width.
        win = rp["prev"][1] * rp["T"] / max(rp["prev"][0], 1)
    else:
        win = 0.0
    B = _layer_bytes(glu)
    whole = max(win, B / (rp["rate_w"] or rp["rate_u"]))
    if res is None:
        # A tie reads ahead, since the union read waits for the routing, and the residents a union read would copy are left in its bytes (a few %).
        return whole <= win + last_u * B / rp["rate_u"]
    # The union read overlaps the layer's time after routing only, so just the time before it (`pre`) is serial with it, and it reads the routed bytes but the residents'.
    pre = rp["pre"][:]
    if pre:
        pre = sum(pre) / len(pre)
    elif rp["prev_pre"]:
        pre = rp["prev_pre"][1] * rp["T"] / max(rp["prev_pre"][0], 1)
    else:
        pre = 0.0
    pre = min(pre, win)
    return whole <= pre + max(win - pre, last_u * (1 - res) * B / rp["rate_u"])


def _lm_close():
    """End of a layer-major pass (its last layer, or the stack's exit on an error): the reader stops and the PLE rows read ahead go."""
    for r in _LM_READ.values():
        r.close()
    _LM_READ.clear()
    exl3_experts.pass_end()
    for s in _LM_PLE:
        s.ahead = None
    _LM_PLE.clear()
    ane_overlap.close()             # MOE_ANE_SPLIT / MOE_ANE_OVERLAP: their surfaces leave the budget before decode (no-op when off)
    ane_experts.close()             # MOE_ANE_EXPERTS: its programs and surfaces go (no-op when off)


_LM_PLE = []            # StreamingPLE tables holding a pass's rows read ahead


def _lm_family(layer):
    """'q4' for a qwen4_exp DecoderLayer (Flash-Next), 'half' for the layers _halves splits (Qwen3-Next/3.6, Qwen2-MoE), else None."""
    if all(hasattr(layer, a) for a in ("attn_hyper_connection", "mlp_hyper_connection",
                                       "ple", "mlp", "layer_type")):
        return "q4"
    return "half" if _halves(layer) is not None else None


def _lm_parts(layer, args, kwargs):
    """(T, pre(a, b), moe_in(h), moe_out(y, keep), caches) of one layer call cut at the MoE block in the model's own code (DecoderLayer.__call__, _halves)."""
    if _lm_family(layer) == "q4":
        h, rope, mask, conv_mask, cache, idx_cache, ids, prev_ctx = args
        if conv_mask is not None or not (mask is None or isinstance(mask, str)):
            raise RuntimeError("layer-major prefill: a padded or batched pass")
        ctx = prev_ctx.shape[1] if prev_ctx is not None else 0
        if layer.__dict__.get("_lgp") and layer_glue.rows_ok(h):
            return (h.shape[1], *_lm_glue_parts(layer, h, rope, mask, conv_mask, cache, idx_cache, ids,
                                                prev_ctx, ctx), [cache])

        def pre(a, b):
            hc = h[:, a:b]
            if layer.ple is not None:
                hc = hc + layer.ple(hc, ids[:, a:b], prev_ctx if a == 0 else ids[:, a - ctx:a],
                                    cache)
            x, hyper, inject = layer.attn_hyper_connection(hc)
            if layer.layer_type == "linear_attention":
                x = layer.linear_attn(x, conv_mask, cache)
            else:
                # A one-position piece had no mask (create_attention_mask).
                x = layer.self_attn(x, rope, mask if b - a > 1 else None, cache, idx_cache)
            return hyper + (x[..., None, :] * inject[..., None]).reshape(*x.shape[:-1], -1)

        def moe_in(hm):
            x, hyper, inject = layer.mlp_hyper_connection(hm)
            return x, (hyper, inject)

        def moe_out(y, keep):
            hyper, inject = keep
            return hyper + (y[..., None, :] * inject[..., None]).reshape(*y.shape[:-1], -1)
        return h.shape[1], pre, moe_in, moe_out, [cache]
    first = _halves(layer)[0]
    x = args[0]
    mask = kwargs.get("mask", args[1] if len(args) > 1 else None)
    cache = kwargs.get("cache", args[2] if len(args) > 2 else None)
    if not (mask is None or isinstance(mask, str)):
        raise RuntimeError("layer-major prefill: a padded or batched pass")
    n2 = layer.post_attention_layernorm
    return (x.shape[1],
            lambda a, b: first(x[:, a:b], mask if b - a > 1 else None, cache),
            lambda hm: (n2(hm), hm),
            lambda y, hm: hm + y,
            [cache])


def _lm_glue_parts(layer, h, rope, mask, conv_mask, cache, idx_cache, ids, prev_ctx, ctx):
    """MOE_GLUE_PREFILL: _lm_parts' pre, moe_in and moe_out for Flash-Next with layer_glue's row kernels, the attention half's combine inside the MLP half's norm kernel."""
    lin = layer.layer_type == "linear_attention"
    gdn = lin and layer.linear_attn.__dict__.get("_lgp")

    def pre(a, b):
        hc = h[:, a:b]
        if layer.ple is not None:
            hc = hc + layer.ple(hc, ids[:, a:b], prev_ctx if a == 0 else ids[:, a - ctx:a], cache)
        x, hyper, inject = layer_glue.hc_rows(layer.attn_hyper_connection, hc)
        if gdn and layer_glue.gdn_rows_ok(layer.linear_attn, x, cache):
            x = layer_glue.gdn_rows(layer.linear_attn, x, cache)
        elif lin:
            x = layer.linear_attn(x, conv_mask, cache)
        else:
            x = layer.self_attn(x, rope, mask if b - a > 1 else None, cache, idx_cache)
        return hyper, x, inject

    def moe_in(t):
        x, hm, inject = layer_glue.hc_rows(layer.mlp_hyper_connection, None, comb=t)
        return x, (hm, inject)

    def moe_out(y, keep):
        return layer_glue.comb_rows(*keep, y)
    return pre, moe_in, moe_out


def _lm_eval(arrays, caches):
    """Force a chunk and what its caches hold (an attention cache's indexer keys too), so its transient is gone before the next chunk is built."""
    idx = [getattr(getattr(c, "indexer", None), "keys", None) for c in caches]
    mx.eval(arrays, [a for a in idx if a is not None])
    _eval_caches(caches)


def _lm_start(layer, glu, args):
    """First layer of a layer-major pass: each store is released and its reader starts on its layers from this one on, and on Flash-Next the PLE rows of the whole pass are hashed and read ahead, deduplicated, on a thread."""
    bud = glu._budget
    later = bud.all[bud.all.index(glu):]
    for key, gs in bud.groups.items():
        store = bud.store_for(gs[0])
        res = {}
        for g in gs:
            g._await_pending()
            if (_RP_MODE != "whole" or _LM_COPY_WHOLE) and g._slot_np is not None:
                e = np.flatnonzero(g._slot_np >= 0)
                res[id(g)] = dict(zip(e.tolist(), g._slot_np[e].tolist()))
        store.release()
        _LM_READ[id(store)] = _LMReader(store, [g for g in later if g._skey == key], res)
    if glu._exl3 is not None:
        exl3_experts.pass_begin()       # E1: MLX's cache through the pass (MemBudget._need plans it)
    ple = layer.__dict__.get("_lm_ple")
    if ple is not None and _lm_family(layer) == "q4":
        streamer = ple.ngram_embedding.__dict__.get("_ple")
        if streamer is not None:
            streamer.grab = []
            try:
                ple(args[6], args[7])           # NGramEmbedding(ids, prev_ctx): the ids only
            finally:
                gid, streamer.grab = streamer.grab[-1], None
            streamer.read_ahead(gid)
            _LM_PLE.append(streamer)


def _lm_layer_call(layer, args, kwargs):
    """One layer of a layer-major prefill pass (see LM_W above): dense part and routing chunk by chunk, the experts once over every row of the pass and then each chunk's combine with its rows, the block running its own code both times."""
    T, pre, moe_in, moe_out, caches = _lm_parts(layer, args, kwargs)
    glu, mlp = _glu_of(layer), _moe_of(layer)
    rp = _LMRP
    if not _LM_READ:
        if rp["win"]:
            rp["prev"] = (rp["T"], sum(rp["win"]) / len(rp["win"]))
        if rp["pre"]:
            rp["prev_pre"] = (rp["T"], sum(rp["pre"]) / len(rp["pre"]))
        rp["win"], rp["pre"], rp["T"] = [], [], T
        _lm_start(layer, glu, args)
    t0, w0 = time.perf_counter(), TIMERS["lm_wait"]
    rp["t0"] = t0
    # The width the budget planned this pass at (MemBudget.plan), LM_W unless the keys left no room for a chunk that wide.
    w = getattr(_MEASURING, "lm_w", LM_W) if _MEASURING is not None else LM_W
    bounds = [(a, min(a + w, T)) for a in range(0, T, w)]
    xs, keeps, inds = [], [], []

    def capture(x, idx):
        inds.append(idx)
        return mx.zeros((*idx.shape, x.shape[-1]), dtype=x.dtype)
    glu.__dict__["_lm"] = capture

    def step(t):
        x, keep = moe_in(t)
        mlp(x)
        _lm_eval([x, keep, inds[-1]], caches)
        xs.append(x)
        keeps.append(keep)
        return x
    ane_split.ACTIVE = True                 # MOE_ANE_SPLIT's groups take this pass's chunks (nothing without it)
    try:
        if len(bounds) > 1 and ane_overlap.takes(layer, args[0]):     # one chunk has nothing to overlap
            ane_overlap.chunks(layer, args, bounds, step)      # MOE_ANE_OVERLAP: the ANE a chunk ahead
        else:
            for a, b in bounds:
                step(pre(a, b))
    finally:
        ane_split.ACTIVE = False
        glu.__dict__.pop("_lm", None)
    cat = (lambda v: v[0] if len(v) == 1 else mx.concatenate(v, axis=1))
    y = glu(cat(xs), cat(inds))
    mx.eval(y)
    del inds
    out = None
    for i, (a, b) in enumerate(bounds):
        # A chunk's input and kept state go once its output is written.
        x, keep, xs[i], keeps[i] = xs[i], keeps[i], None, None
        glu.__dict__["_lm"] = (lambda x_, idx_, ya=y[:, a:b]: ya)
        try:
            o = moe_out(mlp(x), keep)
        finally:
            glu.__dict__.pop("_lm", None)
        if len(bounds) == 1:
            out = o
        else:
            if out is None:
                out = mx.zeros((o.shape[0], T) + o.shape[2:], dtype=o.dtype)
            out[:, a:b] = o
        mx.eval(out)
        if i == 0:
            _MEASURING.lm_rows()
    # The window a read ahead hides behind (_lm_whole): this layer's time outside its waits for reads, with the process's first left out (_lm_count).
    if _lm_count("win") > 1:
        rp["win"].append(time.perf_counter() - t0 - (TIMERS["lm_wait"] - w0))
    if layer.__dict__.get("_prefill_last"):
        _lm_close()
    return out


def lm_supported(model):
    """Whether a budget store's prefill runs layer-major on this model: Flash-Next's layers by default, the _halves ones with MOE_LM_PREFILL=1, none with 0."""
    stack = _layer_stack_of(model)
    layers = stack["layers"] if stack is not None else []
    fam = {_lm_family(l) for l in layers if _moe_of(l) is not None}
    ok = ((fam == {"q4"} and _LM_ENV != "0") or (fam == {"half"} and _LM_ENV == "1"))
    ok = ok and all(l.__dict__.get("_prefill_on") for l in layers)
    if ok:
        ple = next((l.ple.ple_embedding for l in layers
                    if getattr(l, "ple", None) is not None), None)
        for l in layers:
            l.__dict__["_lm_on"] = True
            l.__dict__["_lm_ple"] = ple
    return ok


def _prefill_layer_call(orig, layer, args, kwargs):
    """One decoder layer of a prefill pass, its lazy tails forced before the next layer is built, so a lazy slice of a long input is taken inside the layer and the MoE tail does not coexist with the next layer's attention."""
    global _IN_PREFILL, _PREFILL_AT
    bud = _MEASURING
    if bud is not None and bud.lm_pass and layer.__dict__.get("_lm_on"):
        return _lm_layer_call(layer, args, kwargs)
    if PREFILL_STOP is not None:
        _PREFILL_AT = _layer_at(layer, args, kwargs)
    _prefill_stop()
    halves = _halves(layer)
    _IN_PREFILL = True
    try:
        if halves is None:
            out = orig(layer, *args, **kwargs)
        else:
            h = halves[0](*args, **kwargs)
            if PREFILL_STOP is None:
                mx.async_eval(h, _recurrent_states(args, kwargs))
            else:
                # A stoppable pass: the dense half (attention, GatedDeltaNet) is a stretch of its own a request would wait for.
                mx.eval(h, _recurrent_states(args, kwargs))
                _prefill_stop()
            out = halves[1](h)
    finally:
        _IN_PREFILL = False
    mx.eval(out, _recurrent_states(args, kwargs))
    _PREFILL_AT = None
    if layer.__dict__.get("_prefill_last"):
        _WIDE_SCRATCH.clear()
        _drop_layer_sets()
    return out


def install_prefill_sync(model):
    """Prefill passes go through _prefill_layer_call and decode passes straight through, on the layer CLASS routed by per-instance marks so model swaps stay independent."""
    stack = _layer_stack_of(model)
    if stack is None:
        return
    layers = stack["layers"]
    _set_widest_unsorted(model)
    lcls = type(layers[0])
    if not getattr(lcls, "_prefill_wrapped", False):
        lorig = lcls.__call__

        def layer_call(self, *args, **kwargs):
            if PREFILL_SKIP is not None and self.__dict__.get("_prefill_idx", 1 << 30) < PREFILL_SKIP[0]:
                return PREFILL_SKIP[1]
            if not self.__dict__.get("_prefill_on") or args[0].shape[1] <= _DEFER_MAX_T:
                return lorig(self, *args, **kwargs)
            return _prefill_layer_call(lorig, self, args, kwargs)

        lcls.__call__ = layer_call
        lcls._prefill_wrapped = True
    for i, l in enumerate(layers):
        l.__dict__["_prefill_on"] = True
        l.__dict__["_prefill_last"] = l is layers[-1]
        l.__dict__["_prefill_idx"] = i
    # A GatedDeltaNet's dense half is one stretch with no check (0.37 s at 12k positions on the 35B), so a stoppable pass checks again between its projections and its scan, wrapped once per mlx_lm model module that calls gated_delta_update as chunkattn wraps sdpa.
    for m in model.modules():
        mod = sys.modules.get(type(m).__module__)
        f = getattr(mod, "gated_delta_update", None)
        if f is not None and not hasattr(f, "unstoppable"):
            def gated_delta_update(*a, _f=f, **kw):
                if PREFILL_STOP is not None and _IN_PREFILL:
                    mx.eval([x for x in a if isinstance(x, mx.array)])
                    _prefill_stop()
                return _f(*a, **kw)
            gated_delta_update.unstoppable = f
            mod.gated_delta_update = gated_delta_update


def used_experts():
    """Distinct experts each layer actually selected, once _TRACK_USED was on."""
    return sorted(len(v) for v in _USED.values())


def _exl3_window(cfg, packed):
    """The WIN template argument of an EXL3 expert pack (exl3_experts.spec), None for an affine one, from config.json's expert_quant."""
    index = (packed or {}).get("index", {})
    kinds = {m["parts"][0]["name"] for m in index.values()}
    if "trellis" not in kinds:
        return None
    eq = cfg.get("expert_quant") or {}
    ns = {m["parts"][0]["shape"][-1] for m in index.values()}
    cuda = BACKEND == "cuda"
    if (kinds != {"trellis"} or eq.get("format") != "exl3"
            or eq.get("codebook") not in (("mcg", "mul1") if cuda else ("mcg",))
            or not ns <= ({48, 64, 96} if cuda else {48, 64}) or BACKEND not in ("metal", "cuda")):
        raise ValueError(f"EXL3 experts: decoded here as MCG trellis records of 48 or 64 "
                         f"halfwords a tile, every routed projection, on Metal or CUDA (mul1 and "
                         f"96 on CUDA only); got records {sorted(kinds)}, expert_quant {eq}, "
                         f"n {sorted(ns)}, backend {BACKEND}")
    return exl3_experts.spec(eq)


def _gguf_types(packed, gate_key):
    """The gate, up and down ggml types of a layer of GGUF block records (gguf_experts.py, tools/gguf_repack.py) as one "Q4_K/Q4_K/Q5_1" string (nn.Module would keep a tuple as a parameter), None for other records."""
    index = (packed or {}).get("index", {})
    m = index.get(gate_key)
    if m is None or m["parts"][0]["name"] != "gguf":
        return None
    base = gate_key[:-len("gate_proj")]
    types = tuple(index[base + p]["parts"][0]["ggml_type"] for p in ("gate_proj", "up_proj", "down_proj"))
    if not set(types) <= set(gguf_experts.TYPES) or BACKEND not in ("metal", "cuda"):
        raise ValueError(f"GGUF experts: decoded here in {sorted(gguf_experts.TYPES)} on Metal or CUDA; got "
                         f"{types} on {BACKEND}")
    return "/".join(types)


def dense_to_device(model):
    """MOE_DENSE_VRAM: every array a module holds outside the experts (and outside a streamed embedding table)
    replaced by a bit copy made on the GPU, which MLX allocates in device memory; returns the bytes copied."""
    as_bits = {1: mx.uint8, 2: mx.uint16, 4: mx.uint32, 8: mx.uint64}
    done, moved = {}, 0
    for name, m in model.named_modules():
        if "switch_mlp" in name or (name.endswith("embed_tokens") and not EMBED_VRAM):
            continue
        new = {}
        for k, v in m.items():
            if isinstance(v, mx.array) and v.itemsize in as_bits:
                if id(v) not in done:       # a tied array is copied once
                    done[id(v)] = (v, mx.bitwise_or(v.view(as_bits[v.itemsize]), 0).view(v.dtype))
                    moved += v.nbytes
                new[k] = done[id(v)][1]
        if new:
            mx.eval(list(new.values()))
            for k, c in new.items():
                setattr(m, k, c)
    done.clear()
    mx.clear_cache()
    return moved


def install(model, model_path, resident=0, look_m=0, look_depth=1, cache_bias=0.0,
            defer=False, mtp=False, native=None):
    global _NATIVE, _RESIDENT
    if _NATIVE is not None:         # a model installed before this one
        _NATIVE.release()
        _NATIVE.agent.close()
        _NATIVE = None
    native = _NATIVE_ENV if native is None else native
    if (native or _NATIVE_ASKED) and BACKEND not in ("metal", "cuda"):
        print(f"  native: MOE_NATIVE=1 needs Metal or CUDA; the Python path runs on {BACKEND}",
              flush=True)
        native = False
    if native:
        # The native agent replaces the per-layer sync these hang off and owns the LRU while it runs, so the lookahead and run-ahead would claim behind its back: "auto" defer and a lookahead are defaults rather than requests and go quietly, while the rest is refused.
        if defer is True or fused_router.on():
            raise ValueError("MOE_NATIVE=1 cannot be combined with --defer-mode "
                             "always or the fused router")
        if defer or look_m:
            note("  native: run-ahead and the depth lookahead are off", flush=True)
        defer, look_m = False, 0
    if BACKEND == "cpu" and defer == "auto":
        # No GPU to overlap on MLX's CPU device: a run-ahead's rollbacks re-run whole layers.
        defer = False
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear

    if _DENSE_VRAM:
        note(f"  dense weights: {dense_to_device(model) / 1e9:.2f} GB copied into device memory (MOE_DENSE_VRAM)",
             flush=True)
    shard_map = build_shard_map(model_path)
    st = SafetensorsMap(sorted(set(shard_map.values())))
    st.packed = load_packed(model_path)
    open_stripe(model_path, st.packed)
    # The resident store's rows live in device memory, which only the CUDA staging's copies fill.
    _RESIDENT = _RESIDENT_ASKED
    if _RESIDENT and (not _CUDA_STAGE_ON or _STRIPE is not None):
        print("  resident: MOE_RESIDENT=1 needs MOE_CUDA_STAGE on and no stripe sidecar; off for this load", flush=True)
        _RESIDENT = False
    try:
        _cfg = json.loads((Path(model_path) / "config.json").read_text())
    except (OSError, ValueError):
        _cfg = {}
    _cfg_q = _cfg.get("quantization", {})
    win = _exl3_window(_cfg, st.packed)
    if win is not None:
        note(f"  experts: EXL3 trellis records ({_cfg['expert_quant'].get('codebook', 'mcg').upper()}, "
             f"window {win & 255}) through exl3_experts", flush=True)
        gate = next(m for k, m in st.packed["index"].items() if k.endswith("gate_proj"))
        it, ot, nhw = gate["parts"][0]["shape"]
        exl3_experts.calibrate(it * 16, ot * 16, nhw, win)
    gg = {k: _gguf_types(st.packed, k) for k in (st.packed or {}).get("index", {}) if k.endswith("gate_proj")}
    gg = {k: v for k, v in gg.items() if v is not None}
    if gg:
        mixes = Counter(gg.values())
        ne = st.packed["index"][next(iter(gg))]["parts"][0]["ne"]
        note(f"  experts: GGUF block records through gguf_experts, gate/up/down layers {dict(mixes)}", flush=True)
        gguf_experts.calibrate(ne[0], ne[1], mixes.most_common(1)[0][0].split("/"))
    install_ple(model, model_path, _cfg_q)
    # One quantized_matmul for the projections that read the same row (delta-net inputs, attention q/k/v, the shared expert's gate/up), CUDA only (_FUSE_PROJ).
    n_fused = fuse_proj.install_fuse_proj(model) if _FUSE_PROJ else 0
    if n_fused:
        mx.clear_cache()        # the parts' own buffers, before any store is sized
        note(f"  fused projection groups: {n_fused}", flush=True)
    n_fused = fuse_proj.install_fuse_hc(model)
    if n_fused:
        note(f"  hyper-connections: inject stacked under down in {n_fused}", flush=True)
    route_k.install(model, _moe_of)         # MOE_ROUTE_K: top_k becomes k1 before anything sizes by it
    n_glue = layer_glue.install(model)      # MOE_GLUE (on unless =0), Metal: decode-row glue kernels
    if n_glue and layer_glue.ON:
        note(f"  glue: {n_glue} layers' decode rows through layer_glue's kernels (MOE_GLUE on)", flush=True)
    if n_glue and layer_glue.PREFILL:
        note(f"  glue: {n_glue} layers' prefill rows through layer_glue's kernels (MOE_GLUE_PREFILL on)",
             flush=True)
    n_ane = ane_split.install(model, LM_W)  # MOE_ANE_SPLIT=1, Metal: dense projections shared with the ANE in prefill
    if n_ane:
        note(f"  ane split: {n_ane} projection groups, layer-major prefill only: {ane_split.summary()}", flush=True)
    n_ov = ane_overlap.install(model, LM_W)  # MOE_ANE_OVERLAP=1, Metal: the ANE a chunk ahead in the layer-major prefill
    if n_ov:
        note(f"  ane overlap: {n_ov} layers' projections, layer-major prefill only: {ane_overlap.summary()}", flush=True)
    if st.packed:
        note("  packed experts.bin found: one read per expert instead of three",
              flush=True)
    def resolve(probe):
        """Prefix of one layer's expert tensors whatever the checkpoint calls it, with the projection name stripped in this one place so a multimodal nesting does not double-strip it on the miss path."""
        packed = getattr(st, "packed", None)
        strip = lambda k: k[: -len(".gate_proj")] if k.endswith(".gate_proj") else k
        if not packed:
            return strip(probe)
        if probe in packed["index"]:
            return strip(probe)
        tail = "layers." + probe.split("layers.", 1)[-1]
        for k in packed["index"]:
            if k.endswith(tail):
                return strip(k)
        return strip(probe)

    n, fused_layers = 0, []
    for i, layer in enumerate(model.layers):
        switch, key_base = find_switch_mlp(layer, i)
        if switch is None:
            continue
        key_base = resolve(f"{key_base}.gate_proj")
        if not isinstance(getattr(switch, "gate_proj", None), QuantizedSwitchLinear):
            continue
        block = _moe_of(layer)
        k = int(getattr(block, "top_k", None) or getattr(block, "k", None)
                or getattr(block, "num_experts_per_tok", None) or 8)
        block.switch_mlp = StreamingSwitchGLU(
            st, key_base, switch, resident=resident, slots=k)
        block.switch_mlp._exl3 = win
        block.switch_mlp._gguf = _gguf_types(st.packed, f"{key_base}.gate_proj")
        fused_layers.append(i)
        n += 3
    if _CUDA_COMBINE and fused_layers:
        n_tail = route_cuda.install_tail([_moe_of(model.layers[i]) for i in fused_layers])
        if n_tail:
            note(f"  cuda combine: {n_tail} MoE blocks' decode rows in two kernels (MOE_CUDA_COMBINE)", flush=True)

    # Chain the MoE blocks so each can start the next one's reads: the router lives on the parent mlp and the arena on the switch_mlp, so both are wired, and a depth of `look_depth` widens the window proportionally at the cost of predicting from a hidden state further away (a layer's compute is ~1.3 ms against ~2.5 ms for 12 records of 1.47 MB, so one layer ahead hides only half the read).
    # Cache bias wraps the gate itself, so both uses pick it up (the block's own routing and the lookahead that predicts it), since predicting a biased router with an unbiased one would make the prediction worse, and where the block already separates selection from weighting through a correction term the bias goes into that term: bailing_moe keeps it on the gate (expert_bias) and MiniMax on the block (e_score_correction_bias, DeepSeek-V3 style sigmoid scores).
    first = _moe_of(model.layers[fused_layers[0]]) if fused_layers else None
    term = None
    if first is not None:
        if hasattr(getattr(first, "gate", None), "expert_bias"):
            term = ("gate", "expert_bias")
        elif hasattr(first, "e_score_correction_bias"):
            term = (None, "e_score_correction_bias")
    if cache_bias > 0.0 and term is not None:
        for i in fused_layers:
            mlp = _moe_of(model.layers[i])
            glu = mlp.switch_mlp
            if glu._slot_np is None:
                glu._init_arena()
            owner = getattr(mlp, term[0]) if term[0] else mlp
            CacheBiasedExpertBias(owner, term[1], glu, cache_bias, glu._n_exp,
                                  getattr(owner, "top_k", None)
                                  or getattr(mlp, "num_experts_per_tok", 8))
        note(f"  cache bias {cache_bias} via the block's own correction term "
              f"({term[1]}) on {len(fused_layers)} layers", flush=True)
    elif cache_bias > 0.0:
        for j, i in enumerate(fused_layers):
            w = CacheBiasedMoE(_moe_of(model.layers[i]), cache_bias)
            if j == 0:
                g = _moe_of(model.layers[i]).gate
                # The router is quantised (8-bit on Qwen3.6), so weight.shape is the packed width, not the hidden size.
                dim = g.weight.shape[-1] * (32 // getattr(g, "bits", 32)
                                            if hasattr(g, "bits") else 1)
                check_equivalence(_moe_of(model.layers[i]), w, dim)
                # The check's random pass claimed experts into this layer's arena, so the cold prompt then took the split path here alone (+394 MB peak) with an output that depended on what that random pass happened to route, and the layer is left as untouched as every other one.
                glu = _moe_of(model.layers[i]).switch_mlp
                if isinstance(glu, StreamingSwitchGLU):
                    glu.clear_arena()
                    glu._seen = None
                _WIDE_SCRATCH.clear()
            setattr(model.layers[i], _moe_attr(model.layers[i]), w)

    if defer == "auto" and not glus_of(model):
        # A dense model has no streaming GLU for run-ahead to overlap, and "auto" is the default rather than a request, so plain sync.
        defer = False
        note("  defer: auto is off, no streaming layers (dense model)",
              flush=True)
    # No pin under mtp any more: DeferPolicy.step samples every pass up to _DEFER_MAX_T positions with a pace-relative gap filter, MTP's verify passes (T=1+gamma) included, since run-ahead pinned at a 21-slot arena decoded long prose at 112.5 ms a token against 49.6 layer by layer.
    if defer and look_m and look_depth < 2:
        # The prediction is built when layer L runs and read on L's check one layer later, by which time L+1 has been submitted, so two layers is the first depth that can see the bytes ahead.
        raise ValueError("defer needs look_depth >= 2: the prediction is read "
                         "back on the deferred check, one layer after the "
                         "layer that built it, and L+1 is already submitted")
    global _AUTO, _DEFER
    # defer takes three values: False/True keep their old meaning (never or always run-ahead) and "auto" starts in defer and hands the per-token choice to DeferPolicy from then on.
    _DEFER = bool(defer)
    _AUTO = DeferPolicy() if defer == "auto" else None
    if defer:
        install_defer(model)
    stack = _layer_stack_of(model)
    if BACKEND == "cuda" and stack and os.environ.get("MOE_CUDA_SUBMIT", "0" if _RESIDENT else "1") == "1":
        # Earlier submission in sync decode (cuda_stage.install_submit), on by default under CUDA with MOE_CUDA_SUBMIT=0 off; off under MOE_RESIDENT, whose token is host-bound.
        import cuda_stage
        cuda_stage.install_submit([l for l in stack["layers"] if len(l.children())],
                                  lambda: _DEFER_ACTIVE)
    if _CUDA_STAGE_ON:
        # The staging buffers now, before the memory budget measures anything.
        gs = glus_of(model)
        if gs and all(p._packed is not None for p in gs[0]._projs):
            _cuda_stage().reserve(gs[0]._projs, top_k_of(model))
    if n:
        install_prefill_sync(model)
    _EXPERTS_PER_LAYER.clear()
    if fused_layers and _glu_of(model.layers[fused_layers[0]])._projs[0]._packed:
        _EXPERTS_PER_LAYER.append(
            _glu_of(model.layers[fused_layers[0]])._projs[0]._packed["n_experts"])

    for a, b in zip(fused_layers, fused_layers[1:]):
        # The prefill read-ahead's next layer (_whole_layer).
        _glu_of(model.layers[a]).__dict__["_wide_next"] = _glu_of(model.layers[b])
    for a, b in zip(fused_layers, fused_layers[look_depth:]):
        glu = _glu_of(model.layers[a])
        glu._next_gate = _ranker_of(_moe_of(model.layers[b]))
        glu._next_glu = _glu_of(model.layers[b])
        glu._look_m = look_m
    if native and fused_layers and _glu_of(model.layers[fused_layers[0]])._projs[0]._packed:
        install_native(model, [_glu_of(model.layers[i]) for i in fused_layers],
                       _POOL._max_workers)
        note(f"  native: T=1 decode through the native agent on {len(fused_layers)} "
              f"layers (MOE_NATIVE=1)", flush=True)
    elif native:
        print("  native: no packed streaming layers, Python path", flush=True)
    # MOE_STAGING is decided here, before any store is built and charged, since enable_budget wires the guesses and only a budget store has the rows.
    if _STAGE != "0":
        if _NATIVE is None:
            _stage_off("no native agent")
        elif not hasattr(_NATIVE.agent, "set_staging"):
            _stage_off(f"this moe_native predates it (native/build.sh {sys.executable})")
        elif len({_shape_key(g) for g in _NATIVE.glus}) != 1:
            _stage_off("the experts come in more than one shape, so more than one store")
    return n, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--max-tokens", type=int, default=40)
    ap.add_argument("--prompt", default="Explain in two sentences why the ocean is salty.")
    ap.add_argument("--wired-limit-gb", type=float, default=8.0)
    ap.add_argument("--cache-limit-gb", type=float, default=0.5)
    ap.add_argument("--idle-baseline", type=int, default=10)
    # Speculative decoding attacks both costs at once: the draft proposes k tokens and one pass verifies them.
    ap.add_argument("--draft", default=None,
                    help="draft model for speculative decoding, e.g. mlx-community/ERNIE-4.5-0.3B-PT-4bit")
    ap.add_argument("--num-draft-tokens", type=int, default=4)
    # Bytes a token scales with top_k, so dropping it is the cheapest byte cut, at a quality cost the text has to be checked against.
    ap.add_argument("--top-k", type=int, default=None, help="override router top_k")
    # F_NOCACHE tells the kernel not to retain what it reads, reproducing a device without page-cache slack.
    # Spending the idle memory on resident experts converts memory into reads that never happen.
    ap.add_argument("--resident", type=int, default=0,
                    help="experts kept resident per projection (0=off, needs experts.bin)")
    # Three projections of a layer are indexed by the same experts, so their reads can go out together instead of as three waves of depth top_k.
    # Layer L+1's router applied to layer L's hidden state predicts most of its experts, and reading them during layer L's compute overlaps I/O with the GPU.
    ap.add_argument("--look", type=int, default=0,
                    help="experts speculatively read ahead")
    # Cache-conditional routing (arXiv 2412.00099): the lookahead settings above were tuned against the unbiased router, so look and depth need re-tuning at whatever bias is used.
    ap.add_argument("--cache-bias", type=float, default=0.0,
                    help="bias the router toward resident experts; 0 = off")
    ap.add_argument("--track-used", action="store_true",
                    help="count distinct experts per layer; autotune needs it to "
                         "tell cache-aware routing from a silently smaller model")
    ap.add_argument("--look-depth", type=int, default=1,
                    help="how many layers ahead to speculate")
    ap.add_argument("--threads", type=int, default=8, help="reader threads")
    # Fewer prompt positions per prefill call means fewer unique experts per call, so the transient prefill buffer and the RSS peak shrink.
    ap.add_argument("--prefill-step", type=int, default=0,
                    help="prompt positions per prefill call (0=library default)")
    # Adaptive arena: the cap stops being a constant and follows free memory.
    ap.add_argument("--adaptive", action="store_true",
                    help="grow/shrink the arena with the machine's free memory")
    ap.add_argument("--arena-min", type=int, default=0,
                    help="floor in slots; 0 = arena_floor(top_k) (D3), the "
                         "default every entry point must share since a fixed "
                         "one here used to override it")
    ap.add_argument("--arena-max", type=int, default=512,
                    help="upper bound on slots; the memory signal stops growth")
    ap.add_argument("--reserve-pct", type=int, default=10,
                    help="share of the MACHINE to leave free even when idle")
    # D1: the process's own ceiling on top of --reserve-pct's machine one (ArenaGovernor.budget_gb).
    ap.add_argument("--mem-pct", type=int, default=90,
                    help="process memory budget: MLX memory + --host-gb stays "
                         "under this %% of the machine (default 90)")
    ap.add_argument("--host-gb", type=float, default=0.6,
                    help="python/numpy/MLX-runtime overhead the budget cannot "
                         "see directly (D2 headroom)")
    ap.add_argument("--no-page-cache", action="store_true",
                    help="bypass the OS page cache (simulates a memory-tight device)")
    args = ap.parse_args()

    set_threads(args.threads)
    mx.set_wired_limit(int(args.wired_limit_gb * 1e9))
    mx.set_cache_limit(int(args.cache_limit_gb * 1e9))
    print(f"guards: wired<={args.wired_limit_gb}GB | disk={free_disk_gb():.0f}GB",
          flush=True)

    # A local directory (e.g. the requantised 2-bit build) is used as-is and anything else is resolved through the HF cache.
    local = Path(args.model).expanduser()
    if local.is_dir():
        model_path, args.model = local, str(local)
    else:
        from huggingface_hub import snapshot_download
        model_path = Path(snapshot_download(args.model))

    import resource as _r
    _rss0 = lambda: _r.getrusage(_r.RUSAGE_SELF).ru_maxrss / 1e9
    print(f"  rss before load: {_rss0():.2f}GB", flush=True)
    idle = idle_baseline(args.idle_baseline)
    tele = Telemetry(RESULTS / "streamfast_telemetry.jsonl")
    tele.start()

    t0 = time.time()
    model, tokenizer, slim = load_slim_aware(model_path, args.model)
    if slim:
        print("  slim checkpoint: experts live only in experts.bin", flush=True)
    n_replaced, _st = install(model, model_path, resident=args.resident,
                              look_m=args.look, look_depth=args.look_depth,
                              cache_bias=args.cache_bias)
    global _TRACK_USED
    _TRACK_USED = args.track_used
    if args.no_page_cache:
        import fcntl
        F_NOCACHE = 48
        fds = [f.fileno() for f in _st._fds.values()]
        # experts.bin is opened separately by load_packed, and without this it would keep its page cache while the shards lose theirs, measuring cache state rather than layout.
        if getattr(_st, "packed", None):
            fds.append(_st.packed["fd"].fileno())
        if _STRIPE is not None:
            fds.append(_STRIPE.fd)
        for fd in fds:
            fcntl.fcntl(fd, F_NOCACHE, 1)
        print(f"  F_NOCACHE set on {len(fds)} files: every read is cold", flush=True)
    if args.top_k:
        n_k = set_top_k(model, args.top_k)
        print(f"  router top_k -> {args.top_k} on {n_k} layers", flush=True)
    # Evaluated before the governor is built, since its startup budget check reads mx.get_active_memory() and that has to already include the skeleton or it undercounts held_gb.
    mx.eval(model.parameters())
    if args.adaptive:
        glus = glus_of(model)
        if not glus:
            print("  adaptive arena: no streaming layers found, --adaptive is "
                  "a no-op", flush=True)
        k = top_k_of(model)
        if not glus or k is None:
            raise RuntimeError(
                f"--adaptive cannot read this model: {len(glus)} streaming "
                f"layers, top_k={k}. Both are needed to size the arena.")
        n_exp = n_experts_of(model)
        make_governor(glus, n_exp, k, args.mem_pct, args.reserve_pct,
                      args.host_gb, cap_min=args.arena_min or None,
                      cap_max=args.arena_max)
    print(f"  rss after load+install: {_rss0():.2f}GB", flush=True)
    load_s = round(time.time() - t0, 1)
    print(f"  {n_replaced} projections streaming, resident "
          f"{mx.get_active_memory()/1e9:.2f}GB, load {load_s}s", flush=True)

    draft_model = None
    if args.draft:
        draft_model, _ = mlx_lm.load(args.draft)
        print(f"  draft {args.draft} loaded, k={args.num_draft_tokens}, "
              f"resident now {mx.get_active_memory()/1e9:.2f}GB", flush=True)

    prompt = args.prompt
    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            add_generation_prompt=True, tokenize=False)

    import resource as _res

    def _rss():
        return _res.getrusage(_res.RUSAGE_SELF).ru_maxrss / 1e9

    tg = time.time()
    n, first_s, chunks, rss_decode, mlx_active_decode = 0, None, [], 0.0, 0.0
    gen_kw = {"max_tokens": args.max_tokens}
    if args.prefill_step:
        gen_kw["prefill_step_size"] = args.prefill_step
    if draft_model is not None:
        gen_kw["draft_model"] = draft_model
        gen_kw["num_draft_tokens"] = args.num_draft_tokens
    for resp in mlx_lm.stream_generate(model, tokenizer, prompt, **gen_kw):
        if first_s is None:
            first_s = round(time.time() - tg, 1)
            print(f"  rss after first token: {_rss0():.2f}GB", flush=True)
            print(f"  first token after {first_s}s", flush=True)
        chunks.append(resp.text)
        n += 1
        if n == 3:
            # Baseline taken a few tokens in, after prefill's buffers are gone.
            rss_decode = _rss()
            # D2: ru_maxrss does not see Metal buffers, so autotune's OVER BUDGET gate, which compares against a budget computed from mx.get_active_memory(), was comparing the wrong instrument to the right ceiling.
            mlx_active_decode = mx.get_active_memory() / 1e9
        if n % 5 == 0:
            print(f"  {n} tok, {n/(time.time()-tg):.3f} tok/s, "
                  f"active={mx.get_active_memory()/1e9:.1f}GB", flush=True)

    gen_s = time.time() - tg
    tps = round(n / gen_s, 3) if n else None
    tele.stop()

    dur = len(tele.samples) or 1
    total_mb = sum(s["ssd_read_mb_s"] for s in tele.samples)
    net_mb = max(0.0, total_mb - idle["mb_per_s"] * dur)
    # "hits" means expert-projections the blocking path did not have to read, whether the arena still held them or the lookahead got there first, which is the quantity that decides speed.
    lookups = STATS["need_block"]
    hit_rate = round(100 * STATS["hits"] / lookups, 1) if lookups else 0.0
    out = "".join(chunks)

    # Prefill reads every expert of every layer and lands in the same wall clock as generation, so a single tok/s number hides which of the two is slow and TTFT and decode rate are what a chat feels.
    dec_tps = round((n - 1) / (gen_s - first_s), 2) if n > 1 and gen_s > first_s else None
    print(f"\ntokens={n}  overall={tps} tok/s  DECODE={dec_tps} tok/s  ttft={first_s}s")
    # sync and numpy are measured INSIDE decode and wait_pred INSIDE fetch, so summing every key hid 30-40% of the token in a negative correction and tuned the engine against a profile that did not add up.
    top = sum(v for k, v in TIMERS.items() if k not in _NESTED_TIMERS)
    print("phase timings (s):", {k: round(v, 2) for k, v in TIMERS.items()},
          f"\n  (nested: {', '.join(_NESTED_TIMERS)})"
          f"  measured {top:.1f}s of {gen_s:.1f}s"
          f" -> unmeasured {gen_s - top:.1f}s ({100*(gen_s-top)/max(gen_s,1e-9):.0f}%)")
    if STATS.get("k_calls"):
        print(f"average experts per token: "
              f"{STATS['k_used']/STATS['k_calls']:.2f}")
    print(f"rows blocking={STATS['rows_block']} of needed {STATS['need_block']} "
          f"| rows speculative={STATS['rows_warm']} "
          f"| overlap share={100*STATS['rows_warm']/max(STATS['rows_warm']+STATS['rows_block'],1):.0f}%")
    print(f"expert lookups={lookups}  disk rows={STATS['rows_read']}  "
          f"lru hits={STATS['hits']} ({hit_rate}%)  idx reuse={STATS['index_reuse']}")
    print(f"SSD page-ins net {net_mb:.0f} MB -> {net_mb/max(n,1):.0f} MB/token "
          f"(baseline was 22576)")
    # The 4 GB cap is on the PROCESS, not on MLX's own accounting: the arena, the Python heap and the interpreter count while the OS page cache does not, and ru_maxrss is in bytes on macOS.
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
    print(f"mlx peak={mx.get_peak_memory()/1e9:.2f}GB  process rss={rss:.2f}GB  "
          f"disk={free_disk_gb():.0f}GB")
    print(f"rss at decode start={rss_decode:.2f}GB  mlx active={mlx_active_decode:.2f}GB "
          f"(peak includes prefill transients)")
    if _GOV is not None:
        print(f"arena ended at {_GOV.cap} slots after {len(_GOV.history)} moves: "
              f"{_GOV.history[-4:]}")
    print(f"--- output ---\n{out[:500]}")

    RESULTS.mkdir(exist_ok=True)
    rec = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "mode": "persistent mmap + byte-range slices",
        "lru_size": 0, "projections": n_replaced,
        "draft_model": args.draft, "num_draft_tokens": args.num_draft_tokens if args.draft else None,
        "load_s": load_s, "ttft_s": first_s,
        "gen_tps": tps, "decode_tps": dec_tps, "gen_tokens": n,
        "gen_s": round(gen_s, 1),
        "expert_lookups": lookups, "disk_rows": STATS["rows_read"],
        "lru_hits": STATS["hits"], "cache_hit_rate": hit_rate,
        "pagein_net_mb": round(net_mb, 1),
        "pagein_mb_per_token": round(net_mb / max(n, 1), 1),
        "resident_per_projection": args.resident,
        "mlx_peak_gb": round(mx.get_peak_memory() / 1e9, 2),
        "process_rss_gb": round(rss, 2),
        "disk_after_gb": free_disk_gb(), "swap_after_mb": round(swap_used_mb()),
        "text": out[:600],
    }
    (RESULTS / "stream_fast.json").write_text(
        json.dumps(rec, indent=2, ensure_ascii=False))

    # One machine-readable line for autotune.py: the byte counters and the expert count do not care how loaded the machine is, unlike tok/s, so a tuner can rank configurations on them when it cannot trust the clock.
    used = used_experts()
    need = max(STATS["need_block"], 1)
    print("TUNE " + json.dumps({
        "decode_tps": dec_tps, "rss_gb": rss_decode,
        "mlx_active_gb": round(mlx_active_decode, 2),
        "disk_pct": round(100 * (STATS["rows_block"] + STATS["rows_warm"]) / need, 1),
        "blocking_pct": round(100 * STATS["rows_block"] / need, 1),
                "experts_median": used[len(used) // 2] if used else None,
        "n_experts": _EXPERTS_PER_LAYER[0] if _EXPERTS_PER_LAYER else None}))
    print("wrote stream_fast.json")


if __name__ == "__main__":
    main()
