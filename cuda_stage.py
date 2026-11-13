"""MOE_CUDA_STAGE (on by default when MLX runs on CUDA, MOE_CUDA_STAGE=0 turns it off, never on
Metal or CPU): expert reads for MLX on CUDA land in a staging buffer whose pages stay on the
host, so a read wave neither migrates store pages nor waits on the GPU, with the same bytes in
every slot as a plain read gives."""
import ctypes
import os
import threading
from concurrent.futures import as_completed

import numpy as np
import mlx.core as mx
from engine_log import note

RECORDS = 32            # staging room, in expert records: one decode wave is <= top_k x 3 projections
THREADS = 512           # per block; one block copies one part of one record
BATCHES = 3             # copy launches per wave at most (one more for a straggler)
PAGE = 4096             # O_DIRECT's alignment, which the drive's logical block is at most
# MOE_CUDA_SPANS (on by default, 0 turns it off): a record of any size is read through O_DIRECT as its page-aligned span, and the layer-major prefill's groups go through stream_fast._LMReader's barrier-free reader queue.
SPANS = os.environ.get("MOE_CUDA_SPANS", "1") == "1"

# One block per table row (dst part, dst slot, src offset in 16 B, 16 B units, src slot): the
# part's row of 16-byte units from the staging buffer, or (src slot >= 0) from another slot of
# the same part (a store resident kept by a layer-major union read).
_SRC = """
    const uint32_t r = blockIdx.x;
    const int32_t* t = table + 5 * r;
    uint4* b;
    switch (t[0]) {
%s
        default: return;
    }
    uint4* d = b + (uint64_t)t[1] * (uint64_t)t[3];
    const uint4* s = t[4] < 0 ? (const uint4*)src + (uint64_t)t[2]
                              : (const uint4*)b + (uint64_t)t[4] * (uint64_t)t[3];
    for (uint32_t i = threadIdx.x; i < (uint32_t)t[3]; i += blockDim.x) d[i] = s[i];
    if (r == 0 && threadIdx.x == 0) done[0] = 1;
"""


def room(stride):
    """A staging record's bytes: under MOE_CUDA_SPANS a page multiple that holds the record's aligned span wherever it starts, else the record."""
    return -(-stride // PAGE) * PAGE + PAGE if SPANS else stride


def lead(d, off):
    """Where the record at file offset `off` starts in its room: its place in its page when it goes through O_DIRECT fd `d` and stays 16-byte aligned for the copy kernel, else 0."""
    return off % PAGE if d is not None and off % 16 == 0 else 0


def read_span(job):
    """One staging read (fd, room, offset, bytes, O_DIRECT fd or None) that puts the record at lead() in its page-aligned room, through O_DIRECT as native/moe_pipe_cuda.cpp's pread_direct does or else through the page cache, and says whether O_DIRECT served it."""
    fd, buf, off, n, d = job
    at = lead(d, off)
    if d is not None and off % 16 == 0:
        a, need = off - at, at + n
        span, got = -(-need // PAGE) * PAGE, 0
        try:
            while got < need:
                r = os.preadv(d, [buf[got:span]], a + got)
                if r <= 0 or (got + r < need and r % PAGE):
                    break
                got += r
        except OSError:
            pass
        if got >= need:
            return True
    got = os.preadv(fd, [buf[at:at + n]], off)
    if got != n:
        raise OSError(f"short staging read {got} of {n} at {off}")
    return False


def _cuda():
    """libcuda for cuMemAdvise and the copy engine, or None, when plain managed memory and the copy
    kernel still give the same bytes."""
    try:
        cu = ctypes.CDLL("libcuda.so.1")
        P = ctypes.c_void_p
        cu.cuMemAdvise.argtypes = [ctypes.c_uint64, ctypes.c_size_t, ctypes.c_int, ctypes.c_int]
        cu.cuMemHostAlloc.argtypes = [ctypes.POINTER(P), ctypes.c_size_t, ctypes.c_uint]
        cu.cuMemcpyHtoDAsync_v2.argtypes = [ctypes.c_uint64, P, ctypes.c_size_t, P]
        cu.cuStreamCreate.argtypes = [ctypes.POINTER(P), ctypes.c_uint]
        cu.cuStreamSynchronize.argtypes = [P]
        dev = ctypes.c_int()
        if cu.cuCtxGetDevice(ctypes.byref(dev)) != 0:
            return None, None
        return cu, dev.value
    except OSError:
        return None, None


class Staging:
    def __init__(self, do_read):
        self.do_read = do_read          # stream_fast._do_read, looked up per call
        self.stride = 0
        self.arr = self.view = None
        self.direct = {}                # buffered fd -> O_DIRECT fd, or None where it cannot be used
        self.kernels = {}
        self.aligned = False
        self.cu, self.dev = _cuda()
        self.advise, self.lm, self.lm_busy = None, None, False
        # MOE_CUDA_DMA (on by default; =0 for the copy kernel): decode waves land in pinned host
        # memory and the copy engine moves each record to its slot as soon as it is read.
        self.dma = self.cu is not None and os.environ.get("MOE_CUDA_DMA", "1") == "1"
        self.stream = None
        self.stats = {"span_direct": 0, "span_buffered": 0}
        self.lock = threading.Lock()

    def read(self, job):
        """read_span with its path counted."""
        k = "span_direct" if read_span(job) else "span_buffered"
        with self.lock:
            self.stats[k] += 1

    def _managed(self, nbytes):
        """A staging buffer of managed memory whose pages stay on the host and that the GPU reads
        in place, as (array, uint8 view)."""
        arr = mx.zeros(((nbytes + 15) // 16 * 4,), dtype=mx.uint32)
        mx.eval(arr)
        view = np.array(arr, copy=False).view(np.uint8)     # managed from here on
        p = view.ctypes.data
        adv = None
        if self.cu is not None:
            adv = (self.cu.cuMemAdvise(p, view.nbytes, 3, -1), self.cu.cuMemAdvise(p, view.nbytes, 5, self.dev))
        view[:] = 0                                          # its pages on the host
        self.advise = adv
        return arr, view

    def reserve(self, projs, top_k):
        """Both staging buffers at install, before the memory budget measures anything, since
        allocated inside a prefill they would read as that prefill's growth."""
        strides = [p._packed["stride"] for p in projs]
        self._room(max(strides))
        if self.lm is None:
            self.lm = self._managed(LMStage.REGIONS * top_k * sum(room(s) for s in strides))

    def _room(self, stride):
        if stride <= self.stride:
            return
        self.stride = stride
        self.arr = self.view = None
        if self.dma:
            self.view = self._pinned(RECORDS * room(stride))
            self.dma = self.view is not None
        if not self.dma:
            self.arr, self.view = self._managed(RECORDS * room(stride))
        p = self.view.ctypes.data
        self.aligned = p % PAGE == 0 and (SPANS or stride % PAGE == 0)
        how = "pinned, copy engine" if self.dma else f"managed, copy kernel, advise {self.advise}"
        on = "on (aligned spans)" if SPANS else "on"
        note(f"  cuda stage: {RECORDS} x {stride / 1e6:.2f} MB staging ({how}), "
              f"O_DIRECT {on if self.aligned else 'off (alignment)'}", flush=True)

    def _pinned(self, nbytes):
        """Page-locked host memory (cuMemHostAlloc) as a uint8 view and the copy stream, or None if
        the driver refuses either, never freed as the record size only grows once per process."""
        P = ctypes.c_void_p
        hp, st = P(), P()
        if self.cu.cuMemHostAlloc(ctypes.byref(hp), nbytes, 0) != 0:
            return None
        if self.stream is None:
            if self.cu.cuStreamCreate(ctypes.byref(st), 1) != 0:       # CU_STREAM_NON_BLOCKING
                return None
            self.stream = st
        return np.ctypeslib.as_array((ctypes.c_uint8 * nbytes).from_address(hp.value))

    def _direct_fd(self, fd):
        if fd not in self.direct:
            d = None
            if self.aligned and hasattr(os, "O_DIRECT"):
                try:
                    d = os.open(os.readlink(f"/proc/self/fd/{fd}"), os.O_RDONLY | os.O_DIRECT)
                except OSError:
                    d = None
            self.direct[fd] = d
        return self.direct[fd]

    def _kernel(self, n):
        k = self.kernels.get(n)
        if k is None:
            cases = "\n".join(f"        case {i}: b = (uint4*)dst{i}; break;" for i in range(n))
            k = self.kernels[n] = mx.fast.cuda_kernel(
                name=f"nn_rows_to_slots_{n}", input_names=["src"] + [f"dst{i}" for i in range(n)] + ["table"],
                output_names=["done"], source=_SRC % cases)
        return k

    def wave(self, jobs, ar, pool, by_stride=False, buffered=False):
        """Read `jobs` ((fd, iov, offset, stride) as _claimed builds them, every iov a slot of one
        of `ar`'s projections) and return when their slots are written; `by_stride` (MOE_RESIDENT, whose
        store the host cannot write) chunks each record size apart so none takes the plain read, and
        `buffered` reads through the page cache instead of O_DIRECT."""
        if by_stride:
            for st in sorted({j[3] for j in jobs}):
                one = [j for j in jobs if j[3] == st]
                for i in range(0, len(one), RECORDS):
                    self._chunk(one[i:i + RECORDS], ar, pool, buffered)
            return
        for i in range(0, len(jobs), RECORDS):
            self._chunk(jobs[i:i + RECORDS], ar, pool, buffered)

    def _chunk(self, jobs, ar, pool, buffered=False):
        stride = jobs[0][3]
        self._room(stride)
        # Where each projection's parts live: slot-0 address and row bytes of every part.
        proj = []
        for bufs, iov, fd, off, st in ar:
            v0 = iov[0]
            proj.append(([v.ctypes.data for v in v0], [v.nbytes for v in v0], len(iov), bufs))
        first = [sum(len(q[1]) for q in proj[:j]) for j in range(len(proj))]   # index into dsts
        rows, staged, plain, dma, starts = [], [], [], [], []
        for job in jobs:
            fd, jiov, foff, st = job
            p0 = jiov[0].ctypes.data
            hit = None
            for j, (bases, nb, slots, _) in enumerate(proj):
                if bases[0] <= p0 < bases[0] + nb[0] * slots:
                    sl = (p0 - bases[0]) // nb[0]
                    if (st == stride and sum(nb) == st and len(jiov) == len(nb)
                            and all(jiov[k].ctypes.data == bases[k] + sl * nb[k] and nb[k] % 16 == 0
                                    for k in range(len(nb)))):
                        hit = (j, sl)
                    break
            if hit is None:
                plain.append(job)       # not a slot of this arena: the ordinary read, straight in
                continue
            j, sl = hit
            n = len(staged)
            staged.append((fd, foff, n))
            starts.append(len(rows))
            src = n * room(stride) + lead(None if buffered else self._direct_fd(fd), foff) if SPANS else n * stride
            for k, b in enumerate(proj[j][1]):      # parts lie in the record in iov order
                rows.append((first[j] + k, sl, src // 16, b // 16, -1))
                dma.append((n, jiov[k].ctypes.data, src, b))
                src += b
        dfd = {fd: None if buffered else self._direct_fd(fd) for fd, _, _ in staged}

        def run(w):
            if w[0] == "p":
                self.do_read(w[1])
                return
            fd, foff, n = w[1]
            if SPANS:
                r = room(stride)
                self.read((fd, self.view[n * r:(n + 1) * r], foff, stride, dfd[fd]))
                return
            view = self.view[n * stride:(n + 1) * stride]
            if dfd[fd] is not None:
                try:
                    self.do_read((dfd[fd], [view], foff, stride))
                    return
                except (OSError, AssertionError):
                    pass                       # read again through the page cache
            self.do_read((fd, [view], foff, stride))

        work = [("s", x) for x in staged] + [("p", x) for x in plain]
        if not rows:
            if len(work) > 1:
                list(pool.map(run, work))
            elif work:
                run(work[0])
            return
        if self.dma:
            self._dma(work, dma, pool, run)
            return
        # Copies start as records land, a batch at a time, so the GPU moves the first records over
        # PCIe while the SSD reads the last and the wave only waits for its last batch's copy, each
        # record's rows going in exactly one batch.
        dsts = [b for (_, _, _, bufs) in proj for b in bufs]
        kern = self._kernel(len(dsts))
        starts.append(len(rows))        # record n's rows are rows[starts[n]:starts[n + 1]]
        batch = max(1, -(-len(staged) // BATCHES))
        futs = {pool.submit(run, w): w for w in work}
        ready, outs = [], []
        for f in as_completed(futs):
            f.result()
            w = futs[f]
            if w[0] == "s":
                ready.append(w[1][2])
            if len(ready) >= batch or (ready and all(x.done() for x in futs)):
                sel = [r for n in sorted(ready) for r in rows[starts[n]:starts[n + 1]]]
                ready = []
                outs += kern(inputs=[self.arr, *dsts, mx.array(np.array(sel, dtype=np.int32))],
                             grid=(len(sel) * THREADS, 1, 1), threadgroup=(THREADS, 1, 1),
                             output_shapes=[(1,)], output_dtypes=[mx.int32])
                mx.async_eval(outs[-1])
        if ready:
            sel = [r for n in sorted(ready) for r in rows[starts[n]:starts[n + 1]]]
            outs += kern(inputs=[self.arr, *dsts, mx.array(np.array(sel, dtype=np.int32))],
                         grid=(len(sel) * THREADS, 1, 1), threadgroup=(THREADS, 1, 1),
                         output_shapes=[(1,)], output_dtypes=[mx.int32])
        mx.eval(outs)

    def copy_in(self, copies):
        """MOE_RESIDENT: (device address, pinned host address, bytes) copies on the copy stream, waited for."""
        cu, P = self.cu, ctypes.c_void_p
        if cu is None:
            raise RuntimeError("cuda stage: no libcuda for the tier's copies")
        if self.stream is None:
            st = P()
            if cu.cuStreamCreate(ctypes.byref(st), 1) != 0:        # CU_STREAM_NON_BLOCKING
                raise RuntimeError("cuda stage: cuStreamCreate failed")
            self.stream = st
        for d, s, n in copies:
            rc = cu.cuMemcpyHtoDAsync_v2(d, s, n, self.stream)
            if rc != 0:
                raise RuntimeError(f"cuda stage: cuMemcpyHtoDAsync failed ({rc})")
        rc = cu.cuStreamSynchronize(self.stream)
        if rc != 0:
            raise RuntimeError(f"cuda stage: cuStreamSynchronize failed ({rc})")

    def _dma(self, work, dma, pool, run):
        """The copy engine: each staged record's parts go to their slots (cuMemcpyHtoDAsync on a
        stream of their own) as its read returns, with one stream sync at the end, no kernel and
        the copy kernel's exactness."""
        by = {}
        for n, dst, off, b in dma:
            by.setdefault(n, []).append((dst, off, b))
        base, cu, st = self.view.ctypes.data, self.cu, self.stream
        futs = {pool.submit(run, w): w for w in work}
        for f in as_completed(futs):
            f.result()
            w = futs[f]
            if w[0] != "s":
                continue
            for dst, off, b in by[w[1][2]]:          # off: the part's place in the staging buffer
                rc = cu.cuMemcpyHtoDAsync_v2(dst, base + off, b, st)
                if rc != 0:
                    raise RuntimeError(f"cuda stage: cuMemcpyHtoDAsync failed ({rc})")
        rc = cu.cuStreamSynchronize(st)
        if rc != 0:
            raise RuntimeError(f"cuda stage: cuStreamSynchronize failed ({rc})")


def install_submit(layers, run_ahead):
    """MOE_CUDA_SUBMIT (on by default under CUDA, =0 turns it off): in one-position decode outside
    run-ahead, hand each decoder layer's input and its attention block's output to the GPU as
    soon as they are built, changing only the order of submission and `run_ahead()` keeps the
    layers it owns."""
    done = set()

    def wrap(cls, before):
        if cls in done:
            return
        done.add(cls)
        orig = cls.__call__

        def first(self, *a, **k):
            x = a[0] if a else None
            if x is not None and x.ndim >= 2 and x.shape[1] == 1 and not run_ahead():
                mx.async_eval(x)
            return orig(self, *a, **k)

        def after(self, *a, **k):
            out = orig(self, *a, **k)
            if isinstance(out, mx.array) and out.ndim >= 2 and out.shape[1] == 1 and not run_ahead():
                mx.async_eval(out)
            return out

        cls.__call__ = first if before else after

    for layer in layers:
        wrap(type(layer), True)
        for name in ("linear_attn", "self_attn"):
            m = getattr(layer, name, None)
            if m is not None:
                wrap(type(m), False)
    note(f"  cuda submit: layer inputs and attention outputs go to the GPU as they are built", flush=True)


class LMStage:
    """MOE_CUDA_STAGE, the layer-major prefill (stream_fast._LMReader): the reader thread reads a
    group of experts into one of REGIONS host-resident staging regions, take() queues their copy
    into the range's store rows, and give() frees the region, so the store's pages never come to
    the host."""
    REGIONS = 4
    queued = SPANS          # the groups go through stream_fast._LMReader's reader queue

    def __init__(self, staging, store):
        import threading
        from concurrent.futures import ThreadPoolExecutor
        self.store, self.g = store, store.per_half
        # Each projection's record (a GGUF layer's gate / up and down can differ), its staging room and parts.
        glus = getattr(store, "glus", None)
        self.strides = ([p._packed["stride"] for p in glus[0]._projs] if glus else
                        [store.slot_bytes // len(store.bufs)] * len(store.bufs))
        self.rooms = [room(s) for s in self.strides]
        self.nproj = len(store.bufs)
        self.parts = [[v[0].nbytes for v in vs] for vs in store.views]
        self.region = self.g * sum(self.rooms)
        n = self.REGIONS * self.region
        self.staging, self.shared = staging, False
        if staging.lm is not None and staging.lm[1].nbytes >= n and not staging.lm_busy:
            self.arr, self.view = staging.lm                 # reserved at install
            self.shared = staging.lm_busy = True
        else:
            self.arr, self.view = staging._managed(n)
        self.aligned = self.view.ctypes.data % PAGE == 0 and (SPANS or all(s % PAGE == 0 for s in self.strides))
        self.kern = staging._kernel(sum(len(b) for b in store.bufs))
        self.pool = ThreadPoolExecutor(8)
        self.cv = threading.Condition()
        self.free = list(range(self.REGIONS))
        self.ready, self.owner = {}, {}
        self.stop = False

    def _at(self, s, n, j, i):
        """Staging offset of record i of projection j in region s holding n experts: projection j's n rooms follow
        the earlier projections' (with equal records, (j * n + i) * room as before)."""
        return s * self.region + n * sum(self.rooms[:j]) + i * self.rooms[j]

    def jobs(self, glu, ids, r0, copies, skip=()):
        """Reader thread: a free region for glu's experts `ids` and the reads (fd, room, offset,
        bytes, O_DIRECT fd) that fill it, but for those in `copies` (position in ids to store
        slot) which take() copies slot to slot and those in `skip`."""
        with self.cv:
            while not self.free and not self.stop:
                self.cv.wait()
            if self.stop:
                raise RuntimeError("layer-major prefill closed")
            s = self.free.pop()
        n, jobs, src = len(ids), [], {}
        for j, q in enumerate(glu._projs):
            pk, st, r = q._packed, self.strides[j], self.rooms[j]
            if pk["stride"] != st:
                raise ValueError(f"cuda stage: a layer whose projection {j} record ({pk['stride']} B) is not its "
                                 f"store's ({st} B)")
            fd = pk["fd"]
            d = self.staging._direct_fd(fd) if self.aligned else None
            for i, e in enumerate(ids.tolist()):
                if i in copies or i in skip:
                    continue
                at, off = self._at(s, n, j, i), pk["offset"] + e * st
                src[j, i] = at + lead(d, off) if SPANS else at
                jobs.append((fd, self.view[at:at + r], off, st, d))
        self.ready[r0] = (s, n, copies, skip, src)
        return jobs

    def read(self, glu, ids, r0, copies, skip=()):
        """Reader thread: jobs() read here, returning the bytes read."""
        jobs = self.jobs(glu, ids, r0, copies, skip)
        list(self.pool.map(self.one, jobs))
        return sum(j[3] for j in jobs)

    def one(self, job):
        """One read of jobs(), whoever runs it."""
        if SPANS:
            self.staging.read(job)
            return
        fd, v, off, st, d = job
        if d is not None:
            try:
                if os.preadv(d, [v], off) == st:
                    return
            except OSError:
                pass                                    # read again through the page cache
        got = os.preadv(fd, [v], off)
        if got != st:
            raise OSError(f"short prefill read {got} of {st}")

    def take(self, r0):
        """Model thread: queue the copies of r0's group from staging to store rows without waiting,
        the gathers that read those rows being queued after them."""
        s, n, copies, skip, at = self.ready.pop(r0)
        self.owner[r0] = s
        rows, first = [], 0
        for j in range(self.nproj):
            for i in range(n):
                if i in skip:
                    continue
                src = at.get((j, i), self._at(s, n, j, i))
                for k, b in enumerate(self.parts[j]):
                    rows.append((first + k, r0 + i, src // 16, b // 16, copies.get(i, -1)))
                    src += b
            first += len(self.parts[j])
        dsts = [b for bufs in self.store.bufs for b in bufs]
        out = self.kern(inputs=[self.arr, *dsts, mx.array(np.array(rows, dtype=np.int32))],
                        grid=(len(rows) * THREADS, 1, 1), threadgroup=(THREADS, 1, 1),
                        output_shapes=[(1,)], output_dtypes=[mx.int32])
        mx.async_eval(out)

    def give(self, r0):
        """Model thread, after r0's gathers were evaluated (so its copy ran too)."""
        s = self.owner.pop(r0)
        with self.cv:
            self.free.append(s)
            self.cv.notify_all()

    def close(self):
        with self.cv:
            self.stop = True
            self.cv.notify_all()
        self.pool.shutdown(wait=True)
        if self.shared:
            self.staging.lm_busy = self.shared = False
