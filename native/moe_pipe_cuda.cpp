// Native expert agent for the MLX CUDA backend, the moe_native.cpp agent ported to Linux and CUDA.
//
// This file is plain C++ with a C ABI for ctypes, with no MLX headers and no nanobind, and libcuda comes in through dlopen, so nothing here is
// pinned to the wheel's MLX or compiler.
//
// The GPU hands each MoE layer's router ids to a host thread through a mapped pinned block, where the host pointer is also the device pointer, and
// the thread claims slots with the engine's LRU, writes the layer's copy plan and raises `claimed`, then reads the misses into mapped pinned
// staging and flags each record as it lands.
//
// The wait kernel's blocks copy each record into its slot as soon as its flag is up, so reads and copies overlap, and the gather runs after the
// wait kernel, whose kernels are mx.fast.cuda_kernels in moe_pipe_cuda.py.
//
// The agent makes no CUDA call while it serves a job, because a copy-engine copy into the engine's managed arena did not run while our kernel held
// the GPU, so nothing here may depend on what the driver does while the GPU is parked.
//
// Block words: 0 posted the GPU's last job, 1 done the agent's last answered job, 2 release the sticky word that frees every wait, 3 gave the job
// of a wait that hit its bound, 4 beat the heartbeat, 5 sent the jobs the host committed, 6 beating 1 while a heartbeat is live, 7 claimed the
// agent's last job whose claim and plan are written, and 10 noticed the agent's last posted job it has seen.
//
// Words 12 to 14 hold the heartbeat's longest gap in us, the beat thread's longest gap between bumps and its gaps over 1 ms.
//
// Job v sits at 64 + 128 * (v % RING) as [v, layer, n, ids[n]], its miss counts at 1536 + (v % 128) as misses | tier hits << 8 | experts put in the
// tier << 16, and the wait's log under MOE_PIPE_LOG=1 at 2048 + 16 * (v % 128) with 8 uint64 a job.
//
// The agent's log at 4096 + 10 * (v % 128) holds 5 uint64 a job for v, noticed, claimed, records landed and answered in ns, and 6144 + r holds the
// job's v once record r has landed in staging or is in the tier.
//
// The plan at 8192 is [v, copies, records, 0, copies x (dst low, dst high, src low, src high, bytes, record, save low, save high)], one copy a
// chunk of a part from staging or the tier, with save (0 for none) where the chunk's old bytes go first, the evicted expert's tier slot.
//
// Only the GPU raises posted, with its own counter, so a job built but never run takes no number, and only the agent raises claimed and done.
//
// A job's ring entry is written and fenced before posted moves, and the agent answers jobs in order, so the GPU is never more than one job ahead
// while the waits hold.
//
// After a release or a wait that gave up the waits fall through, the GPU runs ahead and the agent answers without claiming, and a plan that is not
// the job's own is never copied since that pass fails at its boundary.
//
// A killed process whose kernel still runs a few ms later leaves the GPU busy at 100% and 29 W until another CUDA context is made, and
// moe_pipe_cuda.py's guard process makes one.
//
// Opt-in instead, MOE_PIPE_BEAT_MS=2 gives a heartbeat bumped every 0.25 ms and a wait that gives up when it has not moved for that long, so a
// killed engine's wait ends about 2 ms after its last beat, but the beat thread's own gaps reach about 3 ms and a live wait that gives up fails its
// pass.
//
// Reads go through an O_DIRECT fd on the same file unless MOE_PIPE_DIRECT=0 asks for buffered reads, and each record's 4 KiB-aligned span lands in
// its own aligned staging record with the copy pointed inside it, so the slot gets the same bytes.
//
// The victim tier of set_tier is pinned mapped host memory with one expert a tier slot and its records back to back as in the file, and an expert
// the store evicts goes in when the wait kernel's block that writes the new part first copies the old part out, exclusive of the store with the
// oldest put out first.
//
// Staging (set_staging, MOE_PIPE_STAGING=1): the GPU posts its guess of layer L + D's experts at 7168 + 32 * (v % RING) as [v, n, ids[n]]
// right after job v of layer L (D = MOE_PIPE_GUESS_D, 1 or 2), and once job v's own records have landed the agent queues the guessed experts
// layer L + D and the tier do not hold.
//
// The guess threads, one for each guess record allowed in flight, read them into guess records past the staging records, D + 1 groups used
// by turns, oldest batch first and never while a job's own records are read, and L + D's claim copies a guessed miss from its guess record,
// promotes one still in flight and drops those not started.
//
// The claims, the tables and the tier are the same with staging on or off, so only where a miss's bytes come from changes.
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#ifdef __linux__
#include <linux/io_uring.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#endif
#include <sched.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

static uint64_t now_ns() {
  timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return uint64_t(t.tv_sec) * 1000000000ull + t.tv_nsec;
}

// ---------------------------------------------------------------------------
// The libcuda driver calls this file uses, loaded through dlopen.

typedef int CUresult;
typedef void* CUcontext;
typedef unsigned long long CUdeviceptr;

static struct {
  CUresult (*init)(unsigned);
  CUresult (*device_get)(int*, int);
  CUresult (*ctx_get)(CUcontext*);
  CUresult (*ctx_set)(CUcontext);
  CUresult (*primary_retain)(CUcontext*, int);
  CUresult (*host_alloc)(void**, size_t, unsigned);
  CUresult (*host_free)(void*);
  CUresult (*host_devptr)(CUdeviceptr*, void*, unsigned);
} cu;

static std::string g_err; // The last C ABI error, which Python reads while it holds the GIL.

static void load_cuda() {
  void* h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
  if (!h)
    throw std::runtime_error(std::string("dlopen libcuda.so.1: ") + dlerror());
  auto sym = [&](const char* name) {
    void* p = dlsym(h, name);
    if (!p)
      throw std::runtime_error(std::string("libcuda has no ") + name);
    return p;
  };
  cu.init = (CUresult(*)(unsigned))sym("cuInit");
  cu.device_get = (CUresult(*)(int*, int))sym("cuDeviceGet");
  cu.ctx_get = (CUresult(*)(CUcontext*))sym("cuCtxGetCurrent");
  cu.ctx_set = (CUresult(*)(CUcontext))sym("cuCtxSetCurrent");
  cu.primary_retain = (CUresult(*)(CUcontext*, int))sym("cuDevicePrimaryCtxRetain");
  cu.host_alloc = (CUresult(*)(void**, size_t, unsigned))sym("cuMemHostAlloc");
  cu.host_free = (CUresult(*)(void*))sym("cuMemFreeHost");
  cu.host_devptr = (CUresult(*)(CUdeviceptr*, void*, unsigned))sym("cuMemHostGetDevicePointer_v2");
}

static void check(CUresult rc, const char* what) {
  if (rc != 0)
    throw std::runtime_error(std::string(what) + " failed (" + std::to_string(rc) + ")");
}

// ---------------------------------------------------------------------------
// The block, the stop words and the exit signals.

enum { kPosted = 0, kDone = 1, kRelease = 2, kGave = 3, kBeat = 4, kSent = 5, kBeating = 6, kClaimed = 7,
       kNoticed = 10, kRing = 64, kMiss = 1536, kLog = 2048, kAgentLog = 4096, kReady = 6144, kGuess = 7168, kPlan = 8192 };
static const int RING = 8, ENTRY = 128, MAX_IDS = ENTRY - 3, BLOCK_BYTES = 65536;
static const int STAGE_RECORDS = 48;  // Staging room: a top-10 layer of three projections is 30 records.
static const int GUESS_WORDS = 32, GUESS_RECORDS = 30;  // A guess entry's words, and the records of one half of the guess room.
static const int PLAN_WORDS = 8, PLAN_COPIES = 1020;  // The most copies a job may have, since 8192 + 4 + 8 x 1020 words fit the block.
static const size_t PAGE = 4096;      // O_DIRECT's alignment, which the drive's logical block is at most.

static CUcontext g_ctx = nullptr;
static std::atomic<uint32_t>* g_blk = nullptr;
static CUdeviceptr g_blk_dev = 0;

static std::atomic<uint32_t>& W(int i) {
  return g_blk[i];
}

// Every wait of this process falls through now and later, and this stays async-signal-safe.
static void release_all() {
  if (g_blk)
    W(kRelease).store(1, std::memory_order_release);
}

static bool pending() {
  return g_blk && W(kSent).load() > W(kDone).load();
}

// SIGTERM and SIGHUP release every wait and then chain to what was there before, and a later
// signal.signal for the same signal replaces this handler.
static struct sigaction g_prev[NSIG];

static void on_exit_signal(int sig, siginfo_t* info, void* ctx) {
  static const char a[] = "moe_pipe_cuda: exit signal, GPU waits released (one was pending)\n";
  static const char b[] = "moe_pipe_cuda: exit signal, GPU waits released (none pending)\n";
  release_all();
  ssize_t ignored = pending() ? write(2, a, sizeof(a) - 1) : write(2, b, sizeof(b) - 1);
  (void)ignored;
  auto& p = g_prev[sig];
  if (p.sa_flags & SA_SIGINFO) {
    p.sa_sigaction(sig, info, ctx);
  } else if (p.sa_handler != SIG_DFL && p.sa_handler != SIG_IGN) {
    p.sa_handler(sig);
  } else {
    signal(sig, SIG_DFL);
    raise(sig);
  }
}

static void hook_exit_signals() {
  for (int sig : {SIGTERM, SIGHUP}) {
    struct sigaction prev = {}, sa = {};
    sigaction(sig, nullptr, &prev);
    if (!(prev.sa_flags & SA_SIGINFO) && prev.sa_handler == SIG_IGN)
      continue;
    g_prev[sig] = prev;
    sa.sa_sigaction = on_exit_signal;
    sa.sa_flags = SA_SIGINFO | SA_RESTART;
    sigemptyset(&sa.sa_mask);
    sigaction(sig, &sa, nullptr);
  }
}

// Run once per process on the thread that holds MLX's context, which the caller has made.
static void init() {
  if (g_blk)
    return;
  load_cuda();
  check(cu.init(0), "cuInit");
  check(cu.ctx_get(&g_ctx), "cuCtxGetCurrent");
  if (!g_ctx) {
    int d = 0;
    check(cu.device_get(&d, 0), "cuDeviceGet");
    check(cu.primary_retain(&g_ctx, d), "cuDevicePrimaryCtxRetain");
    check(cu.ctx_set(g_ctx), "cuCtxSetCurrent");
  }
  void* h = nullptr;
  check(cu.host_alloc(&h, BLOCK_BYTES, 0x01 | 0x02), "cuMemHostAlloc (mapped block)"); // PORTABLE | DEVICEMAP, so the host pointer maps to the device.
  memset(h, 0, BLOCK_BYTES);
  check(cu.host_devptr(&g_blk_dev, h, 0), "cuMemHostGetDevicePointer");
  g_blk = static_cast<std::atomic<uint32_t>*>(h);
  hook_exit_signals();
}

// ---------------------------------------------------------------------------
// The reader pool runs n jobs a wave and the caller reads too.

class Pool {
 public:
  explicit Pool(int workers) {
    for (int i = 0; i < workers; i++)
      th_.emplace_back([this] { loop(); });
  }
  ~Pool() {
    close();
  }
  void close() {
    {
      std::lock_guard lk(m_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto& t : th_)
      t.join();
    th_.clear();
  }
  void run(size_t n, const std::function<void(size_t)>& f) {
    if (!n)
      return;
    auto w = std::make_shared<Wave>(n, f);
    if (n > 1 && !th_.empty()) {
      {
        std::lock_guard lk(m_);
        cur_ = w;
        gen_++;
      }
      cv_.notify_all();
    }
    work(*w);
    while (w->done.load(std::memory_order_acquire) < n) {
    }
  }

 private:
  // A wave is its own object, so a late worker finishes only its own jobs.
  struct Wave {
    Wave(size_t n, const std::function<void(size_t)>& f) : n(n), f(f) {}
    size_t n;
    std::function<void(size_t)> f;
    std::atomic<size_t> next{0}, done{0};
  };
  static void work(Wave& w) {
    for (size_t i; (i = w.next.fetch_add(1)) < w.n;) {
      w.f(i);
      w.done.fetch_add(1, std::memory_order_release);
    }
  }
  void loop() {
    uint64_t seen = 0;
    std::unique_lock lk(m_);
    for (;;) {
      cv_.wait(lk, [&] { return stop_ || gen_ != seen; });
      if (stop_)
        return;
      seen = gen_;
      auto w = cur_;
      lk.unlock();
      work(*w);
      w.reset();
      lk.lock();
    }
  }
  std::vector<std::thread> th_;
  std::mutex m_;
  std::condition_variable cv_;
  std::shared_ptr<Wave> cur_;
  uint64_t gen_ = 0;
  bool stop_ = false;
};

// ---------------------------------------------------------------------------
// The agent: its layers, tables and threads.

struct Layer {
  int cap = 0, n_exp = 0;
  int32_t* slot_of = nullptr; // The engine's expert to slot table in managed memory, with -1 for absent.
  std::vector<int> owner;     // Slot, or quota position in a store, to expert, with -1 for empty.
  std::vector<uint64_t> stamp; // Slot, or quota position in a store, to the time of its last use.
  std::vector<uint32_t> seen; // Expert to the claims since take_seen.
  uint64_t clock = 0;
  // Store layers also keep the quota to store row map, its reverse with -1 for rows outside
  // the quota, the ring half and every claim's ids since take_log.
  int ring = -1;
  std::vector<int> qslot, qpos;
  std::vector<int32_t> log_ids, log_len;
  struct Proj {
    off_t off;
    size_t stride;
    std::vector<std::pair<uint64_t, size_t>> parts; // The address of the slot 0 row and the row's bytes.
    bool direct = false; // Read through the O_DIRECT fd when records are 16-byte aligned, which keeps the GPU's copy wide.
    size_t at = 0;       // Where its record sits in a tier slot.
  };
  std::vector<Proj> projs;
  bool tier = false;     // True when its experts fit a tier slot.

  int slot_at(int p) const {
    return ring < 0 ? p : qslot[p];
  }
  int pos_of(int s) const {
    return ring < 0 ? s : qpos.at(s);
  }
};

struct Ring {
  int base = 0, next = 0;
  std::vector<std::pair<int, int>> owner; // The layer and expert it holds, with layer -1 for empty.
};

static bool pread_full(int fd, char* p, size_t n, off_t off) {
  while (n) {
    ssize_t got = pread(fd, p, n, off);
    if (got <= 0)
      return false;
    p += got;
    n -= got;
    off += got;
  }
  return true;
}

static size_t round_up(size_t n, size_t a) {
  return (n + a - 1) / a * a;
}

// Read `span` bytes from the aligned offset `a` through O_DIRECT, needing at least `need`, and
// return false so the caller reads buffered.
static bool pread_direct(int fd, char* p, size_t span, off_t a, size_t need) {
  size_t got = 0;
  while (got < need) {
    ssize_t r = pread(fd, p + got, span - got, a + off_t(got));
    if (r <= 0 || (got + r < need && (r % PAGE)))
      return false;
    got += r;
  }
  return true;
}

// One thread's io_uring through raw syscalls (no liburing), for the pieces of a record: every piece queued at once, so the drive sees the
// same requests as one read of the whole span, and each piece reported as it lands; ok stays false where the kernel has none.
struct Uring {
  bool ok = false, tried = false;
#ifdef __linux__
  int fd = -1;
  unsigned *sq_tail = nullptr, *sq_mask = nullptr, *sq_array = nullptr, *cq_head = nullptr, *cq_tail = nullptr, *cq_mask = nullptr;
  io_uring_sqe* sqes = nullptr;
  io_uring_cqe* cqes = nullptr;
  void* maps[3] = {nullptr, nullptr, nullptr};
  size_t lens[3] = {0, 0, 0};

  void init(unsigned entries) {
    tried = true;
    io_uring_params p = {};
    fd = int(syscall(__NR_io_uring_setup, entries, &p));
    if (fd < 0)
      return;
    size_t sl = p.sq_off.array + p.sq_entries * sizeof(unsigned), cl = p.cq_off.cqes + p.cq_entries * sizeof(io_uring_cqe);
    bool one = p.features & IORING_FEAT_SINGLE_MMAP;
    if (one)
      sl = cl = std::max(sl, cl);
    auto map = [&](int k, size_t n, off_t what) {
      void* m = mmap(nullptr, n, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, what);
      if (m == MAP_FAILED)
        return (char*)nullptr;
      maps[k] = m;
      lens[k] = n;
      return (char*)m;
    };
    char* sq = map(0, sl, IORING_OFF_SQ_RING);
    char* cq = one ? sq : map(1, cl, IORING_OFF_CQ_RING);
    char* se = map(2, p.sq_entries * sizeof(io_uring_sqe), IORING_OFF_SQES);
    if (!sq || !cq || !se)
      return;
    sq_tail = (unsigned*)(sq + p.sq_off.tail);
    sq_mask = (unsigned*)(sq + p.sq_off.ring_mask);
    sq_array = (unsigned*)(sq + p.sq_off.array);
    cq_head = (unsigned*)(cq + p.cq_off.head);
    cq_tail = (unsigned*)(cq + p.cq_off.tail);
    cq_mask = (unsigned*)(cq + p.cq_off.ring_mask);
    cqes = (io_uring_cqe*)(cq + p.cq_off.cqes);
    sqes = (io_uring_sqe*)se;
    ok = true;
  }
  ~Uring() {
    for (int k = 0; k < 3; k++)
      if (maps[k])
        munmap(maps[k], lens[k]);
    if (fd >= 0)
      ::close(fd);
  }
  // Queue a read of n bytes at off into buf, tagged; only this thread touches the submission tail.
  void push(int f, char* buf, unsigned n, off_t off, uint64_t tag) {
    unsigned t = *sq_tail, i = t & *sq_mask;
    io_uring_sqe& e = sqes[i];
    memset(&e, 0, sizeof(e));
    e.opcode = IORING_OP_READ;
    e.fd = f;
    e.addr = uint64_t(uintptr_t(buf));
    e.len = n;
    e.off = uint64_t(off);
    e.user_data = tag;
    sq_array[i] = i;
    __atomic_store_n(sq_tail, t + 1, __ATOMIC_RELEASE);
  }
  // Submit `submit` queued reads and wait for at least `wait` completions, -1 on an error other than a signal.
  int enter(unsigned submit, unsigned wait) {
    for (;;) {
      long r = syscall(__NR_io_uring_enter, fd, submit, wait, wait ? IORING_ENTER_GETEVENTS : 0, nullptr, 0);
      if (r >= 0 || errno != EINTR)
        return int(r);
    }
  }
  bool pop(uint64_t& tag, int& res) {
    unsigned h = *cq_head;
    if (h == __atomic_load_n(cq_tail, __ATOMIC_ACQUIRE))
      return false;
    const io_uring_cqe& c = cqes[h & *cq_mask];
    tag = c.user_data;
    res = c.res;
    __atomic_store_n(cq_head, h + 1, __ATOMIC_RELEASE);
    return true;
  }
#else
  void init(unsigned) {
    tried = true;
  }
  void push(int, char*, unsigned, off_t, uint64_t) {}
  int enter(unsigned, unsigned) {
    return -1;
  }
  bool pop(uint64_t&, int&) {
    return false;
  }
#endif
};

static const int MAX_PIECES = 64, READY_WORDS = 1024;   // Pieces a record, and ready words a job (6144 up to the guess entries at 7168).

class Agent {
 public:
  Agent(int fd, int n_layers, int readers) : fd_(fd), pool_(std::max(0, readers - 1)), layers_(n_layers) {
    // Let through a job that no agent answered, which happens only after a release.
    seen_ = W(kPosted).load();
    if (W(kDone).load() < seen_)
      W(kDone).store(seen_, std::memory_order_release);
    if (W(kClaimed).load() < seen_)
      W(kClaimed).store(seen_, std::memory_order_release);
    last_ = now_ns();
    const char* hb = getenv("MOE_PIPE_BEAT_MS");
    beat_on_ = hb && atof(hb) > 0;
    W(kBeating).store(beat_on_, std::memory_order_release);
    // For tests, MOE_PIPE_SLOW_US holds every job that long before it is answered.
    if (const char* e = getenv("MOE_PIPE_SLOW_US"))
      slow_us_ = atoi(e);
    // Open the same file again with O_DIRECT, falling back to buffered reads on a file system
    // that lacks it, and read MOE_PIPE_CHUNK as the bytes the GPU copies at a time.
    const char* c = getenv("MOE_PIPE_CHUNK");
    chunk_ = c ? size_t(atol(c)) / 16 * 16 : 131072;
    if (!chunk_)
      chunk_ = ~size_t(0) / 2;
    // MOE_PIPE_PIECES=1 reads an O_DIRECT record in MOE_PIPE_PIECE-byte pieces (128 KB, the drive's largest IO) with a ready word each, so
    // the wait kernel copies a record while the rest of it is read.
    const char* pc = getenv("MOE_PIPE_PIECES");
    pieces_ = pc && !strcmp(pc, "1");
    if (const char* ps = getenv("MOE_PIPE_PIECE"))
      piece_ = round_up(std::max<size_t>(size_t(atol(ps)), PAGE), PAGE);
    const char* d = getenv("MOE_PIPE_DIRECT");
    if (!(d && !strcmp(d, "0"))) {
#ifdef __linux__
      std::string p = "/proc/self/fd/" + std::to_string(fd);
      dfd_ = open(p.c_str(), O_RDONLY | O_DIRECT);
#else
      dfd_ = dup(fd);   // No O_DIRECT off Linux (tests/test_pipe_staging_emu.py on macOS): the aligned-span reads on a second fd.
#endif
    }
    th_ = std::thread([this] { loop(); });
    if (beat_on_)
      beat_th_ = std::thread([this] { beat(); });
  }
  ~Agent() {
    close();
  }
  void close() {
    if (th_.joinable()) {
      {
        std::lock_guard lk(m_);
        stop_ = true;
      }
      arm_cv_.notify_all();
      th_.join();
      if (beat_th_.joinable())
        beat_th_.join();
      W(kBeating).store(0, std::memory_order_release);
    }
    pool_.close();
    {
      std::lock_guard lk(gm_);
      gstop_ = true;
    }
    gcv_.notify_all();
    for (auto& t : gth_)
      t.join();
    gth_.clear();
    if (stage_mem_) {
      cu.host_free(stage_mem_);
      stage_mem_ = staging_ = nullptr;
    }
    for (auto& c : tier_mem_)
      cu.host_free(c.host);
    tier_mem_.clear();
    if (dfd_ >= 0) {
      ::close(dfd_);
      dfd_ = -1;
    }
  }

  bool direct() const {
    return dfd_ >= 0;
  }

  // Set the victim tier as n slots of slot_bytes with one expert each, pinned and mapped in chunks of `per` slots (0: one chunk, or the
  // chunks it has), on an idle agent; called again it resizes, keeping the experts of the chunks it keeps, and returns the slots it holds,
  // fewer than n when the host runs out of pinned memory while it grows.
  int set_tier(size_t slot_bytes, int n, int per) {
    idle_only();
    bool first = !tier_slot_;
    if (n < 0 || slot_bytes == 0 || (tier_slot_ && slot_bytes != tier_slot_) || (first && n == 0))
      throw std::invalid_argument("tier: slots of one size");
    if (first) {
      tier_slot_ = slot_bytes;
      tier_of_.resize(layers_.size());
    }
    per = per > 0 ? per : tier_per_ > 0 ? tier_per_ : std::max(n, 1);
    if (per != tier_per_)
      while (!tier_mem_.empty())
        tier_drop();
    tier_per_ = per;
    // Chunk c holds slots [c * per, min((c + 1) * per, n)), so only the last chunks change: dropped, then made at their new count.
    auto count = [&](size_t c) { return std::min(per, n - int(c) * per); };
    size_t want = (n + per - 1) / per;
    while (!tier_mem_.empty() && (tier_mem_.size() > want || tier_mem_.back().n != count(tier_mem_.size() - 1)))
      tier_drop();
    int kept = tier_held();
    while (tier_mem_.size() < want) {
      int c = count(tier_mem_.size());
      void* h = nullptr;
      CUdeviceptr d = 0;
      if (cu.host_alloc(&h, slot_bytes * size_t(c), 0x01 | 0x02) != 0) {
        if (first)
          throw std::runtime_error("cuMemHostAlloc (tier) failed");     // The first tier fails as it always did.
        break;
      }
      check(cu.host_devptr(&d, h, 0), "cuMemHostGetDevicePointer (tier)");
      tier_mem_.push_back({static_cast<char*>(h), d, c});
    }
    int held = tier_held();
    tier_key_.resize(held, {-1, -1});
    tier_stamp_.resize(held, 0);
    // Slots past the chunks kept are gone from every list; the new ones are free.
    auto gone = [&](int t) { return t >= kept; };
    tier_free_.erase(std::remove_if(tier_free_.begin(), tier_free_.end(), gone), tier_free_.end());
    tier_freed_.erase(std::remove_if(tier_freed_.begin(), tier_freed_.end(), gone), tier_freed_.end());
    std::deque<std::pair<int, uint64_t>> live;
    for (auto& x : tier_fifo_)
      if (x.first < kept && tier_stamp_[x.first] == x.second)
        live.push_back(x);
    tier_fifo_.swap(live);
    for (int t = held - 1; t >= kept; t--)
      tier_free_.push_back(t);
    tier_n_ = held;
    return held;
  }
  int tier_held() const {
    return tier_mem_.empty() ? 0 : int(tier_mem_.size() - 1) * tier_per_ + tier_mem_.back().n;
  }
  // Free the last chunk, its experts out of the tier first.
  void tier_drop() {
    int t0 = int(tier_mem_.size() - 1) * tier_per_;
    for (int t = t0; t < t0 + tier_mem_.back().n; t++) {
      auto [l, e] = tier_key_[t];
      if (l >= 0)
        tier_of_[l][e] = -1;
      tier_key_[t] = {-1, -1};
      tier_stamp_[t] = 0;
    }
    cu.host_free(tier_mem_.back().host);
    tier_mem_.pop_back();
  }
  uint64_t tier_addr(int t) const {
    return tier_mem_[t / tier_per_].dev + uint64_t(t % tier_per_) * tier_slot_;
  }
  char* tier_host(int t) const {
    return tier_mem_[t / tier_per_].host + size_t(t % tier_per_) * tier_slot_;
  }

  // Resident mode (stream_fast's MOE_RESIDENT): expert e of layer l takes a free tier slot, put as an eviction is, and its host
  // address comes back for the caller to read the expert's records into; 0 when no slot is free or e is in the tier already.
  uint64_t tier_fill(int l, int n_exp, int e) {
    idle_only();
    if (!tier_slot_ || l < 0 || l >= int(tier_of_.size()) || e < 0 || e >= n_exp)
      throw std::invalid_argument("tier_fill: no tier, or no such expert");
    auto& m = tier_of_[l];
    if (m.empty())
      m.assign(n_exp, -1);      // set_layer keeps a table of this size
    if (m[e] >= 0 || tier_free_.empty())
      return 0;
    int t = tier_free_.back();
    tier_free_.pop_back();
    m[e] = t;
    tier_key_[t] = {l, e};
    tier_stamp_[t] = ++tier_clock_;
    tier_fifo_.push_back({t, tier_clock_});
    return reinterpret_cast<uint64_t>(tier_host(t));
  }
  // The host address of each expert's tier slot, 0 where the tier does not hold it, on an idle agent.
  void tier_find(int l, const int32_t* ids, int n, uint64_t* out) {
    idle_only();
    const std::vector<int32_t>* m = l >= 0 && l < int(tier_of_.size()) ? &tier_of_[l] : nullptr;
    for (int k = 0; k < n; k++) {
      int t = m && ids[k] >= 0 && ids[k] < int(m->size()) ? (*m)[ids[k]] : -1;
      out[k] = t < 0 ? 0 : reinterpret_cast<uint64_t>(tier_host(t));
    }
  }

  // Turn staging on with `threads` guess threads, so at most that many guess records are in flight, and guesses posted beside a layer's job
  // for the layer `dist` on, once, on an idle agent and before any layer is set, since set_layer sizes the guess room with the staging room.
  void set_staging(int threads, int dist) {
    idle_only();
    if (!gth_.empty() || threads <= 0 || dist < 1 || dist > 4 || pitch_)
      throw std::invalid_argument("staging: set once, with threads and a distance of 1-4, before any layer");
    gdist_ = dist;
    gs_.assign((dist + 1) * GUESS_RECORDS, Guess());      // D + 1 groups, one a batch by turns.
    // For tests, MOE_PIPE_STAGING_FAULT="n,us" fails every n-th guess read and slows each one by us microseconds.
    if (const char* f = getenv("MOE_PIPE_STAGING_FAULT"))
      sscanf(f, "%d,%d", &fault_every_, &fault_us_);
    for (int t = 0; t < threads; t++)
      gth_.emplace_back([this] { guess_loop(); });
  }

  // Wait until every committed job is answered and the thread is idle or gone, with error()
  // saying why, since the mutex is the happens-before edge for the tables the host reads next.
  bool drain(double timeout_s) {
    std::unique_lock lk(m_);
    return idle_cv_.wait_for(lk, std::chrono::duration<double>(timeout_s),
                             [&] { return dead_ || idle_locked(); });
  }

  // A pass is about to start, so spin for its jobs instead of sleeping.
  void arm() {
    {
      std::lock_guard lk(m_);
      armed_ = true;
      last_ = now_ns();
    }
    arm_cv_.notify_all();
  }

  // projs holds, per projection, its file offset, its stride and its parts, and slot_of is an
  // int32 table of n_exp entries that a store layer completes with set_ring.
  void set_layer(int i, int32_t* slot_of, int n_exp, int cap, std::vector<Layer::Proj> projs,
                 std::vector<int> quota, int ring) {
    idle_only();
    guess_drop();
    Layer L;
    L.n_exp = n_exp;
    L.slot_of = slot_of;
    L.cap = cap;
    size_t most = pitch_, at = 0;
    for (auto& p : projs) {
      size_t sum = 0;
      for (auto& pr : p.parts)
        sum += pr.second;
      if (sum != p.stride)
        throw std::invalid_argument("parts do not add up to the stride");
      // With O_DIRECT a staging record holds the record's aligned span.
      p.direct = dfd_ >= 0 && p.off % 16 == 0 && p.stride % 16 == 0;
      most = std::max(most, p.direct ? round_up(p.stride, PAGE) + PAGE : round_up(p.stride, 16));
      p.at = at;
      at += p.stride;
    }
    L.tier = tier_slot_ && at <= tier_slot_;
    if (L.tier && int(tier_of_.at(i).size()) != n_exp)
      tier_of_[i].assign(n_exp, -1);    // A layer's expert count never changes in one install.
    L.projs = std::move(projs);
    int n = L.cap;
    if (ring >= 0) {
      L.ring = ring;
      L.qslot = std::move(quota);
      L.qpos.assign(L.cap, -1);
      for (int p = 0; p < int(L.qslot.size()); p++)
        L.qpos.at(L.qslot[p]) = p;
      n = L.qslot.size();
    }
    L.owner.assign(n, -1);
    L.stamp.assign(n, 0);
    L.seen.assign(L.n_exp, 0);
    if (most > pitch_) {
      // Staging holds STAGE_RECORDS records of the largest room, then the guess room when staging is on, pinned and mapped
      // for the wait kernel, and each record starts page-aligned for O_DIRECT.
      if (stage_mem_)
        cu.host_free(stage_mem_);
      stage_mem_ = staging_ = nullptr;
      most = round_up(most, PAGE);
      void* h = nullptr;
      CUdeviceptr d = 0;
      check(cu.host_alloc(&h, (STAGE_RECORDS + gs_.size()) * most + PAGE, 0x01 | 0x02), "cuMemHostAlloc (staging)");
      stage_mem_ = static_cast<char*>(h);
      size_t lead = round_up(uintptr_t(h), PAGE) - uintptr_t(h);    // A small allocation need not start a page.
      staging_ = stage_mem_ + lead;
      check(cu.host_devptr(&d, h, 0), "cuMemHostGetDevicePointer (staging)");
      stage_dev_ = d + lead;
      pitch_ = most;
    }
    layers_.at(i) = std::move(L);
  }

  void set_ring(int r, int base, int next, std::vector<std::pair<int, int>> owner) {
    idle_only();
    if (owner.empty() || next < 0 || next >= int(owner.size()))
      throw std::invalid_argument("ring: no rows or bad next");
    if (r >= int(rings_.size()))
      rings_.resize(r + 1);
    rings_[r] = Ring{base, next, std::move(owner)};
  }

  std::pair<int, std::vector<std::pair<int, int>>> ring_state(int r) {
    idle_only();
    auto& R = rings_.at(r);
    return {R.next, R.owner};
  }

  std::pair<std::vector<int32_t>, std::vector<int32_t>> take_log(int i) {
    idle_only();
    auto& L = live(i);
    auto out = std::make_pair(std::move(L.log_ids), std::move(L.log_len));
    L.log_ids.clear();
    L.log_len.clear();
    return out;
  }

  void clear_layer(int i) {
    idle_only();
    guess_drop();
    layers_.at(i) = Layer();
  }

  // Set the table from slot and expert pairs, least recently used first, and write slot_of
  // too, while a store layer's ring members go into slot_of only.
  void set_state(int i, const std::vector<std::pair<int, int>>& lru,
                 const std::vector<std::pair<int, int>>& ring) {
    idle_only();
    auto& L = live(i);
    if (!ring.empty() && L.ring < 0)
      throw std::invalid_argument("ring members for an arena layer");
    std::fill(L.slot_of, L.slot_of + L.n_exp, -1);
    std::fill(L.owner.begin(), L.owner.end(), -1);
    std::fill(L.stamp.begin(), L.stamp.end(), 0);
    L.clock = 0;
    for (auto [s, e] : lru) {
      int p = L.pos_of(s);
      L.owner.at(p) = e;
      L.slot_of[e] = s;
      L.stamp[p] = ++L.clock;
    }
    for (auto [s, e] : ring)
      L.slot_of[e] = s;
  }

  std::vector<std::pair<int, int>> lru(int i) {
    idle_only();
    auto& L = live(i);
    std::vector<std::pair<int, int>> out;
    for (int p = 0; p < int(L.owner.size()); p++)
      if (L.owner[p] >= 0)
        out.push_back({p, L.owner[p]});
    std::sort(out.begin(), out.end(), [&](auto a, auto b) { return L.stamp[a.first] < L.stamp[b.first]; });
    for (auto& pe : out)
      pe.first = L.slot_at(pe.first);
    return out;
  }

  std::vector<uint32_t> take_seen(int i) {
    idle_only();
    auto& L = live(i);
    auto out = L.seen;
    std::fill(L.seen.begin(), L.seen.end(), 0);
    return out;
  }

  // Fill out with nine words: layers, hits, misses, miss layers, park ns and read ns since the
  // last call, where park is a job's claim and reads and read is its records landing, then
  // tier hits, experts put in the tier and records read buffered after O_DIRECT failed.
  void stats(uint64_t* out) {
    std::lock_guard lk(m_);
    uint64_t v[9] = {n_layers_, n_hits_, n_miss_, n_miss_layers_, park_ns_, read_ns_,
                     n_tier_hits_.exchange(0), n_tier_saves_.exchange(0), n_dio_fell_.exchange(0)};
    std::copy(v, v + 9, out);
    n_layers_ = n_hits_ = n_miss_ = n_miss_layers_ = park_ns_ = read_ns_ = 0;
  }

  // Fill out with eleven words since the last call, moe_native's stage_stats order: guessed experts queued, guess records read, guess records
  // a claim used, of those the ones still in flight at the claim, records read and not used, ns the jobs waited for promoted records, records
  // dropped unstarted, 0 (no rows to put back here), guess reads that failed, then the most guess reads in flight at once and the guess reads
  // started while a job's own reads were in flight (0 by design).
  void stage_stats(uint64_t* out) {
    std::lock_guard lk(gm_);
    uint64_t wrong = s_rec_ > s_prom_ ? s_rec_ - s_prom_ : 0;
    uint64_t v[11] = {s_guess_, s_rec_, s_prom_, s_part_, wrong, s_wait_ns_, s_drop_, 0, s_bad_, s_peak_, s_ahead_};
    std::copy(v, v + 11, out);
    s_guess_ = s_rec_ = s_prom_ = s_part_ = s_wait_ns_ = s_drop_ = s_bad_ = s_peak_ = s_ahead_ = 0;
  }

  std::string error() {
    std::lock_guard lk(m_);
    return error_;
  }

  // A miss records the slot it takes, the expert and what the slot held before, or -1.
  struct Miss {
    int s, e, vl, ve;
  };

  // Claim as moe_native's Agent::claim does: hits refresh their slot, and misses take the
  // lowest empty slot in order, else the least recently used one.
  static std::vector<Miss> claim(int li, Layer& L, std::vector<int> ids) {
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    std::vector<int> pend;
    for (int e : ids) {
      if (e < 0 || e >= L.n_exp)
        throw std::runtime_error("expert id out of range");
      L.seen[e]++;
      int s = L.slot_of[e];
      if (s >= 0)
        L.stamp[s] = ++L.clock;
      else
        pend.push_back(e);
    }
    if (int(ids.size()) > L.cap)
      throw std::runtime_error("more experts than arena slots");
    std::vector<Miss> miss;
    for (int e : pend) {
      int best = -1;
      for (int s = 0; s < L.cap; s++) {
        if (L.owner[s] < 0) {
          best = s;
          break;
        }
        if (best < 0 || L.stamp[s] < L.stamp[best])
          best = s;
      }
      int ve = L.owner[best];
      if (ve >= 0)
        L.slot_of[ve] = -1;
      L.owner[best] = e;
      L.slot_of[e] = best;
      L.stamp[best] = ++L.clock;
      miss.push_back({best, e, ve >= 0 ? li : -1, ve});
    }
    return miss;
  }

  // Claim as moe_native's Agent::claim_store does, which places one decode pass.
  std::vector<Miss> claim_store(int li, Layer& L, std::vector<int> ids) {
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    auto& R = rings_.at(L.ring);
    int nq = L.qslot.size(), n_ring = R.owner.size(), hits = 0;
    std::vector<int> pend, keep;
    for (int e : ids) {
      if (e < 0 || e >= L.n_exp)
        throw std::runtime_error("expert id out of range");
      L.seen[e]++;
      int s = L.slot_of[e];
      if (s < 0) {
        pend.push_back(e);
        continue;
      }
      hits++;
      int p = s < int(L.qpos.size()) ? L.qpos[s] : -1;
      if (p >= 0)
        L.stamp[p] = ++L.clock;
      else
        keep.push_back(s);
    }
    if (int(pend.size()) > nq + n_ring - hits)
      throw std::runtime_error("more experts than quota + ring half");
    int quota_room = nq - (hits - int(keep.size()));
    std::vector<Miss> miss;
    for (int n = 0; n < int(pend.size()); n++) {
      int e = pend[n], s = -1, vl = -1, ve = -1;
      if (n >= quota_room) {
        for (int t = 0; t < n_ring && s < 0; t++) {
          int c = R.base + R.next;
          R.next = (R.next + 1) % n_ring;
          if (std::find(keep.begin(), keep.end(), c) == keep.end())
            s = c;
        }
        if (s < 0)
          throw std::runtime_error("ring half full of this call's own hits");
        auto [ol, oe] = R.owner[s - R.base];
        if (ol >= 0 && layers_.at(ol).slot_of && layers_[ol].slot_of[oe] == s)
          layers_[ol].slot_of[oe] = -1;
        if (ol >= 0)
          vl = ol, ve = oe;       // A ring row holds what its last claim wrote there.
        R.owner[s - R.base] = {li, e};
      } else {
        int best = -1;
        for (int p = 0; p < nq; p++) {
          if (L.owner[p] < 0) {
            best = p;
            break;
          }
          if (best < 0 || L.stamp[p] < L.stamp[best])
            best = p;
        }
        if (L.owner[best] >= 0) {
          L.slot_of[L.owner[best]] = -1;
          vl = li, ve = L.owner[best];
        }
        L.owner[best] = e;
        L.stamp[best] = ++L.clock;
        s = L.qslot[best];
      }
      L.slot_of[e] = s;
      miss.push_back({s, e, vl, ve});
    }
    L.log_ids.insert(L.log_ids.end(), ids.begin(), ids.end());
    L.log_len.push_back(ids.size());
    return miss;
  }

 private:
  struct Rec {
    const Layer::Proj* p;
    int s, e;   // The slot and the expert.
    int tier;   // The tier slot it is copied from, or -1 when it is read from the file.
    int guess;  // The guess record it is copied from, or -1.
    bool now;   // Its bytes are in place at the claim, so its flag goes up with it.
    int pb = 0, np = 1;   // Its first ready word and how many: one, or MOE_PIPE_PIECES' pieces of ps bytes of its aligned span.
    size_t ps = 0;
  };

  // A guess record: what it holds or will hold, its batch, its state (0 none, 1 queued, 2 reading, 3 landed, 4 failed), whether a read is
  // writing its bytes whatever the batch, and the job and plan record a promoted read flags when it lands.
  struct Guess {
    int layer = -1, e = -1, p = -1;
    uint64_t batch = 0;
    int state = 0;
    bool busy = false;
    uint32_t prom_v = 0;
    int prom_i = -1;
  };

  // A batch: one posted guess, the layer it guesses, its records in read order and the next one to start.
  struct Batch {
    uint64_t id;
    int layer;
    std::vector<int> q;
    size_t next = 0;
  };

  bool idle_locked() const {
    return !busy_ && W(kDone).load() >= W(kSent).load() && W(kPosted).load() == seen_;
  }
  void idle_only() {
    std::lock_guard lk(m_);
    if (!idle_locked())
      throw std::runtime_error("agent is busy: drain() first");
  }
  Layer& live(int i) {
    auto& L = layers_.at(i);
    if (!L.slot_of)
      throw std::runtime_error("agent layer not set");
    return L;
  }

  // The tier maps layer and expert to a tier slot, a miss found there is copied from it while
  // its slot frees again from the next job on, an evicted expert gets a free slot or else the
  // oldest one but never one put in this same job, and -1 comes back when there is no room or
  // its bytes are there already.
  int tier_take(int l, int e) {
    auto& m = tier_of_.at(l);
    int t = m.empty() ? -1 : m[e];
    if (t < 0)
      return -1;
    m[e] = -1;
    tier_key_[t] = {-1, -1};
    tier_stamp_[t] = 0;
    tier_freed_.push_back(t);
    n_tier_hits_++;
    return t;
  }
  int tier_put(int l, int e, uint64_t first) {
    if (l < 0 || l >= int(tier_of_.size()) || tier_of_[l].empty() || tier_of_[l][e] >= 0)
      return -1;
    int t;
    if (!tier_free_.empty()) {
      t = tier_free_.back();
      tier_free_.pop_back();
    } else {
      while (!tier_fifo_.empty() && tier_stamp_[tier_fifo_.front().first] != tier_fifo_.front().second)
        tier_fifo_.pop_front();   // The slot was taken by a hit since.
      if (tier_fifo_.empty() || tier_fifo_.front().second >= first)
        return -1;
      t = tier_fifo_.front().first;
      tier_fifo_.pop_front();
      auto [ol, oe] = tier_key_[t];
      tier_of_[ol][oe] = -1;
    }
    tier_key_[t] = {l, e};
    tier_of_[l][e] = t;
    tier_stamp_[t] = ++tier_clock_;
    tier_fifo_.push_back({t, tier_clock_});
    if (tier_fifo_.size() > 2 * size_t(tier_n_) + 64) {
      // Hits leave stale entries behind, which this drops once in a while.
      std::deque<std::pair<int, uint64_t>> live;
      for (auto& x : tier_fifo_)
        if (tier_stamp_[x.first] == x.second)
          live.push_back(x);
      tier_fifo_.swap(live);
    }
    n_tier_saves_++;
    return t;
  }

  // Turn job v's misses into records, each miss per projection, and write the plan the wait
  // kernel copies them by, one copy for each chunk of a part with the row's old bytes first
  // into the evicted expert's tier slot.
  std::vector<Rec> plan(uint32_t v, int li, Layer& L, const std::vector<Miss>& miss) {
    tier_free_.insert(tier_free_.end(), tier_freed_.begin(), tier_freed_.end());   // The last job's copies have run by now.
    tier_freed_.clear();
    uint64_t first = tier_clock_ + 1;
    std::vector<Rec> recs;
    std::vector<int> save;
    int nready = 0;     // Ready words so far: one a record, or a piece each under MOE_PIPE_PIECES.
    std::unique_lock glk(gm_, std::defer_lock);
    if (!gs_.empty())
      glk.lock();
    for (auto& m : miss) {
      int th = L.tier ? tier_take(li, m.e) : -1;
      int tv = L.tier && m.vl >= 0 ? tier_put(m.vl, m.ve, first) : -1;
      job_tier_ += th >= 0;
      job_puts_ += tv >= 0;
      for (int p = 0; p < int(L.projs.size()); p++) {
        bool now = th >= 0;
        int g = th < 0 && glk ? guess_use(li, m.e, p, v, nready, now) : -1;
        Rec r{&L.projs[p], m.s, m.e, th, g, now};
        r.pb = nready;
        if (pieces_ && th < 0 && g < 0 && r.p->direct) {
          off_t at = r.p->off + off_t(m.e) * off_t(r.p->stride);
          size_t span = round_up(size_t(at) % PAGE + r.p->stride, PAGE);
          size_t np = std::min<size_t>((span + piece_ - 1) / piece_, MAX_PIECES);
          r.ps = round_up((span + np - 1) / np, PAGE);
          r.np = int((span + r.ps - 1) / r.ps);
        }
        nready += r.np;
        recs.push_back(r);
        save.push_back(tv);
      }
    }
    if (glk) {
      guess_end(li);
      for (auto& r : recs)
        gpause_ |= r.tier < 0 && r.guess < 0;   // No guess read starts until this job's own reads are in.
    }
    uint32_t* P = reinterpret_cast<uint32_t*>(g_blk) + kPlan;
    uint32_t n = 0;
    if (recs.size() > size_t(STAGE_RECORDS))
      throw std::runtime_error("more records than staging rows");
    if (nready > READY_WORDS)
      throw std::runtime_error("more pieces than ready words");
    for (size_t i = 0; i < recs.size(); i++) {
      auto& r = recs[i];
      off_t at = r.p->off + off_t(r.e) * off_t(r.p->stride);
      size_t room = r.guess >= 0 ? STAGE_RECORDS + r.guess : i;
      uint64_t src = r.tier >= 0 ? tier_addr(r.tier) + r.p->at
                                 : stage_dev_ + room * pitch_ + (r.p->direct ? size_t(at) % PAGE : 0);
      uint64_t sv = save[i] >= 0 ? tier_addr(save[i]) + r.p->at : 0;
      size_t x = r.np > 1 ? size_t(at) % PAGE : 0;   // Under pieces, where in the aligned span the part's chunk starts.
      for (auto [base, row] : r.p->parts) {
        // A part moves in chunks of one block each, since many blocks move faster, and a chunk never spans two pieces.
        for (size_t o = 0, c; o < row; o += c) {
          if (n == PLAN_COPIES)
            throw std::runtime_error("more copies than the plan holds");
          c = std::min(chunk_, row - o);
          size_t k = r.np > 1 ? (x + o) / r.ps : 0;
          if (r.np > 1)
            c = std::min(c, (k + 1) * r.ps - (x + o));
          uint64_t dst = base + uint64_t(r.s) * row + o, s = src + o, v = sv ? sv + o : 0;
          uint32_t* E = P + 4 + PLAN_WORDS * n++;
          E[0] = uint32_t(dst);
          E[1] = uint32_t(dst >> 32);
          E[2] = uint32_t(s);
          E[3] = uint32_t(s >> 32);
          E[4] = uint32_t(c);
          E[5] = uint32_t(r.pb + k);
          E[6] = uint32_t(v);
          E[7] = uint32_t(v >> 32);
        }
        src += row;
        x += row;
        if (sv)
          sv += row;
      }
    }
    P[1] = n;
    P[2] = nready;
    __atomic_store_n(&P[0], v, __ATOMIC_RELEASE); // The plan belongs to job v now.
    return recs;
  }

  // Read each record into its staging record, with its aligned span under O_DIRECT, and raise
  // its flag so a block of the wait kernel copies it to its slot, while a failed read raises
  // its flag too and the error fails the pass.
  std::string read_records(uint32_t v, const std::vector<Rec>& recs) {
    std::vector<size_t> io;
    for (size_t i = 0; i < recs.size(); i++)
      if (recs[i].tier < 0 && recs[i].guess < 0)
        io.push_back(i);
    std::atomic<int> bad{0};
    pool_.run(io.size(), [&](size_t k) {
      size_t i = io[k];
      auto& r = recs[i];
      if (r.np > 1 ? !read_pieces(r, staging_ + i * pitch_, v) : !read_one(*r.p, r.e, staging_ + i * pitch_))
        bad++;
      for (int q = 0; q < r.np; q++)
        W(kReady + r.pb + q).store(v, std::memory_order_release);
      if (beat_on_)
        W(kBeat).fetch_add(1, std::memory_order_relaxed);
    });
    return bad ? "short read" : "";
  }

  // MOE_PIPE_PIECES: record r's aligned span read through O_DIRECT in its pieces, all queued at once on this thread's io_uring, each piece's
  // ready word raised as it lands; a short piece reads the record again buffered, and without io_uring the pieces go one after another.
  bool read_pieces(const Rec& r, char* dst, uint32_t v) {
    static thread_local Uring u;
    if (!u.tried)
      u.init(MAX_PIECES);
    off_t at = r.p->off + off_t(r.e) * off_t(r.p->stride), a = at / off_t(PAGE) * off_t(PAGE);
    size_t need = size_t(at - a) + r.p->stride, span = round_up(need, PAGE);
    auto len = [&](int q) { return std::min(r.ps, span - q * r.ps); };
    auto want = [&](int q) { return std::min(r.ps, need - q * r.ps); };
    bool ok = true;
    if (u.ok) {
      for (int q = 0; q < r.np; q++)
        u.push(dfd_, dst + q * r.ps, unsigned(len(q)), a + off_t(q * r.ps), uint64_t(q));
      int left = u.enter(r.np, 1);
      if (left < r.np) {
        // What the kernel did not take stays in a ring this thread never submits from again, and the record is read as one span below.
        u.ok = ok = false;
        left = std::max(left, 0);
      }
      while (left > 0) {
        uint64_t q;
        int res;
        while (u.pop(q, res)) {
          left--;
          if (res >= 0 && size_t(res) >= want(int(q)))
            W(kReady + r.pb + int(q)).store(v, std::memory_order_release);
          else
            ok = false;
        }
        if (left > 0 && u.enter(0, 1) < 0)
          sched_yield();     // The reads are queued and land anyway, so poll.
      }
    } else {
      for (int q = 0; q < r.np && ok; q++) {
        ok = pread_direct(dfd_, dst + q * r.ps, len(q), a + off_t(q * r.ps), want(q));
        if (ok)
          W(kReady + r.pb + q).store(v, std::memory_order_release);
      }
    }
    if (ok)
      return true;
    n_dio_fell_++;
    return pread_full(fd_, dst + (at - a), r.p->stride, at);
  }

  // Read expert e's record of projection p into the staging record at dst, its aligned span under O_DIRECT, and return false on a short read.
  bool read_one(const Layer::Proj& p, int e, char* dst) {
    off_t at = p.off + off_t(e) * off_t(p.stride);
    bool ok = false;
    if (p.direct) {
      off_t a = at / off_t(PAGE) * off_t(PAGE);
      size_t need = size_t(at - a) + p.stride;
      ok = pread_direct(dfd_, dst, round_up(need, PAGE), a, need);
      n_dio_fell_ += !ok;     // The buffered read below fills the same place.
      dst += at - a;
    }
    return ok || pread_full(fd_, dst, p.stride, at);
  }

  // Drop the guesses and wait for the guess reads in flight, before a layer's tables or the staging memory change.
  void guess_drop() {
    std::unique_lock lk(gm_);
    gbatch_++;     // No record of an earlier batch matches from here on.
    gb_.clear();
    gcv_.wait(lk, [&] { return ginflight_ == 0; });
  }

  // Under gm_: batch b's records not started are dropped, while those in flight finish and land.
  void guess_kill(const Batch& b) {
    for (size_t k = b.next; k < b.q.size(); k++)
      if (gs_[b.q[k]].state == 1) {
        gs_[b.q[k]].state = 0;
        s_drop_++;
      }
  }

  // Queue the guess G posted beside a job of layer li, its experts that layer li + D and the tier do not hold, in the group of the guess room
  // that batch b - D - 1 used, whose copies are done once the job after its target's has been posted, which comes before this guess.
  void guess_take(int li, const uint32_t* G) {
    int t = li + gdist_;
    auto& T = layers_[t];
    int n = std::min<int>(G[1], GUESS_WORDS - 2), np = T.projs.size();
    std::lock_guard lk(gm_);
    Batch B{++gbatch_, t};
    int h = B.id % (gdist_ + 1), r = 0;
    for (int k = 0; k < n && r + np <= GUESS_RECORDS; k++) {
      int e = int(G[2 + k]);
      if (e < 0 || e >= T.n_exp || T.slot_of[e] >= 0 || (T.tier && tier_of_[t][e] >= 0))
        continue;
      s_guess_++;
      for (int p = 0; p < np; p++, r++) {
        int j = h * GUESS_RECORDS + r;
        auto& g = gs_[j];
        g.layer = t;
        g.e = e;
        g.p = p;
        g.batch = B.id;
        g.state = 1;
        g.prom_v = 0;
        g.prom_i = -1;
        B.q.push_back(j);
      }
    }
    gb_.push_back(std::move(B));
    while (int(gb_.size()) > gdist_) {     // A batch whose target was never claimed (a pass that ended early).
      guess_kill(gb_.front());
      gb_.pop_front();
    }
    gcv_.notify_all();
  }

  // Under gm_, at job v's claim of layer li: the guess record of a live batch for li that holds expert e's projection p, landed (now) or in
  // flight (promoted, so its read flags ready word i), and -1 when there is none or it was not started, which drops it.
  int guess_use(int li, int e, int p, uint32_t v, int i, bool& now) {
    for (auto& b : gb_) {
      if (b.layer != li)
        continue;
      for (int j : b.q) {
        auto& g = gs_[j];
        if (g.e != e || g.p != p || g.batch != b.id)
          continue;
        if (g.state == 3) {
          now = true;
        } else if (g.state == 2) {
          g.prom_v = v;
          g.prom_i = i;
          s_part_++;
        } else {
          return -1;
        }
        s_prom_++;
        return j;
      }
    }
    return -1;
  }

  // Under gm_, after a claim of layer li: the batches for li and earlier end, their records not started dropped.
  void guess_end(int li) {
    while (!gb_.empty() && gb_.front().layer <= li) {
      guess_kill(gb_.front());
      gb_.pop_front();
    }
  }

  // Under gm_: the oldest live batch with a record to start, or null while a job's own reads run or that record's bytes are still being
  // written by an older read.
  Batch* guess_next() {
    if (gpause_)
      return nullptr;
    for (auto& b : gb_)
      if (b.next < b.q.size())
        return gs_[b.q[b.next]].busy ? nullptr : &b;
    return nullptr;
  }

  // A guess thread: take the next queued record whose bytes no older read still writes, read it, and flag the plan record it was promoted to.
  void guess_loop() {
    std::unique_lock lk(gm_);
    for (;;) {
      gcv_.wait(lk, [&] { return gstop_ || guess_next(); });
      if (gstop_)
        return;
      Batch* nb = guess_next();
      int j = nb->q[nb->next++];
      auto& g = gs_[j];
      g.state = 2;
      g.busy = true;
      uint64_t b = g.batch;
      const Layer::Proj* p = &layers_[g.layer].projs[g.p];   // Layers change only after guess_drop, which waits for this read.
      int e = g.e;
      s_peak_ = std::max<uint64_t>(s_peak_, ++ginflight_);
      s_ahead_ += demand_.load(std::memory_order_relaxed);
      bool fault = fault_every_ > 0 && ++fault_n_ % fault_every_ == 0;
      lk.unlock();
      if (fault_us_)
        std::this_thread::sleep_for(std::chrono::microseconds(fault_us_));
      bool ok = read_one(*p, e, staging_ + (STAGE_RECORDS + j) * pitch_) && !fault;
      lk.lock();
      ginflight_--;
      g.busy = false;
      s_rec_++;
      s_bad_ += !ok;
      if (g.batch == b && g.state == 2) {
        g.state = ok ? 3 : 4;
        if (ok && g.prom_v)
          W(kReady + g.prom_i).store(g.prom_v, std::memory_order_release);
      }
      gcv_.notify_all();
    }
  }

  // After job v's own reads: wait for its promoted guess records and read again on demand any whose guess read failed, flag up either way.
  std::string guess_settle(uint32_t v) {
    std::vector<int> failed;
    {
      std::unique_lock lk(gm_);
      auto open = [&] {
        for (auto& g : gs_)
          if (g.prom_v == v && g.state == 2)
            return true;
        return false;
      };
      if (open()) {
        uint64_t t = now_ns();
        gcv_.wait(lk, [&] { return !open(); });
        s_wait_ns_ += now_ns() - t;
      }
      for (int j = 0; j < int(gs_.size()); j++)
        if (gs_[j].prom_v == v && gs_[j].state == 4)
          failed.push_back(j);
    }
    std::string err;
    for (int j : failed) {
      auto& g = gs_[j];   // A failed record of a claimed batch, which no guess thread touches again until two batches on.
      if (!read_one(layers_[g.layer].projs[g.p], g.e, staging_ + (STAGE_RECORDS + j) * pitch_))
        err = "short read";
      W(kReady + g.prom_i).store(v, std::memory_order_release);
    }
    return err;
  }

  // Wait up to 2 ms for the guess the GPU posts beside job v and queue it, reading the stop words first since the guess lands before the job
  // after v is posted.
  void guess_wait(uint32_t v, int li) {
    const uint32_t* G = reinterpret_cast<uint32_t*>(g_blk) + kGuess + GUESS_WORDS * (v % RING);
    for (uint64_t t0 = now_ns();;) {
      bool last = W(kPosted).load() != v || W(kRelease).load() || W(kGave).load() || now_ns() - t0 > 2000000ull;
      if (__atomic_load_n(&G[0], __ATOMIC_ACQUIRE) == v)
        break;
      if (last)
        return;
    }
    guess_take(li, G);
  }

  // While a wait may be spinning, bump the word every 0.25 ms and otherwise look again every
  // 10 ms or at arm(), with word 13 keeping the longest gap between two bumps in us and word
  // 14 counting the gaps over 1 ms.
  void beat() {
    std::unique_lock lk(m_);
    uint64_t prev = 0;
    while (!stop_) {
      if (now_ns() - last_ < 2000000000ull || W(kSent).load() > W(kDone).load()) {
        lk.unlock();
        W(kBeat).fetch_add(1, std::memory_order_relaxed);
        uint64_t t = now_ns();
        if (prev) {
          uint32_t gap = (t - prev) / 1000;
          if (gap > W(13).load())
            W(13).store(gap);
          if (gap > 1000)
            W(14).fetch_add(1);
        }
        prev = t;
        std::this_thread::sleep_for(std::chrono::microseconds(250));
        lk.lock();
      } else {
        prev = 0;
        arm_cv_.wait_for(lk, std::chrono::milliseconds(10));
      }
    }
  }

  void loop() {
    try {
      serve();
    } catch (std::exception& e) {
      die(e.what());
    } catch (...) {
      die("unknown exception");
    }
  }

  // The thread is going with jobs it may not have answered, so let every wait through.
  void die(const std::string& why) {
    release_all();
    std::lock_guard lk(m_);
    if (error_.empty())
      error_ = "agent thread gone: " + why;
    dead_ = true;
    busy_ = false;
    idle_cv_.notify_all();
  }

  // Spin while jobs come, then look every 50 us and sleep until arm() or 10 ms pass, returning
  // false when stopped with nothing posted.
  bool next_job(uint32_t& v) {
    for (uint32_t spins = 0;; spins++) {
      if (W(kPosted).load(std::memory_order_acquire) != seen_) {
        v = seen_ + 1;
        return true;
      }
      uint64_t idle = now_ns() - last_.load(std::memory_order_relaxed);
      if (idle < 5000000ull) {
        if (beat_on_ && (spins & 255) == 0)
          W(kBeat).fetch_add(1, std::memory_order_relaxed); // A second heartbeat while it spins.
        continue;
      }
      std::unique_lock lk(m_);
      if (stop_)
        return false;
      if (armed_) {
        armed_ = false;
        last_ = now_ns();
        continue;
      }
      if (idle < 200000000ull) {
        lk.unlock();
        std::this_thread::sleep_for(std::chrono::microseconds(50));
      } else {
        arm_cv_.wait_for(lk, std::chrono::milliseconds(10), [&] { return stop_ || armed_; });
      }
    }
  }

  void serve() {
    for (;;) {
      uint32_t v;
      if (!next_job(v)) {
        if (W(kDone).load() < W(kSent).load()) {
          die("closed before the GPU reached a job");
          return;
        }
        return;
      }
      {
        std::lock_guard lk(m_);
        busy_ = true;
      }
      W(kNoticed).store(v, std::memory_order_release);
      uint64_t t0 = now_ns(), t_claim = 0, t_read = 0, read_ns = 0;
      int hits = 0, misses = 0, layer = -1;
      std::string err;
      std::vector<Rec> recs;
      uint32_t* e = reinterpret_cast<uint32_t*>(g_blk) + kRing + ENTRY * (v % RING);
      uint32_t posted = W(kPosted).load(std::memory_order_acquire);
      bool stopped = W(kRelease).load() || W(kGave).load();
      if (posted - seen_ > uint32_t(RING) || __atomic_load_n(&e[0], __ATOMIC_ACQUIRE) != v) {
        // The GPU ran ahead, so answer every posted job since the pass fails anyway.
        if (!stopped)
          err = "job ring overrun";
        v = posted;
      } else if (!stopped) {
        {
          std::lock_guard lk(m_);
          err = error_;
        }
        if (err.empty()) {
          try {
            int li = e[1], n = e[2];
            if (li < 0 || li >= int(layers_.size()) || n < 0 || n > MAX_IDS)
              throw std::runtime_error("agent job: bad layer or ids");
            auto& L = live(li);
            std::vector<int> ids(e + 3, e + 3 + n);
            auto miss = L.ring >= 0 ? claim_store(li, L, std::move(ids)) : claim(li, L, std::move(ids));
            misses = miss.size();
            hits = n - misses;
            recs = plan(v, li, L, miss);
            layer = li;
          } catch (std::exception& ex) {
            err = ex.what();
            recs.clear();
          }
        } else {
          err.clear(); // The error is kept already, and later jobs only signal.
        }
      }
      if (slow_us_)
        std::this_thread::sleep_for(std::chrono::microseconds(slow_us_));
      // The claim and the plan are in place or this job has no plan of its own, and the wait
      // kernel goes on either way so the GPU never spins into its bound.
      for (size_t i = 0; i < recs.size(); i++)
        if (recs[i].now)
          W(kReady + recs[i].pb).store(v, std::memory_order_release);   // A record in the tier or a landed guess is copied as soon as the GPU sees it.
      W(kClaimed).store(v, std::memory_order_release);
      t_claim = now_ns();
      if (!recs.empty()) {
        demand_.store(true, std::memory_order_relaxed);
        auto r = read_records(v, recs);
        demand_.store(false, std::memory_order_relaxed);
        if (!gs_.empty()) {
          auto g = guess_settle(v);    // Every promoted record's flag goes up, an error included, as a demand record's does.
          if (r.empty())
            r = g;
        }
        if (!r.empty())
          err = r;
        t_read = now_ns();
        read_ns = t_read - t_claim;
      }
      if (!gs_.empty()) {
        std::lock_guard glk(gm_);
        gpause_ = false;
        gcv_.notify_all();
      }
      W(kMiss + v % 128).store(std::min(misses, 255) | std::min(job_tier_, 255) << 8 | std::min(job_puts_, 255) << 16,
                               std::memory_order_relaxed);
      job_tier_ = job_puts_ = 0;
      uint64_t* lg = reinterpret_cast<uint64_t*>(g_blk + kAgentLog) + 5 * (v % 128);
      uint64_t t_end = now_ns();
      lg[1] = t0;
      lg[2] = t_claim;
      lg[3] = t_read;
      lg[4] = t_end;
      lg[0] = v;
      if (!gs_.empty() && layer >= 0 && err.empty() && layer + gdist_ < int(layers_.size()) && layers_[layer + gdist_].slot_of)
        guess_wait(v, layer);
      W(kDone).store(v, std::memory_order_release);
      std::lock_guard lk(m_);
      last_ = now_ns();
      seen_ = v;
      if (!err.empty() && error_.empty())
        error_ = err;
      n_layers_++;
      n_hits_ += hits;
      n_miss_ += misses;
      n_miss_layers_ += misses > 0;
      park_ns_ += t_end - t0;
      read_ns_ += read_ns;
      busy_ = false;
      idle_cv_.notify_all();
    }
  }

  int fd_;
  Pool pool_;
  std::vector<Layer> layers_;
  std::vector<Ring> rings_;
  char* stage_mem_ = nullptr; // The staging memory as allocated, with staging_ at its first page boundary.
  char* staging_ = nullptr;
  uint64_t stage_dev_ = 0;
  size_t pitch_ = 0;          // A staging record's room, as a multiple of a page.
  size_t chunk_ = 0;          // The most bytes one copy moves, from MOE_PIPE_CHUNK.
  bool pieces_ = false;       // MOE_PIPE_PIECES: records read in pieces with a ready word each.
  size_t piece_ = 131072;     // A piece's bytes, from MOE_PIPE_PIECE, a multiple of a page.
  int job_tier_ = 0, job_puts_ = 0;   // This job's tier hits and puts, for the miss word.
  int dfd_ = -1;              // The O_DIRECT fd on the same file, or -1 when it is not used.
  // The victim tier of set_tier in chunks of tier_per_ slots (the last may hold fewer), with tier_of_ mapping experts to slots.
  struct TierChunk {
    char* host;
    uint64_t dev;
    int n;
  };
  std::vector<TierChunk> tier_mem_;
  uint64_t tier_clock_ = 0;
  size_t tier_slot_ = 0;
  int tier_n_ = 0, tier_per_ = 0;
  std::vector<std::vector<int32_t>> tier_of_;
  std::vector<std::pair<int, int>> tier_key_;
  std::vector<uint64_t> tier_stamp_;              // When each slot was put, or 0 when it is free or taken.
  std::deque<std::pair<int, uint64_t>> tier_fifo_; // Slot and stamp pairs, oldest first, with stale entries skipped.
  std::vector<int> tier_free_, tier_freed_;       // The free slots, and the slots this job's hits took.
  std::atomic<uint64_t> n_tier_hits_{0}, n_tier_saves_{0}, n_dio_fell_{0};
  // Staging of set_staging: the guess records, the live batches oldest first, the batch count, the guess distance D and the pause while a
  // job's own records are read, all under gm_, and the guess threads.
  std::vector<Guess> gs_;
  std::deque<Batch> gb_;
  uint64_t gbatch_ = 0;
  int gdist_ = 1, ginflight_ = 0;
  bool gpause_ = false;
  std::mutex gm_;
  std::condition_variable gcv_;
  std::vector<std::thread> gth_;
  bool gstop_ = false;
  std::atomic<bool> demand_{false};   // True while a job's own records are read.
  uint64_t s_guess_ = 0, s_rec_ = 0, s_prom_ = 0, s_part_ = 0, s_wait_ns_ = 0, s_drop_ = 0, s_bad_ = 0, s_peak_ = 0, s_ahead_ = 0;
  int fault_every_ = 0, fault_us_ = 0, fault_n_ = 0;   // The MOE_PIPE_STAGING_FAULT state, for tests.
  int slow_us_ = 0;
  uint32_t seen_ = 0;        // The last job answered, written by the agent thread under m_.
  std::atomic<uint64_t> last_{0}; // When the last job was answered or the agent armed.
  std::thread th_, beat_th_;
  bool beat_on_ = false;      // True when MOE_PIPE_BEAT_MS asks for the heartbeat.
  std::mutex m_;
  std::condition_variable idle_cv_, arm_cv_;
  bool stop_ = false, busy_ = false, dead_ = false, armed_ = false;
  std::string error_;
  uint64_t n_layers_ = 0, n_hits_ = 0, n_miss_ = 0, n_miss_layers_ = 0, park_ns_ = 0, read_ns_ = 0;
};

// ---------------------------------------------------------------------------
// The C ABI for moe_pipe_cuda.py, whose failures return -1 with mpc_error() set.

template <class F>
static int guarded(F f) {
  try {
    f();
    return 0;
  } catch (std::exception& e) {
    g_err = e.what();
    return -1;
  }
}

extern "C" {

const char* mpc_error() {
  return g_err.c_str();
}
uint64_t mpc_now_ns() {
  return now_ns();
}
int mpc_init() {
  return guarded([&] { init(); });
}
void* mpc_block() {
  return g_blk;
}
uint64_t mpc_block_dev() {
  return g_blk_dev;
}
void mpc_release() {
  release_all();
}
int mpc_pending() {
  return pending();
}

void* mpc_agent_new(int fd, int n_layers, int readers) {
  try {
    if (!g_blk)
      init();
    return new Agent(fd, n_layers, readers);
  } catch (std::exception& e) {
    g_err = e.what();
    return nullptr;
  }
}
void mpc_agent_close(void* a) {
  static_cast<Agent*>(a)->close();
}
int mpc_agent_drain(void* a, double timeout_s) {
  return static_cast<Agent*>(a)->drain(timeout_s);
}
void mpc_agent_arm(void* a) {
  static_cast<Agent*>(a)->arm();
}
int mpc_agent_error(void* a, char* out, int n) {
  auto s = static_cast<Agent*>(a)->error();
  snprintf(out, n, "%s", s.c_str());
  return s.size();
}
void mpc_agent_stats(void* a, uint64_t* out) {
  static_cast<Agent*>(a)->stats(out);
}
int mpc_agent_direct(void* a) {
  return static_cast<Agent*>(a)->direct();
}
// The slots the tier holds after the call, or -1 on an error.
int mpc_agent_tier(void* a, uint64_t slot_bytes, int n, int per) {
  int held = -1;
  if (guarded([&] { held = static_cast<Agent*>(a)->set_tier(slot_bytes, n, per); }) != 0)
    return -1;
  return held;
}
// The tier slot's host address in *host (0: no free slot, or held already).
int mpc_tier_fill(void* a, int l, int n_exp, int e, uint64_t* host) {
  return guarded([&] { *host = static_cast<Agent*>(a)->tier_fill(l, n_exp, e); });
}
int mpc_tier_find(void* a, int l, const int32_t* ids, int n, uint64_t* out) {
  return guarded([&] { static_cast<Agent*>(a)->tier_find(l, ids, n, out); });
}
int mpc_agent_staging(void* a, int threads, int dist) {
  return guarded([&] { static_cast<Agent*>(a)->set_staging(threads, dist); });
}
void mpc_agent_stage_stats(void* a, uint64_t* out) {
  static_cast<Agent*>(a)->stage_stats(out);
}

// part_addr and part_row hold every projection's parts in order.
int mpc_set_layer(void* a, int i, int32_t* slot_of, int n_exp, int cap, int nproj, const int64_t* off,
                  const int64_t* stride, const int* nparts, const uint64_t* part_addr, const uint64_t* part_row,
                  const int* quota, int nq, int ring) {
  return guarded([&] {
    std::vector<Layer::Proj> projs;
    for (int j = 0, k = 0; j < nproj; j++) {
      Layer::Proj p{off_t(off[j]), size_t(stride[j]), {}};
      for (int q = 0; q < nparts[j]; q++, k++)
        p.parts.push_back({part_addr[k], size_t(part_row[k])});
      projs.push_back(std::move(p));
    }
    static_cast<Agent*>(a)->set_layer(i, slot_of, n_exp, cap, std::move(projs),
                                      std::vector<int>(quota, quota + nq), ring);
  });
}
int mpc_clear_layer(void* a, int i) {
  return guarded([&] { static_cast<Agent*>(a)->clear_layer(i); });
}
// lru and ring are flattened slot and expert pairs.
int mpc_set_state(void* a, int i, const int* lru, int nl, const int* ring, int nr) {
  return guarded([&] {
    std::vector<std::pair<int, int>> l, r;
    for (int k = 0; k < nl; k++)
      l.push_back({lru[2 * k], lru[2 * k + 1]});
    for (int k = 0; k < nr; k++)
      r.push_back({ring[2 * k], ring[2 * k + 1]});
    static_cast<Agent*>(a)->set_state(i, l, r);
  });
}
// out takes the pairs with room for cap, and *n gets how many there are.
int mpc_lru(void* a, int i, int* out, int cap, int* n) {
  return guarded([&] {
    auto v = static_cast<Agent*>(a)->lru(i);
    if (int(v.size()) > cap)
      throw std::runtime_error("lru: no room");
    for (size_t k = 0; k < v.size(); k++) {
      out[2 * k] = v[k].first;
      out[2 * k + 1] = v[k].second;
    }
    *n = v.size();
  });
}
int mpc_take_seen(void* a, int i, uint32_t* out, int n_exp) {
  return guarded([&] {
    auto v = static_cast<Agent*>(a)->take_seen(i);
    if (int(v.size()) != n_exp)
      throw std::runtime_error("take_seen: wrong size");
    std::copy(v.begin(), v.end(), out);
  });
}
// Call this twice, first with ids null for the sizes and then for the arrays.
static std::pair<std::vector<int32_t>, std::vector<int32_t>> g_log;
int mpc_take_log(void* a, int i, int32_t* ids, int32_t* lens, int* nids, int* nlens) {
  return guarded([&] {
    if (!ids) {
      g_log = static_cast<Agent*>(a)->take_log(i);
      *nids = g_log.first.size();
      *nlens = g_log.second.size();
    } else {
      std::copy(g_log.first.begin(), g_log.first.end(), ids);
      std::copy(g_log.second.begin(), g_log.second.end(), lens);
    }
  });
}
// owner is flattened layer and expert pairs over n rows.
int mpc_set_ring(void* a, int r, int base, int next, const int* owner, int n) {
  return guarded([&] {
    std::vector<std::pair<int, int>> o;
    for (int k = 0; k < n; k++)
      o.push_back({owner[2 * k], owner[2 * k + 1]});
    static_cast<Agent*>(a)->set_ring(r, base, next, o);
  });
}
int mpc_ring_state(void* a, int r, int* next, int* owner, int cap, int* n) {
  return guarded([&] {
    auto [nx, o] = static_cast<Agent*>(a)->ring_state(r);
    if (int(o.size()) > cap)
      throw std::runtime_error("ring_state: no room");
    *next = nx;
    for (size_t k = 0; k < o.size(); k++) {
      owner[2 * k] = o[k].first;
      owner[2 * k + 1] = o[k].second;
    }
    *n = o.size();
  });
}

}  // This closes the C ABI block.
