// Native expert agent for the single-token decode path of stream_fast.py when MOE_NATIVE=1.
//
// The GPU hands each MoE layer's router ids to a host thread through one shared 32-bit word
// that MLX's fence_update writes and our wait spins on, so no Python and no GIL work sits
// between a layer's router and its gather.
//
// The thread claims arena slots with the engine's LRU, reads the misses into the arena and
// raises the word, and the gather parked behind a wait on it then runs.
//
// Python API (module moe_native): SharedEvent() is the word with .value and .signal(v), and
// signal_after(x, ev, v, agent, layer) is a graph op that makes a layer's int32 ids visible to
// the host, sets the word to v and hands the job to the agent, or to nobody when the agent is
// null.
//
// wait_for(x, dep, ev, v) is a graph op that copies the 32-bit slot table once the word
// reaches v, and dep orders it after the matching signal_after.
//
// release() lets every GPU wait of this process fall through now and later, stopped() returns
// the released flag and the value a wait gave up on, pending() is true while the last wait
// encoded has not passed, and wait_bound() returns the spin rounds of one wait, its rounds per
// second and its seconds.
//
// Agent(ev, fd, layers, readers) is one host thread that waits for a value, reads the ids from
// unified memory, claims slots as stream_fast's _claim does on a flat table, reads the misses
// on a pthread pool and signals the next value.
//
// Every job is answered, an error included, so the GPU never spins into the watchdog, the
// error is kept for Python in .error, and a thread that goes with a job unanswered releases
// every wait first.
//
// A layer is either a per-layer arena with its own tensors and slot equal to row, or, under
// --mem-peak-gb, a quota of a shared store's rows plus a ring half other layers share, placed
// as stream_fast's _place_store does.
//
// Staging rows (MOE_STAGING=1): a job may carry, after its layer's ids, a guess of the next
// layer's ids, and once this layer's misses are in and the GPU is let go, the guessed experts
// the next layer does not hold are read into staging rows, which are physical rows of the
// store past its logical ones, by two threads per drive so at most two guess records are in
// flight on a drive and none queues ahead of a real read.
//
// At the next claim the records not started are dropped, and a miss that was guessed keeps the
// logical row the claim gave it with its bytes now at the staging row it was read into, so the
// row it replaced becomes a staging row and the swap costs no copy.
//
// slot_of, the claims and the bias mask stay the engine's, only the GPU's table (phys_of) and
// the reads change, and unstage() puts every row back at its logical place before Python takes
// the tables.
//
// MOE_CPU_GUESS: a job may carry its layer's MoE input instead of the guess, and a helper thread
// runs the next layer's 8-bit router on it (set_router, cpu_guess.h) while this one claims and
// reads the layer, picking the guess as the GPU's route kernel does with the next layer's quota as
// the bias mask, so the guess is staged once the GPU is let go.
//
// A wait nobody answers must not outlive the process, because an unbounded wait kernel keeps
// the GPU busy with its memory wired after the process is gone.
//
// So the waits are ours (bounded_wait): each gives up after MOE_NATIVE_WAIT_S seconds (default
// 5) of spinning, and a stop word every wait reads lets them all through at once, raised by
// release() (SIGTERM, SIGHUP, stream_fast's exit hooks) or by the first wait that gives up.
//
// Either of those sticks and stream_fast fails the native path on it, while a kill that cannot
// be caught leaves only the bound.
//
// MLX internals used, all exported from libmlx: metal::get_command_encoder, set_buffer,
// set_bytes, barrier, register_output_array and dispatch, the fast-fence kernels fence_update
// and input_coherent in mlx.metallib, metal::device(gpu).mtl_device(), allocator::malloc, and
// array::copy_shared_buffer, set_data and data.
//
// Our own kernels are compiled at SharedEvent() with `#pragma METAL internals` and
// coherent(system) loads, which the wait and the copy need, and eval_gpu runs on the encoding
// thread in tape order.
//
// fence_update stores its value unconditionally, so the host only ever raises the word and
// never signals a job before the GPU has reached it.
//
// The primitives hold raw pointers to the SharedEvent and the Agent, so Python keeps both
// alive for the life of the model.

#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <sys/param.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/qos.h>
#include <sys/uio.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "mlx/allocator.h"
#include "mlx/array.h"
#include "mlx/backend/metal/device.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

#include "cpu_guess.h"

namespace nb = nanobind;
namespace mx = mlx::core;
using namespace nb::literals;

static uint64_t now_ns() {
  return clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
}

// ---------------------------------------------------------------------------
// GPU waits that end, with our kernels, the stop words and the bound.

static MTL::Device* gpu() {
  return mx::metal::device(mx::Device::gpu).mtl_device();
}

static const char* kKernelSrc = R"(
#pragma METAL internals : enable
#include <metal_stdlib>
using namespace metal;
#ifndef __METAL_MEMORY_SCOPE_SYSTEM__
#define __METAL_MEMORY_SCOPE_SYSTEM__ 3
#endif
constexpr constant thread_scope scope_system =
    static_cast<thread_scope>(__METAL_MEMORY_SCOPE_SYSTEM__);

// MLX's fence_wait (kernels/fence.metal, v0.32.2) with a way out: give up
// after `cap` rounds (stop[1] = v: this and every later wait falls through,
// and the host sees it), or at once once the host released (stop[0]).
[[kernel]] void bounded_wait(volatile coherent(system) device uint* ev [[buffer(0)]],
                             volatile coherent(system) device uint* stop [[buffer(1)]],
                             constant uint& v [[buffer(2)]],
                             constant uint& cap [[buffer(3)]]) {
  for (uint i = 0;; i++) {
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, scope_system);
    if (ev[0] >= v || stop[0] != 0 || stop[1] != 0)
      return;
    if (i == cap) {
      stop[1] = v;
      atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, scope_system);
      return;
    }
  }
}

// A copy whose loads go through system coherence, as the wait's do (wait_for).
[[kernel]] void coherent_copy(volatile coherent(system) device uint* src [[buffer(0)]],
                              device uint* dst [[buffer(1)]],
                              constant uint& n [[buffer(2)]],
                              uint i [[thread_position_in_grid]]) {
  if (i < n)
    dst[i] = src[i];
}
)";

static MTL::ComputePipelineState* make_kernel(const char* name) {
  static MTL::Library* lib = [] {
    NS::Error* err = nullptr;
    auto* opts = MTL::CompileOptions::alloc()->init();
    opts->setLanguageVersion(MTL::LanguageVersion3_2);
    auto* l = gpu()->newLibrary(NS::String::string(kKernelSrc, NS::UTF8StringEncoding), opts, &err);
    opts->release();
    if (!l)
      throw std::runtime_error(std::string("moe_native kernels: ") +
                               (err ? err->localizedDescription()->utf8String() : "compile"));
    return l;
  }();
  NS::Error* err = nullptr;
  auto* fn = lib->newFunction(NS::String::string(name, NS::UTF8StringEncoding));
  auto* p = fn ? gpu()->newComputePipelineState(fn, &err) : nullptr;
  if (fn)
    fn->release();
  if (!p)
    throw std::runtime_error(std::string("moe_native: no pipeline for ") + name);
  return p;
}

static MTL::ComputePipelineState* bounded_wait_kernel() {
  static auto* p = make_kernel("bounded_wait");
  return p;
}

static MTL::ComputePipelineState* coherent_copy_kernel() {
  static auto* p = make_kernel("coherent_copy");
  return p;
}

// The stop words every wait reads, [0] set by the host to release and [1] the value a wait
// gave up on, as one pair per process that is never lowered.
static std::atomic<std::atomic<uint32_t>*> g_stop{nullptr};

static MTL::Buffer* stop_buffer() {
  static MTL::Buffer* b = [] {
    auto* b = gpu()->newBuffer(2 * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    if (!b)
      throw std::runtime_error("newBuffer failed");
    auto* w = static_cast<std::atomic<uint32_t>*>(b->contents());
    w[0].store(0);
    w[1].store(0);
    g_stop.store(w);
    return b;
  }();
  return b;
}

// Every wait of this process falls through now and later, and this stays async-signal-safe.
static void release_all() {
  if (auto* w = g_stop.load())
    w[0].store(1, std::memory_order_release);
}

static std::pair<uint32_t, uint32_t> stopped() {
  auto* w = g_stop.load();
  if (!w)
    return {0, 0};
  return {w[0].load(std::memory_order_acquire), w[1].load(std::memory_order_acquire)};
}

struct WaitBound {
  uint32_t cap;   // Spin rounds before a wait gives up.
  double rate;    // Measured spin rounds per second.
  double budget;  // Seconds a wait may spin.
};

// A shader has no clock, so the bound is a round count:
// MOE_NATIVE_WAIT_S seconds times this GPU's spin rounds per
// second, measured once on a word nobody raises at the fastest of
// three runs of at least 20 ms, because a busy or down-clocked
// GPU spins slower and that only makes a real wait's bound
// longer.
static const WaitBound& wait_bound() {
  static WaitBound wb = [] {
    const char* env = getenv("MOE_NATIVE_WAIT_S");
    double budget = env ? atof(env) : 5.0;
    if (!(budget > 0))
      throw std::invalid_argument("MOE_NATIVE_WAIT_S must be > 0 seconds");
    auto* pool = NS::AutoreleasePool::alloc()->init();
    auto* q = gpu()->newCommandQueue();
    auto* scratch = gpu()->newBuffer(4 * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto run = [&](uint32_t n) {
      memset(scratch->contents(), 0, 4 * sizeof(uint32_t));
      auto* cb = q->commandBuffer();
      auto* enc = cb->computeCommandEncoder();
      uint32_t v = 1;
      enc->setComputePipelineState(bounded_wait_kernel());
      enc->setBuffer(scratch, 0, 0);
      enc->setBuffer(scratch, 2 * sizeof(uint32_t), 1);
      enc->setBytes(&v, sizeof(v), 2);
      enc->setBytes(&n, sizeof(n), 3);
      enc->dispatchThreads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
      enc->endEncoding();
      cb->commit();
      // A run that passes 10 s means the cap did not work, so raise the word from here to
      // unstick the GPU and refuse to go on.
      uint64_t t0 = now_ns();
      while (cb->status() < MTL::CommandBufferStatusCompleted && now_ns() - t0 < 10000000000ull)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      bool late = cb->status() < MTL::CommandBufferStatusCompleted;
      static_cast<std::atomic<uint32_t>*>(scratch->contents())[0].store(~0u);
      cb->waitUntilCompleted();
      if (late)
        throw std::runtime_error("bounded_wait did not end at its cap");
      return cb->status() == MTL::CommandBufferStatusCompleted
                 ? cb->GPUEndTime() - cb->GPUStartTime()
                 : -1.0;
    };
    double rate = 0;
    uint32_t n = 1 << 12;
    for (int good = 0; good < 3;) {
      double t = run(n);
      if (t < 0)
        break;
      if (t >= 0.02) {
        rate = std::max(rate, n / t);
        good++;
      } else if (n <= (1u << 28)) {
        n *= 4;
      } else {
        break;
      }
    }
    scratch->release();
    q->release();
    pool->release();
    if (!(rate > 0))
      throw std::runtime_error("bounded_wait: could not time the spin on this GPU");
    return WaitBound{uint32_t(std::min(budget * rate, 4e9)), rate, budget};
  }();
  return wb;
}

// SIGTERM and SIGHUP release every wait and then chain to what was there before, the default
// action or a handler that a later signal.signal for the same signal replaces while the bound
// still holds.
//
// SIGINT stays Python's, because a KeyboardInterrupt leaves the agent running and exiting runs
// stream_fast's atexit hook, and a signal that was ignored stays ignored.
static struct sigaction g_prev[NSIG];
// The last wait encoded and its word, whose value shows whether the GPU is on a wait that only
// the agent would have answered.
static std::atomic<std::atomic<uint32_t>*> g_last_word{nullptr};
static std::atomic<uint32_t> g_last_wait{0};

static bool wait_pending() {
  auto* w = g_last_word.load();
  return w && w->load() < g_last_wait.load();
}

static void on_exit_signal(int sig, siginfo_t* info, void* ctx) {
  static const char a[] = "moe_native: exit signal, GPU waits released (one was pending)\n";
  static const char b[] = "moe_native: exit signal, GPU waits released (none pending)\n";
  release_all();
  ssize_t ignored = wait_pending() ? write(2, a, sizeof(a) - 1) : write(2, b, sizeof(b) - 1);
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

// One uint32 in a shared MTLBuffer that fence_update writes,
// bounded_wait spins on and the host loads and raises, and values must
// fit in 32 bits.
struct SharedEvent {
  NS::SharedPtr<MTL::Buffer> flag;

  SharedEvent() {
    stop_buffer();
    wait_bound(); // Time and compile the wait now, so the encode does not pay for it.
    flag = NS::TransferPtr(gpu()->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared));
    if (!flag)
      throw std::runtime_error("newBuffer failed");
    word()->store(0);
  }
  ~SharedEvent() {
    auto* w = word();
    g_last_word.compare_exchange_strong(w, nullptr);
  }
  SharedEvent(const SharedEvent&) = delete;
  std::atomic<uint32_t>* word() const {
    return static_cast<std::atomic<uint32_t>*>(flag->contents());
  }
  uint64_t value() const {
    return word()->load(std::memory_order_acquire);
  }
  // Raise the word to v and never lower it.
  void signal(uint64_t v) {
    auto* w = word();
    uint32_t cur = w->load(std::memory_order_relaxed);
    while (cur < uint32_t(v) &&
           !w->compare_exchange_weak(cur, uint32_t(v), std::memory_order_release))
      ;
  }
};

// ---------------------------------------------------------------------------
// The reader pool runs one wave of reads at a time, and the caller reads too.

static bool flag(const char* name) {
  const char* v = getenv(name);
  return v && v[0] == '1';
}

// The same file on its own descriptor with the file cache off, or the same descriptor when
// that fails.
static int nocache_fd(int fd) {
  char path[MAXPATHLEN];
  int f = fcntl(fd, F_GETPATH, path) == 0 ? open(path, O_RDONLY) : -1;
  if (f < 0)
    return fd;
  fcntl(f, F_NOCACHE, 1);
  return f;
}

struct ReadJob {
  int fd;
  int niov;
  iovec iov[4];
  off_t off;
  size_t len;
  int fd2 = -1;   // For a sidecar job, where to read the record when this read fails.
  off_t off2 = 0;
};

class Pool {
 public:
  explicit Pool(int workers) {
    for (int i = 0; i < workers; i++)
      th_.emplace_back([this] { loop(); });
  }
  ~Pool() {
    close();
  }
  // Stop and join the workers, so run() reads alone, and calling this twice is safe.
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

  int workers() const {
    return th_.size();
  }

  // Side work for idle workers under MOE_STAGING_READERS=pool: a worker with no
  // wave calls side() until it returns false and then waits for the next poke() or
  // wave.
  void set_side(std::function<bool()> f) {
    std::lock_guard lk(m_);
    side_ = std::move(f);
  }
  void poke() {
    {
      std::lock_guard lk(m_);
      side_gen_++;
    }
    cv_.notify_all();
  }
  // A wave is its own object, so a worker that wakes late finishes only jobs of the wave it
  // picked up and never repeats them into a later one.
  struct Wave {
    explicit Wave(std::vector<ReadJob> j) : jobs(std::move(j)) {}
    std::vector<ReadJob> jobs;
    std::atomic<size_t> next{0}, done{0};
    std::atomic<int> bad{0}, fell{0}; // bad counts short reads and fell counts reads that failed and then came from fd2.
    std::atomic<uint64_t> t_end{0};   // When the last read finished.
  };

  // Hand a wave to the workers alone and return, for a second device's queue run beside run()
  // on this agent's own pool, and wait() collects it.
  std::shared_ptr<Wave> start(std::vector<ReadJob> jobs) {
    auto w = std::make_shared<Wave>(std::move(jobs));
    {
      std::lock_guard lk(m_);
      cur_ = w;
      gen_++;
    }
    cv_.notify_all();
    return w;
  }
  static void wait(const Wave& w) {
    while (w.done.load(std::memory_order_acquire) < w.jobs.size()) {
    }
  }

  // Return how many reads came back short.
  int run(std::vector<ReadJob> jobs) {
    if (jobs.empty())
      return 0;
    auto w = std::make_shared<Wave>(std::move(jobs));
    if (w->jobs.size() > 1 && !th_.empty()) {
      {
        std::lock_guard lk(m_);
        cur_ = w;
        gen_++;
      }
      cv_.notify_all();
    }
    work(*w);
    while (w->done.load(std::memory_order_acquire) < w->jobs.size()) {
    }
    return w->bad.load();
  }

 private:
  static void work(Wave& w) {
    for (size_t i; (i = w.next.fetch_add(1)) < w.jobs.size();) {
      auto& j = w.jobs[i];
      if (preadv(j.fd, j.iov, j.niov, j.off) != ssize_t(j.len)) {
        // When the sidecar's drive is gone or unreadable, the same record comes from
        // experts.bin and overwrites the iovecs whole.
        if (j.fd2 >= 0 && preadv(j.fd2, j.iov, j.niov, j.off2) == ssize_t(j.len))
          w.fell++;
        else
          w.bad++;
      }
      uint64_t t = now_ns(), cur = w.t_end.load();
      while (cur < t && !w.t_end.compare_exchange_weak(cur, t)) {
      }
      w.done.fetch_add(1, std::memory_order_release);
    }
  }
  void loop() {
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    uint64_t seen = 0, sseen = 0;
    std::unique_lock lk(m_);
    for (;;) {
      cv_.wait(lk, [&] { return stop_ || gen_ != seen || (side_ && side_gen_ != sseen); });
      if (stop_)
        return;
      if (gen_ != seen) {
        seen = gen_;
        auto w = cur_;
        lk.unlock();
        work(*w);
        w.reset();
        lk.lock();
        continue;
      }
      uint64_t g = side_gen_;
      auto f = side_;
      lk.unlock();
      bool did = f();
      lk.lock();
      if (!did)
        sseen = g; // Nothing was left to take as of poke g.
    }
  }
  std::vector<std::thread> th_;
  std::mutex m_;
  std::condition_variable cv_;
  std::shared_ptr<Wave> cur_;
  uint64_t gen_ = 0, side_gen_ = 0;
  bool stop_ = false;
  std::function<bool()> side_;
};

// ---------------------------------------------------------------------------
// The agent: the thread that serves jobs and the tables it keeps.

struct Layer {
  int cap = 0, n_exp = 0;
  int32_t* slot_of = nullptr; // The GPU-visible expert to slot table, with -1 for an absent expert.
  int32_t* phys_of = nullptr; // With staging on, expert to physical row, which is what the GPU gathers.
  std::vector<int> owner;     // Slot, or quota position in a store, to expert, with -1 for empty.
  std::vector<uint64_t> stamp; // Slot, or quota position in a store, to the time of its last use.
  std::vector<uint32_t> seen; // Expert to the claims since take_seen.
  uint64_t clock = 0;
  // Store layers also keep the quota position to store row map, the store row to quota
  // position map with -1 for rows outside the quota, the ring half and every claim's ids since
  // take_log for Python's ghost LRU.
  int ring = -1;
  std::vector<int> qslot, qpos;
  std::vector<int32_t> log_ids, log_len;
  struct Proj {
    off_t off;
    size_t stride;
    std::vector<std::pair<uint8_t*, size_t>> parts; // The address of the slot 0 row and its bytes.
  };
  std::vector<Proj> projs;
  std::vector<mx::array> keep; // The arrays behind the raw pointers.

  int slot_at(int p) const {
    return ring < 0 ? p : qslot[p];
  }
  int pos_of(int s) const {
    return ring < 0 ? s : qpos.at(s);
  }
};

// One half of a store's ring, rows [base, base + n) handed out in turn, each remembering the
// layer and expert it holds.
struct Ring {
  int base = 0, next = 0;
  std::vector<std::pair<int, int>> owner; // The layer and expert it holds, with layer -1 for empty.
};

// One staging row, which holds the guess read into it and the state of its records.
enum : int8_t { kQueued = 1, kFlight, kDone, kDropped, kBad };
struct Stage {
  int layer = -1, e = -1; // The guess read into it, where layer -1 means free once nothing is in flight.
  int row = -1;           // The physical store row it is read into.
  int flight = 0;         // The records being read into it.
  int8_t rec[4] = {};     // The state of each projection's record.
};

// MOE_CPU_GUESS: a layer's router as set_router hands it over, MLX's buffers read in place.
struct Router {
  const uint8_t* w = nullptr;            // E rows of D bytes.
  const void *s = nullptr, *b = nullptr; // Scales and biases, E x D / gs, bf16 (bf) or fp32.
  bool bf = false;
  int E = 0, D = 0, gs = 0, k = 0;
  std::vector<mx::array> keep;
};

class Agent {
 public:
  Agent(SharedEvent& ev, int fd, int n_layers, int readers)
      : ev_(ev), fd_(fd), pool_(std::max(0, readers - 1)), layers_(n_layers) {
    th_ = std::thread([this] { loop(); });
  }
  ~Agent() {
    close();
    if (gfd_ >= 0 && gfd_ != fd_)
      ::close(gfd_);
    if (gmfd_ >= 0 && gmfd_ != mfd_)
      ::close(gmfd_);
  }
  void close() {
    if (!th_.joinable())
      return;
    {
      std::lock_guard lk(m_);
      stop_ = true;
    }
    cv_.notify_all();
    gcv_.notify_all();
    th_.join();
    for (auto& t : gth_)
      t.join();
    gth_.clear();
    if (hth_.joinable()) {
      {
        std::lock_guard lk(hm_);
        hstop_ = true;
      }
      hcv_.notify_all();
      hth_.join();
    }
    // Readers that may run side work go before the state that work reads.
    pool_.close();
    if (ext_)
      ext_->close();
  }

  // Wait until every job pushed so far is answered and the thread is idle or gone, with .error
  // saying why and no guess being read, since the mutex is the happens-before edge for the
  // tables the host reads next.
  bool drain(double timeout_s) {
    std::unique_lock lk(m_);
    return idle_cv_.wait_for(lk, std::chrono::duration<double>(timeout_s),
                             [&] { return dead_ || idle_locked(); });
  }

  // projs holds, per projection, its file offset, its stride and its part arrays of cap rows,
  // which must be evaluated and row contiguous, with slot_of as int32 of n_exp entries, and a
  // store layer passes its quota's rows and its ring half through set_ring while the table
  // starts empty for set_state to fill.
  void set_layer(
      int i, mx::array slot_of,
      std::vector<std::tuple<int64_t, int64_t, std::vector<mx::array>>> projs,
      std::vector<int> quota, int ring, std::optional<mx::array> phys) {
    idle_only();
    Layer L;
    if (slot_of.dtype() != mx::int32 || !slot_of.flags().row_contiguous)
      throw std::invalid_argument("slot_of must be contiguous int32");
    L.n_exp = slot_of.size();
    L.slot_of = slot_of.data<int32_t>();
    L.keep.push_back(slot_of);
    if (phys) {
      if (phys->dtype() != mx::int32 || !phys->flags().row_contiguous ||
          int(phys->size()) != L.n_exp)
        throw std::invalid_argument("phys_of must be contiguous int32 like slot_of");
      L.phys_of = phys->data<int32_t>();
      L.keep.push_back(*phys);
    }
    for (auto& [off, stride, parts] : projs) {
      Layer::Proj p{off_t(off), size_t(stride), {}};
      size_t sum = 0;
      for (auto& a : parts) {
        if (!a.flags().row_contiguous || a.ndim() < 2)
          throw std::invalid_argument("arena part must be contiguous");
        if (L.cap && L.cap != a.shape(0))
          throw std::invalid_argument("arena parts disagree on slots");
        L.cap = a.shape(0);
        size_t row = a.nbytes() / a.shape(0);
        p.parts.push_back({a.data<uint8_t>(), row});
        L.keep.push_back(a);
        sum += row;
      }
      if (sum != size_t(stride) || p.parts.size() > 4)
        throw std::invalid_argument("parts do not add up to the stride");
      L.projs.push_back(std::move(p));
    }
    if (L.projs.size() > 4) // Stage::rec holds four projections.
      throw std::invalid_argument("more than four projections");
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
    layers_.at(i) = std::move(L);
  }

  // Set a ring half's first row and its layer and expert per row, with layer -1 for empty, on
  // an idle agent.
  void set_ring(int r, int base, int next, std::vector<std::pair<int, int>> owner) {
    idle_only();
    if (owner.empty() || next < 0 || next >= int(owner.size()))
      throw std::invalid_argument("ring: no rows or bad next");
    if (r >= int(rings_.size()))
      rings_.resize(r + 1);
    rings_[r] = Ring{base, next, std::move(owner)};
  }

  // Return a ring half's next row and owners, on an idle agent.
  std::pair<int, std::vector<std::pair<int, int>>> ring_state(int r) {
    idle_only();
    auto& R = rings_.at(r);
    return {R.next, R.owner};
  }

  // Return a store layer's claims since the last call as int32 bytes of ids and of ids per
  // claim, on an idle agent.
  std::pair<nb::bytes, nb::bytes> take_log(int i) {
    idle_only();
    auto& L = live(i);
    std::pair<nb::bytes, nb::bytes> out{
        nb::bytes(reinterpret_cast<const char*>(L.log_ids.data()), L.log_ids.size() * 4),
        nb::bytes(reinterpret_cast<const char*>(L.log_len.data()), L.log_len.size() * 4)};
    L.log_ids.clear();
    L.log_len.clear();
    return out;
  }

  // Drop the layer and the arrays it holds, on an idle agent.
  void clear_layer(int i) {
    idle_only();
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

  // Return the table as slot and expert pairs, least recently used first, on an idle agent.
  std::vector<std::pair<int, int>> lru(int i) {
    idle_only();
    auto& L = live(i);
    std::vector<std::pair<int, int>> out;
    for (int p = 0; p < int(L.owner.size()); p++)
      if (L.owner[p] >= 0)
        out.push_back({p, L.owner[p]});
    std::sort(out.begin(), out.end(), [&](auto a, auto b) {
      return L.stamp[a.first] < L.stamp[b.first];
    });
    for (auto& pe : out)
      pe.first = L.slot_at(pe.first);
    return out;
  }

  // Return claims per expert since the last call as uint32 bytes, on an idle agent.
  nb::bytes take_seen(int i) {
    idle_only();
    auto& L = live(i);
    nb::bytes out(reinterpret_cast<const char*>(L.seen.data()),
                  L.seen.size() * sizeof(uint32_t));
    std::fill(L.seen.begin(), L.seen.end(), 0);
    return out;
  }

  // Set up a sidecar on a second drive: its fd, the experts.bin
  // offset, sidecar offset and bytes of each record it holds,
  // both drives' starting rates and its reader count, on an idle
  // agent.
  void set_mirror(int fd, std::vector<std::tuple<int64_t, int64_t, int64_t>> recs,
                  double bw_int, double bw_ext, int readers) {
    idle_only();
    mfd_ = fd;
    mir_.clear();
    for (auto [m, e, n] : recs)
      mir_[m] = {e, n};
    ema_[0][0] = bw_int * 1e-3;
    ema_[1][0] = bw_ext * 1e-3;
    ema_[0][1] = ema_[1][1] = 1e-3;
    mir_dead_ = false;
    ext_ = std::make_unique<Pool>(std::max(1, readers));
  }

  // Return the read waves since the last call as experts.bin bytes, sidecar bytes, experts.bin
  // ns, sidecar ns and reads that fell back to experts.bin.
  std::tuple<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t> mirror_stats() {
    return {int_b_.exchange(0), ext_b_.exchange(0), int_ns_.exchange(0),
            ext_ns_.exchange(0), fell_.exchange(0)};
  }

  // Return layers, hits, misses, miss layers, park ns and read ns since the last call.
  std::tuple<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t> stats() {
    std::lock_guard lk(m_);
    auto out = std::make_tuple(n_layers_, n_hits_, n_miss_, n_miss_layers_,
                               park_ns_, read_ns_);
    n_layers_ = n_hits_ = n_miss_ = n_miss_layers_ = park_ns_ = read_ns_ = 0;
    return out;
  }

  // Give the shared store `rows` staging rows past its `slots` logical
  // rows, which the tensors must already have and which every store layer
  // reaches through a phys_of, and 0 rows turns staging off, while the
  // layout must be the logical one, fresh or after unstage, on an idle
  // agent.
  void set_staging(int slots, int rows) {
    idle_only();
    perm_.clear();
    stage_.clear();
    if (rows <= 0)
      return;
    store_ = -1;
    for (int i = 0; i < int(layers_.size()); i++) {
      auto& L = layers_[i];
      if (!L.slot_of || L.ring < 0)
        continue;
      if (!L.phys_of)
        throw std::invalid_argument("staging: a store layer without phys_of");
      if (store_ < 0)
        store_ = i;
      else if (L.projs[0].parts[0].first != layers_[store_].projs[0].parts[0].first)
        throw std::invalid_argument("staging: the layers are on more than one store");
    }
    if (store_ < 0 || layers_[store_].cap < slots + rows)
      throw std::invalid_argument("staging: no store, or no room past its rows");
    perm_.resize(slots);
    std::iota(perm_.begin(), perm_.end(), 0);
    stage_.assign(rows, Stage{});
    for (int r = 0; r < rows; r++)
      stage_[r].row = slots + r;
    if (gfd_ < 0) {
      // Guess reads spend the rest of the token, so MOE_STAGING_NOCACHE=1 reads them
      // past the file cache on descriptors of their own and MOE_STAGING_READERS=pool has
      // the readers that exist take them between waves instead of four threads of their
      // own.
      gfd_ = flag("MOE_STAGING_NOCACHE") ? nocache_fd(fd_) : fd_;
      gmfd_ = flag("MOE_STAGING_NOCACHE") && ext_ ? nocache_fd(mfd_) : mfd_;
      // For tests, MOE_STAGING_FAULT="n,us" fails every n-th guess read and slows each one by
      // us microseconds.
      if (const char* f = getenv("MOE_STAGING_FAULT"))
        sscanf(f, "%d,%d", &fault_every_, &fault_us_);
      // Side work on the pool only when it has readers, since drain() waits for the guess
      // queue to empty.
      const char* rd = getenv("MOE_STAGING_READERS");
      if (rd && !strcmp(rd, "pool") && pool_.workers() > 0) {
        pool_.set_side([this] { return side_read(false); });
        if (ext_)
          ext_->set_side([this] { return side_read(true); });
      } else {
        // Two threads per drive, so at most two guess records are in flight on each.
        for (int k = 0; k < 2; k++)
          gth_.emplace_back([this] { guess_loop(false); });
        if (ext_)
          for (int k = 0; k < 2; k++)
            gth_.emplace_back([this] { guess_loop(true); });
      }
    }
  }

  // Put every logical row back at its own physical row, so Python's view of
  // the store is its layout again, and return the rows copied, on an idle
  // agent.
  int unstage() {
    idle_only();
    if (perm_.empty())
      return 0;
    int slots = perm_.size(), n = slots + stage_.size();
    std::vector<int> inv(n, -1); // Physical row to the logical row it holds.
    for (int x = 0; x < slots; x++)
      inv[perm_[x]] = x;
    int copies = 0;
    auto put = [&](int x, int to) { // Move logical row x's bytes to the free physical row `to`.
      copy_row(perm_[x], to);
      inv[perm_[x]] = -1;
      perm_[x] = to;
      inv[to] = x;
      copies++;
    };
    // The rows move one at a time on this
    // thread.
    for (int x = 0; x < slots; x++) {
      if (perm_[x] == x)
        continue;
      // Follow the chain of rows from x to a free row and move each home from
      // the end, and when the chain returns to x, move x out to a free row
      // first.
      std::vector<int> chain{x};
      for (int c = inv[x]; c >= 0; c = inv[c]) {
        if (c == x) {
          put(x, int(std::find(inv.begin(), inv.end(), -1) - inv.begin()));
          break;
        }
        chain.push_back(c);
      }
      for (int i = int(chain.size()) - 1; i >= 0; i--)
        put(chain[i], chain[i]);
    }
    for (int r = 0; r < int(stage_.size()); r++)
      stage_[r] = Stage{-1, -1, slots + r};
    s_unstaged_ += copies;
    return copies;
  }

  // Return, since the last call, the guessed experts queued, the guess records read, the
  // guessed misses promoted and how many of those needed records read on demand, the guesses
  // left unclaimed, the ns the claims waited for guess records, the records dropped unstarted,
  // the rows copied back by unstage and the guess reads that failed and were read again on
  // demand.
  std::vector<uint64_t> stage_stats() {
    std::lock_guard lk(m_);
    std::vector<uint64_t> out{s_guess_, s_rec_,  s_prom_,     s_part_, s_wrong_,
                              s_wait_ns_, s_drop_, s_unstaged_, s_bad_};
    s_guess_ = s_rec_ = s_prom_ = s_part_ = s_wrong_ = s_wait_ns_ = s_drop_ = s_unstaged_ =
        s_bad_ = 0;
    return out;
  }

  // MOE_CPU_GUESS: layer i's router (MLX's evaluated 8-bit affine weight, scales and biases) and its top k,
  // ranked on this thread for the jobs of layer i - 1 that carry their MoE input, on an idle agent.
  void set_router(int i, mx::array w, mx::array scales, mx::array biases, int group_size, int k) {
    idle_only();
    bool bf = scales.dtype() == mx::bfloat16;
    int E = w.ndim() == 2 ? w.shape(0) : 0, D = w.ndim() == 2 ? w.shape(1) * 4 : 0;
    if (w.dtype() != mx::uint32 || !E || (!bf && scales.dtype() != mx::float32) ||
        biases.dtype() != scales.dtype() || !w.flags().row_contiguous ||
        !scales.flags().row_contiguous || !biases.flags().row_contiguous || group_size <= 0 ||
        D % group_size || scales.size() != size_t(E) * (D / group_size) ||
        biases.size() != scales.size() || k <= 0 || k > std::min(E, 32))
      throw std::invalid_argument("router: an 8-bit affine weight with bf16 or fp32 scales");
    routers_.resize(layers_.size());
    if (!hth_.joinable())
      hth_ = std::thread([this] { guess_helper(); });
    routers_.at(i) = Router{w.data<uint8_t>(), scales.data<uint8_t>(), biases.data<uint8_t>(),
                            bf, E, D, group_size, k, {w, scales, biases}};
  }

  // Return, since the last call, the guesses ranked here, their ns and the most one took, the
  // jobs whose input could not be ranked, and under a GPU guess too (MOE_CPU_GUESS=check) the
  // guesses compared, those equal in ids and order and those equal only as sets.
  std::vector<uint64_t> cpu_guess_stats() {
    return {cg_n_.exchange(0),   cg_ns_.exchange(0),  cg_max_.exchange(0), cg_skip_.exchange(0),
            cg_cmp_.exchange(0), cg_same_.exchange(0), cg_set_.exchange(0)};
  }

  // For tests, set the logical to physical map to `perm`, whose rows are distinct and in
  // range, the staging rows become the rows it leaves, and unstage() must then put every
  // logical row's bytes home on an idle agent.
  void set_perm(std::vector<int> perm) {
    idle_only();
    int n = perm_.size() + stage_.size();
    if (perm.size() != perm_.size())
      throw std::invalid_argument("set_perm: one row per logical row");
    std::vector<char> used(n, 0);
    for (int p : perm) {
      if (p < 0 || p >= n || used[p])
        throw std::invalid_argument("set_perm: rows must be distinct and in range");
      used[p] = 1;
    }
    perm_ = std::move(perm);
    int r = 0;
    for (int p = 0; p < n; p++)
      if (!used[p])
        stage_[r++] = Stage{-1, -1, p};
  }

  std::string error() {
    std::lock_guard lk(m_);
    return error_;
  }

  // Called from SignalAfter::eval_gpu while MLX encodes, before the GPU can reach v, with the
  // last `guess` ids as the next layer's guess, and it never throws because even a bad job is
  // answered with an error. h (MOE_CPU_GUESS): the layer's MoE input, which the next layer's
  // router ranks here under bias lam and budget tau, where an input it cannot take means no guess.
  void push(int layer, uint64_t v, const mx::array& ids, int guess, const mx::array* h = nullptr,
            float lam = 0.0f, float tau = 0.0f) {
    Job j{layer, v, ids.data_shared_ptr(), nullptr, 0, 0};
    std::string bad;
    if (layer < 0 || layer >= int(layers_.size()) || ids.dtype() != mx::int32 ||
        !ids.flags().row_contiguous || ids.size() > 64 || guess < 0 ||
        guess >= int(ids.size()))
      bad = "agent job: bad layer or ids";
    else {
      j.ids = ids.data<int32_t>();
      j.n = ids.size() - guess;
      j.guess = guess;
      const Router* R = h && layer + 1 < int(routers_.size()) ? &routers_[layer + 1] : nullptr;
      if (R && R->w && h->flags().row_contiguous && int(h->size()) == R->D &&
          (h->dtype() == mx::bfloat16 || h->dtype() == mx::float32)) {
        j.hkeep = h->data_shared_ptr();
        j.h = h->data<uint8_t>();
        j.hbf = h->dtype() == mx::bfloat16;
        j.lam = lam;
        j.tau = tau;
      } else if (h) {
        cg_skip_++;
      }
    }
    {
      std::lock_guard lk(m_);
      if (!bad.empty() && error_.empty())
        error_ = bad;
      q_.push_back(std::move(j));
    }
    cv_.notify_one();
  }

  // Claim on a flat table as stream_fast._claim's arena path
  // does: hits refresh their slot, and misses take the
  // lowest empty slot in order, else the least recently used
  // one.
  static std::vector<std::pair<int, int>> claim(Layer& L, std::vector<int> ids) {
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
    std::vector<std::pair<int, int>> miss;
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
      if (L.owner[best] >= 0)
        L.slot_of[L.owner[best]] = -1;
      L.owner[best] = e;
      L.slot_of[e] = best;
      L.stamp[best] = ++L.clock;
      miss.push_back({best, e});
    }
    return miss;
  }

  // Place one decode pass the way stream_fast._place_store does, with the ids sorted and
  // distinct: quota hits refresh, ring hits are kept, misses fill the quota at the lowest free
  // position or its least recently used member, and what the quota cannot take goes to the
  // ring half, whose previous holder of any layer loses that row.
  //
  // That holder's gather is already done, because its layer feeds this layer's router.
  std::vector<std::pair<int, int>> claim_store(int li, Layer& L, std::vector<int> ids) {
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
    std::vector<std::pair<int, int>> miss;
    for (int n = 0; n < int(pend.size()); n++) {
      int e = pend[n], s = -1;
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
        if (L.owner[best] >= 0)
          L.slot_of[L.owner[best]] = -1;
        L.owner[best] = e;
        L.stamp[best] = ++L.clock;
        s = L.qslot[best];
      }
      L.slot_of[e] = s;
      miss.push_back({s, e});
    }
    L.log_ids.insert(L.log_ids.end(), ids.begin(), ids.end());
    L.log_len.push_back(ids.size());
    return miss;
  }

 private:
  struct Job {
    int layer;
    uint64_t v;
    std::shared_ptr<mx::array::Data> keep;
    const int32_t* ids;
    int n, guess; // The layer's ids, followed by the next layer's guess.
    std::shared_ptr<mx::array::Data> hkeep; // MOE_CPU_GUESS: the MoE input, bf16 or fp32.
    const void* h = nullptr;
    bool hbf = false;
    float lam = 0.0f, tau = 0.0f;
  };

  bool idle_locked() const {
    return q_.empty() && !busy_ && g_flight_ == 0 && gq_.empty();
  }
  void idle_only() {
    std::lock_guard lk(m_);
    if (!idle_locked())
      throw std::runtime_error("agent is busy: drain() first");
  }

  // Staging rows, which set_staging turns on.

  void copy_row(int from, int to) {
    for (auto& p : layers_.at(store_).projs)
      for (auto [base, row] : p.parts)
        memcpy(base + size_t(to) * row, base + size_t(from) * row, row);
  }

  // Build the read of projection k of expert e of layer li into physical row `row`.
  ReadJob record(int li, int k, int row, int e) {
    auto& p = layers_[li].projs[k];
    ReadJob j{fd_, int(p.parts.size()), {}, p.off + off_t(e) * off_t(p.stride), p.stride};
    for (int i = 0; i < j.niov; i++)
      j.iov[i] = {p.parts[i].first + size_t(row) * p.parts[i].second, p.parts[i].second};
    return j;
  }

  // Return the sidecar's offset of a record of experts.bin, or -1 when it has none.
  int64_t mirrored(const ReadJob& j) const {
    if (mir_dead_)
      return -1;
    auto it = mir_.find(j.off);
    return it != mir_.end() && size_t(it->second.second) == j.len ? it->second.first : -1;
  }

  // With m_ held, take the next queue record a drive may read, best guess first and for the
  // sidecar the first one it holds, and mark it in flight.
  bool take(bool ext, ReadJob& j, int& si, int& k) {
    for (size_t at = 0; at < gq_.size(); at++) {
      auto [s, kk] = gq_[at];
      ReadJob r = record(stage_[s].layer, kk, stage_[s].row, stage_[s].e);
      int64_t eoff = ext ? mirrored(r) : -1;
      if (ext && eoff < 0)
        continue;
      if (ext) {
        r.fd2 = r.fd;
        r.off2 = r.off;
        r.fd = gmfd_;
        r.off = eoff;
      } else {
        r.fd = gfd_;
      }
      gq_.erase(gq_.begin() + at);
      j = r;
      si = s;
      k = kk;
      auto& S = stage_[si];
      S.rec[k] = kFlight;
      S.flight++;
      g_flight_++;
      gdrive_[ext]++;
      return true;
    }
    return false;
  }

  // Read a taken record without the lock and mark it once m_ is held again.
  void read_taken(std::unique_lock<std::mutex>& lk, bool ext, const ReadJob& j, int si, int k) {
    int nth = ++fault_n_;
    lk.unlock();
    if (fault_us_ > 0)
      std::this_thread::sleep_for(std::chrono::microseconds(fault_us_));
    bool ok = preadv(j.fd, j.iov, j.niov, j.off) == ssize_t(j.len), fell = false;
    if (!ok && j.fd2 >= 0)
      ok = fell = preadv(j.fd2, j.iov, j.niov, j.off2) == ssize_t(j.len);
    if (fault_every_ > 0 && nth % fault_every_ == 0)
      ok = false; // MOE_STAGING_FAULT makes this read look short.
    if (fell && !mir_dead_.exchange(true)) {
      // A failed guess read retires the sidecar from here on.
      fell_++;
      fprintf(stderr, "  native: sidecar read failed; experts.bin from here on\n");
    }
    lk.lock();
    auto& S = stage_[si];
    S.rec[k] = ok ? kDone : kBad;
    S.flight--;
    g_flight_--;
    gdrive_[ext]--;
    s_rec_++;
    s_bad_ += !ok;
    scv_.notify_all();
    if (idle_locked())
      idle_cv_.notify_all();
  }

  // The loop of a guess thread, two of which run per drive.
  void guess_loop(bool ext) {
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    std::unique_lock lk(m_);
    for (;;) {
      ReadJob j{};
      int si = 0, k = 0;
      gcv_.wait(lk, [&] { return stop_ || take(ext, j, si, k); });
      if (stop_)
        return;
      read_taken(lk, ext, j, si, k);
    }
  }

  // Under MOE_STAGING_READERS=pool, an idle reader of a drive's pool takes one guess record
  // while fewer than two are in flight there.
  bool side_read(bool ext) {
    std::unique_lock lk(m_);
    ReadJob j{};
    int si = 0, k = 0;
    if (stop_ || gdrive_[ext] >= 2 || !take(ext, j, si, k))
      return false;
    read_taken(lk, ext, j, si, k);
    return true;
  }

  // At a job's start, drop the guess records not yet started, since a claimed guess reads its
  // records with the demand.
  void drop_guesses() {
    std::lock_guard lk(m_);
    for (auto [si, k] : gq_)
      stage_[si].rec[k] = kDropped;
    s_drop_ += gq_.size();
    gq_.clear();
  }

  // Read a store layer's misses with staging on: a guessed miss is promoted by swapping
  // its logical row onto the staging row it was read into, with its dropped records read
  // there alongside the rest, and every other miss is read into its row's physical
  // place.
  int staged_reads(int li, Layer& L, const std::vector<std::pair<int, int>>& miss) {
    std::vector<ReadJob> jobs;
    std::vector<std::pair<int, int>> plain;
    std::vector<int> mine;
    {
      std::lock_guard lk(m_);
      for (auto [s, e] : miss) {
        int si = 0;
        while (si < int(stage_.size()) && !(stage_[si].layer == li && stage_[si].e == e))
          si++;
        if (si == int(stage_.size())) {
          plain.push_back({perm_[s], e});
          continue;
        }
        auto& S = stage_[si];
        std::swap(perm_[s], S.row);
        mine.push_back(si);
        bool part = false;
        for (int k = 0; k < int(L.projs.size()); k++)
          if (S.rec[k] == kDropped) {
            jobs.push_back(record(li, k, perm_[s], e));
            part = true;
          }
        s_prom_++;
        s_part_ += part;
      }
      // Free the guesses nobody claimed, this layer's or left from a pass that stopped, once
      // their reads in flight land.
      for (int si = 0; si < int(stage_.size()); si++)
        if (stage_[si].layer >= 0 && std::find(mine.begin(), mine.end(), si) == mine.end()) {
          stage_[si].layer = -1;
          s_wrong_++;
        }
    }
    auto rest = jobs_for(L, plain);
    jobs.insert(jobs.end(), rest.begin(), rest.end());
    int bad = jobs.empty() ? 0 : read_wave(std::move(jobs));
    std::vector<ReadJob> again;
    {
      std::unique_lock lk(m_);
      uint64_t t = now_ns();
      scv_.wait(lk, [&] {
        for (int si : mine)
          if (stage_[si].flight)
            return false;
        return true;
      });
      s_wait_ns_ += now_ns() - t;
      for (int si : mine) {
        auto& S = stage_[si];
        for (int k = 0; k < int(L.projs.size()); k++)
          if (S.rec[k] == kBad) // A guess read that failed is read on demand now.
            again.push_back(record(li, k, perm_[L.slot_of[S.e]], S.e));
        S.layer = -1;
      }
    }
    if (!again.empty())
      bad += pool_.run(std::move(again));
    return bad;
  }

  // MOE_CPU_GUESS: hand job j's guess to the helper at the job's start, with the bias mask read
  // here: the next layer's quota members as stream_fast's _keys reads them on a native pass (its
  // slot below the pool), which this layer's claim does not change, and no bias where it holds none.
  void start_guess(const Job& j) {
    auto& R = routers_[j.layer + 1];
    auto& L = layers_[j.layer + 1];
    std::vector<uint8_t> mask(R.E, 0);
    bool any = false;
    if (j.lam > 0.0f && L.slot_of && L.n_exp == R.E)
      for (int e = 0; e < R.E; e++) {
        int s = L.slot_of[e];
        mask[e] = s >= 0 && s < int(L.qpos.size()) && L.qpos[s] >= 0;
        any |= mask[e];
      }
    {
      std::lock_guard lk(hm_);
      hjob_ = &j;
      hmask_ = std::move(mask);
      hany_ = any;
      hdone_ = false;
    }
    hcv_.notify_all();
  }

  // The guess start_guess asked for, best first, once the helper has it.
  std::vector<int> wait_guess() {
    std::unique_lock lk(hm_);
    hcv_.wait(lk, [&] { return hdone_; });
    hjob_ = nullptr;
    return std::move(hout_);
  }

  // The helper thread: the next layer's router on the job's MoE input, then the selection.
  void guess_helper() {
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    std::unique_lock lk(hm_);
    for (;;) {
      hcv_.wait(lk, [&] { return hstop_ || (hjob_ && !hdone_); });
      if (hstop_)
        return;
      const Job& j = *hjob_;
      lk.unlock();
      uint64_t t = now_ns();
      auto& R = routers_[j.layer + 1];
      std::vector<float> x(R.D), lg(R.E);
      for (int i = 0; i < R.D; i++)
        x[i] = cpu_guess::load(j.h, i, j.hbf);
      cpu_guess::logits8(R.w, R.s, R.b, R.bf, x.data(), R.E, R.D, R.gs, lg.data(), j.hbf);
      std::vector<int32_t> out(R.k);
      cpu_guess::pick(lg.data(), hmask_.data(), R.E, R.k, hany_ ? j.lam : 0.0f,
                      hany_ ? j.tau : 0.0f, out.data());
      uint64_t dt = now_ns() - t;
      cg_n_++;
      cg_ns_ += dt;
      for (uint64_t m = cg_max_.load(); dt > m && !cg_max_.compare_exchange_weak(m, dt);) {
      }
      lk.lock();
      hout_.assign(out.begin(), out.end());
      hdone_ = true;
      hcv_.notify_all();
    }
  }

  // Once the GPU is let go, put the guessed experts layer li does not hold into free staging
  // rows, best first, until the rows run out.
  void stage_guesses(int li, const std::vector<int>& guess) {
    if (li >= int(layers_.size()))
      return;
    auto& L = layers_[li];
    if (!L.slot_of || L.ring < 0)
      return;
    std::lock_guard lk(m_);
    for (int e : guess) {
      if (e < 0 || e >= L.n_exp || L.slot_of[e] >= 0)
        continue;
      int free = -1;
      bool dup = false;
      for (int si = 0; si < int(stage_.size()); si++) {
        dup |= stage_[si].layer == li && stage_[si].e == e;
        if (free < 0 && stage_[si].layer < 0 && !stage_[si].flight)
          free = si;
      }
      if (dup)
        continue;
      if (free < 0)
        break;
      auto& S = stage_[free];
      S.layer = li;
      S.e = e;
      for (int k = 0; k < int(L.projs.size()); k++) {
        S.rec[k] = kQueued;
        gq_.push_back({free, k});
      }
      s_guess_++;
    }
    gcv_.notify_all();
  }
  Layer& live(int i) {
    auto& L = layers_.at(i);
    if (!L.slot_of)
      throw std::runtime_error("agent layer not set");
    return L;
  }

  // Build one preadv per projection of each expert, each scattering the projection's stride
  // into its part rows, which is what _claimed builds.
  std::vector<ReadJob> jobs_for(Layer& L, const std::vector<std::pair<int, int>>& se) {
    std::vector<ReadJob> jobs;
    for (auto [s, e] : se)
      for (auto& p : L.projs) {
        ReadJob j{fd_, int(p.parts.size()), {}, p.off + off_t(e) * off_t(p.stride),
                  p.stride};
        for (int k = 0; k < j.niov; k++)
          j.iov[k] = {p.parts[k].first + size_t(s) * p.parts[k].second,
                      p.parts[k].second};
        jobs.push_back(j);
      }
    return jobs;
  }

  // Read one layer's records, sending the sidecar's records to its own pool while that drive
  // would finish its share before experts.bin finishes the rest, so both drives read at once,
  // and each drive's bytes and time decay per wave as in stream_fast._Stripe while records
  // split whole as _Stripe.split does.
  int read_wave(std::vector<ReadJob> jobs) {
    if (!ext_)
      return pool_.run(std::move(jobs));
    std::vector<ReadJob> own, ext;
    std::vector<std::pair<size_t, off_t>> cand;
    double ib = 0, eb = 0;
    for (size_t k = 0; k < jobs.size(); k++) {
      auto it = mir_dead_ ? mir_.end() : mir_.find(jobs[k].off);
      if (jobs[k].fd == fd_ && it != mir_.end() && size_t(it->second.second) == jobs[k].len)
        cand.push_back({k, off_t(it->second.first)});
      else {
        own.push_back(jobs[k]);
        ib += jobs[k].len;
      }
    }
    double bi = ema_[0][0] / ema_[0][1], be = ema_[1][0] / ema_[1][1];
    for (auto [k, eoff] : cand) {
      ReadJob j = jobs[k];
      if ((eb + j.len) / be < (ib + j.len) / bi) {
        j.fd2 = j.fd;
        j.off2 = j.off;
        j.fd = mfd_;
        j.off = eoff;
        ext.push_back(j);
        eb += j.len;
      } else {
        own.push_back(j);
        ib += j.len;
      }
    }
    uint64_t t0 = now_ns();
    std::shared_ptr<Pool::Wave> w;
    if (!ext.empty())
      w = ext_->start(std::move(ext));
    int bad = pool_.run(std::move(own));
    if (ib > 0)
      note(0, ib, now_ns() - t0);
    if (w) {
      Pool::wait(*w);
      bad += w->bad.load();
      note(1, eb, w->t_end.load() - t0);
      if (int f = w->fell.load()) {
        fell_ += f;
        mir_dead_ = true;
        fprintf(stderr, "  native: sidecar read failed; experts.bin from here on\n");
      }
    }
    return bad;
  }
  void note(int d, double bytes, uint64_t ns) {
    ema_[d][0] = kDecay * ema_[d][0] + bytes;
    ema_[d][1] = kDecay * ema_[d][1] + ns * 1e-9;
    (d ? ext_b_ : int_b_) += uint64_t(bytes);
    (d ? ext_ns_ : int_ns_) += ns;
  }

  // Spin while the GPU is close, since the next layer is about a millisecond away, and then
  // poll gently because a job can be queued long before its command buffer runs, returning
  // false only on close().
  bool wait_gpu(uint64_t v) {
    uint64_t t0 = now_ns();
    while (ev_.value() < v) {
      if (now_ns() - t0 > 5000000) {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
        std::lock_guard lk(m_);
        if (stop_)
          return false;
      }
    }
    std::atomic_thread_fence(std::memory_order_acquire);
    return true;
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

  // The thread is going with jobs it may not have answered, so let every GPU wait through
  // instead of spinning to its bound and keep the reason.
  void die(const std::string& why) {
    release_all();
    std::lock_guard lk(m_);
    if (error_.empty())
      error_ = "agent thread gone: " + why;
    dead_ = true;
    busy_ = false;
    idle_cv_.notify_all();
  }

  void serve() {
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    for (;;) {
      Job j;
      {
        std::unique_lock lk(m_);
        cv_.wait(lk, [&] { return stop_ || !q_.empty(); });
        if (q_.empty())
          return;
        j = std::move(q_.front());
        q_.pop_front();
        busy_ = true;
      }
      if (!wait_gpu(j.v)) {
        die("closed before the GPU reached a job");
        return;
      }
      uint64_t t0 = now_ns(), read_ns = 0;
      int hits = 0, misses = 0;
      bool staged = false;
      std::string err;
      {
        std::lock_guard lk(m_);
        err = error_;
      }
      if (!perm_.empty())
        drop_guesses();
      // MOE_CPU_GUESS: the helper ranks the next layer's guess while this layer is claimed and read.
      bool ranking = j.h && err.empty() && !perm_.empty();
      if (ranking)
        start_guess(j);
      if (err.empty() && j.ids) {
        try {
          auto& L = live(j.layer);
          std::vector<int> ids(j.ids, j.ids + j.n);
          staged = !perm_.empty() && L.ring >= 0;
          auto miss = L.ring >= 0 ? claim_store(j.layer, L, std::move(ids))
                                  : claim(L, std::move(ids));
          misses = miss.size();
          hits = j.n - misses;
          if (staged) {
            uint64_t t = now_ns();
            if (staged_reads(j.layer, L, miss))
              err = "short read";
            read_ns = now_ns() - t;
            for (int k = 0; k < j.n; k++)
              L.phys_of[j.ids[k]] = perm_[L.slot_of[j.ids[k]]];
          } else if (misses) {
            uint64_t t = now_ns();
            if (read_wave(jobs_for(L, miss)))
              err = "short read";
            read_ns = now_ns() - t;
          }
        } catch (std::exception& e) {
          err = e.what();
        }
      }
      std::vector<int> guess;
      if (staged && err.empty())
        guess.assign(j.ids + j.n, j.ids + j.n + j.guess);
      // The bytes are in place or the error is kept, and the gather runs either way so the GPU
      // never hangs on a wait nobody will answer.
      ev_.signal(j.v + 1);
      uint64_t park = now_ns() - t0;
      if (ranking) {
        auto mine = wait_guess();
        if (!staged || !err.empty())
          mine.clear();
        else if (j.guess) { // MOE_CPU_GUESS=check: the GPU's guess is read, this one compared with it.
          cg_cmp_++;
          if (mine == guess)
            cg_same_++;
          else if (std::is_permutation(mine.begin(), mine.end(), guess.begin(), guess.end()))
            cg_set_++;
        } else {
          guess = std::move(mine);
        }
      }
      j.keep.reset();
      j.hkeep.reset();
      if (!guess.empty()) {
        stage_guesses(j.layer + 1, guess);
        pool_.poke(); // Wake the pool's side work when MOE_STAGING_READERS=pool is set, which does nothing otherwise.
        if (ext_)
          ext_->poke();
      }
      {
        std::lock_guard lk(m_);
        if (!err.empty() && error_.empty())
          error_ = err;
        n_layers_++;
        n_hits_ += hits;
        n_miss_ += misses;
        n_miss_layers_ += misses > 0;
        park_ns_ += park;
        read_ns_ += read_ns;
        busy_ = false;
        if (idle_locked())
          idle_cv_.notify_all();
      }
    }
  }

  SharedEvent& ev_;
  int fd_;
  Pool pool_;
  std::vector<Layer> layers_;
  std::vector<Ring> rings_;
  std::thread th_;
  std::mutex m_;
  std::condition_variable cv_, idle_cv_;
  std::deque<Job> q_;
  bool stop_ = false, busy_ = false, dead_ = false;
  std::string error_;
  uint64_t n_layers_ = 0, n_hits_ = 0, n_miss_ = 0, n_miss_layers_ = 0;
  uint64_t park_ns_ = 0, read_ns_ = 0;
  // The sidecar state of set_mirror, used only by read_wave on the agent thread.
  static constexpr double kDecay = 0.98; // The per-wave decay of stream_fast._STRIPE_DECAY.
  int mfd_ = -1;
  std::unordered_map<int64_t, std::pair<int64_t, int64_t>> mir_;
  double ema_[2][2] = {{0, 1}, {0, 1}}; // Experts.bin then sidecar, each as bytes and seconds.
  std::atomic<bool> mir_dead_{false};
  std::unique_ptr<Pool> ext_;
  std::atomic<uint64_t> int_b_{0}, ext_b_{0}, int_ns_{0}, ext_ns_{0}, fell_{0};
  // Staging rows of set_staging, all under m_: the store's logical to physical row map, the
  // staging rows and their guesses, the records not started as stage and projection, the
  // records in flight, the guess threads and store_, the first layer on the store.
  std::vector<int> perm_;
  std::vector<Stage> stage_;
  std::deque<std::pair<int, int>> gq_;
  int g_flight_ = 0, store_ = -1;
  int gdrive_[2] = {0, 0};   // Guess records in flight on each drive.
  int fault_every_ = 0, fault_us_ = 0, fault_n_ = 0; // The MOE_STAGING_FAULT state, for tests.
  uint64_t s_bad_ = 0;
  int gfd_ = -1, gmfd_ = -1; // The guess reads' experts.bin and sidecar descriptors.
  std::vector<std::thread> gth_;
  std::condition_variable gcv_, scv_;
  uint64_t s_guess_ = 0, s_rec_ = 0, s_prom_ = 0, s_part_ = 0, s_wrong_ = 0, s_wait_ns_ = 0,
           s_drop_ = 0, s_unstaged_ = 0;
  // MOE_CPU_GUESS: the routers by layer (set_router), the helper thread and its one task under
  // hm_ (the job, its bias mask, the guess and whether it is ready), and cpu_guess_stats' counters.
  std::vector<Router> routers_;
  std::thread hth_;
  std::mutex hm_;
  std::condition_variable hcv_;
  const Job* hjob_ = nullptr;
  std::vector<uint8_t> hmask_;
  std::vector<int> hout_;
  bool hany_ = false, hdone_ = true, hstop_ = false;
  std::atomic<uint64_t> cg_n_{0}, cg_ns_{0}, cg_max_{0}, cg_skip_{0}, cg_cmp_{0}, cg_same_{0},
      cg_set_{0};
};

// ---------------------------------------------------------------------------
// Graph ops that do what mlx::core::Fence does under MLX_METAL_FAST_SYNCH, with our word
// instead of the fence's.

static void encode_signal(mx::Stream s, SharedEvent& ev, uint64_t v, const mx::array& x,
                          const mx::array* h = nullptr) {
  auto& enc = mx::metal::get_command_encoder(s);
  auto& d = mx::metal::device(s.device);
  // Make x (and h, MOE_CPU_GUESS's MoE input) visible to the CPU, then publish v.
  uint32_t val = v;
  for (const mx::array* a : {&x, h}) {
    if (!a)
      continue;
    uint32_t n = (a->data_size() * a->itemsize() + 3) / 4;
    enc.set_compute_pipeline_state(d.get_kernel("input_coherent"));
    enc.set_input_array(*a, 0);
    enc.set_bytes(n, 1);
    enc.dispatch_threadgroups(MTL::Size((n + 1023) / 1024, 1, 1), MTL::Size(1024, 1, 1));
  }
  enc.barrier();
  enc.set_compute_pipeline_state(d.get_kernel("fence_update"));
  enc.set_buffer(ev.flag.get(), 0);
  enc.set_bytes(val, 1);
  enc.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
}

// Spin until ev reaches v or a stop word, then copy
// `in`, written by the host, into `out` with
// coherent_copy, whose loads use system coherence
// because a plain read after the spin can see stale
// slot lines the GPU still holds, and the copy only
// reads because input_coherent writes the words back
// and that races the agent's next claim on the same
// table.
static void encode_wait(mx::Stream s, SharedEvent& ev, uint64_t v, const mx::array& in,
                        mx::array& out) {
  auto& enc = mx::metal::get_command_encoder(s);
  uint32_t val = v, n = in.nbytes() / 4, cap = wait_bound().cap;
  g_last_word.store(ev.word());
  g_last_wait.store(val);
  enc.set_compute_pipeline_state(bounded_wait_kernel());
  enc.set_buffer(ev.flag.get(), 0);
  enc.set_buffer(stop_buffer(), 1);
  enc.set_bytes(val, 2);
  enc.set_bytes(cap, 3);
  enc.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
  enc.barrier();
  enc.set_compute_pipeline_state(coherent_copy_kernel());
  enc.set_input_array(in, 0);
  enc.set_output_array(out, 1); // Later readers wait for the copy.
  enc.set_bytes(n, 2);
  enc.dispatch_threads(MTL::Size(n, 1, 1), MTL::Size(std::min(n, 1024u), 1, 1));
}

class SignalAfter : public mx::Primitive {
 public:
  SignalAfter(mx::Stream s, SharedEvent* ev, uint64_t v, Agent* agent, int layer, int guess,
              float lam, float tau)
      : mx::Primitive(s), ev_(ev), v_(v), agent_(agent), layer_(layer), guess_(guess), lam_(lam),
        tau_(tau) {}
  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error("signal_after needs a GPU stream");
  }
  void eval_gpu(const std::vector<mx::array>& in, std::vector<mx::array>& out) override {
    out[0].copy_shared_buffer(in[0]);
    const mx::array* h = in.size() > 1 ? &in[1] : nullptr;
    if (agent_)
      agent_->push(layer_, v_, in[0], guess_, h, lam_, tau_);
    encode_signal(stream(), *ev_, v_, in[0], h);
  }
  const char* name() const override {
    return "SignalAfter";
  }

 private:
  SharedEvent* ev_;
  uint64_t v_;
  Agent* agent_;
  int layer_, guess_;
  float lam_, tau_;
};

class WaitFor : public mx::Primitive {
 public:
  WaitFor(mx::Stream s, SharedEvent* ev, uint64_t v) : mx::Primitive(s), ev_(ev), v_(v) {}
  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error("wait_for needs a GPU stream");
  }
  void eval_gpu(const std::vector<mx::array>& in, std::vector<mx::array>& out) override {
    if (in[0].nbytes() % 4 || !in[0].flags().row_contiguous)
      throw std::invalid_argument("wait_for: x must be contiguous 32-bit words");
    out[0].set_data(mx::allocator::malloc(out[0].nbytes()));
    encode_wait(stream(), *ev_, v_, in[0], out[0]);
  }
  const char* name() const override {
    return "WaitFor";
  }

 private:
  SharedEvent* ev_;
  uint64_t v_;
};

static mx::array signal_after(const mx::array& x, SharedEvent& ev, uint64_t v, Agent* agent,
                              int layer, int guess, std::optional<mx::array> h, float lam,
                              float tau, mx::StreamOrDevice s) {
  std::vector<mx::array> in{x};
  if (h)
    in.push_back(*h);
  return mx::array(x.shape(), x.dtype(),
                   std::make_shared<SignalAfter>(mx::to_stream(s), &ev, v, agent, layer, guess,
                                                 lam, tau),
                   std::move(in));
}

static mx::array wait_for(const mx::array& x, const mx::array& dep, SharedEvent& ev,
                          uint64_t v, mx::StreamOrDevice s) {
  return mx::array(x.shape(), x.dtype(),
                   std::make_shared<WaitFor>(mx::to_stream(s), &ev, v), {x, dep});
}

NB_MODULE(moe_native, m) {
  hook_exit_signals();
  m.def("now_ns", &now_ns);
  m.def("release", &release_all);
  m.def("stopped", &stopped);
  m.def("pending", &wait_pending);
  m.def("wait_bound", [] {
    auto& b = wait_bound();
    return std::make_tuple(b.cap, b.rate, b.budget);
  });
  nb::class_<SharedEvent>(m, "SharedEvent")
      .def(nb::init<>())
      .def_prop_ro("value", &SharedEvent::value)
      .def("signal", &SharedEvent::signal, "v"_a);
  nb::class_<Agent>(m, "Agent")
      .def(nb::init<SharedEvent&, int, int, int>(), "ev"_a, "fd"_a, "layers"_a,
           "readers"_a = 8, nb::keep_alive<1, 2>())
      .def("set_layer", &Agent::set_layer, "i"_a, "slot_of"_a, "projs"_a,
           "quota"_a = std::vector<int>{}, "ring"_a = -1, "phys"_a = nb::none())
      .def("set_staging", &Agent::set_staging, "slots"_a, "rows"_a)
      .def("unstage", &Agent::unstage)
      .def("stage_stats", &Agent::stage_stats)
      .def("set_router", &Agent::set_router, "i"_a, "w"_a, "scales"_a, "biases"_a,
           "group_size"_a, "k"_a)
      .def("cpu_guess_stats", &Agent::cpu_guess_stats)
      .def("set_perm", &Agent::set_perm, "perm"_a)
      .def("clear_layer", &Agent::clear_layer, "i"_a)
      .def("set_state", &Agent::set_state, "i"_a, "lru"_a,
           "ring"_a = std::vector<std::pair<int, int>>{})
      .def("set_ring", &Agent::set_ring, "r"_a, "base"_a, "next"_a, "owner"_a)
      .def("ring_state", &Agent::ring_state, "r"_a)
      .def("take_log", &Agent::take_log, "i"_a)
      .def("lru", &Agent::lru, "i"_a)
      .def("take_seen", &Agent::take_seen, "i"_a)
      .def("stats", &Agent::stats)
      .def("set_mirror", &Agent::set_mirror, "fd"_a, "records"_a, "bw_int"_a, "bw_ext"_a,
           "readers"_a)
      .def("mirror_stats", &Agent::mirror_stats)
      .def_prop_ro("error", &Agent::error)
      .def("drain", &Agent::drain, "timeout_s"_a = 30.0,
           nb::call_guard<nb::gil_scoped_release>())
      .def("close", &Agent::close, nb::call_guard<nb::gil_scoped_release>());
  m.def("signal_after", &signal_after, "x"_a, "ev"_a, "v"_a, "agent"_a.none(), "layer"_a,
        nb::kw_only(), "guess"_a = 0, "h"_a = nb::none(), "lam"_a = 0.0f, "tau"_a = 0.0f,
        "stream"_a = nb::none());
  m.def("wait_for", &wait_for, "x"_a, "dep"_a, "ev"_a, "v"_a, nb::kw_only(),
        "stream"_a = nb::none());
}
