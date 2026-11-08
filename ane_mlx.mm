// Neural Engine programs on IOSurfaces that MLX arrays alias with no copy, run in MLX's graph (MOE_ANE_SPLIT, ane_split.py).
// Build: native/build.sh (clang++, MLX's headers and libmlx, nanobind in MLX's domain).
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>

#include <atomic>
#include <memory>

#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "mlx/allocator.h"
#include "mlx/array.h"
#include "mlx/backend/metal/device.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace nb = nanobind;
namespace mx = mlx::core;

static Class gDesc, gModel, gReq, gIO;

static double now_ms() {
  static mach_timebase_info_data_t tb;
  if (!tb.denom) mach_timebase_info(&tb);
  return (double)mach_absolute_time() * tb.numer / tb.denom / 1e6;
}

static void init_classes() {
  if (gDesc) return;
  if (!dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW))
    throw std::runtime_error("AppleNeuralEngine.framework not found");
  gDesc = NSClassFromString(@"_ANEInMemoryModelDescriptor");
  gModel = NSClassFromString(@"_ANEInMemoryModel");
  gReq = NSClassFromString(@"_ANERequest");
  gIO = NSClassFromString(@"_ANEIOSurfaceObject");
  if (!(gDesc && gModel && gReq && gIO)) throw std::runtime_error("ANE classes missing");
}

static IOSurfaceRef make_surface(size_t bytes) {
  size_t alloc = (bytes + 16383) / 16384 * 16384;  // whole pages, so Metal can wrap the base with no copy
  return IOSurfaceCreate((__bridge CFDictionaryRef) @{
    (id)kIOSurfaceWidth : @(bytes), (id)kIOSurfaceHeight : @1, (id)kIOSurfaceBytesPerElement : @1,
    (id)kIOSurfaceBytesPerRow : @(bytes), (id)kIOSurfaceAllocSize : @(alloc), (id)kIOSurfacePixelFormat : @0});
}

// MLX's open command buffer on stream s, its compute encoder ended so a command can go between encoders.
static id<MTLCommandBuffer> command_buffer(mx::Stream s) {
  auto& enc = mx::metal::get_command_encoder(s);
  enc.end_encoding();
  auto* cb = enc.get_command_buffer();
  if (!cb) throw std::runtime_error("no open MLX command buffer");
  return (__bridge id<MTLCommandBuffer>)(void*)cb;
}

// The loaded program, its surfaces and the two events that order it against MLX's GPU stream.
struct Core {
  id model = nil, request = nil;
  IOSurfaceRef in = nullptr, out = nullptr;
  std::string tmpdir;
  id<MTLSharedEvent> ready = nil, done = nil;  // ready: the GPU wrote the input; done: the ANE wrote the output
  uint64_t submitted = 0;
  dispatch_queue_t queue = nullptr;
  std::atomic<int> errors{0};
  std::atomic<double> ane_ms{0}, wait_ms{0};

  bool eval() {
    @autoreleasepool {
      NSError* e = nil;
      return ((BOOL(*)(id, SEL, unsigned int, id, id, NSError**))objc_msgSend)(
          model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, request, &e);
    }
  }

  // On the ANE's queue: wait for the GPU's input, run, release the GPU's wait (always, so the GPU never hangs).
  void host_run(uint64_t v) {
    double t0 = now_ms();
    BOOL ok = [ready waitUntilSignaledValue:v timeoutMS:30000];
    double t1 = now_ms();
    if (!ok || !model || !eval()) errors++;
    double t2 = now_ms();
    wait_ms = wait_ms + (t1 - t0);
    ane_ms = ane_ms + (t2 - t1);
    done.signaledValue = v;
  }

  // New input and output surfaces and the request that binds them (the loaded program is kept).
  void bind_io(size_t in_bytes, size_t out_bytes) {
    @autoreleasepool {
      release_io();
      in = make_surface(in_bytes);
      out = make_surface(out_bytes);
      id wi = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(gIO, @selector(objectWithIOSurface:), in);
      id wo = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(gIO, @selector(objectWithIOSurface:), out);
      request = ((id(*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(
          gReq, @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
          @[ wi ], @[ @0 ], @[ wo ], @[ @0 ], nil, nil, @0);
    }
  }

  // The surfaces go (freed once no MLX alias holds them); the program stays loaded.
  void release_io() {
    if (queue) dispatch_sync(queue, ^{});  // the last submitted run has finished
    request = nil;
    if (in) CFRelease(in);
    if (out) CFRelease(out);
    in = out = nullptr;
  }

  void close() {
    @autoreleasepool {
      release_io();
      if (model) {
        NSError* e = nil;
        ((BOOL(*)(id, SEL, unsigned int, NSError**))objc_msgSend)(model, @selector(unloadWithQoS:error:), 21, &e);
        model = nil;
      }
      if (!tmpdir.empty()) [[NSFileManager defaultManager] removeItemAtPath:@(tmpdir.c_str()) error:nil];
      tmpdir.clear();
    }
  }

  ~Core() { close(); }
};

// Signals `ready` after its inputs (the GPU's writes into the input surface) and queues the ANE run.
class AneSubmit : public mx::UnaryPrimitive {
 public:
  AneSubmit(mx::Stream s, std::shared_ptr<Core> c, uint64_t v) : mx::UnaryPrimitive(s), c_(c), v_(v) {}
  void eval_cpu(const std::vector<mx::array>&, mx::array&) override {
    throw std::runtime_error("AneSubmit: GPU stream only");
  }
  void eval_gpu(const std::vector<mx::array>&, mx::array& out) override {
    out.set_data(mx::allocator::malloc(out.nbytes()));
    [command_buffer(stream()) encodeSignalEvent:c_->ready value:v_];
    auto c = c_;
    uint64_t v = v_;
    dispatch_async(c->queue, ^{ c->host_run(v); });
  }
  const char* name() const override { return "AneSubmit"; }

 private:
  std::shared_ptr<Core> c_;
  uint64_t v_;
};

// The GPU waits for `done` at its run's value; the output is the output surface's alias, no copy.
class AneWait : public mx::UnaryPrimitive {
 public:
  AneWait(mx::Stream s, std::shared_ptr<Core> c, uint64_t v) : mx::UnaryPrimitive(s), c_(c), v_(v) {}
  void eval_cpu(const std::vector<mx::array>&, mx::array&) override {
    throw std::runtime_error("AneWait: GPU stream only");
  }
  void eval_gpu(const std::vector<mx::array>& in, mx::array& out) override {
    [command_buffer(stream()) encodeWaitForEvent:c_->done value:v_];
    out.copy_shared_buffer(in[0]);
  }
  const char* name() const override { return "AneWait"; }

 private:
  std::shared_ptr<Core> c_;
  uint64_t v_;
};

struct Program {
  std::shared_ptr<Core> c;
  double compile_ms = 0, load_ms = 0;
  std::string tmpdir;

  Program(nb::bytes mil, size_t in_bytes, size_t out_bytes) {
    init_classes();
    c = std::make_shared<Core>();
    @autoreleasepool {
      NSError* e = nil;
      NSData* milData = [NSData dataWithBytes:mil.c_str() length:mil.size()];
      id desc = ((id(*)(Class, SEL, id, id, id))objc_msgSend)(
          gDesc, @selector(modelWithMILText:weights:optionsPlist:), milData, @{}, nil);
      if (!desc) throw std::runtime_error("ANE descriptor failed");
      id mdl = ((id(*)(Class, SEL, id))objc_msgSend)(gModel, @selector(inMemoryModelWithDescriptor:), desc);
      if (!mdl) throw std::runtime_error("ANE model failed");
      NSString* hx = ((id(*)(id, SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
      NSString* td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
      [[NSFileManager defaultManager] createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
                                withIntermediateDirectories:YES attributes:nil error:nil];
      [milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
      c->tmpdir = tmpdir = td.UTF8String;
      double t0 = now_ms();
      BOOL ok = ((BOOL(*)(id, SEL, unsigned int, id, NSError**))objc_msgSend)(
          mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e);
      double t1 = now_ms();
      if (ok)
        ok = ((BOOL(*)(id, SEL, unsigned int, id, NSError**))objc_msgSend)(
            mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);
      double t2 = now_ms();
      if (!ok) {
        std::string msg = std::string("ANE compile/load: ") + (e ? e.description.UTF8String : "?");
        c->close();
        throw std::runtime_error(msg);
      }
      compile_ms = t1 - t0;
      load_ms = t2 - t1;
      c->model = mdl;
    }
    c->bind_io(in_bytes, out_bytes);
    id<MTLDevice> dev = (__bridge id<MTLDevice>)(void*)mx::metal::device(mx::Device::gpu).mtl_device();
    c->ready = [dev newSharedEvent];
    c->done = [dev newSharedEvent];
    c->queue = dispatch_queue_create("ane_mlx", DISPATCH_QUEUE_SERIAL);
  }

  // Fresh surfaces for a loaded program (after release()).
  void bind(size_t in_bytes, size_t out_bytes) {
    if (!c->model) throw std::runtime_error("ANE program closed");
    c->bind_io(in_bytes, out_bytes);
  }

  // The surfaces go, the program stays loaded (MOE_ANE_OVERLAP: kept across prompts, its surfaces only in a pass).
  void release() {
    @autoreleasepool {
      c->release_io();
    }
  }

  // Synchronous run on the surfaces; the caller has made the input's GPU writes complete.
  void run() {
    if (!c->model) throw std::runtime_error("ANE program closed");
    bool ok;
    {
      nb::gil_scoped_release nogil;
      ok = c->eval();
    }
    if (!ok) throw std::runtime_error("ANE eval failed");
  }

  // Lazy: a token whose evaluation (after deps) queues the ANE run; returns (token, run value) at once.
  std::pair<mx::array, uint64_t> submit(std::vector<mx::array> deps) {
    if (!c->model || !c->request) throw std::runtime_error("ANE program closed or its surfaces released");
    uint64_t v = ++c->submitted;
    auto s = mx::default_stream(mx::Device::gpu);
    return {mx::array({1}, mx::int32, std::make_shared<AneSubmit>(s, c, v), std::move(deps)), v};
  }

  // Lazy: y (an output alias view) once run v has written it; deps are evaluated before the wait.
  mx::array wait(const mx::array& y, uint64_t v, std::vector<mx::array> deps) {
    std::vector<mx::array> in{y};
    in.insert(in.end(), deps.begin(), deps.end());
    auto s = mx::default_stream(mx::Device::gpu);
    return mx::array(y.shape(), y.dtype(), std::make_shared<AneWait>(s, c, v), std::move(in));
  }

  // An fp16 MLX array over the input (which=0) or output (1) surface at a byte offset, sharing its memory.
  mx::array alias(int which, std::vector<int> shape, size_t offset) {
    IOSurfaceRef s = which ? c->out : c->in;
    if (!s) throw std::runtime_error("ANE program closed");
    size_t n = 2;
    for (int d : shape) n *= d;
    size_t alloc = IOSurfaceGetAllocSize(s);
    if (offset + n > alloc) throw std::runtime_error("alias past the surface");
    auto buf = mx::allocator::make_buffer(IOSurfaceGetBaseAddress(s), alloc);
    if (!buf.ptr()) throw std::runtime_error("Metal could not wrap the surface without a copy");
    CFRetain(s);
    mx::array a(buf, mx::Shape{(int)(alloc / 2)}, mx::float16, [s](mx::allocator::Buffer b) {
      mx::allocator::release(b);
      CFRelease(s);
    });
    auto flat = mx::slice(a, {(int)(offset / 2)}, {(int)((offset + n) / 2)});
    return mx::reshape(flat, mx::Shape(shape.begin(), shape.end()));
  }

  nb::dict stats() {
    nb::dict d;
    d["errors"] = c->errors.load();
    d["ane_ms"] = c->ane_ms.load();
    d["wait_ms"] = c->wait_ms.load();
    d["submitted"] = c->submitted;
    return d;
  }

  void close() { c->close(); }
};

NB_MODULE(_ane_mlx, m) {
  nb::class_<Program>(m, "Program")
      .def(nb::init<nb::bytes, size_t, size_t>(), nb::arg("mil"), nb::arg("in_bytes"), nb::arg("out_bytes"))
      .def("run", &Program::run)
      .def("submit", &Program::submit, nb::arg("deps"))
      .def("wait", &Program::wait, nb::arg("y"), nb::arg("v"), nb::arg("deps"))
      .def("alias", &Program::alias, nb::arg("which"), nb::arg("shape"), nb::arg("offset") = 0)
      .def("stats", &Program::stats)
      .def("close", &Program::close)
      .def("bind", &Program::bind, nb::arg("in_bytes"), nb::arg("out_bytes"))
      .def("release", &Program::release)
      .def_ro("compile_ms", &Program::compile_ms)
      .def_ro("load_ms", &Program::load_ms)
      .def_ro("tmpdir", &Program::tmpdir);
}
