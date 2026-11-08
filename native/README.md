# moe_native: the native expert agent (`MOE_NATIVE=1`)

**Stopping.** The GPU waits through `bounded_wait` (gives up after `MOE_NATIVE_WAIT_S`, default 5 s),
and TERM/SIGHUP, atexit, `os._exit` and an agent-thread death release every wait first;
without that, a TERM mid-generation could hang the GPU until a reboot. `kill -9` still holds the GPU
for up to the bound.

A C++ MLX extension behind `stream_fast._Native`: one-position decode passes hand each
MoE layer's router ids to a host thread that claims arena slots and reads the misses,
with no Python between a layer's router and its gather. The slot table is copied coherently in
`wait_for`. A layer is a per-layer arena or, under `--mem-peak-gb`, a quota of the shared store
plus a ring half (Qwen3.8-Flash-Next).

## CUDA (`native/moe_pipe_cuda.cpp`, `moe_pipe_cuda.py`)

On MLX's CUDA backend `MOE_NATIVE=1` loads `moe_pipe_cuda` instead: a C ABI library through
ctypes, built by g++ at first use (no build step, no MLX headers, no tie to the MLX version), with moe_native's API.
The handshake is two `mx.fast.cuda_kernel`s (signal posts the ids, wait spins on mapped memory with a `%globaltimer`
bound and copies the records the agent preads into mapped staging). `MOE_PIPE_GUARD=0`: no guard process (it ends a
KILLed engine's busy GPU); `MOE_PIPE_BEAT_MS=2`: the opt-in heartbeat. Tests: `tests/test_native.py` (bitwise),
`tests/test_native_exit_cuda.py`. Its read path: `MOE_PIPE_DIRECT=0` turns O_DIRECT off,
`MOE_PIPE_CHUNK` sets the copy chunk, `MOE_PIPE_TIER_GB` the RAM victim tier (default a third of MemAvailable above
2.5 GB at install; 0 = off). `MOE_PIPE_STAGING=1` (off by default): each layer posts the next layer's gate top-k and
the agent reads its misses into host guess records ahead, `MOE_PIPE_STAGING_CAP` (2) in flight
(`tests/test_pipe_staging_emu.py`), `MOE_PIPE_GUESS_D=2` the guess two layers ahead. Also off by default:
`MOE_PIPE_PIECES=1` reads each record in 128 KB pieces on io_uring with a ready word each, so the
wait kernel copies a record while the rest of it is read; `MOE_PIPE_HITS_FIRST=1` (EXL3 packs) runs a layer's held
experts inside its park and the missed ones after the wait.

## Build

    native/build.sh <path-to-python>      # e.g. native/build.sh .venv/bin/python

It refuses any mlx other than 0.32.2, installs `nanobind==2.15.0` and `cmake` into that
env if they are missing (another nanobind builds, but does not share the `mx.array`
type with mlx), builds in a temporary directory and copies `moe_native*.so` into the
env's site-packages. The last step imports it from there and builds one graph op.
Rerun it after any change to `moe_native.cpp` and after any mlx upgrade.

## How the engine finds it

`stream_fast.install_native` (called when `MOE_NATIVE=1`) does `import moe_native`
with `native/build` appended to `sys.path`:

1. the copy in the running env's site-packages (what `build.sh` installs) wins;
2. otherwise a manual build in this tree's `native/build`
   (`cmake -S native -B native/build -DPython_EXECUTABLE=...`, see `CMakeLists.txt`).

Neither: `MOE_NATIVE=1` stops with an ImportError naming the `build.sh` command. An
extension built for another env must not be loaded: it would bring that env's libmlx.
A stale copy in site-packages also shadows a newer `native/build`, so rebuild with
`build.sh`.

## Tests

    python3 tests/test_native_coherent.py   # extension only: slot table seen after the host's writes
    python3 tests/test_native.py            # toy model: native decode == Python, bitwise (arena and store)
