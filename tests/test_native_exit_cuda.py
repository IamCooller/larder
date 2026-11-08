#!/usr/bin/env python3
"""MOE_NATIVE=1 on MLX's CUDA backend (native/moe_pipe_cuda) cannot leave the GPU spinning after the process is
gone: the Metal exit test's cases on an NVIDIA PC, plus a lone KILL. Each case is a child process; the parent
makes no CUDA context of its own (it reads the GPU through nvidia-smi), so nothing but the child's own exit can
end the child's GPU work.

  bound   a wait nobody answers (no agent) gives up after MOE_NATIVE_WAIT_S (2 s here); the give-up sticks, a
          later wait falls through at once, and a native pass of the toy model then fails loudly
  dead    the agent closes with a job the GPU has not reached (parked behind a 1.5 s kernel): every wait is
          released, the job's wait falls through when the GPU gets there, not after the bound (10 s)
  plain   the baseline: SIGTERM while the toy model (test_store's packed MoE) decodes with the pipeline off.
          nvidia-smi's utilization is an average over its last sample period (up to 1 s), so after a decode it
          reads busy for up to ~1.6 s whatever ended the work (0.2-1.1 s here); this case records that tail
  term    SIGTERM while the toy model decodes natively: from the parent twice, and once from the child itself
          during a park (every job held 0.3 s, MOE_PIPE_SLOW_US): the handler releases, the process dies of the
          signal at once, and the GPU reads idle within 2 s of the exit (an orphan reads 100% for good)
  kill    SIGKILL from the child itself during a 0.3 s park, three times: the GPU is left busy, moe_pipe_cuda's
          guard sees it busy for 2.1 s and makes a CUDA context, and the GPU is idle within the bound (5 s)
  noguard the control: the same KILL without the guard (MOE_PIPE_GUARD=0), i.e. the wait as the design had it.
          Expected to leave the GPU busy; the parent then ends it by making a CUDA context
          (as the next process would) and reports how long it read busy
  beat    the opt-in alternative: the same KILL without the guard, with MOE_PIPE_BEAT_MS=2 (the wait gives up 2 ms
          after the agent's heartbeat stops), twice: the GPU is idle within the bound
  soak    (named only) two minutes of native decode, and no wait may give up; with MOE_PIPE_BEAT_MS set in the
          environment, the heartbeat's cost (a wait that gives up while the process lives fails that pass)

    python3 tests/test_native_exit_cuda.py [--json OUT] [bound dead plain term kill noguard beat soak]
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
ARGS = sys.argv[1:]
CASE = ARGS.pop(ARGS.index("--case") + 1) if "--case" in ARGS else None
OUT = ARGS.pop(ARGS.index("--json") + 1) if "--json" in ARGS else None
ALL = ["bound", "dead", "plain", "term", "kill", "noguard", "beat", "soak"]
NAMES = [a for a in ARGS if a in ALL] or ALL[:-1]
IDLE_TERM = 2.0        # s after a TERM'd exit: the counter's tail, not a spin
SLOW = 300000          # us a job is held in the park cases


def nv():
    sys.path.insert(0, str(HERE.parent / "native"))
    import moe_pipe_cuda
    return moe_pipe_cuda


def toy_model(out, native=True):
    import stream_fast
    import test_store as ts
    ts.build(out)
    model = ts.stock_model(out)
    stream_fast.install(model, out, resident=ts.RES, cache_bias=0.0, defer=False, native=native)
    assert not native or (stream_fast._NATIVE is not None and stream_fast._NATIVE.cuda), \
        "native did not install on CUDA"
    return model, ts.PROMPTS[0]


# ---------------------------------------------------------------- the children

def case_bound():
    m = nv()
    import mlx.core as mx
    print(f"bound: {m.wait_bound()[2]:g} s", flush=True)
    ids, tab = mx.array([[[1, 2, 3, 4]]], dtype=mx.uint32), mx.zeros(8, dtype=mx.int32)
    mx.eval(ids, tab)
    m.warm(4)
    t0 = time.time()
    mx.eval(m.wait_rows(m.signal(ids, 0), ids, tab))
    dt = time.time() - t0
    st = m.stopped()
    print(f"bound: the unanswered wait returned after {dt:.2f} s, stopped={st}", flush=True)
    assert st[0] == 0 and st[1] > 0, st
    assert 0.5 * m.wait_bound()[2] < dt < 2 * m.wait_bound()[2], dt
    t0 = time.time()
    mx.eval(m.wait_rows(m.signal(ids, 0), ids, tab))
    dt = time.time() - t0
    print(f"bound: a later wait fell through in {dt * 1e3:.1f} ms", flush=True)
    assert dt < 0.5, dt
    from mlx_lm.generate import generate_step
    with tempfile.TemporaryDirectory() as d:
        model, prompt = toy_model(Path(d) / "toy")
        try:
            for _ in generate_step(mx.array(prompt, dtype=mx.int32), model, max_tokens=6):
                pass
        except RuntimeError as e:
            print(f"bound: the native pass failed loudly: {e}", flush=True)
            assert "gave up" in str(e), e
        else:
            raise AssertionError("native decode ran on after a wait gave up")
    print("ok bound", flush=True)


DELAY = r"""
  if (threadIdx.x == 0) {
    unsigned long long t0, t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); } while (t - t0 < 1500000000ull);
  }
  __syncthreads();
  if (threadIdx.x < 4) out[threadIdx.x] = ids[threadIdx.x];
"""


def case_dead():
    m = nv()
    import mlx.core as mx
    ids, tab = mx.array([[[1, 2, 3, 4]]], dtype=mx.uint32), mx.zeros(8, dtype=mx.int32)
    mx.eval(ids, tab)
    m.warm(4)
    delay = mx.fast.cuda_kernel(name="exit_delay", input_names=["ids"], output_names=["out"], source=DELAY)
    with open(os.devnull, "rb") as f:
        agent = m.Agent(m.SharedEvent(), f.fileno(), 1, 1)
        late = delay(inputs=[ids], grid=(32, 1, 1), threadgroup=(32, 1, 1), output_shapes=[ids.shape],
                     output_dtypes=[mx.uint32])[0]                  # the GPU parks here for 1.5 s
        y = m.wait_rows(m.signal(late, 0), late, tab)
        t0 = time.time()
        mx.async_eval(y)
        m.committed()
        time.sleep(0.5)
        agent.close()                       # its thread goes, the job unanswered
        mx.eval(y)
        dt = time.time() - t0
    print(f"dead: the wait done {dt:.2f} s after the commit (the GPU got there at 1.5 s), "
          f"stopped={m.stopped()}, error={agent.error!r}", flush=True)
    assert m.stopped()[0] == 1, m.stopped()
    assert "closed before the GPU reached a job" in agent.error, agent.error
    assert dt < 3.0, dt
    print("ok dead", flush=True)


def stop_in_park(m, sig):
    """Once a job is committed and unanswered for 0.1 s (the GPU is in its wait), send `sig` to ourselves."""
    while True:
        while not m.pending():
            time.sleep(0.0005)
        t = time.time()
        while m.pending() and time.time() - t < 0.1:
            time.sleep(0.001)
        if m.pending():
            print(f"stopping with {signal.Signals(sig).name} in a park", flush=True)
            os.kill(os.getpid(), sig)
            time.sleep(10)


def case_decode(self_sig=None, native=True):
    m = nv()
    import mlx.core as mx
    from mlx_lm.generate import generate_step
    import stream_fast
    with tempfile.TemporaryDirectory() as d:
        model, prompt = toy_model(Path(d) / "toy", native)
        for n, _ in enumerate(generate_step(mx.array(prompt, dtype=mx.int32), model, max_tokens=100000)):
            ran = stream_fast.TIMERS["native_park"] > 0 or not native
            if n % 5 == 0:
                print(f"decoding {n} native={int(ran)} beat gaps {m.beat_gaps()}", flush=True)
            if self_sig and ran and n == 3:
                threading.Thread(target=stop_in_park, args=(m, self_sig), daemon=True).start()
            if n > 2000:
                break


def case_soak(seconds=120.0):
    m = nv()
    import mlx.core as mx
    from mlx_lm.generate import generate_step
    from mlx_lm.models.cache import make_prompt_cache
    import stream_fast
    import test_store as ts
    t0, tokens, turns = time.time(), 0, 0
    with tempfile.TemporaryDirectory() as d:
        model, _ = toy_model(Path(d) / "toy")
        while time.time() - t0 < seconds:
            # A fresh conversation each turn: the toy's context stays short.
            for _ in generate_step(mx.array(ts.PROMPTS[turns % 4], dtype=mx.int32), model, max_tokens=500,
                                   prompt_cache=make_prompt_cache(model)):
                tokens += 1
            turns += 1
            assert stream_fast._NATIVE is not None, "native went off"
    n = stream_fast.STATS["reads"]
    print(f"soak: beat gaps (GPU max us, thread max us, thread gaps > 1 ms) {m.beat_gaps()}", flush=True)
    print(f"soak: {tokens} tokens in {turns} turns, {n} native layers, {time.time() - t0:.0f} s, "
          f"stopped={m.stopped()}, beat limit {os.environ.get('MOE_PIPE_BEAT_MS', '2')} ms", flush=True)
    assert m.stopped() == (0, 0), m.stopped()
    print("ok soak", flush=True)


# ---------------------------------------------------------------- the parent

def smi():
    out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,power.draw", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    u, p = out.split(", ")
    return int(u), float(p)


def nudge():
    """Make a CUDA context in a fresh process and leave: what ends a dead process's stuck channel."""
    code = ("import ctypes; cu = ctypes.CDLL('libcuda.so.1'); d, c = ctypes.c_int(), ctypes.c_void_p(); "
            "cu.cuInit(0); cu.cuDeviceGet(ctypes.byref(d), 0); cu.cuDevicePrimaryCtxRetain(ctypes.byref(c), d); "
            "cu.cuCtxSetCurrent(c); cu.cuCtxSynchronize(); cu.cuDevicePrimaryCtxRelease_v2(d)")
    subprocess.run([sys.executable, "-c", code], timeout=30)


def run_case(name, env, until=None, term_after=None, watch=6.0):
    env = {**os.environ, "PYTHONUNBUFFERED": "1", **env}
    p = subprocess.Popen([sys.executable, __file__, "--case", name], env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True)
    lines, t_sig = [], None
    killer = threading.Timer(240, p.kill)
    killer.start()
    for line in p.stdout:
        lines.append(line.rstrip())
        print(f"  {name}| {line.rstrip()}", flush=True)
        if term_after is not None and t_sig is None and line.startswith("decoding") and "native=1" in line \
                and int(line.split()[1]) >= 20:
            time.sleep(term_after)
            t_sig = time.time()
            p.send_signal(signal.SIGTERM)
    rc = p.wait()
    killer.cancel()
    t_exit = time.time()
    util = []
    while time.time() - t_exit < watch:
        util.append((round(time.time() - t_exit, 2),) + smi())
        if until is not None and util[-1][1] == 0 and len(util) > 3 and all(u[1] == 0 for u in util[-3:]):
            break
        time.sleep(0.1)
    busy = [t for t, u, _ in util if u > 0]
    idle_at = None if (busy and busy[-1] == util[-1][0]) else (busy[-1] + 0.1 if busy else 0.0)
    nudged = None
    if idle_at is None:
        t = time.time()
        nudge()
        after = []
        for _ in range(10):
            after.append(smi())
            time.sleep(0.1)
        nudged = {"s": round(time.time() - t, 2), "util_after": after}
    r = {"case": name, "rc": rc, "env": {k: v for k, v in env.items() if k.startswith("MOE_")},
         "exit_after_signal_s": None if t_sig is None else round(t_exit - t_sig, 3),
         "idle_after_exit_s": idle_at, "nudged": nudged, "util": util, "said": lines[-6:]}
    print(f"  {name}: rc={rc}, GPU idle {'never (nudged)' if idle_at is None else f'{idle_at:.2f} s'} "
          f"after the exit" + ("" if t_sig is None else f", exited {r['exit_after_signal_s']} s after TERM"),
          flush=True)
    return r, lines


def main():
    if shutil.which("nvidia-smi") is None:
        print("test_native_exit_cuda: skipped (needs an NVIDIA GPU and nvidia-smi)")
        return
    res = []
    try:
        cases(res)
    finally:
        if OUT:
            json.dump(res, open(OUT, "w"), indent=1)


def cases(res):
    u0 = [smi() for _ in range(5)]
    assert all(u == 0 for u, _ in u0), f"the GPU is not idle: {u0}"
    if "bound" in NAMES:
        r, lines = run_case("bound", {"MOE_NATIVE": "1", "MOE_NATIVE_WAIT_S": "2"}, until=0)
        res.append(r)
        assert r["rc"] == 0 and "ok bound" in lines, "bound failed: no kill test before it passes"
    if "dead" in NAMES:
        r, lines = run_case("dead", {"MOE_NATIVE": "1", "MOE_NATIVE_WAIT_S": "10"}, until=0)
        res.append(r)
        assert r["rc"] == 0 and "ok dead" in lines, "dead failed"
    tail = None
    if "plain" in NAMES:
        r, lines = run_case("plain", {"MOE_NATIVE": "0"}, until=0, term_after=0.0)
        res.append(r)
        assert r["rc"] == -signal.SIGTERM and r["idle_after_exit_s"] is not None, r["rc"]
        tail = r["idle_after_exit_s"]
    if "term" in NAMES:
        for after, name, extra in ((0.0, "term", {}), (0.37, "term", {}),
                                   (None, "term-self", {"MOE_PIPE_SLOW_US": str(SLOW)})):
            r, lines = run_case(name, {"MOE_NATIVE": "1", **extra}, until=0, term_after=after)
            res.append(r)
            assert r["rc"] == -signal.SIGTERM, f"{name}: rc {r['rc']}"
            assert any("GPU waits released" in l for l in lines), f"{name}: no release on SIGTERM"
            assert r["idle_after_exit_s"] is not None and r["idle_after_exit_s"] <= IDLE_TERM, \
                (r["idle_after_exit_s"], tail)
            assert after is None or r["exit_after_signal_s"] < 2.0
        assert any("(one was pending)" in l for r in res if r["case"].startswith("term") for l in r["said"]), \
            "no TERM landed with a wait pending: proved nothing"
    if "kill" in NAMES:
        for _ in range(3):
            r, lines = run_case("kill", {"MOE_NATIVE": "1", "MOE_PIPE_SLOW_US": str(SLOW)}, until=0)
            res.append(r)
            assert r["rc"] == -signal.SIGKILL, r["rc"]
            assert any("in a park" in l for l in lines), "the KILL did not land in a park"
            assert r["idle_after_exit_s"] is not None and r["idle_after_exit_s"] < 5.0, r["idle_after_exit_s"]
    if "noguard" in NAMES:
        r, lines = run_case("noguard", {"MOE_NATIVE": "1", "MOE_PIPE_SLOW_US": str(SLOW), "MOE_PIPE_GUARD": "0",
                                        "MOE_NATIVE_WAIT_S": "60"}, watch=8.0)
        res.append(r)
        assert r["rc"] == -signal.SIGKILL, r["rc"]
        print(f"  noguard (the design's wait): GPU busy "
              f"{'until nudged' if r['idle_after_exit_s'] is None else 'ended by itself'}", flush=True)
    if "beat" in NAMES:
        for _ in range(2):
            r, lines = run_case("beat", {"MOE_NATIVE": "1", "MOE_PIPE_SLOW_US": str(SLOW), "MOE_PIPE_GUARD": "0",
                                         "MOE_PIPE_BEAT_MS": "2"}, until=0)
            res.append(r)
            assert r["rc"] == -signal.SIGKILL, r["rc"]
            assert any("in a park" in l for l in lines), "the KILL did not land in a park"
            assert r["idle_after_exit_s"] is not None and r["idle_after_exit_s"] < 5.0, r["idle_after_exit_s"]
    if "soak" in NAMES:
        r, lines = run_case("soak", {"MOE_NATIVE": "1"}, until=0)
        res.append(r)
        assert r["rc"] == 0 and "ok soak" in lines, "soak failed"
    print("ok")


if __name__ == "__main__":
    if CASE:
        {"bound": case_bound, "dead": case_dead, "term": case_decode, "plain": lambda: case_decode(native=False),
         "term-self": lambda: case_decode(signal.SIGTERM), "kill": lambda: case_decode(signal.SIGKILL),
         "noguard": lambda: case_decode(signal.SIGKILL), "beat": lambda: case_decode(signal.SIGKILL),
         "soak": case_soak}[CASE]()
    else:
        main()
