#!/usr/bin/env python3
"""A MOE_NATIVE=1 process can no longer leave the GPU spinning after it is
gone. Each case runs in a child
process whose watchdog thread raises the words itself, so even a broken fix
cannot hang the GPU while the child lives:

  bound  a wait nobody answers (signal_after with no agent) gives up after
         MOE_NATIVE_WAIT_S (2 s here) on its own; the stop word sticks, a
         later wait falls through at once, and a native pass of the toy model
         then fails loudly instead of decoding on stale slots;
  dead   the agent's thread goes (close()) with a job the GPU has not reached
         yet: every wait is released at once, not after the bound (10 s);
  term   SIGTERM while the toy model decodes natively (test_store's packed
         MoE, bound 10 s): from the parent at three moments, and twice from
         the child itself the moment moe_native.pending() shows a wait not
         yet passed. moe_native's handler says whether one was pending, i.e.
         where the old engine hung the GPU; at least one TERM must land so.
         The process dies of the signal at once.

After each case the GPU (ioreg "Device Utilization %") is idle within 4 s of
the exit -- a wait left spinning would hold it for the 10 s bound -- and the
GPU's memory in use (ioreg "In use system memory": the 7.1 GB the hung
command buffer kept on 2026-10-11) is back where it started.

    python3 tests/test_native_exit.py [dir holding moe_native] [bound dead term]

Named cases run alone (bound first: it is what makes the others safe).
"""
import os
import re
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
ARGS = [a for a in ARGS if a != "--case"]
NAMES = [a for a in ARGS if a in ("bound", "dead", "term")] or ["bound", "dead", "term"]
NATIVE_DIR = next((a for a in ARGS if a not in NAMES), None)
IDLE_WITHIN = 4.0      # s after the exit; a wait left spinning holds the GPU for the bound


def native():
    if NATIVE_DIR:
        sys.path.insert(0, NATIVE_DIR)
    else:
        sys.path.append(str(HERE.parent / "native" / "build"))
    import moe_native
    return moe_native


# ---------------------------------------------------------------- the children

RESCUED = []


def watchdog(nv, events, after):
    """Raise the words ourselves after `after` s unless done is set first."""
    done = threading.Event()

    def run():
        if not done.wait(after):
            RESCUED.append(time.time())
            for ev in events:
                ev.signal(1 << 31)
            nv.release()
            print("watchdog: raised the words itself", flush=True)
    threading.Thread(target=run, daemon=True).start()
    return done


def toy_model(out):
    import stream_fast
    import test_store as ts
    ts.build(out)
    model = ts.stock_model(out)
    stream_fast.install(model, out, resident=ts.RES, cache_bias=0.0,
                        defer=False, native=True)
    assert stream_fast._NATIVE is not None, "native did not install"
    return model, ts.PROMPTS[0]


def case_bound():
    nv = native()
    import mlx.core as mx
    ev = nv.SharedEvent()
    cap, rate, budget = nv.wait_bound()
    print(f"bound: {budget:g} s = {cap} rounds at {rate / 1e6:.2f} M rounds/s", flush=True)
    done = watchdog(nv, [ev], 2.5 * budget)
    ids, table = mx.array([1, 2], dtype=mx.int32), mx.zeros(8, dtype=mx.int32)
    mx.eval(ids, table)
    t0 = time.time()
    mx.eval(nv.wait_for(table, nv.signal_after(ids, ev, 1, None, 0), ev, 2))
    dt = time.time() - t0
    print(f"bound: the unanswered wait returned after {dt:.2f} s, stopped={nv.stopped()}",
          flush=True)
    assert not RESCUED, "the watchdog had to raise the word: the bound did not fire"
    assert nv.stopped() == (0, 2), nv.stopped()
    assert 0.5 * budget < dt < 2 * budget, dt
    t0 = time.time()
    mx.eval(nv.wait_for(table, nv.signal_after(ids, ev, 3, None, 0), ev, 4))
    dt = time.time() - t0
    done.set()
    print(f"bound: a later wait fell through in {dt * 1e3:.1f} ms", flush=True)
    assert dt < 0.5, dt
    from mlx_lm.generate import generate_step
    with tempfile.TemporaryDirectory() as d:
        model, prompt = toy_model(Path(d) / "toy")
        try:
            for _ in generate_step(mx.array(prompt, dtype=mx.int32), model, max_tokens=4):
                pass
        except RuntimeError as e:
            print(f"bound: the native pass failed loudly: {e}", flush=True)
            assert "gave up" in str(e), e
        else:
            raise AssertionError("native decode ran on after a wait gave up")
    print("ok bound", flush=True)


def case_dead():
    nv = native()
    import mlx.core as mx
    ev, gate = nv.SharedEvent(), nv.SharedEvent()
    budget = nv.wait_bound()[2]
    done = watchdog(nv, [ev, gate], 5.0)
    ids, table = mx.array([1, 2], dtype=mx.int32), mx.zeros(8, dtype=mx.int32)
    mx.eval(ids, table)
    with open(os.devnull, "rb") as f:
        agent = nv.Agent(ev, f.fileno(), 1, 1)
        g = nv.wait_for(ids, ids, gate, 1)          # the GPU parks here: nobody raises gate
        y = nv.wait_for(table, nv.signal_after(g, ev, 1, agent, 0), ev, 2)
        mx.async_eval(y)                            # job 1 is with the agent now
        time.sleep(0.5)
        t0 = time.time()
        agent.close()                               # its thread goes, job 1 unanswered
        mx.eval(y)
        dt = time.time() - t0
    done.set()
    print(f"dead: waits done {dt * 1e3:.0f} ms after close(), stopped={nv.stopped()}, "
          f"error={agent.error!r}", flush=True)
    assert not RESCUED, "the watchdog had to raise the words: close() did not release"
    assert nv.stopped() == (1, 0), nv.stopped()
    assert "closed before the GPU reached a job" in agent.error, agent.error
    assert agent.drain(1.0)
    assert dt < min(1.0, budget / 2), dt
    print("ok dead", flush=True)


def kill_when_pending(nv):
    while not nv.pending():
        time.sleep(0.0001)
    os.kill(os.getpid(), signal.SIGTERM)


def case_term(self_kill=False):
    nv = native()
    import mlx.core as mx
    from mlx_lm.generate import generate_step
    import stream_fast
    with tempfile.TemporaryDirectory() as d:
        model, prompt = toy_model(Path(d) / "toy")
        # Never set: if the parent's TERM did not end us, release and go.
        watchdog(nv, [], 25.0)
        for n, _ in enumerate(generate_step(mx.array(prompt, dtype=mx.int32), model,
                                            max_tokens=100000)):
            if RESCUED:
                os._exit(3)
            ran = stream_fast.TIMERS["native_park"] > 0
            if n % 20 == 0:
                print(f"decoding {n} native={int(ran)}", flush=True)
            if self_kill and ran and n == 40:
                threading.Thread(target=kill_when_pending, args=(nv,), daemon=True).start()


# ---------------------------------------------------------------- the parent

def ioreg(key):
    out = subprocess.run(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"],
                         capture_output=True, text=True).stdout
    m = re.findall(rf'"{key}"=(\d+)', out)
    return max(map(int, m)) if m else -1


def gpu_util():
    return ioreg("Device Utilization %")


def wired_mb():
    """What the GPU holds of system memory (ioreg), MB."""
    return ioreg("In use system memory") / 2**20


def rss_mb(pid):
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True,
                         text=True).stdout.strip()
    return int(out) / 1024 if out else 0


def run_case(name, budget, idle, wired0, term_after=None):
    env = {**os.environ, "MOE_NATIVE_WAIT_S": str(budget), "PYTHONUNBUFFERED": "1"}
    argv = [sys.executable, __file__, "--case", name] + ([NATIVE_DIR] if NATIVE_DIR else [])
    p = subprocess.Popen(argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True)
    lines, peak, t_term = [], [0], [None]

    def sample():
        while p.poll() is None:
            peak[0] = max(peak[0], gpu_util())
            if rss_mb(p.pid) > 4096:            # a toy never gets here
                print(f"  {name}: child over 4 GB, TERM", flush=True)
                p.send_signal(signal.SIGTERM)
                return
            time.sleep(0.2)
    threading.Thread(target=sample, daemon=True).start()
    for line in p.stdout:
        lines.append(line.rstrip())
        print(f"  {name}| {line.rstrip()}", flush=True)
        if term_after is not None and t_term[0] is None and line.startswith("decoding") \
                and "native=1" in line and int(line.split()[1]) >= 40:
            time.sleep(term_after)
            t_term[0] = time.time()
            p.send_signal(signal.SIGTERM)
    rc = p.wait(timeout=60)
    t_exit = time.time()
    t_idle = None
    while time.time() - t_exit < budget + 5:     # measured past the limit, for the record
        if gpu_util() <= idle:
            t_idle = time.time() - t_exit
            break
        time.sleep(0.1)
    dw = wired_mb() - wired0
    print(f"  {name}: rc={rc}, GPU peak {peak[0]}%, idle (<= {idle}%) "
          f"{'never' if t_idle is None else f'{t_idle:.1f} s'} after exit, "
          f"GPU memory in use {dw:+.0f} MB"
          + ("" if t_term[0] is None else f", exited {t_exit - t_term[0]:.2f} s after TERM"),
          flush=True)
    assert t_idle is not None and t_idle < IDLE_WITHIN, f"{name}: GPU idle after {t_idle} s"
    assert dw < 256, f"{name}: the GPU holds {dw:.0f} MB more than before"
    return rc, lines, (None if t_term[0] is None else t_exit - t_term[0])


def main():
    try:
        native()
    except ImportError:
        print("test_native_exit: skipped (needs moe_native built by native/build.sh; pass its directory as an argument)")
        return
    samples = []
    for _ in range(10):
        samples.append(gpu_util())
        time.sleep(0.2)
    base = sorted(samples)[len(samples) // 2]
    idle, wired0 = max(10, base + 10), wired_mb()
    print(f"GPU before: {samples} -> idle means <= {idle}%; memory in use {wired0:.0f} MB",
          flush=True)
    assert base >= 0 and wired0 >= 0, "no IOAccelerator statistics in ioreg"

    if "bound" in NAMES:
        rc, lines, _ = run_case("bound", 2, idle, wired0)
        assert rc == 0 and "ok bound" in lines, "bound failed: no kill test before it passes"
    if "dead" in NAMES:
        rc, lines, _ = run_case("dead", 10, idle, wired0)
        assert rc == 0 and "ok dead" in lines, "dead failed"
    if "term" not in NAMES:
        print("ok")
        return
    pending = []
    for after in (0.0, 0.37, 1.1, None, None):
        name = "term" if after is not None else "term-self"
        rc, lines, dt = run_case(name, 10, idle, wired0, term_after=after)
        assert rc == -signal.SIGTERM, f"{name}: rc {rc}"
        assert any("GPU waits released" in l for l in lines), f"{name}: no release on SIGTERM"
        assert after is None or (dt is not None and dt < 2.0), f"term: exited {dt} s after TERM"
        pending.append((name, any("(one was pending)" in l for l in lines)))
    print(f"term: a wait pending at the TERM: {pending}", flush=True)
    assert any(p for n, p in pending), "no TERM landed with a wait pending: proved nothing"
    print("ok")


if __name__ == "__main__":
    if CASE:
        {"bound": case_bound, "dead": case_dead, "term": case_term,
         "term-self": lambda: case_term(True)}[CASE]()
    else:
        main()
