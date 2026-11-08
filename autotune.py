#!/usr/bin/env python3
"""
Work out the settings instead of sweeping for them.

Tuning by hand meant ~30 full generations per model with a 10-15% spread, but the
quantities that decide the settings are cheap to measure directly.

    stride, layers, experts, top_k   read from experts_index.json  (instant)
    SSD bandwidth at our block size  random preadv probe           (~20 s)
    baseline RSS without the arena   one load                      (~10 s)

From those:

    max_resident = (rss_budget - baseline) / bytes_per_slot
    look         = top_k          measured rule: at top_k 8, width 4 left the
                                  SSD 40% overlapped where width 8 gives 77%
    depth        2 or 3           the only two worth trying; 1 underfills the
                                  window, beyond 3 predicts from too far back

So the search collapses from a grid to a few candidates near the memory
ceiling, and only those get a real generation (median of several runs, with a
load check first, so a busy machine cannot poison the result).

Usage:
    python3 autotune.py --model ~/models/<name> --top-k 8
"""

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
RESULTS = HERE / "results"
PY = sys.executable


def sysctl_int(name):
    out = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True)
    try:
        return int(out.stdout.strip())
    except ValueError:
        return None


def cpu_busy_pct():
    """Share of CPU taken by everything except our own python processes."""
    lines = subprocess.run(["ps", "-A", "-o", "%cpu,comm"],
                           capture_output=True, text=True).stdout.splitlines()[1:]
    total = 0.0
    for line in lines:
        try:
            pct, comm = line.strip().split(None, 1)
        except ValueError:
            continue
        if "python" in comm.lower():
            continue
        total += float(pct)
    return total


# Set from --allow-busy. Default False so a loaded machine stops the run.
_ALLOW_BUSY = False


def wait_for_quiet(limit=60.0, tries=8, pause=15):
    """Refuse to measure on a loaded machine, because a busy CPU spreads repeated
    runs enough to void any conclusion drawn from them."""
    for attempt in range(tries):
        busy = cpu_busy_pct()
        if busy < limit:
            return busy
        print(f"  machine busy ({busy:.0f}% CPU elsewhere), waiting {pause}s "
              f"[{attempt + 1}/{tries}]", flush=True)
        time.sleep(pause)
    busy = cpu_busy_pct()
    if _ALLOW_BUSY:
        print(f"  --allow-busy: measuring at {busy:.0f}% anyway. These numbers "
              f"are not comparable with anything.", flush=True)
        return busy
    raise SystemExit(
        f"\nRefusing to measure: {busy:.0f}% of the CPU is used elsewhere "
        f"(limit {limit:.0f}%).\n"
        f"Close whatever is compiling -- Xcode has ruined two series already -- "
        f"or pass --allow-busy\nto record numbers that cannot be compared.")


def model_facts(path):
    idx = json.loads((path / "experts_index.json").read_text())
    per_layer, strides = {}, []
    for name, meta in idx.items():
        strides.append(meta["stride"])
        m = re.search(r"layers\.(\d+)\.", name)
        if m:
            per_layer[int(m.group(1))] = per_layer.get(int(m.group(1)), 0) + meta["stride"]
    return {
        "projections": len(idx),
        "moe_layers": len(per_layer),
        "n_experts": next(iter(idx.values()))["n_experts"],
        # One expert held resident in every layer, all three projections.
        "bytes_per_slot": sum(per_layer.values()),
        "median_stride": statistics.median(strides),
    }


def probe_bandwidth(path, stride, threads=8, n_reads=1500, binf=None, direct=True):
    """GB/s for random reads of exactly our record size, page cache bypassed
    (F_NOCACHE on macOS, O_DIRECT on Linux; direct=False reads buffered on Linux, for the
    O_DIRECT-or-not choice). `binf`: the file to read, by default path's experts.bin
    (machine.py points it at a scratch file)."""
    import mmap
    import os
    import random
    import threading
    from concurrent.futures import ThreadPoolExecutor

    binf = Path(binf) if binf else path / "experts.bin"
    stride = int(stride)
    if sys.platform == "darwin":
        import fcntl
        fd = os.open(binf, os.O_RDONLY)
        fcntl.fcntl(fd, 48, 1)              # F_NOCACHE
    else:
        # O_DIRECT wants offsets and lengths on 4 KB: a record rounded up.
        fd = os.open(binf, os.O_RDONLY | (getattr(os, "O_DIRECT", 0) if direct else 0))
        stride = -(-stride // 4096) * 4096
    size = binf.stat().st_size
    offs = [random.randrange(0, max(size - stride, 1) // stride) * stride
            for _ in range(n_reads)]
    # One page-aligned buffer per worker thread, because btrfs checks a direct read's checksum in the caller's buffer and a shared one fails it.
    local = threading.local()

    def one(i):
        if not hasattr(local, "buf"):
            local.buf = mmap.mmap(-1, stride)
        os.preadv(fd, [local.buf], offs[i])

    t = time.perf_counter()
    with ThreadPoolExecutor(max_workers=threads) as p:
        list(p.map(one, range(n_reads)))
    el = time.perf_counter() - t
    os.close(fd)
    return n_reads * stride / el / 1e9


def baseline_rss(path, top_k, probe_slots=8, slot_gb=None):
    """RSS of the model with no arena: the fixed part of the budget.

    Measured by running a small arena and subtracting it, not by loading with
    resident=0. Loading with no arena and calling mx.eval on every parameter
    leaves transient buffers in the high-water mark that a running engine never
    holds: it reported 2.33 GB where the real engine peaks at 2.24 GB WITH 21
    slots already resident, i.e. it overstated the fixed part by about 1.4 GB.
    That single number is what capped resident at 21 when the budget allows
    around 39, and 45 slots measured 24.97 tok/s against 21.19 at 21.
    """
    # Deliberately not made cold: chat.py and server.py run warm, and the
    # budget should describe how the engine actually runs rather than how
    # run_engine measures it.
    code = (
        "import sys, json, resource, pathlib;"
        f"sys.path.insert(0, {str(HERE)!r});"
        "import mlx.core as mx;"
        "from stream_fast import load_slim_aware, install, set_top_k;"
        "from mlx_lm.models.cache import make_prompt_cache;"
        f"p = pathlib.Path({str(path)!r});"
        "m, t, _ = load_slim_aware(p, str(p));"
        f"install(m, p, resident={probe_slots});"
        f"set_top_k(m, {top_k});"
        "ids = t.encode('Explain how a CPU cache works.');"
        "c = make_prompt_cache(m);"
        "h = m.language_model.model(mx.array([ids]), c);"
        "[mx.eval(m.language_model.lm_head("
        "m.language_model.model(mx.array([[int(mx.argmax("
        "m.language_model.lm_head(h)[0, -1]))]]), c))) for _ in range(8)];"
        "print(json.dumps({'rss': resource.getrusage("
        "resource.RUSAGE_SELF).ru_maxrss / 1e9}))"
    )
    out = subprocess.run([PY, "-c", code], capture_output=True, text=True)
    for line in reversed(out.stdout.splitlines()):
        if line.startswith("{"):
            rss = json.loads(line)["rss"]
            return max(0.1, rss - probe_slots * (slot_gb or 0.0))
    raise RuntimeError(f"baseline probe failed:\n{out.stderr[-800:]}")


def run_engine(path, resident, look, depth, top_k, tokens, cache_bias=0.0,
               track_used=False):
    """One generation; returns the engine's TUNE record, or None if it failed."""
    cmd = [PY, str(HERE / "stream_fast.py"), "--model", str(path),
           "--max-tokens", str(tokens), "--top-k", str(top_k),
           "--resident", str(resident), "--look", str(look),
           "--look-depth", str(depth), "--cache-limit-gb", "0.05",
           "--cache-bias", str(cache_bias),
           "--no-page-cache", "--idle-baseline", "1"]
    if track_used:
        cmd.append("--track-used")
    out = subprocess.run(cmd, capture_output=True, text=True)
    m = re.search(r"^TUNE (.+)$", out.stdout, re.M)
    return json.loads(m.group(1)) if m else None


def measure(path, resident, look, depth, top_k, tokens, repeats,
            cache_bias=0.0):
    """Real generations; median decode tok/s and worst-case RSS."""
    recs = [run_engine(path, resident, look, depth, top_k, tokens, cache_bias)
            for _ in range(repeats)]
    recs = [r for r in recs if r and r.get("decode_tps")]
    if not recs:
        return None
    tps = [r["decode_tps"] for r in recs]
    return {"decode_tps": round(statistics.median(tps), 2),
            "runs": [round(x, 2) for x in tps],
            "rss_gb": round(max(r["rss_gb"] for r in recs), 2),
            # D2: ru_maxrss (rss_gb) does not see Metal buffers, but budget_gb
            # is computed from mx.get_active_memory() -- compare against both
            # and let whichever reads higher trip the gate below.
            "mlx_active_gb": round(max(r.get("mlx_active_gb", 0.0) or 0.0
                                      for r in recs), 2),
            "disk_pct": recs[-1]["disk_pct"],
            "blocking_pct": recs[-1]["blocking_pct"]}


def probe_bias(path, resident, look, depth, top_k, tokens, biases,
               min_diversity=0.75):
    """Pick how hard to bias the router toward resident experts.

    Cache-conditional routing trades expert diversity for bytes, and the failure
    to fear is that it stops meaning "prefer a cached expert the router almost
    liked" and starts meaning "only ever use these few" -- a smaller model
    wearing a big model's name. Six prompts will not show that; the count of
    distinct experts will.

    Ranked on byte counters and the expert count, both of which ignore how loaded
    the machine is, so this part of the tuning is trustworthy even when the clock
    is not. Diversity is judged against the unbiased run rather than an absolute
    threshold, because it also depends on how many tokens were generated.
    """
    rows, base = [], None
    print(f"\n{'bias':>6} {'disk':>8} {'blocking':>9} {'experts':>9} "
          f"{'diversity':>10} {'tok/s':>7}")
    for b in biases:
        r = run_engine(path, resident, look, depth, top_k, tokens, b,
                       track_used=True)
        if not r:
            print(f"{b:>6.2f}   FAILED")
            continue
        if base is None:
            base = r["experts_median"] or 1
        div = (r["experts_median"] or 0) / max(base, 1)
        rows.append({**r, "cache_bias": b, "diversity": round(div, 3),
                     "ok": div >= min_diversity})
        print(f"{b:>6.2f} {r['disk_pct']:>7.1f}% {r['blocking_pct']:>8.1f}% "
              f"{str(r['experts_median']):>9} {div:>9.2f}x "
              f"{r['decode_tps']:>7}"
              + ("" if div >= min_diversity else "   too narrow"))
    good = [r for r in rows if r["ok"]]
    best = min(good, key=lambda r: r["disk_pct"]) if good else None
    if best:
        print(f"  -> cache_bias {best['cache_bias']}: disk {best['disk_pct']}% "
              f"of demand at {best['diversity']}x the unbiased diversity")
    return (best["cache_bias"] if best else 0.0), rows


def over_budget(got, budget_gb):
    """True if rss_gb (ru_maxrss, blind to Metal buffers) or mlx_active_gb (the
    instrument budget_gb is computed from) reads over the budget."""
    measured = max(got.get("rss_gb") or 0.0, got.get("mlx_active_gb") or 0.0)
    return bool(measured and measured > budget_gb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--top-k", type=int, default=8)
    # The budget is sized as a share of the machine like every other entry
    # point, and --rss-budget stays as an explicit override for a number
    # measured on some other machine (a watch, a phone) that this one
    # cannot read from hw.memsize.
    ap.add_argument("--mem-pct", type=int, default=90,
                    help="process memory budget as %% of hw.memsize (default "
                         "90; ignored if --rss-budget is also given)")
    ap.add_argument("--rss-budget", type=float, default=None,
                    help="hard ceiling on process memory, GB -- overrides "
                         "--mem-pct")
    ap.add_argument("--host-gb", type=float, default=0.6,
                    help="python/numpy/MLX-runtime overhead the budget cannot "
                         "see directly")
    ap.add_argument("--tokens", type=int, default=100)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--tag", default=None)
    # 20 is the level at which numbers hold still, because a looser limit
    # let foreign load penalise the arms with many waves and void whole
    # sweeps.
    ap.add_argument("--busy-limit", type=float, default=20.0,
                    help="refuse to measure above this %% of CPU used elsewhere")
    # The engine now defaults to cache_bias 0.25, so tuning at 0 would optimise a
    # configuration nobody runs: with the bias on, blocking reads fall from 13.4%
    # of demand to 5.1% and the arena's marginal value changes with them.
    ap.add_argument("--cache-bias", type=float, default=-1.0,
                    help="-1 = probe for it, which is the point of this tool")
    ap.add_argument("--biases", default="0,0.1,0.25,0.5")
    ap.add_argument("--min-diversity", type=float, default=0.75,
                    help="reject a bias that narrows expert use below this "
                         "fraction of the unbiased run")
    ap.add_argument("--allow-busy", action="store_true",
                    help="measure anyway on a loaded machine; results are junk")
    args = ap.parse_args()

    path = Path(args.model).expanduser()
    tag = args.tag or path.name

    facts = model_facts(path)
    print(f"model: {facts['moe_layers']} MoE layers, {facts['n_experts']} experts, "
          f"{facts['projections']} projections")
    print(f"one resident expert across all layers = "
          f"{facts['bytes_per_slot']/1e6:.0f} MB")

    global _ALLOW_BUSY
    _ALLOW_BUSY = args.allow_busy

    # How hard to bias comes first, and it does NOT need a quiet machine: it is
    # ranked on byte counters and the expert count, neither of which cares how
    # loaded the box is. Only the tok/s sweep below needs quiet, so a busy
    # machine still gets the answer that matters most.
    bias_rows = []
    if args.cache_bias < 0:
        args.cache_bias, bias_rows = probe_bias(
            path, 15, args.top_k, 2, args.top_k, args.tokens,
            [float(x) for x in args.biases.split(",")], args.min_diversity)

    busy = wait_for_quiet(args.busy_limit)
    print(f"cpu used elsewhere: {busy:.0f}%   free memory level: "
          f"{sysctl_int('kern.memorystatus_level')}%")

    bw = probe_bandwidth(path, facts["median_stride"])
    print(f"ssd at our record size ({facts['median_stride']/1e6:.2f} MB): "
          f"{bw:.2f} GB/s")

    slot_gb = facts["bytes_per_slot"] / 1e9
    base = baseline_rss(path, args.top_k, slot_gb=slot_gb)
    total_gb = (sysctl_int("hw.memsize") or 0) / 1e9
    budget_gb = args.rss_budget
    if budget_gb is None:
        budget_gb = total_gb * args.mem_pct / 100
    # D1: min(machine, budget) -- the machine-safety reserve caps the process
    # budget too, whether it came from --mem-pct or an explicit --rss-budget.
    # Without this, --rss-budget (or a generous --mem-pct on a small machine)
    # could ask for more slots than the machine can hand out without swapping,
    # same failure this reserve exists to prevent everywhere else. 15% matches
    # ArenaGovernor's own default reserve_pct -- the two are meant to agree.
    budget_gb = min(budget_gb, total_gb * (100 - 15) / 100)
    # Same pure function the engine's own governor uses (D4), not a
    # reimplementation: headroom is one layer's wide-read transient
    # (n_experts * slot_gb / moe_layers -- _batch_wide's buffer is per layer,
    # not the whole model) plus --host-gb for the interpreter/numpy/MLX
    # runtime itself. 0.3 GB of headroom was not enough on its own: a run at
    # 40 slots peaked at 4.08 GB where 45 slots peaked at 3.74.
    from stream_fast import budget_slots
    headroom_gb = facts["n_experts"] * slot_gb / facts["moe_layers"] + args.host_gb
    # Clamped by the model too: a layer only has facts['n_experts'] experts,
    # so a budget roomier than the layer itself (a generous --mem-pct against
    # a small model) must not ask _init_arena to allocate more resident slots
    # than there are experts to fill them.
    max_res = max(2, min(facts["n_experts"],
                         budget_slots(budget_gb, base, slot_gb, headroom_gb)))
    print(f"baseline rss without arena: {base:.2f} GB  ->  max resident within "
          f"{budget_gb:.2f} GB "
          f"({'--rss-budget' if args.rss_budget is not None else str(args.mem_pct) + '% of RAM'}"
          f", machine-reserved)"
          f": {max_res}")

    rows = facts["moe_layers"] * 3 * args.top_k
    bytes_tok = rows * facts["median_stride"]
    print(f"bytes/token at top_k {args.top_k}: {bytes_tok/1e6:.0f} MB  ->  "
          f"{bw*1e9/bytes_tok:.1f} tok/s with no cache and no overlap")

    # Cluster the candidates near the ceiling: below it the arena is the binding
    # constraint, and the manual sweeps showed the curve is flat above it.
    residents = sorted({max(2, max_res // 2), max(2, int(max_res * 0.75)), max_res})
    # Empirical guard, not a derived rule: wide prefetch degrades badly on
    # the models measured for a cause that is unknown (it is not the page
    # cache), while keeping width under the arena costs nothing on roomy
    # budgets and avoids the cliff on tight ones.
    look_of = lambda r: min(args.top_k, max(1, r - 1))
    cands = [(r, look_of(r), d) for r in residents for d in (2, 3)]
    capped = {r: look_of(r) for r in residents if look_of(r) < args.top_k}
    print(f"\ntesting {len(cands)} candidates at cache_bias {args.cache_bias} "
          f"(look = top_k {args.top_k}, held below the arena"
          + (f"; capped to {capped} for the small arenas" if capped else "")
          + f"), {args.repeats} runs each:\n")

    tried = []
    for r, look, d in cands:
        wait_for_quiet(args.busy_limit)
        got = measure(path, r, look, d, args.top_k, args.tokens,
                      args.repeats, args.cache_bias)
        if not got:
            print(f"  r={r:>3} look={look} depth={d}: FAILED")
            continue
        over = over_budget(got, budget_gb)
        print(f"  r={r:>3} look={look} depth={d}: {got['decode_tps']:>6} tok/s"
              f"  rss {got['rss_gb']} GB  mlx active {got['mlx_active_gb']} GB"
              f"{'  OVER BUDGET' if over else ''}"
              f"  {got['runs']}")
        tried.append({"resident": r, "look": look, "depth": d,
                      "over_budget": over, **got})

    if not tried:
        raise SystemExit("every candidate failed to run")

    # Speed rises with the arena right up to the memory ceiling, so the fastest
    # candidate is usually the largest -- and on Qwen3.5 that one overshot the
    # budget (4.37 GB against 4.0) while the next size down was 3 tok/s slower.
    # The useful setting sits between them, which a fixed grid cannot see.
    # Bisect on RSS: the arena is the only term that grows with `resident`.
    ok = [t for t in tried if not t["over_budget"]]
    over = [t for t in tried if t["over_budget"]]
    if over and ok:
        lo = max(t["resident"] for t in ok)
        hi = min(t["resident"] for t in over)
        depth = max(ok, key=lambda t: t["decode_tps"])["depth"]
        for _ in range(2):
            if hi - lo <= 1:
                break
            mid = (lo + hi) // 2
            wait_for_quiet(args.busy_limit)
            got = measure(path, mid, args.top_k, depth, args.top_k,
                          args.tokens, args.repeats, args.cache_bias)
            if not got:
                break
            fits = not over_budget(got, budget_gb)
            print(f"  r={mid:>3} look={args.top_k} depth={depth}: "
                  f"{got['decode_tps']:>6} tok/s  rss {got['rss_gb']} GB "
                  f"mlx active {got['mlx_active_gb']} GB"
                  f"{'' if fits else '  OVER BUDGET'}  {got['runs']}")
            row = {"resident": mid, "look": args.top_k, "depth": depth,
                   "over_budget": not fits, **got}
            tried.append(row)
            if fits:
                ok.append(row)
                lo = mid
            else:
                hi = mid

    ok = ok or tried
    best = max(ok, key=lambda t: t["decode_tps"])

    # Does memory actually buy speed on THIS model? It is not a given.
    # Measured: on ERNIE going from 24 to 44 slots changed nothing (-2%),
    # because at 64 experts per layer the arena already held most of what the
    # router asked for. On Qwen3.5, with 256 experts per layer, the same move
    # was +29%. So the answer decides the deployment mode rather than a knob:
    # if memory pays, let the governor claim what the machine can spare; if it
    # does not, pin a small arena and leave the rest of the machine alone.
    # Compare like with like, because taking the extremes of a size-sorted list
    # lets the stable sort pick the depth and can flip the answer against the
    # threshold.
    fits = [t for t in tried if not t.get("over_budget")] or tried
    per_arm = {}
    for t in fits:
        arm = (t["look"], t["depth"])
        cur = per_arm.setdefault(arm, {})
        if t["decode_tps"] > cur.get(t["resident"], 0):
            cur[t["resident"]] = t["decode_tps"]
    spans = [(arm, min(sizes), max(sizes)) for arm, sizes in per_arm.items()
             if len(sizes) >= 2]
    if spans:
        # Widest arena span wins; it is the one the elasticity is about.
        arm, r_lo, r_hi = max(spans, key=lambda s: s[2] - s[1])
        sizes = per_arm[arm]
        lo = {"resident": r_lo, "decode_tps": sizes[r_lo]}
        hi = {"resident": r_hi, "decode_tps": sizes[r_hi]}
    else:
        by_size = sorted(fits, key=lambda t: t["resident"])
        lo, hi = by_size[0], by_size[-1]
    gb_delta = (hi["resident"] - lo["resident"]) * slot_gb
    gain_pct = (hi["decode_tps"] / lo["decode_tps"] - 1) * 100 if lo["decode_tps"] else 0
    per_gb = gain_pct / gb_delta if gb_delta > 0.05 else 0.0
    # Known limit: 5.0 %/GB is inherited and has not been validated
    # against an outcome, so it stays until something measures where the
    # deployment mode actually flips.
    pays = per_gb >= 5.0

    print(f"\nmemory elasticity: {lo['resident']} -> {hi['resident']} slots "
          f"(+{gb_delta:.2f} GB) changes speed by {gain_pct:+.1f}% "
          f"= {per_gb:+.1f}% per GB")
    if pays:
        # Smallest arena that still fits, so the governor has room to grow.
        floor = min(t["resident"] for t in ok)
        print("  -> memory PAYS on this model: run the adaptive governor, it "
              "will take what the machine can spare and hand it back under "
              "pressure")
        recommend = (f"chat.py --model {path} --top-k {args.top_k} "
                     f"--look {best['look']} --look-depth {best['depth']} "
                     f"--resident {floor} --adaptive --mem-pct {args.mem_pct}")
    else:
        print("  -> memory does NOT pay on this model: pin the arena and leave "
              "the rest of the machine alone")
        # Cheapest setting within noise of the best, so nothing is spent for
        # a gain that is not there.
        near = [t for t in ok if t["decode_tps"] >= best["decode_tps"] * 0.97]
        thrifty = min(near, key=lambda t: t["resident"])
        recommend = (f"chat.py --model {path} --top-k {args.top_k} "
                     f"--look {thrifty['look']} --look-depth {thrifty['depth']} "
                     f"--resident {thrifty['resident']}")
    print(f"\nbest within budget: resident={best['resident']} "
          f"look={best['look']} depth={best['depth']} -> "
          f"{best['decode_tps']} tok/s at {best['rss_gb']} GB")
    print(f"\n  {PY} {recommend}")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"autotune_{tag}.json").write_text(json.dumps({
        "model": str(path), "top_k": args.top_k, "facts": facts,
        "bandwidth_gbs": round(bw, 2), "baseline_rss_gb": round(base, 2),
        "max_resident": max_res, "bytes_per_token_mb": round(bytes_tok / 1e6),
        "mem_pct": args.mem_pct, "budget_gb": round(budget_gb, 2),
        "host_gb": args.host_gb,
        "cpu_busy_pct": round(busy), "cache_bias": args.cache_bias,
        "bias_probe": bias_rows,
        "candidates": tried, "best": best,
        "slot_gb": round(slot_gb, 4),
        "elasticity_pct_per_gb": round(per_gb, 1),
        "memory_pays": bool(pays),
        "recommended": recommend,
    }, indent=2))
    print(f"wrote {RESULTS.name}/autotune_{tag}.json")


if __name__ == "__main__":
    main()
