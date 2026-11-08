#!/usr/bin/env python3
"""Paging, swap and free-disk readings of this machine, shared by the engine and the benchmark harness."""

import json
import re
import subprocess
import threading
import time
from datetime import datetime

PAGE_SIZE = 16384  # Apple Silicon; verified with `sysctl hw.pagesize`

# The disk floor in GiB below which nothing of ours writes a cache.
MIN_FREE_DISK_GB = 5.0


def vm_stat_counters():
    """Cumulative page counters from vm_stat."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    wanted = {
        "free": r"Pages free:\s+(\d+)",
        "pageins": r"Pageins:\s+(\d+)",
        "pageouts": r"Pageouts:\s+(\d+)",
        "swapins": r"Swapins:\s+(\d+)",
        "swapouts": r"Swapouts:\s+(\d+)",
        "compressions": r"Compressions:\s+(\d+)",
    }
    return {k: int(m.group(1)) for k, p in wanted.items() if (m := re.search(p, out))}


def swap_used_mb():
    out = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True).stdout
    m = re.search(r"used\s*=\s*([\d.]+)M", out)
    return float(m.group(1)) if m else 0.0


def free_disk_gb():
    out = subprocess.run(["df", "-g", "/"], capture_output=True, text=True).stdout
    return float(out.strip().splitlines()[-1].split()[3])


def idle_baseline(seconds=15):
    """Page-ins per second of the machine at rest, because vm_stat counts them system-wide and a run is reported net of them."""
    print(f"measuring idle baseline for {seconds}s (leave the machine alone)...")
    a = vm_stat_counters()
    t0 = time.time()
    time.sleep(seconds)
    b = vm_stat_counters()
    dt = time.time() - t0
    rate = (b.get("pageins", 0) - a.get("pageins", 0)) / dt
    mb_s = rate * PAGE_SIZE / 1e6
    print(f"  idle page-ins: {rate:.0f} pages/s = {mb_s:.1f} MB/s")
    return {"pageins_per_s": round(rate, 1), "mb_per_s": round(mb_s, 2), "seconds": seconds}


class Telemetry(threading.Thread):
    """Samples paging counters once a second into a jsonl file."""

    def __init__(self, path):
        super().__init__(daemon=True)
        self.path = path
        self.stop_flag = threading.Event()
        self.samples = []

    def run(self):
        t0 = time.time()
        prev = vm_stat_counters()
        while not self.stop_flag.wait(1.0):
            cur = vm_stat_counters()
            d = {k: cur.get(k, 0) - prev.get(k, 0) for k in cur}
            prev = cur
            self.samples.append(
                {
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "t_rel_s": round(time.time() - t0, 1),
                    "pageins_delta": d.get("pageins", 0),
                    "ssd_read_mb_s": round(d.get("pageins", 0) * PAGE_SIZE / 1e6, 2),
                    "pageouts_delta": d.get("pageouts", 0),
                    "swapins_delta": d.get("swapins", 0),
                    "swapouts_delta": d.get("swapouts", 0),
                    "compressions_delta": d.get("compressions", 0),
                    "swap_used_mb": swap_used_mb(),
                    "free_pages": cur.get("free", 0),
                }
            )

    def stop(self):
        self.stop_flag.set()
        self.join(timeout=5)
        with open(self.path, "w") as f:
            for s in self.samples:
                f.write(json.dumps(s) + "\n")
