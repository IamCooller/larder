# The research record

This folder is the measured record behind the engine: what was tried, what worked, what did not, and why. It is written for someone who wants to check a claim or build on it, so every number says where it came from.

| file | what is in it |
|---|---|
| [WORKED.md](WORKED.md) | 97 levers that landed, by theme: what each does, MoE-only or dense too, the measured gain, the quality check it passed, and its status now |
| [CLOSED.md](CLOSED.md) | 197 ideas that were measured and closed, by theme, each with what was measured and why it was closed. The negative results are the larger and probably the more useful half |
| [HARDWARE.md](HARDWARE.md) | the two machines, their drives and read rates, the models and their packs, where the time of one token goes, and the comparison with llama.cpp |

The record covers 2026-09-04 to 2026-11-07 on the two machines described below; see "What is left out" for what is not here.

## The machines

Two machines carry every number in this folder:

- **16 GB M4 MacBook Pro** ("the M4"): Apple M4 with a 10-core GPU, 16 GB of unified memory, the internal SSD, and an external SSD in a Thunderbolt enclosure for the largest model.
- **Linux PC, RTX 5060 Ti 16 GB, 15 GB RAM, one NVMe SSD** ("the PC").

Both are shared, everyday machines, not lab hardware. Other programs ran beside many measurements; the rules below exist because of that.

**The 35B mixed pack.** Qwen3.6-35B-A3B numbers measured on the M4 ran packs made with this engine's packer, not the published checkpoint; in its final layout the pack holds 20 of the 40 layers' experts at 4-bit and 20 at 2-bit, 16.1 GB, and older rows may have run an earlier layout of the same packer. The launcher installs the published 4-bit pack instead, and that pack was measured only on the PC (2026-11-07). Every row that quotes a 35B number from the M4 says "35B mixed pack".

## The rules every result obeyed

**Quality.** A lever is kept only if the reply stays the model's own. There are three ways to show that:

1. **Greedy-identical.** At `--cache-bias 0` (exact routing) the lever's replies are byte-identical to the engine without it, or they part only at bf16 ties. A tie is checked, not assumed: at the first differing token, the two candidates' logits are compared in an fp32 replay, and a gap of at most two bf16 steps is a tie. Many checks also compare cache arrays or first-token logits bit for bit.
2. **A paired benchmark gate.** When a lever changes the computation on purpose (the cache bias, a pack with fewer bits, a kernel that is not bit-exact), the same tasks run with and without it: GSM8K (100 or 300 tasks) and HumanEval (164 tasks), paired by task, greedy. The lever passes when the exact McNemar test on the discordant pairs is not significant. Perplexity alone is never enough (see the lessons below).
3. **An error or KL bar.** For a kernel that only changes rounding: its output must be no further from an fp32 reference than the stock kernel's, or the teacher-forced KL of its logits must stay under a bar written before the run (against the stock path, or against the run-to-run noise of two plain runs). The Neural Engine prefill path also passed a paired set of long-prompt tasks.

Lossy levers were out of scope by rule: a lever that changes replies needs a passed check (a gate or an error or KL bar) and an explicit decision to adopt it. The defaults that are not bit-exact are listed in the main README's Quality table with the check each one passed: the cache bias, the EXL3 pack, `MOE_FUSE_HC`, folded verify attention, the Neural Engine prefill path, the CUDA resident experts, conversation reuse (it parts only at near-ties), the CUDA GatedDeltaNet kernel and the CUDA decode attention over the QSA picks. One configuration has no such check: the 35B mixed pack (fewer bits on 20 layers) was never gated against the 4-bit pack; it is a research configuration the launcher does not install. Sampling, the thinking switch and a 3-bit copy the user picks are the places where replies change by design.

**A golden set.** A fixed set of 7 or 8 requests per setup (the 35B, Flash-Next, the 2 GB-peak setup of the 35B, the PC) whose replies at bias 0 must stay byte-identical across any change that claims to be exact. It is the merge gate for engine changes.

**Screening.** A new idea first gets one repeat of 10 to 30 minutes, with a kill bar written down before the run: the speedup it must reach, and what would stop it. When a row in [CLOSED.md](CLOSED.md) or [WORKED.md](WORKED.md) names a bar, that bar was written before the numbers existed. An idea that passes the screen is then measured in interleaved rounds (A, B, A, B or ABBA) of fresh server processes, so a slow machine moment hits both arms.

**Measured and estimated.** Every number is measured unless it is marked "(est.)". Estimates come from a cost model with measured constants, an offline replay of recorded runs, or a no-model probe, and the row says which.

**Machine, setup and date.** Every number names the machine, the setup (model and pack, flags, prompt, cache bias, arena or memory budget) and the date it was measured. Two numbers measured on different days are not compared as if they were one experiment; when two sources disagree, the later measurement is used and the difference is noted.

**Universal levers only.** Nothing was trained for one model: no draft heads, adapters or predictors fit to a particular checkpoint. A model's own multi-token-prediction head and a published same-family draft model are used when they exist.

**No fixed thresholds.** Where a lever has a switch point (a read rate, a context length, a memory size), the engine measures it online or once at install and caches it. A different disk or a busier computer changes the choice.

**The machine must never swap itself into a reboot.** On the M4 every probe and harness aborts itself above 11 GB of footprint, on 40 MB of new swapouts, or when the margin to macOS's swap line falls under 1 GB. One GPU job runs at a time on each machine.

## Reading a row

- **Scope.** "MoE only" means the lever needs routed experts (the expert cache, the reads, the routing bias). "Dense too" or "any model" means it also helps a model that fits in memory. "CUDA only" or "Metal only" names the backend. Dense levers (drafts, the batch, attention kernels) carry over to MoE models only where the row says so.
- **Speeds.** "tok/s" is decode speed after the prompt has been read. "First token" is the time from the request to the first token, which includes reading the prompt. "ms a token" is the decode time per token. "ru / en / py" are the three standard prompts: Russian prose, English prose and Python code.
- **The cache bias.** `--cache-bias 0.5` (the default on the gated MoE models) lets routing prefer an expert already in memory; `--cache-bias 0` is exact routing. Exactness claims are checked at bias 0, because under the bias a reply depends on what the cache holds, which differs between processes.
- **Status.** "Default" is on without a flag, "opt-in" needs a flag or an environment variable, "recommended" is in the recommended command but not a server default, "superseded" names what replaced it.
- **Evidence topics.** A row may name an evidence topic such as "arena_sweep". That is the name of the folder of raw logs and notes behind the number.

## Where the raw logs are

The raw logs, per-run JSON files and scripts behind every row are kept in a private research tree, one folder per topic. They are not published because they hold machine paths, host names and account names throughout. The development history (commits and branches) is private for the same reason, so the record cites dates and topic names, not commits. The numbers in these files were copied from those notes and checked against them.

## What is left out

- **Results not tied to these two machines.** A result is listed only when it was measured on the M4 or the PC. Some foundations of the design (one record per expert, one read wave per layer, the directly addressed cache, zero copy) were never measured alone on them, so [WORKED.md](WORKED.md) lists them without a gain.
- **Work across two machines.** A design for one model across two computers over Wi-Fi was explored and parked; it is left out.
- **Tooling and presentation.** The web pages, the menu and cleanups changed no reply and are listed only briefly.

## Lessons about measuring

These cost a wrong conclusion at least once, and they shaped the rules above.

1. **Perplexity does not measure an assistant.** A low-bit pack can keep its perplexity and still answer in the wrong language or refuse to write code. Every lossy change gets a behaviour check and a paired benchmark gate.
2. **A miss rate is not a quality measure.** Dropping 5% of experts at random is harmless; dropping the 5% the cache never holds is not, because real misses always hit the same rare experts. A lever that masks misses was closed after long conversations degenerated, though short evals passed.
3. **Six prompts are not enough.** Configurations that passed a short eval broke in a live conversation. Any configuration gets a long conversation and a paired gate before acceptance.
4. **The order of arms inside one process is a result in itself.** On the PC (2026-10-27) a no-draft decode loop looked 1.130x faster than plain decode when it ran second in the same process; with the order reversed it was 1.007x, and in separate processes 0.987x. The first turn pays for its own experts. Since then every A/B runs fresh processes or both orders.
5. **Profilers inflate small kernels.** A profiler's per-kernel durations put the PC's dense matvec lane at 213 GB/s; the wall clock showed 338 GB/s against a measured ceiling of 411 GB/s, because each launch was about 2 µs too long in the trace (2026-10-27). The wall clock is what a token pays.
6. **Calibration is state.** A harness that re-measured the engine's kernel-route calibration for each arm made two identical servers disagree (2026-10-26). Calibration files are now snapshotted and restored around experiments.
