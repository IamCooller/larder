# Hardware, models and the cost of a token

The physical facts every speed result in this record depends on: the two machines, their drives, the models and their packs, where the time of one decode token goes, and how the 35B compares with llama.cpp on both machines. Every number names the machine, the setup and the date. "(est.)" marks an estimate; everything else was measured. The evidence topic, where one is named, is the folder of raw logs and notes behind the number; those logs are kept private because they hold machine paths.

Two machines carry every number here:

- **16 GB M4 MacBook Pro**, "the M4" below.
- **Linux PC, RTX 5060 Ti 16 GB, 15 GB RAM, one NVMe SSD**, "the PC" below.

**The 35B mixed pack.** Qwen3.6-35B-A3B numbers measured on the M4 ran packs made with this engine's packer; in its final layout the pack holds 20 of the 40 layers' experts at 4-bit and 20 at 2-bit, 16.1 GB of experts, and older rows may have run an earlier layout. It reads fewer bytes per token than the published 4-bit pack the launcher installs. The launcher's own 4-bit 35B pack was measured only on the PC (2026-11-07, topic: launcher35_pc).

## 1. The machines

| | 16 GB M4 MacBook Pro | Linux PC, RTX 5060 Ti 16 GB |
|---|---|---|
| processor | Apple M4 (base), 10-core GPU | AMD Ryzen 5 7600X, 6 cores, AVX-512, 4.5-5.3 GHz under load |
| memory | 16 GB unified, 120 GB/s nominal | 2 x 8 GB DDR5, configured at 4800 MT/s (EXPO off); the OS sees 15.39 GB, 512 MiB of it reserved for the unused integrated GPU; 95-102 ns random latency, 48-50 GB/s read (2026-11-05) |
| GPU memory | shared with the CPU; Metal working set 12.7 GB (2026-10-08) | 16 GB GDDR7, 128-bit, 448 GB/s by specification; **411 GB/s measured pure read** (2026-10-27, topic: pc_fifty/qmv); the desktop holds 369 MiB of it |
| link to the GPU | unified memory, no copy | PCIe gen4 x8, 15.75 GB/s a direction (the card does gen5, the slot gives gen4); Resizable BAR on |
| expert drive | internal SSD, 460 GiB | WD_BLACK SN770 2 TB, PCIe 4.0 x4, under LUKS2 and btrfs |
| second drive | external SSD in a Thunderbolt/USB4 enclosure (OWC Express 1M2 at 40 Gb/s, exFAT), used for the second half of Flash-Next's expert reads | none used by this engine |
| software | MLX 0.32.2 (the main environment) and 0.32.3 (a second one), mlx_lm 0.31.3, both listed 2026-10-09; Flash-Next's 4-bit pack first ran on mlx_lm 0.32.0 with the qwen4_exp model code | Linux kernel 7.1.9, NVIDIA driver 610.57.04, CUDA 13.3, mlx-cuda 0.32.4 (2026-11-05) |
| power and heat | | 50 W of 180 during decode, no throttle reason in 224 samples; the SN770 never throttled in 3,183 hours (2026-11-05) |

Notes:

- Results on the M4 name their MLX version. 0.32.3 ran the arena sweep, the first llama.cpp comparison and the first MTP check of the recommended command (all 2026-10-08 to 10-09); 0.32.2 ran most others. The shipped requirements pin MLX 0.32.2 on macOS and 0.32.4 (CUDA 13) on Linux.
- Every PC number before 2026-11-05 09:57 local time ran with the dm-crypt read and write workqueues on; after that they are off. Reads without the workqueues measured 1.001x of the earlier rate (2026-11-05, topic: pc_hw/queue).
- On the PC, MLX's CUDA backend allocates arrays made on the host (weights loaded from files) in managed memory. Beside a full resident expert store the driver moved part of them to host RAM, and a kernel then read them over PCIe: the 35B's shared down projection ran ~35 µs in 29 of 40 layers against 3.0-3.7 µs from device memory (2026-11-06, topic: decode35_cuda).
- On the PC, two fresh processes of the same command agree on no position's logits bit for bit at an 8k context (teacher-forced KL 0.0050, top-1 agreement 0.973; 2026-11-06, topic: cuda_resident). Exactness checks on CUDA compare replies, not raw logits.

### The M4's Neural Engine

Called directly through a private macOS interface (2026-11-02, M4, macOS 26.6.2, topic: ane_direct):

- It takes fp16, int8 with a per-row scale, and 4-bit lookup-table weights, but never parameters per input block, so MLX's 4-bit group-32 format has no exact form there.
- Big projections run at 9-11 TFLOPS in fp16 and 14-16 with 4-bit tables, against the GPU's 3.36 on 4-bit weights. A dispatch costs 0.09 ms.
- It shares DRAM with the GPU: beside a bandwidth-bound GPU kernel it runs 2.26x slower and costs that kernel 20%.
- Per projection its error to float64 is 2.4-3.4e-4 against MLX's 2.1e-3.
- Since 2026-11-04 it runs part of Flash-Next's long prefill beside the GPU on Metal by default: 16k prompt 1.206x, +0.5-0.6 GB.

## 2. SSD rates and the read path

The quoted rates measure different things: device ladders (a probe with no model), engine-level fits, and the share the operating system's file cache serves.

### Device rates

| drive | rate | setup | date | topic |
|---|---|---|---|---|
| M4, internal | **3.4-3.6 GB/s** at any record size (0.5-16 MB) and queue depth (1-64); 3.53 at the engine's point (1.47 MB records, depth 8) | F_NOCACHE, random offsets in an expert file | 2026-09-05 | |
| M4, internal | 3.42 GB/s and 0.227 ms a read | the launcher's machine probe | 2026-10-26 | launcher |
| M4, external Thunderbolt SSD | **3.38 GB/s** alone (random 1.02 MB records, depth 4; 2.91 at 3.07 MB, depth 4-32); with the internal drive at once 3.64 + 3.39 = **7.02 GB/s** | F_NOCACHE, random, a 1 GB region | 2026-10-13 | ext_ssd_tb |
| M4, external Thunderbolt SSD | 3.37 GB/s | the launcher's machine probe | 2026-10-26 | launcher |
| M4, both drives, whole files | 3.3-3.6 GB/s internal and 2.8-2.9 external, so ~6.5 GB/s together is the ceiling, not 7.02 (the 7.02 came from a 1 GB region) | no model, F_NOCACHE | 2026-10-19 | flash_next/read_rate |
| M4, external USB SSD (an earlier test) | 0.98 GB/s at 3.07 MB records, depth 4-16 (0.90 at depth 1); with the internal drive at once 4.68-4.76 GB/s | 512 GB NVMe in a USB 3.2 Gen 2 (10 Gb/s) enclosure, F_NOCACHE | 2026-10-11 | ext_ssd |
| M4, internal, small reads | **228k IOPS** at 4 KB (depth 64 or more, 935 MB/s; p50 91 µs at depth 1), 170k at 8 KB, 112k at 16 KB, 38k at 64 KB (2.5 GB/s) | C threads with pread, F_NOCACHE, random aligned offsets | 2026-10-12 | ssd_iops |
| M4, internal, 8 KB reads | 74k / 116k / 157k IOPS at depth 8 / 16 / 32; a batch of n 8 KB reads ~109 µs + 9.5 µs each | | 2026-10-22 | kv_select |
| PC, SN770 | random **4.35-4.36 GB/s** at 1 MB records and depth 16-32 (1.59 at depth 1), 4.62 at 3 MB records and depth 8; sequential 5.11 GB/s | O_DIRECT, random offsets in the pack | 2026-10-17 | cuda_pc |
| PC, SN770, many small records | 4.35-4.5 GB/s with 64 records of 620 KB in flight, 16-64 readers; a lone record 0.42-0.44 ms | O_DIRECT | 2026-10-22 | pc_small |
| PC, SN770, the data's age | a fresh copy of the expert file reads 4.65-4.66 GB/s in 64-record waves and 0.41 ms a record (p50); the original copy 3.85 GB/s and 0.56 ms, interleaved; rewriting the pack in place restores 4.71-4.78 GB/s and 0.43 ms | after a reboot, cool drive | 2026-11-05 and 11-06 | pc_hw/queue, read_hide |
| PC, PCIe | the copy engine moves an expert from pinned host memory at 14.3 GB/s, from pageable memory at 10.3 | | 2026-11-06 | cuda_resident |

The SN770 read 17-32% slower on 2026-11-03 to 11-05 than on 2026-10-22 with the same probe and file. The cause was the data's age, not the drive: a fresh copy read at full speed, and the rewritten pack took the coder's 40k-context token from 125.6 to 108.0 ms (1.164x), byte-identical (2026-11-06, topic: read_hide).

### Reader threads and queue depth

- 8 reader threads are the optimum on the M4's drives. 12, 16 and 24 were slower; rechecked 2026-10-21 on both drives with EXL3 waves, 8 / 12 / 16 were equal within 0.3% (topic: exl3_two).
- On the PC, replaying the coder's 40k-context read waves, 16 and 24 readers read at 0.997x and 0.988x of 8 (2026-11-05, topic: pc_hw/queue). At 8 readers the drive is the limit.
- Read-path tuning (queue depth, record size, contiguity, the buffer pool, F_NOCACHE) has nothing to win on the M4's internal SSD, which saturates at the same rate at every point.

### The engine's read cost

| what | machine | value | setup | date | topic |
|---|---|---|---|---|---|
| read cost, fit | M4 | 0.29 ms a wave + 0.29 ms per MB from the SSD (about 3.4 GB/s) + 0.04 ms per MB from the file cache; fit over 279 waves, R² 0.97 | 35B mixed pack, the 2 GB-peak setup (`--mem-peak-gb 2`) and its saved requests | 2026-09-30 | ram2gb |
| file cache | M4 | keeps 5.1-5.8 GB of the 35B's expert file between requests; 80% of read bytes came from it | the same setup | 2026-09-30 | ram2gb |
| effective on miss reads | M4 | about 2-3 GB/s (1.8-1.9 ms per rollback, about 2 experts) | Qwen3-Next-80B-A3B, run-ahead rollbacks | 2026-09-08 | |
| a Flash-Next read wave | M4 | ~0.66 ms + 0.58 ms per expert on the two drives (1 expert 1.07-1.21 ms, 2 experts 1.76-2.00 ms); 27-57% of layers miss nothing | 4-bit pack, 6 GB peak, prose | 2026-10-11 | flash_next/lookahead_replay |
| a Flash-Next EXL3 decode wave | M4 | 0.62 MB records, F_NOCACHE: external drive 0.42 + 0.76 ms an expert, internal 0.30 + 0.62; both drives split ~0.4 + 0.36-0.39 (0.58-0.59 of one drive's time) | recommended Flash-Next command | 2026-10-21 | |
| Flash-Next token, per-pass fit | M4 | token = 59-60 ms + 0.51-0.60 ms per layer that reads + 0.24 ms per MB from the drives + 0.05-0.06 ms per MB from the file cache (R² 0.987-0.998); the file cache serves 19-21% (prose) and 15-16% (code) of the expert bytes asked | 4-bit pack, native reader, 6 GB, cache bias 0.5 / budget 0.2, both drives, 128 tokens, 2 runs | 2026-10-13 | flash_next/pcache_dup |
| file cache under F_NOCACHE | M4 | with the store on half the layers, 46-58% of internal-drive bytes came from RAM; with F_NOCACHE 2-17%, and the token went 231 -> 348 ms (prose) and 269 -> 337 ms (code). Every M4 read number for Flash-Next includes these hits | 4-bit pack, 6 GB, cache bias 0.5 | 2026-10-12 | |
| a Flash-Next wide prefill pass | M4 | reads all 75.5 GB of the 4-bit pack's experts at 5.2 GB/s (internal 3.03, external 2.45), because the reader waited for both drives after every group of 10 experts; without that barrier 6.10-6.13 GB/s (internal ~3.4, external ~2.72) | recommended command | 2026-10-19 | flash_next/ttft_split, flash_next/lm_pipeline |
| reads through the file cache | M4 | each 1 MB guess record read through the page cache while the GPU computes adds ~31 µs to the token (the copy competes for the memory bus); past the cache about half that | Flash-Next, native reader | 2026-10-17 | layer_wait |
| the PC's decode reads | PC | a token asks 480 experts: 363 store hits, 31.7 from the pinned host tier (78.5 MB over PCIe), **85.3 from the NVMe (211 MB)**, about 2.99 GB/s over a 70.7 ms wait (est.) | K4 pack, cache bias 0, the coder setting | 2026-11-04 | ideas_pc_decode |
| the PC's tail reads | PC | K4's 825,600-byte records are not 4 KiB multiples, so buffered reads go through the page cache at 2.89-3.27 GB/s; as page-aligned O_DIRECT spans behind one queue, 3.96-4.22 GB/s, near the drive's 4.35-4.5 | K4, cache bias 0, 32.8k cached | 2026-10-31 | tail_read |

## 3. Models and their layouts

| model | layout | pack and sizes | source, date |
|---|---|---|---|
| **Qwen3.6-35B-A3B**, the main target | 40 layers (30 GatedDeltaNet + 10 attention), 256 experts x top-8 + 1 shared, hidden 2048, expert d_ff 512 | **the 35B mixed pack**: expert file 16.11 GB; layers 0-3 and 24-39 at 4-bit group 32 (1.966 MB an expert), layers 4-23 at 2-bit group 32 (1.180 MB); 3.0 payload bits on average, 4.0 on disk with scales; dense part 1101 MB; one arena slot on every layer 62.9 MB, so 21 slots = 1.32 GB | listed 2026-10-09 |
| Qwen3.6-35B-A3B, as the launcher installs it | the same model | the published 4-bit pack, group 64, with 8-bit embeddings and gates: expert file 18.1 GB + dense 1.39 GB = 19 GB on disk, 20.4 GB to download; about 1.77 MB an expert (computed: 18.1 GB / 10,240 experts) | PC, 2026-11-07, topic: launcher35_pc |
| the 35B's MTP head | one extra prediction layer | 475 MB; resident with `--mtp`, +0.64 GB measured | M4, 2026-10-09, topic: coding_cmd_mtp |
| the 35B's vision tower | depth 27, hidden 1152, patch 16, merge 2: one image token per 32 x 32 pixels | 0.893 GB bf16, inside the 35B mixed pack (the launcher's packer leaves it out, so the launcher's 35B is text-only); on the M4 it takes 0.53 / 0.88 / 1.81 s for 340 / 532 / 950 image tokens, read from the pack in 0.26 s | M4, MLX 0.32.2, 2026-10-26, topic: vision |
| **Qwen3.8-Flash-Next** | 48 layers (36 GatedDeltaNet + 12 attention with a sparse-attention indexer), 512 experts x top-10 + shared, hidden 2560, experts 640 x 2560, hyper-connections, a per-layer n-gram embedding table | **4-bit pack** (mlx-community, 4-bit group 32, routers 8-bit): expert file 75.5 GB (3.07 MB an expert), n-gram table 32.0 GB, dense 3.12 GB | 2026-10-11, topic: flash_next |
| Flash-Next, **3-bit EXL3 experts** (the recommended pack since 2026-10-21) | the same model | expert file 45.8 GB (1.86 MB an expert), built from a 42.6 GiB 3-bit EXL3 release; about 82 GB to download in all; on the M4 read from the internal SSD and a copy on the external Thunderbolt SSD; the PC's copy is bit-identical | topics: exl3, exl3_cuda, exl3_two |
| Flash-Next, K4 | the same model | EXL3 4 bits an expert weight, 60.9 GB, 825,600-byte records; used on the PC as the coder | PC, 2026-10-30 |
| Flash-Next, K6 | the same model | EXL3 6.05 bits, 91.07 GB; read-back error against the 4-bit weights 0.085 against K4's 0.112 | PC, 2026-11-05, topic: coder_diag/t4 |
| Flash-Next, an 8-bit trunk with K4's experts | the same model | trunk 5.565 GB against 3.124; expert store 5,506 MB against 7,235 at the same budget | PC, 2026-11-05 to 11-06, topic: coder_diag/t5 |
| Flash-Next's MTP head | one full-attention layer of the pack's shape with its own 512-expert MoE, then the main lm_head | 1.6 GB of files, 1.63 GB resident, its experts not streamed: 1,087 of 4,619 expert slots at the PC's 16 GB command | PC, 2026-10-26 to 10-27, topic: pc_fifty/p1 |
| Qwen3-Next-80B-A3B | 48 layers, 512 experts x top-10 | 1.77 MB an expert, 43.5 GB of experts; MTP head 0.93 GB | M4, September 2026 |
| Qwen1.5-MoE-A2.7B-Chat | 60 experts in each of 24 layers | 4-bit, 8.5 GB download | M4, 2026-10-26, topic: launcher |
| Qwen3-4B, Qwen3-8B (dense) | dense | MLX 4-bit; Qwen3-4B 2.3 GB | M4 |
| Qwen3-0.6B, Qwen3-1.7B | dense drafts for the Qwen3 dense models | MLX 4-bit | M4 |

Facts about the models that bound what an experiment can show:

- **Flash-Next's KL floor.** The same routing math in another operation order (weights ~1e-7 apart) already gives KL 0.008 and top-1 agreement 95.4% against stock through 48 bf16 layers. A teacher-forced KL bar for any lever on this model should sit above ~0.01 (PC, K4, 768 teacher-forced positions of repo text in Russian, English and Python, 2026-11-03, topic: half_k/debug).
- **Flash-Next's router mass.** The top 5 experts carry 62-70% of the top 10's router mass on average, at least 0.50 (same run). Ranks 9-10 carry 11.5-11.8% (8-14% per layer; M4, 4-bit pack, 2026-10-11).
- **Flash-Next's routing by topic.** 99% of a topic's picks need a median 362-447 of a layer's 512 experts (math 362, code 400, Russian 426, English 447); 95% need 258-356; the topics' 95% sets overlap 0.42-0.71 (Jaccard) and their union is 93% of the experts (cache bias 0, held-out tokens, 2026-10-23, topic: sphere_cover).
- **Consecutive Flash-Next tokens** share 47% of their top-10 per layer (0.16 at layer 0, 0.81 at layer 45; M4, 4-bit pack, 2026-10-11).
- **The router's near-misses predict little.** On Flash-Next (EXL3, cache bias 0, a 40k code prompt) an expert ranked 11-16 is chosen at the next token with probability 0.146, ranks 17-32 with 0.070 (topic: store_score).

## 4. The cost of a decode token

### 4.1 The 35B on the M4 at a 21-slot arena

Instrumented split, 35B mixed pack, default server flags, the 2 GB-peak setup's saved requests, layer-by-layer path, 137 passes, 2026-09-30 (topic: ram2gb). The timers add ~14%.

| term | ms a token | what |
|---|---|---|
| blocking expert reads | 19.2 | 76.7 MB a token (15% of the 503 MB expert demand), 43.5% of it from the file cache, 19 read waves: waves 5.5 + SSD 12.2 + file cache 1.4 |
| host waits for the GPU | 28.9 | 40 evaluations a token: GPU work (~22 ms, est.: bytes 16.5-18.4 + a per-layer constant ~5) plus 40 round trips (~7-11 ms) |
| host, other | 8.3 | graph build, slot claims, Python, ~0.21 ms a layer |
| **wall** | **56.5** | about 49-50 uninstrumented |

In one line: the GPU does ~22 ms of real work in a 50-56 ms token. Reads, round trips and host build exist only because about 19 of 40 layers miss each token.

- Run-ahead at 21 slots: 62.5 ms (GPU wait 23.3, host 19.7, 99 evaluations, ~19 of 40 layers rolled back each token), so the automatic mode settles on layer by layer.
- Multi-row passes at 21 slots: 4 rows forced 185.9 ms and 443 MB, 5 rows 308.0 ms and 765 MB. The union of misses for 2 positions is 1.90x one position (trace replay).

### 4.2 Which 35B speed number is which (21 slots, M4, 35B mixed pack)

| ms a token | workload and setup | date | topic |
|---|---|---|---|
| 48.9 (median, range 38.9-57.0) | the 2 GB-peak setup's requests, warm plain turns, default flags, 10 repeats; 59.5 at 3 repeats | 2026-09-30 | defer_auto |
| 49.7 (median of 11) | a long code review, defaults, no MTP; 50.6 with `--mtp` | 2026-09-30 | decode_policy |
| 50.3 | the same code prompt, the recommended Mac command + `--resident 21`, median gap between tokens | 2026-10-08 | arena_sweep |
| 56.5 | instrumented, section 4.1 | 2026-09-30 | ram2gb |
| 57.2 (1k) / 57.0 (4k) | 1k / 4k summary prompt, the recommended Mac command + `--resident 21`, MLX 0.32.3, after a 1k warm-up, median gap (16.8-17.3 tokens/s) | 2026-10-08 | arena_sweep |
| 70.5 / 67.1 / 71.0 | 1k / 4k / 16k summary, the recommended Mac command without `--mtp`, MLX 0.32.3, the first request of a fresh server (14.19 / 14.9 / 14.08 tokens/s) | 2026-10-08 | llamacpp_vs_ours |
| 63.12 | 30 tool calls, default 21-slot arena, plain decode | 2026-09-28 | |
| 98.7 | long code, run-ahead forced | 2026-09-30 | decode_policy |

- 63 ms is the default before a fix of 2026-09-30: until then the automatic mode never chose layer by layer on this model, and every pass ran ahead (63.5 ms median on warm plain turns).
- 57.2 against 70.5 on the same prompt and command: the llama.cpp comparison ran one request per fresh process, with the arena and the file cache cold (~24.8 GB read). Explained by setup, not re-measured as an A/B.
- Prose costs more than code at every arena size (57.2 against 50.3 ms at 21). Why has not been decomposed.
- At 96 slots (2026-09-11): run-ahead 22.38 ms code and 22.47 prose, sync 33.45 and 32.78.

### 4.3 Arena size (the 35B on the M4)

Sweep, 35B mixed pack, the recommended Mac command + `--resident N`, MLX 0.32.3, 2 repeats, median gap between tokens, sync path at every N, 0 MB of swap-outs (2026-10-08, topic: arena_sweep):

| slots | 1k prompt, ms | 4k prompt, ms | code, ms | peak footprint, GB | blocking rows a token (1k) |
|---|---|---|---|---|---|
| 21 | 57.2 | 57.0 | 50.3 | 4.92 | 213 |
| 29 | 52.5 | 52.6 | 46.5 | 5.36 | 161 |
| 37 | 49.8 | 49.4 | 44.4 | 5.93 | 125 |
| 48 | 45.3 | 45.7 | 41.4 | 5.89 | 86 |
| 56 | 44.1 | 44.0 | 41.0 | 6.46 | 68 |
| 64 | 41.4 | 41.7 | 38.7 | 6.96 | 51 |

Smooth, no knee: ~5-7 ms a token per GB of arena, ~0.1 ms per blocking row saved; 64 slots = 1.38x (24.2 tok/s on the 1k prose prompt, 25.8 on code).

- Offline LRU read cost against arena size (trace replay, no cache bias): 21 slots 1.00, 26 0.90, 29 0.85, 37 0.73, 48 0.61, 64 0.46, 96 0.28.
- Misses per token scale as coverage^-2.07, the same exponent on the 35B and the 80B (fitted on 3-37% coverage); without cache bias there are 14x more misses.
- The adaptive arena from 21 to 64 slots with RAM free settles at 64 slots: 1.405x over a plain 21 (default bias), footprint 6.95-7.02 GB. Capped at 60 slots for the margin to the swap line: 1.435x, footprint 6.66-6.70 GB (2026-10-10, topic: elastic).
- 96 slots cost +4-5 GB of RAM against 21.

### 4.4 Context length (the 35B on the M4)

35B mixed pack, arena 21, 2026-10-11 (topic: attn_share): decode 56.8 / 61.9 / 67.2 ms a token (median) at 1k / 8k / 32k prompt tokens. The 10 full-attention layers take 2.65 / 4.28 / 11.74 ms of it (4.7% / 6.9% / 17.5%), K/V 20 / 164 / 655 MB; the 30 GatedDeltaNet layers take ~12.5 ms at every length. Chunked prefill 6.2 / 26.3 / 128.5 s, footprint peak 4.24 / 4.82 / 5.89 GB.

### 4.5 GPU byte floor (the M4)

- **35B bytes per position** (35B mixed pack, 2026-09-08): GatedDeltaNet 568.4 MB + experts 503.3 + lm_head 286.1 + attention 153.4 + shared experts 70.8 + routers 22.3 = 1604 MB, plus 126 MB of GatedDeltaNet state traffic = 1730 MB. **Floor 16.5-18.4 ms (54-61 tokens/s) at 105-94 GB/s.**
- **80B**: 1183 + 849 = 2032 MB + 201 MB of state, floor 18.6 ms.
- **Measured bandwidth** (chained kernels): the expert gather at the decode shape 94 GB/s, dense quantized matmul 105-119, lm_head 107. A lone evaluation over-reads by 0.22-0.28 ms of launch time.
- **Above the floor**: 118-138 µs per layer per pass (35B and 80B). Host: 4.35 µs per MLX call, ~65 calls a layer on the 35B.
- **A trellis-coded (EXL3) expert gather runs at its bytes, like the affine one** (MLX 0.32.2, no model, Flash-Next shapes, top-10 of 64 resident a layer, 1 row, 2026-10-20, topic: exl3): affine 4-bit 334 µs a layer (92 GB/s), EXL3 3 bits 224 µs (0.67x, 83 GB/s), 4 bits 295 µs (0.88x, 84 GB/s); 2 rows cost 1.87x one row for both. The decode chain is not ALU-bound on the M4: time follows bytes.
- **Dense Qwen3-4B**: stock pipelined step 24.0 ms, 41.7 tok/s (sync 24.9); verify cost c_v(T) = max(24.6, 24.6 + 2.3(T-3)) ms for T rows (2026-09-12).
- A chain of tiny dependent kernels costs ~1.7 µs each on the M4 (2026-10-17, topic: gpu90).

### 4.6 Flash-Next on the M4

| what | value | setup | date | topic |
|---|---|---|---|---|
| memory budget | `--mem-peak-gb 6`: 3.2% of experts held, 53% of routed experts served from memory, 689 MB read a token (565 from the SSD, 124 from the file cache), 289 ms = reads 64% + GPU wait 32% + host 4%. At 8 GB: 5.9% held, 66% served, 495 MB a token | 4-bit pack, cache bias 0, sync path | 2026-10-11 | flash_next |
| store size | at 6 / 7 / 8 GB: prose 145.8 / 129.0 / 112.7 ms, code 201.6 / 168.7 / 141.1; experts read a token 124 / 94 / 76 (prose), 169 / 129 / 100 (code). Each GB of store is ~16 ms of prose and ~30 ms of code; the rest stays ~57-60 ms | 4-bit pack, native reader, cache bias 0.5 / budget 0.2, both drives | 2026-10-11 | flash_next/elastic_store |
| a token with no reads | 56.3 / 55.9 ms of GPU time (Russian / Python) for 4.50 GB moved; floor 44.95 ms at 100 GB/s, so 80% of the floor; 5613 kernels | recommended command, 6 GB | 2026-10-17 | gpu90 |
| where the non-read time goes | 59 ms: GatedDeltaNet 17.2 (projections 13.1 at ~100 GB/s), routed gathers 15.4 (96 GB/s), hyper-connections 10.6, attention + indexer 4.3, lm_head 3.8, router + bias 2.8, shared expert ~1, n-gram table 0.7; ~44 ms of it is the byte floor | 4-bit pack, native reader + cache bias, 6 GB | 2026-10-12 | flash_next/rest_profile |
| an EXL3 decode token | Russian 90.7 ms = reads the GPU waits for 39.2 (61 experts, 114 MB, 18.4 layers with a read) + GPU 51.6 (GatedDeltaNet 17.0, EXL3 experts 10.3, hyper-connections 5.6, attention 4.5, lm_head 3.9, the staging guess 3.2); Python 107.1 = 55.5 + 51.7 | recommended Flash-Next command | 2026-10-21 | |
| a prefill pass | reads every expert once, ~78 GB, in ~20 s from both drives (3.8 GB/s, internal alone 3.2); up to ~2k tokens the first token is one pass of reads; at 25k the GPU limits it at ~8.4 ms per prompt token | 4-bit pack, recommended command, 6 GB, layer-major prefill | 2026-10-12 | flash_next/prefill |
| prefill compute | ~1.95 TFLOPS (8k: 54.4 s of compute, 13.3 GFLOP a token); a pass holds ~146 KB per position | 4-bit pack | 2026-10-13 | flash_next/prefill_gpu |
| long prompts | a pass is GPU-bound with reads hidden, ~12 s + 6.4 ms a position + 163 ns a (position, key); first token at 16k 135 s, 32k 328 s, 64k 847-891 s; the per-key attention term is 15% of 16k, 30% of 32k, 46% of 64k | EXL3, recommended command, MLX 0.32.2, an 8 GB cap | 2026-11-02 | |
| long-context decode | 8k +8 ms a token, 24k +28 ms over a short context | | 2026-10-16 | flash_next/long_decode |

The largest footprint with at least 1 GB of margin to the swap line on an idle M4 is ~9 GB. With other apps open, two 8 GB loads swapped 362 and 384 MB, so 8 GB is not safe there and 6 GB is (2026-10-11, topic: flash_next/spec_probe).

### 4.7 The PC

| what | value | setup | date | topic |
|---|---|---|---|---|
| Flash-Next decode token at the 16 GB command | **32.2 / 30.0 / 39.2 ms** (Russian / English / Python) on the GPU clock = GPU work 20.3 / 21.4 / 20.5 + parked on miss-layer reads 8.9 / 5.2 / 15.5 + head 1.5 / 1.8 / 1.6 + tail 1.4 / 1.5 / 1.5. The host is 0.13-0.42 ms behind, so the GPU sets the pace. 4,194 kernels and 262-268 graph launches a token. The store (4,340 slots) answers 463 of 480 asks; the SSD serves 14.8 / 11.7 / 29.1 experts a token (27.5 / 21.8 / 54.2 MB at 3.1-4.2 GB/s in flight). No lookahead reads on CUDA: a read issues when its layer's router ids exist | EXL3, native reader, `--mem-peak-gb 16 --cache-bias 0.5 --bias-budget 0.2` | 2026-10-26 | pc_fifty |
| the same token, warm, family by family | **25.17 ms** = 21.01 of chain between the 48 waits + 1.83 of parks + 1.06 of head + 1.19 of tail + 0.10. The chain is 17.51 of GPU work + 3.4 of idle; the GPU work is 12.55 at the measured floors (3,826 MB at 411 GB/s plus 0.8 µs a kernel launch) plus 4.96 above them | the same command, Russian, warm store | 2026-10-27 | pc_fifty/p2 |
| a kernel launch | a standalone tiny kernel costs **1.69 µs of GPU and 1.68 µs of CPU** a launch; a custom CUDA kernel costs the host 0.93 µs at 2 arguments, 1.59 at 16, 4.37 at 66, 7.90 at 130 | MLX 0.32.4, no model | 2026-10-27 | pc_fifty/p2 |
| the dense 4-bit matvec lane | 772 tensors and 2,323.7 MB a token (1,868.4 MB packed weights + 455.3 MB bf16 scales and biases); 6.87 ms back to back, 338 GB/s; its floor at 411 GB/s is 5.49 ms | MLX 0.32.4, no model, the model's shapes | 2026-10-27 | pc_fifty/qmv |
| the handshake with the reader | a layer whose experts all hit costs 8.4 µs of handshake on the GPU clock, 0.40 ms a token | the same command, warm store | 2026-10-28 | pc_fifty/p3 |
| the coder's token (K4, 40k context) | 132.6 ms = reads in flight 96.0 (first records 19.5 at 0.483 ms a job, bytes 76.6 at 3.83 GB/s) + GPU 36.6, strictly serial; the drive is idle 0.84 ms before every reading layer (a layer's ids exist only after its router), 32.6 ms a token | K4, cache bias 0 | 2026-11-05 | pc_hw/queue |
| the same token after the pack rewrite | 125.6 -> 108.0 ms (1.164x); 98.1 ms with two more read changes (1.281x); byte-identical | K4, 40k code prompt | 2026-11-06 | read_hide |
| one more store slot a layer | ~119 MB saves ~2 store misses a token at 47-55 slots; each extra NVMe expert costs ~0.85 ms | K4 with an 8-bit trunk, claim logs replayed | 2026-11-07 | coder_diag/gap |
| the 35B held resident (VRAM plus pinned host RAM) | 12.7-12.95 GB of experts in VRAM and 5.1-5.4 GB in the pinned host tier, 0-216 of 10,240 left to the SSD; the fill reads 18.2 GB at 3.4 GB/s in 5.2-5.5 s; decode 73-79 tokens/s after a short prompt, 56-71 after 8k | a Q4_0 GGUF-derived 35B pack, `MOE_RESIDENT=1` | 2026-11-06 | cuda_resident |
| its decode token | 10.41 ms with the exact decode fixes (embedding in VRAM, router and combine as two kernels) against 12.67 before; 1,892 -> 1,424 kernels a token. llama.cpp b11719 on the same card: 9.86 ms (6.7 of GPU work in 1,482 kernels + 3.1 of CPU experts) | the same pack | 2026-11-06 | decode35_cuda |
| the launcher's Q4_0 recipe for the 35B (what it installs on CUDA since 2026-11-11) | 146.5 / 147.3 tokens/s on short turns (turns 2 and 3, pass 1 / pass 2), 100.4 / 106.0 after an 8k prompt; reads an 8k prompt at 933 / 935 tokens/s, first token 8.67 / 8.65 s; with exact routing (cache bias 0, no head) 115.2 / 114.8, 91.6 / 91.7, 809 / 844 tokens/s | the Q4_0 GGUF's experts (18.20 GB after the repack) held resident across VRAM and a pinned host-RAM tier (`MOE_RESIDENT=1`), cache bias 0.5 at budget 0.2, the model's own MTP head (a 475 MB pack); against the published 4-bit pack, GSM8K-400 331 against 338 (p 0.25), HumanEval-164 149 against 148; installed in 6 min 25 s with no manual step; the launcher's own start within 1.7% of the hand-made command | 2026-11-11 | pc_vs_llama_1111, launcher_q40_pc, q40_quality_pc |
| the launcher's 4-bit 35B (what it installed on the PC before 2026-11-11) | 61 to 70 tokens/s after a short prompt, 50 to 54 after an 8k prompt; reads an 8k prompt at 306 to 324 tokens/s; process peak 2.4-3.2 GB, 9.5-10.1 GB of VRAM in use (the desktop's ~0.6 GB included) | the launcher's own plan through `./larder` with a fresh home folder: the pack's 18.1 GB of experts resident in VRAM plus a pinned host-RAM tier (`MOE_RESIDENT=1`), cache bias 0.5 (gated on the M4's mixed pack, not on this pack), two repeats | 2026-11-07 | launcher35_pc |
| dense Qwen3-8B | decode 80.1 tok/s (128 tokens after a 512-token prompt), prefill 2,388 tok/s, 6.0 GB of VRAM; llama.cpp on a Q4_0 of the same model 83.3 / 87.7 tok/s (flash attention off / on) | plain MLX on mlx-community's 4-bit weights, MLX 0.32.4, median of 3 | 2026-10-18 | pc_kernels |

Facts about MLX's CUDA backend that shaped the PC's design (MLX 0.32.4):

- Its quantized gather offset a gathered expert in 32 bits: with Flash-Next's 640 x 2560 experts every slot from 2622 on read other weights (ml-explore/mlx#4726, 2026-10-19). Metal's offsets are 64-bit.
- It runs GatedDeltaNet through a per-position loop of fp32 steps (Metal has a fused kernel). One such layer grows 1021 / 2459 / 3790 / 7575 MB at 205 / 410 / 820 / 2012 positions, which caps a prefill pass at 205 positions under a 13 GB cap (2026-10-19).
- Within a 32k prefill pass its memory pool reserves ~2x MLX's buffers (12.75 GB for ~6.5), and the driver's free-memory call counts freed pool memory as used (2026-10-25).
- A one-thread kernel spinning on a word in mapped pinned memory holds the rest of a graph until the host writes it: round trip 1.35 µs, 1.9 µs with ten ids (2026-10-21). This is how the native reader hands experts to the GPU on CUDA.

## 5. Memory on macOS (the M4)

The engine's guard aborts any run above 11 GB of footprint, on +40 MB of swap-outs, or when the margin to the swap line falls under 1 GB. These facts set that margin:

- `kern.memorystatus_level` counts only non-compressed memory N (active + inactive + free + speculative). With C the pages the compressor occupies, pressure level 2 came at N = C and the first swap-out at N = 0.70 C, 0.94 GB of extra load later, while the level still read 4.81 GB. The margin N - 2C/3 fell ~1.8 GB per GB of load and reached ~0 at the first swap-out (macOS 26.6.2, one test under the 21-slot arena, 2026-10-10, topic: elastic).
- macOS swapped while that level still reported 5.84 GB available, so it overstates safe headroom (2026-10-10).
- macOS un-wires an idle MLX server's GPU memory 1.1-2.1 s after a reply: wired memory falls 8.1-8.8 -> 1.68-1.71 GB and the margin reads 13.5-14.4 GB instead of 6.0-7.4 while decoding. An idle reading is no reference for a pass's own memory (Flash-Next and the 35B, 2026-10-18, topics: flash_next/store_hold, arena_idle).
- macOS takes a dropped MLX buffer's pages back 60-255 ms after MLX lets it go, whatever its size (1 or 4 GB); a buffer allocated meanwhile stands on top of them in the footprint. A rebuilt expert store went 1.0 GB over its budget that way (2026-10-13, topic: flash_next/elastic_cap).
- The margin does not see a freshly loaded model until the GPU first runs it: with Flash-Next it read 11.4-11.9 GB after load and fell 5.5-5.9 GB in the first prefill (12 of 12 runs, 2026-10-13, topic: flash_next/elastic_grow).
- Purgeable Metal buffers leave the footprint (2.15 -> 0.00 GB); a state change costs 0.15 µs, and under pressure macOS emptied all 32 (2.15 GB) with 0 MB of swap-outs (no model, 2026-10-09, topic: purgeable).
- The GPU can read stale host-written memory inside one command buffer: a table the host rewrote after an earlier kernel in the same buffer read it was seen with the old values in 3000 of 3000 rounds through plain loads, and in 0 through coherent system loads (2026-10-10, topic: native_coherent).
- The M4 runs at the free-list floor (0.01-0.05 GB free, 5.8-8.5 GB of file cache) whatever the margin says; rare swap-out bursts came with a store release at the start of a prompt (2026-10-18, topic: swap_prefill).

## 6. The recommended Mac command

`python3 server.py --model <35B mixed pack> --prefill-step 0 --keep-convs 4 --mtp`

- Implied defaults: an adaptive arena of 21 to 60 slots since 2026-10-11 (before: a fixed 21; with RAM free it starts and stays at 60, footprint ~6.7 GB), `--cache-bias 0.5`, `--bias-budget 0.2`, `--defer-mode auto`.
- First check with `--mtp` (2026-10-09, MLX 0.32.3, topic: coding_cmd_mtp): the head loads (475 MB), code correct, first code reply 19.75 tokens/s, 98% of drafts accepted, peak 3.52 GB.

## 7. Against llama.cpp

The same model, Qwen3.6-35B-A3B, against llama.cpp b11719 (the official prebuilt build) at the best setting of a sweep, greedy (temperature 0), thinking off on both sides. Quality was not compared in these tables (the 2026-11-11 recipe has its own paired gate, listed in 4.7). The M4 comparison and the PC's 2026-11-06 one are not like for like: the two sides ran different quantisations, and on the M4 this engine also used the model's MTP head and read most of its bytes from another drive (the internal SSD; about 11 GB a request came from the same external SSD).

### The M4 (2026-11-06)

- **llama.cpp**: unsloth's `Qwen3.6-35B-A3B-UD-IQ4_XS.gguf` (17.7 GB, a mixed quantisation of about 4.25 bits a weight), on the external USB4 SSD (exFAT), plain decoding; `-c 20480 -np 1`, default micro-batch and flash attention.
- **This engine**: the 35B mixed pack (16.1 GB of experts, about 3 payload bits a weight) on the internal SSD, with a partial copy on the same external SSD that it also read (about 11 GB a request); the recommended Mac command (`--prefill-step 0 --keep-convs 4 --mtp`, cache bias 0.5), so decoding used the MTP head; a fresh server for every prompt size.
- **Prompts**: 1k / 4k / 16k tokens (1,090 / 4,162 / 16,450 with the chat template) of repository text with a passphrase in the middle, 200 reply tokens; every reply found the passphrase.
- **One run per cell**, no repeats, and the file cache could not be emptied between arms: treat differences under about 10% as noise.

| arm | first token 1k / 4k / 16k, s | decode 1k / 4k / 16k, tok/s | peak footprint, GB | read per request, GB |
|---|---|---|---|---|
| this engine, the Mac command | 4.2 / 10.7 / 46.4 | 35.8 / 34.4 / 30.2 | 5.7-6.0 | 23-24 |
| this engine, the same with the native agent and native MTP verify forced on (the default since 2026-11-04) | 3.9 / 10.7 / 46.8 | 36.3 / 34.3 / 30.4 | 5.7-6.0 | 23-24 |
| llama.cpp, everything on the GPU (`-ngl 99`) | no reply: the GPU ran out of memory | | | |
| llama.cpp hybrid, `-ngl 99 --n-cpu-moe 32` (the best of N = 24 to 40 and `-ot exps=CPU` at 1k) | 33.2 / 109.4 / 429.3 | 9.1 / 8.1 / 7.9 | 0.9-1.5 (its file pages are not counted) | 27 / 82 / 310 |
| llama.cpp CPU only, `-ngl 0 -t 4` (better than 10 threads) | 39.5 / 142.5 / 643.1 | 11.0 / 9.5 / 7.2 | 3.4-4.0 | 22 / 82 / 359 |

Against llama.cpp's best arm at each size (CPU only at 1k and 4k, hybrid at 16k): decode 3.3x / 3.6x / 3.8x, first token 7.9x / 10x / 9.2x. llama.cpp maps the 17.7 GB file and leaves paging to macOS, so on 16 GB the weights do not stay in memory and a request re-reads 22-30 GB at 1k and 310-360 GB at 16k. This engine reads its routed experts into its own cache, 23-24 GB a request at every size. This engine was not run on the same GGUF: the repack did not fit the free disk.

### The PC (2026-11-11)

The earlier comparison (2026-11-06 and 2026-11-07, below) put llama.cpp ahead of the launcher's then 4-bit setup by 1.4x to 1.9x on decode and 2.6x to 2.7x on reading the prompt. The launcher's 35B on CUDA has since become the Q4_0 recipe, and the comparison was repeated in one session.

- **llama.cpp** (2026-11-11): the same bartowski Q4_0 GGUF, hybrid `-ngl 99 --n-cpu-moe 11 -t 6` for the decode and prompt cells. `--n-cpu-moe 10`, the best setting on 2026-11-09, ran out of memory in the cuBLAS workspace (the desktop held 649 to 687 MiB of the card against 553 to 605 MiB then) and 9 failed to allocate; 11 is the best that fits. At 8k, 11 was faster than the earlier 12 (97.4 against 94.8 tokens/s decode, 891 against 826 reading).
- **This engine** (2026-11-11): the launcher's recipe as its dry run prints it (Q4_0 experts resident, cache bias 0.5 at budget 0.2, the model's MTP head), and the same pack with exact routing (cache bias 0, no head). A fresh server an arm, the pack out of the page cache before each; one warm request, then three 512-token replies to the short prompt and a 256-token reply after the 8k prompt.
- One alternation in both orders (pass 1 / pass 2); a cell is that order's value, no best-of. Quality was not compared.

| | recipe | exact routing | llama.cpp, `--n-cpu-moe 11` | recipe / llama.cpp | exact / llama.cpp |
|---|---|---|---|---|---|
| decode on short turns (turns 2 and 3), tokens/s | 146.5 / 147.3 | 115.2 / 114.8 | 99.6 / 100.7 | 1.47x / 1.46x | 1.16x / 1.14x |
| decode, 256 tokens after an 8k prompt, tokens/s | 100.4 / 106.0 | 91.6 / 91.7 | 97.4 / 97.3 | 1.03x / 1.09x | 0.94x / 0.94x |
| reading the 8k prompt, tokens/s | 933 / 935 | 809 / 844 | 891 / 904 | 1.05x / 1.03x | 0.91x / 0.93x |
| first token after the 8k prompt, s | 8.67 / 8.65 | 9.99 / 9.58 | 9.07 / 8.94 | | |
| first token after the short prompt (turn 1), s | 1.06 / 1.00 | 0.40 / 0.35 | 1.31 / 1.33 | | |

Differences under 10% are level: the recipe after 8k and exact routing's 6% to 9% behind are stated with their signs, not as wins or losses. This engine's short prompt was 144 tokens against llama.cpp's 132, its first token includes the HTTP round trip, its 8k reply followed three short turns on the same server while llama.cpp's was a fresh process, and llama.cpp's expert placement is fixed by `--n-cpu-moe` while this engine's store sizes itself from the free VRAM. The recipe's replies differ run to run (cache bias 0.5 makes routing depend on the store's state); exact routing's short replies were identical in all six turns. Against `--n-cpu-moe 10` at the earlier 102.4 to 103.2 tokens/s the recipe would be 1.42x to 1.44x and exact routing 1.11x to 1.12x on short turns. Evidence: pc_vs_llama_1111 (private); the recipe's gate and install are in q40_quality_pc and launcher_q40_pc.

### The PC (2026-11-06 and 2026-11-07, history: the launcher's 4-bit setup)

- **llama.cpp** (2026-11-06): bartowski's `Qwen_Qwen3.6-35B-A3B-Q4_0.gguf` (20.8 GB, experts in Q4_0 / Q4_1, about 4.5 bits a weight), hybrid `-ngl 99 --n-cpu-moe N -t 6`.
- **This engine** (2026-11-07): the launcher's published 4-bit pack (group 64, 8-bit embeddings and gates), started by `./larder` with a fresh home folder: experts resident in VRAM plus a pinned host-RAM tier, cache bias 0.5 (its paired gate ran on the M4's mixed pack, not on this pack on CUDA), no MTP head. One fresh server a prompt, two repeats.

| | llama.cpp, best hybrid | this engine, repeat 1 / 2 | llama.cpp faster by |
|---|---|---|---|
| decode, 128 tokens after a short prompt (llama.cpp 132 tokens, this engine 144 with the template) | 99.73 (N = 10) | 70.13 / 60.68 | 1.4x / 1.6x |
| decode, 64 tokens after an 8k prompt (8,074 / 8,086 tokens) | 93.36 (N = 12) | 49.80 / 53.85 | 1.9x / 1.7x |
| reading the 8k prompt, tok/s | 839.7 (9.6 s) | 306 / 324 (first token 26.4 / 25.0 s) | 2.7x / 2.6x |

On the M4's prompts (1k / 4k / 16k, 200 tokens) the PC's launcher 35B decoded at 61.4-63.2 / 60.8-63.3 / 54.1-56.7 tok/s (repeat 1 / 2); llama.cpp was not run at those sizes on the PC. The first short repeat included a fresh home folder's first-request costs (first token 12.0 s), which is why the two short repeats differ by 15%.

## 8. Open questions

- **Arena 64 against 21.** An early engine measured 1.05-1.19x at 64 slots and 1.12x at 96 (relative to the cache-bias rows of that time); the 2026-10-08 sweep measured 1.38x at 64. Different engines and baselines; no rerun on a common setup.
- **The uninstrumented split at 21 slots**: only the total (~49-50 ms on code) is known.
- **Why prose costs more than code** at the same arena (57.2 against 50.3 ms at 21).
- **No GPU timeline of a 21-slot 35B token**: per-sync latency and host build are not separated.
