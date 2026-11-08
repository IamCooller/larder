# Tests and checks

This folder holds the tests that run without weights (toy tests), the tests that need a checkpoint (model tests) and the browser tests. Two tools in `tools/` check the claims that a unit test cannot: that a change does not alter the replies (`golden_check.py`) and that a lossy lever does not lower quality (`quality_gate.py`).

## Run everything that needs no weights

```
python3 tests/run_toys.py              # every toy test, one child process each, one line per test
python3 tests/run_toys.py store tui    # only the tests whose file name contains one of these words
```

The list is `tests/toys.txt`, one path per line. A line that says `skipped` means the test needs something this machine lacks and names it: a tokenizer, a CUDA device, a built extension. A skip is not a pass. To run the tests that skip for a tokenizer, point them at any Qwen checkpoint directory (`test_persist.py` builds its own tiny pack from it with `synth_minimax.py`):

```
MOE_TOY_TOKENIZER=<a Qwen3 checkpoint directory> python3 tests/run_toys.py
```

The toy tests that need `qwen4_exp` need mlx-lm from the pull request named in `requirements.txt`. Tests marked "runs on a CUDA PC" do their real work on a CUDA machine and skip on a Mac.

## Model tests

These load weights. Each takes the checkpoint directory as its first argument or from `MOE_TEST_MODEL`, and prints one line saying what is missing when it is absent. Run them one at a time, never beside another model process.

```
python3 tests/test_demo.py CKPT                          # any chat checkpoint
python3 tests/test_server_demo.py CKPT
python3 tests/test_tools_e2e.py CKPT                     # a checkpoint that calls tools
python3 synth_qwen4.py --out /tmp/tiny_q4 --tokenizer-from CKPT   # a tiny random pack, then:
python3 tests/test_ple.py /tmp/tiny_q4
python3 tests/test_native_qwen4.py /tmp/tiny_q4-slim
```


## The two checks

The Quality section of the main README gives two rules: a reply is kept only if it is the model's own choice up to exact bfloat16 ties, or, for a lever that may change replies, a paired test on GSM8K and HumanEval finds no significant difference. These tools enforce them against any OpenAI-compatible server, including this one.

**No-regression check.** Start the server you trust and record its greedy replies to a fixed prompt set (plain chat, code, a second turn, a system message, a tool call). Start the new build with the same model and flags and check it against the record. The first field that differs is printed and the exit code is 2.

```
python3 server.py --model CKPT --port 8790          # the build you trust
python3 tools/golden_check.py record --url http://127.0.0.1:8790/v1 --out golden.json
# stop it, start the new build on the same port, then:
python3 tools/golden_check.py check golden.json --url http://127.0.0.1:8790/v1
python3 tools/golden_check.py diff before.json after.json    # or compare two saved records
```

Both runs need the same flags and a comparable memory state: a speculative arm that is chosen by timing can change a reply without a regression. Self-test: `python3 tests/test_golden_check.py`.

**Paired quality gate.** Get GSM8K `test.jsonl` and `HumanEval.jsonl.gz` from their public repositories (`tools/quality_gate.py --help` names them; the tool downloads nothing). Run both endpoints at once, or one after the other with `--arm` and `--compare` when they do not fit in memory together:

```
python3 tools/quality_gate.py --baseline http://127.0.0.1:8790/v1 --candidate http://127.0.0.1:8791/v1 \
    --gsm8k gsm8k_test.jsonl --humaneval HumanEval.jsonl.gz --n 100 --no-thinking --json gate.json

python3 tools/quality_gate.py --baseline http://127.0.0.1:8790/v1 --gsm8k gsm8k_test.jsonl --n 100 --no-thinking --arm base.json
python3 tools/quality_gate.py --baseline http://127.0.0.1:8791/v1 --gsm8k gsm8k_test.jsonl --n 100 --no-thinking --arm cand.json
python3 tools/quality_gate.py --compare base.json cand.json
```

It sends every item greedy to both endpoints, scores it (GSM8K by the number after `####`, HumanEval by running the model's code against the problem's tests in a child process, with no sandbox; this executes model-written code, so run the gate in a container or a throwaway VM), and prints for each benchmark and pooled: right on both, only baseline, only candidate, the exact two-sided McNemar p, and the paired accuracy difference (candidate minus baseline) with its 95% confidence interval (Newcombe method 10). The verdict is PASS when the pooled difference is not significant at 0.05, FAIL otherwise (exit code 2). A pass says nothing about how large a drop the test could miss; read the interval, or give `--margin 0.03` for a non-inferiority verdict: PASS only if the interval's lower bound is above -0.03. With few items nothing is significant; the published numbers use GSM8K 100 to 300 and all 164 HumanEval problems. Self-test: `python3 tests/test_quality_gate.py`.

## What proves what

| Claim in the README | Check |
|---|---|
| A speed lever gives the model's own tokens | the Exactness tests below (bit for bit on toys), then `golden_check.py` on the real model |
| Memory-aware routing and the 3-bit pack do not lower quality | `quality_gate.py`, baseline against candidate |
| Sampling matches plain sampling's distribution | `test_sampling.py` (chi-square against the exact joint distribution) |
| The chat page, the API and editor clients change no reply | `golden_check.py` through each client's endpoint, `test_webchat.py`, `test_anthropic_api.py` |
| Long-context kernels leave the cache bit-identical | `test_qsa_attn.py`, `test_qsa_pool.py`, `test_foldattn.py`, `test_chunkattn.py` |
| Memory stays under the budget | `test_store*.py`, `test_governor.py`, `test_server_memory.py` |

## Tools

| File | What it is |
|---|---|
| `tools/golden_check.py` | record and compare greedy replies, byte for byte (above) |
| `tools/quality_gate.py` | the paired GSM8K and HumanEval gate with the exact McNemar p (above) |
| `tools/gguf_repack.py` | repack a GGUF into the engine's expert layout; `test_gguf_repack.py` checks it |
| `synth_qwen4.py`, `synth_minimax.py` | build a tiny random pack from a real tokenizer, for the model tests |
| `tests/support.py`, `tests/tool_fixture.py` | toy engine, scripted drafter and tool schemas the tests share |
| `tests/run_toys.py` | run all toy tests (above) |
| `tests/toys.txt` | the toy list `run_toys.py` reads, one test path per line |

## Every test

Times are from one run on a 16 GB M-series Mac without a tokenizer or CUDA; a skipped test took only as long as its start-up. "Needs a build" means `native/build.sh` or `cpu_ext/build.sh` was run for this python.

### Exactness: every speed lever must give the model's own answer

| File | What it checks | Kind | Needs | Time | Run |
|---|---|---|---|---|---|
| `test_foldattn.py` | Folded GQA attention for multi-row verify passes against stock SDPA and an fp32 reference. | toy | - | 13 s | `python3 tests/test_foldattn.py` |
| `test_chunkattn.py` | Query-chunked prefill attention for head dim 256 against the full computation, numerics and peak memory. | toy | - | under 10 s | `python3 tests/test_chunkattn.py` |
| `test_qsa_attn.py` | QSA picks-only decode attention equals MLX's masked SDPA over the picks, bit for bit. | toy, needs mlx-lm with qwen4_exp | - | under 10 s | `python3 tests/test_qsa_attn.py` |
| `test_qsa_pool.py` | The QSA indexer's kept pooled keys equal stock's re-pooled keys. | toy, needs mlx-lm with qwen4_exp | - | under 10 s | `python3 tests/test_qsa_pool.py` |
| `test_layer_glue.py` | layer_glue kernels equal the stock lines they replace, bit for bit, at Flash-Next shapes. | toy, needs mlx-lm with qwen4_exp | - | under 10 s | `python3 tests/test_layer_glue.py` |
| `test_prefill_glue.py` | The prefill-row glue kernels equal the stock lines over 1-2048 rows, each with a mutant that must be caught. | toy, needs mlx-lm with qwen4_exp | - | under 10 s | `python3 tests/test_prefill_glue.py` |
| `test_route_kernel.py` | The fused routing kernel picks the same experts in the same order with bit-equal weights. | toy | - | under 10 s | `python3 tests/test_route_kernel.py` |
| `test_route_k.py` | MOE_ROUTE_K: unset equals k and k1 = k2 = k, bit for bit. | toy, needs mlx-lm with qwen4_exp | - | under 10 s | `python3 tests/test_route_k.py` |
| `test_exl3_dense_fast.py` | The EXL3 dense path's fast kernels write what today's kernels write. | toy | - | under 10 s | `python3 tests/test_exl3_dense_fast.py` |
| `test_ane_experts.py` | ANE expert programs' GPU kernels against fp32 GEMMs, to fp16 precision. | toy | - | under 10 s | `python3 tests/test_ane_experts.py` |
| `test_gguf_repack.py` | tools/gguf_repack.py on a synthetic two-shard GGUF: records, layout, resume. | toy | - | under 10 s | `python3 tests/test_gguf_repack.py` |
| `test_gguf_engine.py` | GGUF routed experts through the engine equal the reference model's tokens on a toy MoE. | toy | - | 10 s | `python3 tests/test_gguf_engine.py` |
| `test_cpu_guess.py` | The CPU routing guess (native/cpu_guess.h) against numpy and MLX. | toy, needs a build | clang++ | under 10 s | `python3 tests/test_cpu_guess.py` |
| `test_cpu_kern.py` | cpu_kern against a float64 numpy reference and MLX's CPU ops. | toy, needs a build | cpu_ext/build.sh | under 10 s | `python3 tests/test_cpu_kern.py` |
| `test_kv_compat.py` | KV cache keys and values round-trip through mlx-lm's two-value and three-value state setters. | toy | - | 10 s | `python3 tests/test_kv_compat.py` |
| `test_bias_qwen4.py` | CacheBiasedMoE equals the stock sparse block at lambda 0 and only re-ranks near ties above it. | toy, needs mlx-lm with qwen4_exp | - | under 10 s | `python3 tests/test_bias_qwen4.py` |
| `test_kv_host_emu.py` | MOE_KV_HOST's homed cache against the flag-off cache, its five CUDA kernels emulated. | toy, needs mlx-lm with qwen4_exp | - | 17 s | `python3 tests/test_kv_host_emu.py` |

### Engine behaviour: speculation, caches, prefill, tools, tokenizer

| File | What it checks | Kind | Needs | Time | Run |
|---|---|---|---|---|---|
| `test_batching.py` | The server's batch loop on a toy dense target: batched replies equal single replies. | toy, needs a tokenizer | - | under 10 s | `MOE_TOY_TOKENIZER=CKPT python3 tests/test_batching.py` |
| `test_spec_rounds.py` | The speculator's round contract: whatever the draft guesses, the cache and the reply follow the target. | toy | - | under 10 s | `python3 tests/test_spec_rounds.py` |
| `test_mtp_seam.py` | mtp= through the whole engine on a toy model and a scripted draft. | toy | - | 15 s | `python3 tests/test_mtp_seam.py` |
| `test_mtp_next.py` | The Qwen3-Next head path of mtp_pack.py and mtp.py on a tiny synthetic head. | toy | - | 14 s | `python3 tests/test_mtp_next.py` |
| `test_sampling.py` | The sampled speculative round against the target's exact joint distribution (chi-square). | toy, needs a tokenizer | - | 75 s | `MOE_TOY_TOKENIZER=CKPT python3 tests/test_sampling.py` |
| `test_forced_rows.py` | Grammar-forced verify rows: proposers, gating and exactness on toys. | toy, needs a tokenizer | - | 17 s | `MOE_TOY_TOKENIZER=CKPT python3 tests/test_forced_rows.py` |
| `test_draft_auto.py` | --draft auto discovery and the auto-draft arm on a fake models tree. | toy | - | 11 s | `python3 tests/test_draft_auto.py` |
| `test_window_cache.py` | The draft-only sink+window KV cache. | toy, needs a tokenizer | - | under 10 s | `MOE_TOY_TOKENIZER=CKPT python3 tests/test_window_cache.py` |
| `test_defer.py` | The run-ahead (--defer) wrappers: rollback and last-layer handling, no model. | toy | - | under 10 s | `python3 tests/test_defer.py` |
| `test_reuse.py` | Conversation reuse: only what an agent added is prefilled, replies unchanged. | toy, needs a tokenizer | - | under 10 s | `MOE_TOY_TOKENIZER=CKPT python3 tests/test_reuse.py` |
| `test_shadow.py` | Shadow render: the history the next turn re-renders is prefilled ahead. | toy, needs a tokenizer | - | under 10 s | `MOE_TOY_TOKENIZER=CKPT python3 tests/test_shadow.py` |
| `test_stub_modes.py` | The four think-stub modes on the toy engine and a tokenizer. | toy, needs a tokenizer | - | under 10 s | `MOE_TOY_TOKENIZER=CKPT python3 tests/test_stub_modes.py` |
| `test_empty_stream.py` | A turn whose decode yields no token still ends the turn. | toy, needs a tokenizer | - | under 10 s | `MOE_TOY_TOKENIZER=CKPT python3 tests/test_empty_stream.py` |
| `test_stop_abandoned.py` | A client that hangs up on a streamed turn stops the decode. | toy, needs a tokenizer | - | under 10 s | `MOE_TOY_TOKENIZER=CKPT python3 tests/test_stop_abandoned.py` |
| `test_think_budget.py` | The thinking budget on scripted samplers, the toy engine and the routes. | toy, needs a tokenizer | - | under 10 s | `MOE_TOY_TOKENIZER=CKPT python3 tests/test_think_budget.py` |
| `test_prefill_logits.py` | Prefill does not materialize every position's logits. | toy | - | under 10 s | `python3 tests/test_prefill_logits.py` |
| `test_prefill_mem.py` | Prefill memory levers are exact on toys. | toy, needs a tokenizer | - | 24 s | `MOE_TOY_TOKENIZER=CKPT python3 tests/test_prefill_mem.py` |
| `test_layer_prefill.py` | Layer-major prefill under a budget store equals the plain path. | toy | - | 10 s | `python3 tests/test_layer_prefill.py` |
| `test_readpath.py` | Prefill read path: a layer read ahead or only its gathered rows give the same set. | toy | - | under 10 s | `python3 tests/test_readpath.py` |
| `test_union_lm.py` | Layer-major read path (_LMReader): whole layer or union only, same buffers. | toy | - | 35 s | `python3 tests/test_union_lm.py` |
| `test_lm_copy_whole.py` | MOE_LM_COPY_WHOLE: a layer-major layer read whole equals the default read. | toy | - | 22 s | `python3 tests/test_lm_copy_whole.py` |
| `test_lm_rule.py` | The whole-layer rule is unchanged when the reader is not queued and corrected when it is. | toy | - | 10 s | `python3 tests/test_lm_rule.py` |
| `test_metal_plan.py` | Metal's long-prefill plan: slope, piece rule and chunk widths from a recorded cold history. | toy | - | 13 s | `python3 tests/test_metal_plan.py` |
| `test_plan_cuda.py` | CUDA's prefill plan replayed offline: two fresh passes must not set the slope. | toy | - | under 10 s | `python3 tests/test_plan_cuda.py` |
| `test_embed_stream.py` | Streamed embedding rows equal the resident table; tokens unchanged. | toy | - | 18 s | `python3 tests/test_embed_stream.py` |
| `test_detok.py` | The Rust streaming detokenizer emits exactly mlx-lm's BPE streaming text. | toy, needs a tokenizer | - | under 10 s | `MOE_TOY_TOKENIZER=CKPT python3 tests/test_detok.py` |
| `test_content_specials.py` | Client text that spells a control token reaches the model as text. | toy, needs a tokenizer | - | under 10 s | `MOE_TOY_TOKENIZER=CKPT python3 tests/test_content_specials.py` |
| `test_tools_parse.py` | tools.parse on 38 replies a coding agent breaks on (cut calls, bare blocks, quoted values). | toy | - | under 10 s | `python3 tests/test_tools_parse.py` |
| `test_vision.py` | Image patches and the vision tower against transformers, fp32 on CPU. | toy, needs a tokenizer | transformers, torch | under 10 s | `MOE_TOY_TOKENIZER=CKPT python3 tests/test_vision.py` |
| `test_memory_core.py` | The --memory path: GapKVCache, probes and routes, no checkpoint. | toy | - | 25 s | `python3 tests/test_memory_core.py` |

### Memory and storage: stores, budgets, packs, stored fronts

| File | What it checks | Kind | Needs | Time | Run |
|---|---|---|---|---|---|
| `test_store.py` | Budget mode: the shared expert store equals the per-layer arena and stock mlx-lm on a toy packed MoE. | toy | - | 30 s | `python3 tests/test_store.py` |
| `test_store_cap.py` | gather_slots at Flash-Next shapes; quotas keep a store under its slot cap. | toy | - | under 10 s | `python3 tests/test_store_cap.py` |
| `test_store_keep.py` | MOE_STORE_KEEP over three turns on one cache. | toy | - | under 10 s | `python3 tests/test_store_keep.py` |
| `test_store_resize.py` | The elastic store resized mid-generation keeps greedy tokens. | toy | - | 13 s | `python3 tests/test_store_resize.py` |
| `test_claim_slots.py` | One slot per expert: no two experts share a slot at any width. | toy | - | under 10 s | `python3 tests/test_claim_slots.py` |
| `test_governor.py` | The governor's size decisions (the floor-below-top_k bug), no weights. | toy | - | 12 s | `python3 tests/test_governor.py` |
| `test_membudget_host.py` | MemBudget's saved host term never lowers on a shorter process. | toy | - | 10 s | `python3 tests/test_membudget_host.py` |
| `test_modelhost_mem_pct.py` | ModelHost forwards mem_pct and host_gb to the engine. | toy | - | under 10 s | `python3 tests/test_modelhost_mem_pct.py` |
| `test_server_memory.py` | server.py --memory plumbing, routes and completion fields. | toy, needs a tokenizer | - | under 10 s | `MOE_TOY_TOKENIZER=CKPT python3 tests/test_server_memory.py` |
| `test_front_ssd.py` | The stored front: an exact SSD round trip within and across processes. | toy, needs a tokenizer | - | under 10 s | `MOE_TOY_TOKENIZER=CKPT python3 tests/test_front_ssd.py` |
| `test_fronts.py` | The stored fronts' index: the directory is the index. | toy | - | 17 s | `python3 tests/test_fronts.py` |
| `test_persist.py` | A restarted server restores its stored front from SSD and plans without the probe (builds a tiny pack with synth_minimax.py). | toy, needs a tokenizer | - | under 10 s | `MOE_TOY_TOKENIZER=CKPT python3 tests/test_persist.py` |
| `test_stripe.py` | A sidecar on a second drive: mirror, wave reads, same bytes. | toy | - | under 10 s | `python3 tests/test_stripe.py` |
| `test_safetensors_map.py` | The engine's safetensors index reads exactly what mx.load reads. | toy | - | under 10 s | `python3 tests/test_safetensors_map.py` |
| `test_cuda_adaptive.py` | CUDA's elastic memory without a model or a GPU. | toy | - | under 10 s | `python3 tests/test_cuda_adaptive.py` |
| `test_cuda_resident.py` | MOE_RESIDENT without a model or a GPU. | toy | - | under 10 s | `python3 tests/test_cuda_resident.py` |
| `test_cuda_spans.py` | MOE_CUDA_SPANS: page-unaligned records read exactly (--pack for the PC run). | toy | - | under 10 s | `python3 tests/test_cuda_spans.py` |
| `test_resident_units.py` | MOE_RESIDENT's GGUF prefill units and the out-of-memory store build. | toy | - | under 10 s | `python3 tests/test_resident_units.py` |
| `test_autotune_first.py` | tuner.py: decisions, the cache round trip and the disk probe. | toy | - | under 10 s | `python3 tests/test_autotune_first.py` |
| `test_pack_refresh.py` | pack_refresh.rewrite: identical bytes after a rewrite, resume after a stop. | toy | - | under 10 s | `python3 tests/test_pack_refresh.py` |

### Native and CUDA paths

| File | What it checks | Kind | Needs | Time | Run |
|---|---|---|---|---|---|
| `test_native.py` | MOE_NATIVE=1 against the Python path: tokens and logprobs bit-equal, LRU and ring state equal. | toy, needs a build | native/build.sh | 13 s | `python3 tests/test_native.py` |
| `test_native_coherent.py` | wait_for hands the gather the slot table the agent just wrote. | toy, needs a build | native/build.sh | under 10 s | `python3 tests/test_native_coherent.py` |
| `test_native_exit.py` | A MOE_NATIVE=1 process cannot leave the GPU spinning after it is gone (macOS). | toy, needs a build | native/build.sh | under 10 s | `python3 tests/test_native_exit.py` |
| `test_native_exit_cuda.py` | The same exit test on NVIDIA, plus a lone KILL. | toy, runs on a CUDA PC | CUDA | under 10 s | `python3 tests/test_native_exit_cuda.py` |
| `test_staging.py` | MOE_STAGING rows in the native agent equal the Python path. | toy, needs a build | native/build.sh | under 10 s | `python3 tests/test_staging.py` |
| `test_pipe_staging_emu.py` | MOE_PIPE_STAGING and PIECES with moe_pipe_cuda built for the host and a stand-in libcuda. | toy, needs a build | g++ | 46 s | `python3 tests/test_pipe_staging_emu.py` |
| `test_gdn_cuda.py` | MOE_CUDA_GDN: the CUDA gated delta kernel; off-CUDA it must never load. | toy | - | under 10 s | `python3 tests/test_gdn_cuda.py` |
| `test_route_cuda.py` | MOE_CUDA_ROUTE: the cache-bias route as one CUDA kernel; off-CUDA it must never load. | toy | - | under 10 s | `python3 tests/test_route_cuda.py` |
| `test_qsa_cuda.py` | The CUDA QSA decode path and its switch point. | toy, runs on a CUDA PC | - | under 10 s | `python3 tests/test_qsa_cuda.py` |
| `test_kv_host.py` | MOE_KV_HOST on CUDA: flag-off equality. | toy, runs on a CUDA PC | - | 14 s | `python3 tests/test_kv_host.py` |

### Server, launcher and front ends

| File | What it checks | Kind | Needs | Time | Run |
|---|---|---|---|---|---|
| `test_engine_args.py` | chat.py and server.py read the engine flags the same way. | toy | - | under 10 s | `python3 tests/test_engine_args.py` |
| `test_quiet_start.py` | The default start-up output of server.py and chat.py. | toy | - | under 10 s | `python3 tests/test_quiet_start.py` |
| `test_webchat.py` | The HTTP layer: chat page, assets, OpenAI-compatible shapes. | toy | - | under 10 s | `python3 tests/test_webchat.py` |
| `test_anthropic_api.py` | /v1/messages and count_tokens through the real handler. | toy | - | under 10 s | `python3 tests/test_anthropic_api.py` |
| `test_share.py` | The share service: pairing code lifecycle and UDP beacon on loopback. | toy | - | under 10 s | `python3 tests/test_share.py` |
| `test_launcher.py` | The launcher's pieces on tiny synthetic checkpoints: catalog, flags, decode estimate. | toy | - | 21 s | `python3 tests/test_launcher.py` |
| `test_tui.py` | The full-screen menu without a terminal. | toy | - | under 10 s | `python3 tests/test_tui.py` |

### Model tests: need a checkpoint, skip with one line without it

| File | What it checks | Kind | Needs | Time | Run |
|---|---|---|---|---|---|
| `test_demo.py` | Turns differ, the KV cache is reused, a system message reaches the model. | model | any chat checkpoint | under 10 s | `python3 tests/test_demo.py CKPT` |
| `test_server_demo.py` | Every HTTP route answers on a real model; a streamed reply arrives in pieces. | model | any chat checkpoint | under 10 s | `python3 tests/test_server_demo.py CKPT` |
| `test_tools_e2e.py` | The model is offered functions, calls one, and the tool block is cached. | model | a tool-calling checkpoint (the 35B) | under 10 s | `python3 tests/test_tools_e2e.py CKPT` |
| `test_ple.py` | Streamed PLE rows equal the model's lookup; tokens unchanged. | model | tiny pack from synth_qwen4.py | under 10 s | `python3 tests/test_ple.py PACK` |
| `test_native_qwen4.py` | MOE_NATIVE=1 on a tiny Flash-Next-shaped pack: logprobs identical. | model | tiny pack from synth_qwen4.py | 12 s | `python3 tests/test_native_qwen4.py PACK` |

### Browser tests

They need headless Chrome and playwright and are not in the toy list. Each starts a stub engine, so no model is involved.

| File | What it checks | Run |
|---|---|---|
| `test_settings_browser.py` | the Settings page: sections, models and variants, a prepare action | `python3 tests/test_settings_browser.py` |
| `test_webchat_browser.py` | the chat page: Thinking switch, images, conversations | `python3 tests/test_webchat_browser.py` |
| `test_status_browser.py` | the Status page: tiles, charts, tooltips, token bar | `python3 tests/test_status_browser.py` |
| `test_chat_features_browser.py` | the context bar, Compact and related features | `python3 tests/test_chat_features_browser.py` |
| `test_a11y_browser.py` | names, language, skip link and focus on the three pages | `python3 tests/test_a11y_browser.py` |

