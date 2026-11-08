# Frequently asked questions

Short answers to the questions that come up most often. The main README has the measured speeds; `docs/HOW_IT_WORKS.md` explains the machinery. Every number here is in `research/` with its machine, setup and date.

## Why is the first reply slow?

Before the model can write one word, it has to read the whole prompt. This is a pass over every prompt token through every layer, and it reads far more expert weights than a single reply token does. On the 16 GB M4 MacBook Pro, a 16k-token prompt took 45.6 s with the 35B's mixed research pack (2026-10-07) and 136.2 s with Flash-Next (2026-10-25); on the RTX 5060 Ti PC, the same prompt took 34.4 s with Flash-Next (2026-10-26). A short question is much quicker: 2.5 to 4.5 s for a question of 19 to 54 tokens with Flash-Next on the M4 (2026-10-14).

The second message of the same conversation is fast again, because the server keeps the conversation's state and reads only the new part.

## Why does it read from the SSD?

The model is bigger than the memory of an ordinary computer. The published 4-bit Flash-Next pack has 75.5 GB of expert weights, and the recommended 3-bit EXL3 pack has 45.8 GB (2026-10-21). A token uses only a few experts in each layer, so the engine keeps the small always-needed part in RAM and reads the chosen experts from disk as they are needed. On the 16 GB M4 with Flash-Next at a 6 GB expert cache, 53% to 66% of the experts a token asked for came from memory, and 64% of the token's time was waiting for the rest to arrive from the drive (565 MB per token at about 3.4 GB/s, 2026-10-11).

A fast drive helps: the M4's internal SSD measured 3.42 GB/s (2026-10-26), an external Thunderbolt SSD 3.38 GB/s (2026-10-13), and the RTX 5060 Ti PC's NVMe drive 4.35 GB/s in random 1 MB reads (2026-10-17).

## Will streaming from the SSD wear it out?

Reading a file does not wear an SSD the way writing to it does. The engine reads model records and writes only its small cache, logs and state files; it does not rewrite the model. The drive is busy while a model runs, so a fast drive still makes replies faster.

## Can I use my own model?

Yes, if it is an MLX safetensors model and either:

- a mixture-of-experts model with the expert layout the engine can stream (the tested models use `switch_mlp` experts), or
- a dense model that fits in the machine's memory.

`./larder --model publisher/repo` takes a Hugging Face repository, and `./larder --model /path/to/folder` takes a model folder on your computer. The launcher checks the model type, the disk and the memory before it downloads anything, and says why when a model cannot run. A model type that the installed `mlx-lm` does not know is refused.

## Can I use a GGUF model from another program?

No. GGUF, PyTorch `.bin`, GPTQ, AWQ and FP8 checkpoints are not supported; look for an MLX conversion of the model. The engine reads MLX safetensors, and a streamed mixture-of-experts model needs the expert tensors in the layout the packer writes.

## How much memory do I need?

16 GB of RAM is what we tested. The launcher reads the memory that is actually free at that moment, sizes the expert cache from it, and refuses a model whose weights and overhead cannot fit. The smallest tested model, Qwen1.5-MoE, needs the least room of the list.

## How much disk space do I need?

The smallest tested model is 8.5 GB on disk. The launcher keeps 10 GB free for the system after a download and refuses to start if that room is not there. Flash-Next is the big one: an 82 GB download and 45.8 GB of expert weights, so a 16 GB machine with a small internal drive puts the experts on an external SSD.

## Can several requests run at once?

Dense models can. Several requests in flight decode as rows of one forward pass, so the total throughput rises: on the 16 GB M4, Qwen3-4B served 4 requests at once at 1.97x to 2.02x the throughput of running them one after another, and 8 at once at 3.20x to 3.23x; Qwen3-8B gave 1.77x to 1.84x and 2.93x to 2.98x (2026-09-30). A mixture-of-experts model answers one request at a time, because each request needs the expert reader.

## Does it need the internet?

To install, and to list and download models. Opening the menu reads Hugging Face metadata for the tested models that are not on disk yet, and a search queries Hugging Face. A model already on disk starts without the network: the launcher reads its notes from disk, the server never calls out, and the chat page and its libraries are served locally.

## Is my data sent anywhere?

Your prompts and replies stay on your computer. The server listens on the loopback address by default, and it refuses browser requests that arrive under another host name or from another web page. Editor extensions talk to it from the same computer. The engine has no telemetry and no update check.

Other network use: the installer fetches the code and packages, the menu reads model metadata from Hugging Face and downloads the model you pick, `./larder share` listens on your LAN and sends a beacon, and the devices screen scans your LAN only after you confirm it and reaches your own computers over ssh. The README's "Privacy and network" section lists each one, and `SECURITY.md` describes share's pairing and encryption.

## Can it call tools and work in an agent?

The API accepts a `tools` list, passes it to the model's chat template, and turns a tool call back into an OpenAI `tool_calls` reply. How well it works depends on the model; the tested 35B is the strongest of the list at tool calls. A reply to a request with tools arrives whole, not token by token. Continue, Cline and Copilot settings are in `docs/VSCODE.md`.

## Can it read images?

The API and the chat page can pass images to a model that has a vision part, and the page greys out its attach button when the loaded model cannot see. The launcher's packer leaves the vision tower out, so every model in the launcher's tested list is text-only once installed, the 35B included.

Images were tested with the 35B mixed research pack, which keeps the checkpoint's own tower: on the 16 GB M4, a 950-token chart gave its first text in 5.9 s, 2.2 s of it in the vision tower, and a follow-up on the same image gave its first text in 1.06 s (2026-10-26). `tools/stream_pack.py --keep-vision` keeps the tower in a pack you make yourself; that path was not tested end to end. The server turns images off under `--mem-peak-gb`.

## How do I stop it, or choose another model?

`./larder stop` stops the server the launcher started, and `./larder status` shows what is running. `./larder --model <id>` starts another listed model, and `./larder --list` shows the ids and this machine's estimated speed. The launcher starts at port 8800 and moves to the next free port; a server started by hand with `server.py` uses 8765 by default.

## Why do two runs give different replies?

The default is greedy decoding. On the same machine, the same prompt read in the same prefill passes, with the same weights and settings, gives the same words, apart from exact bfloat16 ties that can break either way.

Two runs can still part at a near-tied token. On CUDA, two fresh processes do not compute bit-identical logits: at an 8k prompt their top tokens agreed at 97.3% of positions, KL 0.005 (the 35B at cache bias 0, RTX 5060 Ti PC, 2026-11-06, `research/HARDWARE.md`). On any machine, a prompt split into prefill passes differently can tip a near-tie; conversation reuse and the memory budget (`--mem-peak-gb`) change that split. Two plain runs of a 10-step agent loop differed on 4 of the 10 steps (35B mixed pack, cache bias 0, 16 GB M4, 2026-09-30, `research/WORKED.md`).

A reply can also differ when you change one of the options that is meant to change it: the memory-aware routing used by default on the 35B and Flash-Next, the 3-bit EXL3 experts, a 3-bit copy, sampling, or the Thinking switch. A few default kernels also round differently from the stock ones, such as the Neural Engine path on Flash-Next's long prompts and two CUDA kernels. The README's quality table names each one and the check behind it.

## Can I run it on Windows, an Intel Mac, or an AMD GPU?

No. The supported computers are Apple Silicon Macs (Metal) and Linux x86_64 with an NVIDIA GPU (CUDA). Windows, Intel Macs and AMD or Intel GPUs are not supported.

## Why are there only a few models in the list?

The list holds models that were actually run and measured on the test machines, each pinned to a Hugging Face commit. `./larder --search` searches Hugging Face for more, and `./larder --model` takes any compatible model. Other mixture-of-experts layouts in the same family are listed but marked untested.

## What is the licence?

The project is released under the MIT licence (`LICENSE`). Third-party notices are in `THIRD_PARTY_NOTICES.md`, and each model has its own licence from its publisher, shown in the launcher.

## Does it change or retrain the model?

No. It reads the published weights. It does not train, fine-tune or modify them. A 3-bit copy made by the launcher is a separate set of files, made only when you explicitly choose it.
