# The launcher: one command from nothing to a running model

```bash
./larder
```

That is the whole thing. `larder` is a small shell script (it uses `.venv/bin/python` when the
repo has one, else `python3`) that runs `launch.py`. In a terminal it opens the full-screen menu
(see "The menu" below): the first run measures the computer, lists the tested models with their
variants and the speed estimated for this computer, and the choice is installed, started and
shown with the chat address and the API base. The cursor starts on the model used last time
(`~/.cache/moe-ssd-offload/last.json`), or else on the tested model with the best estimated
speed that fits the disk and the memory (a model already on disk wins over a faster one that
is not), so Enter, Enter starts it. With `--auto`, or when no terminal is attached (a script),
nothing is asked: that same model is prepared if needed, the server starts with this
machine's flags, the web chat opens and a short banner prints: the chat URL, the API base and
how to stop.

From a fresh clone: `python3 -m venv .venv` then `.venv/bin/pip install -r requirements.txt`,
then `./larder`.

```bash
./larder stop        # TERM to the server it started, waits up to 60 s, removes its state file
./larder status      # the running server: chat URL, API base, model, port, log
```

The server is saved in `~/.cache/moe-ssd-offload/server.json`, so `stop` and `status` work from
another terminal; both check that the process is still the `server.py` this launcher started
before touching it. Nothing else is killed.

A checkout may add preset machines through an optional preset script; this tree has none.

## The flags

```bash
./larder --list                      # the list with this machine's speeds, then exit
./larder --list --json               # the same as one JSON object, for another computer's menu
./larder --list --json --no-measure  # ... using cached measurements: runs no GPU or SSD test
./larder --list --json --rescan      # ... after looking at the disks again for model folders
./larder --search "org/repo" --json  # a Hugging Face search (words, owner/name or a link), judged here, as JSON
./larder --delete ID --variant KEY --yes   # delete a model the launcher downloaded or made (see "Deleting")
./larder --model qwen3.6-35b         # skip the menu: a listed id,
./larder --model owner/repo          #   a Hugging Face repo,
./larder --model /path/to/folder     #   or a model folder on this computer
./larder --pick                      # the full menu
./larder --search "qwen3 moe"        # Hugging Face search, filtered to what runs
./larder --models-dir /Volumes/X/models   # put models on another disk
./larder --model ID --no-browser --no-chat   # serve only
./larder --model ID --dry-run        # print the engine command and why, then exit
./larder --model qwen1.5-moe --variant q3    # a 3-bit copy made here (see Variants)
./larder --devices                   # the menu also lists your other computers at start (see The menu)
./larder --volumes                   # also look for model folders on other disks (see Found copies)
./larder share                       # experimental: let the menu on another computer use this one (see Sharing)
```

`./larder` with no flags opens the menu when stdin and stdout are a terminal, and runs `--auto` when
they are not; `--dry-run` also means `--auto`. An explicit `--model`, `--pick`, `--list`,
`--search` or `--auto` takes over. `--text` is the menu as plain text (`1b` to pick a model's
variant b), used automatically where curses is missing (Windows) or the terminal cannot show
the full-screen one. In auto mode nothing is asked: the model is downloaded and prepared when
the recommendation says so.

## The menu

`tui.py`, plain curses (nothing to install). One flow: pick a computer, pick a model it can
run, the model starts there and its chat opens here. It knows nothing about models or ssh: it
shows what `launch.TuiBackend` gives it, so `tests/test_tui.py` drives it key by key on a text
screen with a fake backend, and `python3 tui.py --demo` runs it on made-up computers. Set
`MOE_TUI_FROZEN=1` (or `NO_COLOR`) for a still, colourless screen.

- **Devices**: this computer only, until you ask for more. `r` finds the others: any preset machines
  (from the optional preset script), hosts from `~/.ssh/config`, Tailscale peers, shared computers
  heard by their beacon as `<hostname> (shared)` (see "Sharing this computer"), and a sweep of the
  local /24 (`devices.py`: a TCP attempt to port 22 of each address, then the ARP table and the
  resolver). The sweep runs only after you confirm it. Hosts from the preset script, `~/.ssh/config` and
  Tailscale are checked over ssh (one call each) when devices are turned on, or, with devices off,
  after you confirm the sweep. A host only the sweep found gets no ssh until you open it, and a
  shared computer is never reached over ssh. `./larder --devices` lists all but the
  sweep at start. Each row has its hardware and status in colour: `ready`, `no ssh key` (the row says
  `ssh-copy-id ...`), `engine not installed` (one install line), `running: <model>`, `offline`,
  `ssh open`. Enter opens a device's models; on a shared computer it first asks for the pairing code
  shown there.
- **Models**: the rows are the same as the old Models screen, but computed by that computer: the
  menu runs its own launcher there (`--list --json`), caches the answer and `r` asks again (and
  looks at that computer's disks again). Each variant shows size, status (`ready`, `on disk at
  ~/models/<name>`, `download 20.4 GB`, `~40 min to make ...`, `needs 41 GB free on disk, 12.6 GB
  free` in amber, `can't run: ...` only for memory and hardware), a speed bar and its speed: what
  this computer measured (`13.8 tok/s measured here`, green), a preset's measured range
  (`[13.8-14.2 tok/s measured]`, green) or an estimate (`~17 tok/s (estimate)`, dim). A preset's
  own commands are marked ready because they run a prepared cache directly. Enter starts a ready
  model on that computer at once; a model that needs a download asks first on a computer whose
  launcher can prepare it, and a preset says its command cannot prepare models. `/` searches
  Hugging Face on any computer, `d` deletes, and on this computer `f` takes a folder. The footer
  lists the keys.
- **Confirm**: for a model not prepared here: the download, the room on disk, where it goes,
  the speed, and for a 3-bit copy made here the quality note.
- **Install**: a bar from the bytes on disk against the size expected, the speed and time left,
  and the tools' output under it. The install runs in a child process (`launch.py --prepare`), so
  Esc stops it for real; the same choice resumes where it stopped.
- **Start**: the engine's log while it loads; on another computer its own lines, and the menu
  opens the tunnel here only after that computer says it is serving.
- **Ready**: the local chat address (`http://127.0.0.1:8800/`, the same for every model and
  computer; for another device it is the local end of a tunnel), the API base, the model id and
  the measured speed; `o` opens the browser, `c` chats in the terminal, `s` stops it there (TERM;
  a preset's lock is released), `m` goes back to the devices with it running, `q` stops it and
  quits. A server an earlier `./larder` left running opens on this screen, and `q` leaves it running.
- Esc or Left goes back one level. The footer always lists the keys for the screen.

The exact commands:

- this computer: the same as `./larder` (child process, saved state, `./larder stop`).
- a preset machine (only where a checkout adds a preset script): the script's own command, run in
  one `ssh -tt -L local:127.0.0.1:remote` session, so its GPU lock, its stop trap and TERM behave
  as they do when the script runs directly.
- another computer: `cd ~/larder && ./larder --auto --model <id> --no-browser --port <p>` over
  the same kind of session, stopped afterwards with `./larder stop` there and TERM here.

`./larder --list --json` prints this computer's half of that (machine, models, variants, statuses,
speeds) as one JSON object; `--no-measure` uses cached measurements and runs no GPU or SSD
test. It is what a menu on another computer reads.

The same list, the same installs and the same per-model flags are reachable from the browser:
the web chat's Settings view (`web_settings.py`, endpoints in docs/VSCODE.md) asks
`TuiBackend` for the models and runs an install as the same `launch.py --prepare` child, then
switches a model through `POST /admin/load` with `engine_plan`'s flags. A person with the chat
open needs no terminal and no flag.

## Models already on the computer, measured speed, deleting and searching

All four work the same in the menu (on any device, through that device's own launcher) and in the
web chat's Settings; `library.py` holds the rules.

- **Found copies.** Besides the launcher's models folder, it looks at the top-level folders of
  `~/.cache` that hold a `config.json`, the Hugging Face cache's snapshots and, with `./larder --volumes`,
  the top of every other volume `machine.py` lists (and its `models/`); reading a removable volume can
  make macOS ask for access, so that part is off by default. A folder is matched to a catalog variant by
  model type, layers, hidden size, experts and the experts' bits (the commonest per-module entry
  of its `quantization`), and EXL3 only with EXL3 (`expert_quant` in its config), so an EXL3 pack
  is never the 4-bit variant. Unfinished folders (a `.pack_state.json`, a shard the index names
  but is missing) and ones made from another here (`.made`) are skipped. A match reads `on disk at
  <path>` and starts without a download; a raw MoE snapshot is packed in place first. The scan is
  kept in `~/.cache/moe-ssd-offload/found.json` for an hour and until the roots change; `r` in the
  menu, `--rescan` and Settings' "Refresh the list" look again.
- **Measured here.** A server that serves a model adds the decode speed of each reply of 64 tokens
  or more to `~/.cache/moe-ssd-offload/measured.json`: under the machine's fingerprint
  (`machine.fingerprint()`) and the model folder (its name and config signature), the median of
  the last 20 replies, their count and the date. A variant finds its record by what its folder
  was, so `measured here: 13.8 tok/s (12 replies, 2026-10-27)` shows for a model that is no longer
  on disk and again after a re-download. A run does not make a model "tested": that is still
  `CURATED`. A remote device returns its own entries in `--list --json`.
- **Deleting.** `d` on a variant that is on disk asks `Delete <name> <variant> at <path> (<size>)?
  type yes`; Settings has a Delete button with a confirm step. Only folders inside the launcher's
  models folder go (what it downloaded or made). A copy elsewhere is left alone and the answer says
  where it is. Never the running model, a folder a made-here copy links into, a link out or a path
  that climbs out. A remote device runs its own `launch.py --delete <id> --variant <key> --yes`
  over the same ssh as the listing.
- **Searching.** `/` asks on any device for words, `owner/name` or a `https://huggingface.co/...`
  link (`/tree/...`, `/blob/...` and query strings are cut); a repo goes straight to its variants.
  The device's own launcher answers (`launch.py --search Q --json`), so the results are judged for
  its memory and disk, and Enter prepares and starts the result there with `--model owner/name`.
  A preset device (its commands in the checkout's own preset script) judges and lists results too, but starts only
  those commands, which hold its GPU lock. Settings has a search box with the same behaviour for
  its own machine.

## What `./larder` picks

Each tested model's published variant is judged for this machine (`machine.settings`, then
`machine.estimate`). Models that cannot run now are dropped with the reason; among the rest, a
prepared model beats one that still needs a download, and within that group the best estimate
wins. A model already on disk is read from its `launcher.json`, so a machine with prepared
models needs no network to choose. When the last model is still on disk and can run, it is
used without looking at the others (`--pick` or `--model` to choose another).

If nothing fits the memory free at that moment, `--dry-run` answers anyway: it prints the choice
for an idle computer and two lines saying that, while a real start stops with the reason.

## What happens

1. **The machine** (`machine.py`): OS, chip, RAM, GPU and VRAM, MLX version, free disk per volume,
   the GPU's read bandwidth on a decode-shaped 4-bit matmul (~1 s), and the read speed of the
   volume the models go on (random expert-sized reads past the page cache on a 1 GB scratch file,
   ~1 s, deleted after). Cached in `~/.cache/moe-ssd-offload/machine-<key>.json`, the key being
   hardware + OS + MLX version: a change of any of them measures again, `--refresh` forces it.
   Free memory (the swap line on macOS, `MemAvailable` on Linux) and free disk are read live on
   every call, so the engine or a tuner can call `machine.profile()` while running.
2. **The list**: tested models (below), a Hugging Face search, or a local folder. Each model is
   built with mlx-lm's own classes from its `config.json`, lazily (no weights, no memory), and the
   engine's `find_switch_mlp` says whether its experts can stream from the SSD. The row shows what
   it is, the speed estimate for this machine, the download and the license; a model that does
   not fit the disk or the memory says why instead.
3. **Download and prepare** (`tools/stream_pack.py`): shard by shard, each shard's routed experts
   appended to `experts.bin` (one record per expert), the rest saved as slim shards, the download
   deleted before the next. The disk never holds more than the result plus two shards. A stopped
   run resumes where it stopped (`.pack_state.json`); every download is pinned to a commit. For
   the EXL3 Flash-Next pack the base repo's dense part is fetched by HTTP range requests without
   its experts (36 GB of 112), then the EXL3 expert shards are packed one by one
   (`repack.pack_shard`).
4. **Settings** (`machine.settings(profile, model)`, a pure function): below.
5. **The server** (`server.py` as a subprocess, its log in `~/.cache/moe-ssd-offload/server-*.log`),
   on the launcher's chat port (8800 by default, the same for every model; starting a model stops
   the previous one the launcher started first). When it answers `/v1/models`, the launcher prints:

```
  Ready: Qwen3.6-35B-A3B

  Web chat     http://127.0.0.1:8800/
  API base     http://127.0.0.1:8800/v1        (OpenAI-compatible, no key needed)
  Model id     /path/to/models/qwen3.6-35b
  Speed        ~25 tok/s (estimate for this machine)

  VS Code with Continue: add this to ~/.continue/config.yaml
    ...
```

In auto mode (`./larder --auto`, or no terminal) the same point prints four lines instead (the chat URL, the API base, the
stop command, the model and its estimate), the browser opens, and there is no terminal chat:
the process stays in the foreground serving until `./larder stop` or Ctrl+C.

## What runs

| | runs | how |
|---|---|---|
| MLX safetensors, MoE, experts at `mlp.switch_mlp` or `block_sparse_moe.switch_mlp`, quantized | yes, any size the disk holds | experts stream from the SSD. Run with this engine during its development: Qwen3.6-35B, Qwen3.8-Flash-Next, Qwen3-Next-80B, Qwen1.5-MoE, MiniMax, Ling, ERNIE-4.5 MoE; other mlx-lm types with this layout (Qwen3-MoE, DeepSeek, GLM-4 MoE, OLMoE, Mixtral, PhiMoE, ...) are listed but untested |
| the same with one tensor per expert (older conversions) | yes | packed into the same records (`tools/stream_pack.py`) |
| dense MLX models, and MoE layouts the engine does not stream (LFM2-MoE, gpt-oss, ...) | if they fit in memory | resident, mlx-lm's own path plus the engine's dense levers |
| Qwen3.8-Flash-Next with 3-bit EXL3 experts | yes (tested) | the curated entry only: EXL3 experts from a converted pack over the 4-bit MLX base |
| GGUF, PyTorch `.bin`, GPTQ / AWQ / FP8 checkpoints (`quantization_config`) | no | the launcher says so; look for an MLX conversion |
| a model type the installed mlx-lm does not know, gated repos | no | filtered out of the search, refused as a folder |

Machines: Apple Silicon with macOS (Metal) and Linux with an NVIDIA GPU (MLX's CUDA backend).

Tested models (`launch.CURATED`, each pinned to a Hugging Face commit):

| id | model | source | settings |
|---|---|---|---|
| `qwen3.6-35b` | Qwen3.6-35B-A3B, 256 experts x 40 layers | `mlx-community/Qwen3.6-35B-A3B-4bit` | Apple: the adaptive expert arena, `--prefill-step 0 --keep-convs 4`, cache bias 0.5. CUDA: id `qwen3.6-35b-q40`, the published trunk + Q4_0 GGUF experts repacked + the model's MTP head, cache bias 0.5 at budget 0.2 (the 4-bit pack stays a variant) |
| `flash-next-exl3` | Qwen3.8-Flash-Next, 512 experts x 48 layers, 3-bit EXL3 experts | `mlx-community/Qwen3.8-Flash-Next-4bit` (dense part) + `beamster/Qwen3.8-Flash-Next-Sushi-3bpw` (experts) | `--mem-peak-gb` / `--mem-peak-max-gb`, cache bias 0.5, the native reader |
| `qwen1.5-moe` | Qwen1.5-MoE-A2.7B-Chat, small | `mlx-community/Qwen1.5-MoE-A2.7B-Chat-4bit` | `--mem-peak-gb`, exact routing |
| `qwen3-8b`, `qwen3-4b` | dense Qwen3 | `mlx-community/Qwen3-8B-4bit`, `-4B-4bit` | resident, a draft model if one is on disk |

## The settings it chooses

`machine.settings(profile, model)` takes only measured inputs: the machine's RAM, the GPU's working
set (MLX's own number), the memory free before the machine would swap now, VRAM, and the model's
sizes.

| case | flags |
|---|---|
| streamed MoE, Apple | `--mem-peak-gb G --mem-peak-max-gb N`: G = the engine's own prefill share of RAM (35%, `moe_stream.PREFILL_PCT`), the store's floor; N = what the swap line allows now, `(free - 1 GB) / 1.4` (one GB of the engine's footprint costs ~1.4 GB of swap-line margin), capped at the GPU working set and at the whole model. The engine's governor grows the store toward N while memory allows and gives it back under pressure |
| the 35B on Apple | its tested arena path: `--prefill-step 0 --keep-convs 4` (the arena sizes itself from free memory) |
| streamed MoE, CUDA | `--mem-peak-gb` = VRAM - 1 GB (it bounds the whole device's use); `CUDA_HOME` when a toolkit is found and it is unset (MLX compiles the engine's kernels at run time) |
| cache bias | `--cache-bias 0.5 --bias-budget 0.2` only where it passed a paired quality gate (GSM8K / HumanEval); exact routing elsewhere |
| native reader | `MOE_NATIVE=1` for the tested Flash-Next entry when `moe_native` imports, after one `native/build.sh` attempt (Xcode command line tools); on CUDA when `g++` is there |
| dense | none: the server's defaults; refused when the weights + 1 GB do not fit the usable memory |

## Variants: the same model at other bit widths

Each model in the list shows its variants side by side: size on disk, the estimated speed on this
machine, what it takes to get it (ready / a download / minutes to make), and a quality note. The
first one, the model as published, is the default; every other variant is an explicit pick
(`3b` in the menu, `--variant q3` or `--variant owner/repo` on the command line), and making one
asks first. The engine itself never lowers quality.

| variant | where it comes from | offered when |
|---|---|---|
| as published | the repo or folder picked | always, first |
| the same model at other bits | the same publisher's plain `-Nbit` / `-bf16` repos on Hugging Face (up to 4) | they exist |
| N-bit, made here | `tools/quantpack.py --from-pack`: plain affine rounding of the published weights at their own group size, to 4 or 3 bits (only below the published bits): a streamed MoE's routed experts record by record from `experts.bin` (64 experts at a time), a dense model's every quantized tensor; resumable; the rest of the model linked, not copied | the published weights are MLX affine (not EXL3, not unquantized) |
| GPTQ 3-bit | not made here: it needs a calibration capture of the model's activations that tripped the swap guard on a 16 GB machine | shown as a note (Flash-Next) |
| EXL3 | only as a ready pack (making one is ~1,400 core-hours) | the curated Flash-Next entry |

- **Disk** is checked first (the new files, plus the source if it is not on disk yet, plus 10 GB).
- **Validated**: a variant made here is offered as ready only after the engine loaded it and
  answered one request (`launcher.json` is written then); until then it says
  "made; checked with one reply at its first start".
- **Time to make**: (bytes read + bytes written) / (0.39 x the drive's measured rate), from the
  launcher's own run (Qwen1.5-MoE 7.0 -> 5.4 GB in 11 s at 2.9 GB/s on a 16 GB M4); a 75 GB pack
  would take ~2 min at that rate (an earlier tool took 4 min on Flash-Next).
- **Quality notes** are measured where we have numbers, generic otherwise:
  - Qwen1.5-MoE 3-bit (rounded from 4-bit): perplexity +7.6%, 6 of 6 instruction prompts right
    (2026-09-12).
  - Flash-Next, against its 4-bit: EXL3 3-bit +2.45% perplexity with GSM8K-300 / HumanEval-164 not
    significantly different; plain 3-bit rounding +4.13%, GPTQ 3-bit +1.39%.
  - Generic: 8-bit close to the original, 6-bit a very small loss, 5-bit a small loss, 4-bit
    usually a small loss, 3-bit a noticeable loss, 2-bit a large loss; a variant rounded from the
    published 4-bit is "a little worse than one made from the original weights".
- **Speed per bit width**: bytes scale with bits (bits/8 + a bf16 scale and bias per group), both
  the GPU's and the SSD's, and more experts fit in the same memory. Measured on a 16 GB M4
  in earlier runs (2026-10-26): Qwen1.5-MoE's 3-bit copy decoded 1.12x (code) to
  1.35x (prose) the 4-bit, against an estimate of 1.09x, so the estimate is low for 3-bit copies
  (0.80x of the measured warm speed); an older measurement on the arena path (2026-09-12) had
  found 1.05x.

## How the speed estimate is computed

A decode token = the GPU reading its bytes + the expert reads the memory misses
(`machine.estimate`):

- **GPU**: (dense bytes read a token + top-k x MoE layers x one expert's bytes) / (measured
  bandwidth x η). η is the share of the decode kernel's bandwidth a whole token gets: Apple MoE
  0.73, Apple dense 0.99, CUDA MoE 0.36, CUDA dense 0.77.
- **Reads**: the share of the experts the memory holds, `held` = store size / all experts, gives the
  share of routed experts that miss, `m = a x held^-b` (with the cache bias a = 0.0123,
  b = 0.96; without a = 0.076, b = 0.53). Read time = layers with a miss x the drive's measured
  per-read cost + missed bytes / the drive's measured rate.
- The constants are fitted to measured points; against the measured decode speeds of earlier runs
  the estimate lands at 0.88-1.23x.
  It is printed as "~N tok/s (estimate)": the first prompt, long contexts, speculative decoding
  and a second SSD are not in it.

## Sharing this computer (experimental)

`./larder share` turns this computer into a device the menu on another computer can use: it starts a
small control service (`share.py`, not `server.py`) on the LAN and prints a 12-character pairing code
and its certificate's fingerprint (the code is valid 10 minutes, then a new one is printed; 5 wrong tries
lock pairing until you restart `./larder share`). A UDP beacon announces it, and the menu on another
computer shows `<hostname> (shared)`; Enter asks for the code and shows the fingerprint to compare
before it pairs. `SECURITY.md` sums up what share exposes.

- **Pairing.** Both sides prove they know the code (PBKDF2 plus an HMAC challenge in both
  directions), then the device issues a long token kept on both sides (the device in
  `~/.cache/moe-ssd-offload/shared-clients.json`, the menu in
  `~/.cache/moe-ssd-offload/paired.json`). Every later call is signed with that token over the
  method, path, body hash, a nonce and the time, accepted only within 60 seconds and never twice.
- **What goes over the LAN.** Without TLS the pairing handshake, the model list, the machine facts,
  the prepare progress, and the model server's prompts and replies all travel as plain HTTP, and so
  do the pairing code and the model server's api key.
- **TLS.** When `openssl` is on the shared computer, the first `./larder share` makes a self-signed
  certificate (`~/.cache/moe-ssd-offload/share-cert.pem`, its key next to it, mode 600). The control
  service, the pairing and the model server then speak HTTPS, and the menu pins the certificate's
  sha256 fingerprint while pairing, so a connection with a different certificate is refused. The
  menu's local proxy still listens on 127.0.0.1 and reaches the shared model server over that
  HTTPS, adding the api key; it refuses a request whose Origin or Host is not this computer, as the
  server does, so a web page cannot borrow the key. Without openssl `./larder share` does not start;
  `./larder share --no-tls` serves plain HTTP and says that the pairing and the token then travel in clear.
  A control call's body is capped at 64 KB before anything is read.
- **Revoking a pairing.** Delete the token from `~/.cache/moe-ssd-offload/shared-clients.json` on
  the shared computer and restart `./larder share` (it keeps the tokens in memory), or delete that
  device's row from `~/.cache/moe-ssd-offload/paired.json` on the menu computer. Either side alone
  ends the pairing; a new pairing issues a new token.

## Files

- `larder`: the one command; picks the Python (the repo's `.venv` or `python3`) and hands over.
- `launch.py`: the recommendation and the text menu, the catalog, download / prepare, the server,
  the banners, the terminal chat, `--stop` / `--status`, `TuiBackend` for the full-screen menu.
- `tui.py`: the full-screen menu (curses); `devices.py`: the network scan behind its Devices tab.
- `share.py`: `./larder share`, the pairing code and beacon, the signed control calls, the local proxy
  and the TLS certificate; `remote.py`: the ssh and shared-device rows the menu shows.
- `machine.py`: the profile, the settings, the estimate (importable; no engine import at load).
- `library.py`: model folders found outside the models folder and matched to the catalog, the
  measured speeds, the rules for what may be deleted.
- `tools/stream_pack.py`, `tools/repack.py`: the packers the launcher drives.
- `~/.cache/moe-ssd-offload/`: the profile, server logs, the engine's calibrations;
  `~/.cache/moe-ssd-offload/models/<id>/`: models (`--models-dir` moves them), each with a
  `launcher.json` once ready; `last.json` (the model of the last run) and `server.json` (the
  running server, for `./larder stop` / `./larder status`); `found.json` (the scan of other folders) and
  `measured.json` (the speed measured per machine and model).
- Tests (no model, no network): `tests/test_launcher.py`, `tests/test_tui.py`,
  `tests/test_safetensors_map.py`.

## Uninstall and where files live

Nothing is installed outside these places:

- **The engine**: the folder it was installed into (`~/larder` from `install.sh`, or your clone) with its
  `.venv`. The native extensions (`moe_native`, `_ane_mlx`) are built into the Python that runs the engine,
  which is that `.venv` when there is one.
- **Models**: `~/.cache/moe-ssd-offload/models/<id>/`, or the folder you gave `--models-dir`. Flash-Next takes up
  to ~82 GB. `./larder --delete ID --variant KEY --yes` removes one model.
- **Everything else the launcher and the engine keep**: `~/.cache/moe-ssd-offload/` (the machine profile and its
  measurements, server logs, calibrations, measured speeds, and the share certificate and pairings).
- **Saved expert profiles and memory states**: `~/.cache/moe-brain/`, written only when a program uses the
  library's profile or memory features (`eprofile.py`, `kvmem.py`).
- **Hugging Face's cache**: `~/.cache/huggingface/` holds the small config files the launcher reads before a
  download. Other programs share it; leave it if you use them.

To remove everything:

```bash
./larder stop
rm -rf ~/.cache/moe-ssd-offload ~/.cache/moe-brain ~/larder     # or the folder you cloned into
```
