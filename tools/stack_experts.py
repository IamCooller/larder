#!/usr/bin/env python3
"""Stack per-expert tensors into the `.switch_mlp.` layout the engine reads.

Qwen3 checkpoints ship one tensor per projection holding every expert
(`mlp.switch_mlp.gate_proj.weight`, shape [E, out, in]). Qwen1.5/Qwen2 MoE ship
one tensor per expert instead (`mlp.experts.7.gate_proj.weight`, shape
[out, in]) and let mlx_lm stack them in memory at load time. repack.py,
quantpack.py and stream_fast's reader all key off the string `.switch_mlp.`
and read straight from the safetensors index, so the older layout never
reaches them: the engine dies with

    KeyError: 'model.layers.0.mlp.switch_mlp.gate_proj.weight'

Stacking is a concatenation, not a requantisation. weight, scales and biases
are already quantised at one group size and are copied through unchanged, so
the result is the same model bit for bit, so there is no quality question to
answer about the output. mlx_lm's own sanitize() returns a stacked checkpoint
untouched (it only rewrites when `experts.0.up_proj.weight` is present), so
what comes out still loads the ordinary way as well.

Only routed experts move. The router (`mlp.gate.weight`), the shared expert
(`mlp.shared_expert.*`) and everything outside the MoE block are copied as they
are, since the engine streams routed experts and nothing else.

Memory: one projection's experts are held at a time (~10 MB at 60 experts),
and output is flushed to a new shard every ~1.5 GB, so this runs in well under
a gigabyte no matter how large the checkpoint is.

Usage:
    python3 stack_experts.py --src ~/models/<name> \
                             --out ~/models/<name>-stacked
"""

import argparse
import json
import re
import shutil
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

# model.layers.0.mlp.experts.7.gate_proj.weight
#   -> prefix "model.layers.0.mlp", expert 7, projection "gate_proj", part "weight"
EXPERT_RE = re.compile(
    r"^(?P<prefix>model\.layers\.\d+\.mlp)\.experts\.(?P<idx>\d+)"
    r"\.(?P<proj>[^.]+)\.(?P<part>weight|scales|biases)$")

SHARD_BYTES = 1_500_000_000

# Everything a loader needs beside the tensors. Copied verbatim; config.json
# does not change, because stacking does not change a single hyperparameter.
SIDECARS = ("config.json", "tokenizer.json", "tokenizer_config.json",
            "vocab.json", "merges.txt", "added_tokens.json",
            "special_tokens_map.json", "chat_template.jinja")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    src = Path(args.src).expanduser()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    wmap = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]

    # (prefix, projection, part) -> {expert index: tensor key}
    groups = defaultdict(dict)
    passthrough = []
    for key in wmap:
        m = EXPERT_RE.match(key)
        if m:
            groups[(m["prefix"], m["proj"], m["part"])][int(m["idx"])] = key
        else:
            passthrough.append(key)

    if not groups:
        raise SystemExit(
            f"  no `.experts.N.` tensors in {src.name}: either this checkpoint is\n"
            f"  already stacked, or it is not a MoE the engine knows how to read.")

    # Expert count has to be the same everywhere, and the indices have to be a
    # dense 0..E-1 range: a gap would stack the wrong expert into a slot and
    # the model would still load, silently answering as a different model.
    counts = {len(v) for v in groups.values()}
    if len(counts) != 1:
        raise SystemExit(f"  projections disagree on expert count: {sorted(counts)}")
    n_exp = counts.pop()
    for names, byidx in groups.items():
        if sorted(byidx) != list(range(n_exp)):
            raise SystemExit(f"  expert indices are not 0..{n_exp - 1} for {names}")

    print(f"  {src.name}: {len(wmap)} tensors, {n_exp} experts per projection")
    print(f"  stacking {len(groups)} projections, copying {len(passthrough)} tensors")

    state = {"buf": {}, "bytes": 0, "shard": 0}
    files = {}

    def flush():
        if not state["buf"]:
            return
        state["shard"] += 1
        name = f"model-{state['shard']:05d}.safetensors"
        save_file(state["buf"], str(out / name))
        for k in state["buf"]:
            files[k] = name
        print(f"    wrote {name}  {state['bytes'] / 1e9:.2f} GB, "
              f"{len(state['buf'])} tensors", flush=True)
        state["buf"], state["bytes"] = {}, 0

    def add(key, arr):
        state["buf"][key] = arr
        state["bytes"] += arr.nbytes
        if state["bytes"] >= SHARD_BYTES:
            flush()

    with ExitStack() as stack:
        readers = {
            fname: stack.enter_context(safe_open(str(src / fname), framework="numpy"))
            for fname in sorted(set(wmap.values()))
        }

        for (prefix, proj, part), byidx in sorted(groups.items()):
            arrs = [readers[wmap[byidx[i]]].get_tensor(byidx[i]) for i in range(n_exp)]
            add(f"{prefix}.switch_mlp.{proj}.{part}", np.stack(arrs, axis=0))

        for key in passthrough:
            add(key, readers[wmap[key]].get_tensor(key))

    flush()

    total = sum(Path(out / f).stat().st_size for f in sorted(set(files.values())))
    (out / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": files}, indent=2))

    for name in SIDECARS:
        if (src / name).exists():
            shutil.copy2(src / name, out / name)

    print(f"  {out}: {len(files)} tensors in {state['shard']} shards, "
          f"{total / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
