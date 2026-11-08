#!/usr/bin/env python3
"""
Requantise the experts and write experts.bin in ONE pass. Also: per-projection bits.

The old route was three separate steps (requantise, repack, strip the experts),
which puts three copies of the expert weights on disk at once: the new
safetensors, the packed binary, and the original. That filled the disk to 1.6 GB
free mid-write and produced a truncated checkpoint, which only surfaced later as
"invalid data offsets exceeding the size of the file". Writing the packed layout
directly needs one copy.

It also makes mixed precision practical, which the two-step route did not: bits
are chosen PER PROJECTION. 2-bit experts hold perplexity (+1.3%) but lose
instruction following: the 2-bit build answers a Python question in Chinese
with "I cannot write code" while the 4-bit reference returns working code. The
hypothesis worth testing is that down_proj, which writes back into the residual
stream, is what cannot take 2 bits, while gate/up can.

    --bits 3                 every projection at 3 bits
    --bits 2 --down-bits 4   gate/up at 2, down_proj at 4

Bytes per token scale with the average, so 2/2/4 costs 33% more than 2/2/2 and
33% less than 4/4/4.

Usage:
    python3 quantpack.py --out ~/models/<name> --bits 2 --down-bits 4
"""

import argparse
import json
import os
import re
import shutil
import struct
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

SRC = "mlx-community/ERNIE-4.5-21B-A3B-PT-4bit"
EXPERT_MARK = ".switch_mlp."
_DT = {mx.uint32: "U32", mx.uint16: "U16", mx.uint8: "U8",
       mx.float16: "F16", mx.bfloat16: "BF16", mx.float32: "F32"}


_NP = {"U8": np.uint8, "U16": np.uint16, "U32": np.uint32, "I32": np.int32,
       "F16": np.float16, "F32": np.float32, "BF16": np.uint16}
_MX = {"U8": mx.uint8, "U16": mx.uint16, "U32": mx.uint32, "I32": mx.int32,
       "F16": mx.float16, "F32": mx.float32, "BF16": mx.bfloat16}


def selftest():
    """Synthetic double-quantization round-trip, no checkpoint needed.

    Mirrors the real path: bf16 -> 4bit (stand-in for the already-quantized
    source) -> dequantize -> 3bit (the new target) -> dequantize. Checks the
    3-bit pack has the expected shape and that its error is worse than 4-bit
    but still bounded, i.e. double quantization degrades gracefully instead
    of blowing up.
    """
    mx.random.seed(0)
    n_exp, out_dim, in_dim, gs = 4, 64, 128, 64
    x = mx.random.normal((n_exp, out_dim, in_dim)).astype(mx.bfloat16)

    q4, s4, b4 = mx.quantize(x, group_size=gs, bits=4)
    deq4 = mx.dequantize(q4, s4, b4, group_size=gs, bits=4)
    err4 = float(mx.mean(mx.abs(deq4.astype(mx.float32) - x.astype(mx.float32))))

    q3, s3, b3 = mx.quantize(deq4, group_size=gs, bits=3)
    deq3 = mx.dequantize(q3, s3, b3, group_size=gs, bits=3)
    err3 = float(mx.mean(mx.abs(deq3.astype(mx.float32) - x.astype(mx.float32))))

    assert q3.shape == (n_exp, out_dim, in_dim * 3 // 32), f"bad 3bit pack shape {q3.shape}"
    assert s3.shape == b3.shape == (n_exp, out_dim, in_dim // gs), "bad scales/biases shape"
    assert err3 > err4, f"3bit error ({err3:.4f}) should exceed 4bit error ({err4:.4f})"
    assert err3 < 1.0, f"3bit error suspiciously large: {err3:.4f}"
    print(f"selftest OK: err4={err4:.4f} err3={err3:.4f} "
          f"(x{err3/err4:.2f}) q4={q4.shape} q3={q3.shape}")


def shard_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def build_index(src):
    """name -> (path, dtype, shape, byte offset, length), across all shards.

    Loading shard-by-shard broke on Qwen3: a projection's `biases` can live in
    a different shard than its `weight`, because the sharder splits on tensor
    boundaries and does not keep a quantised triple together. Reading tensors
    by name from wherever they are also keeps peak memory at one projection
    (~100 MB) instead of one shard (~4.5 GB).
    """
    index = {}
    for shard in sorted(src.glob("*.safetensors")):
        hdr = shard_header(shard)
        with open(shard, "rb") as f:
            data_off = 8 + struct.unpack("<Q", open(shard, "rb").read(8))[0]
        for name, meta in hdr.items():
            if name == "__metadata__":
                continue
            lo, hi = meta["data_offsets"]
            index[name] = (shard, meta["dtype"], tuple(meta["shape"]),
                           data_off + lo, hi - lo)
    return index


def load_tensor(index, name):
    path, dt, shape, start, length = index[name]
    with open(path, "rb") as f:
        buf = os.pread(f.fileno(), length, start)
    arr = mx.array(np.frombuffer(buf, dtype=_NP[dt]).reshape(shape))
    return arr.view(_MX[dt]) if dt == "BF16" else arr


def _module_q(cfg, base):
    """({group_size, bits}, whether a per-module entry) of a module from config.json's quantization."""
    q = cfg.get("quantization") or {}
    from stream_pack import module_key
    for k in (base, module_key(base)):
        if isinstance(q.get(k), dict):
            return q[k], True
    return {"group_size": q["group_size"], "bits": q["bits"]}, False


def _link_rest(src, out, own):
    """Every file of src except `own` into out: a symlink, or a copy where links fail."""
    for p in sorted(src.iterdir()):
        dst = out / p.name
        if p.name in own or p.name.startswith((".", "_")) or dst.exists() or dst.is_symlink():
            continue
        try:
            dst.symlink_to(p.resolve())
        except OSError:
            shutil.copytree(p, dst) if p.is_dir() else shutil.copy2(p, dst)


def requant_pack(src, out, bits, chunk=64):
    """Round a prepared model again to `bits` at its own group sizes (the experts of a
    streamed MoE `chunk` records at a time, else every quantized tensor), resumable from
    `.requant_state.json`, with the other files linked from src."""
    src, out = Path(src).expanduser(), Path(out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((src / "config.json").read_text())
    if not (cfg.get("quantization") or {}).get("bits"):
        raise SystemExit("the source is not an MLX-quantized model: nothing to round again")
    moe = (src / "experts.bin").exists() and (src / "experts_index.json").exists()
    own = {"config.json", "launcher.json"} | ({"experts.bin", "experts_index.json"} if moe else
                                               {p.name for p in src.glob("*.safetensors")})
    _link_rest(src, out, own)
    state_p = out / ".requant_state.json"
    st = json.loads(state_p.read_text()) if state_p.exists() else {"done": [], "written": 0, "index": {}, "q": {}}
    t0 = time.time()
    if moe:
        _requant_experts(src, out, cfg, bits, chunk, st, state_p)
    else:
        _requant_dense(src, out, cfg, bits, st, state_p)
    q = cfg["quantization"]
    for b, e in st["q"].items():
        q[b] = e
    if not moe:
        q["bits"] = bits
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    state_p.unlink()
    print(f"rounded to {bits} bits in {time.time() - t0:.0f} s: {out}", flush=True)


def _requant_experts(src, out, cfg, bits, chunk, st, state_p):
    from stream_pack import module_key
    index = json.loads((src / "experts_index.json").read_text())
    fd = os.open(src / "experts.bin", os.O_RDONLY)
    with open(out / "experts.bin", "r+b" if st["done"] else "wb") as dst:
        dst.truncate(st["written"])
        dst.seek(st["written"])
        for i, (base, e) in enumerate(index.items()):
            if base in st["done"]:
                continue
            names = [p["name"] for p in e["parts"]]
            if names != ["weight", "scales", "biases"]:
                raise SystemExit(f"{base}: records of {names}, not affine weight/scales/biases")
            sq, _ = _module_q(cfg, base)
            gs, sb = sq["group_size"], sq["bits"]
            parts, new = e["parts"], None
            for e0 in range(0, e["n_experts"], chunk):
                n = min(chunk, e["n_experts"] - e0)
                raw = np.frombuffer(os.pread(fd, n * e["stride"], e["offset"] + e0 * e["stride"]),
                                    np.uint8).reshape(n, e["stride"])
                arrs, at = [], 0
                for p in parts:
                    a = mx.array(np.ascontiguousarray(raw[:, at:at + p["bytes"]]).view(_NP[p["dtype"]])
                                 .reshape(n, *p["shape"]))
                    arrs.append(a.view(_MX[p["dtype"]]) if p["dtype"] == "BF16" else a)
                    at += p["bytes"]
                w = mx.dequantize(*arrs, group_size=gs, bits=sb)
                q3 = mx.quantize(w, group_size=gs, bits=bits)
                mx.eval(q3)
                if new is None:
                    new = [{"name": nm, "bytes": a.nbytes // n, "shape": list(a.shape[1:]), "dtype": _DT[a.dtype]}
                           for nm, a in zip(names, q3)]
                rows = [np.frombuffer(memoryview(a.view(mx.uint16) if a.dtype == mx.bfloat16 else a), np.uint8)
                        .reshape(n, -1) for a in q3]
                dst.write(np.concatenate(rows, axis=1).tobytes())
            stride = sum(p["bytes"] for p in new)
            st["index"][base] = {"offset": st["written"], "stride": stride, "n_experts": e["n_experts"], "parts": new}
            st["written"] += stride * e["n_experts"]
            entry = {"group_size": gs, "bits": bits}
            st["q"][base] = st["q"][module_key(base)] = entry
            dst.flush()
            os.fsync(dst.fileno())
            st["done"].append(base)
            state_p.write_text(json.dumps(st))
            print(f"  [{i + 1}/{len(index)}] {base}: {sb} -> {bits} bits", flush=True)
    os.close(fd)
    (out / "experts_index.json").write_text(json.dumps(st["index"], indent=1))


def _requant_dense(src, out, cfg, bits, st, state_p):
    """Every quantized tensor at the model's default bits rounded to `bits`, the whole model read
    at once because a triple may span shards (a resident model fits memory)."""
    shards = sorted(src.glob("*.safetensors"))
    w = {}
    for shard in shards:
        w.update(mx.load(str(shard)))
    for k in [k for k in w if k.endswith(".scales")]:
        base = k[:-len(".scales")]
        sq, override = _module_q(cfg, base)
        if override or sq["bits"] <= bits:
            continue                            # per-module overrides (routers at 8 bits) stay
        deq = mx.dequantize(w[f"{base}.weight"], w[k], w.get(f"{base}.biases"),
                            group_size=sq["group_size"], bits=sq["bits"])
        w[f"{base}.weight"], w[k], w[f"{base}.biases"] = mx.quantize(deq, group_size=sq["group_size"], bits=bits)
    for shard in shards:
        if shard.name in st["done"]:
            continue
        keys = shard_header(shard).keys() - {"__metadata__"}
        mx.save_safetensors(str(out / shard.name), {k: w[k] for k in keys}, metadata={"format": "mlx"})
        st["done"].append(shard.name)
        state_p.write_text(json.dumps(st))
        print(f"  {shard.name}: rounded to {bits} bits", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out")
    ap.add_argument("--from-pack", default=None, metavar="DIR",
                    help="round a model the launcher prepared (experts.bin, or a dense MLX "
                         "checkpoint) again to --bits, at its own group sizes, into --out")
    ap.add_argument("--selftest", action="store_true",
                    help="run a synthetic dequant/requant round-trip and exit, "
                         "no checkpoint or network needed")
    ap.add_argument("--bits", type=int, default=2, help="bits for gate/up")
    ap.add_argument("--down-bits", type=int, default=0,
                    help="bits for down_proj (0 = same as --bits)")
    ap.add_argument("--group-size", type=int, default=32)
    # Sensitivity also varies by DEPTH, not just by projection. The first and
    # last blocks sit closest to the embedding and the output head, where an
    # error has no later layer to correct it; the middle is where redundancy
    # lives. Mixing 2/2/4 across projections failed (3/6 vs 6/6 for plain
    # 3-bit), so this tries the other axis.
    ap.add_argument("--edge-layers", type=int, default=0,
                    help="how many first AND last layers get --edge-bits")
    ap.add_argument("--edge-bits", type=int, default=4)
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if not args.out:
        ap.error("--out is required")
    if args.from_pack:
        requant_pack(args.from_pack, args.out, args.bits)
        return

    down_bits = args.down_bits or args.bits

    def bits_for(base):
        if args.edge_layers:
            m = re.search(r"layers\.(\d+)\.", base)
            if m:
                i = int(m.group(1))
                if i < args.edge_layers or i >= n_layers - args.edge_layers:
                    return args.edge_bits
        return down_bits if base.endswith("down_proj") else args.bits

    local_src = Path(args.src).expanduser()
    if local_src.exists():
        # Already on disk (e.g. a stack_experts.py checkpoint): snapshot_download
        # only understands hub repo ids, not paths, and would otherwise try to
        # download args.src as if it were "org/repo".
        src = local_src
    else:
        from huggingface_hub import snapshot_download
        src = Path(snapshot_download(args.src))
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((src / "config.json").read_text())
    # Multimodal checkpoints keep the text tower's shape under text_config, and
    # a silent default here is not a typo, it is a wrong checkpoint: on Qwen3.6
    # the fallback of 28 made bits_for protect layers 0-3 and 24-39, so 20 of 40
    # layers were written at edge precision instead of 8. That cost 15% of the
    # bytes per token and 18% of the arena slots per GB, unnoticed for a day.
    n_layers = (cfg.get("num_hidden_layers")
                or cfg.get("text_config", {}).get("num_hidden_layers"))
    if not n_layers:
        raise SystemExit("config.json has no num_hidden_layers, at top level or "
                         "under text_config -- refusing to guess")
    src_bits, src_gs = cfg["quantization"]["bits"], cfg["quantization"]["group_size"]
    print(f"source {src_bits}-bit g{src_gs} -> gate/up {args.bits}-bit, "
          f"down {down_bits}-bit, g{args.group_size}")

    index, written, overrides = {}, 0, {}
    kept_index, kept_bytes = {}, 0
    t0 = time.time()

    tmap = build_index(src)
    all_keys = list(tmap)
    bases = sorted({k[: -len(".weight")] for k in all_keys
                    if k.endswith(".weight") and EXPERT_MARK in k},
                   key=lambda b: (int(re.search(r"layers\.(\d+)\.", b).group(1)), b))

    with open(out / "experts.bin", "wb") as dst:
        if True:
            for base in bases:
                bits = bits_for(base)
                if f"{base}.scales" in tmap:
                    # Source projection is already quantized (the normal case: an
                    # mlx-4bit checkpoint, or the 4bit stack_experts.py output).
                    # Dequantize at the source's own bits/group_size before
                    # requantizing at the target, which IS the double
                    # quantization step (e.g. 4bit -> 3bit).
                    w, s, b = (load_tensor(tmap, f"{base}.{p}")
                               for p in ("weight", "scales", "biases"))
                    deq = mx.dequantize(w, s, b, group_size=src_gs, bits=src_bits)
                else:
                    # Plain bf16/fp16 projection, nothing to dequantize first.
                    deq = load_tensor(tmap, f"{base}.weight")
                q, qs, qb = mx.quantize(deq, group_size=args.group_size, bits=bits)
                mx.eval(q, qs, qb)
                del deq

                n_exp = q.shape[0]
                parts, stride = [], 0
                for name, arr in (("weight", q), ("scales", qs), ("biases", qb)):
                    rb = arr.nbytes // n_exp
                    parts.append({"name": name, "bytes": rb,
                                  "shape": list(arr.shape[1:]),
                                  "dtype": _DT[arr.dtype]})
                    stride += rb
                index[base] = {"offset": written, "stride": stride,
                               "n_experts": n_exp, "parts": parts}
                overrides[base] = {"group_size": args.group_size, "bits": bits}

                # One contiguous record per expert: weight, scales, biases.
                raw = [bytes(memoryview(a)) for a in (q, qs, qb)]
                sizes = [p["bytes"] for p in parts]
                for e in range(n_exp):
                    for buf, rb in zip(raw, sizes):
                        dst.write(buf[e * rb:(e + 1) * rb])
                    written += stride
                del q, qs, qb, raw
                if base.endswith("down_proj"):
                    print(f"  {base}: {bits}-bit", flush=True)

    # Non-expert tensors are copied through, grouped by their original shard.
    for shard in sorted(src.glob("*.safetensors")):
        keep_names = [k for k in all_keys
                      if EXPERT_MARK not in k and tmap[k][0] == shard]
        if not keep_names:
            continue
        keep = {k: load_tensor(tmap, k) for k in keep_names}
        mx.save_safetensors(str(out / shard.name), keep)
        kept_bytes += (out / shard.name).stat().st_size
        for k in keep:
            kept_index[k] = shard.name
        print(f"  {shard.name}: {len(keep)} non-expert tensors kept", flush=True)
        del keep

    (out / "experts_index.json").write_text(json.dumps(index, indent=1))

    # Per-module entries already in the source must survive: Qwen3 keeps its
    # routers at 8 bits while everything else is 4, and dropping that made
    # load_model build them at 4 and fail in quantized_matmul.
    src_per = {k: v for k, v in cfg["quantization"].items() if isinstance(v, dict)}
    cfg["quantization"] = {"group_size": src_gs, "bits": src_bits,
                           **src_per, **overrides}
    cfg["quantization_config"] = {"group_size": src_gs, "bits": src_bits}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    idx_path = src / "model.safetensors.index.json"
    if idx_path.exists():
        idx = json.loads(idx_path.read_text())
        idx["weight_map"] = kept_index
        idx.setdefault("metadata", {})["total_size"] = kept_bytes
        (out / "model.safetensors.index.json").write_text(json.dumps(idx, indent=1))

    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                 "added_tokens.json", "chat_template.jinja", "generation_config.json",
                 "tokenizer.model"):
        if (src / name).exists():
            shutil.copy(src / name, out / name)

    print(f"\n{len(index)} projections -> experts.bin {written/1e9:.2f} GB, "
          f"other weights {kept_bytes/1e9:.2f} GB, in {time.time()-t0:.0f}s")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
