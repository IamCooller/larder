#!/usr/bin/env python3
"""Download a sharded MLX checkpoint and pack it the engine's way, one shard at
a time, so the disk never holds more than the packed result plus one shard.

The separate repack and strip steps need the whole checkpoint on disk first, which
for Qwen3.8-Flash-Next (111.5 GB) would mean 220 GB at the peak. Here each
shard is downloaded, its routed experts appended to experts.bin (the exact
record layout the repack step writes: per expert weight, scales, biases back to
back, one index entry per projection), its PLE n-gram embedding shards
appended to ple.bin with their own index, its remaining tensors saved to a
slim shard, and the download deleted before the next shard starts.

A checkpoint that stores one tensor per expert (`mlp.experts.7.gate_proj.*`,
Qwen1.5-MoE, OLMoE) is packed the same way: each expert's tensors become its
record, the bytes stack_experts.py would have stacked.

--skip-experts fetches only the tensors that stay (dense part, PLE) by HTTP
range requests and writes no experts.bin: the base of a pack whose experts
come from elsewhere (an EXL3 pack, launch.py). Its config still gets the
experts' quantization entries.

A run that stops is resumed by running it again: `.pack_state.json` records
the shards done and the bytes written, and the next run truncates to that.

What is dropped: the vision tower (a text-only engine), named in the
report so nobody wonders where 0.9 GB went. What is kept: everything else,
including the MTP head, which the model file ignores today and a speculator
may want tomorrow.

config.json gets the quantization entries the loader needs for the stripped
experts and PLE shards (their .scales are gone from the safetensors; mlx_lm
builds a module as quantised only if its path is listed).

    python3 stream_pack.py --repo mlx-community/Qwen3.8-Flash-Next-4bit --out ~/models/<name>
"""
import argparse
import json
import os
import re
import shutil
import struct
import time
import urllib.request
from pathlib import Path

import numpy as np

EXPERT_MARK = ".switch_mlp."
# One tensor per expert: prefix, expert, projection, part.
PER_EXPERT = re.compile(r"^(?P<pre>.+)\.experts\.(?P<e>\d+)\.(?P<proj>[^.]+)\.(?P<part>weight|scales|biases)$")
# The released conversion names the PLE shards `ngram_embedding.shards.K`;
# the mlx-lm model file (PR #1788) names its modules `ngram_embedding.shard_K`.
# Both forms are recognised, and the index is written in the module's form,
# which is what install_ple matches against.
PLE_MARK = ".ngram_embedding.shard"
_SHARDS = re.compile(r"\.ngram_embedding\.shards\.(\d+)$")
PARTS = ("weight", "scales", "biases")
# Range requests further apart than this are separate requests; closer, the gap is read and dropped.
GAP = 8 << 20


def module_key(base):
    """A checkpoint key in the form the mlx-lm module tree will have after
    the model's sanitize: `language_model.` stripped, `shards.K` -> `shard_K`."""
    if base.startswith("model.language_model."):
        base = "model." + base[len("model.language_model."):]
    elif base.startswith("language_model."):
        base = base[len("language_model."):]
    return _SHARDS.sub(r".ngram_embedding.shard_\1", base)
DROP_PREFIXES = ("model.visual.", "visual.", "vision_tower.", "language_model.visual.")


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def hf_key(k):
    """The official checkpoint's key layout, from the released conversion's."""
    if k.startswith("language_model.model."):
        return "model.language_model." + k[len("language_model.model."):]
    if k.startswith("language_model.lm_head."):
        return "lm_head." + k[len("language_model.lm_head."):]
    return k


def q_entry(hdr, base, group_size):
    """{group_size, bits} for a quantised tensor triple, from its shapes."""
    w, sc = hdr.get(f"{base}.weight"), hdr.get(f"{base}.scales")
    if w is None or sc is None or w["dtype"] != "U32":
        return None
    bits = 32 * w["shape"][-1] // (sc["shape"][-1] * group_size)
    return {"group_size": group_size, "bits": bits}


def expert_groups(keys):
    """{stacked base: [{part: key} per expert, in expert order]} for per-expert keys."""
    groups = {}
    for k in keys:
        m = PER_EXPERT.match(k)
        if m:
            base = f"{m['pre']}{EXPERT_MARK}{m['proj']}"
            groups.setdefault(base, {}).setdefault(int(m["e"]), {})[m["part"]] = k
    return {b: [ex[e] for e in sorted(ex)] for b, ex in groups.items()
            if all(set(r) == set(PARTS) for r in ex.values())}


def is_expert(k):
    return EXPERT_MARK in k or PER_EXPERT.match(k) is not None


def pack_triples(parts_at, dst, written, index, bases, members=None):
    """Append the tensors of `bases` (each a weight/scales/biases triple) as
    per-row records: for expert projections a row is one expert, for PLE
    shards a row is one embedding row. `parts_at[key]` = (open file, header
    entry, data offset), the three parts of a triple may sit in different
    shards (the 35B and this checkpoint both split some). `members[base]`:
    the triple is one tensor per expert instead, each expert's three are its
    record. Returns bytes written so far. Small triples are interleaved in
    memory in one go; large ones (a 512-expert projection is 4.5 GB) row by row."""
    for base in bases:
        if members and base in members:
            ex = members[base]
            first = {p: parts_at[ex[0][p]][1] for p in PARTS}
            size = {p: first[p]["data_offsets"][1] - first[p]["data_offsets"][0] for p in PARTS}
            index[module_key(base)] = {
                "offset": written, "stride": sum(size.values()), "n_experts": len(ex),
                "parts": [{"name": p, "bytes": size[p], "shape": list(first[p]["shape"]),
                           "dtype": first[p]["dtype"]} for p in PARTS]}
            for rec in ex:
                for p in PARTS:
                    f, m, data_off = parts_at[rec[p]]
                    lo, hi = m["data_offsets"]
                    assert hi - lo == size[p], f"{rec[p]}: experts of one projection differ in size"
                    f.seek(data_off + lo)
                    dst.write(f.read(hi - lo))
            written += len(ex) * sum(size.values())
            continue
        metas = {p: parts_at[f"{base}.{p}"] for p in PARTS}
        n = metas["weight"][1]["shape"][0]
        rows = {}
        for p, (f, m, data_off) in metas.items():
            lo, hi = m["data_offsets"]
            rows[p] = (f, data_off + lo, (hi - lo) // n, hi - lo)
        stride = sum(rb for _, _, rb, _ in rows.values())
        index[module_key(base)] = {
            "offset": written, "stride": stride, "n_experts": n,
            "parts": [{"name": p, "bytes": rows[p][2],
                       "shape": list(metas[p][1]["shape"][1:]),
                       "dtype": metas[p][1]["dtype"]} for p in PARTS],
        }
        total = sum(t for _, _, _, t in rows.values())
        if total <= (1 << 30):
            bufs = []
            for p in PARTS:
                f, start, rb, t = rows[p]
                f.seek(start)
                bufs.append(np.frombuffer(f.read(t), dtype=np.uint8).reshape(n, rb))
            dst.write(np.concatenate(bufs, axis=1).tobytes())
        else:
            for e in range(n):
                for p in PARTS:
                    f, start, rb, _ = rows[p]
                    f.seek(start + e * rb)
                    dst.write(f.read(rb))
        written += n * stride
    return written


def _http(url, lo, hi):
    """Bytes [lo, hi] of url, as a response to read from."""
    req = urllib.request.Request(url, headers={"Range": f"bytes={lo}-{hi}",
                                               "User-Agent": "moe-ssd-offload/stream_pack"})
    r = urllib.request.urlopen(req, timeout=60)
    if r.status != 206:
        raise IOError(f"{url}: no range support (HTTP {r.status})")
    return r


def _pipe(r, n, out=None):
    """Read n bytes of r, into `out` when given."""
    while n:
        b = r.read(min(n, 8 << 20))
        if not b:
            raise IOError("connection closed early")
        if out is not None:
            out.write(b)
        n -= len(b)


def fetch_partial(url, dest, drop):
    """The safetensors file at url with only the tensors drop(key) is false for,
    fetched by range requests and written to dest as a valid safetensors file.
    Returns the full header (shapes of what was dropped included)."""
    with _http(url, 0, 7) as r:
        n = struct.unpack("<Q", r.read(8))[0]
    with _http(url, 8, 8 + n - 1) as r:
        hdr = json.loads(r.read(n))
    keep = sorted((m["data_offsets"], k) for k, m in hdr.items()
                  if k != "__metadata__" and not drop(k))
    new, pos = {}, 0
    for (lo, hi), k in keep:
        new[k] = {**hdr[k], "data_offsets": [pos, pos + hi - lo]}
        pos += hi - lo
    if "__metadata__" in hdr:
        new["__metadata__"] = hdr["__metadata__"]
    head = json.dumps(new).encode()
    head += b" " * (-len(head) % 8)
    spans = []
    for (lo, hi), k in keep:
        if spans and lo - spans[-1][1] <= GAP:
            spans[-1][1] = max(spans[-1][1], hi)
            spans[-1][2].append((lo, hi))
        else:
            spans.append([lo, hi, [(lo, hi)]])
    tmp = dest.with_name(dest.name + ".part")
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(head)) + head)
        for lo, hi, parts in spans:
            with _http(url, 8 + n + lo, 8 + n + hi - 1) as r:
                cur = lo
                for a, b in parts:
                    _pipe(r, a - cur)
                    _pipe(r, b - a, f)
                    cur = b
    tmp.replace(dest)
    return hdr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=None)
    ap.add_argument("--revision", default=None, help="the repo's commit to fetch (default: main)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep-vision", action="store_true")
    ap.add_argument("--hf-layout", action="store_true",
                    help="write the dense tensors under the HF key layout "
                         "(model.language_model.*, lm_head.*). qwen4_exp's model file "
                         "folds the +1 of its zero-centred norm gains only for that "
                         "layout; the released MLX conversion uses language_model.* "
                         "and stores the gains zero-centred, so without this it decodes "
                         "garbage. Not for Qwen3.5/Next, whose module tree IS language_model.*")
    ap.add_argument("--from-dir", default=None,
                    help="pack a checkpoint already on disk instead of downloading; "
                         "the source is left in place")
    ap.add_argument("--skip-experts", action="store_true",
                    help="download and keep everything but the routed experts (range "
                         "requests), no experts.bin; their quantization entries still go "
                         "into config.json")
    args = ap.parse_args()

    import mlx.core as mx
    out = Path(os.path.expanduser(args.out))
    out.mkdir(parents=True, exist_ok=True)
    small = ["config.json", "model.safetensors.index.json", "tokenizer.json",
             "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja",
             "generation_config.json", "vocab.json", "merges.txt", "added_tokens.json", "README.md"]
    full_hdr = {}                    # skip-experts: each shard's whole header
    if args.from_dir:
        src = Path(os.path.expanduser(args.from_dir))
        tmp = None

        def fetch(name):
            return src / name
        for f in small:
            if (src / f).exists():
                shutil.copy(src / f, out / f)
    else:
        from huggingface_hub import hf_hub_download, hf_hub_url
        tmp = out / "_download"
        tmp.mkdir(exist_ok=True)

        def fetch(name):
            if args.skip_experts and name.endswith(".safetensors"):
                dest, hp = tmp / name, tmp / (name + ".header.json")
                if not (dest.exists() and hp.exists()):
                    for attempt in range(5):
                        try:
                            h = fetch_partial(hf_hub_url(args.repo, name, revision=args.revision),
                                              dest, is_expert)
                            hp.write_text(json.dumps(h))
                            break
                        except OSError as e:
                            print(f"  {name}: {e}; retrying", flush=True)
                            time.sleep(5 * (attempt + 1))
                    else:
                        raise SystemExit(f"{name}: download failed")
                full_hdr[name] = json.loads(hp.read_text())
                print(f"  {name}: {dest.stat().st_size / 1e9:.2f} GB kept of the shard", flush=True)
                return dest
            return Path(hf_hub_download(args.repo, name, revision=args.revision, local_dir=str(tmp)))
        for f in small:
            try:
                shutil.copy(fetch(f), out / f)
            except Exception:
                pass
    state_p = out / ".pack_state.json"
    state = json.loads(state_p.read_text()) if state_p.exists() else None
    if state is not None:
        idx, cfg = state["idx"], state["cfg"]
    else:
        idx = json.loads((out / "model.safetensors.index.json").read_text())
        cfg = json.loads((out / "config.json").read_text())
    shards = sorted(set(idx["weight_map"].values()))
    q = cfg.setdefault("quantization", {})
    g = int(q.get("group_size", 64))
    # The released config names its per-module overrides (8-bit routers and
    # shared-expert gates) with the checkpoint prefix; the mlx-lm model file
    # strips it in sanitize and the loader then compares MODULE paths, so
    # without the second name an 8-bit router is built 4-bit and the first
    # matmul dies on the scales shape. Every override under both names.
    for k, v in list(q.items()):
        if isinstance(v, dict) and module_key(k) != k:
            q.setdefault(module_key(k), v)

    exp_index, ple_index, kept_map = {}, {}, {}
    exp_written = ple_written = kept_bytes = dropped = 0
    done = []
    if state is not None:
        exp_index, ple_index, kept_map = state["exp_index"], state["ple_index"], state["kept_map"]
        exp_written, ple_written = state["exp_written"], state["ple_written"]
        kept_bytes, dropped, done = state["kept_bytes"], state["dropped"], state["done"]
        print(f"  resuming after {len(done)} of {len(shards)} shards", flush=True)
    wm = idx["weight_map"]
    members = expert_groups(wm)
    # Which shards each triple touches, and the last shard (in order) that
    # needs each shard's bytes: a shard is deleted only after that point.
    shard_pos = {sh: i for i, sh in enumerate(shards)}

    def triple_shards(base):
        if base in members:
            return {wm[k] for rec in members[base] for k in rec.values()}
        return {wm[f"{base}.{p}"] for p in PARTS if f"{base}.{p}" in wm}
    last_use = {sh: i for i, sh in enumerate(shards)}
    firsts = {members[b][0]["weight"]: b for b in members}
    for k in wm:
        if k.endswith(".weight") and (EXPERT_MARK in k or PLE_MARK in k or k in firsts):
            base = firsts.get(k) or k[:-len(".weight")]
            here = shard_pos[wm[k]]
            for other in triple_shards(base):
                last_use[other] = max(last_use[other], here)
    opened = {}                      # shard name -> (file, header, data offset)

    def get(sh):
        if sh not in opened:
            path = fetch(sh)
            hdr, data_off = read_header(path)
            opened[sh] = (open(path, "rb"), hdr, data_off, path)
        return opened[sh]

    def reopen(name, size):
        f = open(out / name, "r+b" if state is not None and (out / name).exists() else "wb")
        f.truncate(size)
        f.seek(size)
        return f

    t0 = time.perf_counter()
    with reopen("experts.bin", exp_written) as fe, reopen("ple.bin", ple_written) as fp:
        for si, sh in enumerate(shards):
            if sh in done:
                continue
            t = time.perf_counter()
            f, hdr, data_off, path = get(sh)
            keys = [k for k in hdr if k != "__metadata__"]
            exp_bases = [] if args.skip_experts else sorted(
                {k[:-len(".weight")] for k in keys if k.endswith(".weight") and EXPERT_MARK in k}
                | {firsts[k] for k in keys if k in firsts})
            ple_bases = sorted({k[:-len(".weight")] for k in keys if k.endswith(".weight") and PLE_MARK in k})
            parts_at = {}
            for b in exp_bases + ple_bases:
                for key in ([k for rec in members[b] for k in rec.values()] if b in members
                            else [f"{b}.{p}" for p in PARTS]):
                    of, ohdr, ooff, _ = get(wm[key])
                    parts_at[key] = (of, ohdr[key], ooff)
            exp_written = pack_triples(parts_at, fe, exp_written, exp_index, exp_bases, members)
            ple_written = pack_triples(parts_at, fp, ple_written, ple_index, ple_bases)
            # Under both namings: mlx_lm's loader compares the MODULE path
            # (post-sanitize) with config["quantization"], and a stripped
            # tensor has no .scales left to be recognised by. Skipped experts
            # are named from the shard's whole header.
            fh = full_hdr.get(sh, hdr)
            heads = {b: {p: parts_at[r[p]][1] for p in PARTS}
                     for b, r in ((b, members[b][0] if b in members else
                                   {p: f"{b}.{p}" for p in PARTS}) for b in exp_bases + ple_bases)}
            if args.skip_experts:
                heads.update({k[:-len(".weight")]: {p: fh.get(k[:-len("weight")] + p) for p in PARTS}
                              for k in fh if k.endswith(".weight") and EXPERT_MARK in k})
                heads.update({b: {p: fh[ex[0][p]] for p in PARTS} for b, ex in expert_groups(fh).items()})
            for b, h in heads.items():
                e = q_entry({f"{b}.{p}": m for p, m in h.items() if m}, b, g)
                if e:
                    q.setdefault(b, e)
                    q.setdefault(module_key(b), e)
            moved = {k for k in keys if any(k.startswith(b + ".") for b in exp_bases + ple_bases)
                     or is_expert(k) or PLE_MARK in k}
            keep = [k for k in keys if k not in moved
                    and (args.keep_vision or not k.startswith(DROP_PREFIXES))]
            dropped += sum(1 for k in keys if k not in moved and k not in keep)
            if keep:
                loaded = mx.load(str(path))
                slim = out / sh
                dense = {hf_key(k) if args.hf_layout else k: loaded[k] for k in keep}
                # A different file from the one being read: mx.load is lazy.
                mx.save_safetensors(str(slim), dense)
                del loaded, dense
                kept_bytes += slim.stat().st_size
                for k in keep:
                    kept_map[hf_key(k) if args.hf_layout else k] = sh
            for done_sh in [x for x, last in last_use.items() if last <= si and x in opened]:
                of, _, _, opath = opened.pop(done_sh)
                of.close()
                if tmp is not None:
                    opath.unlink()
            fe.flush()
            fp.flush()
            os.fsync(fe.fileno())
            os.fsync(fp.fileno())
            done.append(sh)
            state_p.write_text(json.dumps({
                "idx": idx, "cfg": cfg, "exp_index": exp_index, "ple_index": ple_index,
                "kept_map": kept_map, "exp_written": exp_written, "ple_written": ple_written,
                "kept_bytes": kept_bytes, "dropped": dropped, "done": done}))
            state = True
            print(f"  [{si + 1}/{len(shards)}] {sh}: {len(exp_bases)} expert proj, "
                  f"{len(ple_bases)} PLE shards, {len(keep)} dense kept | experts.bin "
                  f"{exp_written / 1e9:.1f} GB, ple.bin {ple_written / 1e9:.1f} GB | "
                  f"{time.perf_counter() - t:.0f}s", flush=True)
    (out / "experts_index.json").write_text(json.dumps(exp_index, indent=1))
    (out / "ple_index.json").write_text(json.dumps(ple_index, indent=1))
    idx["weight_map"] = kept_map
    idx.setdefault("metadata", {})["total_size"] = kept_bytes
    (out / "model.safetensors.index.json").write_text(json.dumps(idx, indent=1))
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    state_p.unlink()
    if tmp is not None:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\ndone in {(time.perf_counter() - t0) / 60:.0f} min: experts.bin {exp_written / 1e9:.1f} GB "
          f"({len(exp_index)} projections), ple.bin {ple_written / 1e9:.1f} GB ({len(ple_index)} shards), "
          f"dense {kept_bytes / 1e9:.2f} GB, dropped {dropped} vision tensors; "
          f"config.json: {len(q)} quantization entries")


if __name__ == "__main__":
    main()
