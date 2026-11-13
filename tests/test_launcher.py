#!/usr/bin/env python3
"""Checks the launcher's pieces on tiny synthetic checkpoints (a few MB) without a model or the
network, from the catalog filter and flag choice to the decode estimate and stream_pack's new paths."""
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

import launch
import machine
import stream_pack

GB = 1e9
MOE_CFG = {"model_type": "qwen3_moe", "hidden_size": 64, "num_hidden_layers": 2, "intermediate_size": 128,
           "num_attention_heads": 2, "num_experts": 8, "num_experts_per_tok": 2, "decoder_sparse_step": 1,
           "mlp_only_layers": [], "moe_intermediate_size": 64, "rms_norm_eps": 1e-6, "vocab_size": 256,
           "num_key_value_heads": 1, "head_dim": 32, "rope_theta": 1e4, "tie_word_embeddings": False,
           "max_position_embeddings": 128, "norm_topk_prob": True}
DENSE_CFG = {"model_type": "qwen3", "hidden_size": 64, "num_hidden_layers": 2, "intermediate_size": 128,
             "num_attention_heads": 2, "rms_norm_eps": 1e-6, "vocab_size": 256, "num_key_value_heads": 1,
             "max_position_embeddings": 128, "rope_theta": 1e4, "head_dim": 32, "tie_word_embeddings": True}
Q = {"group_size": 64, "bits": 4}


def raises(fn, *words):
    try:
        fn()
    except launch.Unsupported as e:
        assert all(w.lower() in str(e).lower() for w in words), (str(e), words)
        return str(e)
    raise AssertionError(f"no Unsupported raised, wanted {words}")


def check_catalog_filter():
    assert launch.known_type("qwen3_moe") and launch.known_type("qwen3") and launch.known_type("mixtral")
    assert not launch.known_type("no_such_model") and not launch.known_type(None)
    assert not launch.known_type("../evil")
    moe = launch.inspect_model({**MOE_CFG, "quantization": Q})
    assert moe["kind"] == "moe" and moe["moe_layers"] == 2 and moe["experts"] == 8 and moe["top_k"] == 2
    # 4-bit g64 with bf16 scales and biases: 3 projections of 64 x 64.
    assert moe["expert_bytes"] == 3 * (64 * 64 // 2 + 2 * 2 * 64 * 64 // 64), moe["expert_bytes"]
    assert moe["dense_bytes"] < moe["total_bytes"] - 8 * 2 * moe["expert_bytes"] + 1
    # Unquantized experts are not streamed: the whole model stays in memory.
    assert launch.inspect_model(dict(MOE_CFG))["kind"] == "resident"
    d = launch.inspect_model({**DENSE_CFG, "quantization": Q})
    assert d["kind"] == "resident" and d["read_bytes"] == d["total_bytes"]          # tied: lm_head is the table
    raises(lambda: launch.inspect_model({**DENSE_CFG, "quantization_config": {"quant_method": "gptq"}}),
           "gptq", "mlx-community")
    raises(lambda: launch.inspect_model({**DENSE_CFG, "model_type": "no_such_model"}), "not in the installed")
    raises(lambda: launch.inspect_model({"hidden_size": 4}), "model_type")
    # The EXL3 record size matches the converted pack's 1,862,400 bytes at 3 bits.
    assert launch.exl3_expert_bytes({"hidden": 2560, "d_ff": 640}, 3) == 1862400
    print("1. catalog filter: known model types, MoE vs dense, PyTorch-quantized / unknown refused")


def tiny_checkpoint(out, per_expert=False, cfg=MOE_CFG):
    """A quantized tiny model saved the way mlx_lm.convert saves it (stacked
    switch_mlp), or with one tensor per expert (the older layout)."""
    from mlx_lm.utils import _get_classes
    mx.random.seed(0)                       # the same weights in either layout
    M, A = _get_classes(cfg)
    model = M(A.from_dict(cfg))
    model.set_dtype(mx.bfloat16)
    nn.quantize(model, group_size=64, bits=4)
    mx.eval(model.parameters())
    w = dict(tree_flatten(model.parameters()))
    if per_expert:
        for k in [k for k in w if ".switch_mlp." in k]:
            pre, rest = k.split(".switch_mlp.")
            for e in range(w[k].shape[0]):
                w[f"{pre}.experts.{e}.{rest}"] = w[k][e]
            del w[k]
    out.mkdir(parents=True)
    keys = sorted(w)
    half = len(keys) // 2
    shards = {"model-00001-of-00002.safetensors": keys[:half], "model-00002-of-00002.safetensors": keys[half:]}
    for name, ks in shards.items():
        mx.save_safetensors(str(out / name), {k: w[k] for k in ks})
    (out / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {k: n for n, ks in shards.items() for k in ks}}))
    (out / "config.json").write_text(json.dumps({**cfg, "quantization": Q}))
    return w


def pack(src, out, *extra):
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "stream_pack.py"), "--from-dir", str(src), "--out", str(out),
                        *extra], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr


def check_local_folders(tmp):
    g = tmp / "gguf"
    g.mkdir()
    (g / "model-q4.gguf").write_bytes(b"GGUF")
    raises(lambda: launch.local_entry(g), "gguf")
    n = tmp / "empty"
    n.mkdir()
    raises(lambda: launch.local_entry(n), "config.json")
    pt = tmp / "torch"
    pt.mkdir()
    (pt / "config.json").write_text(json.dumps(DENSE_CFG))
    (pt / "pytorch_model.bin").write_bytes(b"0")
    raises(lambda: launch.local_entry(pt), "safetensors")
    src = tmp / "moe_src"
    tiny_checkpoint(src)
    e = launch.local_entry(src)
    assert e["facts"]["kind"] == "moe" and e.get("needs_pack"), e
    packed = tmp / "moe_packed"
    pack(src, packed)
    e2 = launch.local_entry(packed)
    assert not e2.get("needs_pack") and e2["facts"]["expert_bytes"] == e["facts"]["expert_bytes"], (e2, e)
    print("2. local folders: GGUF / no config / PyTorch .bin refused; an unpacked MoE gets packed, a packed one is ready")
    return src


def check_pack(tmp, stacked_src):
    """One tensor per expert packs to the same experts.bin as the stacked layout;
    --skip-experts keeps no expert bytes but their quantization entries; a run
    stopped after a shard and run again gives the same pack."""
    per = tmp / "per_src"
    tiny_checkpoint(per, per_expert=True)
    a, b = tmp / "pk_stacked", tmp / "pk_per"
    pack(stacked_src, a)
    pack(per, b)
    assert (a / "experts.bin").read_bytes() == (b / "experts.bin").read_bytes()
    ia, ib = (json.loads((d / "experts_index.json").read_text()) for d in (a, b))
    assert ia == ib, (ia, ib)
    wm = json.loads((b / "model.safetensors.index.json").read_text())["weight_map"]
    assert not any(".experts." in k or ".switch_mlp." in k for k in wm)
    qa = json.loads((a / "config.json").read_text())["quantization"]
    qb = json.loads((b / "config.json").read_text())["quantization"]
    assert qa == qb and "model.layers.0.mlp.switch_mlp.gate_proj" in qb

    s = tmp / "pk_skip"
    pack(stacked_src, s, "--skip-experts")
    assert (s / "experts.bin").stat().st_size == 0
    assert json.loads((s / "config.json").read_text())["quantization"] == qa
    assert json.loads((s / "model.safetensors.index.json").read_text())["weight_map"] == \
        json.loads((a / "model.safetensors.index.json").read_text())["weight_map"]

    # Resume: stop after the first shard (a state file is what a stopped run leaves).
    r = tmp / "pk_resume"
    code = (f"import sys; sys.path[:0] = [{str(ROOT)!r}, {str(ROOT / 'tools')!r}]; "
            f"sys.argv = ['stream_pack.py', '--from-dir', "
            f"{str(stacked_src)!r}, '--out', {str(r)!r}]\n"
            "import stream_pack\n"
            "real = stream_pack.pack_triples\n"
            "calls = []\n"
            "def stop(*a, **k):\n"
            "    calls.append(1)\n"
            "    if len(calls) > 2: raise SystemExit(3)\n"
            "    return real(*a, **k)\n"
            "stream_pack.pack_triples = stop\n"
            "stream_pack.main()\n")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
    assert p.returncode == 3 and (r / ".pack_state.json").exists(), p.stdout + p.stderr
    pack(stacked_src, r)
    assert not (r / ".pack_state.json").exists()
    for name in ("experts.bin", "experts_index.json", "model.safetensors.index.json", "config.json"):
        assert (r / name).read_bytes() == (a / name).read_bytes(), name
    print("3. stream_pack: per-expert == stacked (experts.bin, index, config), --skip-experts, resume")


def check_partial_fetch(tmp, src):
    """fetch_partial over byte ranges gives a valid file of the kept tensors."""
    shard = src / "model-00001-of-00002.safetensors"
    blob = shard.read_bytes()

    class R(io.BytesIO):
        status = 206

    def http(url, lo, hi):
        return R(blob[lo:hi + 1])
    real = stream_pack._http
    stream_pack._http, stream_pack.GAP = http, 64      # small gap: several spans
    try:
        out = tmp / "partial.safetensors"
        hdr = stream_pack.fetch_partial("x", out, stream_pack.is_expert)
    finally:
        stream_pack._http, stream_pack.GAP = real, 8 << 20
    got, want = mx.load(str(out)), mx.load(str(shard))
    assert set(hdr) - {"__metadata__"} == set(want)
    assert set(got) == {k for k in want if not stream_pack.is_expert(k)} and got
    assert all(mx.array_equal(got[k], want[k]).item() for k in got)
    print(f"4. partial fetch: {len(got)} of {len(want)} tensors, byte ranges, equal to the source")


def prof(backend="metal", ram=16, avail=14.0, gpu_mem=None, ws=12.7):
    return {"backend": backend, "ram": ram * 2**30, "gpu_mem": (gpu_mem or ram) * 1e9,
            "working_set": ws * 1e9, "gpu_bw": 94.9, "now": {"avail_gb": avail}}


FN = {"kind": "moe", "dense_bytes": 2.6985e9, "read_bytes": 2.6985e9, "moe_layers": 48, "experts": 512,
      "top_k": 10, "expert_bytes": 1862400, "recipe": "store", "gated": True, "native": True,
      "total_bytes": 50e9}
Q35 = {"kind": "moe", "dense_bytes": 1.1033e9, "read_bytes": 1.1033e9, "moe_layers": 40, "experts": 256,
       "top_k": 8, "expert_bytes": 1769472, "recipe": "arena", "gated": True, "total_bytes": 20e9}
Q4B = {"kind": "resident", "dense_bytes": 2.26e9, "read_bytes": 2.26e9, "total_bytes": 2.26e9}


def check_settings():
    import moe_stream
    assert machine.PREFILL_PCT == moe_stream.PREFILL_PCT
    s = machine.settings(prof(), {**FN, "native_ok": True})
    # 16 GB: the store's floor is the engine's 35% share (6.0 GB), its ceiling what
    # the swap line allows now; cache bias as gated; the native reader.
    assert s["ok"] and s["args"][:4] == ["--mem-peak-gb", "6", "--mem-peak-max-gb", "9.2"], s
    assert s["args"][4:] == ["--cache-bias", "0.5", "--bias-budget", "0.2"] and s["env"] == {"MOE_NATIVE": "1", "ENGINE_BIAS_GATED": "1"}
    busy = machine.settings(prof(avail=8.0), {**FN, "native_ok": False})
    assert busy["args"][:2] == ["--mem-peak-gb", "5"] and "--mem-peak-max-gb" not in busy["args"]
    assert busy["env"] == {"ENGINE_BIAS_GATED": "1"} and busy["store_gb"] < s["store_gb"]
    assert not machine.settings(prof(ram=8, avail=5.0, ws=5.4), FN)["ok"]
    ungated = machine.settings(prof(), {**FN, "gated": False, "native": False})
    assert "--cache-bias" not in ungated["args"] and "ENGINE_BIAS_GATED" not in ungated["env"]
    a = machine.settings(prof(), Q35)
    assert a["args"][:4] == ["--prefill-step", "0", "--keep-convs", "4"] and a["ok"] and "--mem-peak-gb" not in a["args"]
    assert "MOE_NATIVE" not in a["env"]     # the arena recipe takes the native agent only when it is ready
    an = machine.settings(prof(), {**Q35, "native_ok": True})["env"]
    assert an["MOE_NATIVE"] == "1" and an["MOE_NATIVE_MTP"] == "1" and "MOE_NATIVE_MTP" not in a["env"]
    c = machine.settings(prof("cuda", ram=14, gpu_mem=17.1, avail=9.0), FN)
    # CUDA: the store gives VRAM back down to no pool (dense + overhead, 8 GB) and takes up to the fixed plan (16 GB).
    assert c["args"][:4] == ["--mem-peak-gb", "8", "--mem-peak-max-gb", "16"], c["args"]
    assert machine.settings(prof(), Q4B)["ok"] and machine.settings(prof(), Q4B)["args"] == []
    assert not machine.settings(prof(avail=3.0), Q4B)["ok"]
    cpu = machine.settings(prof("cpu"), FN)                     # Linux without NVIDIA: a plain answer, no Metal plan
    assert not cpu["ok"] and not cpu["args"] and "NVIDIA" in cpu["why"][-1], cpu
    print(f"6. flag choice: 16 GB idle {' '.join(s['args'][:4])}, busy {' '.join(busy['args'][:2])}, "
          f"8 GB refused, arena recipe, CUDA {' '.join(c['args'][:4])}, dense fits / not")


def check_estimate():
    m4 = {"backend": "metal", "gpu_bw": 94.9}
    two, one = {"gbps": 6.1, "lat_ms": 0.18}, {"gbps": 3.42, "lat_ms": 0.227}
    s = {"store_gb": 4.045, "args": ["--cache-bias", "0.5"]}
    e = machine.estimate(m4, FN, s, two, corr={})["tps"]     # the physics alone: no measured runs here
    assert 0.8 < e / 13.69 < 1.25, e                        # measured ru/en/py 13.83/14.19/13.04
    assert machine.estimate(m4, FN, s, one, corr={})["tps"] < e        # a slower drive
    assert machine.estimate(m4, FN, {**s, "store_gb": 8}, one, corr={})["tps"] > \
        machine.estimate(m4, FN, s, one, corr={})["tps"]
    unb = machine.estimate(m4, FN, {"store_gb": 4.045, "args": []}, one, corr={})
    assert unb["tps"] < machine.estimate(m4, FN, s, one, corr={})["tps"] and unb["miss"] > \
        machine.estimate(m4, FN, s, one, corr={})["miss"]
    d = machine.estimate(m4, Q4B, {}, None, corr={})["tps"]
    assert 0.9 < d / 41.7 < 1.1, d                           # measured 24.0 ms a token
    assert machine.miss_share(1.0, True) == 0 and machine.miss_share(1e-6, False) == 1.0
    # a range when the store's size is only known as floor..start, one number when it is fixed
    r = machine.estimate(m4, FN, {**s, "store_lo_gb": 1.0}, one, corr={})
    assert r["lo_tps"] < r["tps"] == r["hi_tps"] and r["runs"] == 0, r
    print(f"7. estimate: Flash-Next EXL3 {e:.1f} tok/s (measured 13.0-14.2), Qwen3-4B {d:.1f} (41.7)")


def check_profile_cache(tmp):
    real_cache, real_facts = machine.CACHE, machine.facts
    machine.CACHE = tmp / "cache"
    try:
        p1 = machine.profile(measure=False)
        f1 = list(machine.CACHE.glob("machine-*.json"))
        assert len(f1) == 1 and "now" not in json.loads(f1[0].read_text()) and "now" in p1
        machine.profile(measure=False)
        assert list(machine.CACHE.glob("machine-*.json")) == f1        # same key: reused
        machine.facts = lambda: {**real_facts(), "mlx": "9.9.9"}
        p2 = machine.profile(measure=False)
        assert p2["key"] != p1["key"] and len(list(machine.CACHE.glob("machine-*.json"))) == 2
    finally:
        machine.CACHE, machine.facts = real_cache, real_facts
    print("5. profile: cached per key, live readings not stored, a new MLX version measures again")


def entry(facts, **kw):
    return {"id": "m", "name": "M", "repo": "owner/M-4bit", "facts": facts, "disk": 8.5e9, "download": 8.5e9,
            "transient": 1e9, "recipe": "store", **kw}


Q15 = {"kind": "moe", "dense_bytes": 0.87e9, "read_bytes": 0.87e9, "moe_layers": 24, "experts": 60, "top_k": 4,
       "expert_bytes": 4866048, "bits": 4, "group": 64, "total_bytes": 8.5e9}


def check_variants(tmp):
    """Published first; Hugging Face siblings as given; made here only below the published
    bits, never for EXL3 or unquantized weights; measured notes where the entry has them."""
    sib = entry(dict(Q15, bits=8, expert_bytes=int(4866048 * 1.0625 / 0.5625)), id="m8", repo="owner/M-8bit")
    vs = launch.variants(entry(Q15, notes={"q3": "measured: +7.6%"}), tmp, [sib])
    assert [v["key"] for v in vs] == ["published", "owner/M-8bit", "q3"], vs
    assert vs[2]["note"] == "measured: +7.6%" and "close to the original" in vs[1]["note"]
    q3 = vs[2]["entry"]
    ratio = machine.bytes_per_weight(3, 64) / machine.bytes_per_weight(4, 64)
    assert q3["facts"]["expert_bytes"] == int(4866048 * ratio) and q3["make_bits"] == 3 and not q3["gated"]
    moved = 4866048 * 60 * 24
    assert q3["disk"] == int(moved * ratio) + 8.5e9 and q3["make_bytes"] == int(moved * (1 + ratio))
    v8 = launch.variants(entry(dict(Q15, bits=8, group=64)), tmp, [])
    assert [v["key"] for v in v8] == ["published", "q4", "q3"] and "rounded from the published 8-bit" in v8[1]["note"]
    assert [v["key"] for v in launch.variants(entry(dict(Q15, bits=3, format="exl3")), tmp, [])] == ["published"]
    assert [v["key"] for v in launch.variants(entry(dict(Q4B, bits=16, group=0)), tmp, [])] == ["published"]
    dense = launch.variants(entry(dict(Q4B, bits=4, group=64)), tmp, [])
    assert dense[1]["entry"]["facts"]["total_bytes"] == int(2.26e9 * ratio)
    print("8. variants: published first, siblings, made only below the published bits, notes")


def check_bits_estimate():
    """Bytes scale with bits: with every expert in memory a 3-bit copy gains its routed bytes,
    with little memory it also misses less; 8 bits cost their bytes."""
    m4 = {"backend": "metal", "gpu_bw": 94.9}
    one = {"gbps": 3.42, "lat_ms": 0.227}
    q3 = launch.made_entry(entry(Q15), 3, Path("/nonexistent"))["facts"]
    full = {"store_gb": 7.5, "args": []}            # every expert in memory: no reads
    a, b = machine.estimate(m4, Q15, full, one, corr={}), machine.estimate(m4, q3, full, one, corr={})
    r4, r3 = (f["read_bytes"] + f["top_k"] * f["moe_layers"] * f["expert_bytes"] for f in (Q15, q3))
    assert a["read_ms"] == b["read_ms"] == 0 and abs(b["tps"] / a["tps"] - r4 / r3) < 1e-9, (a, b)
    full_gain = b["tps"] / a["tps"]
    tight = {"store_gb": 1.0, "args": []}           # the same memory holds more 3-bit experts
    a, b = machine.estimate(m4, Q15, tight, one, corr={}), machine.estimate(m4, q3, tight, one, corr={})
    assert b["miss"] < a["miss"] and b["tps"] / a["tps"] > full_gain, (a, b)
    d4 = dict(Q4B, bits=4, group=64)
    d8 = dict(d4, bits=8, read_bytes=2.26e9 * 1.0625 / 0.5625)
    t4, t8 = (machine.estimate(m4, d, {}, None, corr={})["tps"] for d in (d4, d8))
    assert abs(t4 / t8 - 1.0625 / 0.5625) < 1e-6, (t4, t8)
    print(f"9. estimate per bits: 3-bit {full_gain:.2f}x with all experts held, {b['tps'] / a['tps']:.2f}x "
          f"when reads bind; 8-bit dense {t8:.0f} vs 4-bit {t4:.0f} tok/s")


def check_requant(tmp, src):
    """quantpack --from-pack: every 3-bit record is mx.quantize of the dequantized 4-bit one,
    the config names 3 bits, the rest is linked, a stopped run resumes to the same bytes, and
    the engine decodes the 3-bit pack exactly as mlx-lm does with those weights in memory."""
    import quantpack
    import stream_fast
    from mlx_lm.generate import generate_step
    from mlx_lm.utils import _get_classes
    p4 = tmp / "rq_src"
    pack(src, p4)
    q3 = tmp / "rq_q3"
    quantpack.requant_pack(p4, q3, 3, chunk=3)
    i4, i3 = (json.loads((d / "experts_index.json").read_text()) for d in (p4, q3))
    cfg = json.loads((q3 / "config.json").read_text())
    with open(p4 / "experts.bin", "rb") as f4, open(q3 / "experts.bin", "rb") as f3:
        for base, e in i4.items():
            assert cfg["quantization"][base] == {"group_size": 64, "bits": 3}
            f4.seek(e["offset"])
            raw = f4.read(e["stride"] * e["n_experts"])
            n, at, arrs = e["n_experts"], 0, []
            import numpy as np
            rec = np.frombuffer(raw, np.uint8).reshape(n, e["stride"])
            for p in e["parts"]:
                a = mx.array(np.ascontiguousarray(rec[:, at:at + p["bytes"]]).view(
                    np.uint32 if p["dtype"] == "U32" else np.uint16).reshape(n, *p["shape"]))
                arrs.append(a.view(mx.bfloat16) if p["dtype"] == "BF16" else a)
                at += p["bytes"]
            want = mx.quantize(mx.dequantize(*arrs, group_size=64, bits=4), group_size=64, bits=3)
            f3.seek(i3[base]["offset"])
            got = f3.read(i3[base]["stride"] * n)
            rows = [np.frombuffer(memoryview(a.view(mx.uint16) if a.dtype == mx.bfloat16 else a), np.uint8)
                    .reshape(n, -1) for a in want]
            assert got == np.concatenate(rows, axis=1).tobytes(), base
    assert (q3 / "model-00001-of-00002.safetensors").is_symlink()
    # Resume: a state file left after one projection gives the same experts.bin.
    r = tmp / "rq_resume"
    code = (f"import sys; sys.path[:0] = [{str(ROOT)!r}, {str(ROOT / 'tools')!r}]\n"
            "import quantpack, mlx.core as mx\n"
            "real, n = mx.quantize, [0]\n"
            "def stop(*a, **k):\n"
            "    n[0] += 1\n"
            "    if n[0] > 4: raise SystemExit(3)\n"
            "    return real(*a, **k)\n"
            "quantpack.mx.quantize = stop\n"
            f"quantpack.requant_pack({str(p4)!r}, {str(r)!r}, 3, chunk=3)\n")
    pr = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
    assert pr.returncode == 3 and (r / ".requant_state.json").exists(), pr.stdout + pr.stderr
    quantpack.requant_pack(p4, r, 3, chunk=3)
    assert (r / "experts.bin").read_bytes() == (q3 / "experts.bin").read_bytes()
    # The engine on the 3-bit pack against mlx-lm with the same 3-bit weights resident.
    M, A = _get_classes(MOE_CFG)
    ref = M(A.from_dict(MOE_CFG))
    ref.set_dtype(mx.bfloat16)
    nn.quantize(ref, group_size=64, bits=4, class_predicate=lambda p, m: hasattr(m, "to_quantized") and (
        {"group_size": 64, "bits": 3} if ".switch_mlp." in p else True))
    w = {}
    for name in sorted(p for p in src.glob("*.safetensors")):
        w.update(mx.load(str(name)))
    for k in [k for k in w if ".switch_mlp." in k and k.endswith(".weight")]:
        b = k[:-len(".weight")]
        w[f"{b}.weight"], w[f"{b}.scales"], w[f"{b}.biases"] = mx.quantize(
            mx.dequantize(w[k], w[f"{b}.scales"], w[f"{b}.biases"], group_size=64, bits=4), group_size=64, bits=3)
    ref.load_weights(list(w.items()))
    prompt = mx.array([3, 17, 42, 5, 90, 11], dtype=mx.int32)
    want = [int(t) for t, _ in generate_step(prompt, ref, max_tokens=12)]
    model = _load_slim_no_tokenizer(q3)
    n, _ = stream_fast.install(model, q3, resident=4)
    assert n > 0
    got = [int(t) for t, _ in generate_step(prompt, model, max_tokens=12)]
    assert got == want, (got, want)
    # A dense model: every quantized tensor rounded, the config's default bits 3.
    dsrc, d3 = tmp / "rq_dense_src", tmp / "rq_dense_q3"
    w4 = tiny_checkpoint(dsrc, cfg=DENSE_CFG)
    quantpack.requant_pack(dsrc, d3, 3)
    assert json.loads((d3 / "config.json").read_text())["quantization"]["bits"] == 3
    w3 = {}
    for name in d3.glob("*.safetensors"):
        w3.update(mx.load(str(name)))
    for k in [k for k in w4 if k.endswith(".scales")]:
        b = k[:-len(".scales")]
        want3 = mx.quantize(mx.dequantize(w4[f"{b}.weight"], w4[k], w4[f"{b}.biases"], group_size=64, bits=4),
                            group_size=64, bits=3)
        assert all(mx.array_equal(x, y).item() for x, y in zip(want3, (w3[f"{b}.weight"], w3[k], w3[f"{b}.biases"])))
    dm = _load_slim_no_tokenizer(d3)
    dm.load_weights(list(w3.items()))
    assert len(list(generate_step(prompt, dm, max_tokens=4))) == 4
    print(f"10. quantize step: {len(i4)} projections 4 -> 3 bits, records == mx.quantize of the "
          f"dequantized 4-bit, resume, engine == mlx-lm on the 3-bit weights ({got[:6]}...); "
          f"dense: every quantized tensor rounded, loads and decodes")


def _load_slim_no_tokenizer(d):
    """The tiny pack has no tokenizer: load_slim_aware's model half only."""
    from mlx_lm.utils import load_model
    model, _ = load_model(Path(d), lazy=True, strict=False)
    return model


def check_auto(tmp):
    """One command: the recommendation prefers a model already on disk and then the
    best estimate; the last model is remembered; status and stop read the saved server."""
    import threading
    p = prof()
    disk = {"gbps": 3.42, "lat_ms": 0.227}
    models = tmp / "auto_models"
    models.mkdir()
    small = {"disk": 1e9, "download": 1e9, "transient": 0}
    on_disk = entry(Q15, id="on_disk", name="On disk", **small)
    slower = entry(Q15, id="slower", name="Slower", **small)
    fast = entry(dict(Q15, expert_bytes=1000000), id="fast", name="Fast", **small)
    no = entry(Q15, id="no", name="No", **small)
    (models / "on_disk").mkdir()
    (models / "on_disk" / "launcher.json").write_text(json.dumps(on_disk))
    table = [(on_disk, {"ok": True, "est": {"tps": 10.0}}),
             (slower, {"ok": True, "est": {"tps": 8.0}}),
             (fast, {"ok": True, "est": {"tps": 99.0}}),
             (no, {"ok": False, "est": None, "why": "does not fit"})]
    assert launch.recommend(table, models)[0]["id"] == "on_disk"          # on disk beats faster absent
    for e in (fast, slower):
        (models / e["id"]).mkdir()
        (models / e["id"] / "launcher.json").write_text(json.dumps(e))
    assert launch.recommend(table, models)[0]["id"] == "fast"             # then the best estimate
    assert launch.recommend([table[3]], models) is None

    real_cache, real_pid_command = machine.CACHE, launch.pid_command
    machine.CACHE = tmp / "auto_cache"
    try:
        assert launch.last_entry(models) is None
        launch.remember(fast, "q3")
        assert launch.last_entry(models)["id"] == "fast"
        assert launch.load_json(launch.cache_file("last.json"))["variant"] == "q3"
        e, variant, tight = launch.auto_pick(p, models, disk)
        assert e["id"] == "fast" and variant == "q3" and not tight     # the last model comes first
        launch.cache_file("last.json").unlink()
        fetch = lambda c: fast if c["id"] == launch.CURATED[0]["id"] else \
            entry(Q15, id=c["id"], name=c["name"], **small)
        e, variant, tight = launch.auto_pick(p, models, disk, fetch=fetch)
        assert e["id"] == "fast" and variant == "published" and not tight  # else the recommendation
        e, variant, tight = launch.auto_pick(prof(avail=0.2), models, disk, fetch=fetch)
        assert e["id"] == "fast" and tight                                # busy: the idle choice
        said = []
        launch.auto_pick(p, models, disk, fetch=lambda c: entry(Q15, id=c["id"], name=c["name"], **small) if c["id"] != launch.CURATED[-1]["id"]
                         else entry(Q15, id=c["id"], name="Huge", disk=1e15, download=1e15, transient=0), log=said.append)
        assert any("Skipped Huge" in m for m in said), said               # a larger model left out says why
        assert isinstance(launch.hf_problem(ModuleNotFoundError("mlx")), str) and "not installed" in launch.hf_problem(ModuleNotFoundError("mlx"))
        assert "Hugging Face" in launch.hf_problem(OSError("net"))        # only a real failure blames the network
        shown = io.StringIO()
        with contextlib.redirect_stdout(shown):
            launch.show([(entry(Q15, id=launch.CURATED[0]["id"], name="Tested one"), [])], models, disk)
        assert f"--model {launch.CURATED[0]['id']}" in shown.getvalue()   # --list names the id --model takes
        assert launch.stop_hint().startswith("./larder stop")
        assert launch.engine_log.verbose is False                         # the launcher prints no engine notes

        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        threading.Thread(target=proc.wait, daemon=True).start()
        launch.record_server(fast, models / "fast", 8899, proc, tmp / "server.log")
        st, ours = launch.server_info()
        assert st["pid"] == proc.pid and not ours                         # not a server.py: not ours
        launch.pid_command = lambda pid: f"{sys.executable} server.py --model {models / 'fast'}"
        st, ours = launch.server_info()
        assert ours and st["port"] == 8899 and st["name"] == "Fast"
        assert launch.show_status() == 0
        assert launch.stop_server() == 0 and proc.poll() is not None
        assert not launch.cache_file("server.json").exists()
        launch.clear_server()
        assert launch.stop_server() == 0                                  # nothing left to stop
    finally:
        machine.CACHE, launch.pid_command = real_cache, real_pid_command
    print("11. one command: on disk first, then the best estimate, the last model remembered, "
          "status and stop on the saved server")


def check_menu_backend(tmp):
    """The full-screen menu's backend turns the launcher's judged variants into what the menu shows,
    reads the saved state, counts installed bytes, and runs an install in a child process."""
    import types
    models = tmp / "menu_models"
    models.mkdir()
    real_sib, real_cache, real_du = launch.siblings, machine.CACHE, shutil.disk_usage
    launch.siblings = lambda e, limit=4: []                  # no network in a test
    machine.CACHE = tmp / "menu_cache"
    shutil.disk_usage = lambda p: real_du(p)._replace(free=1 << 40)   # this computer's free disk must not decide the test
    try:
        be = launch.TuiBackend(types.SimpleNamespace(refresh=False, port=8800), models)
        be.prof, be.disk = prof(), {"gbps": 3.42, "lat_ms": 0.227}
        # Without --devices the menu starts with this computer only and reads no other computer's list.
        import remote
        real_list = remote.device_list
        remote.device_list = lambda: (_ for _ in ()).throw(AssertionError("other computers listed at start"))
        try:
            off = launch.TuiBackend(types.SimpleNamespace(refresh=False, port=8800, devices=False), models)
            off.prof = {**prof(), "chip": "a test chip", "arch": "arm64"}
            assert [d.kind for d in off.device_list()] == ["local"]
        finally:
            remote.device_list = real_list
        e = entry(Q15, id="m", name="M", about="a test model", license="mit")
        m = be._model(e, True)
        assert m.name == "M" and m.tested and m.kind == "MoE, 60 experts x 24 layers" and m.license == "mit"
        pub, q3 = m.variants
        assert (pub.key, q3.key) == ("published", "q3") and pub.ok and q3.ok
        assert pub.status == "download 8.5 GB" and not pub.ready and pub.entry_id == "m" and pub.tps > 0
        assert pub.expected == 8.5e9 and q3.make_bits == 3 and q3.entry_id == "m-q3" and q3.expected == q3.disk
        assert "to make after a 8.5 GB download" in q3.status
        assert be.dest(pub) == str(models / "m") and be.last() is None and be.attached() is None
        (models / "m").mkdir()
        (models / "m" / "experts.bin").write_bytes(b"x" * 1000)
        assert be.progress(pub) == 1000
        (models / "m" / "launcher.json").write_text(json.dumps(e))
        again = be._model(e, True).variants[0]
        assert again.ready and again.status == "ready"
        launch.remember(e, "published")
        assert be.last() == "m"
        try:
            be.local(str(tmp / "no_such_folder"))
            raise AssertionError("a missing folder was accepted")
        except ValueError:
            pass
        # an install runs `launch.py --prepare` in a child (here a model that is already prepared)
        job = types.SimpleNamespace(say=lambda t: None, cancelled=lambda: False)
        be.install(again, job)
        assert be.child.returncode == 0
        # a start: a stand-in server answers /v1/models; the result carries the addresses and the state is saved
        stub = ("import http.server, json, sys\n"
                "class H(http.server.BaseHTTPRequestHandler):\n"
                "    def do_GET(self):\n"
                "        self.send_response(200); self.end_headers(); self.wfile.write(json.dumps({'data': [{'id': 'fake'}]}).encode())\n"
                "    def log_message(self, *a): pass\n"
                "http.server.HTTPServer(('127.0.0.1', int(sys.argv[1])), H).serve_forever()\n")
        real_start, real_plan = launch.start_server, launch.engine_plan
        launch.start_server = lambda d, s, port, log, quiet=False: subprocess.Popen(
            [sys.executable, "-c", stub, str(port)], stdout=open(log, "w"), stderr=subprocess.STDOUT)
        launch.engine_plan = lambda e, prof, disk, tight=False, native=True, d=None: (
            {"args": [], "env": {}}, {"tps": 12.3})
        try:
            be.args.port = 8870
            phases = []
            job = types.SimpleNamespace(say=phases.append, cancelled=lambda: False)
            got = be.start(again, job)
            assert got["name"] == "M" and got["model_id"] == "fake" and got["tps"] == 12.3
            assert got["url"] == f"http://127.0.0.1:{be.proc.args[-1]}" and got["api"] == got["url"] + "/v1"
            assert launch.load_json(launch.cache_file("server.json"))["pid"] == be.proc.pid and phases
            proc = be.proc
            be.stop()
            assert proc.poll() is not None and be.proc is None and not launch.cache_file("server.json").exists()
            job = types.SimpleNamespace(say=lambda t: None, cancelled=lambda: True)
            assert be.start(again, job) is None and be.proc is None          # stopped while loading: nothing left
        finally:
            launch.start_server, launch.engine_plan = real_start, real_plan
        # unavailable models are listed with the reason, not dropped
        real_hf = launch.hf_entry
        launch.hf_entry = lambda c: (_ for _ in ()).throw(launch.Unsupported("model type 'x' is not in the installed mlx-lm"))
        try:
            got = [x for x in be.models(lambda t: None)]
        finally:
            launch.hf_entry = real_hf
        assert len(got) == len(launch.CURATED) and all(g.unavailable and g.tested for g in got)
        assert "requirements.txt" in got[0].unavailable
    finally:
        launch.siblings, machine.CACHE = real_sib, real_cache
        shutil.disk_usage = real_du
    print("12. menu backend: variants as the menu shows them, ready / progress / last, an install in a child "
          "process, unavailable models listed with the reason")


def check_listing_json(tmp):
    """`--list --json`: one object another computer's menu can read, with the machine line and
    every model's variants, or the reason a model is unavailable, never a traceback."""
    models = tmp / "json_models"
    models.mkdir()
    p = dict(prof(), chip="Apple M4", arch="arm64", os="macOS 27.0", gpu="Apple M4",
             now={"avail_gb": 14.0, "disks": [{"dev": machine.volume_of(models), "mount": "/", "free": 200e9},
                                               {"dev": -1, "mount": "/Volumes/Other", "free": 123e9}]})
    disk = {"gbps": 3.42, "lat_ms": 0.227}
    real = launch.CURATED, launch.siblings, launch.hf_entry
    real_du = shutil.disk_usage
    shutil.disk_usage = lambda p: real_du(p)._replace(free=1 << 40)   # this machine's free disk must not decide the test
    launch.CURATED = [entry(Q15, id="m", name="M", about="a test model", license="mit")]
    launch.siblings = lambda e, limit=4: []
    launch.hf_entry = lambda c: c
    try:
        got = launch.listing(p, models, disk)
        assert set(got) == {"machine", "engine", "models"}, got
        assert set(got["engine"]) == {"protocol", "commit"} and got["engine"]["protocol"] == 1
        mach, m = got["machine"], got["models"][0]
        assert mach["chip"] == "Apple M4" and mach["ram_gb"] == 16 and mach["os"] == "macOS 27.0"
        assert mach["models_dir"] == str(models) and mach["ssd_gbps"] == 3.42 and mach["free_gb"] > 0
        assert mach["gpu"] == "Apple M4" and mach["others"] == [("/Volumes/Other", 123.0)]
        assert (m["id"], m["name"], m["tested"], m["unavailable"]) == ("m", "M", True, "")
        assert m["kind"] == "MoE, 60 experts x 24 layers" and m["license"] == "mit"
        keys = [v["key"] for v in m["variants"]]
        assert keys == ["published", "q3"], keys
        fields = {"key", "label", "size", "tps", "ok", "why", "status", "note", "ready", "make_bits",
                  "download", "disk", "expected", "entry_id"}
        assert all(fields <= set(v) for v in m["variants"])
        pub, q3 = m["variants"]
        assert pub["entry_id"] == "m" and pub["ready"] is False and pub["status"].startswith("download")
        assert pub["disk"] == 8.5e9 and pub["expected"] == 8.5e9 and pub["tps"] and pub["ok"]
        assert q3["make_bits"] == 3 and q3["download"] == 8.5e9 and q3["expected"] == q3["disk"] > 8.5e9
        assert json.loads(json.dumps(got))["models"][0]["variants"][0]["key"] == "published"
        # a model the installed engine cannot run is a row that says why, not an exception
        def unsupported(c):
            raise launch.Unsupported("model type 'x' is not in the installed mlx-lm")
        launch.hf_entry = unsupported
        bad = launch.listing(p, models, disk)["models"][0]
        assert bad["unavailable"].startswith("model type 'x'") and "requirements.txt" in bad["unavailable"]
        assert bad["variants"] == [] and bad["tested"] is True
    finally:
        launch.CURATED, launch.siblings, launch.hf_entry = real
        shutil.disk_usage = real_du
    print("13. --list --json: the machine line, variants with ready/download/make and speeds, unavailable "
          "models with the reason, one JSON object")


def check_remote_commands(tmp):
    """The exact ssh commands per device kind, the preset script kept whole, the listing parser,
    the readiness filter and a checkout with no preset script."""
    import remote
    import tui
    studio = tui.Device("studio", "studio", "preset", "presets.sh", ("studio",), "engine at ~/engine",
                        "Apple M4, 16 GB", "ready", measured=(13.8, 14.2), preset="studio", dir="~/engine",
                        python="~/venv/bin/python",
                        commands={"flash": {"port": 8790, "id": "flash-next-exl3"}})
    desk = tui.Device("desk", "desk", "tailscale", "tailscale", (), "engine at ~/larder", "linux", "ready")
    assert remote.list_argv(studio)[-1] == ("cd ~/engine && ~/venv/bin/python launch.py --list --json")
    assert remote.list_argv(desk)[-1] == "cd ~/larder && ./larder --list --json"
    assert remote.list_argv(studio)[:len(remote.SSH)] == remote.SSH
    # a preset: its script (lock, stop trap) arrives from the preset script and runs in one ssh session
    seen = []

    def fake_run(argv, timeout=20):
        seen.append(argv)
        return 0, "set -e\ncd ~/engine\nuntil mkdir /tmp/studio_gpu.lock; do sleep 20; done\n", ""
    argv, port = remote.start_argv(studio, "flash-next-exl3", 8800, "flash", run=fake_run)
    assert seen and seen[0][:3] == ["bash", str(remote.HERE / remote.PRESET_SCRIPT), "--print"]
    assert seen[0][3:] == ["studio", "flash"] and port == 8790
    n = len(remote.SSH_START)
    assert argv[:n] == remote.SSH_START and argv[n] == "-L" and argv[n + 1] == "8800:127.0.0.1:8790"
    assert argv[n + 2] == "studio" and argv[n + 3].startswith("set -e") and "studio_gpu.lock" in argv[n + 3]
    # another computer: its launcher downloads if needed, and its port is the same as the forward
    argv, port = remote.start_argv(desk, "demo-4b", 8801, run=fake_run)
    assert port == 8801 and argv[n + 1] == "8801:127.0.0.1:8801"
    assert argv[n + 3] == "cd ~/larder && ./larder --auto --model demo-4b --no-browser --port 8801"
    assert remote.stop_argv(studio) is None and remote.stop_argv(desk)[-1] == "cd ~/larder && ./larder stop"
    # readiness: only the remote's own serving line, never ssh noise or the line before it serves
    assert remote.serving_line("serving on http://127.0.0.1:8790", 8790)
    assert remote.serving_line("Uvicorn running: ready: 127.0.0.1:8790", 8790)
    assert not remote.serving_line("channel 3: open failed: connect failed: Connection refused", 8790)
    assert not remote.serving_line("the chat is at http://127.0.0.1:8790/", 8790)
    assert not remote.serving_line("serving on 127.0.0.1:8791", 8790)
    # its --list --json answer, with the measured speed only on the preset's flagship
    data = {"machine": {"chip": "Apple M4"}, "models": [
        {"id": "flash-next-exl3", "name": "Flash", "kind": "MoE", "tested": True,
         "variants": [{"key": "published", "label": "as published, 4-bit", "size": 50e9, "tps": 14, "ok": True,
                       "why": "", "status": "ready", "ready": True, "entry_id": "flash-next-exl3"},
                      {"key": "q3", "label": "3-bit", "size": 40e9, "tps": 17, "ok": True, "why": "",
                       "status": "~40 min to make", "make_bits": 3, "disk": 40e9, "entry_id": "flash-next-exl3-q3"}]},
        {"id": "qwen3.6-35b", "name": "Qwen", "kind": "MoE", "tested": True,
         "variants": [{"key": "published", "label": "as published, 4-bit", "size": 20e9, "tps": 30, "ok": False,
                       "why": "needs 41 GB free, 12 GB free", "status": "can't run: needs 41 GB free",
                       "ready": False, "entry_id": "qwen3.6-35b"}]}]}
    noisy = "looking around...\n" + json.dumps(data) + "\n"
    got = remote.parse_listing(noisy)
    assert got["machine"]["chip"] == "Apple M4" and len(got["models"]) == 2
    for bad in ("nothing here", "{\"models\": 3}"):
        try:
            remote.parse_listing(bad)
            raise AssertionError("bad listing accepted")
        except ValueError:
            pass
    rows = remote.models_from_listing(got, (13.8, 14.2),
                                      ready_ids=("flash-next-exl3", "qwen3.6-35b"))
    flash, qwen = rows
    assert flash.name == "Flash" and flash.variants[0].measured == (13.8, 14.2)
    assert flash.variants[1].measured == () and flash.variants[1].make_bits == 3
    # a preset command runs a prepared cache directly, whatever the launcher's models folder says
    assert qwen.variants[0].ready and qwen.variants[0].ok and qwen.variants[0].status == "ready"
    assert qwen.variants[0].measured == () and qwen.variants[0].tps == 30
    real_tables = remote.PRESET_NAME, remote.MEASURED, remote.HARDWARE
    remote.PRESET_NAME = {"studio": "studio", "rig": "the rig"}
    remote.MEASURED = {"studio": (13.8, 14.2), "rig": (32.0, 34.0)}
    remote.HARDWARE = {"studio": "Apple M4, 16 GB", "rig": "RTX 5060 Ti 16 GB, 14 GB RAM"}
    try:
        devs = remote.parse_presets(json.dumps({"presets": [
            {"kind": "studio", "host": "studio", "dir": "~/engine", "python": "~/venv/bin/python",
             "models": {"flash": {"port": 8790, "id": "flash-next-exl3"}}},
            {"kind": "studio", "host": "studio", "dir": "~/engine", "python": "~/venv/bin/python",
             "models": {"35b": {"port": 8791, "id": "qwen3.6-35b"}}},
            {"kind": "rig", "host": "me@rig", "dir": "~/engine", "python": "~/venv/bin/python",
             "models": {"flash": {"port": 8792, "id": "flash-next-exl3"}}}]}))
    finally:
        remote.PRESET_NAME, remote.MEASURED, remote.HARDWARE = real_tables
    assert [d.name for d in devs] == ["studio", "the rig"], devs
    assert set(devs[0].commands) == {"flash", "35b"} and devs[1].commands["flash"]["port"] == 8792
    assert devs[0].measured == (13.8, 14.2) and devs[1].measured == (32.0, 34.0)
    # with no preset tables a preset still lists, under its key and without a measured speed
    bare = remote.parse_presets(json.dumps({"presets": [
        {"kind": "lab", "host": "lab", "dir": "~/engine", "models": {"flash": {"port": 8790, "id": "x"}}}]}))
    assert [(d.name, d.measured, d.hardware) for d in bare] == [("lab", (), "")], bare
    # a checkout with no preset script (the public tree) has no presets, and starting one says so
    real_here, remote.HERE = remote.HERE, Path(tmp)
    try:
        assert remote.presets() == [] and remote.device_list(config=[], peers=[]) == []
        try:
            remote.start_argv(studio, "flash-next-exl3", 8800, "flash")
            raise AssertionError("a preset started with no script")
        except RuntimeError as e:
            assert "failed" in str(e), e
    finally:
        remote.HERE = real_here
    print("14. remote: exact ssh argv per kind, the preset script kept whole, readiness, JSON parsing, "
          "measured speed only on the flagship, no presets without the script")


def check_version_handshake(tmp):
    """`--version` and `--list --json` report the engine's protocol and commit; the probe turns a
    missing or behind protocol into "engine is older" and a newer one into "update this computer first"."""
    import engine_version
    import remote
    import tui
    assert engine_version.PROTOCOL == 1 and set(engine_version.version()) == {"protocol", "commit"}
    vjson = engine_version.version_json()
    assert json.loads(vjson)["protocol"] == 1
    desk = tui.Device("desk", "desk", "tailscale", "tailscale", (), "", "", "ready")
    preset = tui.Device("shed", "shed", "preset", "presets.sh", ("shed",), "", "", "ready",
                        preset="shed", dir="~/engine", python="~/venv/bin/python")
    assert remote.version_argv(desk)[-1] == "cd ~/larder && ./larder --version"
    assert remote.version_argv(preset)[-1] == "cd ~/engine && ~/venv/bin/python launch.py --version"
    assert remote.parse_version("junk") is None
    assert remote.parse_version("noise\n" + vjson + "\n")["protocol"] == 1
    assert remote.version_state({}) == "older"
    assert remote.version_state({"protocol": 0}) == "older"
    assert remote.version_state({"protocol": 1}) == "ok"
    assert remote.version_state({"protocol": 2}) == "newer"
    # a probe whose engine has no --version (no VER line) reads older, not ready
    d = tui.Device("old", "old", "tailscale", "tailscale", (), "", "", "checking")
    remote.apply_probe(d, 0, "ENGINE\nHW some chip, 8 GB\n", "")
    assert d.status == "engine-old" and d.engine_state == "older" and "u updates it" in d.detail, (d.status, d.detail)
    # a newer protocol tells the menu to update first
    d = tui.Device("new", "new", "tailscale", "tailscale", (), "", "", "checking")
    remote.apply_probe(d, 0, "ENGINE\nHW some chip, 8 GB\nVER " + json.dumps({"protocol": 2, "commit": "x"}) + "\n", "")
    assert d.status == "engine-new" and "update this computer" in d.detail, (d.status, d.detail)
    # the same protocol stays ready
    d = tui.Device("same", "same", "tailscale", "tailscale", (), "", "", "checking")
    remote.apply_probe(d, 0, "ENGINE\nHW some chip, 8 GB\nVER " + vjson + "\n", "")
    assert d.status == "ready" and d.engine["protocol"] == 1
    # --list --json carries the engine object too
    print("15. version handshake: --version and --list --json report protocol and commit, the menu reads "
          "older / newer / ok and says what to do")


def check_update_engine(tmp):
    """`u`: the ssh script fast-forwards from the device's origin, else from a bundle of this HEAD
    piped in, and refuses local changes or a diverged checkout with the reason."""
    import remote
    import subprocess
    import tui
    desk = tui.Device("desk", "desk", "tailscale", "tailscale", (), "", "", "ready")
    preset = tui.Device("shed", "shed", "preset", "presets.sh", ("shed",), "", "", "ready",
                        preset="shed", dir="~/engine", python="~/venv/bin/python")
    assert remote.update_argv(desk)[:len(remote.SSH)] == remote.SSH
    assert "mktemp -d" in remote.update_argv(desk)[-1] and "update.bundle" in remote.update_argv(desk)[-1]
    assert "$HOME/engine" in remote.update_argv(preset)[-1]
    ok, msg = remote.update_device(desk, bundle=b"fake",
                                   run=lambda argv, data, timeout=300: (0, "updated from origin\n", ""))
    assert ok and msg == "updated from origin"
    ok, msg = remote.update_device(desk, bundle=b"fake",
                                   run=lambda argv, data, timeout=300: (3, "", "update refused: the checkout there has diverged from this computer's engine\n"))
    assert not ok and "diverged" in msg
    # the script itself, against real local checkouts: bundle fast-forward, local changes and divergence
    def run(args):
        return subprocess.run(args, capture_output=True, text=True)
    src, dst = tmp / "upd_src", tmp / "upd_dst"
    run(["git", "init", "-q", str(src)])
    run(["git", "-C", str(src), "config", "user.email", "t@t"])
    run(["git", "-C", str(src), "config", "user.name", "t"])
    (src / "a").write_text("one")
    run(["git", "-C", str(src), "add", "a"])
    run(["git", "-C", str(src), "commit", "-qm", "one"])
    (src / "a").write_text("two")
    run(["git", "-C", str(src), "commit", "-qam", "two"])
    run(["git", "clone", "-q", str(src), str(dst)])
    run(["git", "-C", str(dst), "remote", "remove", "origin"])
    run(["git", "-C", str(dst), "reset", "-q", "--hard", "HEAD~1"])
    bundle = tmp / "upd.bundle"
    run(["git", "-C", str(src), "bundle", "create", str(bundle), "HEAD"])
    # the real stdin runner carries the binary bundle and hands text back
    code, out, err = remote._run_stdin(["bash", "-c", "wc -c; echo done >&2"], b"\x00\x01abc")
    assert code == 0 and out.strip() == "5" and "done" in err, (code, out, err)
    head = run(["git", "-C", str(src), "rev-parse", "HEAD"]).stdout.strip()
    script = remote.UPDATE.replace("@DIR@", '"' + str(dst) + '"').replace("@SHA@", head)
    (dst / "untracked").write_text("x")              # a stray file there does not block an update
    r = subprocess.run(["bash", "-c", script], input=bundle.read_bytes(), capture_output=True)
    out = (r.stdout + r.stderr).decode("utf-8", "replace")
    assert r.returncode == 0 and "updated from the bundle" in out, (r.returncode, out)
    r = subprocess.run(["bash", "-c", script], input=bundle.read_bytes(), capture_output=True)
    out = (r.stdout + r.stderr).decode("utf-8", "replace")
    assert r.returncode == 0 and "already up to date" in out, (r.returncode, out)
    (dst / "untracked").unlink()
    (dst / "a").write_text("edited there")
    r = subprocess.run(["bash", "-c", script], input=bundle.read_bytes(), capture_output=True)
    out = (r.stdout + r.stderr).decode("utf-8", "replace")
    assert r.returncode != 0 and "local changes" in out, (r.returncode, out)
    run(["git", "-C", str(dst), "checkout", "-q", "--", "a"])
    (src / "a").write_text("three")
    run(["git", "-C", str(src), "commit", "-qam", "three"])
    run(["git", "-C", str(src), "bundle", "create", str(bundle), "HEAD"])
    head = run(["git", "-C", str(src), "rev-parse", "HEAD"]).stdout.strip()
    script = remote.UPDATE.replace("@DIR@", '"' + str(dst) + '"').replace("@SHA@", head)
    (dst / "b").write_text("local")
    run(["git", "-C", str(dst), "add", "b"])
    run(["git", "-C", str(dst), "commit", "-qm", "local"])
    r = subprocess.run(["bash", "-c", script], input=bundle.read_bytes(), capture_output=True)
    out = (r.stdout + r.stderr).decode("utf-8", "replace")
    assert r.returncode != 0 and "diverged" in out, (r.returncode, out)
    print("16. update: the ssh script fast-forwards from origin or a stdin bundle and refuses local changes or a diverged checkout")


def check_install_engine(tmp):
    """`i`: a configured public URL runs install.sh over ssh; without one the bundle and the
    install.sh steps are shipped, run with MOE_REPO_URL pointing at the bundle, and the output streams."""
    import remote
    import tui
    desk = tui.Device("desk", "desk", "tailscale", "tailscale", (), "", "", "ready")
    preset = tui.Device("shed", "shed", "preset", "presets.sh", ("shed",), "", "", "ready",
                        preset="shed", dir="~/engine", python="~/venv/bin/python")
    fake_script = ('RAW_URL="${MOE_RAW_URL:-https://raw.githubusercontent.com/REPLACE-ME/larder/main}"\n'
                   'REPO_URL="${MOE_REPO_URL:-https://github.com/REPLACE-ME/larder.git}"\n')
    assert remote.install_urls(fake_script) == (None, None)
    raw, repo = remote.install_urls(
        'RAW_URL="${MOE_RAW_URL:-https://raw.githubusercontent.com/u/larder/main}"\n'
        'REPO_URL="${MOE_REPO_URL:-https://github.com/u/larder.git}"\n')
    assert raw.endswith("larder/main") and repo.endswith("larder.git")
    url_argv = remote.install_argv(desk, raw_url=raw)
    assert url_argv[:len(remote.SSH)] == remote.SSH
    assert url_argv[-1] == (f"curl -fsSL {raw.rstrip('/')}/install.sh | "
                            'INSTALL_DIR="$HOME/larder" bash')
    bundle_argv = remote.install_argv(desk)
    assert "__MOE_INSTALL_SCRIPT__" in bundle_argv[-1]
    assert "mktemp -d" in bundle_argv[-1] and "install.bundle" in bundle_argv[-1]
    assert "pip install" in bundle_argv[-1] and "native" in bundle_argv[-1]
    assert "$HOME/engine" in remote.install_argv(preset)[-1]
    seen = {}
    real_stream = remote._stream_install
    remote._stream_install = lambda argv, data, job, out, timeout=1800: (
        seen.update(argv=argv, data=data), (True, "installed ~/larder"))[1]
    try:
        ok, msg = remote.install_device(desk, bundle=b"fake", urls=(None, None))
        assert ok and msg == "installed ~/larder" and seen["data"] == b"fake"
    finally:
        remote._stream_install = real_stream
    # install.sh does its work in main() called on its last line, so a download cut short runs no step
    text = (remote.HERE / "install.sh").read_text()
    assert remote.install_urls(text) == remote.install_urls() and all(remote.install_urls())
    lines = text.splitlines(keepends=True)
    assert lines[-1].strip() == 'main "$@"', lines[-1]
    for n in list(range(1, len(lines), 9)) + [len(lines) - 1]:
        r = subprocess.run(["bash", "-s", "--", "--dry-run"], input="".join(lines[:n]), capture_output=True,
                           text=True, timeout=30)
        assert "[1/7]" not in r.stdout and "would run" not in r.stdout, (n, r.stdout[-300:])
    print("17. install: a configured URL runs the public installer, otherwise the bundle plus "
          "install.sh steps are shipped; the output streams and is cancellable; a cut download runs nothing")


def check_device_safety(tmp):
    """No ssh destination can be an option, a Tailscale peer is reached at its address, and the menu's r
    reads only the known devices, sweeps the network only after a y and never checks a sweep-only host."""
    import types
    import devices
    import remote
    import tui
    never = lambda *a, **k: (_ for _ in ()).throw(AssertionError("ssh ran"))
    # a destination that ssh would read as an option, or that is not a plain host, builds no argv at all
    for t in ("-oProxyCommand=touch /tmp/owned", "-p2222", "me@-oProxyCommand", "host;id", "a b", "$(id)",
              "h`id`", "host/x", "me@host\n-oX", "studio\n"):
        d = tui.Device("bad", t, "ssh", "test")
        assert remote.safe_target(t) is None, t
        argvs = (remote.version_argv(d), remote.probe_argv(d), remote.list_argv(d), remote.stop_argv(d),
                 remote.delete_argv(d, "m", "published"), remote.search_argv(d, "q"), remote.update_argv(d),
                 remote.install_argv(d), remote.install_argv(d, raw_url="https://example.com/x"))
        assert argvs == (None,) * len(argvs), (t, argvs)
        try:
            remote.start_argv(d, "m", 8800, run=never)
            raise AssertionError(f"started on {t!r}")
        except RuntimeError as err:
            assert "not a plain ssh address" in str(err), err
        assert remote.probe(d, run=never).status == "offline" and "not a plain ssh address" in d.detail
    n = len(remote.SSH)
    for t in ("studio", "me@rig", "rig.local", "192.168.1.14", "100.64.0.2", "fe80::1%en0", "me@[2001:db8::1]",
              "my-host_2"):
        d = tui.Device("good", t, "ssh", "test")
        assert remote.list_argv(d)[n] == t and remote.probe_argv(d)[:n + 1] == remote.SSH + [t], t
        argv, _ = remote.start_argv(d, "m", 8800)
        assert argv[len(remote.SSH_START) + 2] == t, argv
    # a peer names itself, so it is reached at its Tailscale address; a hostile name without one is refused
    peers = remote.tailscale_devices([
        {"name": "-oProxyCommand=touch /tmp/owned", "ip": "100.64.0.7", "os": "linux", "online": True},
        {"name": "desk", "ip": "100.64.0.2", "os": "linux", "online": True},
        {"name": "-oProxyCommand=x", "ip": "", "os": "linux", "online": True}])
    assert [p.target for p in peers] == ["100.64.0.7", "100.64.0.2", "-oProxyCommand=x"], peers
    assert remote.probe_argv(peers[0])[n] == "100.64.0.7" and remote.probe_argv(peers[2]) is None
    cfg = remote.parse_ssh_config("Host -oProxyCommand=x\n  HostName example.com\nHost box\n  User me\n")
    assert [remote.list_argv(d) is None for d in cfg] == [True, False], cfg
    # the running model a preset's lock names reads through the private names, else as its key
    real_names = remote.MODEL_NAME
    try:
        remote.MODEL_NAME = {}
        assert remote.classify(0, "ENGINE\nRUN Sun Nov 8 moe lab big 4242\n", "")[2] == "big"
        remote.MODEL_NAME = {"big": "Big Model"}
        assert remote.classify(0, "ENGINE\nRUN Sun Nov 8 moe lab big 4242\n", "")[2] == "Big Model"
    finally:
        remote.MODEL_NAME = real_names

    # The menu: r reads this computer (and, with --devices, the known devices) again and asks before a sweep.
    models = tmp / "safety_models"
    models.mkdir()
    calls = []
    found = {"me": "192.168.1.2", "subnet": "192.168.1.0/24", "error": None, "tailscale": [],
             "devices": [{"ip": "192.168.1.14", "name": "studio.local", "mac": "", "ports": [22], "note": "ssh"}]}
    real = devices.scan, devices.beacons, devices.subnet, remote.device_list, remote.probe
    devices.scan = lambda progress=None, **k: calls.append("sweep") or found
    devices.beacons = lambda timeout=1.5, port=None: calls.append("beacons") or []
    devices.subnet = lambda me=None: "192.168.1.0/24"
    remote.device_list = lambda: calls.append("known") or [tui.Device("desk", "desk", "ssh", "~/.ssh/config", ("desk",))]
    remote.probe = lambda dev, run=None: calls.append(("ssh", dev.target)) or dev

    def settle(app):
        end = time.time() + 20
        while time.time() < end:
            app.tick()
            if app.mode != "boot" and not any([app.boot_job, app.probe_job, app.scan_job, app.refresh_job]):
                return
            time.sleep(0.005)
        raise AssertionError(f"the menu did not settle ({app.mode})")

    def menu(devices_on):
        be = launch.TuiBackend(types.SimpleNamespace(refresh=False, port=8800, devices=devices_on), models)
        be.prof = {**prof(), "chip": "a test chip", "arch": "arm64"}
        be.info = lambda: {"chip": "a test chip", "ram_gb": 16, "gpu": "", "os": "", "models_dir": str(models),
                           "free_gb": 100, "ssd_gbps": 3, "others": []}
        app = tui.App(be, frozen=True)
        settle(app)
        return app

    try:
        app = menu(False)
        assert [d.kind for d in app.dev_rows] == ["local"] and calls == [], calls
        app.key("r")
        assert app.mode == "input" and "192.168.1.0/24" in app.input["prompt"] and "y/n" in app.input["prompt"]
        app.key("ESC")
        settle(app)
        assert calls == [], f"r without a yes swept, listed or ran ssh: {calls}"
        app.key("r")
        settle(app)
        app.input["buf"] = "y"
        app.key("ENTER")
        settle(app)
        assert "sweep" in calls and "known" in calls, calls
        assert {"studio.local", "desk"} <= {d.name for d in app.dev_rows}
        assert [c[1] for c in calls if isinstance(c, tuple)] == ["desk"], f"a sweep-only host was checked: {calls}"
        scan_row = next(d for d in app.dev_rows if d.name == "studio.local")
        assert app.b.device_probe(scan_row) is scan_row and calls.count(("ssh", "192.168.1.14")) == 0
        # --devices (the private default): the known devices at start, as before; r still asks before a sweep
        calls.clear()
        app = menu(True)
        assert calls.count("known") == 1 and ("ssh", "desk") in calls, calls
        app.key("r")
        app.key("ESC")
        settle(app)
        assert calls.count("known") == 2 and calls.count(("ssh", "desk")) == 2 and "sweep" not in calls, calls
    finally:
        devices.scan, devices.beacons, devices.subnet, remote.device_list, remote.probe = real
    print("17b. no ssh destination can be an option or a non-plain host, a Tailscale peer is reached at its "
          "address, r sweeps the network only after a y, and a host only the sweep found is never checked")


def check_disk_short(tmp):
    """A disk too full for a download is its own state ("needs 41 GB free on disk, 12.6 GB free"),
    never "can't run"; a memory or hardware limit keeps that wording, even on a full disk."""
    import remote
    models = tmp / "short_models"
    models.mkdir()
    disk = {"gbps": 3.42, "lat_ms": 0.227}
    real_du = shutil.disk_usage
    shutil.disk_usage = lambda p: real_du(p)._replace(free=int(12.6e9))
    try:
        big = entry(Q15, disk=31.2e9, download=31.2e9, transient=0)
        r = launch.judge(big, prof(), models, disk)
        assert not r["ok"] and r["short"], r
        assert r["why"] == "needs 41 GB free on disk, 12.6 GB free", r["why"]
        assert launch.status(r, models, disk) == r["why"]
        row = launch.variant_json({**r, "key": "published", "label": "x", "note": ""}, models, disk)
        assert row["disk_short"] is True and row["ok"] is False and "can't run" not in row["status"]
        # the same model on a computer that lacks the memory is a hardware limit
        hw = launch.judge(big, prof(ram=4, avail=2.0, ws=2.5), models, disk)
        assert not hw["ok"] and not hw["short"] and launch.status(hw, models, disk).startswith("can't run: ")
        # a download that fits is neither
        fine = launch.judge(entry(Q15, disk=1e9, download=1e9, transient=0), prof(), models, disk)
        assert fine["ok"] and not fine["short"] and launch.status(fine, models, disk) == "download 1.0 GB"
    finally:
        shutil.disk_usage = real_du
    # a listing from a launcher that predates the state is still read as a disk problem
    old = {"models": [{"id": "m", "name": "M", "variants": [
        {"key": "published", "label": "x", "size": 1e9, "ok": False,
         "why": "needs 41.2 GB free on this disk, 12.6 GB free",
         "status": "can't run: needs 41.2 GB free on this disk, 12.6 GB free"},
        {"key": "q3", "label": "y", "size": 1e9, "ok": False, "why": "3.2 GB usable now",
         "status": "can't run: 3.2 GB usable now"}]}]}
    a, b = remote.models_from_listing(old)[0].variants
    assert a.disk_short and a.status == "needs 41 GB free on disk, 12.6 GB free" and a.why == a.status
    assert not b.disk_short and b.status.startswith("can't run")
    print("18. disk too full is its own state (needs 41 GB free on disk, 12.6 GB free); memory and hardware "
          "limits stay can't run; an older launcher's listing is read the same way")


def check_estimate_learns(tmp):
    """Estimates learn: a start stores the estimate it was made on, the run's reply stores the speed it
    measured beside it, and the next estimate is corrected by this machine's median over 2+ runs of the
    same recipe, bounded to 0.3-1.5x."""
    import library
    real = machine.CACHE, list(library._FP)
    machine.CACHE = tmp / "ecache"
    library._FP[:] = ["fp-est"]
    try:
        m4 = {"backend": "metal", "gpu_bw": 94.9}
        s = {"store_gb": 4.045, "args": ["--cache-bias", "0.5"]}
        disk = {"gbps": 6.1, "lat_ms": 0.18}
        raw = machine.estimate(m4, FN, s, disk, corr={})["tps"]
        assert library.correction("store") == {}                       # no runs yet
        # a launcher start notes its estimate; the first reply of 64+ tokens pairs the two
        folder = fake_folder(tmp / "estmdl" / "q35", {
            "model_type": "fake_moe", "num_hidden_layers": 24, "hidden_size": 64, "num_experts": 60,
            "quantization": {"group_size": 64, "bits": 4}})
        note = machine.estimate(m4, FN, s, disk, corr={})
        library.note_estimate(folder, note, "store", {"power": "battery", "cpu_busy": 1.8})
        rec = library.record(folder, 101, 10.0, today="2026-10-27")
        run = rec["runs"][0]
        assert run["est"] == round(note["tps"], 2) and run["raw"] == round(raw, 2), run
        assert run["recipe"] == "store" and run["cond"]["power"] == "battery" and rec["est"] == run["est"]
        assert len(library.record(folder, 101, 10.0)["runs"]) == 1      # one run per start, not per reply
        assert library.correction("store") == {}                       # one run is not enough
        # a second start on the same machine and recipe
        time.sleep(0.01)                                               # a later start, a later note
        library.note_estimate(folder, note, "store", {"power": "ac"})
        rec = library.record(folder, 102, 20.0, today="2026-10-28")
        assert len(rec["runs"]) == 2 and rec["runs"][1]["tps"] == 5.05
        c = library.correction("store")
        assert c["runs"] == 2 and abs(c["factor"] - (10.0 / round(raw, 2) + 5.05 / round(raw, 2)) / 2) < 0.002, c
        e = machine.estimate(m4, FN, s, disk, corr=c)
        assert abs(e["tps"] - round(raw, 2) * c["factor"]) < 0.01 and e["raw_tps"] == raw and e["runs"] == 2
        # bounded both ways, and another recipe does not count
        lo = {"fp-est": {"a": {"runs": [{"recipe": "store", "raw": 10, "tps": 1}] * 2}},
              "fp-b": {"a": {"runs": [{"recipe": "store", "raw": 10, "tps": 90}] * 2}}}
        assert library.correction("store", fp="fp-est", data=lo)["factor"] == 0.3
        assert library.correction("store", fp="fp-b", data=lo)["factor"] == 1.5
        assert library.correction("arena", fp="fp-est", data=lo) == {}
        # the launcher's own start path notes the estimate it was planned on
        e35 = entry(Q35, id="q35", name="35B", recipe="arena")
        launch.note_estimate(e35, folder, machine.estimate(prof(), Q35, {"store_gb": 1.0}, disk, corr={}), {})
        assert library.note_for(library.folder_key(folder, library.signature(folder)))["model"] == "q35"
    finally:
        machine.CACHE = real[0]
        library._FP[:] = real[1]
    print("18. estimates learn: the estimate stored at start, the measured speed beside it, a correction "
          "from 2+ runs of the same recipe (median, 0.3-1.5x), the listing reads the corrected estimate")


def check_mtp_head(tmp):
    """The launcher adds the model's own MTP head when a 4-bit pack on this computer fits its config,
    and says what a head would add and how to make one when there is none, downloading nothing."""
    import types
    folder = fake_folder(tmp / "mtp_models" / "q35",
                         {"model_type": "fake_qwen", "text_config": {"hidden_size": 2048}})
    heads, other = tmp / "mtp_heads", tmp / "mtp_other"
    heads.mkdir()
    other.mkdir()

    def pack(at, hidden):
        hdr = json.dumps({"mtp.fc.weight": {"dtype": "BF16", "shape": [hidden, 2 * hidden],
                                           "data_offsets": [0, 0]}}).encode()
        p = at / launch.MTP_PACK
        p.write_bytes(len(hdr).to_bytes(8, "little") + hdr)
        (at / (p.stem + "-quant.json")).write_text("{}")
        return p

    def fake_module(at):
        return types.SimpleNamespace(
            head_args=lambda cfg: types.SimpleNamespace(hidden_size=cfg["text_config"]["hidden_size"]),
            DEFAULT_MTP=str(at))

    good = pack(heads, 2048)
    assert launch.mtp_for(folder, {}, mod=fake_module(heads))["path"] == str(good)
    assert "about 25% more" in launch.mtp_for(folder, {}, mod=fake_module(heads))["note"]
    # a head for another model, or a model with no recipe, is not used and says why
    pack(other, 1024)
    miss = launch.mtp_for(folder, {}, mod=fake_module(other))
    assert miss["path"] == "" and "tools/mtp_pack.py" in miss["note"], miss
    no_recipe = types.SimpleNamespace(head_args=lambda cfg: (_ for _ in ()).throw(ValueError("no recipe")),
                                      DEFAULT_MTP=str(heads))
    assert launch.mtp_for(folder, {}, mod=no_recipe) == {"path": "", "note": ""}
    # the plan carries --mtp right beside the model, and the estimate is unchanged by it
    real = launch.mtp_for
    launch.mtp_for = lambda d, e, mod=None: {"path": str(good), "note": "its MTP head is on this computer "
                                                                       "and is used: about 25% more"}
    try:
        e = entry(Q35, id="q35", name="35B", recipe="arena")
        s, est = launch.engine_plan(e, prof(), {"gbps": 6.1, "lat_ms": 0.18}, d=folder)
        assert s["args"][-2:] == ["--mtp", str(good)], s["args"]
        assert any("MTP head" in w for w in s["why"]) and est["tps"] > 0
        assert s["mtp"]["path"] == str(good)
    finally:
        launch.mtp_for = real
    print("19. the model's own MTP head on disk is added to the engine command (--mtp), matched to the "
          "model's config by the pack's shapes; without one the plan says what it would add and how to make it")


Q36 = {"kind": "moe", "model_type": "qwen3_5_moe", "layers": 40, "hidden": 2048, "d_ff": 512, "moe_layers": 40,
       "experts": 256, "top_k": 8, "expert_bytes": 1769472, "bits": 4, "group": 64, "dense_bytes": 1.1033e9,
       "read_bytes": 1.1033e9, "total_bytes": 20e9, "embed_bytes": 0}


def check_cuda_q40(tmp):
    """On CUDA the curated 35B installs as the fast recipe (the base's trunk, a pinned Q4_0 GGUF's experts, the
    model's own MTP head) with the 4-bit pack kept as a variant; on Metal the entry is the one it was."""
    import types
    import huggingface_hub
    import library
    cur = launch.CURATED[0]
    assert launch.for_backend(cur, "metal") is cur and "gguf" not in launch.for_backend(cur, "metal")
    c = launch.for_backend(cur, "cuda")
    assert c["id"] == "qwen3.6-35b-q40" and c["repo"] == cur["repo"] and c["gated"] and c["native"]
    assert c["gguf"]["file"] == "Qwen_Qwen3.6-35B-A3B-Q4_0.gguf" and len(c["gguf"]["rev"]) == 40

    # sizes from the API's file list alone: four shards of the 4-bit repo, the GGUF and the head from the entry
    shards = [5.3e9, 5.3e9, 5.3e9, 4.5e9]
    info = types.SimpleNamespace(sha="r", siblings=[types.SimpleNamespace(rfilename=f"model-{i}.safetensors", size=int(s))
                                                    for i, s in enumerate(shards)])
    saved = huggingface_hub.HfApi, launch._hf_json, launch.inspect_model
    huggingface_hub.HfApi = lambda: types.SimpleNamespace(model_info=lambda *a, **k: info)
    launch._hf_json, launch.inspect_model = lambda *a: {}, lambda cfg, names=None: dict(Q36)
    try:
        e, m = launch.hf_entry(c), launch.hf_entry(launch.for_backend(cur, "metal"))
    finally:
        huggingface_hub.HfApi, launch._hf_json, launch.inspect_model = saved
    size = sum(int(s) for s in shards)
    assert m["download"] == m["disk"] == size and "format" not in m["facts"]          # Metal: as published
    base = size - 1769472 * 256 * 40
    f = e["facts"]
    assert f["format"] == "gguf" and f["bits"] == 4.52 and f["expert_bytes"] == 18203279360 // (256 * 40), f
    assert e["download"] == base + 20836243072 + 1689281536, e["download"]
    assert e["disk"] == base + 18203279360 + int(1689281536 * 0.5625 / 2) and e["transient"] == 20836243072

    # variants: the recipe first, the 4-bit pack beside it, nothing rounded from GGUF records; an installed
    # 4-bit folder is the 4-bit variant's and never the recipe's
    models = tmp / "q40_models"
    old = fake_folder(models / "qwen3.6-35b", {"model_type": "qwen3_5_moe", "num_hidden_layers": 40, "hidden_size": 2048,
                                                "num_experts": 256, "quantization": {"group_size": 64, "bits": 4}},
                      packed=True, extra=["launcher.json"])
    sib = {**{k: c[k] for k in ("repo", "rev", "recipe", "gated", "native")}, "id": "mlx-community--Qwen3.6-35B-A3B-4bit",
           "name": "Qwen3.6-35B-A3B-4bit", "facts": dict(Q36), "download": size, "disk": size, "transient": 0}
    library.scan(models, refresh=True)
    vs = launch.variants(e, models, [sib])
    assert [v["key"] for v in vs] == ["published", "mlx-community/Qwen3.6-35B-A3B-4bit"], vs
    assert vs[0]["label"] == "as published, GGUF Q4_0, Q4_1 experts" and "166 tok/s" in vs[0]["note"]
    assert "measured against" in vs[1]["note"] and "path" not in vs[0]["entry"]
    assert vs[1]["entry"]["path"] == str(old), vs[1]["entry"].get("path")
    assert launch.model_dir(vs[0]["entry"], models) == models / "qwen3.6-35b-q40"

    # folder identity by the real signature: a GGUF pack (by its config, or by its index alone) is never the
    # 4-bit pack, a 4-bit pack never the recipe, and the recipe's own folder is the recipe's
    m2 = tmp / "q40_ident"
    m2.mkdir()
    cfg4 = json.loads((old / "config.json").read_text())
    fast = fake_folder(m2 / "fast", dict(cfg4, expert_quant={"format": "gguf", "bits": 4.52, "source": "s"}), packed=True)
    hand = fake_folder(m2 / "hand", cfg4, packed=True)
    (hand / "experts_index.json").write_text(json.dumps({"model.layers.0.mlp.switch_mlp.gate_proj":
                                                         {"parts": [{"name": "gguf", "ggml_type": "Q4_0"}]}}))
    linked = fake_folder(m2 / "linked", dict(cfg4, expert_quant={"format": "gguf", "source": "s", "types": ["Q4_0"]}),
                         packed=True)                       # a hand-made GGUF pack's marker, with no bits
    sf, sh, so, sl = (library.signature(p) for p in (fast, hand, old, linked))
    assert (sf["format"], sf["bits"], sh["format"], so["format"]) == ("gguf", 4.52, "gguf", None), (sf, sh, so)
    assert library.matches(sf, e["facts"]) and not library.matches(sf, Q36)
    assert not library.matches(sh, Q36) and not library.matches(so, e["facts"])
    assert sl["format"] == "gguf" and not library.matches(sl, Q36) and not library.matches(sl, e["facts"])
    assert library.folder_key(fast, sf).endswith("/4.52g") and library.folder_key(old, so).endswith("/4")
    library.scan(m2, refresh=True)
    assert library.adopt(e, m2)["path"] == str(fast) and "path" not in library.adopt(sib, m2)
    shutil.rmtree(fast)
    library.scan(m2, refresh=True)
    assert "path" not in library.adopt(e, m2)

    # --model takes the listed id and the CUDA one on CUDA, the listed one only on Metal
    real_hf = launch.hf_entry
    launch.hf_entry = lambda x: {**x, "facts": dict(Q36, format="gguf") if x.get("gguf") else dict(Q36),
                                 "download": 1, "disk": 1, "transient": 0}
    try:
        for name in ("qwen3.6-35b", "qwen3.6-35b-q40"):
            args = types.SimpleNamespace(model=name, variant="published")
            assert launch.pick(args, prof("cuda", gpu_mem=17.1), models, None)["id"] == "qwen3.6-35b-q40"
        assert launch.pick(types.SimpleNamespace(model="qwen3.6-35b", variant="published"), prof(), models,
                           None)["id"] == "qwen3.6-35b"
        try:
            launch.pick(types.SimpleNamespace(model="qwen3.6-35b-q40", variant="published"), prof(), models, None)
            raise AssertionError("the CUDA id taken on Metal")
        except SystemExit as err:
            assert "not a listed id (qwen3.6-35b, flash-next-exl3" in str(err), err
    finally:
        launch.hf_entry = real_hf

    # the install: trunk aside by stream_pack, the GGUF into the folder, repacked and deleted, the head packed
    calls, fail = [], {"repack": True}
    big = {"repo": e["repo"], "file": c["gguf"]["file"]}

    def fake_run(cmd):
        tool = Path(cmd[1]).name
        calls.append(tool)
        out = Path(cmd[cmd.index("--out") + 1])
        if tool == "stream_pack.py":
            assert cmd[-1] == "--skip-experts" and out.name == "_trunk"
            out.mkdir(parents=True)
            (out / "config.json").write_text(json.dumps({"model_type": "qwen3_5_moe"}))
            (out / "model-00001-of-00001.safetensors").write_bytes(b"t")
            (out / "experts.bin").write_bytes(b"")
            (out / "experts_index.json").write_text("{}")
        elif tool == "gguf_repack.py":
            assert Path(cmd[2]).name == big["file"] and cmd[-2:] == ["--layers", "40"] and out.name == "experts.bin"
            if fail["repack"]:
                fail["repack"] = False
                raise SystemExit("stopped")
            out.write_bytes(b"g" * 64)
            out.with_name("experts_index.json").write_text(json.dumps({"k": {"parts": [{"name": "gguf"}]}}))
        elif tool == "mtp_pack.py":
            assert (Path(cmd[cmd.index("--src") + 1]) / "mtp.safetensors").read_bytes() == b"head"
            out.write_bytes(b"h")
            out.with_name(out.stem + "-quant.json").write_text("{}")

    def fake_download(repo, name, revision=None, local_dir=None):
        assert (repo, name, revision) == (c["gguf"]["repo"], big["file"], c["gguf"]["rev"])
        calls.append("download")
        p = Path(local_dir) / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"GGUF")
        return str(p)

    def fake_fetch(repo, rev, dst):
        assert (repo, rev) == (c["mtp"]["repo"], c["mtp"]["rev"])
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"head")

    saved = launch._run, huggingface_hub.hf_hub_download, launch._fetch_mtp, launch.tuner.record_baseline
    launch._run, huggingface_hub.hf_hub_download, launch._fetch_mtp = fake_run, fake_download, fake_fetch
    launch.tuner.record_baseline = lambda *a: None
    try:
        e = vs[0]["entry"]
        try:
            launch.prepare(e, models)
            raise AssertionError("the stopped repack went on")
        except SystemExit:
            pass
        d = models / "qwen3.6-35b-q40"
        assert calls == ["stream_pack.py", "download", "gguf_repack.py"] and not launch.is_ready(d)
        assert launch.prepare(e, models) == d                 # resumes: no second trunk
        assert calls[3:] == ["download", "gguf_repack.py", "mtp_pack.py"], calls
        assert sorted(p.name for p in d.iterdir()) == ["config.json", "experts.bin", "experts_index.json",
                                                       "launcher.json", "model-00001-of-00001.safetensors",
                                                       "mtp-4bit-quant.json", "mtp-4bit.safetensors"]
        assert json.loads((d / "config.json").read_text())["expert_quant"]["format"] == "gguf"
        assert (old / "launcher.json").exists() and (old / "experts.bin").stat().st_size == 5000   # the 4-bit pack kept
        # finished experts already in an empty folder (another pack's, linked in) are kept: trunk and head only
        d2 = models / "linked"
        d2.mkdir()
        os.link(d / "experts.bin", d2 / "experts.bin")
        shutil.copy(d / "experts_index.json", d2)
        del calls[:]
        launch._pack_gguf(e, d2)
        assert calls == ["stream_pack.py", "mtp_pack.py"] and (d / "experts.bin").read_bytes() == b"g" * 64
    finally:
        launch._run, huggingface_hub.hf_hub_download, launch._fetch_mtp, launch.tuner.record_baseline = saved

    # the recipe's command: the CUDA store flags, the gated cache bias and the head beside the model
    real_mtp = launch.mtp_for
    hidden = lambda cfg: types.SimpleNamespace(hidden_size=2048)
    launch.mtp_for = lambda d, e, mod=None: real_mtp(d, e, mod=types.SimpleNamespace(head_args=hidden, DEFAULT_MTP=None))
    hdr = json.dumps({"mtp.fc.weight": {"dtype": "U8", "shape": [2048, 4096], "data_offsets": [0, 0]}}).encode()
    (d / launch.MTP_PACK).write_bytes(len(hdr).to_bytes(8, "little") + hdr)
    try:
        s, est = launch.engine_plan(e, prof("cuda", ram=15, gpu_mem=17.1, avail=12.0), {"gbps": 6.0, "lat_ms": 0.1}, d=d)
        (d / "mtp-4bit-quant.json").unlink()
        bare, _ = launch.engine_plan(e, prof("cuda", ram=15, gpu_mem=17.1, avail=12.0), {"gbps": 6.0, "lat_ms": 0.1}, d=d)
    finally:
        launch.mtp_for = real_mtp
    assert s["args"][-6:] == ["--cache-bias", "0.5", "--bias-budget", "0.2", "--mtp", str(d / launch.MTP_PACK)], s["args"]
    assert s["env"]["ENGINE_BIAS_GATED"] == "1" and s["env"]["MOE_RESIDENT"] == "1" and est["tps"] > 0, s["env"]
    assert "--mtp" not in bare["args"]

    # --delete on CUDA: either id names the recipe's folder; the 4-bit key the 4-bit pack where it was found;
    # the question always names the folder that goes
    import argparse
    asked, answer = [], {"a": "no"}
    saved = (machine.facts, launch.ask, launch.running_models, launch.hf_entry, launch.siblings)
    machine.facts = lambda: {"backend": "cuda"}
    launch.ask = lambda prompt, default="": asked.append(prompt) or answer["a"]
    launch.running_models, launch.siblings = lambda: set(), lambda x, limit=4: [sib]
    launch.hf_entry = lambda x: {**x, "facts": e["facts"], "download": 1, "disk": 1, "transient": 0}
    ns = lambda i, v: argparse.Namespace(delete=i, variant=v, yes=False, models_dir=str(models))
    try:
        for i in ("qwen3.6-35b", "qwen3.6-35b-q40"):
            assert launch.delete_cli(ns(i, "mlx-community/Qwen3.6-35B-A3B-4bit")) == 1
            assert asked.pop().startswith(f"Delete {i} mlx-community/Qwen3.6-35B-A3B-4bit at {old} (")
            assert launch.delete_cli(ns(i, "published")) == 1 and asked.pop().startswith(f"Delete {i} published at {d} (")
        assert old.exists() and d.exists()
        answer["a"] = "yes"
        assert launch.delete_cli(ns("qwen3.6-35b-q40", "published")) == 0 and not d.exists() and old.exists()
        machine.facts = lambda: {"backend": "metal"}
        assert launch.delete_cli(ns("qwen3.6-35b", "published")) == 0 and not old.exists()   # Metal: the listed folder
    finally:
        machine.facts, launch.ask, launch.running_models, launch.hf_entry, launch.siblings = saved

    # the head's tensors range-read from the checkpoint's shards into one file, bit for bit
    w1 = {"mtp.fc.weight": mx.arange(12, dtype=mx.bfloat16).reshape(3, 4), "model.x": mx.ones((5,))}
    w2 = {"mtp.norm.weight": mx.full((7,), 2.5), "lm_head.weight": mx.zeros((2, 2))}
    rd = tmp / "fake_repo"
    rd.mkdir()
    for n, w in (("a.safetensors", w1), ("b.safetensors", w2)):
        mx.save_safetensors(str(rd / n), w)
    (rd / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {**{k: "a.safetensors" for k in w1}, **{k: "b.safetensors" for k in w2}}}))
    fs = types.SimpleNamespace(read_text=lambda p, revision=None: (rd / Path(p).name).read_text(),
                               open=lambda p, mode="rb", revision=None: open(rd / Path(p).name, mode))
    launch._fetch_mtp("o/r", "x", tmp / "head" / "mtp.safetensors", fs=fs)
    got = mx.load(str(tmp / "head" / "mtp.safetensors"))
    assert sorted(got) == ["mtp.fc.weight", "mtp.norm.weight"]
    assert mx.array_equal(got["mtp.fc.weight"], w1["mtp.fc.weight"]) and mx.array_equal(got["mtp.norm.weight"], w2["mtp.norm.weight"])
    print("23. CUDA 35B: the Q4_0 GGUF recipe (pinned GGUF, sizes, install steps that resume, head packed, --mtp), "
          "the 4-bit pack kept as a variant; Metal unchanged")


def check_cuda_serve_exec(tmp):
    """On CUDA a start that imported MLX (128 MiB of VRAM in a CUDA context) serves from a fresh process: the plan
    goes to a file and the launcher execs itself with --serve; on Metal it serves in place, as before."""
    import types
    models = tmp / "exec_models"
    d = fake_folder(models / "m", {"model_type": "x"}, extra=["launcher.json"])
    e = entry(Q15, id="m", name="M")
    s, est = {"args": ["--cache-bias", "0.5"], "env": {"MOE_NATIVE": "1"}, "why": [], "mtp": {"path": ""}}, {"tps": 9.0}
    calls = {}

    class Exec(Exception):
        pass

    def fake_exec(py, argv):
        calls["argv"] = argv
        raise Exec()
    saved = (machine.profile, machine.ssd, launch.pick, launch.judge, launch.engine_plan, launch.serve, os.execv, sys.argv)
    machine.ssd = lambda *a, **k: {"gbps": 5.0, "lat_ms": 0.1}
    launch.pick, launch.judge = lambda *a: e, lambda *a: {"ok": True}
    launch.engine_plan = lambda *a, **k: (s, est)
    launch.serve = lambda *a: calls.setdefault("serve", a)
    os.execv = fake_exec
    try:
        for backend in ("metal", "cuda"):
            calls.clear()
            machine.profile = lambda **k: {**prof(backend), "chip": "c", "arch": "a", "os": "o", "gpu": "g", "now": {"avail_gb": 14.0, "disks": []}}
            sys.argv = ["launch.py", "--model", "m", "--models-dir", str(models), "--no-browser", "--no-chat"]
            try:
                launch.main()
            except Exec:
                pass
            if backend == "metal":
                assert "argv" not in calls and calls["serve"][1:] == (e, d, s, est, "published", False, est), calls
        argv = calls["argv"]
        assert "serve" not in calls and argv[1].endswith("launch.py") and argv[2:-2] == sys.argv[1:] and argv[-2] == "--serve"
        plan = json.loads(Path(argv[-1]).read_text())
        assert (plan["e"]["id"], plan["d"], plan["s"], plan["est"]) == ("m", str(d), s, est)
        assert plan["fp"] == machine.fingerprint() and plan["shown"] == {"tps": None, "label": "not measured yet"}
        calls.clear()
        sys.argv = ["launch.py", "--serve", argv[-1]]
        launch.main()                              # the fresh process: straight to serving, the plan file read once
        assert calls["serve"][1:] == (plan["e"], d, s, est, "published", False, plan["shown"])
        assert not Path(argv[-1]).exists()
        real_facts = machine.facts                 # its key comes from the plan: facts (and MLX) are never read
        machine.facts = lambda: (_ for _ in ()).throw(AssertionError("facts read in the served process"))
        assert machine.KEY == [plan["fp"]] and machine.fingerprint() == plan["fp"]
        machine.facts = real_facts
    finally:
        machine.KEY[:] = []
        machine.profile, machine.ssd, launch.pick, launch.judge, launch.engine_plan, launch.serve, os.execv, sys.argv = saved
    print("25. CUDA start serves from a fresh process (exec with --serve and the plan), so the launcher holds no "
          "CUDA context beside its server; Metal serves in place")


def check_first_reply(tmp):
    """The first start's first-use work goes to a warm-up reply before ready, which the server's counters and
    measured speeds leave out; the Speed line's reference on CUDA is a measured speed or none; our own server is
    not "other apps" on the GPU; a served CUDA process never tunes (tuning imports MLX), the launcher tunes first."""
    import contextlib
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import library
    import server
    import tui
    # 26. the warm-up: one 8-token reply marked X-Warmup, said and timed; a failed one never stops the start
    got = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            got.append((self.headers.get("X-Warmup"), json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
            body = json.dumps({"choices": [{"message": {"content": ""}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            assert launch.warm_up(srv.server_address[1])                # an empty answer (all thinking) is fine
            assert not launch.warm_up(srv.server_address[1] + 1)        # nothing there: said, not raised
    finally:
        srv.shutdown()
        srv.server_close()
    assert got[0][0] == "1" and got[0][1]["max_tokens"] == 8, got
    assert "warming up: one short reply..." in out.getvalue() and "failed (no reply" in out.getvalue(), out.getvalue()
    # the server leaves it out of /status and of the measured speeds; decode speed is tokens after the first over
    # the time after the first token, so a slow first token does not lower it
    c = server.Counters()
    c.add({"warmup": True, "completion_tokens": 8, "ttft_s": 100.0, "decode_s": 10.0})
    assert c.report()["requests"] == 0 and c.report()["decode_tok_s"] is None and not c.history()
    c.add({"completion_tokens": 201, "ttft_s": 30.0, "decode_s": 2.0})
    r = c.report()
    assert (r["requests"], r["decode_tok_s"], r["decode_replies"], r["first_token_last_s"]) == (1, 100.0, 1, 30.0), r
    recorded, real_record = [], server.library.record
    server.library.record = lambda *a: recorded.append(a)
    try:
        host = server.ModelHost.__new__(server.ModelHost)
        host.model_name = "m"
        host._measure({"warmup": True, "completion_tokens": 100, "decode_s": 1.0})
        host._measure({"completion_tokens": 100, "decode_s": 1.0})
    finally:
        server.library.record = real_record
    assert recorded == [("m", 100, 1.0)], recorded
    # what a start says while it loads: the last start's time here, or that this is the first
    d = tmp / "starts_model"
    assert launch.start_seconds(d) is None and "first start of this model on this computer" in launch.start_note(None)
    assert launch.start_seconds(d, 41.26) == 41.3 and launch.start_seconds(d) == 41.3
    assert launch.start_note(41.3) == "  about 41 s to ready, as the last start here took"
    print("26. a start warms up with one 8-token reply (X-Warmup) before ready, which /status and the measured "
          "speeds leave out; decode speed excludes the first token's wait; the wait says the last start's time")

    # 27. the Speed line's reference: the estimate on Metal; on CUDA a speed measured here before, else no number
    e, est = {"facts": {"kind": "moe"}}, {"tps": 57.0, "lo_tps": 50.0, "hi_tps": 57.0, "runs": 0}
    real_lookup = library.lookup
    try:
        library.lookup = lambda f: {}
        assert launch.speed_ref(e, est, "metal") is est
        ref = launch.speed_ref(e, est, "cuda")
        assert ref == {"tps": None, "label": "not measured yet"}
        assert launch.speed_text({}, ref["tps"], ref["label"]) == "no reply timed yet; speed not measured yet on this computer"
        assert launch.speed_text({"decode_tok_s": 147.2, "decode_replies": 2}, ref["tps"], ref["label"]) == \
            "147.2 tok/s measured over 2 replies"
        library.lookup = lambda f: {"tps": 146.7, "n": 5, "date": "2026-11-11", "name": "m"}
        ref = launch.speed_ref(e, est, "cuda")
        assert launch.speed_text({}, ref["tps"], ref["label"]) == "no reply timed yet; ~147 tok/s measured here before, 5 replies"
        assert launch.speed_text({"decode_tok_s": 140.0, "decode_replies": 1}, ref["tps"], ref["label"]) == \
            "140.0 tok/s measured over 1 reply (the ~147 tok/s measured here before, 5 replies)"
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            launch.short_banner("M", 8800, ref)
            launch.short_banner("M", 8800, {"tps": None, "label": "not measured yet"})
            launch.short_banner("M", 8800, est)
        lines = [ln for ln in out.getvalue().splitlines() if "Ready:" in ln]
        assert lines == ["  Ready: M   (~147 tok/s measured here before, 5 replies)", "  Ready: M   (not measured yet)",
                         f"  Ready: M   ({tui.estimate_text(57.0, 50.0, 57.0, 0)})"], lines
    finally:
        library.lookup = real_lookup
    print("27. Speed line before replies: the estimate on Metal; on CUDA this computer's measured replies of the "
          "model, else 'not measured yet' (no 2.6x-off bandwidth number)")

    # 28. "other apps use the GPU" counts other processes' SM share from nvidia-smi pmon, not our own server's
    head = "# gpu         pid   type     sm    mem    enc    dec    command\n# Idx           #    C/G      %      %      %      %    name\n"
    mine = f"    0  {os.getpid()}     C     40     12      -      -    python3\n"
    assert machine._nvidia_busy(pmon=head + mine) == 0.0
    assert machine._nvidia_busy(pmon=head + mine + "    0  1     G      7      1      -      -    Xorg\n") == 7.0
    assert machine._nvidia_busy(pmon=head + f"    0  {os.getpid()}     C      -      -      -      -    python3\n") == 0.0
    assert "GPU" not in machine.conditions_text({"gpu_busy": machine._nvidia_busy(pmon=head + mine)})
    print("28. the GPU note counts other processes only (nvidia-smi pmon per pid, this process group left out)")

    # 29. the first run on a machine: tuning (which reads the machine through MLX) happens before the exec, and a
    # served process with no tuning cached never tunes and never imports MLX
    models = tmp / "tune_models"
    d = fake_folder(models / "m", {"model_type": "x"}, extra=["launcher.json", "experts.bin"])
    e = entry(Q15, id="m", name="M")
    order = []

    class Exec(Exception):
        pass
    saved = (machine.profile, machine.ssd, launch.pick, launch.judge, launch.engine_plan, os.execv, sys.argv,
             launch.tuner.load, launch.tuner.tune)
    machine.ssd = lambda *a, **k: {"gbps": 5.0, "lat_ms": 0.1}
    launch.pick, launch.judge = lambda *a: e, lambda *a: {"ok": True}
    launch.engine_plan = lambda *a, **k: ({"args": [], "env": {}, "why": [], "mtp": {"path": ""}}, {"tps": 9.0})
    launch.tuner.load = lambda d: None
    launch.tuner.tune = lambda d, lp=None: order.append("tune") or {}
    os.execv = lambda py, argv: order.append("exec") or (_ for _ in ()).throw(Exec())
    machine.profile = lambda **k: {**prof("cuda"), "chip": "c", "arch": "a", "os": "o", "gpu": "g",
                                   "now": {"avail_gb": 14.0, "disks": []}}
    sys.argv = ["launch.py", "--model", "m", "--models-dir", str(models), "--no-browser", "--no-chat"]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                launch.main()
            except Exec:
                pass
    finally:
        (machine.profile, machine.ssd, launch.pick, launch.judge, launch.engine_plan, os.execv, sys.argv,
         launch.tuner.load, launch.tuner.tune) = saved
    assert order == ["tune", "exec"], order
    probe = ("import sys\n"
             "class Block:\n"
             "    def find_spec(self, name, path=None, target=None):\n"
             "        if name.split('.')[0] == 'mlx':\n"
             "            raise ImportError('mlx imported in the served process')\n"
             "sys.meta_path.insert(0, Block())\n"
             f"sys.path.insert(0, {str(ROOT)!r})\n"
             "import subprocess, launch, machine\n"
             "machine.KEY[:] = ['fp']\n"
             f"machine.CACHE = __import__('pathlib').Path({str(tmp / 'served_cache')!r})\n"
             "launch.tuner.tune = lambda *a, **k: sys.exit('tuned in the served process')\n"
             "subprocess.Popen = lambda *a, **k: 'proc'\n"
             f"assert launch.start_server({str(d)!r}, {{'args': [], 'env': {{}}}}, 8800, '/dev/null', quiet=True) == 'proc'\n"
             "assert not [m for m in sys.modules if m.split('.')[0] == 'mlx']\n")
    r = subprocess.run([sys.executable, "-c", probe], cwd=ROOT, capture_output=True, text=True, timeout=120,
                       stdin=subprocess.DEVNULL)
    assert r.returncode == 0, r.stdout + r.stderr
    print("29. first run on CUDA: the disk tuning runs before the exec, and the served process neither tunes nor "
          "imports MLX (no CUDA context beside the server)")

    # 30. MLX's CUDA kernel cache in the launcher's cache (MLX_PTX_CACHE_DIR), keyed, one key kept; Metal untouched
    cu = {**prof("cuda"), "mlx": "0.32.4", "gpu": "NVIDIA GeForce RTX 5060 Ti"}
    kd = machine.kernel_dir(cu)
    assert kd.parent == machine.CACHE / "mlx-kernels" and kd.name.startswith("0.32.4-"), kd
    assert machine.kernel_dir({**cu, "gpu": "other"}) != kd and machine.kernel_dir({**cu, "mlx": "0.33.0"}) != kd
    assert machine.kernel_dir(prof("metal")) is None
    e = entry(Q15, id="m", name="M")
    real_env = os.environ.pop("MLX_PTX_CACHE_DIR", None)
    try:
        assert launch.engine_plan(e, cu, None, native=False)[0]["env"]["MLX_PTX_CACHE_DIR"] == str(kd)
        assert "MLX_PTX_CACHE_DIR" not in launch.engine_plan(e, prof("metal"), None, native=False)[0]["env"]
        os.environ["MLX_PTX_CACHE_DIR"] = str(tmp / "users_own")        # the user's own setting wins
        assert "MLX_PTX_CACHE_DIR" not in launch.engine_plan(e, cu, None, native=False)[0]["env"]
    finally:
        os.environ.pop("MLX_PTX_CACHE_DIR", None)
        if real_env is not None:
            os.environ["MLX_PTX_CACHE_DIR"] = real_env
    old = machine.CACHE / "mlx-kernels" / "0.32.3-aaaaaaaaaaaa"
    old.mkdir(parents=True)
    (old / "k.ptx").write_text("x")
    s = {"env": {"MLX_PTX_CACHE_DIR": str(kd)}}
    assert launch.kernel_cache(s) and kd.is_dir() and not old.exists()      # empty: built this start; old key gone
    (kd / "k.ptx").write_text("x")
    assert not launch.kernel_cache(s) and (kd / "k.ptx").exists()           # kept: nothing to build
    own = tmp / "users_own"
    own.mkdir()
    assert launch.kernel_cache({"env": {"MLX_PTX_CACHE_DIR": str(own)}}) is None and launch.kernel_cache({"env": {}}) is None
    assert "builds its GPU kernels once" in launch.start_note(40.2, True)
    assert launch.start_note(40.2, False).startswith("  about 40 s")
    assert "builds its kernels once" in launch.start_note(None, True) and "builds" in launch.start_note(None, None)
    assert "kept from an earlier start" in launch.start_note(None, False)     # the start after a cold one
    print("30. MLX's CUDA kernels kept in mlx-kernels/<MLX version>-<key of GPU, driver, nvrtc> (MLX_PTX_CACHE_DIR), "
          "other keys removed, the user's own setting kept; a start with an empty folder says it builds them; "
          "Metal sets nothing")


def check_exl3_resume(tmp, src):
    """An EXL3 install stopped after its first experts shard resumes into the same experts.bin: the rerun
    neither makes the trunk again (stream_pack would truncate experts.bin) nor pads it with zeros."""
    import huggingface_hub
    import repack
    models = tmp / "exl3_resume"
    e = {"id": "x3", "name": "X3", "repo": "o/X-4bit", "rev": "r", "facts": {"kind": "moe"}, "experts_repo": "o/X-exl3",
         "experts_rev": "x", "exl3_files": ["L0.safetensors", "L1.safetensors"], "expert_quant": {"format": "exl3", "k": 3},
         "disk": 1, "download": 1, "transient": 0}
    calls, stop = [], {"L1.safetensors": True}

    def fake_run(cmd):
        calls.append(Path(cmd[1]).name)
        i = cmd.index("--repo")
        r = subprocess.run([*cmd[:i], "--from-dir", str(src), *cmd[i + 4:]], cwd=ROOT, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr

    def fake_download(repo, name, revision=None, local_dir=None):
        if stop.pop(name, False):
            raise OSError("the network went away")
        p = Path(local_dir) / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(name[:2].encode() * 50)
        return str(p)

    def fake_pack(p, dst, written, index):
        dst.write(Path(p).read_bytes())
        index[Path(p).name] = written
        return written + Path(p).stat().st_size

    saved = launch._run, huggingface_hub.hf_hub_download, repack.pack_shard, launch.tuner.record_baseline
    launch._run, huggingface_hub.hf_hub_download, repack.pack_shard = fake_run, fake_download, fake_pack
    launch.tuner.record_baseline = lambda *a: None
    try:
        try:
            launch.prepare(e, models)
            raise AssertionError("the stopped download went on")
        except OSError:
            pass
        d = models / "x3"
        assert (d / "experts.bin").read_bytes() == b"L0" * 50
        launch.prepare(e, models)
    finally:
        launch._run, huggingface_hub.hf_hub_download, repack.pack_shard, launch.tuner.record_baseline = saved
    assert (d / "experts.bin").read_bytes() == b"L0" * 50 + b"L1" * 50, (d / "experts.bin").read_bytes()[:8]
    assert calls == ["stream_pack.py"] and json.loads((d / "experts_index.json").read_text()) == \
        {"L0.safetensors": 0, "L1.safetensors": 100}
    print("24. EXL3 install resumes after a stopped experts shard: the trunk not made again, no zeros in experts.bin")


def check_live_speed():
    """The launcher reads the running server's /status: the measured speed over its replies beside the
    estimate it made, and the conditions under it; the same number the web header and the monitor show."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    status = {"decode_tok_s": 12.8, "decode_replies": 3, "conditions_text": "on battery: expect slower",
              "arena": {"slots": 21, "gb": 1.49}, "store": {"mb": None, "slots": None},
              "experts": {"rows": 396, "rows_per_token": 132.0}}

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps(status if self.path == "/status" else {}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        port = srv.server_address[1]
        st = launch.live_status(port)
        assert st == status, st
        assert launch.speed_text(st, 28) == ("12.8 tok/s measured over 3 replies (the ~28 tok/s estimate) "
                                             "· on battery: expect slower")
        assert launch.speed_text({"decode_replies": 0}, 28) == \
            "no reply timed yet; the ~28 tok/s estimate for this computer"
        assert launch.speed_text({}, None) == "no reply timed yet"
        assert launch.live_status(port + 1, timeout=0.2) is None
        # the banner's watched line prints the measured speed once replies are timed
        out = io.StringIO()
        watch = launch.SpeedWatch(port, 28, every=0.05)
        real_out = sys.stdout
        sys.stdout = out
        try:
            watch.start()
            time.sleep(1.6)
            watch.stop()
        finally:
            sys.stdout = real_out
        assert any("12.8 tok/s measured over 3 replies" in ln for ln in out.getvalue().splitlines()), out.getvalue()
        # how an estimate reads: a range before this computer's runs, one adjusted number after
        import tui
        assert tui.estimate_text(28, 13, 29) == "~13-29 tok/s (estimate for this machine)"
        assert tui.estimate_text(13, 13, 29, 3) == "~13 tok/s (adjusted to this computer's measured runs)"
        assert tui.estimate_text(28) == "~28 tok/s (estimate for this machine)"
    finally:
        srv.shutdown()
        srv.server_close()
    print("20. live speed: /status read by the launcher, '12.8 tok/s measured over 3 replies (the ~28 tok/s "
          "estimate)' beside the conditions, the banner line, a range before measured runs")


def fake_folder(p, cfg, packed=False, extra=()):
    """A model folder with a config and a dummy weights file (nothing here is ever loaded)."""
    p.mkdir(parents=True)
    (p / "config.json").write_text(json.dumps(cfg))
    (p / "model-00001-of-00001.safetensors").write_bytes(b"x" * 1000)
    if packed:
        (p / "experts.bin").write_bytes(b"x" * 5000)
        (p / "experts_index.json").write_text("{}")
    for name in extra:
        (p / name).write_text("{}")
    return p


def check_found_copies(tmp):
    """Folders beside the launcher's own are matched to a catalog variant by their config (type, layers,
    hidden size, experts, bits, EXL3), so an EXL3 pack never reads as the 4-bit one; a match is on disk
    and starts without a download; the scan is cached and looks again on request."""
    import library
    home, models, vol = tmp / "fhome", tmp / "fmodels", tmp / "fvol"
    models.mkdir()
    hub = home / ".cache" / "huggingface" / "hub"
    base = {"model_type": "fake_moe", "num_hidden_layers": 24, "hidden_size": 64, "num_experts": 60,
            "quantization": {"group_size": 64, "bits": 4}}
    exl3 = dict(base, expert_quant={"format": "exl3", "k": 3})
    a = fake_folder(home / ".cache" / "flash-exl3", exl3, packed=True)
    b = fake_folder(home / ".cache" / "flash-4bit", base, packed=True)
    fake_folder(home / ".cache" / "stopped", base, packed=True, extra=[".pack_state.json"])
    half = fake_folder(home / ".cache" / "half", base)
    (half / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"k": "missing.safetensors"}}))
    fake_folder(home / ".cache" / "other-size", dict(base, hidden_size=128), packed=True)
    fake_folder(home / ".cache" / "huggingface" / "not-a-model", base)         # the cache's own folder is skipped
    raw = fake_folder(hub / "models--owner--Flash-4bit" / "snapshots" / "abc", base)
    dense_cfg = {"model_type": "fake_dense", "num_hidden_layers": 6, "hidden_size": 32,
                 "quantization": {"group_size": 64, "bits": 8}}
    far = fake_folder(vol / "models" / "dense8", dense_cfg)
    fake_folder(models / "by-hand", dense_cfg)
    saved = (library.HOME, library.volumes, machine.CACHE, os.environ.get("HF_HUB_CACHE"), shutil.disk_usage,
             library.VOLUMES)
    library.HOME, library.volumes, machine.CACHE = home, lambda m: [str(vol)], tmp / "fcache"
    os.environ["HF_HUB_CACHE"] = str(hub)
    shutil.disk_usage = lambda p: saved[4](p)._replace(free=int(12.6e9))
    try:
        sa, sb = library.signature(a), library.signature(b)
        assert (sa["bits"], sa["exl3"]) == (3, True) and (sb["bits"], sb["exl3"]) == (4, False), (sa, sb)
        library.VOLUMES = False                                     # the public default: other volumes stay unread
        assert "dense8" not in [Path(c["path"]).name for c in library.scan(models, refresh=True)]
        library.VOLUMES = True                                      # ./larder --volumes
        got = {Path(c["path"]).name: c for c in library.scan(models, refresh=True)}
        assert set(got) == {"flash-exl3", "flash-4bit", "other-size", "abc", "dense8", "by-hand"}, sorted(got)
        assert got["abc"]["repo"] == "owner/Flash-4bit" and got["abc"]["where"] == "hf" and not got["abc"]["packed"]
        assert got["dense8"]["where"] == "volume" and got["by-hand"]["where"] == "models" and got["flash-4bit"]["packed"]

        facts = dict(Q15, model_type="fake_moe", layers=24, hidden=64)
        pub = entry(dict(facts, bits=3, format="exl3"), id="flash", name="Flash", repo="owner/Flash-4bit",
                    disk=75e9, download=80e9)
        sib = entry(facts, id="owner--Flash-4bit", name="Flash-4bit", repo="owner/Flash-4bit", disk=100e9, download=110e9)
        disk = {"gbps": 3.42, "lat_ms": 0.227}
        vs = launch.variants(pub, models, [sib])
        assert [v["key"] for v in vs] == ["published", "owner/Flash-4bit"], vs
        # the EXL3 variant gets the EXL3 pack and the 4-bit one the 4-bit pack: never the other way round
        assert vs[0]["entry"]["path"] == str(a) and vs[1]["entry"]["path"] == str(b), [v["entry"].get("path") for v in vs]
        rows = [{**launch.judge(v["entry"], prof(), models, disk), **v} for v in vs]
        # 12.6 GB free is not "needs 85 GB": a folder already there is not downloaded
        assert all(r["ok"] and not r["short"] for r in rows)
        assert launch.status(rows[0], models, disk) == "on disk at ~/.cache/flash-exl3", launch.status(rows[0], models, disk)
        j = launch.variant_json(rows[1], models, disk)
        assert j["ready"] and j["path"] == str(b) and j["own"] is False and j["status"] == "on disk at ~/.cache/flash-4bit"
        assert launch.prepare(rows[1]["entry"], models) == b                       # starts as it is, downloads nothing
        # only a raw snapshot of the 4-bit repo left: it is packed in place (no download)
        shutil.rmtree(b)
        library.scan(models, refresh=True)
        raw_e = launch.variants(pub, models, [sib])[1]["entry"]
        assert raw_e["path"] == str(raw) and raw_e["needs_pack"] and raw_e["download"] == 0
        r = launch.judge(raw_e, prof(), models, disk)
        assert r["short"] and launch.status(r, models, disk) == "needs 111 GB free on disk, 12.6 GB free"
        shutil.disk_usage = lambda p: saved[4](p)._replace(free=1 << 40)       # the packed copy needs room
        r = launch.judge(raw_e, prof(), models, disk)
        assert launch.status(r, models, disk) == "on disk at ~/.cache/huggingface/hub/models--owner--Flash-4bit/" \
            "snapshots/abc; packs first, no download"
        assert not launch.variant_json({**r, "key": "k", "label": "l", "note": ""}, models, disk)["ready"]
        # a dense entry matches by its type, layers, hidden size and bits; another bit width does not
        dfacts = dict(Q4B, model_type="fake_dense", layers=6, hidden=32, bits=8, group=64)
        d8 = launch.variants(entry(dfacts, id="d8", name="D8"), models, [])[0]["entry"]
        assert d8["path"] == str(models / "by-hand") and d8["download"] == 0     # the nearer of two equal copies
        shutil.rmtree(models / "by-hand")
        library.scan(models, refresh=True)
        assert launch.variants(entry(dfacts, id="d8", name="D8"), models, [])[0]["entry"]["path"] == str(far)
        assert "path" not in launch.variants(entry(dict(dfacts, bits=4), id="d4", name="D4"), models, [])[0]["entry"]
        assert "path" not in launch.variants(entry(dict(dfacts, hidden=64), id="d5", name="D5"), models, [])[0]["entry"]
        # without a hidden size (an older cached entry) the weights' size has to agree
        nohid = {k: v for k, v in dfacts.items() if k != "hidden"}
        assert library.matches(library.signature(far), dict(nohid, total_bytes=1000), 1000)
        assert not library.matches(library.signature(far), dict(nohid, total_bytes=9000), 1000)
        # one already prepared in the launcher's folder keeps its place
        (models / "flash").mkdir()
        (models / "flash" / "launcher.json").write_text("{}")
        assert library.adopt(pub, models) is pub
        # a copy on disk beats a faster model that is not
        mine = launch.variants(entry(dfacts, id="d8b", name="D8b"), models, [])[0]["entry"]
        slow = {"ok": True, "est": {"tps": 5.0}}
        fast = {"ok": True, "est": {"tps": 50.0}}
        assert launch.recommend([(entry(Q15, id="z"), fast), (mine, slow)], models)[0] is mine

        # the scan is cached, forgets a folder that is gone, and looks again on request
        seen = []
        real_read = library.read_folder
        library.read_folder = lambda w, p: seen.append(p) or real_read(w, p)
        try:
            library.scan(models, refresh=True)
            n = len(seen)
            assert n >= 4
            library.scan(models)
            assert len(seen) == n                                       # from found.json
            shutil.rmtree(a)
            assert str(a) not in [c["path"] for c in library.scan(models)] and len(seen) == n
            library.scan(models, refresh=True)
            assert len(seen) > n
            library.volumes = lambda m: []                             # other roots: a new scan
            before = len(seen)
            assert "dense8" not in [Path(c["path"]).name for c in library.scan(models)] and len(seen) > before
            library.invalidate()
            assert not library.cache_path().exists()
        finally:
            library.read_folder = real_read
    finally:
        library.HOME, library.volumes, machine.CACHE, env, shutil.disk_usage, library.VOLUMES = saved
        os.environ.pop("HF_HUB_CACHE", None) if env is None else os.environ.__setitem__("HF_HUB_CACHE", env)
    print("19. found copies: an EXL3 pack and a 4-bit pack each match their own variant, on disk at <path> "
          "with no download, a raw snapshot packs in place, half-downloaded and unfinished folders are "
          "skipped, the scan is cached and looks again on refresh")


def check_base_model(tmp):
    """A folder with the curated entry's architecture is that model only when its README does not name
    another base model; no README is adopted as before; an old scan cache is read again."""
    import library
    cfg = {"model_type": "fake_moe", "num_hidden_layers": 24, "hidden_size": 64, "num_experts": 60,
           "quantization": {"group_size": 64, "bits": 4}}
    e = entry(dict(Q15, model_type="fake_moe", layers=24, hidden=64), id="q35", repo="owner/Q35-4bit",
              base="Qwen/Q35")

    def copy(name, readme=None):
        p = fake_folder(tmp / "bm" / name, cfg)
        if readme is not None:
            (p / "README.md").write_text(readme)
        return library.read_folder("cache", p)

    def card(*lines):
        return "---\nlicense: apache-2.0\n" + "\n".join(lines) + "\ntags:\n- mlx\n---\n# title\n"

    other = copy("other", card("base_model: InternScience/Sage-A1.5", "base_model_relation: quantized"))
    same = copy("same", card("base_model: Qwen/Q35"))
    bare = copy("bare")
    listed = copy("listed", card("base_model:", "- Some/Other", "- qwen/q35"))
    tuned = copy("tuned", card("base_model: Qwen/Q35", "base_model_relation: finetune"))
    assert other["base_model"] == ["internscience/sage-a1.5"] and listed["base_model"] == ["some/other", "qwen/q35"]
    assert tuned["relation"] == "finetune" and bare["base_model"] == [] and copy("broken", "---\nbase_model")["base_model"] == []
    assert library.find([other], e) is None                                    # (a) another model
    assert library.find([same], e) is same                                     # (b) its own base
    assert library.find([bare], e) is bare                                     # (c) no README
    assert library.find([listed], e) is listed                                 # (d) the list form
    assert library.find([tuned], e) is None                                    # (e) a fine-tune
    assert library.find([bare, same, other], e) is same                        # (f) evidence ranks first
    assert library.find([other], dict(e, base=None)) is other                  # no base: as before
    longer = copy("longer", card("base_model: Qwen/Q35-Instruct"))
    assert library.find([longer], e) is None                                   # a longer name is another model
    assert library.find([other], dict(e, repo="")) is None                     # no repo: the base alone decides
    assert library.find([same], dict(e, repo="")) is same
    real = library.HOME, machine.CACHE
    library.HOME, machine.CACHE = tmp / "bmhome", tmp / "bmcache"
    try:
        models = tmp / "bmmodels"
        models.mkdir()
        fake_folder(tmp / "bmhome" / ".cache" / "m", cfg)
        library.scan(models, refresh=True)
        old = json.loads(library.cache_path().read_text())
        old.pop("v")
        old["folders"] = [{k: v for k, v in c.items() if k not in ("base_model", "relation")} for c in old["folders"]]
        library.cache_path().write_text(json.dumps(old))
        assert "base_model" in library.scan(models)[0]                         # (g) an old cache is refreshed
    finally:
        library.HOME, machine.CACHE = real
    print("21. base model: another model on the same architecture is not adopted, no README still is, "
          "the match ranks first, an old scan cache is refreshed")


def check_measured(tmp):
    """Replies of 64 or more tokens add their decode speed to a record per machine and model folder (the
    median of the last 20, the count, the date); the record outlives the folder and is found again by
    what the folder was; listings carry it."""
    import library
    real = machine.CACHE, list(library._FP)
    machine.CACHE = tmp / "mcache"
    library._FP[:] = ["fp-here"]
    try:
        assert machine.fingerprint() == machine._key(machine.facts())          # the key the profile is cached under
        folder = fake_folder(tmp / "mdl" / "q35-edge", {
            "model_type": "fake_moe", "num_hidden_layers": 24, "hidden_size": 64, "num_experts": 60,
            "quantization": {"group_size": 64, "bits": 4}})
        facts = dict(Q15, model_type="fake_moe", layers=24, hidden=64)
        assert library.lookup(facts) == {}
        for tokens, secs in ((63, 5.0), (0, 5.0), (None, 5.0), (80, 0.0), (80, -1.0)):
            assert library.record(folder, tokens, secs, today="2026-10-27") is None     # too short or no time
        assert not library.measured_path().exists()
        rec = library.record(folder, 65, 4.0, today="2026-10-27")                       # 64 tokens in 4 s
        assert rec["last"] == [16.0] and rec["tps"] == 16.0 and rec["n"] == 1 and rec["name"] == "q35-edge"
        for i in range(1, 25):                                                        # 24 more: 20 are kept
            rec = library.record(folder, 101, 100 / (10.0 + i), today="2026-10-28")
        assert len(rec["last"]) == 20 and rec["n"] == 20 and rec["date"] == "2026-10-28"
        assert rec["last"] == [round(10.0 + i, 2) for i in range(5, 25)] and rec["tps"] == 24.5, rec
        data = json.loads(library.measured_path().read_text())
        assert list(data) == ["fp-here"] and list(data["fp-here"]) == ["q35-edge@fake_moe/24/64/60/4"]
        assert not library.measured_path().with_suffix(".tmp").exists()
        got = library.lookup(facts)
        assert got == {"tps": 24.5, "n": 20, "date": "2026-10-28", "name": "q35-edge"}, got
        # another bit width, an EXL3 variant and another machine have no record
        assert library.lookup(dict(facts, bits=3)) == {} and library.lookup(dict(facts, bits=3, format="exl3")) == {}
        assert library.lookup(facts, fp="fp-other") == {}
        # deleted: the record stays, and a re-download under another name finds it again
        shutil.rmtree(folder)
        assert library.lookup(facts)["tps"] == 24.5
        fake_folder(tmp / "mdl2" / "qwen-redownload", {
            "model_type": "fake_moe", "num_hidden_layers": 24, "hidden_size": 64, "num_experts": 60,
            "quantization": {"group_size": 64, "bits": 4}})
        again = library.record(tmp / "mdl2" / "qwen-redownload", 65, 4.0, today="2026-10-29")
        assert again["n"] == 1 and library.lookup(facts)["name"] == "qwen-redownload"   # the newer record leads
        # a folder without a config is kept by its name and matches no catalog entry
        (tmp / "mdl2" / "bare").mkdir()
        assert library.record(tmp / "mdl2" / "bare", 65, 4.0)["sig"] is None
        # the listings carry it, for a variant that is not on disk too
        models = tmp / "mmodels"
        models.mkdir()
        e = entry(facts, id="m", name="M")
        r = {**launch.judge(e, prof(), models, {"gbps": 3.42, "lat_ms": 0.227}), "key": "published", "label": "x", "note": ""}
        assert launch.variant_json(r, models, {"gbps": 3.42, "lat_ms": 0.227})["measured_here"]["name"] == "qwen-redownload"
        import types
        be = launch.TuiBackend(types.SimpleNamespace(refresh=False, port=8800), models)
        be.prof, be.disk = prof(), {"gbps": 3.42, "lat_ms": 0.227}
        assert be._variant(r).measured_here["tps"] == 16.0
        import remote
        row = {"models": [{"id": "m", "name": "M", "variants": [
            {"key": "published", "label": "x", "size": 1e9, "ok": True,
             "measured_here": {"tps": 13.8, "n": 12, "date": "2026-10-27"}}]}]}
        assert remote.models_from_listing(row)[0].variants[0].measured_here["n"] == 12
    finally:
        machine.CACHE = real[0]
        library._FP[:] = real[1]
    print("20. measured speed: replies of 64+ tokens, the median of the last 20 with count and date per machine "
          "and folder, kept after the folder is deleted, found by what the folder was, in the listings")


def check_delete(tmp):
    """Only a folder inside the launcher's models folder is deleted, after a yes: never one elsewhere (it
    says where), the models folder itself, a link or a path that climbs out, a running model, or a folder
    a copy made here links into; the same through --delete and through another computer's launcher."""
    import argparse
    import types
    import library
    import remote
    import tui
    models, home = tmp / "del_models", tmp / "del_home"
    models.mkdir()
    cfg = {"model_type": "fake_moe", "num_hidden_layers": 24, "hidden_size": 64, "num_experts": 60,
           "quantization": {"group_size": 64, "bits": 4}}
    pub = fake_folder(models / "m", cfg, packed=True)
    q3 = fake_folder(models / "m-q3", cfg, packed=True)
    q3.joinpath("model-00001-of-00001.safetensors").unlink()
    q3.joinpath("model-00001-of-00001.safetensors").symlink_to(pub / "model-00001-of-00001.safetensors")
    mine = fake_folder(home / ".cache" / "mine", cfg, packed=True)
    # outside the models folder, its link, a path that climbs out, the folder itself: nothing happens
    for p in (mine, models / ".." / "del_home" / ".cache" / "mine", models):
        ok, why = launch.delete_folder(p, models, running=[])
        assert not ok and mine.exists(), (p, why)
    assert "outside the launcher's models folder" in launch.delete_folder(mine, models, running=[])[1]
    (models / "sym").symlink_to(mine)
    assert not launch.delete_folder(models / "sym", models, running=[])[0] and mine.exists()
    (models / "sym").unlink()
    assert launch.delete_folder(models / "nope", models, running=[]) == (False, f"{models / 'nope'} is not on disk")
    # a running model, and the model a copy made here is linked to
    ok, why = launch.delete_folder(pub, models, running=[str(pub)])
    assert not ok and "is running" in why and pub.exists()
    ok, why = launch.delete_folder(pub, models, running=[])
    assert not ok and "m-q3 was made from it" in why and pub.exists()
    ok, why = launch.delete_folder(q3, models, running=[])                     # the copy goes; its source is intact
    assert ok and why.startswith(f"deleted {q3} (") and not q3.exists() and (pub / "experts.bin").exists()
    # which server.py is running comes from the process list and from ./larder's own state
    real_run = subprocess.run
    subprocess.run = lambda *a, **k: types.SimpleNamespace(stdout=(
        f"/usr/bin/python3 server.py --model {pub} --port 8800 --mtp\nvim server.py\n"
        "python3 server.py --port 9000\npython3 other.py --model /x\n"))
    try:
        assert launch.running_models() == {os.path.realpath(pub)}
        ok, why = launch.delete_folder(pub, models)
        assert not ok and "is running" in why
    finally:
        subprocess.run = real_run
    # --delete: the folder a variant lives in, after a yes
    again = fake_folder(models / "m-q4", cfg)
    ns = lambda **kw: argparse.Namespace(**{"delete": "m", "variant": "q4", "yes": True, "models_dir": str(models), **kw})
    real_ask, real_running = launch.ask, launch.running_models
    launch.running_models = lambda: set()
    try:
        launch.ask = lambda prompt, default="": "no"
        assert launch.delete_cli(ns(yes=False)) == 1 and again.exists()
        asked = []
        launch.ask = lambda prompt, default="": asked.append(prompt) or "yes"
        assert launch.delete_cli(ns(yes=False)) == 0 and not again.exists()
        assert asked and asked[0].startswith(f"Delete m q4 at {models / 'm-q4'} (") and asked[0].endswith("? type yes: ")
        fake_folder(models / "owner--M-8bit", cfg)
        assert launch.delete_cli(ns(variant="owner/M-8bit")) == 0 and not (models / "owner--M-8bit").exists()
        assert launch.delete_cli(ns(delete="../del_home/.cache/mine", variant="published")) == 1 and mine.exists()
        # a model the list found elsewhere is left alone and the answer says where it is; one found
        # inside the models folder under another name goes the same way as in the menu
        far = fake_folder(home / ".cache" / "far", dict(cfg, hidden_size=96), packed=True)
        near = fake_folder(models / "hand-placed", dict(cfg, hidden_size=128), packed=True)
        real_cur, real_hf, real_sibs = launch.CURATED, launch.hf_entry, launch.siblings
        real_lib = library.HOME, library.volumes, machine.CACHE
        library.HOME, library.volumes, machine.CACHE = home, lambda m: [], tmp / "del_cache"
        launch.CURATED = [entry(dict(Q15, model_type="fake_moe", layers=24, hidden=h), id=i, name=i.upper())
                          for i, h in (("far", 96), ("near", 128))]
        launch.hf_entry, launch.siblings = lambda c: c, lambda e, limit=4: []
        out = io.StringIO()
        real_out, sys.stdout = sys.stdout, out
        try:
            assert launch.delete_cli(ns(delete="far", variant="published")) == 1
            where = library.short_path(far)
            assert launch.delete_cli(ns(delete="near", variant="published")) == 0
        finally:
            sys.stdout = real_out
            launch.CURATED, launch.hf_entry, launch.siblings = real_cur, real_hf, real_sibs
            library.HOME, library.volumes, machine.CACHE = real_lib
        said = out.getvalue().splitlines()
        assert said[0].startswith(f"far published is at {where}, outside the launcher's models folder"), said
        assert far.exists() and said[1].startswith(f"deleted {near} (") and not near.exists()
    finally:
        launch.ask, launch.running_models = real_ask, real_running
    # the menu backend: here the folder its list showed, there the device's own launcher over ssh
    be = launch.TuiBackend(types.SimpleNamespace(refresh=False, port=8800), models)
    local = tui.Device("this computer")
    model = tui.Model("m", "M")
    v = tui.Variant("published", "x", 1, None, True, "", "ready", path=str(pub), own=True)
    real_running = launch.running_models
    launch.running_models = lambda: set()
    try:
        assert "deleted" in be.delete(local, model, v) and not pub.exists()
        try:
            be.delete(local, model, tui.Variant("published", "x", 1, None, True, "", "ready", path=str(mine)))
            raise AssertionError("a folder elsewhere was deleted")
        except RuntimeError as err:
            assert "outside the launcher's models folder" in str(err)
    finally:
        launch.running_models = real_running
    shed = tui.Device("shed", "shed", "preset", "presets.sh", ("shed",), "", "", "ready", preset="shed",
                     dir="~/moe-ssd-offload", python="~/venv38/bin/python")
    desk = tui.Device("desk", "desk", "tailscale", "tailscale", (), "", "", "ready")
    assert remote.delete_argv(shed, "flash-next-exl3", "published")[-1] == (
        "cd ~/moe-ssd-offload && ~/venv38/bin/python launch.py --delete flash-next-exl3 --variant published --yes")
    assert remote.delete_argv(desk, "m", "owner/M-8bit")[-1] == \
        "cd ~/larder && ./larder --delete m --variant owner/M-8bit --yes"
    assert remote.delete_argv(desk, "m; rm -rf ~", "$(x)")[-1].endswith(
        "--delete 'm; rm -rf ~' --variant '$(x)' --yes")
    assert remote.delete_argv(desk, "m", "q3")[:len(remote.SSH)] == remote.SSH and remote.delete_argv(tui.Device("x"), "m", "q3") is None
    real_remote = remote._run
    seen = []
    remote._run = lambda argv, timeout=20: (seen.append(argv), (0, "some ssh noise\ndeleted /home/u/m (4.0 GB)\n", ""))[1]
    try:
        assert be.delete(desk, model, v) == "deleted /home/u/m (4.0 GB)" and seen[0][-1].endswith("--delete m --variant published --yes")
        remote._run = lambda argv, timeout=20: (1, "", "m is running: stop it first\n")
        try:
            be.delete(desk, model, v)
            raise AssertionError("a refusal was swallowed")
        except RuntimeError as err:
            assert str(err) == "m is running: stop it first"
    finally:
        remote._run = real_remote
    print("21. delete: only inside the models folder, after a yes; elsewhere says where it is; never the running "
          "model, a link out, a path that climbs out, or a source a made copy links to; --delete and the ssh form")


def check_search(tmp):
    """A name or a link goes to its repo and words to a Hugging Face search, which every computer runs and judges with its own launcher (--search Q --json) and starts with --model owner/name."""
    import types
    import remote
    import tui
    for text, want in (("owner/name", "owner/name"), ("  owner/Name-4bit \n", "owner/Name-4bit"),
                       ("https://huggingface.co/mlx-community/Qwen3-4B-4bit", "mlx-community/Qwen3-4B-4bit"),
                       ("https://huggingface.co/org/repo/tree/main", "org/repo"),
                       ("https://huggingface.co/org/repo/blob/main/config.json?download=true#top", "org/repo"),
                       ("https://huggingface.co/org/repo/resolve/abc123/model.safetensors", "org/repo"),
                       ("huggingface.co/org/my.repo_v2?x=1", "org/my.repo_v2"),
                       ("http://www.huggingface.co/org/repo/", "org/repo"), ("https://hf.co/org/repo", "org/repo"),
                       ("qwen3 moe", None), ("qwen3", None), ("", None), ("a/b/c", None),
                       ("https://huggingface.co/datasets/org/data", None), ("https://huggingface.co/models?search=qwen", None),
                       ("https://example.com/org/repo", None)):
        assert launch.parse_repo(text) == want, (text, launch.parse_repo(text), want)
    calls = []
    real = launch.hf_entry, launch.hf_search
    launch.hf_entry = lambda c: calls.append(("repo", c["repo"])) or entry(Q15, id=c["id"], name=c["name"], repo=c["repo"])
    launch.hf_search = lambda q, limit=8: calls.append(("search", q)) or ([], ["x"])
    try:
        got, skipped = launch.search_models("https://huggingface.co/owner/Foo-4bit/tree/main")
        assert calls == [("repo", "owner/Foo-4bit")] and got[0]["id"] == "owner--Foo-4bit" and skipped == []
        launch.search_models("qwen3 moe")
        assert calls[-1] == ("search", "qwen3 moe")
    finally:
        launch.hf_entry, launch.hf_search = real
    # the launcher on the device answers --search Q --json with its own judgement; the menu reads it
    shed = tui.Device("shed", "shed", "preset", "presets.sh", ("shed",), "", "", "ready", preset="shed",
                     dir="~/moe-ssd-offload", python="~/venv38/bin/python")
    desk = tui.Device("desk", "desk", "tailscale", "tailscale", (), "", "", "ready")
    assert remote.search_argv(shed, "qwen3 moe")[-1] == \
        "cd ~/moe-ssd-offload && ~/venv38/bin/python launch.py --search 'qwen3 moe' --json"
    assert remote.search_argv(desk, "owner/Foo")[-1] == "cd ~/larder && ./larder --search owner/Foo --json"
    assert remote.search_argv(desk, "x; rm -rf ~")[-1].endswith("--search 'x; rm -rf ~' --json")
    assert remote.search_argv(desk, "a")[:len(remote.SSH)] == remote.SSH and remote.search_argv(tui.Device("x"), "a") is None
    be = launch.TuiBackend(types.SimpleNamespace(refresh=False, port=8800), tmp / "search_models")
    row = {"id": "owner--Foo-4bit", "name": "owner/Foo-4bit", "repo": "owner/Foo-4bit", "kind": "MoE", "tested": False,
           "variants": [{"key": "published", "label": "as published, 4-bit", "size": 8e9, "tps": 30.0, "ok": False,
                         "why": "needs 18 GB free on disk, 12.9 GB free", "disk_short": True,
                         "status": "needs 18 GB free on disk, 12.9 GB free", "entry_id": "owner--Foo-4bit"}]}
    answer = json.dumps({"machine": {"chip": "Apple M4", "free_gb": 12.9}, "skipped": 3, "error": "", "models": [row]})
    real_run, seen = remote._run, []
    try:
        remote._run = lambda argv, timeout=20: (seen.append(argv), (0, "warning: hub\n" + answer + "\n", ""))[1]
        models, skipped = be.device_search(shed, "https://huggingface.co/owner/Foo-4bit", lambda t: None)
        assert seen[0][-1].endswith("--search https://huggingface.co/owner/Foo-4bit --json") and skipped == 3
        m = models[0]
        assert m.found and m.repo == "owner/Foo-4bit" and m.variants[0].disk_short and not m.variants[0].ok
        assert shed.info == {"chip": "Apple M4", "free_gb": 12.9}
        for out, code, err, want in (
                (json.dumps({"models": [], "error": "model type 'x' is not in the installed mlx-lm"}), 0, "",
                 "model type 'x' is not in the installed mlx-lm"),
                ("", 255, "ssh: connect to host shed port 22: Connection refused\n", "shed: ssh returned no answer (ssh: connect"),
                ("nothing json", 0, "", "shed: the launcher there printed no JSON")):
            remote._run = lambda argv, timeout=20, a=(code, out, err): a
            try:
                be.device_search(shed, "owner/nope", lambda t: None)
                raise AssertionError("an error was swallowed")
            except RuntimeError as e:
                assert want in str(e), (str(e), want)
        # starting a result on a computer: its launcher gets the repo and the variant, a tested model its id
        n = len(remote.SSH_START)
        argv, port = remote.start_argv(desk, "owner/Foo-4bit", 8801)
        assert argv[n + 3] == "cd ~/larder && ./larder --auto --model owner/Foo-4bit --no-browser --port 8801"
        argv, port = remote.start_argv(desk, "flash-next-exl3", 8801, variant="q3")
        assert argv[n + 3] == "cd ~/larder && ./larder --auto --model flash-next-exl3 --variant q3 --no-browser --port 8801"
    finally:
        remote._run = real_run
    # the whole command: --search Q --json prints the rows --list --json prints, judged here, with the reason on a failure
    import contextlib
    import library
    models_dir = tmp / "search_models"
    models_dir.mkdir(exist_ok=True)
    saved = (sys.argv, launch.search_models, launch.siblings, machine.CACHE, library.HOME, library.volumes,
             shutil.disk_usage, machine.profile, machine.ssd)
    machine.CACHE, library.HOME, library.volumes = tmp / "search_cache", tmp / "no_home", lambda m: []
    shutil.disk_usage = lambda p: saved[6](p)._replace(free=int(500e9))
    here = dict(prof(), chip="Apple M4", arch="arm64", os="macOS 27.0", gpu="Apple M4",
                now={"avail_gb": 14.0, "disks": []})
    machine.profile, machine.ssd = (lambda **kw: here), (lambda *a, **kw: {"gbps": 3.42, "lat_ms": 0.227})
    launch.siblings = lambda e, limit=4: []
    try:
        for found, why in (([entry(dict(Q15, model_type="fake_moe", layers=24, hidden=64), id="owner--Foo-4bit",
                                   name="owner/Foo-4bit", repo="owner/Foo-4bit")], ""), ([], "no safetensors weights in the repo")):
            def fake_search(q, limit=8, found=found, why=why):
                if why:
                    raise launch.Unsupported(why)
                return found, ["a", "b"]
            launch.search_models = fake_search
            sys.argv = ["launch.py", "--search", "owner/Foo-4bit", "--json", "--no-measure", "--models-dir", str(models_dir)]
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                launch.main()
            data = json.loads(buf.getvalue())
            assert data["error"] == why and set(data) == {"machine", "engine", "skipped", "error", "models"}, data
            if not why:
                row = data["models"][0]
                assert row["repo"] == "owner/Foo-4bit" and row["tested"] is False and data["skipped"] == 2
                assert [v["key"] for v in row["variants"]] == ["published", "q3"] and row["variants"][0]["ok"]
            else:
                assert data["models"] == []
    finally:
        (sys.argv, launch.search_models, launch.siblings, machine.CACHE, library.HOME, library.volumes,
         shutil.disk_usage, machine.profile, machine.ssd) = saved
    print("22. search: a name or a link (strip /tree, /blob, queries) goes to its repo, words to a search; every "
          "computer searches and judges with its own launcher (--search --json) and starts the result with --model")


def main():
    mx.random.seed(0)
    check_catalog_filter()
    import library
    real_lib = library.HOME, library.volumes, machine.CACHE, os.environ.get("HF_HUB_CACHE")
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        # this computer's own folders must not decide a test: an empty home, no other volumes
        library.HOME, library.volumes, machine.CACHE = tmp / "no_home", lambda m: [], tmp / "main_cache"
        os.environ["HF_HUB_CACHE"] = str(tmp / "no_hub")
        src = check_local_folders(tmp)
        check_pack(tmp, src)
        check_partial_fetch(tmp, src)
        check_profile_cache(tmp)
        check_variants(tmp)
        check_requant(tmp, src)
        check_auto(tmp)
        check_menu_backend(tmp)
        check_listing_json(tmp)
        check_remote_commands(tmp)
        check_version_handshake(tmp)
        check_update_engine(tmp)
        check_install_engine(tmp)
        check_device_safety(tmp)
        check_disk_short(tmp)
        check_found_copies(tmp)
        check_base_model(tmp)
        check_measured(tmp)
        check_estimate_learns(tmp)
        check_mtp_head(tmp)
        check_delete(tmp)
        check_search(tmp)
        check_cuda_q40(tmp)
        check_exl3_resume(tmp, src)
        check_cuda_serve_exec(tmp)
        check_first_reply(tmp)
        library.HOME, library.volumes, machine.CACHE = real_lib[:3]
        os.environ.pop("HF_HUB_CACHE", None) if real_lib[3] is None else os.environ.__setitem__("HF_HUB_CACHE", real_lib[3])
    check_settings()
    check_estimate()
    check_bits_estimate()
    check_live_speed()
    print("test_launcher: ok")


if __name__ == "__main__":
    main()
