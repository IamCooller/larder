#!/usr/bin/env python3
"""A sidecar on a second drive (mirror_experts.py, stream_fast._Stripe), no
real model:

  1. A synthetic pack (2 layers x 3 projections x 16 experts): mirror_experts
     copies half the experts and writes the json; _read_wave fills every
     buffer with experts.bin's bytes while both devices read; a sidecar whose
     reads start failing mid-run (an empty file put under its fd) falls back
     to experts.bin with the same bytes, once noted, and later waves skip it;
     a sidecar that is not a copy, made for another pack, found by the
     default search on experts.bin's own drive, or MOE_STRIPE=0 is not opened.
  2. test_store's toy MoE, Python path, arena and budget mode: greedy tokens
     and logprobs with the sidecar equal those without it, the sidecar did
     serve reads, and a sidecar that fails after the first turn changes
     nothing either.
  3. The same with MOE_NATIVE (when moe_native has set_mirror): the agent
     reads from the sidecar and falls back the same way.

    python3 tests/test_stripe.py
"""
import contextlib
import gc
import io
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np
import mlx.core as mx
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache

import mirror_experts
import stream_fast as sf
import test_store as ts

PARTS = (4096, 512, 256)         # one record's weight, scales, biases bytes


def even(s):
    """Same rate on both devices, so the split gives the sidecar a share."""
    s.ema["ext"] = list(s.ema["int"])


def mirror(pack, drive, fraction):
    # The guards are the run's, not this toy copy's.
    mirror_experts.main(["--model", str(pack), "--drive", str(drive), "--seconds", "0.05",
                         "--fraction", str(fraction), "--min-margin-gb=-1e9",
                         "--max-swap-mb", "1e9"])
    return Path(drive) / Path(pack).resolve().name / mirror_experts.META


def break_sidecar(s, tmp):
    """Reads of the sidecar return nothing from here on, as from a drive gone."""
    empty = Path(tmp) / "empty"
    empty.write_bytes(b"")
    fd = os.open(empty, os.O_RDONLY)
    os.dup2(fd, s.fd)
    os.close(fd)


def check_synthetic(tmp):
    pack = Path(tmp) / "synth"
    pack.mkdir()
    rng = np.random.default_rng(0)
    stride, n_exp, index, off = sum(PARTS), 16, {}, 0
    for layer in range(2):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            index[f"model.layers.{layer}.mlp.switch_mlp.{proj}"] = {
                "offset": off, "stride": stride, "n_experts": n_exp,
                "parts": [{"name": nm, "bytes": b, "shape": [b], "dtype": "U8"}
                          for nm, b in zip(("weight", "scales", "biases"), PARTS)]}
            off += stride * n_exp
    data = rng.integers(0, 256, off, dtype=np.uint8).tobytes()
    (pack / "experts.bin").write_bytes(data)
    (pack / "experts_index.json").write_text(json.dumps(index))
    meta_p = mirror(pack, Path(tmp) / "drive", 0.5)
    meta = json.loads(meta_p.read_text())
    assert 0 < len(meta["records"]) < 6 * n_exp, len(meta["records"])
    groups = {}
    for g, x, o, n in mirror_experts.all_records(index):
        groups.setdefault((g, x), []).append(mirror_experts.picked(g, x, 0.5))
    assert all(len(set(v)) == 1 for v in groups.values()), "a layer's experts split up"

    os.environ["MOE_STRIPE"] = str(meta_p)
    packed = sf.load_packed(pack)
    s = sf.open_stripe(pack, packed)
    assert s is not None and len(s.at) == len(meta["records"])
    even(s)
    fd = packed["fd"].fileno()

    def wave(ids):
        bufs, jobs = [], []
        for key, e in ids:
            b = np.zeros(stride, dtype=np.uint8)
            cut = np.cumsum(PARTS)[:-1]
            jobs.append((fd, [memoryview(p) for p in np.split(b, cut)],
                         index[key]["offset"] + e * stride, stride))
            bufs.append((b, index[key]["offset"] + e * stride))
        sf._read_wave(jobs)
        for b, o in bufs:
            assert b.tobytes() == data[o:o + stride], f"record at {o} differs"

    keys = list(index)
    ids = [(keys[k], int(e)) for k in range(len(keys)) for e in rng.permutation(n_exp)[:6]]
    st0 = dict(sf.STATS)
    for _ in range(3):
        wave(ids)
    ext = sf.STATS["stripe_ext_b"] - st0["stripe_ext_b"]
    own = sf.STATS["stripe_int_b"] - st0["stripe_int_b"]
    assert ext > 0 and own > 0, (ext, own)
    assert ext + own == 3 * len(ids) * stride
    break_sidecar(s, tmp)
    wave(ids)
    assert s.dead and sf.STATS["stripe_fell"] > st0["stripe_fell"]
    e1 = sf.STATS["stripe_ext_b"]
    wave(ids)
    assert sf.STATS["stripe_ext_b"] == e1, "a dead sidecar still got reads"
    print(f"1. synthetic pack: {len(meta['records'])} of {6 * n_exp} records mirrored; "
          f"3 waves {ext} B sidecar + {own} B experts.bin, bytes equal; after a failure "
          f"{sf.STATS['stripe_fell'] - st0['stripe_fell']} reads fell back, bytes equal")

    # Not opened: a copy that is not one, another pack, switched off.
    sc = meta_p.with_name(meta["file"])
    raw = bytearray(sc.read_bytes())
    raw[0] ^= 1
    sc.write_bytes(bytes(raw))
    assert sf.open_stripe(pack, packed) is None
    raw[0] ^= 1
    sc.write_bytes(bytes(raw))
    assert sf.open_stripe(pack, packed) is not None
    (pack / "experts_index.json").write_text(json.dumps(index, indent=1))
    assert sf.open_stripe(pack, packed) is None
    (pack / "experts_index.json").write_text(json.dumps(index))
    # The default search finds the sidecar (here under tmp, as under /Volumes)
    # on experts.bin's own volume and skips it.
    vol, sf._VOLUMES = sf._VOLUMES, Path(tmp)
    os.environ["MOE_STRIPE"] = "auto"
    said = io.StringIO()
    try:
        with contextlib.redirect_stdout(said):
            assert sf.open_stripe(pack, packed) is None
    finally:
        sf._VOLUMES = vol
    assert "on experts.bin's drive" in said.getvalue(), said.getvalue()
    os.environ["MOE_STRIPE"] = "0"
    assert sf.open_stripe(pack, packed) is None
    print("   a changed record, another pack's index, found on experts.bin's drive and "
          "MOE_STRIPE=0: not opened")


def run(out, meta_p, peak, native, fail_after=None, tmp=None):
    """Four turns of greedy decode; (tokens, logprobs, (sidecar bytes in all,
    of them the native agent's, reads that fell back))."""
    os.environ["MOE_STRIPE"] = str(meta_p) if meta_p else "0"
    model = ts.stock_model(out)
    sf.install(model, out, resident=ts.RES, cache_bias=0.0,
               defer=False, native=native)
    if peak:
        sf.enable_budget(model, ts.budget_gb())
    if sf._STRIPE is not None:
        even(sf._STRIPE)
        if native:
            sf._NATIVE.agent.set_mirror(sf._STRIPE.fd, [(m, e, n) for m, (e, n)
                                                        in sf._STRIPE.at.items()],
                                        1e9, 1e9, sf._STRIPE.readers)
    py_ext = [0]
    note = sf._Stripe.note

    def counted(self, dev, nbytes, secs):
        py_ext[0] += nbytes if dev == "ext" else 0
        note(self, dev, nbytes, secs)
    sf._Stripe.note = counted
    e0, f0 = sf.STATS["stripe_ext_b"], sf.STATS["stripe_fell"]
    cache, toks, lps = make_prompt_cache(model), [], []
    try:
        for i, (p, n) in enumerate(zip(ts.PROMPTS, ts.GEN)):
            if i == fail_after:
                sf.native_sync()
                break_sidecar(sf._STRIPE, tmp)
            for t, lp in generate_step(mx.array(p, dtype=mx.int32), model,
                                       max_tokens=n, prompt_cache=cache):
                toks.append(int(t))
                lps.append(np.array(lp.astype(mx.float32)))
        sf.native_sync()
    finally:
        sf._Stripe.note = note
    if native:
        assert sf._NATIVE is not None and not sf._NATIVE.agent.error, "native agent error"
    ext = sf.STATS["stripe_ext_b"] - e0
    got = ext, ext - py_ext[0], sf.STATS["stripe_fell"] - f0
    del model, cache
    ts.settle()
    return toks, lps, got


def check_engine(tmp, peak, native):
    out = Path(tmp) / "toy"
    if not out.exists():
        ts.build(out)
    meta_p = Path(tmp) / "drive" / "toy" / mirror_experts.META
    if not meta_p.exists():
        mirror(out, Path(tmp) / "drive", 1.0)
    a = run(out, None, peak, native)
    b = run(out, meta_p, peak, native)
    c = run(out, meta_p, peak, native, fail_after=1, tmp=tmp)
    what = f"{'native' if native else 'python'}, {'budget' if peak else 'arena'}"
    for name, x in (("sidecar", b), ("sidecar failing after turn 1", c)):
        assert a[0] == x[0], f"{what}, {name}: tokens differ\n{a[0]}\n{x[0]}"
        bad = [i for i, (p, q) in enumerate(zip(a[1], x[1])) if not np.array_equal(p, q)]
        assert not bad, f"{what}, {name}: logprobs differ at steps {bad}"
    assert a[2] == (0, 0, 0) and b[2][0] > 0 and b[2][2] == 0, (a[2], b[2])
    assert c[2][0] > 0 and c[2][2] > 0, c[2]
    if native:
        assert b[2][1] > 0, f"the agent never read the sidecar: {b[2]}"
    print(f"{3 if native else 2}. {what}: {len(a[0])} tokens and logprobs equal with the "
          f"sidecar ({b[2][0]} B read there, {b[2][1]} by the agent) and with it failing "
          f"after turn 1 ({c[2][2]} reads fell back)")


def main():
    try:
        import moe_native
        has = hasattr(moe_native.Agent, "set_mirror")
    except ImportError:
        has = False
    with tempfile.TemporaryDirectory() as tmp:
        check_synthetic(tmp)
        # Native installs before any budget: a budget wraps the layer stack's
        # class, and the native wrapper must sit inside it (install_native).
        for peak in (False, True):
            check_engine(tmp, peak, native=False)
            if has:
                check_engine(tmp, peak, native=True)
        if not has:
            print("3. skipped: no moe_native with set_mirror here")
    os.environ.pop("MOE_STRIPE", None)
    gc.collect()
    print("ok")


if __name__ == "__main__":
    main()
