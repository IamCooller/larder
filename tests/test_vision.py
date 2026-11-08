#!/usr/bin/env python3
"""Images (vision.py, StreamingLLM.supports_images), no checkpoint opened.

Against transformers (the model's reference implementation), fp32 on the CPU:
  1. Pixels: vision.patches equals Qwen2VLImageProcessorPil's pixel_values and
     grid on images of several sizes, an RGBA one over white.
  2. Positions: vision.positions equals Qwen3_5MoeModel.get_rope_index on a
     prompt with two images, the delta too.
  3. M-RoPE: the _MRoPE hook over stock nn.RoPE (head 256, 64 rotated dims,
     sections [11, 11, 10]) matches transformers' interleaved rotary on a pass
     of text and image positions; a text pass is stock's to the bit.
  4. Tower: vision._Tower with tiny random weights matches
     Qwen3_5MoeVisionModel.
On the toy hybrid engine (support.toy_llm: GatedDeltaNet + KV layers, the
real Qwen3.6 tokenizer and chat template) with a tiny tower, CPU:
  5. An image turn's reply and its kept cache equal a reference: one pass of
     the whole prompt with the image rows spliced by hand and positions from
     transformers' get_rope_index through an independent RoPE, greedy decode by
     recomputing the whole sequence.
  6. Kept conversations key on the image: the next turn with the same image
     is taken up past it (no tower run); with another image of the same size
     it is not, and its reply is a fresh engine's. A text-only conversation on
     an engine whose hooks are installed is bit-identical to a plain engine.
  7. server.clean_messages keeps image parts only for a model that takes them.

    perl -e 'alarm shift; exec @ARGV' 280 python3 tests/test_vision.py
"""
import base64
import contextlib
import io
import shutil
import sys
import tempfile
import types
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch
from PIL import Image

import moe_stream
import pcache
import vision
import support
from support import gdn, kv, toy_llm
from mlx_lm.models.cache import make_prompt_cache

from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as hf
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeTextConfig, Qwen3_5MoeVisionConfig)

VC = {"depth": 2, "hidden_size": 64, "intermediate_size": 128, "num_heads": 4,
      "out_hidden_size": 32, "patch_size": 4, "spatial_merge_size": 2,
      "temporal_patch_size": 2, "num_position_embeddings": 64, "in_channels": 3,
      "model_type": "qwen3_5_moe", "deepstack_visual_indexes": []}
PRE = {"size": {"shortest_edge": 64, "longest_edge": 48 * 48},
       "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5]}
SECTION = [2, 1, 1]          # the toy rotates 8 dims: 4 frequencies


def picture(seed, h, w, mode="RGB"):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 256, (h, w, len(mode)), dtype=np.uint8)
    return Image.fromarray(a, mode)


def data_url(img):
    b = io.BytesIO()
    img.save(b, format="PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def check_pixels():
    real = dict(VC, patch_size=16)
    pre = {"size": {"shortest_edge": 65536, "longest_edge": 16777216},
           "image_mean": [0.5] * 3, "image_std": [0.5] * 3}
    ref = Qwen2VLImageProcessorPil(size=pre["size"], patch_size=16, merge_size=2,
                                   temporal_patch_size=2, image_mean=[0.5] * 3,
                                   image_std=[0.5] * 3)
    for k, (h, w) in enumerate([(300, 457), (64, 900), (1080, 1920), (517, 333)]):
        img = picture(k, h, w)
        got, grid = vision.patches(img, real, pre)
        want = ref(images=[img], return_tensors="np")
        assert tuple(want["image_grid_thw"][0]) == grid, (want["image_grid_thw"], grid)
        d = float(np.abs(want["pixel_values"] - got).max())
        assert d < 1e-6, (h, w, d)
    # Transparent pixels over white, then the same processor.
    rgba = picture(9, 120, 200, "RGBA")
    _, img = vision._decode(data_url(rgba))
    a = np.asarray(rgba).astype(np.float64)
    over = a[..., :3] * a[..., 3:] / 255 + 255 * (1 - a[..., 3:] / 255)
    assert np.abs(np.asarray(img).astype(np.float64) - over).max() <= 1.0
    print("  1. pixels: equal to Qwen2VLImageProcessorPil (max |d| < 1e-6), "
          "alpha over white")


def check_positions():
    pad = 7
    ids = [1, 2, 3, 9, pad, 8, 4, 5, 9, pad, 8, 6]
    grids = [(1, 6, 10), (1, 8, 4)]
    lay = vision.Layout(ids, pad, [None, None], grids, ["a", "b"], 2)
    t = torch.tensor([lay.ids])
    mm = (t == pad).int()
    me = types.SimpleNamespace(config=types.SimpleNamespace(
        vision_config=types.SimpleNamespace(spatial_merge_size=2)))
    me.get_vision_position_ids = lambda *a, **k: hf.Qwen3_5MoeModel.get_vision_position_ids(me, *a, **k)
    want, delta = hf.Qwen3_5MoeModel.get_rope_index(
        me, t, mm, image_grid_thw=torch.tensor(grids))
    assert (want[:, 0].numpy() == lay.P).all()
    assert int(delta) == lay.delta, (int(delta), lay.delta)
    print(f"  2. positions: equal to get_rope_index ({lay.n} ids, delta {lay.delta})")
    return lay


def hf_rope(x, P, cfg):
    rot = hf.Qwen3_5MoeTextRotaryEmbedding(cfg)
    xt = torch.tensor(np.array(x))
    cos, sin = rot(xt, torch.tensor(P)[:, None])
    q, _ = hf.apply_rotary_pos_emb(xt, xt, cos, sin)
    return q.numpy()


def check_mrope():
    D, base, sec = 256, 1e7, [11, 11, 10]
    cfg = Qwen3_5MoeTextConfig(head_dim=D, rope_parameters={
        "rope_type": "default", "rope_theta": base, "partial_rotary_factor": 0.25,
        "mrope_section": sec, "mrope_interleaved": True})
    sel = np.zeros(32, np.int32)
    sel[1:33:3], sel[2:30:3] = 1, 2
    box = vision._Box()
    inner = nn.RoPE(64, traditional=False, base=base)
    rope = vision._MRoPE(inner, box, np.concatenate([sel, sel]))
    pad = 7
    ids = [1] * 20 + [pad] + [2] * 30
    box.layout = lay = vision.Layout(ids, pad, [None], [(1, 12, 20)], ["a"], 2)
    x = mx.random.normal((1, 4, lay.n + 5, D))
    got = rope(x, 0)                       # every prompt position and 5 after
    P = np.concatenate([lay.P, np.arange(lay.n, lay.n + 5) + lay.delta + np.zeros((3, 1), np.int64)], axis=1)
    want = hf_rope(x, P, cfg)
    d = float(np.abs(np.array(got) - want).max())
    assert d < 2e-4, d
    # A text pass after the image is stock's rotation at the shifted offset.
    y = mx.random.normal((1, 4, 3, D))
    a, b = rope(y, lay.n + 2), inner(y, lay.n + 2 + lay.delta)
    assert mx.array_equal(a, b)
    box.layout = None
    assert mx.array_equal(rope(y, 9), inner(y, 9))
    print(f"  3. M-RoPE: matches transformers (max |d| {d:.1e}); text passes "
          f"bit-identical to stock")


def tiny_tower(seed=0):
    torch.manual_seed(seed)
    cfg = Qwen3_5MoeVisionConfig(**{k: v for k, v in VC.items()
                                    if k not in ("model_type", "deepstack_visual_indexes")},
                                 hidden_act="gelu_pytorch_tanh")
    ref = hf.Qwen3_5MoeVisionModel(cfg).float().eval()
    with torch.no_grad():
        for p in ref.parameters():
            p.normal_(0, 0.05)
    t = vision._Tower(VC)
    w = []
    for k, v in ref.state_dict().items():
        a = mx.array(v.numpy())
        if k == "patch_embed.proj.weight":
            k, a = "patch_embed.weight", a.transpose(0, 2, 3, 4, 1)
        w.append((k.replace("patch_embed.proj.", "patch_embed."), a))
    t.load_weights(w)
    return t, ref


def check_tower():
    t, ref = tiny_tower()
    gh, gw = 8, 12
    px = np.random.default_rng(1).normal(size=(gh * gw, 3 * 2 * 4 * 4)).astype(np.float32)
    got = np.array(t(mx.array(px), (1, gh, gw)))
    with torch.no_grad():
        want = ref(torch.tensor(px), torch.tensor([[1, gh, gw]])).pooler_output.numpy()
    d = float(np.abs(got - want).max() / np.abs(want).max())
    assert got.shape == want.shape and d < 1e-4, (got.shape, want.shape, d)
    rows, vision.ROWS = vision.ROWS, 40     # a big image's pieces, on a small one
    try:
        cut = np.array(t(mx.array(px), (1, gh, gw)))
    finally:
        vision.ROWS = rows
    d2 = float(np.abs(cut - want).max() / np.abs(want).max())
    assert d2 < 1e-4, d2
    print(f"  4. tower: matches Qwen3_5MoeVisionModel (rel |d| {d:.1e}; in "
          f"40-patch pieces {d2:.1e})")
    return t


def with_vision(llm, tower):
    """The toy engine with the tiny tower: vision.attach's work, by hand."""
    stack = vision._stack_of(llm.model)
    ropes = [l.self_attn for l in stack.layers if hasattr(l, "self_attn")]
    sel = np.zeros(4, np.int32)
    sel[1:3 * SECTION[1]:3], sel[2:3 * SECTION[2]:3] = 1, 2
    pad = llm.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    cfg = {"vision_config": VC, "image_token_id": pad}
    v = vision.Vision(support.MODEL_LABEL, llm.model, cfg, {}, 0, ropes,
                      np.concatenate([sel, sel]))
    v.pre = PRE
    v.runs = 0

    def load():
        v.runs += 1
        return tower
    v._load = load
    llm._vision, llm.supports_images = v, True
    return v


class RefRoPE(nn.Module):
    """transformers' interleaved M-RoPE formula over a whole pass: cos and sin
    per position row, picked per frequency, rotate_half."""

    def __init__(self, dims, base, P):
        super().__init__()
        self.dims, self.P = dims, P
        self.inv = 1.0 / (base ** (np.arange(0, dims, 2, dtype=np.float32) / dims))

    def __call__(self, x, offset=0):
        sel = np.zeros(self.dims // 2, np.int64)
        sel[1:3 * SECTION[1]:3], sel[2:3 * SECTION[2]:3] = 1, 2
        f = (self.P[sel].T * self.inv).astype(np.float32)      # T, dims/2
        emb = np.concatenate([f, f], axis=-1)
        cos, sin = mx.array(np.cos(emb)), mx.array(np.sin(emb))
        r, rest = x[..., :self.dims], x[..., self.dims:]
        h = self.dims // 2
        rot = mx.concatenate([-r[..., h:], r[..., :h]], axis=-1)
        return mx.concatenate([r * cos + rot * sin, rest], axis=-1)


def reference(llm, v, msgs, n_new):
    """Greedy tokens and the prompt's one-pass cache, by hand: ids widened,
    rows spliced, positions from transformers, the whole sequence recomputed
    for every token."""
    stack = vision._stack_of(llm.model)
    ids = llm.tokenizer.encode(llm._raw_render(msgs))
    urls = vision.images_of(msgs)
    imgs = [vision._decode(u)[1] for u in urls]
    px = [vision.patches(i, VC, PRE) for i in imgs]
    pad = v.pad
    wide, rows = [], []
    for t in ids:
        if t == pad:
            p, g = px[len(rows)]
            n = g[1] * g[2] // 4
            wide += [pad] * n
            rows.append(v._load()(mx.array(p), g))
        else:
            wide.append(t)
    grids = torch.tensor([g for _, g in px])
    me = types.SimpleNamespace(config=types.SimpleNamespace(
        vision_config=types.SimpleNamespace(spatial_merge_size=2)))
    me.get_vision_position_ids = lambda *a, **k: hf.Qwen3_5MoeModel.get_vision_position_ids(me, *a, **k)
    attn = [l.self_attn for l in stack.layers if hasattr(l, "self_attn")]
    saved = [a.rope for a in attn]
    out, cache = [], None
    try:
        for step in range(n_new + 1):
            seq = wide + out
            t = torch.tensor([seq])
            P, _ = hf.Qwen3_5MoeModel.get_rope_index(me, t, (t == pad).int(), image_grid_thw=grids)
            for a in attn:
                a.rope = RefRoPE(a.rope.dims if hasattr(a.rope, "dims") else 8, 100000.0, P[:, 0].numpy())
            h = stack.embed_tokens(mx.array(seq)[None])
            at = [i for i, x in enumerate(seq) if x == pad]
            j = 0
            for r in rows:
                h[:, at[j]:at[j] + r.shape[0]] = r[None]
                j += r.shape[0]
            c = make_prompt_cache(llm.model) if step == 0 else None
            logits = llm.model(mx.array(seq)[None], cache=c, input_embeddings=h)
            if step == 0:
                cache = c
            if step < n_new:
                out.append(int(mx.argmax(logits[0, -1])))
    finally:
        for a, r in zip(attn, saved):
            a.rope = r
    return out, cache, wide


def max_abs(a, b):
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))


def check_engine(tower):
    img1, img2 = picture(3, 24, 40), picture(4, 24, 40)
    msgs = [{"role": "system", "content": "You look at pictures."},
            {"role": "user", "content": [
                {"type": "text", "text": "What is in"},
                {"type": "image_url", "image_url": {"url": data_url(img1)}},
                {"type": "text", "text": "this picture?"}]}]
    llm = toy_llm("slice", max_tokens=5)
    llm.reuse, llm.keep_convs = True, 2
    v = with_vision(llm, tower)
    assert llm.supports_images
    llm.ask(messages=msgs, max_tokens=5)
    st = llm.stats()
    got = llm._seen[len(llm._seen) - st["tokens"]:]
    assert st["images"] == 1 and st["image_tokens"] == 15, st
    box = v.box.layout
    v.box.layout = None                     # the reference brings its own positions
    want, ref_cache, wide = reference(llm, v, msgs, len(got))
    v.box.layout = box
    assert got == want, (got, want)
    n = len(wide)
    worst = 0.0
    for (a, b), (c, d) in zip(kv(llm._cache), kv(ref_cache)):
        worst = max(worst, max_abs(a[..., :n, :], c), max_abs(b[..., :n, :], d))
    assert worst < 1e-4, worst
    front = st["cached_tokens"]
    print(f"  5. image turn: reply {got} equals the reference; prompt KV within "
          f"{worst:.1e} of one hand-spliced pass")

    # 6. The next turn, same image: taken up past it, no tower run.
    runs = v.runs
    nxt = msgs + [{"role": "assistant", "content": llm.history[-1]["content"]},
                  {"role": "user", "content": "And the colours?"}]
    r_same = llm.ask(messages=nxt, max_tokens=5)
    st2 = llm.stats()
    end = llm._img[0][1]
    assert st2["cached_tokens"] > end and v.runs == runs, (st2["cached_tokens"], end, v.runs)
    # Another picture of the same size where the first was: not taken up past it.
    other = [dict(m) for m in nxt]
    other[1] = {"role": "user", "content": [
        {"type": "text", "text": "What is in"},
        {"type": "image_url", "image_url": {"url": data_url(img2)}},
        {"type": "text", "text": "this picture?"}]}
    r_other = llm.ask(messages=other, max_tokens=5)
    st3 = llm.stats()
    start = llm._img[0][0]
    assert st3["cached_tokens"] <= start and v.runs == runs + 1, (st3["cached_tokens"], start)
    # Both replies are a fresh engine's.
    for m, r in ((nxt, r_same), (other, r_other)):
        fresh = toy_llm("slice", max_tokens=5)
        with_vision(fresh, tower)
        assert fresh.ask(messages=m, max_tokens=5) == r, m[-1]
    print(f"  6. kept conversation: same image taken up at {st2['cached_tokens']} "
          f"(image ends {end}), another image at {st3['cached_tokens']} (starts "
          f"{start}); replies a fresh engine's")

    # `message` goes on from an image turn's cache: its positions keep the delta.
    solo = toy_llm("slice", max_tokens=3)
    with_vision(solo, tower)
    solo.ask(messages=msgs, max_tokens=3)
    lay = solo._vision.box.layout
    solo.ask(message="And the colours?", max_tokens=3)
    assert solo._vision.box.layout is lay and lay.delta < 0
    solo.reset()
    assert solo._vision.box.layout is None

    # Text only: hooks installed, still a plain engine's tokens and caches.
    plain = toy_llm("slice", max_tokens=5)
    plain.reuse = llm.reuse
    text = [{"role": "system", "content": "You look at pictures."},
            {"role": "user", "content": "Name three colours."}]
    for turn in range(2):
        a, b = llm.ask(messages=text, max_tokens=5), plain.ask(messages=text, max_tokens=5)
        assert a == b and llm._seen == plain._seen, turn
        for (k1, v1), (k2, v2) in zip(kv(llm._cache), kv(plain._cache)):
            assert mx.array_equal(k1, k2) and mx.array_equal(v1, v2)
        for x, y in zip(gdn(llm._cache), gdn(plain._cache)):
            assert mx.array_equal(x, y)
        assert "images" not in llm.stats()
        text = text + [{"role": "assistant", "content": a},
                       {"role": "user", "content": "One more."}]
    print("     text-only turns with the hooks installed: tokens and caches "
          "bit-identical to a plain engine")
    return front


def check_server():
    import server
    url = "data:image/png;base64," + base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\0" * 16).decode()
    msgs = [{"role": "user", "content": [{"type": "text", "text": "hi"},
                                          {"type": "image_url", "image_url": {"url": url}}]}]
    try:
        server.clean_messages(msgs)
        raise AssertionError("an image reached a model without vision")
    except ValueError as e:
        assert "no vision part" in str(e), e
    kept = server.clean_messages(msgs, vision=True)
    assert kept[0]["content"] == [{"type": "text", "text": "hi"},
                                  {"type": "image_url", "image_url": {"url": url}}]
    text = [{"role": "user", "content": [{"type": "text", "text": "a"},
                                          {"type": "text", "text": "b"}]}]
    assert server.clean_messages(text, vision=True) == [{"role": "user", "content": "a\nb"}]
    print("  7. server: image parts kept only for a model that takes them, refused otherwise")


def main():
    if not support.has_tokenizer():
        print("SKIPPED (no tokenizer: set MOE_TOY_TOKENIZER)")
        return
    gen = sys.modules["mlx_lm.generate"]
    mx.set_default_device(mx.cpu)
    gen.generation_stream = mx.new_thread_local_stream(mx.cpu)
    gen.wired_limit = lambda *a, **kw: contextlib.nullcontext()
    # The CPU's compiled kernels need a working C++ toolchain; uncompiled
    # they compute the same functions.
    mx.disable_compile()
    pcache.FRONTS = Path(tempfile.mkdtemp(prefix="moe-vision-test-"))
    try:
        check_pixels()
        check_positions()
        check_mrope()
        tower = check_tower()
        check_engine(tower)
        check_server()
    finally:
        shutil.rmtree(pcache.FRONTS, ignore_errors=True)
    print("test_vision: all checks passed")


if __name__ == "__main__":
    main()
