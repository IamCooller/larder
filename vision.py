#!/usr/bin/env python3
"""Images for the Qwen3.5 / Qwen3.6 vision-language checkpoints (model_type
qwen3_5 and qwen3_5_moe, dense and MoE alike).

The checkpoint's own vision tower turns an image into one embedding per image
token; those take the place of the `<|image_pad|>` ids the chat template
writes (one per image, widened here to the image's token count); and the
full-attention layers rotate by the 3D positions (interleaved M-RoPE) the model
was trained with, text and image alike. What StreamingLLM needs from here:

  attach(path, model)     the image path of a loaded engine, or None and why
  images_of(messages)     the image URLs of an OpenAI messages list, in order
  Vision.layout(urls, ids)  the ids with every image's tokens, their
                          positions, and the cache key of each image
  Vision.encode(layout, start)  the tower over the images at or after `start`

Nothing here runs for a request without images: the hooks are installed on
the first image, and they step aside when the turn has none (box.layout None),
so a text-only turn calls the model exactly as before. The tower is read from
the checkpoint for each request that needs it and freed after (0.89 GB of
bf16 on the 35B), so it is never resident beside the experts.

Positions follow transformers' Qwen3_5MoeModel.get_rope_index: a text token
takes the next position in all three rows; an image of h x w merged patches
starting at position p takes (p, p + row, p + col), and the text after it
starts at p + max(h, w). The pixels follow Qwen2VLImageProcessorPil (smart
resize to a multiple of patch x merge pixels, bicubic, (x / 255 - mean) / std,
then patches in merge-block order).

The tower (_Tower and the pieces under it) is ported from mlx-vlm
(https://github.com/Blaizzy/mlx-vlm, mlx_vlm/models/qwen3_vl/vision.py),
whose license follows:

    MIT License

    Copyright (c) 2025 Prince Canuma

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the
    "Software"), to deal in the Software without restriction, including
    without limitation the rights to use, copy, modify, merge, publish,
    distribute, sublicense, and/or sell copies of the Software, and to permit
    persons to whom the Software is furnished to do so, subject to the
    following conditions:

    The above copyright notice and this permission notice shall be included
    in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
    OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
    NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
    DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
    OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
    USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import base64
import binascii
import hashlib
import importlib.util
import io
import json
import math
import struct
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# The image processor of this family when a pack carries no
# preprocessor_config.json (Qwen/Qwen3.6-35B-A3B's own file).
PREPROCESS = {"size": {"shortest_edge": 65536, "longest_edge": 16777216},
              "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5]}
TOWER_TYPES = ("qwen3_5", "qwen3_5_moe")
# The tower's transient bytes a patch until a run has measured them (Vision.rate), from one block's 77 KB at 3800 patches.
PRIOR_PATCH_BYTES = 80e3
# Patches one piece of a block's per-patch work takes, so a big image holds ~20 KB a patch plus a piece instead of ~77 KB
# (known limit: the piece is fixed, so size it from the room if a tower runs beside a tight budget).
ROWS = 4096


# ------------------------------------------------------------- the messages
def images_of(messages):
    """Every image URL in `messages` in the template's order, raising on a part the engine cannot read."""
    out = []
    for m in messages or ():
        content = m.get("content") if isinstance(m, dict) else None
        if not isinstance(content, list):
            continue
        for p in content:
            if not isinstance(p, dict):
                continue
            kind = p.get("type")
            if kind == "image_url" or "image_url" in p:
                u = p.get("image_url")
                u = u.get("url") if isinstance(u, dict) else u
                if not isinstance(u, str):
                    raise ValueError("an image_url part without a url")
                out.append(u)
            elif kind in ("image", "video", "video_url") or "image" in p or "video" in p:
                raise ValueError(f"content part {kind!r}: only image_url parts "
                                 "with a data: URL are read")
    return out


def _decode(url):
    """(digest, RGB PIL image) of a data: URL: EXIF orientation applied (as
    transformers' load_image), alpha over white (as vLLM; transformers drops
    the alpha, which turns a transparent screenshot's background black)."""
    if not url.startswith("data:") or "," not in url:
        raise ValueError("an image must be a data: URL (base64); URLs are not fetched")
    head, data = url.split(",", 1)
    try:
        raw = (base64.b64decode(data, validate=False) if head.endswith(";base64")
               else data.encode())
    except (binascii.Error, ValueError) as e:
        raise ValueError(f"image data URL: {e}") from None
    from PIL import Image, ImageOps
    try:
        img = Image.open(io.BytesIO(raw))
        img = ImageOps.exif_transpose(img)
    except Exception as e:      # PIL raises several unrelated types on bad bytes
        raise ValueError(f"unreadable image ({type(e).__name__}: {e})") from None
    if img.mode != "RGB":
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255))
        img = Image.alpha_composite(bg, rgba).convert("RGB")
    return hashlib.sha256(raw).hexdigest(), img


# --------------------------------------------------------------- the pixels
def smart_resize(h, w, factor, min_px, max_px):
    """transformers' qwen2_vl smart_resize: both sides a multiple of `factor`,
    the area within [min_px, max_px], the aspect ratio kept."""
    if max(h, w) / min(h, w) > 200:
        raise ValueError(f"image aspect ratio {max(h, w) / min(h, w):.0f} is over 200")
    hb, wb = round(h / factor) * factor, round(w / factor) * factor
    if hb * wb > max_px:
        beta = math.sqrt(h * w / max_px)
        hb = max(factor, math.floor(h / beta / factor) * factor)
        wb = max(factor, math.floor(w / beta / factor) * factor)
    elif hb * wb < min_px:
        beta = math.sqrt(min_px / (h * w))
        hb = math.ceil(h * beta / factor) * factor
        wb = math.ceil(w * beta / factor) * factor
    return hb, wb


def patches(img, vc, pre):
    """(pixel_values [N, C*T*P*P] float32, (1, gh, gw)) of an RGB PIL image,
    as Qwen2VLImageProcessorPil makes them."""
    from PIL import Image
    p, m, tp = vc["patch_size"], vc["spatial_merge_size"], vc["temporal_patch_size"]
    size = pre["size"]
    h, w = smart_resize(img.height, img.width, p * m,
                        size.get("shortest_edge", size.get("min_pixels")),
                        size.get("longest_edge", size.get("max_pixels")))
    x = np.asarray(img.resize((w, h), resample=Image.BICUBIC), dtype=np.float32)
    x = x * np.float32(1 / 255)
    x = (x - np.array(pre["image_mean"], np.float32)) / np.array(pre["image_std"], np.float32)
    x = np.repeat(x.transpose(2, 0, 1)[None], tp, axis=0)          # T, C, H, W
    gh, gw = h // p, w // p
    x = x.reshape(1, tp, 3, gh // m, m, p, gw // m, m, p)
    x = x.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    return np.ascontiguousarray(x.reshape(gh * gw, 3 * tp * p * p)), (1, gh, gw)


def grid_of(img, vc, pre):
    """The (1, gh, gw) patch grid `patches` would make, without the pixels."""
    p, m = vc["patch_size"], vc["spatial_merge_size"]
    size = pre["size"]
    h, w = smart_resize(img.height, img.width, p * m,
                        size.get("shortest_edge", size.get("min_pixels")),
                        size.get("longest_edge", size.get("max_pixels")))
    return 1, h // p, w // p


# ------------------------------------------------------------ the positions
def positions(n, spans, merge):
    """(3 x n positions, delta): text one step a token in all three rows, an image
    (start, end, (t, gh, gw)) on its merged grid from where it starts, the text after
    it from that start plus max(rows, cols), and delta what a later token adds to its index."""
    P = np.empty((3, n), np.int64)
    cur = i = 0
    for s, e, (t, gh, gw) in spans:
        lh, lw = gh // merge, gw // merge
        assert t == 1 and e - s == lh * lw, (t, gh, gw, s, e)
        P[:, i:s] = np.arange(cur, cur + s - i)
        cur += s - i
        P[0, s:e] = cur
        P[1, s:e] = cur + np.repeat(np.arange(lh), lw)
        P[2, s:e] = cur + np.tile(np.arange(lw), lh)
        cur += max(lh, lw)
        i = e
    P[:, i:] = np.arange(cur, cur + n - i)
    return P, int(cur + n - i - n)


class Layout:
    """One request's ids with its images widened, and what the hooks read:
    positions by token index, feature rows by token index, and `spans`
    (start, end, digest), the cache key of every image."""

    def __init__(self, ids, pad, images, grids, digests, merge):
        """`ids` with each single `pad` widened to its image's token count."""
        at = [i for i, t in enumerate(ids) if t == pad]
        if len(at) != len(images):
            raise ValueError(f"{len(images)} image(s) in the messages, {len(at)} "
                             "image placeholder(s) in the prompt")
        out, spans, prev = [], [], 0
        for i, g, d in zip(at, grids, digests):
            out += ids[prev:i]
            k = g[0] * (g[1] // merge) * (g[2] // merge)
            spans.append((len(out), len(out) + k, d))
            out += [pad] * k
            prev = i + 1
        out += ids[prev:]
        self.ids, self.n, self.images = out, len(out), images
        self.spans, self.feats = tuple(spans), {}
        self.P, self.delta = positions(
            self.n, [(s, e, g) for (s, e, _), g in zip(spans, grids)], merge)

    def tokens(self):
        return sum(e - s for s, e, _ in self.spans)

    def pos(self, o, T):
        """Positions of tokens o..o+T: an int when they are text in a row
        (the offset stock RoPE takes), else a (3, T) array."""
        if not any(s < o + T and e > o for s, e, _ in self.spans):
            return int(self.P[0, o]) if o < self.n else o + self.delta
        idx = np.arange(o, o + T)
        out = np.empty((3, T), np.int64)
        inside = idx < self.n
        out[:, inside] = self.P[:, idx[inside]]
        out[:, ~inside] = idx[~inside] + self.delta
        return out

    def rows(self, o, T):
        """[(a, b, features)]: the image rows of tokens o..o+T, a and b
        relative to o; [] when there are none."""
        out = []
        for k, (s, e, _) in enumerate(self.spans):
            a, b = max(s, o), min(e, o + T)
            if a >= b:
                continue
            f = self.feats.get(k)
            if f is None:
                raise RuntimeError(f"image {k} reached the model without its features")
            out.append((a - o, b - o, f[a - s:b - s]))
        return out


# ---------------------------------------------------------------- the tower
def _rotate_half(x):
    h = x.shape[-1] // 2
    return mx.concatenate([-x[..., h:], x[..., :h]], axis=-1)


def _fused_sdpa(q, k, v, scale):
    """SDPA with the head dim padded to one MLX fuses (72 -> 80 here): the
    fallback would hold the whole score matrix (mlx-vlm's ensure_fused_sdpa)."""
    d = q.shape[-1]
    t = next((x for x in (64, 80, 128) if d <= x), d)
    if t != d:
        pad = [(0, 0)] * (q.ndim - 1) + [(0, t - d)]
        q, k, v = mx.pad(q, pad), mx.pad(k, pad), mx.pad(v, pad)
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)[..., :d]


class _Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads, self.scale = heads, (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def qkv_of(self, x, cos, sin):
        """[3, heads, rows, head_dim] of rows x: q and k rotated (in fp32,
        as transformers' apply_rotary_pos_emb_vision)."""
        q, k, v = self.qkv(x).reshape(x.shape[0], 3, self.heads, -1).transpose(1, 0, 2, 3)
        q = (q * cos + _rotate_half(q) * sin).astype(x.dtype)
        k = (k * cos + _rotate_half(k) * sin).astype(x.dtype)
        return mx.stack([q, k, v]).transpose(0, 2, 1, 3)


class _MLP(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.linear_fc1 = nn.Linear(dim, hidden, bias=True)
        self.linear_fc2 = nn.Linear(hidden, dim, bias=True)

    def __call__(self, x):
        return self.linear_fc2(nn.gelu_approx(self.linear_fc1(x)))


class _Block(nn.Module):
    def __init__(self, vc):
        super().__init__()
        d = vc["hidden_size"]
        self.norm1 = nn.LayerNorm(d, eps=1e-6)
        self.norm2 = nn.LayerNorm(d, eps=1e-6)
        self.attn = _Attention(d, vc["num_heads"])
        self.mlp = _MLP(d, vc["intermediate_size"])

    def __call__(self, x, cos, sin):
        """Evaluated here piece by piece, since one lazy graph over all blocks held 853 MB at 3800 patches and one block 294."""
        a, n = self.attn, x.shape[0]
        cut = range(0, n, ROWS)
        parts = [a.qkv_of(self.norm1(x[i:i + ROWS]), cos[i:i + ROWS], sin[i:i + ROWS])
                 for i in cut]
        for p in parts:
            mx.eval(p)
        qkv = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=2)
        del parts
        o = _fused_sdpa(qkv[0][None], qkv[1][None], qkv[2][None], a.scale)[0]
        mx.eval(o)
        del qkv
        out = []
        for i in cut:
            y = x[i:i + ROWS] + a.proj(o[:, i:i + ROWS].transpose(1, 0, 2).reshape(-1, x.shape[1]))
            y = y + self.mlp(self.norm2(y))
            mx.eval(y)
            out.append(y)
        return out[0] if len(out) == 1 else mx.concatenate(out)


class _Merger(nn.Module):
    def __init__(self, vc):
        super().__init__()
        d = vc["hidden_size"]
        self.wide = d * vc["spatial_merge_size"] ** 2
        self.norm = nn.LayerNorm(d, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.wide, self.wide)
        self.linear_fc2 = nn.Linear(self.wide, vc["out_hidden_size"])

    def __call__(self, x):
        x = self.norm(x).reshape(-1, self.wide)
        return self.linear_fc2(nn.gelu(self.linear_fc1(x)))


class _PatchEmbed(nn.Module):
    """The tower's Conv3d (kernel = stride) as one matmul a patch, its weight in MLX's conv layout [out, kT, kH, kW, in]."""

    def __init__(self, vc):
        super().__init__()
        self.p, self.t, self.c = vc["patch_size"], vc["temporal_patch_size"], vc["in_channels"]
        p, t, c = self.p, self.t, self.c
        self.weight = mx.zeros((vc["hidden_size"], t, p, p, c))
        self.bias = mx.zeros((vc["hidden_size"],))

    def __call__(self, x):
        x = x.reshape(-1, self.c, self.t, self.p, self.p).moveaxis(1, 4).reshape(x.shape[0], -1)
        return x @ self.weight.reshape(self.weight.shape[0], -1).T + self.bias


class _Tower(nn.Module):
    """Qwen3-VL's vision tower without deepstack (Qwen3.5 has none)."""

    def __init__(self, vc):
        super().__init__()
        self.merge = vc["spatial_merge_size"]
        d = vc["hidden_size"]
        self.patch_embed = _PatchEmbed(vc)
        self.pos_embed = nn.Embedding(vc["num_position_embeddings"], d)
        self.side = int(vc["num_position_embeddings"] ** 0.5)
        self.blocks = [_Block(vc) for _ in range(vc["depth"])]
        self.merger = _Merger(vc)
        hd = d // vc["num_heads"] // 2
        self.inv_freq = 1.0 / (10000.0 ** (np.arange(0, hd, 2, dtype=np.float32) / hd))

    def _rope(self, gh, gw):
        """cos, sin [N, 1, head_dim] of every patch, in merge-block order."""
        m = self.merge
        r = (np.arange(gh // m)[:, None, None, None] * m + np.arange(m)[None, None, :, None])
        c = (np.arange(gw // m)[None, :, None, None] * m + np.arange(m)[None, None, None, :])
        r = np.broadcast_to(r, (gh // m, gw // m, m, m)).reshape(-1)
        c = np.broadcast_to(c, (gh // m, gw // m, m, m)).reshape(-1)
        f = np.concatenate([np.outer(r.astype(np.float32), self.inv_freq),
                            np.outer(c.astype(np.float32), self.inv_freq)], axis=-1)
        f = mx.array(np.concatenate([f, f], axis=-1).astype(np.float32))[:, None]
        return mx.cos(f), mx.sin(f)

    def _pos(self, gh, gw):
        """The learned 48 x 48 position table, bilinear to gh x gw, in
        merge-block order (mlx-vlm's fast_pos_embed_interpolate)."""
        s, m = self.side, self.merge
        hi, wi = np.linspace(0, s - 1, gh), np.linspace(0, s - 1, gw)
        h0, w0 = hi.astype(np.int64), wi.astype(np.int64)
        h1, w1 = np.minimum(h0 + 1, s - 1), np.minimum(w0 + 1, s - 1)
        dh, dw = (hi - h0).astype(np.float32), (wi - w0).astype(np.float32)
        idx = [(a[:, None] * s + b[None, :]).reshape(-1)
               for a, b in ((h0, w0), (h0, w1), (h1, w0), (h1, w1))]
        wts = [(a[:, None] * b[None, :]).reshape(-1)
               for a, b in ((1 - dh, 1 - dw), (1 - dh, dw), (dh, 1 - dw), (dh, dw))]
        dt = self.pos_embed.weight.dtype
        e = sum(self.pos_embed(mx.array(i.astype(np.int32))) * mx.array(w)[:, None].astype(dt)
                for i, w in zip(idx, wts))
        d = e.shape[-1]
        return (e.reshape(gh // m, m, gw // m, m, d).transpose(0, 2, 1, 3, 4)
                .reshape(-1, d))

    def __call__(self, pixels, grid):
        t, gh, gw = grid
        assert t == 1, "one frame an image"
        x = self.patch_embed(pixels) + self._pos(gh, gw)
        cos, sin = self._rope(gh, gw)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        return self.merger(x)


def _header(f):
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n))


def _tower_key(k):
    """A checkpoint key of the tower, as _Tower names it; None for others."""
    for pre in ("vision_tower.", "model.visual.", "visual."):
        if k.startswith(pre):
            return k[len(pre):].replace("patch_embed.proj.", "patch_embed.")
    return None


# ------------------------------------------------------------------- hooks
class _Box:
    layout = None


class _MRoPE(nn.Module):
    """A full-attention layer's RoPE with 3D positions: transparent without images, stock at
    the offset plus the images' delta on text, and on a pass with image tokens three stock
    rotations (one per position row) picked per frequency, so text turns bit for bit as stock."""

    def __init__(self, inner, box, pick):
        super().__init__()
        self._inner, self._box = inner, box
        self.dims, self.base = inner.dims, inner.base
        self.traditional, self.scale = inner.traditional, inner.scale
        self._pick = pick

    def __call__(self, x, offset=0):
        lay = self._box.layout
        if lay is None:
            return self._inner(x, offset)
        if not isinstance(offset, int):
            return self._inner(x, offset + lay.delta)
        p = lay.pos(offset, x.shape[2])
        if isinstance(p, int):
            return self._inner(x, p)
        B, H, T, D = x.shape
        y = x.transpose(0, 2, 1, 3).reshape(B * T, H, 1, D)
        r = [self._inner(y, mx.array(np.tile(p[j], B).astype(np.int32))) for j in range(3)]
        pick = mx.array(np.pad(self._pick, (0, D - len(self._pick))))
        out = mx.where(pick == 1, r[1], mx.where(pick == 2, r[2], r[0]))
        return out.reshape(B, T, H, D).transpose(0, 2, 1, 3)


def _stack_of(model):
    lm = getattr(model, "language_model", model)
    return getattr(lm, "model", lm)


class Vision:
    """A checkpoint's image path: its config, where the tower's weights are,
    and the hooks, installed on the first image."""

    def __init__(self, path, model, cfg, files, nbytes, ropes, pick):
        self.path, self.model, self.cfg = Path(path), model, cfg
        self.vc = cfg["vision_config"]
        self.pad = cfg["image_token_id"]
        self.merge = self.vc["spatial_merge_size"]
        f = self.path / "preprocessor_config.json"
        self.pre = json.loads(f.read_text()) if f.exists() else PREPROCESS
        self.files, self.nbytes = files, nbytes
        self._ropes, self._pick = ropes, pick
        self.box = _Box()
        self.rate, self.measured = PRIOR_PATCH_BYTES, False
        self.last = {}
        self._installed = False

    def install(self):
        """Wrap, once, the layer stack's call (image rows go in as input_embeddings, which the
        stack's other wrappers read to leave such a pass to the stock path) and every full-attention RoPE."""
        if self._installed:
            return
        stack = _stack_of(self.model)
        cls = type(stack)
        if not getattr(cls, "_vision_wrapped", False):
            orig = cls.__call__

            def call(this, inputs, *args, **kwargs):
                box = this.__dict__.get("_vision_box")
                lay = box.layout if box is not None else None
                if (lay is None or inputs is None or inputs.ndim != 2
                        or kwargs.get("input_embeddings") is not None):
                    return orig(this, inputs, *args, **kwargs)
                cache = args[0] if args else kwargs.get("cache")
                o = next((c.offset for c in cache or () if hasattr(c, "offset")), None)
                parts = lay.rows(o, inputs.shape[1]) if isinstance(o, int) else []
                if not parts:
                    return orig(this, inputs, *args, **kwargs)
                h = this.embed_tokens(inputs)
                for a, b, f in parts:
                    h[:, a:b] = f.astype(h.dtype)[None]
                return orig(this, inputs, *args, **{**kwargs, "input_embeddings": h})

            cls.__call__ = call
            cls._vision_wrapped = True
        stack.__dict__["_vision_box"] = self.box
        for attn in self._ropes:
            attn.rope = _MRoPE(attn.rope, self.box, self._pick)
        self._installed = True

    def layout(self, urls, ids):
        """The request's Layout: every image decoded (for its size and its
        digest), its placeholder widened to its token count."""
        got = [_decode(u) for u in urls]
        grids = [grid_of(img, self.vc, self.pre) for _, img in got]
        return Layout(list(ids), self.pad, [img for _, img in got], grids,
                      [d for d, _ in got], self.merge)

    def _load(self):
        tower = _Tower(self.vc)
        w = {}
        for f, keys in self.files.items():
            arrs = mx.load(str(f))
            for k in keys:
                a = arrs[k]
                name = _tower_key(k)
                if name == "patch_embed.weight" and a.shape[1] == self.vc["in_channels"]:
                    a = a.transpose(0, 2, 3, 4, 1)      # torch Conv3d layout -> MLX
                w[name] = a
            del arrs
        tower.load_weights(list(w.items()))
        return tower

    def encode(self, lay, start):
        """Run the tower over every image of `lay` that ends after `start`
        (the ones a kept cache does not hold already); seconds."""
        todo = [k for k, (s, e, _) in enumerate(lay.spans) if e > start and k not in lay.feats]
        self.last = {"vision_parked_gb": 0.0, "vision_park_s": 0.0}
        if not todo:
            return 0.0
        import stream_fast
        t0 = time.perf_counter()
        px = {k: patches(lay.images[k], self.vc, self.pre) for k in todo}
        n = max(p.shape[0] for p, _ in px.values())
        # It may take what a short prompt's prefill takes unchecked (one of
        # MIN_PIECE positions, moe_stream._prefill); past that the prefill
        # budget's room decides, and the arena is parked as for a long one.
        import moe_stream
        need, room = self.nbytes + self.rate * n, stream_fast.prefill_room()
        fixed, per = stream_fast.prefill_need()
        parked, tp = 0, time.perf_counter()
        if room is not None and room < need and need > fixed + per * moe_stream.MIN_PIECE:
            parked = stream_fast.park_arenas(self.model)
        tp = time.perf_counter() - tp
        try:
            mx.reset_peak_memory()
            a0 = mx.get_active_memory()
            tower = self._load()
            for k in todo:
                p, g = px[k]
                # In the weights' dtype, as transformers' patch embedding casts.
                lay.feats[k] = tower(mx.array(p).astype(tower.patch_embed.weight.dtype), g)
                mx.eval(lay.feats[k])
            grew = mx.get_peak_memory() - a0 - self.nbytes
            self.rate = max(self.rate if self.measured else 0.0, grew / n)
            self.measured = True
            del tower
            mx.clear_cache()
        finally:
            if parked:
                tu = time.perf_counter()
                stream_fast.unpark_arenas(self.model)
                tp += time.perf_counter() - tu
            self.last = {"vision_parked_gb": round(parked / 1e9, 2), "vision_park_s": round(tp, 2)}
        return time.perf_counter() - t0


def tower_files(path):
    """{shard: [tower keys]} of a checkpoint directory, and their bytes."""
    path = Path(path)
    idx = path / "model.safetensors.index.json"
    if idx.exists():
        wm = json.loads(idx.read_text())["weight_map"]
        shards = sorted({wm[k] for k in wm if _tower_key(k)})
    else:
        shards = [f.name for f in path.glob("*.safetensors")]
    files, nbytes = {}, 0
    for s in shards:
        h = _header(path / s)
        keys = [k for k in h if k != "__metadata__" and _tower_key(k)]
        if keys:
            files[path / s] = keys
            nbytes += sum(h[k]["data_offsets"][1] - h[k]["data_offsets"][0] for k in keys)
    return files, nbytes


def attach(path, model, budget=False, layers=None):
    """(Vision, why) for a loaded engine, or (None, why not)."""
    path = Path(path)
    try:
        cfg = json.loads((path / "config.json").read_text())
    except (OSError, ValueError):
        return None, "no config.json"
    vc = cfg.get("vision_config")
    if not vc:
        return None, "no vision_config"
    if vc.get("model_type") not in TOWER_TYPES or vc.get("deepstack_visual_indexes"):
        return None, f"vision tower {vc.get('model_type')!r} is not one this engine runs"
    if budget:
        return None, "--mem-peak-gb: the peak budget does not count the tower yet"
    if layers:
        return None, "layer-range mode"
    if importlib.util.find_spec("PIL") is None:
        return None, "Pillow is not installed (pip install Pillow)"
    files, nbytes = tower_files(path)
    if not files:
        return None, "the pack has no vision tower weights"
    tc = cfg.get("text_config", cfg)
    rp = tc.get("rope_parameters") or tc.get("rope_scaling") or {}
    section = rp.get("mrope_section")
    if not section or not rp.get("mrope_interleaved", False):
        return None, "the text model has no interleaved M-RoPE"
    stack = _stack_of(model)
    ropes = [l.self_attn for l in getattr(stack, "layers", ())
             if getattr(l, "self_attn", None) is not None]
    half = sum(section)
    if (not ropes or not hasattr(stack, "embed_tokens")
            or not all(type(a.rope) is nn.RoPE and a.rope.dims == 2 * half
                       and not a.rope.traditional for a in ropes)):
        return None, "the text model's attention is not the one this hook knows"
    # Which position row each rotated dim takes (transformers'
    # apply_interleaved_mrope): T, then H every third from 1, W from 2.
    sel = np.zeros(half, np.int32)
    sel[1:3 * section[1]:3] = 1
    sel[2:3 * section[2]:3] = 2
    v = Vision(path, model, cfg, files, nbytes, ropes, np.concatenate([sel, sel]))
    return v, f"tower {nbytes / 1e9:.2f} GB, read per request"
