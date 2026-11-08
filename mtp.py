"""The Multi-Token-Prediction head of Qwen3.6, as a loadable module.

Structure recovered from the official checkpoint's tensor shapes:

    h' = fc( concat( pre_fc_norm_embedding(embed(x_t+1)),
                     pre_fc_norm_hidden(h_t) ) )         fc: [2048, 4096]
         -> one decoder layer: full attention + MoE(256, top-8) + shared expert
         -> norm -> the MAIN model's lm_head

Nothing of the model is reimplemented: the decoder layer is mlx_lm's own
DecoderLayer with a layer_idx that makes it the full-attention variant, so the
attention and the MoE block are the classes the target model already uses.

Which half of the fused gate_up_proj is the gate, and which order fc expects its
two inputs in, are conventions that shapes cannot settle, so both are flags and a wrong
choice shows up as acceptance near zero rather than as a crash.
"""
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.qwen3_5 import DecoderLayer, TextModelArgs
from mlx_lm.models.qwen3_next import ModelArgs as NextArgs
from mlx_lm.models.qwen3_next import Qwen3NextDecoderLayer

import defaults

# The packed head a bare --mtp means, from local_defaults.py; None when this checkout has none.
DEFAULT_MTP = defaults.MTP


def head_args(cfg):
    """Model args for the head from a checkpoint's config.json dict.

    Qwen3.6 is a multimodal wrapper (text_config inside); Qwen3-Next-80B is a
    flat qwen3_next config. Same head shape in both checkpoints -- fc over
    concat(embed, hidden), one full-attention decoder layer, the model's
    lm_head -- only the layer class differs.
    """
    if "text_config" in cfg:
        return TextModelArgs.from_dict(cfg["text_config"])
    if cfg.get("model_type") == "qwen3_next":
        return NextArgs.from_dict(cfg)
    raise ValueError(f"no MTP head recipe for model_type {cfg.get('model_type')!r}")


def _layer_class(args):
    return Qwen3NextDecoderLayer if isinstance(args, NextArgs) else DecoderLayer


class MTPHead(nn.Module):
    def __init__(self, args: TextModelArgs, hidden_first=False):
        super().__init__()
        H, eps = args.hidden_size, args.rms_norm_eps
        self.pre_fc_norm_embedding = nn.RMSNorm(H, eps=eps)
        self.pre_fc_norm_hidden = nn.RMSNorm(H, eps=eps)
        self.fc = nn.Linear(2 * H, H, bias=False)
        # layer_idx = interval - 1 makes is_linear False, i.e. full attention,
        # which is what the checkpoint's self_attn.* keys say this layer is.
        self.layers = [_layer_class(args)(args, args.full_attention_interval - 1)]
        self.norm = nn.RMSNorm(H, eps=eps)
        self._hidden_first = hidden_first

    def __call__(self, hidden, embed, cache=None, mask=None):
        """hidden: (B, S, H) states from the target. embed: (B, S, H) embeddings
        of the tokens the target produced at those positions."""
        e = self.pre_fc_norm_embedding(embed)
        h = self.pre_fc_norm_hidden(hidden)
        pair = (h, e) if self._hidden_first else (e, h)
        x = self.fc(mx.concatenate(pair, axis=-1))
        if mask is None:
            mask = create_attention_mask(x, cache[0] if cache else None)
        x = self.layers[0](x, mask=mask, cache=cache[0] if cache else None)
        return self.norm(x)


def load_mtp(path, args, hidden_first=False):
    """Build the head and load the 4-bit pack written by tools/mtp_pack.py."""
    path = Path(path)
    f = path if path.is_file() else path / "mtp-4bit.safetensors"
    meta = json.loads((f.parent / (f.stem + "-quant.json")).read_text())
    bits = {k[len("mtp."):]: v for k, v in meta.items()}

    head = MTPHead(args, hidden_first=hidden_first)
    nn.quantize(head, class_predicate=lambda p, m: bits.get(p, False))

    w = mx.load(str(f))
    head.load_weights([(k[len("mtp."):], v) for k, v in w.items()])
    mx.eval(head.parameters())
    return head


def demo():
    """Self-check: the head builds, loads, and produces finite logits."""
    import mlx_lm
    if not (defaults.MODEL and defaults.MTP):
        print("demo skipped: local_defaults.py names no checkpoint and head")
        return
    src = Path(defaults.MODEL)
    cfg = json.loads((src / "config.json").read_text())
    args = TextModelArgs.from_dict(cfg["text_config"])
    head = load_mtp(defaults.MTP, args)
    B, S, H = 1, 3, args.hidden_size
    x = mx.random.normal((B, S, H)).astype(mx.float16)
    e = mx.random.normal((B, S, H)).astype(mx.float16)
    out = head(x, e)
    mx.eval(out)
    assert out.shape == (B, S, H), out.shape
    assert bool(mx.all(mx.isfinite(out))), "non-finite output"
    print(f"ok: {out.shape}, finite, "
          f"std {float(mx.std(out.astype(mx.float32))):.3f}")


if __name__ == "__main__":
    demo()
