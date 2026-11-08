"""MTP head of Flash-Next (qwen4_exp), loaded beside the main model to measure how often it guesses
the next tokens. Off by default: nothing here is imported without `--mtp-probe DIR` (chat.py), and
the main model's weights, caches and call path are not touched (a one-line tap on the final
hyper-connection mixer records its input, nothing more).

The head, as far as its tensors say (no reference code ships with the checkpoint; the
qwen4_exp modelling files ignore `mtp.*`):

    hidden  = the main model's final hyper-connection stream [.., 4 * H], BEFORE its mixer
              (pre_fc_norm_hidden has 4 * H weights, one statistic per stream)
    h_s     = fc_hidden(norm(hidden)_s) + fc_embedding(pre_fc_norm_embedding(embed(next token)))
              for each of the 4 streams s  (fc_* are [H, H])
    h       -> one full-attention decoder layer, 512-expert MoE, own hyper-connections, own KV cache
            -> own hyper_connection_mixer -> the main model's lm_head

`next token` is the token the main model emitted at the position that produced `hidden`; the head
then predicts the token after it (Qwen3-Next convention). Depth k > 1 feeds the head its own
previous stream and its own previous guess.

gate_up_proj halves: gate = first, up = second (qwen4_exp.sanitize in mlx_lm). Norm weights are
zero-centred in the checkpoint (y = x_hat * (1 + w)), folded here like the main pack's.

The head's experts live in memory (1.6 GB at 4 bit), not streamed.
"""
import glob
import os
import time

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.qwen4_exp import DecoderLayer, GatedResidual, RMSNorm, _AttnCache

_FOLD = ("hc_norm.weight", "q_norm.weight", "k_norm.weight", "q_layernorm.weight",
         "k_layernorm.weight", "pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight")


class MTPHead(nn.Module):
    def __init__(self, targs):
        super().__init__()
        H, eps, hc = targs.hidden_size, targs.rms_norm_eps, targs.hc_count
        self.hc, self.H = hc, H
        self.pre_fc_norm_embedding = RMSNorm(H, eps=eps)
        self.pre_fc_norm_hidden = RMSNorm(hc * H, group_size=H, eps=eps)
        self.fc_embedding = nn.Linear(H, H, bias=False)
        self.fc_hidden = nn.Linear(H, H, bias=False)
        # the layer index only picks the layer type: 3 is full attention in every layer_types
        self.layers = [DecoderLayer(targs, targs.full_attention_interval - 1)]
        self.hyper_connection_mixer = GatedResidual(targs, use_combine=False)

    def stream(self, stack, hidden, ids, cache):
        """hidden [B,S,hc*H] (main model's final stream, or this head's previous output), ids [B,S]
        the tokens that follow those positions. Returns this head's output stream [B,S,hc*H]."""
        B, S, _ = hidden.shape
        e = self.fc_embedding(self.pre_fc_norm_embedding(stack.embed_tokens(ids)))
        h = self.fc_hidden(self.pre_fc_norm_hidden(hidden).reshape(B, S, self.hc, self.H))
        h = (h + e[:, :, None, :]).reshape(B, S, self.hc * self.H)
        mask = create_attention_mask(h, cache)
        return self.layers[0](h, stack.rope, mask, None, cache, cache.indexer, ids, None)

    def logits(self, lm_head, out):
        return lm_head(self.hyper_connection_mixer(out))


def load(llm, path):
    """Build the head for llm's config and load the 4-bit files of `path` (one safetensors per
    tensor, q4 layout)."""
    w = {}
    for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        w.update(mx.load(f))
    w = {k[len("mtp."):]: v for k, v in w.items()}
    head = MTPHead(llm.model.args.text)

    def pred(p, m):
        if not hasattr(m, "to_quantized") or f"{p}.scales" not in w:
            return False
        inner = m.weight.shape[-1]
        return {"bits": w[f"{p}.weight"].shape[-1] * 32 // inner,
                "group_size": inner // w[f"{p}.scales"].shape[-1]}

    nn.quantize(head, class_predicate=pred)
    w = {k: (1.0 + v.astype(mx.float32)).astype(v.dtype) if k.endswith(_FOLD) else v
         for k, v in w.items()}
    head.load_weights(list(w.items()), strict=True)
    mx.eval(head.parameters())
    return head


class _Tap:
    """Records the input of the main model's final mixer (the last layer's stream) on the stack
    instance; the class wrapper is a dict lookup for every other GatedResidual."""


def _install_tap(stack):
    mixer = stack.hyper_connection_mixer
    cls = type(mixer)
    if not getattr(cls, "_mtp_tap", False):
        orig = cls.__call__

        def call(self, x):
            if self.__dict__.get("_tap") is not None:
                self.__dict__["_tap"].x = x
            return orig(self, x)

        cls.__call__ = call
        cls._mtp_tap = True
    tap = _Tap()
    tap.x = None
    mixer.__dict__["_tap"] = tap
    return tap


class Probe:
    """llm.mtp_probe. generate() is a plain greedy decode of the staged engine (stage.Stage over
    all layers) that, with mtp=True, also runs the head after every emitted token and records its
    depth-1..D guesses."""

    def __init__(self, llm, path, depth=3):
        try:
            import stage
        except ImportError:
            raise RuntimeError("--mtp-probe needs stage.py, which this install does not include") from None
        self.llm, self.depth = llm, depth
        self.stg = stage.Stage(llm)
        self.stack = llm.model.model
        self.tap = _install_tap(self.stack)
        t = time.time()
        self.head = load(llm, path)
        self.load_s = time.time() - t

    def _guesses(self, hidden, ids, cache, n_extra):
        """Run the head on `hidden`/`ids` (committing len(ids) positions to `cache`), then chain
        n_extra more guesses from its last position (those cache entries are trimmed again).
        Returns [g1, ..., g_{1+n_extra}] as lazy [1] int arrays."""
        out = self.head.stream(self.stack, hidden, ids, cache)
        last = out[:, -1:]
        g = mx.argmax(self.head.logits(self.llm.model.lm_head, last)[:, 0], axis=-1)
        gs = [g]
        for _ in range(n_extra):
            last = self.head.stream(self.stack, last, g[:, None].astype(mx.int32), cache)
            g = mx.argmax(self.head.logits(self.llm.model.lm_head, last)[:, 0], axis=-1)
            gs.append(g)
        if n_extra:
            cache.trim(n_extra)
        return gs

    def generate(self, ids, max_tokens, eos, mtp=True, guard=None):
        """ids: prompt token ids (list). Returns dict with tokens, margins, per-token times, and
        (mtp) guesses[i] = [g1..gD] made right after token i was emitted, i.e. g_k guesses
        tokens[i + k]."""
        mx_ = mx
        stg = self.stg
        stg.reset()
        cache = _AttnCache()
        D = self.depth
        x = mx_.array([ids], dtype=mx_.int32)
        out, margins, times, guesses = [], [], [], []
        t0 = time.perf_counter()
        lg = stg.forward(x)
        for i in range(max_tokens):
            top = mx_.topk(lg, 2)
            tok = mx_.argmax(lg)
            g = None
            if mtp:
                hid = self.tap.x
                if i == 0:      # the prompt: head positions 0..n-1 pair hidden t with token t+1
                    nxt = mx_.concatenate([x[:, 1:], tok.reshape(1, 1).astype(mx_.int32)], axis=1)
                else:
                    nxt = tok.reshape(1, 1).astype(mx_.int32)
                g = self._guesses(hid, nxt, cache, D - 1)
                mx_.eval(tok, top, *g)
            else:
                mx_.eval(tok, top)
            a, b = top.tolist()
            t = int(tok.item())
            times.append(time.perf_counter() - t0)
            out.append(t)
            margins.append(abs(a - b))
            if g is not None:
                guesses.append([int(v.item()) for v in g])
            if t in eos:
                break
            if guard is not None and i % 25 == 24:
                guard()          # a memory guard of the caller; it raises to stop
            x = mx_.array([[t]], dtype=mx_.int32)
            lg = stg.forward(x)
        return {"tokens": out, "margins": margins, "times": times, "guesses": guesses}


def attach(llm, path, depth=3):
    llm.mtp_probe = Probe(llm, path, depth)
    return llm.mtp_probe
