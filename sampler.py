"""Sampling (temperature, top-k, top-p, min-p) and the exact speculative-sampling rule of Leviathan
et al. 2023 (arXiv 2211.17192) and Chen et al. 2023 (arXiv 2302.01318) for the MTP round.
Greedy, temperature 0 or no Sampling at all, never comes to Sampling: each caller keeps its own argmax.
ThinkBudget wraps either as the sampler of a turn with a thinking budget.
"""
import math
from dataclasses import dataclass

import mlx.core as mx


@dataclass(frozen=True)
class Sampling:
    """One request's settings. An instance is also mlx_lm's `sampler`: log-probs in, ids out."""
    temperature: float
    top_k: int = 0
    top_p: float = 1.0
    min_p: float = 0.0
    seed: int = None

    @classmethod
    def from_request(cls, body):
        """An OpenAI-style body's fields (top_k and min_p as vLLM and llama.cpp name them).

        None, i.e. greedy, when temperature is absent or 0; ValueError on a value out of range.
        """
        t = body.get("temperature")
        if t is None or float(t) == 0:
            return None
        seed = body.get("seed")
        s = cls(float(t), int(body.get("top_k") or 0), float(body.get("top_p") or 1.0),
                float(body.get("min_p") or 0.0), None if seed is None else int(seed))
        if not (s.temperature > 0 and s.top_k >= 0 and 0 < s.top_p <= 1 and 0 <= s.min_p < 1):
            raise ValueError(f"sampling fields out of range: {s}")
        return s

    def process(self, lp):
        """Log-probs after temperature, then top-k, top-p and min-p (Hugging Face's order), float32,
        where a tie at the top-k cut keeps every tied token (exact k needs an argpartition scatter)."""
        x = lp.astype(mx.float32) * (1 / self.temperature)
        top = x
        if 0 < self.top_k < x.shape[-1]:
            top = mx.topk(x, self.top_k, axis=-1)
            x = mx.where(x < top.min(axis=-1, keepdims=True), -mx.inf, x)
        if self.top_p < 1:
            srt = mx.sort(top, axis=-1)[..., ::-1]
            p = mx.softmax(srt, axis=-1)
            # A token stays while the mass above it is short of top_p: the smallest such set.
            cut = mx.where(mx.cumsum(p, axis=-1) - p < self.top_p, srt, mx.inf)
            x = mx.where(x < cut.min(axis=-1, keepdims=True), -mx.inf, x)
        if self.min_p > 0:
            x = mx.where(x < x.max(axis=-1, keepdims=True) + math.log(self.min_p), -mx.inf, x)
        return x - mx.logsumexp(x, axis=-1, keepdims=True)

    def logprobs(self, logits):
        """process() of raw logits, normalized first the way mlx_lm.generate_step hands its sampler
        log-probs, so the speculative round and plain decode see the same numbers."""
        return self.process(logits - mx.logsumexp(logits, axis=-1, keepdims=True))

    def __call__(self, lp):
        return mx.random.categorical(self.process(lp))


def verify(p, q, drafts):
    """One round's verdict on the GPU: p [n+1, V] target and q [n, V] draft log-probs, drafts [n] ~ q
    (a one-hot q row for a certain draft). Draft j stands with probability min(1, p_j(x)/q_j(x)), the
    first that does not becomes fix[j] ~ normalize(max(0, p_j - q_j)), and bonus ~ p_n ends a round
    whose drafts all stood; returns (stands, fix, bonus)."""
    n = drafts.shape[0]
    bonus = mx.random.categorical(p[n])
    if n == 0:
        return mx.zeros((0,), mx.bool_), mx.zeros((0,), mx.uint32), bonus
    d = drafts.astype(mx.int32)[:, None]
    px = mx.take_along_axis(p[:n], d, axis=-1)[:, 0]
    qx = mx.take_along_axis(q, d, axis=-1)[:, 0]
    stands = mx.log(mx.random.uniform(shape=(n,))) < px - qx
    r = mx.maximum(mx.exp(p[:n]) - mx.exp(q), 0.0)
    # p == q leaves no residual, but then no draft is ever refused: any row will do.
    r = mx.where(r.sum(axis=-1, keepdims=True) > 0, mx.log(r), p[:n])
    return stands, mx.random.categorical(r), bonus


# Qwen's early-stopping text for a thinking budget, verbatim from the Qwen3 docs (qwen.readthedocs.io, Quickstart, "Thinking Budget").
EARLY_STOP = ("\n\nConsidering the limited time by the user, I have to give the solution based on "
              "the thinking directly now.\n</think>\n\n")


class ThinkBudget:
    """mlx_lm's `sampler` for a reply that starts inside a think block: `inner`'s ids (argmax when None)
    until `budget` tokens, then `forced` one id a call unless the reply wrote `close` itself, then `inner`'s again.
    One call is one reply token, which holds for plain decode only."""

    def __init__(self, inner, budget, forced, close):
        self.inner, self.budget, self.forced, self.close = inner, budget, list(forced), close
        self.seed = getattr(inner, "seed", None)
        self.k, self.recent, self.closed, self.cut = 0, [], False, False

    def __call__(self, lp):
        k = self.k
        self.k += 1
        if k == self.budget and not self.closed:
            # The one wait a turn: the newest thinking ids, the last maybe still being decoded.
            self.closed = self.close in [y.item() for y in self.recent]
            self.cut = not self.closed
        if self.cut and k - self.budget < len(self.forced):
            return mx.full(lp.shape[:-1], self.forced[k - self.budget], mx.uint32)
        y = self.inner(lp) if self.inner is not None else mx.argmax(lp, axis=-1)
        if k < self.budget and not self.closed:
            # Read two calls late, once the decode loop has read it too, so it costs no wait.
            if len(self.recent) == 2:
                self.closed = self.recent.pop(0).item() == self.close
            self.recent.append(y)
        return y
