#!/usr/bin/env python3
"""The cache and text pieces the long-term memory shares with the engine and the probes."""

import re
import time
from collections import Counter

import mlx.core as mx
import numpy as np

from mlx_lm.models.cache import (ArraysCache, KVCache, QuantizedKVCache,
                                 make_prompt_cache)

# The acknowledgement that closes a kept session.
ACK = "Понял."


def cache_arrays(c):
    """Every live array of one cache object, in a fixed order."""
    if isinstance(c, ArraysCache):
        return [a for a in c.cache if a is not None]
    ks, vs = getattr(c, "keys", None), getattr(c, "values", None)
    return [a for a in (ks, vs) if a is not None]


def kv_arrays(c):
    """A KVCache's keys and values up to its offset, read the same way on every mlx-lm version."""
    if c.offset == c.keys.shape[2]:
        return c.keys, c.values
    return c.keys[..., :c.offset, :], c.values[..., :c.offset, :]


def set_kv(c, keys, values):
    """Put keys and values into a KVCache with its offset at their length, whatever its state setter takes."""
    c.keys, c.values, c.offset = keys, values, keys.shape[2]


def clone_one(c):
    """An independent copy of one cache object, because a cache writes into its own arrays."""
    if isinstance(c, ArraysCache):
        d = ArraysCache(size=len(c.cache))
        d.cache = [None if a is None else mx.array(a) for a in c.cache]
        return d
    d = KVCache()
    if getattr(c, "keys", None) is None:
        return d
    ks, vs = kv_arrays(c)
    set_kv(d, mx.array(ks), mx.array(vs))
    return d


def clone_cache(cache):
    """An independent copy of a whole cache list."""
    return [clone_one(c) for c in cache]


def clone_cache_keeping_gap(cache):
    """clone_cache that keeps each KV cache's gap, so the next question lands at the position the conversation reached."""
    out = []
    for c in cache:
        if isinstance(c, ArraysCache):
            out.append(clone_one(c))
            continue
        d = GapKVCache(getattr(c, "gap", 0))
        if getattr(c, "keys", None) is not None:
            k, v = real_state(c)
            set_kv(d, mx.array(k), mx.array(v))
            d.offset += d.gap
        out.append(d)
    return out


class GapKVCache(KVCache):
    """A cache that reports `gap` extra positions to RoPE while writing at the real offset."""

    def __init__(self, gap=0):
        super().__init__()
        self.gap = gap

    def update_and_fetch(self, keys, values):
        self.offset -= self.gap
        out = super().update_and_fetch(keys, values)
        self.offset += self.gap
        return out

    def to_quantized(self, group_size=64, bits=4):
        """The same cache in 4 bits, with the gap kept."""
        q = GapQuantizedKVCache(self.gap, group_size=group_size, bits=bits)
        q.offset = self.offset
        if self.keys is not None:
            q.keys = mx.quantize(self.keys, group_size=group_size, bits=bits)
            q.values = mx.quantize(self.values, group_size=group_size, bits=bits)
        return q


class GapQuantizedKVCache(QuantizedKVCache):
    """The 4-bit twin of GapKVCache, with the same two lines."""

    def __init__(self, gap, group_size=64, bits=4):
        super().__init__(group_size=group_size, bits=bits)
        self.gap = gap

    def update_and_fetch(self, keys, values):
        self.offset -= self.gap
        out = super().update_and_fetch(keys, values)
        self.offset += self.gap
        return out


def turn_ids(llm, text, first):
    """The tokens of one user turn, built the way the engine builds them."""
    t = llm._tpl
    if t is None:
        raise SystemExit("the engine could not learn this template's "
                         "separators; the memory cannot build turns without "
                         "them")
    import tools as toolfmt
    return toolfmt.encode(llm.tokenizer, (t["head"] if first else t["between"])
                          + toolfmt.escape(text, llm.tokenizer) + t["tail"],
                          add_special_tokens=False)


def zero_state(cache):
    """Set every recurrent state array of a cache to zeros."""
    for c in cache:
        if isinstance(c, ArraysCache):
            c.cache = [None if a is None else mx.zeros_like(a) for a in c.cache]


def prefill_chunked(model, ids, chunk=512, gap=0, cache=None, log=None,
                    zero=True):
    """Prefill `ids` through the model a chunk at a time into one cache."""
    if cache is None:
        cache = make_prompt_cache(model)
        if gap:
            cache = [c if isinstance(c, ArraysCache) else GapKVCache(gap)
                     for c in cache]
            for c in cache:
                if isinstance(c, GapKVCache):
                    c.offset = gap
    t0 = time.perf_counter()
    for i in range(0, len(ids), chunk):
        part = mx.array(np.array(ids[i:i + chunk], dtype=np.int32)[None])
        mx.eval(model(part, cache=cache))
        if log:
            print(f"    chunk {i // chunk + 1}/{-(-len(ids) // chunk)} "
                  f"({i + part.shape[1]}/{len(ids)} tokens) "
                  f"{time.perf_counter() - t0:.1f}s "
                  f"peak {mx.get_peak_memory() / 1e9:.2f} GB", flush=True)
    mx.eval([a for c in cache for a in cache_arrays(c)])
    if zero:
        zero_state(cache)
    return cache, round(time.perf_counter() - t0, 2)


def real_state(c):
    """(keys, values) trimmed to the positions actually stored."""
    n = c.offset - getattr(c, "gap", 0)
    return c.keys[..., :n, :], c.values[..., :n, :]


_WORD = re.compile(r"\w+", re.UNICODE)


def content_words(text):
    """Lowercased words of four characters or more, which is the whole stopword cut."""
    return [w for w in (x.lower() for x in _WORD.findall(text)) if len(w) >= 4]


# Document frequency, not median IDF: on a small transcript the median puts
# "rare" at "appears once", and a word that signals an update appears twice.
def rare_words(sessions, max_doc_frac=0.25):
    """Content words appearing in at most `max_doc_frac` of the sessions."""
    df = Counter()
    for x in sessions:
        df.update(set(content_words(x["body"])))
    if not df:
        return set()
    cap = max(1, int(max_doc_frac * len(sessions)))
    return {w for w, n in df.items() if n <= cap}


# Two stems is the measured default and not a law: an update that restates one
# concept is missed there, so the threshold stays a caller's argument.
def supersede(lines_by_session, rare, min_shared=2):
    """Drop an earlier line when a later one shares `min_shared` rare stems with it."""
    order = [(i, ln) for i, lines in lines_by_session for ln in lines]
    sets = [{w[:6] for w in content_words(ln) if w in rare} for _, ln in order]
    drop = set()
    for a in range(len(order)):
        for b in range(a + 1, len(order)):
            if len(sets[a] & sets[b]) >= min_shared:
                drop.add(a)
                break
    out, k = [], 0
    for i, lines in lines_by_session:
        kept = []
        for _ in lines:
            if k not in drop:
                kept.append(order[k][1])
            k += 1
        if kept:
            out.append((i, kept))
    return out, len(drop)


# Phrases with which a reply declines to assert a value.
REFUSALS = (
    "не знаю", "не могу", "не указан", "не указано", "нет информации",
    "не говорил", "не сказал", "не уверен", "не помню", "неизвестно",
    "don't know", "do not know", "unknown", "not sure",
    "не упоминал", "не предоставили", "не предоставил", "не говорили",
    "не указали", "не сообщали", "не рассказывали", "не делились",
    "нет упоминани", "не упоминается", "нет данных", "нет сведений",
    "не указывал", "не было указано", "не было упомянуто", "нет доступа",
    "не содержится", "не содержит информац", "нет этой информации",
    "нет такой информации", "не было информации", "не упомянул",
    "не располагаю", "не имею информации", "не имею данных",
    "did not mention", "haven't mentioned", "no information about")


def is_refusal(reply):
    """True when the reply declines, including when it asks the question back."""
    low = reply.strip().lower()
    return bool(low) and (any(r in low for r in REFUSALS)
                          or low.endswith("?"))


# One absmax scale per tensor, not per head, which is the cheaper scale and the
# first thing to revisit if int8 ever costs recall.
def degrade_state(a, bits):
    """The array after a quantise and dequantise round trip at `bits`."""
    if bits == 16:
        return a.astype(mx.bfloat16).astype(mx.float32)
    scale = float(mx.max(mx.abs(a.astype(mx.float32))).item()) / 127.0
    if scale == 0.0:
        return a
    q = mx.clip(mx.round(a.astype(mx.float32) / scale), -127, 127)
    return (q * scale).astype(a.dtype)