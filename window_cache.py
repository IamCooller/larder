#!/usr/bin/env python3
"""A KV cache that physically stores only [sink | recent window] for the DRAFT model, never the target, so the draft's bytes per step stay flat at long context while the target keeps its full cache and the emitted tokens stay bit-identical to ordinary decode (usage: `to_window(draft_cache, sink=64, window=512)`, self-test `python3 tests/test_window_cache.py`)."""

import mlx.core as mx
from mlx_lm.models.cache import KVCache

# Fixed by the definition of the cache (the first tokens act as attention sinks,
# and 64 is one RoPE-friendly block), not a tunable.
SINK = 64


class WindowKVCache(KVCache):
    """KVCache that keeps only the first `sink` and the last `window` rows.

    `offset` is the absolute position count, exactly as in KVCache, and is what
    the model reads for RoPE. `stored` is how many rows are physically held and
    is what the returned keys/values are sized by -- the two diverge as soon as
    the first compaction runs, which is why `make_mask` cannot be inherited:
    mlx_lm builds its mask from `offset`.
    """

    def __init__(self, window, sink=SINK):
        super().__init__()
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window} "
                             f"(0 means 'no window' -- do not build this cache)")
        self._sink = sink
        self._window = window
        self._n = 0                       # stored rows, the write position

    # ------------------------------------------------------------ geometry
    @property
    def sink(self):
        return self._sink

    @property
    def window(self):
        return self._window

    @property
    def stored(self):
        """Rows physically held. Bounded by sink + window + window//4."""
        return self._n

    @property
    def _slack(self):
        return max(self._window // 4, 1)

    def _stored_after(self, incoming):
        """How many OLD rows survive once `incoming` new ones are added.

        A pure function of (_n, sink, window, incoming), so `make_mask` can
        predict it before `update_and_fetch` runs it -- mlx_lm builds the mask
        once, off cache[0], before any layer touches its cache, and every layer
        then makes the same decision from the same numbers.
        """
        cap = self._sink + self._window
        n = self._n
        if n + incoming <= cap + self._slack:
            return n
        # Never cut into the sink, and never drop more than is stored: a prompt
        # chunk wider than the window leaves storage temporarily over cap, and
        # the next step (which is a single token) brings it back down.
        return n - min(n + incoming - cap, max(n - self._sink, 0))

    def _compact(self, incoming):
        drop = self._n - self._stored_after(incoming)
        if drop <= 0:
            return
        s = self._sink
        # Rebuild rather than slide rows within the buffer: an overlapping
        # in-place copy is not something mx guarantees, and this runs once per
        # ~window/4 tokens. The buffer comes out exactly `_n` long, so the next
        # append takes KVCache's ordinary grow-by-`step` path.
        self.keys = mx.concatenate(
            [self.keys[..., :s, :], self.keys[..., s + drop:self._n, :]], axis=2)
        self.values = mx.concatenate(
            [self.values[..., :s, :], self.values[..., s + drop:self._n, :]],
            axis=2)
        self._n -= drop

    # --------------------------------------------------------------- cache
    def update_and_fetch(self, keys, values):
        S = keys.shape[2]
        # Before the append, not after: the mask for this pass was already
        # built from the same prediction, and a compaction after the write
        # would hand back fewer keys than the mask is wide.
        self._compact(S)
        prev = self._n
        if self.keys is None or (prev + S) > self.keys.shape[2]:
            B, n_kv_heads, _, k_head_dim = keys.shape
            v_head_dim = values.shape[3]
            n_steps = (self.step + S - 1) // self.step
            new_k = mx.zeros((B, n_kv_heads, n_steps * self.step, k_head_dim),
                             keys.dtype)
            new_v = mx.zeros((B, n_kv_heads, n_steps * self.step, v_head_dim),
                             values.dtype)
            if self.keys is not None:
                if prev < self.keys.shape[2]:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v

        self._n = prev + S
        self.offset += S
        self.keys[..., prev:self._n, :] = keys
        self.values[..., prev:self._n, :] = values
        return self.keys[..., :self._n, :], self.values[..., :self._n, :]

    def make_mask(self, N, return_array=False, window_size=None):
        """Boolean (N, stored + N) mask for a pass of N new rows.

        Needed because mlx_lm's create_attention_mask sizes its mask to
        `offset` -- right for a cache whose storage IS its history, wrong the
        moment one row has been dropped. Every stored row is visible to every
        new row (they are all in the past, whatever their absolute position);
        among the new rows it is plain causal.
        """
        if window_size is not None:
            # A model that already restricts its own attention (sliding-window
            # architectures) needs positions this cache no longer knows: the
            # stored rows are not contiguous in absolute position.
            raise ValueError("WindowKVCache does not support a model-side "
                             "sliding window; use RotatingKVCache for those")
        if N == 1:
            return None                   # one row, everything stored is past
        rows = mx.arange(N)
        block = rows[:, None] >= rows[None]
        s = self._stored_after(N)
        if s == 0:
            return block
        return mx.concatenate([mx.ones((N, s), dtype=mx.bool_), block], axis=1)

    def trim(self, n):
        """Rewind by `n` absolute positions, dropping `n` stored rows.

        Both counters move together, which is what makes trim-then-refeed
        exact: the surviving old rows are untouched, the tail is rewritten with
        the same keys at the same slots and the same absolute RoPE offsets, and
        a refeed can never trigger a compaction the original run did not (its
        stored count is strictly below the pre-trim one at every step).

        Clamped to what is physically stored. A caller that asks for more gets
        a smaller number back, which is exactly the signal
        trim_prompt_cache's callers already check (moe_stream rebuilds the
        draft cache from scratch when the trim comes up short).
        """
        n = min(n, self.offset, self._n)
        self.offset -= n
        self._n -= n
        return n

    # --------------------------------------------------------------- state
    @property
    def state(self):
        if self.keys is not None and self._n != self.keys.shape[2]:
            return self.keys[..., :self._n, :], self.values[..., :self._n, :]
        return self.keys, self.values

    @state.setter
    def state(self, v):
        self.keys, self.values = v
        self._n = 0 if self.keys is None else self.keys.shape[2]

    @property
    def meta_state(self):
        # `offset` has to travel: it is absolute and no longer recoverable from
        # the array shapes the way KVCache's is.
        return tuple(map(str, (self.offset, self._sink, self._window)))

    @meta_state.setter
    def meta_state(self, v):
        self.offset, self._sink, self._window = map(int, v)

    def to_quantized(self, group_size: int = 64, bits: int = 4):
        # KVCache's version would copy `offset` into the quantised cache's
        # storage index and silently score the wrong rows.
        raise NotImplementedError("WindowKVCache cannot be quantised")


def to_window(cache, sink=SINK, window=512):
    """Convert plain KVCache layers to WindowKVCache IN PLACE, and compact.

    In place, and by re-classing the objects rather than rebuilding the list,
    for tree_spec.as_tree_cache's reason: a caller that keeps a cache across
    turns passes `list(self._draft_cache)`, a copy of the list whose ELEMENTS
    are the objects it reads `offset` off afterwards.

    Takes one cache or a list of them; returns what it was given.
    """
    layers = cache if isinstance(cache, (list, tuple)) else [cache]
    for c in layers:
        if type(c) is WindowKVCache:
            c._sink, c._window = sink, window
        elif type(c) is KVCache:
            c.__class__ = WindowKVCache
            c._sink, c._window = sink, window
            c._n = c.offset          # a plain KVCache stores every position
        else:
            raise ValueError(f"a draft window needs plain KVCache layers, got "
                             f"{type(c).__name__}")
        c._compact(0)
    return cache


if __name__ == "__main__":
    print(__doc__)
