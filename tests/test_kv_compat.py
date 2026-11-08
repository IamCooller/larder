"""KV cache keys and values go in and out the same way on mlx-lm's two-value and three-value state setters."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx
from mlx_lm.models.cache import KVCache

from memory_core import GapKVCache, clone_one, kv_arrays, set_kv


class ThreeValueKVCache(KVCache):
    """KVCache with mlx-lm 0.32's state: the whole buffer plus the offset, set from three values."""

    @property
    def state(self):
        return self.keys, self.values, self.offset

    @state.setter
    def state(self, v):
        self.keys, self.values, self.offset = v


def check(cls):
    k = mx.arange(2 * 3 * 4 * 5, dtype=mx.float32).reshape(2, 3, 4, 5)
    c = cls()
    set_kv(c, k, k + 1)
    assert c.offset == 4 and mx.array_equal(kv_arrays(c)[0], k) and mx.array_equal(kv_arrays(c)[1], k + 1)
    # a buffer longer than the offset is read only up to the offset
    c.update_and_fetch(k[..., :1, :], k[..., :1, :])
    ks, vs = kv_arrays(c)
    assert c.offset == 5 and ks.shape[2] == 5 and mx.array_equal(ks[..., :4, :], k)
    d = clone_one(c)
    assert d.offset == 5 and mx.array_equal(kv_arrays(d)[0], ks)
    # a stub-sliced gap cache keeps RoPE's offset while it stores fewer positions
    g = GapKVCache(2)
    set_kv(g, ks[..., :3, :], vs[..., :3, :])
    g.offset += 2
    g.update_and_fetch(k[..., :1, :], k[..., :1, :])
    assert g.offset == 6 and g.keys.shape[2] >= 4


def main():
    check(KVCache)
    check(ThreeValueKVCache)
    print("test_kv_compat: ok")


if __name__ == "__main__":
    main()
