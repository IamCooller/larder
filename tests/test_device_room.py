#!/usr/bin/env python3
"""MOE_RESIDENT's _device_room without a model or a GPU (MLX's calls stubbed): a pass that fits is not touched; one
short of room clears the pool and reads again, and only a shortfall the cleared pool does not cover shrinks the store.

    perl -e 'alarm shift; exec @ARGV' 280 python3 tests/test_device_room.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import stream_fast as sf  # noqa: E402


class FakeMx:
    def __init__(self, free, cache, freed_by_clear):
        self.free, self.cache, self.freed = free, cache, freed_by_clear
        self.calls = []

    def device_info(self):
        return {"free_memory": self.free}

    def get_cache_memory(self):
        return self.cache

    def get_active_memory(self):
        return 0

    def synchronize(self):
        self.calls.append("sync")

    def clear_cache(self):
        self.calls.append("clear")
        self.free += self.cache + self.freed
        self.cache = 0


def room(fake, need):
    real, real_others = sf.mx, sf.vram_others
    sf.mx, sf.vram_others = fake, lambda: 0
    try:
        return sf._device_room(need)
    finally:
        sf.mx, sf.vram_others = real, real_others


def main():
    m = FakeMx(300, 0, 0)
    assert room(m, 250) and m.calls == [], "a pass that fits must not touch the pool"
    m = FakeMx(150, 60, 100)            # 210 of 250 counted, the pool's kept blocks hold 100 more
    assert room(m, 250) and m.calls == ["sync", "clear"], m.calls
    m = FakeMx(150, 60, 0)              # nothing behind it: the store shrinks
    assert not room(m, 250) and m.calls == ["sync", "clear"]
    assert room(FakeMx(0, 0, 0), None), "an unknown need is room"
    print("device room: ok")


if __name__ == "__main__":
    main()
