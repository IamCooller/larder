#!/usr/bin/env python3
"""memory_core.py and the --memory path: model-free, no checkpoint, no probes.

    python3 tests/test_memory_core.py

Five checks:

 1. cache    -- GapKVCache reports a shifted position to RoPE while its stored
                keys stay compact, real_state trims to what is stored, and a
                clone does not share buffers with its source.
 2. text     -- the write gate's word cut, document frequency, supersession and
                the refusal detector, on a handful of literal sessions.
 3. int8     -- the stored format is the reference round trip, and zero is a
                fixed point of it.
 4. store    -- journal -> plan -> render -> write -> load -> attach through
                kvmem.Store with a fake engine, every KV array compared bit for
                bit before and after the file.
 5. import   -- importing kvmem, the engine and both servers in a process where the
                research modules cannot be imported at all still works.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import ArraysCache, KVCache

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import kvmem                                                    # noqa: E402
import memory_core as MC                                        # noqa: E402


def test_cache():
    c = MC.GapKVCache(5)
    c.offset = 5
    k = mx.ones((1, 2, 2, 3))
    c.update_and_fetch(k, k)
    assert c.offset == 7, c.offset
    ks, vs = MC.real_state(c)
    assert ks.shape == (1, 2, 2, 3) and vs.shape == (1, 2, 2, 3)
    assert bool(mx.all(ks == 1).item())

    a = ArraysCache(size=2)
    a.cache = [mx.zeros((1, 2, 2)), None]
    b = MC.clone_one(a)
    assert len(MC.cache_arrays(a)) == 1 and len(MC.cache_arrays(b)) == 1
    a.cache[0][0, 0, 0] = 3.0
    assert float(b.cache[0][0, 0, 0].item()) == 0.0, "the clone shares its buffer"

    c2 = MC.clone_cache([a, c])
    assert len(c2) == 2 and isinstance(c2[1], KVCache)
    assert c2[1].offset == 7


def test_text():
    assert MC.content_words("Меня зовут Антон.") == ["меня", "зовут", "антон"]
    filler = ["Роутер выбирает восемь экспертов из набора.",
              "Синхронизация по слоям стоит больше байтов.",
              "Арена держит двадцать один эксперт на слой.",
              "Предвыборка не убирает синхронизацию по слою.",
              "Раскладка файла имеет два разных шага.",
              "Отложенная проверка пересчитывает токены."]
    sessions = [{"body": b} for b in filler] + [
        {"body": "Мой движок называется Кальмар."},
        {"body": "Я переименовал движок: теперь он называется Наутилус."}]
    rare = MC.rare_words(sessions)
    assert "движок" in rare and "называется" in rare, rare
    kept, dropped = MC.supersede(
        [(0, ["Мой движок называется Кальмар."]),
         (1, ["Я переименовал движок: теперь он называется Наутилус."])],
        rare)
    assert dropped == 1 and kept == [(1, ["Я переименовал движок: теперь он "
                                          "называется Наутилус."])], kept
    assert MC.is_refusal("Не знаю.") and MC.is_refusal("А как вас зовут?")
    assert not MC.is_refusal("Меня зовут Антон.")
    assert not MC.is_refusal("   ")


def test_int8():
    with mx.stream(mx.cpu):
        a = mx.array(np.linspace(-3, 3, 12, dtype=np.float32).reshape(1, 4, 3))
        q, scale = kvmem._quant8(a)
        assert q.dtype == mx.int8 and scale > 0
        assert bool(mx.all(kvmem._dequant8(q, scale, a.dtype)
                           == MC.degrade_state(a, 8)).item())
        z = mx.zeros((2, 2))
        assert bool(mx.all(MC.degrade_state(z, 8) == z).item())


def test_store():
    root = Path(tempfile.mkdtemp(prefix="memory-core-test-"))
    try:
        store = kvmem.Store("/nowhere/toy-model", tag="t", root=root)
        store.append_session(["Меня зовут Антон.", "Я живу в Казани."])
        store.append_session("Моего кота зовут Бублик.")
        store.append_session("Мой движок называется Кальмар.")
        plan = store.plan()
        assert plan["keep"], "the gate kept nothing"

        fake = kvmem._FakeLLM()
        cache = fake.fill(seed=2)
        ids = list(range(20, 27))
        before = {i: MC.real_state(c) for i, c in enumerate(cache)
                  if not isinstance(c, ArraysCache)}
        info = store._write(cache, ids, plan, 0.5, fake)
        assert info["written"] and store.core_path.exists()
        got, got_ids, meta = store.load()
        assert got_ids == ids and meta["tokens"] == len(ids)
        assert json.loads(store.meta_path.read_text())["layers"] == meta["layers"]
        for i, c in enumerate(got):
            if isinstance(c, ArraysCache):
                continue
            k, v = MC.real_state(c)
            k0, v0 = before[i]
            assert bool(mx.all(k == k0).item()) and bool(mx.all(v == v0).item())

        fresh = kvmem._FakeLLM()
        assert store.attach(fresh) is True
        assert fresh._seen == ids and fresh.history[-1]["content"] == MC.ACK
        assert store.stats()["core"] is True
        assert store.core_text() == "\n".join(
            ln for _, ls in plan["keep"] for ln in ls)
    finally:
        import shutil
        shutil.rmtree(root, ignore_errors=True)


def test_import_without_probes():
    """A process where the research modules do not exist imports the engine, the servers and kvmem fine."""
    private = ("kv_scale", "lifelong_probe", "state_memory", "lora_grad", "bench", "stage")
    code = (
        "import sys\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        f"        if name in {private!r}:\n"
        "            raise ImportError(name + ' is private')\n"
        "        return None\n"
        "sys.meta_path.insert(0, Block())\n"
        "import kvmem, memory_core, moe_stream, stream_fast, pcache, server, chat\n"
        f"bad = [m for m in {private!r} if m in sys.modules]\n"
        "assert not bad, bad\n"
        "assert kvmem.Store and kvmem.ground and kvmem.FIRST_PERSON\n"
        "print('imports clean')\n")
    out = subprocess.run([sys.executable, "-c", code],
                         cwd=str(Path(__file__).resolve().parent.parent),
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "imports clean" in out.stdout, out.stdout


if __name__ == "__main__":
    test_cache()
    test_text()
    test_int8()
    test_store()
    test_import_without_probes()
    print("test_memory_core OK")