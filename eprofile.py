#!/usr/bin/env python3
"""Expert profiles that outlive the process, the engine's long-term memory: a profile on disk, decayed per session and built from the undecayed `_seen` demand counts of `stream_fast.py`, seeds the arena at startup with `warm()` and folds the session in at sleep with `update()` (`python3 eprofile.py` runs a demo that needs no model)."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

STORE = Path.home() / ".cache/moe-brain"

# A habit survives about ten sessions of not being used. Chosen, not measured:
# it is the knob that says how fast the system forgets, and there is no way to
# measure it without months of real use. Everything downstream reads this one
# constant so the whole system forgets at one rate.
HALF_LIFE_SESSIONS = 10.0

# What a seeded expert is worth against live demand. With DECAY 0.8 an expert
# wanted on every call settles at 1/(1-0.8) = 5.0, so seeding the profile's peak
# at 5.0 puts a remembered habit on equal footing with a currently hot expert --
# and no higher. It washes out either way: 0.8^10 = 0.107 after ten tokens.
WARM_SCALE = 5.0


def decayed(value, age, half_life):
    """The one forgetting law in this system.

    Used for experts here and for facts in memory.py, so that "important stays,
    unimportant fades" is one formula and not two that drift apart.
    """
    return value * 0.5 ** (age / half_life)


def glus_of(model):
    """The streaming MoE blocks, whether or not the MoE block is wrapped."""
    from stream_fast import CacheBiasedMoE

    out = []
    for layer in model.layers:
        mlp = getattr(layer, "mlp", None)
        inner = mlp.block if isinstance(mlp, CacheBiasedMoE) else mlp
        glu = getattr(inner, "switch_mlp", None)
        if glu is not None and hasattr(glu, "_projs"):
            out.append(glu)
    return out


def index_sha(model_path):
    """Identity of the packed expert file.

    Expert ids mean nothing on their own -- they are offsets into one particular
    `experts.bin`. A profile carried to a different packing would silently pin
    the wrong weights, so every profile records which packing it came from.
    """
    idx = Path(model_path).expanduser() / "experts_index.json"
    return hashlib.sha256(idx.read_bytes()).hexdigest()[:16]


def store_path(model_path, tag):
    name = Path(model_path).expanduser().name
    return STORE / name / f"profile-{tag}.npz"


def harvest(model):
    """Undecayed demand per expert this process has seen, as (layers, experts).

    Returns None before the arena exists: it is built lazily on the first decode,
    so a process that only ever prefilled has nothing to contribute.
    """
    import stream_fast
    stream_fast.native_sync()       # MOE_NATIVE=1 keeps the counts on the agent
    glus = glus_of(model)
    rows = [g._seen for g in glus]
    if not rows or any(r is None for r in rows):
        return None
    return np.asarray(rows, dtype=np.float32)


def normalise(freq):
    """Each layer to sum 1.0, so a long session does not outvote a short one."""
    freq = np.asarray(freq, dtype=np.float64)
    total = freq.sum(axis=1, keepdims=True)
    return np.divide(freq, total, out=np.zeros_like(freq),
                     where=total > 0).astype(np.float32)


def merge(old, new, sessions=1.0, half_life=HALF_LIFE_SESSIONS):
    """Fold one session's demand into the long profile.

    The old profile decays by however many sessions have passed; the new one is
    normalised first, so ten minutes of chat and three hours of it carry the
    same weight. Habit strength should follow how OFTEN something is done, not
    how long the session that did it happened to run.
    """
    new = normalise(new)
    if old is None:
        return new
    old = np.asarray(old, dtype=np.float32)
    if old.shape != new.shape:
        raise ValueError(f"profile shape {old.shape} against model {new.shape}")
    return (decayed(old, sessions, half_life) + new).astype(np.float32)


def save(model_path, tag, freq, sessions):
    p = store_path(model_path, tag)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Write beside the target and rename: rename is atomic on the same
    # filesystem, so a kill during the write leaves the previous profile whole
    # instead of replacing it with a truncated one.
    # `.tmp.npz`, not `.npz.tmp`: np.savez_compressed APPENDS ".npz" when the
    # name does not already end in it, so the temp file landed as
    # "profile-x.npz.tmp.npz" and the rename looked for a name that never
    # existed. The hardening broke the thing it was hardening, and brain_eval
    # caught it on the next run.
    tmp = p.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp, freq=np.asarray(freq, dtype=np.float32),
        meta=json.dumps({
            "index_sha": index_sha(model_path),
            "sessions": int(sessions),
            "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "half_life_sessions": HALF_LIFE_SESSIONS,
        }))
    tmp.replace(p)
    return p


def load(model_path, tag, strict=True):
    """The saved profile, or None. Refuses a profile from a different packing."""
    p = store_path(model_path, tag)
    if not p.exists():
        return None, None
    try:
        with np.load(p, allow_pickle=False) as z:
            freq, meta = z["freq"], json.loads(str(z["meta"]))
    except Exception as e:
        # np.savez writes straight to the final path, so a process killed
        # mid-write leaves a truncated archive -- and this machine kills
        # processes for memory. A semantic mismatch below already degrades
        # gracefully; file-level damage is the likelier failure and had no
        # handling at all, so a half-written profile took the whole session
        # down at startup.
        print(f"  ignoring profile {p.name}: unreadable ({type(e).__name__})")
        return None, None
    want = index_sha(model_path)
    if meta.get("index_sha") != want:
        msg = (f"{p.name} was built on packing {meta.get('index_sha')}, this "
               f"model is {want}. Expert ids do not carry across packings.")
        if strict:
            raise ValueError(msg)
        print(f"  ignoring profile: {msg}")
        return None, None
    return freq, meta


def apply(model, freq):
    """Hand the profile to the GLUs; `_init_arena` seeds `_freq` from it.

    Seeding the counter rather than the LRU directly means the engine's ordinary
    `_retain()` does the filling, so warm and cold start share one code path.
    """
    glus = glus_of(model)
    freq = np.asarray(freq, dtype=np.float64)
    if freq.shape[0] != len(glus):
        raise ValueError(f"profile has {freq.shape[0]} layers, model has "
                         f"{len(glus)}")
    # The expert axis too. `_init_arena` slices `_warm[:n_exp]`, so a short row
    # would be padded by reading past it and a long one silently truncated --
    # a profile from another packing warming the wrong experts, with no error.
    # `_n_exp` on the GLU is 0 until the first decode builds the arena, so it
    # cannot be used at startup; the packing metadata on the projection can.
    packed = getattr(glus[0]._projs[0], "_packed", None)
    n_exp = packed["n_experts"] if packed else None
    if n_exp is not None and freq.shape[1] != n_exp:
        raise ValueError(f"profile has {freq.shape[1]} experts per layer, "
                         f"model has {n_exp}")
    for g, row in zip(glus, freq):
        peak = row.max()
        g._warm = (row / peak * WARM_SCALE) if peak > 0 else row
    return len(glus)


def warm(model, model_path, tag, strict=False):
    """Startup: load the profile for `tag` and seed the arena from it."""
    freq, meta = load(model_path, tag, strict=strict)
    if freq is None:
        return None
    apply(model, freq)
    return meta


def update(model, model_path, tag):
    """Sleep: fold this process's demand into the long profile and save it."""
    new = harvest(model)
    if new is None:
        return None
    old, meta = load(model_path, tag, strict=False)
    sessions = (meta or {}).get("sessions", 0) + 1
    merged = merge(old, new)
    save(model_path, tag, merged, sessions)
    return {"tag": tag, "sessions": sessions,
            "layers": int(merged.shape[0]), "experts": int(merged.shape[1])}


class _StubGlu:
    def __init__(self, seen):
        self._projs = [object()]
        self._seen = seen
        self._warm = None


class _StubModel:
    """Enough of a model for harvest/apply, so the maths is testable offline."""

    def __init__(self, seen):
        import types
        self.layers = [
            types.SimpleNamespace(mlp=types.SimpleNamespace(
                switch_mlp=_StubGlu(row))) for row in seen]


def demo():
    """The decay law and the round trip, on numbers whose answers are known."""
    assert decayed(1.0, 0, 10) == 1.0
    assert abs(decayed(1.0, 10, 10) - 0.5) < 1e-12, "one half-life halves it"
    assert abs(decayed(1.0, 20, 10) - 0.25) < 1e-12

    # normalise: rows sum to one; an all-zero row stays zero rather than NaN.
    n = normalise(np.array([[1.0, 3.0], [0.0, 0.0]]))
    assert abs(n[0].sum() - 1.0) < 1e-6 and abs(n[0, 1] - 0.75) < 1e-6, n
    assert n[1].sum() == 0.0, "an unused layer must not become NaN"

    # merge: a session's length must not change how much it counts.
    short = np.array([[1.0, 0.0, 0.0]])
    long_ = np.array([[1000.0, 0.0, 0.0]])
    assert np.allclose(merge(None, short), merge(None, long_)), \
        "session length is leaking into habit strength"

    # merge: an old habit halves over one half-life of disuse.
    old = np.array([[1.0, 0.0]], dtype=np.float32)
    quiet = merge(old, np.array([[0.0, 1.0]]), sessions=HALF_LIFE_SESSIONS)
    assert abs(quiet[0, 0] - 0.5) < 1e-6, quiet

    # harvest/apply round trip through the stub.
    seen = np.array([[0.0, 4.0, 1.0], [2.0, 0.0, 0.0]])
    m = _StubModel(seen)
    got = harvest(m)
    assert got.shape == (2, 3) and np.allclose(got, seen), got
    apply(m, normalise(seen))
    warmed = [g._warm for g in glus_of(m)]
    assert abs(warmed[0].max() - WARM_SCALE) < 1e-6, "peak must equal WARM_SCALE"
    assert warmed[0][0] == 0.0 and warmed[0][1] > warmed[0][2], warmed[0]

    # An arena that never existed contributes nothing rather than crashing.
    assert harvest(_StubModel([])) is None
    empty = _StubModel(np.zeros((1, 3)))
    empty.layers[0].mlp.switch_mlp._seen = None
    assert harvest(empty) is None

    # A profile whose layer count disagrees with the model must be refused.
    try:
        apply(m, np.zeros((5, 3)))
        raise AssertionError("mismatched layer count was accepted")
    except ValueError:
        pass
    # ... and so must one whose expert count disagrees with the packing.
    class _Proj:
        _packed = {"n_experts": 3}
    m.layers[0].mlp.switch_mlp._projs = [_Proj()]
    try:
        apply(m, np.zeros((2, 4)))
        raise AssertionError("mismatched expert count was accepted")
    except ValueError:
        pass
    apply(m, np.zeros((2, 3)))          # the right width still goes through
    print("demo ok")


if __name__ == "__main__":
    demo()
