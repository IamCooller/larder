#!/usr/bin/env python3
"""Long-term memory with no prompt in it: the user's own words, kept as the attention KV and recurrent state of the sessions that carried something.

A core is rebuilt from the journal by one prefill at consecutive positions, so its offset is its own token count and never grows with the years.

    store = kvmem.Store(model_path, tag="default")
    store.append_session(["Меня зовут Антон.", "Я живу в Казани."])
    store.plan()          # model-free: which sessions and lines form the core
    store.rebuild(llm)    # prefill, quantise, write core.safetensors atomically
    store.attach(llm)     # load it into the engine, exactly as ask_all does
    store.stats()

    python3 kvmem.py --selftest     # model-free, a few seconds
    python3 kvmem.py --stats        # what is on disk right now
"""

import argparse
import hashlib
import re
from collections import Counter
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_lm.models.cache import ArraysCache

from engine_log import note
import eprofile
from memory_core import (ACK, GapKVCache, cache_arrays, content_words,
                         degrade_state, is_refusal, prefill_chunked, rare_words,
                         real_state, set_kv, supersede, turn_ids)

STORE = eprofile.STORE

# A line is kept when one of its words occurs in at most this fraction of the
# sessions. MEASURED on the probe's synthetic year, 8 seeds each (fact lines
# kept / filler lines kept): 20x10 sessions 120/120 and 3 of 1480; 100x10
# 120/120 and 0 of 7880; 200x10 120/120 and 0 of 15880. The probe's own 0.25 --
# which is the right cut for supersession stems, where missing a shared concept
# is the expensive error -- keeps half the filler at 20 sessions, because the
# filler is combinatorial and a rare-ish word is not a rare word.
#
# Below ~12 sessions the gate is degenerate and keeps most of the transcript:
# with a few dozen lines in total almost every word is a hapax, so document
# frequency has nothing to separate. That is a floor on the transcript, not a
# threshold to tune -- see selftest.
WRITE_DF_FRAC = 0.05
# A fact about the user is said in the first person, and since rarity alone is
# a perfect gate only on generated filler (which says "мы" a lot) and keeps
# nearly every filler line of real text, the gate is the conjunction of both.
FIRST_PERSON = re.compile(
    r"(?<![а-яё])(я|мне|меня|мной|мною|мой|моя|моё|мое|мои|моего|моей|моих|"
    r"моему|моим|моими|мы|нас|нам|наш|наша|наше|наши|нашего|нашей|нашим|наших)"
    r"(?![а-яё])", re.I)
# The pronouns say "about the user", never "the same fact": "Меня зовут Антон"
# was retired by an unrelated sentence containing "как меня зовут" because
# "меня" counted as a shared rare stem. They are excluded from supersession.
PRONOUNS = frozenset(
    "я мне меня мной мною мой моя моё мое мои моего моей моих моему моим моими "
    "мы нас нам наш наша наше наши нашего нашей нашим наших".split())

# Rare-word cut for SUPERSESSION, the probe's own default. A word that signals
# a restatement appears in both the old line and the new one, so it can never
# be a hapax; 0.25 is what makes "движок" count as rare in a short transcript.
SUPERSEDE_DF_FRAC = 0.25

_DTYPES = {"float32": mx.float32, "bfloat16": mx.bfloat16,
           "float16": mx.float16}


def store_dir(model_path, tag, root=None):
    """~/.cache/moe-brain/<checkpoint name>/kvmem-<tag>/, beside the profiles."""
    name = Path(model_path).expanduser().name
    return (Path(root) if root else STORE) / name / f"kvmem-{tag}"


def index_sha(model_path):
    """Identity of the packed checkpoint, or None when it is not packed.

    Same hash `eprofile` pins its expert profiles with. A KV core built on
    other weights is not merely stale, it is meaningless -- the keys are that
    model's projections of those tokens -- so the core records which checkpoint
    produced it and `attach` refuses a mismatch. Returns None rather than
    raising for a plain (unpacked) directory, which is what the selftest and a
    toy engine have.
    """
    try:
        return eprofile.index_sha(model_path)
    except (FileNotFoundError, NotADirectoryError, OSError):
        return None


# --- grounding: the memory's answer must be made of the memory -----------------

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"\+?\d[\d\s()-]{5,}\d")
_CAP = re.compile(r"[A-ZА-ЯЁ][\w-]{2,}")
# Capitalised function words that are not names: the probe's refusals put
# "Какой" after a colon and grounding flagged it as an unknown entity.
_CAP_STOP = frozenset(w.lower() for w in (
    "Какой Какая Какое Какие Каком Какую Как Что Где Когда Почему Зачем Если Это "
    "Вы Ваш Ваша Ваше Ваши Вам Вас Вашего Вашей Ваших Однако Также Например "
    "Кстати Пожалуйста Спасибо Судя Скорее Вероятно Возможно Полагаю Извините "
    "Уточните Расскажите Сообщите Хорошо Понял Понятно Итак Поэтому Сейчас").split())
_NEGATIVE = re.compile(r"(?i)у вас (?:нет|отсутству\w+|не было)[^.!?\n]*")
# "Судя по всему, вы предпочитаете спокойную музыку" -- a guess about the user
# dressed as an answer. No entity, no number, so the rules above are blind to
# it; the hedge itself is the tell. Unfounded when nothing of substance in the
# sentence has a stem in the core.
_HEDGE = re.compile(r"(?i)(?:судя по всему|скорее всего|вероятно|по всей видимости|"
                    r"возможно|полагаю|предположу|могу предположить|похоже,? что|"
                    r"думаю,? что)[^.!?\n]*")
# Numerals, as digits or as Russian words: "У вас двое детей" is an invented
# count. Word and digit forms of one number are interchangeable when checking
# against the core ("девять лет" in the core grounds "9 лет" in the reply).
_NUM_WORDS = {"ноль": 0, "один": 1, "одна": 1, "одного": 1, "два": 2, "две": 2,
              "двое": 2, "двух": 2, "три": 3, "трое": 3, "трёх": 3, "четыре": 4,
              "четверо": 4, "пять": 5, "пятеро": 5, "шесть": 6, "семь": 7,
              "восемь": 8, "девять": 9, "десять": 10, "одиннадцать": 11,
              "двенадцать": 12, "тринадцать": 13, "четырнадцать": 14,
              "пятнадцать": 15, "шестнадцать": 16, "семнадцать": 17,
              "восемнадцать": 18, "девятнадцать": 19, "двадцать": 20}
_NUM_RE = re.compile(r"(?<![\w@.])(\d{1,4}|" + "|".join(sorted(_NUM_WORDS, key=len, reverse=True))
                     + r")(?![\w@])", re.I)


def _numbers(text):
    out = set()
    for m in _NUM_RE.finditer(text.lower()):
        tok = m.group(1)
        out.add(int(tok) if tok.isdigit() else _NUM_WORDS[tok])
    return out
_NEG_STOP = {"отсутствует", "отсутствуют", "было", "этой", "этого", "такой"}


# A question about a third person: if the core has no line about that person,
# a confident answer is a stored fact transplanted onto them ("ваш брат
# занимается плаванием" when the swimmer is the user). Relation words, stemmed.
_RELATIONS = ("брат", "сестр", "мать", "матер", "мама", "отец", "отц", "папа",
              "жена", "жены", "жене", "муж", "сын", "доч", "бабушк", "дедушк",
              "дед", "друг", "подруг", "коллег", "начальник", "шеф", "сосед",
              "ребён", "ребен", "дет", "тёт", "тет", "дяд", "племян", "внук",
              "внуч", "партнёр", "партнер", "девушк", "парн", "супруг")


def _relation_of(question):
    q = question.lower()
    return next((r for r in _RELATIONS if re.search(r"(?<![а-яё])" + r, q)), None)


# Optional: a Russian lemmatiser turns the "invented common noun" class into a
# rule ("Вы играете на гитаре" -- no guitar in the core). Stems could not do it:
# 26% false flags from inflection; lemmas measured 0.8% (2 of 240 correct fact
# answers) while catching 27 of 33 inventions offline. Off when pymorphy3 is
# not installed, so the module never depends on it.
try:
    import pymorphy3
    _MORPH = pymorphy3.MorphAnalyzer()
except Exception:                      # not installed: the rule is skipped
    _MORPH = None
# Meta-nouns a reply uses to talk about the conversation itself, not facts.
_NOUN_STOP = frozenset("информация сообщение данные вопрос ответ память беседа разговор "
                       "история пользователь имя человек тема пример деталь запрос помощь "
                       "уточнение сведение контекст факт число значение слово просьба "
                       "случай образ порядок момент начало конец часть список пункт "
                       "вариант версия форма формат вид".split())


def _noun_lemmas(text, confident=True):
    out = set()
    if _MORPH is None:
        return out
    for w in re.findall(r"[а-яё]{4,}", text.lower().replace("*", "")):
        p = _MORPH.parse(w)[0]
        if confident and (p.tag.POS != "NOUN" or p.score < 0.5):
            continue
        if len(p.normal_form) < 4:
            # "кота" lemmatises to "кот", which the core side never indexes
            # (its regex takes 4+ letters): a flag by construction, not a fact.
            continue
        out.add(p.normal_form)
    return out


def _all_lemmas(text):
    out = set()
    if _MORPH is None:
        return out
    for w in re.findall(r"[а-яё]{4,}", text.lower()):
        out.add(_MORPH.parse(w)[0].normal_form)
    return out


def ground(reply, core_text, question=None):
    """Names, addresses and numbers in a reply about the user must come from
    the core. Returns (ok, offending).

    The 16-question runs showed what a memory cannot do on its own: asked for
    a car it never heard of, the model said "Mercedes-Benz G-Class"; asked for
    an e-mail, it made one up. Those are not memory errors -- the core holds no
    such fact -- they are the model filling a gap, and a lexical detector of
    refusals cannot stop them. This can: every capitalised word that is not the
    first word of its sentence, every e-mail and every phone-like number in
    the reply must appear in the core text (by a five-character stem, because
    Russian inflects). Markdown emphasis is stripped first. Cheap, deterministic,
    and it says exactly which token was invented.
    """
    if is_refusal(reply):
        return True, []            # a refusal asserts nothing to ground
    text = reply.replace("*", "")
    low = core_text.lower()
    if question is not None:
        rel = _relation_of(question)
        if rel is not None and not re.search(r"(?<![а-яё])" + rel, low):
            # The core knows nothing about this person; whatever the reply
            # asserts about them came from somewhere else.
            return False, [f"о «{rel}» в памяти ничего нет"]
    digits = re.sub(r"\D", "", core_text)
    bad = []
    for m in _EMAIL.finditer(text):
        if m.group(0).lower() not in low:
            bad.append(m.group(0))
    for m in _PHONE.finditer(text):
        if re.sub(r"\D", "", m.group(0)) not in digits:
            bad.append(m.group(0).strip())
    for sent in re.split(r"(?<=[.!?:;\n])\s+|\n", text):
        words = sent.split()
        for w in words[1:]:                      # a sentence-initial capital says nothing
            m = _CAP.match(w.strip("«»\"'(),.!?:;"))
            if m and m.group(0).lower() not in _CAP_STOP \
                    and m.group(0)[:5].lower() not in low:
                bad.append(m.group(0))
    # "У вас нет детей" -- a confident negative about the user is an assertion
    # too, and the core said nothing about children. Unfounded when no content
    # word of the sentence has a stem in the core.
    core_nums = _numbers(core_text)
    for n in _numbers(text) - core_nums:
        bad.append(str(n))
    for m in _NEGATIVE.finditer(text):
        sent = m.group(0)
        if not any(w[:5].lower() in low for w in re.findall(r"[а-яёa-z]{4,}", sent.lower())
                   if w not in _NEG_STOP):
            bad.append(sent.strip())
    # A hedge stands only on an entity or a number the core knows: a long
    # guess always shares some common word ("работы") with the core, so common
    # words do not count here.
    for m in _HEDGE.finditer(text):
        sent = m.group(0)
        anchored = any(w.group(0)[:5].lower() in low for w in _CAP.finditer(sent)) \
            or bool(_numbers(sent) & core_nums)
        if not anchored:
            bad.append(sent.strip())
    if _MORPH is not None:
        # Every confident noun of the reply must be a lemma the core contains.
        core_lem = _all_lemmas(core_text)
        for lem in sorted(_noun_lemmas(text) - core_lem - _NOUN_STOP):
            bad.append(lem)
    return not bad, bad


REFUSAL = "В моей памяти об этом ничего нет."


def render_sha(llm):
    """Identity of the RENDERING, not of the text.

    The core's tokens were produced by one particular chat template's
    separators plus one particular acknowledgement. Feed them to an engine that
    renders turns differently and the memory sits at token boundaries the model
    never sees, which is the failure mode that reads as "it just forgot".
    """
    tpl = getattr(llm, "_tpl", None) or {}
    blob = json.dumps({"tpl": tpl, "acks": list(ACKS)}, sort_keys=True,
                      ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# --------------------------------------------------------------------------
# int8 for the recurrent state
# --------------------------------------------------------------------------

def _quant8(a):
    """(int8 quotient, absmax scale) -- `memory_core.degrade_state(a, 8)` split.

    One scale per tensor, symmetric, clipped to +-127: exactly the arithmetic
    of the `int8conv` row that passed acceptance at 20, 100 and 200 sessions,
    so the bytes on disk are the bytes that were measured. A per-head or
    per-row scale is finer and is the upgrade path if int8 ever costs recall;
    it is not what was measured, so it is not what is written.
    """
    f = a.astype(mx.float32)
    scale = float(mx.max(mx.abs(f)).item()) / 127.0
    if scale == 0.0:
        return mx.zeros(a.shape, dtype=mx.int8), 0.0
    return mx.clip(mx.round(f / scale), -127, 127).astype(mx.int8), scale


def _dequant8(q, scale, dtype):
    return (q.astype(mx.float32) * scale).astype(dtype)


# --------------------------------------------------------------------------
# rendering one session the way the acceptance rendered it
# --------------------------------------------------------------------------

# One complete acknowledgement per kept session. Not one fixed sentence: with
# 13 identical "Понял." closing 13 sessions, the 100-session run answered
# "Понял." to 1-2 of 16 never-stated questions -- the template leaking back as
# a reply, the same induction the soaks showed for fixed short acks. Varied by
# session index so the core holds no repeated line, complete so each user turn
# is still closed by a real assistant turn (the boundary that recall needs).
ACKS = ("Понял.", "Хорошо, запомнил.", "Принято, буду иметь в виду.",
        "Спасибо, учту это.", "Записал, спасибо.", "Понятно, запомню.",
        "Хорошо, это я учёл.", "Ясно, буду помнить.")


def render(llm, keep, ack=None):
    """[(token ids, position)] for the kept sessions, at CONSECUTIVE positions.

    `turn_ids(llm, body, first) + encode(ack)` is character for character
    what the acceptance probe's transcript builder puts in a session and what
    its oracle re-renders. Reproduced rather than re-invented: the
    engine renders the first turn with `head` and every later one with
    `between`, and getting that boundary wrong is silent -- the tokens still
    decode, the model just stops treating the memory as a finished exchange.
    `ack=None` rotates through ACKS by kept-session index; pass a string to pin
    one (the probe's oracle uses ACK).
    """
    enc = lambda s: llm.tokenizer.encode(s, add_special_tokens=False)
    out, pos = [], 0
    for i, (_, lines) in enumerate(keep):
        a = ACKS[i % len(ACKS)] if ack is None else ack
        ids = turn_ids(llm, "\n".join(lines), not out) + enc(a)
        out.append((ids, pos))
        pos += len(ids)
    return out


def _atomic(path, write):
    """Write beside the target and rename, the way `eprofile.save` does.

    rename(2) is atomic on one filesystem, so a kill in the middle leaves the
    PREVIOUS core whole instead of a truncated one. This machine kills
    processes for memory, and a core is tens of megabytes -- the window is not
    theoretical.

    `core.tmp.safetensors` and not `core.safetensors.tmp`, the trap
    `eprofile.save` already fell into once: `mx.save_safetensors` APPENDS the
    extension when the name does not end in it, so the temp file landed as
    `core.safetensors.tmp.safetensors` and the rename looked for a name that
    never existed -- the hardening broke the thing it was hardening.
    """
    tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")
    try:
        write(tmp)
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


class Store:
    """A journal on disk, and the KV core built from it.

    Three files under `store_dir`:

        journal.jsonl      one JSON object per session, APPEND ONLY
        core.safetensors   the KV (bf16) + the int8 state + the token ids
        core.json          which sessions it holds, and what it was built on

    Nothing here ever deletes a user's words. The journal only grows;
    supersession retires a line from the CORE and records it in `core.json`, so
    what the model no longer sees is still on disk and still readable.
    """

    def __init__(self, model_path, tag="default", root=None):
        self.model_path = str(model_path)
        self.tag = tag
        self.dir = store_dir(model_path, tag, root)
        self.journal_path = self.dir / "journal.jsonl"
        self.core_path = self.dir / "core.safetensors"
        self.meta_path = self.dir / "core.json"

    # ---- the journal ----------------------------------------------------

    def append_session(self, user_texts, ack_text=None):
        """Add one session's user turns. Returns its id.

        `ack_text` is recorded but does not enter the core: the core always
        renders `ACK`, which is the acknowledgement the acceptance passed
        with. The real reply is kept because the journal is the record and a
        later policy may want it -- discarding it would be the one thing this
        file must never do.
        """
        if isinstance(user_texts, str):
            user_texts = [user_texts]
        texts = [t.strip() for t in user_texts if t and t.strip()]
        if not texts:
            raise ValueError("a session needs at least one non-empty user turn")
        rec = {"id": len(self.journal()),
               "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "user": texts, "ack": ack_text}
        self.dir.mkdir(parents=True, exist_ok=True)
        with self.journal_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec["id"]

    def journal(self):
        """Every session, oldest first. A half-written last line is skipped.

        Skipped and reported, never rewritten: an append killed halfway leaves
        a partial line, and truncating the file to "repair" it would delete a
        session to save a byte.
        """
        if not self.journal_path.exists():
            return []
        out = []
        for n, line in enumerate(self.journal_path.read_text(
                encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"  kvmem: journal line {n + 1} is unreadable, skipping "
                      f"it (the line is left on disk)")
        return out

    # ---- selection: pure, model-free, testable ---------------------------

    def plan(self):
        """Which sessions and which lines form the core. No model, no engine.

        Two lexical rules, in this order:

          write gate    a line survives when one of its words appears in at
                        most WRITE_DF_FRAC of the sessions. Rarity is the
                        signal that measured AUC 1.000 as a write gate, ahead
                        of the model's own surprise (0.985) and free.
          supersession  among the surviving lines, a later one that shares two
                        rare stems with an earlier one retires the earlier one.
                        Without it the model has to guess which of two
                        statements is current, and it guessed wrong on 2 of 3
                        updates.

        Returns the kept sessions, the lines each keeps, the lines
        supersession retired (flagged, not deleted) and a sha of the core text
        so a caller can ask "did this change" without rebuilding.
        """
        recs = self.journal()
        sessions = []
        for r in recs:
            lines = [ln.strip() for t in r["user"] for ln in t.split("\n")
                     if ln.strip()]
            sessions.append({"i": r["id"], "body": "\n".join(lines),
                             "lines": lines})
        if not sessions:
            return {"keep": [], "sessions_total": 0, "superseded": [],
                    "lines_seen": 0, "sha": ""}

        rare_write = rare_words(sessions, WRITE_DF_FRAC) - PRONOUNS
        gated = []
        for x in sessions:
            # A fact is a declarative: a user's questions ("Как меня зовут?")
            # carry the pronoun and rare words but state nothing.
            keep = [ln for ln in x["lines"]
                    if FIRST_PERSON.search(ln)
                    and not ln.rstrip().endswith("?")
                    and set(content_words(ln)) & rare_write]
            if keep:
                gated.append((x["i"], keep))

        rare_sup = rare_words(sessions, SUPERSEDE_DF_FRAC) - PRONOUNS
        kept, n_dropped = supersede(gated, rare_sup, 2)
        # supersede reports only how many it dropped, so recover the lines by
        # difference, by multiset rather than set because real journals repeat
        # lines verbatim and a set difference would under-count what it dropped.
        after = {i: Counter(ls) for i, ls in kept}
        superseded = []
        for i, ls in gated:
            left = Counter(after.get(i, ()))
            for ln in ls:
                if left[ln] > 0:
                    left[ln] -= 1
                else:
                    superseded.append({"session": i, "line": ln})
        assert len(superseded) == n_dropped, (len(superseded), n_dropped)

        body = "\n".join(f"{i}\n" + "\n".join(ls) for i, ls in kept)
        return {"keep": kept, "sessions_total": len(sessions),
                "superseded": superseded,
                "lines_seen": sum(len(x["lines"]) for x in sessions),
                "sha": hashlib.sha256(body.encode()).hexdigest()[:16]}

    # ---- the engine path -------------------------------------------------

    def rebuild(self, llm, chunk=512):
        """Prefill the core, quantise the state, write it atomically.

        A FRESH prefill, so the recurrent state saw exactly the tokens the core
        holds and nothing that was gated out. `zero=False` because the state is
        half the memory: zeroing it took recall from 8/8 to 0/8.
        """
        p = self.plan()
        if not p["keep"]:
            note("  kvmem: nothing passes the write gate yet, core unchanged")
            return dict(p, tokens=0, written=False)
        chunks = render(llm, p["keep"])
        ids = [t for c, _ in chunks for t in c]
        t0 = time.perf_counter()
        cache, _ = prefill_chunked(llm.model, ids, chunk, zero=False)
        build_s = round(time.perf_counter() - t0, 2)
        return self._write(cache, ids, p, build_s, llm)

    def _write(self, cache, ids, plan, build_s, llm):
        """core.safetensors + core.json. Degrades the live state to match disk.

        The in-memory state is replaced by its own int8 round trip, so the
        cache the caller keeps holding is bit-identical to the one `attach`
        would load. Otherwise a rebuild-then-ask measures fp32 and the next
        launch measures int8, and the two disagree by an amount nobody bounded.
        """
        tensors = {"ids": mx.array(np.array(ids, dtype=np.int32))}
        layers, gdn_dtypes, kv_bytes, state_bytes = [], {}, 0, 0
        for i, c in enumerate(cache):
            if isinstance(c, ArraysCache):
                layers.append(len(c.cache))
                slots = []
                for slot, a in enumerate(c.cache):
                    if a is None:
                        slots.append(None)
                        continue
                    q, s = _quant8(a)
                    name = str(a.dtype).rsplit(".", 1)[-1]
                    if name not in _DTYPES:
                        # Refuse to write a core `load` could not restore. A
                        # dtype nobody anticipated is a checkpoint change, and
                        # discovering it at the next launch -- as a KeyError on
                        # a memory the user thinks is saved -- is the worst
                        # place to discover it.
                        raise TypeError(f"the recurrent state is {name}; kvmem "
                                        f"can store {sorted(_DTYPES)}")
                    tensors[f"gdn.{i}.{slot}.q"] = q
                    tensors[f"gdn.{i}.{slot}.s"] = mx.array(
                        np.array([s], dtype=np.float32))
                    gdn_dtypes[f"{i}.{slot}"] = name
                    state_bytes += q.size + 4      # +4 for the scale
                    slots.append(_dequant8(q, s, a.dtype))
                c.cache = slots
            else:
                layers.append("kv")
                k, v = real_state(c)
                tensors[f"kv.{i}.keys"], tensors[f"kv.{i}.values"] = k, v
                kv_bytes += k.nbytes + v.nbytes
        mx.eval(list(tensors.values()))
        mx.eval([a for c in cache for a in cache_arrays(c)])

        meta = {
            "model": self.model_path, "tag": self.tag,
            "index_sha": index_sha(self.model_path),
            "render_sha": render_sha(llm),
            "layers": layers, "gdn_dtypes": gdn_dtypes,
            "tokens": len(ids), "kv_bytes": int(kv_bytes),
            "state_bytes": int(state_bytes),
            "total_bytes": int(kv_bytes + state_bytes),
            "sessions": [i for i, _ in plan["keep"]],
            "sessions_total": plan["sessions_total"],
            "superseded": plan["superseded"],
            "plan_sha": plan["sha"], "build_s": build_s,
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self.dir.mkdir(parents=True, exist_ok=True)
        _atomic(self.core_path,
                lambda t: mx.save_safetensors(str(t), tensors))
        _atomic(self.meta_path,
                lambda t: t.write_text(json.dumps(meta, indent=2,
                                                  ensure_ascii=False)))
        return dict(meta, written=True)

    def load(self):
        """(cache, ids, meta) straight off disk, or (None, None, why).

        No model needed: which layer is attention and which is a GatedDeltaNet
        block is recorded in `core.json`, and `attach` is what checks that
        against the engine actually in front of it.
        """
        if not (self.core_path.exists() and self.meta_path.exists()):
            return None, None, "no core saved yet"
        try:
            meta = json.loads(self.meta_path.read_text())
            d = mx.load(str(self.core_path))
        except Exception as e:
            return None, None, f"core is unreadable ({type(e).__name__})"
        ids = [int(x) for x in np.array(d["ids"])]
        if len(ids) != meta.get("tokens"):
            # core.json and core.safetensors are renamed one after the other,
            # so a kill between the two renames leaves them one version apart.
            # Cheap to detect, and a stale pairing is not repairable here.
            return None, None, (f"core.json says {meta.get('tokens')} tokens "
                                f"and core.safetensors holds {len(ids)}; "
                                f"rebuild")
        cache = []
        for i, kind in enumerate(meta["layers"]):
            if kind == "kv":
                c = GapKVCache(0)
                set_kv(c, d[f"kv.{i}.keys"], d[f"kv.{i}.values"])
            else:
                c = ArraysCache(size=int(kind))
                c.cache = [
                    _dequant8(d[f"gdn.{i}.{s}.q"],
                              float(np.array(d[f"gdn.{i}.{s}.s"])[0]),
                              _DTYPES[meta["gdn_dtypes"][f"{i}.{s}"]])
                    if f"gdn.{i}.{s}.q" in d else None
                    for s in range(int(kind))]
            cache.append(c)
        return cache, ids, meta

    def attach(self, llm):
        """Install the core into the engine. True when it took.

        The four lines at the end are the acceptance probe's, and they are
        four because the engine needs all four: the cache holds the keys,
        `_seen` says how many tokens they are (the invariant `offset - gap ==
        len(_seen)` is asserted inside `stub="slice"` after every turn), a
        non-empty `history` is what makes the next question render as a
        CONTINUATION of a finished exchange rather than a first turn, and
        `_ended_cleanly` False makes the separator complete the assistant turn.
        """
        cache, ids, meta = self.load()
        if cache is None:
            note(f"  kvmem: {meta}")
            return False
        llm.reset()
        live = ["kv" if not isinstance(c, ArraysCache) else len(c.cache)
                for c in llm._cache]
        why = None
        if live != meta["layers"]:
            why = (f"the core has {len(meta['layers'])} layers, this model has "
                   f"{len(live)}")
        elif meta.get("index_sha") != index_sha(self.model_path):
            why = (f"the core was built on packing {meta.get('index_sha')}, "
                   f"this model is {index_sha(self.model_path)}")
        elif meta.get("render_sha") != render_sha(llm):
            why = "the core was built with a different chat template or ack"
        if why:
            print(f"  kvmem: ignoring the core -- {why}. Rebuild with /sleep.")
            return False
        llm._cache = cache
        llm._seen = list(ids)
        llm.history = [{"role": "user", "content": "..."},
                       {"role": "assistant", "content": ACK}]
        llm._ended_cleanly = False
        return True

    # ---- reporting -------------------------------------------------------

    def ensure(self, llm):
        """Attach the core, rebuilding it from the journal when disk refuses.

        `attach` refuses a core built by another rendering (template, ack set,
        layer layout) -- that guard is right, but a refusal used to leave the
        engine with no memory until a NEW fact triggered a rebuild: changing
        ACKS on 2026-09-02 silently disabled every existing core. The journal
        holds everything the core is made of, so the fix is to rebuild here.
        """
        if self.attach(llm):
            return True
        if not self.plan()["keep"]:
            return False
        note("  kvmem: rebuilding the core from the journal", flush=True)
        self.rebuild(llm)
        return self.attach(llm)

    def core_text(self):
        """The kept lines as one text: what `ground()` checks a reply against."""
        return "\n".join(ln for _, ls in self.plan()["keep"] for ln in ls)

    def stats(self):
        """What is on disk, without opening the model."""
        recs = self.journal()
        out = {"tag": self.tag, "dir": str(self.dir), "core": False,
               "tokens": 0, "mb": 0.0, "sessions_kept": 0,
               "sessions_total": len(recs), "superseded": 0,
               "last_rebuilt": None, "plan_sha": None}
        if not self.meta_path.exists():
            return out
        try:
            meta = json.loads(self.meta_path.read_text())
        except json.JSONDecodeError:
            return out
        out.update(core=True, tokens=meta.get("tokens", 0),
                   mb=round(meta.get("total_bytes", 0) / 1e6, 1),
                   sessions_kept=len(meta.get("sessions", [])),
                   superseded=len(meta.get("superseded", [])),
                   last_rebuilt=meta.get("ts"), plan_sha=meta.get("plan_sha"))
        return out

    def summary(self):
        """One line for a startup banner."""
        s = self.stats()
        if not s["core"]:
            return (f"memory [{s['tag']}]: no core yet, "
                    f"{s['sessions_total']} sessions journalled")
        return (f"memory [{s['tag']}]: {s['tokens']} tokens / {s['mb']} MB from "
                f"{s['sessions_kept']} of {s['sessions_total']} sessions, "
                f"{s['superseded']} lines superseded, built {s['last_rebuilt']}")


# --------------------------------------------------------------------------
# selftest: no model, no GPU work beyond a few kilobytes of toy arrays
# --------------------------------------------------------------------------

class _CharTokenizer:
    """One id per character, enough for `render` and nothing else."""

    def encode(self, s, add_special_tokens=True):
        return [1 + (ord(c) % 100) for c in s]


class _FakeLLM:
    """Tokenizer, template and cache layout, which is all that `attach` touches."""

    def __init__(self, layers=(2, "kv", "kv", 2), tokens=7, tpl=None):
        self.tokenizer = _CharTokenizer()
        self._tpl = tpl or {"head": "<u>", "after_user": "</u><a>",
                            "between": "</a><u>", "tail": "</u><a>"}
        self._layout, self._tokens = layers, tokens
        self.reset()

    def reset(self):
        self._cache = []
        for kind in self._layout:
            if kind == "kv":
                self._cache.append(GapKVCache(0))
            else:
                c = ArraysCache(size=int(kind))
                c.cache = [None] * int(kind)
                self._cache.append(c)
        self._seen, self.history, self._ended_cleanly = [], [], False
        self._gap = 0

    def fill(self, seed=0):
        """A cache with plausible contents, built on the CPU stream."""
        rng = np.random.default_rng(seed)
        with mx.stream(mx.cpu):
            for c in self._cache:
                if isinstance(c, ArraysCache):
                    c.cache = [mx.array(rng.normal(0, 1, (1, 4, 6)
                                                   ).astype(np.float32)),
                               mx.array(rng.normal(0, 3, (1, 2, 8, 8)
                                                   ).astype(np.float32))]
                else:
                    shape = (1, 2, self._tokens, 4)
                    set_kv(c, mx.array(rng.normal(0, 1, shape).astype(np.float32)),
                           mx.array(rng.normal(0, 1, shape).astype(np.float32)))
            mx.eval([a for c in self._cache for a in cache_arrays(c)])
        return self._cache


def _probe_fixture():
    """An optional acceptance fixture, or None when the probes are not installed."""
    try:
        import lifelong_probe
    except ImportError:
        return None
    return lifelong_probe


def _probe_sessions(n=20, statements=10):
    """The acceptance transcript, built through the probe's own fixture."""
    import random

    LP = _probe_fixture()
    llm = LP._fast_fixture()
    LP._ORACLE_LLM[0] = llm

    class A:
        sessions, statements_, facts, updates = n, statements, 12, 3
        budget_tokens, budget_mb, min_recall = 400, 64.0, 10
        chunk, max_tokens, kv_group = 32, 4, 32
        verbose, supersede, idf_min = False, False, 2.0
        supersede_shared = 2

    A.statements = statements
    return llm, A, *LP.build_transcript(llm, A(), random.Random(0))


def _transcript_gate(root):
    """Sections 2 to 4b of the selftest, on the probe's synthetic year."""
    import random

    if _probe_fixture() is None:
        note("  transcript: skipped, the optional probes are not installed")
        return None
    # The write gate on 20 sessions of 10 statements, with 12 facts and 3 updates.
    llm, A, sessions, facts = _probe_sessions(20, 10)
    st = Store("/nowhere/toy-model", tag="acc", root=root)
    for x in sessions:
        st.append_session(x["body"].split("\n"))
    p = st.plan()
    # The gate keeps exactly the fact lines, and supersession empties two sessions.
    want = {x["i"]: x["fact_lines"] for x in sessions if x["fact_lines"]}
    dropped = {(d["session"], d["line"]) for d in p["superseded"]}
    expect = [(i, [ln for ln in want[i] if (i, ln) not in dropped])
              for i in sorted(want)]
    expect = [(i, ls) for i, ls in expect if ls]
    assert p["keep"] == expect, (p["keep"], expect)
    assert len(p["keep"]) == len(want) - 2, \
        (len(p["keep"]), len(want), "supersession emptied a different number "
         "of sessions")
    assert p["lines_seen"] == sum(len(x["body"].split("\n")) for x in sessions)

    # At two shared rare stems supersession catches the engine and colour
    # updates, not the sport one.
    caught = {d["line"] for d in p["superseded"]}
    olds = {f["key"]: f["text"] for f in facts if "old_exp" in f}
    assert olds["engine"] in caught and olds["colour"] in caught, caught
    assert olds["sport"] not in caught, "the one-concept pair should not catch"
    assert len(p["superseded"]) == 2, p["superseded"]
    # Retired, not deleted: every superseded line is still in the journal.
    body = st.journal_path.read_text(encoding="utf-8")
    for d in p["superseded"]:
        assert d["line"] in body, d

    # With supersession off the core is the oracle's plan, token for token.
    LP = _probe_fixture()
    LP._ORACLE_LLM[0] = llm
    orc, _ = LP.policy_plan("oracle_rebuilt", sessions, facts, A(),
                            random.Random(1))
    mine = render(llm, [(i, want[i]) for i in sorted(want)], ack=ACK)
    assert mine == orc, "kvmem renders a session differently from the oracle"
    # The default rotates acknowledgements, so kept sessions do not close
    # the same way.
    varied = render(llm, [(i, want[i]) for i in sorted(want)])
    assert len(varied) == len(mine) and varied != mine
    tails = [tuple(v[0][len(m[0]) - len(ACK):]) for v, m in zip(varied, mine)]
    assert len(set(tails[:len(ACKS)])) == min(len(tails), len(ACKS)), tails
    with_sup = render(llm, p["keep"], ack=ACK)
    assert sum(len(i) for i, _ in with_sup) < sum(len(i) for i, _ in orc), \
        "supersession did not shrink the core"
    assert with_sup[0][1] == 0 and all(
        b[1] == a[1] + len(a[0]) for a, b in zip(with_sup, with_sup[1:])), \
        "the core must be at consecutive positions"

    # On a 12 by 4 transcript document frequency has nothing to separate,
    # so the gate keeps most of the filler.
    llm2, _, small, _ = _probe_sessions(12, 4)
    tiny = Store("/nowhere/toy-model", tag="tiny", root=root)
    for x in small:
        tiny.append_session(x["body"].split("\n"))
    tp = tiny.plan()
    kept_small = sum(len(ls) for _, ls in tp["keep"])
    assert 0.3 * tp["lines_seen"] < kept_small < 0.6 * tp["lines_seen"], \
        (kept_small, tp["lines_seen"], "the small-transcript floor moved")

    return p


def selftest():
    import shutil
    import tempfile

    root = Path(tempfile.mkdtemp(prefix="kvmem-selftest-"))
    try:
        _selftest_body(root)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _selftest_body(root):
    # 1. Journal round trip. Append-only, ids dense, a session read back
    #    exactly as it went in -- including a multi-line turn, which `plan`
    #    later splits and the journal must not.
    s = Store("/nowhere/toy-model", tag="t", root=root)
    assert s.journal() == [] and s.stats()["sessions_total"] == 0
    assert s.append_session("Меня зовут Антон.") == 0
    assert s.append_session(["Я живу в Казани.", "Моего кота зовут Барсик."],
                            ack_text="Записал.") == 1
    assert s.append_session("а\nб") == 2
    j = s.journal()
    assert [r["id"] for r in j] == [0, 1, 2], j
    assert j[1]["user"] == ["Я живу в Казани.", "Моего кота зовут Барсик."]
    assert j[1]["ack"] == "Записал." and j[0]["ack"] is None
    assert datetime.fromisoformat(j[0]["ts"]).tzinfo is not None
    try:
        s.append_session(["   "])
        raise AssertionError("an empty session was accepted")
    except ValueError:
        pass
    # A truncated final line (a kill mid-append) must cost that line and
    # nothing else.
    with s.journal_path.open("a", encoding="utf-8") as f:
        f.write('{"id": 3, "user": ["обр')
    assert [r["id"] for r in s.journal()] == [0, 1, 2], "a good line was lost"

    p = _transcript_gate(root)

    # 4c. Real prose instead of the generator: technical sentences are dense
    #     in rare words, so rarity alone kept 985 of 985 of them at 100
    #     sessions. The conjunction must drop every third-person line, keep
    #     every first-person fact, and a pronoun-only overlap must not retire
    #     a fact ("Как меня зовут?" shares {меня, зовут} with "Меня зовут
    #     Антон" and is about nothing).
    prose = ["Роутер выбирает восемь экспертов из двухсот пятидесяти шести.",
             "Синхронизация по слоям стоит больше, чем байты с накопителя.",
             "Арена держит двадцать один эксперт на слой и вытесняет по частоте.",
             "Как меня зовут?",
             "Квантование состояния в восемь бит проходит все шесть критериев.",
             "Предвыборка по лукахеду не убирает синхронизацию по слою.",
             "Меня зовут Антон.", "Мой кот Барсик любит спать на клавиатуре.",
             "Раскладка файла экспертов имеет два разных шага строки.",
             "Отложенная проверка пересчитывает шестьдесят процентов токенов."]
    real = Store("/nowhere/toy-model", tag="prose", root=root)
    for k in range(0, len(prose), 2):
        real.append_session(prose[k:k + 2])
    rp = real.plan()
    kept_prose = [ln for _, ls in rp["keep"] for ln in ls]
    assert set(kept_prose) == {"Меня зовут Антон.",
                               "Мой кот Барсик любит спать на клавиатуре."}, kept_prose
    assert not rp["superseded"], rp["superseded"]

    # 5. Grounding: invented names, e-mails and confident negatives are caught,
    #    answers made of the core are not.
    core = "Меня зовут Антон.\nМоего кота зовут Барсик, ему девять лет.\n" \
           "У меня ноутбук ThinkPad Carbon X1 на 16 гигабайт."
    for ok_reply in ("Вашего кота зовут **Барсик**.", "У вас ThinkPad Carbon X1 на 16 гигабайт.",
                     "Вашему коту Барсику девять лет.", "Вас зовут Антон."):
        assert ground(ok_reply, core)[0], ok_reply
    for bad_reply in ("У вас автомобиль **Mercedes-Benz G-Class**.",
                      "Ваш адрес электронной почты: anton@example.com",
                      "У вас нет детей.", "Ваш номер телефона: +7 999 123-45-67."):
        ok, why = ground(bad_reply, core)
        assert not ok, bad_reply
    assert ground("У вас нет кота по имени Мурзик, у вас Барсик.", core)[0] is False
    assert ground("Вашему коту 9 лет.", core)[0], "digit form of a core number"
    assert not ground("У вас **двое детей**.", core)[0], "an invented count"
    assert ground("У вас 16 гигабайт памяти.", core)[0]
    assert not ground("Судя по всему, вы предпочитаете спокойную музыку.", core)[0]
    # Refusals are not checked, and capitalised function words are not names.
    assert ground("Вы не указали, какой у вас размер обуви: 36 или 45?", core)[0]
    assert ground("Не знаю. Какой у вас автомобиль?", core)[0]
    # A stored fact transplanted onto an unstated relative is caught by the
    # question's subject, not by the words (they are all in the core).
    core2 = core + "\nЯ занимаюсь плаванием."
    assert not ground("Ваш брат занимается плаванием.", core2,
                      question="Каким спортом занимается мой брат?")[0]
    assert ground("Вы занимаетесь плаванием.", core2,
                  question="Каким спортом я занимаюсь?")[0]
    assert ground("Вашего брата зовут Олег.", core2 + "\nМоего брата зовут Олег.",
                  question="Как зовут моего брата?")[0]
    if _MORPH is not None:
        assert not ground("Вы играете на гитаре.", core)[0], "an invented noun"
        assert ground("Вашему коту Барсику девять лет.", core)[0]
        assert ground("Вы живёте с котом Барсиком.", core)[0], "inflection must not flag"
        assert not ground("У вас две собаки.", core)[0]
    assert ground("Это не указано. Если хотите, сообщите: Пожалуйста, уточните.", core)[0]
    assert ground("Судя по вашим словам, ваш кот Барсик любит спать.", core)[0]

    # 5. The stored int8 must match the reference round trip exactly, or the
    #    bytes on disk change meaning.
    with mx.stream(mx.cpu):
        rng = np.random.default_rng(3)
        for shape in ((1, 4, 6), (1, 2, 8, 8), (3,)):
            a = mx.array(rng.normal(0, 2.5, shape).astype(np.float32))
            q, sc = _quant8(a)
            assert q.dtype == mx.int8 and bool(
                mx.all(mx.abs(q.astype(mx.float32)) <= 127).item())
            mine_q = _dequant8(q, sc, a.dtype)
            theirs = degrade_state(a, 8)
            assert bool(mx.all(mine_q == theirs).item()), \
                "kvmem's int8 is not the reference round trip"
        z = mx.zeros((2, 2))
        assert bool(mx.all(_dequant8(*_quant8(z), mx.float32) == z).item())

    # 6. The core round trip: KV back bit for bit, the state back to its own
    #    int8 value, the token ids back as integers.
    fake = _FakeLLM()
    cache = fake.fill(seed=1)
    ids = list(range(11, 11 + 7))
    core = Store("/nowhere/toy-model", tag="core", root=root)
    core.append_session("Меня зовут Антон.")
    before = {i: [None if a is None else mx.array(a) for a in c.cache]
              for i, c in enumerate(cache) if isinstance(c, ArraysCache)}
    kv_before = {i: real_state(c) for i, c in enumerate(cache)
                 if not isinstance(c, ArraysCache)}
    info = core._write(cache, ids, core.plan(), 1.0, fake)
    assert info["written"] and core.core_path.exists()
    got, got_ids, meta = core.load()
    assert got_ids == ids, (got_ids, ids)
    assert meta["tokens"] == len(ids) and meta["render_sha"] == render_sha(fake)
    for i, c in enumerate(got):
        if isinstance(c, ArraysCache):
            # the live cache was degraded in place, so disk == memory exactly
            for a, b in zip(before[i], c.cache):
                assert bool(mx.all(degrade_state(a, 8) == b).item()), \
                    f"state slot of layer {i} did not round-trip"
            for a, b in zip(cache[i].cache, c.cache):
                assert bool(mx.all(a == b).item()), \
                    "the live cache and the file disagree"
        else:
            k, v = real_state(c)
            k0, v0 = kv_before[i]
            assert k.shape == k0.shape and v.shape == v0.shape
            assert bool(mx.all(k == k0).item()) and bool(mx.all(v == v0).item()), \
                f"KV of layer {i} is not bit-identical after a round trip"
            assert c.offset == len(ids), (c.offset, len(ids))
    # Byte accounting: int8 plus one scale per tensor, and KV unquantised.
    n_state = sum(a.size for c in cache if isinstance(c, ArraysCache)
                  for a in c.cache if a is not None)
    n_tensors = sum(1 for c in cache if isinstance(c, ArraysCache)
                    for a in c.cache if a is not None)
    assert info["state_bytes"] == n_state + 4 * n_tensors, info["state_bytes"]
    assert info["kv_bytes"] == sum(k.nbytes + v.nbytes
                                   for k, v in kv_before.values())

    # 7. attach installs all four pieces, and refuses anything that does not
    #    match this engine.
    fresh = _FakeLLM()
    assert core.attach(fresh) is True
    assert fresh._seen == ids
    assert len(fresh.history) == 2 and fresh.history[-1]["content"] == ACK
    assert fresh._ended_cleanly is False
    for c in fresh._cache:
        if not isinstance(c, ArraysCache):
            # the invariant stub="slice" asserts after every turn
            assert c.offset - getattr(c, "gap", 0) == len(fresh._seen)
    assert Store("/nowhere/toy-model", tag="empty", root=root).attach(
        _FakeLLM()) is False, "attaching a missing core must be a no-op"
    assert core.attach(_FakeLLM(layers=("kv", "kv"))) is False, \
        "a model with a different layer layout must be refused"
    assert core.attach(_FakeLLM(tpl={"head": "[U]", "after_user": "[A]",
                                     "between": "[U]", "tail": "[A]"})) is False, \
        "a different chat template must be refused"
    # A core whose two files drifted apart (killed between the two renames).
    bad = json.loads(core.meta_path.read_text())
    bad["tokens"] = 999
    core.meta_path.write_text(json.dumps(bad))
    assert core.attach(_FakeLLM()) is False, "a torn core must be refused"
    core.meta_path.write_text(json.dumps(dict(bad, tokens=len(ids))))

    # 8. The atomic write: a writer that dies leaves the OLD core untouched and
    #    no debris beside it.
    keep_bytes = core.core_path.read_bytes()
    keep_meta = core.meta_path.read_text()

    def die(tmp):
        tmp.write_bytes(b"half a core")
        raise KeyboardInterrupt("killed mid-write")

    try:
        _atomic(core.core_path, die)
        raise AssertionError("_atomic swallowed the interrupt")
    except KeyboardInterrupt:
        pass
    assert core.core_path.read_bytes() == keep_bytes, "the old core was damaged"
    assert not list(core.dir.glob("*.tmp.*")), "a tmp file was left behind"
    assert core.attach(_FakeLLM()) is True, "the old core is no longer usable"
    assert core.meta_path.read_text() == keep_meta

    # 9. stats, and the one line a session prints at startup.
    s9 = core.stats()
    assert s9["core"] and s9["tokens"] == len(ids) and s9["sessions_total"] == 1
    assert s9["mb"] == round(info["total_bytes"] / 1e6, 1)
    assert s9["last_rebuilt"] and "kvmem" in s9["dir"]
    assert "no core yet" in Store("/x", tag="n", root=root).summary()
    assert core.summary().startswith("memory [core]:")

    print(f"selftest ok -- core round-trips {len(ids)} tokens, "
          f"{info['kv_bytes']} B KV + {info['state_bytes']} B state"
          + (f"; the gate keeps {len(p['keep'])} of {p['sessions_total']} "
             f"sessions and {sum(len(l) for _, l in p['keep'])} of "
             f"{p['lines_seen']} lines on the acceptance transcript, "
             f"supersedes {len(p['superseded'])}, rendering matches the oracle"
             if p is not None else "; the transcript was skipped"))


def main():
    ap = argparse.ArgumentParser()
    from defaults import MODEL
    ap.add_argument("--model", default=MODEL, required=MODEL is None, help="checkpoint directory")
    ap.add_argument("--tag", default="default")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--plan", action="store_true",
                    help="print what the write gate would keep, model-free")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    store = Store(args.model, args.tag)
    if args.plan:
        p = store.plan()
        for i, lines in p["keep"]:
            print(f"  session {i}:")
            for ln in lines:
                print(f"      {ln}")
        for d in p["superseded"]:
            print(f"  superseded (session {d['session']}): {d['line']}")
    print(store.summary())


if __name__ == "__main__":
    main()
