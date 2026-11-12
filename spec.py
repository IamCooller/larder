"""Speculative decoding for the streaming MoE engine with the model's own
multi-token-prediction head: the head proposes, the target verifies, and every emitted
token is the target's argmax."""
import time
from dataclasses import dataclass, field

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, KVCache, make_prompt_cache

import sampler


def _snapshot(cache):
    """What the round can undo: an offset per trimmable cache and the state list for the
    GatedDeltaNet blocks, which copy nothing and stay exact for a GapKVCache."""
    snap = []
    for c in cache:
        if isinstance(c, ArraysCache):
            snap.append(list(c.cache))
        elif c.is_trimmable():
            snap.append(c.offset)
        else:
            raise ValueError(f"speculation cannot roll a {type(c).__name__} back")
    return snap


def _restore(cache, snap):
    for c, s in zip(cache, snap):
        if isinstance(s, list):
            # A COPY, because one snapshot is restored more than once and ArraysCache
            # rebinds the list it is handed, so a later rollback would otherwise land on
            # an earlier one's state.
            c.cache = list(s)
        else:
            c.trim(c.offset - s)
            assert c.offset == s, (
                f"{type(c).__name__} refused to give {c.offset - s} positions back")


class _Discard(Exception):
    """A round built ahead whose round before it was rejected (MOE_SPEC_AHEAD), raised at a hand-off of its build."""


@dataclass
class Stats:
    rounds: int = 0
    accepted: int = 0
    tokens: int = 0
    replays: int = 0
    rebuilds: int = 0
    draft_s: float = 0.0
    verify_s: float = 0.0
    # Per draft position, the rounds that got as far as draft d and the ones that
    # accepted it, with gamma=1 making hit[0] the accepted count.
    seen: list = field(default_factory=list)
    hit: list = field(default_factory=list)
    # Grammar-forced rows (forced_rows.py) sent to verify passes and the ones accepted,
    # kept out of the head's own counts.
    forced: int = 0
    forced_accepted: int = 0
    # MOE_SPEC_AHEAD: rounds built ahead that became the next round, that were dropped, and of those cut at a hand-off.
    ahead_used: int = 0
    ahead_dropped: int = 0
    ahead_cut: int = 0

    def note(self, d, ok):
        while len(self.seen) <= d:
            self.seen.append(0)
            self.hit.append(0)
        self.seen[d] += 1
        self.hit[d] += bool(ok)
        self.accepted += bool(ok)

    @property
    def acceptance(self):
        """Accepted drafts per round, the fraction of rounds at gamma=1 and the mean
        accepted prefix at gamma>1."""
        return self.accepted / max(self.rounds, 1)

    @property
    def per_depth(self):
        """Conditional acceptance of drafts 1, 2, ..., the numbers an offline depth scan
        predicts."""
        return [h / max(s, 1) for s, h in zip(self.seen, self.hit)]

    @property
    def per_round(self):
        return self.tokens / max(self.rounds, 1)


@dataclass
class Piece:
    """The three fields moe_stream.stream() reads off an mlx_lm response.

    Not a GenerationResponse: the rest of that dataclass is per-token
    telemetry mlx_lm computes and this engine's caller ignores, and filling it
    in with plausible numbers would be inventing measurements.
    """
    token: int
    text: str
    finish_reason: str = None


class Speculator:
    """Speculative decoding, greedy and sampled alike: the target's argmax or its exact
    sampled distribution (sampler.verify)."""

    # The longest verification pass this will build, since the kernels stay flat to four
    # positions and every branch below keeps T within it.
    max_window = 4

    def __init__(self, model, head, gamma=1):
        if not 1 <= gamma <= self.max_window:
            raise ValueError(f"gamma must be 1..{self.max_window}; at gamma "
                             f"the pass is up to {self.max_window + 1} "
                             f"positions wide and the kernels stop paying")
        self.gamma = gamma
        self.model = model
        # The text model under a multimodal wrapper, inlined the way lora_grad.parts_of
        # does it.
        self.lm = getattr(model, "language_model", model)
        self.core = getattr(self.lm, "model", self.lm)
        self.embed = self.core.embed_tokens
        # A tied head (Qwen3-4B) is the embedding table read backwards.
        self.lm_head = (self.lm.lm_head if "lm_head" in self.lm
                        else self.embed.as_linear)
        self.head = head

    # sampler.Sampling for the turn, or None for greedy (moe_stream sets it per turn).
    sampling = None
    _q = ()               # the round's draft distributions, a list while a sampled round drafts
    # forced_rows.Forcer for the turn, or None, fed every emitted token and verified like
    # a draft.
    forcer = None

    def _logit_argmax(self, h):
        return int(mx.argmax(self.lm_head(h)[0, -1]))

    def _logit_argmax_rows(self, h, also=()):
        """The target's argmax at every position of a slab in one lm_head call and one host
        round trip."""
        am = mx.argmax(self.lm_head(h), axis=-1)
        mx.eval(am, *also)      # `also`: arrays the round wants in the same sync
        return am.tolist()

    def _draft_ids(self, h):
        """The draft's argmax as a one-element array, a proposal only."""
        if self.sampling is not None:
            # A sampled draft keeps the q it was drawn from for the verdict.
            q = self.sampling.logprobs(self.lm_head(h[:, -1]))
            self._q.append(q)
            return mx.random.categorical(q)
        return mx.argmax(self.lm_head(h[:, -1]), axis=-1)

    def _draft(self, pairs, hcache, st):
        """The head's guesses at the tokens after the last accepted one, its own method so
        a test can script them."""
        hs = mx.concatenate([p[0] for p in pairs], axis=1)
        toks = [p[1] for p in pairs]
        if any(isinstance(t, mx.array) for t in toks):
            # A round built ahead of its verdict (MOE_SPEC_AHEAD): the tokens are the last round's lazy ids.
            es = self.embed(mx.concatenate([t.reshape(1, -1).astype(mx.int32) for t in toks], axis=1))
        else:
            es = self.embed(mx.array([toks]))
        h = self.head(hs, es, cache=hcache)
        kept = hcache[0].offset
        # No round trip: `gamma` ids as one int32 array; generate() feeds
        # it to the verify pass as is and reads it back with the verdict.
        out = [self._draft_ids(h)]
        for _ in range(self.gamma - 1):
            h = self.head(h[:, -1:], self.embed(out[-1][None]), cache=hcache)
            out.append(self._draft_ids(h))
        if len(out) > 1:
            hcache[0].trim(hcache[0].offset - kept)
        return mx.concatenate(out).astype(mx.int32)

    def _judge(self, rows, drafts, n_head, st, eos, max_tokens):
        """A sampled round's verdict over the pass's rows: (drafts as ids, tokens to emit,
        drafts the cache keeps), with the same eos and budget stops as the greedy loop."""
        p = self.sampling.logprobs(self.lm_head(rows))
        n = len(drafts)
        d = drafts if isinstance(drafts, mx.array) else mx.array(drafts, dtype=mx.int32)
        q = None
        if n:
            # Head drafts were drawn from self._q; forced, carried or scripted ones are
            # certain.
            q = mx.where(mx.arange(p.shape[-1])[None] == d[:, None], 0.0, -mx.inf)
            m = min(len(self._q), n_head)
            if m:
                q = mx.concatenate([*self._q[:m], q[m:]], axis=0)
        stands, fix, bonus = sampler.verify(p, q, d)
        mx.eval(stands, fix, bonus, d)
        stands, fix, drafts = stands.tolist(), fix.tolist(), d.tolist()
        out, a = [], 0
        while a < n:
            if a < n_head:
                st.note(a, stands[a])
            else:
                st.forced_accepted += stands[a]
            if not stands[a]:
                out.append(fix[a])
                return drafts, out, a
            out.append(drafts[a])
            if drafts[a] in eos or st.tokens + len(out) >= max_tokens:
                return drafts, out, a       # out[a] is owed, as a correction would be
            a += 1
        out.append(bonus.item())
        return drafts, out, a

    # MOE_SPEC_AHEAD (stream_fast sets these on CUDA's native T-row pass): the next round is built while the GPU finishes
    # this one, as if every draft stands; `watch(fn)` has fn called after each hand-off of that build (None removes it),
    # where this round's verdict is looked for with `peek(n, count)` (its ids once `post(am, drafts)` -> (array, n) has
    # written them, else None) and read, waiting, at the `ahead`-th.
    ahead = 0
    watch = post = peek = None

    def _round(self, pending, pairs, hcache, cache, gdn, st, snap, lead=0):
        """MOE_SPEC_AHEAD: one greedy round built and handed to the GPU, its argmax rows and drafts lazy; `pending` is
        a [1, k] int32 array, `lead` the tokens the round before emits if every draft stands (what _draft is told)."""
        t0 = time.perf_counter()
        st.tokens += lead
        try:
            drafts = self._draft(pairs, hcache, st)
        finally:
            st.tokens -= lead
        if not isinstance(drafts, mx.array):
            drafts = mx.array(drafts, dtype=mx.int32)
        st.draft_s += time.perf_counter() - t0
        t0 = time.perf_counter()
        k = pending.shape[1]
        H, tape = self._taped(mx.concatenate([pending, drafts[None]], axis=1), cache, gdn)
        am = mx.argmax(self.lm_head(H[0, k - 1:k + drafts.size]), axis=-1)
        n = None
        if self.post is not None:
            mark, n = self.post(am, drafts)
            mx.async_eval(am, drafts, mark)
        else:
            mx.async_eval(am, drafts)
        st.verify_s += time.perf_counter() - t0
        return {"k": k, "drafts": drafts, "H": H, "tape": tape, "snap": snap, "am": am, "n": n, "verdict": None}

    def _verdict(self, r, st, eos, max_tokens, wait=True):
        """The round's tokens as the greedy loop below judges them: (out, a, each judged draft's hit, whether the
        round built ahead of it is the next one), read once; None while the GPU has not written them and `wait` is off."""
        if r["verdict"] is None:
            g = r["drafts"].size
            ids = self.peek(r["n"], 2 * g + 1) if r["n"] is not None else None
            if ids is not None:
                cand, drafts = ids[:g + 1], ids[g + 1:]
            elif not wait:
                return None
            else:
                mx.eval(r["am"], r["drafts"])
                cand, drafts = r["am"].tolist(), r["drafts"].tolist()
            out, a, hits = [cand[0]], 0, []
            while a < len(drafts) and out[a] not in eos and st.tokens + len(out) < max_tokens:
                hits.append(drafts[a] == out[a])
                if drafts[a] != out[a]:
                    break
                a += 1
                out.append(cand[a])
            goes = a == len(drafts) and out[-1] not in eos and st.tokens + len(out) < max_tokens
            r["verdict"] = (out, a, hits, goes)
        return r["verdict"]

    # Off replays a rejection instead of rebuilding, which is also what a model gets
    # whose recurrent caches are not a GatedDeltaNet pair.
    rebuild = True

    def _gdn_of(self, cache):
        """[(cache index, GatedDeltaNet, its in_proj* names)] if every recurrent cache is
        one layer's GatedDeltaNet (conv, state), else None. [] for attention only."""
        if not any(isinstance(c, ArraysCache) for c in cache):
            return []
        layers = getattr(self.lm, "layers", ())
        if len(layers) != len(cache):
            return None
        out = []
        for i, (layer, c) in enumerate(zip(layers, cache)):
            if isinstance(c, ArraysCache):
                m = getattr(layer, "linear_attn", None)
                if m is None or len(c.cache) != 2:
                    return None
                out.append((i, m, [n for n in m if n.startswith("in_proj")]))
        return out

    def _taped(self, inp, cache, gdn):
        """The verify pass, with every GatedDeltaNet's input and projections kept per cache
        index."""
        tape = {}
        for i, m, names in gdn:
            rec = tape[i] = {}
            for n in names:
                def proj(x, f=m[n], n=n, rec=rec):
                    rec["x"], rec[n] = x, f(x)
                    return rec[n]
                # Instance attribute: found before the module's child of the same name.
                m.__dict__[n] = proj
        try:
            return self.core(inp, cache=cache), tape
        finally:
            for _, m, names in gdn:
                for n in names:
                    del m.__dict__[n]

    def _rebuild(self, cache, snap, n, gdn, tape):
        """The cache as `snap` plus the pass's first n rows: attention keeps them and each
        GatedDeltaNet reruns its own call over them."""
        for c, s in zip(cache, snap):
            if not isinstance(s, list):
                c.trim(c.offset - s - n)
                assert c.offset == s + n, f"{type(c).__name__} kept {c.offset - s} of {n}"
        for i, m, names in gdn:
            rec, c = tape[i], cache[i]
            c.cache = list(snap[i])      # a copy, for _restore's reason
            m.__dict__.update({k: (lambda _, y=rec[k][:, :n]: y) for k in names})
            try:
                m(rec["x"][:, :n], None, c)
            finally:
                for k in names:
                    del m.__dict__[k]

    def generate(self, ids, max_tokens, eos=(), stats=None, cache=None,
                 prefill_step=0):
        """Greedy speculative decoding over a caller's live cache or over one this
        generator builds and discards."""
        st = stats if stats is not None else Stats()
        own = cache is None
        if own:
            cache = make_prompt_cache(self.model)
        hcache = [KVCache()]

        # Chunked for the reason moe_stream._prefill gives: a wide prefill makes
        # _batch_wide allocate a buffer spanning nearly every expert.
        step = prefill_step or len(ids) or 1
        for j in range(0, len(ids), step):
            h = self.core(mx.array([ids[j:j + step]]), cache=cache)[:, -1:]
        tok = (self._logit_argmax(h) if self.sampling is None else int(mx.random.categorical(
            self.sampling.logprobs(self.lm_head(h)[0, -1]))))

        # `emitted` is every token yielded and `fed` how many the target has seen, so
        # emitted[fed:] is the tail still owed to the cache.
        emitted, fed = [], 0
        snap = _snapshot(cache)
        gdn = self._gdn_of(cache) if self.rebuild else None
        pairs = [(h, tok)]              # not yet seen by the draft head
        forcer = self.forcer
        ahead = bool(self.ahead and self.watch is not None and self.sampling is None and forcer is None
                     and gdn is not None)
        cur = None      # MOE_SPEC_AHEAD: the round handed to the GPU and not yet judged
        held = None     # (snapshot, head cache offset) under a round built ahead and still in the cache
        try:
            if forcer is not None:
                forcer.feed(tok)
            yield tok
            emitted.append(tok)
            st.tokens += 1

            while st.tokens < max_tokens and emitted[-1] not in eos:
                if ahead:
                    held = None
                    if cur is None:
                        cur = self._round(mx.array([emitted[fed:]], dtype=mx.int32), pairs, hcache, cache, gdn, st,
                                          snap)
                    g = cur["drafts"].size
                    nxt = pre = None
                    if st.tokens + g + 1 < max_tokens:
                        # The next round on this one's lazy ids as if every draft stands, built while the GPU finishes
                        # this one; each hand-off looks for this round's verdict, the `ahead`-th waits for it, and a
                        # rejection stops the build there.
                        pre, hoff = _snapshot(cache), hcache[0].offset
                        held, seen = (pre, hoff), [0]

                        def check(r=cur, seen=seen):
                            seen[0] += 1
                            v = self._verdict(r, st, eos, max_tokens, wait=seen[0] >= self.ahead)
                            if v is not None and not v[3]:
                                raise _Discard
                        toks = mx.concatenate([cur["drafts"], cur["am"][g:].astype(mx.int32)])
                        opt = [(cur["H"][:, cur["k"] - 1 + j:cur["k"] + j], toks[j:j + 1]) for j in range(g + 1)]
                        self.watch(check)
                        try:
                            nxt = self._round(toks[None, g:], opt, hcache, cache, gdn, st, pre, lead=g + 1)
                            # Its verdict kernel may overwrite this round's ids in the block: read the arrays from here.
                            cur["n"] = None
                        except _Discard:
                            st.ahead_cut += 1
                        finally:
                            self.watch(None)
                    out, a, hits, goes = self._verdict(cur, st, eos, max_tokens)
                    st.rounds += 1
                    for j, ok in enumerate(hits):
                        st.note(j, ok)
                    if nxt is not None and goes:
                        # Every draft stood: the round built ahead is the next one, its snapshot the one the loop
                        # below would take now.
                        fed = len(emitted) + a
                        snap, cur = pre, nxt
                        st.ahead_used += 1
                    else:
                        if pre is not None:
                            _restore(cache, pre)
                            hcache[0].trim(hcache[0].offset - hoff)
                            held = None
                            st.ahead_dropped += 1
                        pairs = [(cur["H"][:, cur["k"] - 1 + j:cur["k"] + j], out[j]) for j in range(len(out))]
                        if a < g:
                            self._rebuild(cache, cur["snap"], cur["k"] + a, gdn, cur["tape"])
                            st.rebuilds += 1
                        fed = len(emitted) + a
                        snap, cur = _snapshot(cache), None
                    for t in out:
                        yield t
                        emitted.append(t)
                        st.tokens += 1
                    continue
                t0 = time.perf_counter()
                pending = emitted[fed:]
                k = len(pending)
                # Rows the pass can still take, max_window + 1 positions with `pending`
                # included, checked every round.
                room = self.max_window + 1 - k
                forced = forcer.propose()[:room] if forcer is not None else []
                g = 0 if forced else self.gamma
                # When the grammar decides the round's rows the head is not
                # asked; the positions it has not seen wait in `pairs`.
                self._q = []
                drafts = self._draft(pairs, hcache, st) if g else []
                drafts = drafts[:min(g, room)] if g else []
                n_head = len(drafts)
                if n_head and n_head < room and forcer is not None and forcer.active:
                    # The grammar may force what follows the head's guesses, and that
                    # span rides in the same pass.
                    f = forcer.fork()
                    drafts = drafts.tolist() if isinstance(drafts, mx.array) else list(drafts)
                    for t in drafts:
                        f.feed(t)
                    drafts += f.propose()[:room - n_head]
                drafts = forced or drafts
                st.forced += len(drafts) - n_head
                st.draft_s += time.perf_counter() - t0

                t0 = time.perf_counter()
                on_gpu = isinstance(drafts, mx.array)
                if on_gpu:
                    inp = mx.concatenate([mx.array([pending], dtype=mx.int32),
                                          drafts[None]], axis=1)
                else:
                    inp = mx.array([pending + drafts])
                H, tape = self._taped(inp, cache, gdn or [])
                st.rounds += 1
                # Position k-1+j holds the target's own token after the first j drafts,
                # so the loop stops at the first mismatch and everything emitted is
                # greedy's.
                rows = H[0, k - 1:k + len(drafts)]
                if self.sampling is not None:
                    drafts, out, a = self._judge(rows, drafts, n_head, st, eos, max_tokens)
                    cand = ()
                else:
                    also = (drafts,) if on_gpu else ()
                    cand = self._logit_argmax_rows(rows, also)
                    if on_gpu:
                        drafts = drafts.tolist()    # evaluated above; no sync
                    out = [cand[0]]
                    a = 0
                    while (a < len(drafts) and out[a] not in eos
                           and st.tokens + len(out) < max_tokens):
                        # eos and the token budget end the round BEFORE the draft is
                        # judged: a draft the round never needed is not a rejection,
                        # and counting it as one is what would make `acceptance`
                        # drift away from the true acceptance.
                        if a < n_head:
                            st.note(a, drafts[a] == out[a])
                        else:
                            st.forced_accepted += drafts[a] == out[a]
                        if drafts[a] != out[a]:
                            break
                        a += 1
                        out.append(cand[a])
                new = [(H[:, k - 1 + j:k + j], out[j]) for j in range(len(out))]
                pairs = new if g else pairs + new
                # The round is resolved before anything is yielded, so a consumer that
                # walks away never finds a rejected position in its cache.
                if a == len(drafts):
                    # Every draft verified, so every position of the pass is
                    # state the target would have reached on its own: nothing to
                    # roll back. out[a] is the bonus token, still owed.
                    fed = len(emitted) + a
                    snap = _snapshot(cache)
                elif gdn is None and k + a + len(drafts) <= self.max_window:
                    # No rebuild: drop the pass and repeat these positions with the
                    # newly accepted tokens.
                    _restore(cache, snap)
                else:
                    # The cache moves to the accept point, rebuilt from this pass or
                    # replayed once before the next one would pass the kernel cliff.
                    if gdn is not None:
                        self._rebuild(cache, snap, k + a, gdn, tape)
                        st.rebuilds += 1
                    else:
                        _restore(cache, snap)
                        self.core(mx.array([pending + out[:a]]), cache=cache)
                        st.replays += 1
                    fed = len(emitted) + a
                    snap = _snapshot(cache)
                st.verify_s += time.perf_counter() - t0

                for t in out:
                    if forcer is not None:
                        forcer.feed(t)
                    yield t
                    emitted.append(t)
                    st.tokens += 1
        finally:
            if held is not None:
                # A round built ahead and not yet judged leaves the cache first.
                _restore(cache, held[0])
                hcache[0].trim(hcache[0].offset - held[1])
            # The tail the target never saw, without which the caller's cache is short
            # of `_seen`.
            rest = emitted[fed:]
            if rest and not own:
                self.core(mx.array([rest]), cache=cache)

    def stream_response(self, tokenizer, ids, max_tokens, cache=None,
                        stats=None, prefill_step=0):
        """generate() wrapped to look like mlx_lm.stream_generate, one response per token
        and the eos token carrying no text."""
        det = tokenizer.detokenizer
        eos = set(tokenizer.eos_token_ids)
        tok, n, done = None, 0, False
        # Same wired-memory window mlx_lm.stream_generate opens; without it the
        # speculative arm would be timed against a differently configured GPU.
        from mlx_lm.generate import generation_stream, wired_limit
        with wired_limit(self.model, [generation_stream]):
            for t in self.generate(ids, max_tokens, eos, stats, cache,
                                   prefill_step):
                if done:
                    continue        # never break: generate()'s tail is what
                tok, n = t, n + 1   # puts the last token in the caller's cache
                if t in eos:
                    done = True
                    continue
                det.add_token(t)
                if n == max_tokens:
                    done = True
                    continue
                yield Piece(t, det.last_segment)
        det.finalize()
        yield Piece(tok, det.last_segment,
                    "stop" if tok in eos else "length")


class ForcedSpeculator(Speculator):
    """No drafter: the only proposals are the rows the grammar forces, and a round without
    any is an ordinary one-position decode."""
    def __init__(self, model):
        super().__init__(model, None)

    def _draft(self, pairs, hcache, st):
        return []


