#!/usr/bin/env python3
"""The streaming engine as a library: build an instance, then ask or stream, with
conversation state, stats and an SSD front cache per instance."""

import inspect
import os
import statistics
import time
from pathlib import Path

import mlx.core as mx
import mlx_lm
from mlx_lm.generate import generate_step, speculative_generate_step
from mlx_lm.models.cache import (ArraysCache, KVCache, can_trim_prompt_cache,
                                make_prompt_cache, trim_prompt_cache)

import defaults
import sampler
import spec
import forced_rows
import tools as toolfmt
import spec_draft
import stream_fast
import vision
import window_cache
from memory_core import GapKVCache, cache_arrays, clone_cache, set_kv
from stream_fast import install, load_slim_aware, set_top_k
from engine_log import note

DEFAULT_MODEL = defaults.MODEL

# How many prefilled fronts to keep and how many front token ids in total,
# since a front costs its own length (pcache.prune_fronts).
FRONT_SLOTS = 8
FRONT_TOKEN_BUDGET = 12_000

# The share of the machine a streamed MoE's prompt prefill stays within unless
# prefill_gb says otherwise (StreamingLLM._prefill).
PREFILL_PCT = 35
MIN_PIECE = 2048

# MLX's own buffer cache, small on purpose: the streamed experts must not be
# held twice, and a freed pass buffer is worth less than the RAM it keeps.
MLX_CACHE_GB = 0.05


def _has_recurrent(cache):
    """Whether a cache carries a GatedDeltaNet-style state beside its K/V."""
    return any(isinstance(c, ArraysCache) for c in cache)


def _mlx_prefill_step(lib=generate_step):
    """The prefill chunk mlx_lm itself would use, for a caller that set none."""
    return inspect.signature(lib).parameters["prefill_step_size"].default


class ForcedPolicy:
    """Per-turn choice among the decode arms available this turn, which all emit the
    target's greedy tokens and differ only in speed."""
    KEEP = 8          # turns remembered per arm: a resized arena is followed
    EVERY_MAX = 32    # the longest a slower arm goes without a probe

    def __init__(self, arms=(True, False)):
        self.arms = tuple(arms)
        self.ms = {a: [] for a in self.arms}    # arm -> ms/token of recent turns
        self.last = {a: 0 for a in self.arms}   # the turn each arm last ran
        self.turn = 0
        self.best = None
        self.history = []                # (turn, best, {arm: mean ms}) per change

    def pick(self):
        """The arm to take this turn."""
        counts = {a: len(self.ms[a]) for a in self.arms}
        lo = min(counts.values())
        if lo < 2:
            return next(a for a in self.arms if counts[a] == lo)
        means = {a: statistics.fmean(self.ms[a]) for a in self.arms}
        spread = statistics.fmean(
            [statistics.variance(self.ms[a]) for a in self.arms]) ** 0.5
        best = min(means, key=means.get)
        # Each slower arm's own re-probe schedule: z*z turns of its own gap to
        # the best, z in units of the arms' pooled spread (see class doc).
        every = {a: min(self.EVERY_MAX, max(
                    1.0, (abs(means[a] - means[best]) / spread) ** 2
                    if spread else float("inf")))
                 for a in self.arms if a != best}
        if best != self.best:
            self.best = best
            self.history.append(
                (self.turn, best, {a: round(m, 1) for a, m in means.items()}))
            # The no-MTP engine's arms are True/False: print them by name.
            name = lambda a: {True: "forced-only", False: "plain"}.get(a, a)
            note(f"  decode arm: {name(best)} wins, "
                  + ", ".join(f"{name(a)} {means[a]:.1f}" for a in self.arms)
                  + " ms/token over "
                  + "+".join(str(counts[a]) for a in self.arms)
                  + " turns, re-probed every "
                  + ", ".join(f"{name(a)} {every[a]:.0f}" for a in self.arms if a != best)
                  + " turns",
                  flush=True)
        # When several arms are due at once the most overdue runs, but only on a turn
        # right after the best arm's own, so a probe never chains into a second one.
        due = [(self.turn - self.last[a] - every[a], a) for a in every]
        due = [(over, a) for over, a in due if over >= 0]
        return max(due)[1] if due and self.last[best] == self.turn else best

    def record(self, arm, ms):
        self.turn += 1
        self.last[arm] = self.turn
        self.ms[arm] = (self.ms[arm] + [ms])[-self.KEEP:]

    def outclassed(self, arm):
        """Whether `arm` trails the best by sqrt(EVERY_MAX) pooled spreads or more, where
        pick() re-probes it only at its longest interval."""
        if any(len(self.ms[a]) < 2 for a in self.arms):
            return False
        means = {a: statistics.fmean(self.ms[a]) for a in self.arms}
        gap = means[arm] - min(means.values())
        if gap <= 0:
            return False
        spread = statistics.fmean(
            [statistics.variance(self.ms[a]) for a in self.arms]) ** 0.5
        return not spread or gap / spread >= self.EVERY_MAX ** 0.5


class LeadPolicy(ForcedPolicy):
    """MOE_SPEC_AHEAD's arm choice (CUDA): every arm once, plain first, then the fastest by its turns' mean; the arm in
    use changes only when another beats it by more than the arms' pooled spread, and a slower arm is re-probed on
    ForcedPolicy's schedule, so a head server does not run plain on every other turn."""

    def pick(self):
        counts = {a: len(self.ms[a]) for a in self.arms}
        new = [a for a in self.arms if not counts[a]]
        if new:
            return "plain" if "plain" in new else new[0]
        means = {a: statistics.fmean(self.ms[a]) for a in self.arms}
        vs = [statistics.variance(self.ms[a]) for a in self.arms if counts[a] > 1]
        spread = statistics.fmean(vs) ** 0.5 if vs else 0.0
        best = min(means, key=means.get)
        if self.best in means and means[self.best] - means[best] <= spread:
            best = self.best        # hysteresis: within the spread the arm in use stays
        every = {a: min(self.EVERY_MAX, max(1.0, (abs(means[a] - means[best]) / spread) ** 2
                                            if spread else float(self.EVERY_MAX)))
                 for a in self.arms if a != best}
        if best != self.best:
            self.best = best
            self.history.append((self.turn, best, {a: round(m, 1) for a, m in means.items()}))
            note(f"  decode arm: {best} runs, " + ", ".join(f"{a} {means[a]:.1f}" for a in self.arms)
                 + " ms/token, re-probed every " + ", ".join(f"{a} {every[a]:.0f}" for a in every) + " turns",
                 flush=True)
        due = [(self.turn - self.last[a] - every[a], a) for a in every]
        due = [(over, a) for over, a in due if over >= 0]
        return max(due)[1] if due and self.last[best] == self.turn else best


class StreamingLLM:
    """A MoE model whose experts are read from SSD as the router asks for them, with
    cache_bias, the arena and the prefill budget as its measured working point."""

    # Function definitions for the current conversation, set per stream(messages=)
    # call, on the class because the toy engines skip __init__.
    _tools = None
    # The SSD directory of prefilled fronts written by _build_front and read back by
    # _reuse_front, shared by every engine with the same fingerprint or private to
    # this one without one (pcache.front_dir).
    _front_fp = None
    # That directory, resolved lazily by _front_root, on the class for the same
    # reason.
    _front_dir = None
    _front_sys = None       # leading system messages of the last `messages` call
    # External draft model for mlx_lm's own speculative decode, on the class because
    # the toy engines skip __init__.
    _draft_model = None
    draft_tokens = 3
    # None for mlx_lm's fixed chain, or the draft-margin threshold that ends a chain
    # early with draft_tokens as the cap (spec_draft.py).
    draft_tau = None
    # ForcedPolicy, made on the first turn that needs it (per engine: the
    # toy engines skip __init__, and the choice depends on this arena).
    forced_policy = None
    # One policy per turn kind, since a turn with a forcer and one without have
    # different winners (see ForcedPolicy).
    decode_policy_call = None    # turn with a forcer: mtp|draft/forced_only/plain
    decode_policy_free = None    # turn without one: mtp|draft/plain
    # An auto draft runs only on the turns the policy gives it and an explicit one on
    # every turn, while draft_path says where either came from.
    _draft_auto = False
    draft_path = None
    # Sink+window cap on the draft's KV cache, 0 = full (window_cache.py).
    draft_window = 0
    # The draft's own prompt cache and the token ids it mirrors, None and () until a
    # draft is loaded (see _sync_draft_cache).
    _draft_cache = None
    _draft_seen = ()
    # stream_fast.MemBudget under mem_peak_gb, else None (same __new__ reason).
    _budget = None
    # Thinking tokens a reply may write before the early-stop text (set_thinking), None for no cap.
    think_budget = None
    # A `messages` call takes up a kept conversation's cache where its ids still agree
    # instead of prefilling it again, within keep_convs conversations and keep_gb GB
    # (server.py turns it on; _take_conv, _convs).
    reuse = False
    keep_convs = 1
    keep_gb = 2.0
    _convs = ()
    # Under --mem-peak-gb: whether the budget's room covered a kept
    # conversation when the last turn ended (_keep_room); None before one has.
    _room_ok = None
    # This turn's prefill (_prefill): the widths of its passes, the arena
    # bytes parked for them, the kept conversations dropped to fit them.
    _pieces = ()
    _parked = _dropped = 0
    # Images (vision.py): the engine's image path, None without one, and
    # this turn's image spans (start, end, digest), a kept cache's key too.
    supports_images = False
    _vision = None
    _vision_why = "built without a checkpoint"
    _img = ()
    # A stopped shadow this turn took up and finished (_resume): the layer
    # its pass went on from, the pass's ids, the seconds.
    _resumed = None

    def __init__(self, model=DEFAULT_MODEL, top_k=0, resident=21,
                 adaptive=False, max_tokens=512, thinking=False, cache_bias=None,
                 prefill_step=None, profile=None, stub="slice", mtp=None,
                 bias_budget=0.0, defer=False, mem_pct=90, reserve_pct=15,
                 host_gb=0.6, draft=None, draft_tokens=None, draft_tau=None,
                 draft_window=0, mem_peak_gb=None,
                 prefill_gb=None, layers=None):
        # Every argument as given before anything below rebinds one, which is what the
        # stored fronts and the budget's calibration are keyed on (pcache.fingerprint).
        settings = {k: v for k, v in locals().items() if k != "self"}
        # `defer` passes straight to stream_fast.install, and cache_bias None means 0.5,
        # or 0 under mem_peak_gb where the pool is too small for the bias to have been
        # soaked.
        if cache_bias is None:
            cache_bias = 0.0 if mem_peak_gb else 0.5
        elif mem_peak_gb and cache_bias > 0.0:
            if not os.environ.get("ENGINE_BIAS_GATED"):     # the launcher sets it for a model whose bias passed its gate
                print(f"  WARNING: --mem-peak-gb with --cache-bias {cache_bias}: the "
                      f"router is biased toward a quota of 0-17 slots, where no "
                      f"soak has been run -- routing, and so the replies, follow "
                      f"the pool; --cache-bias 0 is the exact budget mode", flush=True)
        if model is None:
            raise ValueError("no model given: pass the checkpoint directory (--model)")
        path = Path(model).expanduser()
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Build one with:\n"
                f"  python3 quantpack.py --src <hf-repo> --out {path} "
                f"--bits 2 --edge-layers 4 --edge-bits 4")
        import pcache
        self._front_fp = pcache.fingerprint(path, settings)
        note(f"  prefilled fronts: {pcache.front_dir(path, self._front_fp)}")

        # A same-family sibling checkpoint proposes draft_tokens tokens a round for
        # mlx_lm's own draft_model path, checked and loaded before the slow target load.
        if draft_window and not draft:
            raise ValueError("draft_window needs a draft model: the target's "
                             "cache is never windowed -- its logits ARE the "
                             "reply")
        if draft_window < 0:
            raise ValueError(f"draft_window must be >= 0, got {draft_window}")
        # "auto" is a request to look, not a draft: it steps aside for mtp,
        # and it is resolved once the target is loaded (below).
        auto = draft == "auto"
        if draft and mtp and not auto:
            raise ValueError(
                "draft and mtp cannot be combined: both drive "
                "mlx_lm.stream_generate's speculative-decode loop, and only "
                "one drafter can propose a round")
        # k=2 for a draft found on disk: the best fixed chain measured with the
        # 0.6B on the 4B and the 8B (README "Dense speculative decode"); an
        # explicit draft keeps the 3 it has always had.
        if draft_tokens is None:
            draft_tokens = 2 if auto else 3
        if draft and not auto:
            draft_path = Path(draft).expanduser()
            if not draft_path.exists():
                raise FileNotFoundError(f"draft model {draft_path} not found")
            draft_vocab = self._load_draft(path, draft_path, draft_window)
            note(f"  draft {draft_path} loaded, k={draft_tokens} tokens, "
                  f"vocab_size {draft_vocab} matches target")
        if draft_tau is not None and not draft:
            raise ValueError("draft_tau needs a draft model: there is no "
                             "chain to shorten without draft=<path>")
        self.draft_tokens = draft_tokens
        self.draft_tau = draft_tau
        self.draft_window = draft_window

        # Resident stays 21: a decode-only sweep said more slots pay and an end-to-end
        # A/B through ask() over four turns said otherwise.

        # Prompt positions per prefill call, since the library default feeds the whole
        # prompt at once and a wide _batch_wide spikes the RSS high-water mark.
        self.prefill_step = prefill_step
        mx.set_cache_limit(int(MLX_CACHE_GB * 1e9))
        self.model, self.tokenizer, _ = load_slim_aware(path, str(path))
        if layers:
            # Layer-range mode (stage.py): this process runs layers LO..HI only, set
            # before install().
            if mtp or (draft and draft != "auto"):
                raise ValueError("layers cannot be combined with mtp or draft")
            try:
                import stage
            except ImportError:
                raise ValueError("layers needs stage.py, which this install does not include") from None
            stage.prepare(self.model, *layers)
        # Before any cache is made: the pool lives in qwen4_exp's indexer cache.
        stream_fast.install_qsa_pool(self.model)
        # The adaptive arena on Metal takes the native agent by default, as the launcher's recipe does.
        native = stream_fast.native_default(defer, arena=not (mem_peak_gb or layers))
        streamed, st = install(self.model, path, resident=resident,
                               cache_bias=cache_bias, defer=defer, mtp=bool(mtp), native=native)
        if layers:
            stage.after_install(self.model)
        if streamed and not stream_fast.EMBED_VRAM:
            # A streamed MoE pack also streams embed_tokens, an ordinary dense table in
            # the same shards that the same reader frees (stream_fast.install_embed).
            stream_fast.install_embed(self.model, st)
        if top_k:
            set_top_k(self.model, top_k)
        if not streamed and stream_fast.BACKEND != "metal":
            note(f"  target matmul: stock (rowqmm and foldattn are Metal kernels; "
                  f"{stream_fast.BACKEND} here)")
        elif not streamed:
            # A dense target runs its multi-row passes through rowqmm's kernels where
            # install() timed them faster than stock, calibrated on the first load and
            # reused from disk.
            import rowqmm
            layout = rowqmm.install(self.model)
            routes = {}
            for key, cand in layout.items():
                routes.setdefault(rowqmm._label(cand), set()).add(key[0])
            note("  target matmul: rowqmm "
                  + (", ".join(f"{lbl} at T={sorted(rows)}"
                               for lbl, rows in sorted(routes.items()))
                     or "nothing (stock was faster)"))
            # Its multi-row attention likewise goes through foldattn's kernel
            # (each K/V read once per pass, not once per query row and head)
            # for the (context, T) buckets install() timed faster than stock.
            import foldattn
            foldattn.install(self.model)
        # Prefill's own transient sized against the same process budget the arena
        # answers to, and on a streamed MoE also held to prefill_gb
        # (stream_fast.prefill_cap_bytes, PREFILL_PCT).
        total_gb = (stream_fast._sysctl_int("hw.memsize", width=8) or 0) / 1e9
        rows = stream_fast.moe_row_elems(self.model)
        budget = mem_peak_gb or total_gb * mem_pct / 100.0
        if rows and not mem_peak_gb:
            budget = min(budget, prefill_gb or total_gb * PREFILL_PCT / 100.0)
        stream_fast.set_prefill_budget_gb(
            budget, host_gb=host_gb, row_elems=rows,
            fixed=stream_fast.whole_layer_bytes(self.model))
        import chunkattn
        chunkattn.install(self.model)
        import qsa_tiles
        qsa_tiles.install(MLX_CACHE_GB)
        # What one layer's prefill attention materialises at this model's
        # shapes and backend's kernels, so a piece is planned with its scores
        # in it: without the term a 64k prompt's [24, T, keys] scores outgrew
        # the budget nothing had reserved for them. chunkattn's calibrated
        # width is the SDPA's own chunk, which the model counts separately
        # from the indexer's, since only the indexer's is left uncut.
        # CUDA's only: on the PC the term is what made 64k to 128k answer,
        # while on Metal nothing ever tripped and the same reservation moved
        # the plan off the piece and chunk the pass already fit, which
        # measured slower.
        sdpa_rows = 0
        attn = None
        if stream_fast.BACKEND == "cuda":
            sdpa_rows = min((v[1] for v in chunkattn.layout.values()), default=0)
            attn = chunkattn.prefill_attn_slope(self.model, sdpa_rows=sdpa_rows)
        stream_fast.set_prefill_attention(attn)
        if attn:
            note(f"  prefill attention: {attn[0]:.1f} B/row/key of scores at "
                 f"{sdpa_rows or 'no'} rows, {attn[1]:.1f} of indexer, "
                 f"{attn[2] / 1e3:.0f} KB/row, {attn[3] / 1e3:.1f} KB/key")
        # `draft="auto"`: a dense target takes the draft draft_find.py picks off the
        # disk, loaded after the target because only install() says which one this is.
        if auto and not streamed and not mtp:
            import draft_find
            budget = ((mem_peak_gb or total_gb * mem_pct / 100.0) - host_gb) * 1e9
            a0, found = mx.get_active_memory(), None
            try:
                found = draft_find.find(path, budget=budget)
                if found is not None:
                    self._load_draft(path, found[0], draft_window)
            except Exception as e:
                # Directories on disk nobody named on the command line: if
                # one of them breaks the scan or the load, the engine runs as
                # it would have without a draft rather than not at all.
                self._draft_model = self._draft_cache = None
                print(f"  draft: auto -> {found[0] if found else 'scan'} "
                      f"failed ({type(e).__name__}: {e}); none")
            else:
                if found is None:
                    note("  draft: auto -> none found")
                else:
                    self._draft_auto = True
                    note(f"  draft: auto -> {found[0]} (same tokenizer), "
                          f"k={draft_tokens}, "
                          f"{(mx.get_active_memory() - a0) / 1e9:.2f} GB resident")
        # A saved expert profile seeded into the demand counters before the first decode
        # so the arena starts warm, kept so callers that persist state need not pass the
        # path twice.
        self.model_path = str(path)
        self.profile = profile
        if profile:
            import eprofile
            meta = eprofile.warm(self.model, path, profile)
            note(f"  profile '{profile}': "
                  + (f"{meta['sessions']} sessions, updated {meta['updated']}"
                     if meta else "none saved yet, starting cold"))
        # Evaluated before the governor is built (not after, as this used to
        # do): the governor's startup budget check reads
        # mx.get_active_memory(), and that has to already include the skeleton
        # or it undercounts held_gb and prints a budget roomier than it is.
        mx.eval(self.model.parameters())
        if mem_peak_gb and streamed:
            # A PEAK budget, what Activity Monitor shows: the experts move into one
            # shared store whose pool is sized from the budget live, and --resident and
            # the governor step aside (stream_fast.MemBudget).
            if adaptive:
                note("  adaptive arena: off -- --mem-peak-gb sizes a shared "
                      "expert store instead")
                adaptive = False
            # Its measurements outlive the process in a cache keyed on the fronts'
            # fingerprint and every setting besides.
            from rowqmm import LAYOUT_DIR
            calib = LAYOUT_DIR / f"membudget-{pcache.layout_key((), self._front_fp, settings)}.json"
            self._budget = stream_fast.enable_budget(
                self.model, mem_peak_gb, cache_gb=MLX_CACHE_GB,
                t_max=2 if mtp else 1, calib=calib)
            if self._budget is not None:
                # The budget cuts its own passes before any prefill budget does, the
                # first one included.
                stream_fast.set_prefill_budget_gb(None, row_elems=rows)
            if self.prefill_step is None and self._budget is not None and self._budget.layer_major:
                # Every pass of a layer-major prefill reads every expert: one
                # pass, which the budget cuts where it does not fit, instead
                # of mlx_lm's 2048-token chunks (its chunks run inside it).
                self.prefill_step = 0
        if adaptive and not streamed:
            # A dense model has no expert arena to size; --adaptive is chat's
            # default, not a request, so it is a no-op here (as defer's auto).
            note("  adaptive arena: dense model, nothing to govern")
        elif adaptive:
            glus = stream_fast.glus_of(self.model)
            note(f"  adaptive arena: governing {len(glus)} layers"
                  + ("" if glus else " -- NOTHING TO GOVERN, --adaptive is a no-op"))
            # Ceiling raised from 40 to 96 after a paired A/B at cache_bias 0.5 measured
            # blocking reads falling and decode rising at resident 21, 64 and 96, with
            # no degeneration at 64.
            k = stream_fast.top_k_of(self.model)
            n_exp = stream_fast.n_experts_of(self.model)
            if not glus or n_exp is None:
                # Without layers there is nothing to govern and without an expert count
                # no ceiling, so failing here means the model was never instrumented.
                raise RuntimeError(
                    f"--adaptive found {len(glus)} streaming layers and "
                    f"experts/layer={n_exp}; the model was not instrumented.")
            # top_k is only the starting floor, the real one being the high-water mark
            # the layers report as they run, and the cap every expert in the layer
            # (stream_fast.arena_floor).
            stream_fast.make_governor(glus, n_exp, k, mem_pct, reserve_pct,
                                     host_gb)

        self.max_tokens = max_tokens
        self.resident = resident
        # Qwen3.5/3.6 open a think block by default, which spends a short question's
        # whole budget on notes, so thinking is off unless a caller asks for it.
        self.thinking = thinking
        stream_fast.set_bias_budget(bias_budget)
        self._setup_chat(stub)
        # After _setup_chat, which is what clears `_spec`: a toy engine reaches
        # this class through _setup_chat alone.
        # mtp and defer combine: the run-ahead takes passes up to 63 // top_k
        # positions wide (stream_fast._DEFER_MAX_T) and every layer is resolved
        # before the pass returns, so the speculator's round-level rollback
        # always acts on an exact cache.
        if mtp:
            self._spec = self._load_speculator(mtp)
            # MOE_SPEC_AHEAD: the head's rounds are warmed after the first turn (_warm_spec).
            self._spec_cold = self._spec if spec.Speculator.ahead else None
        # Last: its hooks go on top of every wrapper install() put on the
        # layer stack, and only once an image comes.
        self._vision, self._vision_why = vision.attach(
            path, self.model, budget=bool(mem_peak_gb), layers=layers)
        self.supports_images = self._vision is not None
        if self._vision is not None or self._vision_why not in ("no vision_config", "no config.json"):
            note(f"  images: {'on' if self.supports_images else 'off'} "
                  f"({self._vision_why})")

    # What to do about the think stub the template emits for the current turn only: keep
    # leaves it, slice cuts its positions out of the caches, refeed rebuilds the turn
    # from a snapshot, none never feeds it.
    STUB_MODES = ("keep", "slice", "refeed", "none")

    def _setup_chat(self, stub="slice"):
        """Conversation state, split out of __init__ so an engine can be built
        around a toy model without opening a checkpoint."""
        if stub not in self.STUB_MODES:
            raise ValueError(f"stub must be one of {self.STUB_MODES}, "
                             f"not {stub!r}")
        self.stub_mode = stub
        # Cleared here, not in __init__, because a toy engine is built with
        # __new__ and reaches this class through _setup_chat alone.
        # __init__ installs the speculator immediately after calling this.
        self._spec = None
        self._tpl = self._learn_template()
        self._stub, self._stub_ids = self._learn_stub()
        if stub != "keep" and not self._stub_ids:
            # Silence here would be the worst outcome: a run that reports
            # `--stub refeed` and behaves exactly like `keep`.
            note(f"  stub={stub}: this template's generation prompt has no "
                  f"think stub, so the mode is a no-op")
        self.history = []
        self._last = {}
        self.reset()

    def set_thinking(self, on, budget=None):
        """Render the prompts that follow with the template's enable_thinking on or off,
        as server.py sets it before each request; a string turns it on at that reasoning_effort,
        and `budget` caps the thinking at that many tokens (_think_cap)."""
        self.thinking = bool(on)
        self.effort = on if isinstance(on, str) else None
        self.think_budget = budget

    def _think_cap(self, sampling):
        """This turn's sampler.ThinkBudget around `sampling`, or None: no budget, thinking off, or a
        prompt that does not leave the reply inside a think block."""
        if not self.think_budget or not self.thinking:
            return None
        opened, close, forced = (self.tokenizer.encode(t, add_special_tokens=False)
                                 for t in ("<think>", "</think>", sampler.EARLY_STOP))
        tail = self._seen[-4:]      # the generation prompt's end, `<think>\n` on Qwen's templates
        if len(opened) != 1 or len(close) != 1 or opened[0] not in tail or close[0] in tail:
            return None
        return sampler.ThinkBudget(sampling, self.think_budget, forced, close[0])

    def _load_speculator(self, path):
        """Build the MTP speculator, or say out loud why decoding stays ordinary."""
        import json
        import mtp as mtp_mod
        p = Path(path).expanduser()
        f = p if p.is_file() else p / "mtp-4bit.safetensors"
        try:
            cfg = json.loads(
                (Path(self.model_path) / "config.json").read_text())
            head = mtp_mod.load_mtp(p, mtp_mod.head_args(cfg))
        except Exception as e:
            print(f"  mtp: {f} unusable ({type(e).__name__}: {e}); "
                  f"decoding the ordinary way")
            return None
        note(f"  mtp head {f}: {f.stat().st_size / 1e6:.1f} MB loaded, "
              f"speculative decode on")
        return spec.Speculator(self.model, head)

    def _warm_spec(self, sp):
        """MOE_SPEC_AHEAD: a short speculative decode on a cache of its own after the first turn, so a verify pass's
        first-use kernel builds stay out of the MTP turns the decode policy times."""
        t = time.perf_counter()
        ids = list(self._seen[-16:])        # the first turn's own ids
        # Out of the memory budget's sight: it keeps the largest prefill growth a position it measures.
        stack = stream_fast._layer_stack_of(self.model)
        bud = stack.__dict__.pop("_mem_budget", None)
        try:
            for _ in sp.generate(ids, 24):
                pass
            mx.synchronize()
        finally:
            if bud is not None:
                stack.__dict__["_mem_budget"] = bud
            mx.clear_cache()
            mx.reset_peak_memory()
        note(f"  mtp: speculative rounds warmed in {time.perf_counter() - t:.1f} s")

    def _learn_stub(self):
        """The stub's text and its exact token ids, read out of the template: what only the
        current turn gets beyond a past assistant turn."""
        t = self._tpl
        if not t or not t["tail"].startswith(t["after_user"]):
            return "", []
        stub = t["tail"][len(t["after_user"]):]
        if not stub:
            return "", []
        tail = self.tokenizer.encode(t["tail"], add_special_tokens=False)
        head = self.tokenizer.encode(t["after_user"], add_special_tokens=False)
        if tail[:len(head)] != head:
            # The stub is not a clean suffix of the generation prompt, so degrade to
            # keep rather than cut the wrong tokens out of the cache.
            return "", []
        ids = tail[len(head):]
        hits = sum(1 for i in range(len(tail) - len(ids) + 1)
                   if tail[i:i + len(ids)] == ids)
        assert hits == 1, (f"the think stub {ids} occurs {hits} times in the "
                           f"generation prompt {tail}; cannot locate it")
        return stub, ids

    def _prefill(self, ids, step=None):
        """Push `ids` through the model into the live cache without sampling, within the
        prefill budget on a streamed MoE."""
        cap = step or self.prefill_step or len(ids) or 1
        auto = not (step or self.prefill_step)
        long = len(ids) >= MIN_PIECE
        parked, dropped, widths, j, t_park = 0, 0, [], 0, [0.0, 0.0]
        try:
            while j < len(ids):
                left = len(ids) - j
                n = min(cap, left)
                fit = stream_fast.prefill_fit(self._cache) if long else None
                if fit is not None and fit < n and not parked:
                    t = time.perf_counter()
                    parked = stream_fast.park_arenas(self.model)
                    t_park[0] = time.perf_counter() - t
                    fit = stream_fast.prefill_fit(self._cache)
                while fit is not None and fit < min(n, MIN_PIECE) and self._convs:
                    self._convs = self._convs[1:]
                    dropped += 1
                    mx.clear_cache()
                    fit = stream_fast.prefill_fit(self._cache)
                if auto and fit is not None and fit < n:
                    # As many equal passes as the widest that fits would take, since
                    # each pass reads every expert again and equal ones peak lowest.
                    k = -(-left // max(fit, MIN_PIECE // 4))
                    n = -(-left // k)
                # A pass on an empty cache measures the rate the budget
                # plans with (prefill_fit); attention's transient, capped at
                # the room on a longer cache, stays out of it.
                measure = (fit is not None and n >= 4096
                           and not any(getattr(c, "offset", 0) for c in self._cache))
                if measure:
                    mx.reset_peak_memory()
                    a0 = mx.get_active_memory()
                # Only the cache is evaluated: the logits of every position are
                # L x vocab, 2.7 GB for a 5.5k-token front, and nothing reads them.
                self.model(mx.array(ids[j:j + n])[None], cache=self._cache)
                mx.eval([stream_fast.cache_state(c) for c in self._cache])
                if measure:
                    stream_fast.note_prefill(n, mx.get_peak_memory() - a0)
                widths.append(n)
                j += n
                if n < len(ids):
                    mx.clear_cache()
        except stream_fast.PrefillStopped:
            # A request waits on a shadow (shadow()): the arena comes back
            # at its first claim instead.
            parked = 0
            raise
        finally:
            if parked:
                t = time.perf_counter()
                stream_fast.unpark_arenas(self.model)
                t_park[1] = time.perf_counter() - t
            self._pieces += tuple(widths)
            self._parked = max(self._parked, parked)
            self._dropped += dropped
        if parked or dropped or (auto and len(widths) > 1):
            note(f"  prefill: {len(ids)} ids in {len(widths)} pass"
                  f"{'es' if len(widths) > 1 else ''} {widths} within "
                  f"{stream_fast._PREFILL_BUDGET_GB:.2f} GB"
                  + (f", arena parked ({parked / 1e9:.2f} GB, {t_park[0]:.2f} s; read "
                     f"back in {t_park[1]:.2f} s)" if parked else "")
                  + (f", {dropped} kept conversation(s) dropped" if dropped else ""),
                  flush=True)

    def _load_draft(self, path, draft_path, draft_window):
        """Load the draft for the target at `path`, before it when explicit and after it
        when auto, and return its vocab_size."""
        import json
        tgt_vocab = json.loads((path / "config.json").read_text())["vocab_size"]
        draft_vocab = json.loads(
            (draft_path / "config.json").read_text())["vocab_size"]
        if tgt_vocab != draft_vocab:
            # A vocab mismatch would make verification compare different words, so
            # refusing beats scoring garbage.
            raise ValueError(
                f"draft {draft_path} has vocab_size {draft_vocab}, target "
                f"{path} has {tgt_vocab}: the draft must share the "
                f"target's tokenizer or speculative decode verifies "
                f"against the wrong distribution")
        self._draft_model, _ = mlx_lm.load(str(draft_path))
        if not draft_window:
            # The draft decodes through draftstep's fused kernels where install() timed
            # each piece faster than stock, bit-identical except a screened lm_head that
            # keeps only the top three ids exact.
            import draftstep
            fused = draftstep.install(self._draft_model)
            note(f"  draft decode step: fused {sorted(fused) or 'nothing'}")
        # The draft's own prompt cache, kept in lockstep with self._cache by a token-id
        # diff (see _sync_draft_cache).
        self.draft_window = draft_window
        self._draft_cache = self._new_draft_cache()
        self._draft_seen = []
        self.draft_path = str(draft_path)
        return draft_vocab

    def _drop_losing_draft(self):
        """Free a draft found on disk once the decode policy ranks it outclassed on every
        kind of turn it has seen."""
        pols = [p for p in (self.decode_policy_call, self.decode_policy_free)
                if p is not None]
        if not pols or not all(p.outclassed("draft") for p in pols):
            return
        means = {a: round(statistics.fmean(v), 1) for a, v in pols[-1].ms.items() if v}
        self._draft_model = None
        self._draft_cache, self._draft_seen = None, ()
        self._draft_auto = False
        self.decode_policy_call = self.decode_policy_free = None
        for conv in self._convs:
            conv["draft"] = (None, ())
        mx.clear_cache()
        note(f"  draft: auto -> released, slower than decoding without it "
              f"here ({means} ms/token)", flush=True)

    def _new_draft_cache(self):
        """A fresh cache for the draft model, windowed if draft_window is set."""
        c = make_prompt_cache(self._draft_model)
        return window_cache.to_window(c, window=self.draft_window) \
            if self.draft_window else c

    def _sync_draft_cache(self, prefix):
        """Keep the draft's own KV cache a prefix-exact mirror of the target's past tokens,
        diffed by id against self._draft_seen."""
        j = 0
        while (j < len(self._draft_seen) and j < len(prefix)
               and self._draft_seen[j] == prefix[j]):
            j += 1
        if j < len(self._draft_seen):
            drop = len(self._draft_seen) - j
            if trim_prompt_cache(self._draft_cache, drop) != drop:
                self._draft_cache, j = self._new_draft_cache(), 0
        rest = prefix[j:]
        if rest:
            step = self.prefill_step or len(rest)
            for k in range(0, len(rest), step):
                self._draft_model(mx.array(rest[k:k + step])[None],
                                  cache=self._draft_cache)
                mx.eval([c.state for c in self._draft_cache])
        self._draft_seen = list(prefix)

    def _slice_stub(self, at):
        """Cut the stub's positions out of every attention cache after the turn, leaving
        the survivors at the RoPE positions they were written at."""
        L = len(self._stub_ids)
        t0 = time.perf_counter()
        for j, c in enumerate(self._cache):
            if not isinstance(c, KVCache):
                continue                        # GDN state: cannot be un-fed
            gap = getattr(c, "gap", 0)
            stored = c.offset - gap
            assert stored == len(self._seen), (
                f"cache {j} holds {stored} positions against "
                f"{len(self._seen)} tokens in _seen -- already diverged")
            k, v = c.keys[..., :stored, :], c.values[..., :stored, :]
            g = GapKVCache(gap + L)
            # The state setter puts offset at the new storage length; the bump
            # takes it back to exactly the offset this cache already had.
            set_kv(g, mx.concatenate([k[..., :at, :], k[..., at + L:, :]], axis=2),
                   mx.concatenate([v[..., :at, :], v[..., at + L:, :]], axis=2))
            g.offset += gap + L
            mx.eval(g.keys, g.values)
            self._cache[j] = g
        del self._seen[at:at + L]
        self._gap += L
        self._slice_s += time.perf_counter() - t0

    def _refeed(self, snap, seen, gen):
        """Put the cache back where the official rendering would have it, from the snapshot
        taken before the stub plus the reply."""
        t0 = time.perf_counter()
        eos = getattr(self.tokenizer, "eos_token_ids", ())
        while gen and gen[-1] in eos:
            gen = gen[:-1]
        self._cache, self._seen = snap, list(seen)
        if gen:
            self._prefill(gen)
            self._seen += gen
        self._refeed_s += time.perf_counter() - t0

    def _learn_template(self):
        """Extract the template's separators once, by rendering unique markers and
        splitting on them."""
        U1, A1, U2 = "\u0001u1\u0001", "\u0001a1\u0001", "\u0001u2\u0001"
        probe = self._raw_render([
            {"role": "user", "content": U1},
            {"role": "assistant", "content": A1},
            {"role": "user", "content": U2}])
        try:
            head, rest = probe.split(U1, 1)
            mid, rest = rest.split(A1, 1)
            between, tail = rest.split(U2, 1)
        except ValueError:
            return None
        return {"head": head, "after_user": mid, "between": between, "tail": tail}

    def _raw_render(self, hist):
        """Render the conversation, tools included, through the checkpoint's own template, client text escaped: encode it with toolfmt.encode."""
        kw = {"add_generation_prompt": True, "tokenize": False}
        # A file the agent read that quotes <|im_end|> must not end the turn in the model's eyes.
        hist = toolfmt.escape(hist, self.tokenizer)
        if self._tools:
            kw["tools"] = toolfmt.escape(self._tools, self.tokenizer)
            # OpenAI sends tool-call arguments as a JSON string and the template
            # iterates them as a mapping, reshaped here so every caller gets it.
            hist = toolfmt.for_template(hist)
        effort = getattr(self, "effort", None) if self.thinking else None
        try:
            return self.tokenizer.apply_chat_template(
                hist, enable_thinking=self.thinking, **({"reasoning_effort": effort} if effort else {}), **kw)
        except TypeError:
            return self.tokenizer.apply_chat_template(hist, **kw)

    def _next_chunk(self, user, first_turn):
        """Text to feed for this turn: only what the cache does not hold."""
        t = self._tpl
        if t is None:                       # unknown template: re-render whole
            return self._raw_render(self.history), True
        user = toolfmt.escape(user, self.tokenizer)
        if first_turn:
            return t["head"] + user + t["tail"], False
        # `between` opens with whatever closes an assistant turn.
        return t["between"] + user + t["tail"], False

    def _continue_chunk(self, messages, mark="\u0001c\u0001"):
        """Render a whole messages list as a continuation of the cached turn by cutting it
        at a marker user turn."""
        probe = self._raw_render([{"role": "user", "content": mark}]
                                 + [dict(m) for m in messages])
        front, rest = probe.split(mark, 1)
        if not self._tools:
            return rest
        # The tool block renders ahead of the system message and the marker, so behind a
        # memory core it goes in as its own turn, whose boundary is learned from the
        # template's own separators (see _tool_block).
        block = self._tool_block(front)
        between, head = (self._tpl["between"], self._tpl["head"]) \
            if self._tpl else ("", "")
        close = between[:-len(head)] if head and between.endswith(head) else ""
        if not block or not close or not rest.startswith(close):
            # No boundary to insert at means no honest way to show the model
            # the functions, and dropping them silently is the exact failure
            # this branch exists to repair.
            raise RuntimeError(
                "tools= and resume= (server.py --memory) cannot be combined "
                "on this template: there is no turn boundary to put the tool "
                f"block at (block {len(block)} chars, close {close!r}). Serve "
                "tools without --memory rather than have the model answer as "
                "if no functions existed.")
        return rest[:len(close)] + block + rest[len(close):]

    def _tool_block(self, front=None):
        """The complete system turn the template writes for the current tool list, and
        nothing else."""
        if not self._tools or not self._tpl:
            return ""
        if front is None:
            mark = "\u0001t\u0001"
            front = self._raw_render(
                [{"role": "user", "content": mark}]).split(mark, 1)[0]
        head = self._tpl["head"]
        return front[:-len(head)] if head and front.endswith(head) else ""

    def _front_text(self):
        """Everything the template writes before the first non-system message: the tool
        block and the request's leading system messages."""
        lead = list(self._front_sys or [])
        if not self._tpl or (not lead and not self._tools):
            return ""
        mark = "\u0001t\u0001"
        front = self._raw_render(
            lead + [{"role": "user", "content": mark}]).split(mark, 1)[0]
        head = self._tpl["head"]
        return front[:-len(head)] if head and front.endswith(head) else ""

    def _front_ids(self, ids):
        """`ids` cut down to the front, or [] when the front is not a prefix of them,
        checked as token ids and not as text."""
        block = self._front_text()
        front = toolfmt.encode(self.tokenizer, block) if block else []
        return front if front and ids[:len(front)] == front else []

    def _front_root(self):
        """The SSD directory this engine's fronts live in, shared by every engine with the
        same fingerprint or private to this one."""
        if self._front_dir is None:
            import pcache
            self._front_dir = (pcache.front_engine_dir(self.model_path, self)
                               if self._front_fp is None else
                               pcache.front_dir(self.model_path, self._front_fp))
        return self._front_dir

    def _build_front(self, ids):
        """Prefill the tool block alone and write the cache it made to SSD, returning how
        many tokens are now in it."""
        front = self._front_ids(ids)
        if not front:
            return 0
        import pcache
        # Routing off the cache bias: the front must not depend on this
        # process's arena, or a restore elsewhere gives other replies.
        with stream_fast.bias_off(getattr(self, "model", None)):
            self._prefill(front)
            mx.eval([a for c in self._cache for a in cache_arrays(c)])
        if self._budget is not None:
            # The save below would copy the whole front's KV on top of the
            # peak budget (stream_fast.trim_kv says why).
            stream_fast.trim_kv(self._cache)
        try:
            pcache.save_front(self._cache, front, self._front_root())
            pcache.prune_fronts(self._front_root(), front, FRONT_SLOTS,
                                FRONT_TOKEN_BUDGET)
        except (OSError, RuntimeError) as e:
            # The cache directory is unwritable or below the free-disk floor save_front
            # keeps, which costs only this front's chance to skip the next prefill.
            print(f"  front cache not kept ({type(e).__name__}: {e})")
        return len(front)

    def _reuse_front(self, ids):
        """Load a copy of the longest stored front that is a prefix of this request from
        SSD, returning its length or 0."""
        import pcache
        d = self._front_root()
        match = max((f for f in pcache.list_fronts(d) if ids[:len(f)] == f),
                    key=len, default=None)
        if match is None:
            return 0
        cache = pcache.load_front(pcache.front_key(match), match, d)
        if cache is None:
            # Evicted since the listing, or unreadable: a real prefill, like
            # a cold front.
            return 0
        self._cache = cache
        return len(match)

    @staticmethod
    def _conv_reach(conv, ids, img=()):
        """How many of `ids` a kept conversation's cache holds where it can be taken up,
        touching nothing."""
        seen, marks, cache = conv["seen"], conv["marks"], conv["cache"]
        top = min(len(seen), len(ids) - 1)
        m = 0
        while m < top and seen[m] == ids[m]:
            m += 1
        diff = [s for s, _, _ in set(img) ^ set(conv.get("img", ())) if s < m]
        if diff:
            m = min(diff)
        if not any(n <= m for n, _, _ in marks):
            return 0
        if m == len(seen):
            return m
        if conv.get("resume"):
            return 0
        if not _has_recurrent(cache):
            return m if can_trim_prompt_cache(cache) else 0
        return max(n for n, _, _ in marks if n <= m)

    @staticmethod
    def _conv_bytes(conv):
        """What a kept conversation holds in arrays: its cache, the recurrent
        states its marks hold, its draft's cache, a stopped shadow's hidden
        states."""
        held = [a for _, h, _ in conv["marks"] if h
                for s in h if isinstance(s, list) for a in s if a is not None]
        return (sum(c.nbytes for c in conv["cache"])
                + sum(c.nbytes for c in conv["draft"][0] or ())
                + sum(a.nbytes for a in held)
                + (conv["resume"][1].nbytes if conv.get("resume") else 0))

    def _take_conv(self, ids):
        """Take up the kept conversation that holds the most of `ids`, the most recent of
        equals, and return how many it holds, 0 for none."""
        convs = list(self._convs)
        reach = [self._conv_reach(c, ids, self._img) for c in convs]
        b = max(reach, default=0)
        resume = None
        if b:
            conv = convs.pop(max(j for j, r in enumerate(reach) if r == b))
            cache, seen = conv["cache"], conv["seen"]
            if b < len(seen) and _has_recurrent(cache):
                spec._restore(cache, next(h for n, h, _ in conv["marks"] if n == b))
            elif b < len(seen):
                trim_prompt_cache(cache, len(seen) - b)
            self._cache = cache
            self._draft_cache, self._draft_seen = conv["draft"]
            self._marks = [mk for mk in conv["marks"] if mk[0] <= b]
            resume = conv.get("resume")
        # This conversation as the turn will leave it counts too, since counted after
        # the turn a 48k-token one came on top of keep_gb.
        mine = len(ids) * max((sum(c.nbytes / c.offset for c in cv["cache"]
                                   if isinstance(c, KVCache) and c.offset)
                               for cv in convs[-1:]), default=0.0)
        # Known limit: sizes are summed again per eviction, fine while keep_convs is a handful.
        while convs and (len(convs) >= self.keep_convs or
                         sum(map(self._conv_bytes, convs)) + mine > self.keep_gb * 1e9):
            convs.pop(0)
        self._convs = tuple(convs)
        if resume is not None:
            self._resume(ids[:b], *resume)
        return b

    def _resume(self, seen, k, h):
        """Run the rest of the pass a request stopped a shadow in (shadow())."""
        n, t = h.shape[1], time.perf_counter()
        with stream_fast.prefill_skip(self._cache, k, h):
            self._prefill(seen[-n:], n)
        stream_fast.unpark_arenas(self.model)
        held = (spec._snapshot(self._cache)
                if _has_recurrent(self._cache) else None)
        self._marks = [(len(seen), held, True)]
        self._resumed = {"layer": k, "ids": n, "s": round(time.perf_counter() - t, 2)}

    def forget(self):
        """Drop the kept conversations, which reset() leaves: every `messages`
        call runs it and still takes one up."""
        self._convs = ()

    def shadow(self, messages, tools=None, stop=None):
        """Prefill now, while the user reads, the history the next user message will
        render, as a kept conversation of its own."""
        if not (self.reuse and self._budget is None and self._convs
                and self._draft_model is None and stream_fast.prefill_stoppable(self.model)):
            return None
        # An image's ids are its token count in pads, which only a request's
        # layout knows; such a history waits for its request.
        if vision.images_of(messages):
            return None
        if self._vision is not None:
            self._vision.box.layout = None
        prev, self._tools = self._tools, tools
        try:
            a, b = (toolfmt.encode(self.tokenizer, self._raw_render(
                list(messages) + [{"role": "user", "content": s}])) for s in "ab")
        finally:
            self._tools = prev
        n = 0
        while n < min(len(a), len(b)) and a[n] == b[n]:
            n += 1
        special = set(self.tokenizer.all_special_ids)
        ids = a[:max((i for i in range(n) if a[i] in special), default=0)]
        convs = list(self._convs)
        reach = [self._conv_reach(c, ids + [-1]) for c in convs]
        r = max(reach, default=0)
        if not r:
            return None
        js = max(j for j, x in enumerate(reach) if x == r)
        src = convs[js]
        if r >= len(src["seen"]) or r >= len(ids):
            return None         # the render rewrites nothing the cache holds
        # Its size once complete (_conv_bytes: the end mark holds the state
        # the cache does), and the older conversations that must go for it.
        state = sum(c.nbytes for c in src["cache"] if isinstance(c, ArraysCache))
        est = 2 * state + len(ids) * sum(c.nbytes / c.offset for c in src["cache"]
                                         if isinstance(c, KVCache) and c.offset)
        drop = 0
        while drop < js and (len(convs) - drop >= self.keep_convs or sum(
                map(self._conv_bytes, convs[drop:])) + est > self.keep_gb * 1e9):
            drop += 1
        if (len(convs) - drop >= self.keep_convs
                or sum(map(self._conv_bytes, convs[drop:])) + est > self.keep_gb * 1e9):
            return None
        snap = next((h for m, h, _ in src["marks"] if m == r), None)
        cache = make_prompt_cache(self.model)
        for c, s, h in zip(cache, src["cache"], snap or [None] * len(cache)):
            if isinstance(s, ArraysCache):
                c.cache = list(h)
            else:
                kv = (s.keys[..., :r, :], s.values[..., :r, :])
                if hasattr(s, "indexer"):
                    c.state = kv + (s.indexer.keys[:, :r],)     # qwen4_exp's cache holds its indexer's keys too
                else:
                    set_kv(c, *kv)
        # The request's own prefill up to its mark would run this (stream()).
        step = (None if self._spec is not None or self.prefill_step is not None
                else _mlx_prefill_step())
        info = {"from": r, "ids": len(ids) - r}
        if stop is not None and stop():
            return None         # before the arena is parked for it
        live, self._cache, self._convs, self._pieces = self._cache, cache, (), ()
        t0, stopped = time.perf_counter(), None
        stream_fast.PREFILL_STOP = stop
        try:
            self._prefill(ids[r:], step)
        except stream_fast.PrefillStopped as e:
            stopped = e.args
        finally:
            stream_fast.PREFILL_STOP = stream_fast._PREFILL_AT = None
            self._cache, self._convs = live, tuple(convs)
            info["s"] = round(time.perf_counter() - t0, 2)
            mx.clear_cache()
        n, resume = len(ids), None
        info["passes"] = list(self._pieces)
        if stopped is not None:
            # Stopped in layer k of a pass: the layers before it hold the ids and this
            # one holds their input (see _resume).
            k, h = stopped
            n = r + sum(self._pieces) + (h.shape[1] if k else 0)
            resume = (k, h) if k else None
            info.update(stopped=True, layer=k, kept_ids=n - r)
            if n == r:
                return info
        held = spec._snapshot(cache) if state and resume is None else None
        self._convs = tuple(convs[drop:]) + ({"seen": list(ids[:n]), "cache": cache,
                                             "marks": [(n, held, True)],
                                             "draft": (None, ()), "resume": resume},)
        info.update(dropped=drop,
                    kept_mb=[round(self._conv_bytes(c) / 1e6, 1) for c in self._convs])
        return info

    def _keep_room(self, marks):
        """What the budget's pool could still take with what is held now and `marks` more
        marks, read without folding anything into the budget's measurements."""
        b = self._budget
        need = b._need(b.min_w - 1, False, False)
        if need is None:
            return None
        held = mx.get_active_memory() - sum(st.nbytes for st in b.stores.values())
        return (b.peak - b.cache - b._host_top - held - need - b.ring_bytes
                - marks * (b.state or 0.0))

    def reset(self):
        """Drop the conversation and its KV cache."""
        if self._budget is not None:
            self._budget.reset()
        self.history.clear()
        self._cache, self._seen = make_prompt_cache(self.model), []
        self._marks = []
        if self._vision is not None:
            self._vision.box.layout = None      # the positions went with the cache
        if self._draft_model is not None:
            self._draft_cache = self._new_draft_cache()
            self._draft_seen = []
        self._ended_cleanly = False
        # Positions removed by stub="slice": the RoPE hole the survivors sit after and
        # the difference between a cache's offset and len(_seen).
        self._gap = 0
        self._refeed_s = self._slice_s = 0.0

    def stats(self):
        """Speed and memory of the most recent turn."""
        return dict(self._last)

    def stream(self, message=None, messages=None, max_tokens=None,
               resume=None, tools=None, response_format=None, reply=None,
               prefill_only=False, sampling=None, keep_on_close=False):
        """Yield reply text as it is produced, from `message`, a full `messages` list, a
        resumed core, a picked-up reply or a prefill-only turn."""
        # A `messages` call must render the WHOLE list, since the incremental path
        # builds a turn as head + history[-1] + tail and silently dropped every system
        # message ever passed.

        # Tools belong to the request and not to the object, so a stale list left on
        # `self` never reaches the next turn.

        # Images ride a plain `messages` turn only, since a memory core and the batch
        # number positions by cache offset alone.
        imgs = vision.images_of(messages) if messages is not None else []
        if imgs and not self.supports_images:
            raise ValueError(f"this model takes no images ({self._vision_why})")
        if imgs and (resume is not None or prefill_only):
            raise ValueError("images go in a plain messages turn: not with a "
                             "memory core (--memory) or the batch")
        if self._vision is not None and message is None:
            # `message` goes on from the cache the last turn left: its
            # positions go on past that turn's images (Layout.pos).
            self._vision.box.layout = None
        self._img, lay, vision_s = (), None, 0.0
        if reply is not None:
            self._tools = tools         # nothing renders; the forcer reads them
        elif messages is not None:
            lead = []
            for m in messages:
                if m.get("role") != "system":
                    break
                lead.append({"role": "system", "content": str(m.get("content", ""))})
            # A different list or sheet cannot use the newest front but need not drop
            # the kept ones, which made an alternating application pay a full prefill
            # every call.
            self._tools = tools
            self._front_sys = lead
        elif tools is not None:
            raise ValueError("tools go with messages=, not with message=: "
                             "changing them mid-conversation would rewrite a "
                             "prefix the KV cache already holds")

        # Kept conversations are this call's to take up, and under --mem-peak-gb only
        # while the last turn found room for one (_keep_room, _KEEP_FIRST).
        first = (stream_fast._KEEP_FIRST and self._budget is not None
                 and self._room_ok is None and bool(self._budget.room_ok))
        keep = (self.reuse and (self._budget is None or self._room_ok or first)
                and messages is not None
                and resume is None and reply is None
                and all(isinstance(c, (KVCache, ArraysCache)) for c in self._cache))
        if not keep:
            self._convs = tuple(c for c in self._convs if c["cache"] is not self._cache)
        self._pieces, self._parked, self._dropped, self._resumed = (), 0, 0, None
        full_render = False
        if reply is not None:
            self.reset()
            self._cache = reply["cache"]
        elif messages is not None:
            # Same teardown as reset(): a fresh history is a new conversation whichever
            # door it came in by, since a sliced cache cannot be spliced against a
            # re-render.
            self.reset()
            if resume is not None:
                # A core replaces the cache outright, since its tokens are the user's
                # own past sessions and the id diff below would trim the whole memory
                # away.
                self._cache = resume["cache"]
                self._seen = list(resume["seen"])
                self.history = ([dict(m) for m in resume["history"]]
                                + [dict(m) for m in messages])
                self._ended_cleanly = False
            else:
                self.history = list(messages)
                full_render = True
        elif message is not None:
            self.history.append({"role": "user", "content": message})
        else:
            raise ValueError("pass either message or messages")

        # Append, never re-derive: the cache holds exactly the tokens the model
        # has seen, and this turn adds only the separator, the new message and
        # the generation prompt.
        if reply is not None:
            chunk, full = None, False
        elif full_render:
            chunk, full = self._raw_render(self.history), True
        elif messages is not None:
            chunk, full = self._continue_chunk(messages), False
        else:
            chunk, full = self._next_chunk(self.history[-1]["content"],
                                           not self._seen)
        # Tool-block tokens prefilled by _build_front and what the block cost this turn,
        # both carried into ttft by hand because they happen before the clock starts.
        warm, front_s = 0, 0.0
        if full:
            ids = toolfmt.encode(self.tokenizer, chunk)
            if imgs:
                # Every image's placeholder widened to its token count; the
                # spans key the kept caches (_conv_reach), and a front, all
                # text, never reaches past the first image.
                lay = self._vision.layout(imgs, ids)
                ids, self._img = lay.ids, lay.spans
                self._vision.install()
                self._vision.box.layout = lay
            plain = ids[:self._img[0][0]] if self._img else ids
            i = 0
            while i < len(self._seen) and i < len(ids) and self._seen[i] == ids[i]:
                i += 1
            if self._gap:
                # A sliced cache cannot be reused against a re-render: matching ids,
                # different RoPE, so start over rather than splice two numberings.
                self._cache, i, self._gap = make_prompt_cache(self.model), 0, 0
            elif i < len(self._seen):
                drop = len(self._seen) - i
                if trim_prompt_cache(self._cache, drop) != drop:
                    self._cache, i = make_prompt_cache(self.model), 0
            if not i:
                # Nothing matched, so the cache is empty and the tool block can go
                # straight in, after the kept conversations this turn has no room for
                # are dropped.
                t_front = time.perf_counter()
                i = self._take_conv(ids) if keep else 0
                if not i:
                    i = self._reuse_front(plain)
                if not i:
                    # First request with these tools: the block is prefilled
                    # for real, so it counts as prompt and not as cache.
                    warm = i = self._build_front(plain)
                front_s = time.perf_counter() - t_front
            kept, suffix = ids[:i], ids[i:]
            if lay is not None:
                # The tower over the images the cache does not hold yet.
                vision_s = self._vision.encode(lay, i)
        elif reply is not None:
            kept, suffix = list(reply["seen"][:-1]), list(reply["seen"][-1:])
            i = len(kept)
        else:
            kept = list(self._seen)
            suffix = toolfmt.encode(self.tokenizer, chunk, add_special_tokens=False)
            i = len(kept)

        # The stub is the last thing in every generation prompt, whichever of
        # the three paths above built it, so one test covers them all.
        L = len(self._stub_ids)
        has_stub = bool(L) and suffix[-L:] == self._stub_ids
        if has_stub and self.stub_mode == "none":
            suffix, has_stub = suffix[:-L], False
        self._seen = kept + suffix
        stub_at = len(self._seen) - L if has_stub else None
        if prefill_only:
            # Chunked as generate_step would have, because a newcomer's first token
            # through llm.stream made every running row wait.
            self._prefill(suffix[:-1], None if self.prefill_step is not None else
                          _mlx_prefill_step())
            return

        self._refeed_s = self._slice_s = 0.0
        # Same per-turn-difference trick for the adaptive draft's round
        # counters: the decision is made inside spec_draft's loop, and what a
        # caller wants back is what THIS reply's chains did.
        d0 = dict(spec_draft.STATS)
        # Expert rows are counted by stream_fast.STATS; the first token's snapshot leaves the
        # prefill's own reads out of what a decode token costs.
        rows0 = None
        t0, first, n, parts, snap = time.perf_counter(), None, 0, [], None
        gen_prompt, mark = suffix, None
        if has_stub and self.stub_mode == "refeed" and len(suffix) > L and not keep:
            # Everything but the stub goes in first and the cache is frozen
            # there: that snapshot IS the official rendering of this turn, up
            # to and including `<|im_start|>assistant\n`. The stub is fed only
            # to start the generation, and dies with the restore.
            self._prefill(suffix[:-L])
            ts = time.perf_counter()
            snap = clone_cache(self._cache)
            self._refeed_s += time.perf_counter() - ts
            gen_prompt = suffix[-L:]

        # A thinking budget counts one sampler call a reply token, so its turn decodes the plain way:
        # no MTP head, draft or forced rows, whose rounds call the sampler once per row.
        cap = self._think_cap(sampling) if reply is None else None
        if cap is not None:
            sampling = cap
            if self._spec is not None or self._draft_model is not None or self._draft_auto:
                note("  thinking budget: this turn decodes without the MTP head or draft")
        # `draft` is this turn's: None on a turn the decode policy below runs without an
        # auto draft, whose cache then waits for the next turn that has it
        # (_sync_draft_cache diffs ids, so it catches up there); known limit: a sampled
        # turn decodes without the draft model.
        draft = None if sampling else self._draft_model
        # Grammar-forced rows (forced_rows.py) ride a verify pass that is already
        # running, or one of their own through spec.ForcedSpeculator, which pays or not
        # by turn (ForcedPolicy).
        forcer_ok = cap is None and (draft is not None or bool(gen_prompt))
        forcer = (forced_rows.make(self.tokenizer, self._tools, response_format)
                  if forcer_ok else None)
        if forcer is not None and reply is not None:
            # The grammar stands where the reply does, its last token included.
            for t in reply["seen"][len(reply["seen"]) - reply["emitted"]:]:
                forcer.feed(t)
        sp = self._spec if cap is None else None
        forced_only = None      # the no-MTP policy's arm this turn, if it chose one
        decode_arm = None       # the MTP/auto-draft policy's arm this turn, likewise
        decode_policy = None    # whichever of the two policies above ran
        if sp is None and draft is None and forcer is not None:
            if self.forced_policy is None:
                self.forced_policy = ForcedPolicy()
            forced_only = self.forced_policy.pick()
            if forced_only:
                sp = spec.ForcedSpeculator(self.model)
            else:
                forcer = None
        elif gen_prompt and (sp is not None or (self._draft_auto and not sampling)):
            # An MTP head is loaded: the MTP round is one arm among plain and
            # forced-only, and a draft found on disk is another that runs only while it
            # measures faster.
            lead = "mtp" if sp is not None else "draft"
            if forcer is not None:
                if self.decode_policy_call is None:
                    self.decode_policy_call = ForcedPolicy(
                        (lead, "forced_only", "plain"))
                decode_policy = self.decode_policy_call
            else:
                if self.decode_policy_free is None:
                    # MOE_SPEC_AHEAD (CUDA) runs these MTP turns, and LeadPolicy chooses for them.
                    pol = LeadPolicy if lead == "mtp" and spec.Speculator.ahead else ForcedPolicy
                    self.decode_policy_free = pol((lead, "plain"))
                decode_policy = self.decode_policy_free
            decode_arm = decode_policy.pick()
            if decode_arm == "plain":
                sp = forcer = draft = None
            elif decode_arm == "forced_only":
                draft = None
                sp = spec.ForcedSpeculator(self.model)
            # decode_arm == "mtp"/"draft": the drafter stays, and the forcer
            # rides its rounds exactly as it always has.
        if keep:
            # A mark: where the next request can take this conversation up again, at the
            # end of this prompt less its stub, with its recurrent state held by
            # reference (spec._snapshot).
            cut = len(self._seen) - (L if has_stub else 1)
            ahead = cut - len(kept)
            if ahead >= 0:
                held = None
                if _has_recurrent(self._cache):
                    if ahead:
                        lib = (speculative_generate_step if draft is not None
                               else generate_step)
                        step = (None if sp is not None or self.prefill_step is not None
                                else _mlx_prefill_step(lib))
                        self._prefill(gen_prompt[:ahead], step)
                        gen_prompt = gen_prompt[ahead:]
                    held = spec._snapshot(self._cache)
                mark = (cut, held, messages[-1].get("role") == "user")
        elif draft is None and sp is None and len(gen_prompt) > MIN_PIECE:
            fit = stream_fast.prefill_fit(self._cache)
            if fit is not None and fit < len(gen_prompt):
                # Over the prefill budget, which mlx_lm's own prefill knows
                # nothing of: every id but the last goes in here, in the
                # chunks it would have cut (_prefill parks and cuts).
                self._prefill(gen_prompt[:-1], None if self.prefill_step is not None else
                              _mlx_prefill_step())
                gen_prompt = gen_prompt[-1:]
        # The speculative arm decodes over THIS cache, so spec.generate leaves it
        # holding the prefill plus exactly the accepted tokens, and a turn with nothing
        # to prefill falls back.
        if sampling is not None and sampling.seed is not None and reply is None:
            mx.random.seed(sampling.seed)   # a resumed reply goes on with the draws it had
        mtp = spec.Stats() if (sp is not None and gen_prompt) else None
        if sp is not None and mtp is None:
            note("  mtp: this turn has nothing to prefill; ordinary decode")
        if mtp is not None:
            self._last_mtp = mtp          # the round records, for harnesses
            sp.forcer = forcer
            sp.sampling = sampling
            stream = sp.stream_response(
                self.tokenizer, list(gen_prompt),
                max_tokens or self.max_tokens, self._cache, mtp,
                self.prefill_step)
        else:
            prompt_cache = self._cache
            if draft is not None:
                # speculative_generate_step splits ONE flat list at len(model.layers),
                # so the draft's half must already hold the same prefix self._cache
                # does.
                self._sync_draft_cache(
                    self._seen[: len(self._seen) - len(gen_prompt)])
                prompt_cache = list(self._cache) + list(self._draft_cache)
            # With draft_tau set, spec_draft.stream decides the chain length per token
            # from the draft's own margin with draft_tokens as the cap, and None keeps
            # mlx_lm's own call.
            adaptive = (draft is not None
                        and (self.draft_tau is not None or forcer is not None))
            # prefill_step None leaves each loop its own chunking, while 0 means one
            # pass here too, at the price of a prefill-sized peak.
            step_kw = ({} if self.prefill_step is None else
                       {"prefill_step_size": self.prefill_step or len(gen_prompt)})
            # No draft: spec_draft.plain, mlx_lm's stream with the first
            # token out before the forward that feeds it back in.
            stream = (spec_draft.stream if adaptive
                      else mlx_lm.stream_generate if draft is not None
                      else spec_draft.plain)(
                self.model, self.tokenizer, mx.array(gen_prompt),
                max_tokens=max_tokens or self.max_tokens,
                prompt_cache=prompt_cache, **step_kw,
                **({"sampler": sampling} if sampling else {}),
                # T4: mlx_lm's own speculative decode, the draft model loaded
                # once in __init__. num_draft_tokens is stream_generate's own
                # kwarg name for k; omitted entirely when there is no draft so
                # ordinary decode's kwargs stay exactly what they always were.
                **({"draft_model": draft,
                    "num_draft_tokens": self.draft_tokens}
                   if draft is not None else {}),
                **({"draft_tau": self.draft_tau} if adaptive else {}),
                **({"forcer": forcer} if forcer is not None else {}))
        draft_hits = 0
        try:
            for resp in stream:
                if first is None:
                    first = time.perf_counter() - t0
                    rows0 = stream_fast.STATS["rows_read"]
                self._seen.append(resp.token)
                parts.append(resp.text)
                n += 1
                if draft is not None:
                    # GenerationResponse.from_draft is the only acceptance signal
                    # mlx_lm hands back (no per-round count exists), so acceptance
                    # is the share of emitted tokens the draft actually proposed.
                    draft_hits += int(resp.from_draft)
                yield resp.text
        except GeneratorExit:
            # The caller hung up, so the turn ends as a max_tokens one and a cache still
            # off `_seen` is dropped rather than kept.
            if not keep_on_close:
                raise
            stream.close()
            if any(isinstance(c, KVCache) and
                   len(self._seen) - (c.offset - getattr(c, "gap", 0)) not in (0, draft is not None)
                   for c in self._cache):
                self.reset()
                return

        el = time.perf_counter() - t0
        if n == 0:
            # Nothing was decoded, so the caches never saw this prompt: take it back out
            # of `_seen`, close the turn empty and leave stats() as it was.
            stored = {c.offset - getattr(c, "gap", 0) for c in self._cache
                      if isinstance(c, KVCache)}
            if len(stored) == 1:
                del self._seen[stored.pop():]
            self.history.append({"role": "assistant", "content": ""})
            self._ended_cleanly = False
            return
        # A `reply` is the tail of a turn that shared forwards with others:
        # not a turn of either arm, so not a sample.
        if forced_only is not None and reply is None and n > 1 and el > first:
            self.forced_policy.record(forced_only, (el - first) / (n - 1) * 1e3)
        if decode_arm is not None and reply is None and n > 1 and el > first:
            decode_policy.record(decode_arm, (el - first) / (n - 1) * 1e3)
            if self._draft_auto:
                self._drop_losing_draft()
        self._ended_cleanly = bool(resp.finish_reason == "stop")
        gen = self._seen[len(self._seen) - n:] if n else []
        # `self._seen` right now is exactly what the draft cache was fed this turn,
        # captured before _slice_stub and _refeed rewrite it in ways the draft's storage
        # never saw.
        draft_fed = list(self._seen) if draft is not None else None
        # Whether the model opened a think block of its own, which only the none mode
        # can provoke.
        opened = bool(gen and self._stub_ids and gen[0] == self._stub_ids[0])
        if draft is not None:
            # A round that ends on a target-produced token never feeds it back, so heal
            # the one-token gap here before anything downstream assumes the cache tracks
            # `_seen`.
            c0 = self._cache[0]
            stored = c0.offset - getattr(c0, "gap", 0)
            if stored < len(self._seen):
                self._prefill(self._seen[stored:])
        if self.stub_mode == "slice" and stub_at is not None and not keep:
            self._slice_stub(stub_at)
        elif snap is not None:
            self._refeed(snap, self._seen[:stub_at], gen)

        if draft is not None:
            # The draft's rollback bookkeeping leaves its cache up to one token off, so
            # read the cache's own offset instead of assuming every position of
            # `draft_fed` landed.
            off = self._draft_cache[0].offset if self._draft_cache else 0
            self._draft_seen = draft_fed[:off]

        if keep:
            # What a later `messages` call may take up: this cache as it was generated
            # and exactly the ids it holds.
            for c in self._cache:
                if isinstance(c, KVCache):
                    assert c.offset == len(self._seen), (c.offset, len(self._seen))
            marks = self._marks
            if mark is not None:
                # The newest mark and the latest one of a turn that answered a user
                # query, which is where a re-render then stops agreeing.
                marks = ([] if mark[2] else [mk for mk in marks if mk[2]][-1:]) + [mark]
            # The newest: _take_conv made its room when the turn began.
            self._convs += ({"seen": list(self._seen), "cache": self._cache,
                             "marks": marks, "img": self._img,
                             "draft": (self._draft_cache, self._draft_seen)},)
            # The turn's own list goes too: held here, a mark this one dropped
            # would keep its recurrent state (~64 MB) until the next request.
            self._marks = []
        room = None
        if self._budget is not None and self.reuse:
            # Read with the turn over and what it keeps evaluated, and no room means
            # nothing stays and the next turn makes no mark.
            room = self._keep_room(0 if keep else 2)
            ok = room is not None and room >= 0
            if not ok:
                self._convs, self._marks = (), []
            if ok != self._room_ok:
                note(f"  conversation reuse under the budget: {'on' if ok else 'off'}, "
                      f"room beside the pool {room / 1e6:.0f} MB" if room is not None
                      else "  conversation reuse under the budget: off", flush=True)
            self._room_ok = ok
            if room is not None and ok != self._budget.room_ok:
                # The next process's first turn goes by it (_KEEP_FIRST).
                self._budget.room_ok = ok
                self._budget._save_calibration()

        gov = stream_fast._GOV
        self._last = {
            "tokens": n,
            # `front_s` is prefilling or copying the tool block before the clock starts,
            # which a client waits for and ttft must therefore carry.
            "ttft_s": round((first or 0.0) + front_s + vision_s, 2),
            "decode_tps": round((n - 1) / (el - first), 2)
            if n > 1 and el > first else 0.0,
            "rss_gb": round(stream_fast.max_rss() / 1e9, 2),
            # What Activity Monitor calls Memory, now and at its most so far:
            # rss_gb above misses Metal buffers, so a person reading the chat
            # line needs these.
            "mem_gb": round(stream_fast.phys_footprint() / 1e9, 2),
            "mem_peak_gb": round(stream_fast.phys_footprint(peak=True) / 1e9, 2),
            # `warm` is the tool block on the request that prefilled it: prompt, not
            # cache.
            "prompt_tokens": len(suffix) + warm,
            "cached_tokens": i - warm,
            # Conversations kept for reuse, least recently used first, and
            # what each holds; keep_gb caps them, the last counted.
            "kept_convs": len(self._convs),
            "kept_mb": [round(self._conv_bytes(c) / 1e6, 1) for c in self._convs],
            # The passes this turn's prefill ran (_prefill's; mlx_lm's own
            # are not listed), the arena parked for them, the kept
            # conversations dropped before or for them.
            "prefill_pieces": list(self._pieces),
            "arena_parked_gb": round(self._parked / 1e9, 2),
            "convs_dropped": self._dropped,
            "resumed": self._resumed,
            # Under the budget: its room beside the pool with what this turn
            # keeps (_keep_room); a conversation is kept while it is >= 0.
            "keep_room_mb": None if room is None else round(room / 1e6, 1),
            "arena": ("store" if self._budget else gov.cap if gov
                      else self.resident),
            # The arena's own size, what /status shows beside the store's, and the expert rows
            # read from disk while this reply decoded (stream_fast.STATS, prefill excluded).
            "arena_slots": gov.cap if gov else None,
            "arena_gb": round(gov.cap * gov.slot_gb, 2) if gov else None,
            "expert_rows": None if rows0 is None else stream_fast.STATS["rows_read"] - rows0,
            "expert_rows_per_token": (round((stream_fast.STATS["rows_read"] - rows0) / (n - 1), 2)
                                      if rows0 is not None and n > 1 else None),
            # The process budget and the live instrument behind it, since
            # ru_maxrss above does not see Metal buffers at all: a host
            # application checking whether it is within --mem-pct needs this
            # number, not rss_gb.
            "budget_gb": (round(self._budget.peak / 1e9, 2) if self._budget
                          else round(gov.budget_gb, 2) if gov else None),
            # --mem-peak-gb: the shared store's size and the terms its pool
            # was computed from (MemBudget.report).
            "store": self._budget.report() if self._budget else None,
            "mlx_active_gb": round(mx.get_active_memory() / 1e9, 2),
            "profile": self.profile,
            "stub": self.stub_mode,
            "opened_think": opened,
            "refeed_s": round(self._refeed_s, 3),
            "slice_s": round(self._slice_s, 3),
            "front_s": round(front_s, 3),
            "kv_gap": self._gap,
        }
        if mtp is not None and sp is self._spec is not None:
            # Added only when the MTP head itself ran the round, so stats() with the
            # flag off is the dict it has always been.
            self._last.update(
                mtp_acceptance=round(mtp.acceptance, 3),
                mtp_rounds=mtp.rounds,
                # Per draft position, so a tokens/round that misses its target
                # says WHICH draft collapsed instead of only that one did.
                mtp_depth=[round(p, 3) for p in mtp.per_depth],
                mtp_tokens_per_round=round(mtp.per_round, 3),
                # Gamma drafts are gamma sequential passes of the head, so their time is
                # split out or a gamma sweep cannot say which side got slower.
                mtp_draft_s=round(mtp.draft_s, 3))
        if draft is not None:
            # Added only when a draft model ran this turn, same convention as
            # the mtp_* keys above: stats() with the flag off stays the dict
            # it has always been.
            self._last.update(
                draft_acceptance=round(draft_hits / max(n, 1), 3),
                draft_k=self.draft_tokens)
        if draft is not None and self.draft_tau is not None:
            # What the gate did this turn: mean_k is the rows sent to the verify pass
            # per round and steps_wasted the draft step a cut chain paid for.
            rounds = spec_draft.STATS["rounds"] - d0["rounds"]
            drafted = spec_draft.STATS["drafted"] - d0["drafted"]
            steps = spec_draft.STATS["steps"] - d0["steps"]
            self._last.update(
                draft_tau=self.draft_tau,
                draft_rounds=rounds,
                draft_mean_k=round(drafted / max(rounds, 1), 2),
                draft_steps_wasted=steps - drafted,
                draft_tokens_per_round=round(n / max(rounds, 1), 2))
        if lay is not None:
            # Only on a turn with images, as the mtp_* keys: a text turn's
            # stats() is the dict it has always been.
            self._last.update(images=len(lay.spans), image_tokens=lay.tokens(),
                              vision_s=round(vision_s, 2), **self._vision.last)
        if forced_only is not None:
            self._last["forced_only"] = forced_only
        if cap is not None:
            # Only on a turn with a thinking budget: whether it cut the thinking short.
            self._last["think_cut"] = cap.cut
            if cap.cut:
                note(f"  thinking budget: {cap.budget} tokens reached, the early-stop text forced")
        if decode_arm is not None:
            self._last["decode_arm"] = decode_arm
        if forcer is not None:
            self._last.update(
                forced_rows=(mtp.forced if mtp is not None else
                             spec_draft.STATS["forced"] - d0["forced"]),
                forced_accepted=(mtp.forced_accepted if mtp is not None else
                                 spec_draft.STATS["forced_accepted"]
                                 - d0["forced_accepted"]))
        self.history.append({"role": "assistant", "content": "".join(parts)})
        # Give MLX's freed buffers back while the server idles, with the last response
        # dropped first so its logprobs are not freed into the cache on return.
        del resp
        mx.clear_cache()
        if self._spec_cold is not None:
            sp, self._spec_cold = self._spec_cold, None
            self._warm_spec(sp)

    _spec_cold = None

    def ask(self, message=None, messages=None, max_tokens=None, resume=None,
            tools=None, response_format=None, sampling=None):
        """Blocking version of stream(): returns the whole reply."""
        return "".join(self.stream(message, messages, max_tokens, resume,
                                   tools, response_format, sampling=sampling))
