"""The engine flags chat.py and server.py share, and how they become StreamingLLM keywords."""

import argparse
import sys

import engine_log

# Keep torch and sklearn out of the process: the tokenizer import pulls them in unasked, and forced_rows lifts the block when xgrammar needs torch.
if "sklearn" not in sys.modules:
    sys.modules["sklearn"] = None
if "torch" not in sys.modules:
    sys.modules["torch"] = None

import stream_fast
from moe_stream import DEFAULT_MODEL, StreamingLLM
from mtp import DEFAULT_MTP

DEFER = {"auto": "auto", "always": True, "never": False}


def add_engine_flags(ap):
    """The flags both entry points take, with one default and one meaning each."""
    ap.add_argument("--model", default=DEFAULT_MODEL, required=DEFAULT_MODEL is None,
                    help="checkpoint directory (./larder picks and fetches one)")
    ap.add_argument("--resident", type=int, default=None,
                    help="fixed arena size (21 when pinned); giving it pins the arena "
                         "unless --adaptive is passed too")
    ap.add_argument("--adaptive", action=argparse.BooleanOptionalAction, default=None,
                    help="size the arena from free memory and measured speed (default unless "
                         "--resident or --mem-peak-gb is given; --no-adaptive pins it to --resident)")
    ap.add_argument("--reserve-pct", type=int, default=15,
                    help="share of the machine to leave alone with --adaptive")
    ap.add_argument("--mem-pct", type=int, default=90,
                    help="process memory budget: MLX memory + --host-gb stays under this %% of the "
                         "machine (the arena with --adaptive, a dense target's batch KV always)")
    ap.add_argument("--host-gb", type=float, default=0.6,
                    help="python/numpy/MLX-runtime overhead the budget cannot see directly")
    ap.add_argument("--arena-min", type=int, default=21, help="with --adaptive: the arena's floor in slots")
    ap.add_argument("--arena-max", type=int, default=60,
                    help="with --adaptive: hard ceiling in slots; 0 = the model's expert count")
    ap.add_argument("--mem-peak-gb", type=float, default=None, metavar="G",
                    help="keep the process's peak memory under G GB: shared expert store instead "
                         "of per-layer arenas (replaces --resident and --adaptive)")
    ap.add_argument("--bias-budget", type=float, default=0.2,
                    help="bounded-loss cache bias: max relative router-mass loss per layer "
                         "(0 = unbounded, not safe in long dialogue)")
    ap.add_argument("--cache-bias", type=float, default=None,
                    help="cache-conditional routing; 0 = off, 0.5 = aggressive "
                         "(default 0.5; 0 with --mem-peak-gb)")
    ap.add_argument("--defer-mode", choices=tuple(DEFER), default="auto",
                    help="sync/defer choice: auto measures the crossover in-process (default), "
                         "always/never pin one side")
    ap.add_argument("--stub", default="slice", choices=StreamingLLM.STUB_MODES,
                    help="think-stub handling: keep, slice, refeed, none")
    ap.add_argument("--mtp", nargs="?", const=DEFAULT_MTP or "", default=None, metavar="PATH",
                    help="speculative decode with the packed MTP head at PATH"
                         + (f" (default {DEFAULT_MTP})" if DEFAULT_MTP else ""))
    ap.add_argument("--draft", default="auto", metavar="PATH",
                    type=lambda s: None if s.lower() == "none" else s,
                    help="speculative decode with a sibling checkpoint at PATH (mlx_lm's "
                         "draft_model, not --mtp's own head); auto (default): a smaller model with "
                         "the same tokenizer found on disk, kept while it pays; none: off")
    ap.add_argument("--no-draft", dest="draft", action="store_const", const=None,
                    help="same as --draft none")
    ap.add_argument("--draft-tokens", type=int, default=None, metavar="K",
                    help="draft tokens proposed per round (the cap with --draft-tau); "
                         "default 2 for an auto draft, 3 for --draft PATH")
    ap.add_argument("--draft-tau", type=float, default=None, metavar="TAU",
                    help="adaptive draft chain: keep drafting while the draft's top1-top2 "
                         "probability gap > TAU, up to --draft-tokens (default off)")
    ap.add_argument("--draft-window", type=int, default=0, metavar="W",
                    help="cap the draft's attention at 64 sink tokens + the last W "
                         "(0 = full cache, default)")
    ap.add_argument("--memory", nargs="?", const="default", default=None, metavar="TAG",
                    help="long-term KV memory for TAG (default: `default`), attached before the "
                         "first turn and rebuilt on sleep and at exit. See kvmem.py")
    ap.add_argument("--sleep-after", type=int, default=600, metavar="SECONDS",
                    help="with --memory: consolidate after this long without a finished turn "
                         "(0 = only on an explicit sleep and at exit)")
    ap.add_argument("--verbose", action="store_true",
                    help="print the engine's start-up and per-request notes too: memory budget, arena, "
                         "store, reads, kernels, shadow")


def finish(args):
    """Settle the defaults that depend on other flags, refuse the mixes the engine would refuse later,
    and silence the engine's notes unless --verbose."""
    engine_log.set_verbose(args.verbose)
    if args.mtp == "":
        sys.exit("  --mtp needs the folder of a packed MTP head: --mtp PATH (tools/mtp_pack.py packs one).\n")
    if args.adaptive is None:
        args.adaptive = args.resident is None and args.mem_peak_gb is None
    if args.resident is None:
        args.resident = 21
    # An auto draft only offers itself, so it is never an error beside --mtp.
    if args.draft not in (None, "auto") and args.mtp:
        sys.exit("  --draft and --mtp both drive speculative decode; pick one.\n")
    if args.draft_tokens is not None and not args.draft:
        sys.exit("  --draft-tokens only means something with --draft.\n")
    if args.draft_tau is not None and not args.draft:
        sys.exit("  --draft-tau only means something with --draft: there is no chain to shorten.\n")
    if args.draft_window and not args.draft:
        sys.exit("  --draft-window only means something with --draft: the target's cache is never windowed.\n")
    return args


def engine_kwargs(args):
    """The StreamingLLM keywords for the shared flags, after finish()."""
    return dict(resident=args.resident, adaptive=args.adaptive, bias_budget=args.bias_budget,
                cache_bias=args.cache_bias, stub=args.stub, defer=DEFER[args.defer_mode],
                mem_pct=args.mem_pct, reserve_pct=args.reserve_pct, host_gb=args.host_gb,
                mem_peak_gb=args.mem_peak_gb, mtp=args.mtp, draft=args.draft,
                draft_tokens=args.draft_tokens, draft_tau=args.draft_tau,
                draft_window=args.draft_window)


def attach_governor(llm, mem_pct, reserve_pct, host_gb, arena_min, arena_max):
    """The adaptive arena of both entry points, sized from free memory between arena_min and arena_max slots, since StreamingLLM's own governor has no ceiling and started at 136 slots on a 16 GB Mac, past the swap line."""
    glus = stream_fast.glus_of(llm.model)
    if not glus:        # dense model: no arena to size
        return
    stream_fast.make_governor(glus, stream_fast.n_experts_of(llm.model), stream_fast.top_k_of(llm.model),
                              mem_pct, reserve_pct, host_gb, cap_min=arena_min, cap_max=arena_max)


def memory_sleep(store, llm, turns):
    """Journal this session and rebuild the core only if the plan changed."""
    if not turns:
        return "nothing said this session"
    before = store.stats()["plan_sha"]
    store.append_session(turns)
    if store.plan()["sha"] == before:
        return "journalled; nothing new passed the write gate, core unchanged"
    store.rebuild(llm)
    return store.summary()
