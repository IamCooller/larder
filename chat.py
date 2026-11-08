#!/usr/bin/env python3
"""The terminal chat: every reply is followed by its decode speed, first-token wait, memory and cache use."""

import argparse
import sys
import threading
import time

from engine_args import add_engine_flags, attach_governor, engine_kwargs, finish, memory_sleep
from moe_stream import StreamingLLM


def build_parser():
    ap = argparse.ArgumentParser()
    add_engine_flags(ap)
    ap.add_argument("--top-k", type=int, default=0, help="0 = model default")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--mtp-probe", default=None, metavar="DIR",
                    help="Flash-Next only (with --layers 0-47): load the 4-bit MTP head from DIR and "
                         "let a probe record its guesses (mtp_flash.py); off by default")
    ap.add_argument("--layers", default=None, metavar="LO-HI",
                    help="run only decoder layers LO..HI (inclusive) of a Flash-Next pack: "
                         "one stage of a pipeline over several machines (stage.py)")
    ap.add_argument("--no-sleep", action="store_true",
                    help="skip consolidation on exit (it costs one generation "
                         "per turn of the session)")
    ap.add_argument("--pause-draft", nargs="?", const=1.5, type=float,
                    default=None, metavar="SECONDS",
                    help="when you stop typing for this long (default 1.5), "
                         "prefill the line as a draft so Enter finds the "
                         "prompt already in the cache (TTFT ~0.1 s instead "
                         "of 6-10 s on a long message). Costs one prefill "
                         "per pause, and the GPU is busy while it runs. "
                         "Needs a terminal with readline; not with "
                         "--memory")
    ap.add_argument("--think", action="store_true",
                    help="let the model reason before answering (default: off)")
    return ap


def pause_watcher(llm, busy, committed, state, pause):
    """Thread: after a pause in typing, prefill the half-typed line as a draft so Enter finds a warm cache."""
    import readline
    text, since = "", time.monotonic()
    while True:
        time.sleep(0.2)
        now = time.monotonic()
        cur = readline.get_line_buffer().strip()
        if cur != text:
            text, since = cur, now
            continue
        if (not cur or cur.startswith("/") or cur == state["drafted"]
                or now - since < pause or not state["typing"].is_set()
                or not busy.acquire(blocking=False)):
            continue
        try:
            state["drafted"] = cur
            for _ in llm.stream(messages=committed[0] + [
                    {"role": "user", "content": cur}], max_tokens=1):
                pass
        except Exception as e:
            print(f"\n\033[2m  (draft failed: {e})\033[0m", flush=True)
        finally:
            busy.release()


def build_llm(args):
    """The StreamingLLM for these flags, refusing bad mixes first; probes call it without the REPL."""
    finish(args)
    if args.pause_draft is not None and args.memory:
        sys.exit("  --pause-draft sends the line as a stateless request; "
                 "--memory keeps its own cache.\n")
    layers = None
    if args.layers:
        try:
            import stage
        except ImportError:
            sys.exit("  --layers needs stage.py, which this install does not include.\n")
        layers = stage.parse(args.layers)
    llm = StreamingLLM(model=args.model, top_k=args.top_k, max_tokens=args.max_tokens,
                       thinking=args.think, layers=layers, **{**engine_kwargs(args), "adaptive": False})
    # The arena governor as server.py attaches it, held to --arena-min..--arena-max slots.
    if args.adaptive and getattr(llm, "_budget", None) is None:
        attach_governor(llm, args.mem_pct, args.reserve_pct, args.host_gb, args.arena_min, args.arena_max)
    if args.mtp_probe:
        import mtp_flash
        mtp_flash.attach(llm, args.mtp_probe)
    return llm


def main():
    args = build_parser().parse_args()
    llm = build_llm(args)

    # Under --adaptive the arena is not a fixed size, so the number that
    # describes the process is the budget it is held to, not --resident (which
    # --no-adaptive still pins it to and which this banner shows instead).
    arena_desc = (f"peak budget {args.mem_peak_gb} GB, shared expert store"
                  if args.mem_peak_gb else
                  f"budget {args.mem_pct}% of RAM" if args.adaptive
                  else f"arena {args.resident} experts/layer")
    print(f"\n{args.model.rstrip('/').split('/')[-1]} | {arena_desc} "
          f"| thinking {'on' if args.think else 'off'}"
          f" | stub {args.stub}"
          + (" | mtp" if args.mtp else "")
          # The engine's draft, not the flag's: "auto" may have found one or not.
          + (f" | draft{' (auto, kept while faster)' if llm._draft_auto else ''}: "
             f"{llm.draft_path.rstrip('/').split('/')[-1]} "
             + (f"(adaptive, tau {args.draft_tau}, k<={llm.draft_tokens})"
                if args.draft_tau is not None
                else f"({llm.draft_tokens} tokens)") if llm.draft_path else ""))
    print("weights stream from SSD as they are needed; /quit to exit\n")

    store, turns = None, []
    if args.memory:
        import kvmem
        store = kvmem.Store(args.model, tag=args.memory)
        print(f"\033[2m{store.summary()}\033[0m")
        store.ensure(llm)
        print()

    # One engine, two threads: the idle sleeper may only run while the loop
    # sits in input(), and a turn typed mid-rebuild waits for the lock.
    busy = threading.Lock()
    last = [time.monotonic()]

    def idle_sleep():
        while True:
            time.sleep(30)
            if not turns or time.monotonic() - last[0] < args.sleep_after \
                    or not busy.acquire(blocking=False):
                continue
            try:
                pending, turns[:] = list(turns), []
                msg = memory_sleep(store, llm, pending)
                print(f"\n\033[2m  (idle {args.sleep_after}s: {msg})\033[0m\n"
                      f"\033[1myou:\033[0m ", end="", flush=True)
            except Exception as e:
                print(f"\n\033[2m  (idle sleep failed: {e})\033[0m", flush=True)
            finally:
                busy.release()

    if store is not None and args.sleep_after > 0:
        threading.Thread(target=idle_sleep, daemon=True).start()

    # With --pause-draft every turn goes through stream(messages=...) as server.py's do, so a draft and the Enter after it share a kept conversation.
    committed, state = [[]], {"typing": threading.Event(), "drafted": None}
    if args.pause_draft is not None:
        try:
            import readline         # noqa: F401
        except ImportError:
            sys.exit("  --pause-draft needs the readline module.\n")
        if not sys.stdin.isatty():
            sys.exit("  --pause-draft needs a terminal.\n")
        llm.reuse, llm.keep_convs = True, max(llm.keep_convs, 2)
        threading.Thread(target=pause_watcher, daemon=True, args=(
            llm, busy, committed, state, args.pause_draft)).start()

    while True:
        state["typing"].set()
        try:
            user = input("\033[1myou:\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        finally:
            state["typing"].clear()
        if not user:
            continue
        if user in ("/quit", "/exit"):
            break
        if user == "/nosleep":
            args.no_sleep = True
            print("consolidation on exit disabled for this session\n")
            continue
        if user == "/reset":
            llm.reset()
            committed[0] = []
            print("history cleared\n")
            continue
        if user == "/stats":
            print(f"{llm.stats()}\n")
            continue

        if user == "/sleep":
            if store is not None:
                with busy:
                    pending, turns[:] = list(turns), []
                    print(f"{memory_sleep(store, llm, pending)}\n")
            else:
                print("/sleep needs --memory\n")
            continue

        turns.append(user)
        print("\033[1mmodel:\033[0m ", end="", flush=True)
        reply = []
        busy.acquire()
        try:
            gen = (llm.stream(messages=committed[0] + [
                       {"role": "user", "content": user}])
                   if args.pause_draft is not None
                   else llm.stream(user))
            for piece in gen:
                print(piece, end="", flush=True)
                reply.append(piece)
            committed[0] = list(llm.history)
        finally:
            busy.release()
            last[0] = time.monotonic()
        if store is not None and user.rstrip().endswith("?") \
                and kvmem.FIRST_PERSON.search(user):
            # A question about the user is checked against the memory core and flagged, not suppressed.
            ok, why = kvmem.ground("".join(reply), store.core_text(), question=user)
            if not ok:
                print(f"\n\033[33m  memory: not confirmed: "
                      f"{', '.join(why[:3])}\033[0m", end="")
        s = llm.stats()
        print(f"\n\033[2m  {s['decode_tps']} tok/s decode | ttft {s['ttft_s']}s | "
              f"{s['tokens']} tokens | mem {s['mem_gb']} GB (peak {s['mem_peak_gb']}) | "
              f"{s['prompt_tokens']} new / {s['cached_tokens']} cached | "
              f"arena {s['arena']}\033[0m\n")

    if store is not None and not args.no_sleep:
        # Ctrl-C here costs the rebuild, never the words: append_session has
        # already run by then, so the next /sleep picks the session up.
        try:
            print("\033[2mconsolidating memory (Ctrl-C to skip)...\033[0m",
                  flush=True)
            with busy:               # an idle rebuild may be mid-flight
                print(f"\033[2m  {memory_sleep(store, llm, turns)}\033[0m")
        except KeyboardInterrupt:
            print("\nskipped, the session is journalled, /sleep next time")


if __name__ == "__main__":
    main()
