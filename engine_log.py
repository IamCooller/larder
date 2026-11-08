"""The engine's start-up and per-request notes, printed unless an entry point switched them off."""

verbose = True      # library users and tests see every note; chat.py and server.py turn them off without --verbose


def set_verbose(on):
    global verbose
    verbose = bool(on)


def note(*args, **kwargs):
    """print() for what only someone debugging the engine wants to read."""
    if verbose:
        print(*args, **kwargs)
