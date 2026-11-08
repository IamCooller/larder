"""The engine's version: a protocol number bumped whenever the --list --json shape or the flags the menu uses change, plus the checkout's git commit."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROTOCOL = 1


def commit():
    """This checkout's HEAD, or "" when git is not here or the folder is not a checkout."""
    try:
        r = subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=5)
        return r.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def version():
    """The version object every launcher reports for the handshake."""
    return {"protocol": PROTOCOL, "commit": commit()}


def version_json():
    """The one line `launch.py --version` prints."""
    return json.dumps(version())
