#!/usr/bin/env python3
"""moe_stream.LeadPolicy (MOE_SPEC_AHEAD's arm choice) on scripted turn speeds, no model.

    python3 tests/test_lead_policy.py

Every arm runs once, plain first; then the faster arm runs and the slower one is re-probed on ForcedPolicy's
schedule; the arm in use stays while the other is within the arms' spread (hysteresis) and changes when it is not.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moe_stream import LeadPolicy


def play(speed, turns):
    """The arms LeadPolicy picks over `turns` turns when arm a's turn costs speed(a, turn) ms a token."""
    p, out = LeadPolicy(("mtp", "plain")), []
    for t in range(turns):
        a = p.pick()
        out.append(a)
        p.record(a, speed(a, t))
    return out


def test_lead_policy():
    # MTP clearly faster: plain once, then MTP, plain re-probed only at the longest interval.
    fast = play(lambda a, t: (6.0 if a == "mtp" else 7.3) + 0.05 * (t % 3), 40)
    assert fast[:3] == ["plain", "mtp", "mtp"], fast[:3]
    assert fast.count("plain") <= 3, fast
    # MTP slower: one MTP turn, then plain.
    slow = play(lambda a, t: (8.0 if a == "mtp" else 7.3) + 0.05 * (t % 3), 20)
    assert slow[:3] == ["plain", "mtp", "plain"] and slow.count("mtp") <= 3, slow
    # Within the spread the arm in use stays: MTP 7.3 +- 0.4 against plain 7.25 keeps MTP, re-probing plain often.
    near = play(lambda a, t: 7.3 + (-0.4 if t % 2 else 0.4) if a == "mtp" else 7.25 + 0.02 * (t % 2), 20)
    assert near[1] == "mtp" and near.count("mtp") >= 10, near
    # The arm in use changes when the other beats it by more than the spread: MTP gets slow after turn 10.
    turn = play(lambda a, t: (6.0 if t < 10 else 9.0) + 0.02 * (t % 2) if a == "mtp" else 7.3, 60)
    assert turn[-1] == "plain" and turn[-10:].count("plain") >= 8, turn
    print("  first turns", fast[:4], "| mtp slower", slow[:4], "| near", near[:6], "| mtp slows at 10: last ten",
          turn[-10:], " OK")


if __name__ == "__main__":
    test_lead_policy()
    print("test_lead_policy: ok")
