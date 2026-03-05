"""
compare_nsi_old_new.py
--------------------------------------------------------------------
Compare the "old" NSI probabilities (from postquantum_2outcomes.py) with the
"new" event-level NSI-PR probabilities (inflation.distributions.nsi_pr)
for all binary events up to n=3.

Notes on "old" values:
- postquantum_2outcomes.py only defines prob_line(n) / prob_loop(n) for the
  all-zero event of length n.
- For non-all-zero events, old values are undefined. We print "N/A".
"""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Iterable, Optional
import sys

# Ensure repo root is on sys.path so "import inflation" works when running directly.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from inflation.distributions.nsi_pr import prob_event_line, prob_event_loop
from postquantum_2outcomes import prob_line, prob_loop

_SIG_FIGS = 6


def _old_prob_line(event: Iterable[int]) -> Optional[float]:
    evt = tuple(int(x) for x in event)
    if sum(evt) == 0:
        return prob_line(len(evt))
    return None


def _old_prob_loop(event: Iterable[int]) -> Optional[float]:
    evt = tuple(int(x) for x in event)
    if sum(evt) == 0:
        return prob_loop(len(evt))
    return None


def _fmt(val: Optional[float]) -> str:
    if val is None:
        return "N/A"
    return f"{val:.{_SIG_FIGS}g}"


def main() -> None:
    for n in range(1, 4):
        print(f"\n=== n = {n} ===")
        print("event | new_line | old_line | new_loop | old_loop")
        for evt in product((0, 1), repeat=n):
            new_line = prob_event_line(evt)
            new_loop = prob_event_loop(evt)
            old_line = _old_prob_line(evt)
            old_loop = _old_prob_loop(evt)
            evt_str = "".join(str(x) for x in evt)
            print(
                f"{evt_str:>4} | {_fmt(new_line):>9} | {_fmt(old_line):>9} | "
                f"{_fmt(new_loop):>9} | {_fmt(old_loop):>9}"
            )


if __name__ == "__main__":
    main()
