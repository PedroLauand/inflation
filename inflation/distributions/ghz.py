from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import sys
from typing import Iterable

import sympy as sp

if __package__ in (None, ""):
    _REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from inflation.distributions._common import parse_outcomes
else:
    from ._common import parse_outcomes


@lru_cache(maxsize=None)
def _ghz_prob_cached(event: tuple[int, ...]) -> sp.Expr:
    """Only the all-0 and all-1 binary events have support, each with probability 1/2."""
    first = event[0]
    if all(outcome == first for outcome in event):
        return sp.Rational(1, 2)
    return sp.Integer(0)


class GHZDistribution:
    """Binary GHZ distribution with identical loop and line probabilities."""

    @property
    def nof_outcomes(self) -> int:
        return 2

    def prob_event_loop(self, outcomes: Iterable[int]) -> sp.Expr:
        event = parse_outcomes(outcomes, max_outcome=self.nof_outcomes - 1)
        return _ghz_prob_cached(event)

    def prob_event_line(self, outcomes: Iterable[int]) -> sp.Expr:
        event = parse_outcomes(outcomes, max_outcome=self.nof_outcomes - 1)
        return _ghz_prob_cached(event)


__all__ = ["GHZDistribution"]
