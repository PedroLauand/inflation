from __future__ import annotations

from typing import Iterable, Protocol, runtime_checkable

import sympy as sp


@runtime_checkable
class RingDistributionProtocol(Protocol):
    @property
    def nof_outcomes(self) -> int:
        ...

    def prob_event_loop(self, outcomes: Iterable[int]) -> sp.Expr:
        ...

    def prob_event_line(self, outcomes: Iterable[int]) -> sp.Expr:
        ...


__all__ = ["RingDistributionProtocol"]
