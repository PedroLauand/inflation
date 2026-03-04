from __future__ import annotations

from itertools import product
from typing import Callable, Iterable, Iterator, Sequence, Tuple

NativeCoarsen = Tuple[Tuple[int, ...], ...]


def parse_outcomes(outcomes: Iterable[int], *, max_outcome: int) -> Tuple[int, ...]:
    """Parse and validate a non-empty outcome tuple with bounded labels."""
    parsed = tuple(int(x) for x in outcomes)
    if not parsed:
        raise ValueError("Provide at least one outcome.")
    if any((x < 0 or x > max_outcome) for x in parsed):
        raise ValueError(f"Outcomes must be in {{0,1,...,{max_outcome}}}.")
    return parsed


def normalize_coarsen(
    coarsen: Sequence[Sequence[int]] | None,
    *,
    native_outcomes: int = 4,
) -> NativeCoarsen:
    """
    Normalize and strictly validate a coarse-graining partition.

    The groups must be non-empty, disjoint, and an exact cover of
    {0, 1, ..., native_outcomes-1}. Group order defines new outcome labels.
    """
    if coarsen is None:
        return tuple((idx,) for idx in range(native_outcomes))

    groups: list[Tuple[int, ...]] = []
    seen: list[bool] = [False] * native_outcomes
    count_seen = 0

    for group in coarsen:
        group_tuple = tuple(int(x) for x in group)
        if not group_tuple:
            raise ValueError("Coarsen groups must be non-empty.")
        for x in group_tuple:
            if x < 0 or x >= native_outcomes:
                raise ValueError(
                    f"Coarsen references native outcome {x}, "
                    f"but valid labels are 0..{native_outcomes - 1}."
                )
            if seen[x]:
                raise ValueError("Coarsen groups must be disjoint.")
            seen[x] = True
            count_seen += 1
        groups.append(group_tuple)

    if count_seen != native_outcomes:
        raise ValueError(
            "Coarsen groups must form an exact partition of "
            f"{{0,1,...,{native_outcomes - 1}}}."
        )
    return tuple(groups)


def expand_coarse_event(
    coarse_event: Tuple[int, ...],
    coarsen_key: NativeCoarsen,
) -> Iterator[Tuple[int, ...]]:
    """Expand a coarse-labeled event into native events via Cartesian product."""
    expanded_groups = []
    for x in coarse_event:
        expanded_groups.append(coarsen_key[x])
    yield from product(*expanded_groups)


def cyclic_canonical(event: Tuple[int, ...]) -> Tuple[int, ...]:
    """Return the lexicographically minimal cyclic rotation of an event."""
    if len(event) <= 1:
        return event
    doubled = event + event
    best = event
    n = len(event)
    for shift in range(1, n):
        cand = doubled[shift : shift + n]
        if cand < best:
            best = cand
    return best


def line_from_loop(
    event: Tuple[int, ...],
    *,
    loop_prob_fn: Callable[[Tuple[int, ...]], object],
    alphabet_size: int,
) -> object:
    """Compute line probability by summing over a dummy loop extension site."""
    return sum(loop_prob_fn(event + (x,)) for x in range(alphabet_size))
