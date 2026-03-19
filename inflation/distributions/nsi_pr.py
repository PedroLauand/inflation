from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List

import sympy as sp

E1 = sp.Integer(0)
E2 = sp.sqrt(2) - 1
E2_LOOP = sp.Integer(1)
E3 = 3 - 2 * sp.sqrt(2)

@lru_cache(None)
def E_line(n: int) -> sp.Expr:
    if n < 1:
        raise ValueError("E_line(n) requires n >= 1.")
    if n == 1:
        return E1
    if n == 2:
        return E2
    if n == 3:
        return E3
    else:
        return -E_line(n-1) + E_line(n-2) + (1 - E2) * E_line(n-3)


@lru_cache(None)
def E_loop(n: int) -> sp.Expr:
    if n < 1:
        raise ValueError("E_loop(n) requires n >= 1.")
    if n == 1:
        return sp.Integer(0)
    if n == 2:
        return E2_LOOP
    if n == 3:
    #     # return sp.simplify(sp.sqrt(2) - 2 * E_line(2) - E_line(1))
        return sp.Integer(0)
    m = n - 1
    term0 = sp.sqrt(2) ** (m - 1)
    summand = sp.Integer(0)
    for k in range(1, m - 1):
        summand += E_line(k) * sp.Integer(k) * (sp.sqrt(2) ** (m - k - 2))
    expr = term0 - summand - sp.Integer(m) * E_line(m) - sp.Integer(m - 1) * E_line(m - 1)
    return sp.simplify(expr)


def _runs_on_cycle(n: int, subset_bits: int) -> List[int]:
    if subset_bits == 0:
        return []
    if subset_bits == (1 << n) - 1:
        return [n]

    bits = [(subset_bits >> i) & 1 for i in range(n)]
    runs: List[int] = []
    i = 0
    while i < n:
        if bits[i] == 0:
            i += 1
            continue
        j = i
        while j < n and bits[j] == 1:
            j += 1
        runs.append(j - i)
        i = j
    if bits[0] == 1 and bits[-1] == 1 and len(runs) >= 2:
        runs[0] += runs[-1]
        runs.pop()
    return runs


def _runs_on_line(n: int, subset_bits: int) -> List[int]:
    if subset_bits == 0:
        return []
    bits = [(subset_bits >> i) & 1 for i in range(n)]
    runs: List[int] = []
    i = 0
    while i < n:
        if bits[i] == 0:
            i += 1
            continue
        j = i
        while j < n and bits[j] == 1:
            j += 1
        runs.append(j - i)
        i = j
    return runs


def _validate_x_list(x_iter: Iterable[int]) -> List[sp.Integer]:
    parsed = list(x_iter)
    if len(parsed) == 0:
        raise ValueError("Need at least one outcome.")
    out: List[sp.Integer] = []
    for value in parsed:
        vv = sp.Integer(value)
        if vv not in (sp.Integer(-1), sp.Integer(1)):
            raise ValueError("Each xi must be +1 or -1.")
        out.append(vv)
    return out


def _validate_bits(bits_iter: Iterable[int]) -> List[sp.Integer]:
    parsed = list(bits_iter)
    if len(parsed) == 0:
        raise ValueError("Need at least one outcome.")
    out: List[sp.Integer] = []
    for value in parsed:
        vv = int(value)
        if vv not in (0, 1):
            raise ValueError("Each outcome must be 0 or 1.")
        out.append(sp.Integer(vv))
    return out


def _subset_products(x: List[sp.Integer]) -> List[sp.Integer]:
    n = len(x)
    prods: List[sp.Integer] = [sp.Integer(1)] * (1 << n)
    for subset in range(1, 1 << n):
        lsb = subset & -subset
        idx = lsb.bit_length() - 1
        prods[subset] = prods[subset ^ lsb] * x[idx]
    return prods


def P_loop(x_iter: Iterable[int]) -> sp.Expr:
    x = _validate_x_list(x_iter)
    n = len(x)
    full_mask = (1 << n) - 1
    prod_x = _subset_products(x)

    total = sp.Integer(0)
    for subset in range(0, 1 << n):
        if subset == 0:
            corr = sp.Integer(1)
        elif subset == full_mask:
            corr = E_loop(n)
        else:
            corr = sp.Integer(1)
            for run_len in _runs_on_cycle(n, subset):
                corr *= E_line(run_len)
        total += prod_x[subset] * corr
    return sp.simplify(total / (sp.Integer(2) ** n))


def P_line(x_iter: Iterable[int]) -> sp.Expr:
    x = _validate_x_list(x_iter)
    n = len(x)
    prod_x = _subset_products(x)

    total = sp.Integer(0)
    for subset in range(0, 1 << n):
        if subset == 0:
            corr = sp.Integer(1)
        else:
            corr = sp.Integer(1)
            for run_len in _runs_on_line(n, subset):
                corr *= E_line(run_len)
        total += prod_x[subset] * corr
    return sp.simplify(total / (sp.Integer(2) ** n))


class NSIPRDistribution:
    """Exact NSI-PR distribution in binary outcomes."""

    @property
    def nof_outcomes(self) -> int:
        return 2

    def prob_event_loop(self, outcomes: Iterable[int]) -> sp.Expr:
        bits = _validate_bits(outcomes)
        x_pm = [sp.Integer(1) if bit == 0 else sp.Integer(-1) for bit in bits]
        return P_loop(x_pm)

    def prob_event_line(self, outcomes: Iterable[int]) -> sp.Expr:
        bits = _validate_bits(outcomes)
        x_pm = [sp.Integer(1) if bit == 0 else sp.Integer(-1) for bit in bits]
        return P_line(x_pm)


__all__ = ["NSIPRDistribution", "P_loop", "P_line", "E_line", "E_loop"]

if __name__ == "__main__":
    for n in range(1, 5):
        print(f"n={n}, E_loop(n)={E_loop(n)}={float(E_loop(n))}")
