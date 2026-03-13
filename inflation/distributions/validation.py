from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
import sys
from typing import Iterable, Iterator, Sequence

import sympy as sp

# Ensure repo root is on sys.path so "import inflation" works when running directly.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from inflation.distributions.protocols import RingDistributionProtocol


@dataclass(frozen=True)
class ValidationFailure:
    validator: str
    family: str
    length: int
    event: tuple[int, ...] | None
    split_index: int | None
    lhs: sp.Expr
    rhs: sp.Expr
    difference: sp.Expr
    message: str


@dataclass(frozen=True)
class ValidationReport:
    validator: str
    passed: bool
    checks_run: int
    failures: tuple[ValidationFailure, ...]

    def summary_lines(self) -> list[str]:
        status = "PASS" if self.passed else "FAIL"
        lines = [f"{self.validator}: {status} ({self.checks_run} checks, {len(self.failures)} failures)"]
        for failure in self.failures:
            lines.append(
                f"  - {failure.family} length={failure.length}"
                + (f" event={failure.event}" if failure.event is not None else "")
                + (f" split={failure.split_index}" if failure.split_index is not None else "")
                + f": {failure.message}; lhs={failure.lhs}, rhs={failure.rhs}, diff={failure.difference}"
            )
        return lines


def _family_name(dist: RingDistributionProtocol) -> str:
    return type(dist).__name__


def _iter_events(alphabet_size: int, length: int) -> Iterator[tuple[int, ...]]:
    for event in product(range(alphabet_size), repeat=length):
        yield tuple(int(x) for x in event)


def _expr_equal(lhs: sp.Expr, rhs: sp.Expr) -> tuple[bool, sp.Expr]:
    diff = sp.simplify(lhs - rhs)
    return bool(diff == 0), diff


def _expr_is_negative(expr: sp.Expr) -> tuple[bool, sp.Expr]:
    simplified = sp.simplify(expr)
    return bool(simplified.is_negative is True), simplified


def _make_failure(
    *,
    validator: str,
    family: str,
    length: int,
    event: tuple[int, ...] | None,
    split_index: int | None,
    lhs: sp.Expr,
    rhs: sp.Expr,
    message: str,
) -> ValidationFailure:
    return ValidationFailure(
        validator=validator,
        family=family,
        length=int(length),
        event=event,
        split_index=split_index,
        lhs=sp.simplify(lhs),
        rhs=sp.simplify(rhs),
        difference=sp.simplify(lhs - rhs),
        message=message,
    )


def validate_normalization(dist: RingDistributionProtocol) -> ValidationReport:
    validator = "validate_normalization"
    family = _family_name(dist)
    alphabet = int(dist.nof_outcomes)
    failures: list[ValidationFailure] = []
    checks_run = 0

    loop_sum = sp.simplify(sum(dist.prob_event_loop((a,)) for a in range(alphabet)))
    checks_run += 1
    ok, _diff = _expr_equal(loop_sum, sp.Integer(1))
    if not ok:
        failures.append(
            _make_failure(
                validator=validator,
                family=family,
                length=1,
                event=None,
                split_index=None,
                lhs=loop_sum,
                rhs=sp.Integer(1),
                message="singleton loop probabilities do not normalize to 1",
            )
        )

    line_sum = sp.simplify(sum(dist.prob_event_line((a,)) for a in range(alphabet)))
    checks_run += 1
    ok, _diff = _expr_equal(line_sum, sp.Integer(1))
    if not ok:
        failures.append(
            _make_failure(
                validator=validator,
                family=family,
                length=1,
                event=None,
                split_index=None,
                lhs=line_sum,
                rhs=sp.Integer(1),
                message="singleton line probabilities do not normalize to 1",
            )
        )

    return ValidationReport(
        validator=validator,
        passed=not failures,
        checks_run=checks_run,
        failures=tuple(failures),
    )


def validate_consistency_upto(dist: RingDistributionProtocol, n: int) -> ValidationReport:
    validator = "validate_consistency_upto"
    family = _family_name(dist)
    alphabet = int(dist.nof_outcomes)
    max_n = int(n)
    if max_n < 1:
        raise ValueError("validate_consistency_upto requires n >= 1.")

    failures: list[ValidationFailure] = []
    checks_run = 0

    for length in range(1, max_n):
        for event in _iter_events(alphabet, length):
            lhs = sp.simplify(dist.prob_event_line(event))
            loop_extensions = [sp.simplify(dist.prob_event_loop(event + (a,))) for a in range(alphabet)]
            for outcome, extension_prob in enumerate(loop_extensions):
                checks_run += 1
                is_negative, simplified_prob = _expr_is_negative(extension_prob)
                if is_negative:
                    failures.append(
                        _make_failure(
                            validator=validator,
                            family=f"{family}:loop-extension-nonnegativity",
                            length=length + 1,
                            event=event + (outcome,),
                            split_index=None,
                            lhs=simplified_prob,
                            rhs=sp.Integer(0),
                            message="single-site loop extension has negative probability",
                        )
                    )
            rhs_loop = sp.simplify(sum(loop_extensions))
            checks_run += 1
            ok, _diff = _expr_equal(lhs, rhs_loop)
            if not ok:
                failures.append(
                    _make_failure(
                        validator=validator,
                        family=f"{family}:loop-extension",
                        length=length,
                        event=event,
                        split_index=None,
                        lhs=lhs,
                        rhs=rhs_loop,
                        message="line probability does not equal sum over single-site loop extensions",
                    )
                )

    for length in range(1, max_n - 1):
        for event in _iter_events(alphabet, length):
            lhs = sp.simplify(dist.prob_event_line(event))
            line_extensions = [sp.simplify(dist.prob_event_line(event + (a,))) for a in range(alphabet)]
            for outcome, extension_prob in enumerate(line_extensions):
                checks_run += 1
                is_negative, simplified_prob = _expr_is_negative(extension_prob)
                if is_negative:
                    failures.append(
                        _make_failure(
                            validator=validator,
                            family=f"{family}:line-extension-nonnegativity",
                            length=length + 1,
                            event=event + (outcome,),
                            split_index=None,
                            lhs=simplified_prob,
                            rhs=sp.Integer(0),
                            message="single-site line extension has negative probability",
                        )
                    )
            rhs_line = sp.simplify(sum(line_extensions))
            checks_run += 1
            ok, _diff = _expr_equal(lhs, rhs_line)
            if not ok:
                failures.append(
                    _make_failure(
                        validator=validator,
                        family=f"{family}:line-extension",
                        length=length,
                        event=event,
                        split_index=None,
                        lhs=lhs,
                        rhs=rhs_line,
                        message="line probability does not equal sum over single-site line extensions",
                    )
                )

    return ValidationReport(
        validator=validator,
        passed=not failures,
        checks_run=checks_run,
        failures=tuple(failures),
    )


def validate_factorization(dist: RingDistributionProtocol, n: int) -> ValidationReport:
    validator = "validate_factorization"
    family = _family_name(dist)
    alphabet = int(dist.nof_outcomes)
    max_n = int(n)

    failures: list[ValidationFailure] = []
    checks_run = 0

    length = max_n - 1
    if length >= 3:
        for event in _iter_events(alphabet, length):
            for split_index in range(1, length - 1):
                lhs = sp.Integer(0)
                for mid_outcome in range(alphabet):
                    extended = event[:split_index] + (mid_outcome,) + event[split_index + 1 :]
                    lhs += dist.prob_event_line(extended)
                lhs = sp.simplify(lhs)
                rhs = sp.simplify(
                    dist.prob_event_line(event[:split_index]) * dist.prob_event_line(event[split_index + 1 :])
                )
                checks_run += 1
                ok, _diff = _expr_equal(lhs, rhs)
                if not ok:
                    failures.append(
                        _make_failure(
                            validator=validator,
                            family=family,
                            length=length,
                            event=event,
                            split_index=split_index,
                            lhs=lhs,
                            rhs=rhs,
                            message="maximal line marginalization does not factorize across the chosen interior split",
                        )
                    )

    return ValidationReport(
        validator=validator,
        passed=not failures,
        checks_run=checks_run,
        failures=tuple(failures),
    )


def _print_report(report: ValidationReport) -> None:
    for line in report.summary_lines():
        print(line)


def _run_suite(dist: RingDistributionProtocol, *, consistency_n: int, factorization_n: int) -> None:
    family = _family_name(dist)
    print(f"\n=== {family} ===")
    _print_report(validate_normalization(dist))
    _print_report(validate_consistency_upto(dist, consistency_n))
    _print_report(validate_factorization(dist, factorization_n))


if __name__ == "__main__":
    from inflation.distributions.ejm import EJMDistribution

    _run_suite(EJMDistribution(), consistency_n=4, factorization_n=4)

    from inflation.distributions.rgb import RGBDistribution

    _run_suite(RGBDistribution(), consistency_n=4, factorization_n=4)

    from inflation.distributions.nsi_pr import NSIPRDistribution

    _run_suite(NSIPRDistribution(), consistency_n=6, factorization_n=6)

    from inflation.distributions.ghz import GHZDistribution

    _run_suite(GHZDistribution(), consistency_n=4, factorization_n=4)


__all__ = [
    "ValidationFailure",
    "ValidationReport",
    "validate_consistency_upto",
    "validate_normalization",
    "validate_factorization",
]
