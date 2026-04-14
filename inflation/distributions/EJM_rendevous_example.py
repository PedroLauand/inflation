"""
Direct EJM rendezvous example for an n-party loop.

This file is self-contained and includes a standalone copy of the core
``prob_event_loop`` logic used in ``inflation.distributions.ejm``.
"""

import numpy as np


SQRT2 = np.sqrt(2.0)

# Normalized singlet state written as a 2x2 matrix.
PSI = np.array([[0, 1], [-1, 0]], dtype=complex) / SQRT2

# Normalized EJM measurement effects.
EJM_EFFECTS = (
    np.array([[-1 - 1j, 0], [-2j, -1 + 1j]], dtype=complex) / (2 * SQRT2),
    np.array([[1 - 1j, 2j], [0, 1 + 1j]], dtype=complex) / (2 * SQRT2),
    np.array([[-1 + 1j, 2j], [0, -1 - 1j]], dtype=complex) / (2 * SQRT2),
    np.array([[1 + 1j, 0], [-2j, 1 - 1j]], dtype=complex) / (2 * SQRT2),
)

# These are the transfer matrices M_a = E_a @ PSI.
M = tuple(effect @ PSI for effect in EJM_EFFECTS)


def parse_outcomes(outcomes, max_outcome):
    """Copy of the outcome parsing used by the distribution helpers."""
    parsed = tuple(int(x) for x in outcomes)
    if not parsed:
        raise ValueError("Provide at least one outcome.")
    if any((x < 0 or x > max_outcome) for x in parsed):
        raise ValueError(f"Outcomes must be in {{0,1,...,{max_outcome}}}.")
    return parsed


def cyclic_canonical(event):
    """Copy of the cyclic canonicalization used by prob_event_loop."""
    if len(event) <= 1:
        return event

    doubled = event + event
    best = event
    n = len(event)
    for shift in range(1, n):
        candidate = doubled[shift : shift + n]
        if candidate < best:
            best = candidate
    return best


def prob_loop(outcomes, matrices=M):
    """
    Standalone copy of the core ``prob_event_loop`` idea from ``ejm.py``:

        event = cyclic_canonical(parse_outcomes(outcomes))
        p(event) = |Tr(M[a_1] ... M[a_n])|^2
    """
    event = cyclic_canonical(parse_outcomes(outcomes, max_outcome=len(matrices) - 1))

    product_mat = np.eye(matrices[0].shape[0], dtype=complex)
    for outcome in event:
        product_mat = product_mat @ matrices[outcome]

    amplitude = np.trace(product_mat)
    return float(np.real_if_close(np.abs(amplitude) ** 2))


n = 4  # Change only this line for triangle, 4-ring, 5-ring, ...

all_zero = (0,) * n
cyclic_pattern = tuple(i % 4 for i in range(n))
all_one = (1,) * n

print(f"n = {n}")
print(f"p{all_zero} = {prob_loop(all_zero):.10g}")
print(f"p{cyclic_pattern} = {prob_loop(cyclic_pattern):.10g}")
print(f"p{all_one} = {prob_loop(all_one):.10g}")

all_equal = sum(prob_loop((outcome,) * n) for outcome in range(len(M)))
print(f"P(all equal) = {all_equal:.10g}")
