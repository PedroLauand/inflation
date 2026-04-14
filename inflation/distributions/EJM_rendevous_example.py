"""
Direct EJM rendezvous example for an n-party loop.

Set ``n = 3`` for the triangle, ``n = 4`` for the 4-ring, and so on.
"""

from itertools import product

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

# These are the matrices that appear in the loop formula.
M = tuple(effect @ PSI for effect in EJM_EFFECTS)


def prob_loop(event, matrices=M):
    """
    Probability of one event in the loop:

        p(a_1, ..., a_n) = |Tr(M[a_1] ... M[a_n])|^2
    """
    event = tuple(event)
    if not event:
        raise ValueError("event must contain at least one outcome.")

    nof_outcomes = len(matrices)
    if any(outcome < 0 or outcome >= nof_outcomes for outcome in event):
        raise ValueError(f"Each outcome must be between 0 and {nof_outcomes - 1}.")

    product_mat = np.eye(matrices[0].shape[0], dtype=complex)
    for outcome in event:
        product_mat = product_mat @ matrices[outcome]

    amplitude = np.trace(product_mat)
    return float(np.real_if_close(np.abs(amplitude) ** 2))


n = 4  # Change to n = 3 for the triangle.

event_1 = (0, 1, 2, 3)
event_2 = (3, 0, 1, 2)
event_3 = (2, 3, 0, 1)
event_4 = (1, 2, 3, 0)

print(f"n = {n}")
print(f"p{event_1} = {prob_loop(event_1):.10g}")
print(f"p{event_2} = {prob_loop(event_2):.10g}")
print(f"p{event_3} = {prob_loop(event_3):.10g}")
print(f"p{event_4} = {prob_loop(event_4):.10g}")


all_disagree = prob_loop(event_1) + prob_loop(event_2) + prob_loop(event_3) + prob_loop(event_4)
print(f"P(all disagree) = {all_disagree:.10g}")

