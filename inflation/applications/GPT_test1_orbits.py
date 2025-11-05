from itertools import product
from typing import Dict, List, Tuple, Iterator

def iterate_full_assignments(
    n: int,
    outcomes: int,
    lexi_order: List[List[int]],
    fixed_marginal: List[Tuple[int, int, int, int]]
) -> Iterator[List[int]]:
    """
    Iterate over all full assignments consistent with a fixed marginal.

    Parameters
    ----------
    n : int
        Inflation level (number of i,j values: n^2 operators).
    outcomes : int
        Number of outcomes for each operator.
    lexi_order : list[list[int]]
        The lexicographic ordering of operator-outcome slots.
        Each entry is of the form [1, i, j, 0, k].
        Its position in lexi_order is the slot index.
    fixed_marginal : list[(int,int,int,int)]
        List of fixed operator values as tuples (1, i, j, 0, k).
        Example: [(1, 0, 1, 0, 2), (1, 1, 0, 0, 0)]
        meaning A^{0,1}=2 and A^{1,0}=0.

    Yields
    ------
    list[int]
        A full assignment 'a' of length len(lexi_order), with
        a[slot] giving the value for that slot. 
        Values are 0/1 indicators: 1 if that slot is active in
        the assignment, 0 otherwise.
    """
    N = len(lexi_order)

    # Map fixed constraints to slot indices
    fixed_slots = {}
    for constraint in fixed_marginal:
        if constraint not in lexi_order:
            raise ValueError(f"Constraint {constraint} not found in lexi_order")
        slot_index = lexi_order.index(constraint)
        if slot_index in fixed_slots:
            raise ValueError("Conflicting constraints on slot {slot_index}")
        fixed_slots[slot_index] = 1  # "active" for this outcome

    # Remaining slots = all other positions
    remaining_slots = [s for s in range(N) if s not in fixed_slots]

    # For each completion of the remaining slots, build a full assignment
    for combo in product([0,1], repeat=len(remaining_slots)):
        assignment = [0] * N
        # Set fixed part
        for slot, val in fixed_slots.items():
            assignment[slot] = val
        # Fill in remaining part
        for slot, val in zip(remaining_slots, combo):
            assignment[slot] = val
        yield assignment


# -----------------------
# Example usage
# -----------------------

n, outcomes = 2, 3
lexi_order = []
for i in range(n):
    for j in range(n):
        for k in range(outcomes):
            lexi_order.append([1, i+1, j+1, 0, k])
print(lexi_order)
# Fix A^{0,1}=0
fixed = [[1, 1, 2, 0, 2],[1, 2, 1, 0, 1]]

for a in iterate_full_assignments(n, outcomes, lexi_order, fixed):

    print(a)
