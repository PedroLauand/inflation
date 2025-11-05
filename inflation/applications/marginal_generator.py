# minimal_marginals.py
from typing import List, Iterable, Iterator


# =========================================================
# Integer partitions of n (nonincreasing parts)
# =========================================================
def integer_partitions(n: int) -> Iterator[List[int]]:
    """
    Yield all integer partitions of n as nonincreasing lists of positive integers.
    Example: n=4 -> [4], [3,1], [2,2], [2,1,1], [1,1,1,1]
    """
    def _partitions(rem: int, max_part: int, prefix: List[int]):
        if rem == 0:
            yield prefix[:]
            return
        for p in range(min(rem, max_part), 1 - 1, -1):
            prefix.append(p)
            yield from _partitions(rem - p, p, prefix)
            prefix.pop()

    yield from _partitions(n, n, [])


# =========================================================
# Set partitions of {0,1,...,n-1}
# =========================================================
def set_partitions_indices(n: int) -> List[List[List[int]]]:
    """
    Return all set partitions of the index set {0,...,n-1}.
    Each partition is a list of blocks; each block is a list of indices.
    Order of blocks is arbitrary, but deterministic from construction.
    """
    if n == 0:
        return [[[]]]  # one partition of the empty set: a single empty block
    parts = [[ [0] ]]  # start with element 0 in its own block
    for x in range(1, n):
        new_parts = []
        for part in parts:
            # put x into a new block
            new_parts.append(part + [[x]])
            # put x into each existing block
            for i in range(len(part)):
                new_part = [blk[:] for blk in part]
                new_part[i].append(x)
                new_parts.append(new_part)
        parts = new_parts
    return parts


def canonical_outcome_patterns(n: int, outcomes: int) -> List[List[int]]:
    """
    One canonical outcome vector per equivalence class under outcome relabeling (S_outcomes).
    Only equality patterns matter, i.e., set partitions of positions {0..n-1}.
    We map blocks to labels 0,1,2,... in order of the smallest index in each block
    (to make the representative deterministic). We keep only patterns with
    number_of_blocks <= outcomes.
    Returns a list of length-n integer lists (labels in [0..outcomes-1]).
    """
    patterns: List[List[int]] = []
    for part in set_partitions_indices(n):
        # sort blocks by their smallest index to get a deterministic labeling
        blocks = sorted((sorted(b) for b in part), key=lambda b: b[0] if b else -1)
        if len(blocks) > outcomes:
            continue
        vec = [0] * n
        for label, block in enumerate(blocks):
            for idx in block:
                vec[idx] = label
        patterns.append(vec)
    # sort lexicographically for determinism
    patterns.sort()
    return patterns


# =========================================================
# Build a representative permutation J (1-based one-line form)
# for a given cycle partition of n
# =========================================================
def representative_perm_for_partition(parts: List[int]) -> List[int]:
    """
    Given a partition of n (e.g., [3,1] for n=4), build a canonical permutation J
    in one-line (1-based) form with that cycle structure:
      - Use consecutive labels in order: cycles on [1..l1], then next on [...], etc.
      - A cycle of length L maps i_k -> i_{k+1}, last -> first; length 1 maps to itself.
    """
    J: List[int] = []
    n = sum(parts)
    J = list(range(1, n + 1))  # initialize to identity
    cur = 1
    for L in parts:
        if L <= 1:
            # fixed point: already identity
            cur += L
            continue
        cyc = list(range(cur, cur + L))
        for a, b in zip(cyc, cyc[1:]):
            J[a - 1] = b
        J[cyc[-1] - 1] = cyc[0]
        cur += L
    return J


# =========================================================
# Combine structure (conjugacy class rep) + outcome patterns
# into minimal marginal events compatible with your iterator
# =========================================================
def generate_minimal_marginal_events(n: int, outcomes: int) -> List[List[List[int]]]:
    """
    Main entry point.
    Given n and outcomes, return a list of minimal marginal events.
    Each marginal is a list of operator entries:
        [[1, i, J(i), 0, a_i] for i=1..n]
    where:
      - J is one representative permutation per conjugacy class of S_n
        (i.e., one per integer partition of n),
      - (a_1,...,a_n) is one canonical outcome pattern representative
        (one per set partition of {1..n}, capped to use <= outcomes labels).

    The output format is exactly what your marginal iterator expects:
      List[marginal], where each marginal is List[[1,i,j,0,a]].
    """
    if n <= 0:
        return []

    all_marginals: List[List[List[int]]] = []

    # Outcome patterns depend only on (n, outcomes) → compute once
    outcome_reps = canonical_outcome_patterns(n, outcomes)

    # For each conjugacy-class representative (one per partition of n)
    for parts in integer_partitions(n):
        J = representative_perm_for_partition(parts)  # 1-based one-line
        # Build a marginal for each canonical outcome pattern
        for pat in outcome_reps:
            marginal = [[1, i, J[i - 1], 0, pat[i - 1]] for i in range(1, n + 1)]
            all_marginals.append(marginal)

    return all_marginals


# =========================================================
# Example usage
# =========================================================

# Example: n=3, outcomes=4
n, outcomes = 3, 4
marginals = generate_minimal_marginal_events(n, outcomes)

print(f"Total minimal marginals for n={n}, outcomes={outcomes}: {len(marginals)}\n")
#for k, m in enumerate(marginals):
#    print(k, m)
print(marginals)