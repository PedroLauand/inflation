from itertools import product
from typing import Dict, Iterable, Iterator, List, Tuple

# Each marginal entry is of the form [1, i, j, 0, k]
LexEntry = List[int]

def _detect_indexing(marginal: List[LexEntry], n: int) -> str:
    """
    Detect whether i,j in marginal are 0-based (0..n-1) or 1-based (1..n).
    Returns 'zero' or 'one'. Raises if ambiguous/inconsistent.
    """
    if not marginal:
        return 'zero'  # default
    is_zero = all(0 <= e[1] <= n-1 and 0 <= e[2] <= n-1 for e in marginal)
    is_one  = all(1 <= e[1] <= n   and 1 <= e[2] <= n   for e in marginal)
    if is_zero and not is_one:
        return 'zero'
    if is_one and not is_zero:
        return 'one'
    if is_zero and is_one:
        # n==1 edge case or degenerate; prefer zero-based
        return 'zero'
    raise ValueError("Cannot detect indexing: i,j not in 0..n-1 or 1..n consistently.")

def _slot(i: int, j: int, n: int) -> int:
    """Row-major lex slot for operator (i,j), with 0-based i,j."""
    return i * n + j

def iterate_global_events_containing_marginal(
    n: int,
    outcomes: int,
    marginal: List[LexEntry],
) -> Iterator[List[int]]:
    """
    Yield all global events consistent with the given marginal.

    Representation:
      - Operators are A^{i,j} with i,j in {0,..,n-1}.
      - A global event is a list 'evt' of length n^2 in row-major order:
          evt[_slot(i,j,n)] == chosen outcome k for operator (i,j).
      - The marginal is a list of [1,i,j,0,k] constraints. i,j may be 0-based or 1-based; this
        function auto-detects and normalizes to 0-based internally.

    Number of outputs:
      outcomes^(n^2 - t), where t is the number of distinct (i,j) fixed by the marginal.
    """
    if n <= 0 or outcomes <= 0:
        raise ValueError("n and outcomes must be positive integers.")

    # Normalize marginal indexing (i,j) to 0-based
    indexing = _detect_indexing(marginal, n)
    to_zero = (lambda x: x-1) if indexing == 'one' else (lambda x: x)

    # Parse & validate marginal; build fixed choices (i,j) -> k
    fixed: Dict[Tuple[int,int], int] = {}
    for entry in marginal:
        if len(entry) != 5 or entry[0] != 1 or entry[3] != 0:
            raise ValueError(f"Invalid marginal entry format: {entry} (expect [1,i,j,0,k])")
        _, i_raw, j_raw, _, k = entry
        i, j = to_zero(i_raw), to_zero(j_raw)
        if not (0 <= i < n and 0 <= j < n):
            raise ValueError(f"(i,j)=({i_raw},{j_raw}) out of range for n={n} (indexing: {indexing}-based).")
        if not (0 <= k < outcomes):
            raise ValueError(f"k={k} out of range 0..{outcomes-1}.")
        key = (i, j)
        if key in fixed and fixed[key] != k:
            raise ValueError(f"Conflicting outcomes for (i,j)={key}: {fixed[key]} vs {k}.")
        fixed[key] = k

    # All operators (i,j)
    all_ops = [(i, j) for i in range(n) for j in range(n)]
    unfixed_ops = [op for op in all_ops if op not in fixed]

    # Iterate all completions for unfixed operators
    for ks in product(range(outcomes), repeat=len(unfixed_ops)):
        # Start with fixed selections
        out = [-1] * (n * n)  # will fill with k values
        for (i, j), k in fixed.items():
            out[_slot(i, j, n)] = k
        # Fill unfixed with the current combination
        for (i_j, k_sel) in zip(unfixed_ops, ks):
            i, j = i_j
            out[_slot(i, j, n)] = k_sel
        # Safety assertion: all filled
        # (Optional) if you prefer not to assert in production, remove the following line.
        if any(v == -1 for v in out):
            raise RuntimeError("Internal error: incomplete assignment.")
        yield out

def to_lex_representation(evt: List[int], outcomes: int) -> List[int]:
    """
    Convert a global event into its lexicographic one-hot representation.

    Parameters
    ----------
    evt : list[int]
        Global event: length n^2 list. Entry evt[i] = outcome (0..outcomes-1) 
        chosen for operator at position i (row-major order of (i,j)).
    outcomes : int
        Number of possible outcomes for each operator.

    Returns
    -------
    list[int]
        Lexicographic one-hot representation: length n^2 * outcomes.
        Each block of 'outcomes' entries corresponds to an operator.
        The active outcome slot is 1, others 0.
    """
    n2 = len(evt)
    lex_rep_evt = [0] * (n2 * outcomes)
    for i, k in enumerate(evt):
        if not (0 <= k < outcomes):
            raise ValueError(f"Outcome {k} at position {i} out of range 0..{outcomes-1}")
        lex_rep_evt[k + outcomes * i] = 1
    return lex_rep_evt


def apply_perm_to_lex(lex_rep_evt: List[int], g: List[int]) -> List[int]:
    """
    Apply a permutation g to a one-hot lexicographic event vector.

    Parameters
    ----------
    lex_rep_evt : list[int]
        One-hot vector of length N = n^2 * outcomes.
    g : list[int]
        Permutation of indices 0..N-1 (or 1..N if 1-based).
        Semantics: the value at old index p moves to new index g[p].

    Returns
    -------
    list[int]
        The permuted one-hot vector g(lex_rep_evt).
    """
    N = len(lex_rep_evt)
    if len(g) != N:
        raise ValueError(f"Permutation length {len(g)} != event length {N}")

    # Normalize to 0-based if needed
    if max(g) == N:  # looks like 1..N
        g = [x - 1 for x in g]

    # Validate permutation
    if sorted(g) != list(range(N)):
        raise ValueError("g is not a valid permutation of 0..N-1")

    out = [0] * N
    # Generic (works even if vector isn't perfectly one-hot)
    for p in range(N):
        q = g[p]
        out[q] = lex_rep_evt[p]
    return out
from typing import List

def from_lex_representation(lex_rep_evt: List[int], outcomes: int) -> List[int]:
    """
    Invert to_lex_representation: from one-hot vector back to compact global event.

    Parameters
    ----------
    lex_rep_evt : list[int]
        One-hot representation, length N = n^2 * outcomes.
        Each block of 'outcomes' corresponds to one operator (i,j).
    outcomes : int
        Number of outcomes per operator.

    Returns
    -------
    list[int]
        Global event (length n^2).
        Entry evt[i] = chosen outcome (0..outcomes-1) for operator i.
    """
    N = len(lex_rep_evt)
    if N % outcomes != 0:
        raise ValueError("Length of lex_rep_evt must be divisible by outcomes")

    n2 = N // outcomes
    evt = [-1] * n2

    for i in range(n2):
        block = lex_rep_evt[i*outcomes : (i+1)*outcomes]
        if sum(block) != 1:
            raise ValueError(f"Block {i} is not one-hot: {block}")
        evt[i] = block.index(1)

    return evt
# -----------------------
# Example usage
# -----------------------

n, outcomes = 2, 2

# Example marginal with 1-based i,j:
marginal = [[1, 1, 2, 0, 0], #A^{1,2} = 0
            [1, 2, 1, 0, 1]] #A^{2,1} = 1]

# Expect outcomes^(n^2 - t) = 3^(4 - 2) = 9 global events
count = 0
G=[[0 ,1, 2, 3, 4, 5, 6, 7], [1, 0, 3, 2, 5, 4, 7, 6], [6, 7, 4, 5, 2, 3, 0, 1],[7, 6, 5, 4, 3, 2, 1, 0]]
for evt in iterate_global_events_containing_marginal(n, outcomes, marginal):
    # evt is length n^2 = 4: [k(1,1), k(1,2), k(2,1), k(2,2)] in row-major order (1-based shown here):
    #print(evt, to_lex_representation(evt,outcomes))
    lex_evt=to_lex_representation(evt,outcomes)
    perm_lex_evt=apply_perm_to_lex(lex_evt,G[2])
    print(lex_evt,perm_lex_evt)
    print(evt,from_lex_representation(perm_lex_evt,outcomes))
    count += 1
print("Total:", count)
