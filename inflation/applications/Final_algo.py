# final_selfcontained_pipeline.py
# -------------------------------------------------------------------
# Self-contained pipeline (no external imports from your project).
# It:
#   1) generates symmetry-reduced marginals (operator structure + outcome patterns),
#   2) evaluates each marginal's factorized EJM loop value,
#   3) enumerates global extensions and tallies canonical representatives under a group G.
#
# Output format: a list of items, each item is [e1, e2], where:
#   e1: { <marginal-as-tuple-of-tuples> : float_value }
#   e2: { <canonical-global-rep (tuple of length n^2)> : count }
#
# Example usage is at the bottom (n=2, outcomes=4).
# -------------------------------------------------------------------

from __future__ import annotations
from typing import List, Tuple, Dict, Iterable, Union
from functools import lru_cache
from math import ldexp
import numpy as np
import sys
sys.path.append('/Users/pedrolauand/My_Code/Inflation/inflation')
from inflation import InflationProblem
from collections import defaultdict, OrderedDict
from tqdm import tqdm
from scipy.sparse import coo_array


# =========================
# EJM distribution (4 outcomes)
# =========================
e0_un = np.array([[-1 - 1j, 0,     -2j,  -1 + 1j]], dtype=np.complex128).reshape(2, 2)
e1_un = np.array([[ 1 - 1j, 2j,     0,    1 + 1j]], dtype=np.complex128).reshape(2, 2)
e2_un = np.array([[-1 + 1j, 2j,     0,   -1 - 1j]], dtype=np.complex128).reshape(2, 2)
e3_un = np.array([[ 1 + 1j, 0,     -2j,   1 - 1j]], dtype=np.complex128).reshape(2, 2)
_ejm = np.stack([e0_un, e1_un, e2_un, e3_un])  # (4,2,2)

_psi = np.array([[0, 1, -1, 0]], dtype=np.complex128).reshape(2, 2)

@lru_cache(maxsize=None)
def _M() -> np.ndarray:
    """Precompute M[a] = ejm[a] @ psi, shape (4,2,2)."""
    return np.einsum('aij,jk->aik', _ejm, _psi, optimize=True)

def loop_prob_event(outcomes: Iterable[int]) -> float:
    """
    Probability P[a1,...,an] on an n-site RING (n=len(outcomes)), for outcomes in {0,1,2,3}.
    Uses unnormalized objects and applies the global factor 16^{-n}.
    """
    a = tuple(int(x) for x in outcomes)
    if not a:
        raise ValueError("Provide at least one outcome.")
    if any((x < 0 or x > 3) for x in a):
        raise ValueError("Outcomes must be in {0,1,2,3}.")
    M = _M()
    Pmat = np.eye(2, dtype=np.complex128)
    for x in a:
        Pmat = Pmat @ M[x]
    amp = np.trace(Pmat)
    prob = (amp.real * amp.real + amp.imag * amp.imag) * ldexp(1.0, -4 * len(a))  # 16^{-n}
    return float(prob)
#===================================
#Inflation Problem 
#===================================
def exists_shared_source_modified(inf_indices1: np.ndarray,
                            inf_indices2: np.ndarray) -> bool:
    common_sources = np.logical_and(inf_indices1, inf_indices2)
    if not np.any(common_sources):
        return False
    return not set(inf_indices1[common_sources]).isdisjoint(set(inf_indices2[common_sources]))
def overlap_matrix(all_inflation_indxs: np.ndarray) -> np.ndarray:
    n = len(all_inflation_indxs)
    adj_mat = np.eye(n, dtype=bool)
    for i in range(1, n):
        inf_indices_i = all_inflation_indxs[i]
        for j in range(i):
            inf_indices_j = all_inflation_indxs[j]
            if exists_shared_source_modified(inf_indices_i, inf_indices_j):
                adj_mat[i, j] = True
    adj_mat = np.logical_or(adj_mat, adj_mat.T)
    return adj_mat
def ring_problem(inflation_level: int, nof_outcomes: int = 2) -> InflationProblem:
    inf_prob = InflationProblem(
        dag={"i1": ["A"],
             "i2": ["A"], },
        outcomes_per_party=(nof_outcomes,),
        settings_per_party=(1,),
        classical_sources=None,
        inflation_level_per_source=(inflation_level,inflation_level),
        order=["A"])

    to_stabilize = np.flatnonzero(inf_prob._lexorder[:, 1] == inf_prob._lexorder[:, 2])


    #Fix factorization
    inf_prob._inflation_indices_overlap = overlap_matrix(inf_prob._all_unique_inflation_indices)

    # Fix symmetries
    new_symmetries = np.array([
        perm for perm in inf_prob.symmetries
        if np.array_equal(np.sort(perm[to_stabilize]), to_stabilize)
    ], dtype=int)
    inf_prob.symmetries = new_symmetries
    # inf_prob._interpretation_to_name = name_interpret_always_copy_indices

    return inf_prob

# =========================
# Integer partitions (conjugacy classes of S_n)
# =========================
def integer_partitions(n: int) -> List[List[int]]:
    """All integer partitions of n in nonincreasing order."""
    out: List[List[int]] = []
    def rec(rem: int, mx: int, acc: List[int]):
        if rem == 0:
            out.append(acc[:])
            return
        for p in range(min(rem, mx), 1 - 1, -1):
            acc.append(p)
            rec(rem - p, p, acc)
            acc.pop()
    rec(n, n, [])
    return out

def representative_perm_for_partition(parts: List[int]) -> List[int]:
    """
    Given partition parts of n (e.g., [3,1]), build a canonical 1-line permutation J of {1..n}
    with that cycle structure: consecutive labels per cycle.
    """
    n = sum(parts)
    J = list(range(1, n + 1))
    cur = 1
    for L in parts:
        if L <= 1:
            cur += L
            continue
        cyc = list(range(cur, cur + L))
        for a, b in zip(cyc, cyc[1:]):
            J[a - 1] = b
        J[cyc[-1] - 1] = cyc[0]
        cur += L
    return J

# =========================
# Outcome patterns up to relabeling (S_outcomes)
# =========================
def set_partitions_indices(n: int) -> List[List[List[int]]]:
    """All set partitions of {0,..,n-1} as list of blocks (lists)."""
    if n == 0:
        return [[[]]]
    parts = [[[0]]]
    for x in range(1, n):
        new_parts = []
        for part in parts:
            new_parts.append(part + [[x]])  # new block
            for i in range(len(part)):
                new_part = [blk[:] for blk in part]
                new_part[i].append(x)
                new_parts.append(new_part)
        parts = new_parts
    # normalize each partition blocks
    for p in parts:
        p.sort(key=lambda b: min(b))
        for b in p:
            b.sort()
    return parts

def canonical_outcome_patterns(n: int, outcomes: int) -> List[List[int]]:
    """
    One representative per equivalence class under outcome relabeling.
    Map blocks -> labels 0,1,2,... by block order (increasing min index).
    Only keep patterns with #blocks <= outcomes.
    """
    vecs: List[List[int]] = []
    for part in set_partitions_indices(n):
        if len(part) > outcomes:
            continue
        v = [0] * n
        for lbl, block in enumerate(part):
            for idx in block:
                v[idx] = lbl
        vecs.append(v)
    vecs.sort()
    return vecs

# =========================
# Generate symmetry-reduced marginals
# =========================
def generate_minimal_marginal_events(n: int, outcomes: int) -> List[List[List[int]]]:
    """
    Return a list of marginals in format [[1,i,J(i),0,a_i] for i=1..n],
    for each conjugacy-class representative J of S_n, and each canonical outcome pattern.
    """
    if n <= 0:
        return []
    all_marginals: List[List[List[int]]] = []
    outcome_reps = canonical_outcome_patterns(n, outcomes)
    for parts in integer_partitions(n):
        J = representative_perm_for_partition(parts)  # 1-line
        for pat in outcome_reps:
            marginal = [[1, i, J[i - 1], 0, pat[i - 1]] for i in range(1, n + 1)]
            all_marginals.append(marginal)
    return all_marginals

# =========================
# Global extensions iterator for a marginal
# =========================
def iterate_global_events_containing_marginal(
    n: int,
    outcomes: int,
    marginal: List[List[int]],
):
    """
    Yield all compact global events (length n^2, row-major over (i,j)) that extend the marginal.
    marginal format: [[1,i,j,0,a], ...] with i,j 1-based.
    """
    # fixed slots
    fixed: Dict[int, int] = {}
    for (one, i, j, zero, a) in marginal:
        si = (i - 1) * n + (j - 1)
        if si in fixed and fixed[si] != a:
            return  # inconsistent; yield nothing
        fixed[si] = a
    # build remaining indices
    Nslots = n * n
    remaining = [s for s in range(Nslots) if s not in fixed]
    # iterate assignments
    from itertools import product
    for combo in product(range(outcomes), repeat=len(remaining)):
        evt = [0] * Nslots
        # set fixed
        for s, a in fixed.items():
            evt[s] = a
        # set remaining
        for s, a in zip(remaining, combo):
            evt[s] = a
        yield evt

# =========================
# One-hot lex helpers (for group action on events)
# =========================
def to_lex_representation(evt: List[int], outcomes: int) -> List[int]:
    """Compact event -> one-hot lex vector (length n^2 * outcomes)."""
    n2 = len(evt)
    N = n2 * outcomes
    lex = [0] * N
    for s, k in enumerate(evt):
        if not (0 <= k < outcomes):
            raise ValueError("Outcome out of range")
        lex[k + outcomes * s] = 1
    return lex

def from_lex_representation(lex_evt: List[int], outcomes: int) -> List[int]:
    """One-hot lex vector -> compact event; validates one-hot per block."""
    N = len(lex_evt)
    if N % outcomes != 0:
        raise ValueError("Invalid length")
    n2 = N // outcomes
    evt = [0] * n2
    for s in range(n2):
        block = lex_evt[s * outcomes : (s + 1) * outcomes]
        if sum(block) != 1:
            raise ValueError(f"Block {s} not one-hot: {block}")
        evt[s] = block.index(1)
    return evt

def lex_less(a: List[int], b: List[int]) -> bool:
    """True iff a < b in lexicographic order."""
    for x, y in zip(a, b):
        if x < y:
            return True
        if x > y:
            return False
    return False

# =========================
# SymPy group utilities
# =========================
from sympy.combinatorics.permutations import Permutation
from sympy.combinatorics.perm_groups import PermutationGroup

def _normalize_perm_list(g: List[int], N: int) -> List[int]:
    """Ensure 0-based bijection of range(N) from list (accepts 1- or 0-based)."""
    if len(g) != N:
        raise ValueError(f"Permutation length {len(g)} != {N}")
    if max(g) == N:  # likely 1-based
        g = [x - 1 for x in g]
    if sorted(g) != list(range(N)):
        raise ValueError("Invalid permutation (not a bijection).")
    return g

def build_sympy_group(raw_G: Union[List[List[int]], np.ndarray], N: int) -> PermutationGroup:
    """Build group from raw list permutations (0- or 1-based)."""
    gens = [Permutation(_normalize_perm_list(gl, N)) for gl in raw_G] or [Permutation(list(range(N)))]
    return PermutationGroup(gens)

def prepare_group_chain(G: PermutationGroup, N: int) -> Dict[str, object]:
    """
    Run Schreier–Sims once. Returns a dict with base, orbits, transversals
    for use in the canonicalizer (no recomputation).
    """
    G.schreier_sims()
    return {
        "G": G,
        "base": list(G.base),
        "basic_orbits": list(G.basic_orbits),
        "basic_transversals": list(G.basic_transversals),
        "N": N,
    }

def apply_perm_to_lex(lex_evt: List[int], perm: Permutation) -> List[int]:
    """Apply SymPy permutation to one-hot vector: out[perm(p)] = lex_evt[p]."""
    N = len(lex_evt)
    out = [0] * N
    for p in range(N):
        out[perm(p)] = lex_evt[p]
    return out

def canonical_leximin_coset_chain(
    evt: List[int],
    outcomes: int,
    precomp: Dict[str, object],
) -> List[int]:
    """
    Canonical representative of 'evt' under G using the stabilizer chain:
    scans transversal reps at each level to minimize lex-image in one-hot space.
    """
    x = to_lex_representation(evt, outcomes)
    # G: PermutationGroup = precomp["G"]  # type: ignore
    base = precomp["base"]              # type: ignore
    basic_orbits = precomp["basic_orbits"]          # type: ignore
    basic_transversals = precomp["basic_transversals"]  # type: ignore

    # witness permutation and current best image
    g_star = Permutation(list(range(len(x))))  # identity
    best_vec = x

    # walk the chain
    levels = len(base)
    for k in range(levels):
        current = apply_perm_to_lex(x, g_star)
        cand_vec = None
        cand_U = None
        for u in basic_orbits[k]:
            U = basic_transversals[k][u]
            y = apply_perm_to_lex(current, U)
            if (cand_vec is None) or lex_less(y, cand_vec):
                cand_vec = y
                cand_U = U
        if cand_U is not None:
            g_star = cand_U * g_star
            best_vec = cand_vec  # type: ignore
    rep_evt = from_lex_representation(best_vec, outcomes)
    return rep_evt

# =========================
# Cycle extraction & factorized value for a marginal
# =========================
def _perm_from_marginal(marginal: List[List[int]]) -> List[int]:
    """Extract 1-line permutation J (1..n) from marginal [[1,i,j,0,a],...]."""
    n = len(marginal)
    J = [0] * n
    for (_, i, j, _, _) in marginal:
        J[i - 1] = j
    return J

def _outcomes_from_marginal(marginal: List[List[int]]) -> List[int]:
    """Extract outcome vector a_i from marginal [[1,i,j,0,a],...], in order i=1..n."""
    n = len(marginal)
    a = [0] * n
    for (_, i, _j, _, val) in marginal:
        a[i - 1] = val
    return a

def _cycles_from_J(J: List[int]) -> List[List[int]]:
    """Disjoint cycles of 1-line permutation J on {1..n}; each cycle as a list in cycle order (1-based)."""
    n = len(J)
    seen = [False] * (n + 1)
    cycles: List[List[int]] = []
    for start in range(1, n + 1):
        if seen[start]:
            continue
        cyc = []
        v = start
        while not seen[v]:
            seen[v] = True
            cyc.append(v)
            v = J[v - 1]
        cycles.append(cyc)
    return cycles

def factorized_marginal_value(marginal: List[List[int]]) -> float:
    """
    Multiply EJM loop scalars over the disjoint cycles of J with outcomes taken in cycle order.
    """
    J = _perm_from_marginal(marginal)
    a = _outcomes_from_marginal(marginal)
    val = 1.0
    for cyc in _cycles_from_J(J):
        cyc_out = [a[i - 1] for i in cyc]
        val *= loop_prob_event(cyc_out)
    return val

# =========================
# Canonicalize & count representatives of global extensions
# =========================
def representatives_of_global_extensions(
    n: int,
    outcomes: int,
    marginal: List[List[int]],
    *,
    precomp: Dict[str, object],
) -> List[Tuple[int, ...]]:
    """
    Iterate global extensions of 'marginal', canonicalize each under precomp['G'].
    """
    return [canonical_leximin_coset_chain(evt, outcomes, precomp=precomp)
            for evt in iterate_global_events_containing_marginal(n, outcomes, marginal)]


# =========================
# Top-level pipeline
# =========================
def run_pipeline(
    prob: InflationProblem
) -> Tuple[List[Dict], coo_array]:
    """
    Build:
      [
        [ { marginal_tuple : value_float }, { canonical_global_rep_tuple : count_int ...} ],
        ...
      ]
    over all symmetry-reduced marginals for (n, outcomes).
    """
    n = prob.inflation_level_per_source[0]
    outcomes = prob.outcomes_per_party[0]
    raw_G = prob.symmetries

    # group acts on one-hot coordinates of size N = n^2 * outcomes
    N = (n * n) * outcomes
    G = build_sympy_group(raw_G, N)
    precomp = prepare_group_chain(G, N)

    list_of_all_LP_variables = ["1"]
    marginals = generate_minimal_marginal_events(n, outcomes)

    list_of_marginal_dictionaries = []
    list_of_global_expansions = []
    sparse_matrix_rows = []
    sparse_matrix_cols = []
    sparse_matrix_data = []

    # result = []
    # LOOP 0: Compute the marginal probabilities
    known_values_dict = OrderedDict()
    for marginal in tqdm(marginals,  desc="Computing marginal values..."):
        # --- your existing body per marginal ---
        val = factorized_marginal_value(marginal)  ## This computes the numeric probabilities
        mkey = tuple(prob._lexrepr_to_names[prob.mon_to_lexrepr(marginal)])
        # list_of_marginal_dictionaries.append(marginal ## We don't need this anymore
        # mkey = tuple(tuple(x) for x in marginal)
        list_of_all_LP_variables.append("P_global("+",".join(mkey)+")")
        known_values_dict[mkey] = val
        # e1 = {mkey: val}
        # list_of_marginal_dictionaries.append(e1)

    # LOOP 1: Create the dictionaries of global events and their counts
    for marginal in tqdm(marginals,  desc="Finding global extensions..."):
        list_of_global_expansions.append(
            representatives_of_global_extensions(n, outcomes, marginal, precomp=precomp))
    list_of_global_expansions = np.array(list_of_global_expansions)

    # LOOP 2: Convert canonical global events and their counts to sparse arrays
    global_event_to_idx_dict = defaultdict(int)
    nof_marginals = len(marginals)
    idx = 1 + nof_marginals
    for row_num, global_expansion in enumerate(list_of_global_expansions):
        for event_tuple in map(tuple, global_expansion):
            event_idx = global_event_to_idx_dict[event_tuple]
            if event_idx == 0:
                list_of_all_LP_variables.append("P_global("+",".join(map(str,event_tuple))+")")
                event_idx = idx
                global_event_to_idx_dict[event_tuple] = event_idx
                idx += 1
            sparse_matrix_rows.append(row_num)
            sparse_matrix_cols.append(event_idx)
            sparse_matrix_data.append(1)
    sparse_matrix_rows = np.hstack((np.arange(nof_marginals, dtype=int),
                                   np.array(sparse_matrix_rows, dtype=int)))
    sparse_matrix_cols = np.hstack((np.arange(1,nof_marginals+1, dtype=int),
                                   np.array(sparse_matrix_cols, dtype=int)))
    sparse_matrix_data = np.hstack((-np.ones(nof_marginals, dtype=int),
                                   np.array(sparse_matrix_data, dtype=float)))
    inflation_matrix = coo_array((sparse_matrix_data, (sparse_matrix_rows, sparse_matrix_cols)),
                          shape=(nof_marginals, idx))
    inflation_matrix.sum_duplicates()

    return known_values_dict, inflation_matrix, list_of_all_LP_variables

"""# ---- inside run_pipeline, after you compute `marginals` ----
marginals = generate_minimal_marginal_events(n, outcomes)
total = len(marginals)
bar_width = 30

result = []
for idx, marginal in enumerate(marginals, start=1):
    # progress bar
    filled = int(bar_width * idx / total)
    bar = "#" * filled + "-" * (bar_width - filled)
    pct = (idx * 100) // total
    print(f"\r[{bar}] {pct:3d}%  {idx}/{total} marginals", end="", flush=True)

    # --- your existing body per marginal ---
    val = factorized_marginal_value(marginal)
    mkey = tuple(tuple(x) for x in marginal)
    e1 = {mkey: val}
    e2 = representatives_of_global_extensions(n, outcomes, marginal, precomp=precomp)
    result.append([e1, e2])

print()  # newline after finishing the bar
return result"""

# =========================
# Example of usage
# =========================
if __name__ == "__main__":
    import itertools
    from inflation.lp.lp_utils import solveLP_sparse
    # Example: n=2, outcomes=4
    n, outcomes = 4, 2
    # One small example group on N = n^2 * outcomes = 4 * 4 = 16 coordinates:
    #   - identity
    #   - swap within each outcome block of the four operator slots (toy example)
    prob = ring_problem(n, outcomes)
    prob.add_symmetries(prob._setting_specific_outcome_relabelling_symmetries)
    
    print("done with prob")

    
    knowns_dict, inflation_matrix, list_of_LP_variables = run_pipeline(prob)


    def convert_known_dict_to_sparse(known_values_dict: OrderedDict, nof_variables_total: int):
        data = np.array(list(known_values_dict.values()), dtype=float)
        nof_marginals = len(data)
        row = np.zeros(nof_marginals, dtype=int)
        col = np.arange(1, nof_marginals+1, dtype=int)
        # data = list(itertools.chain.from_iterable(marginals_dict.keys() for marginals_dict in marginals_dicts))
        return coo_array((data, (row, col)), shape=(1, nof_variables_total))

    nof_known, nof_all_LP_vars = inflation_matrix.shape
    known_vars_coo_vec = convert_known_dict_to_sparse(knowns_dict, nof_all_LP_vars)

    for k, v in knowns_dict.items():
        print(f"{k}: {v}")
    # # Print a small summary
    # for idx, (e1, e2) in enumerate(out):
    #     print(f"\nItem {idx}:")
    #     # marginal & value
    #     (marginal_key, val) = next(iter(e1.items()))
    #     print("  marginal:", list(list(t) for t in marginal_key))
    #     print("  value:   ", val)
    #     # reps & counts
    #     print("  reps (compact evt) -> count:")
    #     for rep, cnt in e2.items():
    #         print("   ", list(rep), "->", cnt)

    solution = solveLP_sparse(objective=coo_array(([], ([], [])), shape=(1, nof_all_LP_vars)),
                              known_vars=known_vars_coo_vec,
                              equalities=inflation_matrix,
                              default_non_negative=True,
                              variables=list_of_LP_variables,
                              verbose=True)

    print(solution["status"])
