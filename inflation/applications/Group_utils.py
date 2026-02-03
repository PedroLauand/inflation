from __future__ import annotations

from typing import List, Tuple, Union
import numpy as np
from numba import njit, int64, uint8, types
from numba.typed import List as NumbaList
from sympy.combinatorics.permutations import Permutation
from sympy.combinatorics.perm_groups import PermutationGroup


@njit(types.boolean[:](uint8[:], int64), cache=True, fastmath=True)
def to_lex_representation(evt: np.ndarray, outcomes: int) -> np.ndarray:
    """Compact event -> one-hot lex vector (length n^2 * outcomes)."""
    n2 = evt.shape[0]
    lex = np.zeros(n2 * outcomes, dtype=np.bool_)
    base = np.uint16(outcomes)
    idx = evt + base * np.arange(n2, dtype=np.uint16)
    # Assumes 0 <= evt[s] < outcomes for all s.
    for s in range(n2):
        lex[idx[s]] = 1
    return lex


@njit(uint8[:](types.boolean[:], int64), cache=True, fastmath=True)
def from_lex_representation(lex_evt: np.ndarray, outcomes: int) -> np.ndarray:
    """One-hot lex vector -> compact event."""
    lex_evt = np.ascontiguousarray(lex_evt)
    n2 = lex_evt.shape[0] // outcomes
    evt = np.empty(n2, dtype=np.uint8)
    blocks = lex_evt.reshape(n2, outcomes)
    # Assumes length is divisible by outcomes and blocks are one-hot.
    for s in range(n2):
        idx = 0
        for j in range(outcomes):
            if blocks[s, j] != 0:
                idx = j
                break
        evt[s] = idx
    return evt


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


def prepare_group_chain(G: PermutationGroup, N: int) -> NumbaList:
    """
    Run Schreier-Sims once. Returns inverse-transversal matrices per level
    for use in the canonicalizer (no recomputation).
    """
    G.schreier_sims()
    level_invperms = NumbaList()
    if N <= np.iinfo(np.uint8).max:
        perm_dtype = np.uint8
    else:
        perm_dtype = np.uint16
    for orbits, trans in zip(G.basic_orbits, G.basic_transversals):
        invperm_matrix = np.empty((len(orbits), N), dtype=perm_dtype)
        for i, u in enumerate(orbits):
            perm_arr = np.array(trans[u].array_form, dtype=perm_dtype)
            invperm = np.empty_like(perm_arr)
            invperm[perm_arr] = np.arange(perm_arr.size, dtype=perm_arr.dtype)
            invperm_matrix[i] = invperm
        level_invperms.append(invperm_matrix)
    return level_invperms


@njit(cache=True, fastmath=True)
def lexmin_with_invperms(current: np.ndarray, invperms: np.ndarray) -> Tuple[np.ndarray, int]:
    """Return lex-min vector and its index using inverse-permutation matrix."""
    m, n = invperms.shape
    best_idx = 0
    for i in range(1, m):
        for j in range(n):
            a = current[invperms[i, j]]
            b = current[invperms[best_idx, j]]
            if a < b:
                best_idx = i
                break
            if a > b:
                break
    best_vec = np.empty(n, dtype=current.dtype)
    for j in range(n):
        best_vec[j] = current[invperms[best_idx, j]]
    return best_vec, best_idx


@njit(cache=True, fastmath=True)
def canonical_leximin_coset_chain_uint64(
    evt: np.ndarray,
    outcomes: int,
    level_invperms: NumbaList,
) -> np.uint64:
    """
    Canonical representative of 'evt' under G using the stabilizer chain.
    Returns a uint64 key encoded directly from the lex-min one-hot vector.
    """
    x = to_lex_representation(evt, outcomes)
    current_invperm = np.arange(len(x), dtype=level_invperms[0].dtype)
    best_vec = x
    levels = len(level_invperms)
    for k in range(levels):
        current = x[current_invperm]
        invperm_matrix = level_invperms[k]
        cand_vec, cand_idx = lexmin_with_invperms(current, invperm_matrix)
        current_invperm = current_invperm[invperm_matrix[cand_idx]]
        best_vec = cand_vec

    outcomes_u64 = np.uint64(outcomes)
    acc = np.uint64(0)
    base = np.uint64(1)
    n2 = best_vec.size // outcomes
    offset = 0
    for s in range(n2):
        idx = 0
        for j in range(outcomes):
            if best_vec[offset + j] != 0:
                idx = j
                break
        acc += np.uint64(idx) * base  # Base-`outcomes` accumulation from one-hot blocks.
        if s + 1 < n2:
            base *= outcomes_u64
        offset += outcomes
    return acc
