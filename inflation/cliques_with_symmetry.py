import numpy as np
from collections import deque
from typing import Tuple, Set
import numba as nb


@nb.njit(nb.boolean(nb.boolean[:], nb.boolean[:]), fastmath=True, cache=False)
def is_subset_numba(candidate_mask: np.ndarray, potential_super_mask: np.ndarray) -> bool:
    """
    Checks if `candidate_mask` is a subset of `potential_super_mask`
    using a clean, Pythonic `zip` that Numba compiles efficiently.

    This is equivalent to `set(candidate) <= set(super)`.
    """
    # Numba has a specialized, fast implementation for zipping NumPy arrays.
    # This avoids manual indexing and is highly readable.
    for b_candidate, b_super in zip(candidate_mask, potential_super_mask):
        # If an element is in the candidate but not in the potential superset...
        if b_candidate and not b_super:
            # ...then it's not a subset.
            return False
    # If the loop completes without returning, it must be a subset.
    return True


# The rest of your filtering function remains the same, as it just calls this one.
def filter_maximal_cliques_numpy(clique_masks: np.ndarray) -> np.ndarray:
    if clique_masks.shape[0] < 2:
        return clique_masks

    sizes = np.sum(clique_masks, axis=1)
    desc_sort_indices = np.argsort(-sizes)
    sorted_masks = clique_masks[desc_sort_indices]

    maximal_indices = []
    for i in range(sorted_masks.shape[0]):
        candidate_mask = sorted_masks[i]
        is_dominated = False

        for max_idx in maximal_indices:
            maximal_mask = sorted_masks[max_idx]
            if is_subset_numba(candidate_mask, maximal_mask):
                is_dominated = True
                break

        if not is_dominated:
            maximal_indices.append(i)

    return sorted_masks[maximal_indices]


def find_cliques_symmetric(
        adj_matrix: np.ndarray,
        automorphisms: np.ndarray
) -> np.ndarray:
    """
    Finds ALL and MAXIMAL cliques in a graph using a high-performance,
    boolean-mask-based search that is pruned by graph symmetry and
    ordered-candidate selection.

    Parameters
    ----------
    adj_matrix : np.ndarray
      The adjacency matrix of the undirected graph.
    automorphisms : np.ndarray
      The graph's automorphism group as a (k, n) NumPy array.

    Returns
    -------
    np.ndarray
      A boolean NumPy arrays whose rows are clique bitmasks.
    """
    num_vertices = adj_matrix.shape[0]
    if num_vertices == 0:
        return np.empty((0, 0), dtype=bool)

    nbrs_masks = adj_matrix.astype(bool)
    all_found_cliques = []
    all_found_clique_masks = []
    seen_canonical_subproblems: Set[Tuple[Tuple[int, ...], Tuple[int, ...]]] = set()
    queue = deque()

    # --- 1. Initialize search from one representative vertex per orbit ---
    visited_init = np.zeros(num_vertices, dtype=bool)
    for i in range(num_vertices):
        if not visited_init[i]:
            orbit_of_i = automorphisms[:, i]
            visited_init[orbit_of_i] = True

            base_mask = np.zeros(num_vertices, dtype=bool)
            base_mask[i] = True

            cnbrs_mask = nbrs_masks[i].copy()
            cnbrs_mask[:i + 1] = False

            queue.append((base_mask, cnbrs_mask))

    # --- 2. Main search loop using boolean masks ---
    while queue:
        base_mask, cnbrs_mask = queue.popleft()
        base_indices = np.flatnonzero(base_mask)
        cnbrs_indices = np.flatnonzero(cnbrs_mask)

        # --- A. CANONICAL PRUNING ---
        permuted_bases = np.sort(automorphisms[:, base_indices], axis=1)
        if cnbrs_indices.size == 0:
            combined = permuted_bases
            permuted_cnbrs = None
        else:
            permuted_cnbrs = np.sort(automorphisms[:, cnbrs_indices], axis=1)
            combined = np.hstack([permuted_bases, permuted_cnbrs])

        min_idx = np.lexsort(combined.T[::-1])[0]
        canonical_base_tuple = tuple(permuted_bases[min_idx])
        canonical_cnbrs_tuple = () if permuted_cnbrs is None else tuple(permuted_cnbrs[min_idx])
        canonical_rep = (canonical_base_tuple, canonical_cnbrs_tuple)

        if canonical_rep in seen_canonical_subproblems:
            continue
        seen_canonical_subproblems.add(canonical_rep)

        # --- B. GENERATE CLIQUE ORBIT ---
        unique_in_orbit, where_unique = np.unique(permuted_bases, axis=0, return_index=True)
        all_found_cliques.extend(unique_in_orbit)
        all_found_clique_masks.extend(base_mask[automorphisms[where_unique]])



        # --- C. EXPLORE CHILDREN (WITH ORDERED-CANDIDATE PRUNING) ---
        # This is the corrected and optimized loop.
        for u in cnbrs_indices:
            # Create a mask for the new vertex u
            u_mask = np.zeros(num_vertices, dtype=bool)
            u_mask[u] = True

            # Vectorized extension and intersection
            new_base_mask = base_mask | u_mask
            new_cnbrs_mask = cnbrs_mask & nbrs_masks[u]

            # **CRUCIAL OPTIMIZATION**: Only consider candidates
            # that appear AFTER u to prevent redundant paths.
            new_cnbrs_mask[:u + 1] = False

            queue.append((new_base_mask, new_cnbrs_mask))
    return np.array(all_found_clique_masks, dtype=bool)

if __name__ == '__main__':
    ### Example Usage
    import numpy as np
    from inflation.symmetry_utils import group_elements_from_generators

    petersen_adj = np.array([
        [0, 1, 0, 0, 1, 1, 0, 0, 0, 0],  # 0
        [1, 0, 1, 0, 0, 0, 1, 0, 0, 0],  # 1
        [0, 1, 0, 1, 0, 0, 0, 1, 0, 0],  # 2
        [0, 0, 1, 0, 1, 0, 0, 0, 1, 0],  # 3
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 1],  # 4
        [1, 0, 0, 0, 0, 0, 0, 1, 1, 0],  # 5
        [0, 1, 0, 0, 0, 0, 0, 0, 1, 1],  # 6
        [0, 0, 1, 0, 0, 1, 0, 0, 0, 1],  # 7
        [0, 0, 0, 1, 0, 1, 1, 0, 0, 0],  # 8
        [0, 0, 0, 0, 1, 0, 1, 1, 0, 0]  # 9
    ])

    # For a real application, you'd use a tool like 'nauty' or 'sage' to find the
    # automorphism group. For this example, we will manually create one automorphism
    # (a 5-fold rotation of the outer star and inner pentagon) plus the identity.
    # A small subgroup is sufficient to demonstrate the principle.
    identity = list(range(10))
    # Rotation: 0->1, 1->2, 2->3, 3->4, 4->0 (outer)
    #           5->6, 6->7, 7->8, 8->9, 9->5 (inner)
    rotation = [1, 2, 3, 4, 0, 6, 7, 8, 9, 5]
    # For a full run, you would provide all 120 automorphisms.
    # Here we just use two to show the mechanism works.
    petersen_autos = group_elements_from_generators(np.array([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # identity
        [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],  # rotation
        [4, 3, 2, 1, 0, 9, 8, 7, 6, 5]  # a reflection
    ]))

    print("--- Petersen Graph with NumPy-Optimized Symmetric Discovery ---")
    # All vertices in the Petersen graph are symmetric, so there is only one orbit.
    # The algorithm will start a search from vertex 0, find the edge {0,1},
    # then prune all other searches that would lead to finding other edges.
    all_cliques= find_cliques_symmetric(petersen_adj, petersen_autos)

    print([np.flatnonzero(bm).tolist() for bm in all_cliques])
    max_cliques_petersen = filter_maximal_cliques_numpy(all_cliques)
    print([np.flatnonzero(bm).tolist() for bm in max_cliques_petersen])