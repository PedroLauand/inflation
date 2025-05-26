# import sys
# sys.path.append('/Users/pedrolauand/inflation')
from inflation import InflationProblem, InflationLP, InflationSDP
import numpy as np
"""
Crazy idea: Use dummy intermediate latents to encode causal symmetry (to indicate the different Hilbert spaces of a single source)
"""
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
def name_interpret_always_copy_indices(*args, **kwargs):
    return InflationProblem._interpretation_to_name(*args, include_copy_indices=True)

def ring_problem(n: int) -> InflationProblem:
    inf_prob = InflationProblem(
        dag={"i1": ["A"],
             "i2": ["A"], },
        outcomes_per_party=(4,),
        settings_per_party=(1,),
        classical_sources=None,
        inflation_level_per_source=(n,n),
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

    #Hacks to prevent knowability assumptions
    # inf_prob.is_network = False
    # inf_prob._is_knowable_q_non_networks = (lambda x: False)

    inf_prob._interpretation_to_name = name_interpret_always_copy_indices

    return inf_prob


prob = ring_problem(3)
prob.add_symmetries(prob._setting_specific_outcome_relabelling_symmetries)
ring_SDP = InflationSDP(prob, verbose=2, include_all_outcomes=False)


ring_SDP.generate_relaxation("physical2", max_monomial_length=2)
# ring_SDP.generate_relaxation("npa1")

print("Quantum inflation **nonfanout/commuting** factors:")
print(ring_SDP.physical_atoms)
# print("Quantum inflation **noncommuting** factors:")
# print(sorted(set(ring_4_SDP.atomic_factors).difference(ring_4_SDP.physical_atoms)))
# # # print(ring_4_SDP.momentmatrix)
# # #

# Inputting values
known_values = {}
known_values["P[A^{1,2}=0]"] = 1 / 4
known_values["P[A^{1,2}=0 A^{2,3}=0]"] = 1 / 8
known_values["P[A^{1,2}=0 A^{2,3}=1]"] = 1 / 24
known_values["P[A^{1,2}=0 A^{2,3}=0 A^{3,1}=0]"] = 1 / 8
known_values["P[A^{1,2}=0 A^{2,3}=0 A^{3,1}=1]"] = 1 / 64
known_values["P[A^{1,2}=0 A^{2,3}=1 A^{3,1}=2]"] = 1 / 48
print("Known Values:")
print(known_values)

ring_SDP.update_values(known_values)
ring_SDP.solve(solve_dual=False)
#
