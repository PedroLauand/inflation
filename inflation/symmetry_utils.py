"""
Auxiliary functions for symmetry discovery and interpretation.

@authors: Emanuel-Cristian Boghiu, Elie Wolfe and Alejandro Pozas-Kerstjens
"""

from __future__ import annotations

from typing import Callable, List, Optional, TYPE_CHECKING, Tuple, Union

import numpy as np
from sympy.combinatorics import Permutation, PermutationGroup
from tqdm import tqdm

from .utils import ndarray_bytes_key

if TYPE_CHECKING:
    from .InflationProblem import InflationProblem


def _as_permutation_matrix(
    perms: Union[np.ndarray, List[np.ndarray], List[List[int]], Tuple[Tuple[int, ...], ...]],
) -> np.ndarray:
    arr = np.asarray(perms, dtype=int)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    if arr.ndim != 2:
        raise ValueError("Permutations must be provided as a 2D array-like object.")
    return arr


def _sorted_group_elements(group: PermutationGroup) -> np.ndarray:
    elements = np.array(list(group.generate_schreier_sims(af=True)), dtype=int)
    if elements.ndim == 1:
        elements = elements[np.newaxis, :]
    return elements[np.lexsort(np.rot90(elements))]


def _check_subgroup_search_support() -> None:
    if not hasattr(PermutationGroup, "subgroup_search"):
        raise RuntimeError(
            "SymPy subgroup_search is required for scalable symmetry discovery. "
            "Please upgrade SymPy to a version with PermutationGroup.subgroup_search."
        )


def _build_bsgs_group_from_perms(perms: np.ndarray) -> PermutationGroup:
    perms = np.unique(perms, axis=0)
    group = PermutationGroup([Permutation(perm) for perm in perms])
    group.schreier_sims()
    return group


def discovery_symmetries_from_predicate(
    *,
    stabilizer_predicate: Callable[[np.ndarray], bool],
    scenario: Optional["InflationProblem"] = None,
    initial_generators: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    candidate_generators: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    verbose: bool = True,
    progress_desc: str = "Searching stabilizing subgroup",
    return_group: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, PermutationGroup]]:
    """
    Discover the subgroup that satisfies `stabilizer_predicate` using BSGS search.

    Returns sorted subgroup elements as permutations of lexorder. If
    `return_group=True`, also returns the BSGS subgroup object.
    """

    _check_subgroup_search_support()
    if scenario is None and initial_generators is None and candidate_generators is None:
        raise ValueError("Provide at least one of `scenario`, `initial_generators`, or `candidate_generators`.")

    if initial_generators is None:
        if scenario is None:
            raise ValueError("`initial_generators` is required when `scenario` is not provided.")
        identity = np.arange(scenario._nr_operators, dtype=int)
        initial_arr = identity[np.newaxis, :]
    else:
        initial_arr = _as_permutation_matrix(initial_generators)

    if candidate_generators is None:
        if scenario is not None:
            candidate_arr = np.asarray(scenario.all_possible_symmetries, dtype=int)
        else:
            candidate_arr = initial_arr
    else:
        candidate_arr = _as_permutation_matrix(candidate_generators)

    width = initial_arr.shape[1]
    if candidate_arr.shape[1] != width:
        raise ValueError(
            f"Candidate width {candidate_arr.shape[1]} does not match initial width {width}."
        )
    all_candidates = np.unique(np.vstack((initial_arr, candidate_arr)), axis=0)

    if scenario is not None:
        initial_group = scenario.bsgs_group_from_perms(initial_arr)
        candidate_group = scenario.bsgs_group_from_perms(all_candidates)
    else:
        initial_group = _build_bsgs_group_from_perms(initial_arr)
        candidate_group = _build_bsgs_group_from_perms(all_candidates)

    bar = tqdm(total=None, desc=progress_desc, disable=not verbose)

    def _sympy_predicate(sympy_perm: Permutation) -> bool:
        perm = np.asarray(sympy_perm.array_form, dtype=int)
        bar.update(1)
        return bool(stabilizer_predicate(perm))

    try:
        subgroup = candidate_group.subgroup_search(_sympy_predicate)
    finally:
        bar.close()
    subgroup.schreier_sims()
    subgroup_elements = _sorted_group_elements(subgroup)

    if verbose:
        initial_order = int(initial_group.order())
        candidate_order = int(candidate_group.order())
        stabilizer_order = int(subgroup.order())
        ratio = float(stabilizer_order) / float(max(initial_order, 1))
        print("Stabilizer subgroup summary:")
        print(f"  |G_initial|={initial_order}")
        print(f"  |G_candidate|={candidate_order}")
        print(f"  |G_stabilizer|={stabilizer_order}")
        print(f"  compression factor (|G_stabilizer|/|G_initial|) ~= {ratio:.6g}")

    if return_group:
        return subgroup_elements, subgroup
    return subgroup_elements


def discover_stabilizing_subgroup(
    *,
    stabilizer_predicate: Callable[[np.ndarray], bool],
    scenario: Optional["InflationProblem"] = None,
    initial_generators: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    candidate_generators: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    verbose: bool = True,
    progress_desc: str = "Searching stabilizing subgroup",
) -> Tuple[PermutationGroup, np.ndarray]:
    """
    Backward-compatible helper returning `(subgroup_group, subgroup_elements)`.
    """

    subgroup_elements, subgroup_group = discovery_symmetries_from_predicate(
        stabilizer_predicate=stabilizer_predicate,
        scenario=scenario,
        initial_generators=initial_generators,
        candidate_generators=candidate_generators,
        verbose=verbose,
        progress_desc=progress_desc,
        return_group=True,
    )
    return subgroup_group, subgroup_elements


def discover_distribution_symmetries(
    distribution: np.ndarray,
    scenario: "InflationProblem",
    *,
    initial_generators: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    candidate_generators: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    verbose: bool = True,
    return_group: bool = False,
    progress_desc: str = "Discovering distribution symmetries",
) -> Union[np.ndarray, Tuple[np.ndarray, PermutationGroup]]:
    """
    Discover symmetries of a distribution compatible with a scenario.

    This is a thin wrapper that builds a distribution-preservation predicate and
    delegates to `discovery_symmetries_from_predicate`.
    """

    parties = scenario.nr_parties
    if not isinstance(distribution, np.ndarray):
        raise TypeError("The distribution must be encoded in a numpy array.")
    if len(distribution.shape) != 2 * parties:
        raise ValueError("The distribution must be encoded as a 2*nr_parties-dimensional array.")
    if np.any(distribution.shape[:parties] != scenario.outcomes_per_party):
        raise ValueError("The number of outcomes of the distribution and scenario do not match.")
    if np.any(distribution.shape[parties:] != scenario.settings_per_party):
        raise ValueError("The number of settings of the distribution and scenario do not match.")
    if np.any(distribution < 0):
        raise ValueError("The distribution contains negative values.")
    if not np.allclose(distribution.sum(axis=tuple(range(parties))), 1):
        raise ValueError("The distribution is not normalized for each setting.")

    original_dag_events_order = {
        tuple(op): i for i, op in enumerate(scenario.original_dag_events)
    }
    original_dag_monomials_values = {}
    original_dag_monomials_lexboolvecs = []
    for ins in np.ndindex(*scenario.settings_per_party):
        for outs in np.ndindex(*scenario.outcomes_per_party):
            original_dag_lexboolvec = np.zeros(len(scenario.original_dag_events), dtype=bool)
            for p, (x, a) in enumerate(zip(ins, outs)):
                original_dag_lexboolvec[original_dag_events_order[(p + 1, x, a)]] = True
            original_dag_monomials_lexboolvecs.append(original_dag_lexboolvec)
            original_dag_monomials_values[ndarray_bytes_key(original_dag_lexboolvec)] = distribution[(*outs, *ins)]
    original_dag_monomials_lexboolvecs = np.asarray(original_dag_monomials_lexboolvecs, dtype=bool)
    original_values_1d = np.asarray(
        [original_dag_monomials_values[ndarray_bytes_key(mon)] for mon in original_dag_monomials_lexboolvecs],
        dtype=float,
    )

    def stabilizer_predicate(perm_lexorder: np.ndarray) -> bool:
        perm_orig = lexperm_to_origperm(perm_lexorder, scenario)
        lexboolvecs = original_dag_monomials_lexboolvecs[:, perm_orig]
        new_values_1d = np.asarray(
            [original_dag_monomials_values[ndarray_bytes_key(mon)] for mon in lexboolvecs],
            dtype=float,
        )
        return bool(np.allclose(new_values_1d, original_values_1d))

    return discovery_symmetries_from_predicate(
        stabilizer_predicate=stabilizer_predicate,
        scenario=scenario,
        initial_generators=initial_generators,
        candidate_generators=candidate_generators,
        verbose=verbose,
        progress_desc=progress_desc,
        return_group=return_group,
    )


def group_elements_from_generators(
    generators: Union[np.ndarray, List[np.ndarray]],
) -> np.ndarray:
    """
    Given a set of generators of a permutation group, return all group elements
    in lexicographic order.
    """

    group = _build_bsgs_group_from_perms(_as_permutation_matrix(generators))
    return _sorted_group_elements(group)


def interpret_lexorder_symmetry(perm: np.ndarray, scenario: "InflationProblem") -> dict:
    """Human-readable form of a lexorder permutation."""
    return dict(zip(scenario._lexrepr_to_names, scenario._lexrepr_to_names[perm]))


def interpret_original_symmetry(perm: np.ndarray, scenario: "InflationProblem") -> dict:
    """Human-readable form of a permutation of original observable events."""
    return dict(zip(scenario._original_event_names, scenario._original_event_names[perm]))


def lexperm_to_origperm(lexperm: np.ndarray, scenario: "InflationProblem") -> np.ndarray:
    """
    Convert a lexorder permutation into a permutation on observable DAG events.
    """
    return scenario._lexidx_to_origidx[lexperm][scenario._canonical_lexids]
