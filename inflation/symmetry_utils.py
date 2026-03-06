"""
This file contains auxiliary functions for discovering symmetries

@authors: Emanuel-Cristian Boghiu, Elie Wolfe and Alejandro Pozas-Kerstjens
"""

from __future__ import annotations

from typing import Callable, List, Optional, TYPE_CHECKING, Tuple, Union

import numpy as np

from sympy.combinatorics import Permutation, PermutationGroup
from tqdm import tqdm
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
    group_elements = np.array(list(group.generate_schreier_sims(af=True)), dtype=int)
    if group_elements.ndim == 1:
        group_elements = group_elements[np.newaxis, :]
    return group_elements[np.lexsort(np.rot90(group_elements))]


def _check_subgroup_search_support() -> None:
    if not hasattr(PermutationGroup, "subgroup_search"):
        raise RuntimeError(
            "SymPy subgroup_search is required for scalable symmetry discovery. "
            "Please upgrade SymPy to a version that provides PermutationGroup.subgroup_search."
        )


def _build_bsgs_group_from_perms(perms: np.ndarray) -> PermutationGroup:
    perms = np.unique(perms, axis=0)
    group = PermutationGroup([Permutation(perm) for perm in perms])
    group.schreier_sims()
    return group


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
    Discover the stabilizing subgroup of a candidate permutation group using BSGS search.

    Parameters
    ----------
    stabilizer_predicate : Callable[[numpy.ndarray], bool]
        Predicate over lexorder permutations. True iff permutation is valid.
    scenario : InflationProblem, optional
        Scenario providing lexorder size and BSGS builder helpers.
    initial_generators : numpy.ndarray, optional
        Generators for the baseline symmetry group (used for reporting and
        automatically included in the candidate group).
    candidate_generators : numpy.ndarray, optional
        Generators spanning the candidate superset to be searched.
    verbose : bool, optional
        Whether to show tqdm progress and group-order summary.
    progress_desc : str, optional
        Description for the subgroup-search progress bar.

    Returns
    -------
    Tuple[PermutationGroup, numpy.ndarray]
        The subgroup in BSGS form and its full list of elements.
    """
    _check_subgroup_search_support()
    if scenario is None and initial_generators is None and candidate_generators is None:
        raise ValueError("Provide at least one of `scenario`, `initial_generators`, or `candidate_generators`.")

    if initial_generators is None:
        if scenario is not None:
            identity = np.arange(scenario._nr_operators, dtype=int)
            initial_arr = identity[np.newaxis, :]
        else:
            raise ValueError("`initial_generators` is required when `scenario` is not provided.")
    else:
        initial_arr = _as_permutation_matrix(initial_generators)

    if candidate_generators is None:
        if scenario is not None:
            candidate_arr = np.asarray(scenario.all_possible_symmetry_generators, dtype=int)
        else:
            candidate_arr = initial_arr
    else:
        candidate_arr = _as_permutation_matrix(candidate_generators)

    width = initial_arr.shape[1]
    if candidate_arr.shape[1] != width:
        raise ValueError(
            f"Candidate generator width {candidate_arr.shape[1]} does not match initial width {width}."
        )

    all_candidate_generators = np.unique(np.vstack((initial_arr, candidate_arr)), axis=0)
    if scenario is not None:
        initial_group = scenario.bsgs_group_from_perms(initial_arr)
        candidate_group = scenario.bsgs_group_from_perms(all_candidate_generators)
    else:
        initial_group = _build_bsgs_group_from_perms(initial_arr)
        candidate_group = _build_bsgs_group_from_perms(all_candidate_generators)

    bar = tqdm(
        total=None,
        desc=progress_desc,
        disable=not verbose,
    )

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
    return subgroup, subgroup_elements


def discover_distribution_symmetries(
    distribution: Optional[np.ndarray],
    scenario: "InflationProblem",
    *,
    initial_generators: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    candidate_generators: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    stabilizer_predicate: Optional[Callable[[np.ndarray], bool]] = None,
    atol: float = 1e-9,
    rtol: float = 1e-8,
    verbose: bool = True,
    return_group: bool = False,
    progress_desc: str = "Discovering distribution symmetries",
) -> Union[np.ndarray, Tuple[np.ndarray, PermutationGroup]]:
    """
    Given a distribution, find the symmetries of the distribution that are
    compatible with the symmetries of the scenario. The symmetries are
    represented as permutations of the lexicographic order of the events in
    the scenario.

    Parameters
    ----------
    distribution : numpy.ndarray, optional
        Distribution to be analyzed. It must be encoded as an array of shape
        ``[o1, ..., oN, s1, ..., sN]``. This argument can be omitted when
        `stabilizer_predicate` is supplied.
    scenario : InflationProblem
        The scenario object.

    Returns
    -------
    numpy.ndarray
        The symmetries of the distribution that are compatible with the
        symmetries of the scenario. The symmetries are represented as
        permutations of the lexicographic order of the events in the
        scenario.
    """
    if stabilizer_predicate is None:
        # Sanity checks
        parties = scenario.nr_parties
        if not isinstance(distribution, np.ndarray):
            raise TypeError("The distribution must be encoded in a numpy array.")
        if len(distribution.shape) != 2 * parties:
            raise ValueError("The distribution must be encoded as an 2*nr_parties-dimensional array.")
        if np.any(distribution.shape[:parties] != scenario.outcomes_per_party):
            raise ValueError("The number of outcomes of the distribution and of the scenario do not match.")
        if np.any(distribution.shape[parties:] != scenario.settings_per_party):
            raise ValueError("The number of settings of the distribution and of the scenario do not match.")
        if not np.all(distribution >= 0):
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
                original_dag_monomials_values[original_dag_lexboolvec.tobytes()] = distribution[(*outs, *ins)]
        original_dag_monomials_lexboolvecs = np.asarray(original_dag_monomials_lexboolvecs, dtype=bool)
        original_values_1d = np.asarray(
            [original_dag_monomials_values[mon.tobytes()] for mon in original_dag_monomials_lexboolvecs],
            dtype=float,
        )

        def stabilizer_predicate(perm_lexorder: np.ndarray) -> bool:
            perm_orig = lexperm_to_origperm(perm_lexorder, scenario)
            lexboolvecs = original_dag_monomials_lexboolvecs[:, perm_orig]
            new_values_1d = np.asarray(
                [original_dag_monomials_values[mon.tobytes()] for mon in lexboolvecs],
                dtype=float,
            )
            return bool(np.allclose(new_values_1d, original_values_1d, atol=atol, rtol=rtol))
    elif distribution is None:
        pass

    subgroup_group, subgroup_elements = discover_stabilizing_subgroup(
        stabilizer_predicate=stabilizer_predicate,
        scenario=scenario,
        initial_generators=initial_generators,
        candidate_generators=candidate_generators,
        verbose=verbose,
        progress_desc=progress_desc,
    )
    if return_group:
        return subgroup_elements, subgroup_group
    return subgroup_elements

def group_elements_from_generators(generators: Union[np.ndarray,
                                                List[np.ndarray]]
                                    ) -> np.ndarray:
    """
    Given a set of generators of some permutation group, return the group
    elements in lexicographic order.

    Parameters
    ----------
    generators : Union[numpy.ndarray, List[numpy.ndarray]]
        The generators of the permutation group. Each generator is a permutation
        of the indices ``[0...n]``.

    Returns
    -------
    numpy.ndarray
        The elements of the symmetry group as permutatiosn of th lexicographic
        order.
    """
    G = _build_bsgs_group_from_perms(_as_permutation_matrix(generators))
    return _sorted_group_elements(G)

def interpret_lexorder_symmetry(perm: np.ndarray,
                                scenario: "InflationProblem") -> dict:
    """Gives a human-readable form of a permutation of the lexicographic order
    of the events in the scenario.

    Parameters
    ----------
    perm : numpy.ndarray
        The permutation of the lexicographic order of the events in the
        scenario.
    scenario : InflationProblem
        The scenario object.

    Returns
    -------
    dict
        A dictionary that maps the names of the events in lexicographic order
        to the names of the corresponding events under the symmetry.
    """
    return dict(zip(scenario._lexrepr_to_names,
                    scenario._lexrepr_to_names[perm]))

def interpret_original_symmetry(perm: np.ndarray,
                                scenario: "InflationProblem") -> dict:
    """Gives a human-readable form of a permutation of the observable events in
    the scenario.

    Parameters
    ----------
    perm : numpy.ndarray
        The permutation of the events in the scenario.
    scenario : InflationProblem
        The scenario object.

    Returns
    -------
    dict
        A dictionary that maps the names of the events in lexicographic order
        to the names of the corresponding events under the symmetry.
    """
    return dict(zip(scenario._original_event_names,
                    scenario._original_event_names[perm]))

def lexperm_to_origperm(lexperm: np.ndarray,
                        scenario: "InflationProblem") -> np.ndarray:
    """
    Given a permutation of the lexicographic order of the events in the
    scenario, return the permutation of the observable events in the scenario.

    Parameters
    ----------
    lexperm : numpy.ndarray
        The permutation of the lexicographic order of the events in the
        scenario.
    scenario : InflationProblem
        The scenario object.

    Returns
    -------
    numpy.ndarray
        The corresponding permutation of the observable events in the scenario.
    """
    return scenario._lexidx_to_origidx[lexperm][scenario._canonical_lexids]
