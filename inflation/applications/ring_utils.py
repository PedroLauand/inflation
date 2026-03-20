from __future__ import annotations

from itertools import chain

import numpy as np

from ..InflationProblem import InflationProblem
from ..sdp.fast_npa import commutation_matrix
from ..utils import ndarray_bytes_key

_RING_LEXORDER_CACHED_ATTRS = (
    "_lexorder_lookup",
    "_lexrepr_to_dicts",
    "_lexrepr_to_names",
    "_original_event_names",
    "_lexrepr_to_copy_index_free_names",
    "_lexrepr_to_all_names",
    "_lexrepr_to_symbols",
    "_lexorder_hashable_interpretation_decoder",
    "_party_relabelling_symmetries",
    "_setting_specific_outcome_relabelling_symmetries",
    "_party_specific_setting_relabelling_symmetries",
    "all_possible_symmetry_generators",
    "all_possible_symmetries",
    "inflation_symmetries_one_source",
)


def _clear_ring_cached_attrs(prob: InflationProblem) -> None:
    for attr in _RING_LEXORDER_CACHED_ATTRS:
        prob.__dict__.pop(attr, None)


def _rebuild_ring_measurements(prob: InflationProblem) -> None:
    prob.measurements = []
    for p in range(prob.nr_parties):
        outcomes = np.arange(prob.outcomes_per_party[p], dtype=prob._np_dtype)
        settings = np.arange(prob.settings_per_party[p], dtype=prob._np_dtype)
        inflation_indices = prob.inflation_indices_per_party[p]
        measurements_per_party = np.empty(
            (len(inflation_indices), len(settings), len(outcomes), prob._nr_properties),
            dtype=prob._np_dtype,
        )
        measurements_per_party[:, :, :, 0] = p + 1
        for idx, inf_idxs in enumerate(inflation_indices):
            measurements_per_party[idx, :, :, 1 : (prob.nr_sources + 1)] = inf_idxs
            for setting in settings.flat:
                measurements_per_party[idx, setting, :, -2] = setting
                for outcome in outcomes.flat:
                    measurements_per_party[idx, setting, outcome, -1] = outcome
        prob.measurements.append(measurements_per_party)

    prob.measurements_symbolic = [
        np.apply_along_axis(prob._1d_to_symbol, -1, measurements_per_party)
        for measurements_per_party in prob.measurements
    ]

    prob._ortho_groups_per_party = []
    for p, measurements_per_party in enumerate(prob.measurements):
        outcome_card = prob.outcomes_per_party[p]
        prob._ortho_groups_per_party.append(
            measurements_per_party.reshape((-1, outcome_card, prob._nr_properties))
        )
    prob._ortho_groups = list(chain.from_iterable(prob._ortho_groups_per_party))

    offset = 0
    prob._ortho_idxs_per_party = []
    prob._ortho_idxs = []
    for ortho_groups_of_party in prob._ortho_groups_per_party:
        ortho_idxs_of_party = []
        for ortho_group in ortho_groups_of_party:
            block = np.arange(len(ortho_group)) + offset
            ortho_idxs_of_party.append(block)
            offset += len(ortho_group)
        prob._ortho_idxs_per_party.append(np.array(ortho_idxs_of_party))
        prob._ortho_idxs.extend(ortho_idxs_of_party)
    prob._template_idxs = np.array([ortho_idx_group[0] for ortho_idx_group in prob._ortho_idxs], dtype=int)


def strip_ring_self_loops(prob: InflationProblem) -> InflationProblem:
    """
    Mutate a one-party/two-source ring InflationProblem so only off-diagonal
    copy-index operators remain.
    """
    if prob.nr_parties != 1 or prob.nr_sources != 2 or not prob.really_just_one_source:
        raise ValueError("strip_ring_self_loops only supports one-party/two-source ring scenarios")

    filtered_indices_per_party = []
    for inflation_indices in prob.inflation_indices_per_party:
        mask = inflation_indices[:, 0] != inflation_indices[:, 1]
        filtered = np.asarray(inflation_indices[mask], dtype=prob._np_dtype)
        if filtered.size == 0:
            raise ValueError("Off-diagonal ring semantics require at least one non-self-loop operator")
        filtered_indices_per_party.append(filtered)
    prob.inflation_indices_per_party = filtered_indices_per_party

    prob._all_unique_inflation_indices = np.unique(
        np.vstack(prob.inflation_indices_per_party),
        axis=0,
    ).astype(prob._np_dtype)
    prob._inflation_indices_hash = {
        ndarray_bytes_key(op, dtype=prob._np_dtype): idx
        for idx, op in enumerate(prob._all_unique_inflation_indices)
    }
    prob._inflation_indices_overlap = prob.one_source_overlap_matrix(
        np.asarray(prob._all_unique_inflation_indices, dtype=prob._np_dtype)
    )

    _clear_ring_cached_attrs(prob)
    _rebuild_ring_measurements(prob)

    prob._lexorder = np.vstack(prob._ortho_groups).astype(prob._np_dtype)
    prob.party_from_lexidx = prob._lexorder[:, 0]
    prob.party_from_templateidx = prob.party_from_lexidx[prob._template_idxs]
    prob._nr_operators = len(prob._lexorder)
    prob._lexorder_for_factorization = np.array(
        [
            prob._inflation_indices_hash[ndarray_bytes_key(op, dtype=prob._np_dtype)]
            for op in prob._lexorder[:, 1:-2]
        ],
        dtype=np.intc,
    )

    if prob._nonclassical_sources.any():
        prob._default_notcomm = commutation_matrix(
            prob._lexorder,
            prob.sources_to_check_for_party_pair_commutation,
            False,
        )
    else:
        prob._default_notcomm = np.zeros((prob._nr_operators, prob._nr_operators), dtype=bool)

    lexorder_to_original = prob.rectify_fake_setting(prob._lexorder[:, [0, -2, -1]])
    (
        prob.original_dag_events,
        prob._canonical_lexids,
        prob._lexidx_to_origidx,
    ) = np.unique(
        lexorder_to_original,
        return_index=True,
        return_inverse=True,
        axis=0,
    )

    prob.symmetries = prob.inflation_symmetries_one_source
    return prob


def ring_problem(
    inflation_level: int,
    nof_outcomes: int,
    *,
    classical_sources="all",
) -> InflationProblem:
    prob = InflationProblem(
        dag={"i1": ["A"], "i2": ["A"]},
        outcomes_per_party=(nof_outcomes,),
        settings_per_party=(1,),
        classical_sources=classical_sources,
        inflation_level_per_source=(inflation_level, inflation_level),
        order=["A"],
        really_just_one_source=True,
    )
    return strip_ring_self_loops(prob)
