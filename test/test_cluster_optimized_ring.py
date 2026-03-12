import uuid
import unittest
from unittest import mock

import numpy as np
import sympy as sp
from scipy.sparse import coo_array, csr_array

from inflation.applications.Final_algo_numba import (
    CACHE_FORMAT_VERSION,
    PrepLP,
    _offdiag_slot_index,
)
from inflation.applications.Group_utils import canonical_leximin_coset_chain_uint64
from inflation.lp.lp_utils import solveLP_sparse


class _UniformBinaryDistribution:
    @property
    def nof_outcomes(self):
        return 2

    def prob_event_loop(self, outcomes):
        return sp.Rational(1, 2) ** len(tuple(outcomes))

    def prob_event_line(self, outcomes):
        return sp.Rational(1, 2) ** len(tuple(outcomes))


def _reference_global_keys_and_matrix(prep: PrepLP) -> tuple[np.ndarray, np.ndarray]:
    nof_marginals = prep.nof_marginals
    nof_slots = prep.nof_off_diagonal_slots
    global_event_map: dict[int, int] = {}
    global_keys: list[int] = []
    dense = np.zeros((nof_marginals, 1 + nof_marginals), dtype=np.float64)

    for row_num, marginal in enumerate(prep.marginals):
        fixed: dict[int, int] = {}
        for (_one, i, j, _zero, outcome) in marginal:
            slot = _offdiag_slot_index(int(i), int(j), prep.n)
            fixed[slot] = int(outcome)

        evt = np.zeros(nof_slots, dtype=np.uint8)
        if fixed:
            fixed_idx = np.fromiter(fixed.keys(), dtype=np.int64)
            evt[fixed_idx] = np.fromiter(fixed.values(), dtype=np.uint8)
        else:
            fixed_idx = np.empty(0, dtype=np.int64)

        mask = np.ones(nof_slots, dtype=bool)
        mask[fixed_idx] = False
        remaining = np.nonzero(mask)[0]
        total = pow(prep.outcomes, int(remaining.size))

        dense[row_num, 1 + row_num] = -1.0
        for pos in range(total):
            tmp = pos
            for rem_pos in range(remaining.size - 1, -1, -1):
                idx = remaining[rem_pos]
                evt[idx] = tmp % prep.outcomes
                tmp //= prep.outcomes
            key = int(canonical_leximin_coset_chain_uint64(evt, prep.outcomes, prep.level_invperms))
            column = global_event_map.get(key)
            if column is None:
                column = 1 + nof_marginals + len(global_keys)
                global_event_map[key] = column
                global_keys.append(key)
                dense = np.pad(dense, ((0, 0), (0, 1)))
            dense[row_num, column] += 1.0
    known_rows = np.zeros((nof_marginals, dense.shape[1]), dtype=np.float64)
    for row_num in range(nof_marginals):
        known_rows[row_num, 1 + row_num] = 1.0
    return np.asarray(global_keys, dtype=np.uint64), np.vstack((dense, known_rows))


class TestClusterOptimizedRing(unittest.TestCase):
    def _make_prep(self, n: int, **kwargs) -> PrepLP:
        defaults = {
            "show_progress": False,
            "auto_discover_symmetries": True,
            "compress_rows_under_discovered_group": True,
            "verbose_symmetry_discovery": False,
            "verbose_cache": False,
        }
        defaults.update(kwargs)
        return PrepLP(n, _UniformBinaryDistribution(), **defaults)

    def test_parallel_global_extensions_match_serial_reference(self):
        for n in (3, 4):
            with self.subTest(n=n):
                prep = self._make_prep(n)
                expected_keys, expected_dense = _reference_global_keys_and_matrix(prep)
                np.testing.assert_array_equal(prep.global_keys, expected_keys)
                np.testing.assert_allclose(prep.inflation_matrix.toarray(), expected_dense)

    def test_parallel_pipeline_is_deterministic(self):
        prep_a = self._make_prep(4)
        prep_b = self._make_prep(4)
        np.testing.assert_array_equal(prep_a.global_keys, prep_b.global_keys)
        np.testing.assert_array_equal(prep_a.variable_names, prep_b.variable_names)
        np.testing.assert_allclose(prep_a.inflation_matrix.toarray(), prep_b.inflation_matrix.toarray())

    def test_new_cache_roundtrip_and_old_cache_rejected(self):
        cache_name = f"cluster_opt_{uuid.uuid4().hex}"
        prep = None
        cached = None
        try:
            prep = self._make_prep(3, problem_name=cache_name)
            _ = prep.variable_names
            _ = prep.inflation_matrix
            self.assertIsNotNone(prep.cache_path)
            self.assertTrue(prep.cache_path.exists())

            cached = self._make_prep(3, problem_name=cache_name)
            np.testing.assert_array_equal(cached.global_keys, prep.global_keys)
            np.testing.assert_allclose(cached.inflation_matrix.toarray(), prep.inflation_matrix.toarray())
        finally:
            cache_path = None
            if prep is not None and prep.cache_path is not None:
                cache_path = prep.cache_path
            elif cached is not None and cached.cache_path is not None:
                cache_path = cached.cache_path
            if cache_path is not None and cache_path.exists():
                cache_path.unlink()

        stale_name = f"cluster_opt_stale_{uuid.uuid4().hex}"
        stale_prep = self._make_prep(3, problem_name=stale_name)
        stale_path = stale_prep.cache_path
        self.assertIsNotNone(stale_path)
        stale_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            np.savez_compressed(
                stale_path,
                cache_format_version=np.int64(CACHE_FORMAT_VERSION - 1),
            )
            with self.assertRaisesRegex(ValueError, "Incompatible cache"):
                self._make_prep(3, problem_name=stale_name)
        finally:
            if stale_path.exists():
                stale_path.unlink()

    def test_prep_solve_matches_generic_solver_without_tocsc(self):
        prep = self._make_prep(3)
        self.assertIsInstance(prep.inflation_matrix, csr_array)
        self.assertEqual(prep.inflation_matrix.indptr.dtype, np.int64)
        self.assertEqual(prep.inflation_matrix.data.dtype, np.float64)

        generic_solution = solveLP_sparse(
            objective=prep.blank_objective,
            known_vars=coo_array(([], ([], [])), shape=(1, prep.nof_lp_vars), dtype=np.float64),
            equalities=prep.inflation_matrix.tocoo(copy=False),
            default_non_negative=True,
            variables=prep.variable_names,
            verbose=0,
        )

        with mock.patch("scipy.sparse._csr.csr_array.tocsc", side_effect=AssertionError("unexpected tocsc")):
            direct_solution = prep.solve(
                verbose=0,
            )

        self.assertEqual(direct_solution["status"], generic_solution["status"])
        self.assertEqual(direct_solution["success"], generic_solution["success"])
        self.assertAlmostEqual(
            float(direct_solution["primal_value"]),
            float(generic_solution["primal_value"]),
            places=9,
        )
        self.assertAlmostEqual(
            float(direct_solution["dual_value"]),
            float(generic_solution["dual_value"]),
            places=9,
        )
        np.testing.assert_allclose(
            direct_solution["sparse_certificate"].toarray(),
            generic_solution["sparse_certificate"].toarray(),
            atol=1e-12,
        )

    def test_prep_solve_rejects_unknown_optimizer_name(self):
        prep = self._make_prep(3)
        with self.assertRaisesRegex(ValueError, "Unknown optimizer choice"):
            prep.solve(optimizer="not_a_solver", verbose=0)


if __name__ == "__main__":
    unittest.main()
