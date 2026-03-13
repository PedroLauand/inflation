import tempfile
import uuid
import unittest
import io
from pathlib import Path
from contextlib import redirect_stdout
from unittest import mock

import numpy as np
import sympy as sp
from scipy.sparse import coo_array, csr_array

import inflation.applications.Final_algo_numba as final_algo_numba
from inflation.applications.Final_algo_numba import (
    CACHE_FORMAT_VERSION,
    PrepLP,
    _count_unique_global_extension_keys_per_row,
    _detect_worker_count,
    _detect_total_memory_budget_bytes,
    _fill_unique_global_extension_keys_per_row,
    read_prep_lp_solution,
    _offdiag_slot_index,
    _union_sorted_unique_uint64,
)
from inflation.applications.Group_utils import canonical_leximin_coset_chain_uint64
from inflation.distributions import GHZDistribution, NSIPRDistribution
from inflation.lp.lp_utils import solveLP_sparse


class _UniformBinaryDistribution:
    @property
    def nof_outcomes(self):
        return 2

    def prob_event_loop(self, outcomes):
        return sp.Rational(1, 2) ** len(tuple(outcomes))

    def prob_event_line(self, outcomes):
        return sp.Rational(1, 2) ** len(tuple(outcomes))


class _ParityBinaryDistribution:
    @property
    def nof_outcomes(self):
        return 2

    def prob_event_loop(self, outcomes):
        return sp.Integer(1) if sum(tuple(outcomes)) % 2 == 0 else sp.Integer(0)

    def prob_event_line(self, outcomes):
        return self.prob_event_loop(outcomes)


def _reference_global_keys_and_matrix(prep: PrepLP) -> tuple[np.ndarray, np.ndarray]:
    nof_marginals = prep.nof_marginals
    nof_slots = prep.nof_off_diagonal_slots
    global_event_map: dict[int, int] = {}
    global_keys: list[int] = []
    dense = np.zeros((nof_marginals, 0), dtype=np.float64)

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

        for pos in range(total):
            tmp = pos
            for rem_pos in range(remaining.size - 1, -1, -1):
                idx = remaining[rem_pos]
                evt[idx] = tmp % prep.outcomes
                tmp //= prep.outcomes
            key = int(canonical_leximin_coset_chain_uint64(evt, prep.outcomes, prep.level_invperms))
            column = global_event_map.get(key)
            if column is None:
                column = len(global_keys)
                global_event_map[key] = column
                global_keys.append(key)
                dense = np.pad(dense, ((0, 0), (0, 1)))
            dense[row_num, column] += 1.0
    ordered_keys = np.asarray(global_keys, dtype=np.uint64)
    order = np.argsort(ordered_keys, kind="stable")
    sorted_keys = ordered_keys[order]
    sorted_dense = dense[:, order]
    return sorted_keys, sorted_dense


def _legacy_feasibility_args(prep: PrepLP) -> dict:
    direct = prep.inflation_matrix.tocoo(copy=False)
    nof_marginals = prep.nof_marginals
    total_cols = 1 + nof_marginals + prep.nof_lp_vars

    known_rows = np.arange(nof_marginals, dtype=np.int32)
    known_cols = np.arange(1, nof_marginals + 1, dtype=np.int64)
    global_cols = direct.col.astype(np.int64, copy=False) + np.int64(1 + nof_marginals)
    equalities = coo_array(
        (
            np.concatenate((
                -np.ones(nof_marginals, dtype=np.float64),
                direct.data.astype(np.float64, copy=False),
            )),
            (
                np.concatenate((known_rows, direct.row.astype(np.int32, copy=False))),
                np.concatenate((known_cols, global_cols)),
            ),
        ),
        shape=(nof_marginals, total_cols),
    )
    known_vars = coo_array(
        (
            prep.known_values.astype(np.float64, copy=False),
            (
                np.zeros(nof_marginals, dtype=np.int32),
                known_cols,
            ),
        ),
        shape=(1, total_cols),
    )
    variables = np.asarray(["1", *prep.known_labels, *prep.global_keys.tolist()], dtype=object)
    objective = coo_array(([], ([], [])), shape=(1, total_cols), dtype=np.float64)
    return {
        "objective": objective,
        "known_vars": known_vars,
        "equalities": equalities,
        "variables": variables,
        "default_non_negative": True,
        "verbose": 0,
    }


class TestClusterOptimizedRing(unittest.TestCase):
    def _make_prep(self, n: int, distribution=None, **kwargs) -> PrepLP:
        defaults = {
            "show_progress": False,
            "auto_discover_symmetries": True,
            "compress_rows_under_discovered_group": True,
            "verbose_symmetry_discovery": False,
            "verbose_cache": False,
        }
        defaults.update(kwargs)
        if distribution is None:
            distribution = _UniformBinaryDistribution()
        return PrepLP(n, distribution, **defaults)

    def test_parallel_global_extensions_match_serial_reference(self):
        for n in (3, 4):
            with self.subTest(n=n):
                prep = self._make_prep(n)
                expected_keys, expected_dense = _reference_global_keys_and_matrix(prep)
                np.testing.assert_array_equal(prep.global_keys, expected_keys)
                np.testing.assert_allclose(prep.inflation_matrix.toarray(), expected_dense)

    def test_row_labels_use_grouped_cycle_notation(self):
        prep = self._make_prep(4, distribution=GHZDistribution())
        self.assertTrue(any("[{" in label for label in prep.row_labels.tolist()))
        self.assertFalse(any("A^{" in label for label in prep.row_labels.tolist()))

    def test_print_certificate_explains_incompatible_fraction_threshold(self):
        prep = self._make_prep(4, distribution=GHZDistribution())
        solution = prep.solve(verbose=0)
        self.assertEqual(prep.solve_target, "incompatible_fraction")
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            prep.print_certificate(solution, max_terms=1)
        output = stdout.getvalue()
        self.assertIn("normalized certificate value on knowns:", output)
        self.assertIn("certificate must be at least 0 for incompatible fraction 0", output)
        self.assertIn("violation / negativity:", output)
        self.assertIn("negativity certifies incompatible fraction >=", output)
        self.assertIn("normalized affine certificate:", output)
        self.assertIn("    - 1", output)
        self.assertNotIn("[   0]", output)

    def test_print_certificate_falls_back_to_prep_solve_target(self):
        prep = self._make_prep(4, distribution=GHZDistribution())
        solution = prep.solve(verbose=0)
        solution_without_mode = dict(solution)
        solution_without_mode.pop("mode", None)
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            prep.print_certificate(solution_without_mode, max_terms=1)
        output = stdout.getvalue()
        self.assertIn("certificate must be at least 0 for incompatible fraction 0", output)

    def test_detect_worker_count_prefers_slurm_then_numba_then_fallback(self):
        with mock.patch.dict(final_algo_numba.os.environ, {"SLURM_CPUS_PER_TASK": "5"}, clear=True):
            with mock.patch.object(final_algo_numba, "get_num_threads", return_value=17):
                self.assertEqual(_detect_worker_count(), 5)

        with mock.patch.dict(final_algo_numba.os.environ, {}, clear=True):
            with mock.patch.object(final_algo_numba, "get_num_threads", return_value=6):
                self.assertEqual(_detect_worker_count(), 6)

        with mock.patch.dict(final_algo_numba.os.environ, {}, clear=True):
            with mock.patch.object(final_algo_numba, "get_num_threads", side_effect=RuntimeError("boom")):
                with mock.patch.object(final_algo_numba.os, "sched_getaffinity", new=None, create=True):
                    with mock.patch.object(final_algo_numba.os, "cpu_count", return_value=None):
                        self.assertEqual(_detect_worker_count(), 8)

    def test_worker_count_aligns_numba_threads_only_for_slurm(self):
        with mock.patch.dict(final_algo_numba.os.environ, {"SLURM_CPUS_PER_TASK": "4"}, clear=True):
            with mock.patch.object(final_algo_numba, "set_num_threads") as set_threads:
                prep = self._make_prep(3)
                self.assertEqual(prep.worker_count, 4)
                set_threads.assert_called_once_with(4)

        with mock.patch.dict(final_algo_numba.os.environ, {}, clear=True):
            with mock.patch.object(final_algo_numba, "set_num_threads") as set_threads:
                with mock.patch.object(final_algo_numba, "get_num_threads", return_value=3):
                    prep = self._make_prep(3)
                    self.assertEqual(prep.worker_count, 3)
                set_threads.assert_not_called()

    def test_detect_total_memory_budget_prefers_slurm_then_local(self):
        with mock.patch.dict(final_algo_numba.os.environ, {"SLURM_MEM_PER_NODE": "200000"}, clear=True):
            self.assertEqual(_detect_total_memory_budget_bytes(16), 200000 * 1024 ** 2)

        with mock.patch.dict(final_algo_numba.os.environ, {"SLURM_MEM_PER_CPU": "125G"}, clear=True):
            self.assertEqual(_detect_total_memory_budget_bytes(4), 125 * 4 * 1024 ** 3)

        with mock.patch.dict(final_algo_numba.os.environ, {}, clear=True):
            with mock.patch.object(final_algo_numba, "_detect_local_memory_bytes", return_value=99):
                self.assertEqual(_detect_total_memory_budget_bytes(8), 99)

    def test_two_pass_row_pipeline_emits_exact_sorted_unique_counts(self):
        prep = self._make_prep(4, distribution=NSIPRDistribution())
        (
            _row_entry_ptr,
            row_fixed_ptr,
            fixed_slots_flat,
            fixed_vals_flat,
            row_remaining_ptr,
            remaining_slots_flat,
        ) = prep._row_extension_descriptor_payload
        wave_rows = np.arange(min(4, prep.nof_marginals), dtype=np.int64)
        row_unique_nnz = np.empty(wave_rows.size, dtype=np.int64)
        _count_unique_global_extension_keys_per_row(
            row_unique_nnz,
            wave_rows,
            prep.row_extension_counts.astype(np.int64, copy=False),
            row_fixed_ptr,
            fixed_slots_flat,
            fixed_vals_flat,
            row_remaining_ptr,
            remaining_slots_flat,
            prep.nof_off_diagonal_slots,
            prep.outcomes,
            prep.level_invperms,
        )
        output_ptr = np.empty(wave_rows.size + 1, dtype=np.int64)
        output_ptr[0] = 0
        output_ptr[1:] = np.cumsum(row_unique_nnz, dtype=np.int64)
        flat_keys = np.empty(int(output_ptr[-1]), dtype=np.uint64)
        flat_counts = np.empty(int(output_ptr[-1]), dtype=np.uint64)
        _fill_unique_global_extension_keys_per_row(
            flat_keys,
            flat_counts,
            output_ptr,
            wave_rows,
            prep.row_extension_counts.astype(np.int64, copy=False),
            row_fixed_ptr,
            fixed_slots_flat,
            fixed_vals_flat,
            row_remaining_ptr,
            remaining_slots_flat,
            prep.nof_off_diagonal_slots,
            prep.outcomes,
            prep.level_invperms,
        )
        for local_row, row_num in enumerate(wave_rows.tolist()):
            start = int(output_ptr[local_row])
            stop = int(output_ptr[local_row + 1])
            row_keys = flat_keys[start:stop]
            row_counts = flat_counts[start:stop]
            self.assertEqual(stop - start, int(row_unique_nnz[local_row]))
            self.assertTrue(np.all(row_counts > 0))
            if row_keys.size > 1:
                self.assertTrue(np.all(row_keys[1:] > row_keys[:-1]))
            self.assertEqual(int(row_counts.sum()), int(prep.row_extension_counts[row_num]))

    def test_row_archives_are_written_once_per_row_and_preserve_row_totals(self):
        prep = self._make_prep(4, distribution=NSIPRDistribution())
        archived_rows: list[tuple[int, np.ndarray, np.ndarray]] = []
        original_write = final_algo_numba._write_row_counts_archive

        def capture_write(path, keys, counts):
            row_num = int(Path(path).stem.split("_")[1])
            archived_rows.append(
                (
                    row_num,
                    np.asarray(keys, dtype=np.uint64).copy(),
                    np.asarray(counts, dtype=np.uint64).copy(),
                )
            )
            return original_write(path, keys, counts)

        with mock.patch.object(final_algo_numba, "_write_row_counts_archive", side_effect=capture_write):
            _ = prep.global_keys

        self.assertEqual(len(archived_rows), prep.nof_marginals)
        self.assertEqual(sorted(row_num for row_num, _keys, _counts in archived_rows), list(range(prep.nof_marginals)))
        for row_num, keys, counts in archived_rows:
            self.assertEqual(keys.size, counts.size)
            self.assertTrue(np.all(counts > 0))
            if keys.size > 1:
                self.assertTrue(np.all(keys[1:] > keys[:-1]))
            self.assertEqual(int(counts.sum()), int(prep.row_extension_counts[row_num]))

    def test_sorted_key_union_helper(self):
        union = _union_sorted_unique_uint64(
            np.asarray([1, 4, 8], dtype=np.uint64),
            np.asarray([1, 3, 8, 10], dtype=np.uint64),
        )
        np.testing.assert_array_equal(union, np.asarray([1, 3, 4, 8, 10], dtype=np.uint64))

    def test_largest_row_buffer_budget_violation_fails_fast(self):
        prep = self._make_prep(3)
        with mock.patch.object(final_algo_numba, "_detect_total_memory_budget_bytes", return_value=1):
            with self.assertRaisesRegex(MemoryError, "largest marginal row requires a raw uint64 key buffer"):
                _ = prep.global_keys

    def test_direct_matrix_public_api_shape(self):
        prep = self._make_prep(3)
        self.assertEqual(prep.inflation_matrix.shape, (prep.nof_marginals, prep.nof_lp_vars))
        self.assertEqual(prep.nof_lp_constraints, prep.nof_marginals)
        self.assertEqual(prep.global_keys.size, prep.nof_lp_vars)
        self.assertEqual(prep.known_values.size, prep.nof_lp_constraints)
        self.assertEqual(prep.row_labels.size, prep.nof_lp_constraints)
        self.assertFalse(hasattr(prep, "known_vars"))
        self.assertFalse(hasattr(prep, "known_vars_symbolic"))
        self.assertFalse(hasattr(prep, "blank_objective"))

    def test_parallel_pipeline_is_deterministic(self):
        prep_a = self._make_prep(4)
        prep_b = self._make_prep(4)
        np.testing.assert_array_equal(prep_a.global_keys, prep_b.global_keys)
        np.testing.assert_array_equal(prep_a.global_keys, prep_b.global_keys)
        np.testing.assert_allclose(prep_a.inflation_matrix.toarray(), prep_b.inflation_matrix.toarray())

    def test_new_cache_roundtrip_and_old_cache_rejected(self):
        cache_name = f"cluster_opt_{uuid.uuid4().hex}"
        prep = None
        cached = None
        try:
            prep = self._make_prep(3, problem_name=cache_name)
            _ = prep.global_keys
            _ = prep.inflation_matrix
            self.assertIsNotNone(prep.cache_path)
            self.assertTrue(prep.cache_path.exists())

            cached = self._make_prep(3, problem_name=cache_name)
            self.assertIsNone(cached._cached_inflation_matrix)
            np.testing.assert_array_equal(cached.global_keys, prep.global_keys)
            np.testing.assert_allclose(cached.inflation_matrix.toarray(), prep.inflation_matrix.toarray())
            np.testing.assert_array_equal(cached.global_keys, prep.global_keys)
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

    def test_prep_solve_feasibility_matches_generic_padded_formulation(self):
        prep = self._make_prep(3)
        matrix = prep.inflation_matrix
        self.assertIsInstance(matrix, csr_array)
        self.assertEqual(matrix.indptr.dtype, np.int64)
        self.assertEqual(matrix.data.dtype, np.float64)

        generic_solution = solveLP_sparse(**_legacy_feasibility_args(prep))

        with mock.patch("scipy.sparse._csr.csr_array.tocsc", side_effect=AssertionError("unexpected tocsc")):
            direct_solution = prep.solve(
                mode="feasibility",
                verbose=0,
            )

        self.assertEqual(direct_solution["mode"], "feasibility")
        self.assertEqual(direct_solution["status"], generic_solution["status"])
        self.assertEqual(direct_solution["success"], generic_solution["success"])
        self.assertTrue(direct_solution["solver_success"])
        self.assertAlmostEqual(float(direct_solution["primal_value"]), 0.0, places=9)
        self.assertAlmostEqual(
            float(direct_solution["primal_value"]),
            float(generic_solution["primal_value"]),
            places=9,
        )

        payload_only_prep = self._make_prep(3)
        _ = payload_only_prep.global_keys
        self.assertIsNone(payload_only_prep._cached_inflation_matrix)
        _ = payload_only_prep.solve(mode="feasibility", verbose=0)
        self.assertIsNone(payload_only_prep._cached_inflation_matrix)

    def test_default_mode_reports_zero_incompatible_fraction_on_nsi_n4(self):
        prep = self._make_prep(4, distribution=NSIPRDistribution())
        solution = prep.solve(verbose=0)

        self.assertEqual(solution["mode"], "incompatible_fraction")
        self.assertTrue(solution["solver_success"])
        self.assertTrue(solution["success"])
        self.assertAlmostEqual(float(solution["incompatible_fraction"]), 0.0, places=9)
        self.assertEqual(prep.nof_lp_constraints, prep.nof_marginals)
        self.assertEqual(set(solution["x"]), set(prep.global_keys.tolist()))
        self.assertEqual(set(solution["dual_certificate"]), set(solution["constraint_names"][solution["sparse_certificate"].col]))
        self.assertEqual(solution["sparse_certificate"].shape, (1, prep.nof_lp_constraints))

    def test_relaxed_metrics_positive_on_incompatible_case(self):
        prep = self._make_prep(3, distribution=_ParityBinaryDistribution())

        incompatible_fraction_solution = prep.solve(mode="incompatible_fraction", verbose=0)
        generalized_robustness_solution = prep.solve(mode="generalized_robustness", verbose=0)

        self.assertTrue(incompatible_fraction_solution["solver_success"])
        self.assertFalse(incompatible_fraction_solution["success"])
        self.assertGreater(float(incompatible_fraction_solution["incompatible_fraction"]), 0.0)

        self.assertTrue(generalized_robustness_solution["solver_success"])
        self.assertFalse(generalized_robustness_solution["success"])
        self.assertGreater(float(generalized_robustness_solution["generalized_robustness"]), 0.0)

    def test_ghz_n3_all_equal_distribution_is_feasible(self):
        prep = self._make_prep(3, distribution=GHZDistribution())

        relaxed_solution = prep.solve(mode="incompatible_fraction", verbose=0)
        feasibility_solution = prep.solve(mode="feasibility", verbose=0)

        self.assertTrue(relaxed_solution["solver_success"])
        self.assertTrue(relaxed_solution["success"])
        self.assertAlmostEqual(float(relaxed_solution["incompatible_fraction"]), 0.0, places=9)
        self.assertTrue(feasibility_solution["solver_success"])
        self.assertTrue(feasibility_solution["success"])
        self.assertEqual(feasibility_solution["status"], "optimal")
        self.assertEqual(feasibility_solution["sparse_certificate"].nnz, 0)

    def test_solution_roundtrip_preserves_direct_basis_metadata(self):
        prep = self._make_prep(4, distribution=NSIPRDistribution())
        solution = prep.solve(verbose=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ring_solution"
            archive = prep.save_solution(solution, path)
            restored = read_prep_lp_solution(archive)

        self.assertEqual(restored["mode"], solution["mode"])
        self.assertEqual(restored["success"], solution["success"])
        self.assertEqual(restored["solver_success"], solution["solver_success"])
        self.assertEqual(restored["x"], solution["x"])
        self.assertEqual(restored["dual_certificate"], solution["dual_certificate"])
        np.testing.assert_array_equal(restored["constraint_names"], solution["constraint_names"])
        self.assertAlmostEqual(restored["known_mass"], solution["known_mass"], places=12)
        self.assertAlmostEqual(restored["optimized_mass"], solution["optimized_mass"], places=12)
        self.assertAlmostEqual(
            restored["incompatible_fraction"],
            solution["incompatible_fraction"],
            places=12,
        )

    def test_inflation_matrix_is_reconstructed_lazily_from_payload(self):
        prep = self._make_prep(4)
        _ = prep.global_keys
        self.assertIsNone(prep._cached_inflation_matrix)
        matrix = prep.inflation_matrix
        self.assertIsInstance(matrix, csr_array)
        self.assertIs(prep._cached_inflation_matrix, matrix)

    def test_prep_solve_rejects_unknown_optimizer_name(self):
        prep = self._make_prep(3)
        with self.assertRaisesRegex(ValueError, "Unknown optimizer choice"):
            prep.solve(optimizer="not_a_solver", verbose=0)

    def test_prep_solve_rejects_unknown_mode(self):
        prep = self._make_prep(3)
        with self.assertRaisesRegex(ValueError, "Unknown solve mode"):
            prep.solve(mode="not_a_mode", verbose=0)


if __name__ == "__main__":
    unittest.main()
