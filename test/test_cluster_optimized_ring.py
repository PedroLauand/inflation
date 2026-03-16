import tempfile
import uuid
import unittest
import io
from itertools import combinations, product
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
    _compute_unique_global_extension_keys_for_row,
    _count_reduced_base_candidates,
    _detect_worker_count,
    _detect_total_memory_budget_bytes,
    _cycles_from_J,
    _estimate_active_worker_peak_bytes,
    _format_exact_row_memory_tally_lines,
    _estimate_per_worker_peak_bytes,
    keep_loops_of_length,
    keep_loops_up_to_three,
    _relaxed_mass_gap,
    _relaxed_mass_tolerance,
    _perm_from_marginal,
    read_prep_lp_solution,
    _offdiag_slot_index,
    _iter_reduced_base_supports,
    _union_sorted_unique_uint64,
)
from inflation.applications.Group_utils import (
    build_sympy_group,
    canonical_leximin_coset_chain_uint64,
    canonical_leximin_support_indices,
)
from inflation.distributions import EJMDistribution, GHZDistribution, NSIPRDistribution
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


def _keep_any_two_or_three_cycles(marginal) -> bool:
    cycles = _cycles_from_J(_perm_from_marginal(marginal))
    return len(cycles) > 0 and all(len(cycle) in (2, 3) for cycle in cycles)


def _noninvariant_row_shape_filter(marginal) -> bool:
    return (
        len(marginal) > 1
        and int(marginal[0][2]) == 2
        and int(marginal[1][4]) == 1
    )


def _encode_event_key(evt: np.ndarray, outcomes: int) -> int:
    acc = 0
    base = 1
    for digit in evt.tolist():
        acc += int(digit) * base
        base *= int(outcomes)
    return acc


def _reference_support_canonicalizer(support: np.ndarray, group_elements: np.ndarray) -> tuple[int, ...]:
    best = None
    for perm in group_elements:
        candidate = tuple(sorted(int(perm[int(pos)]) for pos in support.tolist()))
        if best is None or candidate < best:
            best = candidate
    return best or tuple()


def _reference_event_canonicalizer(evt: np.ndarray, outcomes: int, group_elements: np.ndarray) -> int:
    slot_count = evt.size
    best = None
    for perm in group_elements:
        mapped_evt = np.empty(slot_count, dtype=np.uint8)
        for in_slot in range(slot_count):
            coord = in_slot * outcomes + int(evt[in_slot])
            mapped_coord = int(perm[coord])
            out_slot, out_digit = divmod(mapped_coord, outcomes)
            mapped_evt[out_slot] = np.uint8(out_digit)
        candidate = tuple(int(x) for x in mapped_evt.tolist())
        if best is None or candidate < best:
            best = candidate
    return _encode_event_key(np.asarray(best, dtype=np.uint8), outcomes)


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
            key = int(
                canonical_leximin_coset_chain_uint64(
                    evt,
                    prep.outcomes,
                    prep.slot_sources,
                    prep.outcome_maps,
                )
            )
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


def _reference_base_support_keys(
    prep: PrepLP,
    *,
    filter_fn=None,
    filter_on_raw: bool = False,
) -> list[tuple[int, ...]]:
    seen: set[tuple[int, ...]] = set()
    support_keys: list[tuple[int, ...]] = []
    base_outcomes = tuple(range(prep.outcomes))
    copy_labels = tuple(range(1, prep.n + 1))
    for subset_size in range(2, prep.n + 1):
        for subset in combinations(copy_labels, subset_size):
            for image in final_algo_numba._derangements(subset):
                for pat in product(base_outcomes, repeat=subset_size):
                    support = final_algo_numba._marginal_support_from_cycle_cover(
                        subset,
                        image,
                        pat,
                        prep.n,
                        prep.outcomes,
                    )
                    if filter_fn is not None and filter_on_raw:
                        raw_marginal = final_algo_numba._marginal_from_support_key(
                            support,
                            prep.n,
                            prep.outcomes,
                        )
                        if not filter_fn(raw_marginal):
                            continue
                    canonical = canonical_leximin_support_indices(
                        support,
                        prep.N,
                        prep.core_group_perms,
                    )
                    key = tuple(int(x) for x in canonical.tolist())
                    if filter_fn is not None and not filter_on_raw:
                        marginal = final_algo_numba._marginal_from_support_key(
                            key,
                            prep.n,
                            prep.outcomes,
                        )
                        if not filter_fn(marginal):
                            continue
                    if key not in seen:
                        seen.add(key)
                        support_keys.append(key)
    return support_keys


def _reference_reduced_base_support_keys(
    prep: PrepLP,
    *,
    filter_fn=None,
) -> list[np.ndarray]:
    support_keys: list[np.ndarray] = []
    for support in _iter_reduced_base_supports(prep.n, prep.outcomes):
        if filter_fn is not None:
            marginal = final_algo_numba._marginal_from_support_key(
                support,
                prep.n,
                prep.outcomes,
            )
            if not filter_fn(marginal):
                continue
        support_keys.append(support)
    return support_keys


def _support_key_bytes(key) -> bytes:
    return final_algo_numba.ndarray_bytes_key(np.asarray(key, dtype=np.int64), dtype=np.int64)


def _support_key_bytes_list(keys) -> list[bytes]:
    return [_support_key_bytes(key) for key in keys]


def _support_key_bytes_set(keys) -> set[bytes]:
    return set(_support_key_bytes_list(keys))


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

    def test_support_canonicalizer_is_idempotent_and_matches_reference(self):
        prep = self._make_prep(4, distribution=GHZDistribution())
        group = build_sympy_group(prep.core_symmetries, prep.N)
        reference_elements = np.asarray(list(group.generate_schreier_sims(af=True)), dtype=int)
        for key in prep.base_support_keys[: min(12, prep.base_nof_marginals)]:
            support = np.asarray(key, dtype=np.int64)
            canonical = canonical_leximin_support_indices(support, prep.N, prep.core_group_perms)
            np.testing.assert_array_equal(canonical, support)
            self.assertEqual(
                tuple(int(x) for x in canonical.tolist()),
                _reference_support_canonicalizer(support, reference_elements),
            )

    def test_event_canonicalizer_is_idempotent_and_matches_reference(self):
        prep = self._make_prep(4, distribution=GHZDistribution())
        _ = prep.global_keys
        group = build_sympy_group(prep.discovered_symmetries, prep.N)
        reference_elements = np.asarray(list(group.generate_schreier_sims(af=True)), dtype=int)
        for key in prep.global_keys[: min(24, prep.global_keys.size)]:
            evt = final_algo_numba._decode_uint64_event_key(
                np.uint64(key),
                prep.nof_off_diagonal_slots,
                prep.outcomes,
            )
            canonical = canonical_leximin_coset_chain_uint64(
                evt,
                prep.outcomes,
                prep.slot_sources,
                prep.outcome_maps,
            )
            self.assertEqual(int(canonical), int(key))
            self.assertEqual(
                int(key),
                _reference_event_canonicalizer(evt, prep.outcomes, reference_elements),
            )

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

    def test_structural_memory_estimator_defaults_and_filter_metadata(self):
        default_prep = self._make_prep(3)
        self.assertEqual(default_prep.smallest_marginal_size, 2)

        filtered_prep = self._make_prep(4, marginal_filter_fn=keep_loops_of_length([2, 4]))
        self.assertEqual(filtered_prep.smallest_marginal_size, 2)

        max3_prep = self._make_prep(4, marginal_filter_fn=keep_loops_up_to_three)
        self.assertEqual(max3_prep.smallest_marginal_size, 1)

        custom_prep = self._make_prep(4, marginal_filter_fn=lambda _m: True)
        self.assertEqual(custom_prep.smallest_marginal_size, 2)

    def test_base_support_keys_match_reference_on_unfiltered_nsi_small(self):
        for n in (3, 4, 5):
            with self.subTest(n=n):
                prep = self._make_prep(n, distribution=NSIPRDistribution())
                self.assertEqual(_support_key_bytes_set(prep.base_support_keys), _support_key_bytes_set(_reference_base_support_keys(prep)))
                self.assertEqual(len(prep.base_support_keys), len(_reference_base_support_keys(prep)))

    def test_base_nof_marginals_does_not_force_base_marginal_materialization(self):
        prep = self._make_prep(4, distribution=NSIPRDistribution())
        self.assertNotIn("base_support_keys", prep.__dict__)
        self.assertNotIn("base_marginals", prep.__dict__)

        self.assertEqual(prep.base_nof_marginals, len(prep.base_support_keys))
        self.assertIn("base_support_keys", prep.__dict__)
        self.assertNotIn("base_marginals", prep.__dict__)

        _ = prep.base_marginals
        self.assertIn("base_marginals", prep.__dict__)

    def test_symmetry_invariant_helper_filter_matches_reference_canonical_semantics(self):
        prep = self._make_prep(
            4,
            distribution=GHZDistribution(),
            marginal_filter_fn=keep_loops_of_length([2, 4]),
        )
        expected = _reference_base_support_keys(
            prep,
            filter_fn=keep_loops_of_length([2, 4]),
            filter_on_raw=False,
        )
        self.assertEqual(_support_key_bytes_set(prep.base_support_keys), _support_key_bytes_set(expected))
        self.assertEqual(len(prep.base_support_keys), len(expected))

    def test_reduced_base_candidates_match_bruteforce_and_are_fewer(self):
        for n, outcomes in ((4, 2), (5, 2), (4, 4)):
            with self.subTest(n=n, outcomes=outcomes):
                brute_force_total = sum(
                    sum(1 for _ in final_algo_numba._derangements(subset)) * (outcomes ** len(subset))
                    for subset_size in range(2, n + 1)
                    for subset in combinations(tuple(range(1, n + 1)), subset_size)
                )
                reduced_total = _count_reduced_base_candidates(n, outcomes)
                self.assertLess(reduced_total, brute_force_total)

    def test_reduced_base_supports_are_already_core_canonical_on_small_cases(self):
        cases = (
            (3, NSIPRDistribution()),
            (4, NSIPRDistribution()),
            (5, NSIPRDistribution()),
            (6, NSIPRDistribution()),
            (4, GHZDistribution()),
            (4, EJMDistribution()),
        )
        for n, distribution in cases:
            with self.subTest(n=n, outcomes=distribution.nof_outcomes):
                prep = self._make_prep(n, distribution=distribution)
                for support in _iter_reduced_base_supports(prep.n, prep.outcomes):
                    canonical = canonical_leximin_support_indices(
                        support,
                        prep.N,
                        prep.core_group_perms,
                    )
                    np.testing.assert_array_equal(support, canonical)

    def test_non_invariant_filter_uses_generated_representative_semantics(self):
        prep = self._make_prep(
            4,
            distribution=NSIPRDistribution(),
            marginal_filter_fn=_noninvariant_row_shape_filter,
        )
        raw_keys = _reference_reduced_base_support_keys(
            prep,
            filter_fn=_noninvariant_row_shape_filter,
        )
        self.assertEqual(_support_key_bytes_list(prep.base_support_keys), _support_key_bytes_list(raw_keys))
        self.assertGreater(len(raw_keys), 0)

    def test_one_pass_peak_estimators_match_worker_model(self):
        self.assertEqual(_estimate_per_worker_peak_bytes(12), 24 * 12)
        peak = _estimate_active_worker_peak_bytes(
            np.asarray([4, 8, 3, 10], dtype=np.int64),
            worker_count=2,
        )
        self.assertEqual(peak, 24 * (10 + 8))

    def test_exact_row_memory_tally_lines_group_and_sort_descending(self):
        lines = _format_exact_row_memory_tally_lines(
            np.asarray([1 << 28, 1 << 28, 1 << 27], dtype=np.int64),
            [
                [[1, 1, 2, 0, 0], [1, 2, 3, 0, 0], [1, 3, 4, 0, 0], [1, 4, 5, 0, 0], [1, 5, 6, 0, 0], [1, 6, 1, 0, 0]],
                [[1, 1, 2, 0, 0], [1, 2, 3, 0, 0], [1, 3, 1, 0, 0], [1, 4, 5, 0, 0], [1, 5, 6, 0, 0], [1, 6, 4, 0, 0]],
                [[1, 1, 2, 0, 0], [1, 2, 3, 0, 0], [1, 3, 4, 0, 0], [1, 4, 1, 0, 0], [1, 5, 6, 0, 0], [1, 6, 5, 0, 0]],
            ],
        )
        self.assertEqual(
            lines,
            [
                "2 rows at 6.0 GiB each (marginal size 6; types: 1x loop of 6; 2x loop of 3)",
                "1 row at 3.0 GiB each (marginal size 6; type: 1x loop of 4 + 1x loop of 2)",
            ],
        )

    def test_one_pass_row_kernel_emits_exact_sorted_unique_counts(self):
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
        for row_num in wave_rows.tolist():
            row_keys, row_counts = _compute_unique_global_extension_keys_for_row(
                int(row_num),
                prep.row_extension_counts.astype(np.int64, copy=False),
                row_fixed_ptr,
                fixed_slots_flat,
                fixed_vals_flat,
                row_remaining_ptr,
                remaining_slots_flat,
                prep.nof_off_diagonal_slots,
                prep.outcomes,
                prep.slot_sources,
                prep.outcome_maps,
            )
            self.assertTrue(np.all(row_counts > 0))
            if row_keys.size > 1:
                self.assertTrue(np.all(row_keys[1:] > row_keys[:-1]))
            self.assertEqual(int(row_counts.sum()), int(prep.row_extension_counts[row_num]))

    def test_row_archives_are_written_once_per_row_and_preserve_row_totals(self):
        with mock.patch.object(final_algo_numba, "_detect_worker_count", return_value=1):
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
        with mock.patch.object(final_algo_numba, "_detect_total_memory_budget_bytes", return_value=1):
            prep = self._make_prep(3)
            with self.assertRaisesRegex(MemoryError, "structural worst-case marginal row requires a one-pass worker peak"):
                _ = prep.global_keys

    def test_active_worker_budget_violation_fails_fast(self):
        prep = self._make_prep(3)
        with mock.patch.object(
            final_algo_numba,
            "_estimate_active_worker_peak_bytes",
            return_value=prep.usable_memory_budget_bytes + 1,
        ):
            with self.assertRaisesRegex(MemoryError, "exact active-worker bound is based on the top"):
                _ = prep.global_keys

    def test_structural_active_worker_overestimate_is_informational_only(self):
        prep = self._make_prep(3)
        with mock.patch.object(
            PrepLP,
            "worst_case_active_worker_peak_bytes",
            new_callable=mock.PropertyMock,
            return_value=prep.usable_memory_budget_bytes + 1,
        ):
            _ = prep.global_keys

    def test_discovered_row_orbit_validation_is_opt_in(self):
        with mock.patch.object(
            PrepLP,
            "_validated_discovered_row_orbits",
            new_callable=mock.PropertyMock,
            side_effect=AssertionError("validation should be skipped"),
        ):
            prep = self._make_prep(4, validate_discovered_row_orbits=False)
            _ = prep.row_labels

        with mock.patch.object(
            PrepLP,
            "_validated_discovered_row_orbits",
            new_callable=mock.PropertyMock,
            side_effect=AssertionError("validation enabled"),
        ):
            prep = self._make_prep(4, validate_discovered_row_orbits=True)
            with self.assertRaisesRegex(AssertionError, "validation enabled"):
                _ = prep.row_labels

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

    def test_relaxed_success_uses_mass_scaled_tolerance(self):
        known_mass = 1.0e6
        mass_tol = _relaxed_mass_tolerance(known_mass)
        tiny_relative_gap = 9.39099575881e-10 * known_mass
        large_relative_gap = 2.0e-8 * known_mass

        self.assertGreater(mass_tol, tiny_relative_gap)
        self.assertLess(mass_tol, large_relative_gap)

        small_gap = _relaxed_mass_gap(known_mass - tiny_relative_gap, known_mass, sense="upper")
        large_gap = _relaxed_mass_gap(known_mass - large_relative_gap, known_mass, sense="upper")
        self.assertLessEqual(small_gap, mass_tol)
        self.assertGreater(large_gap, mass_tol)

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

    def test_row_compression_matches_uncompressed_on_ghz_filtered_case(self):
        base_kwargs = {
            "marginal_filter_fn": _keep_any_two_or_three_cycles,
            "auto_discover_symmetries": True,
            "show_progress": False,
            "verbose_cache": False,
        }
        prep_uncompressed = self._make_prep(
            4,
            distribution=GHZDistribution(),
            compress_rows_under_discovered_group=False,
            **base_kwargs,
        )
        prep_compressed = self._make_prep(
            4,
            distribution=GHZDistribution(),
            compress_rows_under_discovered_group=True,
            **base_kwargs,
        )

        sol_uncompressed = prep_uncompressed.solve(mode="incompatible_fraction", verbose=0)
        sol_compressed = prep_compressed.solve(mode="incompatible_fraction", verbose=0)

        self.assertFalse(sol_uncompressed["success"])
        self.assertFalse(sol_compressed["success"])
        self.assertGreater(float(sol_uncompressed["incompatible_fraction"]), 0.0)
        self.assertAlmostEqual(
            float(sol_uncompressed["incompatible_fraction"]),
            float(sol_compressed["incompatible_fraction"]),
            places=9,
        )
        self.assertTrue(
            any(sp.simplify(value) == 0 for value in prep_uncompressed.base_known_values_symbolic.tolist())
        )
        self.assertTrue(any(sp.simplify(value) == 0 for value in prep_compressed.known_values_symbolic.tolist()))

    def test_row_compression_matches_uncompressed_on_nsi_n4(self):
        prep_uncompressed = self._make_prep(
            4,
            distribution=NSIPRDistribution(),
            compress_rows_under_discovered_group=False,
        )
        prep_compressed = self._make_prep(
            4,
            distribution=NSIPRDistribution(),
            compress_rows_under_discovered_group=True,
        )

        sol_uncompressed = prep_uncompressed.solve(mode="incompatible_fraction", verbose=0)
        sol_compressed = prep_compressed.solve(mode="incompatible_fraction", verbose=0)

        self.assertEqual(sol_uncompressed["success"], sol_compressed["success"])
        self.assertAlmostEqual(
            float(sol_uncompressed["incompatible_fraction"]),
            float(sol_compressed["incompatible_fraction"]),
            places=9,
        )

    def test_compressed_orbit_members_share_identical_row_signatures(self):
        prep = self._make_prep(
            4,
            distribution=GHZDistribution(),
            marginal_filter_fn=_keep_any_two_or_three_cycles,
            compress_rows_under_discovered_group=True,
        )
        for orbit_members in prep.row_orbit_members:
            rep_keys = None
            rep_counts = None
            for base_idx in orbit_members:
                keys, counts = prep._row_signature_for_marginal(prep.base_marginals[base_idx])
                if rep_keys is None:
                    rep_keys = keys
                    rep_counts = counts
                    continue
                np.testing.assert_array_equal(keys, rep_keys)
                np.testing.assert_array_equal(counts, rep_counts)

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
