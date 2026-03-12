import io
import unittest
import uuid
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

import sympy as sp

import inflation.applications.EJM_numba_test as ejm_numba_test
from inflation.applications.Final_algo_numba import PrepLP
from inflation.progress_utils import make_tqdm, progress_stage


class _NonTtyStringIO(io.StringIO):
    def isatty(self):
        return False


class _UniformBinaryDistribution:
    @property
    def nof_outcomes(self):
        return 2

    def prob_event_loop(self, outcomes):
        return sp.Rational(1, 2) ** len(tuple(outcomes))

    def prob_event_line(self, outcomes):
        return sp.Rational(1, 2) ** len(tuple(outcomes))


class _FakeMatrix:
    def __init__(self, shape):
        self.shape = shape


class _FakePrep:
    def __init__(self, *args, **kwargs):
        self.output_path = None
        self.blank_objective = "blank-objective"
        self._variable_names = ["1", "known"]
        self._known_vars = "known-vars"
        self._inflation_matrix = _FakeMatrix((7, 11))

    @property
    def variable_names(self):
        print("Canonicalizing marginals")
        return self._variable_names

    @property
    def known_vars(self):
        print("Computing marginal values...")
        return self._known_vars

    @property
    def inflation_matrix(self):
        print("Finalizing sparse extension matrix...")
        return self._inflation_matrix


class TestProgressUtils(unittest.TestCase):
    def test_progress_stage_writes_clean_status_lines_to_provided_stream(self):
        progress_stream = _NonTtyStringIO()
        stdout_stream = _NonTtyStringIO()

        with redirect_stdout(stdout_stream):
            with progress_stage(
                "Preparing demo...",
                file=progress_stream,
                end_message=lambda elapsed: f"Demo ready in {elapsed:.2f}s",
            ):
                pass

        output = progress_stream.getvalue()
        self.assertIn("Preparing demo...", output)
        self.assertIn("Demo ready in", output)
        self.assertIn("\n", output)
        self.assertNotIn("\r", output)
        self.assertNotIn("\x1b[A", output)
        self.assertEqual(stdout_stream.getvalue(), "")

    def test_make_tqdm_writes_clean_line_progress_to_provided_stream(self):
        progress_stream = _NonTtyStringIO()
        stdout_stream = _NonTtyStringIO()

        with redirect_stdout(stdout_stream):
            for _ in make_tqdm(
                range(3),
                file=progress_stream,
                desc="demo",
                disable=False,
                mininterval=0,
                miniters=1,
            ):
                pass

        output = progress_stream.getvalue()
        self.assertIn("demo", output)
        self.assertIn("\n", output)
        self.assertNotIn("\r", output)
        self.assertNotIn("\x1b[A", output)
        self.assertEqual(stdout_stream.getvalue(), "")

    def test_prep_lp_progress_uses_stdout_and_reports_finalization_and_cache_status(self):
        stdout_stream = _NonTtyStringIO()
        stderr_stream = _NonTtyStringIO()
        cache_name = f"test_progress_utils_{uuid.uuid4().hex}"

        prep = None
        prep_cached = None
        try:
            with redirect_stdout(stdout_stream), redirect_stderr(stderr_stream):
                prep = PrepLP(
                    3,
                    _UniformBinaryDistribution(),
                    problem_name=cache_name,
                    show_progress=True,
                    auto_discover_symmetries=True,
                    compress_rows_under_discovered_group=True,
                    verbose_symmetry_discovery=True,
                    verbose_cache=True,
                )
                _ = prep.variable_names
                _ = prep.known_vars
                _ = prep.inflation_matrix

            stdout_output = stdout_stream.getvalue()
            stderr_output = stderr_stream.getvalue()

            self.assertIn("Canonicalizing marginals", stdout_output)
            self.assertIn("Computing marginal values...", stdout_output)
            self.assertIn("Finding global extensions...", stdout_output)
            self.assertIn("Finalizing sparse extension matrix...", stdout_output)
            self.assertIn("Constraint matrix finalized:", stdout_output)
            self.assertIn("Saving LP input cache to", stdout_output)
            self.assertIn("Saved LP constraints cache to", stdout_output)
            self.assertNotIn("Discovering ring stabilizing symmetries", stdout_output)
            self.assertNotIn("Stabilizer subgroup summary", stdout_output)
            self.assertNotIn("Canonicalizing marginals", stderr_output)
            self.assertNotIn("Computing marginal values...", stderr_output)
            self.assertNotIn("Finding global extensions...", stderr_output)
            self.assertNotIn("\r", stdout_output)
            self.assertNotIn("\x1b[A", stdout_output)

            cached_stdout_stream = _NonTtyStringIO()
            cached_stderr_stream = _NonTtyStringIO()
            with redirect_stdout(cached_stdout_stream), redirect_stderr(cached_stderr_stream):
                prep_cached = PrepLP(
                    3,
                    _UniformBinaryDistribution(),
                    problem_name=cache_name,
                    show_progress=True,
                    auto_discover_symmetries=True,
                    compress_rows_under_discovered_group=True,
                    verbose_symmetry_discovery=True,
                    verbose_cache=True,
                )

            cached_stdout = cached_stdout_stream.getvalue()
            self.assertIn("Checking LP input cache at", cached_stdout)
            self.assertIn("Loaded cached LP constraints from", cached_stdout)
            self.assertNotIn("\r", cached_stdout)
            self.assertNotIn("\x1b[A", cached_stdout)
        finally:
            cache_path = None
            if prep is not None and prep.cache_path is not None:
                cache_path = prep.cache_path
            elif prep_cached is not None and prep_cached.cache_path is not None:
                cache_path = prep_cached.cache_path
            if cache_path is not None and cache_path.exists():
                cache_path.unlink()

    def test_ejm_entrypoint_materializes_inputs_before_solver_setup(self):
        stdout_stream = _NonTtyStringIO()
        solver_calls = []

        def fake_solve(**kwargs):
            solver_calls.append(kwargs)
            print("Starting pre-processing for the LP solver...")
            print("Optimizer started.")
            return {"status": "optimal"}

        with redirect_stdout(stdout_stream), mock.patch.object(ejm_numba_test, "PrepLP", _FakePrep), mock.patch.object(
            ejm_numba_test,
            "solveLP_sparse",
            side_effect=fake_solve,
        ):
            ejm_numba_test.main(n=3)

        output = stdout_stream.getvalue()
        expected_markers = [
            "PrepLP initialized for n=3; materializing LP inputs before Mosek.",
            "Canonicalizing marginals",
            "LP inputs ready for n=3: rows=7, cols=11. Starting Mosek setup.",
            "Starting pre-processing for the LP solver...",
            "Optimizer started.",
        ]
        positions = [output.index(marker) for marker in expected_markers]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(solver_calls[0]["verbose"], 2)
        self.assertEqual(solver_calls[0]["variables"], ["1", "known"])
        self.assertEqual(solver_calls[0]["known_vars"], "known-vars")
        self.assertIsInstance(solver_calls[0]["equalities"], _FakeMatrix)


if __name__ == "__main__":
    unittest.main()
