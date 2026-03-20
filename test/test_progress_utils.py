import io
import unittest
import uuid
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

import sympy as sp

import inflation.applications.EJM_5_test as ejm_5_test
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
        self._global_keys = [101, 202]
        self._known_values = [0.5, 0.5]
        self._inflation_matrix = _FakeMatrix((7, 11))
        self.solve_calls = []

    @property
    def global_keys(self):
        print("Enumerating base marginals")
        return self._global_keys

    @property
    def known_values(self):
        print("Computing marginal values...")
        return self._known_values

    @property
    def inflation_matrix(self):
        print("Finalizing sparse extension matrix...")
        return self._inflation_matrix

    @property
    def nof_lp_constraints(self):
        return self._inflation_matrix.shape[0]

    @property
    def nof_lp_vars(self):
        return self._inflation_matrix.shape[1]

    def solve(self, **kwargs):
        self.solve_calls.append(kwargs)
        print("Starting pre-processing for the LP solver...")
        print("Optimizer started.")
        return {
            "status": "optimal",
            "success": True,
            "incompatible_fraction": 0.0,
        }


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
            with mock.patch.dict("inflation.applications.Final_algo_numba.os.environ", {"SLURM_CPUS_PER_TASK": "2"}, clear=False):
                with mock.patch("inflation.applications.Final_algo_numba.set_num_threads"):
                    with redirect_stdout(stdout_stream), redirect_stderr(stderr_stream):
                        prep = PrepLP(
                            3,
                            _UniformBinaryDistribution(),
                            problem_name=cache_name,
                            show_progress=True,
                            auto_discover_symmetries=True,
                            compress_rows_under_discovered_group=True,
                        )
                        _ = prep.global_keys
                        _ = prep.known_values
                        _ = prep.inflation_matrix

            stdout_output = stdout_stream.getvalue()
            stderr_output = stderr_stream.getvalue()

            self.assertIn("Build resources:", stdout_output)
            self.assertIn("threads=", stdout_output)
            self.assertIn("usable memory=", stdout_output)
            self.assertNotIn("Structural memory estimate", stdout_output)
            self.assertLess(
                stdout_output.index("Build resources:"),
                stdout_output.index("Core group order:"),
            )
            self.assertLess(
                stdout_output.index("Core group order:"),
                stdout_output.index("Enumerating base marginals"),
            )
            self.assertIn("Core group order:", stdout_output)
            self.assertIn("Enumerating base marginals", stdout_output)
            self.assertIn("Base marginal type summary", stdout_output)
            self.assertIn("rows of type", stdout_output)
            self.assertIn("1x loop of 3", stdout_output)
            self.assertIn("Computing marginal values...", stdout_output)
            self.assertIn("Discovered group order:", stdout_output)
            self.assertIn("Final marginal type summary after discovered symmetry compression", stdout_output)
            self.assertIn("Global extension workload:", stdout_output)
            self.assertIn("rows=", stdout_output)
            self.assertIn("log2 total entries=", stdout_output)
            self.assertIn("log2 max row entries=", stdout_output)
            self.assertIn("initial per-thread peak upper=", stdout_output)
            self.assertIn("using ", stdout_output)
            self.assertIn("concurrent thread", stdout_output)
            self.assertIn("peak memory accross all threads=", stdout_output)
            self.assertIn("Initial row memory upper-bound tally (assumes no compression):", stdout_output)
            self.assertIn("marginal size", stdout_output)
            self.assertIn("loop of", stdout_output)
            self.assertNotIn("Initial active-worker upper bound:", stdout_output)
            self.assertIn("Finding global extensions...", stdout_output)
            self.assertIn("Global extensions task complete:", stdout_output)
            self.assertIn("rows done=", stdout_output)
            self.assertIn("estimated compression=", stdout_output)
            self.assertIn("final compression=", stdout_output)
            self.assertIn("active=", stdout_output)
            self.assertIn("Finalizing direct LP payload...", stdout_output)
            self.assertIn("Exact final payload:", stdout_output)
            self.assertIn("Direct LP payload finalized:", stdout_output)
            self.assertIn("payload=", stdout_output)
            self.assertIn("Saving LP input cache to", stdout_output)
            self.assertIn("Saved LP constraints cache to", stdout_output)
            self.assertNotIn("work_units=", stdout_output)
            self.assertNotIn("chunk_entries=", stdout_output)
            self.assertNotIn("waves=", stdout_output)
            self.assertNotIn("Discovering ring stabilizing symmetries", stdout_output)
            self.assertNotIn("Stabilizer subgroup summary", stdout_output)
            self.assertNotIn("Enumerating base marginals", stderr_output)
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
                    early_memory_estimate=True,
                    auto_discover_symmetries=True,
                    compress_rows_under_discovered_group=True,
                    verbose_cache=True,
                )

            cached_stdout = cached_stdout_stream.getvalue()
            self.assertIn("Build resources:", cached_stdout)
            self.assertIn("Core group order:", cached_stdout)
            self.assertIn("Structural memory estimate", cached_stdout)
            self.assertIn("smallest_marginal_size=", cached_stdout)
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

    def test_verbose_cache_defaults_to_show_progress_but_explicit_false_suppresses_cache_chatter(self):
        stdout_stream = _NonTtyStringIO()
        stderr_stream = _NonTtyStringIO()
        cache_name = f"test_progress_utils_{uuid.uuid4().hex}"
        prep = None
        try:
            with mock.patch.dict("inflation.applications.Final_algo_numba.os.environ", {"SLURM_CPUS_PER_TASK": "2"}, clear=False):
                with mock.patch("inflation.applications.Final_algo_numba.set_num_threads"):
                    with redirect_stdout(stdout_stream), redirect_stderr(stderr_stream):
                        prep = PrepLP(
                            3,
                            _UniformBinaryDistribution(),
                            problem_name=cache_name,
                            show_progress=True,
                            auto_discover_symmetries=True,
                            compress_rows_under_discovered_group=True,
                            verbose_cache=False,
                        )
                        _ = prep.global_keys
                        _ = prep.known_values
                        _ = prep.inflation_matrix

            stdout_output = stdout_stream.getvalue()
            self.assertNotIn("Saving LP input cache to", stdout_output)
            self.assertNotIn("Saved LP constraints cache to", stdout_output)
        finally:
            if prep is not None and prep.cache_path is not None and prep.cache_path.exists():
                prep.cache_path.unlink()

    def test_no_auto_discovery_skips_discovered_group_order_and_second_type_summary(self):
        stdout_stream = _NonTtyStringIO()
        stderr_stream = _NonTtyStringIO()

        with mock.patch.dict("inflation.applications.Final_algo_numba.os.environ", {"SLURM_CPUS_PER_TASK": "2"}, clear=False):
            with mock.patch("inflation.applications.Final_algo_numba.set_num_threads"):
                with redirect_stdout(stdout_stream), redirect_stderr(stderr_stream):
                    prep = PrepLP(
                        3,
                        _UniformBinaryDistribution(),
                        problem_name=None,
                        show_progress=True,
                        auto_discover_symmetries=False,
                        compress_rows_under_discovered_group=True,
                        verbose_cache=False,
                    )
                    _ = prep.global_keys
                    _ = prep.known_values
                    _ = prep.inflation_matrix

        stdout_output = stdout_stream.getvalue()
        self.assertIn("Core group order:", stdout_output)
        self.assertIn("Enumerating base marginals", stdout_output)
        self.assertIn("Base marginal type summary", stdout_output)
        self.assertNotIn("Discovered group order:", stdout_output)
        self.assertNotIn("Final marginal type summary after discovered symmetry compression", stdout_output)

    def test_ejm_entrypoint_materializes_inputs_before_solver_setup(self):
        stdout_stream = _NonTtyStringIO()
        fake_prep = _FakePrep()

        def fake_prep_factory(*args, **kwargs):
            return fake_prep

        with redirect_stdout(stdout_stream), mock.patch.object(
            ejm_5_test,
            "PrepLP",
            side_effect=fake_prep_factory,
        ):
            ejm_5_test.main(n=3)

        output = stdout_stream.getvalue()
        expected_markers = [
            "PrepLP initialized for n=3; materializing LP inputs before Mosek.",
            "Enumerating base marginals",
            "LP inputs ready for n=3: rows=7, cols=11. Starting Mosek setup.",
            "Starting pre-processing for the LP solver...",
            "Optimizer started.",
            "Feasible within tolerance for n=3: True. Incompatible fraction: 0",
        ]
        positions = [output.index(marker) for marker in expected_markers]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(fake_prep.solve_calls[0]["verbose"], 2)
        self.assertEqual(fake_prep.solve_calls[0]["optimizer"], "primal_simplex")


if __name__ == "__main__":
    unittest.main()
