import io
import unittest
from contextlib import redirect_stderr, redirect_stdout

import sympy as sp

from inflation.applications.Final_algo_numba import PrepLP
from inflation.progress_utils import make_tqdm


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


class TestProgressUtils(unittest.TestCase):
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

    def test_prep_lp_progress_uses_stdout_and_hides_symmetry_discovery_output(self):
        stdout_stream = _NonTtyStringIO()
        stderr_stream = _NonTtyStringIO()

        with redirect_stdout(stdout_stream), redirect_stderr(stderr_stream):
            prep = PrepLP(
                3,
                _UniformBinaryDistribution(),
                show_progress=True,
                auto_discover_symmetries=True,
                compress_rows_under_discovered_group=True,
                verbose_symmetry_discovery=True,
                verbose_cache=False,
            )
            _ = prep.variable_names

        stdout_output = stdout_stream.getvalue()
        stderr_output = stderr_stream.getvalue()

        self.assertIn("Canonicalizing marginals", stdout_output)
        self.assertIn("Computing marginal values...", stdout_output)
        self.assertIn("Finding global extensions...", stdout_output)
        self.assertNotIn("Discovering ring stabilizing symmetries", stdout_output)
        self.assertNotIn("Stabilizer subgroup summary", stdout_output)
        self.assertNotIn("Canonicalizing marginals", stderr_output)
        self.assertNotIn("Computing marginal values...", stderr_output)
        self.assertNotIn("Finding global extensions...", stderr_output)
        self.assertNotIn("\r", stdout_output)
        self.assertNotIn("\x1b[A", stdout_output)


if __name__ == "__main__":
    unittest.main()
