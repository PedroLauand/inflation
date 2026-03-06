import importlib.util
import unittest
from itertools import product
from pathlib import Path

import sympy as sp

from inflation.distributions import EJMDistribution, NSIPRDistribution, RGBDistribution


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestStandardizedDistributionAPIs(unittest.TestCase):
    def test_api_availability(self):
        for distribution in (EJMDistribution(), RGBDistribution(), NSIPRDistribution()):
            self.assertTrue(callable(distribution.prob_event_loop))
            self.assertTrue(callable(distribution.prob_event_line))
            self.assertIsInstance(distribution.nof_outcomes, int)

    def test_returns_symbolic_expressions(self):
        self.assertIsInstance(EJMDistribution().prob_event_loop([0]), sp.Expr)
        self.assertIsInstance(RGBDistribution().prob_event_loop([0]), sp.Expr)
        self.assertIsInstance(NSIPRDistribution().prob_event_loop([0]), sp.Expr)

    def test_coarsen_strict_partition_validation(self):
        with self.assertRaises(ValueError):
            EJMDistribution(coarsen=[[0, 1], [1], [2, 3]])
        with self.assertRaises(ValueError):
            EJMDistribution(coarsen=[[0], [1], [2]])
        with self.assertRaises(ValueError):
            EJMDistribution(coarsen=[[0], [1], [2], [4]])

    def test_ejm_loop_cyclic_invariance(self):
        ejm = EJMDistribution()
        ref = ejm.prob_event_loop([0, 1, 2, 3])
        cyc = ejm.prob_event_loop([1, 2, 3, 0])
        self.assertEqual(sp.simplify(ref - cyc), 0)

    def test_rgb_loop_cyclic_invariance(self):
        rgb = RGBDistribution()
        ref = rgb.prob_event_loop([0, 1, 2, 3])
        cyc = rgb.prob_event_loop([1, 2, 3, 0])
        self.assertEqual(sp.simplify(ref - cyc), 0)

    def test_nsi_pr_orderless_loop(self):
        nsi = NSIPRDistribution()
        self.assertEqual(
            sp.simplify(nsi.prob_event_loop([0, 1, 0, 0]) - nsi.prob_event_loop([0, 0, 0, 1])),
            0,
        )

    def test_nsi_pr_sign_convention(self):
        nsi = NSIPRDistribution()
        lhs = nsi.prob_event_loop([0, 1, 0, 0])
        rhs = nsi.prob_event_line([0, 0, 0]) - nsi.prob_event_loop([0, 0, 0, 0])
        self.assertEqual(sp.simplify(lhs - rhs), 0)

    def test_ejm_line_from_loop_consistency(self):
        ejm = EJMDistribution(coarsen=[[0], [1], [2, 3]])
        event = [0, 1, 2]
        lhs = ejm.prob_event_line(event)
        rhs = sum(ejm.prob_event_loop(event + [x]) for x in range(ejm.nof_outcomes))
        self.assertEqual(sp.simplify(lhs - rhs), 0)

    def test_rgb_line_from_loop_consistency(self):
        rgb = RGBDistribution(coarsen=[[0], [1], [2, 3]])
        event = [0, 1, 2]
        lhs = rgb.prob_event_line(event)
        rhs = sum(rgb.prob_event_loop(event + [x]) for x in range(rgb.nof_outcomes))
        self.assertEqual(sp.simplify(lhs - rhs), 0)

    def test_small_n_normalization(self):
        ejm = EJMDistribution()
        ejm_loop_sum = sum(ejm.prob_event_loop([a]) for a in range(ejm.nof_outcomes))
        ejm_line_sum = sum(ejm.prob_event_line([a]) for a in range(ejm.nof_outcomes))
        self.assertEqual(sp.simplify(ejm_loop_sum - 1), 0)
        self.assertEqual(sp.simplify(ejm_line_sum - 1), 0)

        rgb = RGBDistribution()
        rgb_loop_sum = sum(rgb.prob_event_loop([a]) for a in range(rgb.nof_outcomes))
        rgb_line_sum = sum(rgb.prob_event_line([a]) for a in range(rgb.nof_outcomes))
        self.assertEqual(sp.simplify(rgb_loop_sum - 1), 0)
        self.assertEqual(sp.simplify(rgb_line_sum - 1), 0)

        nsi = NSIPRDistribution()
        n = 3
        nsi_loop_sum = sum(nsi.prob_event_loop(evt) for evt in product((0, 1), repeat=n))
        nsi_line_sum = sum(nsi.prob_event_line(evt) for evt in product((0, 1), repeat=n))
        self.assertEqual(sp.simplify(nsi_loop_sum - 1), 0)
        self.assertEqual(sp.simplify(nsi_line_sum - 1), 0)

    def test_migration_smoke(self):
        final_algo_path = REPO_ROOT / "inflation" / "applications" / "Final_algo_numba.py"
        final_mod = _load_module_from_path("final_algo_numba_smoke", final_algo_path)
        value = final_mod.factorized_marginal_value([[1, 1, 1, 0, 0]], EJMDistribution())
        self.assertIsInstance(value, sp.Expr)

        postquantum_path = REPO_ROOT / "inflation" / "applications" / "postquantum_proof_2outcomes.py"
        postquantum_mod = _load_module_from_path("postquantum_smoke", postquantum_path)
        self.assertTrue(callable(postquantum_mod.main))


if __name__ == "__main__":
    unittest.main()
