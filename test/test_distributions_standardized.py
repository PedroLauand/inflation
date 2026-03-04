import importlib.util
import unittest
from itertools import product
from pathlib import Path

from inflation.distributions import ejm, nsi_pr, rgb


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
        for module in (ejm, rgb, nsi_pr):
            self.assertTrue(callable(module.prob_event_loop))
            self.assertTrue(callable(module.prob_event_line))

    def test_coarsen_strict_partition_validation(self):
        with self.assertRaises(ValueError):
            ejm.prob_event_loop([0], coarsen=[[0, 1], [1], [2, 3]])
        with self.assertRaises(ValueError):
            ejm.prob_event_loop([0], coarsen=[[0], [1], [2]])
        with self.assertRaises(ValueError):
            ejm.prob_event_loop([0], coarsen=[[0], [1], [2], [4]])

    def test_ejm_loop_cyclic_invariance(self):
        ref = ejm.prob_event_loop([0, 1, 2, 3])
        cyc = ejm.prob_event_loop([1, 2, 3, 0])
        self.assertAlmostEqual(ref, cyc, places=12)

    def test_rgb_loop_cyclic_invariance(self):
        ref = rgb.prob_event_loop([0, 1, 2, 3])
        cyc = rgb.prob_event_loop([1, 2, 3, 0])
        self.assertAlmostEqual(ref, cyc, places=12)

    def test_nsi_pr_orderless_loop(self):
        self.assertAlmostEqual(
            nsi_pr.prob_event_loop([0, 1, 0, 0]),
            nsi_pr.prob_event_loop([0, 0, 0, 1]),
            places=12,
        )

    def test_nsi_pr_sign_convention(self):
        lhs = nsi_pr.prob_event_loop([0, 1, 0, 0])
        rhs = nsi_pr.prob_event_line([0, 0, 0]) - nsi_pr.prob_event_loop([0, 0, 0, 0])
        self.assertAlmostEqual(lhs, rhs, places=12)

    def test_ejm_line_from_loop_consistency(self):
        coarsen = [[0], [1], [2, 3]]
        event = [0, 1, 2]
        lhs = ejm.prob_event_line(event, coarsen=coarsen)
        rhs = sum(ejm.prob_event_loop(event + [x], coarsen=coarsen) for x in range(3))
        self.assertAlmostEqual(lhs, rhs, places=12)

    def test_rgb_line_from_loop_consistency(self):
        coarsen = [[0], [1], [2, 3]]
        event = [0, 1, 2]
        lhs = rgb.prob_event_line(event, coarsen=coarsen)
        rhs = sum(rgb.prob_event_loop(event + [x], coarsen=coarsen) for x in range(3))
        self.assertAlmostEqual(lhs, rhs, places=12)

    def test_small_n_normalization(self):
        ejm_loop_sum = sum(ejm.prob_event_loop([a]) for a in range(4))
        ejm_line_sum = sum(ejm.prob_event_line([a]) for a in range(4))
        self.assertAlmostEqual(ejm_loop_sum, 1.0, places=12)
        self.assertAlmostEqual(ejm_line_sum, 1.0, places=12)

        rgb_loop_sum = sum(rgb.prob_event_loop([a]) for a in range(4))
        rgb_line_sum = sum(rgb.prob_event_line([a]) for a in range(4))
        self.assertAlmostEqual(rgb_loop_sum, 1.0, places=12)
        self.assertAlmostEqual(rgb_line_sum, 1.0, places=12)

        n = 3
        nsi_loop_sum = sum(nsi_pr.prob_event_loop(evt) for evt in product((0, 1), repeat=n))
        nsi_line_sum = sum(nsi_pr.prob_event_line(evt) for evt in product((0, 1), repeat=n))
        self.assertAlmostEqual(nsi_loop_sum, 1.0, places=12)
        self.assertAlmostEqual(nsi_line_sum, 1.0, places=12)

    def test_migration_smoke(self):
        final_algo_path = REPO_ROOT / "inflation" / "applications" / "Final_algo_numba.py"
        final_mod = _load_module_from_path("final_algo_numba_smoke", final_algo_path)
        value = final_mod.factorized_marginal_value([[1, 1, 1, 0, 0]], ejm.prob_event_loop)
        self.assertIsInstance(value, float)

        postquantum_path = REPO_ROOT / "inflation" / "applications" / "postquantum_proof_2outcomes.py"
        postquantum_mod = _load_module_from_path("postquantum_smoke", postquantum_path)
        self.assertTrue(callable(postquantum_mod.main))
        self.assertIsInstance(postquantum_mod.prob_event_loop([0, 0]), float)


if __name__ == "__main__":
    unittest.main()
