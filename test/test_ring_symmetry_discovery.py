import unittest

import numpy as np
import sympy as sp

from inflation.applications.Final_algo_numba import PrepLP


class _UniformBinaryDistribution:
    @property
    def nof_outcomes(self):
        return 2

    def prob_event_loop(self, outcomes):
        return sp.Rational(1, 2) ** len(tuple(outcomes))

    def prob_event_line(self, outcomes):
        return sp.Rational(1, 2) ** len(tuple(outcomes))


class TestRingSymmetryDiscovery(unittest.TestCase):
    def test_lazy_materialization_and_symbolic_known_values(self):
        distribution = _UniformBinaryDistribution()
        prep = PrepLP(
            3,
            distribution,
            show_progress=False,
            auto_discover_symmetries=True,
            compress_rows_under_discovered_group=True,
            verbose_cache=False,
        )
        self.assertNotIn("variable_names", prep.__dict__)
        self.assertNotIn("inflation_matrix", prep.__dict__)
        self.assertFalse(hasattr(prep, "known_vars"))
        self.assertFalse(hasattr(prep, "known_vars_symbolic"))
        self.assertFalse(hasattr(prep, "blank_objective"))

        symbolic = prep.known_values_symbolic
        numeric = prep.known_values
        self.assertEqual(len(symbolic), len(numeric))
        for sym_val, num_val in zip(symbolic.tolist(), numeric.tolist()):
            self.assertAlmostEqual(float(sp.N(sym_val)), float(num_val), places=12)

    def test_row_orbit_metadata_consistency(self):
        distribution = _UniformBinaryDistribution()
        prep = PrepLP(
            3,
            distribution,
            show_progress=False,
            auto_discover_symmetries=True,
            compress_rows_under_discovered_group=True,
            verbose_cache=False,
        )

        self.assertGreaterEqual(
            len(prep.discovered_symmetries),
            len(prep.core_symmetries),
            "Discovered subgroup must contain the core symmetry group.",
        )
        self.assertEqual(
            int(prep.row_orbit_multiplicities.sum()),
            prep.base_nof_marginals,
            "Orbit multiplicities must account for all base rows.",
        )
        self.assertEqual(
            len(prep.row_orbit_members),
            prep.nof_marginals,
            "Each compressed row must have one orbit-members record.",
        )
        self.assertEqual(
            len(prep.row_orbit_average_labels),
            prep.nof_marginals,
            "Each compressed row must have one average orbit label.",
        )
        self.assertEqual(
            len(prep.row_orbit_member_labels),
            prep.nof_marginals,
            "Each compressed row must track its member labels.",
        )
        for members, multiplicity in zip(prep.row_orbit_members,
                                         prep.row_orbit_multiplicities):
            self.assertEqual(
                len(members),
                int(multiplicity),
                "Orbit size and multiplicity mismatch.",
            )

    def test_disable_row_compression_keeps_base_rows(self):
        distribution = _UniformBinaryDistribution()
        prep = PrepLP(
            3,
            distribution,
            show_progress=False,
            auto_discover_symmetries=True,
            compress_rows_under_discovered_group=False,
            verbose_cache=False,
        )

        self.assertEqual(
            prep.nof_marginals,
            prep.base_nof_marginals,
            "Disabling row compression should keep all base rows.",
        )
        self.assertTrue(
            np.array_equal(
                prep.row_orbit_multiplicities,
                np.ones(prep.base_nof_marginals, dtype=np.int64),
            ),
            "Without compression each row should have multiplicity 1.",
        )
