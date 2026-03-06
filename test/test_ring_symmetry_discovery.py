import unittest

import numpy as np

from inflation.applications.Final_algo_numba import PrepLP, ring_problem


def _uniform_binary_loop_prob(outcomes):
    return 0.5 ** len(tuple(outcomes))


class TestRingSymmetryDiscovery(unittest.TestCase):
    def test_row_orbit_metadata_consistency(self):
        prob = ring_problem(3, 2)
        prep = PrepLP(
            prob,
            event_prob_fn=_uniform_binary_loop_prob,
            show_progress=False,
            auto_discover_symmetries=True,
            compress_rows_under_discovered_group=True,
            verbose_symmetry_discovery=False,
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
        prob = ring_problem(3, 2)
        prep = PrepLP(
            prob,
            event_prob_fn=_uniform_binary_loop_prob,
            show_progress=False,
            auto_discover_symmetries=True,
            compress_rows_under_discovered_group=False,
            verbose_symmetry_discovery=False,
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
