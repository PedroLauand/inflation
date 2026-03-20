import unittest

from inflation import InflationProblem
from inflation.applications.ring_utils import ring_problem


class TestInflationProblemDefaults(unittest.TestCase):
    def test_generic_problem_defaults_to_standard_multi_source_behavior(self):
        problem = InflationProblem(
            {"Lambda": ["A"]},
            outcomes_per_party=[3],
            settings_per_party=[2],
            inflation_level_per_source=[1],
        )

        self.assertFalse(problem.really_just_one_source)
        self.assertEqual(problem._lexrepr_to_names[0], "A_0=0")

    def test_ring_builder_explicitly_uses_one_source_mode(self):
        problem = ring_problem(3, 2, classical_sources="all")

        self.assertTrue(problem.really_just_one_source)


if __name__ == "__main__":
    unittest.main()
