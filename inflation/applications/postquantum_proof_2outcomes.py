from pathlib import Path
import sys

# Ensure repo root is on sys.path so "import inflation" works when running this file directly.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from inflation import InflationProblem, InflationSDP
from inflation.applications.ring_utils import build_off_diagonal_ring_problem
from inflation.distributions import NSIPRDistribution


def ring_problem(inflation_level: int, nof_outcomes: int = 2) -> InflationProblem:
    return build_off_diagonal_ring_problem(
        inflation_level,
        nof_outcomes,
        classical_sources="all",
    )


def main() -> None:
    distribution = NSIPRDistribution()
    prob = ring_problem(4, 2)
    # prob.add_symmetries(prob._setting_specific_outcome_relabelling_symmetries) # NOT SAFE TO USE ON NSI!

    ring_SDP = InflationSDP(prob, verbose=2, include_all_outcomes=False)
    ring_SDP.generate_relaxation("physical2")

    print("Quantum inflation **nonfanout/commuting** factors:")
    print(ring_SDP.physical_atoms)

    values = {
        "P[A^{1,2}=0]": float(distribution.prob_event_line([0])),
        "P[A^{1,2}=0 A^{2,1}=0]": float(distribution.prob_event_loop([0, 0])),
        "P[A^{1,2}=0 A^{2,3}=0]": float(distribution.prob_event_line([0, 0])),
        "P[A^{1,2}=0 A^{2,3}=0 A^{3,1}=0]": float(distribution.prob_event_loop([0, 0, 0])),
        "P[A^{1,2}=0 A^{2,3}=0 A^{3,4}=0]": float(distribution.prob_event_line([0, 0, 0])),
        "P[A^{1,2}=0 A^{2,3}=0 A^{3,4}=0 A^{4,1}=0]": float(distribution.prob_event_loop([0, 0, 0, 0])),
    }
    ring_SDP.update_values(values=values, only_specified_values=False)
    print(ring_SDP.known_moments)

    ring_SDP.solve(solve_dual=False)
    print(ring_SDP.status)


if __name__ == "__main__":
    main()
