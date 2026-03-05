"""
symmetric_inflation_test.py
--------------------------------------------------------------------
Expectation-driven subclass of the canonical PrepLP pipeline.
--------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
from itertools import product
import sys

import numpy as np
from scipy.sparse import coo_array

# Ensure repo root is on sys.path so "import inflation" works when running directly.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from inflation import InflationProblem
from inflation.applications.Final_algo_numba import PrepLP, ring_problem


def inf_problem(inflation_level: int, nof_outcomes: int = 2) -> InflationProblem:
    """
    Build the ring inflation problem used in this scenario.
    """
    return ring_problem(inflation_level, nof_outcomes)


def loop_prob_from_correlators(
    outcomes: Tuple[int, ...] | List[int],
    E_line: dict[int, float],
    E_loop: dict[int, float],
) -> float:
    """
    Probability for a loop of length m = len(outcomes) with binary outcomes {0,1},
    using line correlators E_k (k < m) and loop correlator E^o_m.
    """
    m = len(outcomes)
    if m == 0:
        raise ValueError("outcomes must be non-empty")
    if any(o not in (0, 1) for o in outcomes):
        raise ValueError("this RHS demo assumes binary outcomes {0,1}")

    x = [1 if o == 0 else -1 for o in outcomes]
    total = 1.0

    for k in range(2, m):
        if k not in E_line:
            raise KeyError(f"missing E_line[{k}] for loop length {m}")
        seg_sum = 0.0
        for start in range(m):
            prod = 1
            for t in range(k):
                prod *= x[(start + t) % m]
            seg_sum += prod
        total += E_line[k] * seg_sum

    if m not in E_loop:
        raise KeyError(f"missing E_loop[{m}] for loop length {m}")
    prod_all = 1
    for val in x:
        prod_all *= val
    total += E_loop[m] * prod_all

    return total / (2**m)


def sanity_check_loop_distributions(
    E_line: dict[int, float],
    E_loop: dict[int, float],
    *,
    tol: float = 1e-9,
) -> None:
    """
    Print basic sanity checks for loop distributions implied by correlators.

    For each loop length m in E_loop, we check:
      - normalization: sum_{a in {0,1}^m} P(a) ~= 1
      - non-negativity: min P(a) >= -tol
    """
    lengths = sorted(E_loop.keys())
    if not lengths:
        print("No loop lengths found in E_loop; skipping sanity checks.")
        return

    print("\nLoop distribution sanity checks:")
    for m in lengths:
        outcomes_list = list(product((0, 1), repeat=m))
        probs = np.array(
            [loop_prob_from_correlators(out, E_line, E_loop) for out in outcomes_list],
            dtype=float,
        )
        total = float(probs.sum())
        min_p = float(probs.min(initial=np.inf))
        max_p = float(probs.max(initial=-np.inf))
        neg_count = int((probs < -tol).sum())
        over_count = int((probs > 1.0 + tol).sum())

        print(f"  m={m}: sum={total:.12g} (|sum-1|={abs(total-1.0):.3g})")
        print(
            f"       min={min_p:.12g}, max={max_p:.12g}, "
            f"negatives={neg_count}, >1={over_count}"
        )


class PrepLPExpectations(PrepLP):
    """
    Expectation-parameterized subclass of PrepLP.

    This class only injects an expectation-based event probability function.
    All LP outputs are inherited from PrepLP.
    """

    def __init__(
        self,
        prob: InflationProblem,
        E_line: dict[int, float],
        E_loop: dict[int, float],
        *,
        add_normalization: bool = True,
        show_progress: bool = True,
        marginal_filter_fn=None,
    ) -> None:
        self.E_line = E_line
        self.E_loop = E_loop
        self.add_normalization = add_normalization
        super().__init__(
            prob,
            event_prob_fn=self._event_prob_from_expectations,
            marginal_filter_fn=marginal_filter_fn,
            show_progress=show_progress,
        )

    def _event_prob_from_expectations(self, cycle_outcomes: Tuple[int, ...] | List[int]) -> float:
        """Loop-event probability callable built from E_line/E_loop correlators."""
        return loop_prob_from_correlators(cycle_outcomes, self.E_line, self.E_loop)


if __name__ == "__main__":
    from inflation.lp.lp_utils import solveLP_sparse

    n, outcomes = 3, 2
    include_outcome_relabel_symmetries = False
    cache_name = "lp_cache_expectations_n=3_no_outcome_relabelling.npz"

    prob = inf_problem(n, outcomes)
    if include_outcome_relabel_symmetries:
        prob.add_symmetries(prob._setting_specific_outcome_relabelling_symmetries)
    print("done with prob")

    sqrt2 = np.sqrt(2.0)
    E_line = {
        1: 0.0,
        2: sqrt2 - 1.0,
        # 3: 3.0 - 2.0 * sqrt2,
    }
    E_loop = {
        1: 0.0,
        2: 1.0,
        3: 0,
    }

    sanity_check_loop_distributions(E_line, E_loop, tol=1e-9)

    def _save_cache(
        path: Path,
        variable_names: np.ndarray,
        known_vars_coo_vec: coo_array,
        inflation_matrix: coo_array,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            inflation_matrix_columns_indices=inflation_matrix.col,
            inflation_matrix_row_indices=inflation_matrix.row,
            inflation_matrix_data_entries=inflation_matrix.data,
            variable_names=variable_names,
            known_values=known_vars_coo_vec.data,
            known_positions=known_vars_coo_vec.col,
        )

    def _load_cache(path: Path) -> Tuple[np.ndarray, coo_array, coo_array]:
        with np.load(path, allow_pickle=False) as z:
            row_idx = z["inflation_matrix_row_indices"]
            col_idx = z["inflation_matrix_columns_indices"]
            data = z["inflation_matrix_data_entries"]
            n_rows = int(np.max(row_idx)) + 1 if row_idx.size else 0
            n_cols = int(np.max(col_idx)) + 1 if col_idx.size else 0
            inflation_shape = (n_rows, n_cols)
            inflation_matrix = coo_array((data, (row_idx, col_idx)), shape=inflation_shape)
            known_positions = z["known_positions"]
            known_values = z["known_values"]
            if known_values.size == 0:
                known_vars_coo_vec = coo_array((1, inflation_shape[1]), dtype=float)
            else:
                known_rows = np.broadcast_to(
                    np.array(0, dtype=known_positions.dtype),
                    known_positions.shape,
                )
                known_vars_coo_vec = coo_array(
                    (known_values, (known_rows, known_positions)),
                    shape=(1, inflation_shape[1]),
                )
            variable_names = z["variable_names"]
            return variable_names, known_vars_coo_vec, inflation_matrix

    cache_dir = Path(__file__).resolve().parent / "cache"
    cache_path = cache_dir / cache_name

    if cache_path.exists():
        print(f"Loading cached LP constraints from {cache_path}")
        variable_names, known_vars_coo_vec, inflation_matrix = _load_cache(cache_path)
    else:
        prep = PrepLPExpectations(
            prob,
            E_line,
            E_loop,
            add_normalization=True,
            show_progress=True,
        )
        variable_names = prep.variable_names
        known_vars_coo_vec = prep.known_vars_coo_vec
        inflation_matrix = prep.inflation_matrix
        _save_cache(cache_path, variable_names, known_vars_coo_vec, inflation_matrix)

    nof_all_LP_vars = inflation_matrix.shape[1]
    solution = solveLP_sparse(
        objective=coo_array(([], ([], [])), shape=(1, nof_all_LP_vars)),
        known_vars=known_vars_coo_vec,
        equalities=inflation_matrix,
        default_non_negative=True,
        variables=variable_names,
        verbose=True,
    )
    print(solution["status"])
