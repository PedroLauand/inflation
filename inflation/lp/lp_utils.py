"""
This file contains functions to interact with LP solvers.

@authors: Erica Han, Elie Wolfe
"""

import sys
from pathlib import Path
import mosek
import numpy as np

from typing import List, Dict, Union
from scipy.sparse import coo_array, issparse
from time import perf_counter
from gc import collect
from ..utils import partsextractor, expand_sparse_vec, vstack
from ..progress_utils import progress_stage
from array import array


def drop_zero_rows(coo_mat: coo_array):
    """Drops zero rows from a sparse matrix in place.

    Parameters
    ----------
    coo_mat : coo_array
        Sparse matrix to drop zero rows from.
    """
    if len(coo_mat.shape) == 1:
        coo_mat = coo_mat.reshape((1, coo_mat.shape[0]))
    nz_rows, new_row = np.unique(coo_mat.row, return_inverse=True)
    coo_mat.row = new_row
    coo_mat = coo_mat.reshape((len(nz_rows), coo_mat.shape[1]))
    return coo_mat


def canonical_order(coo_mat: coo_array):
    """Puts a sparse matrix in canonical order in place.

    Parameters
    ----------
    coo_mat : coo_array
        Sparse matrix to put in canonical order.
    """
    if len(coo_mat.shape) == 1:
        coo_mat = coo_mat.reshape((1, coo_mat.shape[0]))
    order = np.lexsort([coo_mat.col, coo_mat.row])
    coo_mat.row = np.asarray(coo_mat.row)[order]
    coo_mat.col = np.asarray(coo_mat.col)[order]
    coo_mat.data = np.asarray(coo_mat.data)[order]
    return coo_mat

def _signed_index_dtype(max_index: int) -> np.dtype:
    # Choose the smallest signed integer dtype that can represent max_index.
    if max_index <= np.iinfo(np.int8).max:
        return np.dtype(np.int8)
    if max_index <= np.iinfo(np.int16).max:
        return np.dtype(np.int16)
    if max_index <= np.iinfo(np.int32).max:
        return np.dtype(np.int32)
    return np.dtype(np.int64)

def _ensure_index_dtype(coo_mat: coo_array, idx_dtype: np.dtype) -> coo_array:
    if coo_mat.row.dtype != idx_dtype:
        coo_mat.row = coo_mat.row.astype(idx_dtype, copy=False)
    if coo_mat.col.dtype != idx_dtype:
        coo_mat.col = coo_mat.col.astype(idx_dtype, copy=False)
    return coo_mat


def _normalize_npz_path(path: Union[str, Path]) -> Path:
    archive_path = Path(path)
    if archive_path.suffix == "":
        return archive_path.with_suffix(".npz")
    if archive_path.suffix.lower() == ".npz":
        return archive_path
    raise ValueError("Archive path must omit the extension or end with '.npz'.")


def _serialize_solution_keys(keys) -> np.ndarray:
    """Preserve integer keys and fall back to strings for generic symbolic labels."""
    key_list = list(keys)
    raw = np.asarray(key_list)
    if raw.dtype.kind in {"u", "i"}:
        return raw
    return np.asarray(key_list, dtype=str)


def save_lp_solution(
    solution: Dict,
    path: Union[str, Path],
    *,
    compression: bool = True,
    allow_pickle: bool = True,
) -> Path:
    """Save the array-serializable subset of an LP solution to an NPZ archive."""
    archive_path = _normalize_npz_path(path)
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    x_keys = list(solution["x"].keys())
    variable_names = _serialize_solution_keys(x_keys)
    x_values = np.asarray([float(solution["x"][name]) for name in x_keys], dtype=float)
    constraint_names = np.asarray(solution.get("constraint_names", variable_names), dtype=str)
    sparse_certificate = canonical_order(solution["sparse_certificate"].tocoo(copy=False))
    nonzero_mask = ~np.isclose(sparse_certificate.data, 0.0)
    certificate_col = sparse_certificate.col[nonzero_mask].astype(np.int64, copy=False)
    certificate_data = sparse_certificate.data[nonzero_mask].astype(float, copy=False)
    term_code, term_desc = solution["term_code"]

    save_fn = np.savez_compressed if compression else np.savez
    save_fn(
        archive_path,
        status=np.asarray(str(solution["status"])),
        success=np.asarray(bool(solution["success"])),
        primal_value=np.asarray(float(solution["primal_value"])),
        dual_value=np.asarray(float(solution["dual_value"])),
        term_code=np.asarray(str(term_code)),
        term_desc=np.asarray(str(term_desc)),
        mode=np.asarray(str(solution.get("mode", ""))),
        solver_success=np.asarray(bool(solution.get("solver_success", solution["success"]))),
        known_mass=np.asarray(float(solution.get("known_mass", np.nan))),
        optimized_mass=np.asarray(float(solution.get("optimized_mass", np.nan))),
        incompatible_fraction=np.asarray(float(solution.get("incompatible_fraction", np.nan))),
        generalized_robustness=np.asarray(float(solution.get("generalized_robustness", np.nan))),
        variable_names=variable_names,
        x_values=x_values,
        constraint_names=constraint_names,
        certificate_col=certificate_col,
        certificate_data=certificate_data,
    )
    _ = allow_pickle
    return archive_path


def read_lp_solution(path: Union[str, Path], *, allow_pickle: bool = True) -> Dict:
    """Read an LP solution archive and reconstruct the solveLP_sparse() solution dictionary."""
    archive_path = _normalize_npz_path(path)
    with np.load(archive_path, allow_pickle=allow_pickle) as z:
        variable_names = np.asarray(z["variable_names"])
        x_values = np.asarray(z["x_values"], dtype=float)
        constraint_names = (
            np.asarray(z["constraint_names"], dtype=str)
            if "constraint_names" in z.files
            else variable_names
        )
        certificate_col = np.asarray(z["certificate_col"], dtype=np.int64)
        certificate_data = np.asarray(z["certificate_data"], dtype=float)
        cert_row = np.zeros(certificate_col.shape[0], dtype=np.int32)
        sparse_certificate = coo_array(
            (certificate_data, (cert_row, certificate_col)),
            shape=(1, constraint_names.size),
        )
        dual_certificate = dict(
            zip(constraint_names[certificate_col].tolist(), certificate_data.tolist())
        )
        return {
            "primal_value": float(np.asarray(z["primal_value"]).item()),
            "dual_value": float(np.asarray(z["dual_value"]).item()),
            "status": str(np.asarray(z["status"]).item()),
            "success": bool(np.asarray(z["success"]).item()),
            "solver_success": bool(
                np.asarray(z["solver_success"]).item() if "solver_success" in z.files else np.asarray(z["success"]).item()
            ),
            "mode": str(np.asarray(z["mode"]).item()) if "mode" in z.files else "",
            "known_mass": float(np.asarray(z["known_mass"]).item()) if "known_mass" in z.files else np.nan,
            "optimized_mass": float(np.asarray(z["optimized_mass"]).item()) if "optimized_mass" in z.files else np.nan,
            "incompatible_fraction": (
                float(np.asarray(z["incompatible_fraction"]).item())
                if "incompatible_fraction" in z.files
                else np.nan
            ),
            "generalized_robustness": (
                float(np.asarray(z["generalized_robustness"]).item())
                if "generalized_robustness" in z.files
                else np.nan
            ),
            "dual_certificate": dual_certificate,
            "sparse_certificate": sparse_certificate,
            "constraint_names": constraint_names,
            "x": dict(zip(variable_names.tolist(), x_values.tolist())),
            "term_code": (
                str(np.asarray(z["term_code"]).item()),
                str(np.asarray(z["term_desc"]).item()),
            ),
        }


def load_lp_solution(path: Union[str, Path], *, allow_pickle: bool = True) -> Dict:
    """Compatibility alias for read_lp_solution()."""
    return read_lp_solution(path, allow_pickle=allow_pickle)

def solveLP(objective: Union[coo_array, Dict] = None,
            known_vars: Union[coo_array, Dict] = None,
            semiknown_vars: Dict = None,
            inequalities: Union[coo_array, List[Dict]] = None,
            equalities: Union[coo_array, List[Dict]] = None,
            variables: Union[List, np.ndarray] = None,
            **kwargs
            ) -> Dict:
    """Wrapper function that converts all dictionaries to sparse matrices to
    pass to the solver.

    Parameters
    ----------
    objective : Union[coo_array, Dict], optional
        Objective function
    known_vars : Union[coo_array, Dict], optional
        Known values of the monomials
    semiknown_vars : Dict, optional
        Semiknown variables
    inequalities : Union[coo_array, List[Dict]], optional
        Inequality constraints
    equalities : Union[coo_array, List[Dict]], optional
        Equality constraints
    variables : List
        Monomials by name in same order as column indices of all other solver
        arguments

    Returns
    -------
    dict
        Primal objective value, dual objective value, problem status, success
        status, dual certificate (as dictionary and sparse matrix), x values,
        and response code.
    """
    # Save solver arguments, unpacking kwargs
    solver_args = locals()
    del solver_args['kwargs']
    solver_args.update(kwargs)

    # Check type for arguments related to the problem
    problem_args = ("objective",
                    "known_vars",
                    "semiknown_vars",
                    "inequalities",
                    "equalities",
                    "lower_bounds",
                    "upper_bounds")
    used_args = {k: v for k, v in solver_args.items()
                 if k in problem_args and v is not None}
    if all(issparse(arg) for arg in used_args.values()):
        assert variables is not None, "Variables must be declared when all " \
                                      "arguments are in sparse matrix form."
    elif all(isinstance(arg, (dict, list)) for arg in used_args.values()):
        if variables is None:
            # Infer variables
            variables = set()
            if objective:
                variables.update(objective)
            if inequalities:
                for ineq in inequalities:
                    variables.update(ineq)
            if equalities:
                for eq in equalities:
                    variables.update(eq)
            if semiknown_vars:
                for x, (c, x2) in semiknown_vars.items():
                    variables.update([x, x2])
            if known_vars:
                variables.update(known_vars)
            variables = sorted(variables)
            solver_args["variables"] = variables
        solver_args.update(convert_dicts(**solver_args))
    else:
        assert variables is not None, "Variables must be declared when " \
                                      "arguments are of mixed form."
        solver_args.update(convert_dicts(**solver_args))
    solver_args.pop("semiknown_vars", None)
    return solveLP_sparse(**solver_args)


blank_coo_array = coo_array((0, 0), dtype=np.int8)
def solveLP_sparse(objective: coo_array = blank_coo_array,
                   known_vars: coo_array = blank_coo_array,
                   inequalities: coo_array = blank_coo_array,
                   equalities: coo_array = blank_coo_array,
                   lower_bounds: coo_array = blank_coo_array,
                   upper_bounds: coo_array = blank_coo_array,
                   solve_dual: bool = False,
                   default_non_negative: bool = True,
                   relax_known_vars: bool = False,
                   relax_inequalities: bool = False,
                   verbose: int = 0,
                   solverparameters: Dict = None,
                   variables: Union[List, np.ndarray] = None
                   ) -> Dict:
    """Internal function to solve an LP with the Mosek Optimizer API using
    sparse matrices. Columns of each matrix correspond to a fixed order of
    variables in the LP.

    Parameters
    ----------
    objective : coo_array, optional
        Objective function with coefficients as matrix entries.
    known_vars : coo_array, optional
        Known values of the monomials with values as matrix entries.
    inequalities : coo_array, optional
        Inequality constraints in matrix form.
    equalities : coo_array, optional
        Equality constraints in matrix form.
    lower_bounds : coo_array, optional
        Lower bounds of variables with bounds as matrix entries.
    upper_bounds : coo_array, optional
        Upper bounds of variables with bounds as matrix entries.
    solve_dual : bool, optional
        Whether to solve the dual (``True``) or primal (``False``) formulation.
        By default, ``False``.
    default_non_negative : bool, optional
        Whether to set default primal variables as non-negative. By default,
        ``True``.
    relax_known_vars : bool, optional
        Do feasibility as optimization where each known value equality becomes
        two relaxed inequality constraints. E.g., P(A) = 0.7 becomes P(A) +
        lambda >= 0.7 and P(A) - lambda <= 0.7, where lambda is a slack
        variable. By default, ``False``.
    relax_inequalities : bool, optional
        Do feasibility as optimization where each inequality is relaxed by the
        non-negative slack variable lambda. By default, ``False``.
    verbose : int, optional
        Verbosity. Higher means more messages. By default, 0.
    solverparameters : dict, optional
        Parameters to pass to the MOSEK solver. For example, to control whether
        presolve is applied before optimization, set
        ``mosek.iparam.presolve_use`` to ``mosek.presolvemode.on`` or
        ``mosek.presolvemode.off``. Or, control which optimizer is used by
        setting an optimizer type to ``mosek.iparam.optimizer``. See `MOSEK's
        documentation`_ for more details.
    variables : list
        Monomials by name in same order as column indices of all other solver
        arguments

    Returns
    -------
    dict
        Primal objective value, dual objective value, problem status, success
        status, dual certificate (as dictionary and sparse matrix), x values,
        and response code.
    """

    inequalities=drop_zero_rows(inequalities)
    equalities=drop_zero_rows(equalities)
    known_vars=canonical_order(known_vars)
    upper_bounds=canonical_order(upper_bounds)
    lower_bounds=canonical_order(lower_bounds)
    objective=canonical_order(objective)

    if verbose > 1:
        t0 = perf_counter()
        t_total = perf_counter()
        print("Starting pre-processing for the LP solver...")


    if relax_known_vars or relax_inequalities:
        default_non_negative = False

    with mosek.Env() as env:
        with mosek.Task(env) as task:
            # Set parameters for the solver depending on value type
            if solverparameters:
                for param, val in solverparameters.items():
                    if isinstance(val, int):
                        task.putintparam(param, val)
                    elif isinstance(val, float):
                        task.putdouparam(param, val)
                    elif isinstance(val, str):
                        task.putstrparam(param, val)
            task.putintparam(mosek.iparam.sim_reformulation,
                             mosek.simreform.aggressive)
            task.putintparam(mosek.iparam.sim_switch_optimizer, mosek.onoffkey.on)
            task.putintparam(mosek.iparam.optimizer, mosek.optimizertype.primal_simplex)
            if solve_dual:
                task.putintparam(mosek.iparam.sim_solve_form,
                                 mosek.solveform.dual)
            else:
                task.putintparam(mosek.iparam.sim_solve_form,
                                 mosek.solveform.primal)
            if verbose > 0:
                # Attach a log stream printer to the task
                task.set_Stream(mosek.streamtype.log, streamprinter)
                task.putintparam(mosek.iparam.log_include_summary,
                                 mosek.onoffkey.on)
                task.putintparam(mosek.iparam.log_storage, 1)
            if verbose < 2:
                task.putintparam(mosek.iparam.log_sim, 0)
                task.putintparam(mosek.iparam.log_intpnt, 0)

            # Initialize constraint matrix
            constraints = vstack((inequalities, equalities))

            nof_primal_inequalities = inequalities.shape[0]
            nof_primal_equalities = equalities.shape[0]

            (nof_primal_constraints, nof_primal_variables) = constraints.shape
            nof_known_vars = known_vars.nnz
            max_size = max(nof_primal_constraints, nof_primal_variables, nof_known_vars * 2, 1)
            idx_dtype = _signed_index_dtype(max_size - 1)

            # Initialize b vector (RHS of constraints)
            b = [0] * nof_primal_constraints

            if relax_inequalities:
                constraints = _ensure_index_dtype(constraints, idx_dtype)
                # Add slack variable lambda to each inequality
                cons_row = np.hstack(
                    (constraints.row, np.arange(nof_primal_inequalities, dtype=idx_dtype)))
                cons_col = np.hstack(
                    (constraints.col, np.full(nof_primal_inequalities,
                                              nof_primal_variables, dtype=idx_dtype)))
                cons_data = np.hstack(
                    (constraints.data, np.repeat(1, nof_primal_inequalities)))
                constraints = coo_array((cons_data, (cons_row, cons_col)),
                                         shape=(nof_primal_constraints,
                                                nof_primal_variables + 1))

            if relax_known_vars:
                known_vars = _ensure_index_dtype(known_vars, idx_dtype)
                # Each known value is replaced by two inequalities with slacks
                kv_row = np.tile(
                    np.arange(nof_known_vars * 2, dtype=idx_dtype),
                    2)
                kv_col = np.hstack((
                    np.tile(known_vars.col, 2),
                    np.full(nof_known_vars * 2, nof_primal_variables, dtype=idx_dtype)
                ))
                kv_data = np.hstack((
                    np.broadcast_to(1, nof_known_vars * 3),
                    np.broadcast_to(-1, nof_known_vars)
                ))

                kv_matrix = coo_array((kv_data, (kv_row, kv_col)),
                                       shape=(nof_known_vars * 2,
                                              nof_primal_variables + 1))
                canonical_order(kv_matrix)
                constraints.resize(*(nof_primal_constraints,
                                     nof_primal_variables + 1))
                b = np.hstack((b, np.tile(known_vars.data, 2)))
            else:
                # Add known values as equalities to the constraint matrix
                kv_matrix = expand_sparse_vec(known_vars, idx_dtype=idx_dtype)
                b = np.hstack((b, known_vars.data))
                nof_primal_equalities += nof_known_vars

            constraints = vstack((constraints, kv_matrix))
            (nof_primal_constraints, nof_primal_variables) = constraints.shape
            if objective.shape[-1] == 0:
                objective = coo_array((1, nof_primal_variables), dtype=np.int8)

            if verbose > 0:
                print(f"Size of constraint matrix: {constraints.shape}")


            constraints = _ensure_index_dtype(constraints, idx_dtype)
            if verbose > 1:
                print("Proceeding with primal initialization...")
                with progress_stage(
                    "Converting constraint matrix to CSC format...",
                    end_message=lambda elapsed: f"CSC conversion complete in {elapsed:.2f}s",
                ):
                    matrix = constraints.tocsc(copy=False)
            else:
                matrix = constraints.tocsc(copy=False)

            if relax_known_vars or relax_inequalities:
                # Maximize lambda
                # (If maximum slack is still negative, then the unrelaxed
                # LP would be infeasible, whereas positive slack solution
                # implies LP solution strictly interior in the polytope.)
                objective_vector = np.zeros(nof_primal_variables)
                objective_vector[-1] = -1
            else:
                objective_vector = objective.toarray().ravel()

            # Set the objective sense
            task.putobjsense(mosek.objsense.maximize)

            # Add all the problem data to the task
            numcon = nof_primal_constraints
            numvar = nof_primal_variables
            if verbose > 1:
                with progress_stage(
                    "Starting task.inputdata in Mosek...",
                    end_message=lambda elapsed: f"Mosek input data loaded in {elapsed:.2f}s",
                ):
                    int32_max = np.iinfo(np.int32).max
                    indptr_max = int(matrix.indptr[-1]) if matrix.indptr.size else 0
                    indices_max = int(matrix.indices.max()) if matrix.indices.size else 0
                    use_64 = (
                        numcon > int32_max
                        or numvar > int32_max
                        or indptr_max > int32_max
                        or indices_max > int32_max
                    )
                    int_dtype = np.int64 if use_64 else np.int32

                    # Set bound keys and values for constraints
                    # Ax >= b where b is 0
                    bkc = np.hstack((np.broadcast_to(mosek.boundkey.lo, nof_primal_inequalities),
                                     np.broadcast_to(mosek.boundkey.fx,
                                                     nof_primal_equalities)
                                     )).astype(int_dtype, copy=False)
                    if relax_known_vars:
                        bkc = np.hstack((bkc,
                                         np.repeat([mosek.boundkey.lo, mosek.boundkey.up],
                                                         nof_known_vars)
                                         )).astype(int_dtype, copy=False)
                    blc = buc = b

                    ub_col = upper_bounds.col
                    ub_data = np.zeros(nof_primal_variables)
                    ub_data[ub_col] = upper_bounds.data
                    lb_col = lower_bounds.col
                    lb_data = np.zeros(nof_primal_variables)
                    lb_data[lb_col] = lower_bounds.data

                    # Set bound keys and bound values for variables
                    blx = np.zeros(nof_primal_variables)
                    bux = np.zeros(nof_primal_variables)
                    ub_col = np.asarray(upper_bounds.col)
                    lb_col = np.asarray(lower_bounds.col)
                    lb_data = lower_bounds.data
                    if default_non_negative:
                        bkx = np.repeat(mosek.boundkey.lo, nof_primal_variables).astype(int_dtype, copy=False)
                        bkx[ub_col] = mosek.boundkey.ra
                    else:
                        bkx = np.repeat(mosek.boundkey.fr, nof_primal_variables).astype(int_dtype, copy=False)
                        bkx[np.setdiff1d(lb_col, ub_col)] = mosek.boundkey.lo
                        bkx[np.setdiff1d(ub_col, lb_col)] = mosek.boundkey.up
                        bkx[np.intersect1d(ub_col, lb_col)] = mosek.boundkey.ra
                    blx[lb_col] = lb_data
                    bux[ub_col] = upper_bounds.data

                    if relax_known_vars or relax_inequalities:
                        bkx[-1] = mosek.boundkey.fr
                    if use_64:
                        aptrb = array("q", matrix.indptr[:-1].astype(np.int64, copy=False))
                        aptre = array("q", matrix.indptr[1:].astype(np.int64, copy=False))
                        asub = array("q", matrix.indices.astype(np.int64, copy=False))
                    else:
                        aptrb = array("i", matrix.indptr[:-1].astype(np.int32, copy=False))
                        aptre = array("i", matrix.indptr[1:].astype(np.int32, copy=False))
                        asub = array("i", matrix.indices.astype(np.int32, copy=False))
                    task.inputdata(# maxnumcon=
                                   numcon,
                                   # maxnumvar=
                                   numvar,
                                   # c=
                                   array("d", objective_vector),
                                   # cfix=
                                   0,
                                   # aptrb=
                                   aptrb,
                                   # aptre=
                                   aptre,
                                   # asub=
                                   asub,
                                   # aval=
                                   matrix.data,
                                   bkc,
                                   blc,
                                   buc,
                                   bkx,
                                   blx,
                                   bux)
            else:
                int32_max = np.iinfo(np.int32).max
                indptr_max = int(matrix.indptr[-1]) if matrix.indptr.size else 0
                indices_max = int(matrix.indices.max()) if matrix.indices.size else 0
                use_64 = (
                    numcon > int32_max
                    or numvar > int32_max
                    or indptr_max > int32_max
                    or indices_max > int32_max
                )
                int_dtype = np.int64 if use_64 else np.int32

                # Set bound keys and values for constraints
                # Ax >= b where b is 0
                bkc = np.hstack((np.broadcast_to(mosek.boundkey.lo, nof_primal_inequalities),
                                 np.broadcast_to(mosek.boundkey.fx,
                                                 nof_primal_equalities)
                                 )).astype(int_dtype, copy=False)
                if relax_known_vars:
                    bkc = np.hstack((bkc,
                                     np.repeat([mosek.boundkey.lo, mosek.boundkey.up],
                                                     nof_known_vars)
                                     )).astype(int_dtype, copy=False)
                blc = buc = b

                ub_col = upper_bounds.col
                ub_data = np.zeros(nof_primal_variables)
                ub_data[ub_col] = upper_bounds.data
                lb_col = lower_bounds.col
                lb_data = np.zeros(nof_primal_variables)
                lb_data[lb_col] = lower_bounds.data

                # Set bound keys and bound values for variables
                blx = np.zeros(nof_primal_variables)
                bux = np.zeros(nof_primal_variables)
                ub_col = np.asarray(upper_bounds.col)
                lb_col = np.asarray(lower_bounds.col)
                lb_data = lower_bounds.data
                if default_non_negative:
                    bkx = np.repeat(mosek.boundkey.lo, nof_primal_variables).astype(int_dtype, copy=False)
                    bkx[ub_col] = mosek.boundkey.ra
                else:
                    bkx = np.repeat(mosek.boundkey.fr, nof_primal_variables).astype(int_dtype, copy=False)
                    bkx[np.setdiff1d(lb_col, ub_col)] = mosek.boundkey.lo
                    bkx[np.setdiff1d(ub_col, lb_col)] = mosek.boundkey.up
                    bkx[np.intersect1d(ub_col, lb_col)] = mosek.boundkey.ra
                blx[lb_col] = lb_data
                bux[ub_col] = upper_bounds.data

                if relax_known_vars or relax_inequalities:
                    bkx[-1] = mosek.boundkey.fr
                if use_64:
                    aptrb = array("q", matrix.indptr[:-1].astype(np.int64, copy=False))
                    aptre = array("q", matrix.indptr[1:].astype(np.int64, copy=False))
                    asub = array("q", matrix.indices.astype(np.int64, copy=False))
                else:
                    aptrb = array("i", matrix.indptr[:-1].astype(np.int32, copy=False))
                    aptre = array("i", matrix.indptr[1:].astype(np.int32, copy=False))
                    asub = array("i", matrix.indices.astype(np.int32, copy=False))
                task.inputdata(# maxnumcon=
                               numcon,
                               # maxnumvar=
                               numvar,
                               # c=
                               array("d", objective_vector),
                               # cfix=
                               0,
                               # aptrb=
                               aptrb,
                               # aptre=
                               aptre,
                               # asub=
                               asub,
                               # aval=
                               matrix.data,
                               bkc,
                               blc,
                               buc,
                               bkx,
                               blx,
                               bux)
            collect()
            if verbose > 1:
                print("Pre-processing took",
                      format(perf_counter() - t0, ".4f"), "seconds.\n")
                t0 = perf_counter()

            if verbose > 2:
                with progress_stage(
                    "Writing problem to debug_lp.ptf...",
                    end_message=lambda elapsed: f"Wrote debug_lp.ptf in {elapsed:.2f}s",
                ):
                    task.writedata("debug_lp.ptf")

            # Solve the problem
            if verbose > 0:
                print("\nSolving the problem...\n")
            trmcode = task.optimize()
            if verbose > 1:
                print("Solving took", format(perf_counter() - t0, ".4f"),
                      "seconds.")
            basic = mosek.soltype.bas
            (problemsta,
             solutionsta,
             skc,
             skx,
             skn,
             xc,
             xx,
             yy,
             slc,
             suc,
             slx,
             sux,
             snx) = task.getsolution(basic)

            # Get objective values, solutions x, dual values y
            xx = np.asarray(xx, dtype=object)
            yy = np.asarray(yy, dtype=object)

            primal = task.getprimalobj(basic)
            dual = task.getdualobj(basic)
            x_values = dict(zip(variables, xx))
            y_values = yy

            if solutionsta == mosek.solsta.optimal:
                success = True
            else:
                success = False
            status_str = solutionsta.__repr__()
            term_tuple = mosek.Env.getcodedesc(trmcode)
            if solutionsta == mosek.solsta.unknown and verbose > 0:
                print("The solution status is unknown.")
                print(f"   Termination code: {term_tuple}")

            # Extract the certificate as a sparse matrix: y.b - c.x <= 0
            if relax_known_vars:
                y_values = y_values[nof_primal_constraints - nof_known_vars*2:]
            else:
                y_values = y_values[nof_primal_constraints - nof_known_vars:]
            cert_row = [0] * nof_primal_variables
            cert_col = [*range(nof_primal_variables)]
            cert_data = np.zeros((nof_primal_variables,))
            obj_data = objective.toarray().ravel()
            objective_unknown_cols = np.setdiff1d(objective.col, known_vars.col)
            cert_data[objective_unknown_cols] = -obj_data[objective_unknown_cols]
            cert_data[known_vars.col] = y_values[:nof_known_vars] # Assumes known values coded as equalities
            if relax_known_vars:
                cert_data[known_vars.col] += y_values[nof_known_vars:(2*nof_known_vars)]
            sparse_certificate = coo_array((cert_data, (cert_row, cert_col)),
                                            shape=(1, nof_primal_variables))

            # Certificate as a dictionary
            certificate = dict(zip(variables,
                                   sparse_certificate.toarray().ravel().tolist()))

            # Clean entries with coefficient zero
            for x in list(certificate):
                if np.isclose(certificate[x], 0):
                    del certificate[x]

            if verbose > 1:
                print("\nTotal execution time:",
                      format(perf_counter() - t_total, ".4f"), "seconds.")

            return {
                "primal_value": primal,
                "dual_value": dual,
                "status": status_str,
                "success": success,
                "dual_certificate": certificate,
                "sparse_certificate": sparse_certificate,
                "x": x_values,
                "term_code": term_tuple
            }


def streamprinter(text: str) -> None:
    """A stream printer to get output from Mosek.
    
    Parameters
    ----------
    text : str
        Text to print.
    """
    sys.stdout.write(text)
    sys.stdout.flush()


###########################################################################
# ROUTINES RELATED TO SPARSE MATRIX CONVERSION                            #
###########################################################################


def to_sparse(argument: Union[Dict, List[Dict]],
              variables: List,
              idx_dtype: np.dtype | None = None) -> coo_array:
    """Convert a solver argument to a sparse matrix to pass to the solver.
    
    Parameters
    ----------
    argument : Union[Dict, List[Dict]]
        Solver argument to convert to sparse matrix.
    variables : List
        Monomials by name in same order as column indices of all other solver
        arguments
    
    Returns
    -------
    coo_array
        Sparse matrix representation of the solver argument.
    """
    if type(argument) == dict:
        var_to_idx = {x: i for i, x in enumerate(variables)}
        data = list(argument.values())
        keys = list(argument.keys())
        col = np.asarray(partsextractor(var_to_idx, keys),
                         dtype=idx_dtype if idx_dtype is not None else None)
        row = np.zeros(len(col), dtype=idx_dtype if idx_dtype is not None else int)
        return coo_array((data, (row, col)), shape=(1, len(variables)))
    else:
        # Argument is a list of constraints
        row = []
        for i, cons in enumerate(argument):
            row.extend([i] * len(cons))
        cols = [to_sparse(cons, variables, idx_dtype=idx_dtype).col for cons in argument]
        col = [c for vec_col in cols for c in vec_col]
        data = [to_sparse(cons, variables, idx_dtype=idx_dtype).data for cons in argument]
        data = [d for vec_data in data for d in vec_data]
        row = np.asarray(row, dtype=idx_dtype if idx_dtype is not None else None)
        col = np.asarray(col, dtype=idx_dtype if idx_dtype is not None else None)
        return coo_array((data, (row, col)),
                          shape=(len(argument), len(variables)))


def convert_dicts(semiknown_vars: Dict = None,
                  variables: List = None,
                  **kwargs) -> Dict:
    """Convert any dictionaries to sparse matrices to send to the solver.
    Semiknown variables are absorbed into the equality constraints.
    
    Parameters
    ----------
    semiknown_vars : Dict, optional
        Semiknown variables
    variables : List
        Monomials by name in same order as column indices of all other solver
        arguments
    
    Returns
    -------
    Dict
        Converted arguments (Unchanged arguments are not returned)
    """

    assert variables is not None, "Variables must be passed."

    args = locals()
    del args['kwargs']
    args.update(kwargs)

    # Arguments converted from dictionaries to sparse matrices
    idx_dtype = _signed_index_dtype(max(len(variables) - 1, 0))
    sparse_args = {k: to_sparse(arg, variables, idx_dtype=idx_dtype) for k, arg in args.items()
                   if isinstance(arg, (dict, list)) and k != "semiknown_vars"
                   and k != "solverparameters" and k != "variables"}

    # Add semiknown variables to equality constraints
    if semiknown_vars:
        nof_semiknown = len(semiknown_vars)
        nof_variables = len(variables)
        var_to_idx = {x: i for i, x in enumerate(variables)}
        row = np.repeat(np.arange(nof_semiknown, dtype=idx_dtype), 2)
        col = [(var_to_idx[x], var_to_idx[x2])
               for x, (c, x2) in semiknown_vars.items()]
        col = np.asarray(list(sum(col, ())), dtype=idx_dtype)
        data = [(1, -c) for x, (c, x2) in semiknown_vars.items()]
        data = list(sum(data, ()))
        semiknown_mat = coo_array((data, (row, col)),
                                   shape=(nof_semiknown, nof_variables))
        if "equalities" in sparse_args:
            sparse_args["equalities"] = vstack(
                (sparse_args["equalities"], semiknown_mat))
        else:
            sparse_args["equalities"] = semiknown_mat
    return sparse_args
