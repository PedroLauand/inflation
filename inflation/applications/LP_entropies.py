import itertools
import pulp

def build_entropy_feasibility_lp_general(N, SA, SB, SAB, solver=None, verbose=False):
    """
    Build and solve the entropic feasibility LP for parties:
        A0,...,AN, B0,...,BN
    with given SA = S(A), SB = S(B), SAB = S(AB),
    under:
      - non-negativity S(X) >= 0,
      - strong subadditivity (SSA),
      - weak monotonicity (WM),
      - conditional entropy non-negativity: S(X|Y) >= 0, S(Y|X) >= 0
        for all disjoint nonempty X, Y.

    N:   number of copies per side minus 1 (A0..AN gives N+1 copies).
    SA:  entropy of each A_i
    SB:  entropy of each B_j
    SAB: entropy of each pair A_i B_j

    Returns:
        (status_string, prob, s_vars, subset_to_name)
        - status_string: 'Optimal' → feasible, else infeasible/other.
        - prob: PuLP problem instance
        - s_vars: dict mapping bitmask -> PuLP variable
        - subset_to_name: function to get a human-readable subset name
    """

    # Number of parties: indices 0..N for A0..AN, N+1..2N+1 for B0..BN
    M = 2 * (N + 1)

    # Party labels
    party_labels = []
    for i in range(N + 1):
        party_labels.append(f"A{i}")
    for j in range(N + 1):
        party_labels.append(f"B{j}")

    def subset_name(mask: int) -> str:
        """Return a string like 'A0A1B0' for a subset bitmask."""
        labels = []
        for k in range(M):
            if mask & (1 << k):
                labels.append(party_labels[k])
        return "".join(labels) if labels else "∅"

    # All nonempty subsets as bitmasks
    subsets = [m for m in range(1, 1 << M)]

    # Create LP
    prob = pulp.LpProblem("EntropyFeasibilityGeneral", pulp.LpMinimize)

    # Entropy variables S(X) >= 0 for all nonempty X
    s_vars = {
        m: pulp.LpVariable(f"S_{subset_name(m)}", lowBound=0.0)
        for m in subsets
    }

    # === Boundary / symmetry constraints ===

    # S(A_i) = SA for all i
    for i in range(N + 1):
        mask = 1 << i
        cname = f"S_A{i}_eq_SA"
        prob += (s_vars[mask] == SA), cname

    # S(B_j) = SB for all j
    for j in range(N + 1):
        idx = (N + 1) + j
        mask = 1 << idx
        cname = f"S_B{j}_eq_SB"
        prob += (s_vars[mask] == SB), cname

    # S(A_i B_j) = SAB for all i, j
    for i in range(N + 1):
        for j in range(N + 1):
            idxA = i
            idxB = (N + 1) + j
            mask = (1 << idxA) | (1 << idxB)
            cname = f"S_A{i}B{j}_eq_SAB"
            prob += (s_vars[mask] == SAB), cname

    # === Strong subadditivity (SSA) ===
    # For all disjoint, nonempty I, J, K:
    #   S(IJ) + S(JK) >= S(J) + S(IJK)
    for I in subsets:
        for J in subsets:
            if I & J:
                continue  # must be disjoint
            for K in subsets:
                if (I & K) or (J & K):
                    continue  # pairwise disjoint
                IJ = I | J
                JK = J | K
                IJK = I | J | K
                cname = f"SSA_{subset_name(I)}_{subset_name(J)}_{subset_name(K)}"
                prob += (
                    s_vars[IJ] + s_vars[JK] - s_vars[J] - s_vars[IJK] >= 0
                ), cname

    # === Weak monotonicity (WM) ===
    # For all disjoint, nonempty X, Y, Z:
    #   S(XY) + S(XZ) >= S(Y) + S(Z)
    for X in subsets:
        for Y in subsets:
            if X & Y:
                continue  # disjoint
            for Z in subsets:
                if (X & Z) or (Y & Z):
                    continue  # pairwise disjoint
                XY = X | Y
                XZ = X | Z
                cname = f"WM_{subset_name(X)}_{subset_name(Y)}_{subset_name(Z)}"
                prob += (
                    s_vars[XY] + s_vars[XZ] - s_vars[Y] - s_vars[Z] >= 0
                ), cname

    # === Conditional entropy non-negativity ===
    # For all disjoint, nonempty X, Y:
    #   S(X|Y) = S(XY) - S(Y) >= 0
    #   S(Y|X) = S(XY) - S(X) >= 0
    for X in subsets:
        for Y in subsets:
            if X & Y:
                continue  # must be disjoint
            XY = X | Y

            #cname1 = f"CondPos_{subset_name(X)}_given_{subset_name(Y)}"
            prob += (s_vars[XY] - s_vars[Y] >= 0)

            #cname2 = f"CondPos_{subset_name(Y)}_given_{subset_name(X)}"
            prob += (s_vars[XY] - s_vars[X] >= 0)

    # Dummy objective: minimize 0 (pure feasibility)
    prob += 0, "Objective"

    # Choose solver
    if solver is None:
        solver = pulp.PULP_CBC_CMD(msg=verbose)

    prob.solve(solver)

    status = pulp.LpStatus[prob.status]
    if verbose:
        print("Solver status:", status)
        if status == "Optimal":
            print("Feasible entropic vector found.")
        else:
            print("No feasible solution under the given constraints.")

    return status, prob, s_vars, subset_name


if __name__ == "__main__":
    # Example: N = 1 gives A0,A1,B0,B1
    N = 3

    # Example parameters – you can plug in Werner values, etc.
    SA = 1.0
    SB = 1.0
    SAB = 1.3

    status, prob, s_vars, name = build_entropy_feasibility_lp_general(
        N, SA, SB, SAB, verbose=True
    )

    if status == "Optimal":
        print("\nSample entropies from a feasible solution:")
        for mask, var in s_vars.items():
            print(f"S({name(mask)}) = {var.value():.6f}")
            # break  # uncomment if you don't want everything
    else:
        print("\nGiven (SA, SB, SAB) is infeasible under these constraints for this N.")
