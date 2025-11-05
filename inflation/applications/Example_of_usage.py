# example_usage.py
from EJM_utils import iterate_global_events_containing_marginal
from Group_tools import build_sympy_group, prepare_group_chain, canonical_leximin_coset_chain # type: ignore

# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------
n, outcomes = 2, 2
N = (n * n) * outcomes

# Example marginal (1-based i,j indices)
marginal = [
    [1, 1, 2, 0, 0],  # A^{1,2} = 0
    [1, 2, 1, 0, 1],  # A^{2,1} = 1
]

# Small permutation group on N=8 coordinates
raw_G = [
    [0, 1, 2, 3, 4, 5, 6, 7],        # identity
    [1, 0, 3, 2, 5, 4, 7, 6],        # swap within outcome blocks
    [6, 7, 4, 5, 2, 3, 0, 1],        # reverse block order
    [7, 6, 5, 4, 3, 2, 1, 0],        # full reversal
]

# ------------------------------------------------------------
# Group construction and preprocessing
# ------------------------------------------------------------
G = build_sympy_group(raw_G, N)
precomp = prepare_group_chain(G, N)

# ------------------------------------------------------------
# Iterate over all full global events consistent with the marginal
# and count their canonical representatives
# ------------------------------------------------------------
rep_counts = {}

for evt in iterate_global_events_containing_marginal(n, outcomes, marginal):
    rep_evt, _ = canonical_leximin_coset_chain(evt, outcomes, precomp=precomp)
    key = tuple(rep_evt)
    rep_counts[key] = rep_counts.get(key, 0) + 1

# ------------------------------------------------------------
# Print results
# ------------------------------------------------------------
print("\nCanonical representatives and counts:")
for rep, count in sorted(rep_counts.items()):
    print(f"{list(rep)} : {count}")

# Optionally, store results as a dictionary object
print("\nSummary dictionary:")
print(rep_counts)
