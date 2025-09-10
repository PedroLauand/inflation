import sys
sys.path.append('/Users/pedrolauand/My_Code/Inflation/inflation')
from inflation import InflationProblem, InflationLP
import numpy as np
from EJM_4outcomes_GPT import build_values

def ring_problem(inflation_level: int, nof_outcomes: int = 2) -> InflationProblem:
    inf_prob = InflationProblem(
        dag={"i1": ["A"],
             "i2": ["A"], },
        outcomes_per_party=(nof_outcomes,),
        settings_per_party=(1,),
        classical_sources="all",
        inflation_level_per_source=(inflation_level,inflation_level),
        order=["A"],
        really_just_one_source=True)
    return inf_prob

print("Initiating InflationProblem instance.")
prob = ring_problem(4, 4)
print("Adding symmetries.")
prob.add_symmetries(prob._setting_specific_outcome_relabelling_symmetries)
print("InflationProblem initiation complete. Now calculating probabilities.")

values = build_values(4)  # for inflation level 4
print("Probability calculations complete, now initiation InflationLP initialization.")
ring_LP = InflationLP(prob, verbose=2, include_all_outcomes=True)
print("Extracting all atomic monomials from the LP.")
at=ring_LP.atomic_monomials

actual_values={}
for k in values:
    for j in at:
        if k == j.name:
            print(f"{k} : {values[k]}")
            actual_values.update({k:values[k]})


    


print("Assigning these values to the LP")
ring_LP.update_values(values=actual_values, only_specified_values=False)
print("Assignment complete. Beginning solve.")


ring_LP.solve(solve_dual=False)
#print(ring_LP.status)

#pedro_keys = set(my_dict.keys())
#lp_keys = set([m.name for m in prob.atomic_monomials])
#settable_keys = pedro_keys.intersection(lp_keys)
#new_dict = {k, my_dict[k] for k in settable_keys}
#new_dict = {k: my_dict[k] for k in settable_keys}