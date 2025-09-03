import sys
sys.path.append('/Users/pedrolauand/My_Code/Distributed_Computing/inflation')
from inflation import InflationProblem, InflationLP
import numpy as np
from EJM_4outcomes import loop_1, loop_2, loop_3, loop_4

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

prob = ring_problem(4, 4)
prob.add_symmetries(prob._setting_specific_outcome_relabelling_symmetries)



oA=4
#Inflation level 4
P_A=loop_1()
P_AB=loop_2()
P_ABC=loop_3()
P_ABCD=loop_4()
values={}
for a1 in range(oA):
    #1-loop
    values.update({"P[A^{1,1}="+str(a1)+"]":P_A[a1]})
    #1-line
    values.update({"P[A^{1,2}="+str(a1)+"]":sum( P_AB[a1,b] for b in range(oA))})
    for a2 in range(oA):
        #2-loop
        values.update({"P[A^{1,2}="+str(a1)+ " " + "A^{2,1}="+str(a2)+"]":P_AB[a1,a2]})
        #2-line
        values.update({"P[A^{1,2}="+str(a1)+ " " + "A^{2,3}="+str(a2)+"]":sum( P_ABC[a1,a2,b] for b in range(oA))})
        for a3 in range(oA):
            #3-loop
            values.update({"P[A^{1,2}=" + str(a1) + " " + "A^{2,3}=" + str(a2) + " " + "A^{3,1}="+str(a3)+"]" : P_ABC[a1,a2,a3]})
            #3-line
            values.update({"P[A^{1,2}=" + str(a1) + " " + "A^{2,3}=" + str(a2) + " " + "A^{3,4}="+str(a3) +"]" : sum(P_ABCD[a1,a2,a3,b] for b in range(oA))})
            for a4 in range(oA):
                #4-loop
                values.update({"P[A^{1,2}=" + str(a1) + " " + "A^{2,3}=" + str(a2) + " " + "A^{3,4}="+str(a3)  + " " + "A^{4,1}="+str(a4) +"]" : P_ABCD[a1,a2,a3,a4]})


#print(values)

#prob.add_symmetries(prob._setting_specific_outcome_relabelling_symmetries)
#Dictionarie for inflation values via index sym
#Inflation level n-> 1-loop: P[A^{1,1} = a] 
# -> 1-line : P[A^{1,2} = a] 
# ->2-loop :  P[A^{1,2}=a A^{2,1}=a'] 
# -> 2-line: P[A^{1,2}=a A{2,3}=a'] 
# ->3 -loop : P[A^{1,2}=a A{2,3}=a' A^{3,1}=a'']
# ->3 -line : P[A^{1,2}=a A{2,3}=a' A^{3,4}=a'']
ring_LP = InflationLP(prob, verbose=2, include_all_outcomes=True)

at=ring_LP.atomic_monomials

actual_values={}
for k in values:
    for j in at:
        if k == j.name:
            actual_values.update({k:values[k]})

print(actual_values)

    



ring_LP.update_values(values=actual_values, only_specified_values=False)



ring_LP.solve(solve_dual=False)
#print(ring_LP.status)

#pedro_keys = set(my_dict.keys())
#lp_keys = set([m.name for m in prob.atomic_monomials])
#settable_keys = pedro_keys.intersection(lp_keys)
#new_dict = {k, my_dict[k] for k in settable_keys}
#new_dict = {k: my_dict[k] for k in settable_keys}