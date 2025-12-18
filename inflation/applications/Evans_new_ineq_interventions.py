import sys
sys.path.append('/Users/pedrolauand/My_Code/Inflation/inflation')
from inflation import InflationProblem, InflationLP
import numpy as np
v=1
P_Q=np.zeros((2,2,2,1,1,1))
for a in range(2):
    for b in range(2):
        for c in range(2):
            P_Q[a,b,c,0,0,0]= 1/8*(1+(1/np.sqrt(2))*(-1)**(a+b*(c+1)))
prob= InflationProblem({"h1": ["A", "B"],
                       "h2": ["B", "C"],
                       "B": ["A", "C"]},
                       classical_sources="all",  
                       outcomes_per_party=[2, 2, 2],
                       settings_per_party=[1, 1, 1],
                       inflation_level_per_source=[2, 2],
                       order=['A', 'B', 'C'])

lp = InflationLP(prob)

lp.set_distribution(P_Q)
dict={'P[A=0|do(B=0)]': 1/2}
lp.update_values(dict)
lp.solve(verbose=2)
print(lp.certificate_as_probs())


#print(lp.certificate_as_dict())
