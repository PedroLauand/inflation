import einops 
import numpy as np
#EJM Measurements
e_0 =(1/np.sqrt(8))*np.array([[-1 - 1j, 0, -2j, -1 + 1j]]).reshape((2, 2))
e_1 = (1/np.sqrt(8))*np.array([[1 - 1j, 2j, 0, 1 + 1j]]).reshape((2, 2))
e_2 = (1/np.sqrt(8))*np.array([[-1 + 1j, 2j, 0, -1 - 1j]]).reshape((2, 2))
e_3 = (1/np.sqrt(8))*np.array([[1 + 1j, 0, -2j, 1 - 1j]]).reshape((2, 2))
EJM = np.array([e_0, e_1, e_2, e_3])
# State
psi = (1/np.sqrt(2))*np.array([[0, 1, -1, 0]]).reshape((2, 2))


#1 loop
def loop_1():
    P_A=np.zeros((4), dtype=float)
    for a in range(4):
        amplitude = einops.einsum(psi, EJM[a],
                             'ABa ABb, ABb ABa -> ')
        P_A[a] = np.real(amplitude*amplitude.conj())
    
    return P_A

def loop_2():
    P_AB=np.zeros((4,4), dtype=float)
    for a in range(4):
        for b in range(4):
            amplitude = einops.einsum(psi, psi, EJM[a], EJM[b],
                             'ABa ABb, BCb BCa, BCa ABa, ABb BCb -> ')
            P_AB[a,b] = np.real(amplitude*amplitude.conj())
            
    
    return P_AB

#Function for n loop
#Cyclic permutation of the last and the first for the subsystems.

#Function for a line (loop then sum to the line).
def loop_3():
    P_ABC = np.zeros((4,4,4), dtype=float)
    for a,b,c in np.ndindex((4, 4, 4)):
        amplitude = einops.einsum(psi, psi, psi, EJM[a], EJM[b], EJM[c], 'ABa ABb, BCb BCc, CAc CAa, CAa ABa, ABb BCb, BCc CAc-> ')
        P_ABC[a, b, c] = np.real(amplitude*amplitude.conj())
    return P_ABC

def loop_4():
    P_ABCD = np.zeros((4,4,4,4), dtype=float)
    for a,b,c,d in np.ndindex((4, 4, 4, 4)):
        amplitude = einops.einsum(psi, psi, psi, psi, EJM[a], EJM[b], EJM[c], EJM[d], 'ABa ABb, BCb BCc, CDc CDd, DAd DAa, DAa ABa, ABb BCb, BCc CDc, CDd DAd-> ')
        P_ABCD[a, b, c, d] = np.real(amplitude*amplitude.conj())
    return P_ABCD
#print(P_ABC)