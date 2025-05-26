import numpy as np
P_I=np.ones((4,4,4,1,1,1), dtype=float)/64
P_NS=np.zeros((4,4,4,1,1,1))


for a in range(4):
    P_NS[a, a, a] = 1 / 8
    for b in range(4):
        for c in range(4):
            if a!=b and b!=c and c!=a:
                P_NS[a,b,c]=1/48
# v=1
# p_v = np.zeros((4, 4, 4, 1, 1, 1))
# p_v=v*P_NS+(1-v)*P_I

print("P_NS(A=i,B=i)=",P_NS[0, 0, :].sum())
print("P_NS(A=i,B=j)=",P_NS[0, 1, :].sum())
# for i in range(3):
#     print('P_NS(a='+str(i)+')=', sum([P_NS[i,b,c] for b in range(4)for c in range(4)]))
#     print('P_NS(b='+str(i)+')=', sum([P_NS[a,i,c] for a in range(4)for c in range(4)]))
#     print('P_NS(c='+str(i)+')=', sum([P_NS[a,b,i] for a in range(4)for b in range(4)]))
#
#     print('P_I(a='+str(i)+')=', sum([P_I[i,b,c] for b in range(4)for c in range(4)]))
#
# for i in range(3):
#     for j in range(3):
#         print('P_NS(a='+str(i)+',b='+str(j)+')=', sum([P_NS[i,j,c] for c in range(4)]))
#         print('P_NS(b='+str(i)+',c='+str(j)+')=', sum([P_NS[a,i,j] for a in range(4)]))
#         print('P_NS(a='+str(i)+',c='+str(j)+')=', sum([P_NS[i,b,j] for b in range(4)]))
#
#
#         print('P_I(a='+str(i)+',b='+str(j)+')=', sum([P_I[i,j,c] for c in range(4)]))
# for i in range(3):
#     for j in range(3):
#         for k in range(3):
#             print('P_NS(a='+str(i)+',b='+str(j)+',c='+str(k)+')=',P_NS[i,j,k] )
print("P_NS(A=i,B=j)=", (1-4*P_NS[0, 0, :].sum())/12)