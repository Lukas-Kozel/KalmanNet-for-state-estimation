# @author: Oliver Kost, kost@ntis.zcu.cz

from MDM_functions import pinv, Ksi_fun, baseMatrix_fun, Upsilon_1_fun, Upsilon_2_fun, kron2_vec, kron2_mat, MDM_nullO_LTI

from GenData import genData_LTI

import sys
import numpy as np

import time
start = time.time()

N = int(1e4)
 
F,G,E,nz,H,D,z,w1,v1,Q,R,nw,nv,u,w,v = genData_LTI(N)    
    
L = int(2)
version = int(2)

r, Awv = MDM_nullO_LTI(L,F,G,E,nz,H,D,z,u,version)    

Np = N-(L-1)     

#w1b = np.ones((1,nw,1))
w1b = np.eye(nw); w1b = w1b.reshape(w1b.shape+(1,)) # dimensions [numberOfBaseVectors,nw,1] the last two dimensions (nw,1) are the dimensions of the base vectors
v1b = np.eye(nv); v1b = v1b.reshape(v1b.shape+(1,))
Upsilon_1 = Upsilon_1_fun(w1b,v1b,L)   

Awv1u = Awv @ Upsilon_1

if Upsilon_1.shape[1] != np.linalg.matrix_rank(Awv1u):
    print(" ")
    raise ValueError(f"Deficient rank in estimating the mean value {np.linalg.matrix_rank(Awv1u)} < {Upsilon_1.shape[1]}")
    sys.exit()

alpha_1 = pinv(Awv1u) @ np.mean(r,0)

if L==1:  
    print(np.round( np.vstack([ alpha_1 , (pinv(np.hstack(v1b)) @ v1).T]) ,decimals=2))
else:  
    print(np.round( np.vstack([ alpha_1 , np.vstack([pinv(np.hstack(w1b)) @ w1, pinv(np.hstack(v1b)) @ v1]).T]) ,decimals=2))

r02 = []
wv1 = Upsilon_1 @ alpha_1 
r0 = r - Awv @ wv1 

nr = r[0].shape[0] 
Ksi = Ksi_fun(nr)
for i in range(Np):     
    r02.append(Ksi @ kron2_vec(r0[i])) 

w2b = [baseMatrix_fun(nw,1)]  # dimensions [timeShiftVCorrelation,numberOfBaseMatrices,nw,nw] the last two dimensions (nw,nw) are the dimensions of the base matrix
v2b = [baseMatrix_fun(nv,1),baseMatrix_fun(nv,0),baseMatrix_fun(nv,0)]
Upsilon_2 = Upsilon_2_fun(w2b,v2b,L)   

Awv2u = Ksi @ kron2_mat(Awv) @ Upsilon_2

if Upsilon_2.shape[1] != np.linalg.matrix_rank(Awv2u):
    print(" ")
    raise ValueError(f"Deficient rank in estimating the second moment {np.linalg.matrix_rank(Awv2u)} < {Upsilon_2.shape[1]}")
    sys.exit()
    
alpha_2 = pinv(Awv2u) @ np.mean(r02,0)

alpha_2_true = np.zeros((0,1))
maxCorrW = min(len(w2b),L-1) - 1
maxCorrV = min(len(v2b),L) - 1
if maxCorrW>=0:
    alpha_2_true = np.vstack([alpha_2_true , Ksi_fun(nw)@np.reshape(Q,(-1,1))])
if maxCorrW>=1:
    alpha_2_true = np.vstack([alpha_2_true , np.zeros((4,1)) ])
if maxCorrW>=2:
    alpha_2_true = np.vstack([alpha_2_true , np.zeros((4,1)) ])
if maxCorrV>=0:    
    alpha_2_true = np.vstack([alpha_2_true , Ksi_fun(nv)@np.reshape(R,(-1,1)) ])
if maxCorrV>=1: 
    alpha_2_true = np.vstack([alpha_2_true , np.reshape(R*2/3,(-1,1)) ])
if maxCorrV>=2:
    alpha_2_true = np.vstack([alpha_2_true , np.reshape(R/3,(-1,1)) ])
if maxCorrV>=3:
    alpha_2_true = np.vstack([alpha_2_true , np.zeros((4,1)) ])    

print(np.round( np.hstack([np.vstack(alpha_2) , alpha_2_true , 100*(np.vstack(alpha_2)-alpha_2_true)/np.vstack(alpha_2) ]) ,decimals=2))

end = time.time()
print(f"Trvalo to {end - start:.6f} s") # 1.51s 1e4 ver 1

WV = np.vstack([v[:,i:i+Np] for i in range(L)])
if L>1:
    WV = np.vstack([ np.vstack([w[:,i:i+Np] for i in range(L-1)]), WV ])
print(np.round(np.cov(WV),decimals=1))

# %prun -s tottime -q -T output.txt -l 30 %run v3_072925.py

'''
N=1e4
L=2
maxCorrW=0
maxCorrV=1

version = 0
[[ 1.    1.02  1.97 -1.04]
 [ 1.    1.    2.   -1.  ]]
[[ 1.91  2.   -4.87]
 [-1.05 -1.    4.41]
 [ 2.03  2.    1.25]
 [ 1.98  2.   -1.22]
 [ 0.98  1.   -1.58]
 [ 3.03  3.    0.98]
 [ 1.32  1.33 -0.8 ]
 [ 0.66  0.67 -0.61]
 [ 0.65  0.67 -3.31]
 [ 2.02  2.    1.12]]
Trvalo to 0.450804 s
[[ 2.  -1.   0.   0.1  0.   0.1]
 [-1.   2.   0.   0.   0.  -0. ]
 [ 0.   0.   2.   1.   1.3  0.6]
 [ 0.1  0.   1.   3.   0.7  2. ]
 [ 0.   0.   1.3  0.7  2.   1. ]
 [ 0.1 -0.   0.6  2.   1.   3. ]]

version = 1
[[ 1.    1.02  1.97 -1.04]
 [ 1.    1.    2.   -1.  ]]
[[ 1.87  2.   -6.79]
 [-1.09 -1.    8.3 ]
 [ 1.98  2.   -1.04]
 [ 1.98  2.   -1.12]
 [ 0.98  1.   -1.65]
 [ 3.03  3.    1.  ]
 [ 1.32  1.33 -0.79]
 [ 0.66  0.67 -0.67]
 [ 0.64  0.67 -3.46]
 [ 2.02  2.    0.98]]
Trvalo to 0.426804 s
[[ 2.  -1.   0.   0.1  0.   0.1]
 [-1.   2.   0.   0.   0.  -0. ]
 [ 0.   0.   2.   1.   1.3  0.6]
 [ 0.1  0.   1.   3.   0.7  2. ]
 [ 0.   0.   1.3  0.7  2.   1. ]
 [ 0.1 -0.   0.6  2.   1.   3. ]]

version = 2
raise ValueError(f"Deficient rank in estimating the mean value {np.linalg.matrix_rank(Awv1u)} < {Upsilon_1.shape[1]}")
ValueError: Deficient rank in estimating the mean value 3 < 4


'''