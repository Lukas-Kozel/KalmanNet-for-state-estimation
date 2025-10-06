# @author: Oliver Kost, kost@ntis.zcu.cz

import sys
import numpy as np
import scipy.linalg as sp

def pinv(X,check=False):
    if check:
        if X.shape[1] != np.linalg.matrix_rank(X):
            print(" ")
            raise ValueError(f"Rank error in 'pinv' calculation {np.linalg.matrix_rank(X)} < {X.shape[1]}", f"X = {X}")
            sys.exit()
    return np.linalg.inv(X.T @ X) @ X.T

def I_XpinvX(X,check=False):
    return np.eye(X.shape[0]) - X @ pinv(X,check)

def Ksi_fun(n): #  Rvu = Ksi @ Rv
    i, j = np.triu_indices(n)
    Ksi = np.zeros((len(i), n*n), int)
    Ksi[np.arange(len(i)), i + j*n] = 1
    return Ksi

def baseMatrix_fun(n , sym = 1):
    if sym==1: # for symetric matrices
        i, j = np.triu_indices(n)
        baseMatrix= np.zeros((len(i), n, n), int)
        baseMatrix[np.arange(len(i)), i,j] = 1
        baseMatrix[np.arange(len(i)), j,i] = 1
    else:
        i, j = np.indices((n,n))
        i = np.hstack(i)
        j = np.hstack(j)
        baseMatrix= np.zeros((n*n, n, n), int)
        baseMatrix[np.arange(len(i)), j, i] = 1
    return baseMatrix

def Upsilon_1_fun(w1b,v1b,L):
    nw = w1b.shape[1]
    nv = v1b.shape[1]

    nwvp = (L-1)*nw + L*nv
    
    Upsilon_1 = np.zeros((nwvp , 0))
    if L>1: 
        for j in range(w1b.shape[0]):
            Upsilon_1 = np.hstack([ Upsilon_1 , np.vstack([ np.kron( np.ones((L-1,1)),w1b[j]), np.zeros((L*nv,1)) ]) ])
    for j in range(v1b.shape[0]):
        Upsilon_1 = np.hstack([ Upsilon_1 , np.vstack([ np.zeros(((L-1)*nw,1)) , np.kron( np.ones((L,1)),v1b[j]) ]) ])
    return Upsilon_1

def Upsilon_2_fun(w2b,v2b,L):
    nw = w2b[0].shape[1]
    nv = v2b[0].shape[1]
    
    nwvp = (L-1)*nw + L*nv
    
    Upsilon_2 = np.zeros((nwvp*nwvp , 0))
    for i in range(min(len(w2b),L-1)):
        for j in range(w2b[i].shape[0]):
            Upsilon_2_part = np.kron(np.eye(L-1,L-1,i),w2b[i][j])
            if i>=1:
                Upsilon_2_part = Upsilon_2_part + Upsilon_2_part.T 
            Upsilon_2 = np.hstack([ Upsilon_2 , np.reshape(blkDiag([ Upsilon_2_part , np.zeros((L*nv,L*nv)) ]),(-1,1)) ])

    for i in range(min(len(v2b),L)):
        for j in range(v2b[i].shape[0]):
            Upsilon_2_part = np.kron( np.eye(L,L,i),v2b[i][j])
            if i>=1:
                Upsilon_2_part = Upsilon_2_part + Upsilon_2_part.T
            Upsilon_2 = np.hstack([ Upsilon_2 , np.reshape(blkDiag([ np.zeros(((L-1)*nw,(L-1)*nw)) , Upsilon_2_part ]),(-1,1)) ])

    return Upsilon_2

'''
def O_Gamma_fun_old(L,F,nz,H,i):
    nx = F.shape[1]

    nzSum = 0
    for k in range(L): # sloupecky
        if k==0:
            part_O_Gamma = H[i]
        else:
            nzSum += nz[i+k-1]    
            part_O_Gamma = np.vstack([np.zeros((nzSum,nx)), H[i+k]])
            
        Fpart = np.eye(nx)
        for j in range(k+1,L): # radky 
            Fpart = F[i+j-1] @ Fpart
            part_O_Gamma = np.vstack([part_O_Gamma, H[i+j] @ Fpart])
        if k==0:
            O = part_O_Gamma 
        elif k==1:
            Gamma = part_O_Gamma
        else:
            Gamma = np.hstack([Gamma, part_O_Gamma]) 
    return O, Gamma
'''

def O_Gamma_fun(L,F,nz,H,i):
    nx = F.shape[1]

    for j in range(L-1,-1,-1): # radky 
        part_O_Gamma_last = H[i+j]
        part_O_Gamma = part_O_Gamma_last
        for k in range(j-1,-1,-1): # sloupecky 
            part_O_Gamma_last = part_O_Gamma_last @ F[i+k] 
            part_O_Gamma = np.hstack([part_O_Gamma_last, part_O_Gamma]) 
        if j==L-1:
            O_Gamma = part_O_Gamma
        else:
            part_O_Gamma = np.hstack([part_O_Gamma, np.zeros((nz[i+j],(L-1-j)*nx))])
            O_Gamma = np.vstack([part_O_Gamma , O_Gamma])  
    return O_Gamma[:,:nx], O_Gamma[:,nx:]

def kron2_vec(x):
    return np.outer(x, x).ravel() # fast np.kron(x, x)
   
def kron2_mat(A):
    m, n = A.shape  
    return np.einsum('ij,kl->ikjl', A, A).reshape(m*m, n*n) # fast np.kron(A,A)  

def blkDiag(A):
    rL = range(len(A))
    rows_cols = [A[i].shape for i in rL]
    
    total_rows = sum(s[0] for s in rows_cols)
    total_cols = sum(s[1] for s in rows_cols)
    
    result = np.zeros((total_rows, total_cols))
    
    row_start = 0
    col_start = 0
    for i in rL:
        rows, cols = rows_cols[i]
        result[row_start:row_start+rows, col_start:col_start+cols] = A[i]
        row_start += rows
        col_start += cols
    return result

def MDM_nullO_LTI(L,F,G,E,nz,H,D,z,u,version):   
    if L<1:
        print(" ")
        raise ValueError("L must be greater than 0")
        sys.exit() 
    elif not(version==0 or version==1 or version==2):    
        print(" ")
        raise ValueError("Version must be 0/1/2")
        sys.exit()      
    elif version == 2 and L==1:    
        print(" ")
        raise ValueError("Version must be 0/1 for L=1")
        sys.exit()  
    
    F = np.tile(F, (L-1, 1, 1))
    G = np.tile(G, (L-1, 1, 1))
    E = np.tile(E, (L-1, 1, 1))
    
    H = np.tile(H, (L, 1, 1))
    D = np.tile(D, (L, 1, 1))
    nz = np.tile(nz, L)
     
    r, Awv, aniMat, Gamma, calG = MDM_nullO(L,F,G,E,nz,H,D,z,u,version) 
    
    N = len(z)   
    for i in range(1,N-(L-1)): 
        Z = np.hstack([z[i + j] for j in range(L)]) 
        if version == 2 or L==1:  
            r.append( aniMat @ Z )
        else:    
            if i==1:
                aniMat_Gamma_calG = aniMat @ Gamma @ calG
            U = np.hstack([u[i + j] for j in range(L-1)])
            r.append( aniMat @ Z - aniMat_Gamma_calG @ U )
            
    return r, Awv[0]

def MDM_nullO(L,F,G,E,nz,H,D,z,u,version):   
    if L<1:
        print(" ")
        raise ValueError("L must be greater than 0")
        sys.exit() 
    elif not(version==0 or version==1 or version==2):    
        print(" ")
        raise ValueError("Version must be 0/1/2")
        sys.exit()      
    elif version == 2 and L==1:    
        print(" ")
        raise ValueError("Version must be 0/1 for L=1")
        sys.exit()  
 
    N = nz.shape[0]

    r = []
    Awv = []
    for i in range(N-(L-1)): 
        if L==1:
            if version == 0:
                aniMat = I_XpinvX(H[i],0) # pinv H
            elif version == 1:    
                aniMat = sp.null_space(H[i].T).T # anihilation matrix H    
            r.append( aniMat @ z[i] )
            Awv.append( aniMat @ D[i] )  # [Aw, Av]
            
            Gamma = [] 
            calG = []
            
        else: # for L>1
            O, Gamma = O_Gamma_fun(L,F,nz,H,i)   
            
            '''
            O_old, Gamma_old = O_Gamma_fun_old(L,F,nz,H,i)   
            error = sum(sum(abs(O_old-O)))+sum(sum(abs(Gamma_old-Gamma)))>1e-13
            if error>1e-15:
                print(error)
            '''
     
            if L==2:
                calG = G[i] 
                calE = E[i]
            else:
                calG = blkDiag(G[i:i+L-1]) 
                calE = blkDiag(E[i:i+L-1]) 

            if version == 0: 
                aniMat = I_XpinvX(O,0) # pinv O
            elif version == 1:    
                aniMat = sp.null_space(O.T).T # anihilation matrix O
            elif version == 2:    
                aniMat = sp.null_space(np.hstack([O, Gamma @ calG]).T).T # anihilation matrix  [O, Gamma @ calG] 

            calD = blkDiag(D[i:i+L])
            Awv.append( aniMat @ np.hstack([Gamma @ calE, calD]) )  # [Aw, Av]
        
            Z = np.hstack([z[i + j] for j in range(L)]) 
            if version == 2:  
                r.append( aniMat @ Z )
            else:    
                U = np.hstack([u[i + j] for j in range(L-1)])
                r.append( aniMat @ ( Z - Gamma @ calG @ U ) )
    return r, Awv, aniMat, Gamma, calG