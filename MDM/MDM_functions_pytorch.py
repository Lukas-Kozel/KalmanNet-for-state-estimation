import torch
import sys

def pinv(X, check=False):
    if check:
        # Rank v PyTorch
        rank = torch.linalg.matrix_rank(X)
        if X.shape[1] != rank:
            print(" ")
            raise ValueError(f"Rank error in 'pinv' calculation {rank} < {X.shape[1]}", f"X = {X}")
    # Ekvivalent np.linalg.inv(X.T @ X) @ X.T
    return torch.linalg.inv(X.T @ X) @ X.T

def I_XpinvX(X, check=False):
    return torch.eye(X.shape[0], device=X.device) - X @ pinv(X, check)

def Ksi_fun(n, device='cpu'):
    # Getting indices of the upper triangle
    i, j = torch.triu_indices(n, n)
    num_elements = i.shape[0]
    Ksi = torch.zeros((num_elements, n * n), dtype=torch.float32, device=device)
    
    # Filling the matrix (equivalent to: Ksi[np.arange(len(i)), i + j*n] = 1)
    Ksi[torch.arange(num_elements), i + j * n] = 1.0
    return Ksi

def baseMatrix_fun(n, sym=1, device='cpu'):
    if sym == 1:
        i, j = torch.triu_indices(n, n)
        num_elements = i.shape[0]
        baseMatrix = torch.zeros((num_elements, n, n), dtype=torch.float32, device=device)
        # Setting 1 at symmetric positions
        baseMatrix[torch.arange(num_elements), i, j] = 1.0
        baseMatrix[torch.arange(num_elements), j, i] = 1.0
    else:
        # Ekvivalent np.indices
        i = torch.arange(n).repeat_interleave(n)
        j = torch.arange(n).repeat(n)
        num_elements = i.shape[0]
        baseMatrix = torch.zeros((n * n, n, n), dtype=torch.float32, device=device)
        baseMatrix[torch.arange(num_elements), j, i] = 1.0
    return baseMatrix

def kron2_vec(x):
    # Fast equivalent of np.outer(x, x).ravel()
    return torch.outer(x, x).reshape(-1)

def kron2_mat(A):
    # Fast equivalent of np.kron(A, A) using torch.kron or einsum
    # einsum is often faster for specific dimensions
    m, n = A.shape
    res = torch.einsum('ij,kl->ikjl', A, A)
    return res.reshape(m * m, n * n)

def blkDiag(A_list):
    """Equivalent of blkDiag built on torch.block_diag."""
    if not A_list:
        return torch.tensor([], device='cpu')
    return torch.block_diag(*A_list)

def O_Gamma_fun(L, F_list, nz, H_list, i):
    """
    Creates the O (Observability) and Gamma (Input-to-Measurement) matrices.
    Inputs F_list and H_list should be lists/tensors of matrices for the given steps.
    """
    nx = F_list[0].shape[1]
    device = F_list[0].device

    O_Gamma = None
    
    for j in range(L - 1, -1, -1):
        part_O_Gamma_last = H_list[i + j]
        part_O_Gamma = part_O_Gamma_last
        
        for k in range(j - 1, -1, -1):
            part_O_Gamma_last = part_O_Gamma_last @ F_list[i + k]
            part_O_Gamma = torch.cat([part_O_Gamma_last, part_O_Gamma], dim=1)
            
        if j == L - 1:
            O_Gamma = part_O_Gamma
        else:
            # Padding with zeros to align dimensions
            zeros = torch.zeros((nz[i + j], (L - 1 - j) * nx), device=device)
            part_O_Gamma = torch.cat([part_O_Gamma, zeros], dim=1)
            O_Gamma = torch.cat([part_O_Gamma, O_Gamma], dim=0)
            
    return O_Gamma[:, :nx], O_Gamma[:, nx:]

def Upsilon_2_fun(w2b_list, v2b_list, L, device='cpu'):
    """Vectorized version of Upsilon_2 on GPU."""
    # The baseMatrix matrices are in a list (as in the original code)
    nw = w2b_list[0].shape[2]
    nv = v2b_list[0].shape[2]
    
    nwvp = (L - 1) * nw + L * nv
    Upsilon_2 = torch.zeros((nwvp * nwvp, 0), device=device)
    
    # Part for the process noise W
    for i in range(min(len(w2b_list), L - 1)):
        w2b = w2b_list[i]
        for j in range(w2b.shape[0]):
            eye_part = torch.eye(L - 1, device=device)[i].reshape(L - 1, 1) @ \
                       torch.eye(L - 1, device=device)[i].reshape(1, L - 1)
            
            # Kron(eye, base)
            Upsilon_2_part = torch.kron(eye_part, w2b[j])
            if i >= 1:
                Upsilon_2_part = Upsilon_2_part + Upsilon_2_part.T
            
            # Block padding with zeros (corresponds to blkDiag([part, zeros]))
            full_block = torch.zeros((nwvp, nwvp), device=device)
            full_block[:(L-1)*nw, :(L-1)*nw] = Upsilon_2_part
            
            Upsilon_2 = torch.cat([Upsilon_2, full_block.reshape(-1, 1)], dim=1)

    # Part for the measurement noise V
    for i in range(min(len(v2b_list), L)):
        v2b = v2b_list[i]
        for j in range(v2b.shape[0]):
            eye_part = torch.eye(L, device=device)[i].reshape(L, 1) @ \
                       torch.eye(L, device=device)[i].reshape(1, L)
            
            Upsilon_2_part = torch.kron(eye_part, v2b[j])
            if i >= 1:
                Upsilon_2_part = Upsilon_2_part + Upsilon_2_part.T
            
            full_block = torch.zeros((nwvp, nwvp), device=device)
            full_block[(L-1)*nw:, (L-1)*nw:] = Upsilon_2_part
            
            Upsilon_2 = torch.cat([Upsilon_2, full_block.reshape(-1, 1)], dim=1)

    return Upsilon_2

import torch
import torch.linalg

def null_space_torch(A, rtol=1e-5):
    """
    Helper function for computing the null space of a matrix (equivalent of scipy.linalg.null_space).
    Uses SVD decomposition.
    """
    u, s, vh = torch.linalg.svd(A, full_matrices=True)
    # vh are the rows of V^H, so the columns of V are vh.mH (Hermitian transposed)
    # Looking for columns of V corresponding to small singular values
    max_s = torch.max(s)
    tol = max_s * rtol
    num_null = A.shape[1] - torch.sum(s > tol)
    null_v = vh[-num_null:, :].mH # The last n rows of vh transposed into columns
    return null_v

def MDM_nullO_LTI(L, F, G, E, nz, H, D, z, u, version):   
    """
    LTI (Linear Time Invariant) wrapper for MDM_nullO.
    Tiles (repeats) the system matrices for all windows.
    """
    if L < 1:
        raise ValueError("L must be greater than 0")
    elif not(version == 0 or version == 1 or version == 2):    
        raise ValueError("Version must be 0/1/2")
    elif version == 2 and L == 1:    
        raise ValueError("Version must be 0/1 for L=1")

    device = F.device
    
    # Preparing lists/tensors for the LTI system (repeating the same matrices)
    F_list = F.unsqueeze(0).repeat(L-1, 1, 1)
    G_list = G.unsqueeze(0).repeat(L-1, 1, 1)
    E_list = E.unsqueeze(0).repeat(L-1, 1, 1)
    
    H_list = H.unsqueeze(0).repeat(L, 1, 1)
    D_list = D.unsqueeze(0).repeat(L, 1, 1)
    nz_list = nz.repeat(L)
     
    # MDM core call
    r, Awv, aniMat, Gamma, calG = MDM_nullO(L, F_list, G_list, E_list, nz_list, H_list, D_list, z, u, version) 
    
    # Padding residuals for the rest of the sequence (if the input is longer than window L)
    N = z.shape[0] if isinstance(z, torch.Tensor) else len(z)
    
    for i in range(1, N - (L - 1)): 
        # Z = [z_i, z_{i+1}, ..., z_{i+L-1}]
        Z = torch.cat([z[i + j].reshape(-1) for j in range(L)], dim=0)
        
        if version == 2 or L == 1:  
            r.append(aniMat @ Z)
        else:    
            # For versions 0 and 1 we must subtract the influence of the control input U
            aniMat_Gamma_calG = aniMat @ Gamma @ calG
            U = torch.cat([u[i + j].reshape(-1) for j in range(L - 1)], dim=0)
            r.append(aniMat @ Z - aniMat_Gamma_calG @ U)
            
    return r, Awv[0]

def MDM_nullO(L, F, G, E, nz, H, D, z, u, version):   
    """
    Core of the MDM algorithm (Measurement Difference Method).
    """
    device = F.device
    N = nz.shape[0]

    r = []
    Awv = []
    
    # Computed only for the first possible window (i=0) - sufficient for online AKF
    # The rest is computed in a loop in MDM_nullO_LTI
    for i in range(1): # In the recursive version we only care about the current window
        if L == 1:
            if version == 0:
                aniMat = I_XpinvX(H[i], 0)
            elif version == 1:    
                # Columns of the null space of H.T, then transpose to rows
                aniMat = null_space_torch(H[i].T).T
            
            # z[i] je tensor (obs_dim,)
            curr_z = z[i].reshape(-1)
            r.append(aniMat @ curr_z)
            Awv.append(aniMat @ D[i])
            
            Gamma = torch.tensor([], device=device)
            calG = torch.tensor([], device=device)
            
        else: # Pro okno L > 1
            # 1. Computation of observability and input transfer matrices
            O, Gamma = O_Gamma_fun(L, F, nz, H, i)   
     
            if L == 2:
                calG = G[i] 
                calE = E[i]
            else:
                # blkDiag ze seznamu matic
                calG = torch.block_diag(*[G[j] for j in range(i, i + L - 1)])
                calE = torch.block_diag(*[E[j] for j in range(i, i + L - 1)])

            # 2. Computation of the annihilation matrix (aniMat)
            if version == 0: 
                aniMat = I_XpinvX(O, 0)
            elif version == 1:    
                aniMat = null_space_torch(O.T).T
            elif version == 2:    
                # Version 2 also annihilates the influence of the control input
                M_v2 = torch.cat([O, Gamma @ calG], dim=1)
                aniMat = null_space_torch(M_v2.T).T

            # 3. Computation of Awv (regressor matrix for noise)
            calD = torch.block_diag(*[D[j] for j in range(i, i + L)])
            # Awv = aniMat @ [Gamma*calE, calD]
            Awv.append(aniMat @ torch.cat([Gamma @ calE, calD], dim=1))
        
            # 4. Computation of the residual r_k
            Z = torch.cat([z[i + j].reshape(-1) for j in range(L)], dim=0)
            if version == 2:  
                r.append(aniMat @ Z)
            else:    
                U = torch.cat([u[i + j].reshape(-1) for j in range(L - 1)], dim=0)
                r.append(aniMat @ (Z - Gamma @ calG @ U))
                
    return r, Awv, aniMat, Gamma, calG