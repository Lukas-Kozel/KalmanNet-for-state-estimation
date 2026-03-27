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
    # Získání indexů horního trojúhelníku
    i, j = torch.triu_indices(n, n)
    num_elements = i.shape[0]
    Ksi = torch.zeros((num_elements, n * n), dtype=torch.float32, device=device)
    
    # Naplnění matice (ekvivalent k: Ksi[np.arange(len(i)), i + j*n] = 1)
    Ksi[torch.arange(num_elements), i + j * n] = 1.0
    return Ksi

def baseMatrix_fun(n, sym=1, device='cpu'):
    if sym == 1:
        i, j = torch.triu_indices(n, n)
        num_elements = i.shape[0]
        baseMatrix = torch.zeros((num_elements, n, n), dtype=torch.float32, device=device)
        # Nastavení 1 na symetrické pozice
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
    # Rychlý ekvivalent np.outer(x, x).ravel()
    return torch.outer(x, x).reshape(-1)

def kron2_mat(A):
    # Rychlý ekvivalent np.kron(A, A) pomocí torch.kron nebo einsum
    # einsum je pro specifické dimenze často rychlejší
    m, n = A.shape
    res = torch.einsum('ij,kl->ikjl', A, A)
    return res.reshape(m * m, n * n)

def blkDiag(A_list):
    """Ekvivalent blkDiag postavený na torch.block_diag."""
    if not A_list:
        return torch.tensor([], device='cpu')
    return torch.block_diag(*A_list)

def O_Gamma_fun(L, F_list, nz, H_list, i):
    """
    Vytvoří matice O (Observability) a Gamma (Input-to-Measurement).
    Vstupy F_list a H_list by měly být seznamy/tenzory matic pro dané kroky.
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
            # Doplnění nulami pro zarovnání rozměrů
            zeros = torch.zeros((nz[i + j], (L - 1 - j) * nx), device=device)
            part_O_Gamma = torch.cat([part_O_Gamma, zeros], dim=1)
            O_Gamma = torch.cat([part_O_Gamma, O_Gamma], dim=0)
            
    return O_Gamma[:, :nx], O_Gamma[:, nx:]

def Upsilon_2_fun(w2b_list, v2b_list, L, device='cpu'):
    """Vektorizovaná verze Upsilon_2 na GPU."""
    # Matice baseMatrix jsou v listu (jako v původním kódu)
    nw = w2b_list[0].shape[2]
    nv = v2b_list[0].shape[2]
    
    nwvp = (L - 1) * nw + L * nv
    Upsilon_2 = torch.zeros((nwvp * nwvp, 0), device=device)
    
    # Část pro šum procesu W
    for i in range(min(len(w2b_list), L - 1)):
        w2b = w2b_list[i]
        for j in range(w2b.shape[0]):
            eye_part = torch.eye(L - 1, device=device)[i].reshape(L - 1, 1) @ \
                       torch.eye(L - 1, device=device)[i].reshape(1, L - 1)
            
            # Kron(eye, base)
            Upsilon_2_part = torch.kron(eye_part, w2b[j])
            if i >= 1:
                Upsilon_2_part = Upsilon_2_part + Upsilon_2_part.T
            
            # Blokové doplnění nulami (odpovídá blkDiag([part, zeros]))
            full_block = torch.zeros((nwvp, nwvp), device=device)
            full_block[:(L-1)*nw, :(L-1)*nw] = Upsilon_2_part
            
            Upsilon_2 = torch.cat([Upsilon_2, full_block.reshape(-1, 1)], dim=1)

    # Část pro šum měření V
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
    Pomocná funkce pro výpočet nulového prostoru matice (ekvivalent scipy.linalg.null_space).
    Používá SVD rozklad.
    """
    u, s, vh = torch.linalg.svd(A, full_matrices=True)
    # vh jsou řádky V^H, tedy sloupce V jsou vh.mH (hermitovsky transponované)
    # Hledáme sloupce V, které odpovídají malým singulárním číslům
    max_s = torch.max(s)
    tol = max_s * rtol
    num_null = A.shape[1] - torch.sum(s > tol)
    null_v = vh[-num_null:, :].mH # Posledních n řádků vh transponovaných na sloupce
    return null_v

def MDM_nullO_LTI(L, F, G, E, nz, H, D, z, u, version):   
    """
    LTI (Linear Time Invariant) wrapper pro MDM_nullO.
    Tile (opakuje) matice systému pro všechna okna.
    """
    if L < 1:
        raise ValueError("L must be greater than 0")
    elif not(version == 0 or version == 1 or version == 2):    
        raise ValueError("Version must be 0/1/2")
    elif version == 2 and L == 1:    
        raise ValueError("Version must be 0/1 for L=1")

    device = F.device
    
    # Příprava listů/tenzorů pro LTI systém (opakování stejných matic)
    F_list = F.unsqueeze(0).repeat(L-1, 1, 1)
    G_list = G.unsqueeze(0).repeat(L-1, 1, 1)
    E_list = E.unsqueeze(0).repeat(L-1, 1, 1)
    
    H_list = H.unsqueeze(0).repeat(L, 1, 1)
    D_list = D.unsqueeze(0).repeat(L, 1, 1)
    nz_list = nz.repeat(L)
     
    # Volání jádra MDM
    r, Awv, aniMat, Gamma, calG = MDM_nullO(L, F_list, G_list, E_list, nz_list, H_list, D_list, z, u, version) 
    
    # Doplnění reziduí pro zbytek sekvence (pokud je vstup z delší než okno L)
    N = z.shape[0] if isinstance(z, torch.Tensor) else len(z)
    
    for i in range(1, N - (L - 1)): 
        # Z = [z_i, z_{i+1}, ..., z_{i+L-1}]
        Z = torch.cat([z[i + j].reshape(-1) for j in range(L)], dim=0)
        
        if version == 2 or L == 1:  
            r.append(aniMat @ Z)
        else:    
            # Pro verze 0 a 1 musíme odečíst vliv řídicího vstupu U
            aniMat_Gamma_calG = aniMat @ Gamma @ calG
            U = torch.cat([u[i + j].reshape(-1) for j in range(L - 1)], dim=0)
            r.append(aniMat @ Z - aniMat_Gamma_calG @ U)
            
    return r, Awv[0]

def MDM_nullO(L, F, G, E, nz, H, D, z, u, version):   
    """
    Jádro MDM algoritmu (Measurement Difference Method).
    """
    device = F.device
    N = nz.shape[0]

    r = []
    Awv = []
    
    # Počítáme jen pro první možné okno (i=0) - pro online AKF to stačí
    # Zbytek se v MDM_nullO_LTI dopočítá v cyklu
    for i in range(1): # V rekurzivní verzi nás zajímá jen aktuální okno
        if L == 1:
            if version == 0:
                aniMat = I_XpinvX(H[i], 0)
            elif version == 1:    
                # Sloupce nulového prostoru H.T, pak transponovat na řádky
                aniMat = null_space_torch(H[i].T).T
            
            # z[i] je tensor (obs_dim,)
            curr_z = z[i].reshape(-1)
            r.append(aniMat @ curr_z)
            Awv.append(aniMat @ D[i])
            
            Gamma = torch.tensor([], device=device)
            calG = torch.tensor([], device=device)
            
        else: # Pro okno L > 1
            # 1. Výpočet matic pozorovatelnosti a přenosu vstupu
            O, Gamma = O_Gamma_fun(L, F, nz, H, i)   
     
            if L == 2:
                calG = G[i] 
                calE = E[i]
            else:
                # blkDiag ze seznamu matic
                calG = torch.block_diag(*[G[j] for j in range(i, i + L - 1)])
                calE = torch.block_diag(*[E[j] for j in range(i, i + L - 1)])

            # 2. Výpočet anihilační matice (aniMat)
            if version == 0: 
                aniMat = I_XpinvX(O, 0)
            elif version == 1:    
                aniMat = null_space_torch(O.T).T
            elif version == 2:    
                # Verze 2 anihiluje i vliv řídicího vstupu
                M_v2 = torch.cat([O, Gamma @ calG], dim=1)
                aniMat = null_space_torch(M_v2.T).T

            # 3. Výpočet Awv (matice regressorů pro šum)
            calD = torch.block_diag(*[D[j] for j in range(i, i + L)])
            # Awv = aniMat @ [Gamma*calE, calD]
            Awv.append(aniMat @ torch.cat([Gamma @ calE, calD], dim=1))
        
            # 4. Výpočet rezidua r_k
            Z = torch.cat([z[i + j].reshape(-1) for j in range(L)], dim=0)
            if version == 2:  
                r.append(aniMat @ Z)
            else:    
                U = torch.cat([u[i + j].reshape(-1) for j in range(L - 1)], dim=0)
                r.append(aniMat @ (Z - Gamma @ calG @ U))
                
    return r, Awv, aniMat, Gamma, calG