# @author: Oliver Kost, kost@ntis.zcu.cz

import numpy as np

def genData_LTV(N):
    np.random.seed(42)

    nx = 2
    nz = np.zeros((N),int);
    nw = 2
    nv = 2
    
    x = np.zeros((N,nx));
    u = [];
    F = np.zeros((N,nx,nx));
    G = [];
    E = np.zeros((N,nx,nw));
    x[0] = np.array([0, 10])
    
    z = []
    H = []
    D = []
    
    Q = np.array([[ 2, -1],
                  [-1,  2]])  
    R = np.array([[2, 1],
                  [1, 3]])  
    
    chQ = np.linalg.cholesky(Q)
    chR = np.linalg.cholesky(R)
    
    w1 = np.vstack([1, 1])
    v1 = np.vstack([2, -1])
    
    w = w1 + chQ @ np.random.randn(nw,N)
    
    #v = v1 + chR @ np.random.randn(nv,N)
    v = 1/np.sqrt(3)*chR@np.random.randn(nv,N+3)   # correlation across 3 time instants
    v = v1 + (v[:,2:] + v[:,1:-1] + v[:,:-2])      # correlation across 3 time instants
    
    for i in range(N):   
        
        F[i] = np.array([[0.99         , 0  ],
                         [np.cos(9*i/N), 0.9]]) + 1e-3*np.random.randn(nx,nx)
     
        E[i] = np.array([[2             ,  np.sin(39*i/N)],
                         [np.sin(6*i/N) , -1             ]]) + np.random.randn(nx,nw) 
         
        if i<N/2:
            u.append( np.array([2 + np.sin(43*i/N)]) + np.random.randn() )    
            G.append( np.array([[np.cos(31*i/N)],
                                [1             ]]) + 0*np.random.randn(nx,1) )
            H.append( np.array([[np.cos(61*i/N) , 1             ],
                                [np.cos(25*i/N) , 1             ],
                                [np.cos(45*i/N) , np.sin(75*i/N)]]) + np.random.randn(3,nv) )
            D.append( np.array([[ 1, 2               ],
                                [ 2, 1+np.sin(33*i/N)],
                                [-1, 3               ]]) + np.random.randn(3,nv) )   
        elif i<N*3/4:
            u.append( np.array([2 + np.sin(43*i/N)]) + np.random.randn() )    
            G.append( np.array([[np.cos(31*i/N)],
                                [1             ]]) + 0*np.random.randn(nx,1) )
            H.append( np.array([[np.cos(61*i/N) , 1             ]]) + np.random.randn(1,nv) )
            D.append( np.array([[ 2, 1+np.sin(33*i/N)]]) + np.random.randn(1,nv) )  
        else:
            u.append( np.array([6 + np.sin(99*i/N), np.random.randn()]) )    
            G.append( np.array([[np.cos(78*i/N)  , 1],
                                 np.random.randn(nx)]) )
            H.append( np.array([[np.cos(12*i/N) , 1   ],
                                [ 2 , 1+np.sin(36*i/N)],
                                [ 1 , 2               ],
                                [ 0 , 3               ]]) + np.random.randn(4,nv) )  
            D.append( np.array([[ 1, 2],
                                [ 0, 2],
                                [ 1, 1],
                                [-1, 3]]) + np.random.randn(4,nv) ) 
        if i<(N-1):
            x[i+1] = F[i] @ x[i] + G[i] @ u[i] + E[i] @ w[:,i]            
        z.append( H[i] @ x[i] + D[i] @ v[:,i] )
        
        nz[i] = H[i].shape[0]
    
    return F,G,E,nz,H,D,z,w1,v1,Q,R,nw,nv,u,w,v


def genData_LTI(N):
    np.random.seed(42)

    nx = 2
    nz = 4
    nw = 2
    nv = 2
    
    x = np.zeros((N,nx));
    u = [];
    x[0] = np.array([0, 10])
    
    z = []
    
    Q = np.array([[ 2, -1],
                  [-1,  2]])  
    R = np.array([[2, 1],
                  [1, 3]])  
    
    chQ = np.linalg.cholesky(Q)
    chR = np.linalg.cholesky(R)
    
    w1 = np.vstack([1, 1])
    v1 = np.vstack([2, -1])
    
    w = w1 + chQ @ np.random.randn(nw,N)
    
    #v = v1 + chR @ np.random.randn(nv,N)
    v = 1/np.sqrt(3)*chR@np.random.randn(nv,N+3)   # correlation across 3 time instants
    v = v1 + (v[:,2:] + v[:,1:-1] + v[:,:-2])      # correlation across 3 time instants
    
    for i in range(N):    
        F = np.array([[0.99, 0  ],
                      [1   , 0.9]])
        E = np.array([[-1, 2],
                      [ 1, 0]])
        G = np.array([[-1],
                      [ 1]])
        u.append( np.array([2 + np.sin(43*i/N)]) + np.random.randn() )   
        
        H = np.array([[ 1, 2],
                      [ 0, 2],
                      [ 1, 1],
                      [-1, 3]])
        
        D = np.array([[ 1, 0],
                      [ 6, 2],
                      [ 0, 5],
                      [-1, 2]])
    
        if i<(N-1):
            x[i+1] = F @ x[i] + G @ u[i] + E @ w[:,i]            
        z.append( H @ x[i] + D @ v[:,i] )
    
    return F,G,E,nz,H,D,z,w1,v1,Q,R,nw,nv,u,w,v

