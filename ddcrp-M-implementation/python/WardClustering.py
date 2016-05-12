import numpy as np
from scipy import spatial
from scipy import cluster
import StatsUtil
import math as mt

# Just a wrapper function that computes the parcellation with n parcels from
#   the linkage matrix Z
def Cluster(Z, n):
    z = cluster.hierarchy.fcluster(Z, t=n, criterion = 'maxclust')
    return z

# Computes the Ward clustering linkage matrix Z for a given connectivity
#   matrix D and spatial adjacency specified by adj_list. This output can then
#   be fed to the Cluster function to get a parcellation of a specific size
#   This is based largely on the Matlab linkage function.
def ClusterTree(D, adj_list):
    if StatsUtil.CheckSymApprox(D):
        X = D
    else:
        X = np.concatenate((D,D.transpose()),axis=1)

    # Compute squared euclidean distance Y between rows
    Qx = np.tile(np.linalg.norm(X, axis=1)**2,(X.shape[0],1))
    Y = Qx + Qx.transpose()-2*np.dot(X, X.transpose())
    Y = spatial.distance.squareform(Y,checks=False)
    Y[Y<0] = 0  # Correct for numerical errors in very similar rows
    
    # Construct adjacency matrix
    N = len(adj_list)
    A = np.zeros([N,N], dtype=bool)
    for i in range(N):
        A[i,adj_list[i]] = True
    connected = spatial.distance.squareform(A).astype(bool)
    

    # Initialize all data structures
    valid_clusts = np.ones(N, dtype=bool)   # which clusters still remain
    col_limits = np.cumsum(np.concatenate((np.array([N-2]), np.arange(N-2, 0, -1))))
    
    # During updating clusters, cluster index is constantly changing, R is
    # a index vector mapping the original index to the current (row, column)
    # index in Y.  C denotes how many points are contained in each cluster.
    m = mt.ceil(mt.sqrt(2*Y.shape[0]))
    C = np.zeros(2*m-1)
    C[0:m] = 1
    R = np.arange(m)
    all_inds = np.arange(Y.shape[0])
    conn_inds = all_inds[connected] # pairs of adjacent clusters that can be merged
    Z = np.zeros([m-1,4])

    for s in range(m-1):
        if conn_inds.size==0:
            # The graph was disconnected (e.g. two hemispheres)
            # Just add all connections to finish up cluster tree
            connected = np.zeros(len(connected))
            conn_inds = []
            valid_clust_inds = np.flatnonzero(valid_clusts)
            
            for i in valid_clust_inds:
                U = valid_clusts
                U[i] = 0
                new_conns = PdistInds(i, N, U)
                connected[new_conns] = True
                conn_inds = np.concatenate((conn_inds, new_conns))
            
            conn_inds = np.unique(conn_inds)

        # Find closest pair of clusters
        v = np.amin(Y[conn_inds])
        k = conn_inds[np.argmin(Y[conn_inds])]
    
        j = np.where(k <= col_limits)[0][0]
        i = N - (col_limits[j] - k) - 1
        
        Z[s,0:3] = np.array([R[i], R[j], v]) # Add row to output linkage
    
        # Update Y with this new cluster i containing old clusters i and j
        U = valid_clusts      
        U[np.array([i,j])] = 0
        I = PdistInds(i, N, U)
        J = PdistInds(j, N, U)
        Y[I] = ((C[R[U]]+C[R[i]])*Y[I] + 
                (C[R[U]]+C[R[j]])*Y[J] - C[R[U]]*v)/(C[R[i]]+C[R[j]]+C[R[U]])
        
        # Add j's connections to new cluster i
        new_conns = connected[J] & ~connected[I]
        connected[I] = connected[I] | new_conns
        conn_inds = np.sort(np.concatenate((conn_inds,I[new_conns])))
        
        # Remove all of j's connections from conn_inds and connected
        U[i]=1
        J = PdistInds(j, N, U)
        conn_inds = conn_inds[np.in1d(conn_inds,J, assume_unique=True,
                                                                invert=True)]
        connected[J] = np.zeros(len(J))
       
        valid_clusts[j] = 0 
        # update m, N, R
        C[m+s] = C[R[i]] + C[R[j]]
        Z[s,3] = C[m+s]
        R[i] = m+s

    Z[:,2] = np.sqrt(Z[:,2])
    return Z

# Compute positions in distance vector (for NxN matrix) for a given row. Results
#   are masked by the valid_flags boolean vector
def PdistInds(row, N, valid_flags):
    if row > 0:
        inds1 = np.concatenate((
            [row-1],
            (row-1) + np.cumsum(np.arange(N-2,N-row-1,-1))))
        I = np.concatenate((
            inds1,
            [-1],
            np.arange(inds1[-1]+N-row,inds1[-1]+2*N-2*row-1)))
    else:
        I = np.arange(N)-1
        
    I = I[valid_flags];
    return I