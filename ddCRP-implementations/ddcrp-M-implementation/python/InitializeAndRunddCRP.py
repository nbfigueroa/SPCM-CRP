import numpy as np
from scipy import cluster
from scipy import sparse
import random
import ddCRP
import StatsUtil

# Main function: Initializes our method using a Ward clustering linkage matrix
#   Z, a (normalized) connectivity matrix D_norm, and adjacency list defining
#   spatial adjacency, the possible numbers of parcels to consider for
#   initialization ("sizes"), hyperparameters alpha, kappa, nu, and sigsq, the
#   number of passes over the dataset MCMC should be run, the ground truth
#   parcellation gt_z (if known, empty otherwise), and a verbose flag which
#   determines whether update information is printed every 1000 iterations.
#   Returns the MAP parcellation map_z, as well as a stats objects with
#   information about the iterations of the model
def InitializeAndRun(Z, D_norm, adj_list, sizes, alpha, kappa, nu, sigsq, pass_limit, gt_z, verbose):

    # Find highest-probability greedy parcellation to initialize
    logp = LogProbWC(D_norm, Z, sizes, alpha, kappa, nu, sigsq)
    max_i = np.argmax(logp)
    z = cluster.hierarchy.fcluster(Z, t=sizes[max_i], criterion = 'maxclust')
  
    # Construct a spanning tree within each cluster as initialization for c
    c = ClusterSpanningTrees(z, adj_list)
    map_z, stats = ddCRP.ddCRP(D_norm, adj_list, c, gt_z, pass_limit, alpha, kappa, nu, sigsq, 1000, verbose)             
    return(map_z, stats)


# Compute probability of each Ward clustering (from Z) of a matrix D at
#   various sizes, using our model with hyperparameters alpha, kappa, nu, sigsq
def LogProbWC(D, Z, sizes, alpha, kappa, nu, sigsq):
    hyp = ddCRP.ComputeCachedLikelihoodTerms(kappa, nu, sigsq)
    logp = np.zeros(len(sizes))
    
    for i in range(len(sizes)):
        z = cluster.hierarchy.fcluster(Z, t=sizes[i], criterion = 'maxclust')

        sorted_i = np.argsort(z)
        sorted_z = np.sort(z)
        parcels = np.split(sorted_i,np.flatnonzero(np.diff(sorted_z))+1)
        
        # Formally we should construct a spanning tree within each cluster so
        #   that we can evaluate the probability. However, the only property of
        #   the "c" links that impacts the probability directly is the number of
        #   self-connections. So we simply add the correct number of self-
        #   connections (equal to the number of parcels) and leave the rest
        #   set to zero
        c = np.zeros(len(z))
        c[0:sizes[i]] = np.arange(sizes[i])
            
        logp[i] = ddCRP.FullProbabilityddCRP(D, c, parcels, alpha, hyp,
                                                StatsUtil.CheckSymApprox(D))

    return logp


# In order to use the Ward clustering z as an initialization to our model, we
#   need to generate voxel links "c" that are consistent with the Ward
#   clustering. There are many way to do this, but a simple one is to
#   construct a minimum spanning tree within each cluster, and set each
#   element's "c" link to point to its parent in the tree.
def ClusterSpanningTrees(z, adj_list):

    # We're going to remove edges from adj_list, so make a copy
    adj_list = adj_list.copy()

    nvox = len(adj_list)
    # Remove all adjacencies that cross clusters
    for i in range(nvox):
        adj_list[i] = adj_list[i][z[adj_list[i]]==z[i]]
        adj_list[i]  = np.random.permutation(adj_list[i])

    # Construct sparse adjacency matrix
    neighbor_count = [len(neighbors) for neighbors in adj_list]
    node_list = np.zeros(sum(neighbor_count))
    next_edge = 0
    for i in range(nvox):
        if neighbor_count[i] > 0:
            node_list[next_edge:(next_edge+neighbor_count[i])] = i
            next_edge = next_edge + neighbor_count[i]
    G = sparse.csc_matrix((np.ones(len(node_list)),
                            (node_list,np.hstack(adj_list))), shape=(nvox,nvox)) 
    
    # Construct spanning tree in each cluster
    minT = sparse.csgraph.minimum_spanning_tree(G)
    c = np.zeros(len(adj_list))
    for clust in np.unique(z):
        clust_vox = np.flatnonzero(z==clust)
        rand_root=clust_vox[random.randint(1,len(clust_vox)-1)]
        _,parents = sparse.csgraph.breadth_first_order(minT,rand_root,
                                                        directed=False) 
        c[clust_vox] = parents[clust_vox] 
    
    # Roots have parent value of -9999, set them to be their own parent
    roots = np.flatnonzero(c==-9999) 
    c[roots] = roots

    return c
