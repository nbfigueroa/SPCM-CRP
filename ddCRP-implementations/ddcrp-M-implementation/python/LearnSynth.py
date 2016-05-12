import numpy as np
import collections
import WardClustering
import StatsUtil
import InitializeAndRunddCRP as initdd
from multiprocessing import Pool

# Format of generated synthetic datasets
SynthData = collections.namedtuple('SynthData',['D','adj_list','z','coords'])

# Main function: Computes a parcellation of synthetic data at different noise
#   levels, using Ward Clustering and our method based on the ddCRP. Each
#   parcellation is evaluated based on its Normalized Mututal Information
#   with the ground truth. The input "type"={'square','stripes','face'}
#   determines the underlying ground truth parcellation.
def LearnSynth(type):
    np.random.seed(1)   # For repeatability
    max_noise = 10;     # Number of noise levels to try
    repeats = 5;       # Number of times to repeat experiments
    
    WC = np.zeros((max_noise,repeats))
    DC = np.zeros((max_noise,repeats))
    DC_K = np.zeros((max_noise,repeats))

    for rep in range(repeats):
        print('Repeat #' + str(rep))
        all_synth = [GenerateSynthData(type, noise_sig) 
                        for noise_sig in range(max_noise)]

        # Run all noise levels in parallel
        p = Pool(processes=max_noise)
        all_res = p.map(LearnSynthForDataset, all_synth)
        p.close()
        p.join()
        WC[:,rep] = [res[0] for res in all_res]
        DC[:,rep] = [res[1] for res in all_res]
        DC_K[:,rep] = [res[2] for res in all_res]

    return (WC, DC, DC_K)


# Compute Ward clustering and our parcellation for a specific synthetic
#   (previously generated) dataset
def LearnSynthForDataset(synth):
    # Hyperparameters
    alpha = 10;
    kappa = 0.0001;
    nu = 1;
    sigsq = 0.01;
    pass_limit = 30;

    D = NormalizeConn(synth.D)  # Normalize connectivity to zero mean, unit var

    # Compute our ddCRP-based parcellation
    Z = WardClustering.ClusterTree(D, synth.adj_list)
    _,dd_stats = initdd.InitializeAndRun(Z, D, synth.adj_list, range(1,21),
                    alpha, kappa, nu, sigsq, pass_limit, synth.z, 0)
    DC = dd_stats['NMI'][-1]
    DC_K = dd_stats['K'][-1]

    # Ward Clustering, using number of clusters discovered from our method
    WC = StatsUtil.NMI(synth.z, WardClustering.Cluster(Z, DC_K))

    return (WC,DC,DC_K)

# Generate synthetic dataset of "type"={'square','stripes','face'} at a given
#   noise level "sig". Returns a SynthData object containing a connectivity
#   matrix D, and adjacency list adj_list, ground truth parcellation z, and
#   element coordinates coords
def GenerateSynthData(type, sig):
    sqrtN = 18

    coords = np.zeros((sqrtN**2,2))
    adj_list = np.empty(sqrtN**2, dtype=object)
    for r in range(0, sqrtN):
        for c in range(0, sqrtN):
            currVox = c + r*sqrtN
            coords[currVox,:] = [r, c]
            curr_adj = []
            if r > 0:
                curr_adj.append(c + (r-1)*sqrtN)
            if r < (sqrtN-1):
                curr_adj.append(c + (r+1)*sqrtN)
            if c > 0:
                curr_adj.append((c-1) + r*sqrtN)
            if c < (sqrtN-1):
                curr_adj.append((c+1) + r*sqrtN)
            adj_list[currVox] = np.array(curr_adj)
    
    if type == 'square':
        z = np.array([
        0,0,0,0,0,0,3,3,3,3,3,3,6,6,6,6,6,6,
        0,0,0,0,0,0,3,3,3,3,3,3,6,6,6,6,6,6,
        0,0,0,0,0,0,3,3,3,3,3,3,6,6,6,6,6,6,
        0,0,0,0,0,0,3,3,3,3,3,3,6,6,6,6,6,6,
        0,0,0,0,0,0,3,3,3,3,3,3,6,6,6,6,6,6,
        0,0,0,0,0,0,3,3,3,3,3,3,6,6,6,6,6,6,
        1,1,1,1,1,1,4,4,4,4,4,4,7,7,7,7,7,7,
        1,1,1,1,1,1,4,4,4,4,4,4,7,7,7,7,7,7,
        1,1,1,1,1,1,4,4,4,4,4,4,7,7,7,7,7,7,
        1,1,1,1,1,1,4,4,4,4,4,4,7,7,7,7,7,7,
        1,1,1,1,1,1,4,4,4,4,4,4,7,7,7,7,7,7,
        1,1,1,1,1,1,4,4,4,4,4,4,7,7,7,7,7,7,
        2,2,2,2,2,2,5,5,5,5,5,5,8,8,8,8,8,8,
        2,2,2,2,2,2,5,5,5,5,5,5,8,8,8,8,8,8,
        2,2,2,2,2,2,5,5,5,5,5,5,8,8,8,8,8,8,
        2,2,2,2,2,2,5,5,5,5,5,5,8,8,8,8,8,8,
        2,2,2,2,2,2,5,5,5,5,5,5,8,8,8,8,8,8,
        2,2,2,2,2,2,5,5,5,5,5,5,8,8,8,8,8,8])
    elif type == 'stripes':
        z = np.array([
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        0,0,0,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,
        0,0,0,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,
        0,0,0,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,
        0,0,0,1,1,1,2,2,2,3,3,3,3,3,3,3,3,3,
        0,0,0,1,1,1,2,2,2,3,3,3,3,3,3,3,3,3,
        0,0,0,1,1,1,2,2,2,3,3,3,3,3,3,3,3,3,
        0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,4,4,4,
        0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,4,4,4,
        0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,4,4,4,
        0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,
        0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,
        0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5])
    elif type == 'face':
        z = np.array([
        0,0,0,0,0,0,3,3,3,3,3,3,6,6,6,6,6,6,
        0,0,0,0,0,0,3,3,3,3,3,3,6,6,6,6,6,6,
        0,0,0,0,0,0,3,3,3,3,3,3,6,6,6,6,6,6,
        0,0,0,0,0,0,3,3,3,3,3,3,6,6,6,6,6,6,
        0,0,0,0,0,0,3,3,3,3,3,3,6,6,6,6,6,6,
        0,0,0,0,0,0,3,3,3,3,3,3,6,6,6,6,6,6,
        1,1,1,1,1,1,4,4,4,4,4,4,7,7,7,7,7,7,
        1,1,1,1,1,1,4,4,4,4,4,4,7,7,7,7,7,7,
        1,1,1,1,1,1,4,4,4,4,4,4,7,7,7,7,7,7,
        1,1,1,1,1,1,4,4,4,4,4,4,7,7,7,7,7,7,
        1,1,1,1,1,1,4,4,4,4,4,4,7,7,7,7,7,7,
        1,1,1,1,1,1,4,4,4,4,4,4,7,7,7,7,7,7,
        2,2,2,2,2,2,5,5,5,5,5,5,8,8,8,8,8,8,
        2,2,2,2,2,2,5,5,5,5,5,5,8,8,8,8,8,8,
        2,2,2,2,2,2,5,5,5,5,5,5,8,8,8,8,8,8,
        2,2,2,2,2,2,5,5,5,5,5,5,8,8,8,8,8,8,
        2,2,2,2,2,2,5,5,5,5,5,5,8,8,8,8,8,8,
        2,2,2,2,2,2,5,5,5,5,5,5,8,8,8,8,8,8])
    
    N = len(z)
    K = len(np.unique(z))
    
    A = np.random.normal(size=(K,K))
    
    D = np.zeros((N,N))
    for v1 in range(0,N):
        for v2 in range(0,N):
            if v1 != v2:
                D[v1,v2] = sig*np.random.normal() + A[z[v1],z[v2]]
    
    synth = SynthData(D, adj_list, z, coords)
    return synth


# Normalize connectivity matrix "D" to have zero mean and unit variance
def NormalizeConn(D):
    D = D.astype('float64')
    N = D.shape[0]
    off_diags = np.logical_not(np.eye(N,dtype='bool'))
    D = D - D[off_diags].mean()
    D = D/D[off_diags].std()
    np.fill_diagonal(D, 0)

    D = D.astype('float32')
    return D

# If this file is run at the command line, perform a demo
if __name__ == "__main__":
    WC,DC,DC_K = LearnSynth('stripes');
    print('WC: ' + str(np.mean(WC,axis=1)))
    print('DC: ' + str(np.mean(DC,axis=1)))
