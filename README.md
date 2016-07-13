# SPCM-CRP

SPCM-CRP : Transform Invariant Chinese Restaurant Process for Covariance Matrices  
Website: http://nbfigueroa.github.io/SPCM-CRP/  
Author: Nadia Figueroa (nadia.figueroafernandez AT epfl.ch)

This repo provides code for running the Non-parametric Spectral Clustering algorithm on Covariance Matrix Datasets (SPCM-CRP) introduced in [1]. In a nutshell, **SPCM-CRP** is a similarity-dependent Chinese Restaurant process. Where the similarity matrix comes from the Spectral Polytope Covariance Matrix Similarity function and the non-parametric clustering is applied on the spectral manifold of the similarity function.

### Illustrative Example
To highlight the power of the proposed method, we consider a dataset of 5 Covariance Matrices of 3-dimensions, which can be illustrated as ellipsoids in 3D space:
<p align="center">
<img src="https://github.com/nbfigueroa/SPCM-CRP/blob/master/img/3d-ellipsoids.png" width="700">
</p>
The dataset is generated by **two distinct Covariance matrices** which are randomly transformed (rotation, scale, noise). The similar matrices are depicted by the colors of their ellipsoids.

**Our goal** is to cluster this dataset with a transform-invariant metric that will give us the two expected clusters.

#### Spectral Polytope Covariance Matrix (SPCM) Similarity Function
Seldom Covariance Matrix similarity functions explictly have the property of transform-invariance. In this work, I propose such a similarity function which uses the idea of a **Spectral Polytope** together with the concept of **homothety**.
<p align="center">
<img src="https://github.com/nbfigueroa/SPCM-CRP/blob/master/img/sigmas_mod.png" width="700">
</p>

The **Spectral Polytope (SP)** is the Convex envelope of the projection of the Eigenvectors scaled by their Eigenvalues (X). The idea is, if the SPs of two covariance matrices have the same shape but are scaled by some homothetic factor, then they are similar (Refer to [1] for the math). By implementing this simple, yet elegant idea, we get robust transform-invariant similarity values (second plot is the B-SPCM - a bounded version of SPCM):

<p align="center">
<img src="https://github.com/nbfigueroa/SPCM-CRP/blob/master/img/SPCM.png" width="300"><img src="https://github.com/nbfigueroa/SPCM-CRP/blob/master/img/BSPCM.png" width="300">
</p>

which are not well recovered by other metrics (RIEM, LERM, KLDM, JBLD):

<p align="center">
<img src="https://github.com/nbfigueroa/SPCM-CRP/blob/master/img/RIEM.png" width="225"><img src="https://github.com/nbfigueroa/SPCM-CRP/blob/master/img/LERM.png" width="225"><img src="https://github.com/nbfigueroa/SPCM-CRP/blob/master/img/KLDM.png" width="225"><img src="https://github.com/nbfigueroa/SPCM-CRP/blob/master/img/JBLD.png" width="225">
</p>

- RIEM: Affine Invariant Riemannian Metric
- LERM: Log-Euclidean Riemannian Metric
- KLDM: Kullback-Liebler Divergence Metric
- JBLD: Jensen-Bregman LogDet Divergence

```
Computing SPCM Similarity Function for 5x5 observations...
Elapsed time is 0.009259 seconds.
*************************************************************
```

```
Computing Spectral Dimensionality Reduction based on SPCM Similarity Function...
Elapsed time is 0.075289 seconds.
*************************************************************
```

```
Clustering via sd-CRP...
*** Initialized with 5 clusters out of 5 observations ***
Iteration 1: Started with 5 clusters --> moved to 3 clusters with logprob = -23.35
Iteration 2: Started with 3 clusters --> moved to 2 clusters with logprob = -22.03
Iteration 3: Started with 2 clusters --> moved to 2 clusters with logprob = -22.03
Iteration 4: Started with 2 clusters --> moved to 2 clusters with logprob = -22.57
Iteration 5: Started with 2 clusters --> moved to 3 clusters with logprob = -23.91
Iteration 6: Started with 3 clusters --> moved to 2 clusters with logprob = -22.22
Iteration 7: Started with 2 clusters --> moved to 2 clusters with logprob = -22.57
Iteration 8: Started with 2 clusters --> moved to 2 clusters with logprob = -21.68
Iteration 9: Started with 2 clusters --> moved to 3 clusters with logprob = -23.89
Iteration 10: Started with 3 clusters --> moved to 2 clusters with logprob = -21.68
Iteration 11: Started with 2 clusters --> moved to 2 clusters with logprob = -22.25
Iteration 12: Started with 2 clusters --> moved to 3 clusters with logprob = -23.04
Iteration 13: Started with 3 clusters --> moved to 2 clusters with logprob = -22.22
Iteration 14: Started with 2 clusters --> moved to 2 clusters with logprob = -22.59
Iteration 15: Started with 2 clusters --> moved to 2 clusters with logprob = -22.59
Iteration 16: Started with 2 clusters --> moved to 2 clusters with logprob = -22.25
Iteration 17: Started with 2 clusters --> moved to 2 clusters with logprob = -22.03
Iteration 18: Started with 2 clusters --> moved to 3 clusters with logprob = -23.61
Iteration 19: Started with 3 clusters --> moved to 2 clusters with logprob = -22.25
Iteration 20: Started with 2 clusters --> moved to 2 clusters with logprob = -21.68
Elapsed time is 0.179722 seconds.
MAP Cluster estimate recovered at iter 8: 2
sd-CRP LP: -2.168154e+01 and Purity: 1.00, NMI Score: 1.00, F measure: 1.00 
*************************************************************
```

### Comparisons

In ```demo_comparisons.m`` I provide extensive comparisons between the SPCM similarity function and 4 other standard Covariance Matrix Similarity functions used in literature for various datasets [1].

- pics here

Also, you can compare the SPCM-CRP to standard Similarity-based clustering algorithms like Affinity Propapgation and Spectral Clustering with k-means [1].

- pics here


#### Dependencies
To run the **comparisons** demo, download the following toolbox and make sure to have it in your MATLAB path:
- [ML_toolbox](https://github.com/epfl-lasa/ML_toolbox): Machine learning toolbox containing a plethora of dimensionality reduction, clustering, classification and regression algorithms accompanying the [Advanced Machine Learning](http://lasa.epfl.ch/teaching/lectures/ML_MSc_Advanced/index.php) course imparted at EPFL by Prof. Aude Billard.

If you're not interested in running comparisons, this step is not needed.

### Publication
[1] Nadia Figueroa and Aude Billard, "Transform Invariant Discovery of Dynamical Primitives: Leveraging Spectral and Bayesian Non-parametric Methods." *In preparation*.
