# SPCM-CRP

SPCM-CRP-MM : Transform Invariant Chinese Restaurant Process Mixture Model for Covariance Matrices  
Website: https://github.com/nbfigueroa/SPCM-CRP  
Author: Nadia Figueroa (nadia.figueroafernandez AT epfl.ch)

This repo provides code for running the Non-parametric Spectral Clustering algorithm on Covariance Matrix Datasets (SPCM-CRP-MM) introduced in [1]. In a nutshell, **SPCM-CRP-MM** is a similarity-dependent Chinese Restaurant Process Mixture Model. Where the similarity matrix comes from the Spectral Polytope Covariance Matrix Similarity function and the non-parametric clustering is applied on the spectral manifold induced from the similarity function.

#### Dependencies
- [LightSpeed Matlab Toolbox](https://github.com/tminka/lightspeed): Tom Minka's lightspeed library which includes highly optimized versions of mathematical functions.
- [ML_toolbox](https://github.com/epfl-lasa/ML_toolbox): Machine learning toolbox containing a plethora of dimensionality reduction, clustering, classification and regression algorithms accompanying the [Advanced Machine Learning](http://lasa.epfl.ch/teaching/lectures/ML_MSc_Advanced/index.php) course imparted at EPFL by Prof. Aude Billard.

To run your own experiments on the DTI datasets you must download the following toolbox:
- [fanDTasia](https://ch.mathworks.com/matlabcentral/fileexchange/26997-fandtasia-toolbox):A Matlab library for Diffusion Weighted MRI (DW-MRI) Processing, Diffusion Tensor (DTI) Estimation, Diffusion Kurtosis (DKI) Estimation, Higher-order Diffusion Tensor Analysis, Tensor ODF estimation, Visualization and more.


### Illustrative Example
To highlight the power of the proposed method, we consider a dataset of 5 Covariance Matrices of 3-dimensions, which can be illustrated as ellipsoids in 3D space:
<p align="center">
<img src="https://github.com/nbfigueroa/SPCM-CRP/blob/master/img/3d-ellipsoids.png" width="700">
</p>
The dataset is generated by **two distinct Covariance matrices** which are randomly transformed (rotation, scale, noise). The similar matrices are depicted by the colors of their ellipsoids.

**Our goal** is to cluster this dataset with a transform-invariant metric that will give us the two expected clusters.

### Spectral Polytope Covariance Matrix (SPCM) Similarity Function
Seldom Covariance Matrix similarity functions explictly have the property of transform-invariance. In this work, I propose such a similarity function which uses the idea of a **Spectral Polytope** together with the concept of **homothety**.
<p align="center">
<img src="https://github.com/nbfigueroa/SPCM-CRP/blob/master/img/sigmas_mod.png" width="700">
</p>

The **Spectral Polytope (SP)** is the Convex envelope of the projection of the Eigenvectors scaled by their Eigenvalues (X). The idea is, if the SPs of two covariance matrices have the same shape but are scaled by some homothetic factor, then they are similar (Refer to [1] for the math). By implementing this simple, yet geometrically intuitive idea, we get robust transform-invariant similarity values (second plot is the B-SPCM - a bounded version of SPCM):

<p align="center">
<img src="https://github.com/nbfigueroa/SPCM-CRP/blob/master/img/SPCM.png" width="300"><img src="https://github.com/nbfigueroa/SPCM-CRP/blob/master/img/BSPCM.png" width="300">
</p>

which are not well recovered by other metrics (RIEM, LERM, KLDM, JBLD):

<p align="center">
<img src="https://github.com/nbfigueroa/SPCM-CRP/blob/master/img/RIEM.png" width="200"><img src="https://github.com/nbfigueroa/SPCM-CRP/blob/master/img/LERM.png" width="200"><img src="https://github.com/nbfigueroa/SPCM-CRP/blob/master/img/KLDM.png" width="200"><img src="https://github.com/nbfigueroa/SPCM-CRP/blob/master/img/JBLD.png" width="200">
</p>

- RIEM: Affine Invariant Riemannian Metric
- LERM: Log-Euclidean Riemannian Metric
- KLDM: Kullback-Liebler Divergence Metric
- JBLD: Jensen-Bregman LogDet Divergence

### Similarity-based Non-parametric clustering (SPCM - CRP Mixture Model)
Now that we have a good similarity function for our task, we want to derive a clustering mechanism that is free of model selection and robust to intializations. Ideally, we could use Similarity-based clustering such as Affinity Propagation or Spectral Clustering, the performance of these methods, however, rely heavily on hyper-parameter tuning. Thus, we choose a variant of the Chinese Resturant Process, namely the **SPCM-CRP** [2] whose priors for cluster assigment are driven by the similarity values and the data is clustered on the Spectral Manifold of the Similarity matrix of the Dataset.

#### SPCM-CRP steps
- Initially, we apply an **Unsupervised Spectral Embedding** [1] algorithm, which automatically selects the dimensionality of the Spectral Manifold by applying a SoftMax on the Eigenvalues of the Laplacian of the Similarity matrix:

  <p align="center">
  <img src="https://github.com/nbfigueroa/SPCM-CRP/blob/master/img/auto-spectral-manifold.png" width="500">
  </p>

- Once we have the points on the Spectral Manifold corresponding to each Covariance Matrix, we apply the **SPCM-CRP**. Which follows the analogy for seating customers in a Chinese Restaurant with infinite number of tables wrt. a similarity between the customers
  <p align="center">
  <img src="https://github.com/nbfigueroa/SPCM-CRP/blob/master/img/ddcrp.png" width="500">
  </p>
  Each customer chooses to sit with another customer or alone according to a prior dependent on the **SPCM similarity function**. Table assignments z(c), then emerge from linked customers, regardless of sequence or ordering. As for any other Non-parametric approach the Posterior of the SPCM-CRP is intractable (refer to [1] for the math) and thus a Gibbs sampler is implemented for inference.

To run the full SPCM-CRP pipeline, follow the scipt ```demo_clust_spcmCRP.m```, for the 3D dataset you should get the following output on your MATLAB terminal:

```
Clustering via SPCM-CRP...
*** Initialized with 5 clusters out of 5 observations ***
Running dd-CRP Mixture Sampler... 
Iteration 1: Started with 5 clusters --> moved to 3 clusters with logprob = -28.06
Iteration 2: Started with 3 clusters --> moved to 2 clusters with logprob = -20.29
Iteration 3: Started with 2 clusters --> moved to 2 clusters with logprob = -19.90
Iteration 4: Started with 2 clusters --> moved to 2 clusters with logprob = -20.29
Iteration 5: Started with 2 clusters --> moved to 2 clusters with logprob = -20.56
Iteration 6: Started with 2 clusters --> moved to 2 clusters with logprob = -20.56
Iteration 7: Started with 2 clusters --> moved to 2 clusters with logprob = -20.17
Iteration 8: Started with 2 clusters --> moved to 2 clusters with logprob = -20.17
Iteration 9: Started with 2 clusters --> moved to 2 clusters with logprob = -20.29
Iteration 10: Started with 2 clusters --> moved to 2 clusters with logprob = -26.15
Iteration 11: Started with 2 clusters --> moved to 2 clusters with logprob = -20.60
Iteration 12: Started with 2 clusters --> moved to 2 clusters with logprob = -20.17
Iteration 13: Started with 2 clusters --> moved to 2 clusters with logprob = -20.32
Iteration 14: Started with 2 clusters --> moved to 2 clusters with logprob = -20.54
Iteration 15: Started with 2 clusters --> moved to 2 clusters with logprob = -20.60
Iteration 16: Started with 2 clusters --> moved to 2 clusters with logprob = -19.90
Iteration 17: Started with 2 clusters --> moved to 2 clusters with logprob = -20.54
Iteration 18: Started with 2 clusters --> moved to 1 clusters with logprob = -24.85
Iteration 19: Started with 1 clusters --> moved to 1 clusters with logprob = -24.14
Iteration 20: Started with 1 clusters --> moved to 2 clusters with logprob = -23.72
Elapsed time is 0.162789 seconds.
*************************************************************
---spcm-CRP-MM Results---
 Iter:3, LP: -1.989588e+01, Clusters: 2 with Purity: 1.00, NMI Score: 1.00, F measure: 1.00 
*************************************************************
```

The result is:

  <p align="center">
  <img src="https://github.com/nbfigueroa/SPCM-CRP/blob/master/img/convergence-3dtoy.png" width="700">
  </p>
  
  <p align="center">
  <img src="https://github.com/nbfigueroa/SPCM-CRP/blob/master/img/results-3dtoy.png" width="700">
  </p>

**without** selecting or optimizing for **ANY** hyper-parameters.

### Comparisons

- In ```demo_spcm_compare.m`` I provide extensive comparisons between the SPCM similarity function and the 4 other standard Covariance Matrix Similarity functions used in literature for datasets of increasing dimensionality and samples [1], for example a 6D Covariance matrix dataset with 30 samples are well discriminated with SPCM and B-SPCM:
 
  <p align="center">
  <img src="https://github.com/nbfigueroa/SPCM-CRP/blob/master/img/SPCM_30.png" width="300"><img src="https://github.com/nbfigueroa/SPCM-CRP/blob/master/img/BSPCM_30.png" width="300">
  </p>
  
  while the othe functions (RIEM, LERM, KLDM, JBLD) do not show apparent partitions to the naked eye:
  
  <p align="center">
  <img src="https://github.com/nbfigueroa/SPCM-CRP/blob/master/img/RIEM_30.png" width="200"><img src="https://github.com/nbfigueroa/SPCM-CRP/blob/master/img/LERM_30.png" width="200"><img src="https://github.com/nbfigueroa/SPCM-CRP/blob/master/img/KLDM_30.png" width="200"><img src="https://github.com/nbfigueroa/SPCM-CRP/blob/master/img/JBLD_30.png" width="200">
  </p>

- Also, you can compare the SPCM-CRP to standard Similarity-based clustering algorithms like Affinity Propapgation and Spectral Clustering with k-means [1].

### References
[1] Nadia Figueroa and Aude Billard, "Transform-Invariant Non-Parametric Clustering of Covariance Matrices and its Application to Unsupervised Joint Segmentation and Action Discovery." *In preparation for Pattern Recognition*.  
