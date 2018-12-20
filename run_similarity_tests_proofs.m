%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Compare similarity functions and clustering algorithms  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% To run some of the function on this script you need 
% the ML_toolbox in your MATLAB path.
clc;  clear all; close all
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                    --Select a Dataset to Test--                       %%     
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1) Toy 3D dataset, 5 Samples, 2 clusters (c1:3, c2:2)
% This function loads the 3-D ellipsoid dataset used to generate Fig. 3, 4 
% and 5 from Section 4 and the results in Section 7 in the accompanying paper.
clc; clear all; close all;
display = 0; randomize = 0; dataset_name = 'Toy 3D';
[sigmas, true_labels] = load_toy_dataset('3d', display, randomize);

%% 2)  Toy 6D dataset, 30 Samples, 3 clusters (c1:10, c2:20, c3: 10)
% This function loads the 6-D ellipsoid dataset used to generate Fig. 6 and 
% from Section 4 and the results in Section 8 in the accompanying paper.
clc; clear all; close all;
display = 0; randomize = 0; dataset_name = 'Toy 6D';
[sigmas, true_labels] = load_toy_dataset('6d', display, randomize);

%% 3) Real 6D dataset, task-ellipsoids, 105 Samples, 3 clusters 
%% Cluster Distibution: (c1:63, c2:21, c3: 21)
% This function loads the 6-D task-ellipsoid dataset used to evaluate this 
% algorithm in Section 8 of the accompanying paper.
%
% Please cite the following paper if you make use of this data:
% El-Khoury, S., de Souza, R. L. and Billard, A. (2014) On Computing 
% Task-Oriented Grasps. Robotics and Autonomous Systems. 2015 

clc; clear all; close all;
data_path = './data/'; randomize = 0; dataset_name = 'Real 6D (Task-Ellipsoids)';
[sigmas, true_labels] = load_task_dataset(data_path, randomize);

%% 4a) Toy 3D dataset, Diffusion Tensors from Synthetic Dataset, 1024 Samples
%% Cluster Distibution: 4 clusters (each cluster has 10 samples)
% This function will generate a synthetic DW-MRI (Diffusion Weighted)-MRI
% This is done following the "Tutorial on Diffusion Tensor MRI using
% Matlab" by Angelos Barmpoutis, Ph.D. which can be found in the following
% link: http://www.cise.ufl.edu/~abarmpou/lab/fanDTasia/tutorial.php
%
% To run this function you should download fanDTasia toolbox in the 
% ~/SPCM-CRP/3rdParty directory, this toolbox is also provided in 
% the tutorial link.

clc; clear all; close all;
data_path = './data/'; type = 'synthetic'; display = 1; randomize = 0; 
[sigmas, true_labels] = load_dtmri_dataset( data_path, type, display, randomize );
dataset_name = 'Synthetic DT-MRI';

%% 4b) Real 3D dataset, Diffusion Tensors from fanTDasia Dataset, 1024 Samples
%% Cluster Distibution: 4 clusters (each cluster has 10 samples)
% This function loads a 3-D Diffusion Tensor Image from a Diffusion
% Weight MRI Volume of a Rat's Hippocampus, the extracted 3D DTI is used
% to evaluate this algorithm in Section 8 of the accompanying paper.
%
% To load and visualize this dataset, you must download the dataset files 
% in the  ~/SPCM-CRP/data directory. These are provided in the online 
% tutorial on Diffusion Tensor MRI in Matlab:
% http://www.cise.ufl.edu/~abarmpou/lab/fanDTasia/tutorial.php
%
% One must also download the fanDTasia toolbox in the ~/SPCM-CRP/3rdParty
% directory, this toolbox is also provided in this link.

clc; clear all; close all;
data_path = './data/'; type = 'real'; display = 1; randomize = 0; 
[sigmas, true_labels] = load_dtmri_dataset( data_path, type, display, randomize );
dataset_name = 'Real DT-MRI';

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Compute Similarity Matrices with different measures   %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%% Set Hyper-parameter %%%%%%%%%%%%%%%%%%%%%%%%
% Tolerance for SPCM decay function 
tau = 1; % [1, 100] Set higher for noisy data, Set 1 for ideal data 

% %%%%%% Compute Confusion Matrix of Similarities %%%%%%%%%%%%%%%%%%
spcm = ComputeSPCMfunctionMatrix(sigmas, tau);  
S_spcm   = spcm(:,:,1);
if exist('h0','var') && isvalid(h0), delete(h0);end
title_str = 'SPCM Similarity Function';
h0 = plotSimilarityConfMatrix(S_spcm, title_str);

%%%%%%%%%%%%%%% Bounded - SPCM function %%%%%%%%%%%%%%%
S_b_spcm  = spcm(:,:,2);
if exist('h0_b','var') && isvalid(h0_b), delete(h0_b);end
title_str = 'Bounded SPCM Similarity Function';
h0_b = plotSimilarityConfMatrix(S_b_spcm, title_str);

%%%%%%%%%%%%%%% Affine Invariant Riemannian Metric %%%%%%%%%%%%%%%
tic;
S_riem = compute_cov_sim( sigmas, 'RIEM' );
toc;
if exist('h1','var') && isvalid(h1), delete(h1);end
title_str = 'Affine Invariant Riemannian Metric (RIEM)';
h1 = plotSimilarityConfMatrix(S_riem, title_str);

%%%%%%%%%%%%%%% 'LERM': Log-Euclidean Riemannina Metric %%%%%%%%%%%%%%%
tic;
S_lerm = compute_cov_sim( sigmas, 'LERM' );
toc;
if exist('h2','var') && isvalid(h2), delete(h2);end
title_str = 'Log-Euclidean Riemannian Metric (LERM)';
h2 = plotSimilarityConfMatrix(S_lerm, title_str);

%%%%%%%%%%%%%%% 'KLDM': Kullback-Liebler Divergence Metric %%%%%%%%%%%%%%%
tic;
S_kldm = compute_cov_sim( sigmas, 'KLDM' );
toc;
if exist('h3','var') && isvalid(h3), delete(h3);end
title_str = 'Kullback-Liebler Divergence Metric (KLDM)';
h3 = plotSimilarityConfMatrix(S_kldm, title_str);

%%%%%%%%%%%%%%% 'JBLD': Jensen-Bregman LogDet Divergence %%%%%%%%%%%%%%%
tic;
S_jbld = compute_cov_sim( sigmas, 'JBLD' );
toc;
if exist('h4','var') && isvalid(h4), delete(h4);end
title_str = 'Jensen-Bregman LogDet Divergence (JBLD)';
h4 = plotSimilarityConfMatrix(S_jbld, title_str);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Dim. Red Option 1: Vector space repr. of SPD matrices using log-Euclidean framework 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N     = length(sigmas{1});
d_log = N*(N+1)/2;
vec_sigmas = zeros(d_log,length(sigmas));
for s=1:length(sigmas)
    sigma     = sigmas{s};
    log_sigma = logm(sigma);   
    
    % logged-SPD to Vector
    vec_sigmas(:,s) = symMat2Vec(log_sigma);
end

[ V, L, Mu ] = my_pca( vec_sigmas );
figure('Color',[1 1 1]);
plot(diag(L),'Color',[1 0 0]); grid on;
title('Eigenvalues of PCA on log-vector space')
[ p ] = explained_variance( L, 0.90 );
[A_p, Y] = project_pca(vec_sigmas, Mu, V, p);
fprintf('Vector-space dim (%d) - lower dimension (%d)\n',d_log,p);

%% %%%%%% Visualize Lower-D Embedding %%%%%%%%
plot_options        = [];
plot_options.labels = true_labels;
plot_options.title  = 'PCA on log-vector space'; 
ml_plot_data(Y',plot_options);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Dim. Red Option 2: Principal Geodesic Analysis on Riemannian Manifold 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N     = length(sigmas{1});
d_log = N*(N+1)/2;

% Compute Instrinsic Mean of Set of SPD matrices
Sigma_bar = intrinsicMean_mat(sigmas, 1e-1, 100)

% [ V, L, Mu ] = my_pca( vec_sigmas );
% figure('Color',[1 1 1]);
% plot(diag(L),'Color',[1 0 0]); grid on;
% title('Eigenvalues of PGA on Riemannian Manifold')
% [ p ] = explained_variance( L, 0.90 );
% [A_p, Y] = project_pca(vec_sigmas, Mu, V, p);
% fprintf('Vector-space dim (%d) - lower dimension (%d)\n',d_log,p);

%% %%%%%% Visualize Lower-D Embedding %%%%%%%%
plot_options        = [];
plot_options.labels = true_labels;
plot_options.title  = 'PGA on Riemannian Manifold'; 
ml_plot_data(Y',plot_options);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       Discover Cluster with different GMM Variants        %%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1: GMM-EM Model Selection via BIC
% 2: CRP-GMM (Collapsed Gibbs Sampler)
est_options = [];
est_options.type             = 1;   % GMM Estimation Algorithm Type   

% If algo 1 selected:
est_options.maxK             = 15;  % Maximum Gaussians for Type 1
est_options.fixed_K          = [];  % Fix K and estimate with EM for Type 1

% If algo 0 or 2 selected:
est_options.samplerIter      = 500;  % Maximum Sampler Iterations
                                     % For type 2: >100 iter are needed
                                    
est_options.do_plots         = 1;   % Plot Estimation Statistics
est_options.sub_sample       = 1;   % Size of sub-sampling of trajectories

% Fit GMM to Trajectory Data
tic;
clear Priors Mu Sigma
[Priors, Mu, Sigma] = fitgmm(Y, est_options);
toc;
% Extract Cluster Labels
est_K           = length(Priors);
[~, est_labels] = my_gmm_cluster(Y, Priors, Mu, Sigma, 'hard', []);
[Purity NMI F]  = cluster_metrics(true_labels, est_labels);
K = length(unique(true_labels));
fprintf('Number of estimated clusters: %d/%d, Purity: %1.2f, NMI Score: %1.2f, F measure: %1.2f \n',est_K,K, Purity, NMI, F);

%% Frank Wood's Implementation
 samplerIter = 500;
 tic;
[class_id, mean_record, covariance_record, K_record, lP_record, alpha_record] = sampler(Y, samplerIter);
[val , Maxiter]  = max(lP_record);
est_labels       = class_id(:,Maxiter);
toc;
%% Mo chen's Implementation

% DP-GMM (Mo-Chen's implementation) -- better mixing sometimes, slower
% (sometimes)
tic;
[est_labels, Theta, w, ll] = mixGaussGb(Y);
Priors = w;
est_K = length(Priors);


%% %%%%%% Euclidean Distances in l-dimensional space %%%%%%%%%%
l_sensitivity = 2;
[~, mode_hist_D, mean_D] = computePairwiseDistances(Y',1);
sigma = sqrt(mode_hist_D/l_sensitivity);
l = 1/(2*sigma^2);
d_Y   = L2_distance(Y,Y);
title_str = 'L-2 Distance on PCA log-vector space';
plotSimilarityConfMatrix(d_Y, title_str);
sim_Y = exp(-l*d_Y)
title_str = 'L-2 Similarity (Kernel) on PCA log-vector space';
plotSimilarityConfMatrix(sim_Y, title_str);

%% %%%%%% Compute Spectral Embedding on SPCM Similarities %%%%%%%%%%%%%%%%%%
M = [];
[Y_s, d, thres, V] = spectral_DimRed(S_spcm, M);
if isempty(M)
    s_norm = normalize_soft(softmax(d));
    M = sum(s_norm <= thres);
end
figure('Color',[1 1 1]);
plot(d,'Color',[1 0 0]);
title('Eigenvalues of SPCM Similarity matrix')
grid on;

%% %%%%%% Visualize Lower-D Embedding %%%%%%%%
plot_options        = [];
plot_options.labels = true_labels;
plot_options.title  = 'SPCM Spectral Manifold'; 
h1 = ml_plot_data(Y_s',plot_options)

%% %%%%%% Compute Spectral Embedding on B-SPCM Similarities %%%%%%%%%%%%%%%%%%
M = [];
[Y_bs, d, thres, V] = spectral_DimRed(S_b_spcm, M);
if isempty(M)
    s_norm = normalize_soft(softmax(d));
    M = sum(s_norm <= thres);
end
figure('Color',[1 1 1]);
plot(d,'Color',[1 0 0]);
title('Eigenvalues of B-SPCM Similarity matrix')
grid on;

%% %%%%%% Visualize Lower-D Embedding %%%%%%%%
plot_options        = [];
plot_options.labels = true_labels;
plot_options.title  = 'Bounded SPCM Spectral Manifold'; 
h1 = ml_plot_data(Y_bs',plot_options)

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Apply Standard Similarity-based Clustering Algorithms on Similarity Matrices
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Choose Similarity Metric (SPCM, RIEM, LERM, KLDM, JBLD ) %%%
% S_type = {'RIEM', 'LERM', 'KLDM', 'JBLD', 'B-SPCM'};
S_type = {'LERM'};

%%% Choose Clustering Algorithm %%%
% 'affinity': Affinity Propagation
% 'spectral': Spectral Clustering w/k-means
% C_type = 'Affinity';
C_type = 'Spectral';

%%% Selection of M-dimensional Spectral Manifold (for Spectral Clustering) %%%




%% Compute Stats for Paper
clc;
for i=1:length(S_type)    
  fprintf('%s Clustering with %s-- K: %1.2f +- %1.2f, Purity: %1.2f +-%1.2f , NMI Score: %1.2f +-%1.2f, F measure: %1.2f+-%1.2f \n', ...
      C_type, S_type{i}, mean(Ks(i,:)), std(Ks(i,:)), mean(Purities(i,:)), std(Purities(i,:)), mean(NMIs(i,:)), std(NMIs(i,:)), ...
      mean(F1s(i,:)), std(F1s(i,:)));
end
