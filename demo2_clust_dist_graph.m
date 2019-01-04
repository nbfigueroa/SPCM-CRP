%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Main demo script for the SPCM-CRP-MM Clustering Algorithm proposed in:
%
% N. Figueroa and A. Billard, “Transform-Invariant Clustering of SPD Matrices 
% and its Application on Joint Segmentation and Action Discovery}”
% Arxiv, 2017. 
%
% Author: Nadia Figueroa, PhD Student., Robotics
% Learning Algorithms and Systems Lab, EPFL (Switzerland)
% Email address: nadia.figueroafernandez@epfl.ch  
% Website: http://lasa.epfl.ch
% November 2016; Last revision: 23-May-2017
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Step 1 (DATA LOADING): Load Datasets %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all; clear all; clc
%%%%%%%%%%%%%%%%%%%%%%%%% Select a Dataset %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1:  Toy Ellipsoid Dataset        (3D) / (9 Samples   c1:3,  c2:3,  c2:3)
% 2:  Toy Ellipsoid Dataset        (6D) / (60 Samples  c1:20, c2:20, c2:20)
% 3:  Real 6D Task-Ellipsoids      (6D) / (105 Samples c1:63, c2:21, c3:21)
% 4:  Synthetic Diffusion Tensors  (3D) / (1024 Samples 4 classes)
% 5:  Real Diffusion Tensors (Rat) (3D) / (1024 Samples 5 classes)
% ...
% 6:  ETH-80 Object Dataset Feats. (18D)  ... TODO (Rotated Objects)
% 7 : HMM Emission Models - Task1  (13D)  ... TODO (Polishing)
% 8 : HMM Emission Models - Task2  (7D)   ... TODO (Grating)
% 9 : HMM Emission Models - Task3  (13D)  ... TODO (Rolling)
% 10: HMM Emission Models - Task4  (26D)  ... TODO (Peeling)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%% Data Loading Parameter Description %%%%%%%%%%%%%%%%%%%%%%
% display:   [0,1]  -- Display Covariance matrices in their own format
% randomize: [0,1]  -- Randomize the Covariance Matrices indices
% pkg_dir:  {'./data/'} -- Path to data folder
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

pkg_dir = '/home/nbfigueroa/Dropbox/PhD_papers/journal-draft/new-code/SPCM-CRP';
display      = 0;       % display SDP matrices (if applicable)
randomize    = 0;       % randomize idx
dataset      = 2;       % choosen dataset from index above
sample_ratio = 1;      % sub-sample dataset [0.0 - 1]
[sigmas, true_labels, dataset_name] = load_SPD_dataset(dataset, pkg_dir, display, randomize, sample_ratio);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Step 2: Compute Similarity Matrix from SPD-Distances for dataset   %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%% Choose SDP distance %%%%%%%%%%%%%%%%%%%%%%%%
choosen_distance = 3;  % -2: Euclidean
                       % -1: Cholesky-Euclidean
                       %  0: Affine-Invariant Riemannian Distance (RIEM)
                       %  1: Log-Euclidean Riemannian Distance (LERM)
                       %  2: KL-Divergence (KLDM)
                       %  3: LogDet-Divergence (JBLD)
                     
[D, distance_name] = computeSDP_distances(sigmas, choosen_distance);
      
%% %%%%% Visualize Bounded Distance (dis-similarity) Matrix %%%%%%%%%%%%%%
if exist('h0','var') && isvalid(h0), delete(h0); end
h0 = plotSimilarityConfMatrix(D, distance_name);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   Step 3: Run Automatic Euclidean Embedding and Dimensionality Reduction  %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%% Embed Objects (Covariance Matrices) in Approximate Euclidean Space %%%%%%
emb_options = [];
emb_options.l_sensitivity = 1; % This changes the embedding/results ALOT!
emb_options.distance_name = distance_name;
emb_options.norm_K        = 1;   % Normalize the Kernel Matrix
emb_options.pow_eigen     = 2;   % K^(pow_eigen) for dimensionality selection
emb_options.show_plots    = 1;   % 0/1 display plots
emb_options.emb_type      = 1;   % 0: Kernel-PCA from Distances 
                                 % 1: Diffusion Maps with Distances 
switch emb_options.emb_type
    case 0

        [x_emb, Y, K, l] = distKernelPCA_Embedding(D, emb_options);
        emb_name = strcat('Kernel PCA on ',{' '}, distance_name);
        
    case 1
        % Time-steps for diffusion process
        emb_options.t              = 4;
        % Markov Chain Probability Matrix Construction Style (see function)
        emb_options.markov_constr  = 1;
        [x_emb, Y, K, l] = diffusionMap_Embedding(D, emb_options);
        eps = 1/l
        emb_name = strcat('Diffusion Maps on ',{' '}, distance_name);
end

M = size(Y,1);
show_emb = 0; 

%%%%%%%% Visualize Approximate Euclidean Embedding %%%%%%%%
plot_options        = [];
plot_options.labels = true_labels;
plot_options.title  = emb_name{1}; 
ml_plot_data(Y',plot_options);
axis equal;

%%%%%%%% Visualize Full Euclidean Embedding %%%%%%%%
if show_emb
    plot_options        = [];
    plot_options.labels = true_labels;
    plot_options.title  = emb_name{1};
    ml_plot_data(x_emb(1:8,:)',plot_options);
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  (Optional) Compute Similarity Matrix from B-SPCM Function for dataset %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% %%%%%%%%%%%%%%%%%%%%% Set Hyper-parameter %%%%%%%%%%%%%%%%%%%%%%%%
% Tolerance for SPCM decay function 
tau = 1; % [1, 100] Set higher for noisy data, Set 1 for ideal data 

%%%%%%%%%%%%%%%%%%% Compute SPCM Similarities %%%%%%%%%%%%%%%%%%
spcm = ComputeSPCMfunctionMatrix(sigmas, tau);  
K    = spcm(:,:,2);
if exist('h0','var') && isvalid(h0), delete(h0); end
h0 = plotSimilarityConfMatrix(K, 'B-SCPM Similarity function');

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%             Step 4: Discover Clusters of Covariance Matrices          %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Discover Clusters with different GMM-based Clustering Variants on Embedding %%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 0: sim-CRP-MM (Collapsed Gibbs Sampler) on Preferred Embedding
% 1: GMM-EM Model Selection via BIC on Preferred Embedding
% 2: CRP-GMM (Gibbs Sampler/Collapsed) on Preferred Embedding

est_options = [];
est_options.type             = 0;
% If algo 1 selected:
est_options.maxK             = 10;  % Ma ximum Gaussians for Type 1
est_options.fixed_K          = [];  % Fix K and estimate with EM for Type 1

% If algo 0 or 2 selected:
est_options.samplerIter      = 100;   % Maximum Sampler Iterations
                                      % For type 0: 50-200 iter are needed
                                      % For type 2: 200-1000 iter are needed

% Plotting options
est_options.do_plots         = 1;              % Plot Estimation Stats
est_options.dataset_name     = dataset_name;   % Dataset name
est_options.true_labels      = true_labels;    % To plot against estimates

% Fit GMM to Trajectory Data
tic;
clear Priors Mu Sigma
[Priors, Mu, Sigma, est_labels, stats] = fitgmm_sdp(K, Y, est_options);
toc;

%%%%%%%%%% Compute Cluster Metrics %%%%%%%%%%%%%
[Purity, NMI, F] = cluster_metrics(true_labels, est_labels');
if exist('true_labels', 'var')
    K = length(unique(true_labels));
end
switch est_options.type
    case 0
        fprintf('---%s Results---\n Iter:%d, LP: %d, Clusters: %d/%d with Purity: %1.2f, NMI Score: %1.2f, F measure: %1.2f \n', ...
            'spcm-CRP-MM (Collapsed-Gibbs)', stats.Psi.Maxiter, stats.Psi.MaxLogProb, length(unique(est_labels)), K,  Purity, NMI, F);
    case 1
        fprintf('---%s Results---\n  Clusters: %d/%d with Purity: %1.2f, NMI Score: %1.2f, F measure: %1.2f \n', ...
            'Finite-GMM (MS-BIC)', length(unique(est_labels)), K,  Purity, NMI, F);
    case 2
        
        if isfield(stats,'collapsed')
            fprintf('---%s Results---\n  Clusters: %d/%d with Purity: %1.2f, NMI Score: %1.2f, F measure: %1.2f \n', ...
                'CRP-GMM (Collapsed-Gibbs)', length(unique(est_labels)), K,  Purity, NMI, F);
        else
            fprintf('---%s Results---\n  Clusters: %d/%d with Purity: %1.2f, NMI Score: %1.2f, F measure: %1.2f \n', ...
                'CRP-GMM (Gibbs)', length(unique(est_labels)), K,  Purity, NMI, F);
        end
end

%% Visualize Estimated Parameters
if M < 4      
    est_options.emb_name = emb_name{1};
    [h_gmm]  = visualizeEstimatedGMM(Y,  Priors, Mu, Sigma, est_labels, est_options);
    axis equal
end

% TODO: Need to re-implement this function/has some problems when |k|>|c|
% [accuracy, est_labels_arr, CM] = calculateAccuracy(est_labels', true_labels);
% h = plotConfMat(CM);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                  Compute/Show GMM-Oracle Results                      %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GMM-Oracle Estimation
[Priors0, Mu0, Sigma0] = gmmOracle(Y, true_labels);
[~, est_labels0]       = my_gmm_cluster(Y, Priors0, Mu0, Sigma0, 'hard', []);
est_K0                 = length(unique(est_labels0));
[Purity NMI F]         = cluster_metrics(true_labels, est_labels0);
fprintf('(GMM-Oracle) Number of estimated clusters: %d/%d, Purity: %1.2f, NMI Score: %1.2f, F measure: %1.2f \n',est_K0,K, Purity, NMI, F);

%% Visualize GMM-Oracle
if M < 4
    est_options = [];
    est_options.type = -1;        
    est_options.emb_name = emb_name{1};
    [h_gmm]  = visualizeEstimatedGMM(Y,  Priors0, Mu0, Sigma0, est_labels0, est_options);
    axis equal
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%%% For Datasets 4a/b: Visualize cluster labels for DTI %%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Visualize Estimated Cluster Labels as DTI
% if exist('h3','var') && isvalid(h3), delete(h3);end
title = 'Estimated Cluster Labels of Diffusion Tensors';
h3 = plotlabelsDTI(est_labels, title);
