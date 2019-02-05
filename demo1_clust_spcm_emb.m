%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Main demo script for the SPCM-CRP-MM Clustering Algorithm proposed in:
%
% N. Figueroa and A. Billard, “Transform-Invariant Clustering of SPD Matrices 
% and its Application on Joint Segmentation and Action Discovery}”
% Arxiv, 2019. 
%
% Author: Nadia Figueroa, PhD Student., Robotics
% Learning Algorithms and Systems Lab, EPFL (Switzerland)
% Email address: nadia.figueroafernandez@epfl.ch  
% Website: http://lasa.epfl.ch
% 23-May-2017; Last revision: 28-Dec-2018;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Step 1 (DATA LOADING): Load Datasets %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all; clear all; clc
%%%%%%%%%%%%%%%%%%%%%%%%% Select a Dataset %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1:  Non-deformed+Deformed Ellips. (3D) / (80 Samples  c1:3,  c2:3,  c2:3  c4:)
% 2:  SPD sampled from Wishart      (3D) / (120 Samples c1:40, c2:40, c2:40)
% 3:  SPD sampled from Wishart      (6D) / (200 Samples c1:50, c2:50, c3:50 c4:50)
% 4:  Real 6D Task-Ellipsoids       (6D) / (105 Samples c1:63, c2:21, c3:21)
% 5:  Real Diffusion Tensors (Rat)  (3D) / (1024 Samples 5 classes)
% 6:  Manipulability Ellipsoids 1   (3D) / (727 Samples X classes)
% ...
% 9:  ETH-80 Object Dataset Feats. (18D)   ... TODO (Rotated Objects)
% 10 : HMM Emission Models - Task1  (13D)  ... TODO (Polishing)
% 11 : HMM Emission Models - Task2  (7D)   ... TODO (Grating)
% 12 : HMM Emission Models - Task3  (13D)  ... TODO (Rolling)
% 13:  HMM Emission Models - Task4  (26D)   ... TODO (Peeling)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%% Data Loading Parameter Description %%%%%%%%%%%%%%%%%%%%%%
% display:   [0,1]  -- Display Covariance matrices in their own format
% randomize: [0,1]  -- Randomize the Covariance Matrices indices
% pkg_dir:  {'./data/'} -- Path to data folder
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

pkg_dir = '/home/nbfigueroa/Dropbox/PhD_papers/journal-draft/new-code/SPCM-CRP';
display      = 0;       % display SDP matrices (if applicable)
randomize    = 0;       % randomize idx
dataset      = 4;       % choosen dataset from index above
sample_ratio = 0.5;      % sub-sample dataset [0.01 - 1]
[sigmas, true_labels, dataset_name] = load_SPD_dataset(dataset, pkg_dir, display, randomize, sample_ratio);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Step 2: Compute Similarity Matrix from B-SPCM Function for dataset   %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% %%%%%%%%%%%%%%%%%%%%%%%%% Set Hyper-parameter %%%%%%%%%%%%%%%%%%%%%%%%%%%
% Tolerance for SPCM decay function 
dis_type = 2; % Choose dis-similarity type 
              % 1:'var' use the variance of homothetic ratios
              % 2:'cv'  use the coefficient of variation of homo. ratios
gamma    = 2;
% %%%%%%%%%%%%%%% Compute Confusion Matrix of Similarities %%%%%%%%%%%%%%%%
spcm = ComputeSPCMfunctionMatrix(sigmas, gamma, dis_type);  
D    = spcm(:,:,1);
S    = spcm(:,:,2);

%%%%%%% Visualize Bounded Similarity Confusion Matrix %%%%%%%%%%%%%%
% if exist('h0','var') && isvalid(h0), delete(h0); end
title_str = 'SPCM Similarity Matrix';
h0 = plotSimilarityConfMatrix(S, title_str);

% if exist('h1','var') && isvalid(h1), delete(h1); end
title_str = 'SPCM Dis-similarity Matrix';
h1 = plotSimilarityConfMatrix(D, title_str);


% Compute Negative Eigenfraction of similarity matrix (NEF)
lambda_S = eig(S);
NEF_S    = sum(abs(lambda_S(lambda_S < 0)))/sum(abs(lambda_S));

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   Step 3: Embed SDP Matrices in Approximate Euclidean Space  %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   
% Choose Embedding implementation
show_plots = 0;          % Show plot of similarity matrices+eigenvalues   
pow_eigen  = 2;          % (L^+)^(pow_eigen) for dimensionality selection 
emb_type   = 0;          % 0: Graph-Subspace Projection
                         % 1: Kernel-PCA on L^+
                         % 2: Kernel-PCA on deformed Kernel with L
switch emb_type
    case 0        
        [x_emb, Y] = graphEuclidean_Embedding(S, show_plots, pow_eigen);
        emb_name = '(SPCM) Graph-Subspace Projection';
        
    case 1        
        norm_K = 1; % Choose to normalize K, ends up being unneccesary for L^+
        [x_emb, Y] = graphKernelPCA_Embedding(S, show_plots, norm_K, pow_eigen);
        emb_name = '(SPCM) Graph Kernel PCA Projection';
    
    case 2
        emb_options = [];
        emb_options.l_sensitivity = 2; % This changes the embedding/results ALOT!
        emb_options.distance_name = 'SPCM';
        emb_options.norm_K        = 1;           % Normalize the Kernel Matrix
        emb_options.pow_eigen     = pow_eigen;   % K^(pow_eigen) for dimensionality selection
        emb_options.show_plots    = show_plots;  % 0/1 display plots
        emb_options.deform        = 0;
        
        %%%%%%%%%%%%%%%%%%%%%% Choose SDP distance %%%%%%%%%%%%%%%%%%%%%%%%
        choosen_distance = 1;  %  1: Log-Euclidean Riemannian Distance (LERM)
                               %  3: LogDet-Divergence (JBLD)                     
        [D, distance_name] = computeSDP_distances(sigmas, choosen_distance);
        [x_emb, Y, K, K_SP, gamma_le]   = deformedKernelPCA_Embedding(D, S, emb_options);
        if emb_options.deform
            emb_name = 'Kernel PCA on deformed $k_{LE}^{SP}(\cdot,\cdot)$';
            if exist('h0a','var') && isvalid(h0a), delete(h0a); end
            title_str = '(un-deformed) log-Euclidean RBF kernel $k_{LE}(\cdot,\cdot)$';
            h0a = plotSimilarityConfMatrix(K, title_str);
            
            if exist('h0b','var') && isvalid(h0b), delete(h0b); end
            title_str = 'Deformed log-Euclidean RBF kernel $k_{LE}^{SP}(\cdot,\cdot)$';
            h0b = plotSimilarityConfMatrix(K_SP, title_str);            
        else
            emb_name = 'Kernel PCA on (un-deformed) $k_{LE}(\cdot,\cdot)$';
        end
end

M = size(Y,1);
show_emb = 0;

%%%%%%%% Visualize Approximate Euclidean Embedding %%%%%%%%
plot_options        = [];
plot_options.labels = true_labels;
plot_options.title  = emb_name; 
ml_plot_data(Y',plot_options);
axis equal;

%%%%%%%% Visualize Full Euclidean Embedding %%%%%%%%
if show_emb
    plot_options        = [];
    plot_options.labels = true_labels;
    plot_options.title  = emb_name;
    ml_plot_data(x_emb',plot_options);
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%             Step 4: Discover Clusters of Covariance Matrices          %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Discover Clusters with GMM-based Clustering Variants on Embedding %%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 0: sim-CRP-MM (Collapsed Gibbs Sampler) on Preferred Embedding
% 1: GMM-EM Model Selection via BIC on Preferred Embedding
% 2: CRP-GMM (Gibbs Sampler/Collapsed) on Preferred Embedding

est_options = [];
est_options.type             = 0;   % Clustering Estimation Algorithm Type   

% If algo 1 selected:
est_options.maxK             = 15;   % Maximum Gaussians for Type 1
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
[Priors, Mu, Sigma, est_labels, stats] = fitgmm_sdp(S, Y, est_options);
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
    est_options.emb_name = emb_name;
    [h_gmm]  = visualizeEstimatedGMM(Y,  Priors, Mu, Sigma, est_labels, est_options);
    axis equal;
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
if M < 4
    oracle_options = [];
    oracle_options.type = -1;        
    oracle_options.emb_name = emb_name;
    [h_gmm]  = visualizeEstimatedGMM(Y,  Priors0, Mu0, Sigma0, est_labels0, oracle_options);
    axis equal
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%%% For Datasets 4a/b: Visualize cluster labels for DTI %%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Visualize Estimated Cluster Labels as DTI
% if exist('h3','var') && isvalid(h3), delete(h3);end
title = 'Estimated Cluster Labels of Diffusion Tensors';
h3 = plotlabelsDTI(est_labels, title);
