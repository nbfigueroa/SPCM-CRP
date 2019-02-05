%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Compare similarity functions and clustering algorithms  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% To run some of the function on this script you need 
% the ML_toolbox in your MATLAB path.
clc;  clear all; close all

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
display      = 0;      % display SDP matrices (if applicable)
randomize    = 0;      % randomize idx
dataset      = 6;      % choosen dataset from index above
sample_ratio = 1;      % sub-sample dataset [0.0 - 1]
[sigmas, true_labels, dataset_name] = load_SPD_dataset(dataset, pkg_dir, display, randomize, sample_ratio);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%   Step 2: Compute Log-Euclidean Embedding of SPD matrices     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%% Embed Objects (Covariance Matrices) in Approximate Euclidean Space %%%%%%
emb_options = [];
emb_options.expl_var      = 0.80;  % Explained variance for eigenvectors to keep
emb_options.show_plots    = 0;     % 0/1 display plots
emb_options.emb_type      = 0;     % 0: PCA on Tangent Space log-Eucl Mapping
                                   % 1: PCA on Tangent Space Riemannian Mapping

                                 
[x_emb, y_emb, pca_params, emb_name] = pcaTangentSpaceEmbedding(sigmas, emb_options);
show_full_emb  = 0;

%%%%%%%% Visualize Tangent-Space Embedding %%%%%%%%
if show_full_emb
    plot_options        = [];
    plot_options.labels = true_labels;
    plot_options.title  = 'Tangent Space Vector Embedding';
    ml_plot_data(x_emb', plot_options);
end

%%%%%%%% Visualize Lower-D Embedding %%%%%%%%
plot_options        = [];
plot_options.labels = true_labels;
plot_options.title  = emb_name; 
ml_plot_data(y_emb',plot_options);
axis equal

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%             Step 2: Discover Clusters of Covariance Matrices          %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Discover Clusters with different GMM-based Clustering Variants on Embedding %%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 0: sim-CRP-MM (Collapsed Gibbs Sampler) on Preferred Embedding
% 1: GMM-EM Model Selection via BIC on Preferred Embedding
% 2: CRP-GMM (Gibbs Sampler/Collapsed) on Preferred Embedding

est_options = [];
est_options.type             = 0;   % Clustering Estimation Algorithm Type   

% If algo 1 selected:
est_options.maxK             = 9;  % Maximum Gaussians for Type 1
est_options.fixed_K          = [];  % Fix K and estimate with EM for Type 1

% If algo 0 or 2 selected:
est_options.samplerIter      = 500;   % Maximum Sampler Iterations
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
    est_options = [];
    est_options.type = -1;        
    [h_gmm]  = visualizeEstimatedGMM(Y,  Priors0, Mu0, Sigma0, est_labels0, est_options);
    axis equal
end

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

