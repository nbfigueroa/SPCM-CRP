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
display = 1;  randomize = 0;
choosen_dataset = 1;
[sigmas, true_labels, dataset_name] = load_SPD_dataset(choosen_dataset, pkg_dir, display, randomize);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Step 2: Compute Similarity Matrix from B-SPCM Function for dataset   %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% %%%%%%%%%%%%%%%%%%%%% Set Hyper-parameter %%%%%%%%%%%%%%%%%%%%%%%%
% Tolerance for SPCM decay function 
tau = 1; % [1, 100] Set higher for noisy data, Set 1 for ideal data 
% Datasets 1-3:  tau = 1;
% Datasets 4a/4b tau = 10;
% Datasets 4a/4b tau = 5;
% Dataset 5: tau = 1;

% %%%%%% Compute Confusion Matrix of Similarities %%%%%%%%%%%%%%%%%%
spcm = ComputeSPCMfunctionMatrix(sigmas, tau);  
D    = spcm(:,:,1);
S    = spcm(:,:,2);

%%%%%%% Visualize Bounded Similarity Confusion Matrix %%%%%%%%%%%%%%
if exist('h0','var') && isvalid(h0), delete(h0); end
title_str = 'Bounded Similarity (B-SPCM) Matrix';
h0 = plotSimilarityConfMatrix(S, title_str);

% if exist('h1','var') && isvalid(h1), delete(h1); end
% title_str = 'Un-Bounded (Dis)-Similarity Function (SPCM) Matrix';
% h1 = plotSimilarityConfMatrix(D, title_str);

% Compute Negative Eigenfraction of similarity matrix (NEF)
lambda_S = eig(S);
NEF_S    = sum(abs(lambda_S(lambda_S < 0)))/sum(abs(lambda_S))

%% Gram-Matrix of (Dis)-Similarity Values
% Compute Gram Matrix of D (make function)
N = size(D,1);
J = eye(N) - (1/N)*ones(N,1)*ones(N,1)';
G = -0.5 *( J * (D.^2) * J);
if exist('h2','var') && isvalid(h2), delete(h2); end
title_str = 'Gram Matrix of (Dis)-Similarity (SPCM) Values';
h2 = plotSimilarityConfMatrix(G, title_str);

% Compute Negative Eigenfraction of similarity matrix (NEF)
lambda_G = eig(G);
NEF_G    = sum(abs(lambda_G(lambda_G < 0)))/sum(abs(lambda_G))
lambda_S = eig(S);
NEF_S    = sum(abs(lambda_S(lambda_S < 0)))/sum(abs(lambda_S))

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%        Step 2: Run Automatic Spectral Dimensionality Reduction        %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% Automatic Discovery of Dimensionality on M Manifold %%%%%%%%%%%
clc;
[x_emb, x_emb_apprx] = spectral_DimRed_v2(S);
%% %%%%%% Visualize Euclidean Embedding %%%%%%%%
plot_options        = [];
plot_options.labels = true_labels;
plot_options.title  = 'Graph Embedding to Euclidean Space'; 
ml_plot_data(x_emb',plot_options);

%% %%%%%% Visualize Approximate Euclidean Embedding %%%%%%%%
plot_options        = [];
plot_options.labels = true_labels;
plot_options.title  = 'Approximate Graph Embedding to Euclidean Space'; 
ml_plot_data(x_emb_apprx',plot_options);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%             Step 3: Discover Clusters with sd-CRP-MM                  %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%% Non-parametric Clustering on Manifold Data with Sim prior %%%%%%%%
% Approximated Embedded data
Y = x_emb_apprx;
M = size(Y,1);

% Setting sampler/model options (i.e. hyper-parameters, alpha, Covariance matrix)
options                 = [];
options.type            = 'full';  % Type of Covariance Matrix: 'full' = NIW or 'Diag' = NIG
options.T               = 200;     % Sampler Iterations 
options.alpha           = 1;     % Concentration parameter

% Standard Base Distribution Hyper-parameter setting
if strcmp(options.type,'diag')
    lambda.alpha_0       = M;                    % G(sigma_k^-1|alpha_0,beta_0): (degrees of freedom)
    lambda.beta_0        = sum(diag(cov(Y')))/M; % G(sigma_k^-1|alpha_0,beta_0): (precision)
end
if strcmp(options.type,'full')
    lambda.nu_0        = M;                           % IW(Sigma_k|Lambda_0,nu_0): (degrees of freedom)
%     lambda.Lambda_0    = eye(M)*sum(diag(cov(Y')))/M; % IW(Sigma_k|Lambda_0,nu_0): (Scale matrix)
    lambda.Lambda_0    = diag(diag(cov(Y')));       % IW(Sigma_k|Lambda_0,nu_0): (Scale matrix)
end
lambda.mu_0             = mean(Y,2);    % hyper for N(mu_k|mu_0,kappa_0)
lambda.kappa_0          = 1;            % hyper for N(mu_k|mu_0,kappa_0)


% Run Collapsed Gibbs Sampler
options.lambda    = lambda;
options.verbose   = 1;
[Psi Psi_Stats]   = run_ddCRP_sampler(Y, S, options);
est_labels        = Psi.Z_C';

%%%%%%%% Visualize Collapsed Gibbs Sampler Stats %%%%%%%%%%%%%%
if exist('h1b','var') && isvalid(h1b), delete(h1b);end
options = [];
options.dataset      = dataset_name;
options.true_labels  = true_labels; 
options.Psi          = Psi;
[ h1b ] = plotSamplerStats( Psi_Stats, options );

%% %%%%%%%% Compute Cluster Metrics %%%%%%%%%%%%%
[Purity NMI F]                 = cluster_metrics(true_labels, est_labels');
[accuracy, est_labels_arr, CM] = calculateAccuracy(est_labels', true_labels);
h = plotConfMat(CM);
fprintf('---%s Results---\n Iter:%d, LP: %d, Clusters: %d with Purity: %1.2f, NMI Score: %1.2f, F measure: %1.2f \n', ...
'spcm-CRP-MM', Psi.Maxiter, Psi.MaxLogProb, length(unique(est_labels)), Purity, NMI, F);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                     Visualize Clustering Results                      %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%% Plot Clustering Results against True Labels %%%%%%%%%%%%%%%
if exist('h2','var') && isvalid(h2), delete(h2);end
options = [];
options.clust_type  = 'spcm-CRP-MM';
options.Psi         = Psi; 
est_labels          = Psi.Z_C';
[ Purity NMI F h2 ] = plotClusterResults( true_labels, est_labels, options ); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% For Datasets 1-3 + 5a/b: Visualize sd-CRP-MM Results on Manifold Data %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Extract Learnt cluster parameters
Mu = Psi.Theta.Mu;
Sigma = Psi.Theta.Sigma;
      
% Visualize Cluster Parameters on Manifold Data
if exist('h3','var') && isvalid(h3), delete(h3);end
h3 = plotClusterParameters( Y, est_labels, Mu, Sigma );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%%% For Datasets 4a/b: Visualize cluster labels for DTI %%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Visualize Estimated Cluster Labels as DTI
if exist('h3','var') && isvalid(h3), delete(h3);end
title = 'Estimated Cluster Labels of Diffusion Tensors';
h3 = plotlabelsDTI(est_labels, title);
