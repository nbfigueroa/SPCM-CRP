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
display = 0;  randomize = 0;
choosen_dataset = 2;
[sigmas, true_labels, dataset_name] = load_SPD_dataset(choosen_dataset, pkg_dir, display, randomize);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Step 2: Compute Similarity Matrix from B-SPCM Function for dataset   %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% %%%%%%%%%%%%%%%%%%%%% Set Hyper-parameter %%%%%%%%%%%%%%%%%%%%%%%%
% Tolerance for SPCM decay function 
tau = 1; % [1, 100] Set higher for noisy data, Set 1 for ideal data 

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
NEF_S    = sum(abs(lambda_S(lambda_S < 0)))/sum(abs(lambda_S));

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%   Step 2: Compute Log-Euclidean Embedding of SPD matrices     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N     = length(sigmas{1});
d_vec = N*(N+1)/2;
vec_sigmas = zeros(d_vec,length(sigmas));
for s=1:length(sigmas)
    % Log matrix of Sigma
    sigma     = sigmas{s};
    log_sigma = logm(sigma);   
    
    % Projecting to the L-2 Norm
    vec_sigmas(:,s) = symMat2Vec(log_sigma);
end
Y = vec_sigmas;
M = size(Y,1);

%% %%%%%% Visualize Lower-D Embedding %%%%%%%%
plot_options        = [];
plot_options.labels = true_labels;
plot_options.title  = 'Log-Euclidean Embedding'; 
ml_plot_data(vec_sigmas',plot_options);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     Dim. Red Option 1: PCA on log-Euclidean Embedding  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[ V, L, Mu_X ] = my_pca( vec_sigmas );
figure('Color',[1 1 1]);
plot(diag(L),'-*r'); grid on;
xlabel('Eigenvalue index')
title('Eigenvalues of PCA on log-vector space','Interpreter','LaTex')
[ p ] = explained_variance( L, 0.90 );
[A_p, Y] = project_pca(vec_sigmas, Mu_X, V, p);
fprintf('Vector-space dim (%d) - lower dimension (%d)\n',d_vec,p);
M = size(Y,1);

%% %%%%%% Visualize Lower-D Embedding %%%%%%%%
plot_options        = [];
plot_options.labels = true_labels;
plot_options.title  = 'PCA on log-Euclidean Embedding'; 
ml_plot_data(Y',plot_options);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Dim. Red Option 2: Principal Geodesic Analysis on Riemannian Manifold 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N     = length(sigmas{1});
d_log = N*(N+1)/2;

% Compute Instrinsic Mean of Set of SPD matrices
sigma_bar    = intrinsicMean_mat(sigmas, 1e-8, 100);

% Calculate the tangent vectors about the mean
vec_sigmas = zeros(d_log,length(sigmas));
for s=1:length(sigmas)
    sigma     = sigmas{s};
    log_sigma = logmap(sigma, sigma_bar);           
    vec_sigmas(:,s) = symMat2Vec(log_sigma);
end

% Construct Covariance matrix of tangent vectors
Cov_x = (1/(N-1))*vec_sigmas*vec_sigmas';

% Perform Eigenanalysis of Covariance Matrix
[V, L] = eig(Cov_x);

% Sort Eigenvalue and get indices
[L_sort, ind] = sort(diag(L),'descend');

% arrange the columns in this order
V = V(:,ind); 

% Vectorize sorted eigenvalues
L = diag(L_sort); 

% X_bar represented on the Tangent Manifold
Mu = symMat2Vec(logm(sigma_bar));

figure('Color',[1 1 1]);
plot(diag(L),'-*r'); grid on;
xlabel('Eigenvalue index')
title('Eigenvalues of PGA of Riemannian Manifold','Interpreter','LaTex')
[ p ]    = explained_variance( L, 0.90 );
[A_p, Y] = project_pca(vec_sigmas, Mu, V, p);
fprintf('Vector-space dim (%d) - lower dimension (%d)\n',d_log,p);
M = size(Y,1);
%% %%%%%% Visualize Lower-D Embedding %%%%%%%%
plot_options        = [];
plot_options.labels = true_labels;
plot_options.title  = 'PGA on Riemannian Manifold'; 
ml_plot_data(Y',plot_options);

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

