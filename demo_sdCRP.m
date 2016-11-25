%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Test SPCM Similarity with CRP Clustering Algorithm
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
clear all
close all

% Set to 1 if you want to display Covariance Matrices
display = 1;

% Set to 1 if you want to randomize the Covariance Matrices indices
randomize = 0;

% Path to data folder
data_path = './data/';

%% Select Dataset:
%% 1) Toy 3D dataset, 5 Samples, 2 clusters (c1:3, c2:2)
[sigmas, true_labels] = load_toy_dataset('3d', display, randomize);

%% 2)  Toy 6D dataset, 60 Samples, 3 clusters (c1:20, c2:20, c3: 20)
[sigmas, true_labels] = load_toy_dataset('6d', display, randomize);

%% 3) Real 6D dataset, task-ellipsoids, 105 Samples, 3 clusters (c1:63, c2:21, c3: 21)
[sigmas, true_labels] = load_task_dataset(data_path); %<== ADD RANDOMIZE OPTION

%% 4) Real 400D dataset, Covariance Features from ETH80 Dataset, 40 Samples, 8 classes/clusters (c1:10, c2:10,.., c8: 10)
split = 10;
[sigmas, true_labels] = load_eth80_dataset(data_path,split); %<== ADD RANDOMIZE OPTION

%% 5) Real XD dataset, Covariance Features from YouTube Dataset, N Samples, Y classes/clusters 
split = 1;
[sigmas, true_labels] = load_youtube_dataset(data_path,split);

%% 6) Real 3D dataset, Diffusion Tensors from fanTDasia Dataset, 1024 Samples, 4 clusters (c1: ... c2: ... c3:... c4:...)
type = 'real';
[sigmas, true_labels] = load_dtmri_dataset( data_path, type, display, randomize );


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute Similarity Matrix from B-SPCM Function for dataset
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% %%%%%%%%%%%%%%%%%%%%% Set Hyper-parameter %%%%%%%%%%%%%%%%%%%%%%%%
% Tolerance for SPCM decay function 
tau = 10; % [1, 100] Set higher for noisy data, Set 1 for ideal data 

% %%%%%% Compute Confusion Matrix of Similarities %%%%%%%%%%%%%%%%%%
N = length(sigmas);    % Number of Covariance Matrices
D = size(sigmas{1},1); % Dimension of Covariance Matrices
fprintf('Computing SPCM Similarity Function for %dx%d Covariance Matrices of %dx%d dimensions...\n',N,N,D,D);
tic;
spcm = ComputeSPCMfunctionMatrix(sigmas, tau);  
toc;
S = spcm(:,:,2); % Bounded Decay SPCM Similarity Matrix
fprintf('*************************************************************\n');

%% %%%%% Visualize Bounded Similarity Confusion Matrix %%%%%%%%%%%%%%
if exist('h0','var') && isvalid(h0), delete(h0);end
title_str = 'Bounded Similarity Function (B-SPCM) Matrix';
h0 = plotSimilarityConfMatrix(S, title_str);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Run Spectral Manifold Algorithm
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% %%%%%% Automatic Discovery of Dimensionality of M Manifold % %%%%%%
fprintf('Computing Spectral Dimensionality Reduction based on SPCM Similarity Function...\n');
tic;
M = [];
[Y, d, thres, V] = spectral_DimRed(S, M);

if isempty(M)
    s_norm = normalize_soft(softmax(d));
    M = sum(s_norm <= thres);
end

toc;
fprintf('*************************************************************\n');

%%%%%%% Plot Spectral Manifold Representation for M=2 or M=3 % %%%%%
if exist('h1','var') && isvalid(h1), delete(h1);end
h1 = plotSpectralManifold(Y, true_labels, d,thres, s_norm, M);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Discover Clusters using sd-CRP-MM %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% %%%%%% Non-parametric Clustering on Manifold Data with Sim prior % %%%%%%
fprintf('Clustering via sd-CRP...\n');
tic;
[Psi_MAP] = run_sdCRPMM(Y, S);
toc;
fprintf('*************************************************************\n');

%% %%%%%%% Plot Clustering Results against True Labels % %%%%%%%%
if exist('h2','var') && isvalid(h2), delete(h2);end
clust_type = 'sd-CRP-MM';
est_labels = Psi_MAP.Z_C';
[ Purity NMI F h2 ] = plotClusterResults( true_labels, est_labels, clust_type );
fprintf('MAP Cluster estimate recovered at iter %d: %d\n', Psi_MAP.iter, length(est_labels));
fprintf('%s LP: %d and Purity: %1.2f, NMI Score: %1.2f, F measure: %1.2f \n', clust_type, Psi_MAP.LogProb, Purity, NMI, F);


% %%%%%%%%% Plot sd-CRP-MM Results on Manifold Data % %%%%%%%%%

% Extract cluster parameters
Mu = Psi_MAP.Cluster_Mu;
Pr = Psi_MAP.Cluster_Pr;
Sigma = zeros(size(Pr,1),size(Pr,1),size(Pr,2));
for i=1:size(Pr,2)
    Sigma(:,:,i) = diag(Pr(:,i));
end        
% Visualize Cluster Parameters on Manifold Data
if exist('h3','var') && isvalid(h3), delete(h3);end
h3 = plotClusterParameters( Y, est_labels, Mu, Sigma );

%% %%%%%%%% Plot cluster labels for DTI %% %%%%%%%%
        
% Generate labels from Fractional Anisotropy Value
figure('Color',[1 1 1]);
imagesc(flipud(reshape(est_labels,[size(DTI,3) size(DTI,4)])))
colormap(pink)
colorbar
axis square
title('Estimated Cluster Labels of Diffusion Tensors')


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Discover Clusters using CRP-MM %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('Clustering via DP(CRP)-MM (MCMC Estimation)...\n');
% %%%%%% Non-parametric Clustering on Manifold Data % %%%%%%
tic;
[labels_dpgmm,Theta_dpgmm,w,llh] = mixGaussGb(Y);
toc;
fprintf('*************************************************************\n');

%% Extract learnt parameters
k_dpgmm  = length(unique(labels_dpgmm));
Mu_dpgmm = zeros(size(Y,1), k_dpgmm);

% Sigma = model_dpgmm.U_


%%
fprintf('Clustering via DP(CRP)-MM (Variational Estimation)...\n');
% ...
% .. test vbdpgmm here
% ...

