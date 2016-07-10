%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compare similarity and clustering algorithms
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% To run some of the function on this script you need 
% the ML_toolbox in your MATLAB path.

clc
clear all
close all

% Set to 1 if you want to display Covariance Matrices
display = 1;

%% Select Dataset:
%% Toy 3D dataset, 5 Samples, 2 clusters (c1:3, c2:2)
[sigmas, true_labels] = load_toy_dataset('3d', display);

%% Toy 4D dataset, 6 Samples, 2 clusters (c1:4, c2:2)
[sigmas, true_labels] = load_toy_dataset('4d', display);

%% Toy 6D dataset, 30 Samples, 3 clusters (c1:10, c2:10, c3: 10)
[sigmas, true_labels] = load_toy_dataset('6d', display);

%% Real 6D dataset, task-ellipsoids, 105 Samples, 3 clusters (c1:63, c2:21, c3: 21)
% Path to data folder
data_path = '/home/nadiafigueroa/dev/MATLAB/SPCM-CRP/data';
[sigmas, true_labels] = load_task_dataset(data_path);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute Similarity Matrix from b-SPCM Function for dataset
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc;
% %%%%%%%%%%%%%%%%%%%%%
% Set Hyper-parameter
% %%%%%%%%%%%%%%%%%%%%%
% Tolerance for SPCM decay function 
tau = 1; % [1, 100] Set higher for noisy data, Set 1 for ideal data 

% Number of datapoints
N = length(sigmas);
fprintf('Computing SPCM Similarity Function for %dx%d observations...\n',N,N);
tic;
spcm = ComputeSPCMfunctionMatrix(sigmas, tau);  
toc;
S_spcm = spcm(:,:,2); % Bounded Decay SPCM Similarity Matrix
fprintf('*************************************************************\n');


% %%%%%% Visualize Bounded Similarity Confusion Matrix %%%%%%%%%%%%%%
figure('Color',[1 1 1])
imagesc(S_spcm)
title('Bounded Decaying Similarity Function (B-SPCM) Matrix')
colormap(pink)
colorbar 
axis square

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute other Similarity Functions for Comparison
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%% Affine Invariant Riemannian Metric %%%%%%%%%%%%%%%
fprintf('Computing RIEM Similarity Function for %dx%d observations...\n',N,N);
tic;
S_riem = compute_cov_sim( sigmas, 'RIEM' );
toc;
fprintf('*************************************************************\n');

%%%%%%%%%%%%%%% 'LERM': Log-Euclidean Riemannina Metric %%%%%%%%%%%%%%%
fprintf('Computing LERM Similarity Function for %dx%d observations...\n',N,N);
tic;
S_lerm = compute_cov_sim( sigmas, 'LERM' );
toc;
fprintf('*************************************************************\n');

%%%%%%%%%%%%%%% 'KLDM': Kullback-Liebler Divergence Metric %%%%%%%%%%%%%%%
fprintf('Computing KLDM Similarity Function for %dx%d observations...\n',N,N);
tic;
S_kldm = compute_cov_sim( sigmas, 'KLDM' );
toc;
fprintf('*************************************************************\n');

%%%%%%%%%%%%%%% 'JBLD': Jensen-Bregman LogDet Divergence %%%%%%%%%%%%%%%
fprintf('Computing JBLD Similarity Function for %dx%d observations...\n',N,N);
tic;
S_jbld = compute_cov_sim( sigmas, 'JBLD' );
toc;
fprintf('*************************************************************\n');

% Plot Results for all metrics
figure('Color',[1 1 1])
subplot(2,2,1)
imagesc(S_riem)
title('Affine Invariant Riemannian Metric (RIEM)')
colormap(pink)
colorbar 
axis square

subplot(2,2,2)
imagesc(S_lerm)
title(' Log-Euclidean Riemannian Metric (LERM)')
colormap(pink)
colorbar 
axis square

subplot(2,2,3)
imagesc(S_kldm)
title(' Log-Euclidean Riemannian Metric (LERM)')
colormap(pink)
colorbar 
axis square

subplot(2,2,4)
imagesc(S_jbld)
title(' Log-Euclidean Riemannian Metric (LERM)')
colormap(pink)
colorbar 
axis square

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Apply Clustering Algorithms on Similarity Matrices
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Option 1: Affinity Propagation on Similarity Matrix

% Choose Similarity Metric (SPCM, RIEM, LERM, KLDM, JBLD )
S_type = {'SPCM', 'RIEM', 'LERM', 'KLDM', 'JBLD'};

% %%% Compute clusters from Similarity Matrix using Affinity Propagation %%%%%%

figure('Color',[1 1 1])

D_aff = S - eye(size(S));
fprintf('Clustering via Affinity Propagation...\n');
tic;
[E K labels_aff idx] = affinitypropagation(D_aff);
toc;
NMI = CalcNMI(true_labels, labels_aff');
fprintf('Number of clusters: %d, NMI Score: %d\n',K, NMI);
fprintf('*************************************************************\n');

imagesc(labels_aff')
title('Clustering from Aff. Prop. on SPCM function')
axis equal tight
colormap(pink)

%% Option 2: kernel Kmeans

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compare CRP to kmeans on projected data from Spectral Manifold
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Project Data to Spectral Manifold

%% Option 1: Use CRP on points projected to spectral manifold

%% Option 2: Use kmeans on points projected to spectral manifold
