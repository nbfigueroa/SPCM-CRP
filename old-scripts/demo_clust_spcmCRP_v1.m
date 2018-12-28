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


%%%%%%%%%%%%%%%%% Data Loading Parameter Description %%%%%%%%%%%%%%%%%%%%%%
% display:   [0,1]  -- Display Covariance matrices in their own format
% randomize: [0,1]  -- Randomize the Covariance Matrices indices
% split:     [1,10] -- Selected Data Split from ETH80 or Youtube Dataset
% type:      {'real', 'synthetic'} -- Type for DT-MRI Dataset
% data_path:  {'./data/'} -- Path to data folder
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                    --Select a Dataset to Test--                       %%    
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1) Toy 3D dataset, 5 Samples, 2 clusters (c1:3, c2:2)
% This function loads the 3-D ellipsoid dataset used to generate Fig. 3, 4 
% and 5 from Section 4 and the results in Section 7 in the accompanying paper.

clc; clear all; close all;
display = 0; randomize = 0; dataset_name = 'Toy 3D';
[sigmas, true_labels] = load_toy_dataset('3d', display, randomize);

%% 2)  Toy 6D dataset, 60 Samples, 3 clusters (c1:20, c2:20, c3: 20)
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

% clc; clear all; close all;
data_path = './data/'; type = 'synthetic'; display = 1; randomize = 0; 
[sigmas, true_labels] = load_dtmri_dataset( data_path, type, display, randomize );
dataset_name = 'Synthetic DT-MRI';
%% 4b) Real 3D dataset, Diffusion Tensors from fanTDasia Dataset, 1024 Samples
%% Cluster Distibution: 4 clusters (each cluster has 10 samples)
% This function loads a 3-D Diffusion Tensor Image from a Diffusion
% Weight MRI Volume of a Rat's Hippocampus, the extracted 3D DTI is used
% to evaluate this algorithm in Section 8 of the accompanying paper.
%untitled
% To load and visualize this dataset, you must download the dataset files 
% in the  ~/SPCM-CRP/data directory. These are provided in the online 
% tutorial on Diffusion Tensor MRI in Matlab:
% http://www.cise.ufl.edu/~abarmpou/lab/fanDTasia/tutorial.php
%
% One must also download the fanDTasia toolbox in the ~/SPCM-CRP/3rdParty
% directory, this toolbox is also provided in this link.

% clc; clear all; close all;
data_path = './data/'; type = 'real'; display = 1; randomize = 0; 
[sigmas, true_labels] = load_dtmri_dataset( data_path, type, display, randomize );
dataset_name = 'Real DT-MRI';
%% 5) Real 400D dataset, Covariance Features from ETH80 Dataset, 40 Samples
%% Cluster Distibution: 8 classes/clusters (each cluster has 10 samples)
% This function loads the 400-D ETH80 Covariance Feature dataset 
% used to evaluate this algorithm in Section 8 of the accompanying paper.
%
%
% You must download this dataset from the following link: 
% http://ravitejav.weebly.com/classification-of-manifold-features.html
% and export it in the ~/SPCM-CRP/data directory
%
% Please cite the following paper if you make use of these features:
% R. Vemulapalli, J. Pillai, and R. Chellappa, “Kernel Learning for Extrinsic 
% Classification of Manifold Features”, CVPR, 2013. 

clc; clear all; close all;
data_path = './data/'; split = 1; randomize = 0; 
[sigmas, true_labels] = load_eth80_dataset(data_path, split, randomize);


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Step 1: Compute Similarity Matrix from B-SPCM Function for dataset   %%
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
NEF_S    = sum(abs(lambda_S(lambda_S < 0)))/sum(abs(lambda_S))

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%        Step 2: Run Automatic Eucliden Embedding and Dimensionality Reduction  %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% Embed Objects (Covariance Matrices) in Approximate Euclidean Space %%%%%%%%%%%
show_emb = 0; show_plots = 1;
[x_emb, Y] = graphEuclidean_Embedding(S, show_plots);
M = size(Y,1);

%%%%%%%% Visualize Approximate Euclidean Embedding %%%%%%%%
plot_options        = [];
plot_options.labels = true_labels;
plot_options.title  = 'Approximate Graph Embedding to Euclidean Space'; 
ml_plot_data(x_emb_apprx',plot_options);

%%%%%%%% Visualize Full Euclidean Embedding %%%%%%%%
if show_emb
    plot_options        = [];
    plot_options.labels = true_labels;
    plot_options.title  = 'Graph Embedding to Euclidean Space';
    ml_plot_data(x_emb',plot_options);
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%             Step 3: Discover Clusters with sd-CRP-MM                  %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Discover Clusters with different GMM-based Clustering Variants on Embedding %%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 0: sim-CRP-MM on Preferred Embedding
% 1: GMM-EM Model Selection via BIC on Preferred Embedding
% 2: CRP-GMM (Collapsed Gibbs Sampler) on Preferred Embedding
% 3: CRP-WIW-MM (Collapsed Gibbs Sampler) directly on SDP matrices
% 4: SPCM-CRP-WIW-MM (Collapsed Gibbs Sampler) directly on SDP matrices

est_options = [];
est_options.type             = 0;   % Clustering Estimation Algorithm Type   

% If algo 1 selected:
est_options.maxK             = 15;  % Maximum Gaussians for Type 1
est_options.fixed_K          = [];  % Fix K and estimate with EM for Type 1

% If algo 0 or 2 selected:
est_options.samplerIter      = 1000;  % Maximum Sampler Iterations
                                      % For type 2: >100 iter are needed
                                    
est_options.do_plots         = 1;   % Plot Estimation Statistics
est_options.sub_sample       = 1;   % Size of sub-sampling of trajectories

% Fit GMM to Trajectory Data
tic;
clear Priors Mu Sigma
[Priors, Mu, Sigma, est_labels] = fitgmm_sdp(S, Y, est_options);
toc;

%%%%%%%%%% Compute Cluster Metrics %%%%%%%%%%%%%
K = length(unique(true_labels));
[Purity NMI F] = cluster_metrics(true_labels, est_labels');
fprintf('---%s Results---\n Iter:%d, LP: %d, Clusters: %d/%d with Purity: %1.2f, NMI Score: %1.2f, F measure: %1.2f \n', ...
'spcm-CRP-MM', Psi.Maxiter, Psi.MaxLogProb, length(unique(est_labels)), K,  Purity, NMI, F);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                     Visualize Clustering Results                      %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%% Plot Clustering Results against True Labels %%%%%%%%%%%%%%%
% if exist('h','var') && isvalid(h), delete(h);end
% [accuracy, est_labels_arr, CM] = calculateAccuracy(est_labels', true_labels);
% h = plotConfMat(CM);

% Visualize Estimated Parameters
if M < 4
    est_options = [];
    est_options.type = 0;        
    [h_gmm]  = visualizeEstimatedGMM(Y,  Priors, Mu, Sigma, est_labels, est_options);
end

% GMM-Oracle Estimation
[Priors0, Mu0, Sigma0] = gmmOracle(Y, true_labels);
[~, est_labels0] = my_gmm_cluster(Y, Priors0, Mu0, Sigma0, 'hard', []);
[Purity NMI F]  = cluster_metrics(true_labels, est_labels0);
est_K0 = length(unique(est_labels0));
fprintf('(GMM-Oracle) Number of estimated clusters: %d/%d, Purity: %1.2f, NMI Score: %1.2f, F measure: %1.2f \n',est_K0,K, Purity, NMI, F);
if M < 4
    est_options = [];
    est_options.type = -1;        
    [h_gmm]  = visualizeEstimatedGMM(Y,  Priors0, Mu0, Sigma0, est_labels0, est_options);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%%% For Datasets 4a/b: Visualize cluster labels for DTI %%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Visualize Estimated Cluster Labels as DTI
% if exist('h3','var') && isvalid(h3), delete(h3);end
title = 'Estimated Cluster Labels of Diffusion Tensors';
h3 = plotlabelsDTI(est_labels, title);
