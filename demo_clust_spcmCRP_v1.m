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

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%        Step 2: Run Automatic Eucliden Embedding and Dimensionality Reduction  %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% Embed Objects (Covariance Matrices) in Approximate Euclidean Space %%%%%%%%%%%
show_emb = 0;
[x_emb, x_emb_apprx] = graphEuclidean_Embedding(S);

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

%%%%%%%% Non-parametric Clustering on Manifold Data with Sim prior %%%%%%%%
% Approximated Embedded data
Y = x_emb_apprx;
M = size(Y,1);

% Setting sampler/model options (i.e. hyper-parameters, alpha, Covariance matrix)
options                 = [];
options.type            = 'full';  % Type of Covariance Matrix: 'full' = NIW or 'Diag' = NIG
options.T               = 200;     % Sampler Iterations 
options.alpha           = 1;     % Concentration parameter [0 - 2]

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
[Psi, Psi_Stats]   = run_ddCRP_sampler(Y, S, options);
est_labels        = Psi.Z_C';

%%%%%%%% Visualize Collapsed Gibbs Sampler Stats %%%%%%%%%%%%%%
if exist('h1b','var') && isvalid(h1b), delete(h1b);end
options = [];
options.dataset      = dataset_name;
options.true_labels  = true_labels; 
options.Psi          = Psi;
[ h1b ] = plotSamplerStats( Psi_Stats, options );

%%%%%%%%%% Extract Learned GMM models %%%%%%%%%%%%%
N = length(sigmas);
unique_labels = unique(est_labels);
est_K      = length(unique_labels);
Priors     = zeros(1, est_K);
singletons = zeros(1, est_K);
for k=1:est_K
    assigned_k = sum(est_labels==unique_labels(k));
    Priors(k) = assigned_k/N;
    singletons(k) = assigned_k < round(N*0.01);
end
Mu     = Psi.Theta.Mu(:,unique_labels);
Sigma  = Psi.Theta.Sigma(:,:,unique_labels);

if any(singletons)
    [~, est_labels] =  my_gmm_cluster(Y, Priors, Mu, Sigma, 'hard', []);
    unique_labels = unique(est_labels);
    singleton_idx = find(singletons == 1);
    Mu(:,singleton_idx) = [];
    Sigma(:,:,singleton_idx) = [];
    unique_labels(singleton_idx) = [];
    Priors  = [];
    est_K = length(Mu);
    for k=1:est_K
        assigned_k = sum(est_labels==unique_labels(k));
        Priors(k) = assigned_k/N;
    end
end

%%%%%%%%%% Compute Cluster Metrics %%%%%%%%%%%%%
[Purity NMI F] = cluster_metrics(true_labels, est_labels');
fprintf('---%s Results---\n Iter:%d, LP: %d, Clusters: %d with Purity: %1.2f, NMI Score: %1.2f, F measure: %1.2f \n', ...
'spcm-CRP-MM', Psi.Maxiter, Psi.MaxLogProb, length(unique(est_labels)), Purity, NMI, F);

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%%% For Datasets 4a/b: Visualize cluster labels for DTI %%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Visualize Estimated Cluster Labels as DTI
if exist('h3','var') && isvalid(h3), delete(h3);end
title = 'Estimated Cluster Labels of Diffusion Tensors';
h3 = plotlabelsDTI(est_labels, title);
