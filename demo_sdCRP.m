%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Main demo script for the SPCM-CRP-MM Clustering Algorithm proposed in:
%
% N. Figueroa and A. Billard, “Discovering Similarities in Transformed Data: 
% Leveraging Spectral and Bayesian Non-parametic Methods for 
% Transform-Invariant Clustering and Segmentation”, Arxiv, 2016. 
%
% Author: Nadia Figueroa, PhD Student., Robotics
% Learning Algorithms and Systems Lab, EPFL (Switzerland)
% Email address: nadia.figueroafernandez@epfl.ch  
% Website: http://lasa.epfl.ch
% November 2016; Last revision: 27-November-2016
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
display = 0; randomize = 1;
[sigmas, true_labels] = load_toy_dataset('3d', display, randomize);

%% 2)  Toy 6D dataset, 60 Samples, 3 clusters (c1:20, c2:20, c3: 20)
% This function loads the 6-D ellipsoid dataset used to generate Fig. 6 and 
% from Section 4 and the results in Section 8 in the accompanying paper.

clc; clear all; close all;
display = 0; randomize = 1;
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
data_path = './data/'; randomize = 1;
[sigmas, true_labels] = load_task_dataset(data_path, randomize);

%% 4a) Toy 3D dataset, Diffusion Tensors from Synthetic Dataset, 1024 Samples
%% Cluster Distibution: 4 clusters (each cluster has 10 samples)

type = 'synthetic';
[sigmas, true_labels] = load_dtmri_dataset( data_path, type, display, randomize );

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
% One must also download the fanDTasia toolbox in the ~/SPCM-CRP/3rdParty
% directory, this toolbox is also provided in this link.

clc; clear all; close all;
data_path = './data/'; type = 'real'; display = 1; randomize = 0; 
[sigmas, true_labels] = load_dtmri_dataset( data_path, type, display, randomize );

%% 5) Real 7D dataset, Multi-Model GMM with 67 components
%% Cluster Distibution is unknown
% This function loads the 7-D (K=67) GMM describing human search strategies
% used to evaluate this algorithm in Section 8 of the accompanying paper.


%% 6a) Real 400D dataset, Covariance Features from ETH80 Dataset, 40 Samples
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
data_path = './data/'; split = 1; randomize = 1; 
[sigmas, true_labels] = load_eth80_dataset(data_path, split, randomize);

%% 6b) Real 900D dataset, Covariance Features from Youtube Dataset, 423 Samples
%% Cluster Distibution: 47 classes/clusters (each cluster has 9 samples)
% This function loads the 900-D YouTube Covariance Feature dataset 
% used to evaluate this algorithm in Section 8 of the accompanying paper.
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
[sigmas, true_labels] = load_youtube_dataset(data_path, split, randomize);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Step 1: Compute Similarity Matrix from B-SPCM Function for dataset   %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% %%%%%%%%%%%%%%%%%%%%% Set Hyper-parameter %%%%%%%%%%%%%%%%%%%%%%%%
% Tolerance for SPCM decay function 
tau = 10; % [1, 100] Set higher for noisy data, Set 1 for ideal data 

% %%%%%% Compute Confusion Matrix of Similarities %%%%%%%%%%%%%%%%%%
spcm = ComputeSPCMfunctionMatrix(sigmas, tau);  
S = spcm(:,:,2);

%%%%%%% Visualize Bounded Similarity Confusion Matrix %%%%%%%%%%%%%%
if exist('h0','var') && isvalid(h0), delete(h0);end
title_str = 'Bounded Similarity Function (B-SPCM) Matrix';
h0 = plotSimilarityConfMatrix(S, title_str);

%% %%% For Dataset 4b) Computing the Similarities takes ages so we %% %%%
%%%%%% can load a precomputed Similarity matrix with this command: %% %%%

%%%%%%% Select the YouTube Dataset %%%%%%%%%%%%%%
clc; clear all; close all;
data_path = './data/'; dataset = 'YouTube';
[S, true_labels] = loadSimilarityConfMatrix(data_path, dataset);

%% %%%%% Visualize Bounded Similarity Confusion Matrix %%%%%%%%%%%%%%
if exist('h0','var') && isvalid(h0), delete(h0);end
title_str = 'Bounded Similarity Function (B-SPCM) Matrix';
h0 = plotSimilarityConfMatrix(S, title_str);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%        Step 2: Run Automatic Spectral Dimensionality Reduction        %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%% Automatic Discovery of Dimensionality on M Manifold %%%%%%%%%%%
M = [];
M = 2;
[Y, d, thres, V] = spectral_DimRed(S, M);
if isempty(M)
    s_norm = normalize_soft(softmax(d));
    M = sum(s_norm <= thres);
end

%%%%%%%% Visualize Spectral Manifold Representation for M=2 or M=3 %%%%%%%%
if exist('h1','var') && isvalid(h1), delete(h1);end
h1 = plotSpectralManifold(Y, true_labels, d,thres, s_norm, M);


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%             Step 3: Discover Clusters with sd-CRP-MM                  %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%% Non-parametric Clustering on Manifold Data with Sim prior %%%%%%%%
options      = [];
options.iter = 100; % Max Sampler Iterations 
[Psi_MAP] = run_sdCRPMM(Y, S);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                     Visualize Clustering Results                      %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%% Plot Clustering Results against True Labels %%%%%%%%%%%%%%%
if exist('h2','var') && isvalid(h2), delete(h2);end
options = [];
options.clust_type = 'sd-CRP-MM';
options.Psi_MAP    = Psi_MAP; 
est_labels = Psi_MAP.Z_C';
[ Purity NMI F h2 ] = plotClusterResults( true_labels, est_labels, options );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%% For Datasets 1-4: Visualize sd-CRP-MM Results on Manifold Data %% %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Extract Learnt cluster parameters
Mu = Psi_MAP.Cluster_Mu;
Pr = Psi_MAP.Cluster_Pr;
Sigma = zeros(size(Pr,1),size(Pr,1),size(Pr,2));
for i=1:size(Pr,2)
    Sigma(:,:,i) = diag(Pr(:,i));
end        

% Visualize Cluster Parameters on Manifold Data
if exist('h3','var') && isvalid(h3), delete(h3);end
h3 = plotClusterParameters( Y, est_labels, Mu, Sigma );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%%% For Datasets 5a/b: Visualize cluster labels for DTI %%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Visualize Estimated Cluster Labels as DTI
figure('Color',[1 1 1]);
if exist('h3','var') && isvalid(h3), delete(h3);end
imagesc(flipud(reshape(est_labels,[sqrt(size(est_labels,2)) sqrt(size(est_labels,2))])))
colormap(pink)
colorbar
axis square
title('Estimated Cluster Labels of Diffusion Tensors')
