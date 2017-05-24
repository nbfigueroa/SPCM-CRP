%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Compare similarity functions and clustering algorithms  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% To run some of the function on this script you need 
% the ML_toolbox in your MATLAB path.

clc;  clear all; close all

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

clc; clear all; close all;
data_path = './data/'; type = 'synthetic'; display = 1; randomize = 0; 
[sigmas, true_labels] = load_dtmri_dataset( data_path, type, display, randomize );
dataset_name = 'Synthetic DT-MRI';
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
%
% One must also download the fanDTasia toolbox in the ~/SPCM-CRP/3rdParty
% directory, this toolbox is also provided in this link.

clc; clear all; close all;
data_path = './data/'; type = 'real'; display = 1; randomize = 0; 
[sigmas, true_labels] = load_dtmri_dataset( data_path, type, display, randomize );
dataset_name = 'Real DT-MRI';
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Compute Similarity Matrix from B-SPCM Function for dataset   %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% %%%%%%%%%%%%%%%%%%%%% Set Hyper-parameter %%%%%%%%%%%%%%%%%%%%%%%%
% Tolerance for SPCM decay function 
tau = 0.5; % [1, 100] Set higher for noisy data, Set 1 for ideal data 

% %%%%%% Compute Confusion Matrix of Similarities %%%%%%%%%%%%%%%%%%
spcm = ComputeSPCMfunctionMatrix(sigmas, tau);  
S_spcm   = spcm(:,:,2);

%%%%%%% Visualize Bounded Similarity Confusion Matrix %%%%%%%%%%%%%%
if exist('h0','var') && isvalid(h0), delete(h0);end
title_str = 'Bounded SPCM (B-SPCM) Similarity Function';
h0 = plotSimilarityConfMatrix(S_spcm, title_str);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%          Compute other Similarity Functions for Comparison          %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%% Affine Invariant Riemannian Metric %%%%%%%%%%%%%%%
tic;
S_riem = compute_cov_sim( sigmas, 'RIEM' );
toc;

%%%%%%% Visualize Bounded Similarity Confusion Matrix %%%%%%%%%%%%%%
if exist('h1','var') && isvalid(h1), delete(h1);end
title_str = 'Affine Invariant Riemannian Metric (RIEM)';
h1 = plotSimilarityConfMatrix(S_riem, title_str);

%%%%%%%%%%%%%%% 'LERM': Log-Euclidean Riemannina Metric %%%%%%%%%%%%%%%
tic;
S_lerm = compute_cov_sim( sigmas, 'LERM' );
toc;

%%%%%%% Visualize Bounded Similarity Confusion Matrix %%%%%%%%%%%%%%
if exist('h2','var') && isvalid(h2), delete(h2);end
title_str = 'Log-Euclidean Riemannian Metric (LERM)';
h2 = plotSimilarityConfMatrix(S_lerm, title_str);

%%%%%%%%%%%%%%% 'KLDM': Kullback-Liebler Divergence Metric %%%%%%%%%%%%%%%
tic;
S_kldm = compute_cov_sim( sigmas, 'KLDM' );
toc;

%%%%%%% Visualize Bounded Similarity Confusion Matrix %%%%%%%%%%%%%%
if exist('h3','var') && isvalid(h3), delete(h3);end
title_str = 'Kullback-Liebler Divergence Metric (KLDM)';
h3 = plotSimilarityConfMatrix(S_kldm, title_str);

%%%%%%%%%%%%%%% 'JBLD': Jensen-Bregman LogDet Divergence %%%%%%%%%%%%%%%
tic;
S_jbld = compute_cov_sim( sigmas, 'JBLD' );
toc;

%%%%%%% Visualize Bounded Similarity Confusion Matrix %%%%%%%%%%%%%%
if exist('h4','var') && isvalid(h3), delete(h3);end
title_str = 'Jensen-Bregman LogDet Divergence (JBLD)';
h4 = plotSimilarityConfMatrix(S_jbld, title_str);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Apply Standard Similarity-based Clustering Algorithms on Similarity Matrices
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Choose Similarity Metric (SPCM, RIEM, LERM, KLDM, JBLD ) %%%
% S_type = {'RIEM', 'LERM', 'KLDM', 'JBLD', 'B-SPCM'};
S_type = {'B-SPCM'};

%%% Choose Clustering Algorithm %%%
% 'affinity': Affinity Propagation
% 'spectral': Spectral Clustering w/k-means
C_type = 'Affinity';
% C_type = 'Spectral';

%%% Selection of M-dimensional Spectral Manifold (for Spectral Clustering) %%%
% mani = 'auto';
mani = 'known';

%%%%%%%%% Compute clusters from Similarity Matrices %%%%%%%%%
clc;
runs = 10;
Purities = zeros(length(S_type),runs);
NMIs     = zeros(length(S_type),runs);
F1s      = zeros(length(S_type),runs);
Ks       = zeros(length(S_type),runs);
for j =1:runs
for i=1:length(S_type)
        
    s_type = S_type{i};
    
    switch s_type 
        case 'B-SPCM' 
            S = S_spcm;
        case 'RIEM' 
            S = S_riem;
        case 'LERM'
            S = S_lerm;
        case 'KLDM'
            S = S_kldm;
        case 'JBLD'
            S = S_jbld;
    end
    
    switch C_type
        case 'Affinity'            
            fprintf('Clustering %s similarities via Affinity Propagation...\n', s_type);
            tic;                       
            % Hacks such that AP works
            max_sim =  max(max(S));
            if strcmp(s_type,'B-SPCM')                
                if (strcmp(dataset_name,'Toy 6D') || strcmp(dataset_name,'Synthetic DT-MRI') || strcmp(dataset_name,'Real DT-MRI'))
                    D_aff = -(S + eye(size(S)));
                else
                    D_aff = (S - eye(size(S)));
                end
            else
                D_aff = (S - 2*eye(size(S))*max_sim);
            end           
            damp = 0.5;
            [E K labels idx] = affinitypropagation(D_aff, damp);
            toc;
            clus_method = 'Affinity Propagation';
            
        case 'Spectral'
            fprintf('Clustering %s similarities via Spectral Clustering...\n', s_type);
            tic;
            
            % Project point to spectral manifold from Similarity Matrix
            % Choose known M for the spectral manifold dimension
            
            switch mani
                case 'auto'
                    % Automatically discover M from the Eigenvalue of the Laplacian
                    [Y, d, thres] = spectral_DimRed(S,[]);
                    s_norm = normalize_soft(softmax(d));    
                    M = sum(s_norm <= thres);                    
                    
                case 'known'
                    % Use M = true # clusters
                    M = length(unique(true_labels));            
                    [Y, d, thres] = spectral_DimRed(S, M);    
            end
            
            % K-means 
            cluster_options             = [];
            cluster_options.method_name = 'kmeans';
            cluster_options.K           = M;
            result                      = ml_clustering(Y',cluster_options,'Distance','sqeuclidean','MaxIter',500);  
            labels = result.labels;
            clus_method = 'Spectral Clustering';
            toc;
    end
    
    [Purity NMI F] = cluster_metrics(true_labels, labels');
    K = length(unique(labels));
    
    Purities(i,j) = Purity;
    NMIs(i,j)     = NMI;
    F1s(i,j)      = F;
    Ks(i,j)       = K;
    
    fprintf('Number of clusters: %d, Purity: %1.2f, NMI Score: %1.2f, F measure: %1.2f \n',K, Purity, NMI, F);
    fprintf('*************************************************************\n');

end
end

%% Compute Stats for Paper
clc;
for i=1:length(S_type)    
  fprintf('%s Clustering with %s-- K: %1.2f +- %1.2f, Purity: %1.2f +-%1.2f , NMI Score: %1.2f +-%1.2f, F measure: %1.2f+-%1.2f \n', ...
      C_type, S_type{i}, mean(Ks(i,:)), std(Ks(i,:)), mean(Purities(i,:)), std(Purities(i,:)), mean(NMIs(i,:)), std(NMIs(i,:)), ...
      mean(F1s(i,:)), std(F1s(i,:)));
end
