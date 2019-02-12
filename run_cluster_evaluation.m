%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Evaluation of Clustering Schemes for SPD Matrices on Different Datasets     %%                        %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
% 13:  HMM Emission Models - Task4  (26D)  ... TODO (Peeling)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%% Data Loading Parameter Description %%%%%%%%%%%%%%%%%%%%%%
% display:   [0,1]  -- Display Covariance matrices in their own format
% randomize: [0,1]  -- Randomize the Covariance Matrices indices
% pkg_dir:  {'./data/'} -- Path to data folder
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

pkg_dir = '/home/nbfigueroa/Dropbox/PhD_papers/journal-draft/new-code/SPCM-CRP';
display      = 0;       % display SDP matrices (if applicable)
randomize    = 0;       % randomize idx
dataset      = 5;       % choosen dataset from index above
sample_ratio = 1;       % sub-sample dataset [0.01 - 1]
[sigmas, true_labels, dataset_name] = load_SPD_dataset(dataset, pkg_dir, display, randomize);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Train different clustering models on current dataset %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
repetitions       = 10; 
cluster_methods   = 4;
embedding_methods = 4;
clust_stats = [];

%%%%%%%%%%%%%%%% Compute Distances/Similarities of Dataset %%%%%%%%%%%%%%%%
% SPCM Distance (d_SP) and Similarity (\kappa_SP)
gamma   = 2;
spcm    = ComputeSPCMfunctionMatrix(sigmas, gamma, 2);  
D_SP    = spcm(:,:,1);
S_SP    = spcm(:,:,2);

% Log-Euclidean Riemannian Distance (d_LE)
[D_LE, distance_name] = computeSDP_distances(sigmas, 1);

% Embedding Hyper-parameters
pow_eigen  = 4;
l_sens     = 2;

for k=1:embedding_methods
    %%%%%%%%%%%%%% Extract Vector-Space Embedding of Dataset %%%%%%%%%%%%%%
    clear x_emb Y
    emb_options = [];
    emb_options.l_sensitivity = l_sens; %
    emb_options.distance_name = 'SPCM';
    emb_options.norm_K        = 1;
    emb_options.pow_eigen     = pow_eigen;
    emb_options.show_plots    = 0;
    switch k
        case 1
            % PCA on Tangent Space log-Eucl Mapping
            emb_options.expl_var      = 0.80;  % Explained variance for eigenvectors to keep
            emb_options.emb_type      = 0;     % 0: PCA on Tangent Space log-Eucl Mapping
            [x_emb, Y, pca_params, emb_name] = pcaTangentSpaceEmbedding(sigmas, emb_options);
            
        case 2
            % Kernel-PCA on (un)-deformed log-Euclidean Kernel
            emb_options.deform        = 0;           
            [x_emb, Y, K, K_SP, gamma_le]   = deformedKernelPCA_Embedding(D_LE, S_SP, emb_options);
            emb_name = 'Kernel PCA on (un-deformed) $k_{LE}(\cdot,\cdot)$';
            
        case 3
            % Kernel-PCA on SPCM deformed log-Euclidean Kernel
            emb_options.deform        = 1;           
            [x_emb, Y, K, K_SP, gamma_le]   = deformedKernelPCA_Embedding(D_LE, S_SP, emb_options);
            emb_name = 'Kernel PCA on SPCM deformed $k_{LE}^{SP}(\cdot,\cdot)$';
            
        case 4
            % Graph-Subspace Projection            
            [x_emb, Y] = graphEuclidean_Embedding(S_SP, 0, pow_eigen);
            emb_name = '(SPCM) Graph-Subspace Projection';
            
    end
    
    %%%%%%%%%%% Run Clustering on Chosen Embedding %%%%%%%%%%%
    Purity_s    = zeros(cluster_methods,repetitions);
    NMI_s       = zeros(cluster_methods,repetitions);
    ARI_s       = zeros(cluster_methods,repetitions);
    F2_s        = zeros(cluster_methods,repetitions);
    K_s         = zeros(cluster_methods,repetitions);
    for j = 1:repetitions
        for i=1:cluster_methods
            fprintf(' +++ Embedding Method %d with Cluster Method %d (Repetition %d) +++ \n',k,i,j);
            % Hyper-parameters for cluster methods
            est_options = [];
            est_options.maxK             = 15;
            est_options.fixed_K          = [];
            est_options.samplerIter      = 100;
            est_options.do_plots         = 0;              % Plot Estimation Stats
            est_options.dataset_name     = dataset_name;   % Dataset name
            est_options.true_labels      = true_labels;    % To plot against estimates
            if i < 4
                switch i
                    case 1
                        % GMM-EM Model Selection via BIC on Preferred Embedding
                        est_options.type        = 1;
                        
                    case 2
                        % CRP-GMM (Gibbs Sampler/Collapsed) on Preferred Embedding
                        est_options.type        = 2;
                        est_options.samplerIter = 500;
                        
                    case 3
                        % SPCM-CRP-MM (Collapsed Gibbs Sampler) on Preferred Embedding
                        est_options.type        = 0;
                        est_options.samplerIter = 200;                       
                        
                end
                
                % Fit GMM to Vector Data
                tic;
                clear Priors Mu Sigma est_labels
                [Priors, Mu, Sigma, est_labels, stats] = fitgmm_sdp(S_SP, Y, est_options);
                toc;
                
            else                
                % GMM-Oracle on Preferred Embedding (for comparison)
                clear Priors Mu Sigma est_labels
                [Priors, Mu, Sigma] = gmmOracle(Y, true_labels);
                [~, est_labels]     = my_gmm_cluster(Y, Priors, Mu, Sigma, 'hard', []);
            end
            
            %%%%%%%%%% Compute Cluster Metrics %%%%%%%%%%%%%
            [Purity_s(i,j), NMI_s(i,j), F2_s(i,j), ARI_s(i,j)] = cluster_metrics(true_labels, est_labels');
            K_s(i,j) =  length(unique(est_labels));
            
        end
    end
    clust_stats{k}.dataset   = dataset_name;
    clust_stats{k}.y_dim     = size(Y,1);
    clust_stats{k}.embedding = emb_name;
    clust_stats{k}.reps      = repetitions;
    clust_stats{k}.Purity    = Purity_s;
    clust_stats{k}.NMI       = NMI_s;
    clust_stats{k}.ARI       = ARI_s;
    clust_stats{k}.F2        = F2_s;
    clust_stats{k}.K         = K_s;    
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Visualize Clustering Results %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc;
[Purity_stats, NMI_stats, ARI_stats, F2_stats, K_stats] = extract_cluster_stats(clust_stats);
h_stats = plot_cluster_stats(Purity_stats, NMI_stats, ARI_stats, F2_stats, K_stats, clust_stats{1}.dataset) 
