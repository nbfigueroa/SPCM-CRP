%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compare similarity functions and clustering algorithms
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% To run some of the function on this script you need 
% the ML_toolbox in your MATLAB path.

clc
clear all
close all

% Set to 1 if you want to display Covariance Matrices
display = 1;

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
tau = 10; % [1, 100] Set higher for noisy data, Set 1 for ideal data 

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
imagesc(spcm(:,:,1))
title('Similarity Function (SPCM) Matrix','Fontsize',16)
colormap(pink)
colorbar 
axis square

figure('Color',[1 1 1])
imagesc(S_spcm)
title('Bounded Similarity Function (B-SPCM) Matrix','Fontsize',16)
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

% Plot Results
figure('Color',[1 1 1])
imagesc(S_riem)
title('Affine Invariant Riemannian Metric (RIEM)','Fontsize',16)
colormap(pink)
colorbar 
axis square

%% %%%%%%%%%%%%% 'LERM': Log-Euclidean Riemannina Metric %%%%%%%%%%%%%%%
fprintf('Computing LERM Similarity Function for %dx%d observations...\n',N,N);
tic;
S_lerm = compute_cov_sim( sigmas, 'LERM' );
toc;
fprintf('*************************************************************\n');


% Plot Results
figure('Color',[1 1 1])
imagesc(S_lerm)
title(' Log-Euclidean Riemannian Metric (LERM)','Fontsize',16)
colormap(pink)
colorbar 
axis square

%% %%%%%%%%%%%%% 'KLDM': Kullback-Liebler Divergence Metric %%%%%%%%%%%%%%%
fprintf('Computing KLDM Similarity Function for %dx%d observations...\n',N,N);
tic;
S_kldm = compute_cov_sim( sigmas, 'KLDM' );
toc;
fprintf('*************************************************************\n');

% Plot Results
figure('Color',[1 1 1])
imagesc(S_kldm)
title(' Kullback-Liebler Divergence Metric (KLDM)','Fontsize',16)
colormap(pink)
colorbar 
axis square

%% %%%%%%%%%%%%% 'JBLD': Jensen-Bregman LogDet Divergence %%%%%%%%%%%%%%%
fprintf('Computing JBLD Similarity Function for %dx%d observations...\n',N,N);
tic;
S_jbld = compute_cov_sim( sigmas, 'JBLD' );
toc;
fprintf('*************************************************************\n');

% Plot Results
figure('Color',[1 1 1])
imagesc(S_jbld)
title('Jensen-Bregman LogDet Divergence (JBLD)','Fontsize',16)
colormap(pink)
colorbar 
axis square
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Apply Standard Similarity-based Clustering Algorithms on Similarity Matrices
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Choose Similarity Metric (SPCM, RIEM, LERM, KLDM, JBLD ) %%%
% S_type = {'B-SPCM','RIEM', 'LERM', 'KLDM', 'JBLD'};
S_type = {'RIEM'};

%%% Choose Clustering Algorithm %%%
% 'affinity': Affinity Propagation
% 'spectral': Spectral Clustering w/k-means
% C_type = 'affinity';
C_type = 'spectral';

%%% Selection of M-dimensional Spectral Manifold (for Spectral Clustering) %%%
% mani = 'auto';
mani = 'known';

%%%%%%%%% Compute clusters from Similarity Matrices %%%%%%%%%
figure('Color',[1 1 1])

%%% Plotting true labels
s_plots = length(S_type) + 1;
subplot(s_plots, 1, 1);
imagesc(true_labels)
title('True Labels', 'FontSize',16)
axis equal tight
colormap(pink)
set(gca,'XTickLabel',[]);
set(gca,'YTickLabel',[]);
grid on

%# create cell arrays of number labels
for jj=1:length(S)
text(jj, 1, num2str(jj),'color','r',...
    'HorizontalAlignment','center','VerticalAlignment','middle','Fontsize',20);
end

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
        case 'affinity'            
            fprintf('Clustering via Affinity Propagation...\n');
            tic;
            max_sim =  max(max(S));
            D_aff = S - eye(size(S))*max_sim;
            damp = 0.15;
            [E K labels idx] = affinitypropagation(D_aff, damp);
            toc;
            clus_method = 'Affinity Propagation';
            
        case 'spectral'
            fprintf('Clustering via Spectral Clustering...\n');
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
    
    fprintf('Number of clusters: %d, Purity: %1.2f, NMI Score: %1.2f, F measure: %1.2f \n',K, Purity, NMI, F);
    fprintf('*************************************************************\n');

    subplot(s_plots, 1, i + 1);
    imagesc(labels')
    
%     title_string = sprintf('Method (%s) Metric (%s)  [K=%d, Purity: %1.2f, NMI: %1.2f, F-measure: %1.2f]', clus_method, s_type, K, Purity, NMI, F);
    title_string = sprintf('Method (%s) Metric (%s) ', clus_method, s_type);
    title(title_string, 'FontSize',16)
    axis equal tight
    colormap(pink)
    set(gca,'XTickLabel',[]);
    set(gca,'YTickLabel',[]);
    grid on
   
%   create cell arrays of number labels
    for jj=1:length(S)
    text(jj, 1, num2str(jj),'color','r',...
        'HorizontalAlignment','center','VerticalAlignment','middle','Fontsize',20);
    end

end


%%

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


