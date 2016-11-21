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

%% Select Dataset:
%% 1) Toy 3D dataset, 5 Samples, 2 clusters (c1:3, c2:2)
[sigmas, true_labels] = load_toy_dataset('3d', display, randomize);

%% 2)  Toy 6D dataset, 60 Samples, 3 clusters (c1:20, c2:20, c3: 20)
[sigmas, true_labels] = load_toy_dataset('6d', display, randomize);

%% 3) Real 6D dataset, task-ellipsoids, 105 Samples, 3 clusters (c1:63, c2:21, c3: 21)
% Path to data folder
data_path = '/home/nadiafigueroa/dev/MATLAB/SPCM-CRP/data/';
[sigmas, true_labels] = load_task_dataset(strcat(data_path,'6D-Grasps.mat'));

%% 4) Real XD dataset, Covariance Features from ETH80 Dataset, N Samples, k clusters (c1:63, c2:21, c3: 21)


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute Similarity Matrix from b-SPCM Function for dataset
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

% %%%%%% Visualize Bounded Similarity Confusion Matrix %%%%%%%%%%%%%%
if exist('h0','var') && isvalid(h0), delete(h0);end
title_str = 'Bounded Similarity Function (B-SPCM) Matrix';
h0 = plotSimilarityConfMatrix(S, title_str);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%
% Run Spectral Manifold Algorithm
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('Computing Spectral Dimensionality Reduction based on SPCM Similarity Function...\n');

% Automatic Discovery of Dimensionality of M Manifold
tic;
[Y, d, thres, V] = spectral_DimRed(S,[]);
s_norm = normalize_soft(softmax(d));    
M = sum(s_norm <= thres);
toc;
fprintf('*************************************************************\n');

%%% Plot Spectral Manifold Representation for M=2 or M=3
if exist('h1','var') && isvalid(h1), delete(h1);end
h1 = plotSpectralManifold(Y, true_labels, d,thres, s_norm, M);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Discover Clusters using sd-CRP-MM %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('Clustering via sd-CRP...\n');
tic;
[Psi_MAP] = run_sdCRPMM(Y, S);
toc;

figure('Color', [1 1 1])
subplot(2,1,1)
imagesc(true_labels)
axis equal tight
colormap(pink)
title('True Labels', 'FontWeight', 'Bold')

subplot(2,1,2)
labels_sdcrp = Psi_MAP.Z_C';
imagesc(labels_sdcrp)
axis equal tight
colormap(pink)
sdcrp_tables = length(unique(labels_sdcrp));
fprintf('MAP Cluster estimate recovered at iter %d: %d\n', Psi_MAP.iter, sdcrp_tables);
[Purity NMI F] = cluster_metrics(true_labels, labels_sdcrp');
title_string = sprintf('Clustering from sdCRP K=%d, Purity: %1.2f, NMI Score: %1.2f, F measure: %1.2f',sdcrp_tables, Purity, NMI, F);
title(title_string, 'FontWeight', 'Bold')
fprintf('sd-CRP LP: %d and Purity: %1.2f, NMI Score: %1.2f, F measure: %1.2f \n', Psi_MAP.LogProb, Purity, NMI, F);
fprintf('*************************************************************\n');

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot sd-CRP-MM Results on Projected Data %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Extract cluster parameters
Mu = Psi_MAP.Cluster_Mu;
Pr = Psi_MAP.Cluster_Pr;
Sigma = zeros(size(Pr,1),size(Pr,1),size(Pr,2));
for i=1:size(Pr,2)
    Sigma(:,:,i) = diag(Pr(:,i));
end        00

% Plot Gaussians on Projected Data
figure('Color', [1 1 1])
if (M == 2) || (M == 3)
    % Plot M-Dimensional Points of Spectral Manifold
    idx_label   = labels_sdcrp;
    pred_clust = length(unique(labels_sdcrp));
    
    if M==2    
        for jj=1:pred_clust
            clust_color = [rand rand rand];                                             
            scatter(Y(1,idx_label==jj),Y(2,idx_label==jj), 50, clust_color, 'filled'); hold on;
            plotGMM(Mu(:,jj), Sigma(:,:,jj), clust_color, 1);
            alpha(.5)
        end 
        xlabel('$y_1$');ylabel('$y_2$');
        colormap(hot)
        grid on
        title('$\Sigma_i$-s Respresented in 2-d Spectral space')
    end

    if M==3
        subplot(3,1,1)
        clust_color = zeros(length(pred_clust),3);
        for jj=1:pred_clust
            clust_color(jj,:) = [rand rand rand];
            scatter(Y(1,idx_label==jj),Y(2,idx_label==jj), 50, clust_color(jj,:), 'filled');hold on  
            plotGMM(Mu(1:2,jj), Sigma(1:2,1:2,jj), clust_color(jj,:), 1);
            alpha(.5)
        end
        xlabel('$y_1$');ylabel('$y_2$');
        axis auto
        colormap(hot)
        grid on
        title('$\Sigma_i$-s Respresented in 2-d [$y_1$-$y_2$] Spectral space', 'Fontsize',14)
        
        subplot(3,1,2)
        for jj=1:pred_clust
            scatter(Y(1,idx_label==jj),Y(3,idx_label==jj), 50, clust_color(jj,:), 'filled');hold on  
            plotGMM(Mu([1 3],jj), Sigma([1 3],[1 3],jj), clust_color(jj,:), 1);
            alpha(.5)
        end
        xlabel('$y_1$');ylabel('$y_3$');
        axis auto
        colormap(hot)
        grid on
        title('$\Sigma_i$-s Respresented in 2-d [$y_1$-$y_3$] Spectral space', 'Fontsize',14)
        
        subplot(3,1,3)
        for jj=1:pred_clust
            scatter(Y(2,idx_label==jj),Y(3,idx_label==jj), 50, clust_color(jj,:), 'filled');hold on  
            plotGMM(Mu(2:3,jj), Sigma(2:3,2:3,jj), clust_color(jj,:), 1);
            alpha(.5)
        end
        xlabel('$y_2$');ylabel('$y_3$');
        axis auto
        colormap(hot)
        grid on
        title('$\Sigma_i$-s Respresented in 2-d [$y_2$-$y_3$] Spectral space', 'Fontsize',14)
        
    end
end


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Discover Clusters using CRP-MM %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all;
fprintf('Clustering via DP(CRP)-MM (MCMC Estimation)...\n');
% .. test dpgmm (mcmc) here
tic;
[labels_dpgmm,Theta_dpgmm,w,llh] = mixGaussGb(Y);
toc;
figure

% Extract learnt parameters
k_dpgmm  = length(w);
Mu_dpgmm = zeros(size(Y,1),k_dpgmm);

% Sigma = model_dpgmm.U_


fprintf('Clustering via DP(CRP)-MM (Variational Estimation)...\n');
% ...
% .. test vbdpgmm here
% ...

% Extract learnt parameters
%%
figure('Color', [1 1 1])
subplot(2,1,1)
imagesc(true_labels)
axis equal tight
colormap(pink)
title('True Labels', 'FontWeight', 'Bold')

subplot(2,1,2)
labels_crp = Psi_crpmm(end).classes';
imagesc(labels_crp)
axis equal tight
colormap(pink)
crp_tables = length(unique(labels_crp));
[Purity NMI F] = cluster_metrics(true_labels, labels_crp');
title_string = sprintf('Clustering from CRP-MM K=%d, Purity: %1.2f, NMI Score: %1.2f, F measure: %1.2f',crp_tables, Purity, NMI, F);
title(title_string, 'FontWeight', 'Bold')
fprintf('CRP-MM Purity: %1.2f, NMI Score: %1.2f, F measure: %1.2f \n', Purity, NMI, F);
fprintf('*************************************************************\n');
