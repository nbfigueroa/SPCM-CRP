%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Test SPCM Similarity with CRP Clustering Algorithm
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute Similarity Matrix from b-SPCM Function for dataset
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
S = spcm(:,:,2); % Bounded Decay SPCM Similarity Matrix
fprintf('*************************************************************\n');

% %%%%%% Visualize Bounded Similarity Confusion Matrix %%%%%%%%%%%%%%
figure('Color',[1 1 1])
imagesc(S)
title('Bounded Similarity Function (B-SPCM) Matrix')
colormap(pink)
colorbar 
axis square

%% %%%%%%%%%%%%%%%%%%%%%%%%%%
% Run Spectral Manifold Algorithm
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('Computing Spectral Dimensionality Reduction based on SPCM Similarity Function...\n');
tic;

% Automatic Discovery of Dimensionality of M Manifold
[Y, d, thres] = spectral_DimRed(S,[]);
s_norm = normalize_soft(softmax(d));    
M = sum(s_norm <= thres);

figure('Color',[1 1 1])
if (M == 2) || (M == 3)
    subplot(2,1,1)
end
plot(s_norm,'-*r'); hold on
plot(thres*ones(1,length(d)),'--k','LineWidth', 2); hold on
xlabel('Eigenvalue Index')
ylabel('Normalized Eigenvalue Softmax')
tit = strcat('Eigenvalue Analysis for Manifold Dimensionality -> M = ', num2str(M));
title(tit)
toc;
fprintf('*************************************************************\n');

if (M == 2) || (M == 3)
    subplot(2,1,2)
    % Plot M-Dimensional Points of Spectral Manifold
    idx_label   = true_labels;
    true_clust = length(unique(true_labels));
    if M==2    
        for jj=1:true_clust
            clust_color = [rand rand rand];
            scatter(Y(1,idx_label==jj),Y(2,idx_label==jj), 100, clust_color, 'filled');hold on                      
        end   
        grid on
        title('Sigmas Respresented in 2-d Spectral space')
    end

    if M==3
        for jj=1:true_clust
            clust_color = [rand rand rand];
            scatter3(Y(1,idx_label==jj),Y(2,idx_label==jj),Y(3,idx_label==jj), 100, clust_color, 'filled');hold on        
        end
        grid on
        title('Sigmas Respresented in 3-d Spectral space')
    end
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Discover Clusters using sd-CRP %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('Clustering via sd-CRP...\n');
tic;
[Psi_MAP] = run_sdCRP(Y, S);
toc;

figure('Color', [1 1 1])
labels_sdcrp = Psi_MAP.Z_C';
imagesc(labels_sdcrp)
axis equal tight
colormap(pink)
sdcrp_tables = length(unique(labels_sdcrp));
fprintf('MAP Cluster estimate recovered at iter %d: %d\n', Psi_MAP.iter, sdcrp_tables);
[Purity NMI F] = cluster_metrics(true_labels, labels_sdcrp');
title_string = sprintf('Clustering from sdCRP K=%d, Purity: %1.2f, NMI Score: %1.2f, F measure: %1.2f',sdcrp_tables, Purity, NMI, F);
title(title_string)
fprintf('sd-CRP LP: %d and Purity: %1.2f, NMI Score: %1.2f, F measure: %1.2f \n', Psi_MAP.LogProb, Purity, NMI, F);
fprintf('*************************************************************\n');
