%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Test SPCM-CRP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
close all

% %%%%%%%%%%%%%%%%%%%%%
% Set Hyper-parameter
% %%%%%%%%%%%%%%%%%%%%%
% Tolerance for SPCM decay function 
tau = 50; % [1, 100] Set higher for noisy data, Set 1 for ideal data 

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute Similarity Matrix from b-SPCM Function for dataset
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Number of datapoints
N = length(sigmas);
fprintf('Computing SPCM Similarity Function for %dx%d observations...\n',N,N);
tic;
spcm = ComputeSPCMfunctionMatrix(sigmas, tau);  
toc;
S = spcm(:,:,2); % Bounded Decay SPCM Similarity Matrix
fprintf('*************************************************************\n');

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Clustering Directly from SPCM and Spectral Manifold
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all

% %%%%%% Visualize Bounded Similarity Confusion Matrix %%%%%%%%%%%%%%
figure('Color',[1 1 1], 'Position',[1300 100 397 908])

subplot(6,1,1)
imagesc(S)
title('Similarity Function Kernel Matrix')
colormap(pink)
colorbar 

% %%% Compute clusters from Similarity Matrix using Affinity Propagation %%%%%%
subplot(6,1,2)
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

% %%% Compute clusters from Similarity Matrix using K-means on Spectral Manifold %%%%%%       

fprintf('Computing Spectral Dimensionality Reduction based on SPCM Similarity Function...\n');
tic;

% Dimensionality of M Manifold
[Y, d, thres] = spectral_DimRed(S,[]);
s_norm = normalize_soft(softmax(d));    
M = sum(s_norm <= thres);

subplot(6,1,3)
plot(s_norm,'-*r'); hold on
plot(thres*ones(1,length(d)),'--k','LineWidth', 2); hold on
xlabel('Eigenvalue Index')
ylabel('Normalized Eigenvalue Softmax')
tit = strcat('Eigenvalue Analysis for Manifold Dimensionality= ', num2str(M));
title(tit)
toc;
fprintf('*************************************************************\n');

%%%% Compute clusters from Similarity Matrix and Spectral Manifold using sd-CRP %%%%%%  
fprintf('Clustering via sd-CRP...\n');
tic;
[Psi_MAP] = run_sdCRP(Y, S);
toc;

subplot(6,1,5)
labels_sdcrp = Psi_MAP.Z_C';
imagesc(labels_sdcrp)
title('Clustering from sdCRP on Spectral Space/SPCM')
axis equal tight
colormap(pink)
sdcrp_tables = length(unique(labels_sdcrp));
fprintf('MAP Cluster estimate recovered at iter %d: %d\n', Psi_MAP.iter, sdcrp_tables);
NMI = CalcNMI(true_labels, labels_sdcrp); %% CHANGE THIS FUNCTION
fprintf('sd-CRP LP: %d and NMI Score: %d\n', Psi_MAP.LogProb, NMI);
fprintf('*************************************************************\n');

% Plot M-Dimensional Points of Spectral Manifold
subplot(6,1,4)
idx_ddcrp = labels_sdcrp;
if M==2    
    for jj=1:sdcrp_tables
        clust_color = [rand rand rand];
        scatter(Y(1,idx_ddcrp==jj),Y(2,idx_ddcrp==jj), 100, clust_color, 'filled')
        hold on            
        scatter(Psi_MAP.Cluster_Mu(1,jj), Psi_MAP.Cluster_Mu(2,jj), 200, clust_color, '*')
        hold on
        
        %%% Plot ellipse when ready
    end   
    grid on
    title('Models Respresented in 2-d Spectral space')
end
   
if M==3
    for jj=1:sdcrp_tables
        clust_color = [rand rand rand];
        scatter3(Y(1,idx_ddcrp==jj),Y(2,idx_ddcrp==jj),Y(3,idx_ddcrp==jj), 100, clust_color, 'filled')
        hold on                     
        scatter3(Psi_MAP.Cluster_Mu(1,jj), Psi_MAP.Cluster_Mu(2,jj), Psi_MAP.Cluster_Mu(3,jj), 200, clust_color, '*')
        hold on       
        
        %%% Plot ellipsoid when ready
    end
    grid on
    title('Models Respresented in 3-d Spectral space')
end

% Apply K-means on M-dimensional spectral space (For comparison purposes)
k = M;
fprintf('Clustering via k-means with k=%d...\n',k);
opts = statset('Display','final');
tic;
[idx,ctrs] = kmeans(Y',k,'Distance','sqEuclidean', 'Replicates', 5,'Options',opts);
toc;

subplot(6,1,6)
labels_speckmeans = idx';
imagesc(labels_speckmeans)
title_kmeans = strcat(strcat('Clustering from K (',num2str(k)),')-means on Spectral Space of SPCM');
title(title_kmeans)
axis equal tight
colormap(pink)
NMI = CalcNMI(true_labels, labels_speckmeans);
fprintf('K-means NMI Score: %d\n', NMI);

