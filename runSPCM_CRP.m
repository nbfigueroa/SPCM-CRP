function labels_sdcrp = runSPCM_CRP(sigmas, options)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
% %%%%%%%%%%%%%%%%%%%%%
% Set Hyper-parameters
% %%%%%%%%%%%%%%%%%%%%%
% Tolerance for SPCM decay function 
tau = options.tau; 

% For Spectral Manifold Algorithm
M = options.M;   % M-dimension of Spectral Manifold

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute Similarity Function for dataset
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N = size(sigmas,1);
fprintf('Computing SPCM Similarity Function for %dx%d observations...\n',N,N);
tic;
spcm = ComputeSPCMfunctionProb(sigmas, tau);  
toc;
S = spcm(:,:,2); % Bounded Decay SPCM Similarity Matrix
fprintf('*************************************************************\n');


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Clustering Directly from SPCM and Spectral Manifold
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% %%%%%% Visualize Bounded Similarity Confusion Matrix %%%%%%%%%%%%%%
figure('Color',[1 1 1], 'Position',[1300 100 597 908])


% %%% Compute clusters from Similarity Matrix using Affinity Propagation %%%%%%
% D_aff = S - eye(size(S));
% fprintf('Clustering via Affinity Propagation...\n');
% tic;
% [E K labels_aff idx] = affinitypropagation(D_aff);
% toc;
% fprintf('*************************************************************\n');
% subplot(4,1,2)
% imagesc(labels_aff')
% title('Clustering from Aff. Prop. on SPCM function')
% axis equal tight
% colormap(pink)

fprintf('Computing Spectral Dimensionality Reduction based on SPCM Similarity Function...\n');
tic;
[Y, d] = spectral_DimRed(S,M);                
toc;
fprintf('*************************************************************\n');

% %%% Compute clusters from Similarity Matrix and Spectral Manifold using sd-CRP %%%%%%  
fprintf('Clustering via sd-CRP...\n');
tic;
[Psi_MAP] = run_sdCRP(Y, S);
toc;
labels_sdcrp = Psi_MAP.Z_C';
sdcrp_tables = length(unique(labels_sdcrp));
fprintf('MAP Cluster estimate recovered at iter %d: %d\n', Psi_MAP.iter, sdcrp_tables)


subplot(3,1,2)
imagesc(labels_sdcrp)
title('Clustering from sdCRP on Spectral Space/SPCM')
axis equal tight
colormap(pink)


subplot(3,1,1)
imagesc(S)
title('Similarity Function Kernel Matrix')
axis equal tight
colormap(pink)
colorbar 


% Plot M-Dimensional Points of Spectral Manifold
subplot(3,1,3)
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
    title('Models Respresented in 2-d Spectral Manifold')
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
    title('Models Respresented in 3-d Spectral Manifold')
end

end

