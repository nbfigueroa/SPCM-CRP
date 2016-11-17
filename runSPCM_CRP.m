function [ labels_sdcrp ] = runSPCM_CRP( sigmas,  options )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here



%%%%%%%%%%%%%%%%%%%%%%
% Set Hyper-parameter
% %%%%%%%%%%%%%%%%%%%%%
% Tolerance for SPCM decay function 
tau = options.tau; % [1, 100] Set higher for noisy data, Set 1 for ideal data 

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
title('Bounded Similarity Function (B-SPCM) Matrix','Fontsize',14)
colormap(pink)
colorbar 
axis square


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Run Spectral Manifold Algorithm
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('Computing Spectral Dimensionality Reduction based on SPCM Similarity Function...\n');
tic;

% Automatic Discovery of Dimensionality of M Manifold
[Y, d, thres, V] = spectral_DimRed(S,[]);
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
title(tit, 'Fontsize',14)
toc;
fprintf('*************************************************************\n');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Discover Clusters using sd-CRP-MM %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all;
fprintf('Clustering via sd-CRP...\n');
tic;
[Psi_MAP] = run_sdCRPMM(Y, S);
toc;

figure('Color', [1 1 1])
labels_sdcrp = Psi_MAP.Z_C';
imagesc(labels_sdcrp)
axis equal tight
colormap(pink)
sdcrp_tables = length(unique(labels_sdcrp));
fprintf('MAP Cluster estimate recovered at iter %d: %d\n', Psi_MAP.iter, sdcrp_tables);


end

