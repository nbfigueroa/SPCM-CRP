%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Test SPCM Similarity with CRP Clustering Algorithm
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
clear all
close all

% Set to 1 if you want to display Covariance Matrices
display = 0;

% Set to 1 if you want to randomize the Covariance Matrices indices
randomize = 0;

% Path to data folder
data_path = './data/';

%% Select Dataset:
%% 1) Toy 3D dataset, 5 Samples, 2 clusters (c1:3, c2:2)
[sigmas, true_labels] = load_toy_dataset('3d', display, randomize);

%% 2)  Toy 6D dataset, 60 Samples, 3 clusters (c1:20, c2:20, c3: 20)
[sigmas, true_labels] = load_toy_dataset('6d', display, randomize);

%% 3) Real 6D dataset, task-ellipsoids, 105 Samples, 3 clusters (c1:63, c2:21, c3: 21)
[sigmas, true_labels] = load_task_dataset(data_path); %<== ADD RANDOMIZE OPTION

%% 4) Real 400D dataset, Covariance Features from ETH80 Dataset, 40 Samples, 8 classes/clusters (c1:10, c2:10,.., c8: 10)
split = 10;
[sigmas, true_labels] = load_eth80_dataset(data_path,split); %<== ADD RANDOMIZE OPTION

%% 5) Real XD dataset, Covariance Features from YouTube Dataset, N Samples, Y classes/clusters 
split = 1;
[sigmas, true_labels] = load_youtube_dataset(data_path,split);

%% 6) Real 3D dataset, Diffusion Tensors from fanTDasia Dataset, 1024 Samples, 10/5 clusters

% Load MRI Image and Parameters
S = openFDT(strcat(data_path,'./fandtasia_demo/fandtasia_demo.fdt'));
params = textread(strcat(data_path,'./fandtasia_demo/fandtasia_demo.txt'));

% Extract and plot Gradient orientations
GradientOrientations=params(:,[1:3]);
b_value=params(:,4);
g=GradientOrientations([2:47],:);

% Estimate DTI from Gradient Orientation and b_value
G=constructMatrixOfMonomials(GradientOrientations, 2);
C=constructSetOf81Polynomials(2)';
P=G*C;P=[-diag(b_value)*P ones(size(GradientOrientations,1),1)];
DTI=zeros(3,3,size(S,1),size(S,2));S0=zeros(size(S,1),size(S,2));
for i=1:size(S,1)
   for j=1:size(S,2)
      y=log(squeeze(S(i,j,1,:)));
      x=lsqnonneg(P, y);
      T = C * x([1:81]);
      UniqueTensorCoefficients(:,i,j)=T;
      DTI(:,:,i,j)=[T(6) T(5)/2 T(4)/2
      T(5)/2 T(3) T(2)/2
      T(4)/2 T(2)/2 T(1)];
      S0(i,j)=exp(x(82));
   end
end

%% Plot DTI and Mean Diffusivity Values

figure('Color',[1 1 1]);
plotDTI(DTI,0.002);
title('REAL DT-MRI of Rat Hippocampi')

% Compute Mean Diffusivity
mean_diffusivity = zeros(size(DTI,3),size(DTI,4));
for i=1:size(DTI,3)
    for j=1:size(DTI,4)
        mean_diffusivity(i,j)=trace(DTI(:,:,i,j))/3;
    end
end
mean_diffusivity = flipud(mean_diffusivity);

% Plot Mean Diffusivity of Diffusion Tensors
figure('Color',[1 1 1]);
imagesc(mean_diffusivity)
colormap(pink)
colorbar
axis square
title('Mean Diffusivity of Diffusion Tensors')

% Compute Fractional Anisotropy
frac_anisotropy = zeros(size(DTI,3),size(DTI,4));
for i=1:size(DTI,3)
    for j=1:size(DTI,4)
        [eigenvectors,l] = eig(DTI(:,:,i,j));
        m=(l(1,1)+l(2,2)+l(3,3))/3;
        frac_anisotropy(i,j)=sqrt(3/2)*sqrt((l(1,1)-m)^2+(l(2,2)-m)^2+(l(3,3)-m)^2)/sqrt(l(1,1)^2+l(2,2)^2+l(3,3)^2);;
    end
end
frac_anisotropy = flipud(frac_anisotropy);

% Plot Fractional Anisotropy of Diffusion Tensors
figure('Color',[1 1 1]);
imagesc(frac_anisotropy)
colormap(pink)
colorbar
axis square
title('Fractional Anisotropy of Diffusion Tensors')


% Create Tensor Dataset to Cluster
% ...

% Generate labels from Fractional Anisotropy Value
% ...

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute Similarity Matrix from B-SPCM Function for dataset
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% %%%%%%%%%%%%%%%%%%%%% Set Hyper-parameter %%%%%%%%%%%%%%%%%%%%%%%%
% Tolerance for SPCM decay function 
tau = 1; % [1, 100] Set higher for noisy data, Set 1 for ideal data 

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

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Run Spectral Manifold Algorithm
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% %%%%%% Automatic Discovery of Dimensionality of M Manifold % %%%%%%
fprintf('Computing Spectral Dimensionality Reduction based on SPCM Similarity Function...\n');
tic;
M = [];
[Y, d, thres, V] = spectral_DimRed(S, M);

if isempty(M)
    s_norm = normalize_soft(softmax(d));
    M = sum(s_norm <= thres);
end

toc;
fprintf('*************************************************************\n');

% %%%%%% Plot Spectral Manifold Representation for M=2 or M=3 % %%%%%
if exist('h1','var') && isvalid(h1), delete(h1);end
h1 = plotSpectralManifold(Y, true_labels, d,thres, s_norm, M);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Discover Clusters using sd-CRP-MM %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% %%%%%% Non-parametric Clustering on Manifold Data with Sim prior % %%%%%%
fprintf('Clustering via sd-CRP...\n');
tic;
[Psi_MAP] = run_sdCRPMM(Y, S);
toc;
fprintf('*************************************************************\n');

% %%%%%%%% Plot Clustering Results against True Labels % %%%%%%%%
if exist('h2','var') && isvalid(h2), delete(h2);end
clust_type = 'sd-CRP-MM';
est_labels = Psi_MAP.Z_C';
[ Purity NMI F h2 ] = plotClusterResults( true_labels, est_labels, clust_type );
fprintf('MAP Cluster estimate recovered at iter %d: %d\n', Psi_MAP.iter, length(est_labels));
fprintf('%s LP: %d and Purity: %1.2f, NMI Score: %1.2f, F measure: %1.2f \n', clust_type, Psi_MAP.LogProb, Purity, NMI, F);


% %%%%%%%%% Plot sd-CRP-MM Results on Manifold Data % %%%%%%%%%

% Extract cluster parameters
Mu = Psi_MAP.Cluster_Mu;
Pr = Psi_MAP.Cluster_Pr;
Sigma = zeros(size(Pr,1),size(Pr,1),size(Pr,2));
for i=1:size(Pr,2)
    Sigma(:,:,i) = diag(Pr(:,i));
end        
% Visualize Cluster Parameters on Manifold Data
if exist('h3','var') && isvalid(h3), delete(h3);end
h3 = plotClusterParameters( Y, est_labels, Mu, Sigma );


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

