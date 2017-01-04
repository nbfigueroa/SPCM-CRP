function [ labels_sdcrp ] = clust_spcmCRP( sigmas,  options )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Step 1: Compute Similarity Matrix from B-SPCM Function for dataset   %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% %%%%%%%%%%%%%%%%%%%%% Set Hyper-parameter %%%%%%%%%%%%%%%%%%%%%%%%
% Tolerance for SPCM decay function 
tau = 1; % [1, 100] Set higher for noisy data, Set 1 for ideal data 

% %%%%%% Compute Confusion Matrix of Similarities %%%%%%%%%%%%%%%%%%
spcm = ComputeSPCMfunctionMatrix(sigmas, tau);  
S = spcm(:,:,2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%        Step 2: Run Automatic Spectral Dimensionality Reduction        %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%% Automatic Discovery of Dimensionality on M Manifold %%%%%%%%%%%
M = [];
[Y, d, thres, V] = spectral_DimRed(S, M);
if isempty(M)
    s_norm = normalize_soft(softmax(d));
    M = sum(s_norm <= thres);
end

%%%%%%%% Visualize Spectral Manifold Representation for M=2 or M=3 %%%%%%%%
if plots
    if exist('h1','var') && isvalid(h1), delete(h1);end
    h1 = plotSpectralManifold(Y, true_labels, d,thres, s_norm, M);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             Step 3: Discover Clusters with sd-CRP-MM                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%% Non-parametric Clustering on Manifold Data with Sim prior %%%%%%

% Chosen hyper parameters (Default values)
options       = [];
hyper.alpha     = 1;     % Concentration parameter
hyper.mu0       = 0;      % hyper for N(mu_k|mu_0,kappa_0)
hyper.kappa0    = 1;      % hyper for N(mu_k|mu_0,kappa_0)
hyper.a0        = M;      % hyper for IW(Sigma_k|Lambda_0,nu_0): (degrees of freedom)
hyper.b0        = M*0.5;  % hyper for IW(Sigma_k|Lambda_0,nu_0): (Scale matrix)
options.hyper = hyper;    % Setting hyper-parameters
options.niter = 100;     % Sampler Iterations 

% Run Gibb Sampler
[Psi Psi_Stats] = run_ddCRP_sampler(Y, S, options);


end

