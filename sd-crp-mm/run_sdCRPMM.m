function [Psi_MAP] = run_sdCRPMM(Y,S)
% Similarity Depenendent Chinese Restaurant Process Mixture Model.
% Implementation of Algorithm 2. from Socher11 paper (Spectral Chinese Restaurant Processes: Clustering Based on Similarities)
% **Inputs**
%          Y: projected M-dimensional points  Y (y1,...,yN) where N = dim(S),
%          S: Similarity Matrix where s_ij=1 is full similarity and
%          s_ij=0 no similarity between observations
%          
% **Outputs**
%          Psi_MAP (MAP Markov Chain State)
%          Psi_MAP.LogProb:
%          Psi_MAP.Z_C:
%          Psi_MAP.clust_params:
%          Psi_MAP.iter:
%
% Author: Nadia Figueroa, PhD Student., Robotics
% Learning Algorithms and Systems Lab, EPFL (Switzerland)
% Email address: nadia.figueroafernandez@epfl.ch
% Website: http://lasa.epfl.ch
% February 2016; Last revision: 29-Feb-2016
%
% If you use this code in your research please cite:
% "Transform Invariant Discovery of Dynamical Primitives: Leveraging
% Spectral and Nonparametric Methods"; N. Figueroa and A. Billard; Pattern
% Recognition
% (2016)


%%% Setting up Initial Markov Chain State and Cluster Likelihoods %%%
%  Data Dimensionality
% (M = reduced spectral space, N = # of observations)
[M, N] = size(Y);

%%%% Default Sampler Options %%%%
niter = 20;

%%%% Default Hyperparameters %%%%
hyper.alpha     = 1;
hyper.mu0       = 0;
hyper.kappa0    = 1;
hyper.a0        = M;   
hyper.b0        = M*0.5;

%%% Define Priors %%%
delta = num2cell(S,2);

% Remove when changed everything to similarities ---- Change this fucker LPddCRP_NG
A = {1:(N)};A = A(ones(N,1));

%%% Compute Initial Cluster Assignments %%%
C = 1:N;
clust_members = cell(N,1);
Z_C=extract_TableIds(C); %% CHANGE THIS FUNCTION ----->
K = max(Z_C);
clust_LLs = zeros(size(Z_C));
for k = 1:K
    clust_members{k} = find(Z_C==k);    
    clust_logLiks(k) = table_logLik_NIG(Y(:,Z_C==k), hyper.a0, hyper.b0, hyper.mu0, hyper.kappa0);
%     clust_logLiks(k) = table_logLik_NIW(Y(:,Z_C==k), hyper.a0, hyper.b0, hyper.mu0, hyper.kappa0);%% CHANGE THIS FUNCTION -----> to NIW
    clust_params(k)  = hyper;
end
fprintf('*** Initialized with %d clusters out of %d observations ***\n', K, N)

%%% Load initial variables  %%%
Psi.LogProb        = -inf;
Psi.C              = C;
Psi.Z_C            = Z_C;
Psi.clust_members  = clust_members;
Psi.clust_params   = clust_params;
Psi.clust_logLiks  = clust_logLiks;
Psi_MAP.LogProb    = -inf;

%%% Run Gibbs Sampler for niter iterations %%%
for i = 1:niter
    fprintf('Iteration %d: Started with %d clusters ', i, max(Psi.Z_C));
    
    %%% Draw Sample sd(SPCM)-CRP %%%
    [Psi.C, Psi.Z_C, Psi.clust_members, Psi.clust_params, Psi.clust_logLiks] = sample_sdCRPMM(Y, delta, Psi);
    
    
    %%% Update the LogProbability with the Priors %%%
    Psi.LogProb = logPr_sdCRPMM(Y, delta, Psi); %% CHANGE THIS FUNCTION ----->
    
    %%% Re-sample table (cluster) parameters %%%
    % with Eq. 8 from spectral chinese restaurant
%     [Psi_MAP.Cluster_Mu, Psi_MAP.Cluster_Pr, Psi_MAP.clust_params] = resample_TableParams(Y, Psi_MAP.Z_C, Psi_MAP.clust_params);
    
    fprintf('--> moved to %d clusters with logprob = %4.2f\n', max(Psi.Z_C) , Psi.LogProb);
    
    %%% If current posterior is higher than previous update MAP estimate %%%
    if (Psi.LogProb > Psi_MAP.LogProb && max(Psi.Z_C) > 1)
        Psi_MAP.LogProb = Psi.LogProb;
        Psi_MAP.Z_C = Psi.Z_C;
        Psi_MAP.clust_params = Psi.clust_params;
        Psi_MAP.iter = i;
    end    
end

%%% Re-sample table parameters %%%
% with Eq. 8 from spectral chinese restaurant
[Psi_MAP.Cluster_Mu, Psi_MAP.Cluster_Pr, Psi_MAP.clust_params] = resample_TableParams(Y, Psi_MAP.Z_C, Psi_MAP.clust_params);

end

function [Mu, Pr, new_hypers] = resample_TableParams(Y, Z_C, hypers_n)

    % Updating hyper-parameters
    hypers = hypers_n(1);
    new_hypers = resample_Hypers(Y, Z_C, hypers);
    
    % New cluster means
    Mu = new_hypers.mu_n;
    
    % Computing new cluster precision matrices
    s2 = bsxfun(@rdivide,new_hypers.b_n,(new_hypers.a_n.*new_hypers.kappa_n));
    t = tinv(0.975, 2 * new_hypers.a_n);    
    Pr = bsxfun(@times,t./(2.*new_hypers.a_n+1),sqrt(s2));
    
end


function [new_hypers] = resample_Hypers(Y, Z_C, hypers)
    K = max(Z_C);
    Z = bsxfun(@eq,Z_C,1:K);
    
    Nks = sum(Z);
    YbarN = Y*Z; % Ybar*N
    Ybar = bsxfun(@rdivide,YbarN,Nks);
    
    % Updating Means
    new_hypers.mu_n = bsxfun(@rdivide,hypers.kappa0.*hypers.mu0 + YbarN, hypers.kappa0+Nks);
    
    % Computing hypers for Precision Matrix
    new_hypers.kappa_n = Nks + hypers.kappa0;
    new_hypers.a_n = hypers.a0 + Nks./2;
    new_hypers.b_n = hypers.b0 + 0.5 * ((Y-YbarN(:,Z_C)).^2)*Z + bsxfun(@rdivide,hypers.kappa0.* bsxfun(@times,Nks, (Ybar-hypers.mu0).^2),2.*(hypers.kappa0+Nks));      
       
end