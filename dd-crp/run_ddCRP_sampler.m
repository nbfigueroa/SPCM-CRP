function [Psi, Psi_Stats] = run_ddCRP_sampler(Y,S, options)
% Distance Depenendent Chinese Restaurant Process Mixture Model.
% **Inputs**
%          Y: projected M-dimensional points  Y (y1,...,yN) where N = dim(S),
%          S: Similarity Matrix where s_ij=1 is full similarity and
%          s_ij=0 no similarity between observations
%          
% **Outputs**
%          Psi (MAP Markov Chain State)
%          Psi.LogProb:
%          Psi.Z_C:
%          Psi.clust_params:
%          Psi.iter:
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
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           Parse Sampler Options and Set Aurxiliary Variables            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Data Dimensionality
% (M = reduced spectral space, N = # of observations)
[M, N] = size(Y);

%%%% Default Hyperparameters %%%%
alpha            = 1;
lambda.mu0       = 0;
lambda.kappa0    = 1;
lambda.a0        = M;   
lambda.b0        = M*0.5;

%%%% Default Covariance Matrix Type %%%
type = 'diag';

%%%% Parse Sampler Options %%%%
if nargin < 3
    T = 100; % number of iterations
else
    T = options.T;
    
    if isfield(options, 'lambda')
        alpha            = options.alpha;
        lambda.mu0       = options.lambda.mu0;
        lambda.kappa0    = options.lambda.kappa0;
        lambda.alpha0    = options.lambda.alpha0;
        lambda.beta0     = options.lambda.beta0;
    end
        
    if isfield(options, 'type')
        type = options.type;
    end
end

%%% Initialize Stats Variabes  %%%
Psi_Stats.JointLogProbs = zeros(1,T);
Psi_Stats.TotalClust    = zeros(1,T);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%              Define Initial Markov Chain State Psi^{t-1}               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Augment Similarity Matrix with alpha on diagonal %%%
S = S + eye(N)*(alpha-1);
S_alpha = num2cell(S,2);

%%% Compute Initial Cluster Assignments and Likelihoods %%%
C = 1:N;
clust_members = cell(N,1);
Z_C   = extract_TableIds(C); %% CHANGE THIS FUNCTION ----->
K = max(Z_C);
for k = 1:K
    clust_members{k} = find(Z_C==k);    
    clust_logLiks(k) = table_logLik(Y(:,Z_C==k), lambda, type);
end

fprintf('*** Initialized with %d clusters out of %d observations ***\n', K, N)

%%% Load initial variables  %%%
Psi.C              = C;
Psi.Z_C            = Z_C;
Psi.lambda         = lambda;
Psi.alpha          = alpha;
Psi.type           = type;
Psi.clust_members  = clust_members;
Psi.clust_logLiks  = clust_logLiks;
Psi.LogProb        = -inf;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                   Run Gibbs Sampler for dd-CRP                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Running dd-CRP Mixture Sampler... \n');
tic;

for i = 1:T
    fprintf('Iteration %d: Started with %d clusters ', i, max(Psi.Z_C));
    
    %%% Draw Sample sd(SPCM)-CRP %%%
    [Psi.C, Psi.Z_C, Psi.clust_members, Psi.clust_logLiks] = sample_ddCRPMM(Y, S_alpha, Psi);    
    
    %%% Update the LogProbability with the Priors %%%
    LogProb = logPr_sdCRPMM(Y, S_alpha, Psi); %% CHANGE THIS FUNCTION ----->    
    fprintf('--> moved to %d clusters with logprob = %4.2f\n', max(Psi.Z_C) , LogProb);
    
    %%% Store Stats %%%
    Psi_Stats.JointLogProbs(i) = LogProb;
    Psi_Stats.TotalClust(i)    = max(Psi.Z_C);
    
    %%% If current posterior is higher than previous update MAP estimate %%%
    if (LogProb > Psi.LogProb && max(Psi.Z_C) > 1)
        Psi.LogProb = LogProb;
        Psi.Z_C = Psi.Z_C;
        Psi.iter = i;
    end    
end

%%% Re-sample table parameters %%%
% Eq. 39 
[Psi.Theta] = resample_TableParams(Y, Psi.Z_C, lambda, type);

toc;
fprintf('*************************************************************\n');

end
