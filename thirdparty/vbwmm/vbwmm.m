function [expectations, other,priors] = vbwmm( C, nu , K, varargin)
%% Variational Bayes Wishart Mixture Model
%
% INPUT:
%       C    - array of size p x p x L
%       nu   - array of size 1 x L (degrees of freedom for each window)
%       K    - number of clusters
%
% OPTIONAL INPUTS:
%       maxiter - number of iterations (default: 100)
%       rel_tol - relative tolerance in lower bound (stopping criteria)
%                   (default: 1e-9)
%       force_iter - force algorithm to run a number of iterations before
%                    stopping
%       verbose - on/off (off still yields a few lines of text)
%       init_method - method of initialization
%                       'kmeans' (default) - clustering is initialized to
%                       the kmeans solution
%                       'random' - clustering is initialized randomly (i.e.
%                       each window is assigned to one of the K clusters)
%                       'uniform' - expectation of z (clustering) is
%                       initialized to 1/K. 
%       init_z - initialize z to a specific clustering of your choice
%           (default is empty which results in standard initializtion as
%           specified by "init_method")
%
%       runGPU - bool that determines if GPU should be used 
%                (default is false - NOT THOROUGHLY TESTED)
%       delay_hyper - number of iterations that should run before updating
%                   hyperparameters (in this case only eta)
%       update_z - how should z (clustering) be updated
%                       'expect' (default) - expectation step
%                       'max' - maximization step
%                       'stochastic_search' - sampling according to current
%                       "posterior" (mimicking Gibbs sampling) (NOT TESTED
%                       THOROUGHLY)
%       
%       If you use this code for academic purposes please cite my paper
%
%       Nielsen, S. F. V., Madsen, K. H., Schmidt, M. N., & M�rup, M. (2017) 
%       "Modeling Dynamic Functional Connectivity 
%       using a Wishart Mixture Model" 
%       In 2017 International Workshop on 
%       Pattern Recognition in NeuroImaging (PRNI), IEEE.
%       
%
%   Written by: S�ren F�ns Vind Nielsen (sfvn at dtu dot dk)
%   June, 2017
%   See LICENSE file in repository (MIT License)

%% Initialization
% Global variables
global fix_pi fix_Sigma fix_Z max_Sigma fix_eta max_eta symmetric_tol update_z

% Get optional arguments and set defaults
opts = mgetopt(varargin);
maxiter = mgetopt(opts,'maxiter',100);
rel_tol = mgetopt(opts, 'rel_tol', 1e-9 );
symmetric_tol = mgetopt(opts, 'symmetric_tol', 1e-12);
force_iter = mgetopt(opts,'force_iter',0);
verbose = mgetopt(opts,'verbose', 'full');
init_method = mgetopt(opts, 'init_method', 'random' );
init_z = mgetopt(opts,'init_z', []);
runGPU = mgetopt(opts,'run_gpu', false);
delay_hyper = mgetopt(opts,'delay_hyper', 0);
update_z = mgetopt(opts,'update_z', 'expect'); % possible values ['expect', 'max', 'stochastic_search' ]

% Port data to GPU if needed
if runGPU
    C = gpuArray(C);
end

if size(nu,2)~=1
   nu = nu'; 
end

[p,~,L] = size(C);
scale = mean(C(:));

if ~strcmp(verbose,'off')
    fprintf('%70s\n', '-------------------------------------------------------'  )
    fprintf('%70s\n', ' Variational Bayes Wishart Mixture Model ')
    fprintf('%70s\n', sprintf(' Running a %i state model - Number of windows: %d', K, L) )
    fprintf('%70s\n', '-------------------------------------------------------'  )
end

% priors
eta_a0 = mgetopt(opts, 'eta_a0', 2 );
eta_b0 = mgetopt(opts, 'eta_b0', 1e6 );
alpha0 = mgetopt(opts, 'alpha0', 1 ); alpha = alpha0*ones(1,K, 'like', C);
nu0 = mgetopt(opts, 'nu0', p );

% create initialization struct
eta_inv_init = mgetopt(opts,'eta_inv',[]);
% TODO: add options to hard initialize other variables
init.eta_inv = eta_inv_init;
init.method = init_method;
init.z = init_z;


% debuggin purposes
fix_Sigma = mgetopt(opts, 'fix_Sigma', false);
fix_Z = mgetopt(opts, 'fix_Z', false);
fix_pi = mgetopt(opts, 'fix_pi', false);
fix_eta = mgetopt(opts, 'fix_eta', false);
max_Sigma = mgetopt(opts, 'max_Sigma', false);
max_eta = mgetopt(opts, 'max_eta', false);


%%% Prior struct
priors.p = p;
priors.K = K;
priors.nu0 = nu0;
priors.alpha = alpha;
priors.nu = nu;
priors.a0 = eta_a0;
priors.b0 = eta_b0;
priors.scale = scale;

%%% Initialize expectations
if strcmp(verbose,'full')
    disp('Initialization...')
    tic_init = tic;
end
[expectations,entropy] = initializeExpectations(C,K,priors,init);
[~,other.z_init] = max(expectations.Z,[],2);

if strcmp(verbose,'full')
    toc(tic_init)
end

% Constant term in lower bound
priors.lnDetC = nan(L,1);
no_errors = true; % flag if determinant contribution should be incorporated
for l = 1:L
    try
        priors.lnDetC(l) = 2*sum(log(diag(chol(C(:,:,l)))));
    catch ME
        if strcmp(ME.identifier,'MATLAB:posdef')
            priors.lnDetC(l) = nan;
             if any(strcmp(verbose,{'full','minimal'}))
                 warning('Data contains non-positive definite scatter matrices - terms ignored in lower bound')
             end
             no_errors = false;
            break
        end
    end
end

if no_errors
    priors.const_z = (nu-p-1)/2.*priors.lnDetC - nu*p/2*log(2) - mvgammaln(p,nu/2);
    priors.lb_const_term = K*(-priors.nu0*p/2*log(2) - mvgammaln(p,priors.nu0/2) ) ...
    - gammaln(sum(priors.alpha)) + sum(gammaln(priors.alpha));
else
    priors.lb_const_term = 0;
    priors.const_z = 0;
end


% other results
other.lower_bound = nan(1,maxiter, 'like',C);

other.lower_bound(1) = calculateLowerBound(C,expectations,priors,entropy);

%% Main Loop

it = 1; rel_lb_diff = Inf;

if strcmp(verbose, 'full')
    fprintf('%70s\n', '-------------------------------------------------------'  )
    fprintf( '%15s | %15s | %15s | %15s |\n', 'Iterations' , 'Lowerbound (LB)' ,'Abs Diff in LB', 'Rel Diff in LB')
    fprintf('%70s\n', '-------------------------------------------------------'  )
end
start = tic;
while (it<maxiter && rel_lb_diff>rel_tol) || it < force_iter
    % Update Z
    if ~fix_Z
        logR =  -1/2*squeeze(sum(sum( ...
            bsxfun(@times, permute(expectations.SigmaInv,[1,2,4,3]), C  ),1),2) );
        logR = bsxfun(@plus, logR, priors.const_z) ;
        logR = bsxfun(@plus,logR, expectations.lnPi)+ nu/2*expectations.lnDetSigmaInv;
        logR = bsxfun(@minus,logR,lsum(logR,2));
        switch update_z
            case 'expect'
                expectations.Z = exp(logR);
                expectations.Z(expectations.Z < 1e-16)  = 0;
                % Calc entropy
                ent_Z = -sum(sum(expectations.Z.*logR));
            case 'max'
                [~,z] = max(logR,[],2);
                expectations.Z = createAssignmentMatrix(z,K)';
                ent_Z = 0;
            case 'stochastic_search'
                % do stuff
                PZ = exp(logR);
                % Sample hard assignments
                [~,idx] = sort(rand(L,1)<cumsum(PZ,2),2,'descend');
                expectations.Z = createAssignmentMatrix(idx(:,1),K)'; %find(rand(L,1)<cumsum(PZ,2),1,'first');
                ent_Z = 0;
        end
    else
        ent_Z = 0;
    end
    
    % Update pi
    ak = sum(expectations.Z,1) + priors.alpha;
    expectations.Pi = ak/sum(ak);
    expectations.lnPi = psi(0,ak) - psi(0,sum(ak));
    % Calc entropy
    ent_pi = dirichlet_entropy(ak);
    
    % Update Sigma
    if ~fix_Sigma
        if ~max_Sigma
            [expectations,ent_S] = expect_Sigma(C,expectations,priors);
        else
            [expectations,ent_S] = maximize_Sigma(C,expectations,priors);
        end
    else
        ent_S = 0;
    end
    
    % Update eta
    if ~fix_eta && (it>delay_hyper)
        if ~max_eta
            [expectations,ent_eta] = expect_eta(expectations,priors);
        else
            [expectations,ent_eta] = maximize_eta(expectations,priors);
        end
    else
        ent_eta = 0;
    end
    
    % calculate lower bound
    entropy = ent_S + ent_Z + ent_pi + ent_eta;
    other.lower_bound(it+1) = calculateLowerBound(C,  expectations , priors, entropy);
    
    % check difference
    lb_diff = other.lower_bound(it+1)-other.lower_bound(it);
    rel_lb_diff = abs(lb_diff)/abs(other.lower_bound(it));
    
    % print stuff
    if mod(it,50)==0 && strcmp(verbose,'full')
        fprintf('%70s\n', '-------------------------------------------------------'  )
        fprintf( '%15s | %15s | %15s | %15s |\n', 'Iterations' , 'Lowerbound (LB)' ,'Abs Diff in LB', 'Rel Diff in LB')
        fprintf('%70s\n', '-------------------------------------------------------'  )
    end
    
    if strcmp(verbose,'full')
        fprintf( '%15i | %15.4d | %15.4d | %15.4d | \n', it , other.lower_bound(it+1) ,lb_diff, rel_lb_diff)
    end
    
    if lb_diff<0  &&  rel_lb_diff>rel_tol && force_iter==0 && ~strcmp(update_z,'stochastic_search')
        error('MyErrors:lbdiv','Lower Bound Diverging')
    elseif lb_diff<0 && force_iter>0
        if strcmp(verbose,{'full', 'minimal'})
            warning('MyErrors:lbdiv','Lower Bound Diverging')
        end
    end
    it = it + 1;
end

other.lower_bound(isnan(other.lower_bound)) = [];
% TODO: Return extra nice stuff
% Expectations of covariance (i.e. inverse of what we have been doing)
% Print final message (time,...)

if any(strcmp(verbose, {'full','minimal'}))
    toc(start)
end
%eof
end




%% Subfunctions
function lb = calculateLowerBound( C,  expectations , priors, entropy)
global max_eta
p = size(C,1);
K = priors.K;
lb = 0;

% Contribution from data and Sigma
for k = 1:K
    lb = lb + expectations.Z(:,k)'*( priors.const_z ... % log-likelihood
        + priors.nu/2*expectations.lnDetSigmaInv(k) ...
        - 1/2*squeeze(sum(sum(bsxfun(@times, expectations.SigmaInv(:,:,k), C) ,2),1)) )  ...
        -1/2*sum(sum( (expectations.EtaInv*eye(p)) .* expectations.SigmaInv(:,:,k))) ... % prior on Sigma
        + (priors.nu0 - p - 1)/2*expectations.lnDetSigmaInv(k) ...
        - priors.nu0*p/2*expectations.lnEta;
end

% Contribtuion from eta
lb = lb + ~max_eta*(-gammaln(priors.a0) + priors.a0*log(priors.b0)...
    -(priors.a0+1)*expectations.lnEta -priors.b0*expectations.EtaInv);


% Contribution from Z
lb = lb + sum(sum(bsxfun(@times,expectations.Z, expectations.lnPi)));

% Contribution from pi
lb = lb + dot(priors.alpha-1,expectations.lnPi);

lb = lb + entropy + priors.lb_const_term;
%eof
end

% -------------------------------------------------------------------------
function [expectations,entropy] = expect_Sigma(C,expectations, priors)
K=priors.K;
p = priors.p;
vk = nan(1,K, 'like',C);
for k = 1:K
    vk(k) = priors.nu0 + dot(expectations.Z(:,k),priors.nu');
    expectations.SigmaInv(:,:,k) = vk(k)*inv( expectations.EtaInv*eye(p) + ...
        squeeze(sum(bsxfun(@times,permute(C,[3,1,2]),expectations.Z(:,k)) ,1)) );    
    %expectations.SigmaInv(:,:,k) = vk(k)*my_inv(...
    %    squeeze(sum(bsxfun(@times,permute(C,[3,1,2]),expectations.Z(:,k)) ,1))...
    %    ,expectations.EtaInv);    
end
[expectations,flag] = symmetrizeSigma(expectations,C);
% Calc entropy
[entropy,lnDetS_Inv] = wishart_entropy( permute(bsxfun(@times,permute(expectations.SigmaInv,[3,1,2])...
    ,1./vk'),[2,3,1]),vk);
expectations.lnDetSigmaInv = lnDetS_Inv;
expectations.pars.vk = vk;
end

% -------------------------------------------------------------------------
function [expectations,entropy] = maximize_Sigma(C,expectations, priors)
K=priors.K;
p = priors.p;
vk = nan(1,K);
for k = 1:K
    vk(k) = priors.nu0 + dot(expectations.Z(:,k),priors.nu');
    expectations.SigmaInv(:,:,k) = (vk(k)- p -1 )*eye(p)/( expectations.EtaInv*eye(p) + ...
        squeeze(sum(bsxfun(@times,permute(C,[3,1,2]),expectations.Z(:,k)) ,1)) );
    expectations.lnDetSigmaInv(k) = 2*sum(log(diag(chol( expectations.SigmaInv(:,:,k)  ))));
end
entropy=0;
end

% -------------------------------------------------------------------------
function [expectations,entropy] = expect_eta(expectations, priors)
a = priors.a0 + priors.nu0*priors.p*priors.K/2;
b = priors.b0;
for k = 1:priors.K
    b = b + 1/2*sum(diag(expectations.SigmaInv(:,:,k)));
end

% expectations
expectations.Eta = b/(a-1);
expectations.EtaInv = a/b;
expectations.lnEta = log(b) - psi(0,a);

entropy = inversegamma_entropy(a,b);
end

% -------------------------------------------------------------------------
function [expectations,entropy] = maximize_eta(expectations, priors)
a = priors.nu0*priors.p*priors.K/2;
b = 0;
for k = 1:priors.K
    b = b + 1/2*sum(diag(expectations.SigmaInv(:,:,k)));
end

% expectations
expectations.Eta = b/a;
expectations.EtaInv = inv(expectations.Eta);
expectations.lnEta = log(expectations.Eta);
entropy = 0;
end

% -------------------------------------------------------------------------
function [expectations,entropy] = initializeExpectations(C,K,priors,init)
global fix_pi fix_Sigma fix_eta fix_Z max_Sigma update_z

[p,~,L] = size(C);
if isempty(init.z)
    switch init.method
        case 'random' % Random sampling of cluster centres in data
            % Initialize Z randomly
            rand_assign = randsample(1:K,L, true);
            while true
                if length(unique(rand_assign))~=K
                    rand_assign = randsample(1:K,L, true);
                else
                    break
                end
            end
            [expectations,ent_Z] = initialize_Z(rand_assign,C);
            
            % initialize eta from prior
            [expectations,ent_eta] = initialize_eta(expectations,priors.a0,priors.b0, init.eta_inv);
            
            
            % perform standard expectation on Sigma and Pi
            [expectations,ent_S] = expect_Sigma(C,expectations,priors);
            
            ak = sum(expectations.Z,1) + priors.alpha;
            expectations.Pi = ak/sum(ak);
            expectations.lnPi = psi(0,ak) - psi(0,sum(ak));
            
            % Entropy
            entropy = ~fix_pi*dirichlet_entropy(ak) ...
                + ~fix_Sigma*ent_S-(~fix_Z)*ent_Z...
                + ~fix_eta*ent_eta;
            
        case 'kmeans'
            % extract upper triangular part
            Y = nan(p*(p-1)/2 + p , L, 'like', C);
%             for l = 1:L
%                 tmp = triu(C(:,:,l));
%                 Y(:,l) = tmp(abs(tmp)>0);
%             end
            for l = 1:L
                Y(:,l) = symMat2Vec(C(:,:,l));
            end
            
            % run kmeans
            states = kmeans(Y', K,'Replicates',10);
            
            % Initalize Z as the state sequence suggests
            [expectations,ent_Z] = initialize_Z(states,C);
                        
            % Initialize pi
            ak = sum(expectations.Z,1) + priors.alpha;
            expectations.Pi = ak/sum(ak);
            expectations.lnPi = psi(0,ak) - psi(0,sum(ak));
            % Calc entropy
            ent_pi = ~fix_pi*dirichlet_entropy(ak);
            
            % Initialize eta
            [expectations,ent_eta] = initialize_eta(expectations,priors.a0,priors.b0, init.eta_inv);
            
            % Initialize Sigma
            if ~max_Sigma
                [expectations,ent_S] = expect_Sigma(C,expectations, priors);
            else
                [expectations,ent_S] = maximize_Sigma(C,expectations, priors);
            end
            
            entropy = ent_Z + ent_pi + ~fix_Sigma*ent_S + ~fix_eta*ent_eta;
            
        case 'uniform'
           % Initialize Z uniformly
           expectations.Z = ones(L,K, 'like', C).*1/K;
           switch update_z
               case {'max','stochastic_search'}
                   ent_Z = 0;
               case 'expect'
                   ent_Z = ~fix_Z*-sum(sum(expectations.Z.*log(expectations.Z) ));
           end
            % initialize eta from prior
            [expectations,ent_eta] = initialize_eta(expectations,priors.a0,priors.b0,init.eta_inv);
            
            
            % perform standard expectation on Sigma and Pi
            [expectations,ent_S] = expect_Sigma(C,expectations,priors);
            
            ak = sum(expectations.Z,1) + priors.alpha;
            expectations.Pi = ak/sum(ak);
            expectations.lnPi = psi(0,ak) - psi(0,sum(ak));
            
            % Entropy
            entropy = ~fix_pi*dirichlet_entropy(ak) ...
                + ~fix_Sigma*ent_S-(~fix_Z)*ent_Z...
                + ~fix_eta*ent_eta; 
    end
    
else
    % Initalize Z as the give state sequence
    z_reg = 0.001;
    [expectations,ent_Z] = initialize_Z(init.z,C);
    
    % Initialize pi
    ak = sum(expectations.Z,1) + priors.alpha;
    expectations.Pi = ak/sum(ak);
    expectations.lnPi = psi(0,ak) - psi(0,sum(ak));
    % Calc entropy
    ent_pi = ~fix_pi*dirichlet_entropy(ak);
    
    % Initialize eta
    [expectations,ent_eta] = initialize_eta(expectations,priors.a0,priors.b0,init.eta_inv);
    
    % Initialize Sigma
    [expectations,ent_S] = expect_Sigma(C,expectations, priors);
    
    entropy = ent_Z + ent_pi + ~fix_Sigma*ent_S + ~fix_eta*ent_eta;
end




end

% -------------------------------------------------------------------------
function [expectations,ent_Z] = initialize_Z(z,C)
global update_z fix_Z
z_reg = 0.001;
K = max( unique(z) );
L = size(C,3);
expectations.Z = nan(L,K, 'like', C);
expectations.Z = createAssignmentMatrix(z)';
if K>1 % regularize assignment matrix
    expectations.Z(expectations.Z==1) = 1-z_reg;
    expectations.Z(expectations.Z==0) = z_reg/(K-1);
end
switch update_z
    case {'max','stochastic_search'}     
        ent_Z = 0;
    case 'expect'
        ent_Z = ~fix_Z*-sum(sum(expectations.Z.*log(expectations.Z) ));
end
%eof
end



% -------------------------------------------------------------------------
function [expectations,entropy] = initialize_eta(expectations,a,b,init_eta_inv)
global max_eta fix_eta
if isempty(init_eta_inv) % if eta is uninitialized
    if ~max_eta
        expectations.Eta = b/(a-1);
        expectations.EtaInv = a/b;
        expectations.lnEta = log(b) - psi(0,a);
    else
        expectations.Eta = b/a;
        expectations.EtaInv = inv(expectations.Eta);
        expectations.lnEta = log(expectations.Eta);
    end
else
    expectations.EtaInv = init_eta_inv;
    expectations.Eta = inv(expectations.EtaInv);
    expectations.lnEta = log(expectations.Eta);
end
entropy = ~(max_eta | fix_eta)*inversegamma_entropy(a,b);
end
% -------------------------------------------------------------------------
function entropy = dirichlet_entropy(alph)
% Bishop:sum(bsxfun(@times,permute(C,[3,1,2]),expectations.Z(:,k)) ,1)
%logNormalizer = gammaln(sum(alph))-sum(gammaln(alph));
%entropy = -dot(alph-1,psi(0,alph)-psi(0,sum(alph)))-logNormalizer;

% Wikipedia:
logCalpha = gammaln(sum(alph))-sum(gammaln(alph));
entropy = -dot(alph-1,psi(0,alph))-(length(alph)-sum(alph))*psi(0,sum(alph))-logCalpha;
end

% -------------------------------------------------------------------------
function [entropy, expectation_lnDetX] = wishart_entropy(Omega,v)
% Naive implementation
entropy = 0;
expectation_lnDetX = nan(1,size(Omega,3), 'like', Omega);
p = size(Omega,1);
for k = 1:size(Omega,3)
    logdet = 2*sum(log(diag(chol(Omega(:,:,k)))));
    expectation_lnDetX(k) = sum(psi(0,(v(k)+1-(1:p))/2)) + p*log(2) + logdet;
    entropy = entropy + v(k)/2*logdet  + v(k)*p/2*log(2)...
        + mvgammaln(p,v(k)/2) -(v(k)-p-1 )/2*expectation_lnDetX(k) ...
        + v(k)*p/2;
end
%eof
end


% -------------------------------------------------------------------------
function entropy = inversegamma_entropy(a,b)
% Used definition from Wikipedia
entropy = a + log(b) + gammaln(a) - (1+a)*psi(0,a);
end


% -------------------------------------------------------------------------
function [expectations, exitflag] =  symmetrizeSigma(expectations,C)
global symmetric_tol
exitflag = 0;
symdiff = expectations.SigmaInv - permute(expectations.SigmaInv,[2,1,3]);
for k = 1:size(symdiff,3)
   assert( (norm(symdiff(:,:,k),'fro')^2 ...
   /norm(expectations.SigmaInv(:,:,k),'fro')^2)<symmetric_tol    );
end
expectations.SigmaInv = (expectations.SigmaInv + ...
    permute(expectations.SigmaInv,[2,1,3]))/2;
end

% -------------------------------------------------------------------------
function out = mgetopt(varargin)
% MGETOPT Parser for optional arguments
%
% Usage
%   Get a parameter structure from 'varargin'
%     opts = mgetopt(varargin);
%
%   Get and parse a parameter:
%     var = mgetopt(opts, varname, default);
%        opts:    parameter structure
%        varname: name of variable
%        default: default value if variable is not set
%
%     var = mgetopt(opts, varname, default, command, argument);
%        command, argument:
%          String in set:
%          'instrset', {'str1', 'str2', ... }
%
% Example
%    function y = myfun(x, varargin)
%    ...
%    opts = mgetopt(varargin);
%    parm1 = mgetopt(opts, 'parm1', 0)
%    ...

% Copyright 2007 Mikkel N. Schmidt, ms@it.dk, www.mikkelschmidt.dk

if nargin==1
    if isempty(varargin{1})
        out = struct;
    elseif isstruct(varargin{1})
        out = varargin{1}{:};
    elseif isstruct(varargin{1}{1})
        out = varargin{1}{1};
    else
        out = cell2struct(varargin{1}(2:2:end),varargin{1}(1:2:end),2);
    end
elseif nargin>=3
    opts = varargin{1};
    varname = varargin{2};
    default = varargin{3};
    validation = varargin(4:end);
    if isfield(opts, varname)
        out = opts.(varname);
    else
        out = default;
    end
    
    for narg = 1:2:length(validation)
        cmd = validation{narg};
        arg = validation{narg+1};
        switch cmd
            case 'instrset',
                if ~any(strcmp(arg, out))
                    fprintf(['Wrong argument %s = ''%s'' - ', ...
                        'Using default : %s = ''%s''\n'], ...
                        varname, out, varname, default);
                    out = default;
                end
            case 'dim'
                if ~all(size(out)==arg)
                    fprintf(['Wrong argument dimension: %s - ', ...
                        'Using default.\n'], ...
                        varname);
                    out = default;
                end
            otherwise,
                error('Wrong option: %s.', cmd);
        end
    end
end
%eof
end