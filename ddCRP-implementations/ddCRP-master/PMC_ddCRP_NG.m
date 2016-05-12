function [MAP, samples] = PMC_ddCRP_NG(X,A,opts)

%PMC_ddCRP_NG samples from posterior cluster distribution
%
%   MAP = PMC_ddCRP_NG(X,A) for data matrix X and neigbourhood list A
%   returns 100 samples using default hyperparameters. The first column of
%   the cell array samples contains the customer assignments for each of
%   the samples. The second column contains the corresponding cluster
%   assignments for each sample. The data matrix is clustered along the
%   second dimension and the neighbourhood list A is a cell array of length
%   N. Any empty cells in A are regarded as elements to be ignored, be 
%   careful not to include them in the neighbourhood definition of nodes
%   that should be clustered.
%
%   [MAP, samples] = PMC_ddCRP_NG(X,A,opts) returns the samples with settings as
%   specified in the structure variable opts. This variable takes the
%   following fields (fields not set by the user are set to the default
%   values noted in parentheses):
%     steps     = (100) the number of samples to return.
%     thinning  = (1) if thinning=N, keep only the Nth sample. Note that 
%                 this does not affect the number of samples that are 
%                 returned and discarded samples are all sampled at the
%                 same temperature (see below).
%     cLinks    = (1:N; i.e. each of N elements is assigned to themselves.)
%                 N-dimensional vector of starting customer assignments.
%     Pi        = (derived from cLinks) The starting partition represented
%                 as a vector where Pi(n) indictates to which cluster the
%                 nth element belongs.
%     weights   = (all weights set to 1) a cell array containing weights
%                 representing the closeness of element pairs. The array
%                 has the same dimensions as A and each cell contains the
%                 closeness of that element to the corresponding element as
%                 indexed in A.
%     temp      = (1) temperature at which to sample, i.e. probabilities
%                 are raised to this power. When this is a vector it is a
%                 temperature schedule and the length should match the
%                 number of samples requested
%     hyp       = hyperparameter settings, this is a structure variable
%                 that takes the following fields:
%       alpha   = (1) ddCRP concentration parameter.
%       mu0     = (0) mean of the normal-gamma prior. Only change this
%                 setting if your data matrix has not been Z-transformed
%                 along the first dimension (which should be the case).
%       kappa0  = (1) related to the precision of normal portion of the
%                 normal-gamma prior. Higher values increases shrinkage
%                 towards mu0.
%       a0      = (2) [] parameter of the gamma portion of the normal-gamma
%                 distribution. Together with b0 this encodes the prior
%                 on the within cluster precision.
%       b0      = (1) [] parameter of the gamma portion of the normal-gamma
%                 distribution.
%
%	If you use this code in your research, please cite the following paper:
% 
%	Janssen, R. J., Jyl�nki, P., Kessels, R. P. C., & van Gerven, M. A. J. 
% (2015). Probabilistic model-based functional parcellation reveals a 
% robust, fine-grained subdivision of the striatum. NeuroImage, 119, 
% 398–405. http://doi.org/10.1016/j.neuroimage.2015.06.084

%

[T, N] = size(X)
weights = cellfun(@ones,num2cell(cellfun(@length,A)),num2cell(ones(size(A))),'UniformOutput',false);
MAP.LP = -inf;

for i = 1:length(A)
  A{i} = A{i}(:)';
  weights{i} = weights{i}(:)';
end

A
weights

steps = 100;
thinning = 1;
cLinks = 1:N;
temp = 1;
exclude = false(size(A));
for n = 1:N
  exclude(n) = isempty(A{n});
end
exclude = find(exclude);

if nargin > 2
  fld = fieldnames(opts);
  for i = 1:length(fld)
    if strcmp('hyp',fld{i})
      hypfld = fieldnames(opts.hyp);
      for j = 1:length(hypfld)
        eval(sprintf('hyp.%s = opts.hyp.%s;',hypfld{j},hypfld{j}))
      end
    else
      eval(sprintf('%s = opts.%s;',fld{i},fld{i}))
    end
  end
end

opt.weights = weights;
cLinks(exclude) = exclude;

if exist('Pi','var')
  if ~isempty(exclude)&&any(Pi(exclude)>0)
    Pi = link2Pi(cLinks);
    list = unique(Pi(exclude));
    Pi(exclude) = 0;
    for i = 0:(length(list)-1)
        Pi(Pi>list(end-i)) = Pi(Pi>list(end-i))-1;
    end
  else
    K = max(Pi);
    ClustMembers = cell(1,K);
    for k = 1:K
      ClustMembers{k} = find(Pi==k)';
    end
  end
else
  ClustMembers = cell(N,1);
  Pi=link2Pi(cLinks);
  if ~isempty(exclude)
    Pi(exclude) = 0;
    list = unique(Pi(exclude));
    for i = 0:(length(list)-1)
      Pi(Pi>list(end-i)) = Pi(Pi>list(end-i))-1;
    end
  end
  K = max(Pi);
  for k = 1:K
    ClustMembers{k} = find(Pi==k);
  end
end

opt.steps = thinning;
opt.Pi = Pi;
opt.ClustMembers = ClustMembers;

if numel(temp)==1
  temp = temp(ones(steps,1));
elseif numel(temp)<steps
  warning('Number of temperature steps do not match sampling steps.\nMissing temperatures are set to last temperature.')
  temp(numel(temp):steps) = temp(end);
elseif numel(temp)>steps
  warning('More temperature steps provided than samples requested.\nFinal temperature steps will be disregarded')
end
K = max(Pi);
if nargout>1
  samples = cell(steps,2);
end

%%%% Simplest Initialization %%%%
% hyperparameters
hyp.alpha = 1;
hyp.mu0 = 0;
hyp.kappa0 = 1;
hyp.a0 = 2;
hyp.b0 = 1;

% Initialization
[M, N] = size(X)
cLinks = 1:N;
steps = 100;
thinning = 1;
temp = 1;
exclude = false(size(A));
for n = 1:N
  exclude(n) = isempty(A{n});
end
exclude = find(exclude);


ClustMembers = cell(N,1);
Pi=link2Pi(cLinks);
if ~isempty(exclude)
Pi(exclude) = 0;
list = unique(Pi(exclude));
for i = 0:(length(list)-1)
  Pi(Pi>list(end-i)) = Pi(Pi>list(end-i))-1;
end
end
K = max(Pi);
for k = 1:K
ClustMembers{k} = find(Pi==k);
end
fprintf('Initialised with %d clusters\n',K)

% options
opt.steps = thinning;
opt.Pi = Pi;
opt.ClustMembers = ClustMembers;


for s = 1:steps
  fprintf('Generating sample %d',s)
  opt.temp = temp(ones(thinning,1)*s);
  [cLinks, opt.Pi, opt.ClustMembers, opt.allLLs] = ddCRP_NGgensamp(X, cLinks, A, hyp, opt);
  if nargout>1
    samples{s,1} = cLinks;
    samples{s,2} = opt.Pi;
  end
  LP = LPddCRP_NG(X,A,cLinks,opt.Pi,hyp,weights);
  if LP>MAP.LP
    MAP.LP = LP;
    MAP.Pi = opt.Pi;
  end
  fprintf('Sample contains %d clusters; logprob = %4.2f\n',max(opt.Pi),LP)
end

% Reconstruct cluster timecourses for the MAP parcellation
K = max(MAP.Pi);
Z = bsxfun(@eq,MAP.Pi,1:K);
Nks = sum(Z);
XbarN = X*Z; % Xbar*N
Xbar = bsxfun(@rdivide,XbarN,Nks);
mu_n = bsxfun(@rdivide,hyp.kappa0.*hyp.mu0+ XbarN,hyp.kappa0+Nks);
kappa_n = Nks+hyp.kappa0;
a_n = hyp.a0+Nks./2;
b_n = hyp.b0 + 0.5 * ((X-XbarN(:,MAP.Pi)).^2)*Z + bsxfun(@rdivide,hyp.kappa0.*bsxfun(@times,Nks,(Xbar-hyp.mu0).^2),2.*(hyp.kappa0+Nks));

s2 = bsxfun(@rdivide,b_n,(a_n.*kappa_n));
t = tinv(0.975,2*a_n);

MAP.ClusterTCs = mu_n;
MAP.ClusterCI = bsxfun(@times,t./(2.*a_n+1),sqrt(s2));



