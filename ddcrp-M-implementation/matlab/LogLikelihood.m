% Computes sum of log-likelihood terms for given sufficient statistics (in Nx3
%   matrix, with columns [count, mean, sum of squared dev]) and vectorized
%   hyperparameters
function logp = LogLikelihood(stats, hyp)

stats = stats(stats(:,1)>1,:);

% stats = [N | mu | sumsq]
% hyp = [mu0 kappa0 nu0 sigsq0 nu0*sigsq0 const_logp_terms]

kappa = hyp(2) + stats(:,1);
nu = hyp(3) + stats(:,1);
%nu_sigsq = hyp(5) + sumSqX + (n*hyp(2)) / (hyp(2)+n) * (hyp(1) - meanX)^2;
%Assume mu0=0 and kappa0 << n
nu_sigsq = hyp(5) + stats(:,3) + hyp(2) * stats(:,2).^2;

logp = sum(hyp(6) + gammaln(nu/2)- 0.5*log(kappa) - (nu/2).*log(nu_sigsq)- (stats(:,1)/2)*log(pi));
end

