% Computes sum of log-likelihood terms for given sufficient statistics (in Nx3
%   matrix, with columns [count, mean, sum of squared dev]) and vectorized
%   hyperparameters
function i = ChooseFromLP(lp)

max_lp = max(lp);
normLogp = lp - (max_lp + log(sum(exp(lp-max_lp))));
p = exp(normLogp);
p(~isfinite(p)) = 0;
cumP = cumsum(p);
i = find(cumP>rand,1);
end

