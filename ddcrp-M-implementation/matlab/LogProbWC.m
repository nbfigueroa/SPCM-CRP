% Compute probability of each Ward clustering (from Z) of a matrix D at
%   various sizes, using our model with hyperparameters alpha, kappa, nu, sigsq
function logp = LogProbWC(D, Z, sizes, alpha, kappa, nu, sigsq)
hyp = ComputeCachedLikelihoodTerms(kappa, nu, sigsq);

logp = zeros(length(sizes),1);
for i = 1:length(sizes)
    z = cluster(Z, 'maxclust', sizes(i))';
    [sorted_z, sorted_i] = sort(z);
    parcels = mat2cell(sorted_i, 1, diff(find(diff([0 sorted_z (max(z)+1)]))));
    
    % Formally we should construct a spanning tree within each cluster so
    %   that we can evaluate the probability. However, the only property of
    %   the "c" links that impacts the probability directly is the number of
    %   self-connections. So we simply add the correct number of self-
    %   connections (equal to the number of parcels) and leave the rest
    %   set to zero
    c = zeros(length(z),1);
    c(1:sizes(i)) = 1:sizes(i);
    
    logp(i) = FullProbabilityddCRP(D, c, parcels, alpha, hyp, CheckSymApprox(D));
end