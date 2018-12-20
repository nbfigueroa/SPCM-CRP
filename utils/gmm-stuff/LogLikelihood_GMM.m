function [ loglik ] = LogLikelihood_GMM(X,Priors,Mu,Sigma,w)
%LOGLIKELIHOOD_GMM computes the likelihood of data given a GMM
% Used as a utility function for GMM_BIC

%   input------------------------------------------------------------------
%
%       o X : (D x N), set of N data points of dimension  D
%
%       o w : (1 x N), set of weights if the data points are weighted
%
%   output ----------------------------------------------------------------
%
%       o loglik : (N x 1) , loglikelihood
%


N = size(X,2);
K = size(Priors,2);

if isempty(w)
    w = ones(1,length(X));
end

Pxi = zeros(N,K);
for i=1:K
    Pxi(:,i) = my_gaussPDF(X, Mu(:,i), Sigma(:,:,i));
end

if exist('w','var') && ~isempty(w)
    %               (N x K)
    %             (N x 1)
    F            = (repmat(w(:),1,K) .* Pxi) * Priors';
else
    
    %Compute the log likelihood
    F            = Pxi*Priors';
   
end

F(F<realmin) = realmin;
F            = Pxi*Priors';
loglik       = sum(log(F));


end