function [ bic ] = GMM_BIC(Data,w,Priors,Mu,Sigma,cov_type)
%GMM_BIC Computes the Bayesian Information criterion
%
%  input ------------------------------------------------------------------
%
%   o Data:     D x N array representing N datapoints of D dimensions.
%
%   o w:        1 x N, set of weights associated with each data point.
%
%   o Priors:   1 x K array representing the prior probabilities of the
%               K GMM components.
%   o Mu:       D x K array representing the centers of the K GMM components.
%
%   o Sigma:    D x D x K array representing the covariance matrices of the
%               K GMM components.
%
%   o cov_type: string, covariance type = 'full','diag' or 'isot'
%
%  output -----------------------------------------------------------------
%
%   o bic   : (1 x 1)
%
%




[D,N]       = size(Data);
K           = length(Priors);
num_param   = K-1 + K * D;

if strcmp(cov_type,'full') == true
    num_param = num_param + K * ( D * ( D - 1)/2 );
elseif strcmp(cov_type,'diag') == true
    num_param = num_param + K * D;
elseif strcmp(cov_type,'isot') == true
    num_param = num_param + K * 1;
else
   error(['no such covariance type: ' cov_type '  only full | diag | isot ']); 
end
    

% compute the loglikelihood of the data given the model
% loglik: (N x 1) 

loglik = LogLikelihood_GMM(Data,Priors,Mu,Sigma,w);
bic    = - 2 * loglik + num_param * log(N);

end