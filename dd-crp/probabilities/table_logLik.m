function LL = table_logLik(Y, lambda, type)
% log likelihood of table parametrs given data for Conjugate Prior
% Marginal log likelihood of data given hyper-parameters

[M,N] = size(Y);
Ybar = mean(Y,2);

mu_0    = lambda.mu0;
kappa_0 = lambda.kappa0;

switch type
    case 'diag'                
        % Marginal log likelihood of NG distribution
        alpha_0 = lambda.alpha0;
        beta_0 = lambda.beta0;
        
        % Compute posterior update parameters Eq. 86-89
        % p(\mu, \lambda) = NG(\mu,\lambda | \mu_N, \kappa_n, \alpha_n,
        % \beta_n ) (page 8 . Conjugate Bayesian analysis of the Gaussian distribution 'Murphy')
        alpha_N = alpha_0 + N/2;
        kappa_N = kappa_0 + N;
        beta_N  = beta_0 + 0.5*sum(bsxfun(@minus,Ybar,Y).^2,2) + (kappa_0*N*(Ybar-mu_0).^2)./(2*(kappa_N));
  
        % Marginal Likelihood p(Y|\lambda) Eq. 95
        % (page 9 . Conjugate Bayesian analysis of the Gaussian distribution 'Murphy')
        LL = M*(gammaln(alpha_N)- gammaln(alpha_0) + alpha_0*log(beta_0) + 0.5*log(kappa_0)-0.5*log(kappa_N)-0.5*N*log(2*pi));
        LL = LL + sum(-alpha_N*log(beta_N));
        
    case 'full'        
        % Predicted log likelihood of NIW distribution
        % p(\mu, \Sigma) = NIW(\mu,\Sigma | \mu_N, \kappa_n, \nu_n, \Lambda_n )

        nu_0     = lambda.nu0;
        Lambda_0 = lambda.Lambda0;
        
        
end






