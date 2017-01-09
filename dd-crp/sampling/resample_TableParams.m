function [Theta] = resample_TableParams(Y, Z_C, lambdas, type)

    % Updating lambda-parameters
    new_lambdas = resample_lambdas(Y, Z_C, lambdas, type);
    
    % New Cluster Means \mu = {mu_1, ... \mu_K}
    Mu = new_lambdas.mu_n;
    
    switch type
        case 'diag'
            % Computing new precision values \lambda = {\lambda_1, ... \lambda_K}
            s2 = bsxfun(@rdivide,new_lambdas.beta_n,(new_lambdas.alpha_n.*new_lambdas.kappa_n));
            t = tinv(0.975, 2 * new_lambdas.alpha_n);
            Pr = bsxfun(@times,t./(2.*new_lambdas.alpha_n+1),sqrt(s2));
            
        case 'full'     
            % Computing new cluster Covariance matrices \Sigma = {\Sigma_1, ... \Sigma_K}
            
    end
    
    Theta.Mu = Mu;
    Theta.Pr = Pr;
        
end
