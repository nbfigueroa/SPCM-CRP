function [new_lambdas] = resample_lambdas(Y, Z_C, lambdas, type)
    K = max(Z_C);
    Z = bsxfun(@eq,Z_C,1:K);
    
    Nks = sum(Z);
    YbarN = Y*Z; % Ybar*N
    Ybar = bsxfun(@rdivide,YbarN,Nks);
    
    % Updating Means
    new_lambdas.mu_n = bsxfun(@rdivide,lambdas.kappa0.*lambdas.mu0 + YbarN, lambdas.kappa0+Nks);
    
    switch type
        case 'diag'
            % Computing lambdas for Precision (NG)
            new_lambdas.kappa_n = Nks + lambdas.kappa0;
            new_lambdas.a_n = lambdas.a0 + Nks./2;
            new_lambdas.b_n = lambdas.b0 + 0.5 * ((Y-YbarN(:,Z_C)).^2)*Z + bsxfun(@rdivide,lambdas.kappa0.* bsxfun(@times,Nks, (Ybar-lambdas.mu0).^2),2.*(lambdas.kappa0+Nks));
            
        case 'full'
            % Computing lambdas for Covariance Matrix (NIW)
    end
    
   
end