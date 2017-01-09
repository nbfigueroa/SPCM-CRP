function [new_lambdas] = resample_lambdas(Y, Z_C, lambdas, type)
    K = max(Z_C);
    Z = bsxfun(@eq,Z_C,1:K);
    
    Nks = sum(Z);
    YbarN = Y*Z; % Ybar*N
    Ybar = bsxfun(@rdivide,YbarN,Nks);
    
    % Updating Mean Parameters
    new_lambdas.mu_n = bsxfun(@rdivide,lambdas.kappa_0.*lambdas.mu_0 + YbarN, lambdas.kappa_0+Nks);
    new_lambdas.kappa_n = Nks + lambdas.kappa_0;
    
    switch type
        case 'diag'
            % Update Precision Parameters (NG)
            new_lambdas.alpha_n = lambdas.alpha_0 + Nks./2;
            new_lambdas.beta_n  = lambdas.beta_0 + 0.5 * ((Y-YbarN(:,Z_C)).^2)*Z + bsxfun(@rdivide,lambdas.kappa_0.* bsxfun(@times,Nks, (Ybar-lambdas.mu_0).^2),2.*(lambdas.kappa_0+Nks));
            
        case 'full'
            % Update Covariance Parameters (NIW)
    end
    
   
end