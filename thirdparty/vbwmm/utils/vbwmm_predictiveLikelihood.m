function logP = vbwmm_predictiveLikelihood(Ctest,nu_test, expectations, priors, simple)
% Calculates predictive likelihood for the Variational Bayes Wishart 
% Mixture Model
K = priors.K; p = priors.p;
logP_pr_K=nan(size(Ctest,3),priors.K);
for k=1:K        
    vk = expectations.pars.vk(k);
    Phi_k=(1/vk)*expectations.SigmaInv(:,:,k);
    R=chol(Phi_k);
    for i=1:size(Ctest,3)
        RtestInv=chol( inv(Phi_k)+Ctest(:,:,i) );
        logP_pr_K(i,k)=-(vk+nu_test(i))*sum(log(diag(RtestInv)))+mvgammaln(p,(vk+nu_test(i))/2);
        if ~simple
            logP_pr_K(i,k)= logP_pr_K(i,k)+(nu_test(i)-p-1)/2*log(det(Ctest(:,:,i)))-mvgammaln(p,nu_test(i)/2);
        end
    end
    logP_pr_K(:,k)=logP_pr_K(:,k)-mvgammaln(p,vk/2)-vk*sum(log(diag(R))); % leftovers from the training-posterior
end
logP= bsxfun(@plus,logP_pr_K,log(expectations.Pi));
logP = lsum(logP,2);
%eof
end