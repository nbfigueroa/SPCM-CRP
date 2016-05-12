function LL = singleClustLL(X,a0,b0,mu0,kappa0)

[T,N] = size(X);
% Xm = sum(X,2)./N;
Xm = X*ones(N,1)./N;

ak = a0+N./2;
kappak = kappa0+N;
bkt = b0 + 0.5*sum(bsxfun(@minus,Xm,X).^2,2) + (kappa0*N*(Xm-mu0).^2)./(2*(kappak));

LL = T*(gammaln(ak)-gammaln(a0)+a0*log(b0)+0.5*log(kappa0)-0.5*log(kappak)-0.5*N*log(2*pi));
LL = LL + sum(-ak*log(bkt));
