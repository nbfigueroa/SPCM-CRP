function LL = table_logLik_NIG(Y,a0,b0,mu0,kappa0)
% log likelihood of table parametrs given data for Normal-Inverse-Gamma

[M,N] = size(Y);
% Ym = sum(Y,2)./N;
Ym = Y*ones(N,1)./N;

ak = a0+N./2;
kappak = kappa0+N;
bkt = b0 + 0.5*sum(bsxfun(@minus,Ym,Y).^2,2) + (kappa0*N*(Ym-mu0).^2)./(2*(kappak));

LL = M*(gammaln(ak)-gammaln(a0)+a0*log(b0)+0.5*log(kappa0)-0.5*log(kappak)-0.5*N*log(2*pi));
LL = LL + sum(-ak*log(bkt));