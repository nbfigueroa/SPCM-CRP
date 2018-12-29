function [X,X_test,zt] = generateSynthData(p,N,ZS,R)
% generate covariance parameters
assert(length(N)==length(ZS))

if nargin<4
for n = 1:length(unique(ZS))
    R{n}=triu(randn(p));
end
else
   assert(length(R)==max(ZS))
end

% generate data X
 X=zeros(p,sum(N));
 X_test=zeros(p,sum(N));
 zt = [];
 
for n=1:length(N)
    X(:,sum(N(1:n-1))+1:sum(N(1:n)))=R{ZS(n)}'*randn(p,N(n));
    X_test(:,sum(N(1:n-1))+1:sum(N(1:n)))=R{ZS(n)}'*randn(p,N(n));
    zt = [zt ZS(n)*ones(1,N(n))];
end

%eof
end