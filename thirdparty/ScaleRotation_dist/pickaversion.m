function [U,D] =pickaversion(M) 
% PICKAVERSION creates a version of $M$. .
% [U,D] = pickaversion(M), for a p x p SPD matrix M, returns U in SO(p) and 
%         D in Diag+(p). Eigenvalues are in descending order
% 
% June 9, 2015 Sungkyu Jung.

[p1,~]=size(M);
[U,D]=svd(M);
d = diag(D); [~, d1perm] = sort(d,'descend');  % make sure D is in descending order
ii = zeros(p1);
for j = 1:p1;    ii(j,d1perm(j)) = 1;   end
U = U*ii'; D = ii*D*ii';  
if det(U)<0; % make sure U is in SO(p)
    U(:,1) = -U(:,1);
end
