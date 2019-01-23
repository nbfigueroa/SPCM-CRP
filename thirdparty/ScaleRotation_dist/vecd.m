function vecArray = vecd(Array,option)
% inputs
%  Array: array of p x p pd matrices, with length n, Array = p x p x n;
%  option 0 or 1 (0 for multiplying \sqrt(2) (default), 1 for not)       
% output vectorized version : put diagonal first, then off-diagonal

if nargin == 1;
    option = 0;
end
if option == 0;
    tau = sqrt(2);
else
    tau = 1;
end
sizeArray = size(Array);
if length(sizeArray) == 2
    [p p]= size(Array);
    offdiag = [];
    for i = 1:p-1
        for j = i+1:p
            offdiag = [offdiag ; Array(i,j)];
        end
    end    
    vecArray = [diag(Array) ;tau*offdiag];
else
    [p p n]= size(Array);
    vecArray = zeros(p*(p+1)/2,n);
    for kk = 1:n
    aa = Array(:,:,kk);
    offdiag = [];
    for i = 1:p-1
        for j = i+1:p
            offdiag = [offdiag ; aa(i,j)];
        end
    end    
    vecArray(:,kk) = [diag(aa) ;tau*offdiag];
    end
end