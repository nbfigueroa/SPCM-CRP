function Array = matd(vecArray,option)
% inputs
%  Array: array of p x p pd matrices, with length n, Array = p x p x n;
%  option 0 or 1 (0 for multiplying \sqrt(2), 1 for not)
% output vectorized version : put diagonal first, then off-diagonal

if nargin == 1;
    option = 0;
end
if option == 0;
    tau = 1/sqrt(2);
else
    tau = 1;
end

sizeArray = size(vecArray);
p = (sqrt(sizeArray(1)*8+1) - 1) / 2;

for k = 1:sizeArray(2)
    Array(:,:,k) = diag(vecArray(1:p,k));
    ind = p+1;
    for i = 1:p-1
        for j = i+1:p
            Array(i,j,k) =  tau*vecArray(ind,k);
            Array(j,i,k) =  tau*vecArray(ind,k);
            ind = ind+1;
        end
    end
end