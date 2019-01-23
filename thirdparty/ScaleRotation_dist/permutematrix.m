function [P, nperm]= permutematrix(p);
% PERMUTEMATRIX for dimension p = 2,3,4,... , 
% P = permutematrix(p) is a cell array of pxp matrices, where each cell is
% a permutation matrix with determinant 1. 
% The length of cell array is p! = the number of all permutations.
%
% June 9, 2015 Sungkyu Jung.

nperm = factorial(p);
allperm=perms(1:p);
P = cell(nperm,1); 

for i = 1:nperm
    ii = zeros(p);
    for j = 1:p
    ii(j,allperm(i,j)) = 1;
    end
    ii(1,allperm(i,1)) = det(ii);
    P{i} = ii;
    % det(ii) % this should be 1
end
    

