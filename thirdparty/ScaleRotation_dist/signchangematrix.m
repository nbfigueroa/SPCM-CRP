function [Is, nsignchange]= signchangematrix(p);
% SIGNCHANGEMATRIX for dimension p = 2 and 3
% Is = signchangematrix(p) is a cell array of pxp matrices, where each cell 
% is a sign change matrix with determinant 1. 
% The length of cell array is 2^(p-1) = the number of all sign changes.
% only works for  p = 2 and p = 3.
%
% June 9, 2015 Sungkyu Jung.

I =eye(p);
nsignchange = 2^(p-1);
Is =cell(nsignchange,1);
if p == 2;
    Is{1} = I;
    Is{2} = -I;
end
if p == 3;
    Is{1} = I; 
    isn = 2;
    for i = 1:2;
        for j = i+1:3;
            ii = [1,1,1];
            ii(i) = -1;
            ii(j) = -1;
            Is{isn} = diag(ii);
            isn = isn+1;
        end
    end
end
    