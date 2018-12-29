function Z = createAssignmentMatrix(z,kmax)
% Function creates assignment matrix of size k x N
% k is the number of different labels in partioning z and N is the number
% of data points. Each column of output Z cotains only a single 1 the rest
% of the entries are zero.
[~,~,zz] = unique(z);
if nargin < 2
    kmax = max(zz);
end
Z = full(sparse(zz,1:length(zz),ones(1,length(zz)),kmax,length(zz)));
%eof
end