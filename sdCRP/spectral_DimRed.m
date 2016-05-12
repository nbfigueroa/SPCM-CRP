function [X, D_sort] = spectral_DimRed(S , M)
% This algorithm maps the data into M-dimensional spectral space from
% similarity matrix S using the method from (On Spectral Clustering: Analysis and an algorithm. Andrew Ng.)
% Implementation of Algorithm 1. from Socher11a paper (Spectral Chinese Restaurant Processes)

% Input: Similarity Matrix S

% Output: M-dimensional points (x1,...,,xN) where N = dim(S)

% Example: blah blah

% Author: Nadia Figueroa, PhD Student., Robotics
% Learning Algorithms and Systems Lab, EPFL (Switzerland)
% Email address: nadia.figueroafernandez@epfl.ch  
% Website: http://lasa.epfl.ch
% February 2016; Last revision: 12-Feb-2016


% Remove Diagonal Values (self-similarities)
S = S - eye(size(S));

% Compute Diagonal Degree Matrix:
D = diag(sum(S,2));

% Compute Un-normalized Graph Laplacian
L = D - S;

% Compute Symmetric Normalized Laplacian
L_sym = D^(-1/2)*L*D^(-1/2);

% Compute Random-Walk Normalized Laplacian
% L_sym = D^(-1)*L;

% Compute Eigen Decomposition of L_sym
[V,D] = eig(L_sym);

D_sort = diag(sort(diag(D),'ascend')); % make diagonal matrix out of sorted diagonal values of input D
[c, ind]=sort(diag(D_sort),'ascend'); % store the indices of which columns the sorted eigenvalues come from
V_sort=V(:,ind); % arrange the columns in this order

% Evaluate Eigengap
% figure('Color',[1 1 1])
d = diag(D_sort);
plot(d,'*')
hold on 
d_gap = abs(d - [0;d(1:end-1)]); 
plot(d_gap, 'Color', [1 0 0])

% Choose M eigenvectors
V_M = V_sort(:,1:M);

% Normalize rows of V to unit length
V_M = bsxfun(@rdivide, V_M, sum(V_M,2));

% Define MxN observation vector X = (x1, ..., xN) as rows of V
X = V_M';

end
