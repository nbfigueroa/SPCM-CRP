function [Y, d, thres, V] = spectral_DimRed(S , M)
% This algorithm maps the data into M-dimensional spectral space from
% similarity matrix S using the method from (On Spectral Clustering: Analysis and an algorithm. Andrew Ng.)
% Implementation of Algorithm 1. from Socher11a paper (Spectral Chinese Restaurant Processes)

% Input: Similarity Matrix S and M \in R [1,100] correpsonding to manifold
% dimensionality

% Output: M-dimensional points (y1,...,,yN) where N = dim(S), M is given or
% computed

% Example: blah blah

% Author: Nadia Figueroa, PhD Student., Robotics
% Learning Algorithms and Systems Lab, EPFL (Switzerland)
% Email address: nadia.figueroafernandez@epfl.ch  
% Website: http://lasa.epfl.ch
% February 2016; Last revision: 07-July-2016


% Remove Diagonal Values (self-similarities)
% S = S - eye(size(S));


% fprintf('Computing Spectral Dimensionality Reduction based on SPCM Similarity Function...\n');
% tic;

[N] = length(S);

% Compute Diagonal Degree Matrix:
D = diag(sum(S,2));

% Compute Un-normalized Graph Laplacian
L = D - S;

% Compute Symmetric Normalized Laplacian
L_sym = D^(-1/2)*L*D^(-1/2);

% Compute Eigen Decomposition of L_sym
[V,D] = eig(L_sym);

D_sort = diag(sort(diag(D),'ascend')); % make diagonal matrix out of sorted diagonal values of input D
[~, ind]=sort(diag(D_sort),'ascend'); % store the indices of which columns the sorted eigenvalues come from
V_sort=V(:,ind); % arrange the columns in this order

% Vectorize eigenvalues
d = real(diag(D_sort));

% If M is not given, find optimal threshold using softmax + attractive
% adaptation
thres  = 0;
eps_i  = 0.2;
eps_ii = 0.2;

if isempty(M)   
    s = softmax(d);
    s_norm = normalize_soft(s);    
    M_cut = sum(s_norm < thres);

    % Attractive Threshold adaptation
    if (abs(s_norm(M_cut)) < eps_i) || (abs(s_norm(M_cut+1)) < eps_i)
        if M_cut > 2 
            f = [ s_norm(M_cut-2) s_norm(M_cut-1) s_norm(M_cut) s_norm(M_cut+1)];
            thres = thres + mean(f);
        elseif N > 3
            f = [s_norm(M_cut) s_norm(M_cut+1) s_norm(M_cut+1) s_norm(M_cut+2)];
            thres = thres + mean(f);
        end        
        M_cut = sum(s_norm < thres);
    end
    
        if abs(s_norm(M_cut + 1) - thres) < eps_ii
            M_cut = M_cut + 1;
            thres = s_norm(M_cut);
        end
    M = M_cut;
end

% Choose M eigenvectors
V_M = V_sort(:,1:M);

% Scale by square eigenvalues
lambda_sqrt = repmat(d(1:M)',N,1);
V_M = V_M.*lambda_sqrt;

% Normalize rows of V to unit length
% V_M = bsxfun(@rdivide, V_M, sum(V_M,2));

% Define NxM observation vector Y = [y1; ...; yN] as rows of V
Y = V_M;

% Transpose to have columns as observations
Y = Y';


% toc;
% fprintf('*************************************************************\n');

end
