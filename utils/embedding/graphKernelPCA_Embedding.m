function [x_emb, x_emb_apprx] = graphKernelPCA_Embedding(S, show_plots, norm_K, pow_eigen)
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
% December 2018;

% Adjacency matrix is my similarity matrix
A = S;

% Number of nodes
n = length(A);

% Compute Diagonal Degree Matrix:
D = diag(sum(S,2));

% Compute Un-normalized Graph Laplacian
L = D - S;

% Compute Symmetric Normalized Laplacian
L_sym = D^(-1/2)*L*D^(-1/2);
% plotSimilarityConfMatrix(L_sym, 'Symmetric Normalized Laplacian');

% Compute Eigen Decomposition of L_sym
[V,D] = eig(L_sym);

D_sort = diag(sort(diag(D),'ascend')); % make diagonal matrix out of sorted diagonal values of input D
[~, ind]=sort(diag(D_sort),'ascend'); % store the indices of which columns the sorted eigenvalues come from
% V_sort=V(:,ind); % arrange the columns in this order

% Vectorize eigenvalues
d = real(diag(D_sort));

% Compute Pseudo-Inverse of Symmetric Normalized Laplacian
L_pseudo      = pinv(L_sym);
if show_plots
    plotSimilarityConfMatrix(L_pseudo, 'Laplacian Pseudo-Inverse');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% Apply Kernel-PCA on L^+ %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic;

% The PseudoInverse of the Laplacian is a Kernel Matrix
K = L_pseudo;

% Store the number of training and test points
ell = size(K, 1);
    
% Normalize kernel matrix K
if norm_K == true
    disp('Normalizing kernel (Gram) matrix...');
    column_sums = sum(K) / ell;                  % column sums
    total_sum   = sum(column_sums) / ell;        % total sum
    J = ones(ell, 1) * column_sums;              % column sums (in matrix)
    K = K - J - J';
    K = K + total_sum;
end

if show_plots
    plotSimilarityConfMatrix(K, 'Kernel Matrix');
end

% Compute first no_dims eigenvectors and store these in V, store corresponding eigenvalues in L
disp('Eigenanalysis of kernel matrix...');
K(isnan(K)) = 0;
K(isinf(K)) = 0;
[V, L] = eig(K);

% Sort eigenvalues and eigenvectors in descending order
[L, ind] = sort(diag(L), 'descend');

%%%%%% Computing the full embedding (no reduction) %%%%%%
no_dims = n-1;
L_full = L(1:no_dims);
V_full = V(:,ind(1:no_dims));

% Compute inverse of eigenvalues matrix L
disp('Computing final embedding...');

% Compute square root of eigenvalues matrix L
sqrtL = diag(sqrt(L_full));

% Compute inverse of square root of eigenvalues matrix L
invsqrtL = diag(1 ./ diag(sqrtL));

% Compute the new embedded points for both K and Ktest-data
% x_emb = sqrtL * V_full';    
x_emb = invsqrtL * V_full' * K;

%%%%%% Computing the reduced embedding %%%%%%

% Vectorize eigenvalues
d_pseudo    = real(L)

% Find the optimal number of dimensions using the power heuristic
pow = pow_eigen;
[~, D_pow] = eig(mpower(K,pow));
D_sort_pow    = real(diag(sort(diag(D_pow),'descend'))); % make diagonal matrix out of sorted diagonal values of input D
d_L_pow = diag(D_sort_pow)
[~, opt_ids_der]  = ml_curve_opt(d_L_pow','derivatives');
[~, opt_ids_line] = ml_curve_opt(d_L_pow','line');
k_options = sort([opt_ids_der opt_ids_line],'ascend');
k_dim = k_options(1)

no_dims = k_dim;
L_red = L(1:no_dims);
V_red = V(:,ind(1:no_dims));

% Compute inverse of eigenvalues matrix L
disp('Computing final embedding...');
invL = diag(1 ./ L_red);

% Compute square root of eigenvalues matrix L
sqrtL = diag(sqrt(L_red));

% Compute inverse of square root of eigenvalues matrix L
invsqrtL = diag(1 ./ diag(sqrtL));

% Compute the new embedded points of K
%y = 1/lambda * sum(alpha)'s * Kernel
% y = sqrtL(1:p,1:p) * V(:,1:p)';
x_emb_apprx = sqrtL * V_red';          % = invsqrtL * V'* K
% x_emb_apprx = invsqrtL * V_red'* K;

% Plot eigenvalues of matrices
if show_plots
    figure('Color',[1 1 1]);
    subplot(1,3,1);
    plot(d,'-*r'); hold on
    grid on
    xlabel('Eigenvalue Index $\lambda_i$','Interpreter','Latex','FontSize',14)
    ylabel('Eigenvalues of $\mathcal{L}_{sym}$','Interpreter','Latex','FontSize',14)
    tit = strcat('Eigenvalues of Symmetric Normalized Laplacian');
    title(tit, 'Interpreter','Latex','FontSize',14)
    
    subplot(1,3,2);
    plot(d_pseudo,'-*r'); hold on
    grid on
    xlabel('Eigenvalue Index $\lambda_i$','Interpreter','Latex','FontSize',14)
    ylabel('Eigenvalues of $\mathcal{K}$','Interpreter','Latex','FontSize',14)
    tit = strcat('Eigenvalues of Kernel Matrix');
    title(tit, 'Interpreter','Latex','FontSize',14)
    
    subplot(1,3,3);
    plot(d_L_pow,'-*r'); hold on
    grid on
    xlabel('Eigenvalue Index $\lambda_i$','Interpreter','Latex','FontSize',14)
    ylabel('Eigenvalues of ${\mathcal{L}_{sym}^{+}}^{(N)}$','Interpreter','Latex','FontSize',14)
    tit = strcat('Eigenvalues Powered Kernel Matrix $\mathcal{K}^{(N)}$ ');
    title(tit, 'Interpreter','Latex','FontSize',14)
end


end
