function [x_emb, x_emb_apprx, S, S_SP, gamma] = deformedKernelPCA_Embedding(D, S_sp, emb_options)
% This algorithm maps the data into M-dimensional spectral space from
% similarity matrix S using the method from (On Spectral Clustering: Analysis and an algorithm. Andrew Ng.)
% Implementation of Algorithm 1. from Socher11a paper (Spectral Chinese Restaurant Processes)

% Input: Distance Matrix S and M \in R [1,100] correpsonding to manifold
% dimensionality

% Output: M-dimensional points (y1,...,,yN) where N = dim(S), M is given or
% computed

% Example: blah blah

% Author: Nadia Figueroa, PhD Student., Robotics
% Learning Algorithms and Systems Lab, EPFL (Switzerland)
% Email address: nadia.figueroafernandez@epfl.ch  
% Website: http://lasa.epfl.ch
% December 2018;

l_sensitivity = emb_options.l_sensitivity;
norm_K        = emb_options.norm_K;
pow_eigen     = emb_options.pow_eigen;
show_plots    = emb_options.show_plots; 
distance_name = emb_options.distance_name;
deform_kernel =  emb_options.deform;

% Contruct Kernel Matrix
sigma = sqrt(mean(D(:))/l_sensitivity);
gamma = 1/(2*sigma^2)
K = exp(-gamma*D.^2);
S = K;
if show_plots
    title_str = strcat('Kernel of ',{' '}, distance_name);
    plotSimilarityConfMatrix(K, title_str);
end

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
    S_SP = K;
end

% Manifold adaptive kernel
if (deform_kernel == 1)
    % Construct Laplacian from SPCM similarity (kernel)
    A = S_sp;                    % Adjacency comes from similarity
    n = length(A);               % Number of nodes
    D = diag(sum(A,2));          % Compute Diagonal Degree Matrix:
    L = D - A;                   % Compute Un-normalized Graph Laplacian
    L_sym = D^(-1/2)*L*D^(-1/2); % Compute Symmetric Normalized Laplacian
%     K_def = inv((inv(K) + L_sym)); % Deformed Kernel with mu=1
    mu = 1;
    K_def = K - mu*(K'*inv(eye(n) + L_sym*K)*(L_sym*K)); % Deformed Kernel
    K = K_def;
    S_SP = K;
else
    S_SP = [];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% Apply Kernel-PCA %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic;


K(isnan(K)) = 0;
K(isinf(K)) = 0;

% Compute first no_dims eigenvectors and store these in V, store corresponding eigenvalues in L
disp('Eigenanalysis of kernel matrix...');
[V, L] = eig(K);

% Sort eigenvalues and eigenvectors in descending order
[L, ind] = sort(diag(L), 'descend');

%%%%%% Computing the full embedding (no reduction) %%%%%%
no_dims = ell;
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
d_pseudo    = real(L);

% Find the optimal number of dimensions using the power heuristic
pow = pow_eigen;
[~, D_pow] = eig(mpower(K,pow));
D_sort_pow    = real(diag(sort(diag(D_pow),'descend'))); % make diagonal matrix out of sorted diagonal values of input D
d_L_pow = diag(D_sort_pow);
[~, opt_ids_der]  = ml_curve_opt(d_L_pow','derivatives');
[~, opt_ids_line] = ml_curve_opt(d_L_pow','line');
k_options = sort([opt_ids_der opt_ids_line],'ascend');
k_dim = k_options(1);

no_dims = k_dim;
eig_diff = L_full(no_dims-1) - L_full(no_dims);
if eig_diff < 0.01
    no_dims = k_dim -1
end
    
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
    subplot(1,2,1);
    plot(d_pseudo,'-*r'); hold on
    grid on
    xlabel('Eigenvalue Index $\lambda_i$','Interpreter','Latex','FontSize',14)
    ylabel('Eigenvalues of $\mathcal{K}$','Interpreter','Latex','FontSize',14)
    tit = strcat('Eigenvalues of Kernel Matrix');
    title(tit, 'Interpreter','Latex','FontSize',14)
    
    subplot(1,2,2);
    plot(d_L_pow,'-*r'); hold on
    grid on
    xlabel('Eigenvalue Index $\lambda_i$','Interpreter','Latex','FontSize',14)
    ylabel('Eigenvalues of ${\mathcal{L}_{sym}^{+}}^{(N)}$','Interpreter','Latex','FontSize',14)
    tit = strcat('Eigenvalues Powered Kernel Matrix $\mathcal{K}^{(N)}$ ');
    title(tit, 'Interpreter','Latex','FontSize',14)
end


end
