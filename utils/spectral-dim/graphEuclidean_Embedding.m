function [x_emb, x_emb_apprx] = graphEuclidean_Embedding(S)
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
plotSimilarityConfMatrix(L_pseudo, 'Laplacian Pseudo-Inverse'); 

% Compute Eigen Decomposition of L_sym
[V_pseudo,D_pseudo] = eig(L_pseudo);

D_sort_pseudo = real(diag(sort(diag(D_pseudo),'descend'))); % make diagonal matrix out of sorted diagonal values of input D
[~, ind]=sort(diag(D_sort_pseudo),'descend'); % store the indices of which columns the sorted eigenvalues come from
V_sort_pseudo= real(V_pseudo(:,ind)); % arrange the columns in this order

% Vectorize eigenvalues
d_pseudo    = real(diag(D_sort_pseudo));

%%%%% Maximum Variance Subpace Projection of the Node Vectors %%%%% 
% Construct Resistance Distance Matrix
Vol_graph = sum(real(diag(D)));
N_eucl = zeros(n,n);
% N      = zeros(n,n);

% Create Euclidean embedded vectors
E     = eye(n);
x_emb = zeros(n,n);
for i=1:n
        x_i = V_sort_pseudo'*E(:,i);
        x_emb(:,i) = (D_sort_pseudo.^.5)*x_i;
end

% Compute distance matrix
for i=1:n
    for j=1:n
%         N(i,j)      = Vol_graph * (L_pseudo(i,i) + L_pseudo(j,j) - 2*L_pseudo(i,j));
        N_eucl(i,j) = Vol_graph * (x_emb(:,i)-x_emb(:,j))'*(x_emb(:,i)-x_emb(:,j));
    end
end    
% plotSimilarityConfMatrix(N, '$N$ = Resistance Distance induced by B-SPCM Graph (eq)'); 
% plotSimilarityConfMatrix(N_eucl, '$N$ = Resistance Distance induced by B-SPCM Graph (emb)'); 

% Compute eigenvalues of powered distance matrix
pow = 4;
[V_pow,D_pow] = eig(mpower(L_pseudo,pow));
D_sort_pow    = real(diag(sort(diag(D_pow),'descend'))); % make diagonal matrix out of sorted diagonal values of input D
[~, ind]      = sort(diag(D_sort_pow),'descend'); % store the indices of which columns the sorted eigenvalues come from
d_L_pow = diag(D_sort_pow);


[~, opt_ids_der]  = ml_curve_opt(d_L_pow','derivatives');
[~, opt_ids_line] = ml_curve_opt(d_L_pow','line');
k_options = sort([opt_ids_der opt_ids_line],'ascend');
k_dim = k_options(1)

x_emb_apprx = zeros(k_dim,n);
D_apprx     = zeros(n,n); V_apprx   = zeros(n,n);
D_apprx(1:k_dim, 1:k_dim) = D_sort_pseudo(1:k_dim, 1:k_dim);
V_apprx(:, 1:k_dim)       = V_sort_pseudo(:, 1:k_dim);
for i=1:n
        x_i_full         = V_apprx'*E(:,i);
        x_emb_apprx_full = (D_apprx.^.5)*x_i_full;
        x_emb_apprx(:,i) = x_emb_apprx_full(1:k_dim,1);
end

% Compute approximate distance matrix
N_approx = zeros(n,n);
for i=1:n
    for j=1:n
        N_approx(i,j) = Vol_graph * (x_emb_apprx(:,i)-x_emb_apprx(:,j))'*(x_emb_apprx(:,i)-x_emb_apprx(:,j));
    end
end 
plotSimilarityConfMatrix(N_approx, '$\tilde{N}$ = Approximate Resistance Distance'); 


% Plot eigenvalues of matrices
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
ylabel('Eigenvalues of $\mathcal{L}_{sym}^{+}$','Interpreter','Latex','FontSize',14)
tit = strcat('Eigenvalues Laplacian PseudoInverse');
title(tit, 'Interpreter','Latex','FontSize',14)

subplot(1,3,3);
plot(d_L_pow,'-*r'); hold on
grid on
xlabel('Eigenvalue Index $\lambda_i$','Interpreter','Latex','FontSize',14)
ylabel('Eigenvalues of ${\mathcal{L}_{sym}^{+}}^{(4)}$','Interpreter','Latex','FontSize',14)
tit = strcat('Eigenvalues Laplacian PseudoInverse Powered(4)');
title(tit, 'Interpreter','Latex','FontSize',14)



end
% N      = zeros(n,n);
% N_vec  = zeros(n,n);
% N_eucl = zeros(n,n);
%         N(i,j)      = Vol_graph * (L_pseudo(i,i) + L_pseudo(j,j) - 2*L_pseudo(i,j));
%         N_vec(i,j)  = Vol_graph * (E(:,i)-E(:,j))'*L_pseudo*(E(:,i)-E(:,j));
% plotSimilarityConfMatrix(N, 'Resistance Distance induced by B-SPCM Graph (Direct Equation)'); 
% plotSimilarityConfMatrix(N_vec, 'Resistance Distance induced by B-SPCM Graph (Node Vectors)'); 