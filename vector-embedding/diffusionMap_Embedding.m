function [x_emb, x_emb_apprx, S, l] = diffusionMap_Embedding(D, emb_options)
% This algorithm maps the data into M-dimensional lower-dimensional vector-space.

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
pow_eigen     = emb_options.pow_eigen;
show_plots    = emb_options.show_plots; 
distance_name = emb_options.distance_name;
t             = emb_options.t;
markov_constr = emb_options.markov_constr;

% Contruct Kernel Matrix
sigma = sqrt(mean(D(:))/l_sensitivity);
l = 1/(2*sigma^2)
K = exp(-l*D.^2);
S = K;
if show_plots
    title_str = strcat('Kernel of ',{' '}, distance_name);
    plotSimilarityConfMatrix(K, title_str);
end


% Compute Markov probability matrix with t timesteps
disp(['Compute Markov forward transition probability matrix with ' num2str(t) ' timesteps...']);


% Two ways of constructing the markov probability matrix
% Option 1: symmetry preserving
% Option 2: stochasticity preserving
switch markov_constr
    case 1
        p = sum(K, 1)';        
        % New kernel
        L_alpha = K ./ ((p * p') .^ t);
        
        % Graph Laplacian Normalization on new kernel
        p = sqrt(sum(L_alpha, 1))';
        M_t = L_alpha ./ (p * p');
        
    case 2
        
        %%%%%%% "Classical Way" %%%%%%%
        % Compute Diagonal Degree Matrix
        D_    = diag(sum(K,2)); 
%         d = sum(K, 2)';
        
        % Approximates the Laplace-Beltrami Operator
        alpha = 1;
        
        % Define the new kernel
        L_alpha = (D_^-alpha) * K * (D_^-alpha);
%         L_alpha = K ./ ((d * d') .^ alpha);
        
        % Graph Laplacian Normalization to new kernel
        D_alpha  = diag((sum(L_alpha,2)));
        M_alpha  = D_alpha^(-1) * L_alpha;
        
        % Markov-chain matrix no longer symmetric but preserves positivity and
        % stochasticity property (sum_rows = 1)        
        % Diffuse the process
        M_t  = M_alpha^t;        
end

rank_L = rank(L_alpha)
rank_P = rank(M_t)

if show_plots
    title_str = strcat('Laplacian Matrix of ',{' '}, distance_name);
    plotSimilarityConfMatrix(L_alpha, title_str);
    
    title_str = strcat('t-Transition Matrix of ',{' '}, distance_name);
    plotSimilarityConfMatrix(M_t, title_str);
end

% Perform economy-size SVD
disp('Perform eigendecomposition...');
[U, L, V] = svd(M_t, 0);
d_Mt    = diag(real(L));

% Eigen-vector Normalization step
U = bsxfun(@rdivide, U, U(:,1));
ell = size(M_t, 1);
no_dims = ell-1;
x_emb = U(:,2:no_dims + 1)';


%%%%%% Computing the reduced embedding %%%%%%
% Vectorize eigenvalues
d_Mt    = diag(real(L));


% Check that there is only 1 eigenvalue == 1
% if sum(d_Mt > 0.95) > 1
%     error('More than 1 eigenvalue ~ 1, epsilon might be too small (i.e. l_sensitivity too large)!!')
% end

% switch markov_constr
%     case 1
        % Find the optimal number of dimensions using the power heuristic
        pow = pow_eigen;
        [~, D_pow] = eig(mpower(M_t,pow));
        D_sort_pow    = real(diag(sort(diag(D_pow),'descend'))); % make diagonal matrix out of sorted diagonal values of input D
        d_Mt_pow = diag(D_sort_pow);
        [~, opt_ids_der]  = ml_curve_opt(d_Mt_pow','derivatives');
        [~, opt_ids_line] = ml_curve_opt(d_Mt_pow','line');
        k_options = sort([opt_ids_der opt_ids_line],'ascend');
        k_dim = k_options(1)
        
%     case 2
        % Find the optimal number of dimensions by curve heuristic
        [~, opt_ids_der]  = ml_curve_opt(d_Mt','derivatives');
        [~, opt_ids_line] = ml_curve_opt(d_Mt','line');
        k_options = sort([opt_ids_der opt_ids_line],'ascend');
        k_dim_ = k_options(1)
%         d_Mt_pow = d_Mt;
% end
eig_one = sum(d_Mt > 0.95)
x_emb_apprx = U(:,eig_one+1:k_dim+1)';
% x_emb_apprx = U(:,2:k_dim + 1)';


% Plot eigenvalues of matrices
if show_plots
    figure('Color',[1 1 1]);
    subplot(1,2,1);
    plot(d_Mt,'-*r'); hold on
    grid on
    xlabel('Eigenvalue Index $\lambda_i$','Interpreter','Latex','FontSize',14)
    ylabel('Eigenvalues of $\mathcal{M}^t$','Interpreter','Latex','FontSize',14)
    tit = strcat('Eigenvalues of Markov Transition Matrix $\mathcal{M}^t$');
    title(tit, 'Interpreter','Latex','FontSize',14)
    
    subplot(1,2,2);
    plot(d_Mt_pow,'-*r'); hold on
    grid on
    xlabel('Eigenvalue Index $\lambda_i$','Interpreter','Latex','FontSize',14)
    ylabel('Eigenvalues of ${\mathcal{M}^{t}}^{(N)}$','Interpreter','Latex','FontSize',14)
    tit = strcat('Eigenvalues of Powered Markov Transition Matrix $\mathcal{M}^t$');
    title(tit, 'Interpreter','Latex','FontSize',14)
end


end
