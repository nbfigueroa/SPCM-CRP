function X_bar = intrinsicMean_mat(X, tol, max_iter)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute the intrinsic (Karcher) mean of X
%
%  --- Inputs ---
% o X         cell structure containing the SPD matrices
% o tol       error tolerance to stop computation
% o max_iter  maximum # of iterations
%
%  --- Output ---
%
% o X_bar     Karcher (instrinsic) mean of SPD matrices
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Auxiliary Variables
N     = length(X);    % number of data-points (SPD matrices)
d_mat = length(X{1}); % dimensionality of spd matrices

% Initial Guess for Mean
X_bar =  X{randsample(N,1)};

% Gradient descent/iterative scheme to compute the Karcher Mean
% of a set of point on the Riemannian Manifold
i = 0; dist2mean_old = inf;
while i < max_iter
        
    % Compute the inner matrix for the gradient descent step
    X_sums   = zeros(d_mat,d_mat);
    for n=1:N
        X_sums =  X_sums +  logm( (X_bar^(-1/2)) * X{n} * (X_bar^(-1/2)) );
    end    
    X_sums = 1/N * X_sums;
    
    % Gradient descent step from "Riemannian Tensor Computing" Eq.3
    X_bar_ = X_bar;
    X_bar = (X_bar_^(1/2)) * expm(X_sums) * (X_bar_^(1/2)); 
    
    % Compute Sum of distances to mean; i.e. objective function
    dist2mean = 0;
    for n=1:N
%         dist2mean =  dist2mean +  (norm(logm(X{n}) - logm(X_bar),'fro')^2);
        dist2mean =  dist2mean + (norm(logm(X_bar^(-1/2)*X{n}*X_bar^(-1/2)),'fro')^2);
    end        
    fprintf('Sum-of-squared Distances to mean (iteration %d): %f\n', i, dist2mean);
    
    if dist2mean_old-dist2mean < 1e-6
        break;
    end
    dist2mean_old = dist2mean;
    i = i + 1;    
end
