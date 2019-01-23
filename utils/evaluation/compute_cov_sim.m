function [D] = compute_cov_sim( Sigmas, type )
% This function computes the similiarity value of N covariance matrices
% Input:    Sigmas (1xN or Nx1 cell containing Covariance Matrices)
%           type   (Type of similarity function to compute, default:
%           'KLDM')
%               'RIEM': Affine Invariant Riemannian Metric
%               'LERM': Log-Euclidean Riemannian Metric
%               'KLDM': Kullback-Liebler Divergence Metric
%               'JBLD': Jensen-Bregman LogDet Divergence
% Output:   S (NxN dimensional similarity matrix)

% Example: blah blah

% Author: Nadia Figueroa, PhD Student., Robotics
% Learning Algorithms and Systems Lab, EPFL (Switzerland)
% Email address: nadia.figueroafernandez@epfl.ch  
% Website: http://lasa.epfl.ch
% July 2016; Last revision: January 2019
%%
if isempty(type)
    type = 'KLDM';
end
N = length(Sigmas);
D = zeros (N,N);
n = size(Sigmas{1},1);

fprintf('Computing %s Distances for %dx%d Covariance Matrices of %dx%d dimensions...\n',type,N,N,n,n);
tic;

% Upper triangle of symmetric matrix
for i=1:N
    for j=i:N
        
        X = Sigmas{i};
        Y = Sigmas{j};
                
        switch type
            
            case 'EUCL'
                % Euclidean (Frobenius) Metric
                D(i,j) = norm(X - Y,'fro');
                        
            case 'CHOL'
                % Cholesky-Euclidean (Frobenius) Metric
                D(i,j) = norm(chol(X) - chol(Y),'fro');                
                
            case 'RIEM'
                % Affine Invariant Riemannian Metric
                D(i,j) = norm(logm(X^(-1/2)*Y*X^(-1/2)),'fro');
            
            case 'LERM'
                % Log-Euclidean Riemannian Metric
                D(i,j) = norm(logm(X) - logm(Y),'fro');
            
            case 'KLDM'
                % Kullback-Liebler Divergence Metric (KDLM)
                D(i,j) = 1/2*sqrt(trace(X^-1*Y + Y^-1*X - 2*eye(size(X))));                
            
            case 'JBLD'
                % Jensen-Bregman LogDet Divergence
                D(i,j) = sqrt(logm(det((X+Y)/2)) - 1/2*logm(det(X*Y)));
            
            case 'SROT'
               % Minimum Scale-Rotation Curve Distance
               [dist, ~]=MSRcurve(X,Y);
               D(i,j) = dist;
        end
       
    end
end

% Lower triangular
D = D + D';

% % Diagonal -- should be 0 for all distances
% for i=1:n
%         X = Sigmas{i};
%         Y = Sigmas{i};
%                 
%         switch type
%             case 'RIEM'
%                 % Affine Invariant Riemannian Metric
%                 D(i,i) = (norm(logm(X^(-1/2)*Y*X^(-1/2)),'fro')^2);
%             
%             case 'LERM'
%                 % Log-Euclidean Riemannian Metric
%                 D(i,i) = (norm(logm(X) - logm(Y),'fro')^2);
%             
%             case 'KLDM'
%                 % Kullback-Liebler Divergence Metric (KDLM)
%                 D(i,i) = 1/2*sqrt(trace(X^-1*Y + Y^-1*X - 2*eye(size(X))));                
%             
%             case 'JBLD'
%                 % Jensen-Bregman LogDet Divergence
%                 D(i,i) = sqrt(logm(det((X+Y)/2)) - 1/2*logm(det(X*Y)));
%         end
% end
toc;
fprintf('*************************************************************\n');


end

