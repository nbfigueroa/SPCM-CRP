function [spcm] = ComputeSPCMfunctionMatrix(behavs_theta, tau)

spcm = [];    
spcm = zeros(length(behavs_theta),length(behavs_theta),4);

% Inefficient way
N = length(behavs_theta);    % Number of Covariance Matrices
D = size(behavs_theta{1},1); % Dimension of Covariance Matrices
fprintf('Computing SPCM Similarity Function for %dx%d Covariance Matrices of %dx%d dimensions...\n',N,N,D,D);

tic;
for i=1:length(behavs_theta)
    for j=1:length(behavs_theta)
        
        [b_sim s hom_fact dir] = ComputeSPCMPair(behavs_theta{i},behavs_theta{j}, tau);
        
        spcm(i,j,1) = s;
        spcm(i,j,2) = b_sim;
        spcm(i,j,3) = hom_fact;
        spcm(i,j,4) = dir;
        
    end
end
toc;

fprintf('*************************************************************\n');



