function [spcm] = ComputeSPCMfunctionMatrix(Sigmas, tau)


n    = length(Sigmas);
spcm = zeros(n,n,4);

%%%%%%%%%%%%%%%%% Efficient Similarity matrix computation %%%%%%%%%%%%%%%%%
tic;
% Upper triangular
for i=1:n
    for j=i:n
        [b_sim s hom_fact dir] = ComputeSPCMPair(Sigmas{i},Sigmas{j}, tau);
        
        spcm(i,j,1) = s;
        spcm(i,j,2) = b_sim;
        spcm(i,j,3) = hom_fact;
        spcm(i,j,4) = dir;
        
    end
end

% Lower triangular
for k=1:4
    spcm(:,:,k) = spcm(:,:,k) + spcm(:,:,k)';
end

% Diagonal
for i=1:n
        [b_sim s hom_fact dir] = ComputeSPCMPair(Sigmas{i},Sigmas{i}, tau);
        spcm(i,i,1) = s;
        spcm(i,i,2) = b_sim;
        spcm(i,i,3) = hom_fact;
        spcm(i,i,4) = dir;
end
toc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


end






