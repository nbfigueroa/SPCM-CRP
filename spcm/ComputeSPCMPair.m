function [f_sim, spcm, mean_fact, dir] = ComputeSPCMPair(Sigma_i,Sigma_j,tau, dis_type)
       
        dim = size(Sigma_i,1);
        
        [Vi, Di] = eig(Sigma_i);
        [Vj, Dj] = eig(Sigma_j);
        
        %Ensure eigenvalues are sorted in ascending order
        [Vi, Di] = sortem(Vi,Di);
        [Vj, Dj] = sortem(Vj,Dj);
        
        %Structural of Spectral Polytope
        Xi = Vi*Di^1/2;
        Xj = Vj*Dj^1/2;
                
        %Norms of Spectral Polytope Vectors
        for k=1:length(Dj)
            eig_i(k,1) = norm(Xi(:,k));
            eig_j(k,1) = norm(Xj(:,k));
        end   

        %Homothetic factors and means
        hom_fact_ij = eig_i./eig_j;
        mean_ij = mean(hom_fact_ij);
        
        hom_fact_ji = eig_j./eig_i;               
        mean_ji = mean(hom_fact_ji);
        
        
        %Force Magnification and set directionality
        if (mean_ji >= mean_ij)
            dir = 1;
            hom_fact = hom_fact_ji;
        else
            dir = -1;
            hom_fact = hom_fact_ij;
        end          
                        
        % Homothetic mean factor 
        mean_fact = mean(hom_fact);
        switch dis_type
            case 1
                % Pure similarity value, using the variance
                spcm_val = var(hom_fact);
                
                % Spectral similarity value  (Eq.8 from [1])  +++ function way +++
                delta_ij = mean_ij - mean_ji;
                H = heavy(delta_ij);
                spcm = H*spcm_val + (1-H)*spcm_val;
                
                % Scaling function (Eq.9 from [1])
                upsilon = 10^(tau*exp(-dim));
                
                % B-SPCM f(delta_ij,tau) =
                % 1/( 1 + s(Sigma_i,Sigma_j)*upsilon(tau,dim))
                f_sim = 1/(1+spcm*upsilon);

            
            case 2
                % Pure dis-similarity value, using the coeff of variation
                spcm_val = getCV(hom_fact);

                % Spectral similarity value  (Eq.8 from [1])  +++ function way +++
                delta_ij = mean_ij - mean_ji;
                H = heavy(delta_ij);
                spcm = H*spcm_val + (1-H)*spcm_val;
                
                % B-SPCM f(delta_ij,tau) =
                % 1/( 1 + s(Sigma_i,Sigma_j)*upsilon(tau,dim))
                f_sim = exp(-(dim/2)*spcm);    

            case 3
                % Pure dis-similarity value, using the coeff of variation
                % unbiased estimator
                spcm_val = (1+1/(4*dim))*getCV(hom_fact);

                % Spectral similarity value  (Eq.8 from [1])  +++ function way +++
                delta_ij = mean_ij - mean_ji;
                H = heavy(delta_ij);
                spcm = H*spcm_val + (1-H)*spcm_val;
                
                % B-SPCM f(delta_ij,tau) =
                % 1/( 1 + s(Sigma_i,Sigma_j)*upsilon(tau,dim))
                f_sim = exp(-(dim/2)*spcm);    

            
        end

        
end

function [H] = heavy(x)
H = 1/2 * (1 + sign(x));
end
