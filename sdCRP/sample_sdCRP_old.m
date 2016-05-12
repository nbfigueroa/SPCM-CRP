function [C, Z_C, clust_members, clust_params, clust_logLiks] = sample_sdCRP(Y, delta, Psi)
% Gibbs Sampling of the sd-CRP
% **Inputs** 
%   Y:      Data points in Spectral Space
%   delta:  Similarities in Original Space
%   Psi:    Current Markov Chain State
%
% **Outputs** 
%   C:              new customer assignments
%   Z_C:            new table ids
%   clust_members:
%   clust_params:
%   clust_logLiks:
%
% Author: Nadia Figueroa, PhD Student., Robotics
% Learning Algorithms and Systems Lab, EPFL (Switzerland)
% Email address: nadia.figueroafernandez@epfl.ch
% Website: http://lasa.epfl.ch
% February 2016; Last revision: 29-Feb-2016


%  Data Dimensionality
% (M = reduced spectral space, N = # of observations)
[M, N] = size(Y);

%%% Extracting current markov state %%%
C              = Psi.C;
Z_C            = Psi.Z_C;
clust_members  = Psi.clust_members;
clust_params   = Psi.clust_params;
clust_logLiks  = Psi.clust_logLiks;
K = max(Z_C);

%%% Sample random permutation of observations \tau of [1,..,N] %%%
tau = randperm(N);

%%% For every i-th randomly sampled observation sample a new cluster
%%% assignment c_i

for i=1:N       
        c_i = tau(i);        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%% "remove" the c_i, i.e. the outgoing link of customer i %%%%%
        %%%%% to do this, set its cluster to c_i, and set its connected customers to c_i                      
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        
        old_c_i = C(c_i);                                
        old_customer_cycle = clust_members{Z_C(c_i)};
                
        %%% new assigned connected customer %%%
        C(c_i) = c_i;        
        %%% new connected customers considering the removal of link c_i c_{-i} %%
        new_customer_cycle = sit_Behind(C,c_i); %% CHANGE THIS FUNCTION                    
        
        %%%%% if this removal splits a table update the likelihoods. %%%%%
        % A table seating assignment has changed!
        % Compute log-prob when removing the current c_i
        if length(new_customer_cycle)~=length(old_customer_cycle)            
            
            % Increase number of tables
            K = K+1;            
            
            % Adding new customer cycle as new table and removing other
            % linked customers
            clust_members{K} = new_customer_cycle;            
            idxs = ismember(old_customer_cycle,new_customer_cycle);
            clust_members{Z_C(c_i)}(idxs) = [];            
            
            % Creating new table
            Z_C(new_customer_cycle) = K;
            
            % Likelihood of old table without c_i 
            % (recompute likelihood of customers sitting without c_i)            
            old_table_id = Z_C(old_c_i);            
            hyper = clust_params(old_table_id);
            clust_logLiks(old_table_id) = cluster_logLik(Y(:,Z_C==old_table_id),hyper.a0,hyper.b0,hyper.mu0,hyper.kappa0);
            %%% CHANGE THIS FUNCTION TO normal-inverse Wishart and to take
            %%% updated table parameteres
            
            % Likelihood of new table created by c_i            
            new_table_id = K;            
            % sample a new parameters for this new table
            hyper = clust_params(new_table_id);
            %%%%%%%%%%% INSERT FUNCTION HERE
            clust_params(new_table_id) = hyper;
            % (compute likelihood of new customers sitting with c_i)            
            clust_logLiks(new_table_id) = cluster_logLik(Y(:,Z_C==new_table_id),hyper.a0,hyper.b0,hyper.mu0,hyper.kappa0);                              
        end                             
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%% Compute priors p(c_i = j | \alpha, delta) %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        clust_priors = delta{c_i};   
        clust_priors(1, c_i) = clust_params(c_i).alpha;   
                        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%% Compute the conditional distribution %%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%%%%%%%%%%%%%%%%  IDK WTf this does      
        % Compute Likelihood of Cluster given Data
        customer_ids = 1:N;        
        curr_table = Z_C(c_i);
        
        % Current clusters/tables and LL
        table_ids = unique(Z_C);
        tables = length(table_ids);
        LLs = zeros(tables,3);   
        
        % Log-Likelihoods of data in clusters k
        LLs(:,1) = clust_logLiks(table_ids);       
        
        % Update LL of table assignments
        for j = 1:tables
            k = table_ids(j);
            if curr_table==k
                LLs(j,2) = sum(LLs(:,1));
            else
                others = true(size(table_ids));
                others([j find(table_ids==curr_table)]) = false;
                hyper = clust_params(j);
                LLs(j,3) = cluster_logLik(Y(:,Z_C==k|Z_C==curr_table),hyper.a0,hyper.b0,hyper.mu0,hyper.kappa0);
                LLs(j,2) = sum(LLs(others,1))+LLs(j,3);
            end
        end
        %%%%%%%%%%%%%%%%%
                
       
        new_connected_customers = false(N,1);
        llhood_data = zeros(N,1);
        for n = 1:N
            if ~any(new_customer_cycle==customer_ids(n))
                new_connected_customers(n) = true;
            end
            llhood_data(n) = LLs(table_ids==Z_C(customer_ids(n)),2);
        end
        
        lhood_data = exp(llhood_data-max(llhood_data));
        lhood_data = lhood_data./sum(lhood_data);        
        
        cond_prob = clust_priors(:).*lhood_data;
        cond_prob = cond_prob./sum(cond_prob);
        cond_prob = cumsum(cond_prob);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%% Sample from the distribution %%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % i.e. pick new cluster assignment proportional to conditional probability
        c_i_sample = find(cond_prob > rand , 1, 'first');
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%% Adjust customer seating, table assignments and LogLiks %%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%%%% IDK WTf this does
        % If there is more than 1 customer assigned to a table adjust clustering and logprobs
        if new_connected_customers(c_i_sample)
            % Something
            k1 = Z_C(c_i);k2 = Z_C(customer_ids(c_i_sample));
            
            c_l = min(k1,k2); c_h = max(k1,k2);
            
            %%% Updating table ids
            Z_C(Z_C==c_h) = c_l;
            Z_C(Z_C>c_h) = Z_C(Z_C>c_h)-1;
            
            clust_members{c_l} = [clust_members{[k1 k2]}];
            clust_members(c_h) = [];
            
            clust_logLiks(c_l) = LLs(table_ids==Z_C(c_i_sample),3);
            clust_logLiks(c_h:(K-1)) = clust_logLiks((c_h+1):K);
            K = K - 1;
        end
        %%%%%
        
        %%%%% Update cluster assignment %%%%%
        C(c_i) = customer_ids(c_i_sample);
end
clust_logLiks = clust_logLiks(1:K);
