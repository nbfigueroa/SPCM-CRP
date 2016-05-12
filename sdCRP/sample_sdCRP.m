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
[~, N] = size(Y);

%%% Extracting current markov state %%%
C              = Psi.C;
Z_C            = Psi.Z_C;
clust_members  = Psi.clust_members;
clust_params   = Psi.clust_params;
clust_logLiks  = Psi.clust_logLiks;
K              = max(Z_C);

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
        new_customer_cycle = get_Connections(C,c_i); %% CHANGE THIS FUNCTION                    
        
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
        clust_logPriors = log(clust_priors)';
                        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%% Compute the conditional distribution %%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                
        % Current table assignment              
        table_id_curr = Z_C(c_i);        
        % Current clusters/tables
        table_ids = unique(Z_C);
        tables = length(table_ids);
                                    
        %%% Compute log-likelihood of clusters given data %%% 
        new_logLiks = zeros(tables,1);
        sum_logLiks = zeros(tables,1);        
        old_logLiks  = clust_logLiks(table_ids); 
        %%%%%%%%% REWRITE THESE LINES ------>>
        for j = 1:tables
            k = table_ids(j);
            if table_id_curr==k
                sum_logLiks(j)  =  sum(old_logLiks);
            else
                others = true(size(table_ids));
                others([j find(table_ids==table_id_curr)]) = false;
                 
                hyper = clust_params(j);
                new_logLiks(j) = cluster_logLik(Y(:,Z_C==k|Z_C==table_id_curr),hyper.a0,hyper.b0,hyper.mu0,hyper.kappa0);
                sum_logLiks(j) = sum(old_logLiks(others)) + new_logLiks(j);                
            end
        end
                                
        %%% Compute log-likelihood of data point i and its connectors %%%
        data_logLik = zeros(N,1);
        for ii = 1:N
            data_logLik(ii) = sum_logLiks(table_ids==Z_C(ii));
        end        
        data_logLik = data_logLik - max(data_logLik);
        
        %%% Compute log cond prob of all possible cluster assignments %%%
        log_cond_prob = clust_logPriors + data_logLik;
        
        %%% Compute conditional distribution %%%          
        % convert to probability sans log
        cond_prob = exp(log_cond_prob);
        % normalize
        cond_prob = cond_prob./sum(cond_prob);
        % generate distribution
        cond_prob = cumsum(cond_prob);
                
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%% Sample from the distribution %%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% Pick new cluster assignment proportional to conditional probability
        c_i_sample = find(cond_prob > rand , 1 );
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%% Adjust customer seating, table assignments and LogLiks %%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % If new cluster assignment is part of the new connected customers
        % (new created cycle) then move these customers to that table
        % This happens if a customer joins a pre-existing table, leaving
        % their current table and thus reducing table numbers   
        
        %%%%%%%%% REWRITE THESE LINES ------>>
        customers_in_new_cycle = false(N,1);
        for n = 1:N
            if ~any(new_customer_cycle==n)
                customers_in_new_cycle(n) = true;
            end
        end
        if customers_in_new_cycle(c_i_sample)  
            
            %%% Table id for sampled cluster assign %%%
            table_id_sample  = Z_C(c_i_sample);
                        
            %%% Update Cluster Members %%%
            clust_swap = minmax([table_id_curr table_id_sample]);          
            new_clust_members = [clust_members{[table_id_curr table_id_sample]}];
            clust_members{clust_swap(1)} = new_clust_members;
            clust_members(clust_swap(2)) = [];
            
            %%%%%%%%% REWRITE THESE LINES ------>>
            %%% Update Table Ids %%%
            Z_C(Z_C==clust_swap(2)) = clust_swap(1);
            Z_C(Z_C>clust_swap(2)) = Z_C(Z_C>clust_swap(2))-1;
            
            %%% Update Cluster logLiks %%%
            clust_logLiks(clust_swap(1))       = new_logLiks(table_ids == Z_C(c_i_sample));
            clust_logLiks(clust_swap(2):(K-1)) = clust_logLiks((clust_swap(2)+1):K);
            
            %%% Reduce Number of Tables %%%
            K = K - 1;
        end
        
        %%%%% Update cluster assignment %%%%%
        C(c_i) = c_i_sample;
end
clust_logLiks = clust_logLiks(1:K);
