function [C, Z_C, clust_members, clust_logLiks] = sample_ddCRPMM(Y, S_alpha, Psi, type)
% Gibbs Sampling of the dd-CRP
% **Inputs** 
%   o Y:          Data points in Spectral Space
%   o S_alpha:    Similarities in Original Space (self-simiarity = alpha)
%   o Psi:        Current Markov Chain State
%
% **Outputs** 
%   o C:              Customer Assignments
%   o Z_C:            Table Assignments
%   o table_members:
%   o clust_logLiks:
%
% Author: Nadia Figueroa, PhD Student., Robotics
% Learning Algorithms and Systems Lab, EPFL (Switzerland)
% Email address: nadia.figueroafernandez@epfl.ch
% Website: http://lasa.epfl.ch
% February 2016; Last revision: 6-Jan-2017


%  Data Dimensionality
% (M = reduced spectral space, N = # of observations)
[~, N] = size(Y);

%%% Extracting current markov state %%%
C              = Psi.C;
Z_C            = Psi.Z_C;
alpha          = Psi.alpha;
lambda         = Psi.lambda;
type           = Psi.type;
clust_members  = Psi.clust_members;
clust_logLiks  = Psi.clust_logLiks;
K              = max(Z_C);

%%% Sample random permutation of observations \tau of [1,..,N] %%%
tau = randperm(N);

%%% For every i-th randomly sampled observation sample a new cluster
%%% assignment c_i
for i=1:N       
        c_i = tau(i);        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%% Step 1: "Remove" the c_i, i.e. the outgoing link of customer i   %%%%%
        %%%%% to do this, set its cluster to c_i, and set its connected customers to c_i                      
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%               
        ci_old = C(c_i);                                 
        old_conn_customers = clust_members{Z_C(c_i)};
                
        %%% new assigned connected customer (assign to itself) %%%
        C(c_i) = c_i;        
        
        %%% new connected customers (table) considering the removal of c_i from C; i.e. C_{-i} %%
        % CHANGE THIS FUNCTION                    
        new_conn_customers = get_Connections(C,c_i);
        
        %%%%% Compute likelihood of C_i and ci=j. %%%%%
        if length(new_conn_customers)~=length(old_conn_customers) 
            % Increase number of tables
            K = K+1;            
            
            % Adding new customer cycle as new table and removing other
            % linked customers
            clust_members{K} = new_conn_customers;            
            idxs = ismember(old_conn_customers,new_conn_customers);
            clust_members{Z_C(c_i)}(idxs) = [];            
            
            % Creating new table
            Z_C(new_conn_customers) = K;
            
            % Likelihood of old table without c_i 
            % (recompute likelihood of customers sitting without c_i)            
            old_table_id = Z_C(ci_old);            
            clust_logLiks(old_table_id) = table_logLik(Y(:,Z_C==old_table_id), lambda, type);           
            
            % Likelihood of new table created by c_i            
            new_table_id = K;            
            % (compute likelihood of new customers sitting with c_i)            
            clust_logLiks(new_table_id) = table_logLik(Y(:,Z_C==new_table_id), lambda, type);  
        end                         
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%% Compute priors p(c_i = j | S, \alpha) %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        assign_priors = S_alpha{c_i};   
        assign_logPriors = log(assign_priors)';
                        
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
        %%%% Eq. 30 Likelihood of Partition!        
        for j = 1:tables
            k = table_ids(j);
            if table_id_curr==k
                sum_logLiks(j)  =  sum(old_logLiks);
            else
                others = true(size(table_ids));
                others([j find(table_ids==table_id_curr)]) = false;
                
                new_logLiks(j) = table_logLik(Y(:,Z_C==k|Z_C==table_id_curr), lambda, type);
                sum_logLiks(j) = sum(old_logLiks(others)) + new_logLiks(j);                
            end
        end
                                
        %%% Compute Cluster LogLikes %%%
        data_logLik = zeros(N,1);
        for ii = 1:N
            data_logLik(ii) = sum_logLiks(table_ids==Z_C(ii));
        end        
        data_logLik = data_logLik - max(data_logLik);
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%% STEP 2: Sample new customer assignment from updated conditional %%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                
        %%% Compute log cond prob of all possible cluster assignments %%%
        % Eq. 31
        log_cond_prob = assign_logPriors + data_logLik;
        
        %%% Compute conditional distribution %%%          
        % convert to probability sans log
        cond_prob = exp(log_cond_prob);
        % normalize
        cond_prob = cond_prob./sum(cond_prob);
        % generate distribution
        cond_prob = cumsum(cond_prob);
                
        %%% Pick new cluster assignment proportional to conditional probability
        c_i_sample = find(cond_prob > rand , 1 );
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%% Adjust customer seating, table assignments and LogLiks %%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % If sampled customer assignment leaves table assignments intact do
        % nothing, otherwise it means that it joined another table!
        % Update table parameters and likelihoods
                
        if ~ismember(c_i_sample, new_conn_customers)
            %%% Table id for sampled cluster assign %%%
            table_id_sample  = Z_C(c_i_sample);
                        
            %%% Update Cluster Members %%%
            clust_swap = minmax([table_id_curr table_id_sample]);          
            new_clust_members = [clust_members{[table_id_curr table_id_sample]}];
            clust_members{clust_swap(1)} = new_clust_members;
            clust_members(clust_swap(2)) = [];
            
            %%%%%%%%% REWRITE THESE LINES ------>>
            %%% Update Table Assignments %%%
            Z_C(Z_C==clust_swap(2)) = clust_swap(1);
            Z_C(Z_C>clust_swap(2)) = Z_C(Z_C>clust_swap(2))-1;
            
            %%% Update Table logLiks %%%
            clust_logLiks(clust_swap(1))       = new_logLiks(table_ids == Z_C(c_i_sample));
            clust_logLiks(clust_swap(2):(K-1)) = clust_logLiks((clust_swap(2)+1):K);
            
            %%% Reduce Number of Tables %%%
            K = K - 1;
        end
        
        %%%%% Update Customer assignment %%%%%
        C(c_i) = c_i_sample;
end
clust_logLiks = clust_logLiks(1:K);
