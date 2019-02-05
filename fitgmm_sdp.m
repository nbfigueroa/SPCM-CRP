function [Priors, Mu, Sigma, est_labels, stats] = fitgmm_sdp(S, Y, est_options)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2018 Learning Algorithms and Systems Laboratory,          %
% EPFL, Switzerland                                                       %
% Author:  Nadia Figueroa                                                 % 
% email:   nadia.figueroafernandez@epfl.ch                                %
% website: http://lasa.epfl.ch                                            %
%                                                                         %
% This work was supported by the EU project Cogimon H2020-ICT-23-2014.    %
%                                                                         %
% Permission is granted to copy, distribute, and/or modify this program   %
% under the terms of the GNU General Public License, version 2 or any     %
% later version published by the Free Software Foundation.                %
%                                                                         %
% This program is distributed in the hope that it will be useful, but     %
% WITHOUT ANY WARRANTY; without even the implied warranty of              %
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General%
% Public License for more details                                         %
%                                                                         %
% If you use this code in your research please cite:                      %
% ""     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Parse Options
est_type         = est_options.type;
max_gaussians    = est_options.maxK;
do_plots         = est_options.do_plots;
[M,N]            = size(Y);

if isempty(est_options.fixed_K)
    fixed_K        = 0;
else
    fixed_K = est_options.fixed_K;
end

if est_type ~= 1    
    if isempty(est_options.samplerIter)
        if est_type == 0
            samplerIter = 20;
        end
        if est_type == 2
            samplerIter = 200;
        end
        dataset_name = 'Test data';
    else
        samplerIter = est_options.samplerIter;
        dataset_name = est_options.dataset_name;
    end
end

if isempty(est_options.true_labels)
    true_labels        = [];
else
    true_labels = est_options.true_labels;
end

switch est_type

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%% Option 0: Cluster SDP matrices with SPCM-CRP-MM  %%%%%%  
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    case 0
        % Setting sampler/model options (i.e. hyper-parameters, alpha, Covariance matrix)
        options                 = [];
        options.type            = 'full';          % Type of Covariance Matrix: 'full' = NIW or 'Diag' = NIG
        options.T               = samplerIter;     % Sampler Iterations
        options.alpha           = 1;               % Concentration parameter [0 - 2]
        
        % Standard Base Distribution Hyper-parameter setting
        if strcmp(options.type,'diag')
            lambda.alpha_0       = M;                    % G(sigma_k^-1|alpha_0,beta_0): (degrees of freedom)
            lambda.beta_0        = sum(diag(cov(Y')))/M; % G(sigma_k^-1|alpha_0,beta_0): (precision)
        end
        if strcmp(options.type,'full')
            lambda.nu_0        = M ;                       % IW(Sigma_k|Lambda_0,nu_0): (degrees of freedom)
%             lambda.Lambda_0    = eye(M)*sum(diag(cov(Y')))/M;  % IW(Sigma_k|Lambda_0,nu_0): (Scale matrix)
            lambda.Lambda_0    = 1/M * diag(diag(cov(Y')));         % IW(Sigma_k|Lambda_0,nu_0): (Scale matrix)

        end
        lambda.mu_0             = mean(Y,2);    % hyper for N(mu_k|mu_0,kappa_0)
        lambda.kappa_0          = 1;            % hyper for N(mu_k|mu_0,kappa_0)
        
        lambda.Lambda_0
        
        % Run Collapsed Gibbs Sampler
        options.lambda    = lambda;
        options.verbose   = 1;
        [Psi, Psi_Stats]  = run_ddCRP_sampler(Y, S, options);
        est_labels        = Psi.Z_C';
        
        %%%%%%%% Visualize Collapsed Gibbs Sampler Stats %%%%%%%%%%%%%%
        if do_plots
            if exist('h1b','var') && isvalid(h1b), delete(h1b);end
            options = [];
            options.dataset      = dataset_name;
            options.true_labels  = true_labels;
            options.Psi          = Psi;
            [ h1b ] = plotSamplerStats( Psi_Stats, options );
        end
        
        %%%%%%%%%% Extract Learned GMM models %%%%%%%%%%%%%
        est_labels        = Psi.Z_C';
        N = size(Y,2);
        unique_labels = unique(est_labels);
        est_K      = length(unique_labels);
        Priors     = zeros(1, est_K);
        singletons = zeros(1, est_K);
        for k=1:est_K
            assigned_k = sum(est_labels==unique_labels(k));
            Priors(k) = assigned_k/N;
            singletons(k) = assigned_k < round(N*0.01);
        end
        Mu     = Psi.Theta.Mu(:,unique_labels);
        Sigma  = Psi.Theta.Sigma(:,:,unique_labels);
        
        if any(singletons)
            singleton_idx = find(singletons == 1);
            Mu(:,singleton_idx) = [];
            Sigma(:,:,singleton_idx) = [];
            unique_labels(singleton_idx) = [];
            Priors  = [];
            est_K = length(Mu);
            for k=1:est_K
                assigned_k = sum(est_labels==unique_labels(k));
                Priors(k) = assigned_k/N;
            end            
        end
        clear stats 
        stats.Psi       = Psi;
        stats.Psi_Stats = Psi_Stats;
        [~, est_labels] =  my_gmm_cluster(Y, Priors, Mu, Sigma, 'hard', []);
        
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%% Option 1: Cluster SDP matrices with GMM-EM + BIC Model Selection %%%%%%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
    case 1        
        
        em_type = 'nadia';
        if fixed_K == 0
            repetitions = 10;
            [bic_scores, k] = fit_gmm_bic(Y, max_gaussians, repetitions, em_type, do_plots);
            stats.bic_scores = bic_scores;
            stats.best_k = k;
        else
            k = fixed_K;
            stats.best_k = k;
        end
        
        switch em_type
            case 'matlab'
                % Train GMM with Optimal k
                warning('off', 'all'); % there are a lot of really annoying warnings when fitting GMMs
                %fit a GMM to our data
                GMM_full = fitgmdist([Y]', k, 'Start', 'plus', 'CovarianceType','full', 'Regularize', .000001, 'Replicates', 10);
                warning('on', 'all');
                
                % Extract Model Parameters
                Priors = GMM_full.ComponentProportion;
                Mu = transpose(GMM_full.mu);
                Sigma = GMM_full.Sigma;
                
            case 'nadia'                
                cov_type = 'full';  Max_iter = 500;
                [Priors0, Mu0, ~, Sigma0] = my_gmmInit(Y, k, cov_type);
                [Priors, Mu, Sigma, ~]    = my_gmmEM(Y, k, cov_type, Priors0, Mu0, Sigma0, Max_iter);                
        end
        [~, est_labels] =  my_gmm_cluster(Y, Priors, Mu, Sigma, 'hard', []);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%% Option 2: Cluster SDP matrices CRP MM sampler (CRP-GMM) %%%%%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    case 2       
        % CRP-GMM (Frank-Wood's implementation) which is a Gibbs Sampler       
            [class_id, mean_record, covariance_record, K_record, lP_record, alpha_record] = sampler(Y, samplerIter);
            [max_val, max_id] = max(lP_record);
            est_K             = K_record(max_id);
            est_labels        = class_id(:,max_id);
            samplerIter       = length(lP_record);
            
            % Gather Stats
            clear stats
            stats.lP_record      = lP_record;
            stats.K_record       = K_record;
            stats.Mu_record      = mean_record;
            stats.Sigma_record   = covariance_record;            
            stats.samplerIter    = samplerIter;
            
            % Visualization and plotting options
            if do_plots
                figure('Color',[1 1 1])
                subplot(2,1,1)
                semilogx(1:samplerIter, lP_record'); hold on;
                semilogx(max_id, lP_record(max_id),'ko','MarkerSize',10);
                grid on
                xlabel('Gibbs Iteration','Interpreter','LaTex','Fontsize',20); ylabel('LogPr','Interpreter','LaTex','Fontsize',20)
                xlim([1 samplerIter])
                legend({'$p(Z|Y, \alpha, \lambda)$'},'Interpreter','LaTex','Fontsize',14)
                title(sprintf('CRP-GMM Sampling results, optimal K=%d at iter=%d', est_K, max_id), 'Interpreter','LaTex','Fontsize',20)
                subplot(2,1,2)
                stairs(K_record, 'LineWidth',2);
                set(gca, 'XScale', 'log')
                xlim([1 samplerIter])
                xlabel('Gibbs Iteration','Interpreter','LaTex','Fontsize',20); ylabel('$\Psi$ = Estimated K','Interpreter','LaTex','Fontsize',20);
            end
            
            % Extract Learnt cluster parameters
            unique_labels = unique(est_labels);
            est_K         = length(unique_labels);
            Priors        = zeros(1, est_K);
            singletons    = zeros(1, est_K);
            for k=1:est_K
                assigned_k = sum(est_labels==unique_labels(k));
                Priors(k) = assigned_k/N;
                singletons(k) = assigned_k < 2;
            end
            Mu    = mean_record {max_id};
            Sigma = covariance_record{max_id};
            
            % Remove Singleton Clusters
            if any(singletons)
                [~, est_labels] =  my_gmm_cluster(Y, Priors, Mu, Sigma, 'hard', []);
                unique_labels = unique(est_labels);
                est_K         = length(unique_labels);
                Mu    = Mu(:,unique_labels);
                Sigma = Sigma(:,:,unique_labels);
                Priors  = [];
                for k=1:est_K
                    assigned_k = sum(est_labels==unique_labels(k));
                    Priors(k) = assigned_k/N;
                end
            end
            [~, est_labels] =  my_gmm_cluster(Y, Priors, Mu, Sigma, 'hard', []);
            
            % CRP-GMM (Mo-Chens's implementation) which is a COLLAPSED Gibbs Sampler
            if est_K == 1
                fprintf(2, 'It seems that the Gibbs Sampler did not converge.. trying Collapsed Gibbs Sampler...\n');                
                [est_labels, Theta, w, ll, k_s] = mixGaussGb(Y, samplerIter);                
                [Priors, Mu, Sigma] = gmmOracle(Y, est_labels);
                [max_val, max_id] = max(ll);
                est_K = length(Priors);
                clear stats
                stats.collapsed.ll = ll;
                stats.collapsed.ll = k_s;
                
                if do_plots
                    figure('Color',[1 1 1])
                    subplot(2,1,1)
                    semilogx(1:samplerIter, ll); hold on;
                    semilogx(max_id, ll(max_id),'ko','MarkerSize',10);
                    grid on;
                    xlabel('Collapsed Gibbs Iteration','Interpreter','LaTex','Fontsize',20); ylabel('LogPr','Interpreter','LaTex','Fontsize',20)
                    xlim([1 samplerIter])
                    legend({'$p(Z|Y, \alpha, \lambda)$'},'Interpreter','LaTex','Fontsize',14)
                    title(sprintf('CRP-GMM Sampling results, optimal K=%d at iter=%d', est_K, max_id), 'Interpreter','LaTex','Fontsize',20)
                    subplot(2,1,2)
                    stairs(K_record, 'LineWidth',2);
                    set(gca, 'XScale', 'log')
                    xlim([1 samplerIter])
                    xlabel('Collapsed Gibbs Iteration','Interpreter','LaTex','Fontsize',20); ylabel('$\Psi$ = Estimated K','Interpreter','LaTex','Fontsize',20);
                end
            end
end


end

