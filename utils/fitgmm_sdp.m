function [Priors, Mu, Sigma] = fitgmm_sdp(Xi_ref, est_options)
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
% "A Physically-Consistent Bayesian Non-Parametric Mixture Model for      %
%   Dynamical System Learning."; N. Figueroa and A. Billard; CoRL 2018    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Parse Options
est_type         = est_options.type;
max_gaussians    = est_options.maxK;
do_plots         = est_options.do_plots;
[M,N]            = size(Xi_ref);

if isempty(est_options.fixed_K)
    fixed_K        = 0;
else
    fixed_K = est_options.fixed_K;
end

if ~isempty(est_options.sub_sample)
    sub_sample       = est_options.sub_sample;
    Xi_ref     = Xi_ref(:,1:sub_sample:end);
end

if est_type ~= 1    
    if isempty(est_options.samplerIter)
        if est_type == 0
            samplerIter = 20;
        end
        if est_type == 2
            samplerIter = 200;
        end
    else
        samplerIter = est_options.samplerIter;
    end
end
switch est_type
               
    case 1
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%% Option 1: Cluster SDP matrices with GMM-EM + BIC Model Selection %%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        em_type = 'nadia';
        if fixed_K == 0
            repetitions = 10;
            [bic_scores, k] = fit_gmm_bic(Xi_ref, max_gaussians, repetitions, em_type, do_plots);
        else
            k = fixed_K;
        end
        
        switch em_type
            case 'matlab'
                % Train GMM with Optimal k
                warning('off', 'all'); % there are a lot of really annoying warnings when fitting GMMs
                %fit a GMM to our data
                GMM_full = fitgmdist([Xi_ref]', k, 'Start', 'plus', 'CovarianceType','full', 'Regularize', .000001, 'Replicates', 10);
                warning('on', 'all');
                
                % Extract Model Parameters
                Priors = GMM_full.ComponentProportion;
                Mu = transpose(GMM_full.mu);
                Sigma = GMM_full.Sigma;
                
            case 'nadia'
                
                cov_type = 'full';  Max_iter = 500;
                [Priors0, Mu0, ~, Sigma0] = my_gmmInit(Xi_ref, k, cov_type);
                [Priors, Mu, Sigma, ~]    = my_gmmEM(Xi_ref, k, cov_type, Priors0, Mu0, Sigma0, Max_iter);
                
        end


    case 2
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%% Option3: Cluster Trajectories with Chinese Restaurant Process MM sampler (CRP-GMM) %%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % CRP-GMM (Frank-Wood's implementation) -- faster (does not mix
        % well sometimes)
        do_fw = 1;
        if do_fw
            [class_id, mean_record, covariance_record, K_record, lP_record, alpha_record] = sampler(Xi_ref, samplerIter);
            [val , Maxiter]  = max(lP_record);
            est_labels       = class_id(:,Maxiter);
            % Visualization and plotting options
            if do_plots
                figure('Color',[1 1 1])
                subplot(2,1,1)
                semilogx(1:samplerIter, lP_record'); hold on;
                semilogx(Maxiter,lP_record(Maxiter),'ko','MarkerSize',10);
                grid on
                xlabel('Gibbs Iteration','Interpreter','LaTex','Fontsize',20); ylabel('LogPr','Interpreter','LaTex','Fontsize',20)
                xlim([1 samplerIter])
                legend({'$p(Z|Y, \alpha, \lambda)$'},'Interpreter','LaTex','Fontsize',14)
                title(sprintf('CRP-GMM Sampling results, optimal K=%d at iter=%d', length(unique(est_labels)), Maxiter), 'Interpreter','LaTex','Fontsize',20)
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
            Mu    = mean_record {Maxiter};
            Sigma = covariance_record{Maxiter};
            
            % Remove Singleton Clusters
            if any(singletons)
                [~, est_labels] =  my_gmm_cluster(Xi_ref, Priors, Mu, Sigma, 'hard', []);
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
            
            
        else
            % DP-GMM (Mo-Chen's implementation) -- better mixing sometimes, slower (
            tic;
            [est_labels, Theta, w, ll] = mixGaussGb(Xi_ref);
            Priors = w;
            est_K = length(Priors);
            
            toc;
        end


end


end

