function [ handle ] = plotSamplerStats( Psi_Stats, options )
%UNTITLED10 Summary of this function goes here
%   Detailed explanation goes here

handle = figure('Color', [1 1 1]);

% Set Default Variables
dataset       = 'Test';
true_labels   = [];

% Parse Options
if nargin > 1
    if isfield(options, 'dataset'); dataset  = options.dataset ; end
    if isfield(options, 'true_labels'); true_labels = options.true_labels ; end
    if isfield(options, 'Psi'); Psi = options.Psi ;end
    
    subplot(2,1,1)
    semilogx(1:length(Psi_Stats.JointLogProbs),Psi_Stats.JointLogProbs,'r-+', 'LineWidth',2);
    xlabel('Gibbs Iteration'); ylabel('Posterior LogPr p(C|Y,S)')
    box on;
    grid on;
    if strcmp(type,'diag')
        title(sprintf('Sampling results on %s Dataset with Hypers: \\alpha=%1.2f, \\mu_0=%1.2f, \\kappa_0=%1.2f, \\Lambda_0=%1.2f, \\nu_0=%1.2f', ...
            dataset, alpha, lambda.mu_0, lambda.kappa_0, lambda.beta_0, lambda.alpha_0))
    end
    
    subplot(2,1,2)
    stairs(Psi_Stats.TotalClust, 'LineWidth',2);
    set(gca, 'XScale', 'log')
    xlabel('Gibbs Iteration'); ylabel('\Psi = Estimated K');
    if ~isempty(true_labels)
        hold on;
        plot(1:length(Psi_Stats.TotalClust),length(unique(true_labels))*(ones(1,length(Psi_Stats.TotalClust))),'k-', 'LineWidth',2)
        legend('Estimated Clusters', 'True Clusters')
    end
    box on;
    grid on;
    
end
