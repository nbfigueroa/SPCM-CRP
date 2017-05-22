function [ handle ] = plotSamplerStats( Psi_Stats, options )
%UNTITLED10 Summary of this function goes here
%   Detailed explanation goes here

handle = figure('Color', [1 1 1]);

% Set Default Variables
dataset       = 'Test';
true_labels   = [];
Iterations    = length(Psi_Stats.LogLiks);

% Parse Options
if nargin > 1
    if isfield(options, 'dataset'); dataset  = options.dataset ; end
    if isfield(options, 'true_labels'); true_labels = options.true_labels ; end
    if isfield(options, 'Psi'); Psi = options.Psi ;end
    
    subplot(2,1,1)
    semilogx(1:length(Psi_Stats.PostLogProbs),Psi_Stats.PostLogProbs,'r-*', 'LineWidth',2); hold on;
    semilogx(1:length(Psi_Stats.LogLiks),Psi_Stats.LogLiks,'b-*', 'LineWidth',2);  hold on;
    semilogx(Psi.Maxiter,Psi_Stats.PostLogProbs(Psi.Maxiter),'ko','MarkerSize',10);
    xlim([1 Iterations])
    xlabel('Gibbs Iteration','Interpreter','LaTex'); ylabel('LogPr','Interpreter','LaTex')
    legend({'$p(C|Y,S, \alpha, \lambda)$','$p(Y|\mathbf{Z}(C),\lambda)$'},'Interpreter','LaTex')
    
    box on;
    grid on;   
    title(sprintf('Sampling results on %s Dataset, optimal K=%d at iter=%d',dataset, Psi_Stats.TotalClust(Psi.Maxiter),Psi.Maxiter), 'Interpreter','LaTex')
    
    subplot(2,1,2)
    stairs(Psi_Stats.TotalClust, 'LineWidth',2);
    set(gca, 'XScale', 'log')
    xlim([1 Iterations])
    xlabel('Gibbs Iteration','Interpreter','LaTex'); ylabel('$\Psi$ = Estimated K','Interpreter','LaTex');
    if ~isempty(true_labels)
        hold on;
        plot(1:length(Psi_Stats.TotalClust),length(unique(true_labels))*(ones(1,length(Psi_Stats.TotalClust))),'k-', 'LineWidth',2)
        legend('Estimated Clusters', 'True Clusters')
    end
    box on;
    grid on;
    
end
