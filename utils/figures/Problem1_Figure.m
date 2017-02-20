%% Load 2D dataset for testing GMM-EM & Likelihood
clear all; close all; clc;
load('probl1_gmm.mat')

%% Visualize Covariance Dataset
title_name  = 'INPUT: Dataset $\Theta$ of Covariance Matrices';
plot_labels = {'$x_1$','$x_2$'};
font_size   = 20;

figure('Color',[1 1 1])

colors     = hsv(5);

for i=1:length(gmm.Priors)
    hold on
    plotGMM(gmm.Mu(:,i), gmm.Sigma(:,:,i), colors(i,:),1);   
    alpha(.3)
end
ml_plot_centroid(gmm.Mu',colors);hold on; 
ml_plot_gmm_contour(gca,gmm.Priors,gmm.Mu,gmm.Sigma,colors,1);


xlabel(plot_labels{1},'Interpreter','Latex', 'FontSize',font_size,'FontName','Times', 'FontWeight','Light');            
ylabel(plot_labels{2},'Interpreter','Latex','FontSize',font_size,'FontName','Times', 'FontWeight','Light');
% legend({'$\theta_1$','$\theta_2$','$\theta_3$','$\theta_4$','$\theta_5$'},'Interpreter','Latex','FontSize',font_size)
title (title_name,'Interpreter','Latex','FontSize',font_size,'FontName','Times', 'FontWeight','Light');          
grid on; box on;


%% Visualize GMM pdf from learnt parameters
ml_plot_gmm_pdf([], Priors, Mu, Sigma)
