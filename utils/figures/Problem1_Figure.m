%% Load 2D dataset for testing GMM-EM & Likelihood
clear all; close all; clc;

load('probl1_gmm.mat')

% Visualize Covariance Dataset
title_name  = 'INPUT: Dataset $\Theta$ of Covariance Matrices';
plot_labels = {'$x_1$','$x_2$'};
font_size   = 20;

figure('Color',[1 1 1])

colors     = hsv(4);
ml_plot_centroid(gmm.Mu',colors);hold on; 
ml_plot_gmm_contour(gca,gmm.Priors,gmm.Mu,gmm.Sigma,colors);

xlabel(plot_labels{1},'Interpreter','Latex', 'FontSize',font_size,'FontName','Times', 'FontWeight','Light');            
ylabel(plot_labels{2},'Interpreter','Latex','FontSize',font_size,'FontName','Times', 'FontWeight','Light');
title (title_name,'Interpreter','Latex','FontSize',font_size,'FontName','Times', 'FontWeight','Light');          
grid on; box on;


%% Visualize GMM pdf from learnt parameters
ml_plot_gmm_pdf([], Priors, Mu, Sigma)
