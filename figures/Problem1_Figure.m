%% Load 2D dataset for testing GMM-EM & Likelihood
clear all; close all; clc;
load('probl1_gmm.mat')

%% Visualize Covariance Dataset
title_name  = 'Dataset $\Theta$ of Covariance Matrices';
plot_labels = {'$x_1$','$x_2$'};
font_size   = 18;

figure('Color',[1 1 1])
colored = 1;
% Clustered Colors
if colored
    vivid_c = hsv(3);
    % Thin ellipse (diag covariance)
    colors(1,:) = vivid_c(1,:);
    colors(6,:) = vivid_c(1,:);
    % Circle (isometric covariance)
    colors(2,:) = vivid_c(2,:);
    colors(3,:) = vivid_c(2,:);
    % Fat ellipse (Full covariance)
    colors(4,:) = vivid_c(3,:);
    colors(5,:) = vivid_c(3,:);
else
    % Gray Color
    colors = repmat([0.3    0.3    0.3],[6,1]);
end
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
axis equal
grid on; box on;


%% Visualize GMM pdf from learnt parameters
ml_plot_gmm_pdf([], Priors, Mu, Sigma)
