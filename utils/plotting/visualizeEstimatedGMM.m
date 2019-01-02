function [h_gmm, h_pdf] = visualizeEstimatedGMM(Xi_ref,  Priors, Mu, Sigma, est_labels, est_options)
M = size(Xi_ref,1);
emb_name = est_options.emb_name; 
switch est_options.type
    case -1
        title_string = strcat('GMM-Oracle on ',{' '},emb_name);
    case 0
        title_string = strcat('SPCM-CRP-MM on ',{' '},emb_name);
    case 1
        title_string = strcat('Finite GMM w/BIC Model Selection on ',{' '},emb_name);
    case 2
        title_string = strcat('CRP-GMM on ',  {' '},emb_name);
end


if M == 2
    % Visualize Cluster Parameters Trajectory Data
    [h_gmm] = plotGMMParameters( Xi_ref, est_labels, Mu, Sigma);
    limits = axis;
    title(title_string,'Interpreter','LaTex', 'FontSize',15); 
    
    % Visualize PDF of fitted GMM
    ml_plot_gmm_pdf(Xi_ref, Priors, Mu, Sigma, limits);
    title(title_string,'Interpreter','LaTex', 'FontSize',15); 
    
elseif M == 3
    GMM = [];
    GMM.Priors = Priors; GMM.Mu = Mu; GMM.Sigma = Sigma;
    [h_gmm] = plot3DGMMParameters(Xi_ref, GMM, est_labels);
    title(title_string,'Interpreter','LaTex', 'FontSize',15); 
    view(-120, 30);
    axis equal
    h_pdf = [];
end


end