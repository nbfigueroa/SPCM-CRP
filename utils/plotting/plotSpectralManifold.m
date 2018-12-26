function [handle] = plotSpectralManifold(Y, true_labels, d,thres, s_norm, M)

% Create Figure Handle
handle0 = figure('Color',[1 1 1]);

plot(d,'-*r'); hold on
grid on
xlabel('Eigenvalue Index $\lambda_i$','Interpreter','Latex','FontSize',14)
ylabel('$\mathbf{\lambda}_i$ of $\mathcal{L}$','Interpreter','Latex','FontSize',14)
tit = strcat('Eigenvalues Laplacian ');
title(tit, 'Interpreter','Latex','FontSize',14)

handle = figure('Color',[1 1 1]);
plot(s_norm,'-*r'); hold on
plot(thres*ones(1,length(d)),'--k','LineWidth', 2); hold on
grid on
xlabel('Eigenvalue Index $\lambda_i$','Interpreter','Latex','FontSize',14)
ylabel('$\zeta(\mathbf{\lambda})_i$','Interpreter','Latex','FontSize',14)
tit = strcat('Eigenvalue Analysis for Manifold Dimensionality  M = ', num2str(M));
title(tit, 'Interpreter','Latex','FontSize',14)


if (M == 2) || (M == 3)
    figure('Color',[1 1 1])
    % Plot M-Dimensional Points of Spectral Manifold
    idx_label   = true_labels;
    true_clust = length(unique(true_labels));
    if M==2    
        for jj=1:true_clust
            clust_color = [rand rand rand];
            scatter(Y(1,idx_label==jj),Y(2,idx_label==jj), 100, clust_color, 'filled');hold on                      
        end   
        grid on
        xlabel('$y_1$', 'Interpreter','Latex','FontSize',14);
        ylabel('$y_2$', 'Interpreter','Latex','FontSize',14);        
        title('$\Sigma_i$ Represented in 2-d Spectral space', 'Interpreter','Latex','FontSize',14)
    end

    if M==3
        for jj=1:true_clust
            clust_color = [rand rand rand];
            scatter3(Y(1,idx_label==jj),Y(2,idx_label==jj),Y(3,idx_label==jj), 100, clust_color, 'filled');hold on        
        end
        xlabel('$y_1$', 'Interpreter','Latex','FontSize',14);
        ylabel('$y_2$', 'Interpreter','Latex','FontSize',14);
        zlabel('$y_3$', 'Interpreter','Latex','FontSize',14);
        colormap(hot)
        grid on
        title('$\Sigma_i$ Represented in 3-d Spectral space','Interpreter','Latex','FontSize',14)
    end
    axis equal
elseif M > 3
    
    % Plot result of Laplacian Eigenmaps Projection
    plot_options              = [];
    plot_options.labels       = true_labels;
    plot_options.title        = '$\Sigma_i$ Represented in M-d Spectral space';
    ml_plot_data(Y',plot_options);
    
end

end