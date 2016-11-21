function [handle] = plotSpectralManifold(Y, true_labels, d,thres, s_norm, M)

% Create Figure Handle
handle = figure('Color',[1 1 1]);


% Create Figure Handle
if (M == 2) || (M == 3)
    subplot(2,1,1)
end
plot(s_norm,'-*r'); hold on
plot(thres*ones(1,length(d)),'--k','LineWidth', 2); hold on
xlabel('Eigenvalue Index')
ylabel('Normalized Eigenvalue Softmax')
tit = strcat('Eigenvalue Analysis for Manifold Dimensionality  M = ', num2str(M));
title(tit, 'Fontsize',14)

if (M == 2) || (M == 3)
    subplot(2,1,2)
    % Plot M-Dimensional Points of Spectral Manifold
    idx_label   = true_labels;
    true_clust = length(unique(true_labels));
    if M==2    
        for jj=1:true_clust
            clust_color = [rand rand rand];
            scatter(Y(1,idx_label==jj),Y(2,idx_label==jj), 50, clust_color, 'filled');hold on                      
        end   
        grid on
        title('$\Sigma_i$-s Represented in 2-d Spectral space')
    end

    if M==3
        for jj=1:true_clust
            clust_color = [rand rand rand];
            scatter3(Y(1,idx_label==jj),Y(2,idx_label==jj),Y(3,idx_label==jj), 50, clust_color, 'filled');hold on        
        end
        xlabel('$y_1$');ylabel('$y_2$');zlabel('$y_3$')
        colormap(hot)
        grid on
        title('$\Sigma_i$-s Represented in 3-d Spectral space')
    end
end

end