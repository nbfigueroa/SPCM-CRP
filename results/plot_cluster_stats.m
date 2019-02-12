function [figure1] = plot_cluster_stats(Purity_stats, NMI_stats, ARI_stats, F2_stats, K_stats, dataset_name) 
% Plot Stats for Purity Metric
figure1 = figure('Color',[1 1 1]);

C_f = [
    1 0.85 0.85;
    1 0.65 0.65;
    1 0.45 0.45;
    0.85 0.85 1];
C = [
    1 0.75 0.75;
    1 0.5 0.5;
    1 0.25 0.25;
    0.75 0.75 1];
names_ = {'Tangent','Hilbert','Deformed','Graph'};

%%%%%%%% Plot Purity Stats %%%%%%%%
% Create axes
axes1 = axes('Parent',figure1,...
    'Position',[0.0298040251843457 0.11 0.215362300210497 0.815]);
hold(axes1,'on');

Y = [Purity_stats(1,:,1);
     Purity_stats(1,:,2);
     Purity_stats(1,:,3);
     Purity_stats(1,:,4)];
E = [Purity_stats(2,:,1);
     Purity_stats(2,:,2);
     Purity_stats(2,:,3);
     Purity_stats(2,:,4)];
[HB] = superbar(Y, 'E', E, 'BarFaceColor', permute(C_f,[3 1 2]), 'BarEdgeColor',  permute(C, [3 1 2]));
% Create ylabel
ylabel('Purity','FontAngle','italic','FontSize',14,...
    'FontName','L M Roman10',...
    'Interpreter','latex');
% Create xlabel
xlabel('Vector Space Embedding','FontAngle','italic','FontSize',14,...
    'FontName','L M Roman10',...
    'Interpreter','latex');
xlim(axes1,[0.5 4.5]);
ylim(axes1,[0 1.1]);
box(axes1,'on');
grid(axes1,'on');
% Set the remaining axes properties
set(axes1,'FontAngle','italic','FontName','L M Roman10','XTick',[1 2 3 4],...
    'XTickLabel',names_);

%%%%%%%% Plot NMI Stats %%%%%%%%
% Create axes
axes2 = axes('Parent',figure1,...
    'Position',[0.272419500172427 0.11 0.221135816691161 0.815]);
hold(axes2,'on');

Y = [ARI_stats(1,:,1);
     ARI_stats(1,:,2);
     ARI_stats(1,:,3);
     ARI_stats(1,:,4)];
E = [ARI_stats(2,:,1);
     ARI_stats(2,:,2);
     ARI_stats(2,:,3);
     ARI_stats(2,:,4)];
[HB] = superbar(Y, 'E', E, 'BarFaceColor', permute(C_f,[3 1 2]), 'BarEdgeColor',  permute(C, [3 1 2]));
% Create ylabel
ylabel('ARI Score','FontAngle','italic','FontSize',14,...
    'FontName','L M Roman10',...
    'Interpreter','latex');
% Create xlabel
xlabel('Vector Space Embedding','FontAngle','italic','FontSize',14,...
    'FontName','L M Roman10',...
    'Interpreter','latex');
xlim(axes2,[0.5 4.5]);
ylim(axes2,[0 1.1]);
box(axes2,'on');
grid(axes2,'on');
% Set the remaining axes properties
set(axes2,'FontAngle','italic','FontName','L M Roman10','XTick',[1 2 3 4],...
    'XTickLabel',names_);

%%%%%%%% Plot ARI Stats %%%%%%%%
% Create axes
axes3 = axes('Parent',figure1,...
    'Position',[0.521097237034544 0.11 0.220578380580923 0.815]);
hold(axes3,'on');

Y = [F2_stats(1,:,1);
     F2_stats(1,:,2);
     F2_stats(1,:,3);
     F2_stats(1,:,4)];
E = [F2_stats(2,:,1);
     F2_stats(2,:,2);
     F2_stats(2,:,3);
     F2_stats(2,:,4)];
[HB] = superbar(Y, 'E', E, 'BarFaceColor', permute(C_f,[3 1 2]), 'BarEdgeColor',  permute(C, [3 1 2]));

% Create ylabel
ylabel('$F_\beta$ Score ($\beta=2$)','FontAngle','italic','FontSize',14,...
    'FontName','L M Roman10',...
    'Interpreter','latex');
% Create xlabel
xlabel('Vector Space Embedding','FontAngle','italic','FontSize',14,...
    'FontName','L M Roman10',...
    'Interpreter','latex');
xlim(axes3,[0.5 4.5]);
ylim(axes3,[0 1.1]);
box(axes3,'on');
grid(axes3,'on');
% Set the remaining axes properties
set(axes3,'FontAngle','italic','FontName','L M Roman10','XTick',[1 2 3 4],...
    'XTickLabel',names_);


%%%%%%%% Plot F2 Stats %%%%%%%%
% Create axes
axes4 = axes('Parent',figure1,...
    'Position',[0.774317673597671 0.11 0.218163529409848 0.815]);
hold(axes4,'on');

Y = [K_stats(1,:,1);
     K_stats(1,:,2);
     K_stats(1,:,3);
     K_stats(1,:,4)];
E = [K_stats(2,:,1);
     K_stats(2,:,2);
     K_stats(2,:,3);
     K_stats(2,:,4)];
[HB] = superbar(Y, 'E', E, 'BarFaceColor', permute(C_f,[3 1 2]), 'BarEdgeColor',  permute(C, [3 1 2]));
% Create ylabel
ylabel('Estimated Clusters $K$','FontAngle','italic','FontSize',14,...
    'FontName','L M Roman10',...
    'Interpreter','latex');
% Create xlabel
xlabel('Vector Space Embedding','FontAngle','italic','FontSize',14,...
    'FontName','L M Roman10',...
    'Interpreter','latex');
xlim(axes4,[0.5 4.5]);
% ylim(axes4,[0 1.1]);
box(axes4,'on');
grid(axes4,'on');
% Set the remaining axes properties
set(axes4,'FontAngle','italic','FontName','L M Roman10','XTick',[1 2 3 4],...
    'XTickLabel',names_);

%%%%%%%% Overall Plot Commands %%%%%%%%
legend(HB(1,:),{'GMM+BIC','CRP-GMM','SPCM-CRP-GMM','GMM-Oracle'},'Interpreter','LaTex', 'FontSize',12)
plot_title = strcat('Clustering Performance Metrics on Dataset: ',dataset_name);
axes_ = axes('Parent',figure1,'Tag','suptitle','Position',[0 1 1 1]);
axis off
text('Parent',axes_,'HorizontalAlignment','center','FontSize',22,...
    'Interpreter','latex',...
    'String',plot_title,...
    'Position',[0.512227538543329 -0.0370129870129872 0],...
    'Visible','on');

end