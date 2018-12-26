function [handle] = plotSimilarityConfMatrix(S, title_str)

handle = figure('Color',[1 1 1]);
imagesc(S)
title(title_str,'Interpreter','latex', 'Fontsize',14)


level = 100; n = ceil(level/2);
cmap1 = [linspace(1, 1, n); linspace(0, 1, n); linspace(0, 1, n)]';
cmap2 = [linspace(1, 0, n); linspace(1, 0, n); linspace(1, 1, n)]';
cmap = [cmap1; cmap2(2:end, :)];
colormap(vivid(cmap, [.5, .75]));
% colormap(pink)
% colormap(bone)
% colormap(jet)
colorbar 
axis square

end