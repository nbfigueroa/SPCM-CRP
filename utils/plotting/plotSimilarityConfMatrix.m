function [handle] = plotSimilarityConfMatrix(S, title_str)

handle = figure('Color',[1 1 1]);
imagesc(S)
title(title_str,'Fontsize',14)
colormap(pink)
colorbar 
axis square

end