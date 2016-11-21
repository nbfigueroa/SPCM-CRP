function [handle] = plotSimilarityConfMatrix(S, title_str)

handle = figure('Color',[1 1 1]);
imagesc(S)
title('Bounded Similarity Function (B-SPCM) Matrix','Fontsize',14)
colormap(pink)
colorbar 
axis square

end