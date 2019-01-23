
function h = plotellipsoid(U,dd,colorbydominantaxis)
% PLOTELLIPSOID plot 3D ellipsoid. 
% 
% plotellipsoid(U,dd) plots 3D ellipsoid corresponding to eigenvector
% matrix 'U' and vector of eigenvalues 'dd'. The color of the figure is
% determined by the first column of U. 
% plotellipsoid(U,dd,1) then color of the figure is determined by the
% dominant axis of the ellipsoid. 
%
% June 9, 2015 Sungkyu Jung.


if nargin == 2 
    colorbydominantaxis = false;
end


  dd = sqrt(dd);
  [x,y,z]=ellipsoid(0,0,0,dd(1),dd(2),dd(3));
  xx = U(1,1)*x+U(1,2)*y+U(1,3)*z;
  yy = U(2,1)*x+U(2,2)*y+U(2,3)*z;
  zz = U(3,1)*x+U(3,2)*y+U(3,3)*z;
  if colorbydominantaxis
        [~,id] = max(dd); 
  else
      id = 1;
  end
  coloring = abs(U(:,id));
  coloring(coloring > 1) = 1;
  coloring(coloring < 0 ) = 0;
  
  h = surf(xx,yy,zz,'FaceColor',coloring,'facealpha',0.6,'EdgeColor','none');
  light('position',[1 0 1]);
  axis equal