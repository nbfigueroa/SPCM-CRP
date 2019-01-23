function h = plotellipse(M,c);
% PLOTELLIPSE plot the ellipse on the current figure. 
% h = plotellipse(M,c), corresponding to x M^{-1} x = 1, with color chosen
% by 'c'
% h = plotellipse(M), corresponding to x M^{-1} x = 1, with default color
% 'red'
%
% June 9, 2015 Sungkyu Jung.


% M = matd(vecd(X))

% compute ellipse 1
p = 2;
theta = linspace(0,2*pi,202);
[U, L]=eig(M);
l = sqrt(diag(L));
F = 1;
ell1 = U*[l(1)*cos(theta); l(2)*sin(theta)]*sqrt(F);
if nargin == 2
    h = plot(ell1(1,:),ell1(2,:),'color',c,'Linewidth',2);
else
    h = plot(ell1(1,:),ell1(2,:),'r','Linewidth',2);
end