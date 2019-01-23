function [T, dist, Uarray, Darray,A,L]= scrotcurve(U,D,V,Lambda,t)
% SCROTCURVE computes discrete evaluation of the scaling-rotation (SCAROT) 
% curve from (U,D) to (V,Lambda). 
%
% T = scrotcurve(U,D,V,Lambda,t) is a vecd'ed array of pxp matrices, where 
% each column is vecd(M), where M is the evaluated value of SCAROT curve 
% at the value of t. If t is a vector of length m, T is length m.
% T = scrotcurve(U,D,V,Lambda), with t = linspace(0,1,101) by default.
% [T, dist]= scrotcurve(U,D,V,Lambda,t) returns the geodesic distance.
% [T, dist, Uarray, Darray] = scrotcurve(U,D,V,Lambda,t) returns the 
% eigenvector-eigenvalue sequence. 
% [T, dist, Uarray, Darray,A,L]= scrotcurve(U,D,V,Lambda,t) also returns
% the computed parameters of the SCAROT curve. 
% 
%
% June 9, 2015 Sungkyu Jung.


if nargin == 4;
    t = linspace(0,1,101);
end
 
A = logm(V*U'); % rotation parameter
L = logm(Lambda/D);
[p,~]=size(A);

%Tcell = cell(length(t),1);
T = zeros(p*(p+1)/2,length(t));
for i = 1:length(t);
T(:,i) = vecd(expm(A*t(i))*U*D*expm(L*t(i))*U'*expm(A'*t(i)));
end
dist = sqrt(1/2* trace(A*A') + trace (L*L') );

Uarray = cell(length(t),1);
Darray = cell(length(t),1);
if nargout > 2;
    for i = 1:length(t)
    Uarray{i} = expm(A*t(i))*U;
    Darray{i} = D*expm(L*t(i));
    end
end
    
    
    
    
end