function  [dist, params]= MSRcurve(M,X,tolerance,k)
% MSRCURVE computes the minimal scaling-rotation (MSR) distance from M to X
% dist = MSRcurve(M,X) returns the quotient distance.
% [dist, params] = MSRcurve(M,X)
%         params is a cell array of parameters of the optimal scaling
%         rotation curve (params.U, parmas.D, params.V, params.Lambda,
%         params.A, params.L), from M to X.
%
% [dist, params] = MSRcurve(M,X,tolerance)
%         tolerance (default is 1e-14) is used to adjust allowed
%         difference in eigenvalues (to be recognized as having the same value).
% [dist, params] = MSRcurve(M,X,tolerance,k)
%         scaling-factor k (default = 1)
%         See Section 3.2, Equation (3.4) (http://arxiv.org/abs/1406.3361)
%         for the definition of the scaling-factor. 
%
% June 9, 2015 Sungkyu Jung.


% preprocessing
[p1,p2]=size(M);
[p11,p22]=size(X);
if p1 ~= p2 || p11~=p22
    disp('input matrix must be symmetric postive definite matrices');
    return;
end
if p1 ~= p11
    disp('input matrices must agree in size');
    return;
end
if p1 > 3;
    disp('Only p =2 or p = 3 is allowed.');
    return;
end

tol = 1e-14;
if nargin > 2
    tol = tolerance;
end

K = 1; 
if nargin > 3
    K = k;
end
 
% create a version of $M$, eigenvalues are in descending order.
[U,D]=svd(M);
d = diag(D); [d1, d1perm] = sort(d,'descend');  % make sure D is in descending order
ii = zeros(p1);
for j = 1:p1;    ii(j,d1perm(j)) = 1;   end
U = U*ii'; D = ii*D*ii'; d = d1;
if det(U)<0; % make sure U is in SO(p)
    U(:,1) = -U(:,1);
end

% create a version of $X$, eigenvalues are in descending order.
[V,Lambda]=svd(X);
lambda= diag(Lambda); [lambda1, l1perm] = sort(lambda,'descend');  % make sure Lambda is in descending order
ii = zeros(p1);
for j = 1:p1;    ii(j,l1perm(j)) = 1;   end
V = V*ii'; Lambda = ii*Lambda*ii'; lambda = lambda1;
if det(V)<0; % make sure V is in SO(p)
    V(:,1) = -V(:,1);
end

switch p1
    case 2
        if d(1) > d(2)+tol && lambda(1) > lambda(2)+tol
            % both sets of eigenvalues are all distinctive (case 1)
            [Is, nsignchange]= signchangematrix(p1);
            [P, nperm]= permutematrix(p1);
            distvec = zeros(1,nsignchange*nperm);
            for i =1:nsignchange;
                for j = 1:nperm;
                    Ustar = U*P{j}*Is{i};
                    Dstar = P{j}'*D*P{j};
                    A = logm(V*Ustar'); % rotation parameter
                    L = logm(Lambda/Dstar);
                    distvec(nperm*(i-1)+j) = sqrt(K/2* trace(A*A') + trace (L*L') );
                end
            end
            [dist, minid] = min(distvec);
            i = ceil(minid/nperm);
            j = minid - (i-1)*nperm;
            
            params.U   = U*P{j}*Is{i};
            params.D  = P{j}'*D*P{j};
            params.V = V;
            params.Lambda = Lambda;
            params.A = logm(V*params.U'); % rotation parameter
            params.L = logm(Lambda/params.D);
            return;
        elseif abs( d(1) - d(2) ) < tol
            % M is spherical (case 2)
            params.U = V;
            params.D = D;
            params.V = V;
            params.Lambda = Lambda;
            params.A = zeros(p1); % logm(V*V');
            params.L = logm(Lambda/D);
            dist = sqrt( trace (params.L*params.L') );
            return;
        else
            % then X is spherical, but M is not. (case 2)
            params.U = U;
            params.D = D;
            params.V = U;
            params.Lambda = Lambda;
            params.A = zeros(p1); % logm(U*U');
            params.L = logm(Lambda/D);
            dist = sqrt( trace (params.L*params.L') );
            return;
        end
    case 3
        
        i1 = d(1) > d(2) + tol;
        i2 = d(2) > d(3) + tol;
        if i1  && i2
            eigenvaluecased = 1; % no permutation
        elseif (~i1 && i2)
            eigenvaluecased = 2; % no permutation
        elseif (i1 && ~i2)
            eigenvaluecased = 2; % permute 1-3
            l1perm = [3 2 1];
            ii = zeros(p1);
            for j = 1:p1;    ii(j,l1perm(j)) = 1;   end
            ii(1,l1perm(1)) = det(ii);
            U = U*ii'; D = ii*D*ii'; d = d(l1perm);
        else
            eigenvaluecased = 3; % no permute
        end
        
        
        i1 = lambda(1) > lambda(2) + tol;
        i2 = lambda(2) > lambda(3) + tol;
        % Preparation for Lambda
        if i1  && i2
            eigenvaluecasel = 1; % no permutation
            
        elseif (~i1 && i2)
            eigenvaluecasel = 2; % no permutation
            
        elseif (i1 && ~i2)
            eigenvaluecasel = 2; % permute 1-3
            l1perm = [3 2 1];
            ii = zeros(p1);
            for j = 1:p1;    ii(j,l1perm(j)) = 1;   end
            ii(1,l1perm(1)) = det(ii);
            V = V*ii'; Lambda = ii*Lambda*ii'; lambda = lambda(l1perm);
        else
            eigenvaluecasel = 3; % no permute
        end
        
        % Four cases here
        if eigenvaluecased == 1 && eigenvaluecasel == 1
            % both sets of eigenvalues are all distinctive (case 1)
            [Is, nsignchange]= signchangematrix(p1);
            [P, nperm]= permutematrix(p1);
            distvec = zeros(1,nsignchange*nperm);
            for i =1:nsignchange;
                for j = 1:nperm;
                    Ustar = U*P{j}'*Is{i};
                    Dstar = P{j}*D*P{j}';
                    A = logm(V*Ustar'); % rotation parameter
                    L = logm(Lambda/Dstar);
                    distvec(nperm*(i-1)+j) = sqrt(K/2* trace(A*A') + trace (L*L'));
                end
            end
            [dist, minid] = min(distvec);
            i = ceil(minid/nperm);
            j = minid - (i-1)*nperm;
            
            params.U   = U*P{j}'*Is{i};
            params.D  = P{j}*D*P{j}';
            params.V = V;
            params.Lambda = Lambda;
            params.A = logm(V*params.U'); % rotation parameter
            params.L = logm(Lambda/params.D);
            return;
            
        elseif  eigenvaluecased == 3
            % M is spherical (case 4)
            params.U = V;
            params.D = D;
            params.V = V;
            params.Lambda = Lambda;
            params.A = zeros(p1); % logm(V*V');
            params.L = logm(Lambda/D);
            dist = sqrt( trace (params.L*params.L') );
            return;
            
        elseif eigenvaluecasel == 3
            % X is spherical (case 4) change the role of M and X
            params.U = U;
            params.D = D;
            params.V = U;
            params.Lambda = Lambda;
            params.A = zeros(p1); % logm(U*U');
            params.L = logm(Lambda/D);
            dist = sqrt( trace (params.L*params.L') );
            return;
        elseif eigenvaluecased == 2 && eigenvaluecasel == 1;
            % (case 2)
            [Ustar, Dstar, A,L, dist]=case2optimalrotation(U,D,V,Lambda,K);
            params.U = Ustar;
            params.D = Dstar;
            params.V = V;
            params.Lambda = Lambda;
            params.A = A;
            params.L = L;
            return;
        elseif eigenvaluecased == 1 && eigenvaluecasel == 2;
            % (case 2)  change the role of M and X
            
            [Vstar, Lstar, ~,~, dist]=case2optimalrotation(V,Lambda,U,D,K);
            params.U = U;
            params.D = D;
            params.V = Vstar;
            params.Lambda = Lstar;
            params.A =  logm(Vstar*U');
            params.L =  logm(Lstar/D);
            return;
        else
            % (case 3)
            p = 3;
            % three permutations and two sign changes;
            allperm = [1 2 3; 3 1 2; 1 3 2];
            nperm = 3;
            P = cell(nperm,1);
            for i = 1:nperm
                ii = zeros(p);
                for j = 1:p
                    ii(j,allperm(i,j)) = 1;
                end
                ii(1,allperm(i,1)) = det(ii);
                P{i} = ii;
            end
            nsignchange = 2;
            Is = cell(nsignchange,1);
            Is{1} = diag([1,1,1]);
            Is{2} = diag([-1,1,-1]);
            
            dist = 10000;
            for i =1:nsignchange;
                for j = 1:nperm;
                    [UR, VR]=case3optimalrotation(U,V,Is{i},P{j},0);   %%%%%%%%
                    Unow = UR*Is{i}*P{j}';
                    Dnow = P{j}*D*P{j}';
                    Vnow = VR ;
                    A = logm(Vnow*Unow'); % rotation parameter
                    L = logm(Lambda/Dnow);
                    dist_ij = sqrt(K/2* trace(A*A') + trace (L*L'))  ;
                    if  dist_ij < dist % then update
                        dist = dist_ij;
                        params.U = Unow;
                        params.D = Dnow;
                        params.V = Vnow;
                        params.Lambda = Lambda;
                        params.A =  logm(Vnow*Unow');
                        params.L =  logm(Lambda/Dnow);
                    end
                end
            end
            
        end
        
end

end




function [Ustar, Dstar, A,L, dist] = case2optimalrotation(U,D,V,Lambda,K)
% D must be arranged as diag(d,d,d3) since the optimal rotation is U*Rhat
% times sign change and permutation.

p = 3;
% three permutations and two sign changes;
allperm = [1 2 3; 3 1 2; 1 3 2];
nperm = 3;
P = cell(nperm,1);
for i = 1:nperm
    ii = zeros(p);
    for j = 1:p
        ii(j,allperm(i,j)) = 1;
    end
    ii(1,allperm(i,1)) = det(ii);
    P{i} = ii;
end
nsignchange = 2;
Isigncell = cell(nsignchange,1);
Isigncell{1} = diag([1,1,1]);
Isigncell{2} = diag([-1,1,-1]);

Ustar = U;
Dstar = D;
A = logm(V*Ustar');
L = logm(Lambda/Dstar);
dist = sqrt(K/2* trace(A*A') + trace (L*L') );
for i = 1:nsignchange
    for j = 1:nperm
        Gamma = Isigncell{i}*P{j}'*V'*U;
        
        G = [Gamma(1,1) + Gamma(2,2) ; Gamma(1,2) - Gamma(2,1)];
        
        if norm(G) > 0
            G = G / norm(G);
            Rhat = [G(1) -G(2) 0 ;
                G(2) G(1) 0 ;
                0 0 1];
        else
            %    Gamma = Isigncell{i}*P{j}'*V'*U;
            [E1, ~, E2] = svd(Gamma(1:2,1:2));
            if det(Gamma(1:2,1:2)) < 0;
                E1(:,2) = -E1(:,2);
            end
            Rhat = [E2*E1'  zeros(2,1);
                zeros(1,2)    1  ];
        end
        
        Ustar_ij = U*Rhat*Isigncell{i}*P{j}';
        Dstar_ij = P{j}*D*P{j}';
        
        A_ij = logm(V*Ustar_ij');
        L_ij = logm(Lambda/Dstar_ij);
        dist_ij = sqrt(K/2* trace(A_ij*A_ij') + trace (L_ij*L_ij') );
        
        if  dist_ij < dist
            % then update
            A = A_ij;
            L = L_ij;
            dist = dist_ij;
            Ustar = Ustar_ij;
            Dstar = Dstar_ij;
        end
    end
end
end


function [UR, VR]=case3optimalrotation(U,V,Isign,Perm,idisplay);

% Preparation
IsP = Isign*Perm ;
VtU = V'*U ;


% initialize
cnt = 0;
TOL = 1e-15;
objfunvalue = trace(VtU * IsP);
Rphit = eye(3);
if idisplay
    theta = 0;
    phi = 0;
    angletrace = [theta phi];
end

while true
    % Given phi, update theta;
    Gamma  = IsP * Rphit * VtU ;
    G = [Gamma(1,1) + Gamma(2,2) ; Gamma(1,2) - Gamma(2,1)]; 
    if abs(sum(G.^2)) < TOL
        Rtheta = eye(3);
    else 
    G = G / norm(G);
    
    Rtheta = [G(1) -G(2) 0 ;
        G(2) G(1) 0 ;
        0 0 1];
    end
    % Given theta, update phi
    
    Gamma = VtU *  Rtheta * IsP ;
    Gt = [Gamma(1,1) + Gamma(2,2) ; Gamma(1,2) - Gamma(2,1)]; 
    if abs(sum(Gt.^2)) < TOL
        Rphit = eye(3);
    else 
    Gt = Gt / norm(Gt);
    
    Rphit = [Gt(1) -Gt(2) 0 ;
        Gt(2) Gt(1) 0 ;
        0 0 1];
    end
    
    cnt = cnt + 1;
    
    objfunvalue(cnt+1) = trace( Gamma * Rphit) ;
    if idisplay
        phi = mod(-atan2(Gt(2),Gt(1)), 2*pi) ;
        theta = mod(atan2(G(2),G(1)), 2*pi);
        angletrace(cnt+1,:) = [theta phi] ;
    end
    if objfunvalue(cnt+1) - objfunvalue(cnt) < TOL;         break;     end
    if cnt > 50;                                            break;     end
end

UR = U* Rtheta;
VR = V * Rphit';


if idisplay % display on current figure;
    
    % objective function to be maximized for angles [theta phi].
    ObjFun = @(angthephi) (trace(VtU * ...
        [cos(angthephi(1)) -sin(angthephi(1)) 0;
        sin(angthephi(1)) cos(angthephi(1)) 0;
        0 0 1] * ...
        IsP * ...
        [cos(angthephi(2)) sin(angthephi(2)) 0 ;
        -sin(angthephi(2)) cos(angthephi(2)) 0;
        0 0 1])     );
    
    clf;
    subplot(1,2,1);
    K = 41;
    anglevec = linspace(0,2*pi,K);
    Objfunvaluemat = zeros(K,K);
    
    for k = 1:K;
        for k2 = 1:K;
            phi = anglevec(k);
            theta = anglevec(k2);
            Objfunvaluemat(k,k2) =  ObjFun([theta phi]);
        end
    end
    contour(anglevec(1:K),anglevec(1:K),Objfunvaluemat,'ShowText','on')
    xlabel('theta');
    ylabel('phi');
    hold on;
    angletrace(angletrace<0) = angletrace(angletrace<0) + 2*pi;
    plot(angletrace(:,1),angletrace(:,2),'.-k')
    
    subplot(1,2,2);
    plot(objfunvalue)
    xlabel('Iteration');
    ylabel('objective function');
end



end


