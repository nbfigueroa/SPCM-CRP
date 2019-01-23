function  [Ustar,Dstar,dist]= optver(V,Lambda,M,tolerance)
% OPTVER returns the optimal (minimal) version (U,D) of M with resprect to
% (V,Lambda). (works for 2x2 or 3x3 case)
% [U,D]= optver(V,Lambda,M), where M is an SPD matrix.
%
% June 9, 2015 Sungkyu Jung.

[p1,p2]=size(V);
[p11,p22]=size(M);
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

tol = 1e-15;
if nargin > 3
    tol = tolerance;
end

K = 1; 
if nargin > 4
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



switch p1
    case 2
        if d(1) > d(2) + tol
            [Is, nsignchange]= signchangematrix(p1);
            [P, nperm]= permutematrix(p1);
            distvec = zeros(1,nsignchange*nperm);
            for i =1:nsignchange;
                for j = 1:nperm;
                    Ustar = U*P{j}*Is{i};
                    Dstar = P{j}'*D*P{j};
                    A = logm(V*Ustar');
                    L = logm(Lambda/Dstar);
                    distvec(nperm*(i-1)+j) = sqrt(K/2* trace(A*A') + trace (L*L') );
                end
            end
            [dist, minid] = min(distvec);
            i = ceil(minid/nperm);
            j = minid - (i-1)*nperm;
            
            Ustar  = U*P{j}*Is{i};
            Dstar  = P{j}'*D*P{j};
            return;
            
        else
            % M is spherical (case 2)
            Ustar = V;
            Dstar = D;
            if nargout > 2;
                params.L = logm(Lambda/D);
                dist = sqrt( trace (params.L*params.L') );
            end
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
        
        
        % Four cases here
        if eigenvaluecased == 1
            
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
            
            Ustar  = U*P{j}*Is{i};
            Dstar  = P{j}'*D*P{j};
            return;
            
        elseif  eigenvaluecased == 3  % M is spherical
            Ustar = V;
            Dstar = D;
            if nargout > 2;
                params.L = logm(Lambda/D);
                dist = sqrt( trace (params.L*params.L') );
            end
            return;
        else
            [Ustar, Dstar, ~,~, dist]=case2optimalrotation(U,D,V,Lambda,K);
            return;
            
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






% 
% function [Ustar, Dstar, A,L, dist] = case2optimalrotation(U,D,V,Lambda)
% % D must be arranged as diag(d,d,d3) since the optimal rotation is U*Rhat
% % times sign change and permutation.
% 
% p = 3;
% % three permutations and two sign changes;
% allperm = [1 2 3; 3 1 2; 1 3 2];
% nperm = 3;
% P = cell(nperm,1);
% for i = 1:nperm
%     ii = zeros(p);
%     for j = 1:p
%         ii(j,allperm(i,j)) = 1;
%     end
%     ii(1,allperm(i,1)) = det(ii);
%     P{i} = ii;
% end
% nsignchange = 2;
% Isigncell = cell(nsignchange,1);
% Isigncell{1} = diag([1,1,1]);
% Isigncell{2} = diag([-1,1,-1]);
% 
% Ustar = U;
% Dstar = D;
% A = logm(V*Ustar');
% L = logm(Lambda/Dstar);
% dist = sqrt(1/2* trace(A*A') + trace (L*L') );
% for i = 1:nsignchange
%     for j = 1:nperm
%         Gamma = Isigncell{i}*P{j}'*V'*U;
%         
%     G = [Gamma(1,1) + Gamma(2,2) ; Gamma(1,2) - Gamma(2,1)];  G = G / norm(G);
%     
%     Rhat = [G(1) -G(2) 0 ;
%         G(2) G(1) 0 ;
%         0 0 1]; 
%         
%         %             Gamma = Isigncell{i}*P{j}'*V'*U;
%         %         [E1, ~, E2] = svd(Gamma(1:2,1:2));
%         %         if det(Gamma(1:2,1:2)) < 0;
%         %             E1(:,2) = -E1(:,2);
%         %         end
%         %         Rhat = [E2*E1'  zeros(2,1);
%         %             zeros(1,2)    1  ]; 
%         
%         Ustar_ij = U*Rhat*Isigncell{i}*P{j}';
%         Dstar_ij = P{j}*D*P{j}';
%         
%         A_ij = logm(V*Ustar_ij');
%         L_ij = logm(Lambda/Dstar_ij);
%         dist_ij = sqrt(1/2* trace(A_ij*A_ij') + trace (L_ij*L_ij') );
%         
%         if  dist_ij < dist
%             % then update
%             A = A_ij;
%             L = L_ij;
%             dist = dist_ij;
%             Ustar = Ustar_ij;
%             Dstar = Dstar_ij;
%         end
%     end
% end
% end
% 
% 
