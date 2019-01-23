% Spectral Quaternion distance and interpolation 
% by 
% 
% @Article{collard2012anisotropy,
%   Title                    = {An anisotropy preserving metric for DTI processing},
%   Author                   = {Collard, Anne and Bonnabel, Silv{\`e}re and Phillips, Christophe and Sepulchre, Rodolphe},
%   Journal                  = {arXiv preprint arXiv:1210.2826},
%   Year                     = {2012}
% }
function  [dist, interp, eigenvaluesONLY, eigenvectorsONLY,k,angle]= SQcurve(Y,X,Ninterp)
% interpolation from Y to X
% 
% needs rot2quat, quat2rot functions

if nargin < 3 
    Ninterp = 7;
end
% preprocessing
[p1,p2]=size(Y);
[p11,p22]=size(X);
if p1 ~= p2 || p11~=p22
    disp('input matrix must be symmetric postive definite matrices');
    return;
end
if p1 ~= p11
    disp('input matrices must agree in size');
    return;
end
if p1 ~= 3;
    disp('Only p =2 or p = 3 is allowed.');
    return;
end

tol = 1e-14;


% create a version of $M$, eigenvalues are in descending order.
[U,D]=svd(Y); 
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

%% Pre-processing (handling equal eigenvalues)

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
        
        if eigenvaluecasel == 3; 
            V = U; 
        end
        
        if eigenvaluecased == 3;
            U = V;
        end
       

%% 1. Rotation distances 
 quatU = rot2quat(U);
 Is=signchangematrix(3); 
 quatVvec = zeros(4,4);
 for k = 1:4 
       quatVvec(:,k) = rot2quat(V*Is{k});
 end
 % collection of all quaternions possible from V (and sign-changed Vs)
 quatVvec = [quatVvec -quatVvec]; 

 innerproducts = quatU' * quatVvec ; 
 [q1rqs, indexqs]=max(innerproducts);
 % optimal quaternion of V w.r.t quatU.
 quatV = quatVvec(:,indexqs);
 
 squared_dSO3 = 2 - 2 * q1rqs ; 
 angle = acos(q1rqs) * 2 ; % rotation angle
 %% 2. Eigenvalues 
 
 squared_dDiag = sum((log(lambda ./ d )).^2);
 
 
 %% scale factor "k"
 ha1 = log(lambda(1)/lambda(3));
 ha2 = log(d(1)/d(3));
 k = (1 + tanh( 3 * ha1 * ha2 - 7 ) ) / 2 ; 
 
 
 %% distance 
 
 dist = sqrt(  k *  squared_dSO3 +   squared_dDiag ); 
 
 %% interpolations
 weights = (0:(Ninterp-1)) / (Ninterp-1); 
 interp = zeros(3,3,Ninterp);
 eigenvaluesONLY = zeros(3,3,Ninterp);
 eigenvectorsONLY = zeros(3,3,Ninterp);
for i = 1: Ninterp
    qnow = ( 1 - weights(i) ) * quatU  +  weights(i)  * quatV;
    R = quat2rot(qnow / norm(qnow));
    Eigenvalues = diag(exp(( 1 - weights(i) ) * log(d) + weights(i) * log(lambda)));
    interp(:,:,i) = R * Eigenvalues * R' ; 
    eigenvaluesONLY(:,:,i) = Eigenvalues;
    eigenvectorsONLY(:,:,i) = R; 
end


