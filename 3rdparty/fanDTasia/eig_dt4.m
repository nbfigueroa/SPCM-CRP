function [eigvec,eigval]=eig_dt4(dt4)
%-fanDTasia ToolBox------------------------------------------------------------------
% This Matlab script is part of the fanDTasia ToolBox: a Matlab library for Diffusion 
% Weighted MRI (DW-MRI) Processing, Diffusion Tensor (DTI) Estimation, High-order 
% Diffusion Tensor Analysis, Tensor ODF estimation, Visualization and more.
%
% A Matlab Tutorial on DW-MRI can be found in:
% http://www.cise.ufl.edu/~abarmpou/lab/fanDTasia/tutorial.php
%------------------------------------------------------------------------------------
%This function takes as an input a 4th-order tensor in 3-dimensions, formated as a 15dim-vector.
%The order of coefficients is the same as the one used by printTensors(dt4)
%
%It produces a list of eigenvectors (of size Nx3) and a list of the corresponding 
%eigenvalues. The eigenvectors correspond to the critical orientations (i.e. maxima,
%minima, etc.) and the eigenvalues correspond to the value of the tensor-polynomial
%at the orientation x=eigenvector. i.e. eigvalue=<D o x o x o x o x>.
%
%This script uses the symbolic toolbox of Matlab, that solves systems of equations.
%
%If you have any question, I will be happy to assist you. Feel free to
%e-mail me at: abarmpou@cise.ufl.edu
%
%Author: Angelos Barmpoutis, Ph.D.
%
%-----------------------------------------------------------------------------------
%
% If you use this software please cite the following papers on 4th-order tensors:
% 1) A. Barmpoutis et al. "Regularized Positive-Definite Fourth-Order Tensor Field
% Estimation from DW-MRI", NeuroImage, Vol. 45(1 sup.1), Page(s): 153-162, 2009 
% 2) A. Barmpoutis and B.C. Vemuri, "A Unified Framework for Estimating Diffusion 
% Tensors of any order with Symmetric Positive-Definite Constraints", 
% In the Proceedings of ISBI, 2010
%
%-----------------------------------------------------------------------------------

W=convertD2W(dt4([15 5 1 12 3 10 11 8 7 14 13 9 4 6 2]));

D=eye(3);
Wbar=convertW2Wbar(W,D);

n_eigval=0;
eigval=[];
eigvec=[];

if (Wbar(2,1,1,1)==0) & (Wbar(3,1,1,1)==0)
    
    n_eigval=n_eigval+1;
    eigval=[eigval Wbar(1,1,1,1)/D(1,1)];    
    eigvec=[eigvec;sqrt(1/D(1,1)) 0 0];
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
syms t;

eq1=-Wbar(2,1,1,1)*t^4+(Wbar(1,1,1,1)-3*Wbar(2,1,1,2))*t^3+3*(Wbar(1,1,1,2)-Wbar(2,1,2,2))*t^2+(3*Wbar(1,1,2,2)-Wbar(2,2,2,2))*t+W(1,2,2,2);

eq2=Wbar(3,1,1,1)*t^3+3*Wbar(3,1,1,2)*t^2+3*Wbar(3,1,2,2)*t+Wbar(3,2,2,2);


sol=solve(eq2);

for i=1:length(sol)
    x=double(sol(i));
    y=-Wbar(2,1,1,1)*x^4+(Wbar(1,1,1,1)-3*Wbar(2,1,1,2))*x^3+3*(Wbar(1,1,1,2)-Wbar(2,1,2,2))*x^2+(3*Wbar(1,1,2,2)-Wbar(2,2,2,2))*x+W(1,2,2,2);
    
    if abs(imag(y))<0.00000000001 & abs(real(y))<0.000000000001
        n_eigval=n_eigval+1;
        eigvec=[eigvec;[x 1 0]/sqrt(D(1,1)*x*x+2*D(1,2)*x+D(2,2))];
        eigval=[eigval evalW(W,eigvec(n_eigval,:))];    
        
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
syms u v;

eq1=-Wbar(3,1,1,1)*u^4-3*Wbar(3,1,1,2)*u^3*v+(Wbar(1,1,1,1)-3*Wbar(3,1,1,3))*u^3-3*Wbar(3,1,2,2)*u^2*v^2+(3*Wbar(1,1,1,2)-6*Wbar(3,1,2,3))*u^2*v ...
    +(3*Wbar(1,1,1,3)-3*Wbar(3,1,3,3))*u^2-3*Wbar(3,2,2,3)*u*v^2-Wbar(3,2,2,2)*u*v^3+3*Wbar(1,1,2,2)*u*v^2+(6*Wbar(1,1,2,3)-3*Wbar(3,2,3,3))*u*v ...
    +(3*Wbar(1,1,3,3)-Wbar(3,3,3,3))*u+Wbar(1,2,2,2)*v^3+3*Wbar(1,2,2,3)*v^2+3*Wbar(1,2,3,3)*v+Wbar(1,3,3,3);

eq2=-Wbar(3,1,1,1)*u^3*v+Wbar(2,1,1,1)*u^3-3*Wbar(3,1,1,2)*u^2*v^2+(3*Wbar(2,1,1,2)-3*Wbar(3,1,1,3))*u^2*v+3*Wbar(2,1,1,3)*u^2-3*Wbar(3,1,2,2)*u*v^3 ...
    +(3*Wbar(2,1,2,2)-6*Wbar(3,1,2,3))*u*v^2+(6*Wbar(2,1,2,3)-3*Wbar(3,1,3,3))*u*v+3*Wbar(2,1,3,3)*u+3*Wbar(2,2,2,3)*v^2-Wbar(3,2,2,2)*v^4 ...
    +(Wbar(2,2,2,2)-3*Wbar(3,2,2,3))*v^3-3*Wbar(3,2,3,3)*v^2+(3*Wbar(2,2,3,3)-Wbar(3,3,3,3))*v+Wbar(2,3,3,3);

sol=solve(eq1,eq2);

for i=1:length(sol.u)
    u=double(sol.u(i));
    v=double(sol.v(i));
    
    if abs(imag(v))<0.000000000000001 & abs(imag(u))<0.0000000000000000001
        n_eigval=n_eigval+1;
        eigvec=[eigvec;[u v 1]/sqrt(D(1,1)*u*u+2*D(1,2)*u*v+2*D(1,3)*u+D(2,2)*v*v+2*D(2,3)*v+D(3,3))];
        eigval=[eigval evalW(W,eigvec(n_eigval,:))];
    end
end

%sorting the eigvectors according to the eigenvalues
[s,indx]=sort(eigval);
eigvec=eigvec(indx(length(eigval):-1:1),:);
eigval=eigval(indx(length(eigval):-1:1));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Wbar=convertW2Wbar(W,D)
Dinv=inv(D);
for i1=1:3
    for i2=1:3
        for i3=1:3
            for i4=1:3
                Wbar(i1,i2,i3,i4)=0;
                for h=1:3
                    Wbar(i1,i2,i3,i4)=Wbar(i1,i2,i3,i4)+Dinv(i1,h)*W(h,i2,i3,i4);
                end
            end
        end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function out=evalW(W,x)
out=0;
for i1=1:3
    for i2=1:3
        for i3=1:3
            for i4=1:3
                    out=out+W(i1,i2,i3,i4)*x(i1)*x(i2)*x(i3)*x(i4);              
            end
        end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function W=convertD2W(D)
%converts a vectorized version D([1:15]) with unique coeeficients
%to a fully symmetric fourth order tensor W(:,:,:,:)
for i1=1:3
    for i2=1:3
        for i3=1:3
            for i4=1:3
                
                ix=0;
                iy=0;
                iz=0;
                
                if i1==1
                    ix=ix+1;
                elseif i1==2
                    iy=iy+1;
                elseif i1==3
                    iz=iz+1;
                end
                
                if i2==1
                    ix=ix+1;
                elseif i2==2
                    iy=iy+1;
                elseif i2==3
                    iz=iz+1;
                end
                
                if i3==1
                    ix=ix+1;
                elseif i3==2
                    iy=iy+1;
                elseif i3==3
                    iz=iz+1;
                end
                
                if i4==1
                    ix=ix+1;
                elseif i4==2
                    iy=iy+1;
                elseif i4==3
                    iz=iz+1;
                end
                
             
                if (ix==4)&(iy==0)&(iz==0)
                    W(i1,i2,i3,i4)=D(1)/computeFactor(4,0,0);
                elseif (ix==0)&(iy==4)&(iz==0)
                    W(i1,i2,i3,i4)=D(2)/computeFactor(0,4,0);
                elseif (ix==0)&(iy==0)&(iz==4)
                    W(i1,i2,i3,i4)=D(3)/computeFactor(0,0,4);
                elseif (ix==2)&(iy==2)&(iz==0)
                    W(i1,i2,i3,i4)=D(4)/computeFactor(2,2,0);
                elseif (ix==0)&(iy==2)&(iz==2)
                    W(i1,i2,i3,i4)=D(5)/computeFactor(0,2,2);
                elseif (ix==2)&(iy==0)&(iz==2)
                    W(i1,i2,i3,i4)=D(6)/computeFactor(2,0,2);
                elseif (ix==2)&(iy==1)&(iz==1)
                    W(i1,i2,i3,i4)=D(7)/computeFactor(2,1,1);
                elseif (ix==1)&(iy==2)&(iz==1)
                    W(i1,i2,i3,i4)=D(8)/computeFactor(1,2,1);
                elseif (ix==1)&(iy==1)&(iz==2)
                    W(i1,i2,i3,i4)=D(9)/computeFactor(1,1,2);
                elseif (ix==3)&(iy==1)&(iz==0)
                    W(i1,i2,i3,i4)=D(10)/computeFactor(3,1,0);
                elseif (ix==3)&(iy==0)&(iz==1)
                    W(i1,i2,i3,i4)=D(11)/computeFactor(3,0,1);
                elseif (ix==1)&(iy==3)&(iz==0)
                    W(i1,i2,i3,i4)=D(12)/computeFactor(1,3,0);
                elseif (ix==0)&(iy==3)&(iz==1)
                    W(i1,i2,i3,i4)=D(13)/computeFactor(0,3,1);
                elseif (ix==1)&(iy==0)&(iz==3)
                    W(i1,i2,i3,i4)=D(14)/computeFactor(1,0,3);
                elseif (ix==0)&(iy==1)&(iz==3)
                    W(i1,i2,i3,i4)=D(15)/computeFactor(0,1,3);
                end             
            end
        end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function counter=computeFactor(x,y,z)
counter=0;
for i1=1:3
    for i2=1:3
        for i3=1:3
            for i4=1:3
                ix=0;
                iy=0;
                iz=0;
                if i1==1
                    ix=ix+1;
                elseif i1==2
                    iy=iy+1;
                elseif i1==3
                    iz=iz+1;
                end
                
                if i2==1
                    ix=ix+1;
                elseif i2==2
                    iy=iy+1;
                elseif i2==3
                    iz=iz+1;
                end
                
                if i3==1
                    ix=ix+1;
                elseif i3==2
                    iy=iy+1;
                elseif i3==3
                    iz=iz+1;
                end
                
                if i4==1
                    ix=ix+1;
                elseif i4==2
                    iy=iy+1;
                elseif i4==3
                    iz=iz+1;
                end
                if (ix==x)&(iy==y)&(iz==z)
                    counter=counter+1;
                end
            end
        end
    end
end
