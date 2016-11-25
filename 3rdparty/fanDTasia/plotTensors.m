function plotTensors(UniqueTensorCoefficients,delta,params,image_data)
%-fanDTasia ToolBox------------------------------------------------------------------
% This Matlab script is part of the fanDTasia ToolBox: a Matlab library for Diffusion 
% Weighted MRI (DW-MRI) Processing, Diffusion Tensor (DTI) Estimation, High-order 
% Diffusion Tensor Analysis, Tensor ODF estimation, Visualization and more.
%
% A Matlab Tutorial on DW-MRI can be found in:
% http://www.cise.ufl.edu/~abarmpou/lab/fanDTasia/tutorial.php
%
%-CITATION---------------------------------------------------------------------------
% If you use this software please cite the following work:
%
% If you work on Higher-Order Tensors please cite the following articles:
% 1) A. Barmpoutis et al. "Regularized Positive-Definite Fourth-Order Tensor Field
% Estimation from DW-MRI", NeuroImage, Vol. 45(1 sup.1), Page(s): 153-162, 2009 
% 2) A. Barmpoutis and B.C. Vemuri, "A Unified Framework for Estimating Diffusion 
% Tensors of any order with Symmetric Positive-Definite Constraints", 
% In the Proceedings of ISBI, 2010
%
% If you work on DTI please cite the following articles:
% 1) A. Barmpoutis, B. C. Vemuri, T. M. Shepherd, and J. R. Forder "Tensor splines for 
% interpolation and approximation of DT-MRI with applications to segmentation of 
% isolated rat hippocampi", IEEE TMI: Transactions on Medical Imaging, Vol. 26(11), 
% pp. 1537-1546 
% 2) A. Barmpoutis and B.C. Vemuri, "A Unified Framework for Estimating Diffusion Tensors 
% of any order with Symmetric Positive-Definite Constraints", In Proc. of ISBI, 2010.
%
%-DESCRIPTION------------------------------------------------------------------------
% This function plots a 2D field of 3D even-order tensors as spherical functions. 
% The 3D tensors must be in the form of a list of unique tensor coefficients using
% the same ordering as defined by the function printTensor. The field can 
% contain either a single tensor, or a row of tensors or a 2D field of tensors.
% Each tensor is colored according to the orientation of maximum value. The orientation
% components X,Y,Z are assigned to the color components R,G,B.
%
%-USE--------------------------------------------------------------------------------
% example 1: plotTensors(UniqueTensorCoefficients)
% where UniqueTensorCoefficients is of size C or CxN or CxNxM, where C is the number
% of unique tensor coefficients in the tensor.
%
% example 2: plotTensors(UniqueTensorCoefficients,delta)
% where delta is a scalar that controls the size 
% of a voxel in the field. Default: delta=1
%
% example 2: plotTensors(UniqueTensorCoefficients,delta,params)
% where params is a 1D, 2D or 3D dimensional vector.
% params(1) takes the value 81 or 321 and controls the angular resolution of the plot.
% params(2) takes the value 1 or 0 for scaling or not each tensor to fit in 1 voxel
% params(3) takes a real in the range [0,1) for enchancing visualy the anisotropy
% of the tensors. For 0, the minimum value is subtracted from the tensors. For 0.999, 
% the largest value will be subtracted from the tensors, i.e. nothing will be displayed.
%
% example 3: plotTensors(UniqueTensorCoefficients,delta,params,image_data)
% where all the input arguments are same as in example 2, and image_data is a scalar 
% valued image shown on the background, behind the tensor field. 
%
% Here is an example: plotTensors(UniqueTensorCoefficients,0.9,[81 1 0]);
%
%-DISCLAIMER-------------------------------------------------------------------------
% You can use this source code for non commercial research and educational purposes 
% only without licensing fees and is provided without guarantee or warrantee expressed
% or implied. You cannot repost this file without prior written permission from the 
% authors. If you use this software please cite the following work:
%
% If you work on Higher-Order Tensors please cite the following articles:
% 1) A. Barmpoutis et al. "Regularized Positive-Definite Fourth-Order Tensor Field
% Estimation from DW-MRI", NeuroImage, Vol. 45(1 sup.1), Page(s): 153-162, 2009 
% 2) A. Barmpoutis and B.C. Vemuri, "A Unified Framework for Estimating Diffusion 
% Tensors of any order with Symmetric Positive-Definite Constraints", 
% In the Proceedings of ISBI, 2010
%
% If you work on DTI please cite the following articles:
% 1) A. Barmpoutis, B. C. Vemuri, T. M. Shepherd, and J. R. Forder "Tensor splines for 
% interpolation and approximation of DT-MRI with applications to segmentation of 
% isolated rat hippocampi", IEEE TMI: Transactions on Medical Imaging, Vol. 26(11), 
% pp. 1537-1546 
% 2) A. Barmpoutis and B.C. Vemuri, "A Unified Framework for Estimating Diffusion Tensors 
% of any order with Symmetric Positive-Definite Constraints", In Proc. of ISBI, 2010.
%
%-AUTHOR-----------------------------------------------------------------------------
% Angelos Barmpoutis, PhD
% Computer and Information Science and Engineering Department
% University of Florida, Gainesville, FL 32611, USA
% abarmpou at cise dot ufl dot edu
%------------------------------------------------------------------------------------


if nargin==1
    delta=1;
    n=81;
    normalize=0;
    aniso=-1;
end

if nargin==2
    n=81;
    normalize=0;
    aniso=-1;
end

if nargin>=3
    if length(params)>=1
        n=params(1);
        normalize=0;
        aniso=-1;
    end
    if length(params)>=2
        normalize=params(2);
        aniso=-1;
    end
    if length(params)>=3
        aniso=params(3);
    end
end

if n~=81 & n~=321
    error('The resolution defined in params(1) must be either 81 or 321.');
end

if normalize~=0 & normalize~=1
    error('The normalization flag defined in params(2) must be either 0 or 1.');
end

if aniso~=-1 & (aniso<0 | aniso>=1)
    error('The visual enchancement of anisotropy defined in params(3) must be a real number in the range [0,1).');
end

sz=size(UniqueTensorCoefficients);
if length(sz)==1
    nx=1;ny=1;id=1;
elseif length(sz)==2
    nx=sz(2);ny=1;id=2;
elseif length(sz)==3
    nx=sz(2);ny=sz(3);id=3;
else
    error('UniqueTensorCoefficients must be 1D, 2D, or 3D matrix');
end

o=sz(1);
if o==6
    order=2;
elseif o==15
    order=4;
elseif o==28
    order=6;
elseif o==45
    order=8;
end

UnitVectors;
XYZ=[g([1:n],:);-g([1:n],:)];
TRI=SphericalMesh(n);
G=constructMatrixOfMonomials(g([1:n],:),order);
CLR=g([1:n],:).^2;CLR=[CLR;[0:0.01:1]' [0:0.01:1]' [0:0.01:1]'];
tensor=zeros(o,1);

if id==2 & nargin==4
    img_min=min(image_data(:));
    img_max=max(image_data(:));
    img_range=img_max-img_min;
    if img_range==0
        img_range=1;
    end
elseif id==3 & nargin==4
    img_min=min(min(image_data(:,:)));
    img_max=max(max(image_data(:,:)));
    img_range=img_max-img_min;
    if img_range==0
        img_range=1;
    end
end


for i=1:nx
    for j=1:ny
        if id==1
            tensor(:)=UniqueTensorCoefficients(:);
        elseif id==2
            tensor(:)=UniqueTensorCoefficients(:,i);
            if nargin==4
                img_value=image_data(i);
            end
        elseif id==3
            tensor(:)=UniqueTensorCoefficients(:,i,j);
            if nargin==4
                img_value=image_data(i,j);
            end
        end
        S=G*tensor;[mx,c]=max(S);
        if aniso~=-1
            mn=min(S);S=max(S-mn-aniso*(mx-mn),0);
        end
        if normalize==1
            S=S/max(S);
        end
        S=[S;S];
        trimesh(TRI,XYZ(:,1).*S+((i-1)*(2*delta)),  XYZ(:,2).*S+((j-1)*(2*delta)),  XYZ(:,3).*S,  round(c)*ones(size(XYZ,1),1),'CDataMapping','direct','FaceColor','interp','FaceLighting','phong','EdgeColor','none');
        if i==1 & j==1
            hold
        end
        if (nargin==4) & (id==2 | id==3 )
            trimesh([1 2 3;1 4 3],(i-1+[-0.5;+0.5;+0.5;-0.5])*(2*delta),(j-1+[-0.5;-0.5;+0.5;+0.5])*(2*delta),-ones(4,1),n+1+round(99*((img_value-img_min)/img_range))*ones(4,1),'CDataMapping','direct','FaceColor','interp','FaceLighting','none','EdgeColor','none');
        end
    end
end
axis equal
colormap(CLR);
view([0 90]);
set(gca,'GridLineStyle','none')
set(gca,'XTick',[])
set(gca,'YTick',[])
set(gca,'ZTick',[])
%lighting phong
light('Position',[0 0 1],'Style','infinite','Color',[ 1.000 1.00 1.00]);
hold

fprintf(1,'\nIf you use this software please cite the following papers:\n');
fprintf(1,'1) A. Barmpoutis et al. "Regularized Positive-Definite Fourth-Order Tensor Field\n');
fprintf(1,'Estimation from DW-MRI", NeuroImage, Vol. 45(1 sup.1), Page(s): 153-162\n');
fprintf(1,'2) A. Barmpoutis and B.C. Vemuri, "A Unified Framework for Estimating Diffusion Tensors\n'); 
fprintf(1,'of any order with Symmetric Positive-Definite Constraints",\n');
fprintf(1,'In the Proceedings of ISBI, 2010\n');