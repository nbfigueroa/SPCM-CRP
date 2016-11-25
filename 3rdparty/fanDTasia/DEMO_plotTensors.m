function DEMO_plotTensors()
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
% This demo script shows how to use plotTensors.m in order to plot a 2D field of 
% 3D even-order tensors as spherical functions. The 3D tensors must be in the form 
% of a list of unique tensor coefficients using the same ordering as defined by the
% function printTensor. The field can contain either a single tensor, or a row of 
% tensors or a 2D field of tensors. Each tensor is colored according to the orientation 
% of maximum value. The orientation components X,Y,Z are assigned to the color 
% components R,G,B.
%
%-USE--------------------------------------------------------------------------------
% DEMO_plotTensors;
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


%we open a demo tensor field
fprintf(1,'Do you want to open:\n');
fprintf(1,' 1. a 2nd-order tensor field.\n');
fprintf(1,' 2. a 4th-order tensor field.\n');
a1=input('Answer [1,2]:');

if a1==1
    D=openFDT('2nd_order_tensors.fdt');
elseif a1==2
    D=openFDT('4th_order_tensors.fdt');
else
    error('Your answer must be either 1 or 2.');
end
    
fprintf(1,'What do you want to do?\n');
fprintf(1,' 1. Plot a single tensor.\n');
fprintf(1,' 2. Plot a row of tensors.\n');
fprintf(1,' 3. Plot a 2D field of tensors.\n');
a2=input('Answer [1,2,3]:');

if a2==1
    plotTensors(D(:,[22],[22]),0.9,[321 1]);
elseif a2==2
    plotTensors(D(:,[10:22],[22]),0.9, [321 1]);
elseif a2==3
    plotTensors(D(:,[1:2:32],[1:2:32]),0.9, [321 1]);
end