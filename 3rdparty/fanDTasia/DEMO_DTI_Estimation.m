function D=DEMO_DTI_Estimation()
%-fanDTasia ToolBox------------------------------------------------------------------
% This Matlab script is part of the fanDTasia ToolBox: a Matlab library for Diffusion 
% Weighted MRI (DW-MRI) Processing, Diffusion Tensor (DTI) Estimation, High-order 
% Diffusion Tensor Analysis, Tensor ODF estimation, Visualization and more.
%
% A Matlab Tutorial on DW-MRI can be found in:
% http://www.cise.ufl.edu/~abarmpou/lab/fanDTasia/tutorial.php
%
%-CITATION---------------------------------------------------------------------------
% If you use this software please cite the following papers:
% 1) A. Barmpoutis, B. C. Vemuri, T. M. Shepherd, and J. R. Forder "Tensor splines for 
% interpolation and approximation of DT-MRI with applications to segmentation of 
% isolated rat hippocampi", IEEE TMI: Transactions on Medical Imaging, Vol. 26(11), 
% pp. 1537-1546 
% 2) A. Barmpoutis and B.C. Vemuri, "A Unified Framework for Estimating Diffusion Tensors 
% of any order with Symmetric Positive-Definite Constraints", In Proc. of ISBI, 2010.
%
%-DESCRIPTION------------------------------------------------------------------------
% This demo script shows how to compute a Diffusion Tensor (DTI) from a given DW-MRI
% (Diffusion-Weighted MRI) dataset. The method guarantees that the estimated tensor
% is positive-definite or at least positive semi-definite. Here the given demo dataset 
% consists of 1 voxel, 21 gradient directions, and it was synthesized using the on-line 
%  DW-MRI simulator, which is available through the web-site of Angelos Barmpoutis.
%
%-USE--------------------------------------------------------------------------------
% D=DEMO_DTI_Estimation;
%
% D: is the computed Diffusion Tensor in the form of a 3x3 matrix
%
%-DISCLAIMER-------------------------------------------------------------------------
% You can use this source code for non commercial research and educational purposes 
% only without licensing fees and is provided without guarantee or warrantee expressed
% or implied. You cannot repost this file without prior written permission from the 
% authors. If you use this software please cite the following papers:
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


order=2;%In standard DTI the order is 2

%Here is a sample demo DW-MRI dataset of 1 voxel (21 gradient orientations)
%This dataset was synthesized using the on-line DW-MRI simulator, which is available through the web-site of Angelos Barmpoutis.
%To generate a synthetic DW-MRI dataset in MATLAB please download SimulateDWMRI.m developed by Angelos Barmpoutis.
b_value=[1500;1500;1500;1500;1500;1500;1500;1500;1500;1500;1500;1500;1500;1500;1500;1500;1500;1500;1500;1500;1500];
S=[0.39481;0.43774;0.12879;0.31532;0.31744;0.36900;0.59490;0.35280;0.36880;0.44046;0.48088;0.17118;0.22700;0.34665;0.26000;0.25414;0.21642;0.34456;0.26625;0.20723;0.30364];
S0=1.0;
GradientOrientations=[0.1639 0.5115 0.8435;0.1176 -0.5388 0.8342;0.5554 0.8278 -0.0797;-0.4804 0.8719 0.0948;0.9251 -0.0442 0.3772;0.7512 -0.0273 -0.6596;0.1655 -0.0161 0.9861;0.6129 -0.3427 0.7120;0.6401 0.2747 0.7175;-0.3724 -0.3007 0.8780;-0.3451 0.3167 0.8835;0.4228 0.7872 0.4489;0.0441 0.9990 0.0089;-0.1860 0.8131 0.5515;0.8702 0.4606 0.1748;-0.7239 0.5285 0.4434;-0.2574 -0.8032 0.5372;0.3515 -0.8292 0.4346;-0.7680 -0.4705 0.4346;0.8261 -0.5384 0.1660;0.9852 -0.0420 -0.1660];

%Construct all possible monomials of a specific order
G=constructMatrixOfMonomials(GradientOrientations, order); %computes G from section 5.1 (ISBI'10)
%Construct set of polynomial coefficients C
C=constructSetOf321Polynomials(order)'; %computes C from section 5.1 (ISBI'10)
P=G*C;
P=-diag(b_value)*P;

%The next lines implement the core of the algorithm. 
%It should be repeated for every voxel in the DW-MRI dataset (here there is only 1 voxel).
start_time=cputime;
y=log(S/S0);
x=lsqnonneg(P, y);
UniqueTensorCoefficients = C * x;
end_time=cputime;
fprintf(1,'\nTotal estimation time: %.0f ms\n\n',(end_time-start_time)*1000);

%Print out the result
printTensor(UniqueTensorCoefficients,order);

%Put the result in the form of a 3x3 matrix
D=[UniqueTensorCoefficients(6) UniqueTensorCoefficients(5)/2 UniqueTensorCoefficients(4)/2
   UniqueTensorCoefficients(5)/2 UniqueTensorCoefficients(3) UniqueTensorCoefficients(2)/2
   UniqueTensorCoefficients(4)/2 UniqueTensorCoefficients(2)/2 UniqueTensorCoefficients(1)]

% If you want to plot a tensor or a tensor field as an ellipsoid or a field of ellipsoids
% you have to download the plotDTI.m function developed by Angelos Barmpoutis, Ph.D.
% and then uncomment the following line.
%
% plotDTI(D);
%
% or if you want to plot a tensor or a tensor field as spherical functions
% you have to download the plotTensors.m function developed by Angelos Barmpoutis, Ph.D.
% and then uncomment the following line.
%
% plotTensors(UniqueTensorCoefficients,1,[321 1]);

fprintf(1,'\nIf you use this software for DTI estimation please cite the following papers:\n');
fprintf(1,'1) A. Barmpoutis, B. C. Vemuri, T. M. Shepherd, and J. R. Forder "Tensor splines for\n');
fprintf(1,'interpolation and approximation of DT-MRI with applications to segmentation of\n');
fprintf(1,'isolated rat hippocampi", IEEE TMI: Transactions on Medical Imaging, Vol. 26(11),\n');
fprintf(1,'pp. 1537-1546\n');
fprintf(1,'2) A. Barmpoutis and B.C. Vemuri, "A Unified Framework for Estimating Diffusion Tensors\n'); 
fprintf(1,'of any order with Symmetric Positive-Definite Constraints",\n');
fprintf(1,'In the Proceedings of ISBI, 2010\n');