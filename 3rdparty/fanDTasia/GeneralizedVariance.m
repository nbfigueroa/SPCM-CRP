function variance=GeneralizedVariance(TensorCoefficients)
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
% A. Barmpoutis et al. "Regularized Positive-Definite Fourth-Order Tensor Field
% Estimation from DW-MRI", NeuroImage, Vol. 45(1 sup.1), Page(s): 153-162, 2009 
%
%-DESCRIPTION------------------------------------------------------------------------
% This function computes the Generalized Variance from a 4th-order tensor. The 15
% coefficients of the 4th-order tensor are given as an input to the function. The
% implementation is based on the 4th-order tensor distance defined in A.Barmpoutis et
% al. NeuroImage 2009.
%
%-USE--------------------------------------------------------------------------------
% variance=GeneralizedVariance(TensorCoefficients);
%
% TensorCoefficients: is a 15-dimensional vector that contains the coefficients of a
% 4th-order tensor. The ordering of the coefficients is the same as the one described
% by the function printTensor(TensorCoefficients).
%
% variance: is the computed generalized variance
%
%-DISCLAIMER-------------------------------------------------------------------------
% You can use this source code for non commercial research and educational purposes 
% only without licensing fees and is provided without guarantee or warrantee expressed
% or implied. You cannot repost this file without prior written permission from the 
% authors. If you use this software please cite the following work:
% A. Barmpoutis et al. "Regularized Positive-Definite Fourth-Order Tensor Field
% Estimation from DW-MRI", NeuroImage, Vol. 45(1 sup.1), Page(s): 153-162, 2009
%
%-AUTHOR-----------------------------------------------------------------------------
% Angelos Barmpoutis, PhD
% Computer and Information Science and Engineering Department
% University of Florida, Gainesville, FL 32611, USA
% abarmpou at cise dot ufl dot edu
%------------------------------------------------------------------------------------

dd=TensorCoefficients([15 5 1 12 3 10 11 8 7 14 13 9 4 6 2]);
a=1/9;%0.1100;
b=1/105;%0.0095;
c=1/63;
d=1/315;%94/296

sum=d*(dd(1)+dd(2)+dd(3)+dd(4)+dd(5)+dd(6))^2+...
(b-d)*(dd(1)+dd(2)+dd(3))^2+...
(c-d)*((dd(1)+dd(4))^2+(dd(1)+dd(6))^2)+...
(c-d)*((dd(2)+dd(4))^2+(dd(2)+dd(5))^2)+...
(c-d)*((dd(3)+dd(5))^2+(dd(3)+dd(6))^2)+...
(a-b-2*(c-d))*(dd(1)^2+dd(2)^2+dd(3)^2)+...
(b-d-2*(c-d))*(dd(4)^2+dd(5)^2+dd(6)^2)+...
d*(dd(7)+dd(13)+dd(15))^2+...
+d*(dd(8)+dd(11)+dd(14))^2+...
+d*(dd(9)+dd(10)+dd(12))^2+...
+(b-d)*(dd(10)+dd(12))^2+...
+(b-d)*(dd(11)+dd(14))^2+...
+(b-d)*(dd(13)+dd(15))^2+...
+(c-b)*(dd(10)^2+dd(11)^2+dd(12)^2+dd(13)^2+dd(14)^2+dd(15)^2);

tmp=(dd(1)+dd(2)+dd(3)+(dd(4)+dd(5)+dd(6))/3)/5;
		 
variance=(sum/(tmp*tmp)-1)/9;