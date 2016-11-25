function TensorODF=DEMO_tensorODF_Estimation()
%-fanDTasia ToolBox------------------------------------------------------------------
% This Matlab script is part of the fanDTasia ToolBox: a Matlab library for Diffusion 
% Weighted MRI (DW-MRI) Processing, Diffusion Tensor (DTI) Estimation, High-order 
% Diffusion Tensor Analysis, Tensor ODF estimation, Visualization and more.
%
% A Matlab Tutorial on DW-MRI can be found in:
% http://www.cise.ufl.edu/~abarmpou/lab/fanDTasia/tutorial.php
%
%-CITATION---------------------------------------------------------------------------
% If you use this software please cite the following papers on Cartesian Tensors:
% 1) Y. Weldeselassie et al."Symmetric Positive-Definite Cartesian Tensor Orientation
% Distribution Functions (CT-ODF)", In the Proceedings of MICCAI, 2010
% 2) A. Barmpoutis and B.C. Vemuri, "A Unified Framework for Estimating Diffusion 
% Tensors of any order with Symmetric Positive-Definite Constraints", 
% In the Proceedings of ISBI, 2010
%
%-DESCRIPTION------------------------------------------------------------------------
% This demo script shows how to compute a Cartesian Tensor ODF from a given DW-MRI 
% (Diffusion-Weighter MRI) dataset (Section 2.2 from MICCAI'10 paper). The method 
% guarantees that the estimated Tensor-ODF is positive-definite or at least positive 
% semi-definite. Here the given demo dataset consists of 1 voxel, 21 gradient directions,
% and it corresponds to a 2-fiber crossing. The dataset was synthesized using the 
% on-line DW-MRI simulator, which is available through the web-site of Angelos Barmpoutis.
%
%-USE--------------------------------------------------------------------------------
% TensorODF=DEMO_tensorODF_Estimation;
%
% TensorODF: is a vector with the computed Coefficients of the Tensor-ODF
%
%-DISCLAIMER-------------------------------------------------------------------------
% You can use this source code for non commercial research and educational purposes 
% only without licensing fees and is provided without guarantee or warrantee expressed
% or implied. You cannot repost this file without prior written permission from the 
% authors. If you use this software please cite the following papers:
% 1) Y. Weldeselassie et al."Symmetric Positive-Definite Cartesian Tensor Orientation
% Distribution Functions (CT-ODF)", In the Proceedings of MICCAI, 2010
% 2) A. Barmpoutis and B.C. Vemuri, "A Unified Framework for Estimating Diffusion 
% Tensors of any order with Symmetric Positive-Definite Constraints", 
% In the Proceedings of ISBI, 2010
%
%-AUTHOR-----------------------------------------------------------------------------
% Angelos Barmpoutis, PhD
% Computer and Information Science and Engineering Department
% University of Florida, Gainesville, FL 32611, USA
% abarmpou at cise dot ufl dot edu
%------------------------------------------------------------------------------------


order=4;%In this demo we use a 4th-order tensor example

%Here is a sample demo DW-MRI dataset of 1 voxel (21 gradient orientations), which corresponds to a 2-fiber crossing.
%The ground truth underlying fiber orientations are [cos(100*pi/180) sin(100*pi/180) 0] and [cos(20*pi/180) sin(20*pi/180) 0] 
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
BG=constructMatrixOfIntegrals(GradientOrientations, order, 100); 
B=BG*C; %computes B from section 2.2 (MICCAI'10)

%The next lines implement the core of the algorithm. 
%It should be repeated for every voxel in the DW-MRI dataset (here there is only 1 voxel).
start_time=cputime;
x=lsqnonneg(B, S/S0);
TensorODF = C * x;
end_time=cputime;
fprintf(1,'\nTotal estimation time: %.0f ms\n\n',(end_time-start_time)*1000);

%Print out the result
printTensor(TensorODF,order);

% If you want to plot a tensor or a tensor field as spherical functions
% you have to download the plotTensors.m function developed by Angelos Barmpoutis, Ph.D.
% and then uncomment the following line.
%
% plotTensors(TensorODF,1,[321 1]);

fprintf(1,'\nDo you want to compute the critical orientations (maxima, minima, etc.)\n');
fprintf(1,'of the estimated TensorODF? For YES type 1, for NO type 0.\n');
a=input('Answer [1,0] :');

if a==1
	%The maxima of the TensorODF correspond to the underlying distinct fiber orientations.
	%The critical orientation (i.e. maxima, minima, etc.) of a 4th-order tensorODF can be
	%computed using the following:
	[v,l]=eig_dt4(TensorODF);v
	
	%Comparison with the ground truth underlying fiber orientations
	d1=acos(v(1,:)*[cos(100*pi/180);sin(100*pi/180);0])*180/pi;d1=min(abs(d1),abs(180-d1));
	d2=acos(v(2,:)*[cos(20*pi/180);sin(20*pi/180);0])*180/pi;d2=min(abs(d2),abs(180-d2));
	error1=(d1+d2)/2;
	d1=acos(v(2,:)*[cos(100*pi/180);sin(100*pi/180);0])*180/pi;d1=min(abs(d1),abs(180-d1));
	d2=acos(v(1,:)*[cos(20*pi/180);sin(20*pi/180);0])*180/pi;d2=min(abs(d2),abs(180-d2));
	error2=(d1+d2)/2;
	error=min(error1,error2);
	fprintf(1,'\nThe fiber orientation error from the ground truth is %.2f degrees.\n',error);
end

fprintf(1,'\nIf you use this software please cite the following papers on Cartesian Tensors:\n');
fprintf(1,'1) Y. Weldeselassie et al."Symmetric Positive-Definite Cartesian Tensor Orientation\n');
fprintf(1,'Distribution Functions (CT-ODF)", In the Proceedings of MICCAI, 2010\n');
fprintf(1,'2) A. Barmpoutis and B.C. Vemuri, "A Unified Framework for Estimating Diffusion Tensors\n'); 
fprintf(1,'of any order with Symmetric Positive-Definite Constraints",\n');
fprintf(1,'In the Proceedings of ISBI, 2010\n');