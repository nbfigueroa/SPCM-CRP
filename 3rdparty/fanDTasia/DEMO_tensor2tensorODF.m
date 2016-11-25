function TensorODF=DEMO_tensor2tensorODF()
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
% This demo script shows how to compute a Cartesian Tensor ODF from a given Higher-order 
% Diffusion Tensor (Section 2.3 from MICCAI'10 paper). The method guarantees that the 
% estimated Tensor-ODF is positive-definite or at least positive semi-definite. Here the 
% given demo high-order tensor corresponds to a 2-fiber crossing and it was estimated 
% using the DT4_Estimation.m developed by Angelos Barmpoutis.
%
%-USE--------------------------------------------------------------------------------
% TensorODF=DEMO_tensor2tensorODF;
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

%Here is a sample 4th order tensor that corresponds to a synthesized 2-fiber crossing.
%The ground truth underlying fiber orientations are [cos(100*pi/180) sin(100*pi/180) 0] and [cos(20*pi/180) sin(20*pi/180) 0] 
%It was estimated using the DT4_Estimation.m developed by Angelos Barmpoutis.
tensor=[0.00033117;-0.00000545;0.00159264;0.00000564;0.00093217;-0.00001132;0.00026464;0.00004150;0.00114423;0.00143355;0.00003415;0.00273919;0.00000615;-0.00059943;0.00077108];

UnitVectors;
GradientOrientations=g([1:81],:); %Define a set of new Gradient Orientations
bvalue=1500; %Define a new b-value


%Construct all possible monomials of a specific order
G=constructMatrixOfMonomials(GradientOrientations, order); %computes G from section 5.1 (ISBI'10)
%Construct set of polynomial coefficients C
C=constructSetOf321Polynomials(order)'; %computes C from section 5.1 (ISBI'10)
BG=constructMatrixOfIntegrals(GradientOrientations, order, 100);
B=BG*C; %computes B from section 2.2 (MICCAI'10)

%The next lines implement the core of the algorithm. 
%It should be repeated for every voxel in the DW-MRI dataset (here there is only 1 voxel).
start_time=cputime;
x=lsqnonneg(B, exp(-bvalue*G*tensor));
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