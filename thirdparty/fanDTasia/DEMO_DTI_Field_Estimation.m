function D=DEMO_DTI_Field_Estimation()
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
% A. Barmpoutis and B.C. Vemuri, "A Unified Framework for Estimating Diffusion Tensors 
% of any order with Symmetric Positive-Definite Constraints", 
% In the Proceedings of ISBI, 2010
%
%-DESCRIPTION------------------------------------------------------------------------
% This demo script shows how to compute a field of Diffusion Tensors from a given DW-MRI
% (Diffusion-Weighted MRI) dataset. The method guarantees that the estimated tensors
% are positive-definite or at least positive semi-definite. Here the given demo dataset 
% is simulated using the DW-MRI simulator developed by Angelos Barmpoutis.
%
%-USE--------------------------------------------------------------------------------
% D=DEMO_DTI_Field_Estimation;
%
% D: is the computed 2D Diffusion Tensor field in the form of a 3x3xSizeXxSizeY matrix
%
%-DISCLAIMER-------------------------------------------------------------------------
% You can use this source code for non commercial research and educational purposes 
% only without licensing fees and is provided without guarantee or warrantee expressed
% or implied. You cannot repost this file without prior written permission from the 
% authors. If you use this software please cite the following work:
% A. Barmpoutis and B.C. Vemuri, "A Unified Framework for Estimating Diffusion Tensors 
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
%This dataset was synthesized using the DW-MRI simulator developed by Angelos Barmpoutis.
fprintf(1,'Simulating DW-MRI dataset...');
UnitVectors;
GradientOrientations=[1 0 0;g([1:21],:)];
b_value=[10;ones(21,1)*1500];
S=ones(32,32,1,size(GradientOrientations,1));
for i=2:size(GradientOrientations,1)
for x=1:32
   for y=1:32
      f1_flag=0;
      f2_flag=0;
      if x*x+y*y>16*16 & x*x+y*y<32*32
         v=[y/x -1 0];v=v/sqrt(v*v');
         fiber_orientation1=v;f1_flag=1;
      end
      if x<y+10 & x>y-10
         fiber_orientation2=[sqrt(2)/2 sqrt(2)/2 0];f2_flag=1;
      end
      if f1_flag==0 & f2_flag==1
         fiber_orientation1=fiber_orientation2;
      elseif f1_flag==1 & f2_flag==0
         fiber_orientation2=fiber_orientation1;
      elseif f1_flag==0 & f2_flag==0
         fiber_orientation1=[0 0 1];fiber_orientation2=[0 0 1];
      end
      S(x,y,1,i)=S(x,y,1,1)*(SimulateDWMRI(fiber_orientation1,GradientOrientations(i,:))+ SimulateDWMRI(fiber_orientation2,GradientOrientations(i,:)))/2;
   end
end
end
fprintf(1,'done\n');

%Construct all possible monomials of a specific order
G=constructMatrixOfMonomials(GradientOrientations, order); %computes G from section 5.1 (ISBI'10)
%Construct set of polynomial coefficients C
C=constructSetOf81Polynomials(order)'; %computes C from section 5.1 (ISBI'10)
P=G*C;
P=[-diag(b_value)*P ones(size(GradientOrientations,1),1)];

%The next lines implement the core of the algorithm. 
%It should be repeated for every voxel in the DW-MRI dataset (here there is only 1 voxel).
start_time=cputime;

for i=1:size(S,1)
    for j=1:size(S,2)

y=squeeze(log(S(i,j,1,:)));
x=lsqnonneg(P, y);
UniqueTensorCoefficients = C * x([1:81]);

%Put the result in the form of a 3x3 matrix
T=[UniqueTensorCoefficients(6) UniqueTensorCoefficients(5)/2 UniqueTensorCoefficients(4)/2
   UniqueTensorCoefficients(5)/2 UniqueTensorCoefficients(3) UniqueTensorCoefficients(2)/2
   UniqueTensorCoefficients(4)/2 UniqueTensorCoefficients(2)/2 UniqueTensorCoefficients(1)];

D(:,:,i,j)=T;
TensorCoefficients(:,i,j)=UniqueTensorCoefficients;
end
end

end_time=cputime;
fprintf(1,'\nTotal estimation time: %.0f ms\n\n',(end_time-start_time)*1000);

% If you want to plot a tensor or a tensor field as an ellipsoid or a field of ellipsoids
% you have to download the plotDTI.m function developed by Angelos Barmpoutis, Ph.D.
% and then uncomment the following line.
%
% plotDTI(D,0.002);
%
% or if you want to plot a tensor or a tensor field as spherical functions
% you have to download the plotTensors.m function developed by Angelos Barmpoutis, Ph.D.
% and then uncomment the following line.
%
% plotTensors(TensorCoefficients,1,[321 1]);