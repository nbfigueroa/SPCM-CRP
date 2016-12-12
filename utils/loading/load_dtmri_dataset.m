function [sigmas, true_labels] = load_dtmri_dataset( data_path, type, display, randomize )
% Loads a a real DT-MRI Dataset or create a synthetic one, each tensor is
% 3x3 DT-MRI size is real: 32x32 and synthetic real: 32x32
% This code is heavily based on 

switch type
    
    case 'synthetic'
        
        % Specify acquisition parameters (b-values and gradients)
        UnitVectors;
        gnb = 21;
        GradientOrientations=[1 0 0;g([1:gnb],:)];
        b_value=[10;ones(gnb,1)*1500];
        
        % Define Fiber orientations and simulate DW-MR signal response
        S=ones(32,32,1,size(GradientOrientations,1));
        for i=1:size(GradientOrientations,1)
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
        
        % Image Title
        dti_title = 'Synthetic DT-MRI';
        
        % Cluster Step for Fractional Anisotropy
        cl_step = 0.075;

        
    case 'real'
        
        % Load Real DW-MRI Image and Parameters
        S = openFDT(strcat(data_path,'./fandtasia_demo/fandtasia_demo.fdt'));
        params = textread(strcat(data_path,'./fandtasia_demo/fandtasia_demo.txt'));
        
        % Extract and plot Gradient orientations
        GradientOrientations=params(:,[1:3]);
        b_value=params(:,4);
        g=GradientOrientations([2:47],:);
        
        % Image Title
        dti_title = 'REAL DT-MRI of Rat Hippocampi';  
        
        % Cluster Step for Fractional Anisotropy
        cl_step = 0.25;
        
end


% Estimate DTI from Gradient Orientation and b_value
G=constructMatrixOfMonomials(GradientOrientations, 2);
C=constructSetOf81Polynomials(2)';
P=G*C;P=[-diag(b_value)*P ones(size(GradientOrientations,1),1)];
DTI=zeros(3,3,size(S,1),size(S,2));S0=zeros(size(S,1),size(S,2));
for i=1:size(S,1)
    for j=1:size(S,2)
        y=log(squeeze(S(i,j,1,:)));
        x=lsqnonneg(P, y);
        T = C * x([1:81]);
        UniqueTensorCoefficients(:,i,j)=T;
        DTI(:,:,i,j)=[T(6) T(5)/2 T(4)/2
            T(5)/2 T(3) T(2)/2
            T(4)/2 T(2)/2 T(1)];
        S0(i,j)=exp(x(82));
    end
end


% Compute Mean Diffusity
mean_diffusity = zeros(size(DTI,3),size(DTI,4));
for i=1:size(DTI,3)
    for j=1:size(DTI,4)
        mean_diffusity(j,i)=trace(DTI(:,:,i,j))/3;
    end
end


% Compute Fractional Anisotropy
frac_anisotropy = zeros(size(DTI,3),size(DTI,4));
for i=1:size(DTI,3)
    for j=1:size(DTI,4)
        [eigenvectors,l] = eig(DTI(:,:,i,j));
        m=(l(1,1)+l(2,2)+l(3,3))/3;
        frac_anisotropy(j,i)=sqrt(3/2)*sqrt((l(1,1)-m)^2+(l(2,2)-m)^2+(l(3,3)-m)^2)/sqrt(l(1,1)^2+l(2,2)^2+l(3,3)^2);
    end
end

minFA = min(frac_anisotropy(:));
maxFA = max(frac_anisotropy(:));

% Cluster Ranges
cl_ranges = [minFA:cl_step:maxFA];

% Create Tensor Dataset to Cluster
k = 1; true_labels = zeros(1,size(DTI,3)*size(DTI,4));
for i=1:size(DTI,3) % row
    for j=1:size(DTI,4) % column
        
        % Create Sigma Structure from Diffusion Tensors
        sigmas{k}   = DTI(:,:,i,j);
        
        % Compute FA values
        [eigenvectors,l] = eig(DTI(:,:,i,j));
        m=(l(1,1)+l(2,2)+l(3,3))/3;
        FA=sqrt(3/2)*sqrt((l(1,1)-m)^2+(l(2,2)-m)^2+(l(3,3)-m)^2)/sqrt(l(1,1)^2+l(2,2)^2+l(3,3)^2);
                
        % Create fake clusters from FA
        true_labels(k) = sum((FA < cl_ranges) == 0);
        k = k + 1;
    end
end



if display==1

    % Plot DT-MRI Image with tensors
    figure('Color',[1 1 1]);
    plotDTI(DTI,0.002);
    title(dti_title)

    
    % Plot Fractional Anisotropy of Diffusion Tensors
    figure('Color',[1 1 1]);
    imagesc(flipud(frac_anisotropy))
    colormap(pink)
    colorbar
    axis square
    title('Fractional Anisotropy of Diffusion Tensors')
    
    % Generate labels from Fractional Anisotropy Value
    figure('Color',[1 1 1]);
    imagesc(flipud(reshape(true_labels,[size(DTI,3) size(DTI,4)])))
    colormap(pink)
    colorbar
    axis square
    title('True Cluster Labels of Diffusion Tensors')
    
end

end