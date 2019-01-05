function[sigmas, true_labels, dataset_name] = load_SPD_dataset(choosen_dataset, pkg_dir, display, randomize, varargin)

data_path = strcat(pkg_dir,'/datasets/');      
switch choosen_dataset
    case 1
        % 1) Toy 3D dataset, 5 Samples, 2 clusters (c1:3, c2:2)
        % This function loads the 3-D ellipsoid dataset used to generate Fig. 3, 4 
        % and 5 from Section 4 and the results in Section 7 in the accompanying paper.
         dataset_name = 'Toy 3D';
        [sigmas, true_labels] = load_toy_dataset(0, display, randomize);
        
    case 2
        % 2)  Toy 6D dataset, 60 Samples, 3 clusters (c1:20, c2:20, c3: 20)
        % This function loads the 6-D ellipsoid dataset used to generate Fig. 6 and 
        % from Section 4 and the results in Section 8 in the accompanying paper.
        dataset_name = 'Toy 6D';
        [sigmas, true_labels] = load_toy_dataset(1, display, randomize);
        
    case 3
        % 3) Real 6D dataset, task-ellipsoids, 105 Samples, 3 clusters 
        % Cluster Distibution: (c1:63, c2:21, c3: 21)
        % This function loads the 6-D task-ellipsoid dataset used to evaluate this 
        % algorithm in Section 8 of the accompanying paper.
        %
        % Please cite the following paper if you make use of this data:
        % El-Khoury, S., de Souza, R. L. and Billard, A. (2014) On Computing 
        % Task-Oriented Grasps. Robotics and Autonomous Systems. 2015 
        
        dataset_name = 'Real 6D (Task-Ellipsoids)';        
        [sigmas, true_labels] = load_task_dataset(data_path, randomize);
        
    case 4                
        % 4a) Toy 3D dataset, Diffusion Tensors from Synthetic Dataset, 1024 Samples
        % Cluster Distibution: 4 clusters (each cluster has 10 samples)
        % This function will generate a synthetic DW-MRI (Diffusion Weighted)-MRI
        % This is done following the "Tutorial on Diffusion Tensor MRI using
        % Matlab" by Angelos Barmpoutis, Ph.D. which can be found in the following
        % link: http://www.cise.ufl.edu/~abarmpou/lab/fanDTasia/tutorial.php
        %
        % To run this function you should download fanDTasia toolbox in the
        % ~/SPCM-CRP/3rdParty directory, this toolbox is also provided in
        % the tutorial link.
        
        % clc; clear all; close all;
        type = 'synthetic'; 
        [sigmas, true_labels] = load_dtmri_dataset( data_path, type, display, randomize );
        dataset_name = 'Synthetic DT-MRI';
        
    case 5
        % 4b) Real 3D dataset, Diffusion Tensors from fanTDasia Dataset, 1024 Samples
        % Cluster Distibution: 4 clusters (each cluster has 10 samples)
        % This function loads a 3-D Diffusion Tensor Image from a Diffusion
        % Weight MRI Volume of a Rat's Hippocampus, the extracted 3D DTI is used
        % to evaluate this algorithm in Section 8 of the accompanying paper.
        %untitled
        % To load and visualize this dataset, you must download the dataset files 
        % in the  ~/SPCM-CRP/data directory. These are provided in the online 
        % tutorial on Diffusion Tensor MRI in Matlab:
        % http://www.cise.ufl.edu/~abarmpou/lab/fanDTasia/tutorial.php
        %
        % One must also download the fanDTasia toolbox in the ~/SPCM-CRP/3rdParty
        % directory, this toolbox is also provided in this link.

        % clc; clear all; close all;
        type = 'real';  dataset_name = 'Real DT-MRI';
        [sigmas, true_labels] = load_dtmri_dataset( data_path, type, display, randomize );
        
    case 6 
        
        % TODO: REDO THIS DATASET WITH THE ROTATED IMAGE CLUSTERS and 18x18 Cov.Matrx
        % 5) Real 400D dataset, Covariance Features from ETH80 Dataset, 40 Samples
        % Cluster Distibution: 8 classes/clusters (each cluster has 10 samples)
        % This function loads the 400-D ETH80 Covariance Feature dataset 
        % used to evaluate this algorithm in Section 8 of the accompanying paper.
        %
        %
        % You must download this dataset from the following link: 
        % http://ravitejav.weebly.com/classification-of-manifold-features.html
        % and export it in the ~/SPCM-CRP/data directory
        %
        % Please cite the following paper if you make use of these features:
        % R. Vemulapalli, J. Pillai, and R. Chellappa, “Kernel Learning for Extrinsic 
        % Classification of Manifold Features”, CVPR, 2013.       
        [sigmas, true_labels] = load_eth80_dataset(data_path, split, randomize);
        
        
end
        
if nargin == 5    
    % input
    sample_ratio = varargin{1};
    
    % sampling
    classes         = unique(true_labels);
    idx_per_class   = [];
    new_true_labels = [];
    for c=1:length(classes)
        idx_per_class{c} = find(true_labels == classes(c));
        new_idx_class{c} = randsample(idx_per_class{c},ceil(length(idx_per_class{c})*sample_ratio));
        new_true_labels = [new_true_labels ones(1,length(new_idx_class{c}))*classes(c)];
    end
    new_sigmas = []; idx = 1;
    for c=1:length(new_idx_class)
        new_idx = new_idx_class{c};
        for i=1:length(new_idx)
            new_sigmas{1,idx} = sigmas{new_idx(i)};
            idx = idx + 1;
        end
    end
    
    % output
    true_labels = new_true_labels;
    sigmas = new_sigmas;
end

end