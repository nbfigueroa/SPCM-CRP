function[sigmas, true_labels, dataset_name] = load_SPD_dataset(choosen_dataset, pkg_dir, display, randomize, varargin)

data_path = strcat(pkg_dir,'/datasets/');      
df = 100;

switch choosen_dataset
    case 0
        % 1) Non-deformed Ellipsoids (3D) / (40 Samples  c1:40)
        % Load an SPD simulation (from Section 1 and 3)
        sim_type = 1;
        % 1: linear ellipsoid with isotropic scalings + rotations
        % 2: linear ellipsoid with anisotropic scalings + rotations
        % 3: simulation 1 + 2
        % 4: 3D wishart samples from linear, spherical and planar ellipsoids
        % 5: 6D wishart samples 4 different covariance matrices
        [Mu_test, sigma_test, true_labels, dataset_name] = load_SPD_simulations(sim_type, df);
        % Build new Sigmas and compute metrics
        sigmas = [];
        for k=1:length(Mu_test)
            sigmas{k} = sigma_test(:,:,k);
        end
        
    case 1
        % 1) Non-deformed+Deformed Ellipsoids (3D) / (80 Samples  c1:3,  c2:3,  c2:3  c4:)
        % Load an SPD simulation (from Section 1 and 3)
        sim_type = 3;
        % 1: linear ellipsoid with isotropic scalings + rotations
        % 2: linear ellipsoid with anisotropic scalings + rotations
        % 3: simulation 1 + 2
        % 4: 3D wishart samples from linear, spherical and planar ellipsoids
        % 5: 6D wishart samples 4 different covariance matrices
        [Mu_test, sigma_test, true_labels, dataset_name] = load_SPD_simulations(sim_type, df);
        % Build new Sigmas and compute metrics
        sigmas = [];
        for k=1:length(Mu_test)
            sigmas{k} = sigma_test(:,:,k);
        end
        
    case 2
        % 2)  SPD sampled from Wishart      (3D) / (120 Samples c1:40, c2:40, c2:40)
        % Load an SPD simulation (from Section 1 and 3)
        sim_type = 4;
        % 1: linear ellipsoid with isotropic scalings + rotations
        % 2: linear ellipsoid with anisotropic scalings + rotations
        % 3: simulation 1 + 2
        % 4: 3D wishart samples from linear, spherical and planar ellipsoids
        % 5: 6D wishart samples 4 different covariance matrices
        [Mu_test, sigma_test, true_labels, dataset_name] = load_SPD_simulations(sim_type, df);
        % Build new Sigmas and compute metrics
        sigmas = [];
        for k=1:length(Mu_test)
            sigmas{k} = sigma_test(:,:,k);
        end
        
    case 3
        % 2)  SPD sampled from Wishart      (3D) / (120 Samples c1:40, c2:40, c2:40)
        % Load an SPD simulation (from Section 1 and 3)
        sim_type = 5;
        % 1: linear ellipsoid with isotropic scalings + rotations
        % 2: linear ellipsoid with anisotropic scalings + rotations
        % 3: simulation 1 + 2
        % 4: 3D wishart samples from linear, spherical and planar ellipsoids
        % 5: 6D wishart samples 4 different covariance matrices
        [Mu_test, sigma_test, true_labels, dataset_name] = load_SPD_simulations(sim_type, df);
        % Build new Sigmas and compute metrics
        sigmas = [];
        for k=1:length(Mu_test)
            sigmas{k} = sigma_test(:,:,k);
        end
        
    case 4
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
        
    case 5
        % Load Manipulability Ellipsoids from JTDS datasets
        dataset = [];
        rot = 0;
        if rot
            dataset{1} = load(strcat(data_path,'/manipulability/pouring_obst_Rot.mat'));
            dataset{2} = load(strcat(data_path,'/manipulability/foot_motion_Rot.mat'));
            dataset{3} = load(strcat(data_path,'/manipulability/forearm_swing_Rot.mat'));
            dataset{4} = load(strcat(data_path,'/manipulability/backhand_swing_Rot.mat'));
        else
            dataset{1} = load(strcat(data_path,'/manipulability/pouring_obst.mat'));
            dataset{2} = load(strcat(data_path,'/manipulability/foot_motion.mat'));
            dataset{3} = load(strcat(data_path,'/manipulability/forearm_swing.mat'));
            dataset{4} = load(strcat(data_path,'/manipulability/backhand_swing.mat'));
        end
        
        Mu_test = [];
        sigmas = [];
        true_labels = [];
        last_id  = 1;
        for i=1:length(dataset)
            if i == 2
                %         traj_idx   = [dataset{i}.index_train(2):1:dataset{i}.index_train(3)-1];
                traj_idx   = [1:1:dataset{i}.index_train(3)];
            else
                traj_idx   = [1:1:dataset{i}.index_train(3)];
            end
            sigma_test_ = dataset{i}.M_train(:,:,traj_idx);
            sigma_test(:,:,last_id:last_id+length(traj_idx)-1) = sigma_test_;
            
            
            % lower the foot trajectories
            if i == 2
                Mu_test(:,last_id:last_id+length(traj_idx)-1)    = dataset{i}.x_train(:,traj_idx) + [0;0;-0.5];
            else
                Mu_test(:,last_id:last_id+length(traj_idx)-1)    = dataset{i}.x_train(:,traj_idx);
            end
            
            % Build new Sigmas and compute metrics
            for k=1:length(sigma_test_)
                sigmas{last_id+k-1} = sigma_test_(:,:,k);
            end
            last_id = last_id+length(traj_idx);
        end
        dataset_name='Manipulability Ellipsoids';
        me_index = zeros(1,length(sigmas));
        for i=1:length(sigmas)
            me_index(1,i) = sqrt(det(sigmas{i}));
        end
        % Creating labels base on Manipulability Index
        [~,edges] = histcounts(me_index,5);
        true_labels = discretize(me_index,edges);
        
    case 6
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

%     case 4                
%         % 4a) Toy 3D dataset, Diffusion Tensors from Synthetic Dataset, 1024 Samples
%         % Cluster Distibution: 4 clusters (each cluster has 10 samples)
%         % This function will generate a synthetic DW-MRI (Diffusion Weighted)-MRI
%         % This is done following the "Tutorial on Diffusion Tensor MRI using
%         % Matlab" by Angelos Barmpoutis, Ph.D. which can be found in the following
%         % link: http://www.cise.ufl.edu/~abarmpou/lab/fanDTasia/tutorial.php
%         %
%         % To run this function you should download fanDTasia toolbox in the
%         % ~/SPCM-CRP/3rdParty directory, this toolbox is also provided in
%         % the tutorial link.
%         
%         % clc; clear all; close all;
%         type = 'synthetic'; 
%         [sigmas, true_labels] = load_dtmri_dataset( data_path, type, display, randomize );
%         dataset_name = 'Synthetic DT-MRI';
%     case 6 
%         
%         % TODO: REDO THIS DATASET WITH THE ROTATED IMAGE CLUSTERS and 18x18 Cov.Matrx
%         % 5) Real 400D dataset, Covariance Features from ETH80 Dataset, 40 Samples
%         % Cluster Distibution: 8 classes/clusters (each cluster has 10 samples)
%         % This function loads the 400-D ETH80 Covariance Feature dataset 
%         % used to evaluate this algorithm in Section 8 of the accompanying paper.
%         %
%         %
%         % You must download this dataset from the following link: 
%         % http://ravitejav.weebly.com/classification-of-manifold-features.html
%         % and export it in the ~/SPCM-CRP/data directory
%         %
%         % Please cite the following paper if you make use of these features:
%         % R. Vemulapalli, J. Pillai, and R. Chellappa, “Kernel Learning for Extrinsic 
%         % Classification of Manifold Features”, CVPR, 2013.       
%         [sigmas, true_labels] = load_eth80_dataset(data_path, split, randomize);
%         
%         