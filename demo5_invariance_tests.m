%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Demo script to run invariant property tests for dis-similarities/distances %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; close all;
% Load an SPD simulation (from Section 1 and 3)
sim_type = 3; % 1: linear ellipsoid with isotropic scalings + rotations
              % 2: linear ellipsoid with anisotropic scalings + rotations
              % 3: simulation 1 + 2
              % 4: 3D wishart samples from linear, spherical and planar ellipsoids
              % 5: 6D wishart samples 4 different covariance matrices
df = 100;
[Mu_test, sigma_test, true_labels, dataset_name] = load_SPD_simulations(sim_type, df);

% Build new Sigmas and compute metrics
sigmas = [];
for k=1:length(Mu_test)
    sigmas{k} = sigma_test(:,:,k);
end


%% Commands for manipulability ellipsoids
clear all; clc;
dataset = [];
rot = 0
load(strcat(data_path,'6D-Grasps.mat'))
if rot
    dataset{1} = load('pouring_obst_Rot');
    dataset{2} = load('foot_motion_Rot');
    dataset{3} = load('forearm_swing_Rot');
    dataset{4} = load('backhand_swing_Rot');
else
    dataset{1} = load('pouring_obst');
    dataset{2} = load('foot_motion');
    dataset{3} = load('forearm_swing');
    dataset{4} = load('backhand_swing');
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
    last_id = last_id+length(traj_idx)
    true_labels = [true_labels i*ones(1,length(sigma_test_))];
end
dataset_name='Manipulability Ellipsoids';

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%     Plot Sigmas  (with spectral polytopes)   %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot Sigma
iter = 1;
colors = jet(length(Mu_test));
draw_polytopes = 0;

figure('Color',[1 1 1])
for k=1:iter:length(Mu_test)
    [V,D]=eig(sigma_test(1:3,1:3,k)); scale = 0.25; 
    [x,y,z] = created3DgaussianEllipsoid(Mu_test(:,k),V,D, scale); hold on;
    [V, D] = sortem(V,D);
    
    if draw_polytopes
        % Draw frame
        H = eye(4);
        H(1:3,1:3) = eye(3);
        H(1:3,4)   = Mu_test(:,k) + randn(3,1)/50;
        
        % Draw World Reference Frame
        drawframe(H,1); hold on;
        
        % Draw Eigenvectors/Principal Axes
        P = [V(:,1)*D(1,1) V(:,2)*D(2,2) V(:,3)*D(3,3)];
        
        dot_x = dot(P(:,1)/norm(P(:,1)),[1 0 0]);
        dot_y = dot(P(:,2)/norm(P(:,2)),[0 1 0]);
        dot_z = dot(P(:,3)/norm(P(:,3)),[0 0 1]);
        if dot_y < 0                         
            P(:,2) = -P(:,2);
        end
        if dot_z < 0                         
            P(:,3) = -P(:,3);
        end        
%         if (k == 21) || (k == 28)
        if (k == 21) || (k == 30)
            P(:,1) = -P(:,1);
            if (k == 30)
                P(:,1) = -P(:,1);
%                 P(:,2) = -P(:,2);
                P(:,3) = -P(:,3);
            end
        end
        
        arrow3(Mu_test(:,k), P(:,1), 'k'); hold on;
        arrow3(Mu_test(:,k), P(:,2), 'k'); hold on;
        arrow3(Mu_test(:,k), P(:,3), 'k'); hold on;
        P = P +Mu_test(:,k);
        fill3(P(1,:),P(2,:),P(3,:),'k','FaceAlpha',0.5); hold on;
    else
        % Draw frame
        H = eye(4);
        H(1:3,1:3) = eye(3);
        H(1:3,4)   = Mu_test(:,k);
        
        % Draw World Reference Frame
        drawframe(H,0.05); hold on;
        
    end
    
    
    % This makes the ellipsoids beautiful
    surf(x, y, z,'FaceColor',colors(k,:),'FaceAlpha', 0.15, 'FaceLighting','phong','EdgeColor','none'); hold on;
end
camlight;
xlabel('$x_1$', 'Interpreter', 'LaTex', 'FontSize',15);
ylabel('$x_2$', 'Interpreter', 'LaTex','FontSize',15);
zlabel('$x_3$', 'Interpreter', 'LaTex','FontSize',15);
set(gca,'FontSize',16);
grid on;
axis equal;
axis tight;
title('Geometric Invariance Testing 3D Ellipsoids', 'Interpreter', 'LaTex', 'FontSize',15);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%     Compute SPCM distances/similarities for test datasets    %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% %%%%%%%%%%%%%%% Compute Confusion Matrix of Similarities (cv) %%%%%%%%%%%%%%%%
dis_type = 2;
gamma    = 2;
spcm = ComputeSPCMfunctionMatrix(sigmas, gamma, dis_type);  
D_SP    = spcm(:,:,1);
S_SP    = spcm(:,:,2);

if exist('h1a','var') && isvalid(h1a), delete(h1a); end
title_str = 'SPCM  Distance $d_{SP}(\cdot,\cdot)$';
h1a = plotSimilarityConfMatrix(D_SP, title_str);

if exist('h1b','var') && isvalid(h1b), delete(h1b); end
title_str = strcat('SPCM Similarity $\kappa_{SP}(\cdot,\cdot)$ with $\gamma=$',num2str(gamma));
h1b = plotSimilarityConfMatrix(S_SP, title_str);

%% %%%%%%%%%%%%%%%%%%%% Choose SDP distance %%%%%%%%%%%%%%%%%%%%%%%%
% -2: Euclidean
% -1: Cholesky-Euclideandef:affine
%  0: Affine-Invariant Riemannian Distance (RIEM)
%  1: Log-Euclidean Riemannian Distance (LERM)
%  2: KL-Divergence (KLDM)
%  3: LogDet-Divergence (JBLD)
%  4: Minimum Scale Rotation Curve Distance (SROT)
                    
%%%%%%% Visualize Bounded Distance (dis-similarity) Matrix %%%%%%%%%%%%%%
choosen_distance = 1;
[D_LE, distance_name] = computeSDP_distances(sigmas, choosen_distance);
if exist('h2a','var') && isvalid(h2a), delete(h2a); end
% h2a = plotSimilarityConfMatrix(D_LE, distance_name);

% Contruct Kernel Matrix
l_sensitivity = 2;
sigma = sqrt(mean(D_LE(:))/l_sensitivity);
gamma_LE = 1/(2*sigma^2)
K_LE = exp(-gamma_LE*D_LE.^2);
if exist('h2b','var') && isvalid(h2b), delete(h2b); end
distance_name = strcat('Log-Euclidean RBF Kernel with $\gamma=$',num2str(gamma_LE));
h2b = plotSimilarityConfMatrix(K_LE, distance_name);

%%%%%%% Visualize Bounded Distance (dis-similarity) Matrix %%%%%%%%%%%%%%
choosen_distance = 3;
[D_J, distance_name] = computeSDP_distances(sigmas, choosen_distance);
if exist('h3a','var') && isvalid(h3a), delete(h3a); end
% h3a = plotSimilarityConfMatrix(D_J, distance_name);

% Contruct Kernel Matrix
l_sensitivity = 2;
sigma = sqrt(real(mean(D_J(:))/l_sensitivity));
gamma_J = 1/(2*sigma^2)
K_J = exp(-gamma_J*D_J.^2);
if exist('h3b','var') && isvalid(h3b), delete(h3b); end
distance_name = strcat('Root Stein (JBLD) RBF Kernel with $\gamma=$',num2str(gamma_J));
h3b = plotSimilarityConfMatrix(K_J, distance_name);

%% Alternative Plot of Dis-similarity measures
figure('Color',[1 1 1]);
% Plot of distances
subplot(2,1,1)
plot(1:length(new_sigmas),D_SP(1,:),'o-r'); hold on;
plot(1:length(new_sigmas),D_LE(1,:),'s-','Color',[0.1 0.2 1]); hold on;
plot(1:length(new_sigmas),D_J(1,:),'d-','Color',[0.2 0.4 1]); hold on;
xlabel('SPD $\mathbf{S}_i$', 'Interpreter', 'LaTex', 'FontSize',15);
ylabel('$d_{*}(\mathbf{S}_1,\mathbf{S}_i)$', 'Interpreter', 'LaTex', 'FontSize',15);
legend({'$d_{SP}(\mathbf{S}_1,\cdot)$','$d_{LE}(\mathbf{S}_1,\cdot)$','$d_{J}(\mathbf{S}_1,\cdot)$'}, 'Interpreter', 'LaTex', 'FontSize',13);
title('SPD Distances/Dis-similarities', 'Interpreter', 'LaTex', 'FontSize',18)
grid on;

% Plot of similarities/kernels
subplot(2,1,2)
plot(1:length(new_sigmas),S_SP(1,:),'o-r'); hold on;
plot(1:length(new_sigmas),K_LE(1,:),'s-','Color',[0.1 0.2 1]); hold on;
plot(1:length(new_sigmas),K_J(1,:),'d-','Color',[0.2 0.4 1]); hold on;
xlabel('SPD $\mathbf{S}_i$', 'Interpreter', 'LaTex', 'FontSize',15);
ylabel('$k_{*}(\mathbf{S}_1,\mathbf{S}_i)$', 'Interpreter', 'LaTex', 'FontSize',15);
legend({'$\kappa_{SP}(\mathbf{S}_1,\cdot)$','$k_{LE}(\mathbf{S}_1,\cdot)$','$k_{J}(\mathbf{S}_1,\cdot)$'}, 'Interpreter', 'LaTex', 'FontSize',13);
title('SPD Kernels/Similarities', 'Interpreter', 'LaTex', 'FontSize',18)
grid on;
