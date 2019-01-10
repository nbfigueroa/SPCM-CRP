%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Demo script to run invariant property tests for dis-similarities/distances %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%     Load Test 3D Ellipsoid Dataset     %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all; clear all; clc
pkg_dir = '/home/nbfigueroa/Dropbox/PhD_papers/journal-draft/new-code/SPCM-CRP';
display      = 0;       % display SDP matrices (if applicable)
randomize    = 0;       % randomize idx
dataset      = 1;       % choosen dataset from index above
sample_ratio = 1;       % sub-sample dataset [0.01 - 1]
[sigmas, true_labels, dataset_name] = load_SPD_dataset(dataset, pkg_dir, display, randomize, sample_ratio);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%     Test 1: Shear-Invariance     %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Creating test centers
Mu_test      = zeros(3,3);
Mu_test(1,:) = [0 , 2, 4];

% Getting an original "linear" SPD matrix
sigma_test(:,:,1) = sigmas{3};
eigs_test_mtrx    = eig(sigma_test(:,:,1));

% Getting an original "spherical" SPD matrix
sigma_test(:,:,2) = min(eig(sigma_test(:,:,1)))*eye(3);

% Applying a shear transformation to the spherical SPD matrix
eigs_factor       = max(eigs_test_mtrx)/min(eigs_test_mtrx)
Shear_matrix      = eye(3); 
Shear_matrix(1,3) = sqrt(eigs_factor)/2;
Shear_matrix(2,3) = 0;
Sheared_sphere    = Shear_matrix*sigma_test(:,:,2)*Shear_matrix';
sigma_test(:,:,3) = Sheared_sphere;
eig(sigma_test(:,:,3))


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%     Test 2a: Scale-Invariance     %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Creating test centers
Mu_test      = zeros(3,9);
Mu_test(1,:) = [-10:10:10 -10:10:10 -10:10:10];
Mu_test(2,:) = [0 0 0 -10 -10 -10  10 10 10];
scale_t      = 0.1;

% Getting an original "linear" SPD matrix
sigma_test = [];
for k=1:3
    sigma_k = sigmas{k*3};
    sigma_test(:,:,k) = scale_t*sigma_k;
    sigma_test(:,:,k+3) = sigma_k;
    sigma_test(:,:,k+6) = 1/scale_t*sigma_k;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%     Test 2b: Scale-Invariance     %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%
%%     Plot Sigmas     %%
%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot Sigma
colors = jet(length(Mu_test));

figure('Color',[1 1 1])
for k=1:length(Mu_test)
    [V,D]=eig(sigma_test(:,:,k)); scale = 1; 
    [x,y,z] = created3DgaussianEllipsoid(Mu_test(:,k),V,D, scale); hold on;
    
    % Draw frame
    H = eye(4);
    H(1:3,1:3) = eye(3);
    H(1:3,4)   = Mu_test(:,k);
    
    % Draw World Reference Frame
    drawframe(H,2.5); hold on;
    
    % This makes the ellipsoids beautiful
    surf(x, y, z,'FaceColor',colors(k,:),'FaceAlpha', 0.25, 'FaceLighting','phong','EdgeColor','none'); hold on;
end
camlight;
xlabel('$x_1$', 'Interpreter', 'LaTex', 'FontSize',15);
ylabel('$x_2$', 'Interpreter', 'LaTex','FontSize',15);
zlabel('$x_3$', 'Interpreter', 'LaTex','FontSize',15);
set(gca,'FontSize',16);
grid on;
axis equal;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%     Compute dis-similarities/distances for test datasets     %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Build new Sigmas and compute metrics
new_sigmas = [];
for k=1:length(Mu_test)
    new_sigmas{k} = sigma_test(:,:,k);
end

%%%%%%%%%%%%%%%%%%%%%  Compute SPCM (dis)-similarity %%%%%%%%%%%%%%%%%%%%%
% Tolerance for SPCM decay function 
dis_type = 1; % Choose dis-similarity type 
              % 1:'var' use the variance of homothetic ratios
              % 2:'cv'  use the coefficient of variation of homo. ratios

% %%%%%%%%%%%%%%% Compute Confusion Matrix of Similarities (var) %%%%%%%%%%%%%%%%
spcm = ComputeSPCMfunctionMatrix(new_sigmas, 1, dis_type);  
D_var    = spcm(:,:,1);
S_var    = spcm(:,:,2);

% %%%%%%%%%%%%%%% Compute Confusion Matrix of Similarities (cv) %%%%%%%%%%%%%%%%
dis_type = 2;
spcm = ComputeSPCMfunctionMatrix(new_sigmas, 1, dis_type);  
D_cv    = spcm(:,:,1);
S_cv    = spcm(:,:,2);

title_str = 'SPCM (var) Dis-similarity Matrix';
plotSimilarityConfMatrix(D_var, title_str);
title_str = 'SPCM (cv) Dis-similarity Matrix';
plotSimilarityConfMatrix(D_cv, title_str);

title_str = 'SPCM (var) Similarity Matrix';
plotSimilarityConfMatrix(S_var, title_str);
title_str = 'SPCM (cv) Similarity Matrix';
plotSimilarityConfMatrix(S_cv, title_str);

%%%%%%%%%%%%%%%%%%%%%% Choose SDP distance %%%%%%%%%%%%%%%%%%%%%%%%
choosen_distance = 0;  % -2: Euclidean
                       % -1: Cholesky-Euclidean
                       %  0: Affine-Invariant Riemannian Distance (RIEM)
                       %  1: Log-Euclidean Riemannian Distance (LERM)
                       %  2: KL-Divergence (KLDM)
                       %  3: LogDet-Divergence (JBLD)
                     
[D_R, distance_name] = computeSDP_distances(new_sigmas, choosen_distance);
      
%%%%%%% Visualize Bounded Distance (dis-similarity) Matrix %%%%%%%%%%%%%%
if exist('h2a','var') && isvalid(h2a), delete(h2a); end
h2a = plotSimilarityConfMatrix(D_R, distance_name);
choosen_distance = 1;
[D_LE, distance_name] = computeSDP_distances(new_sigmas, choosen_distance);
if exist('h2b','var') && isvalid(h2b), delete(h2b); end
h2b = plotSimilarityConfMatrix(D_LE, distance_name);
choosen_distance = 3;
[D_J, distance_name] = computeSDP_distances(new_sigmas, choosen_distance);
if exist('h2c','var') && isvalid(h2c), delete(h2c); end
h2c = plotSimilarityConfMatrix(D_J, distance_name);

%% Alternative Plot of Dis-similarity measures
figure('Color',[1 1 1]);
% plot(1:length(new_sigmas),D_var(1,:),'*-r'); hold on;
plot(1:length(new_sigmas),D_cv(1,:),'o-r'); hold on;
plot(1:length(new_sigmas),D_R(1,:),'*-b'); hold on;
plot(1:length(new_sigmas),D_LE(1,:),'o-b'); hold on;
plot(1:length(new_sigmas),D_J(1,:),'d-b'); hold on;
xlabel('SPD Index', 'Interpreter', 'LaTex', 'FontSize',15);
ylabel('dis-similarity/distance vs SPD ID:1', 'Interpreter', 'LaTex', 'FontSize',15);
% legend({'spcm (var)', 'spcm (cv)','affine','log-eucl','jbld'}, 'Interpreter', 'LaTex', 'FontSize',15)
legend({'spcm (cv)','affine','log-eucl','jbld'}, 'Interpreter', 'LaTex', 'FontSize',15)
grid on;
