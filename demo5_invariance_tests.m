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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%     Test 1 (Intro): Ellipsoid deformation with iso/anisotropic scalings     %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Creating test centers
Mu_test      = zeros(3,40);
Mu_test(1,:) = [0:5:95,-5:-5:-100];

% Linear ellipsoid
sigma_test = [];
sigma_test(:,:,1) = diag([1.25 1.25 5]);
for k=2:20
    ani_scale         = eye(3);
    ani_scale(2,2)    = 0.5 + sqrt(k/20 * 4.5);
%     ani_scale    = (0.5 + sqrt(k/20 * 4.5) )* ani_scale;
    sigma_test(:,:,k) = ani_scale*sigma_test(:,:,1)*ani_scale';
end

pitch = 0; yaw= 0;
rolls = 0.1570:0.0785:pi/2+0.0785;
for k=21:40
    R_0 = eul2rotm([yaw,pitch,-rolls(k-20)]);
    sigma_test(:,:,k) = R_0*sigma_test(:,:,k-20)*R_0';
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%     Plotting the concept of homothetic ratios     %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure('Color',[1 1 1])
D = eye(3)
lambda_1 = [1 1 2];
lambda_2 = [2 2 2];
origin = zeros(3,1);

% Draw Eigenvectors/Principal Axes
P = [lambda_1(1)*D(:,1) lambda_1(2)*D(:,2) lambda_1(3)*D(:,3)]
arrow3(origin, P(:,1), 'k'); hold on;
arrow3(origin, P(:,2), 'k'); hold on;
arrow3(origin, P(:,3), 'k'); hold on;
fill3(P(1,:),P(2,:),P(3,:),'k','FaceAlpha',0.25); hold on;

P = [lambda_2(1)*D(:,1) lambda_2(2)*D(:,2) lambda_2(3)*D(:,3)]
arrow3(origin, P(:,1), 'r'); hold on;
arrow3(origin, P(:,2), 'r'); hold on;
arrow3(origin, P(:,3), 'r'); hold on;
fill3(P(1,:),P(2,:),P(3,:),'r','FaceAlpha',0.25); hold on;

axis equal
grid off
axis off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%     Test 2 (Metric): Spectral Polytope Representation     %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Creating test centers
Mu_test      = zeros(3,3);
Mu_test(1,:) = [-10 0 10];

% Linear ellipsoid
sigma_test = [];


pitch = pi/6; yaw= pi/6;
rolls = 0.1570:0.0785:pi/2+0.0785;
R_0 = eul2rotm([yaw,pitch,rolls(1)]);

sigma_test(:,:,1) = diag([1 1 5]);

R_0 = eul2rotm([yaw,pitch,rolls(5)]);
sigma_test(:,:,2) = diag([2.5 2.5 2.5]);

yaw= 0;pitch = 0;
R_0 = eul2rotm([yaw,pitch,rolls(1)]);
sigma_test(:,:,3) =diag([1 5 5]);
 

% Plot Sigma with Spectral Polytopes
colors = hsv(length(Mu_test));
% colors = jet(length(Mu_test));

figure('Color',[1 1 1])
for k=1:1:length(Mu_test)
    [V,D]=eig(sigma_test(:,:,k)); scale = 1; 
    [x,y,z] = created3DgaussianEllipsoid(Mu_test(:,k),V,D, scale); hold on;
    
    subplot(1,3,k);
    % Draw frame
    H = eye(4);
    H(1:3,1:3) = eye(3);
    H(1:3,4)   = Mu_test(:,k) + randn(3,1)/50;
    
    % Draw World Reference Frame
    drawframe(H,1); hold on;
    
    % Draw Eigenvectors/Principal Axes  
    P = [V(:,1)*D(1,1) V(:,2)*D(2,2) V(:,3)*D(3,3)];
    arrow3(Mu_test(:,k), P(:,1), 'k'); hold on;
    arrow3(Mu_test(:,k), P(:,2), 'k'); hold on;
    arrow3(Mu_test(:,k), P(:,3), 'k'); hold on;
    P = P +Mu_test(:,k);
    fill3(P(1,:),P(2,:),P(3,:),'k','FaceAlpha',0.5); hold on;   
    
    % This makes the ellipsoids beautiful
    surf(x, y, z,'FaceColor',colors(k,:),'FaceAlpha', 0.25, 'FaceLighting','phong','EdgeColor','none'); hold on;
    camlight;
    xlabel('$x_1$', 'Interpreter', 'LaTex', 'FontSize',15);
    ylabel('$x_2$', 'Interpreter', 'LaTex','FontSize',15);
    zlabel('$x_3$', 'Interpreter', 'LaTex','FontSize',15);
    set(gca,'FontSize',16);
    grid on;
    axis equal;
    % axis tight;
    switch k
        case 1
            title('Linear 3D Ellipsoid','Interpreter', 'LaTex', 'FontSize',15);
        case 2
            title('Spherical 3D Ellipsoid','Interpreter', 'LaTex', 'FontSize',15);
        case 3
            title('Planar 3D Ellipsoid','Interpreter', 'LaTex', 'FontSize',15);
    end
    
end
% suptitle('Different 3D Ellipsoids and their semi-principal axes', 'Interpreter', 'LaTex', 'FontSize',15);




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

% Creating test centers
Mu_test      = zeros(3,20);
Mu_test(1,:) = [0:90:810 -10:-90:-900];
scales_t     = [1:10:100] ;
scales_t     = [scales_t 0.5*scales_t];

% Getting an original "linear" SPD matrix
sigma_test(:,:,1) = sigmas{1};
for k=1:10
    sigma_test(:,:,k) = scales_t(k)*sigma_test(:,:,1);    
end

% Getting an original "spherical" SPD matrix
sigma_test(:,:,11) = sigmas{7};
for k=12:20
    sigma_test(:,:,k) = scales_t(k)*sigma_test(:,:,11);    
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%     Test 3a: Reflection-Invariance     %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Creating test centers
Mu_test      = zeros(3,20);
Mu_test(1,:) = [0:90:810 -10:-90:-900];
scales_t     = [1:10:100] ;

% Getting an original "linear" SPD matrix
sigma_test(:,:,1) = sigmas{1};
for k=1:10
    sigma_test(:,:,k) = scales_t(k)*sigma_test(:,:,1);    
end

% Getting an original "spherical" SPD matrix
Reflection = [1 0 0 ; 0 -1 0; 0 0 1];
for k=11:20
    sigma_test(:,:,k) = Reflection*sigma_test(:,:,k-10)*Reflection';    
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%     Test 3b: Rotation/Scale-Invariance     %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Creating test centers
Mu_test          = zeros(3,40);
Mu_test(1,1:20)  = [0:100:900 -10:-100:-1000];
Mu_test(2,21:40) = [0:100:900 -10:-100:-1000];
scales_t         = [1:10:100] ;
scales_t         = [scales_t 0.5*scales_t];

% Getting an original "linear" SPD matrix
sigma_test(:,:,1) = sigmas{1};
roll = -pi/4; pitch = 0; yaw= 0;
R_0 = eul2rotm([yaw,pitch,roll]);
for k=1:10
    sigma_test(:,:,k) = scales_t(k)*(R_0*sigma_test(:,:,1)*R_0');    
end

% Getting an original "planar" SPD matrix
sigma_test(:,:,11) = sigmas{7};
for k=12:20
    sigma_test(:,:,k) = scales_t(k)*sigma_test(:,:,11);    
end

% Rotate all the "linear" SPD matrices
roll = -pi/4; pitch = pi/2; yaw= 0;
R_1 = eul2rotm([yaw,pitch,roll]);
for k=21:30
    sigma_test(:,:,k) = R_1*sigma_test(:,:,k-20)*R_1';    
end

% Rotate all the "planar" SPD matrices
roll = 0; pitch = 0; yaw= pi/2;
R_2 = eul2rotm([yaw,pitch,roll]);
for k=31:40
    sigma_test(:,:,k) = R_2*sigma_test(:,:,k-20)*R_2';    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%     Test 4: Sampled (Rotated) Covariance Matrices   %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Creating test centers
Mu_test          = zeros(3,80);
Mu_test(1,1:40)  = [0:1:19, -1:-1:-20];
Mu_test(2,41:80) = [0:1:19, -1:-1:-20];
df               = 100;

% Getting an original "linear" SPD matrix
sigma_test(:,:,1) = sigmas{1};
for k=2:20
    sigma_test(:,:,k) = wishrnd(1/df * sigma_test(:,:,1),df);    
end

% Getting an original "planar" SPD matrix
sigma_test(:,:,21) = sigmas{8};
for k=22:40
    sigma_test(:,:,k) =  wishrnd(1/df * sigma_test(:,:,21),df);    
end

sigma_test(:,:,41) = sigmas{1};
roll = 0; pitch = 0; yaw= pi/2;
R_1 = eul2rotm([yaw,pitch,roll]);
for k=42:60
    sigma_test(:,:,k) = wishrnd(1/df * (R_1*sigma_test(:,:,41)*R_1'),df);    
end


sigma_test(:,:,61) = sigmas{8};
roll = 0; pitch = -pi/4; yaw= pi/2;
R_2 = eul2rotm([yaw,pitch,roll]);
for k=62:80
    sigma_test(:,:,k) = wishrnd(1/df * (R_2*sigma_test(:,:,61)*R_2'),df);    
end


%% %%%%%%%%%%%%%%%%%%%%%%
%%     Plot Sigmas     %%
%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot Sigma
colors = jet(length(Mu_test));

draw_polytopes = 1;

figure('Color',[1 1 1])
for k=1:1:length(Mu_test)-1
    [V,D]=eig(sigma_test(:,:,k)); scale = 1; 
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
        drawframe(H,1); hold on;
        
    end
    
    
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
axis tight;
title('Invariance Testing 3D Ellipsoids', 'Interpreter', 'LaTex', 'FontSize',15);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%     Compute dis-similarities/distances for test datasets     %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Build new Sigmas and compute metrics
new_sigmas = [];
for k=1:length(Mu_test)
    new_sigmas{k} = sigma_test(:,:,k);
end

%%%%%%%%%%%%%%%%%%%%%%%%%  Compute SPCM (dis)-similarity %%%%%%%%%%%%%%%%%%%%%
% Tolerance for SPCM decay function 
dis_type = 1; % Choose dis-similarity type 
              % 1:'var' use the variance of homothetic ratios
              % 2:'cv'  use the coefficient of variation of homo. ratios
              % 3:'un-cv'  use the unbiased coefficient of variation of homo. ratios
% %%%%%%%%%%%%%%% Compute Confusion Matrix of Similarities (var) %%%%%%%%%%%%%%%%
spcm = ComputeSPCMfunctionMatrix(new_sigmas, 1, dis_type);  
D_var    = spcm(:,:,1);
S_var    = spcm(:,:,2);

% %%%%%%%%%%%%%%% Compute Confusion Matrix of Similarities (cv) %%%%%%%%%%%%%%%%
dis_type = 2;
spcm = ComputeSPCMfunctionMatrix(new_sigmas, 1, dis_type);  
D_cv    = spcm(:,:,1);
S_cv    = spcm(:,:,2);

% if exist('h1a','var') && isvalid(h1a), delete(h1a); end
title_str = 'SPCM (var) Dis-similarity Matrix';
h1a = plotSimilarityConfMatrix(D_var, title_str);

% if exist('h1b','var') && isvalid(h1b), delete(h1b); end
title_str = 'SPCM (cv) Dis-similarity Matrix';
h1b = plotSimilarityConfMatrix(D_cv, title_str);

% title_str = 'SPCM (var) Similarity Matrix';
% plotSimilarityConfMatrix(S_var, title_str);
% title_str = 'SPCM (cv) Similarity Matrix';
% plotSimilarityConfMatrix(S_cv, title_str);

%% %%%%%%%%%%%%%%%%%%%% Choose SDP distance %%%%%%%%%%%%%%%%%%%%%%%%
choosen_distance = 0;  % -2: Euclidean
                       % -1: Cholesky-Euclideandef:affine
                       %  0: Affine-Invariant Riemannian Distance (RIEM)
                       %  1: Log-Euclidean Riemannian Distance (LERM)
                       %  2: KL-Divergence (KLDM)
                       %  3: LogDet-Divergence (JBLD)
                       %  4: Minimum Scale Rotation Curve Distance (SROT)
                     
[D_R, distance_name] = computeSDP_distances(new_sigmas, choosen_distance);
%%%%%%% Visualize Bounded Distance (dis-similarity) Matrix %%%%%%%%%%%%%%
if exist('h2a','var') && isvalid(h2a), delete(h2a); end
h2a = plotSimilarityConfMatrix(D_R, distance_name);

%%%%%%% Visualize Bounded Distance (dis-similarity) Matrix %%%%%%%%%%%%%%
choosen_distance = 1;
[D_LE, distance_name] = computeSDP_distances(new_sigmas, choosen_distance);
if exist('h2b','var') && isvalid(h2b), delete(h2b); end
h2b = plotSimilarityConfMatrix(D_LE, distance_name);

%%%%%%% Visualize Bounded Distance (dis-similarity) Matrix %%%%%%%%%%%%%%
choosen_distance = 3;
[D_J, distance_name] = computeSDP_distances(new_sigmas, choosen_distance);
if exist('h2c','var') && isvalid(h2c), delete(h2c); end
h2c = plotSimilarityConfMatrix(D_J, distance_name);

%%%%%%% Visualize Bounded Distance (dis-similarity) Matrix %%%%%%%%%%%%%%
choosen_distance = 4;
[D_SR, distance_name] = computeSDP_distances(new_sigmas, choosen_distance);
if exist('h2d','var') && isvalid(h2c), delete(h2c); end
h2c = plotSimilarityConfMatrix(D_SR, distance_name);

%% Alternative Plot of Dis-similarity measures
with_var = 0;

figure('Color',[1 1 1]);
if with_var
    plot(1:length(new_sigmas),D_var(1,:),'*-b'); hold on;
end
plot(1:length(new_sigmas),D_cv(1,:),'o-r'); hold on;
plot(1:length(new_sigmas),D_R(1,:),'*-k'); hold on;
plot(1:length(new_sigmas),D_LE(1,:),'o-k'); hold on;
plot(1:length(new_sigmas),D_J(1,:),'d-k'); hold on;
plot(1:length(new_sigmas),D_SR(1,:),'+-k'); hold on;
xlabel('SPD $\mathbf{S}_i$', 'Interpreter', 'LaTex', 'FontSize',15);
ylabel('$d_{*}(\mathbf{S}_1,\mathbf{S}_i)$', 'Interpreter', 'LaTex', 'FontSize',15);
if with_var
    legend({'spcm (var)', 'spcm (cv)','affine','log-eucl','jbld','srot'}, 'Interpreter', 'LaTex', 'FontSize',13);
else
%     legend({'spcm (cv)','affine','log-eucl','jbld'}, 'Interpreter', 'LaTex', 'FontSize',13);
    legend({'desired','affine','log-eucl','jbld','srot'}, 'Interpreter', 'LaTex', 'FontSize',13);
end
title('SPD Distances/Dis-similarities', 'Interpreter', 'LaTex', 'FontSize',15)
grid on;

%% Alternative Plot of SPCM Dis-similarities only
figure('Color',[1 1 1]);
subplot(2,1,1);
plot(1:length(new_sigmas),D_var(1,:),'*-b'); hold on;
plot(1:length(new_sigmas),D_cv(1,:),'o-r'); hold on;
xlabel('SPD $\mathbf{S}_i$', 'Interpreter', 'LaTex', 'FontSize',15);
ylabel('$d_{*}(\mathbf{S}_1,\mathbf{S}_i)$', 'Interpreter', 'LaTex', 'FontSize',15);
legend({'spcm (var)', 'spcm (cv)'}, 'Interpreter', 'LaTex', 'FontSize',13);
title('SPCM Dis-similarities', 'Interpreter', 'LaTex', 'FontSize',15)
grid on;

subplot(2,1,2);
plot(1:length(new_sigmas),S_var(1,:),'*-b'); hold on;
plot(1:length(new_sigmas),S_cv(1,:),'o-r'); hold on;
xlabel('SPD $\mathbf{S}_i$', 'Interpreter', 'LaTex', 'FontSize',15);
ylabel('$s_{*}(\mathbf{S}_1,\mathbf{S}_i)$', 'Interpreter', 'LaTex', 'FontSize',15);
legend({'spcm (var)', 'spcm (cv)'}, 'Interpreter', 'LaTex', 'FontSize',13);
title('SPCM Similarities', 'Interpreter', 'LaTex', 'FontSize',15)
grid on;