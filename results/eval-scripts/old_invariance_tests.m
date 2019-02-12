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


