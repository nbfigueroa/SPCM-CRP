%% Demo of VB-WMM code
addpath utils
addpath ./utils/plots

clear, close all

SEED = 56894;
rng(SEED)

Kpos = 1:10; % possbile number of clusters
RESTARTS = 5; 

p = 25; % dimensionality of problem
% NB! When dimensionality is high (compared to the number of data-points)
% the prior on eta becomes increasingly important (as problem becomes
% unstable). Consider fixing eta in that case. 
% Set 'fix_eta' to true and set inital values for 'eta_inv' to "something".
% We suggest trying a range of values and use the value with highest
% predictive likelihood on held-out data.

T_SCALE = 1000; % number of time points
N= T_SCALE*[0.4 0.3 0.2 0.1]; % configuration of clustering
ZS = [1,2,3,1]; 
K = length(unique(ZS));

wl = 25; % window length
L = T_SCALE/wl; 
C = nan(p,p,L);
Ctest = nan(p,p,L);

%% Generate Synthetic Data

% generate covariance parameters
for n = 1:length(unique(ZS))
    R{n}=triu(randn(p));
end
[X,X_test,zt] = generateSynthData(p,N,ZS,R);

% Extract scatter matrices - boxcar non-overlapping windows
zl = [];
for l = 1:L
    window_idx = (1:wl)+(l-1)*wl;
    C(:,:,l) = X(:, window_idx )*X(:, window_idx )';
    Ctest(:,:,l) = X_test(:, window_idx )*X_test(:, window_idx )';
    zl=[zl, round(mean(zt(window_idx)))]; 
end
nu = wl*ones(1,L);

%% Run Model
prediction  = nan(length(Kpos),RESTARTS);
found_ss    = cell(length(Kpos),RESTARTS);
lower_bound = nan(length(Kpos),RESTARTS); 

for kk = Kpos
    for r = 1:RESTARTS
        tic
        [expectations, other,priors] = vbwmm( C, nu , kk, 'run_gpu',false,...
                                                          'verbose', 'off',...
                                                          'init_method', 'kmeans',...
                                                          'update_z', 'expect');
       toc
       prediction(kk,r) = sum(vbwmm_predictiveLikelihood(Ctest, nu, expectations, priors, true),1); 
       
       [~,z_vbwmm]  = max(expectations.Z,[],2); % find mode of clustering
       found_ss{kk,r} = z_vbwmm;
       lower_bound(kk,r) = other.lower_bound(end);
    end
end

% Extract clusters and compute metrics
[max_val max_id] = max(mean(prediction,2));
est_K       = max_id;
est_labels  = found_ss{max_id}';
true_labels = zl;
K = length(unique(true_labels));
[Purity, NMI, F] = cluster_metrics(true_labels, est_labels');
fprintf('---%s Results---\n  Clusters: %d/%d with Purity: %1.2f, NMI Score: %1.2f, F measure: %1.2f \n', ...
                'VB-WMM (Variational Bayes)', est_K, K,  Purity, NMI, F);

%% Plots
figure,
subplot(2,1,1)
plot(Kpos,mean(prediction,2), 'bo-'), hold on
line([K K],get(gca,'YLim'),'Color',[0 1 0]), hold off
ylabel({'Mean Log-Predictive Likelihood', 'over Restarts'})

subplot(2,1,2)
plot(Kpos,mean(lower_bound,2), 'bo-'), hold on
line([K K],get(gca,'YLim'),'Color',[0 1 0]), hold off
xlabel('Number of States')
ylabel('Mean ELBO over Restarts')
