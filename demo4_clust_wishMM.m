%% Prepare variables for VB-WMM
N = length(sigmas);
M = length(sigmas{1});

if N < 15
    Kpos_ = N;
else
    Kpos_ = 15;
end
Kpos = 1:Kpos_; % possbile number of clusters
RESTARTS = 5; 

nu = (M+1000)*ones(1,N);
C = zeros(M,M,N);
for n=1:N
    C(:,:,n) = sigmas{n};
end

%% Run Model
prediction  = nan(length(Kpos),RESTARTS);
found_ss    = cell(length(Kpos),RESTARTS);
lower_bound = nan(length(Kpos),RESTARTS); 

for kk = Kpos
    for r = 1:RESTARTS
        [expectations, other,priors] = vbwmm( C, nu , kk, 'run_gpu',false,...
                                                          'verbose', 'off',...
                                                          'init_method', 'kmeans',...
                                                          'update_z', 'expect');
       [~,z_vbwmm]  = max(expectations.Z,[],2); % find mode of clustering
       
       % Found Clusters
       found_ss{kk,r} = z_vbwmm;
       
       % ELBO (evidence lower bound for the marginal likelihood)
       lower_bound(kk,r) = other.lower_bound(end);
    end
end

%% Extract clusters and compute metrics
ELBO_predictions = mean(lower_bound,2);
[~, opt_K]       = ml_curve_opt(-ELBO_predictions', 'derivatives');
ELBO_diff        = abs(opt_K(1) - opt_K(2));
if (ELBO_diff >= 5)    
    est_K       = opt_K(1);
else
    est_K       = opt_K(end);
end

est_labels  = found_ss{est_K}';
K = length(unique(true_labels));
[Purity, NMI, F] = cluster_metrics(true_labels, est_labels');
fprintf('---%s Results---\n  Clusters: %d/%d with Purity: %1.2f, NMI Score: %1.2f, F measure: %1.2f \n', ...
                'VB-WMM (Variational Bayes)', est_K, K,  Purity, NMI, F);

% Plots
figure('Color',[1 1 1]);
plot(Kpos,ELBO_predictions, 'bo-'), hold on
line([K K],get(gca,'YLim'),'Color',[0 1 0]), hold off
grid on;
xlabel('Number of States', 'Interpreter','LaTex')
ylabel('Mean ELBO over Restarts', 'Interpreter','LaTex')
