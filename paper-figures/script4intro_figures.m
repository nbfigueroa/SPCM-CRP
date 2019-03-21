all_data   = [];
all_labels = [];
for d=1:length(Data)
    data_ = Data{d}(:,1:3)';
    all_data   = [all_data data_];
    if d == 1
        labels_ = True_states{d}';
    else
        labels_ = True_states{d}' + d;
    end
    all_labels = [all_labels labels_];
end
sample = 2;
[Priors0, Mu0, Sigma0] = gmmOracle(all_data(:,1:sample:end), all_labels(:,1:sample:end));
[~, est_labels0]       = my_gmm_cluster(all_data(:,1:sample:end), Priors0, Mu0, Sigma0, 'hard', []);
tot_dilation_factor = 7; rel_dilation_fact = 0.01;
Sigma0 = adjust_Covariances(Priors0, Sigma0, tot_dilation_factor, rel_dilation_fact);
% Modify Sigmas for visualization
[V,D] = eig(Sigma0(:,:,2))
D(1,1) = 1.15*D(1,1);
D(2,2) = 7*D(2,2);
D(3,3) = 10*D(3,3);
Sigma0(:,:,2) = V* D * V';

[V,D] = eig(Sigma0(:,:,4))
D(1,1) = 0.95*D(1,1);
D(2,2) = 7*D(2,2);
D(3,3) = 10*D(3,3);
Sigma0(:,:,4) = V* D * V';

[V,D] = eig(Sigma0(:,:,1))
D(1,1) = 2*D(1,1);
D(2,2) = 2*D(2,2);
Sigma0(:,:,1) = V* D * V';

%% Plot Gaussians
oracle_options = [];
oracle_options.type = -1;
oracle_options.emb_name = '';
[h_gmm]  = visualizeEstimatedGMM(all_data(:,1:sample:end),  Priors0, Mu0, Sigma0, est_labels0, oracle_options);
axis equal
axis tight