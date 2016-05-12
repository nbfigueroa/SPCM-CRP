% Computes a parcellation of synthetic data at different noise
%   levels, using Ward Clustering and our method based on the ddCRP. Each
%   parcellation is evaluated based on its Normalized Mututal Information
%   with the ground truth. The input "type"={'square','stripes','face'}
%   determines the underlying ground truth parcellation.
function [WC DC DC_K] = LearnSynth(type)

rng(1); % For repeatability

% Hyperparameters
alpha=10;
kappa=0.0001;
nu=1;
sigsq = 0.01;
pass_limit = 30;
repeats = 5;   % Number of times to repeat experiments

synth_sig = linspace(0,9,10);   % Noise levels to try
num_noise = length(synth_sig);
WC = zeros(num_noise,repeats);
DC = zeros(num_noise,repeats);
DC_K = zeros(num_noise,repeats);

for rep = 1:repeats
    disp(['Repeat # ' num2str(rep)]);
    parfor noise_ind = 1:num_noise
        [D adj_list gt_z] = GenerateSynthData(type, synth_sig(noise_ind));
        D = NormalizeConn(D); % Normalize connectivity to zero mean, unit var
        
        % Compute our ddCRP-based parcellation
        [~, Z] = WardClustering(D, adj_list, 1);
        [~, stats] = InitializeAndRunddCRP(Z, D, adj_list, 1:20, ...
            alpha, kappa, nu, sigsq, pass_limit, gt_z, 0);
        DC(noise_ind, rep) = stats.NMI(end);
        DC_K(noise_ind, rep) = stats.K(end);
        
        % Ward Clustering, using number of clusters discovered from our method
        n_clust = DC_K(noise_ind, rep);
        z = WardClustering(D, adj_list, n_clust);
        WC(noise_ind, rep) = CalcNMI(gt_z, z);
    end
end
end

