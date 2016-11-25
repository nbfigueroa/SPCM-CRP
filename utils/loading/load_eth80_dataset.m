function [sigmas, true_labels] = load_eth80_dataset( data_path, split )
%Loads a split of the ETH80 Covariance Features Dataset
% 1 split is 80 samples of 400x400 dimensional covariance features

load(strcat(data_path, sprintf('/ETH80_subspace_and_covariance_features/split%d_data.mat', split)))

% Both training and testing datasets
sigmas_      = [tr_covariance_features te_covariance_features]; % covariance features
true_labels_ = [tr_labels; te_labels]'; % labels for covariance features


% Group classes of train/test features
[true_labels, idx] = sort(true_labels_);
for i=1:length(sigmas_)
    sigmas{i} = sigmas_{idx(i)};
end

end

