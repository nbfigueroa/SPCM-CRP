function [Sigmas, True_Labels] = load_eth80_dataset( data_path, split, randomize )
%Loads a split of the ETH80 Covariance Features Dataset
% 1 split is 80 samples of 400x400 dimensional covariance features

load(strcat(data_path, sprintf('/ETH80_subspace_and_covariance_features/split%d_data.mat', split)))

% Both training and testing datasets
sigmas      = [tr_covariance_features te_covariance_features]; % covariance features
true_labels = [tr_labels; te_labels]'; % labels for covariance features

if (randomize == 1) 
    fprintf('Randomize Indices: 1 \n');
    [Sigmas True_Labels] = randomize_data(sigmas, true_labels);
elseif (randomize == 0) 
    fprintf('Randomize Indices: 0 \n');
    Sigmas = sigmas;
    True_Labels = true_labels;
end

end

