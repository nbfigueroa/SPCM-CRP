function [Sigmas, True_Labels] = load_youtube_dataset( data_path, split, randomize )
%Loads a split of the YouTube Covariance Features Dataset
% 1 split is 423 samples of 900x900 dimensional covariance features

load(strcat(data_path, sprintf('/YouTube_subspace_and_covariance_features/split%d_data.mat', split)))

% Both training and testing datasets
sigmas_      = [tr_covariance_features te_covariance_features]; % covariance features
true_labels_ = [tr_labels te_labels]; % labels for covariance features


% Group classes of train/test features
[true_labels, idx] = sort(true_labels_);
for i=1:length(sigmas_)
    sigmas{i} = sigmas_{idx(i)};
end


if (randomize == 1) 
    fprintf('Randomize Indices: 1 \n');
    [Sigmas True_Labels] = randomize_data(sigmas, true_labels);
elseif (randomize == 0) 
    fprintf('Randomize Indices: 0 \n');
    Sigmas = sigmas;
    True_Labels = true_labels;
end

end

