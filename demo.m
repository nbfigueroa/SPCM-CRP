%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Test Clustering Algorithm on Toy Datasets
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%%%%%%%
% Load Datasets
%%%%%%%%%%%%%%%%
clc
clear all
close all

% Set to 1 if you want to display Covariance Matrices
display = 1;

%% Select Dataset:
%% Toy 3D dataset, 5 Samples, 2 clusters (c1:3, c2:2)
[sigmas, true_labels] = load_toy_dataset('3d', display);

%% Toy 4D dataset, 6 Samples, 2 clusters (c1:4, c2:2)
[sigmas, true_labels] = load_toy_dataset('4d', display);

%% Toy 6D dataset, 30 Samples, 3 clusters (c1:10, c2:10, c3: 10)
[sigmas, true_labels] = load_toy_dataset('6d', display);

%% Real 6D dataset, task-ellipsoids, 105 Samples, 3 clusters (c1:63, c2:21, c3: 21)
% Path to data folder
data_path = '/home/nadiafigueroa/dev/MATLAB/SPCM-CRP/data';
[sigmas, true_labels] = load_task_dataset(data_path);