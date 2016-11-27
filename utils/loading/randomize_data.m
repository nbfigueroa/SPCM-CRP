function [ Sigmas True_Labels ] = randomize_data( sigmas, true_labels )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here


    rand_ids = randperm(length(sigmas));
    for i=1:length(sigmas)
       Sigmas{i} = sigmas{rand_ids(i)} ;
       True_Labels(i) = true_labels(rand_ids(i));
    end

end

