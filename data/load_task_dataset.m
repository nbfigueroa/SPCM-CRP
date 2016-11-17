function [sigmas, true_labels] = load_task_dataset(data_path)

behavs = [];
behavs_theta = [];
load(data_path)

dim = 6; 
for i=1:size(behavs_theta,1)
    behavs{i,1} = [1:size(behavs_theta,2)] + (i-1)*size(behavs_theta,2);
end

behavs_theta6 = [];
for i=1:size(behavs_theta,1)    
    for j=1:size(behavs_theta,2)    
        behavs_theta6{(i-1)*size(behavs_theta,2) + j} = behavs_theta{i,j}.Sigma;
    end
end

sigmas = behavs_theta6;
samples = 21;
true_labels = [ones(1,samples*3) , ones(1,samples)*2, ones(1,samples)*3];


end