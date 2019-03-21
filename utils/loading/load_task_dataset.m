function [Sigmas, True_Labels] = load_task_dataset(data_path, randomize)

behavs = [];
behavs_theta = [];

load(strcat(data_path,'6D-Grasps.mat'))

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


% Add a slight rotation to the ellipsoids
% a = -pi/2;
% b =  pi/2;
% a = -pi/10;
% b =  pi/10;
% pitches = (b-a).*rand(length(sigmas),1) + a;
% for i=1:length(sigmas)
%     R_0 = eul2rotm([0,pitches(i),0]);
%     R_6D = eye(6);
%     R_6D(1:3,1:3) = R_0;
%     R_6D(4:6,4:6) = R_0;
%     sigmas{i} = R_6D*sigmas{i}*R_6D';
% end

if (randomize == 1) 
    fprintf('Randomize Indices: 1 \n');
    [Sigmas True_Labels] = randomize_data(sigmas, true_labels);
elseif (randomize == 0) 
    fprintf('Randomize Indices: 0 \n');
    Sigmas = sigmas;
    True_Labels = true_labels;
end

end