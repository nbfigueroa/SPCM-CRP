%%  %%%%%%%%%%%%%%%%%%%%%%%%%
% Load Toy Data (3D) %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
figure('Color',[1 1 1]);
angle = pi/2;
%Ellipsoid 1
Cov = ones(3,3) + diag([1 1 1]);
% mu = [1 1 2]';
mu = [0 0 0]';
behavs_theta{1,1} = Cov;
[V1,D1] = eig(Cov);
% CoordRot = rotx(angle/3)*roty(angle/3)*rotz(angle/3);
CoordRot = rotx(-angle);
% CoordRot = eye(3);
% V1_rot = V1* CoordRot;
% Covr1 = V1_rot*D1*inv(V1_rot);
% [V1,D1] = eig(Covr1);
[x,y,z] = created3DgaussianEllipsoid(mu,V1,D1^1/2);
mesh(x,y,z,'EdgeColor','blue','Edgealpha',0.2);
hidden off
hold on;

%Ellipsoid 2: Scale+Noise
D1m = diag(D1)*1.3 + abs(randn(3,1).*[0.35 0.37 0.3]');
D1m = diag(D1)*0.5;
Covs2 = V1*(diag(D1m))*V1';
behavs_theta{1,2} = Covs2;
[V2,D2] = eig(Covs2);
[x,y,z] = created3DgaussianEllipsoid(mu,V2,D2^1/2);
mesh(x,y,z,'EdgeColor','black','Edgealpha',0.2);
hidden off
hold on;


%Ellipsoid 3: Rotated Coordinates
% CoordRot = rotx(angle)*roty(angle*0.5)*rotx(angle*2);
CoordRot = [1 0 0; 0 1 0 ; 0 0 -1];
% CoordRot = eye(3);
% CoordRot = rotx(angle);
V2_rot = CoordRot*V2;
Covs3 = V2_rot*D2*V2_rot';
behavs_theta{1,3} = Covs3;
[V3,D3] = eig(Covs3);
mu = [0 0 0]';
[x,y,z] = created3DgaussianEllipsoid(mu,V3,D3^1/2);
mesh(x,y,z,'EdgeColor','red','Edgealpha',0.2);
hidden off
hold on;

%Ellipsoid 4: Different
[Q R] = qr(randn(3,3));
D = diag([4 3 0.5]);
behavs_theta{1,4} = Q*(D)*Q';
[V4,D4] = eig(behavs_theta{1,4});
mu = [0 0 0]';
[x,y,z] = created3DgaussianEllipsoid(mu,V4,D4^1/2);
mesh(x,y,z,'EdgeColor','magenta','Edgealpha',0.2);
hidden off
hold on;


%Ellipsoid 5: Different Rotated + Scaled
[Q2 R] = qr(randn(3,3));
D = diag([4 3 0.5]*0.75);
behavs_theta{1,5} = (Q2)*(D)*(Q2)';
[V5,D5] = eig(behavs_theta{1,5});
mu = [0 0 0]';
[x,y,z] = created3DgaussianEllipsoid(mu,V5,D5^1/2);
mesh(x,y,z,'EdgeColor','green','Edgealpha',0.2);
hidden off
hold on;


colormap jet
alpha(0.5)
% xlabel('x');ylabel('y');zlabel('z');
% grid off
% axis off
axis equal

%%  %%%%%%%%%%%%%%%%%%%%%%%%%
% Load Toy Data (4D) %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('Color',[1 1 1])
behavs_theta4 = [];
tmp = eye(4);
tmp(1:3,1:3) = behavs_theta{1};
tmp(4,1:3) = ones(1,3);
tmp(1:3,4) = ones(3,1);
tmp(4,4) = 2;
behavs_theta4{1} = tmp;
subplot(2,3,1)
% colormap(copper)
imagesc(tmp)
colormap(pink)
colorbar  


tmp = eye(4);
tmp(1:3,1:3) = behavs_theta{2};
tmp(4,1:3) = ones(1,3)*(0.5);
tmp(1:3,4) = ones(3,1)*(0.5);
behavs_theta4{2} = tmp;
subplot(2,3,2)
% colormap(copper)
imagesc(tmp)
colormap(pink)
colorbar  

% Rotated
tmp = behavs_theta4{2};
[V D] = eig(tmp);
[Q R] = qr(randn(4,4)*.5);
behavs_theta4{3} = cov2cor(Q*tmp*Q');

% Rotated and Scaled 
tmp = behavs_theta4{3};
subplot(2,3,3)
% colormap(copper)
imagesc(tmp)
colormap(pink)
colorbar  

[V D]= eig(tmp);
behavs_theta4{4} = V*(D*2)*V';
subplot(2,3,4)
% colormap(copper)
imagesc(behavs_theta4{4})
colormap(pink)
colorbar  


% Different
[Q R] = qr(randn(4,4));
D = diag([10 4 3 1]);
behavs_theta4{5} = Q*(D)*Q';
subplot(2,3,5)
% colormap(copper)
imagesc(behavs_theta4{5})
colormap(pink)
colorbar  


D = diag([10 4 3 1]*3.5);
behavs_theta4{6} = Q*(D)*Q';
subplot(2,3,6)
% colormap(copper)
imagesc(behavs_theta4{6})
colormap(pink)
colorbar  


%%  %%%%%%%%%%%%%%%%%%%%%%%%%
% Load Toy Data (6D) %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

behavs_theta6toy = [];

tot = 5;
rows = floor(sqrt(tot));
cols = ceil(tot/rows);

figure('Color',[1 1 1])
for i=1:tot
    D = diag(abs(diag(eye(6))*randn));
    [Q R] = qr(randn(6,6));
    behavs_theta6toy{i} = Q*(D)*Q';
    subplot(rows,cols,i)
    imagesc(behavs_theta6toy{i})
    colormap(pink)
    colorbar 
end

lambda = [1 10 10 10 1 1];
figure('Color',[1 1 1])
iter = 1;
for i=tot+1:2*tot
    D = diag(abs(lambda*randn)*0.5);
    [Q R] = qr(randn(6,6));
    behavs_theta6toy{i} = Q*(D)*Q';
    subplot(rows,cols,iter)
    imagesc(behavs_theta6toy{i})
    colormap(pink)
    colorbar 
    iter = iter + 1;
end


lambda = [1 20 30 40 50 60];
figure('Color',[1 1 1])
iter = 1;
for i=2*tot+1:3*tot
    D = diag(abs(lambda*randn*0.5));
    [Q R] = qr(randn(6,6));
    behavs_theta6toy{i} = Q*(D)*Q';
    subplot(rows,cols,iter)
    imagesc(behavs_theta6toy{i})
    colormap(pink)
    colorbar 
    iter = iter + 1;
end

%%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load Task Ellipsoid Data (6D) %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
behavs = [];
behavs_theta = [];
load 6D-Grasps.mat
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

%% Display Data 3x21
tot = 21;
rows = floor(sqrt(tot));
cols = ceil(tot/rows);

figure('Color',[1 1 1])
for i=1:tot    
    behavs_theta6{i} = behavs_theta{1,i}.Sigma;
    subplot(rows,cols,i)
    imagesc(behavs_theta6{i})
    colormap(pink)
    colorbar 
end
title('Writing Task Ellipsoids')


figure('Color',[1 1 1])
for i=1:tot    
    behavs_theta6{tot+i} = behavs_theta{4,i}.Sigma;
    subplot(rows,cols,i)
    imagesc(behavs_theta6{tot+i})
    colormap(pink)
    colorbar 
end
title('Cuting Task Ellipsoids')


figure('Color',[1 1 1])
for i=1:tot    
    behavs_theta6{tot*2+i} = behavs_theta{5,i}.Sigma;
    subplot(rows,cols,i)
    imagesc(behavs_theta6{tot*2+i})
    colormap(pink)
    colorbar 
end
title('Screwing Task Ellipsoids')

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%  Find Clusters via Kernel-Kmeans on Prob. of SPCM %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all
behaviors = [];

% TOY 3d DATA
behaviors = behavs_theta;

% TOY 4d DATA
% behaviors = behavs_theta4;

% TOY 6d DATA
% behaviors = behavs_theta6toy ;

% Task Ellipsoids 6d DATA
% behaviors = behavs_theta6;
% behaviors = behavs_theta6_d;
% alpha = 1; %Grasp Data

% Extract Behaviors
sigmas = [];

% Behavs to compare
for i=1:length(behaviors)
    sigmas{i} =  behaviors{i};
end

% From BPHMM
% sigmas = rec_sigmas;

% %%%%%% Visualize SPCM and Prob. Similarity Confusion Matrix %%%%%%%%%%%%%%
tau = 10;
spcm = ComputeSPCMfunctionProb(sigmas, tau);  
figure('Color', [1 1 1], 'Position',[3283  545  377 549]);           

subplot(3,1,1)
imagesc(spcm(:,:,2))
title('Probability of Similarity Confusion Matrix')
colormap(pink)
colorbar 

% %%% Compute clusters from Similarity Matrix using Affinity Propagation %%%%%%
spcm_aff = log(spcm(:,:,2));
prob_spcm_aff = diag(median(spcm_aff,2)) + spcm_aff;
[E K c idx] = affinitypropagation(prob_spcm_aff);
fprintf('Number of clusters: %d \n', K);
subplot(3,1,2)
imagesc(c)
title('Recovered Clusters after Aff. Prop. on log Prob. of SPCM')
axis equal tight
colormap(pink)

% % %%% Use Kernel-K-means to find the number of clusters from Similarity function  %%%%%%
N_runs = 100;
energy_threshold = 0.5;
[labels energy] = kernel_kmeans(spcm(:,:,2),  N_runs);
K = length(unique(labels));
fprintf('After SPCM and KK-means--->>> \n Number of clusters: %d with total energy %d\n', K, energy);

subplot(3,1,3)
imagesc(labels)
title('Clustering from kernel-kmeans on SPCM Prob. function')
axis equal tight
colormap(pink)

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Construct Hierarchical Gaussian Models with Dendrogram Tree Struct %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Display iterations
dis = 1;

clear Xi
Xi = {}; 

% Set dataset
behaviors = [];
% behavs_theta6_d = behavs_theta6(1,end-42:end);
behaviors = behavs_theta4;
% behaviors = behavs_theta6toy;

% 6D Grasping Data
% behavs_theta6_d = behavs_theta6(1,end-41:end);
% behaviors = behavs_theta6_d;
% alpha = 1; %Grasp Data

% 3D Data
% behaviors = behavs_theta;
% alpha = 0; %Toy Data (Strict Similarity)

% Create tree
root.name = 'Behaviors';
root.idx = 0;
root.Sigma = eye(size(behaviors{1}));

dis_ell = 0;
if size(behaviors{1},1)==3
    dis_ell = 1;
end
dis_ell = 0;

clear t
t = tree(root);
for i=1:length(behaviors)
    name = strcat('Theta_',num2str(i));
    node.name = name;
    node.idx = i; %bphmm idx
    node.Sigma = behaviors{i};
    t  = t.addnode(1,node);
end
% tnames = t.treefun(@getName);
% disp(tnames.tostring)

ii = 1;
clear xi
xi.behavs = t; 

alpha = -0.05; %Toy Data (Strict Similarity)
% alpha = 1; %Grasp Data
use_log = 1;

keep_iter = 1;
node_flag = 0; % 0=Neither have children, 1=Both have children, 2=One node has children
while(keep_iter)
% for i=1:1 

    % Assignment Step of Hierarchical Gaussian Model
    %Get IDs of Nodes in the 1st level
    td = xi.behavs.depthtree;    
    xi.node_ids = find(td==1); 
    
    % Check that level 1 nodes are greater than one
    if (length(xi.node_ids)==1)
        disp('No more similar behaviors');
        if dis       
            tnames = xi.behavs.treefun(@getName);        
            disp(tnames.tostring)        
            figure('Color', [1 1 1],'Position',[1988  674  560 420]);
            tnames.plot 
            text(560, 240, 'Dendogram Tree')
        end       
        keep_iter = 0;
        break;
    end
    
    % Extract Behaviors from 1st level
    sigmas = [];
    % Behavs to compare
    for i=1:length(xi.node_ids)
        sigmas{i} =  xi.behavs.get(xi.node_ids(i)).Sigma;
    end
    
    % Compute Similarity
    spcm = ComputeSPCMfunction(sigmas,use_log);    
    xi.spcm = spcm; 
    if dis       
        tnames = xi.behavs.treefun(@getName);        
        disp(tnames.tostring)        
        figure('Color', [1 1 1],'Position',[1988  674  560 420]);
        tnames.plot 
        text(560, 240, 'Dendogram Tree')
        
        figure('Color', [1 1 1],'Position',[2550 674 560 420]);        
        imagesc(spcm(:,:,1))
        title('log(SPCM) Confusion Matrix')
        colormap(copper)
        colormap(pink)
        colorbar                                
    end
        
    % Find Anchor pair
    spcm_pairs = nchoosek(1:1:length(xi.node_ids),2);
    for i=1:size(spcm_pairs,1)
        spcm_pairs(i,3) = spcm(spcm_pairs(i,1),spcm_pairs(i,2),1);
    end
    
    % Create tree with number of children per node
    ot = tree(xi.behavs, 1); % Create a copy-tree filled with ones
    nc = ot.recursivecumfun(@(x) sum(x) + 1);
    
    % Choose pairs with min dissimilarity
    cand_spcm_ids = spcm_pairs(:,3) < alpha;
    
    % If the min SPCM pair is less than \alpha (allowable dissimilarity)
    % Update Step of Hierarchical Gaussian Model
    if (sum(cand_spcm_ids) > 0)
    
        cand_spcm_pairs = spcm_pairs(cand_spcm_ids,:);
        [~ , ids] = sort(cand_spcm_pairs(:,3));
        sorted_spcm_pairs = cand_spcm_pairs(ids,:);
    
        % Choose min dissimilarity pair whose merge changes the criterion the
        % least
        for jj=1:size(sorted_spcm_pairs,1)
            ni = nc.get(sorted_spcm_pairs(jj,1));
            nj = nc.get(sorted_spcm_pairs(jj,2));
            s  = sorted_spcm_pairs(jj,3);
            de = sqrt(ni*nj/(ni+nj))*s;
            sorted_spcm_pairs(jj,4)= de;
        end

        [~ , ids] = sort(sorted_spcm_pairs(:,4));
        spcm_pairs = sorted_spcm_pairs(ids,:)

        min_spcm = spcm_pairs(1,3)
        anchor = spcm_pairs(1,1:2)
        
        
        %Check if nodes in anchor pair have children
        no_children1 = isempty(xi.behavs.getchildren(xi.node_ids(anchor(1))));
        no_children2 = isempty(xi.behavs.getchildren(xi.node_ids(anchor(2))));
        
        %Neither nodes have children
        if no_children1 && no_children2
            % Find Anchor, Homothetic Ratio and Directionality
            disp('Neither nodes have children, create new group');
            node_flag = 0;
            homo_dir = spcm(anchor(1),anchor(2),3);
            if homo_dir < 0
                anchor = [anchor(2) anchor(1)];
            end    
            homo_ratio = spcm(anchor(1),anchor(2),2);
                        
        %Both nodes have children
        elseif  ~no_children1 && ~no_children2 
            disp('Both nodes have children, merge groups');
            node_flag = 1;
            homo_dir = spcm(anchor(1),anchor(2),3);
            if homo_dir < 0
                anchor = [anchor(2) anchor(1)];
            end    
            homo_ratio = spcm(anchor(1),anchor(2),2);
        
        %One of the Nodes has children    
        else 
            disp('One node has children, merge other node to group');
            node_flag = 2;
            if no_children1
                anchor = [anchor(2) anchor(1)];
                homo_ratio = spcm(anchor(2),anchor(1),2); 
            end            
        end                
        
        fprintf('Choosen Anchor pair [%d  %d] \n',xi.node_ids(anchor(1)),xi.node_ids(anchor(2)));
        
        h(1) = 1;
        h(2) = homo_ratio;
            
        Sigma_a = sigmas{anchor(1)};
        Sigma_b = sigmas{anchor(2)};
        [Va Da] = eig(Sigma_a);
        [Vb Db] = eig(Sigma_b);
        Sigma_b_sc = Vb*(Db*1/h(2))*inv(Vb);


        % Display Transformations
        mu = [0 0 0]';
        if dis_ell
            figure('Color', [1 1 1])
            [x,y,z] = created3DgaussianEllipsoid(mu,Va,Da^1/2);
            mesh(x,y,z,'EdgeColor','black','Edgealpha',0.8);
            hidden off
            hold on;
            axis equal
            [x,y,z] = created3DgaussianEllipsoid(mu,Vb,Db^1/2);
            mesh(x,y,z,'EdgeColor','red','Edgealpha',0.1);
            hidden off   
            pause(1);
            [Vb_sc Db_sc] = eig(Sigma_b_sc);
            [x,y,z] = created3DgaussianEllipsoid(mu,Vb_sc,Db_sc^1/2);
            mesh(x,y,z,'EdgeColor','red','Edgealpha',0.1);
            hidden off
            pause(0.1);
        end 

        % Find transformation R that minimizes objective function J
        conv_thres = 1e-5;
        thres = 1e-3;
        max_iter = 10000;
        iter = 1;
        J_S = 1;

        tic
        S1 = Sigma_a;
        W2 = eye(size(Sigma_a));
        W = [];
        J = [];
        conv_flag = 0;
        while(J_S > thres)     
            S1 = Sigma_a;
            S2 = W2*Sigma_b_sc*W2';  

            if dis_ell && (mod(iter,5)==0)
                [V2 D2] = eig(S2);
                [x,y,z] = created3DgaussianEllipsoid(mu,V2,D2^1/2);
                mesh(x,y,z,'EdgeColor','red','Edgealpha',0.1);
                hidden off   
                pause(1);
            end

            % Objective function Procrustes metric dS(S1,S2) = inf(R)||L1 - L2R||
            
            if det(S1)<0
                disp('S1 not pos def.')
                S1
            end
            
            if det(S2)<0
                disp('S2 not pos def.')
                S2
            end

%             L1 = chol(S1);
%             L2 = chol(S2);
%             
            L1 = matrixSquareRoot(S1);
            L2 = matrixSquareRoot(S2);
            [U,D,V] = svd(L1'*L2);
            R_hat = V*U';    
            J_S = EuclideanNormMat(L1 - L2*R_hat);
            J(iter) = J_S;    
            
            %Check convergence
%             if (iter>1) && (J(iter-1) - J(iter) < conv_thres)
            if dis_ell && (mod(iter,5)==0)
                fprintf('Error: %f \n',J_S)
            end
            if J_S < thres
                disp('Parameter Estimation Converged');
                conv_flag = 1;
                break;
            end
            
            if (iter>max_iter)
                disp('Exceeded Maximum Iterations');               
                break;
            end
            
            % Compute approx rotation
            W2 = W2 * R_hat^-1;
            iter = iter + 1;            
        end

        S = [];
        S(:,:,1) = S1;
        S(:,:,2) = S2;


        W(:,:,1) = eye(size(S1));
        W(:,:,2) = W2;

        if dis_ell
            [V2 D2] = eig(S2);
            [x,y,z] = created3DgaussianEllipsoid(mu,V2,D2^1/2);
            mesh(x,y,z,'EdgeColor','red','Edgealpha',0.8);
            hidden off
            axis equal
            pause(1);
        end
        toc

        fprintf('Finished pair %d and %d at iteration %d - Error: %f \n',xi.node_ids(anchor(1)), xi.node_ids(anchor(2)), iter,J_S)
      
        % Compute weights for Mean Sigma Estimate
        w =[];
        
        ch1 = 1;
        ch2 = 1;        
        
        ch1 = nc.get(xi.node_ids(anchor(1)));
        
        if (node_flag==1)                                            
            ch2 = nc.get(xi.node_ids(anchor(2)));                        
        end            

        tot_ch = ch1 + ch2;
        
        w(1) = ch1/tot_ch;
        w(2) = ch2/tot_ch;               
        
        % Weighted Mean Population Covariance Matrix 
        % Using the Log-Euclidean Distance (Good for Matrix Interpolation) 
        n = size(S,3);        
        Delta_hat = zeros(size(S(:,:,1)));
        for i=1:size(S,3)
            Delta_hat =  Delta_hat + w(i)*matrixLog(S(:,:,i));
        end
        Sigma_hat = matrixExp(Delta_hat);
        
        % Least Square Estimator of Average Covariance Matrix 
        % Using the Matrix Square Root Distance (Good for Matrix Interpolation)       
%         n = size(S,3);        
%         Delta_hat = zeros(size(S(:,:,1)));
%         for i=1:size(S,3)
%             Delta_hat =  Delta_hat + matrixSquare(S(:,:,i));
%         end        
%         Delta_hat = (1/n)*Delta_hat;
%         Sigma_hat = Delta_hat*Delta_hat';
        

        xi.anchor = xi.node_ids(anchor);
        xi.homo = h;
        xi.W = W;
        xi.S = S;
        xi.Parent_node = Sigma_hat;
        Xi{ii} = xi;

        if (node_flag==0)
            %Create New Group Node/tree
            clear t
            t = xi.behavs;
            t_idx = t.treefun(@getIdx);
            idxs = [];
            for jj=1:nnodes(t_idx)
                idxs = [idxs t_idx.get(jj)];
            end
            idxs = sort(idxs,'descend');
            new_node_idx = idxs(1)+1;
            name = strcat('Theta_',num2str(new_node_idx));
            
            clear root
            root.name = name;
            root.idx = new_node_idx; %bphmm idx
            root.Sigma = Sigma_hat;
            ti = tree(root);
            for j=1:2
                node = t.get(xi.anchor(j));
                node.homo = xi.homo(j);
                node.W = xi.W(:,:,j);
                new_comp= [];
                new_comp{1} = Sigma_hat;
                new_comp{2} = node.Sigma;
                new_spcm = ComputeSPCMfunction(new_comp,use_log);
                node.spcm = new_spcm(1,2,:);
                ti = ti.addnode(1,node);
            end
            chop_nodes = sort(xi.anchor,'descend');
            for j=1:2
                t = t.chop(chop_nodes(j));
            end
            t = t.graft(1,ti);        
            
        elseif (node_flag==1)
        % Both nodes have children so merge them into one big happy family            
            clear t
            t = xi.behavs;
            t_idx = t.treefun(@getIdx);
            idxs = [];
            for jj=1:nnodes(t_idx)
                idxs = [idxs t_idx.get(jj)];
            end
            idxs = sort(idxs,'descend');
            new_node_idx = idxs(1)+1;
            name = strcat('Theta_',num2str(new_node_idx));
            
            clear root
            root.name = name;
            root.idx = new_node_idx; %bphmm idx
            root.Sigma = Sigma_hat;
            ti = tree(root);
            
            %Add subtrees to new group
            for s=1:2
                st = t.subtree(xi.anchor(s));
                clear node
                node = st.get(1);
                node.homo = xi.homo(s);
                node.W = xi.W(:,:,s);
                new_comp= [];
                new_comp{1} = Sigma_hat;
                new_comp{2} = node.Sigma;
                new_spcm = ComputeSPCMfunction(new_comp,use_log);
                node.spcm = new_spcm(1,2,:);
                st = st.set(1,node);
                ti = ti.graft(1,st);               
            end            
                
            chop_nodes = sort(xi.anchor,'descend');
            for j=1:2
                t = t.chop(chop_nodes(j));
            end
            t = t.graft(1,ti);    
            
            
        elseif (node_flag==2)
            % One node has children so add the orphan node to the group            
            clear t
            t = xi.behavs;
            t_idx = t.treefun(@getIdx);
            idxs = [];
            for jj=1:nnodes(t_idx)
                idxs = [idxs t_idx.get(jj)];
            end
            idxs = sort(idxs,'descend');
            new_node_idx = idxs(1)+1;
            name = strcat('Theta_',num2str(new_node_idx));
            
            clear root
            root.name = name;
            root.idx = new_node_idx; %bphmm idx
            root.Sigma = Sigma_hat;
            ti = tree(root);
            
            %Add subtree to new group
            st = t.subtree(xi.anchor(1));
            clear node
            node = st.get(1);
            node.homo = xi.homo(1);
            node.W = xi.W(:,:,1);
            new_comp= [];
            new_comp{1} = Sigma_hat;
            new_comp{2} = node.Sigma;
            new_spcm = ComputeSPCMfunction(new_comp,use_log);
            node.spcm = new_spcm(1,2,:);
            st = st.set(1,node);
            ti = ti.graft(1,st);
           
            %Add node to new group
            node = t.get(xi.anchor(2));
            node.homo = xi.homo(2);
            node.W = xi.W(:,:,2);
            new_comp= [];
            new_comp{1} = Sigma_hat;
            new_comp{2} = node.Sigma;
            new_spcm = ComputeSPCMfunction(new_comp,use_log);
            node.spcm = new_spcm(1,2,:);
            ti = ti.addnode(1,node);
                
            chop_nodes = sort(xi.anchor,'descend');
            for j=1:2
                t = t.chop(chop_nodes(j));
            end
            t = t.graft(1,ti);    
            
        end
        
        % Update Tree
        ii = ii + 1;
        clear xi
        xi.behavs = t; 
        Xi{ii} = xi;
    else
        disp('No more similar behaviors');
        keep_iter = 0;
    end
end
 
disp('Final Tree')
treenames = Xi{end}.behavs.treefun(@getName);
disp(treenames.tostring)  
figure('Color', [1 1 1],'Position',[1988  674  560 420]);
treenames.plot

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Construct Hierarchical Gaussian Models with Bayesian Rose Tree Struct %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all
% Display iterations
dis = 1;

clear Xi
Xi = {}; 

% Set dataset
behaviors = [];
% behavs_theta6_d = behavs_theta6(1,end-42:end);
% behaviors = behavs_theta4;
behaviors = behavs_theta6toy;

% 6D Grasping Data
%behavs_theta6_d = behavs_theta6(1,end-41:end);
%behaviors = behavs_theta6_d;
%alpha = 1; %Grasp Data

% 3D Data
% behaviors = behavs_theta;
alpha = 0.5; %Toy Data (Strict Similarity)


% Create tree
root.name = 'Behaviors';
root.idx = 0;
root.Sigma = eye(size(behaviors{1}));

if size(behaviors{1},1)==3
    dis_ell = 1;
end

dis_ell = 0;
clear t
t = tree(root);
for i=1:length(behaviors)
    name = strcat('Theta_',num2str(i));
    node.name = name;
    node.idx = i; %bphmm idx
    node.Sigma = behaviors{i};
    t  = t.addnode(1,node);
end
% tnames = t.treefun(@getName);
% disp(tnames.tostring)

ii = 1;
clear xi
xi.behavs = t; 

% alpha = -0.05; %Toy Data ( Very Strict Similarity)
keep_iter = 1;
node_flag = 0; % 0=Neither have children, 1=Both have children, 2=One node has children
while(keep_iter)

    
    % Assignment Step of Hierarchical Gaussian Model
    %Get IDs of Nodes in the 1st level
    td = xi.behavs.depthtree;    
    xi.node_ids = find(td==1); 
    
    % Check that level 1 nodes are greater than one
    if (length(xi.node_ids)==1)
        disp('No more similar behaviors');
        if dis       
            tnames = xi.behavs.treefun(@getName);        
            disp(tnames.tostring)        
            figure('Color', [1 1 1],'Position',[1988  674  560 420]);
            tnames.plot 
            text(560, 240, 'Phylogenetic Tree')
        end       
        keep_iter = 0;
        break;
    end
    
    % Extract Behaviors from 1st level
    sigmas = [];
    % Behavs to compare
    for i=1:length(xi.node_ids)
        sigmas{i} =  xi.behavs.get(xi.node_ids(i)).Sigma;
    end
    
    % Compute Similarity
    use_log = 0;
    spcm = ComputeSPCMfunction(sigmas,use_log)    
    xi.spcm = spcm; 
    if dis       
        tnames = xi.behavs.treefun(@getName);        
        disp(tnames.tostring)        
        figure('Color', [1 1 1],'Position',[1988  674  560 420]);
        tnames.plot 
        text(560, 240, 'Phylogenetic Tree')
        
        figure('Color', [1 1 1],'Position',[2550 674 560 420]);        
        imagesc(spcm(:,:,1))
        title('SPCM Confusion Matrix')
        colormap(copper)
        colormap(pink)
        colorbar                                
    end
        
    % Find Anchor pair
    spcm_pairs = nchoosek(1:1:length(xi.node_ids),2);
    for i=1:size(spcm_pairs,1)
        spcm_pairs(i,3) = spcm(spcm_pairs(i,1),spcm_pairs(i,2),1);
    end
    
    % Create tree with number of children per node
    ot = tree(xi.behavs, 1); % Create a copy-tree filled with ones
    nc = ot.recursivecumfun(@(x) sum(x) + 1);
    
    % Choose pairs with min dissimilarity
    cand_spcm_ids = spcm_pairs(:,3) < alpha;
    
    % If the min SPCM pair is less than \alpha (allowable dissimilarity)
    % Update Step of Hierarchical Gaussian Model
    if (sum(cand_spcm_ids) > 0)
    
        cand_spcm_pairs = spcm_pairs(cand_spcm_ids,:);
        [~ , ids] = sort(cand_spcm_pairs(:,3));
        sorted_spcm_pairs = cand_spcm_pairs(ids,:);
    
        % Choose min dissimilarity pair whose merge changes the criterion the
        % least
        for jj=1:size(sorted_spcm_pairs,1)
            ni = nc.get(sorted_spcm_pairs(jj,1));
            nj = nc.get(sorted_spcm_pairs(jj,2));
            s  = sorted_spcm_pairs(jj,3);
            de = sqrt(ni*nj/(ni+nj))*s;
            sorted_spcm_pairs(jj,4)= de;
        end

        [~ , ids] = sort(sorted_spcm_pairs(:,4));
        spcm_pairs = sorted_spcm_pairs(ids,:);

        min_spcm = spcm_pairs(1,3);
        anchor = spcm_pairs(1,1:2);       
        
        %Check if nodes in anchor pair have children
        no_children1 = isempty(xi.behavs.getchildren(xi.node_ids(anchor(1))));
        no_children2 = isempty(xi.behavs.getchildren(xi.node_ids(anchor(2))));
        
        %Neither nodes have children
        if no_children1 && no_children2
            % Find Anchor, Homothetic Ratio and Directionality
            disp('Neither nodes have children, create new group');
            node_flag = 0;
            homo_dir = spcm(anchor(1),anchor(2),3);
            if homo_dir < 0
                anchor = [anchor(2) anchor(1)];
            end    
            homo_ratio = spcm(anchor(1),anchor(2),2);
                        
        %Both nodes have children
        elseif  ~no_children1 && ~no_children2 
            disp('Both nodes have children, merge groups');
            node_flag = 1;
            homo_dir = spcm(anchor(1),anchor(2),3);
            if homo_dir < 0
                anchor = [anchor(2) anchor(1)];
            end    
            homo_ratio = spcm(anchor(1),anchor(2),2);
        
        %One of the Nodes has children    
        else 
            disp('One node has children, merge other node to group');
            node_flag = 2;
            if no_children1
                anchor = [anchor(2) anchor(1)];
                homo_ratio = spcm(anchor(2),anchor(1),2); 
            end            
        end                
        
        fprintf('Choosen Anchor pair [%d  %d] \n',xi.node_ids(anchor(1)),xi.node_ids(anchor(2)));
        
        h(1) = 1;
        h(2) = homo_ratio;
            
        Sigma_a = sigmas{anchor(1)};
        Sigma_b = sigmas{anchor(2)};
        [Va Da] = eig(Sigma_a);
        [Vb Db] = eig(Sigma_b);
         Sigma_b_sc = Vb*(Db*1/h(2))*inv(Vb);


        % Display Transformations
        mu = [0 0 0]';
        if dis_ell
            figure('Color', [1 1 1])
            [x,y,z] = created3DgaussianEllipsoid(mu,Va,Da^1/2);
            mesh(x,y,z,'EdgeColor','black','Edgealpha',0.8);
            hidden off
            hold on;
            axis equal
            [x,y,z] = created3DgaussianEllipsoid(mu,Vb,Db^1/2);
            mesh(x,y,z,'EdgeColor','red','Edgealpha',0.1);
            hidden off   
            pause(1);
            [Vb_sc Db_sc] = eig(Sigma_b_sc);
            [x,y,z] = created3DgaussianEllipsoid(mu,Vb_sc,Db_sc^1/2);
            mesh(x,y,z,'EdgeColor','red','Edgealpha',0.1);
            hidden off
            pause(0.1);
        end 

        % Find transformation W that minimizes objective function J
        conv_thres = 1e-5;
        thres = 1e-3;
        thres = 1e-2;
        max_iter = 1000;
        iter = 1;
        J_S = 1;

        tic
        S1 = Sigma_a;
        W2 = eye(size(Sigma_a));
        W = [];
        J = [];
        conv_flag = 0;
        while(J_S > thres)     
            S1 = Sigma_a;
            S2 = W2*Sigma_b_sc*W2';  

            if dis_ell && (mod(iter,5)==0)
                [V2 D2] = eig(S2);
                [x,y,z] = created3DgaussianEllipsoid(mu,V2,D2^1/2);
                mesh(x,y,z,'EdgeColor','red','Edgealpha',0.1);
                hidden off   
                pause(1);
            end

            % Objective function Procrustes metric dS(S1,S2) = inf(R)||L1 - L2R||
            
            if det(S1)<0
                disp('S1 not pos def.')
                S1
            end
            
            if det(S2)<0
                disp('S2 not pos def.')
                S2
            end

%             L1 = chol(S1);
%             L2 = chol(S2);
%             
            L1 = matrixSquareRoot(S1);
            L2 = matrixSquareRoot(S2);
            [U,D,V] = svd(L1'*L2);
            R_hat = V*U';    
            J_S = EuclideanNormMat(L1 - L2*R_hat);
            J(iter) = J_S;    
            
            %Check convergence
%             if (iter>1) && (J(iter-1) - J(iter) < conv_thres)
            if dis_ell && (mod(iter,5)==0)
                fprintf('Error: %f \n',J_S)
            end
            if J_S < thres
                disp('Parameter Estimation Converged');
                conv_flag = 1;
                break;
            end
            
            if (iter>max_iter)
                disp('Exceeded Maximum Iterations');               
                break;
            end
            
            % Compute approx rotation
            W2 = W2 * R_hat^-1;
            iter = iter + 1;            
        end

        S = [];
        S(:,:,1) = S1;
        S(:,:,2) = S2;


        W(:,:,1) = eye(size(S1));
        W(:,:,2) = W2;

        if dis_ell
            [V2 D2] = eig(S2);
            [x,y,z] = created3DgaussianEllipsoid(mu,V2,D2^1/2);
            mesh(x,y,z,'EdgeColor','red','Edgealpha',0.8);
            hidden off
            axis equal
            pause(1);
        end
        toc

        fprintf('Finished pair %d and %d at iteration %d - Error: %f \n',xi.node_ids(anchor(1)), xi.node_ids(anchor(2)), iter,J_S)
      
        % Compute weights for Mean Sigma Estimate
        w =[];
        
        ch1 = 1;
        ch2 = 1; 
        
        if (node_flag==1)
            ch1 = length(xi.behavs.getchildren(xi.node_ids(anchor(1))));
            ch2 = length(xi.behavs.getchildren(xi.node_ids(anchor(2))));            
        elseif (node_flag==2)
            ch1 = length(xi.behavs.getchildren(xi.node_ids(anchor(1))));                   
        end            

        tot_ch = ch1 + ch2;
        
        w(1) = ch1/tot_ch;
        w(2) = ch2/tot_ch;        

        
        % Weighted Mean Population Covariance Matrix 
        % Using the Log-Euclidean Distance (Good for Matrix Interpolation) 
        n = size(S,3);        
        Delta_hat = zeros(size(S(:,:,1)));
        for i=1:size(S,3)
            Delta_hat =  Delta_hat + w(i)*matrixLog(S(:,:,i));
        end
        Sigma_hat = matrixExp(Delta_hat);
        
        % Least Square Estimator of Average Covariance Matrix 
        % Using the Matrix Square Root Distance (Good for Matrix Interpolation)       
%         n = size(S,3);        
%         Delta_hat = zeros(size(S(:,:,1)));
%         for i=1:size(S,3)
%             Delta_hat =  Delta_hat + matrixSquare(S(:,:,i));
%         end        
%         Delta_hat = (1/n)*Delta_hat;
%         Sigma_hat_root = Delta_hat*Delta_hat';
        

        xi.anchor = xi.node_ids(anchor);
        xi.homo = h;
        xi.W = W;
        xi.S = S;
        xi.Parent_node = Sigma_hat;
        Xi{ii} = xi;

        if (node_flag==0)
            %Create New Group Node/tree
            clear t
            t = xi.behavs;
            t_idx = t.treefun(@getIdx);
            idxs = [];
            for jj=1:nnodes(t_idx)
                idxs = [idxs t_idx.get(jj)];
            end
            idxs = sort(idxs,'descend');
            new_node_idx = idxs(1)+1;
            name = strcat('Theta_',num2str(new_node_idx));
            clear root
            root.name = name;
            root.idx = new_node_idx; %bphmm idx
            root.Sigma = Sigma_hat;
            ti = tree(root);
            for j=1:2
                node = t.get(xi.anchor(j));
                node.homo = xi.homo(j);
                node.W = xi.W(:,:,j);
                new_comp= [];
                new_comp{1} = Sigma_hat;
                new_comp{2} = node.Sigma;
                new_spcm = ComputeSPCMfunction(new_comp,use_log);
                node.spcm = new_spcm(1,2,:);
                ti = ti.addnode(1,node);
            end
            chop_nodes = sort(xi.anchor,'descend');
            for j=1:2
                t = t.chop(chop_nodes(j));
            end
            t = t.graft(1,ti);        
            
        elseif (node_flag==1)
        % Both nodes have children so merge them into one big happy family            
            clear t
            t = xi.behavs;                
            new_group_node_idx = sort([t.get(xi.anchor(1)).idx t.get(xi.anchor(2)).idx],'ascend'); 
            old_group_node_idx = t.get(xi.anchor(1)).idx;
            name = strcat('Theta_',num2str(new_group_node_idx(1)))
            
            %Get children of other Anchor Node
            children = t.getchildren(xi.anchor(2));            
            for kk=1:length(children)
                clear node
                node = t.get(children(kk));
                %Compute similarity for new nodes
                new_comp= [];
                new_comp{1} = Sigma_hat;
                new_comp{2} = node.Sigma;
                new_spcm = ComputeSPCMfunction(new_comp,use_log);
                node.spcm = new_spcm(1,2,:);        
                t = t.addnode(xi.anchor(1),node);
            end           
            
            %Remove other Anchor Node
            tidx = t.treefun(@getIdx);            
            old_id = find(tidx==t.get(xi.anchor(2)).idx);          
            t = t.chop(old_id);                                  
            
            %Modify Anchor Group Node            
            tidx = t.treefun(@getIdx);            
            new_id = find(tidx==old_group_node_idx);            
            clear node
            node.name = name;
            node.idx = new_group_node_idx(1); %bphmm idx
            node.Sigma = Sigma_hat;
            t = t.set(new_id,node);
            
        elseif (node_flag==2)
            % One node has children so add the orphan node to the tree
            clear t
            t = xi.behavs;    
            
            %Add Node to Group
            clear node
            node = t.get(xi.anchor(2));
            node.homo = xi.homo(2);
            node.W = xi.W(:,:,2);

            %Compute similarity for new node
            new_comp= [];
            new_comp{1} = Sigma_hat;
            new_comp{2} = node.Sigma;
            new_spcm = ComputeSPCMfunction(new_comp,use_log);
            node.spcm = new_spcm(1,2,:);        
            t = t.addnode(xi.anchor(1),node);
            t = t.chop(xi.anchor(2));
        end
        
        % Check consistency of Group Node IDXs
        
        % Update Tree
        ii = ii + 1;
        clear xi
        xi.behavs = t; 
        Xi{ii} = xi;
    else
        disp('No more similar behaviors');
        keep_iter = 0;
    end
end

disp('Final Tree')
treenames = Xi{end}.behavs.treefun(@getName);
disp(treenames.tostring)  
figure('Color', [1 1 1],'Position',[1988  674  560 420]);
treenames.plot
%% Check tree nodes and leafes consistency
[V,D] = eig(Xi{3}.behavs.get(3).Sigma);Xi{3}.behavs.get(3).W*(V*(D/Xi{3}.behavs.get(3).homo)*V')*Xi{3}.behavs.get(3).W'

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Construct Hierarchical Gaussian Models %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Display iterations
dis = 1;
clear Xi
Xi = {}; 

clear xi
xi.behavs = behavs_theta; 

ii = 1;
 while(length(xi.behavs)>1)
% for ii=1:1

    % Compute Similarity
    spcm = ComputeSPCMfunction(xi.behavs,1);
    xi.spcm = spcm; 
    
    % Find Anchor pair
    spcm_pairs = nchoosek(1:1:length(xi.behavs),2);
    for i=1:size(spcm_pairs,1)
        spcm_pairs(i,3) = spcm(spcm_pairs(i,1),spcm_pairs(i,2),1);
    end
    [min_spcm min_spcm_id] = min(spcm(:,3));
    anchor = spcm_pairs(min_spcm_id,1:2);

    
    % Find Anchor, Homothetic Ratio and Directionality
    homo_dir = spcm(anchor(1),anchor(2),3);
    if homo_dir < 0
        anchor = [anchor(2) anchor(1)];
    end    
    homo_ratio = spcm(anchor(1),anchor(2),2); 

    h(1) = 1;
    h(2) = homo_ratio;

    Sigma_a = behavs_theta{anchor(1)};
    Sigma_b = behavs_theta{anchor(2)};
    [Va Da] = eig(Sigma_a);
    [Vb Db] = eig(Sigma_b);
    Sigma_b_sc = Vb*(Db*1/h(2))*inv(Vb);

 
    % Display Transformations
    mu = [0 0 0]';
    if dis
        figure('Color', [1 1 1])
        [x,y,z] = created3DgaussianEllipsoid(mu,Va,Da^1/2);
        mesh(x,y,z,'EdgeColor','black','Edgealpha',0.8);
        hidden off
        hold on;
        axis equal
        [x,y,z] = created3DgaussianEllipsoid(mu,Vb,Db^1/2);
        mesh(x,y,z,'EdgeColor','red','Edgealpha',0.1);
        hidden off   
        pause(1);
        [Vb_sc Db_sc] = eig(Sigma_b_sc);
        [x,y,z] = created3DgaussianEllipsoid(mu,Vb_sc,Db_sc^1/2);
        mesh(x,y,z,'EdgeColor','red','Edgealpha',0.1);
        hidden off
        pause(1);

    end 

    % Find transformation R that minimizes objective function J
    thres = 1e-5;
    max_iter = 1000;
    iter = 1;
    J_S = 1;

    tic
    S1 = Sigma_a;
    W2 = eye(size(Sigma_a));
    W = [];
    while(J_S > thres || iter > max_iter)     
        S1 = Sigma_a;
        S2 = W2*Sigma_b_sc*W2';  

        if dis && (mod(iter,5)==0)
            [V2 D2] = eig(S2);
            [x,y,z] = created3DgaussianEllipsoid(mu,V2,D2^1/2);
            mesh(x,y,z,'EdgeColor','red','Edgealpha',0.1);
            hidden off   
            pause(1);
        end

        % Objective function Procrustes metric dS(S1,S2) = inf(R)||L1 - L2R||
        L1 = chol(S1);
        L2 = chol(S2);
        [U,D,V] = svd(L1'*L2);
        R_hat = V*U';    
        J_S = EuclideanNormMat(L1 - L2*R_hat);

        % Compute approx rotation
        W2 = W2 * R_hat^-1;
        iter = iter + 1;
    end

    S = [];
    S(:,:,1) = S1;
    S(:,:,2) = S2;


    W(:,:,1) = eye(size(S1));
    W(:,:,2) = W2;

    if dis
        [V2 D2] = eig(S2);
        [x,y,z] = created3DgaussianEllipsoid(mu,V2,D2^1/2);
        mesh(x,y,z,'EdgeColor','red','Edgealpha',0.8);
        hidden off
        axis equal
        pause(1);
    end
    toc

    fprintf('Finished pair %d and %d at iteration %d - Error: %f ',anchor(1), anchor(2), iter,J_S)

    % Least Square Estimator of Average Covariance Matrix 
    % Using the Log-Euclidean Distance (Good for Matrix Interpolation)
    n = size(S,3);
    Delta_hat = zeros(size(S(:,:,1)));
    for i=1:size(S,3)
        Delta_hat =  Delta_hat + log(S(:,:,i));
    end
    Delta_hat = (1/n)*Delta_hat;
    Sigma_hat = real(exp(Delta_hat));

    xi.anchor = anchor;
    xi.homo = h;
    xi.W = W;
    xi.S = S;
    xi.Sigma_hat = Sigma_hat;
    Xi{ii} = xi;
    
    new_behavs = xi.behavs;
    new_behavs(anchor) = [];
    new_behavs = [new_behavs Sigma_hat];
    clear xi
    xi.behavs = new_behavs;
    ii=ii+1;
    Xi{ii} = xi;
 end

%% Compute SPCM function Confusion Matrix
spcm = [];
% for i=1:length(Eig_behavs)
%   for j=1:length(Eig_behavs)     
       
for i=1:length(behavs_theta)
  for j=1:length(behavs_theta)     
      
         % Testing w/ Example Ellipsoids
        [Vi, Di] = eig(behavs_theta{i});
        [Vj, Dj] = eig(behavs_theta{j});
        
        %For Datasets
%         Vi = Eig_behavs{i}.V; Di = Eig_behavs{i}.D;        
%         Vj = Eig_behavs{i}.V; Dj = Eig_behavs{j}.D;

        %Ensure eigenvalues are sorted in ascending order
        [Vi, Di] = sortem(Vi,Di);
        [Vj, Dj] = sortem(Vj,Dj);
        
        %Structural of Sprectral Polytope
        Xi = Vi*Di^1/2;
        Xj = Vj*Dj^1/2;
                
        %Norms of Spectral Polytope Vectors
        for k=1:length(Dj)
            eig_i(k,1) = norm(Xi(:,k));
            eig_j(k,1) = norm(Xj(:,k));
        end
        
        %Homothetic factors
        hom_fact_ij = eig_i./eig_j;
        hom_fact_ji = eig_j./eig_i;
        
        %Magnif
        if (mean(hom_fact_ji) > mean(hom_fact_ij)) || (mean(hom_fact_ji) == mean(hom_fact_ij))
            dir = 1;
            hom_fact = hom_fact_ji;
        else
            dir = -1;
            hom_fact = hom_fact_ij;
        end     
        
        spcm(i,j,1) = std(hom_fact); 
        spcm(i,j,2) = mean(hom_fact); 
        spcm(i,j,3) = dir; 
        
   end
end

figure('Color', [1 1 1]);
spcm_cut = spcm(:,:,1);
cut_thres = 100;
for i=1:size(spcm,1)
    for j=1:size(spcm,2)
        if spcm(i,j)>cut_thres
            spcm_cut(i,j) = cut_thres;
        end
    end
end

imagesc(-log(spcm(:,:,1)))
title('log(SPCM) Confusion Matrix')
colormap(copper)
colormap(pink)
colorbar 

    % Variants of Non-Euclidean Statistics for Covariance Matrices
        % Objective function Root Euclidean dH(S1,S2) = ||(S1)^1/2 - (S2)^1/2||
    %     J_H = EuclideanNormMat(S1^1/2 - S2^1/2)
        % Objective function Cholesky dC(S1,S2) = ||chol(S1) - chol(S2)||
    %     J_C = EuclideanNormMat(chol(S1) - chol(S2))
        % Objective function Cholesky dL(S1,S2) = ||log(S1) - log(S2)||
    %     J_L = EuclideanNormMat(log(S1) - log(S2))   

 
%% Make groups

alpha  = 1.5;
groups = [];
groups = MotionGrouping(spcm(:,:,1),alpha);

% Plot Grasp Ellipsoids
fig_row = floor(sqrt(size(groups,1)));
fig_col = size(groups,1)/fig_row;
figure('Color',[1 1 1])
for ii=1:length(groups)
    g = groups{ii};
    gs = [];
    subplot(fig_row,fig_col,ii)
% figure('Color',[1 1 1])
    mu = [0 0 0]';
    for jj=1:length(g)
        behav.g_id = g(jj);
        behav.bt_id = Ids_behavs{g(jj)};
        behav.seg_id = behavs{behav.bt_id(1)}(behav.bt_id(2));
        behav.mu = behavs_theta{behav.bt_id(1),behav.bt_id(2)}.mu;
        behav.Sigma = behavs_theta{behav.bt_id(1),behav.bt_id(2)}.Sigma;
        behav.V = Eig_behavs{g(jj)}.V;
        behav.D = Eig_behavs{g(jj)}.D;
        
        gs = [gs behav.seg_id];
        grouped_behavs{ii,jj} = behav;
        [x,y,z] = created3DgaussianEllipsoid(mu,behav.V(1:3,1:3),behav.D(1:3,1:3));
        surfl(x,y,z);
%         mu = mu + [0 0.5 0]';
        hold on
    end
    if length(gs)<10
        tit = strcat('Group ', num2str(ii),' --> Behavs: ',num2str(gs));
    else
        tit = strcat('Group ', num2str(ii),' --> Behavs: ',num2str(gs(1)), ' to ', num2str(gs(end)));
    end
    title(tit)
    xlabel('x');ylabel('y');zlabel('z');
    axis equal
end


%% Procrustes
A = [11 39;17 42;25 42;25 40;23 36;19 35;30 34;35 29;...
30 20;18 19];
B = [15 31;20 37;30 40;29 35;25 29;29 31;31 31;35 20;...
29 10;25 18];
 
X = A;
Y = B + repmat([25 0], 10,1); 
figure('Color', [1 1 1])
plot(X(:,1), X(:,2),'r-', Y(:,1), Y(:,2),'b-');
text(X(:,1), X(:,2),('abcdefghij')')
text(Y(:,1), Y(:,2),('abcdefghij')')
legend('X = Target','Y = Comparison','location','SE')
set(gca,'YLim',[0 55],'XLim',[0 65]);

[d, Z, tr] = procrustes(X,Y);
plot(X(:,1), X(:,2),'r-', Y(:,1), Y(:,2),'b-',...
Z(:,1),Z(:,2),'b:');
text(X(:,1), X(:,2),('abcdefghij')')
text(Y(:,1), Y(:,2),('abcdefghij')')
text(Z(:,1), Z(:,2),('abcdefghij')')
legend('X = Target','Y = Comparison','Z = Transformed','location','SW')
set(gca,'YLim',[0 55],'XLim',[0 65]);

%% Example Hierarchical Clustiuurz

% Compute four clusters of the Fisher iris data using Ward linkage
% and ignoring species information, and see how the cluster
% assignments correspond to the three species.
load fisheriris
Z = linkage(meas,'ward','euclidean');
c = cluster(Z,'maxclust',4);
crosstab(c,species)
dendrogram(Z)
%%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Matrix Similarity on Grasps
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
behavs = [];
behavs_theta = [];
load 6D-Grasps.mat
dim = 6; 
for i=1:size(behavs_theta,1)
    behavs{i,1} = [1:size(behavs_theta,2)] + (i-1)*size(behavs_theta,2);
end

% Prepare Data for Full Comparison
% Sigmas =[];
% Eig_behavs = [];
% Ids_behavs = [];
% k=1;
% for i=1:size(behavs_theta,1)
%     for j=1:size(behavs_theta,2)
%         if ~isempty(behavs_theta{i,j})
%             Real Sigmas
%             Sigmas{k}= behavs_theta{i,j}.Sigma(1:dim,1:dim);
%             
%             Simulated Sigmas
%             [V,D] = eig(Sigmas{k});
%                       
%             Behavs ID on behavs_theta
%             Ids_behavs{k} = [i j]; 
%             
%             Eigen Stuff
%             eigstuff.V = V;
%             eigstuff.D = D;
%             Eig_behavs{k} = eigstuff;
%             k = k+1;            
%         end
%     end
% end

tot = 5;
behavs_theta6 = [];
for i=1:tot    
    behavs_theta6{i} = behavs_theta{3,i}.Sigma;
end

for i=1:tot    
    behavs_theta6{tot+i} = behavs_theta{4,i}.Sigma;
end