%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Example Datasets for Clustering on SPCM Kernel Matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%  %%%%%%%%%%%%%%%%%%%%%%%%%
% Load Toy Data (3D) %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all
clear all
figure('Color',[1 1 1]);
angle = pi/2;
%Ellipsoid 1
Cov = ones(3,3) + diag([1 1 1]);
% mu = [1 1 2]';
mu = [0 0 0]';
behavs_theta{1,1} = Cov;
[V1,D1] = eig(Cov);
CoordRot = rotx(-angle);
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
axis equal

sigmas = behavs_theta;
true_labels = [ones(1,3) , ones(1,2)*2];

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

sigmas = behavs_theta4;

true_labels = [ones(1,4) , ones(1,2)*2];


%%  %%%%%%%%%%%%%%%%%%%%%%%%%
% Load Toy Data (6D) %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

behavs_theta6toy = [];

tot = 10;
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

sigmas = behavs_theta6toy;
true_labels = [ones(1,tot) , ones(1,tot)*2, ones(1,tot)*3];


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

sigmas = behavs_theta6;
samples = 21;
true_labels = [ones(1,samples*3) , ones(1,samples)*2, ones(1,samples)*3];

