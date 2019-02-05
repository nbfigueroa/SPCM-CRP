%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%    Plot Spectral Polytope Representation for different Ellispoids  (Figure 4)  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Creating test centers
Mu_test      = zeros(3,3);
Mu_test(1,:) = [-10 0 10];

% Linear ellipsoid
sigma_test = [];

pitch = pi/6; yaw= pi/6;
rolls = 0.1570:0.0785:pi/2+0.0785;
R_0 = eul2rotm([yaw,pitch,rolls(1)]);

sigma_test(:,:,1) = diag([1 1 5]);

R_0 = eul2rotm([yaw,pitch,rolls(5)]);
sigma_test(:,:,2) = diag([2.5 2.5 2.5]);

yaw= 0;pitch = 0;
R_0 = eul2rotm([yaw,pitch,rolls(1)]);
sigma_test(:,:,3) =diag([1 5 5]);
 

% Plot Sigma with Spectral Polytopes
colors = hsv(length(Mu_test));
% colors = jet(length(Mu_test));

figure('Color',[1 1 1])
for k=1:1:length(Mu_test)
    [V,D]=eig(sigma_test(:,:,k)); scale = 1; 
    [x,y,z] = created3DgaussianEllipsoid(Mu_test(:,k),V,D, scale); hold on;
    
    subplot(1,3,k);
    % Draw frame
    H = eye(4);
    H(1:3,1:3) = eye(3);
    H(1:3,4)   = Mu_test(:,k) + randn(3,1)/50;
    
    % Draw World Reference Frame
    drawframe(H,1); hold on;
    
    % Draw Eigenvectors/Principal Axes  
    P = [V(:,1)*D(1,1) V(:,2)*D(2,2) V(:,3)*D(3,3)];
    arrow3(Mu_test(:,k), P(:,1), 'k'); hold on;
    arrow3(Mu_test(:,k), P(:,2), 'k'); hold on;
    arrow3(Mu_test(:,k), P(:,3), 'k'); hold on;
    P = P +Mu_test(:,k);
    fill3(P(1,:),P(2,:),P(3,:),'k','FaceAlpha',0.5); hold on;   
    
    % This makes the ellipsoids beautiful
    surf(x, y, z,'FaceColor',colors(k,:),'FaceAlpha', 0.25, 'FaceLighting','phong','EdgeColor','none'); hold on;
    camlight;
    xlabel('$x_1$', 'Interpreter', 'LaTex', 'FontSize',15);
    ylabel('$x_2$', 'Interpreter', 'LaTex','FontSize',15);
    zlabel('$x_3$', 'Interpreter', 'LaTex','FontSize',15);
    set(gca,'FontSize',16);
    grid on;
    axis equal;
    % axis tight;
    switch k
        case 1
            title('Linear 3D Ellipsoid','Interpreter', 'LaTex', 'FontSize',15);
        case 2
            title('Spherical 3D Ellipsoid','Interpreter', 'LaTex', 'FontSize',15);
        case 3
            title('Planar 3D Ellipsoid','Interpreter', 'LaTex', 'FontSize',15);
    end
    
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%     Plotting the concept of homothetic ratios  (Figure 6)   %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure('Color',[1 1 1])
D = eye(3);
lambda_1 = [1 1 2];
lambda_2 = [2 2 2];
origin = zeros(3,1);

% Draw Eigenvectors/Principal Axes
P = [lambda_1(1)*D(:,1) lambda_1(2)*D(:,2) lambda_1(3)*D(:,3)];
arrow3(origin, P(:,1), 'k'); hold on;
arrow3(origin, P(:,2), 'k'); hold on;
arrow3(origin, P(:,3), 'k'); hold on;
fill3(P(1,:),P(2,:),P(3,:),'k','FaceAlpha',0.25); hold on;

P = [lambda_2(1)*D(:,1) lambda_2(2)*D(:,2) lambda_2(3)*D(:,3)];
arrow3(origin, P(:,1), 'r'); hold on;
arrow3(origin, P(:,2), 'r'); hold on;
arrow3(origin, P(:,3), 'r'); hold on;
fill3(P(1,:),P(2,:),P(3,:),'r','FaceAlpha',0.25); hold on;

axis equal
grid off
axis off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%      Plot for Similarity function analysis (Figure 7)      %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

d_SP  = linspace(0,10,100);
k_SP  = zeros(10,length(d_SP));
gammas = logspace(log10(0.001),log10(100),25);

gamma_leg = cell(1,length(gammas));
for i=1:length(gammas)
    k_SP(i,:) = exp(-gammas(i)*d_SP);    
    gamma_leg{i} = strcat('$\gamma=$', num2str(gammas(i)));
end

figure('Color',[1 1 1])
colors = vivid(length(gammas));
for i=1:length(gammas)
    plot(d_SP,k_SP(i,:),'-d','Color',colors(i,:),'LineWidth',1.5); hold on;
end
xlabel('SPCM distance value $d_{SP}(\cdot,\cdot)$', 'Interpreter', 'LaTex', 'FontSize',15);
ylabel('SPCM similarity value $\kappa_{SP}(\cdot,\cdot)$', 'Interpreter', 'LaTex', 'FontSize',15);
legend(gamma_leg, 'Interpreter', 'LaTex', 'FontSize',13);
title('Effect of $\gamma$ on SPCM similarity $\kappa_{SP}(\cdot,\cdot)=\exp(-\gamma d_{SP}(\cdot,\cdot))$', 'Interpreter', 'LaTex', 'FontSize',15)
grid on;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%      Plot for Similarity function analysis (Figure 8)      %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Build new Sigmas and compute metrics
new_sigmas = [];
for k=1:length(Mu_test)
    new_sigmas{k} = sigma_test(:,:,k);
end

dis_type   = 2;
gamma_test = [2,4,8];
S_cv    = zeros(length(sigma_test),length(sigma_test),length(gamma_test));

spcm = ComputeSPCMfunctionMatrix(new_sigmas, 1, dis_type);  
D_cv    = spcm(:,:,1);

if exist('h1a','var') && isvalid(h1b), delete(h1a); end
title_str = 'SPCM Distance $d_{SP}(\cdot,\cdot)$';
% h1b = plotSimilarityConfMatrix(D_cv, title_str);

for i=1:length(gamma_test)
    spcm = ComputeSPCMfunctionMatrix(new_sigmas, gamma_test(i), dis_type);  
    S_cv(:,:,i)    = spcm(:,:,2);
    title_str = strcat('SPCM Similarity $\kappa_{SP}(\cdot,\cdot)$ with $\gamma=$', num2str(gamma_test(i)));
    plotSimilarityConfMatrix(S_cv(:,:,i), title_str);
end

figure('Color',[1 1 1]);
% distance
yyaxis right;
plot(1:length(new_sigmas),D_cv(1,:),'*-r'); hold on;
ylabel('$d_{SP}(\mathbf{S}_1,\mathbf{S}_i)$', 'Interpreter', 'LaTex', 'FontSize',15);
% legend('$d_{SP}(\mathbf{S}_1,\cdot)$', 'Interpreter', 'LaTex', 'FontSize',13);
    
% similarities with different gammas
yyaxis left;
plot(1:length(new_sigmas),S_cv(1,:,1),'*-','Color',[0.1 0.3 1]); hold on;
plot(1:length(new_sigmas),S_cv(1,:,2),'o-','Color',[0.2 0.3 1]); hold on;
plot(1:length(new_sigmas),S_cv(1,:,3),'d-','Color',[0.3 0.3 1]); hold on;
xlabel('SPD $\mathbf{S}_i$', 'Interpreter', 'LaTex', 'FontSize',15);
ylabel('$\kappa_{SP}(\mathbf{S}_1,\mathbf{S}_i)$', 'Interpreter', 'LaTex', 'FontSize',15);
legend({'$\kappa_{SP}(\mathbf{S}_1,\cdot|\gamma=2)$','$\kappa_{SP}(\mathbf{S}_1,\cdot|\gamma=4)$','$\kappa_{SP}(\mathbf{S}_1,\cdot|\gamma=8)$','$d_{SP}(\mathbf{S}_1,\cdot)$'}, 'Interpreter', 'LaTex', 'FontSize',13);
title('Effect of $\gamma$ on SPCM Similarities', 'Interpreter', 'LaTex', 'FontSize',15)
grid on;

