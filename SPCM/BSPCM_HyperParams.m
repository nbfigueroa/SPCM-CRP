%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot Effects of dimensionality of B-SPCM and Scaling function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
clear all
close all

%%%% Similarity values for SPCM %%%%
S = linspace(0,5,1000);

%%%% d-Dimension of data samples %%%%
dim = [3:1:10];

%%%% Tolerance tau for scaling function %%%%
tau = [1:1:10];
tau_idx = length(tau);

%%%% Scaling function %%%%
alpha = zeros(length(tau),length(dim));
for i=1:length(tau)
    alpha(i,:) = 10.^(tau(i)*exp(-dim));
end

%%%% Bounded SPCM from SPCM f(Sigma_i,Sigma_j,tau) = 
%%%% 1/( 1 + upsilon(tau,dim)*s(Sigma_i,Sigma_j))
b_spcm = zeros(length(dim),length(S));
for i=1:length(dim)
    b_spcm(i,:) = 1./(1+S*alpha(tau(tau_idx),i));
end

figure('Color', [1 1 1])
subplot(2,1,1)
legendinfo = {};
for i=1:length(dim)
    plot(S,b_spcm(i,:),'LineWidth', 2, 'Color',[rand rand rand])    
    legendinfo{i} = sprintf('d=%d',dim(i));
    hold on
end
legend(legendinfo)
ylabel('B-SPCM value f(\Delta_{ij},1)','FontSize', 20)
xlabel('SPCM value \Delta_{ij}','FontSize', 20)
title('Effect of dimensionality on f(\Delta_{ij},\tau) with \tau=1','FontSize', 20)
grid on

subplot(2,1,2)
legendinfo_ = {};
for i=1:length(alpha)
    plot(dim,alpha(i,:),'LineWidth', 2, 'Color',[rand rand rand])    
    legendinfo_{i} = sprintf('tau=%d',tau(i));
    hold on
end
legend(legendinfo_,'Interpreter','latex')
ylabel('Scaling function \upsilon(\tau)','FontSize', 20)
xlabel('Dimensionality d','FontSize', 20)
title('Effect of dimensionality on \upsilon(\tau)','FontSize', 20)
grid on

%% Plot of Bounded function from spcm full
clc
clear all
close all

% d-Dimension of data samples
dim = [6];

% Values from H(Theta_i ~ Theta_j) = s(Sigma_i,Sigma_j)
s = linspace(0,10,100);

% Tolerance tau for scaling function
taus = linspace(0,20,100);

% Probability Function from SPCM p(theta_i~theta_j|Sigma_i,Sigma_j) = 1/( 1 + s(Sigma_i,Sigma_j)*alpha(tau,dim)figure('Color', [1 1 1])
figure('Color', [1 1 1])
for k=1:1:length(dim)
    dim_idx = k;
    b_spcm = zeros(length(s),length(taus));
    for i=1:length(s)
        for j=1:length(taus)
            upsilon = 10^(taus(j)*exp(-dim(dim_idx)));
            b_spcm(i,j) = 1/(1+s(i)*upsilon);
        end
    end
    surf(taus,s,b_spcm)
    hold on
end

alpha(.8)
ylabel('SPCM similarity value \Delta_{ij}','FontSize', 20)
xlabel('Similarity Tolerance \tau','FontSize', 20)
zlabel('f(\Delta_{ij},\tau)','FontSize', 20)
title('Bounded-SPCM for d=6 in \Sigma \in R^{dxd}','FontSize', 20)
colormap(hot)
shading interp
colorbar