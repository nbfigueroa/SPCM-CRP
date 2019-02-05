%% New method: 
% 1) Fit a normal or logistic distribution of Laplacian Eigenvalues
if exist('h2','var') && isvalid(h2), delete(h2);end
h2 = figure('Color',[1 1 1]);
clear title xlabel ylabel

d_ = d;

subplot(3,1,1)
plot(d_','-*r'); hold on
plot(M, d_(M),'ok','MarkerSize',10);
xlabel('Eigenvector Index','Interpreter','Latex');
ylabel('$\lambda$','Interpreter','Latex');
title('Laplacian EigenValues', 'Interpreter','Latex');
grid on;

% Alternative.. fit a normal distribution to the eigenvalues
normal = fitdist(d,'Normal');
ndist = pdf(normal, d_);
M_n = sum(ndist <= 1e-2)

if M_n ==0
    M_n = 1;
end

subplot(3,1,2)
plot(d_,ndist,'-k','linewidth',2); hold on;
plot(d_, 0,'*r','MarkerSize',5); hold on;
plot(d_(M_n), ndist(M_n),'ok','MarkerSize',10)
title('Fitted $\mathcal{N}(\lambda;\mu,\sigma)$ on Laplacian Eigenvalues', 'Interpreter','Latex');
grid on;

% Alternative.. fit a logistic distribution to the eigenvalues
logis = fitdist(d,'logistic');
ldist = pdf(logis, d_);
M_l = sum(ldist <= 1e-4)

if M_l ==0
    M_l = 1;
end

subplot(3,1,3)
plot(d_,ldist,'-k','linewidth',2); hold on;
plot(d_, 0,'*r','MarkerSize',5); hold on;
plot(d_(M_l), ldist(M_l),'oK','MarkerSize',10)
title('Fitted $\log(\lambda;\mu,\sigma)$ on Laplacian Eigenvalues', 'Interpreter','Latex');
grid on;
