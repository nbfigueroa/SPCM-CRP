curve_test = [1 0.95 0.93 0.85 0.78 0.65 0.55 0.45 0.35 0.25 0.17 0.15 0.15 0.15 0.13 0.1 0.1 0.1 0.1 0.01 0.00];
% curve_test = [1 0.85 0.65 0.019 0.015 0.012 0.011 0.0105 0.0001];
curve_test = curve_test.^4;

pow = 6;
d_pseudo = eig(mpower(K,pow));
curve_test = sort(d_pseudo','descend');
curve_test = curve_test(1:50);

% % Compute Diff of BIC Scores
rel_eigen_gap    = [0 abs(diff(curve_test))];
curve_test_diff  = ([0 diff(curve_test)]);
curve_test_diff2 = [0 (diff(curve_test_diff))];

% Find optimal value on RSS curve
[~, opt_Ks_BIC_line] = ml_curve_opt(curve_test,'line');
opt_BIC_vals_line    = curve_test(opt_Ks_BIC_line);

% Other options with the 'derivatives' approach
[max_delta, max_delta_id] = max(rel_eigen_gap);

[~, opt_Ks_BIC_ders] = ml_curve_opt(curve_test,'derivatives')
opt_BIC_vals_ders    = curve_test(opt_Ks_BIC_ders);

% Plot Results
figure('Color', [1 1 1])
subplot(1,2,1)
plot(1:length(curve_test), curve_test, 'k-*','LineWidth',1.5); hold on;
scatter(max_delta_id, curve_test(max_delta_id), 150, [0 1 0],'o','LineWidth',1.5); hold on;
scatter(opt_Ks_BIC_ders(1), opt_BIC_vals_ders(1), 150, [0 0 1],'d','LineWidth',1.5); hold on;
scatter(opt_Ks_BIC_line, opt_BIC_vals_line, 150, [0 0 1],'s','LineWidth',1.5); hold on;
grid on;
title('Eigenvalues of $\mathbf{K}$ ','Interpreter','LaTex','FontSize',14);
xlabel('Eigenvalue $\lambda_i$ Index','Interpreter','LaTex','FontSize',14);
ylabel('Eigenvalues $\lambda_i$','Interpreter','LaTex','FontSize',14);
legend({'$\lambda_i$','$P  = \arg\max_{i}(|\delta|\lambda_i)$','$P_i=\arg\max_{i}(\delta^2\lambda_i)$','$P_l = \arg\max_i(||\vec{\rho}_i||)$'},'Interpreter','LaTex','FontSize',12)
% legend({'$\lambda_i$','$P  = \arg\max_{i}(|\delta|\lambda_i)$'},'Interpreter','LaTex','FontSize',12)
xlim([1 length(curve_test)])

subplot(1,2,2)
plot(1:length(curve_test), rel_eigen_gap,    '-*g','LineWidth',1.5); hold on;
plot(1:length(curve_test), curve_test_diff,  '-*r','LineWidth',1.5); hold on;
plot(1:length(curve_test), curve_test_diff2, '-*b','LineWidth',1.5); hold on;
grid on;
title('Eigengap Curves ','Interpreter','LaTex','FontSize',14);
xlabel('Eigenvalue $\lambda_i$ Index','Interpreter','LaTex','FontSize',14);
legend({'$|\delta|\lambda_i$','$\delta\lambda_i$','$\delta^{2}\lambda_i$'},'Interpreter','LaTex','FontSize',12)
xlim([1 length(curve_test)])