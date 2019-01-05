%% Logistic function tests
dim = 6;
x = 0:0.05:4;
beta = 1;
y = exp(-(dim)*x.^2);
figure('Color',[1 1 1])
plot(x,y,'-*r');
xlabel('SPCM dis-similarity $d(X,Y)$','Interpreter','LaTex');
ylabel('SPCM similarity $s(X,Y)$','Interpreter','LaTex');
grid on;

%%
upsilon = 10^(exp(-dim));
y = 1./(1+x*upsilon);
figure('Color',[1 1 1])
plot(x,y,'-*r');
xlabel('SPCM dis-similarity $d(X,Y)$','Interpreter','LaTex');
ylabel('SPCM similarity $s(X,Y)$','Interpreter','LaTex');
grid on;
