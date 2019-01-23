%% Scaling--Rotation Curve examples(including SQ curves) 

 %% 3x3 example of sca-rot curves
neval = 7;
figure(1);clf;

a = [-24.1247  -31.4399   22.8799]';

X = diag([15,5,1]); 
r = rotM(a,pi/2);
M = r*diag([15,5,1])*r';
rowid = [3 1];iannotate = false;
drawscrotcurve(X,M,neval,rowid,iannotate);

M = diag([15,5,1]); 
X = 3*diag([9,12,8]);
rowid = [3 2];iannotate = false;
drawscrotcurve(X,M,neval,rowid,iannotate);

M = diag([15,5,1]); 
r = rotM(a,pi/2);
X = r*diag([9,12,8])*r';
rowid = [3 3];iannotate = false;
drawscrotcurve(X,M,neval,rowid,iannotate);

figure(1);
for i = 1:21
    subplot(3,7,i);zoom(1.5);
end
 

%% 3x3 example of sca-rot curves compared to other deformations 
neval = 11;
icase = 4;
switch icase
    case 1
a = [-24.1247  -31.4399   22.8799]';
M = diag([15,5,1]); 
r = rotM(a,pi/3);
X = r*diag([15,5,1])*r';
    case 2
M = diag([15,5,1]); 
X = diag([7,12,8]); 
    case 3
M = diag([15,5,1]); 
r = rotM(a,pi/2);
X = r*diag([9,12,8])*r';
    case 4
r = rotM(-a,pi/6);
M = r*diag([15,2,1])*r'; 
r = rotM(a,pi/6);
X = r*diag([100,2,1])*r';
    case 5
r = rotM(-a,pi/6);
M = r*diag([15,2,1])*r'; 
r = rotM(a,pi/6);
X = r*diag([10,6,2])*r';
end

rowid = [4 1];
figure(2);clf; 
[~, paramsscrot]=MSRcurve(M,X);
[~,dist,Uarray,Darray,~,~]= scrotcurve(paramsscrot.U,paramsscrot.D,paramsscrot.V,paramsscrot.Lambda,linspace(0,1,neval));

axisdisplaylength = 1.2*sqrt(max([diag(X);diag(M)]));
rotaxis = real([paramsscrot.A(3,2), paramsscrot.A(1,3), paramsscrot.A(2,1)]);
angle = norm(rotaxis);
rotaxis = axisdisplaylength*rotaxis/angle;
coll = [1 0 0; 0 1 0; 0 0 1];
rowid(2) = 1;
axisdisplaylength = axisdisplaylength/2;
for t = 1:neval
    subplot(rowid(1),neval, neval*(rowid(2)-1)+ t)
    unow = Uarray{t};
    dnow = diag(Darray{t});
    plotellipsoid(real(unow),dnow);
    hold on;
    for k = 1:3;
        plot3([0, sqrt(dnow(k))*unow(1,k)],...
            [0, sqrt(dnow(k))*unow(2,k)],...
            [0, sqrt(dnow(k))*unow(3,k)],...
            'linewidth',1,'color',coll(k,:));
    end
    plot3([-rotaxis(1), rotaxis(1)],...
        [-rotaxis(2), rotaxis(2)],...
        [-rotaxis(3), rotaxis(3)],...
        'k','linewidth',1,'color',[0 0 0]);
    xlim([-axisdisplaylength, axisdisplaylength]);
    ylim([-axisdisplaylength, axisdisplaylength]);
    zlim([-axisdisplaylength, axisdisplaylength]);
    
    axis off
end
rowid(2) = 2; % Now Euclidean! 
for t = 1:neval
    subplot(rowid(1),neval, neval*(rowid(2)-1)+ t)
    EuclideanCurve = (1- (t-1)/(neval-1) )*M + (t-1)/(neval-1)*X;
    [unow dnow]=svd(EuclideanCurve);    
    plotellipsoid(real(unow),diag(dnow));
    hold on;
    for k = 1:3;
        plot3([0, sqrt(dnow(k))*unow(1,k)],...
            [0, sqrt(dnow(k))*unow(2,k)],...
            [0, sqrt(dnow(k))*unow(3,k)],...
            'linewidth',1,'color',coll(k,:));
    end
    plot3([-rotaxis(1), rotaxis(1)],...
        [-rotaxis(2), rotaxis(2)],...
        [-rotaxis(3), rotaxis(3)],...
        'k','linewidth',1,'color',[0 0 0]);
    xlim([-axisdisplaylength, axisdisplaylength]);
    ylim([-axisdisplaylength, axisdisplaylength]);
    zlim([-axisdisplaylength, axisdisplaylength]);
    axis off 
end

rowid(2) = 3; % Now Log-Euclidean! 
for t = 1:neval
    subplot(rowid(1),neval, neval*(rowid(2)-1)+ t) 
    aM = (1- (t-1)/(neval-1) ); 
    aX = (t-1)/(neval-1);
    
    EuclideanCurve =  expm( aM*logm(M) + aX *logm(X));
    [unow dnow]=svd(EuclideanCurve);    
    plotellipsoid(real(unow),diag(dnow));
    hold on;
    for k = 1:3;
        plot3([0, sqrt(dnow(k))*unow(1,k)],...
            [0, sqrt(dnow(k))*unow(2,k)],...
            [0, sqrt(dnow(k))*unow(3,k)],...
            'linewidth',1,'color',coll(k,:));
    end
    plot3([-rotaxis(1), rotaxis(1)],...
        [-rotaxis(2), rotaxis(2)],...
        [-rotaxis(3), rotaxis(3)],...
        'k','linewidth',1,'color',[0 0 0]);
    xlim([-axisdisplaylength, axisdisplaylength]);
    ylim([-axisdisplaylength, axisdisplaylength]);
    zlim([-axisdisplaylength, axisdisplaylength]);
    axis off 
end

rowid(2) = 4; % Now Affine-Ivariant! 
for t = 1:neval
    subplot(rowid(1),neval, neval*(rowid(2)-1)+ t) 
    aM = (1- (t-1)/(neval-1) ); 
    aX = (t-1)/(neval-1);
    Mhalf = M^(1/2);
    Mhalfm = M^(-1/2);
    EuclideanCurve =  Mhalf *expm( aX* logm(Mhalfm * X * Mhalfm) ) * Mhalf ;
    [unow dnow]=svd(EuclideanCurve);    
    plotellipsoid(real(unow),diag(dnow));
    hold on;
    for k = 1:3;
        plot3([0, sqrt(dnow(k))*unow(1,k)],...
            [0, sqrt(dnow(k))*unow(2,k)],...
            [0, sqrt(dnow(k))*unow(3,k)],...
            'linewidth',1,'color',coll(k,:));
    end
    plot3([-rotaxis(1), rotaxis(1)],...
        [-rotaxis(2), rotaxis(2)],...
        [-rotaxis(3), rotaxis(3)],...
        'k','linewidth',1,'color',[0 0 0]);
    xlim([-axisdisplaylength, axisdisplaylength]);
    ylim([-axisdisplaylength, axisdisplaylength]);
    zlim([-axisdisplaylength, axisdisplaylength]);
    axis off 
    
end
for i = 1:44
subplot(4,11,i);
zoom(2)
end

subplot(4,11,1);
text(-10,0,'Scaling-rotation','FontSize',12)
subplot(4,11,12);
text(-10,0,'Euclidean','FontSize',12)
subplot(4,11,23);
text(-10,0,'Log-Euclidean','FontSize',12)
subplot(4,11,34);
text(-10,0,'Affine-invariant','FontSize',12) 
% %%
% print('-dpng','Scaling_Rotation_Paper_Fig2');
% print('-dpsc','Scaling_Rotation_Paper_Fig2'); 


%%
neval = 101;
[~, paramsscrot]=MSRcurve(M,X);
[~,dist,Uarray,Darray,~,~]= scrotcurve(paramsscrot.U,paramsscrot.D,paramsscrot.V,paramsscrot.Lambda,linspace(0,1,neval));
determinants = zeros(4,neval);
maxeval = zeros(4,neval);
mineval = zeros(4,neval);
fa = zeros(4,neval);
md = zeros(4,neval);
%FA = frational anisotropy; 
%MD = mean diffusivity; 

 for t = 1:neval 
    determinants(4,t) = det(Darray{t});
    maxeval(4,t) = max(diag(Darray{t}));
    mineval(4,t) = min(diag(Darray{t}));
    fa(4,t) = FA(diag(Darray{t}));
    md(4,t) = mean(diag(Darray{t}));
    aM = (1- (t-1)/(neval-1) ); 
    aX = (t-1)/(neval-1);
    Euc = (1- (t-1)/(neval-1) )*M + (t-1)/(neval-1)*X;
    determinants(1,t) = det(Euc);
    maxeval(1,t) = max(eig(Euc));
    mineval(1,t) = min(eig(Euc));
    fa(1,t) = FA(eig(Euc));
    md(1,t) = mean(eig(Euc));
    LogEuc = expm( aM*logm(M) + aX *logm(X)) ;
    determinants(2,t) = det(LogEuc );
    maxeval(2,t) = max(eig(LogEuc));
    mineval(2,t) = min(eig(LogEuc));
    fa(2,t) = FA(eig(LogEuc));
    md(2,t) = mean(eig(LogEuc));
    Mhalf = M^(1/2);
    Mhalfm = M^(-1/2);   
    AIR = Mhalf *expm( aX* logm(Mhalfm * X * Mhalfm) ) * Mhalf;
    determinants(3,t) = det( AIR );
    maxeval(3,t) = max(eig(AIR));
    mineval(3,t) = min(eig(AIR));
    fa(3,t) = FA(eig(AIR));
    md(3,t) = mean(eig(AIR));
end

figure(3);clf;
subplot(2,3,1);
tt = linspace(0,1,neval); hold on;
plot(tt, determinants(4,:),'-k','Linewidth',2);
plot(tt, determinants(1,:),':b','Linewidth',2);
plot(tt, determinants(2,:),'-.g','Linewidth',2);
plot(tt, determinants(3,:),'--r','Linewidth',1.5);
%legend('Euclidean','Log-Euclidean','Affine-invariant','Scaling-Rotation');
ylabel('Determinant');xlabel('t');
subplot(2,3,2);
tt = linspace(0,1,neval);hold on;
plot(tt, fa(4,:),'-k','Linewidth',2);
plot(tt, fa(1,:),':b','Linewidth',2);
plot(tt, fa(2,:),'-.g','Linewidth',2);
plot(tt, fa(3,:),'--r','Linewidth',1.5);
%legend('Euclidean','Log-Euclidean','Affine-invariant','Scaling-Rotation');
ylabel('FA');xlabel('t'); 
subplot(2,3,3);
tt = linspace(0,1,neval);hold on;
plot(tt, md(4,:),'-k','Linewidth',2);
plot(tt, md(1,:),':b','Linewidth',2);
plot(tt, md(2,:),'-.g','Linewidth',2);
plot(tt, md(3,:),'--r','Linewidth',1.5);
legend('Scaling-Rotation','Euclidean','Log-Euclidean','Affine-invariant','Location','Best');
ylabel('MD');xlabel('t'); 
%  %%
% print('-dpng','Scaling_Rotation_Paper_Fig3');
% print('-dpsc','Scaling_Rotation_Paper_Fig3'); 



%%
neval = 101;
[~, paramsscrot]=MSRcurve(M,X);
[~,dist,Uarray,Darray,~,~]= scrotcurve(paramsscrot.U,paramsscrot.D,paramsscrot.V,paramsscrot.Lambda,linspace(0,1,neval));
determinants = zeros(4,neval);
maxeval = zeros(4,neval);
mineval = zeros(4,neval);
fa = zeros(4,neval);
md = zeros(4,neval);
rotation_angle = zeros(4,neval);
%FA = frational anisotropy; 
%MD = mean diffusivity; 
[~,maxid_forSR]=max(diag(Darray{1})); 
a = Uarray{1}(:,maxid_forSR);


 for t = 1:neval 
    determinants(4,t) = det(Darray{t});
    maxeval(4,t) = max(diag(Darray{t}));
    mineval(4,t) = min(diag(Darray{t}));
    fa(4,t) = FA(diag(Darray{t}));
    md(4,t) = mean(diag(Darray{t}));
    rotation_angle(4,t) = real(acos(abs(  a'* Uarray{t}(:,maxid_forSR) )));
        
    aM = (1- (t-1)/(neval-1) ); 
    aX = (t-1)/(neval-1);
    Euc = (1- (t-1)/(neval-1) )*M + (t-1)/(neval-1)*X;
    determinants(1,t) = det(Euc);
    [vecEuc,valEuc]=eig(Euc); valEuc = diag(valEuc);
    [~,maxid_forNOW]=max(valEuc); 
    
    maxeval(1,t) = max(valEuc);
    mineval(1,t) = min(valEuc);
    fa(1,t) = FA(valEuc);
    md(1,t) = mean(valEuc);
    
    rotation_angle(1,t) = real(acos(abs( vecEuc(:,maxid_forNOW)' *   a        )));
    
    LogEuc = expm( aM*logm(M) + aX *logm(X)) ;
    determinants(2,t) = det(LogEuc );
    [vecLogEuc,valLogEuc]=eig(LogEuc); valLogEuc = diag(valLogEuc);
    [~,maxid_forNOW]=max(valLogEuc); 
    
    maxeval(2,t) = max(valLogEuc);
    mineval(2,t) = min(valLogEuc);
    fa(2,t) = FA(valLogEuc);
    md(2,t) = mean(valLogEuc);
    
    rotation_angle(2,t) = real(acos(abs( vecLogEuc(:,maxid_forNOW)' *   a        )));
    
    
    
    Mhalf = M^(1/2);
    Mhalfm = M^(-1/2);   
    AIR = Mhalf *expm( aX* logm(Mhalfm * X * Mhalfm) ) * Mhalf;
    determinants(3,t) = det( AIR );
    
    [vecAIR,valAIR]=eig(AIR); valAIR = diag(valAIR);
    [~,maxid_forNOW]=max(valAIR); 
    
    maxeval(3,t) = max(valAIR);
    mineval(3,t) = min(valAIR);
    fa(3,t) = FA(valAIR);
    md(3,t) = mean(valAIR);
    
    rotation_angle(3,t) = real(acos(abs( vecAIR(:,maxid_forNOW)' *   a        )));
    
    
 end
rotation_angle = rotation_angle * 180 / pi ; 
determinants = log10(determinants);
 
figure(3);clf;
subplot(2,4,2);
tt = linspace(0,1,neval); hold on;
plot(tt, determinants(4,:),'-k','Linewidth',2);
plot(tt, determinants(1,:),':b','Linewidth',2);
plot(tt, determinants(2,:),'-.g','Linewidth',2);
plot(tt, determinants(3,:),'--r','Linewidth',1.5);
%legend('Euclidean','Log-Euclidean','Affine-invariant','Scaling-Rotation');
ylabel('log(Determinant)');xlabel('t');
subplot(2,4,3);
tt = linspace(0,1,neval);hold on;
plot(tt, fa(4,:),'-k','Linewidth',2);
plot(tt, fa(1,:),':b','Linewidth',2);
plot(tt, fa(2,:),'-.g','Linewidth',2);
plot(tt, fa(3,:),'--r','Linewidth',1.5);
%legend('Euclidean','Log-Euclidean','Affine-invariant','Scaling-Rotation');
ylabel('FA');xlabel('t'); 
subplot(2,4,4);
tt = linspace(0,1,neval);hold on;
plot(tt, md(4,:),'-k','Linewidth',2);
plot(tt, md(1,:),':b','Linewidth',2);
plot(tt, md(2,:),'-.g','Linewidth',2);
plot(tt, md(3,:),'--r','Linewidth',1.5);
legend('Scaling-Rotation','Euclidean','Log-Euclidean','Affine-invariant','Location','Best');
ylabel('MD');xlabel('t'); 
subplot(2,4,1); 
tt = linspace(0,1,neval);hold on;
plot(tt, rotation_angle(4,:),'-k','Linewidth',2);
plot(tt, rotation_angle(1,:),':b','Linewidth',2);
plot(tt, rotation_angle(2,:),'-.g','Linewidth',2);
plot(tt, rotation_angle(3,:),'--r','Linewidth',1.5);
%legend('Scaling-Rotation','Euclidean','Log-Euclidean','Affine-invariant','Location','Best');
ylabel('Rotation angle');xlabel('t'); 
%  
% print('-dpng','Scaling_Rotation_Paper_Fig3new');
% print('-dpsc','Scaling_Rotation_Paper_Fig3new'); 
% 



%% Now draw a cone 

figure(1);clf;
subplot(3,2,[1 3]);hold on;


% 2x2 scaling-rotation curves example



p = 2;
epsilon = 1; 
X = [exp(epsilon/2) 0; 0 exp(-epsilon/2)];
tt = 60*(pi/180) ;
R = [cos(tt) -sin(tt); sin(tt) cos(tt)];
Y = 2*R*X*R';
% generate versions of $X$
[U,D]=pickaversion(X) ; 

[Is, nsignchange]= signchangematrix(p);
[P, nperm]= permutematrix(p); 
% generate a version of $Y$
[V,Lambda]=pickaversion(Y); 

%%%%% 
%%%%% 
D = D.^2; epsilon = 2; X = [exp(epsilon/2) 0; 0 exp(-epsilon/2)];
Lambda = 2*(Lambda/2).^2; Y = 2*R*X*R'; 
%%%%%
%%%%%


x = vecd(X);scatter3(x(1),x(2),x(3),'ob','filled')
y = vecd(Y);scatter3(y(1),y(2),y(3),'og','filled')
gd = zeros(nsignchange*nperm,1); 
  colorpool =   [0     0     1
     0     1     1
     1     1     0
     1     0     0]; 
cnt = 1; 
for i =1:nsignchange;
    for j = 1:nperm;
        Ustar = U*P{j}*Is{i};        
        Dstar = P{j}'*D*P{j};        
        [T,dist]= scrotcurve(Ustar,Dstar,V,Lambda);
        plot3(T(1,:),T(2,:),T(3,:),'Color',colorpool(cnt,:)*0.8,'LineWidth',2)
        gd(cnt) = dist;
        cnt = cnt+1;
    end
end 

% Create cone
[t,z] = meshgrid((0:0.05:2)*pi, (0:0.1:3));
a = z.*(1 + cos(t));
b = z.*(1 - cos(t));
c = z*sqrt(2).*sin(t);

% figure 
BW = 1;
% set(gcf, 'Position', [0 0 600 600]) % [0 350 800 800]
if BW,
    surf(a,b,c,'FaceColor',0.9*[1 1 1],'EdgeColor',0.4*[1 1 1])
else
    surf(a,b,c,'FaceColor','red','EdgeColor',[0.2 0.2 0.2])
end
camlight; lighting phong
axis equal, axis([0 3 0 3 -1.2 1.8])
alpha(0.7), view([126 30])

% Add axes, labels
line([0 3.0],[0 0],[0 0],'Color','black','LineWidth',2)
line([0 0],[0 3.0],[0 0],'Color','black','LineWidth',2)
line([0 0],[0 0],[-1.9 1.9],'Color','black','LineWidth',2)
% h = text(f*f*Lxy,0,0,'a'); set(h, 'Fontsize', 18)
% h = text(0,f*f*Lxy,0,'b'); set(h, 'Fontsize', 18)
% h = text(0,0,f*f*Lz/2,'c'); set(h, 'Fontsize', 18)

% Add arroweads
%annotation('arrow', [0.5168 0.5168], [0.8 0.926]);
%annotation('arrow', [0.2 0.165], [0.537 0.516]);
%annotation('arrow', [0.835 0.87], [0.537 0.516]);


axis equal;

title('The cone of Sym^+(2)'); 

legend('X','Y',...
       ['\chi_1, length ' num2str(gd(1),2)],...
       ['\chi_2, length ' num2str(gd(2),2)],...
       ['\chi_3, length ' num2str(gd(3),2)],...
       ['\chi_4, length ' num2str(gd(4),2)])
choosei = 4;
   
   
subplot(3,2,[2 4]);hold on;
xyzpoints = zeros(3,4);
xyzpointsG = zeros(3,4);
cnt = 1;
for i =1:nsignchange;
    for j = 1:nperm; 
        Ustar = U*P{j}*Is{i};
        Dstar = P{j}'*D*P{j};
        xyzpoints(:,cnt) = [log(diag(Dstar)); atan2(Ustar(2,1),Ustar(1,1))];
        Ustar = V*P{j}*Is{i};
        Dstar = P{j}'*Lambda*P{j};
        if choosei == cnt;
            VV = Ustar;
            LambdaV = Dstar;
        end
        xyzpointsG(:,cnt) = [log(diag(Dstar)); atan2(Ustar(2,1),Ustar(1,1))];
        cnt = cnt+1;
    end
end

scatter3(xyzpointsG(1,:),xyzpointsG(2,:),xyzpointsG(3,:),'g','filled');
scatter3(xyzpoints(1,1),xyzpoints(2,1),xyzpoints(3,1),'ob','filled');hold on;
scatter3(xyzpoints(1,2),xyzpoints(2,2),xyzpoints(3,2),'+b');hold on;
scatter3(xyzpoints(1,3),xyzpoints(2,3),xyzpoints(3,3),'*b');hold on;
scatter3(xyzpoints(1,4),xyzpoints(2,4),xyzpoints(3,4),'db');hold on;
% scatter3(xyzpoints(1,1),xyzpoints(2,1),2*pi+xyzpoints(3,1),'ob','filled');hold on;
% scatter3(xyzpoints(1,2),xyzpoints(2,2),2*pi+xyzpoints(3,2),'+b');hold on;
% scatter3(xyzpoints(1,3),xyzpoints(2,3),2*pi+xyzpoints(3,3),'*b');hold on;
% scatter3(xyzpoints(1,4),xyzpoints(2,4),2*pi+xyzpoints(3,4),'db');hold on;
% scatter3(xyzpoints(1,1),xyzpoints(2,1),-2*pi+xyzpoints(3,1),'ob','filled');hold on;
% scatter3(xyzpoints(1,2),xyzpoints(2,2),-2*pi+xyzpoints(3,2),'+b');hold on;
% scatter3(xyzpoints(1,3),xyzpoints(2,3),-2*pi+xyzpoints(3,3),'*b');hold on;
% scatter3(xyzpoints(1,4),xyzpoints(2,4),-2*pi+xyzpoints(3,4),'db');hold on;
set(gca,'ZTick',[-pi 0  pi]);
set(gca,'ZTickLabel',{'-pi', '0', 'pi'});
zlabel('SO(2)');
xlabel('log(d1)');
ylabel('log(d2)');
cnt = 1;
 
neval = 101;
for i =1:nsignchange;
    for j = 1:nperm;
        Ustar = U*P{j}*Is{i};
        Dstar = P{j}'*D*P{j}; 
        [T,dist,Uarray,Darray,A,L]= scrotcurve(VV,LambdaV,Ustar,Dstar,linspace(0,1,neval));
        xyzpoints = zeros(3,neval);   
        for t = 1:neval  
            unow = Uarray{t};
            dnow = Darray{t};
            xyzpoints(:,t) = [log(diag(dnow)); atan2(unow(2,1),unow(1,1))];
              if t > 1;
                  [~,ord]=min(abs(xyzpoints(3,t-1) - [xyzpoints(3,t); xyzpoints(3,t)+2*pi;xyzpoints(3,t)-2*pi]));
                  switch ord
                      case 1
                      case 2
                      xyzpoints(3,t) = xyzpoints(3,t)+2*pi;
                      case 3 
                      xyzpoints(3,t) = xyzpoints(3,t)-2*pi;
                  end
              end
        end 
        plot3(xyzpoints(1,:),xyzpoints(2,:),xyzpoints(3,:),'Color',colorpool(cnt,:)*0.8,'LineWidth',2);
        cnt = cnt+1;
       % text(xyzpoints(1,round(neval/2)),xyzpoints(2,round(neval/2)),xyzpoints(3,round(neval/2)),[num2str(dist)]);
    end
end

axis equal
grid on;
view([27 84]);
title('SO \times Diag^+ (2)'); 

neval = 7;
subplot(3,1,3);hold on;
rowid = [3 3];iannotate = false;
drawscrotcurve(Y,X,neval,rowid,iannotate);
subplot(3,7,18);
title('the trajectory of (red) minimal scaling-rotation curve')   






% %%
% print('-dpng','Scaling_Rotation_Paper_Fig4');
% print('-dpsc','Scaling_Rotation_Paper_Fig4'); 

%%  SUPPLEMENT
%%  SUPPLEMENT
%%  SUPPLEMENT
%%  SUPPLEMENT
%%  SUPPLEMENT
%%  SUPPLEMENT
%%  SUPPLEMENT
%%  SUPPLEMENT
%%  SUPPLEMENT
%%  SUPPLEMENT
%%  SUPPLEMENT
%%  SUPPLEMENT
%%  SUPPLEMENT
%%  SUPPLEMENT
%%  SUPPLEMENT
%%  SUPPLEMENT
%%  SUPPLEMENT
%%  SUPPLEMENT
%%  SUPPLEMENT
%%  SUPPLEMENT
%%  SUPPLEMENT
%%  SUPPLEMENT
%%  SUPPLEMENT
%%  SUPPLEMENT
%%  SUPPLEMENT
%%  SUPPLEMENT
%%  SUPPLEMENT
%%  SUPPLEMENT
%%  SUPPLEMENT
%%  SUPPLEMENT
%%  SUPPLEMENT
%%  SUPPLEMENT




%% 3x3 example of sca-rot curves compared to other deformations 
neval = 11;
icase = 6;
switch icase
    case 1
a = [-24.1247  -31.4399   22.8799]';
M = diag([15,5,1]); 
r = rotM(a,pi/3);
X = r*diag([15,5,1])*r';
    case 2
%M = diag([9,6,5]);  X = diag([5,9,7]); 
M = diag([15,5,1]);  X = diag([7,12,8]); 
    case 3
M = diag([15,5,1]); 
r = rotM(a,pi/3);
X = r*diag([9,12,8])*r';
    case 4
a = [-24.1247  -31.4399   22.8799]';
r = rotM(-a,pi/6);
M = r*diag([15,2,1])*r'; 
r = rotM(a,pi/6);
X = r*diag([100,2,1])*r';
    case 5
%a = [-24.1247  -31.4399   22.8799]';
r = rotM(a,pi/6);
M = r*diag([1,15,4])*r'; 
r = rotM(-a,pi/6);
X = r*diag([2,10,8])*r';
    case 6
r = rotM(-a,pi/6);
M = r*diag([4,4,4])*r'; 
r = rotM(a,pi/3);
X = r*diag([11,11,6])*r';
end

rowid = [5 1];
figure(2);clf; 
[~, paramsscrot]=MSRcurve(M,X);
[~,dist,Uarray,Darray,~,~]= scrotcurve(paramsscrot.U,paramsscrot.D,paramsscrot.V,paramsscrot.Lambda,linspace(0,1,neval));
if icase ==2 
    PP = permutematrix(3);
    ipermutemat = 2;
    % PP{ipermutemat} *paramsscrot.D * PP{ipermutemat}'
    for i = 1:neval;
        Uarray{i} = Uarray{i}* PP{ipermutemat}';
        Darray{i} = PP{ipermutemat} * Darray{i}* PP{ipermutemat}';
    end
end
if icase == 3
    PP = permutematrix(3);
    ipermutemat = 4;
    % PP{ipermutemat} *paramsscrot.D * PP{ipermutemat}'
    for i = 1:neval;
        Uarray{i} = Uarray{i}* PP{ipermutemat}';
        Darray{i} = PP{ipermutemat} * Darray{i}* PP{ipermutemat}';
    end
end
     
axisdisplaylength = 1.6*sqrt(max([diag(X);diag(M)]));
rotaxis = real([paramsscrot.A(3,2), paramsscrot.A(1,3), paramsscrot.A(2,1)]);
angle = norm(rotaxis);
rotaxis = axisdisplaylength*rotaxis/angle;
coll = [1 0 0; 0 1 0; 0 0 1];
rowid(2) = 1;
axisdisplaylength = axisdisplaylength/2;
for t = 1:neval
    subplot(rowid(1),neval, neval*(rowid(2)-1)+ t)
    unow = Uarray{t};
    dnow = diag(Darray{t});
    plotellipsoid(real(unow),dnow);
    hold on;
    for k = 1:3;
        plot3([0, sqrt(dnow(k))*unow(1,k)],...
            [0, sqrt(dnow(k))*unow(2,k)],...
            [0, sqrt(dnow(k))*unow(3,k)],...
            'linewidth',1,'color',coll(k,:));
    end
    plot3([-rotaxis(1), rotaxis(1)],...
        [-rotaxis(2), rotaxis(2)],...
        [-rotaxis(3), rotaxis(3)],...
        'k','linewidth',1,'color',[0 0 0]);
    xlim([-axisdisplaylength, axisdisplaylength]);
    ylim([-axisdisplaylength, axisdisplaylength]);
    zlim([-axisdisplaylength, axisdisplaylength]);
    
    axis off
end

rowid(2) = 2; % Now Euclidean! 
for t = 1:neval
    subplot(rowid(1),neval, neval*(rowid(2)-1)+ t)
    EuclideanCurve = (1- (t-1)/(neval-1) )*M + (t-1)/(neval-1)*X;

    [unow dnow]=svd(EuclideanCurve);   
    if icase == 6;
    if t==1 ; 
        unow = paramsscrot.U;
        dnow = paramsscrot.D;
    end
    end
    plotellipsoid(real(unow),diag(dnow));
    hold on;
    for k = 1:3;
        plot3([0, sqrt(dnow(k))*unow(1,k)],...
            [0, sqrt(dnow(k))*unow(2,k)],...
            [0, sqrt(dnow(k))*unow(3,k)],...
            'linewidth',1,'color',coll(k,:));
    end
    plot3([-rotaxis(1), rotaxis(1)],...
        [-rotaxis(2), rotaxis(2)],...
        [-rotaxis(3), rotaxis(3)],...
        'k','linewidth',1,'color',[0 0 0]);
    xlim([-axisdisplaylength, axisdisplaylength]);
    ylim([-axisdisplaylength, axisdisplaylength]);
    zlim([-axisdisplaylength, axisdisplaylength]);
    axis off 
end

rowid(2) = 3; % Now Log-Euclidean! 
for t = 1:neval
    subplot(rowid(1),neval, neval*(rowid(2)-1)+ t) 
    aM = (1- (t-1)/(neval-1) ); 
    aX = (t-1)/(neval-1);
    
    EuclideanCurve =  expm( aM*logm(M) + aX *logm(X));
 
    
    [unow dnow]=svd(EuclideanCurve);   
        if icase == 6;
    if t==1 ; 
        unow = paramsscrot.U;
        dnow = paramsscrot.D;
    end
    end
    plotellipsoid(real(unow),diag(dnow));
    hold on;
    for k = 1:3;
        plot3([0, sqrt(dnow(k))*unow(1,k)],...
            [0, sqrt(dnow(k))*unow(2,k)],...
            [0, sqrt(dnow(k))*unow(3,k)],...
            'linewidth',1,'color',coll(k,:));
    end
    plot3([-rotaxis(1), rotaxis(1)],...
        [-rotaxis(2), rotaxis(2)],...
        [-rotaxis(3), rotaxis(3)],...
        'k','linewidth',1,'color',[0 0 0]);
    xlim([-axisdisplaylength, axisdisplaylength]);
    ylim([-axisdisplaylength, axisdisplaylength]);
    zlim([-axisdisplaylength, axisdisplaylength]);
    axis off 
end

rowid(2) = 4; % Now Affine-Ivariant! 
for t = 1:neval
    subplot(rowid(1),neval, neval*(rowid(2)-1)+ t) 
    aM = (1- (t-1)/(neval-1) ); 
    aX = (t-1)/(neval-1);
    Mhalf = M^(1/2);
    Mhalfm = M^(-1/2);
    EuclideanCurve =  Mhalf *expm( aX* logm(Mhalfm * X * Mhalfm) ) * Mhalf ;
 
        
    [unow dnow]=svd(EuclideanCurve);   
    if icase == 6;
    if t==1 ; 
        unow = paramsscrot.U;
        dnow = paramsscrot.D;
    end
    end 
    plotellipsoid(real(unow),diag(dnow));
    hold on;
    for k = 1:3;
        plot3([0, sqrt(dnow(k))*unow(1,k)],...
            [0, sqrt(dnow(k))*unow(2,k)],...
            [0, sqrt(dnow(k))*unow(3,k)],...
            'linewidth',1,'color',coll(k,:));
    end
    plot3([-rotaxis(1), rotaxis(1)],...
        [-rotaxis(2), rotaxis(2)],...
        [-rotaxis(3), rotaxis(3)],...
        'k','linewidth',1,'color',[0 0 0]);
    xlim([-axisdisplaylength, axisdisplaylength]);
    ylim([-axisdisplaylength, axisdisplaylength]);
    zlim([-axisdisplaylength, axisdisplaylength]);
    axis off 
    
end



rowid(2) = 5; % Now SQ-curve 
 [dist, interp,evaluesONLY,evecONLY]= SQcurve(M,X,neval);
for t = 1:neval
    subplot(rowid(1),neval, neval*(rowid(2)-1)+ t) 
     
    %[unow dnow]=svd(interp(:,:,t));  
    unow = evecONLY(:,:,t);
    dnow = evaluesONLY(:,:,t);
       
    plotellipsoid(real(unow),diag(dnow));
    hold on;
    for k = 1:3;
        plot3([0, sqrt(dnow(k))*unow(1,k)],...
            [0, sqrt(dnow(k))*unow(2,k)],...
            [0, sqrt(dnow(k))*unow(3,k)],...
            'linewidth',1,'color',coll(k,:));
    end
    plot3([-rotaxis(1), rotaxis(1)],...
        [-rotaxis(2), rotaxis(2)],...
        [-rotaxis(3), rotaxis(3)],...
        'k','linewidth',1,'color',[0 0 0]);
    xlim([-axisdisplaylength, axisdisplaylength]);
    ylim([-axisdisplaylength, axisdisplaylength]);
    zlim([-axisdisplaylength, axisdisplaylength]);
    axis off 
    
end




for i = 1:55
subplot(5,11,i);
zoom(2)
end
subplot(5,11,1);
text(-10,0,'Scaling-rotation','FontSize',12)
subplot(5,11,12);
text(-10,0,'Euclidean','FontSize',12)
subplot(5,11,23);
text(-10,0,'Log-Euclidean','FontSize',12)
subplot(5,11,34);
text(-10,0,'Affine-invariant','FontSize',12)
subplot(5,11,45);
text(-10,0,'Scaling-quaternion','FontSize',12)
% %%
% print('-dpng',['Scaling_Rotation_Paper_Fig2_supplment' num2str(icase) 'R']);
% print('-dpsc',['Scaling_Rotation_Paper_Fig2_supplment' num2str(icase) 'R']);
% 

%%
neval = 101;
[~, paramsscrot]=MSRcurve(M,X);
[~,dist,Uarray,Darray,~,~]= scrotcurve(paramsscrot.U,paramsscrot.D,paramsscrot.V,paramsscrot.Lambda,linspace(0,1,neval));
[~,~, SQinterpEvalues,SQinterpEvectors]= SQcurve(M,X,neval);
determinants = zeros(5,neval);
maxeval = zeros(5,neval);
mineval = zeros(5,neval);
fa = zeros(5,neval);
md = zeros(5,neval);
rotation_angle = zeros(5,neval);
%FA = frational anisotropy; 
%MD = mean diffusivity; 

[~,maxid_forSR]=max(diag(Darray{1})); 
a = Uarray{1}(:,maxid_forSR);

[~,maxid_forSQ]=max(diag(SQinterpEvalues(:,:,1))); 
aQ = SQinterpEvectors(:,maxid_forSQ,1); 

 for t = 1:neval 
    determinants(4,t) = det(Darray{t});
    maxeval(4,t) = max(diag(Darray{t}));
    mineval(4,t) = min(diag(Darray{t}));
    fa(4,t) = FA(diag(Darray{t}));
    md(4,t) = mean(diag(Darray{t}));
    %rotation_angle(4,t) = real(acos(abs(  a'* Uarray{t}(:,maxid_forSR) )));
        
    rotation_angle(4,t) = real(acos( (trace(Uarray{t} * Uarray{1}') - 1 ) / 2 ) );
     
    aM = (1- (t-1)/(neval-1) ); 
    aX = (t-1)/(neval-1);
    Euc = (1- (t-1)/(neval-1) )*M + (t-1)/(neval-1)*X;
    determinants(1,t) = det(Euc);
    [vecEuc,valEuc]=eig(Euc); valEuc = diag(valEuc);
    [~,maxid_forNOW]=max(valEuc); 
    
    maxeval(1,t) = max(valEuc);
    mineval(1,t) = min(valEuc);
    fa(1,t) = FA(valEuc);
    md(1,t) = mean(valEuc);
    
    rotation_angle(1,t) = real(acos(abs( vecEuc(:,maxid_forNOW)' *   a        )));
    
    LogEuc = expm( aM*logm(M) + aX *logm(X)) ;
    determinants(2,t) = det(LogEuc );
    [vecLogEuc,valLogEuc]=eig(LogEuc); valLogEuc = diag(valLogEuc);
    [~,maxid_forNOW]=max(valLogEuc); 
    
    maxeval(2,t) = max(valLogEuc);
    mineval(2,t) = min(valLogEuc);
    fa(2,t) = FA(valLogEuc);
    md(2,t) = mean(valLogEuc);
    
    rotation_angle(2,t) = real(acos(abs( vecLogEuc(:,maxid_forNOW)' *   a        )));
    
    
    
    Mhalf = M^(1/2);
    Mhalfm = M^(-1/2);   
    AIR = Mhalf *expm( aX* logm(Mhalfm * X * Mhalfm) ) * Mhalf;
    determinants(3,t) = det( AIR );
    
    [vecAIR,valAIR]=eig(AIR); valAIR = diag(valAIR);
    [~,maxid_forNOW]=max(valAIR); 
    
    maxeval(3,t) = max(valAIR);
    mineval(3,t) = min(valAIR);
    fa(3,t) = FA(valAIR);
    md(3,t) = mean(valAIR);
    
    rotation_angle(3,t) = real(acos(abs( vecAIR(:,maxid_forNOW)' *   a        )));
    
    
    determinants(5,t) = det(SQinterpEvalues(:,:,t));
    maxeval(5,t) = max(diag(SQinterpEvalues(:,:,t)));
    mineval(5,t) = min(diag(SQinterpEvalues(:,:,t)));
    fa(5,t) = FA(diag(SQinterpEvalues(:,:,t)));
    md(5,t) = mean(diag(SQinterpEvalues(:,:,t)));
    
%    rotation_angle(5,t) = real(acos(abs( SQinterpEvectors(:,maxid_forSQ,t)' *   aQ        )));
        rotation_angle(5,t) = real(acos( (trace(SQinterpEvectors(:,:,t) * SQinterpEvectors(:,:,1)') - 1 ) / 2 ) );

 end
rotation_angle = rotation_angle * 180 / pi ; 
determinants = log10(determinants);
 
figure(3);clf;
subplot(2,4,2);
tt = linspace(0,1,neval); hold on;
plot(tt, determinants(4,:),'-k','Linewidth',2);
plot(tt, determinants(1,:),':b','Linewidth',2);
plot(tt, determinants(2,:),'-.g','Linewidth',2);
plot(tt, determinants(3,:),'--r','Linewidth',1.5);
plot(tt, determinants(5,:),':m','Linewidth',1.5);
%legend('Euclidean','Log-Euclidean','Affine-invariant','Scaling-Rotation');
ylabel('log(Determinant)');xlabel('t');
subplot(2,4,3);
tt = linspace(0,1,neval);hold on;
plot(tt, fa(4,:),'-k','Linewidth',2);
plot(tt, fa(1,:),':b','Linewidth',2);
plot(tt, fa(2,:),'-.g','Linewidth',2);
plot(tt, fa(3,:),'--r','Linewidth',1.5);
plot(tt, fa(5,:),':m','Linewidth',1.5);
%legend('Euclidean','Log-Euclidean','Affine-invariant','Scaling-Rotation');
ylabel('FA');xlabel('t'); 
subplot(2,4,4);
tt = linspace(0,1,neval);hold on;
plot(tt, md(4,:),'-k','Linewidth',2);
plot(tt, md(1,:),':b','Linewidth',2);
plot(tt, md(2,:),'-.g','Linewidth',2);
plot(tt, md(3,:),'--r','Linewidth',1.5);
plot(tt, md(5,:),':m','Linewidth',1.5);
legend('Scaling-rotation','Euclidean','Log-Euclidean','Affine-invariant','Scaling-quaternion','Location','Best');
ylabel('MD');xlabel('t'); 
subplot(2,4,1); 
tt = linspace(0,1,neval);hold on;
plot(tt, rotation_angle(4,:),'-k','Linewidth',2);
% plot(tt, rotation_angle(1,:),':b','Linewidth',2);
% plot(tt, rotation_angle(2,:),'-.g','Linewidth',2);
% plot(tt, rotation_angle(3,:),'--r','Linewidth',1.5);

%plot(tt, rotation_angle(4,:),'-k','Linewidth',4);
% scatter(tt, rotation_angle(1,:),'.b','Linewidth',2);
% scatter(tt, rotation_angle(2,:),'.g','Linewidth',2);
% scatter(tt, rotation_angle(3,:),'.r','Linewidth',1.5);
plot(tt, rotation_angle(5,:),':m','Linewidth',1.5);
%legend('Scaling-Rotation','Euclidean','Log-Euclidean','Affine-invariant','Location','Best');
ylabel('Rotation angle');xlabel('t'); 
if icase == 6;
    ylim([-1,1]);
end
  
% %%
% print('-dpng',['Scaling_Rotation_Paper_Fig3_supplment' num2str(icase) 'R']);
% print('-dpsc',['Scaling_Rotation_Paper_Fig3_supplment' num2str(icase) 'R']);
% 
% 
































%% Different scaling - K 

neval = 11;
icase = 9;
switch icase
    case 9
%M = diag([9,6,5]);  X = diag([5,9,7]); 
M = diag([15,5,1]);  X = diag([7,12,8]); 
end

KK = [1, 0.4, .1];  
rowid = [5 1];
figure(2);clf; 
for rowi = 1:3
[~, paramsscrot]=MSRcurve(M,X,1e-14, KK(rowi));
[~,~,Uarray,Darray,~,~]= scrotcurve(paramsscrot.U,paramsscrot.D,paramsscrot.V,paramsscrot.Lambda,linspace(0,1,neval));
if rowi ==1 
    PP = permutematrix(3);
    ipermutemat = 2;
    % PP{ipermutemat} *paramsscrot.D * PP{ipermutemat}'
    for i = 1:neval;
        Uarray{i} = Uarray{i}* PP{ipermutemat}';
        Darray{i} = PP{ipermutemat} * Darray{i}* PP{ipermutemat}';
    end
end


axisdisplaylength = 1.9*sqrt(max([diag(X);diag(M)]));
rotaxis = real([paramsscrot.A(3,2), paramsscrot.A(1,3), paramsscrot.A(2,1)]);
angle = norm(rotaxis);
rotaxis = axisdisplaylength*rotaxis/angle/4;
coll = [1 0 0; 0 1 0; 0 0 1];
rowid(2) = rowi+1;
axisdisplaylength = axisdisplaylength/2;
for t = 1:neval
    subplot(rowid(1),neval, neval*(rowid(2)-1)+ t)
    unow = Uarray{t};
    dnow = diag(Darray{t});
    plotellipsoid(real(unow),dnow);
    hold on;
    for k = 1:3;
        plot3([0, sqrt(dnow(k))*unow(1,k)],...
            [0, sqrt(dnow(k))*unow(2,k)],...
            [0, sqrt(dnow(k))*unow(3,k)],...
            'linewidth',1,'color',coll(k,:));
    end
    plot3([-rotaxis(1), rotaxis(1)],...
        [-rotaxis(2), rotaxis(2)],...
        [-rotaxis(3), rotaxis(3)],...
        'k','linewidth',1,'color',[0 0 0]);
    xlim([-axisdisplaylength, axisdisplaylength]);
    ylim([-axisdisplaylength, axisdisplaylength]);
    zlim([-axisdisplaylength, axisdisplaylength]);
    
    axis off
end
end
 





 
% 
% rowid(2) = 1; % Now Log-Euclidean! 
% for t = 1:neval
%     subplot(rowid(1),neval, neval*(rowid(2)-1)+ t) 
%     aM = (1- (t-1)/(neval-1) ); 
%     aX = (t-1)/(neval-1);
%     
%     EuclideanCurve =  expm( aM*logm(M) + aX *logm(X));
%     [unow dnow]=svd(EuclideanCurve);    
%     plotellipsoid(real(unow),diag(dnow))
%     hold on;
%     for k = 1:3;
%         plot3([0, sqrt(dnow(k))*unow(1,k)],...
%             [0, sqrt(dnow(k))*unow(2,k)],...
%             [0, sqrt(dnow(k))*unow(3,k)],...
%             'linewidth',1,'color',coll(k,:));
%     end 
%     xlim([-axisdisplaylength, axisdisplaylength]);
%     ylim([-axisdisplaylength, axisdisplaylength]);
%     zlim([-axisdisplaylength, axisdisplaylength]);
%     axis off 
% end

rowid(2) = 1; % Now Affine-Ivariant! 
for t = 1:neval
    subplot(rowid(1),neval, neval*(rowid(2)-1)+ t) 
    aM = (1- (t-1)/(neval-1) ); 
    aX = (t-1)/(neval-1);
    Mhalf = M^(1/2);
    Mhalfm = M^(-1/2);
    EuclideanCurve =  Mhalf *expm( aX* logm(Mhalfm * X * Mhalfm) ) * Mhalf ;
    [unow dnow]=svd(EuclideanCurve);    
    plotellipsoid(real(unow),diag(dnow));
    hold on;
    for k = 1:3;
        plot3([0, sqrt(dnow(k))*unow(1,k)],...
            [0, sqrt(dnow(k))*unow(2,k)],...
            [0, sqrt(dnow(k))*unow(3,k)],...
            'linewidth',1,'color',coll(k,:));
    end 
    xlim([-axisdisplaylength, axisdisplaylength]);
    ylim([-axisdisplaylength, axisdisplaylength]);
    zlim([-axisdisplaylength, axisdisplaylength]);
    axis off 
    
end
 
rowid(2) = 5; % Now SQ-curve 
 [dist, interp,evaluesONLY,evecONLY]= SQcurve(M,X,neval);
for t = 1:neval
    subplot(rowid(1),neval, neval*(rowid(2)-1)+ t) 
     
    %[unow dnow]=svd(interp(:,:,t));  
    unow = evecONLY(:,:,t);
    dnow = evaluesONLY(:,:,t);
       
    plotellipsoid(real(unow),diag(dnow));
    hold on;
    for k = 1:3;
        plot3([0, sqrt(dnow(k))*unow(1,k)],...
            [0, sqrt(dnow(k))*unow(2,k)],...
            [0, sqrt(dnow(k))*unow(3,k)],...
            'linewidth',1,'color',coll(k,:));
    end 
    plot3([-rotaxis(1), rotaxis(1)],...
        [-rotaxis(2), rotaxis(2)],...
        [-rotaxis(3), rotaxis(3)],...
        'k','linewidth',1,'color',[0 0 0]);
    xlim([-axisdisplaylength, axisdisplaylength]);
    ylim([-axisdisplaylength, axisdisplaylength]);
    zlim([-axisdisplaylength, axisdisplaylength]);
    axis off  
end
 

for i = 1:55
subplot(5,11,i);
zoom(2.8)
end

subplot(5,11,1);
text(-10,0,'Affine-invariant','FontSize',12)
subplot(5,11,12);
text(-10,0,'Scaling-rotation (k = 1)','FontSize',12)
subplot(5,11,23);
text(-10,0,'Scaling-rotation (k = 0.4)','FontSize',12)
subplot(5,11,34);
text(-10,0,'Scaling-rotation (k = 0.1)','FontSize',12)
subplot(5,11,45);
text(-10,0,'Scaling-quaternion','FontSize',12)
% %%
% print('-dpng',['Scaling_Rotation_Paper_Fig2_supplment' num2str(icase) 'R']);
% print('-dpsc',['Scaling_Rotation_Paper_Fig2_supplment' num2str(icase) 'R']);
% 

%%
neval = 101;
[~, paramsscrot]=MSRcurve(M,X);
[~,dist,Uarray,Darray,~,~]= scrotcurve(paramsscrot.U,paramsscrot.D,paramsscrot.V,paramsscrot.Lambda,linspace(0,1,neval));
[~,~, SQinterpEvalues,SQinterpEvectors]= SQcurve(M,X,neval);
determinants = zeros(5,neval);
maxeval = zeros(5,neval);
mineval = zeros(5,neval);
fa = zeros(5,neval);
md = zeros(5,neval);
rotation_angle = zeros(5,neval);
%FA = frational anisotropy; 
%MD = mean diffusivity; 
[~,maxid_forSR]=max(diag(Darray{1})); 
a = Uarray{1}(:,maxid_forSR);

[~, paramsscrot]=MSRcurve(M,X,1e-14, 0.4);
[~,~,Uarray4,Darray4,~,~]= scrotcurve(paramsscrot.U,paramsscrot.D,paramsscrot.V,paramsscrot.Lambda,linspace(0,1,neval));

[~,maxid_forSR4]=max(diag(Darray4{1})); 
a4 = Uarray4{1}(:,maxid_forSR4);

[~, paramsscrot]=MSRcurve(M,X,1e-14, 0.1);
[~,~,Uarray1,Darray1,~,~]= scrotcurve(paramsscrot.U,paramsscrot.D,paramsscrot.V,paramsscrot.Lambda,linspace(0,1,neval));

[~,maxid_forSR1]=max(diag(Darray1{1})); 
a1 = Uarray1{1}(:,maxid_forSR1);

[~,maxid_forSQ]=max(diag(SQinterpEvalues(:,:,1))); 
aQ = SQinterpEvectors(:,maxid_forSQ,1); 


 for t = 1:neval 
     
    determinants(4,t) = det(Darray{t});
    maxeval(4,t) = max(diag(Darray{t}));
    mineval(4,t) = min(diag(Darray{t}));
    fa(4,t) = FA(diag(Darray{t}));
    md(4,t) = mean(diag(Darray{t}));
    
    rotation_angle(4,t) = real(acos( (trace(Uarray{t} * Uarray{1}') - 1 ) / 2 ) );
    
    %rotation_angle(4,t) = real(acos(abs(  a'* Uarray{t}(:,maxid_forSR) )));
    
    
    determinants(6,t) = det(Darray4{t});
    maxeval(6,t) = max(diag(Darray4{t}));
    mineval(6,t) = min(diag(Darray4{t}));
    fa(6,t) = FA(diag(Darray4{t}));
    md(6,t) = mean(diag(Darray4{t}));
    %rotation_angle(6,t) = real(acos(abs(  a4'* Uarray4{t}(:,maxid_forSR4) )));
    rotation_angle(6,t) = real(acos( (trace(Uarray4{t} * Uarray4{1}') - 1 ) / 2 ) );
   
    
    determinants(7,t) = det(Darray1{t});
    maxeval(7,t) = max(diag(Darray1{t}));
    mineval(7,t) = min(diag(Darray1{t}));
    fa(7,t) = FA(diag(Darray1{t}));
    md(7,t) = mean(diag(Darray1{t}));
    %rotation_angle(7,t) = real(acos(abs(  a1'* Uarray1{t}(:,maxid_forSR1) )));
    rotation_angle(7,t) = real(acos( (trace(Uarray1{t} * Uarray1{1}') - 1 ) / 2 ) );
    
    
       
    aM = (1- (t-1)/(neval-1) ); 
    aX = (t-1)/(neval-1);
    Euc = (1- (t-1)/(neval-1) )*M + (t-1)/(neval-1)*X;
    determinants(1,t) = det(Euc);
    [vecEuc,valEuc]=eig(Euc); valEuc = diag(valEuc);
    [~,maxid_forNOW]=max(valEuc); 
    
    maxeval(1,t) = max(valEuc);
    mineval(1,t) = min(valEuc);
    fa(1,t) = FA(valEuc);
    md(1,t) = mean(valEuc);
    
    rotation_angle(1,t) = real(acos(abs( vecEuc(:,maxid_forNOW)' *   a        )));
    
    LogEuc = expm( aM*logm(M) + aX *logm(X)) ;
    determinants(2,t) = det(LogEuc );
    [vecLogEuc,valLogEuc]=eig(LogEuc); valLogEuc = diag(valLogEuc);
    [~,maxid_forNOW]=max(valLogEuc); 
    
    maxeval(2,t) = max(valLogEuc);
    mineval(2,t) = min(valLogEuc);
    fa(2,t) = FA(valLogEuc);
    md(2,t) = mean(valLogEuc);
    
    rotation_angle(2,t) = real(acos(abs( vecLogEuc(:,maxid_forNOW)' *   a        )));
    
    
    
    Mhalf = M^(1/2);
    Mhalfm = M^(-1/2);   
    AIR = Mhalf *expm( aX* logm(Mhalfm * X * Mhalfm) ) * Mhalf;
    determinants(3,t) = det( AIR );
    
    [vecAIR,valAIR]=eig(AIR); valAIR = diag(valAIR);
    [~,maxid_forNOW]=max(valAIR); 
    
    maxeval(3,t) = max(valAIR);
    mineval(3,t) = min(valAIR);
    fa(3,t) = FA(valAIR);
    md(3,t) = mean(valAIR);
    
    rotation_angle(3,t) = real(acos(abs( vecAIR(:,maxid_forNOW)' *   a        )));
    
    
    determinants(5,t) = det(SQinterpEvalues(:,:,t));
    maxeval(5,t) = max(diag(SQinterpEvalues(:,:,t)));
    mineval(5,t) = min(diag(SQinterpEvalues(:,:,t)));
    fa(5,t) = FA(diag(SQinterpEvalues(:,:,t)));
    md(5,t) = mean(diag(SQinterpEvalues(:,:,t)));
    %rotation_angle(5,t) = real(acos(abs( SQinterpEvectors(:,maxid_forSQ,t)' *   aQ  ))); 
    rotation_angle(5,t) = real(acos( (trace(SQinterpEvectors(:,:,t) * SQinterpEvectors(:,:,1)') - 1 ) / 2 ) );
 end

 determinants = log10(determinants);
 rotation_angle = rotation_angle * 180 / pi ; 

figure(3);clf;
subplot(2,4,1); 
tt = linspace(0,1,neval);hold on;
plot(tt, rotation_angle(3,:),'--r','Linewidth',1.5);
plot(tt, rotation_angle(4,:),'-k','Linewidth',3);
plot(tt, rotation_angle(6,:),'-.b','Linewidth',3);
plot(tt, rotation_angle(7,:),'--g','Linewidth',3); 
plot(tt, rotation_angle(5,:),':m','Linewidth',1.5);
ylabel('Rotation angle');xlabel('t'); 

subplot(2,4,2);
tt = linspace(0,1,neval); hold on;
plot(tt, determinants(3,:),'--r','Linewidth',1.5);
plot(tt, determinants(4,:),'-k','Linewidth',3);
plot(tt, determinants(6,:),'-.b','Linewidth',3);
plot(tt, determinants(7,:),'--g','Linewidth',3);
plot(tt, determinants(5,:),':m','Linewidth',1.5);
%legend('Euclidean','Log-Euclidean','Affine-invariant','Scaling-Rotation');
ylabel('log(Determinant)');xlabel('t');
subplot(2,4,3);
tt = linspace(0,1,neval);hold on;
plot(tt, fa(3,:),'--r','Linewidth',1.5);
plot(tt, fa(4,:),'-k','Linewidth',3);
plot(tt, fa(6,:),'-.b','Linewidth',3);
plot(tt, fa(7,:),'--g','Linewidth',3);
plot(tt, fa(5,:),':m','Linewidth',1.5);
%legend('Euclidean','Log-Euclidean','Affine-invariant','Scaling-Rotation');
ylabel('FA');xlabel('t'); 
subplot(2,4,4);
tt = linspace(0,1,neval);hold on;
plot(tt, md(3,:),'--r','Linewidth',1.5);
plot(tt, md(4,:),'-k','Linewidth',3);
plot(tt, md(6,:),'-.b','Linewidth',3);
plot(tt, md(7,:),'--g','Linewidth',3);
plot(tt, md(5,:),':m','Linewidth',1.5);
legend('Affine-invariant',...
    'Scaling-rotation (k = 1)',...
    'Scaling-rotation (k = .4)',...
    'Scaling-rotation (k = .1)', ...
        'Scaling-quaternion','Location','Best');
ylabel('MD');xlabel('t'); 
% 
% %%
% print('-dpng',['Scaling_Rotation_Paper_Fig3_supplment' num2str(icase) 'R']);
% print('-dpsc',['Scaling_Rotation_Paper_Fig3_supplment' num2str(icase) 'R']);
% 



















