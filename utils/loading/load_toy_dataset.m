function [Sigmas, True_Labels] = load_toy_dataset(type, display, randomize)

if strcmp(type, '3d') || strcmp(type, '4d')  
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Load Toy Data (3D) %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
        if display == 1, figure('Color',[1 1 1]); end  
        angle = pi/2;
        
        %Ellipsoid 1
        Cov = ones(3,3) + diag([1 1 1]);        
        mu = [0 0 0]';
        behavs_theta{1,1} = Cov;
        [V1,D1] = eig(Cov);
        CoordRot = rotx(-angle);
        [x,y,z] = created3DgaussianEllipsoid(mu,V1,D1^1/2);        
        if display==1
            mesh(x,y,z,'EdgeColor','blue','Edgealpha',0.2);
            hidden off
            hold on;
        end

        %Ellipsoid 2: Scale+Noise
        D1m = diag(D1)*1.3 + abs(randn(3,1).*[0.35 0.37 0.3]');
        D1m = diag(D1)*0.5;
        Covs2 = V1*(diag(D1m))*V1';
        behavs_theta{1,2} = Covs2;
        [V2,D2] = eig(Covs2);
        mu = [1 0 0]';
        [x,y,z] = created3DgaussianEllipsoid(mu,V2,D2^1/2);
        if display==1
            mesh(x,y,z,'EdgeColor','blue','Edgealpha',0.2);
            hidden off
            hold on;
        end

        %Ellipsoid 3: Rotated Coordinates
        CoordRot = [1 0 0; 0 1 0 ; 0 0 -1];
        V2_rot = CoordRot*V2;
        Covs3 = V2_rot*D2*V2_rot';
        behavs_theta{1,3} = Covs3;
        [V3,D3] = eig(Covs3);
        mu = [2 0 0]';
        [x,y,z] = created3DgaussianEllipsoid(mu,V3,D3^1/2);
        if display==1
            mesh(x,y,z,'EdgeColor','blue','Edgealpha',0.2);
            hidden off
            hold on;
        end
        
        %Ellipsoid 4: Different
        [Q R] = qr(randn(3,3));
        D = diag([4 3 0.5]);
        Cov = Q*(D)*Q';
        behavs_theta{1,4} = Cov;
        [V4,D4] = eig(behavs_theta{1,4});
        mu = [4 0 0]';
        [x,y,z] = created3DgaussianEllipsoid(mu,V4,D4^1/2);
        if display == 1
            mesh(x,y,z,'EdgeColor','red','Edgealpha',0.2);
            hidden off
            hold on;
        end

        %Ellipsoid 5: Different Rotated + Scaled
        [Q2 R] = qr(randn(3,3));
        D = diag([4 3 0.5]*0.75);
        behavs_theta{1,5} = (Q2)*(D)*(Q2)';
        [V5,D5] = eig(behavs_theta{1,5});
        mu = [6 0 0]';
        [x,y,z] = created3DgaussianEllipsoid(mu,V5,D5^1/2);
        if display == 1
            mesh(x,y,z,'EdgeColor','red','Edgealpha',0.2);
            hidden off
            hold on;
            colormap jet
            alpha(0.75)
            axis equal
            grid off
            axis off
            xlabel('x'); ylabel('y'); zlabel('z'); 
            title('Toy 3D Covariance Matrices Dataset','FontSize', 20)
        end
        
        sigmas = behavs_theta;
        true_labels = [ones(1,3) , ones(1,2)*2];
end

if strcmp(type,'4d')
    close all
    %%%%%%%%%%%%%%%%%%%%%%
    % Load Toy Data (4D) %
    %%%%%%%%%%%%%%%%%%%%%%
    if display == 1, figure('Color',[1 1 1]); end  
    
    behavs_theta4 = [];
    tmp = eye(4);
    tmp(1:3,1:3) = behavs_theta{1};
    tmp(4,1:3) = ones(1,3);
    tmp(1:3,4) = ones(3,1);
    tmp(4,4) = 2;
    behavs_theta4{1} = tmp;
    if display ==1
        subplot(2,3,1)
        imagesc(tmp)
        colormap(pink)
        colorbar
        axis square
    end
    
    tmp = eye(4);
    tmp(1:3,1:3) = behavs_theta{2};
    tmp(4,1:3) = ones(1,3)*(0.5);
    tmp(1:3,4) = ones(3,1)*(0.5);
    behavs_theta4{2} = tmp;
    if display == 1
        subplot(2,3,2)
        imagesc(tmp)
        colormap(pink)
        colorbar
        axis square
    end
    
    % Rotated and Scaled
    tmp = behavs_theta4{2};
    [V D] = eig(tmp);
    [Q R] = qr(randn(4,4)*.5);
    behavs_theta4{3} = cov2cor(Q*tmp*Q');       
    tmp = behavs_theta4{3};
    if display == 1
        subplot(2,3,3)
        imagesc(tmp)
        colormap(pink)
        colorbar
        axis square
    end
    
    [V D]= eig(tmp);
    behavs_theta4{4} = V*(D*2)*V';
    if display == 1
        subplot(2,3,4)
        imagesc(behavs_theta4{4})
        colormap(pink)
        colorbar
        axis square
    end
    
    % Different
    [Q R] = qr(randn(4,4));
    D = diag([10 4 3 1]);
    behavs_theta4{5} = Q*(D)*Q';

    if display == 1
        subplot(2,3,5)
        imagesc(behavs_theta4{5})
        colormap(pink)
        colorbar
        axis square
    end
    
    D = diag([10 4 3 1]*3.5);
    behavs_theta4{6} = Q*(D)*Q';
    if display == 1
        subplot(2,3,6)
        imagesc(behavs_theta4{6})
        suptitle('Toy 4D Covariance Matrices Dataset')
        colormap(pink)
        colorbar
        axis square        
    end
    
    clear sigmas true_labels
    sigmas = behavs_theta4;
    true_labels = [ones(1,4) , ones(1,2)*2];
end


if strcmp(type,'6d')
    
    %%%%%%%%%%%%%%%%%%%%%%
    % Load Toy Data (6D) %
    %%%%%%%%%%%%%%%%%%%%%%
    
    behavs_theta6toy = [];
    
    
    tot_1=20;tot_2=20;tot_3=20;
    
    tot = tot_1;
    rows = floor(sqrt(tot));
    cols = ceil(tot/rows);
    if display == 1, figure('Color',[1 1 1]); suptitle('Toy 6D Covariance Matrices Dataset pt.1'); end    
    for i=1:tot
        D = diag(abs(diag(eye(6))*randn));
        [Q R] = qr(randn(6,6));
        behavs_theta6toy{i} = Q*(D)*Q';
        if display == 1
            subplot(rows,cols,i)
            imagesc(behavs_theta6toy{i})
            colormap(pink)
            colorbar
            axis square
        end
    end
 
    
    lambda = [1 10 10 10 1 1];
    if display == 1, figure('Color',[1 1 1]);suptitle('Toy 6D Covariance Matrices Dataset pt.2'); end
    iter = 1;
    tot = tot_2;
    rows = floor(sqrt(tot));
    cols = ceil(tot/rows);
    for i=tot+1:2*tot
        D = diag(abs(lambda*randn)*0.5);
        [Q R] = qr(randn(6,6));
        behavs_theta6toy{i} = Q*(D)*Q';
        if display == 1
            subplot(rows,cols,iter)
            imagesc(behavs_theta6toy{i})
            colormap(pink)
            colorbar
            axis square
        end
        iter = iter + 1;
    end
    
    
    lambda = [1 20 30 40 50 60];
    if display == 1, figure('Color',[1 1 1]); suptitle('Toy 6D Covariance Matrices Dataset pt.3'); end
    iter = 1;
    tot = tot_3;
    rows = floor(sqrt(tot));
    cols = ceil(tot/rows);
    for i=2*tot+1:3*tot
        D = diag(abs(lambda*randn*0.5));
        [Q R] = qr(randn(6,6));
        behavs_theta6toy{i} = Q*(D)*Q';
        if display == 1
            subplot(rows,cols,iter)
            imagesc(behavs_theta6toy{i})
            colormap(pink)
            colorbar
            axis square
        end
        iter = iter + 1;
    end
    
    
    sigmas = behavs_theta6toy;
    true_labels = [ones(1,tot_1) , ones(1,tot_2)*2, ones(1,tot_3)*3];    
end


if (randomize == 1) 
    fprintf('Randomize Indices: 1 \n');
    [Sigmas True_Labels] = randomize_data(sigmas, true_labels);
elseif (randomize == 0) 
    fprintf('Randomize Indices: 0 \n');
    Sigmas = sigmas;
    True_Labels = true_labels;
end


end