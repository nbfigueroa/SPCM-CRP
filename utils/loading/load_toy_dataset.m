function [Sigmas, True_Labels] = load_toy_dataset(type, display, randomize)


switch type
    case 0
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Load Toy Data (3D) for Illustrative Example %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % Number of samples and clusters
        N = 9; K = 3;
        
        % Initial Values
        Sigma = zeros(3,3,N);
        rand_vals  = randn(1,20);
        rand_vals  = rand_vals(abs(rand_vals) > 0.5);
        base_randn = abs(randsample(rand_vals,1));
        V_1  = eye(3);
        
        % Linear Ellipsoid Cluster (3 linears)
        Lambda      = zeros(3);
        Lambda(3,3) = 2*base_randn;
        Lambda(2,2) = Lambda(3,3);
        Lambda(1,1) = 5*Lambda(2,2);
        Sigma(:,:,1) = V_1*Lambda*V_1';
        roll = -0.7854; pitch = -0.7854; yaw= 0;
        R_2 = eul2rotm([yaw,pitch,roll]);
        V_2 = R_2*V_1;
        Sigma(:,:,2) = V_2*(1.25*Lambda)*V_2';
        roll = 0.7854; pitch = -0.7854; yaw= 1.5484;
        R_3 = eul2rotm([yaw,pitch,roll]);
        V_3 = R_3*V_1;
        Sigma(:,:,3) = V_3*(0.5*Lambda)*V_3';
        
        % Spherical Ellipsoid Cluster (3 spheres)
        lambda    = 3*base_randn;
        Lambda      = lambda*eye(3);
        Sigma(:,:,4) = V_1*Lambda*V_1';
        Sigma(:,:,5) = V_1*(2*Lambda)*V_1';
        Sigma(:,:,6) = V_1*(0.5*Lambda)*V_1';        
        
        % Planar Ellipsoid Cluster (3 planars)
        Lambda      = zeros(3);
        Lambda(1,1) = 2*base_randn;
        Lambda(2,2) = 5*Lambda(1,1);
        Lambda(3,3) = Lambda(2,2);
        Sigma(:,:,7) = V_1*Lambda*V_1';
        roll = 0; pitch = 1.578; yaw= 0;
        R_2 = eul2rotm([yaw,pitch,roll]);
        V_2 = R_2*V_1;
        Sigma(:,:,8) = V_2*(0.75*Lambda)*V_2';
        roll = 0.7854; pitch = -0.7854; yaw= 1.5484;
        R_3 = eul2rotm([yaw,pitch,roll]);
        V_3 = R_3*V_1;
        Sigma(:,:,9) = V_3*(1.25*Lambda)*V_3';
        
        % Data structures
        true_labels = [1 1 1 2 2 2 3 3 3];
        for k=1:N
            sigmas{k}   = Sigma(:,:,k);
        end
        
        if display == 1
            Mu      = zeros(3,N);
            Mu(1,:) = [-5 -5 -5  20 20 20 50 50 50 ];
            Mu(2,:) = [0 -20 20  0 -20 20 0 -20 20];
            
            % Clustered Sigmas GMM
            colors = jet(K);
            
            figure('Color',[1 1 1])
            for k=1:N
                [V,D]=eig(Sigma(:,:,k));
                scale = 1;
                [x,y,z] = created3DgaussianEllipsoid(Mu(:,k),V,D, scale); hold on;
                
                % Draw frame
                H = eye(4);
                H(1:3,1:3) = eye(3);
                H(1:3,4)   = Mu(:,k);
                
                % Draw World Reference Frame
                drawframe(H,2.5); hold on;
                
                % This makes the ellipsoids beautiful
                surf(x, y, z,'FaceColor',colors(true_labels(k),:),'FaceAlpha', 0.25, 'FaceLighting','phong','EdgeColor','none');
                camlight
            end
            axis equal;
            grid on;
            xlabel('$x_1$', 'Interpreter', 'LaTex', 'FontSize',15);
            ylabel('$x_2$', 'Interpreter', 'LaTex','FontSize',15);
            zlabel('$x_3$', 'Interpreter', 'LaTex','FontSize',15);
            set(gca,'FontSize',16);
            title('Toy 3D Covariance Matrices Dataset','FontSize', 20, 'Interpreter','LaTex')
        end
        
    case 1
        %%%%%%%%%%%%%%%%%%%%%%
        % Load Toy Data (6D) %
        %%%%%%%%%%%%%%%%%%%%%%
        
        sigmas = [];                
        tot_1=100;tot_2=100;tot_3=100;
        
        tot = tot_1;
        rows = floor(sqrt(tot));
        cols = ceil(tot/rows);
        if display == 1, figure('Color',[1 1 1]); suptitle('Toy 6D Covariance Matrices Dataset pt.1'); end
        for i=1:tot
            D = diag(abs(diag(eye(6))*randn*100));
%             [Q R] = qr(randn(6,6));
            Q = orth(rand(6,6));
            sigmas{i} = Q*(D)*Q';
            if display == 1
                subplot(rows,cols,i)
                imagesc(sigmas{i})
                colormap(pink)
                colorbar
                axis square
            end
        end
        
        
        lambda = [5 5 5 1 1 1];
        if display == 1, figure('Color',[1 1 1]);suptitle('Toy 6D Covariance Matrices Dataset pt.2'); end
        iter = 1;
        tot = tot_2;
        rows = floor(sqrt(tot));
        cols = ceil(tot/rows);
        for i=tot+1:2*tot
            D = diag(abs(lambda*randn)*50);
%             [Q R] = qr(randn(6,6));
            Q = orth(rand(6,6));
            sigmas{i} = Q*(D)*Q';
            if display == 1
                subplot(rows,cols,iter)
                imagesc(sigmas{i})
                colormap(pink)
                colorbar
                axis square
            end
            iter = iter + 1;
        end        
        
        lambda = [1 10 20 30 40 50];
        if display == 1, figure('Color',[1 1 1]); suptitle('Toy 6D Covariance Matrices Dataset pt.3'); end
        iter = 1;
        tot = tot_3;
        rows = floor(sqrt(tot));
        cols = ceil(tot/rows);
        for i=2*tot+1:3*tot
            D = diag(abs(lambda*randn*10));
%             [Q R] = qr(randn(6,6));
            Q = orth(rand(6,6));
            
            sigmas{i} = Q*(D)*Q';
            if display == 1
                subplot(rows,cols,iter)
                imagesc(sigmas{i})
                colormap(pink)
                colorbar
                axis square
            end
            iter = iter + 1;
        end                
        true_labels = [ones(1,tot_1) , ones(1,tot_2)*2, ones(1,tot_3)*3];

    case 2
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