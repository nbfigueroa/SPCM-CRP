function [Mu_test, sigma_test, true_labels, dataset_name] = load_SPD_simulations(sim_type, df)

switch sim_type
    case 1 
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %     Test 1 (Intro): Ellipsoid deformation with isotropic scalings     %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Creating test centers
        Mu_test      = zeros(3,40);
        Mu_test(1,:) = [0:5:95,-5:-5:-100];
        
        % Linear ellipsoid
        sigma_test = [];
        sigma_test(:,:,1) = diag([1.25 1.25 5]);
        for k=2:20
            ani_scale         = eye(3);            
            ani_scale         = (0.5 + sqrt(k/20 * 4.5) )* ani_scale;
            sigma_test(:,:,k) = ani_scale*sigma_test(:,:,1)*ani_scale';
        end
        
        pitch = 0; yaw= 0;
        rolls = 0.1570:0.0785:pi/2+0.0785;
        for k=21:40
            R_0 = eul2rotm([yaw,pitch,-rolls(k-20)]);
            sigma_test(:,:,k) = R_0*sigma_test(:,:,k-20)*R_0';
        end
        true_labels = ones(1,40);
        dataset_name = 'Isotropic Scalings on Linear Ellipsoid';
        
    case 2
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %     Test 2 (Intro): Ellipsoid deformation with anisotropic scalings     %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Creating test centers
        Mu_test      = zeros(3,40);
        Mu_test(1,:) = [0:5:95,-5:-5:-100];
        
        % Linear ellipsoid
        sigma_test = [];
        sigma_test(:,:,1) = diag([1.25 1.25 5]);
        for k=2:20
            ani_scale         = eye(3);            
            ani_scale(2,2)    = 0.5 + sqrt(k/20 * 4.5);
            sigma_test(:,:,k) = ani_scale*sigma_test(:,:,1)*ani_scale';
        end
        
        pitch = 0; yaw= 0;
        rolls = 0.1570:0.0785:pi/2+0.0785;
        for k=21:40
            R_0 = eul2rotm([yaw,pitch,-rolls(k-20)]);
            sigma_test(:,:,k) = R_0*sigma_test(:,:,k-20)*R_0';
        end
        true_labels = ones(1,40);
        dataset_name = 'Anisotropic Scalings on Linear Ellipsoid';
        
    case 3
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %     Test 1 (Distance): Ellipsoid deformation with iso/anisotropic scalings   %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        [Mu_test_1, sigma_test_1] = load_SPD_simulations(1);
        [Mu_test_2, sigma_test_2] = load_SPD_simulations(2);
        Mu_test(:,1:40) = Mu_test_1;
        Mu_test(1,41:80) = Mu_test_2(2,:);
        Mu_test(2,41:80) = Mu_test_2(1,:);
        Mu_test(3,41:80) = Mu_test_2(3,:);
        sigma_test(:,:,1:40)  = sigma_test_1;
        sigma_test(:,:,41:80) = sigma_test_2;
        true_labels = [ones(1,40) 2*ones(1,4) 3*ones(1,3) 4*ones(1,4) 5*ones(1,3) 6*ones(1,6) ...
                                  2*ones(1,4) 3*ones(1,3) 4*ones(1,4) 5*ones(1,3) 6*ones(1,6)] ;
                              
        % Scale isotropically with rando positive number
        isos = 2*abs(randn(80,1)');
        for i=1:80
            sigma_test(:,:,i) = isos(i)*sigma_test(:,:,i);
        end
                              
        true_labels(41) = 1;
        true_labels(61) = 1;
        dataset_name = 'Mix Isotropic/Anisotropic Scalings on Linear Ellipsoid';
        
    case 4
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %     Test 2: Sampled (Rotated) Covariance Matrices   %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Creating test centers
        Mu_test           = zeros(3,120);
        Mu_test(1,:)      = [0:2:78, 0:2:78,0:2:78];
        Mu_test(2,41:80)  = 10*ones(1,40);
        Mu_test(2,81:120) = 20*ones(1,40);
        
        % Sampling from an axis-aligned linear SPD matrix
        sigma_test(:,:,1) = diag([1 1 5]);
        for k=2:20
            sigma_test(:,:,k) = wishrnd(1/df * sigma_test(:,:,1),df);
        end
        
        % Sampling from an rotated linear SPD matrix
        sigma_test(:,:,21) = diag([1 1 5]);
        roll = pi/2; pitch = 0; yaw= 0;
        R_1 = eul2rotm([yaw,pitch,roll]);
        for k=22:40
            sigma_test(:,:,k) = wishrnd(1/df * (R_1*sigma_test(:,:,21)*R_1'),df);
        end                
        
        % Sampling from an axis-aligned linear SPD matrix
        sigma_test(:,:,41) = diag([1 5 5]);
        for k=42:60
            sigma_test(:,:,k) =  wishrnd(1/df * sigma_test(:,:,41),df);
        end
        % Sampling from a rotated planar SPD matrix
        sigma_test(:,:,61) = diag([1 5 5]);
        roll = 0; pitch = -pi/4; yaw= pi/2;
        R_2 = eul2rotm([yaw,pitch,roll]);
        for k=62:80
            sigma_test(:,:,k) = wishrnd(1/df * (R_2*sigma_test(:,:,61)*R_2'),df);
        end
        
        % Sampling from a spherical SPD matrix
        sigma_test(:,:,81) = diag([2.5 2.5 2.5]);
        for k=82:120
            sigma_test(:,:,k) =  wishrnd(1/df * sigma_test(:,:,81),df);
        end
        true_labels = [ones(1,40) 2*ones(1,40) 3*ones(1,40) ] ;
        dataset_name = '3 Sets Sampled from Wishart Distributions';
        
    case 5
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %     Test 2: Sampled (Rotated) Covariance Matrices   %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Creating test centers
        Mu_test           = zeros(3,200);
%         Mu_test(1,:)      = [0:2:78, 0:2:78,0:2:78];
%         Mu_test(2,41:80)  = 10*ones(1,40);
%         Mu_test(2,81:120) = 20*ones(1,40);
        
        % Sampling Set 1
        sigma_test(:,:,1) = diag([1 1 5 5 2 2]);
        Q = eye(6);
        for k=2:50
            sigma_test(:,:,k) = wishrnd(1/df * Q*sigma_test(:,:,1)*Q',df);
            if mod(k,10) == 0
                % [Q R] = qr(randn(6,6));
                Q = orth(rand(6,6));
            end
        end

        % Sampling Set 2
        sigma_test(:,:,51) = diag([1 1 1 5 5 5]);
        Q = eye(6);
        for k=52:100
            sigma_test(:,:,k) =  wishrnd(1/df * Q*sigma_test(:,:,51)*Q',df);
            if mod(k,10) == 0
                % [Q R] = qr(randn(6,6));
                Q = orth(rand(6,6));
            end
        end
        
        % Sampling Set 3
        sigma_test(:,:,101) = diag([0.5 2 2 2 5 5]);
        Q = eye(6);
        for k=102:150
            sigma_test(:,:,k) =  wishrnd(1/df * Q*sigma_test(:,:,101)*Q',df);
            if mod(k,10) == 0
                % [Q R] = qr(randn(6,6));
                Q = orth(rand(6,6));
            end
        end
        
        % Sampling Set 4
        sigma_test(:,:,151) = diag([5 5 5 5 1 1]);
        Q = eye(6);
        for k=152:200
            sigma_test(:,:,k) =  wishrnd(1/df * Q*sigma_test(:,:,151)*Q',df);
            if mod(k,10) == 0
                % [Q R] = qr(randn(6,6));
                Q = orth(rand(6,6));
            end
        end
        
        true_labels = [ones(1,50) 2*ones(1,50) 3*ones(1,50) 4*ones(1,50)] ;
        dataset_name = '4 Sets Sampled from Wishart Distributions';
        
end