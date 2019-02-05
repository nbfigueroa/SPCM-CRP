function [x_emb, y_emb, pca_params, emb_name] = pcaTangentSpaceEmbedding(sigmas, emb_options)

expl_var   = emb_options.expl_var;   
show_plots = emb_options.show_plots; 
emb_type   = emb_options.emb_type;

pca_params = [];
switch emb_type
    
    % Tangent-space Mapping with Log-Euclidean Framework
    case 0        
        N     = length(sigmas{1});
        d_vec = N*(N+1)/2;
        vec_sigmas = zeros(d_vec,length(sigmas));
        for s=1:length(sigmas)
            % Log matrix of Sigma
            sigma     = sigmas{s};
            log_sigma = logm(sigma);
            
            % Projecting to the L-2 Norm
            vec_sigmas(:,s) = symMat2Vec(log_sigma);
        end
        
        % Tangent Space Vector Embedding w/o Dim. Red.
        x_emb = vec_sigmas;

        % Project to a low-d vector space with PCA
        [ V, L, Mu_X ] = my_pca( vec_sigmas );
        if show_plots
            figure('Color',[1 1 1]);
            plot(diag(L),'-*r'); grid on;
            xlabel('Eigenvalue index')
            title('Eigenvalues of PCA on log-vector space','Interpreter','LaTex')
        end
        [ p ] = explained_variance( L, expl_var);
        [A_p, y_emb] = project_pca(vec_sigmas, Mu_X, V, p);                
        fprintf('Vector-space dim (%d) - lower dimension (%d)\n',d_vec,p);
        emb_name  =  'PCA on log-Eucl. Tangent Space Embedding';
        
    % Tangent-space Mapping with Affine-Invariant Framework (PGA)
    % Principal Geodesic Analysis on Riemannian Manifold %
    case 1
   
        N     = length(sigmas{1});
        d_log = N*(N+1)/2;
        
        % Compute Instrinsic Mean of Set of SPD matrices
        sigma_bar    = intrinsicMean_mat(sigmas, 1e-8, 100);
        
        % Calculate the tangent vectors about the mean
        vec_sigmas = zeros(d_log,length(sigmas));
        for s=1:length(sigmas)
            sigma     = sigmas{s};
            log_sigma = logmap(sigma, sigma_bar);
            vec_sigmas(:,s) = symMat2Vec(log_sigma);
        end
        x_emb = vec_sigmas;
        % Construct Covariance matrix of tangent vectors
        Cov_x = (1/(N-1))*x_emb*x_emb';
        
        % Perform Eigenanalysis of Covariance Matrix
        [V, L] = eig(Cov_x);
        
        % Sort Eigenvalue and get indices
        [L_sort, ind] = sort(diag(L),'descend');
        
        % arrange the columns in this order
        V = V(:,ind);
        
        % Vectorize sorted eigenvalues
        L = diag(L_sort);
        
        % X_bar represented on the Tangent Manifold
        Mu_X = symMat2Vec(logm(sigma_bar));        
        if show_plots
            figure('Color',[1 1 1]);
            plot(diag(L),'-*r'); grid on;
            xlabel('Eigenvalue index')
            title('Eigenvalues of PGA of Riemannian Manifold','Interpreter','LaTex')
        end
        [ p ]    = explained_variance( L, expl_var );
        [A_p, y_emb] = project_pca(vec_sigmas, Mu_X, V, p);
        fprintf('Vector-space dim (%d) - lower dimension (%d)\n',d_log,p);        
        emb_name  =  'PGA on Riemannian Manifold';
end

pca_params.Mu_X = Mu_X;
pca_params.A_p = A_p;