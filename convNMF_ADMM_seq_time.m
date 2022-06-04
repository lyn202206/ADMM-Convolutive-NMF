function [W,H,cost,time] = convNMF_ADMM_seq_time(V,W,H,V_hat,N_iter_max,beta,rho,flag,time_limit)
% Computes the convolutive NMF using the heuristic to update H
%
% Input :
% V : matrix to be approximated
% W,H : latent factors ( W(t) is W(:,:,t+1) )
% V_hat : approximation of V
% N_iter_max : maximum number of iterations
% beta : parameter of the beta-divergence*
% (only beta=0,1,2 are supported)
% rho: ADMM smoothing parameter
% flag: if to update U and X before the update H
% time_limit: if to limit the maxmimun time for iteration
% Author : Yinan Li
% Date: February 14th, 2021

    tic
    cost = zeros(1,N_iter_max); 
    cost(1) = eval_D_beta(V,V_hat,beta); % initial cost
    [M,K,T] = size(W); [~,N] = size(H);
    time = zeros(1,N_iter_max); time(1) = toc;
    flag_time = 1;

    if nargin<9, flag_time=0; end
    if nargin<8, flag=0; end
    if nargin<7, rho=1; end
    
    
    % initialization for X, Ht, Xplus, Wplus, Hplus
    X = zeros(M,N,T);
    for t=0:T-1
        tW = W(:,:,t+1);
        tH = shift_t(H,t);
        % initial X
        X(:,:,t+1) = tW*tH;
    end
    
    % some variable for test
    Xplus = X;
    U=zeros(M,N,T);
    
    alphaX = zeros(M,N,T);
    alphaH = zeros(K,N);
    alphaW = zeros(M,K,T);
    
    Wplus = W;
    Hplus = H;
    
    time(1) = toc;

    for k=2:N_iter_max
        
        % update H by accumulation
        P1 = zeros(K,K);
        P2 = zeros(K,N);
        P3 = zeros(K,N);
        for t=0:T-1
            tW = W(:,:,t+1);
            tX = shift_t(X(:,:,t+1),-t);
            talphaX = shift_t(alphaX(:,:,t+1),-t);
            P1 = P1 + tW'*tW;
            P2 = P2 + tW'*tX;
            P3 = P3 + tW'*talphaX;
        end
        H = (P1 + eye(K))\(P2 + Hplus + 1/rho*(P3-alphaH));
        
        % update Hplus
        Hplus = max(H + 1/rho*alphaH, 0);
        
        % update alphaH
        alphaH = alphaH + rho*(H - Hplus);
      
        % update parameters sequentially
        for t=0:T-1
            % update Xplus in each time slice
            tW = Wplus(:,:,t+1);
            tH = shift_t(Hplus,t);
            % calculate Xplus
            Xplus(:,:,t+1) = tW*tH;
           
            % splite V in each time slice which result in U
            U(:,:,t+1) = (tW*tH + eps).*V./(sum(Xplus,3) + T*eps);
           
            % update X in each time slice
            if beta ==2
                % update for Euclidean distance
                tW = W(:,:,t+1);
                tH = shift_t(H,t);
                tV = U(:,:,t+1);
                % update X
                X(:,:,t+1) = (rho*tW*tH + tV-alphaX(:,:,t+1))/(1+rho);
            elseif beta==1
                % update for Kullback-Leibler divergence
                tW = W(:,:,t+1);
                tH = shift_t(H,t);
                tV = U(:,:,t+1);
                % update X
                b = rho*tW*tH - alphaX(:,:,t+1) - 1;
                X(:,:,t+1) = (b + sqrt(b.^2 + 4*rho*tV))/(2*rho);                
            elseif beta==0
                % update for Itakura-Saito divergence
                tW = W(:,:,t+1);
                tH = shift_t(H,t);
                tV = U(:,:,t+1);
                % parameters for update
                A = alphaX(:,:,t+1)/rho - tW*tH;
                B = 1/(3*rho) - A.^2/9;
                C = - A.^3/27 + A/(6*rho) + tV/(2*rho);
                D = B.^3 + C.^2;
                % update X
                tX = X(:,:,t+1);
                tX(D>=0) = nthroot(C(D>=0)+sqrt(D(D>=0)),3) + ...
                nthroot(C(D>=0)-sqrt(D(D>=0)),3) - ...
                A(D>=0)/3;

                phi = acos(C(D<0) ./ ((-B(D<0)).^1.5));
                tX(D<0) = 2*sqrt(-B(D<0)).*cos(phi/3) - A(D<0)/3;
                X(:,:,t+1) = tX;
            else       
                error('The beta you specified is not currently supported.')
            end
            % update alphaX
            alphaX(:,:,t+1) = alphaX(:,:,t+1) +  rho*(X(:,:,t+1)-tW*tH);
             
            % update W
            tH = shift_t(H,t);
            tX = X(:,:,t+1);
            P = tH*tH' + eye(K);
            Q = tH*tX' + Wplus(:,:,t+1)' + 1/rho*(tH*alphaX(:,:,t+1)' - alphaW(:,:,t+1)');
            W(:,:,t+1) = (P \ Q)';
           
            % update Wplus
            Wplus(:,:,t+1) = max(W(:,:,t+1) + 1/rho*alphaW(:,:,t+1), 0);
           
            % udpate alphaW
            alphaW(:,:,t+1) = alphaW(:,:,t+1) + rho*(W(:,:,t+1) - Wplus(:,:,t+1));          
        end
        
        %  if to update X before the update of U
        if flag==1            
            %  update Xplus in each time slice
            for t=0:T-1
                tW = Wplus(:,:,t+1);
                tH = shift_t(Hplus,t);
                % initial X
                Xplus(:,:,t+1) = tW*tH;
            end
            % splite V in each time slice which result in U
            for t=0:T-1
                tW = Wplus(:,:,t+1);
                tH = shift_t(Hplus,t);
                U(:,:,t+1) = (tW*tH + eps).*V./(sum(Xplus,3) + T*eps);
            end
            % update X in each time slice after the update of the whole W
            if beta ==2
                % update for Euclidean distance
                for t=0:T-1
                    tW = W(:,:,t+1);
                    tH = shift_t(H,t);
                    tV = U(:,:,t+1);
                    % update X
                    X(:,:,t+1) = (rho*tW*tH + tV-alphaX(:,:,t+1))/(1+rho);
                end
            elseif beta==1
                % update for Kullback-Leibler divergence
                for t=0:T-1                    
                    tW = W(:,:,t+1);
                    tH = shift_t(H,t);
                    tV = U(:,:,t+1);
                    % update X
                    b = rho*tW*tH - alphaX(:,:,t+1) - 1;
                    X(:,:,t+1) = (b + sqrt(b.^2 + 4*rho*tV))/(2*rho);                
                end
            elseif beta==0
                % update for Itakura-Saito divergence
                for t=0:T-1
                    tW = W(:,:,t+1);
                    tH = shift_t(H,t);
                    tV = U(:,:,t+1);
                    % parameters for update
                    A = alphaX(:,:,t+1)/rho - tW*tH;
                    B = 1/(3*rho) - A.^2/9;
                    C = - A.^3/27 + A/(6*rho) + tV/(2*rho);
                    D = B.^3 + C.^2;
                    % update X
                    tX = X(:,:,t+1);
                    tX(D>=0) = nthroot(C(D>=0)+sqrt(D(D>=0)),3) + ...
                    nthroot(C(D>=0)-sqrt(D(D>=0)),3) - ...
                    A(D>=0)/3;

                    phi = acos(C(D<0) ./ ((-B(D<0)).^1.5));
                    tX(D<0) = 2*sqrt(-B(D<0)).*cos(phi/3) - A(D<0)/3;
                    X(:,:,t+1) = tX;
                end
            else       
                error('The beta you specified is not currently supported.')
            end          

           % update for dual variables
            for t=0:T-1
                tW = W(:,:,t+1);
                tH = shift_t(H,t);
                % update alphaX
                alphaX(:,:,t+1) = alphaX(:,:,t+1) +  rho*(X(:,:,t+1)-tW*tH);
            end       
        end
        
        
        V_hat = zeros(M,N);
        for t=0:T-1
           tW = W(:,:,t+1);
           tH = shift_t(H,t);
           V_hat = V_hat + tW*tH;
        end
        V_hat = max(V_hat,0);

        cost(k) = eval_D_beta(V,V_hat,beta);
        time(k) = toc;
        disp(['ADMM, Iteration ',num2str(k),' , cost = ',num2str(cost(k))])
        
        if flag_time==1
            if time(k)>time_limit
                cost = cost(1:k);
                time = time(1:k);
                break;
            end
        end
        
        
    end
    W = Wplus;
    H = Hplus;
    [W,H] = renormalize_convNMF(W,H);    
end


