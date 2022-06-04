function [W,H,cost,time] = convNMF_MM1(V,W,H,V_hat,N_iter_max,beta)
% Computes the convolutive NMF using the first MM approach MM1
%
% Input :
% V : matrix to be approximated
% W,H : latent factors ( W(t) is W(:,:,t+1) )
% V_hat : approximation of V
% N_iter_max : maximum number of iterations
% beta : parameter of the beta-divergence
% Author : Dylan Fagot

tic
cost = zeros(1,N_iter_max); cost(1) = eval_D_beta(V,V_hat,beta);
[~,K,T] = size(W); [~,N] = size(V);
time = zeros(1,N_iter_max); time(1) = toc;

for k=2:N_iter_max
    %% Sequential update of H
    for n=1:N
        
        num = zeros(K,1); denom = num;
        if n<=N-T+1
            for n_prime = n:n+T-1 
                num = num + W(:,:,n_prime-n+1)'*((V(:,n_prime)+eps).*(V_hat(:,n_prime)+eps).^(beta-2));
                denom = denom + W(:,:,n_prime-n+1)'*(V_hat(:,n_prime)+eps).^(beta-1);
            end
        else
            for n_prime = n:N
                num = num + W(:,:,n_prime-n+1)'*((V(:,n_prime)+eps).*(V_hat(:,n_prime)+eps).^(beta-2));
                denom = denom + W(:,:,n_prime-n+1)'*(V_hat(:,n_prime)+eps).^(beta-1);
            end
        end
        
        h_n_old = H(:,n);
        H(:,n) = H(:,n).*(num./denom).^gamma_beta(beta);
        
        if n<=N-T+1
            for n_prime = n:n+T-1
                V_hat(:,n_prime) = max(V_hat(:,n_prime) + W(:,:,n_prime-n+1)*(H(:,n)-h_n_old),0);
                % max(.,0) ensures the nonnegativity
            end
        else
            for n_prime = n:N
                V_hat(:,n_prime) = max(V_hat(:,n_prime) + W(:,:,n_prime-n+1)*(H(:,n)-h_n_old),0);
            end
        end
    
    end
    
    
    %% Update of W
    
    for t=0:T-1
        W_t_old = W(:,:,t+1);
        H_shift_t = shift_t(H,t);
        W(:,:,t+1) = W(:,:,t+1).*(((((V+eps).*(V_hat+eps).^(beta-2))*H_shift_t')./((V_hat+eps).^(beta-1)*H_shift_t'))).^gamma_beta(beta);
        V_hat = max(V_hat + (W(:,:,t+1)-W_t_old)*H_shift_t,0); 
        % max(.,0) ensures nonnegativity
    end
    
    [W,H] = renormalize_convNMF(W,H);
    
    % recalate the V_hat
    X_hat = zeros(size(V));
    for t=0:T-1
        tW = W(:,:,t+1);
        tH = shift_t(H,t);
        X_hat = X_hat + tW*tH;
    end
    V_hat = max(X_hat,0);
    
    
    time(k) = toc;
    cost(k) = eval_D_beta(V,V_hat,beta);
    disp(['MM1, Iteration ',num2str(k),' , cost = ',num2str(cost(k))])

end
end
