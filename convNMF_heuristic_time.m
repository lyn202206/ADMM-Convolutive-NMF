function [W,H,cost,time] = convNMF_heuristic_time(V,W,H,V_hat,N_iter_max,beta,time_limit)
% Computes the convolutive NMF using the heuristic to update H
%
% Input :
% V : matrix to be approximated
% W,H : latent factors ( W(t) is W(:,:,t+1) )
% V_hat : approximation of V
% N_iter_max : maximum number of iterations
% beta : parameter of the beta-divergence*
% Author : Dylan Fagot

tic
cost = zeros(1,N_iter_max); cost(1) = eval_D_beta(V,V_hat,beta);
[M,K,T] = size(W); [~,N] = size(H);
time = zeros(1,N_iter_max); time(1) = toc;

flag_time = 1;

if nargin<7, flag_time=0; end


for k=2:N_iter_max   
    %% Heuristic update of H
    u_H = zeros(K,N,T);
    for t=0:T-1
        u_H(:,:,t+1) = H.*((W(:,:,t+1)'*shift_t((V+eps).*(V_hat+eps).^(beta-2),-t))./(W(:,:,t+1)'*(shift_t(V_hat+eps,-t).^(beta-1)))).^gamma_beta(beta);
    end
    H = mean(u_H,3);
    
    V_hat = zeros(M,N);
    for t=0:T-1
        V_hat = V_hat + W(:,:,t+1)*shift_t(H,t);
    end
    
    %% Update of W
    
    for t=0:T-1
        W_t_old = W(:,:,t+1);
        H_shift_t = shift_t(H,t);
        W(:,:,t+1) = W(:,:,t+1).*(((((V+eps).*(V_hat+eps).^(beta-2))*H_shift_t')./((V_hat+eps).^(beta-1)*H_shift_t'))).^gamma_beta(beta);
        V_hat = max(V_hat + (W(:,:,t+1) - W_t_old)*H_shift_t,0); 
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
    
    cost(k) = eval_D_beta(V,V_hat,beta);
    time(k) = toc;
    disp(['Heuristic, Iteration ',num2str(k),' , cost = ',num2str(cost(k))])
    
    if flag_time==1
         if time(k)>time_limit
             cost = cost(1:k);
             time = time(1:k);
             break;
         end
     end
    
end

end



