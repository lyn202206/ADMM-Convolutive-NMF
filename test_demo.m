% the code to test the performance of ADMM CNMF algorithms
% multiplicative updates are used as baseline algorithms

clc
close all
clear all

% Parameters for random date generation
M = 300; % the dimension of date
K = 50; % the number of basis
N = 1000; % the number of samples
T = 5;   % the time dimension

% some parameters needed
N_iter_max = 100;
time_limit = 100;
% parameter settings
% beta = 0,1,2 corresponding to ISD, KLD and ELD, respectively
beta = 2;          
% several step sizes for ADMM method
rho = 1;
rho1 = 2;
rho2 = 5;

% generate the random data for testing
V = zeros(M,N);
W_groundtruth = abs(randn(M,K,T));
H_groundtruth = abs(randn(K,N));

% generate data to approximate
for t=0:T-1
    tW = W_groundtruth(:,:,t+1);
    tH = shift_t(H_groundtruth,t);
    V = V + tW * tH;
end

% the initialization for all the tested algorithms
W = abs(randn(M,K,T));
H = abs(randn(K,N));

% Compute V_hat
V_hat = zeros(M,N);
for t=0:T-1
    V_hat = V_hat + W(:,:,t+1)*shift_t(H,t);
end

% Convolutive NMF algorithms based on multiplicative updates

[W_heuristic,H_heuristic,cost_heuristic,time_heuristic] = ...
    convNMF_heuristic(V,W,H,V_hat,N_iter_max,beta);

[W_MM1,H_MM1,cost_MM1,time_MM1] = ...
    convNMF_MM1(V,W,H,V_hat,N_iter_max,beta);

[W_MM2,H_MM2,cost_MM2,time_MM2] = ...
    convNMF_MM2(V,W,H,V_hat,N_iter_max,beta);

% Convolutive NMF algorithms based on ADMM CNMF Algorithms 1
[W_ADMM,H_ADMM,cost_ADMM,time_ADMM] = convNMF_ADMM_Y(V,W,H,V_hat,N_iter_max,beta,rho);
[W_ADMM1,H_ADMM1,cost_ADMM1,time_ADMM1] = convNMF_ADMM_Y(V,W,H,V_hat,N_iter_max,beta,rho1);
[W_ADMM2,H_ADMM2,cost_ADMM2,time_ADMM2] = convNMF_ADMM_Y(V,W,H,V_hat,N_iter_max,beta,rho2);

% Convolutive NMF algorithms based on ADMM CNMF Algorithms 2
[W_ADMMs,H_ADMMs,cost_ADMMs,time_ADMMs] = convNMF_ADMM_seq(V,W,H,V_hat,N_iter_max,beta,rho);
[W_ADMM1s,H_ADMM1s,cost_ADMM1s,time_ADMM1s] = convNMF_ADMM_seq(V,W,H,V_hat,N_iter_max,beta,rho1);
[W_ADMM2s,H_ADMM2s,cost_ADMM2s,time_ADMM2s] = convNMF_ADMM_seq(V,W,H,V_hat,N_iter_max,beta,rho2);


% setting the parameters for algorithms with respect to the running time
N_iter_max = 1e6;   % reset the max number of iteration to ensure the number of iteration reach its maximum
% several step sizes for ADMM method
rho3 = 5;
rho4 = 10;
rho5 = 25;

% Convolutive NMF algorithms based on multiplicative updates
[~,~,cost_heuristic_t,time_heuristic_t] = ...
    convNMF_heuristic_time(V,W,H,V_hat,N_iter_max,beta,time_limit);

[~,~,cost_MM1_t,time_MM1_t] = ...
    convNMF_MM1_time(V,W,H,V_hat,N_iter_max,beta,time_limit);

[~,~,cost_MM2_t,time_MM2_t] = ...
    convNMF_MM2_time(V,W,H,V_hat,N_iter_max,beta,time_limit);


% Convolutive NMF algorithms based on ADMM CNMF Algorithms 1
[~,~,cost_ADMM_t,time_ADMM_t] = convNMF_ADMM_Y_time(V,W,H,V_hat,N_iter_max,beta,rho3,time_limit);
[~,~,cost_ADMM1_t,time_ADMM1_t] = convNMF_ADMM_Y_time(V,W,H,V_hat,N_iter_max,beta,rho4,time_limit);
[~,~,cost_ADMM2_t,time_ADMM2_t] = convNMF_ADMM_Y_time(V,W,H,V_hat,N_iter_max,beta,rho5,time_limit);

% Convolutive NMF algorithms based on ADMM CNMF Algorithms 2
[~,~,cost_ADMM_ts,time_ADMM_ts] = convNMF_ADMM_seq_time(V,W,H,V_hat,N_iter_max,beta,rho3,1,time_limit);
[~,~,cost_ADMM1_ts,time_ADMM1_ts] = convNMF_ADMM_seq_time(V,W,H,V_hat,N_iter_max,beta,rho4,1,time_limit);
[~,~,cost_ADMM2_ts,time_ADMM2_ts] = convNMF_ADMM_seq_time(V,W,H,V_hat,N_iter_max,beta,rho5,1,time_limit);


figure

subplot(121)
semilogy(cost_heuristic,'-b','LineWidth',2)
hold on
semilogy(cost_MM1,'-.r','LineWidth',2)
semilogy(cost_MM2,':g','LineWidth',2)
semilogy(cost_ADMM,'m','LineWidth',2)
semilogy(cost_ADMM1,'-.m','LineWidth',2)
semilogy(cost_ADMM2,':m','LineWidth',2)
semilogy(cost_ADMMs,'k','LineWidth',2)
semilogy(cost_ADMM1s,'-.k','LineWidth',2)
semilogy(cost_ADMM2s,':k','LineWidth',2)
% grid on
xlabel('Iteration #','Fontname','Times New Roman')
ylabel('Value of objective function','Fontname','Times New Roman')
legend('Heuristic','MM1','MM2',['ADMM \rho = ',num2str(rho)],['ADMM \rho = ',num2str(rho1)],['ADMM \rho = ',num2str(rho2)],['ADMM\_{\it seq} \rho = ',num2str(rho)],['ADMM\_{\it seq} \rho = ',num2str(rho1)],['ADMM\_{\it seq} \rho = ',num2str(rho2)])
title(['\beta = ', num2str(beta), ',    T = ', num2str(T)])
set(gca,'Fontname','Times New Roman');


subplot(122)
semilogy(time_heuristic_t,cost_heuristic_t,'-b','LineWidth',2)
hold on
semilogy(time_MM1_t,cost_MM1_t,'-.r','LineWidth',2)
semilogy(time_MM2_t,cost_MM2_t,':g','LineWidth',2)
semilogy(time_ADMM_t,cost_ADMM_t,'m','LineWidth',2)
semilogy(time_ADMM1_t,cost_ADMM1_t,'-.m','LineWidth',2)
semilogy(time_ADMM2_t,cost_ADMM2_t,':m','LineWidth',2)
semilogy(time_ADMM_ts,cost_ADMM_ts,'k','LineWidth',2)
semilogy(time_ADMM1_ts,cost_ADMM1_ts,'-.k','LineWidth',2)
semilogy(time_ADMM2_ts,cost_ADMM2_ts,':k','LineWidth',2)
% grid on
xlabel('Time (s)','Fontname','Times New Roman')
ylabel('Value of objective function','Fontname','Times New Roman')
legend('Heuristic','MM1','MM2',['ADMM \rho = ',num2str(rho3)],['ADMM \rho = ',num2str(rho4)],['ADMM \rho = ',num2str(rho5)],['ADMM\_{\it seq} \rho = ',num2str(rho3)],['ADMM\_{\it seq} \rho = ',num2str(rho4)],['ADMM\_{\it seq} \rho = ',num2str(rho5)])
title(['\beta = ', num2str(beta), ',    T = ', num2str(T)])
set(gca,'Fontname','Times New Roman');
set(gca,'xlim',[0 time_limit])
% axis([0 time_limit -Inf Inf])













