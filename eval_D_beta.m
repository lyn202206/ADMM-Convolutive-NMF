function div = eval_D_beta(A,B,beta)
% Computes the beta-divergence between matrices A and B see
%
% Févotte, C., & Idier, J. (2011). 
% Algorithms for nonnegative matrix factorization with the ?-divergence. 
% Neural computation, 23(9), 2421-2456.
%
% Input :
% A,B : matrices
% beta : real parameter of the beta-divergence used
% 
% Output : 
% div : beta divergence between A and B
% Author : Dylan Fagot

if beta == 0 
    C = (A+eps)./(B+eps);
    div = sum(C(:)-log(C(:))-1);
elseif beta == 1
    div = sum((A(:)+eps).*log((A(:)+eps)./(B(:)+eps))-(A(:)+eps)+(B(:)+eps));
else
    div = sum(1/(beta*(beta-1))*((A(:)+eps).^beta + (beta-1)*(B(:)+eps).^beta - beta*(A(:)+eps).*(B(:)+eps).^(beta-1)));
end
 
end
