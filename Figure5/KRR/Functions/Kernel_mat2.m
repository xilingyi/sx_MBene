%% This function computes a Kernel matrix such that the K_{i,j} entry is the kernel function k(X_i,T_j).
% Thus Kernel_mat2(train,train,v) would outout K_{train,train}
% Thus Kernel_mat2(test,test,v) would outout K_{test,test}

% input£º
  % X,T£º training data or test datas
  % sigma: the parameter for the gaussian kernel function.

% output:
  % K: K_{train,train} or K_{test,test}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
function K = Kernel_mat2(X, T, sigma)
Xrow = size(X,1);
Trow = size(T,1);
K = zeros(Xrow, Trow);
for ii = 1:Xrow
    K(ii,:) =  exp(-sum((repmat(X(ii,:),Trow,1)-T).^2,2) ./(2*sigma^2)) ;  % Vectorised
end
end