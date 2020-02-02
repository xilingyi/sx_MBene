%% The function is to calculate the Mean Squared Error (MSE) using equation (15)

% input:
  % K : a kernel matrix
  % y : a vector
  % alpha: the dual weights
  
% output:
  % mse: a single value
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
function  mse  = dualcost(K, y, alpha)
l = size(y,1);
mse = (K*alpha - y)'*(K*alpha - y);
mse = mse/l; % calculate the Mean Squared Error (MSE)
end

