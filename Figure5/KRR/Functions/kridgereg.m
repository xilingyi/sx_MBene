%% The function is to perform kernel ridge regression using equation (12)

% input:
  % K : a kernel matrix
  % y : a vector
  % gamma: a scalar ridge parameter
  
% output:
  % alpha: the dual weights
  
% explaination:
  % x = A\B solves the system of linear equations A*x = B
  % x = mldivide(A,B) is an alternative way to execute x = A\B
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
function [alpha] = kridgereg(K, y, gamma)
l = size(y,1); % the row numbers of y
alpha = (K + gamma*l*eye(l))\y; % the original eqn.
%alpha = mldivide((K + gamma*l*eye(l)),y); % revised by ZW
end

