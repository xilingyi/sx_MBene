%% k-fold-cross-validation routine
% This function takes a training set and a pair of parameters. Then performs
% k-fold cross validation and outputs the mean score for this set of parameters.

 % input:
   % gamma: a vector
   % sigma: a vector
   % fold5Score: results of five-fold cross-validation
   
 % output:
   % a figure named 5-fold-cross-validation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %%
function plot_cross_validation_error(gamma, sigma, fold5Score, mm, nn)
[g, s] = meshgrid(gamma, sigma);
Sc = reshape(fold5Score, mm, nn);
surf(log(g), log(s), log(Sc));
box on
xlabel('log(\gamma)');
ylabel('log(\sigma)');
zlabel('log(Score)');
title('5-fold-cross-validation');
colorbar;
end
