%% This function randomly splits a given sample set and ground truth y,
% to 2:1 ratio for training and testing sets.

% input£º
  % X£ºindependent variable
  % y£ºdependent variable
  
% output:
  % X_train: independent variable in the training data
  % y_train: dependent variable in the training data
  % X_test: independent variable in the test data
  % y_test: dependent variable in the test data

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
function [X_train, y_train, X_test,y_test] = splitData(X, y)
%% Shuffle data
ids = randperm(size(X,1));
Xi = X(ids,:);yi = y(ids);
%% 4:1 ratio split hardcoded
train_frac = 4/5;
% test_frac = 1/4;
%% divide data
X_train = Xi(1:floor(size(X,1)*train_frac),:);
y_train = yi(1:floor(size(X,1)*train_frac));
X_test = Xi(floor(size(X,1)*train_frac) + 1:end,:);
y_test = yi(floor(size(X,1)*train_frac) + 1:end);
end