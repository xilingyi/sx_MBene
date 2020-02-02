%% This function is to perform KRR on the training set using K-fold cross-validation

% input:
  % X_tr : independent variable in training data
  % y_tr : dependent variable in training data
  % gamma: a vector
  % sigma: a vector
  % k: number of folding cross-validation
  
% output:
  % score: a matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
function score = kFoldScore(X_tr, y_tr, gamma, sigma, k)
set_size = floor(size(X_tr,1)/k); % size of single fold
ids = cell(k,1); % 


for ii = 1:k % prepare set indeces
    ids{ii} = set_size*(ii - 1) + 1:set_size*ii;
end

ids{k} = [ids{k}  set_size*k + 1:size(X_tr, 1)];
scores = zeros(k,1);
K_tr = Kernel_mat2(X_tr, X_tr, sigma);% compute K for all folds
for ii = 1:k
    ev_y = y_tr(ids{ii},:);
    tr_y = vertcat(y_tr([ids{[1:ii-1 ii + 1:k]}],:));
    
    K = K_tr;% Create temporary K for k-1 training folds
    K(ids{ii},:) = [];
    K(:,ids{ii}) = [];
    
    alpha = kridgereg(K, tr_y, gamma);
    K = K_tr(ids{ii},:);% Create temporary K for the testing
    K(:,ids{ii}) = [];
    scores(ii)  = dualcost(K, ev_y, alpha);
end
score = mean(scores);
end