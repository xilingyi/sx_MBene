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
function score = kFoldScore(X_tr, y_tr, c, g, k)
set_size = floor(size(X_tr,1)/k); % size of single fold
ids = cell(k,1); % 
for ii = 1:k % prepare set indeces
    ids{ii} = set_size*(ii - 1) + 1:set_size*ii;
end
ids{k} = [ids{k}  set_size*k + 1:size(X_tr, 1)];
scores = zeros(k,1);
for ii = 1:k
    ev_y = y_tr(ids{ii},:); % 测试集的Y
    tr_y = vertcat(y_tr([ids{[1:ii-1 ii + 1:k]}],:)); % 训练集的Y
    ev_x = X_tr(ids{ii},:); % 测试集的X
    tr_x = vertcat(X_tr([ids{[1:ii-1 ii + 1:k]}],:)); % 训练集的X
    
    % 训练集
    [pn_train,inputps] = mapminmax(tr_x');
    pn_train = pn_train';
    pn_test = mapminmax('apply',ev_x',inputps);
    pn_test = pn_test';
    %测试集
    [tn_train,outputps] = mapminmax(tr_y');
    tn_train = tn_train';
    tn_test = mapminmax('apply',ev_y',outputps);
    tn_test = tn_test';
    
    cmd = [' -t 2',' -c ',num2str(c),' -g ',num2str(g),' -s 4 -p 0.001'];
    model = svmtrain(tn_train,pn_train,cmd);
% [Predict_1,error_1] = svmpredict(tn_train,pn_train,model);
    [Predict_2,~] = svmpredict(tn_test,pn_test,model);
% predict_1 = mapminmax('reverse',Predict_1,outputps);
    predict_2 = mapminmax('reverse',Predict_2,outputps);
    scores(ii) = ((predict_2-ev_y)'*(predict_2-ev_y))/size(predict_2,1);
end
score = mean(scores);
end