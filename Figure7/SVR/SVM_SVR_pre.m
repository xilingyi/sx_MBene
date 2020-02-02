%% Support Vector Machine 
% predict new materials

clearvars;close all;clc;

%% step1. load data
addpath('Functions'); 
%%
[~, ~, raw_tr] = xlsread('ML_Figure7.xlsx');
data_tr = cell2mat(raw_tr(2:end,2:end));
input_tr = data_tr(:,2:end); 
input_tr(:,[3 4 5 6 7]) = [];
output_tr = data_tr(:,1)'; 
%%
[~, ~, raw_te] = xlsread('predicted_sample.xlsx');
data_te = cell2mat(raw_te(2:end,2:end));
input_te = data_te(:,2:end); 
input_te(:,[3 4 5 6 7]) = [];
output_te = data_te(:,1)'; 

%% step2. build model
bestc = 2.378414230005442;
bestg = 0.088388347648318;
p_train = input_tr;
t_train = output_tr';
p_test = input_te;
t_test = output_te';

% training set 
[pn_train,inputps] = mapminmax(p_train');
pn_train = pn_train';
pn_test = mapminmax('apply',p_test',inputps);
pn_test = pn_test';
% testing set
[tn_train,outputps] = mapminmax(t_train');
tn_train = tn_train';
tn_test = mapminmax('apply',t_test',outputps);
tn_test = tn_test';

% (2) run SVM
cmd = [' -t 2',' -c ',num2str(bestc),' -g ',num2str(bestg),' -s 4 -p 0.0001'];
model = svmtrain(tn_train,pn_train,cmd);
[Predict_1,error_1] = svmpredict(tn_train,pn_train,model);
[Predict_2,error_2] = svmpredict(tn_test,pn_test,model);

% (3). normalized data
predict_1 = mapminmax('reverse',Predict_1,outputps);
predict_2 = mapminmax('reverse',Predict_2,outputps);

[rmse_train, R2_train]=cod (t_train,predict_1);
[rmse_test, R2_test]=cod (t_test,predict_2);
result = [predict_2,t_test];

% save('SVR_pre.mat');