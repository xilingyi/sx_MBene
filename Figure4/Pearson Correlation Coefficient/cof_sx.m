clearvars;close all;clc;

%% produce Pearson Correlation Coefficient
[~, ~, raw] = xlsread('../dataset.xlsx');
data = raw(2:end,3:end); data = cell2mat(data);
Cof = corrcoef(data);
Cof_rlt = [raw(1,3:end)' num2cell(Cof)];
Cof_rlt = [raw(1,[1 3:end]);Cof_rlt];
save('result.mat');
