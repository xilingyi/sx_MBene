clearvars;close all;clc;

[~, ~, raw] = xlsread('../ML_Figure6.xlsx');
data = cell2mat(raw(2:end,2:end));
X = data(:,2:end); 
X(:,[3 4 5 6 7]) = [];
Y = data(:,1); 
Num=size(X,2);
Mi_list=zeros(1,Num);
%  sym lxy
for ii = 1:Num
    [Ixy,lambda]=MutualInfo(X(:,ii),Y);
    Mi_list(:,ii)=Ixy;
end
save('result.mat');