%% Empty all
clc;
clearvars;
close all;

%% load dataset
[~, ~, raw] = xlsread('../dataset.xlsx');
data = raw(2:end,3:end); data = cell2mat(data);
X = data(:,:);

%% feature Regularization
[m,n] = size(X);
% mean
X_mean=mean(X,1);
% std
X_std=std(X,0,1);
% reg
Xi=zeros(m,n);
for a = 1:size(X,2)
    Xi(:,a) = (X(:,a)-X_mean(:,a))/X_std(:,a);
end

%% data pre-process
Xi=Xi(:,1:9);
[m,n]=size(Xi);
mn=m*n;
Y=Xi(:);
Z=num2cell(Y);
group={'F','D','qM','qD','db','LM/D-M','LM/D-B','n','a'};
for ii = 1:mn
    if rem(ii,m)==0
        Z(ii,2)=group(1,floor(ii/m));
    else
        Z(ii,2)=group(1,(floor(ii/m)+1));
    end
end
Z(:,[1 2],:)=Z(:,[2 1]);
Z=cat(1,[{'Group'},{'value'}],Z);

%% error bar
Z_mean=zeros(n,3);
Z_mean(:,1)=(mean(Xi))';
Z_mean(:,2)=(min(Xi))';
Z_mean(:,3)=(max(Xi))';
Z_vio=cat(2,group',num2cell(Z_mean));
Z_vio=cat(1,[{'Group'},{'value'},{'min'},{'max'}],Z_vio);
save('result_dft.mat');