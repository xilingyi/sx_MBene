clearvars;close all;clc;

load RFR…Û∏Â“‚º˚.mat
Y=A;
Y_result=zeros(1,size(Y,2));

for ii = 1:size(Y,2)
    Y_diff=Y{1,ii}(:,1)-Y{1,ii}(:,2);
    Y_result(1,ii)= mae(Y_diff);
end
Y_result=max(Y_result);