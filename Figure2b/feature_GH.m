%% various descriptors vs. ¦¤GH  through simple linear regression

clearvars;close all;clc;
load feature_GH.mat
total([2,4,8,12,17,32,34,35,37,80,85,91,110],:) = [];
rmse=zeros(size(total,2)-1,1);

for ii = 1:size(total,2)-1
    X=total(:,ii);
    Y=total(:,end);
    p=polyfit(X,Y,1);
    Y_pre=X*p(1,1)-p(1,2);
    rmse(ii,1) = sqrt(sum((Y-Y_pre).^2)/max(size(Y)));
end
rmses=[(1:1:size(total,2)-1)' rmse];
rmses=sortrows(rmses,2);
save('result.mat');