%% 平均决定系数R2
function [rmse,R2] = cod(outputR2, ytestR2)
 rmse = sqrt(sum((outputR2-ytestR2).^2)/max(size(outputR2)));
 R2 = 1-norm(outputR2-ytestR2).^2/norm(outputR2-mean(outputR2)).^2;
end