function [dPst1,hPst1,uPst1] = genfeature3(D_in,head_in,unit_in,list,add_flag) 

if (nargin == 4)
    add_flag=1;
end
% 1st stage
%list = {'+abs', '-abs', '/abs', '*abs'};
dP=[];
hP=[];
uP=[];
for i=1:length(list)
[dPTemp,hPTemp,uPTemp] = optc3(D_in,head_in,unit_in,char(list(i)));
%list(i)
dP=[dP dPTemp];
hP=[hP hPTemp];
uP=[uP uPTemp];
end

if (add_flag==1)
    dPst1=[D_in dP];
    hPst1=[head_in hP];
    uPst1=[unit_in uP];
else
    dPst1=dP;
    hPst1=hP;
    uPst1=uP;
end
% Take only unique columns
%[Df,unis1]=optcU(dPst1);
%hf=hPst1(unis1);
%uf=uPst1(unis1);
