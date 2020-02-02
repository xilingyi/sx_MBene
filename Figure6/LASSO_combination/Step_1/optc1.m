function [D_f1,head_f1,u_f1] = optc1(D_p,header,unit_in,opt_flag)
    %opt_flag
    switch opt_flag
        case '^r'
            D_f1=abs(D_p).^(0.5);
            head_f1=strcat('(',header,')',opt_flag);
            u_f1=unit_in-0.5;
        case '^I'
            D_f1=D_p.^(-1);
            head_f1=strcat('(',header,')',opt_flag);
            u_f1=unit_in-1.0;
        case '^2'
            D_f1=D_p.^2;
            head_f1=strcat('(',header,')',opt_flag);
            u_f1=unit_in+2;
        case '^3'
            D_f1=D_p.^3;
            head_f1=strcat('(',header,')',opt_flag);
            u_f1=unit_in+3;
        case 'log'
            D_f1=log(abs(D_p));
            head_f1=strcat('(',header,')',opt_flag);
            u_f1=log(unit_in);
        case 'exp'
            D_f1=exp(D_p);
            head_f1=strcat('(',header,')',opt_flag);
            u_f1=exp(unit_in);
        case 'abs'
            D_f1=abs(D_p);
            head_f1=strcat('(',header,')',opt_flag);
            u_f1=unit_in;            
        otherwise
            disp('unknown operation')
    end
end