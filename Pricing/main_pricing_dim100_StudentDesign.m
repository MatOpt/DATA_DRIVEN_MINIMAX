%% Optimal Pricing for Profit Maximization

dbstop if error

rng('default');

d = 100; 
m = 5000; % 5000 constraints
nu = 4;

%s_0 = 3*rand(d,1);   % d x 1
%s_0 = normrnd(2,0.5,[d,1]);
s_0 = 3+ trnd(nu,[d,1]);
s_0 = abs(s_0);

%S = 3*rand(m,d);     % m x d
%S = normrnd(2,0.5,[m,d]);

S = 2+ trnd(nu,[m,d]);

%p = 10+10*rand(m,1);

p = normrnd(15,1,[m,1]);

c_l = -5;
c_u = -2;


%l_theta = normrnd(2,0.5,[d,1]);
l_theta = 2*rand(d,1);
u_theta = l_theta + 3*rand(d,1);

l_c = -5;
u_c = -2;

Cov = genCovMatrix(d,1,1);

tilde_theta = (u_theta+ l_theta)/2;

tilde_c = -3;


Demand = S*tilde_theta + p*tilde_c;
sigma = 1;


%% Initialize

theta_0 = 3*rand(d,1);
c_0 = -3+ rand();
p_0 = 10 + 10*rand();



N_iter = 1;

    
run_adaptive = 1;
%run_fixed = 1;

K = 5e4; %5000000;

tol = 0.00001;
[p_opt,theta_opt,c_opt] = sequential_LP(s_0,S,p,Demand,l_theta,u_theta,l_c,u_c,theta_0,c_0,28.4597,tol );

%% run adaptive algorithm
run_adaptive = 1;
if run_adaptive == 1

    infeas_adaptive = [];
    index_adaptive = [];
%         error_Interior_adaptive = [];
%         obj_gap_Interior_adaptive = [];
    for iter = 1:N_iter
        fprintf("Run %d / %d adpative algorithm.\n", iter,N_iter)
        [index_list,infeas_list,obj_gap_list] = adaptive_pricing_new(K,s_0,S,p,Demand,l_theta,u_theta,l_c,u_c,sigma,theta_0,c_0,p_0,theta_opt,c_opt,p_opt);

        if iter == 1
            infeas_adaptive = infeas_list;
            obj_gap_adaptive = obj_gap_list;
            %error_Interior_adaptive = error_list;
            %obj_gap_Interior_adaptive = obj_gap_list;
        else   
            infeas_adaptive = infeas_adaptive + infeas_list;      
            obj_gap_adaptive = obj_gap_adaptive + obj_gap_list;
            %error_Interior_adaptive = error_Interior_adaptive + error_list;
            %obj_gap_Interior_adaptive = obj_gap_Interior_adaptive + obj_gap_list;     
        end

    end

    infeas_adaptive = infeas_adaptive/N_iter;          
    obj_gap_adaptive = obj_gap_adaptive/N_iter;
    %error_Interior_adaptive = error_Interior_adaptive/N_iter;
    %obj_gap_Interior_adaptive = obj_gap_Interior_adaptive/N_iter;  
end
%% run fixed_stepsize algorithm

run_fixed = 1;

if run_fixed == 1

    infeas_fixed = [];
    index_fixed = [];
    
%     error_Interior_fixed = [];
%     obj_gap_Interior_fixed = [];
    for iter = 1:N_iter
        fprintf("Run %d / %d fixed algorithm.\n", iter,N_iter)
        [index_list_fixed,infeas_list,obj_gap_list] = fixed_pricing(K,s_0,S,p,Demand,l_theta,u_theta,l_c,u_c,sigma,theta_0,c_0,p_0,theta_opt,c_opt,p_opt);

        if iter == 1
            infeas_fixed = infeas_list;
            %error_Interior_fixed = error_list;
            obj_gap_fixed = obj_gap_list;
        else   

            infeas_fixed = infeas_fixed + infeas_list;               
            %error_Interior_fixed = error_Interior_fixed + error_list;
            obj_gap_fixed = obj_gap_fixed + obj_gap_list;     
        end

    end

    infeas_fixed = infeas_fixed/N_iter;               
    %error_Interior_fixed = error_Interior_fixed/N_iter;
    obj_gap_fixed = obj_gap_fixed/N_iter;  


end
%%

log_x_slope = 9:0.2:15;
log_y = 3 - (log_x_slope - 11)/2;


x = index_list;

log_x = log(x);

log_infeas_adaptive = log(abs(infeas_adaptive));
log_infeas_fixed = log(abs(infeas_fixed));

n_data = length(x);

y_last = log_infeas_adaptive(n_data);

y = y_last + 1.3 + (log_x(n_data) - log_x)*0.5;

log_x_fixed = log(index_list_fixed);


figure
set(gca,'FontSize',30);
plot(log_x,log_infeas_adaptive,'-',log_x_fixed,log_infeas_fixed,':',log_x_slope, log_y,'-.x','lineWidth', 4 );
legend('    Adp-CSPD ', '    Basic-CSPD', '$\log y = -\frac{1}{2}\log N + C$', 'fontSize', 20,'Interpreter','latex');

xlabel('$\log(N)$','fontsize',32, 'Interpreter','latex' );
ylabel('$log(|| H(\bar x_{N})_+||_2)$','fontsize',32,'FontName','Times New Roman','Interpreter','latex');
title("Empirical Log-Feasibility Residual", 'FontSize', 32)


figure
set(gca,'FontSize',30);
plot(x,infeas_adaptive,'-',index_list_fixed,infeas_fixed,':','lineWidth', 4 );
legend('    Adp-CSPD ', '    Basic-CSPD', 'fontSize', 20,'Interpreter','latex');

xlabel('$N$','fontsize',32,'Interpreter','latex');
ylabel('$|| H(\bar x_{k})_+||_2$','fontsize',32,'FontName','Times New Roman','Interpreter','latex');
title("Empirical Feasibility Residual", 'FontSize', 32)

%%

log_obj_adaptive = log(abs(obj_gap_adaptive));
log_obj_fixed = log(abs(obj_gap_fixed));


obj_last = log_obj_adaptive(n_data);

log_obj_y = 3 - (log_x_slope - 15)/2;

figure
set(gca,'FontSize',30);
plot(x,obj_gap_adaptive,'-',index_list_fixed,obj_gap_fixed,':','lineWidth', 4 );
legend('    Adp-CSPD ', '    Basic-CSPD', 'fontSize', 20, 'Interpreter','latex' );

xlabel('$N$','fontsize',32,'Interpreter','latex');
ylabel('$F(\bar x_N,y^*) - F(x^*,\bar x_N)$','fontsize',32,'FontName','Times New Roman','Interpreter','latex');
title("Empirical Objective Gap", 'FontSize', 32)


obj_last = log_obj_adaptive(n_data);

log_obj_y = 4 - (log_x_slope - 15)/2;

figure
set(gca,'FontSize',30);
plot(log_x,log_obj_adaptive,'-',log_x_fixed,log_obj_fixed,':',log_x_slope, log_obj_y,'-.x','lineWidth', 4 );
legend('    Adp-CSPD ', '    Basic-CSPD', '$\log y = -\frac{1}{2}\log k + C$', 'fontSize', 20,'Interpreter','latex');

xlabel('$\log(N)$','fontsize',32,'Interpreter','latex');
ylabel('$\log(|F(\bar x_N,y^*) - F(x^*,\bar x_N)|)$','fontsize',32,'FontName','Times New Roman','Interpreter','latex');
title("Empirical Log-Objective Gap", 'FontSize', 32)
