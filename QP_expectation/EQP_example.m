rng('default');

d = 50; N = 1000; 
m = 15; %15 constraints

M = randn(d); % n x n 

Q = M' * M + eye(d);


x_center = normrnd(0,0.3,[d,1]);
s = randn(d, m);%x0
w = randn(d, m);


sigma = 0.5;

epsilon = normrnd(0,sigma,[m,N]);

gamma = zeros(m, 1);

mu_0 = 3*randn(d,1);

c = 0.1;

% for i = 1:m
%     Max_gamma = norm(epsilon(i, :) - s(:, i)' * w(:, i))^2 / N;
%     Min_gamma = norm(epsilon(i, :) - sum(epsilon(i, :))/N)^2 / N;
%     gamma(i) = (Max_gamma - Min_gamma) * rand + Min_gamma ;
% end

mu_omega = 0.3*rand(d,1);

cvx_begin
    variable xs(d)
    minimize( (xs - x_center)' * Q * (xs - x_center) + c*norm(xs) + xs'*mu_0 + xs'*mu_omega);
cvx_end

for i= 1:m
    gamma(i) = (epsilon(i, :) + (xs - s(:, i))'* w(:, i))*(epsilon(i, :) + (xs - s(:, i))'* w(:, i))' / N ;
end

display('The constraint values are');
display(gamma);

cvx_x = xs;

Cov = genCovMatrix(d,1,1);

%% solve the case where optimal solution falls in the interior

solve_interior = 1;

N_iter = 5;
if solve_interior == 1
    display('solve the case where optimal solution falls in the interior');
    
    y_opt = mu_0 + c*xs/norm(xs);
    gamma_interior = 1.2*gamma;

    %for K = [800000,820000,850000,880000,900000,920000,950000,980000,1000000]
    x_0 = 10*cvx_x;

    lambda_0 = rand(m, 1);
    
    run_adaptive = 1;
    run_fixed = 1;
    
    K = 1e5;
    %% run adaptive algorithm
    if run_adaptive == 1
        
        infeas_Interior_adaptive = [];
        index_Interior_adaptive = [];
        error_Interior_adaptive = [];
        obj_gap_Interior_adaptive = [];
        for iter = 1:N_iter
            fprintf("Run %d / %d adpative algorithm for interior problem.\n", iter,N_iter)
            [index_list,infeas_list,obj_gap_list,error_list] = adaptive_QP(x_0,lambda_0,mu_0,K,cvx_x,y_opt,Q,x_center,gamma_interior,s,w,epsilon,c,mu_omega,Cov,N,m);
            
            if iter == 1
                infeas_Interior_adaptive = infeas_list;
                error_Interior_adaptive = error_list;
                obj_gap_Interior_adaptive = obj_gap_list;
            else   
                
                infeas_Interior_adaptive = infeas_Interior_adaptive + infeas_list;               
                error_Interior_adaptive = error_Interior_adaptive + error_list;
                obj_gap_Interior_adaptive = obj_gap_Interior_adaptive + obj_gap_list;     
            end
                
        end
        
        infeas_Interior_adaptive = infeas_Interior_adaptive/N_iter;               
        error_Interior_adaptive = error_Interior_adaptive/N_iter;
        obj_gap_Interior_adaptive = obj_gap_Interior_adaptive/N_iter;  
    end
    %% run fixed_stepsize algorithm
    
    run_fixed = 1;
    
    if run_fixed == 1
        
        infeas_Interior_fixed = [];
        error_Interior_fixed = [];
        obj_gap_Interior_fixed = [];
        for iter = 1:N_iter
            fprintf("Run %d / %d fixed algorithm for interior problem.\n", iter,N_iter)
            [index_list_fixed,infeas_list,obj_gap_list,error_list] = fixed_QP(x_0,lambda_0,mu_0,K,cvx_x,y_opt,Q,x_center,gamma_interior,s,w,epsilon,c,mu_omega,Cov,N,m);
            
            if iter == 1
                infeas_Interior_fixed = infeas_list;
                error_Interior_fixed = error_list;
                obj_gap_Interior_fixed = obj_gap_list;
            else   
                
                infeas_Interior_fixed = infeas_Interior_fixed + infeas_list;               
                error_Interior_fixed = error_Interior_fixed + error_list;
                obj_gap_Interior_fixed = obj_gap_Interior_fixed + obj_gap_list;     
            end
                
        end
        
        infeas_Interior_fixed = infeas_Interior_fixed/N_iter;               
        error_Interior_fixed = error_Interior_fixed/N_iter;
        obj_gap_Interior_fixed = obj_gap_Interior_fixed/N_iter;  
        
        
    end
    %%
    x = index_list;
    
    n_data = length(x);
    
    
    
    figure
    set(gca,'FontSize',30);
    tmpx_idx = x>0; %((x>=1e4).*(x<=1e5))>0;
    tmplist_idx = index_list_fixed>0; %((index_list_fixed>= 1e4).*(index_list_fixed<=1e5))>0;
    
    plot(x(tmpx_idx),infeas_Interior_adaptive(tmpx_idx), index_list_fixed(tmplist_idx),infeas_Interior_fixed(tmplist_idx), 'lineWidth', 4 );
    legend('    Adp-CSPD ', '    Basic-CSPD', 'fontSize', 20,'Interpreter','latex');

    xlabel('$N$','fontsize',32,'Interpreter','latex');
    ylabel('$|| H(\bar x_{N})_+||_2$','fontsize',32,'FontName','Times New Roman','Interpreter','latex');
    title("Empirical Infeasibility Residual Convergence", 'FontSize', 32)


    %%
    x = index_list;
   
    log_x = log(x);
    
    log_x_fixed = log(index_list_fixed);
    log_obj_gap_Interior_adaptive = log(abs(obj_gap_Interior_adaptive));
    log_obj_gap_Interior_fixed = log(abs(obj_gap_Interior_fixed));
    
    
    n_data = length(x);
    
    y = zeros(n_data);
    y_last = log_obj_gap_Interior_adaptive (n_data);

    y = y_last - 0.3 + (log_x(n_data) - log_x)*0.5;
    figure
    set(gca,'FontSize',30);
    
    plot(log_x,log_obj_gap_Interior_adaptive, '-', log_x_fixed,log_obj_gap_Interior_fixed, ':', log_x,y, '-.x','lineWidth', 4 );
    legend('    Adp-CSPD ', '    Basic-CSPD', '$\log y = -\frac{1}{2}\log N + C$', 'fontSize', 20,'Interpreter','latex');
    % plot(log_x,log_obj_gap_Interior_adaptive, log_x_fixed,log_obj_gap_Interior_fixed, log_x,y, 'lineWidth', 4 );
    % legend('    Adaptive', '   Fixed', ' slope = -0.5', 'fontSize', 20);

    xlabel('$\log(N)$','fontsize',32,'Interpreter','latex');
    ylabel('$\log(|F(\bar x_{N},y^\ast) -F(x^{\ast},\bar y_N)|)$','fontsize',32,'FontName','Times New Roman','Interpreter','latex');
    title("Empirical Objective Log-Convergence", 'FontSize', 32)

    figure
    set(gca,'FontSize',30);

    


    plot(x,obj_gap_Interior_adaptive,'-', index_list_fixed,obj_gap_Interior_fixed,':', 'lineWidth', 4 );
    legend('    Adp-CSPD ', '    Basic-CSPD', 'fontSize', 20,'Interpreter','latex');

    xlabel('$N$','fontsize',32,'Interpreter','latex');
    ylabel('$F(\bar x_{N},y^\ast) -F(x^{\ast},\bar y_N)$','fontsize',32,'FontName','Times New Roman','Interpreter','latex');
    title("Empirical Objective Convergence", 'FontSize', 32)
end



%% solve the case where optimal solution falls at the boundary

solve_boundary = 0;

if solve_boundary == 1
    display('solve the case where optimal solution falls at the boundary');
    gamma_boundary = 0.9 * gamma;
    % for i= [1,5,7,11]
    %     gamma(i) = gamma(i) + normrnd(0,1) ;
    % end
    cvx_begin
        variable x(d)
        minimize( (x - x_center)' * Q * (x - x_center) + c*norm(x) + x'*mu_0 + x'*mu_omega);
        subject to
        for i=1:m
            (epsilon(i, :) + (x - s(:, i))'* w(:, i))*(epsilon(i, :) + (x - s(:, i))'* w(:, i))' / N <= gamma_boundary(i);
        end
    cvx_end

    cvx_x = x;

    all_feasibility(cvx_x, s, w, epsilon, gamma_boundary)


    %for K = [800000,820000,850000,880000,900000,920000,950000,980000,1000000]
    x_0 = 10*cvx_x;

    lambda_0 = rand(m, 1);

    K = 1e6;
    run_adaptive = 1;
    run_fixed = 1;

        if run_adaptive == 1

            infeas_Boundary_adaptive = [];
            index_Boundary_adaptive = [];
            error_Boundary_adaptive = [];
            obj_gap_Boundary_adaptive = [];
            for iter = 1:N_iter
                fprintf("Run %d / %d adpative algorithm for boundary problem.\n", iter,N_iter)
                [index_list,infeas_list,obj_gap_list,error_list] = adaptive_QP(x_0,lambda_0,mu_0,K,cvx_x,y_opt,Q,x_center,gamma_boundary,s,w,epsilon,c,mu_omega,Cov,N,m);

                if iter == 1
                    infeas_Boundary_adaptive = infeas_list;
                    index_Boundary_adaptive = index_list;
                    error_Boundary_adaptive = error_list;
                    obj_gap_Boundary_adaptive = obj_gap_list;
                else   

                    infeas_Boundary_adaptive = infeas_Boundary_adaptive + infeas_list;               
                    index_Boundary_adaptive = index_Boundary_adaptive + index_list;
                    error_Boundary_adaptive = error_Boundary_adaptive + error_list;
                    obj_gap_Boundary_adaptive = obj_gap_Boundary_adaptive + obj_gap_list;     
                end

            end

            infeas_Boundary_adaptive = infeas_Boundary_adaptive/N_iter;               
            index_Boundary_adaptive = index_Boundary_adaptive/N_iter;
            error_Boundary_adaptive = error_Boundary_adaptive/N_iter;
            obj_gap_Boundary_adaptive = obj_gap_Boundary_adaptive/N_iter;  
        end

        if run_fixed == 1

            infeas_Boundary_fixed = [];
            index_Boundary_fixed = [];
            error_Boundary_fixed = [];
            obj_gap_Boundary_fixed = [];
            for iter = 1:N_iter
                fprintf("Run %d / %d fixed algorithm for boundary problem.\n", iter,N_iter)
                [index_list_fixed,infeas_list,obj_gap_list,error_list] = fixed_QP(x_0,lambda_0,mu_0,K,cvx_x,y_opt,Q,x_center,gamma_boundary,s,w,epsilon,c,mu_omega,Cov,N,m);

                if iter == 1
                    infeas_Boundary_fixed = infeas_list;
                    index_Boundary_fixed = index_list;
                    error_Boundary_fixed = error_list;
                    obj_gap_Boundary_fixed = obj_gap_list;
                else   

                    infeas_Boundary_fixed = infeas_Boundary_fixed + infeas_list;               
                    index_Boundary_fixed = index_Boundary_fixed + index_list;
                    error_Boundary_fixed = error_Boundary_fixed + error_list;
                    obj_gap_Boundary_fixed = obj_gap_Boundary_fixed + obj_gap_list;     
                end

            end

            infeas_Boundary_fixed = infeas_Boundary_fixed/N_iter;               
            index_Boundary_fixed = index_Boundary_fixed/N_iter;
            error_Boundary_fixed = error_Boundary_fixed/N_iter;
            obj_gap_Boundary_fixed = obj_gap_Boundary_fixed/N_iter;  
        end


    %%
        x = index_list;
        log_x = log(x);
        log_x_fixed = log(index_list_fixed);
        log_infeas_Boundary_adaptive = log(abs(infeas_Boundary_adaptive));
        log_infeas_Boundary_fixed = log(abs(infeas_Boundary_fixed));

        n_data = length(x);

        y = zeros(n_data);
        y_last = log_infeas_Boundary_adaptive(n_data);

        y = y_last - 0.3 + (log_x(n_data) - log_x)*0.5;

        figure
        set(gca,'FontSize',30);
        plot(log_x,log_infeas_Boundary_adaptive, '-', log_x_fixed,log_infeas_Boundary_fixed, ':', log_x, y, '-.x', 'lineWidth', 4 );
        legend('    Adp-CSPD ', '    Basic-CSPD', '$\log y = -\frac{1}{2}\log N + C$', 'fontSize', 20,'Interpreter','latex');
        xlabel('$\log(N)$','fontsize',32,'Interpreter','latex');
        ylabel('$\log(|| H(\bar x_{N})_+||_2)$','fontsize',32,'FontName','Times New Roman','Interpreter','latex');
        title("Empirical Feasibility Residual Log-Convergence", 'FontSize', 32);


    %%
        x_fixed = index_list_fixed;
        figure
        set(gca,'FontSize',30);
        plot(x,obj_gap_Boundary_adaptive,'-', x_fixed, obj_gap_Boundary_fixed,':', 'lineWidth', 4 );

        legend('    Adp-CSPD ', '    Basic-CSPD', 'fontSize', 20,'Interpreter','latex');
        xlabel('$N$','fontsize',32,'Interpreter','latex');
        ylabel('$F(\bar x_{N},y^\ast) -F(x^{\ast},\bar y_N)$','fontsize',32,'FontName','Times New Roman','Interpreter','latex');
        title("Empirical Objective Convergence", 'FontSize', 32);
        

        log_x_fixed = log(x_fixed);
        
        figure
        set(gca,'FontSize',30);
        plot(log_x,log(abs(obj_gap_Boundary_adaptive)),'-', log_x_fixed, log(abs(obj_gap_Boundary_fixed)), ':', log_x, y, '-.x','lineWidth', 4 );

        legend('    Adaptive ','    Fixed ','fontSize', 20);
        xlabel('$\log N$','fontsize',32,'Interpreter','latex');
        ylabel('$\log(|F(\bar x_{N},y^\ast) -F(x^{\ast},\bar y_N)|)$','fontsize',32,'FontName','Times New Roman','Interpreter','latex');
        title("Empirical Objective Log-Convergence", 'FontSize', 32);

end

