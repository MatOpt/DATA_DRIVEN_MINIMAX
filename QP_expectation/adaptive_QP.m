function [index_list,infeas_list,obj_gap_list,error_list] = adaptive_QP(x_0,lambda_0,mu_0,K,x_opt,y_opt,Q,x_center,gamma,s,w,epsilon,c,mu_omega,Cov,N,m)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

    d = length(x_0);
    
    x_K = x_0;

    lambda_old = lambda_0;
    x_old = x_0;

    y_0 = mu_0;
    y_old = y_0;

    y_K = zeros(d,1);

    %nfea_list = zeros(K/10000, 1);
    %index_list = zeros(K/10000, 1);

    infeas_list = [];
    index_list = [];
    error_list = [];
    obj_gap_list = [];


    for k = 1:K
        p = randi(N);
        q = randi(N);

        c1 = 500;
        c2 = 30;      
        if k <= 1e5


            sk = 10000;
            % for dual variable
            alpha_k = c1 *sqrt(sk+1);  
            rho_k = c1 *(sqrt(sk+2)- sqrt(sk+1) );

            % for primal variables
            beta_k =  c2 * sqrt(sk+2);
            zeta_k = c2 * (sqrt(sk+3) - sqrt(sk+2));

            kappa_k = c2 * sqrt(sk+2);
            phi_k = c2 * (sqrt(sk+3) - sqrt(sk+2));



        else

            alpha_k = c1 *sqrt(k+1);
            rho_k = c1 *(sqrt(k+2)- sqrt(k+1) ) ;


            beta_k = c2 * sqrt(k+2);
            zeta_k = c2 * (sqrt(k+3) - sqrt(k+2));

            kappa_k = c2 * sqrt(k+2);
            phi_k = c2 * (sqrt(k+3) - sqrt(k+2));
        end


        omega = normrnd(0,1,[d,1]);

        omega = mvnrnd(mu_omega,Cov);

        grad_x = 2 * Q * (x_old - x_center) + y_old + omega';
        grad_lambda = zeros(m, 1); 
        for j = 1:m
            grad_x = grad_x + 2*lambda_old(j)*((x_old - s(:, j))'*w(:, j) + epsilon(j, p))*w(:, j);
            grad_lambda(j) = ((x_old - s(:, j))'*w(:, j) + epsilon(j, q))^2 - gamma(j);
        end

        % update dual 
        lambda_new = (alpha_k*lambda_old +  rho_k* lambda_0 + grad_lambda ) /(alpha_k + rho_k);
        lambda_new = max(lambda_new,0);

        % update primal
        x_new = ( beta_k *x_old +  zeta_k * x_0 - grad_x )/(beta_k + zeta_k);

        x_old = x_new;
        lambda_old = lambda_new;


        x_K = (k-1)*x_K/k + x_old/k;


        % update y 
        y_new = (x_old + kappa_k*y_old + phi_k*y_0)/(kappa_k + phi_k);

        %  project on l2 Ball centered at mu_0

        y_tmp = y_new - mu_0;   
        y_tmp_norm = norm(y_tmp);

        if y_tmp_norm > c
            y_new = mu_0 + c*y_tmp/y_tmp_norm;  % normalize the vector and make projection
        end 

        y_old = y_new;
        y_K = (k-1)*y_K/k + y_old/k;


        if mod(k, 10000) == 0
            feasibility = norm(max(all_feasibility(x_K, s, w, epsilon, gamma),0));

            obj_gap = (x_K - x_center)'*Q*(x_K - x_center) + x_K'*y_opt + x_K'*mu_omega - (x_opt - x_center)'*Q*(x_opt - x_center) -  x_opt'*y_K - x_opt'*mu_omega ; 
            error = norm(x_K - x_opt)^2;


            fprintf('In iteration %d, objective value: %f, feasibility %f: \n', k, obj_gap, feasibility);
            %fprintf('feasibility: %f \n', feasibility);
            infeas_list = [infeas_list, feasibility];
            index_list = [index_list, k];
            obj_gap_list = [obj_gap_list, obj_gap];
            error_list = [error_list, error] ;
        end
    end        
end

