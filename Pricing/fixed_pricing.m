function [index_list,infeas_list,obj_gap_list] = fixed_pricing(K,s_0,S,p_s,Demand,l_theta,u_theta,l_c,u_c,sigma,theta_0,c_0,p_0,theta_opt, c_opt,p_opt )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here



% s_0 feature of the target product d x 1
% S feature of the reference products m x d
% p_s fixed prices   m x 1
% Demand m x 1

% theta_0    % d x 1
% c_0  1
% p_0  1
%



list_K = [3000];

for kk = 10:0.5:log(K)

    %% test exp(kk)
    KK = ceil(exp(kk));

    list_K = [list_K, KK ];

end
list_K = [list_K, K];
    
infeas_list = [];
index_list = [];
obj_gap_list = [];

for KK = list_K
    
    m = length(Demand);


theta_old = theta_0;
c_old = c_0;
p_old = p_0;

gamma_old = zeros(m,1);





theta_K = theta_old;
c_K = c_old;
p_K = p_old;



    for k = 1:KK


        %% setup step-sizes 
        C1 = 100;
        C2 = 10;
        beta_k = C1*sqrt(KK);
        %tau_k = C1*(sqrt(k+2) - sqrt(k+1));

        eta_k = C2*sqrt(KK);
        %rho_k = C2*(sqrt(k+3)- sqrt(k+2));

        kappa_k = eta_k;
        %psi_k = rho_k;


        %% Update dual variable gamma

        h_w = S*theta_old + p_s*c_old + normrnd(0,sigma,[m,1]) - Demand;
        h_w = -h_w;


        gamma_new = gamma_old + h_w/beta_k ;    
        gamma_new = max(gamma_new,0);

        %% Update primal variable p, theta, c
        omega = normrnd(0,sigma);


        nabla_g_theta = S;   % m x d
        nabla_g_c = p_s;       % m x 1

        nabla_g_theta = - nabla_g_theta;
        nabla_g_c = - nabla_g_c;

        nabla_F_theta =  p_old*s_0;  % d x 1
        nabla_F_c = p_old^2;    % 1 x 1
        nabla_F_p = (s_0'*theta_old + omega) + 2* c_old * p_old; % 1 x 1


        nabla_L_theta = nabla_F_theta + nabla_g_theta'*gamma_old;  % d x 1

        nabla_L_c = nabla_F_c + nabla_g_c'*gamma_old;   % 1 x 1


        % update p

        p_new = p_old+ nabla_F_p/eta_k;
        p_new = max(p_new,0);


        % update theta
        theta_new = theta_old - nabla_L_theta/kappa_k;
        theta_new = max(l_theta,theta_new);
        theta_new = min(u_theta,theta_new);

        % update c
        c_new = kappa_k*c_old - nabla_L_c/kappa_k;
        c_new = max(l_c, c_new);
        c_new = min(u_c, c_new);



        p_old = p_new;
        theta_old = theta_new;
        c_old = c_new;
        gamma_old = gamma_new;


        %% update weighted average 
        theta_K = (k-1)*theta_K/k + theta_new/k;
        p_K = (k-1)*p_K/k + p_new/k;
        c_K = (k-1)*c_K/k + c_new/k;

        if mod(k, 5000) == 0
            feasibility = norm(max(Demand - (S*theta_K + p_s*c_K),0));
            F1 = p_opt*(s_0'*theta_K + c_K *p_opt);
            F2 = p_K*(s_0'*theta_opt + c_opt*p_K);
            obj_gap = F1 - F2;
        %fprintf('In iteration %d, objective value: %f, feasibility %f: \n', k, obj_gap, feasibility);
        fprintf('In iteration %d, obj_gap: %f, feasibility: %f \n',  k, obj_gap, feasibility);
        
        
            %fprintf('In iteration %d, objective value: %f, feasibility %f: \n', k, obj_gap, feasibility);
            %fprintf('In iteration %d, feasibility: %f \n',  k, feasibility);

    %         obj_gap_list = [obj_gap_list, obj_gap];
    %         error_list = [error_list, error] ;
        end

    end
    feasibility = norm(max(Demand - (S*theta_K + p_s*c_K),0));
    F1 = p_opt*(s_0'*theta_K + c_K *p_opt);
    F2 = p_K*(s_0'*theta_opt + c_opt*p_K);
    obj_gap = F1 - F2;
    
    obj_gap_list = [obj_gap_list,obj_gap];
    infeas_list = [infeas_list, feasibility];
    index_list = [index_list, KK];
    
end

end

