function [p_new,theta_new,c_new] = sequential_LP(s_0,S,p_s,Demand,l_theta,u_theta,l_c,u_c,theta_0,c_0,p_0,tol )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
m = length(Demand);


theta_old = theta_0;
c_old = c_0;
p_old = p_0;

gamma_old = zeros(m,1);


infeas_list = [];
index_list = [];

d = length(theta_0);
theta_K = theta_old;
c_K = c_old;
p_K = p_old;

run = 1;
while run

    
    % given p, solve an LP 
    
    cvx_begin
    variables theta(d) c 
    minimize (p_old *(s_0'*theta) + p_old^2*c);
    subject to
        l_theta <= theta <= u_theta; 
        l_c <= c <= u_c;
        S*theta + p_s * c >= Demand;
    cvx_end
    
    theta_new = theta;
    c_new = c;
    
    p_new = -0.5*s_0'*theta_new/c_new;
    
    fprintf('The price in this round is %f\n',p_new);
    
    error = norm(p_old - p_new) + norm(c_new - c_old) + norm(theta_old - theta_new);
    if error <= tol
        break
    end

    p_old = p_new;
    theta_old = theta_new;
    c_old = c_new;
        
    
end



end

