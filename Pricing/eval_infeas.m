function norm_temp = eval_infeas(S,p_s,Demand,theta_K,c_K)
%UNTITLED Summary of this function goes here
%   Evaluate infeasibility of constraints


temp =  S*theta_K + p_s*c_K - Demand;
temp = max(temp,0);

norm_temp = norm(temp);

end

