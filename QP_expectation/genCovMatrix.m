function covMat = genCovMatrix(n,type,sigma )
% generate the covariance matrix
% 3 types 
tol=10^-1;
covMat=[];

if abs(type-1)<tol
   % type 1, normal indepdent with sigma 
   covMat=sigma*eye(n);
   
elseif abs(type-2)<tol
    % type 2, normal distribution with 0.5 positve correlation
    covMat=0.3*ones(n,n)+0.5*eye(n);
    covMat=covMat*sigma;
elseif abs(type-3)<tol
    % type 3, normal distribution with Toeplitz correlation
    covMat=zeros(n,n);
    rho=2;
    for i=1:n
        for j=1:n
            covMat(i,j)=exp(-abs(i-j)/n*rho);
        end
    end
    covMat=covMat*sigma;
elseif abs(type-4) < tol
    % type 4, 
    
    covMat = zeros(n,n);
    
    for i=1:n 
        for j=1:n
            d= i-j;
            if d- floor(d/2)*2 ==0
                covMat(i,j)=0.5;
            else
                covMat(i,j)=-0.5;
            end
        end
    end
    covMat = covMat + 0.5*eye(n);
    covMat = covMat*sigma;
else
    fprintf('invalid type!');
end


end


