%q1solution_b and q1solution_c are the same
%each of them will solve both part (b) and (c)

function [] = q1solution_b()
A = double(imread('mandrill-small.tiff'));
imshow(uint8(round(A)));

num_cluster = 16;
num_point = size(A,1)*size(A,2);
k=1;
x=zeros(num_point,3);
for i=1:size(A,1)
    for j=1:size(A,2)
        x(k,:) = A(i,j,:);        
        k = k+1;
    end
end

mu=zeros(num_cluster,3);
for i=1:num_cluster
    mu(i,:)=x(randi([1 num_point]),:);
end



iter = 0;
while(iter < 50)
    r=zeros(num_point,num_cluster);
    for i=1:num_point
        r(i,argmin_index(x(i,:)',mu))=1; 
    end
    
    for i=1:num_cluster
        numerator = 0;
        for j=1:num_point
            numerator = numerator+r(j,i)*x(j,:);
        end
        denominator = sum(r(:,i));
        
        mu(i,:)=numerator/denominator;
    end
    
    iter = iter + 1;
    
    if rem(iter,5) == 0
        fprintf('%d%% completed\n',iter*10/5);
    end
        
end

mu

for i=1:size(A,1)
    for j=1:size(A,2)
        A(i,j,:)=argmin_value([A(i,j,1);A(i,j,2);A(i,j,3)],mu)';        
    end
end

figure

imshow(uint8(round(A)));

end


function [result] = argmin_index(xi,mu)

tmpmin = intmax;
for i=1:size(mu,1)
    mui=mu(i,:)';
    tmp = sum((xi-mui).^2);
    if tmp<tmpmin
        result = i;
        tmpmin = tmp;
    end
end
end

function [result] = argmin_value(xi,mu)
tmpmin = intmax;
for i=1:size(mu,1)
    mui=mu(i,:)';
    tmp = sum((xi-mui).^2);
    if tmp<tmpmin
        result = mui;
        tmpmin = tmp;
    end
end
end








