function [  ] = q2solution_d(  )

s_set = [0.2 0.5 1 2 5];
b_set = [0.01 0.03 0.1 0.3];
max_llh = intmin;

completed = 0;

for i = 1:length(s_set)
	for j=1:length(b_set)
		sigma = s_set(i);
		inv_beta = b_set(j);
		[x_new_set,t_new_set,e_new_set,llh] = getllh(sigma,inv_beta);
		if llh > max_llh
			max_llh = llh;
			max_x_new_set = x_new_set;
			max_t_new_set = t_new_set;
			max_e_new_set = e_new_set;
			max_sigma = sigma;
			max_inv_beta = inv_beta;
		end
		
		completed = completed + 5;
		fprintf('%d%% completed\n',completed);
		
	end
end

q2plot(max_x_new_set,max_t_new_set,max_e_new_set,max_sigma,max_inv_beta)
max_llh_sigma = max_sigma
max_llh_inv_beta = max_inv_beta
max_llh


end

function [result] = kernel(xn,xm,sigma)
	result = exp(-1/(2*sigma^2)*(xn-xm)^2);
end

function [result] = make_k(x,new_x,sigma)
	result = zeros(length(x),1);
	for i=1:length(x)
		result(i) = kernel(x(i),new_x,sigma);
	end
end

function [result] = delta(n,m)
	result = (n==m);
end

function [result] = make_C(x,inv_beta,sigma)
	result = zeros(length(x));
	for n = 1:length(x)
		for m=1:length(x)
			result(n,m) = kernel(x(n),x(m),sigma) + inv_beta*delta(n,m);
		end
	end
end

function [result] = make_c(new_x,inv_beta,sigma)
	result = kernel(new_x,new_x,sigma) + inv_beta;
end

function [result] = get_mean(x,t,new_x,inv_beta,sigma)
	k = make_k(x,new_x,sigma);
	C = make_C(x,inv_beta,sigma);
	result = k'/C*t;
end

function [result] = get_mean_C(x,t,new_x,inv_beta,sigma,C)
	k = make_k(x,new_x,sigma);
	result = k'/C*t;
end


function [result] = get_var(x,new_x,inv_beta,sigma)
	k = make_k(x,new_x,sigma);
	C = make_C(x,inv_beta,sigma);
	c = make_c(new_x,inv_beta,sigma);
	result = c - k'/C*k;
end

function [result] = make_Cplus(C,k,c)
	result = [C k;k' c];
end

function [result] = lognormal(x,mu,sigma)
	result = log(1/sqrt(det(sigma)*(2*pi)^3))-1/2*(x-mu)'/sigma*(x-mu);
end

function [x_new_set,t_new_set,e_new_set,llh] = getllh(sigma,inv_beta)

load('q2x.dat');
load('q2y.dat');
x = q2x;
t = q2y;

x_new_set = min(x):0.2:max(x);
t_new_set = zeros(1,length(x_new_set));
e_new_set = zeros(1,length(x_new_set));

result = 0;
C = make_C(x,inv_beta,sigma);

for i=1:length(x_new_set)
	t_new_set(i) = get_mean(x,t,x_new_set(i),inv_beta,sigma);
	e_new_set(i) = get_var(x,x_new_set(i),inv_beta,sigma);
	
	x_new = x_new_set(i);
	t_new = t_new_set(i);
	x_plus = [x;x_new];
	C_plus = make_Cplus(C,make_k(x,x_new,sigma),make_c(x_new,inv_beta,sigma));
	t_plus = [t;t_new];
	mu_plus = zeros(length(t_plus),1);
	
	result = result + lognormal(t_plus,mu_plus,C_plus);
end

llh = result;

end

function []  = q2plot(x_new_set,t_new_set,e_new_set,sigma,inv_beta)
load('q2x.dat');
load('q2y.dat');
x = q2x;
t = q2y;
figure
plot(x,t,'LineStyle','none','Marker','.')
hold on
errorbar(x_new_set,t_new_set,e_new_set)
tt = sprintf('sigma=%d, inv-beta=%d',sigma,inv_beta);
title(tt);
xlabel('x');
ylabel('t');
end



