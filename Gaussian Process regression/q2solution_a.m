function [  ] = q2solution_a(  )

q2plot(1,0.1)
q2plot(0.2,0.5)
q2plot(10,0.01)

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

function [result] = get_var(x,new_x,inv_beta,sigma)
	k = make_k(x,new_x,sigma);
	C = make_C(x,inv_beta,sigma);
	c = make_c(new_x,inv_beta,sigma);
	result = c - k'/C*k;
end

function [] = q2plot(sigma,inv_beta)
figure
load('q2x.dat');
load('q2y.dat');
x = q2x;
t = q2y;
plot(x,t,'LineStyle','none','Marker','.')
hold on
x_new_set = min(x):0.2:max(x);
t_new_set = zeros(1,length(x_new_set));
e_new_set = zeros(1,length(x_new_set));
for i=1:length(x_new_set)
	t_new_set(i) = get_mean(x,t,x_new_set(i),inv_beta,sigma);
	e_new_set(i) = get_var(x,x_new_set(i),inv_beta,sigma);
end
errorbar(x_new_set,t_new_set,e_new_set)

tt = sprintf('sigma=%d, inv-beta=%d',sigma,inv_beta);
title(tt);
xlabel('x');
ylabel('t');

end




