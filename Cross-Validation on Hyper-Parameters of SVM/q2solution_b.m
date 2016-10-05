function [] = q2solution_b()
load q2_data.mat

num_features = size(q2x_train , 2);
num_train = size(q2x_train , 1)/2;
num_val = size(q2x_train , 1)/2;
num_test = size(q2x_test , 1);

x_train = q2x_train(1:num_train,:);
x_val = q2x_train((num_train+1):size(q2x_train , 1),:);
x_test = q2x_test;

t_train = q2t_train(1:num_train,:);
t_val = q2t_train((num_train+1):size(q2t_train , 1),:);
t_test = q2t_test;

tx = t_train * ones(1, num_features);
tx = tx .* x_train;


all_eta = [0.01,0.5,1,10,100];
all_C = [0.01,0.1,1,10,100,1000];
niter = 200;

error = intmax;
chosen_eta = 1;
chosen_C = 1;

for eta_index = 1:size(all_eta,2)
	eta = all_eta(eta_index);
	for C_index = 1:size(all_C,2)
		C = all_C(C_index);
		w = zeros(num_features,1);
		b = 0;
		

		for iter = 1:niter
			alpha = eta/(1+eta*iter);
			for i = 1:num_train
				indicator = (t_train(i,:).*(x_train(i,:)*w+b)) < 1;
				grad_w = w/num_train - C * tx(i,:)' * indicator;
				grad_b = -C * t_train(i,:)' * indicator;
				w = w - alpha * grad_w;
				b = b - alpha * grad_b * 0.01;
			end
		end
		error_val_array = (t_val .* (x_val * w + b)) < 0;
		error_val = sum(error_val_array);
		if error_val < error
			error = error_val;
			chosen_eta = eta;
			chosen_C = C;
		end
		
		fprintf('eta:%d,C:%d,error:%d\n',eta,C,error_val);
		
	end
end

fprintf('\nchosen_eta: %d\nchosen_C: %d\n',chosen_eta,chosen_C);

num_train = size(q2x_train , 1);

x_train = q2x_train;
t_train = q2t_train;

eta = chosen_eta;
C = chosen_C;
niter = 200;

tx = t_train * ones(1, num_features);
tx = tx .* x_train;


w = zeros(num_features,1);
b = 0;

for iter = 1:niter
	alpha = eta/(1+eta*iter);
	for i = 1:num_train
		indicator = (t_train(i,:).*(x_train(i,:)*w+b)) < 1;
		grad_w = w/num_train - C * tx(i,:)' * indicator;
		grad_b = -C * t_train(i,:)' * indicator;
		w = w - alpha * grad_w;
		b = b - alpha * grad_b * 0.01;
	end
end
error_test_array = (t_test .* (x_test * w + b)) < 0;
error_test = sum(error_test_array);



fprintf('test_error: %d (out of %d test examples)\n',error_test,size(t_test,1));





end

