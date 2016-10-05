function [] = q2solution_c()
load q2_data.mat
num_features = size(q2x_train , 2);
num_train = size(q2x_train , 1)/10*9;
num_val = size(q2x_train , 1)/10;
num_test = size(q2x_test , 1);

x_test = q2x_test;
t_test = q2t_test;


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
		
		error_sum = 0;
		
		for fold_index = 1:10
			x_train = q2x_train([1:(num_val*(fold_index-1)),(1+num_val*fold_index):size(q2x_train , 1)],:);
			x_val = q2x_train((num_val*(fold_index-1)+1):(num_val*fold_index),:);
			t_train = q2t_train([1:(num_val*(fold_index-1)),(1+num_val*fold_index):size(q2x_train , 1)],:);
			t_val = q2t_train((num_val*(fold_index-1)+1):(num_val*fold_index),:);
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
			error_val_array = (t_val .* (x_val * w + b)) < 0;
			error_sum = error_sum + sum(error_val_array);
		end

		if error_sum < error
			error = error_sum;
			chosen_eta = eta;
			chosen_C = C;
		end
		fprintf('eta:%d,C:%d,error:%d\n',eta,C,error_sum);
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

