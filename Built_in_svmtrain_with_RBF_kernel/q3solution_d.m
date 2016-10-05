function [] = q3solution_c()

load q3_data.mat

num_train = size(q3x_train , 1)/5*4;
num_val = size(q3x_train , 1)/5;

all_sigma = [0.2, 0.5, 1, 1.5, 2, 2.5, 3];

accurate = -1;

chosen_sigma = 0;

for sigma_index = 1:size(all_sigma,2)
	sigma = all_sigma(1,sigma_index);
	
	accurate_sum = 0;
	
	for fold_index = 1:5
		x_train = q3x_train([1:(num_val*(fold_index-1)),(1+num_val*fold_index):size(q3x_train , 1)],:);
		x_val = q3x_train((num_val*(fold_index-1)+1):(num_val*fold_index),:);
		t_train = q3t_train([1:(num_val*(fold_index-1)),(1+num_val*fold_index):size(q3x_train , 1)],:);
		t_val = q3t_train((num_val*(fold_index-1)+1):(num_val*fold_index),:);
		SVMStruct = svmtrain(x_train,t_train,'kernel_function','rbf','rbf_sigma',sigma);
		Group = svmclassify(SVMStruct,x_val);
		correct_matrix = Group.*t_val;
		accurate_sum = accurate_sum + sum(correct_matrix == 1);
	end

	if accurate_sum > accurate
		accurate = accurate_sum;
		chosen_sigma = sigma;
	end
end

chosen_sigma


end

