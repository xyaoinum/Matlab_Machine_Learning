function [] = q3solution_a()
load q3_data.mat
SVMStruct = svmtrain(q3x_train,q3t_train,'kernel_function','linear','showplot',true);

Group = svmclassify(SVMStruct,q3x_test);

correct_matrix = Group.*q3t_test;

accuracy = sum(correct_matrix == 1)/size(correct_matrix,1)

end
