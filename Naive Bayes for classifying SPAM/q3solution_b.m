function [  ] = q3solution_b(  )
nb_train;

indicator = log_pwspam - log_pwnonspam;

[value,index] = sort(indicator,'descend');

fprintf('top five indices: ')
for i=1:5
    fprintf('%d ',index(i))
end
fprintf('\n')

end

