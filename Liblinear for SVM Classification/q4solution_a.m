function [] = q4solution_a()

addpath('liblinear-1.94/matlab');  % add LIBLINEAR to the path
[sparseTrainMatrix, ~, trainCategory] = readMatrix('MATRIX.TRAIN');

sizes = [50 100 200 400 800 1400];
errors = [0 0 0 0 0 0];

for i=1:6

    numTrainDocs = sizes(i)

    svmTrainCategory = 2*trainCategory-1;
    svmTrainCategory = svmTrainCategory.';
    model = train(svmTrainCategory(1:numTrainDocs,:),sparseTrainMatrix(1:numTrainDocs,:));
    
    [sparseTestMatrix, tokenlist, testCategory] = readMatrix('MATRIX.TEST');

    numTestDocs = size(sparseTestMatrix, 1);

    svmTestCategory = 2*testCategory-1;
    svmTestCategory = svmTestCategory.';

    [output] = predict(svmTestCategory,sparseTestMatrix,model);
    output = (output+1)/2;


    % Compute the error on the test set
    error=0;
    for j=1:numTestDocs
      if (testCategory(j) ~= output(j))
        error=error+1;
      end
    end

    %Print out the classification error on the test set
    error/numTestDocs
    errors(i) = error/numTestDocs;
    
    
    
end

plot(sizes,errors)
title('test set error ~ training set size','FontSize',15);
xlabel('training set size');
ylabel('test set error');

end

