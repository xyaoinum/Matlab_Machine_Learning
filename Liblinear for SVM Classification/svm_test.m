addpath('liblinear-1.94/matlab');  % add LIBLINEAR to the path
[sparseTestMatrix, tokenlist, testCategory] = readMatrix('MATRIX.TEST');

numTestDocs = size(sparseTestMatrix, 1);

svmTestCategory = 2*testCategory-1;
svmTestCategory = svmTestCategory.';

[output] = predict(svmTestCategory,sparseTestMatrix,model);
output = (output+1)/2;


% Compute the error on the test set
error=0;
for i=1:numTestDocs
  if (testCategory(i) ~= output(i))
    error=error+1;
  end
end

%Print out the classification error on the test set
error/numTestDocs
