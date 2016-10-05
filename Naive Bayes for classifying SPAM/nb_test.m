
[spmatrix, tokenlist, category] = readMatrix('MATRIX.TEST');

testMatrix = full(spmatrix);
numTestDocs = size(testMatrix, 1);
numTokens = size(testMatrix, 2);

output = zeros(numTestDocs, 1);

output = (testMatrix*(log_pwspam.')+log_pspam) > (testMatrix*(log_pwnonspam.')+log_pnonspam);

% Compute the error on the test set
error=0;
for i=1:numTestDocs
  if (category(i) ~= output(i))
    error=error+1;
  end
end

%Print out the classification error on the test set
error/numTestDocs


