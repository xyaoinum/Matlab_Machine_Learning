function [] = q3solution_c()

sizes = [50 100 200 400 800 1400];
errors = [0 0 0 0 0 0];

for i=1:6
    
    sizes(i);
    
    [spmatrix, tokenlist, trainCategory] = readMatrix(strcat('MATRIX.TRAIN.',num2str(sizes(i))));

    trainMatrix = full(spmatrix);
    numTrainDocs = size(trainMatrix, 1);
    numTokens = size(trainMatrix, 2);

    spamMatrix = trainMatrix(trainCategory == 1,:);
    nonspamMatrix = trainMatrix(trainCategory == 0,:);

    pspam = size(spamMatrix,1)/(size(spamMatrix,1)+size(nonspamMatrix,1));
    pnonspam = 1 - pspam;

    log_pspam = log(pspam);
    log_pnonspam = log(pnonspam);

    deno_spam = (sum(sum(spamMatrix))+size(spamMatrix,2));
    log_pwspam = log(sum(spamMatrix) + 1) - log(deno_spam);

    deno_nonspam = (sum(sum(nonspamMatrix))+size(nonspamMatrix,2));
    log_pwnonspam = log(sum(nonspamMatrix) + 1) - log(deno_nonspam);

    [spmatrix, tokenlist, category] = readMatrix('MATRIX.TEST');

    testMatrix = full(spmatrix);
    numTestDocs = size(testMatrix, 1);

    output = (testMatrix*(log_pwspam.')+log_pspam) > (testMatrix*(log_pwnonspam.')+log_pnonspam);

    % Compute the error on the test set
    error=0;
    for j=1:numTestDocs
      if (category(j) ~= output(j))
        error=error+1;
      end
    end

    %Print out the classification error on the test set
    errors(i) = error/numTestDocs;

end

sizes
errors
plot(sizes,errors)

title('test set error ~ training set size','FontSize',15);
xlabel('training set size');
ylabel('test set error');


end

