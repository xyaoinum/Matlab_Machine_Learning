
[spmatrix, tokenlist, trainCategory] = readMatrix('MATRIX.TRAIN');

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








