addpath('liblinear-1.94/matlab');  % add LIBLINEAR to the path
[sparseTrainMatrix, tokenlist, trainCategory] = readMatrix('MATRIX.TRAIN');

numTrainDocs = size(sparseTrainMatrix, 1);

svmTrainCategory = 2*trainCategory-1;
svmTrainCategory = svmTrainCategory.';
model = train(svmTrainCategory,sparseTrainMatrix);