path = fullfile("C:\Users\nursa\OneDrive\softcomp\containerx3\");
imds = imageDatastore(path,IncludeSubfolders=true,LabelSource="foldernames");

tbl = countEachLabel(imds);

[imdsTrain,imdsValidation] = splitEachLabel(imds, 200, 'randomize');

inputSize = [254 254 3];
numClasses = 4;

layers = [
    imageInputLayer(inputSize)
    convolution2dLayer(5,20)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm', ...
    'MaxEpochs',6, ...
    'MiniBatchSize',10, ...
    'InitialLearnRate',3e-4, ...
    'Plots','training-progress');

net = trainNetwork(imdsTrain,layers,options);

YPred = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;
accuracy1 = mean(YPred == YValidation);

% inputSize = net.Layers(1).InputSize;
% augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
% augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);
% 
% net = trainNetwork(imdsTrain,layers,options);
% 
% YPred = classify(net,imdsValidation);
% YValidation = imdsValidation.Labels;
% accuracy2 = mean(YPred == YValidation);

