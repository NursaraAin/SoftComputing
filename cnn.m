%%
%Set path to folder
Path = fullfile("containerx3\");
%Assign images using imageDatastore function
imds = imageDatastore(Path,IncludeSubfolders=true,LabelSource="foldernames");
%split the data
%training = 200 images
%testing = remaining images from imds
[imdsTrain,imdsValidation] = splitEachLabel(imds,200,'randomized');
%%
%set input size to images
inputSize = [254 254 3];
numClasses = 4;
%set layers
layers = [
    imageInputLayer(inputSize)
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];
%set training options
options = trainingOptions('sgdm', ...
    'MaxEpochs',6, ...
    'MiniBatchSize',10, ...
    'InitialLearnRate',3e-4, ...
    'Plots','training-progress');
%adjust the images according to input size
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);
%train new network
net = trainNetwork(augimdsTrain,layers,options);
% |========================================================================================|
% |  Epoch  |  Iteration  |  Time Elapsed  |  Mini-batch  |  Mini-batch  |  Base Learning  |
% |         |             |   (hh:mm:ss)   |   Accuracy   |     Loss     |      Rate       |
% |========================================================================================|
% |       1 |           1 |       00:00:04 |       30.00% |       1.4847 |          0.0003 |
% |       1 |          50 |       00:00:24 |       40.00% |      11.0029 |          0.0003 |
% |       2 |         100 |       00:00:44 |       20.00% |       1.4092 |          0.0003 |
% |       2 |         150 |       00:01:03 |       30.00% |       1.2273 |          0.0003 |
% |       3 |         200 |       00:01:22 |       70.00% |       0.9166 |          0.0003 |
% |       4 |         250 |       00:01:41 |       40.00% |       1.1182 |          0.0003 |
% |       4 |         300 |       00:02:00 |        0.00% |       1.4874 |          0.0003 |
% |       5 |         350 |       00:02:19 |       40.00% |       1.3932 |          0.0003 |
% |       5 |         400 |       00:02:39 |       60.00% |       0.9952 |          0.0003 |
% |       6 |         450 |       00:03:00 |       40.00% |       1.2367 |          0.0003 |
% |       6 |         480 |       00:03:11 |       60.00% |       0.9738 |          0.0003 |
% |========================================================================================|
%%
%measure accuracy
[YPred,probs] = classify(net,augimdsValidation);
accuracy1 = mean(YPred == imdsValidation.Labels);
%generate confusion matrix
confMat = confusionmat(YPred,imdsValidation.Labels);
label = unique(imdsValidation.Labels);
figure
confusionchart(confMat,label)
%view predicted images
%%
idx = randperm(numel(imdsValidation.Files),16);
figure
for i = 1:16
    subplot(4,4,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label) + ", " + num2str(100*max(probs(idx(i),:)),3) + "%");
end
%%
guess=imread("NursaraAin.jpg");
adjust=augmentedImageDatastore(inputSize(1:2),guess);

[i,score]=classify(net,adjust);

str = string(i)+" "+score(i)*100+"%";
figure
imshow(guess)
title(str)
