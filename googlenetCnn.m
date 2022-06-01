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
%load pretrained network GoogLeNet
net2 = googlenet;
%adjust the images to network's input size
inputSize = net2.Layers(1).InputSize;
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

lgraph = layerGraph(net2);
% LayerGraph with properties:
% 
%          Layers: [144×1 nnet.cnn.layer.Layer]
%     Connections: [170×2 table]
%      InputNames: {'data'}
%     OutputNames: {'new_classoutput'}

%get the number of classes
numClasses = numel(categories(imdsTrain.Labels));
%create a new layer for transfer learning
newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
%replace the current layer with new layer
lgraph = replaceLayer(lgraph,'loss3-classifier',newLearnableLayer);
newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,'output',newClassLayer);
%set the training options
options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',3e-4, ...
    'Plots','training-progress');
%train the network 
newNet = trainNetwork(augimdsTrain,lgraph,options);
% |========================================================================================|
% |  Epoch  |  Iteration  |  Time Elapsed  |  Mini-batch  |  Mini-batch  |  Base Learning  |
% |         |             |   (hh:mm:ss)   |   Accuracy   |     Loss     |      Rate       |
% |========================================================================================|
% |       1 |           1 |       00:00:11 |       30.00% |       2.4595 |          0.0003 |
% |       1 |          50 |       00:02:08 |       60.00% |       0.9513 |          0.0003 |
% |       2 |         100 |       00:04:01 |      100.00% |       0.0428 |          0.0003 |
% |       2 |         150 |       00:05:56 |      100.00% |       0.0021 |          0.0003 |
% |       3 |         200 |       00:07:42 |       90.00% |       0.1234 |          0.0003 |
% |       4 |         250 |       00:09:27 |       90.00% |       0.3395 |          0.0003 |
% |       4 |         300 |       00:11:12 |      100.00% |       0.0087 |          0.0003 |
% |       5 |         350 |       00:12:57 |      100.00% |       0.0081 |          0.0003 |
% |       5 |         400 |       00:14:42 |      100.00% |       0.0072 |          0.0003 |
% |       6 |         450 |       00:16:28 |      100.00% |       0.0032 |          0.0003 |
% |       6 |         480 |       00:17:32 |      100.00% |       0.0010 |          0.0003 |
% |========================================================================================|
%%
%measure the accuracy of classifier
[YPred,probs] = classify(newNet,augimdsValidation);
accuracy = mean(YPred == imdsValidation.Labels);
%generate confusion matrix
confMat = confusionmat(YPred,imdsValidation.Labels);
label = unique(imdsValidation.Labels);
figure
confusionchart(confMat,label)
%view predicted images
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

[i,score]=classify(newNet,adjust);

str = string(i)+" "+score(i)*100+"%";
figure
imshow(guess)
title(str)