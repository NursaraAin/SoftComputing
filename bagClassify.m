%% 
Path = fullfile("containerx3\");
imds = imageDatastore(Path,IncludeSubfolders=true,LabelSource="foldernames");
tbl = countEachLabel(imds);

%figure
%montage(imds.Files(1:16:end))

[trainingSet, validationSet] = splitEachLabel(imds, 200, 'randomize');

bag = bagOfFeatures(trainingSet);

img = readimage(imds, 1);
featureVector = encode(bag, img);

% Plot the histogram of visual word occurrences
figure
bar(featureVector)
title('Visual word occurrences')
xlabel('Visual word index')
ylabel('Frequency of occurrence')

categoryClassifier = trainImageCategoryClassifier(trainingSet, bag);

confMatrix = evaluate(categoryClassifier, trainingSet);

confMatrix = evaluate(categoryClassifier, validationSet);

% Compute average accuracy
accuracy3 = mean(diag(confMatrix));



%% 

img = imread(fullfile('can.jpg'));
[labelIdx, scores] = predict(categoryClassifier, img);

% Display the string label
categoryClassifier.Labels(labelIdx)
scores

img = imread(fullfile('can2.jpg'));
[labelIdx, scores] = predict(categoryClassifier, img);

% Display the string label
categoryClassifier.Labels(labelIdx)
scores
img = imread(fullfile('air-kotak-drinho.png'));
[labelIdx, scores] = predict(categoryClassifier, img);

% Display the string label
categoryClassifier.Labels(labelIdx)
scores

img = imread(fullfile('cup.jpg'));
[labelIdx, scores] = predict(categoryClassifier, img);

% Display the string label
categoryClassifier.Labels(labelIdx)
scores

img = imread(fullfile('bottle.jpg'));
[labelIdx, scores] = predict(categoryClassifier, img);

% Display the string label
categoryClassifier.Labels(labelIdx)
scores

%%
% Creating Bag-Of-Features.
% -------------------------
% * Image category 1: bottle
% * Image category 2: can
% * Image category 3: cup
% * Image category 4: drink box
% * Selecting feature point locations using the Grid method.
% * Extracting SURF features from the selected feature point locations.
% ** The GridStep is [8 8] and the BlockWidth is [32 64 96 128].
% 
% * Extracting features from 800 images...done. Extracted 10949148 features.
% 
% * Keeping 80 percent of the strongest features from each category.
% 
% * Balancing the number of features across all image categories to improve clustering.
% ** Image category 1 has the least number of strongest features: 700176.
% ** Using the strongest 700176 features from each of the other image categories.
% 
% * Creating a 500 word visual vocabulary.
% * Number of levels: 1
% * Branching factor: 500
% * Number of clustering steps: 1
% 
% * [Step 1/1] Clustering vocabulary level 1.
% * Number of features          : 2800704
% * Number of clusters          : 500
% * Initializing cluster centers...100.00%.
% * Clustering...completed 18/100 iterations (~8.97 seconds/iteration)...converged in 18 iterations.
% 
% * Finished creating Bag-Of-Features
% 
% 
% Encoding images using Bag-Of-Features.
% --------------------------------------
% * Encoding an image...done.
% 
% Training an image category classifier for 4 categories.
% --------------------------------------------------------
% * Category 1: bottle
% * Category 2: can
% * Category 3: cup
% * Category 4: drink box
% 
% * Encoding features for 800 images...done.
% 
% * Finished training the category classifier. Use evaluate to test the classifier on a test set.
% 
% 
% Evaluating image category classifier for 4 categories.
% -------------------------------------------------------
% 
% * Category 1: bottle
% * Category 2: can
% * Category 3: cup
% * Category 4: drink box
% 
% * Evaluating 800 images...done.
% 
% * Finished evaluating all the test sets.
% 
% * The confusion matrix for this test set is:
% 
% 
%                            PREDICTED
% KNOWN        | bottle   can    cup    drink box   
% --------------------------------------------------
% bottle       | 0.72     0.07   0.14   0.07        
% can          | 0.12     0.69   0.12   0.07        
% cup          | 0.05     0.04   0.88   0.04        
% drink box    | 0.12     0.11   0.13   0.65        
% 
% * Average Accuracy is 0.73.
% 
% 
% Evaluating image category classifier for 4 categories.
% -------------------------------------------------------
% 
% * Category 1: bottle
% * Category 2: can
% * Category 3: cup
% * Category 4: drink box
% 
% * Evaluating 396 images...done.
% 
% * Finished evaluating all the test sets.
% 
% * The confusion matrix for this test set is:
% 
% 
%                            PREDICTED
% KNOWN        | bottle   can    cup    drink box   
% --------------------------------------------------
% bottle       | 0.56     0.13   0.22   0.09        
% can          | 0.20     0.47   0.17   0.16        
% cup          | 0.09     0.06   0.81   0.05        
% drink box    | 0.14     0.11   0.19   0.56        
% 
% * Average Accuracy is 0.60.

