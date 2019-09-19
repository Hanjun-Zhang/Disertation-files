doTrainingAndEval = true;
if ~doTrainingAndEval && ~exist('fasterRCNNResNet50VehicleExample.mat','file')
    % Download pretrained detector.
    disp('Downloading pretrained detector (118 MB)...');
    pretrainedURL = 'https://www.mathworks.com/supportfiles/vision/data/fasterRCNNResNet50VehicleExample.mat';
    websave('fasterRCNNResNet50VehicleExample.mat',pretrainedURL);
end

data = load('faceDatasetGroundTruth_all_head.mat');
faceDataset = data.faceDataset;

faceDataset(1:4,:)


% Add the fullpath to the local vehicle data folder.
%faceDataset.imageFilename = fullfile(pwd,faceDataset.imageFilename);

% Read one of the images.
I = imread(faceDataset.imageFilename{9});

% Insert the ROI labels.
I = insertShape(I,'Rectangle',faceDataset.face{9});

% Resize and display image.
I = imresize(I,3);
imshow(I)



% Set random seed to ensure example training reproducibility.
rng(0);

% Randomly split data into a training and test set.
shuffledIndices = randperm(height(faceDataset));
idx = floor(0.6 * length(shuffledIndices) );
trainingData = faceDataset(shuffledIndices(1:idx),:);
testData = faceDataset(shuffledIndices(idx+1:end),:);


% Options for step 1.
options = trainingOptions('sgdm', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 1, ...
    'InitialLearnRate', 1e-4, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.5, ...
    'LearnRateDropPeriod',1, ...
    'CheckpointPath', 'D:\matlabcode\checkpoint', ...
    'ExecutionEnvironment','cpu', ...
    'Shuffle','every-epoch');    

if doTrainingAndEval
    
    % Train Faster R-CNN detector.
    %  * Use 'resnet50' as the feature extraction network. 
    %  * Adjust the NegativeOverlapRange and PositiveOverlapRange to ensure
    %    training samples tightly overlap with ground truth.
    [detector, info] = trainFasterRCNNObjectDetector(trainingData, 'resnet50', options, ...
        'NegativeOverlapRange', [0 0.3], ...
        'PositiveOverlapRange', [0.7 1]);
else
    % Load pretrained detector for the example.
    pretrained = load('faster_rcnn_detector.mat');
    detector = pretrained.detector;
end

% Read a test image.
I = imread(testData.imageFilename{1});

% Run the detector.
[bboxes,scores] = detect(detector,I);

% Annotate detections in the image.
I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
figure
imshow(I)

if doTrainingAndEval
    % Create a table to hold the bounding boxes, scores, and labels output by
    % the detector.
    numImages = height(testData);
    results = table('Size',[numImages 3],...
        'VariableTypes',{'cell','cell','cell'},...
        'VariableNames',{'Boxes','Scores','Labels'});
    
    % Run detector on each image in the test set and collect results.
    for i = 1:numImages
        
        % Read the image.
        I = imread(testData.imageFilename{i});
        
        % Run the detector.
        [bboxes, scores, labels] = detect(detector, I);
        
        % Collect the results.
        % Collect the results.
        results.Boxes{i} = bboxes;
        results.Scores{i} = scores;
        results.Labels{i} = labels;
    end
else
    % Load pretrained detector for the example.
    pretrained = load('fasterRCNNResNet50VehicleExample.mat');
    results = pretrained.results;
end
    

% Extract expected bounding box locations from test data.
expectedResults = testData(:, 2:end);

% Evaluate the object detector using Average Precision metric.
[ap, recall, precision] = evaluateDetectionPrecision(results, expectedResults);

% Plot precision/recall curve
figure
plot(recall,precision)
xlabel('Recall')
ylabel('Precision')
grid on
title(sprintf('Average Precision = %.2f', ap))
