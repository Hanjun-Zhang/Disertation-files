doTraining = true;
if ~doTraining && ~exist('yolov2ResNet50VehicleExample.mat','file')
    % Download pretrained detector.
    disp('Downloading pretrained detector (98 MB)...');
    pretrainedURL = 'https://www.mathworks.com/supportfiles/vision/data/yolov2ResNet50VehicleExample.mat';
    websave('yolov2ResNet50VehicleExample.mat',pretrainedURL);
end


%data = load('faceDatasetGroundTruth_all_head.mat');
data = load('FDDB_2.mat');
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
idx = floor(0.7 * length(shuffledIndices) );
trainingData = faceDataset(shuffledIndices(1:idx),:);
testData = faceDataset(shuffledIndices(idx+1:end),:);


% Define the image input size.
imageSize = [224 224 3];

% Define the number of object classes to detect.
numClasses = width(faceDataset)-1;

anchorBoxes = [
    150 150
    120 120
    100 100
    90 90
    70 70
    50 50
    30 30
];

% Load a pretrained ResNet-50.
baseNetwork = resnet50;

% Specify the feature extraction layer.
featureLayer = 'activation_40_relu';

% Create the YOLO v2 object detection network. 
lgraph = yolov2Layers(imageSize,numClasses,anchorBoxes,baseNetwork,featureLayer);


if doTraining
    
    % Configure the training options. 
    %  * Lower the learning rate to 1e-3 to stabilize training. 
    %  * Set CheckpointPath to save detector checkpoints to a temporary
    %    location. If training is interrupted due to a system failure or
    %    power outage, you can resume training from the saved checkpoint.
    options = trainingOptions('sgdm', ...
        'MiniBatchSize', 8, ....
        'InitialLearnRate',1e-3, ...
        'LearnRateSchedule','piecewise', ...
        'LearnRateDropFactor',0.5, ...
        'LearnRateDropPeriod',2, ...
        'MaxEpochs',10,...
        'CheckpointPath', 'D:\matlabcode\checkpoint', ...
        'ExecutionEnvironment','gpu', ...
        'Shuffle','every-epoch');    
    
    % Train YOLO v2 detector.
    [detector,info] = trainYOLOv2ObjectDetector(trainingData,lgraph,options);
else
    % Load pretrained detector for the example.
    pretrained = load('faster_rcnn_detector.mat');
    %pretrained = load('FDDB_5_detector.mat');
    detector = pretrained.detector;
end





I = imread(testData.imageFilename{1});
%I = imresize(I,[224 224]);
% Run the detector.
[bboxes,scores] = detect(detector,I);

% Annotate detections in the image.
I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
imshow(I)


% Create a table to hold the bounding boxes, scores, and labels output by
% the detector. 
numImages = height(testData);
results = table('Size',[numImages 3],...
    'VariableTypes',{'cell','cell','cell'},...
    'VariableNames',{'Boxes','Scores','Labels'});

% Run detector on each image in the test set and collect results.
for i = 1:numImages
    i
    % Read the image.
    I = imread(testData.imageFilename{i});
    %I = imresize(I,[224 224]);
    % Run the detector.
    [bboxes,scores,labels] = detect(detector,I);
   
    % Collect the results.
    results.Boxes{i} = bboxes;
    results.Scores{i} = scores;
    results.Labels{i} = labels;
end

% Extract expected bounding box locations from test data.
expectedResults = testData(:, 2:end);

% Evaluate the object detector using average precision metric.
[ap, recall, precision] = evaluateDetectionPrecision(results, expectedResults);

% Plot precision/recall curve
plot(recall,precision)
xlabel('Recall')
ylabel('Precision')
grid on
title(sprintf('Average Precision = %.2f', ap))
