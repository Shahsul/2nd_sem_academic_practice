close all;clc;

%link to the dataset folder

datalocation = fullfile('test_dataset');

imds = imageDatastore(datalocation, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

labelCount = countEachLabel(imds);


%get the image size
img = readimage(imds,1);
size(img);

%split the data for training and testing 
numTrainFiles = 900;
[train_set,validation] = splitEachLabel(imds,numTrainFiles,'randomize');


%design the CNN architecture
layers = [
    imageInputLayer([512   512     3])
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(3)
    softmaxLayer
    classificationLayer];


%setting the tarining parameters

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',validation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

%train the network
net = trainNetwork(train_set,layers,options);


%check the validation
YPred = classify(net,validation);
YValidation = imdsValidation.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation)
