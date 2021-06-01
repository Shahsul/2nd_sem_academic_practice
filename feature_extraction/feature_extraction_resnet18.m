close all;
clc;

%import the data

datalocation = fullfile('test_dataset');

imds = imageDatastore(datalocation, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

%find the image size
img = readimage(imds,1);
s=size(img);


%spilt the data for training and test
[train,test] = splitEachLabel(imds,0.8,'randomized');

%initialize the pretrained network
net = resnet18



inputSize = net.Layers(1).InputSize;

%set the default image size
augimdsTrain = augmentedImageDatastore(inputSize(1:2),train);
augimdsTest = augmentedImageDatastore(inputSize(1:2),test);


%get the feature representation
layer = 'pool5';
featuresTrain = activations(net,augimdsTrain,layer,'OutputAs','rows');
featuresTest = activations(net,augimdsTest,layer,'OutputAs','rows');

% >> whos featuresTrain   to visialize the features


%Extract the class labels from the training and test data.

YTrain = train.Labels;
YTest = test.Labels;

classifier = fitcecoc(featuresTrain,YTrain);

YPred = predict(classifier,featuresTest);


%tets
idx = [1 5 10 15];
figure
for i = 1:numel(idx)
    subplot(2,2,i)
    I = readimage(test,idx(i));
    label = YPred(idx(i));
    imshow(I)
    title(char(label))
end

%check accuracy
accuracy = mean(YPred == YTest)

%%train classifier on Shallower Features
layer = 'res3b_relu';
featuresTrain = activations(net,augimdsTrain,layer);
featuresTest = activations(net,augimdsTest,layer);

featuresTrain = squeeze(mean(featuresTrain,[1 2]))';
featuresTest = squeeze(mean(featuresTest,[1 2]))';

%% Train an SVM classifier on the shallower features. Calculate the test accuracy.
classifier = fitcecoc(featuresTrain,YTrain);
YPred = predict(classifier,featuresTest);
accuracy2 = mean(YPred == YTest)

