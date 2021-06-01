datalocation = fullfile('C:','Users','acer','Documents','MATLAB',...
    'Thesis_practice','test_dataset');

imds = imageDatastore(datalocation, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

labelCount = countEachLabel(imds);

%split the data for training and testing 
numTrainFiles = 900;
[train_set,validation] = splitEachLabel(imds,numTrainFiles,'randomize');

%select the pretrained network layers and set new criteria accordingly
net=alexnet

layers = [imageInputLayer([512,512,3])
net(2:end-3)
fullyConnectedLayer(3)     
softmaxLayer         
classificationLayer()
]

%set the training options
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',validation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

train_it = trainNetwork(train_set,layers,options);


%check the validation
YPred = classify(train_it,validation);
YValidation = imdsValidation.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation)
