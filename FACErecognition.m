
numSubjects = 24; 
imageSize = [112, 92, 1]; 

layers = [
    imageInputLayer(imageSize)
    
    convolution2dLayer(5, 16, 'Padding', 'same')
    reluLayer()
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(5, 32, 'Padding', 'same')
    reluLayer()
    maxPooling2dLayer(2, 'Stride', 2)
    
    fullyConnectedLayer(numSubjects)
    softmaxLayer()
    classificationLayer()];

% Define training options
options = trainingOptions('adam', ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 10, ...
    'Shuffle', 'every-epoch', ...
    'InitialLearnRate', 1e-4, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% Training the CNN
net = trainNetwork(imdsTrain, layers, options);

% Test the trained network on the test set
YPred = classify(net, imdsTest);

% Calculate accuracy
accuracy = sum(YPred == imdsTest.Labels) / numel(imdsTest.Labels);
disp(['Recognition Accuracy: ' num2str(accuracy * 100) '%']);

