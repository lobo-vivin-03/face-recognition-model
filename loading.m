
datasetPath = 'C:\Users\ASUS\OneDrive\Desktop\Internship2023\project\test'; 

imageSize = [112, 92]; % Size of the images in the dataset

imds = imageDatastore(datasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames', 'ReadFcn', @(filename)imresize(imread(filename), imageSize));

[imdsTrain, imdsTest] = splitEachLabel(imds, 0.8, 'randomized');
