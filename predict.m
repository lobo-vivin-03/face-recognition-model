userinput = imread("C:\Users\ASUS\OneDrive\Desktop\Internship2023\project\vivin.jpg");

% Convert to grayscale if needed image is in RGB
if size(userinput, 3) == 3
    userinput = rgb2gray(userinput);
end

% Resize to match the network input size
userinput = imresize(userinput, [112, 92]);

% Perform face recognition on the User input image
predictedLabel = classify(net, userinput);

% Find all images of the recognized person in the training set
matchingIndices = find(imdsTrain.Labels == predictedLabel);

% Display the recognized image and 8 matching images in subplots
figure;

% Display the user input image
subplot(1, 9, 1);
imshow(userinput);
title('USER INPUT IMAGE');

% Display the first 8 matching images
for i = 1:min(8, numel(matchingIndices))
    subplot(1, 9, i + 1);
    matchingImage = readimage(imdsTrain, matchingIndices(i));
    imshow(matchingImage);
    title(['Matching Image ' num2str(i)]);
end
