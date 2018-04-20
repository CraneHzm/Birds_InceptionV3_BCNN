% Copyright (c) Hu Zhiming 2018/4/11 JimmyHu@pku.edu.cn All Rights Reserved.

% preprocess the images: split the images into train, validation & test
% sets and augment the train data by randomly flipping, cropping & adding noise to
% the original train data.


load 'ImageInfo/train_test_split.txt';
load 'ImageInfo/image_class_labels.txt';

% make the train, validation & test directories to save the datasets.
mkdir('Train');
mkdir('Validation');
mkdir('Test');

% make the subdirectories for the 200 classes in 'Train/', 'Validation/' &
% 'Test/'.
for i = 1: 200
    dirname = num2str(i);
    dirname1 = ['Train/' dirname];
    dirname2 = ['Validation/' dirname];
    dirname3 = ['Test/' dirname];
    mkdir(dirname1);
    mkdir(dirname2);
    mkdir(dirname3);
end


% read the images and split them into train, validation & test datasets.
imagesfile = fopen('ImageInfo/images.txt', 'r');
% nline is the line's id in the file, i.e., the id of the corresponding image.
nline = 0;
% the number of gray images in the dataset.
numGray = 0;
while ~feof(imagesfile)
    
    % read a line in the file.
    tline = fgetl(imagesfile);
    nline = nline + 1;
    % fprintf('%d', nline);
    % fprintf('\n');
    
    % get the path of the images.
    str = regexp(tline, ' ', 'split');
    str = char(str(2));
    imagepath = ['images/' str];
    
    % read the image.
    image = imread(imagepath);
    % if the original image is a gray image, convert it to rgb format.
    if numel(size(image)) < 3
        image = cat(3, image, image, image);
        numGray = numGray + 1;
    end
    % imshow(image);
    
    % split the images into train, validation & test datasets.
    % if this image is splited to test dataset.
    if train_test_split(nline, 2) == 0
        class_label = image_class_labels(nline, 2);
        % the path of the image.
        imagepath = ['Test/', num2str(class_label), '/', num2str(nline), '.jpg'];
        % save the image.
        imwrite(image, imagepath);
    % if this image is splited to train dataset.
    % We randomly select 10% of the total train images as validation data
    % and the rest images are used as train data.
    else
        % the random flag.
        flag = rand;
        % the image is splited to train or validation dataset.
        if flag > 0.1
            class_label = image_class_labels(nline, 2);
            % the path of the image.
            imagepath = ['Train/', num2str(class_label), '/', num2str(nline), '.jpg'];
            % save the image.
            imwrite(image, imagepath);
        % the image is splited to  dataset.
        else
            class_label = image_class_labels(nline, 2);
            % the path of the image.
            imagepath = ['Validation/', num2str(class_label), '/', num2str(nline), '.jpg'];
            % save the image.
            imwrite(image, imagepath);
        end
    end
    
end

% close the file.
fclose(imagesfile);
fprintf('Number of Gray Images: ');
fprintf('%d', numGray);


% Augment the train data by randomly flipping, cropping & adding noise to the original train images.
% the path of the original train images.
trainPath = 'Train/';


% Randomly flip the original train images.
% make the directories that we need.
mkdir('TrainFlip');
for i = 1: 200
    dirname = num2str(i);
    dirname = ['TrainFlip/' dirname];
    mkdir(dirname);
end

% read all the images in 'Train/' and randomly flip them.
for i = 1: 200
    % the path of the corresponding class.
    classPath = [trainPath, num2str(i), '/'];
    flipPath = ['TrainFlip/', num2str(i), '/'];
    % list all the images in classPath.
    trainDir = dir([classPath, '*.jpg']);
    for j = 1: length(trainDir)
        % the random flag.
        flag = rand;
        if flag > 0.5
            % read all the images in this path.
            name = trainDir(j).name;
            image = imread([classPath, name]);
            % flip the original image.
            image2 = fliplr(image);
            path2=[flipPath, name];
            imwrite(image2, path2);
        end
    end
end


% Randomly crop the original train images.
% make the directories that we need.
mkdir('TrainCrop');
for i = 1: 200
    dirname = num2str(i);
    dirname = ['TrainCrop/' dirname];
    mkdir(dirname);
end

% read all the images in 'Train/' and randomly crop them.
for i = 1: 200
    % the path of the corresponding class.
    classPath = [trainPath, num2str(i), '/'];
    cropPath = ['TrainCrop/', num2str(i), '/'];
    % list all the images in classPath.
    trainDir = dir([classPath, '*.jpg']);
    for j = 1: length(trainDir)
        % the random flag.
        flag = rand;
        if flag > 0.5
            % read all the images in this path.
            name = trainDir(j).name;
            image = imread([classPath, name]);
            % resize the original image.
            image2 = imresize(image, 1.2);
            width = size(image, 2);
            height = size(image, 1);
            % crop the image.
            image2 = imcrop(image2, [width*0.1 height*0.1 width height]);
            name2=[cropPath name];
            imwrite(image2, name2);
        end
    end
end


% Randomly add noise to the original train images.
% make the directories that we need.
mkdir('TrainNoise');
for i = 1: 200
    dirname = num2str(i);
    dirname = ['TrainNoise/' dirname];
    mkdir(dirname);
end

% read all the images in 'Train/' and randomly add noise to them.
for i = 1: 200
    % the path of the corresponding class.
    classPath = [trainPath, num2str(i), '/'];
    noisePath = ['TrainNoise/', num2str(i), '/'];
    % list all the images in classPath.
    trainDir = dir([classPath, '*.jpg']);
    for j = 1: length(trainDir)
        % the random flag.
        flag = rand;
        if flag > 0.5
            % read all the images in this path.
            name = trainDir(j).name;
            image = imread([classPath, name]);
            % add noise to the original image.
            image2 = imnoise(image, 'salt & pepper', 0.02);
            name2=[noisePath name];
            imwrite(image2, name2);
        end
    end
end