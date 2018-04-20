% Copyright (c) Hu Zhiming 2018/4/20 JimmyHu@pku.edu.cn All Rights Reserved.

% Pick Up a small part of the Test Images for our UI test.

% the number of images to pick up in a class.
ImageNum = 4;

% make the directory and its subdirectories.
mkdir('TestPickUp');
for i = 1: 200
    dirname = num2str(i);
    dirname = ['TestPickUp/' dirname];
    mkdir(dirname);
end

% the path of the original test images.
testPath = 'Test/';

% read all the images in 'Test/' and pick up some images and save them in 'TestPickUp/'.
for i = 1: 200
    % the path of the corresponding class.
    classPath = [testPath, num2str(i), '/'];
    pickupPath = ['TestPickUp/', num2str(i), '/'];
    % list all the images in classPath.
    testDir = dir([classPath, '*.jpg']);
    % pick up images.
    for j = 1: ImageNum
        % read all the images in this path.
        name = testDir(j).name;
        image = imread([classPath, name]);
        path2=[pickupPath, name];
        imwrite(image, path2);
    end
end



