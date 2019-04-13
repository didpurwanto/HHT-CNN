%% DOCUMENTATION


% OBJECTIVES
% Create Caffe feature

% DEPENDENCIES
% 1. Caffe architecture

% EVALUATION SETTING
% Input     : videos (.mp4)  
% Output    : caffe feature 

clc
clear
close


grad_pot_caffe = [];
max_pot_caffe = [];
sum_pot_caffe = [];
grad_pot_caffe2 = [];
max_pot_caffe2 = [];
sum_pot_caffe2 = [];

% CNN
caffe.set_mode_cpu();
% load mean
load('D:\LAB\Visud\caffe3\caffe-windows\matlab\+caffe\imagenet\ilsvrc_2012_mean.mat'); 
% create net and load weights
net = caffe.Net('D:\LAB\Visud\caffe3\caffe-windows\models\bvlc_alexnet\deploy.prototxt', ...
    'D:\LAB\Visud\caffe3\caffe-windows\models\bvlc_alexnet\bvlc_alexnet.caffemodel', 'test');
net.blobs('data').reshape([227 227 3 1]);

% Input
srcFiles = dir('C:\Users\didpurwanto\Documents\Dataset\JPL\movie\*.avi');  
totalVideo = length(srcFiles);

for i = 1 : totalVideo
    vector_sequence = [];
    prog0 = sprintf('Processing (%d/%d)', i, totalVideo);
    disp(prog0);
    
    feature = [];
    filename = strcat('C:\Users\didpurwanto\Documents\Dataset\JPL\movie\', srcFiles(i).name);   
    name = srcFiles(i).name(1: end-4);            
    mov = VideoReader(filename);

    opFolder = fullfile(cd, name);
    if ~exist(opFolder, 'dir')
        mkdir(opFolder);
    end
    
    prog1 = sprintf('     - Writing \t %d \tframe and extract features from fc7 ... ', mov.NumberOfFrames);
    disp(prog1);

    for t = 1 : mov.NumberOfFrames
        currFrame = read(mov, t);   
        opBaseFileName = sprintf('%3.3d.tif', t);
        opFullFileName = fullfile(opFolder, opBaseFileName);
        imwrite(currFrame, opFullFileName, 'tif');   

        % load image and forward (do this on each image). Then load image in caffe's data format             
        im_data = caffe.io.load_image(opFullFileName); 
        im_data = imresize(im_data, [256 256]); 
        im_data = im_data(15:241, 15:241, :); 
        res = net.forward({im_data}); 
        feature = net.blobs('fc7').get_data();
        
        vector_sequence = horzcat(vector_sequence, feature);      
    end
    
    dd = sprintf('%s.mat',mov.name);
    save(dd,'vector_sequence');
end