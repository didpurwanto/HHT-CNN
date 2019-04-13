%% DOCUMENTATION


% OBJECTIVES
% Create POT representation

% DEPENDENCIES
% 1. func_pyramid.m

% EVALUATION SETTING
% Input     : nxm 4096x1 caffe feature;  
%             [hht_vector_1_2_10]
% Output    : POT feature representation for caffe feature
%             [caffe_pot_max, caffe_pot_sum, caffe_pot_grad1, caffe_pot_grad2]

clc
clear
close all

% Input
srcFiles = dir('C:\Users\didpurwanto\Documents\Dataset\JPL\CNN\*.mat');  
totalVideo = length(srcFiles);

% pyramid temporal filter's level 
level_pyramid = 4;

caffe_pot_max = [];
caffe_pot_sum = [];
caffe_pot_grad1 = [];
caffe_pot_grad2 = [];

for i = 1:totalVideo
    prog0 = sprintf('Processing (%d/%d)', i, totalVideo);
    disp(prog0);

    filename = strcat('C:\Users\didpurwanto\Documents\Dataset\JPL\CNN\', srcFiles(i).name);   
    name = srcFiles(i).name(1: end-4); 
    disp('     - load data ...');    
    load(filename);
    
    disp('     - do temporal pooling ...');
    
    [pot_grad1, pot_grad2, pot_max2, pot_sum2] = ...
        func_pyramid_pot(vector_sequence, level_pyramid);
   
    caffe_pot_max = horzcat(caffe_pot_max, pot_max2);    
    caffe_pot_sum = horzcat(caffe_pot_sum, pot_sum2);
    caffe_pot_grad1 = horzcat(caffe_pot_grad1, pot_grad1);
    caffe_pot_grad2 = horzcat(caffe_pot_grad2, pot_grad2);
    
end

save('pot_cafee_jpl.mat','caffe_pot_max','caffe_pot_sum','caffe_pot_grad1','caffe_pot_grad2');
