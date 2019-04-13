%% DOCUMENTATION


% OBJECTIVES
% This code is used for extract TDD features based on "action recognition
% with trajectory-pooled deep-convolutional descriptors' paper

%% DEPENDENCIES
% 1. Caffe
% 2. ITF project
% 3. Dense flow
% 4. OpenCV 2.49

% All dependencies located in the same folder with my 'tdd folder'
% from each folder or you can refer to this links:
% [Improved Trajectories] https://github.com/wanglimin/improved_trajectory
% [Dense Flow](https://github.com/wanglimin/dense_flow)
% [parallel caffe toolbox](https://github.com/yjxiong/caffe). 
% ["Temporal net model (v2)"](http://mmlab.siat.ac.cn/tdd/temporal_v2.caffemodel) 

% Model and Configuration
% ["Spatial net model (v1)"](http://mmlab.siat.ac.cn/tdd/spatial.caffemodel) </br> 
% ["Spatial net prototxt (v1)"](http://mmlab.siat.ac.cn/tdd/spatial_cls.prototxt) </br>
% ["Temporal net model (v1)"](http://mmlab.siat.ac.cn/tdd/temporal.caffemodel) </br>
% ["Temporal net prototxt (v1)"](http://mmlab.siat.ac.cn/tdd/temporal_cls.prototxt)

% EXTRACT FLOW 
% Input     : video.avi, flow folder, ITF features
% Output    : TDD features including convolutional for both spatial and
% temporal domain


%%
clc
clear
close all

% Run another code in dependecy folder
cd D:\LAB\MATLAB\cvpr_paper\dependencies\TDD
run('main.m');
