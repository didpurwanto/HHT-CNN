clc
clear 
close all

%% HELP
% This code is used for rename movie file


%% SEARCH VIDEO
srcFiles = dir('/Users/didpurwanto/Documents/dataku/*.mov');  
totalVideo = length(srcFiles);


%% RENAME FILE
% This part for rename file to help programmer more understand the
% structure of video (save 10 first digit)
% Input     : Ball_hime_17263.mov 
% Output    : Ball_hime_1.mov
cd /Users/didpurwanto/Documents/dataku/
for i = 3 : totalVideo
    progIndication = sprintf('Processing file %s', srcFiles(i).name);
    disp(progIndication);    
    
    filename = strcat('/Users/didpurwanto/Documents/dataku/',srcFiles(i).name);
    old_name = srcFiles(i).name;    
    
    temp_new_name = srcFiles(i).name(1: end-16);   
    new_name = sprintf('%s.mov',temp_new_name);
    
    movefile(old_name,new_name);
    
end