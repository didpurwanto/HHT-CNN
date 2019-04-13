clc
clear
close all

% Input
srcFiles = dir('C:\Users\didpurwanto\Documents\Dataset\JPL\TDD\*.mat');  
totalVideo = length(srcFiles);
pyramid_level = 4;
batas = 10000;

OPTIONS.MAXITERATIONS = 200;
OPTIONS.MAXMODES = 10;

for i = 1:totalVideo
    prog0 = sprintf('Preprocessing ............................................. (%d/%d)', i, totalVideo);
    disp(prog0);
    
    filename = strcat('C:\Users\didpurwanto\Documents\Dataset\JPL\TDD\', srcFiles(i).name);   
    name = srcFiles(i).name(1: end-4); 
    
    disp('    load data!');     
    load(filename);      
    
    data = tdd_feature_spatial_conv5_norm_1;
    
    [x,y] = size(data);
    for j = 1:x
        prog1 = sprintf('    emd...........................................%d\t %d/%d - %d', i, j, x, y);
        disp(prog1);
        if y>=batas
            % tmp_s4{i,j} = func_imf(data_s4(j,1:batas),OPTIONS);
            % tmp_s5{i,j} = func_imf(data_s5(j,1:batas));
            tmp_data{i,j} = func_imf(data(j,1:batas),OPTIONS);
            % tmp_t4{i,j} = func_imf(data_t4(j,1:batas));
        else
            % tmp_s4{i,j} = func_imf(data_s4(j,:),OPTIONS);
            % tmp_s5{i,j} = func_imf(data_s5(j,:));
            tmp_data{i,j} = func_imf(data(j,:),OPTIONS);
            % tmp_t4{i,j} = func_imf(data_t4(j,:));
        end
    end
   
    imf_data = tmp_data(i,:);
    
    save_name = sprintf('s51_%s',name);
    save(save_name,'-v7.3','imf_data');
end