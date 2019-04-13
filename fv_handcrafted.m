% DOCUMENTATION 

% 
% OBJECTIVES
% To create feature representation using Fisher Vector
% 
% FV
% Input     : Test with Handcrafted features using
% Fisher Vector 
% Output    : New FV feature representation with number of k is 256 for
% train Gaussian Mixture Model

%
clc
clear
close all

srcFiles = dir('C:\Users\didpurwanto\Documents\Dataset\JPL\IDT\*.bin');  
totalVideo = length(srcFiles);

k = 256;
data_hog = []; data_hof = []; data_mbh = []; data_tra = [];
fv_hog = []; fv_hof = []; fv_mbh = []; fv_tra = [];

disp('- Preprocessing ...');

for i = 1 : totalVideo
    % Dataset folder
    filename = strcat('C:\Users\didpurwanto\Documents\Dataset\JPL\IDT\',srcFiles(i).name);
    name = srcFiles(i).name(1: end-4);

    % Import IDT features    
    IDT = func_import_idt([name, '.bin'],15);    
    tra = IDT.tra;
    hog = IDT.hog;
    hof = IDT.hof;    
    mbhx = IDT.mbhx;
    mbhy = IDT.mbhy;
    mbh = vertcat(mbhx, mbhy);
    
    [dim, num] = size(hog);
    
    progIndication3 = sprintf('     Processing %s length \t %d \t (%d / %d)', srcFiles(i).name, length(mbh), i, totalVideo);       
    disp(progIndication3);            
    cd D:\LAB\MATLAB\paper\code
    
    % Create random 2560 data    
    
    batas = 3047;
    if num > batas
        data_tra = horzcat(data_tra, tra(:,1:batas));
        data_hog = horzcat(data_hog, hog(:,1:batas));
        data_hof = horzcat(data_hof, hof(:,1:batas));
        data_mbh = horzcat(data_mbh, mbh(:,1:batas));
    
    else
        data_tra = horzcat(data_tra, tra);
        data_hog = horzcat(data_hog, hog);
        data_hof = horzcat(data_hof, hof);
        data_mbh = horzcat(data_mbh, mbh);    
    end
end
disp('     Preprocessing done!');

% PCA
disp('- PCA running ...');
[x0,y0] = size(data_tra);
[x1,y1] = size(data_hog);
[x2,y2] = size(data_hof);
[x3,y3] = size(data_mbh);

coeff_tra = pca(data_tra');
choose_tra = coeff_tra(:, 1:x0/2);
new_tra = data_tra'*choose_tra;
new_tra = new_tra';

coeff_hog = pca(data_hog');
choose_hog = coeff_hog(:, 1:x1/2);
new_hog = data_hog'*choose_hog;
new_hog = new_hog';

coeff_hof = pca(data_hof');
choose_hof = coeff_hof(:, 1:x2/2);
new_hof = data_hof'*choose_hof;
new_hof = new_hof';

coeff_mbh = pca(data_mbh');
choose_mbh = coeff_mbh(:, 1:x3/2);
new_mbh = data_mbh'*choose_mbh;
new_mbh = new_mbh';
disp('     PCA done!');

% Fisher Vector    
disp('- Fisher Vector running ....')
tic;
prog12 = sprintf('    Training GMM for HOG | numFeatures: %d dimension: %d', y0, x0/2);
disp(prog12);
[means_tra, covariances_tra, priors_tra] = vl_gmm(new_tra, k);
toc;
tic;
prog2 = sprintf('    Training GMM for HOG | numFeatures: %d dimension: %d', y1, x1/2);
disp(prog2);
[means_hog, covariances_hog, priors_hog] = vl_gmm(new_hog, k);
toc;
tic;
prog3 = sprintf('    Training GMM for HOF | numFeatures: %d dimension: %d', y2, x2/2);
disp(prog3);
[means_hof, covariances_hof, priors_hof] = vl_gmm(new_hof, k);
toc;
tic;
prog4 = sprintf('    Training GMM for HOG | numFeatures: %d dimension: %d', y3, x3/2);
disp(prog4);
[means_mbh, covariances_mbh, priors_mbh] = vl_gmm(new_mbh, k);
disp('     GMM Trained!');
toc;

% Encode FV
for ii = 1 : totalVideo
   filename = strcat('C:\Users\didpurwanto\Documents\Dataset\JPL\IDT\',srcFiles(ii).name);
   name = srcFiles(ii).name(1: end-4); 
   
   % Import IDT features    
   IDT = func_import_idt([name, '.bin'],15);
   tra = IDT.tra;
   hog = IDT.hog;
   hof = IDT.hof;    
   mbhx = IDT.mbhx;
   mbhy = IDT.mbhy;
   mbh = vertcat(mbhx, mbhy);
   
   
   
   cd D:\LAB\MATLAB\paper\code
   
   
   
   en_tra = tra'*choose_tra;
   en_tra = en_tra';
   en_hog = hog'*choose_hog;
   en_hog = en_hog';
   en_hof = hof'*choose_hof;
   en_hof = en_hof';
   en_mbh = mbh'*choose_mbh;
   en_mbh = en_mbh';
   
   prog3 = sprintf('     Encode %s (%d/%d)',srcFiles(ii).name, ii, totalVideo);
   disp(prog3);
   
   encoding_tra = vl_fisher(en_tra, means_tra, covariances_tra, priors_tra);
   encoding_hog = vl_fisher(en_hog, means_hog, covariances_hog, priors_hog);
   encoding_hof = vl_fisher(en_hof, means_hof, covariances_hof, priors_hof);
   encoding_mbh = vl_fisher(en_mbh, means_mbh, covariances_mbh, priors_mbh); 
   
   % L2 Normalization
   norm_tra = encoding_tra'/norm(encoding_tra',2);
   norm_hog = encoding_hog'/norm(encoding_hog',2);
   norm_hof = encoding_hof'/norm(encoding_hof',2);
   norm_mbh = encoding_mbh'/norm(encoding_mbh',2);
   
   % Concanate   
   fv_tra = vertcat(fv_tra, norm_tra);
   fv_hog = vertcat(fv_hog, norm_hog);
   fv_hof = vertcat(fv_hof, norm_hof);
   fv_mbh = vertcat(fv_mbh, norm_mbh);
end
disp('     Encoding FV done!')


save('fv_handcrafted.mat', 'fv_tra','fv_hog', 'fv_hof','fv_mbh');
disp('- DONE!')