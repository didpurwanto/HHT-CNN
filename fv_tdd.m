%% DOCUMENTATION 


% OBJECTIVES
% To create feature representation using Fisher Vector

% FV
% Input     : Test with TDD features using
% Fisher Vector 
% Output    : New FV feature representation with number of k is 256 for
% train Gaussian Mixture Model

%%

clc
clear
close all

srcFiles = dir('C:\Users\didpurwanto\Documents\Dataset\DogCentric\TDD\*.mat');  
totalVideo = length(srcFiles);

k = 256;
data_s41 = []; data_s42 = []; data_s51 = []; data_s52 = [];
data_t31 = []; data_t32 = []; data_t41 = []; data_t42 = [];
fv_s41 = []; fv_s42 = []; fv_s51 = []; fv_s52 = [];
fv_t31 = []; fv_t32 = []; fv_t41 = []; fv_t42 = [];

disp('- Preprocessing ...');

for i = 1 : totalVideo
    % Dataset folder
    filename = strcat('C:\Users\didpurwanto\Documents\Dataset\DogCentric\TDD\',srcFiles(i).name);
    name = srcFiles(i).name(1: end-4);
    
    load(filename);

    tdd_s41 = tdd_feature_spatial_conv4_norm_1;
    tdd_s42 = tdd_feature_spatial_conv4_norm_2;
    tdd_s51 = tdd_feature_spatial_conv5_norm_1;
    tdd_s52 = tdd_feature_spatial_conv5_norm_2;
    tdd_t31 = tdd_feature_temporal_conv3_norm_1;
    tdd_t32 = tdd_feature_temporal_conv3_norm_2;
    tdd_t41 = tdd_feature_temporal_conv4_norm_1;
    tdd_t42 = tdd_feature_temporal_conv4_norm_2;
    
        
    progIndication3 = sprintf('     Processing %s length \t %d \t (%d / %d)', srcFiles(i).name, length(tdd_s41), i, totalVideo);       
    disp(progIndication3);            
    disp('     TDD imported!');
    
    % Create random big matrix data    
    [dim, num] = size(tdd_s41);
    batas = 2000;
    if num > batas
        data_s41 = horzcat(data_s41, tdd_s41(:,1:batas));
        data_s42 = horzcat(data_s42, tdd_s42(:,1:batas));
        data_s51 = horzcat(data_s51, tdd_s51(:,1:batas));
        data_s52 = horzcat(data_s52, tdd_s52(:,1:batas));
        data_t31 = horzcat(data_t31, tdd_t31(:,1:batas));
        data_t32 = horzcat(data_t32, tdd_t32(:,1:batas));
        data_t41 = horzcat(data_t41, tdd_t41(:,1:batas));
        data_t42 = horzcat(data_t42, tdd_t42(:,1:batas));
    else
        data_s41 = horzcat(data_s41, tdd_s41);
        data_s42 = horzcat(data_s42, tdd_s42);
        data_s51 = horzcat(data_s51, tdd_s51);
        data_s52 = horzcat(data_s52, tdd_s52);
        data_t31 = horzcat(data_t31, tdd_t31);
        data_t32 = horzcat(data_t32, tdd_t32);
        data_t41 = horzcat(data_t41, tdd_t41);
        data_t42 = horzcat(data_t42, tdd_t42);
    end
end

% create temp backup data
% save('data_tdd.mat', 'data_s41','data_s42','data_s51','data_s52','data_t31','data_t32','data_t41','data_t42');
% load('data_tdd.mat');

disp('     Preprocessing done!');


% PCA
disp('- PCA running ...');
[x,y] = size(data_s41);

tic;
coeff_s41 = pca(data_s41');
choose_s41 = coeff_s41(:, 1:x/2);
new_s41 = data_s41'*choose_s41;
toc;

coeff_s42 = pca(data_s42');
choose_s42 = coeff_s42(:, 1:x/2);
new_s42 = data_s42'*choose_s42;

coeff_s51 = pca(data_s51');
choose_s51 = coeff_s51(:, 1:x/2);
new_s51 = data_s51'*choose_s51;

coeff_s52 = pca(data_s52');
choose_s52 = coeff_s52(:, 1:x/2);
new_s52 = data_s52'*choose_s41;

coeff_t31 = pca(data_t31');
choose_t31 = coeff_t31(:, 1:x/2);
new_t31 = data_t31'*choose_t31;

coeff_t32 = pca(data_t32');
choose_t32 = coeff_t32(:, 1:x/2);
new_t32 = data_t32'*choose_t32;

coeff_t41 = pca(data_t41');
choose_t41 = coeff_t41(:, 1:x/2);
new_t41 = data_t41'*choose_t41;

coeff_t42 = pca(data_t42');
choose_t42 = coeff_t42(:, 1:x/2);
new_t42 = data_t42'*choose_t42;

disp('     PCA done!');

% Fisher Vector    
disp('- Fisher Vector running ....')
tic;
prog2 = sprintf('    Training GMM for S41 | numFeatures: %d dimension: %d', y, x/2);
disp(prog2);
[means_s41, covariances_s41, priors_s41] = vl_gmm(new_s41', k);
toc;

tic;
prog3 = sprintf('    Training GMM for S42 | numFeatures: %d dimension: %d', y, x/2);
disp(prog3);
[means_s42, covariances_s42, priors_s42] = vl_gmm(new_s42', k);
toc;

tic;
prog4 = sprintf('    Training GMM for S51 | numFeatures: %d dimension: %d', y, x/2);
disp(prog4);
[means_s51, covariances_s51, priors_s51] = vl_gmm(new_s51', k);
toc;

tic;
prog5 = sprintf('    Training GMM for S52 | numFeatures: %d dimension: %d', y, x/2);
disp(prog5);
[means_s52, covariances_s52, priors_s52] = vl_gmm(new_s52', k);
toc;

tic;
prog6 = sprintf('    Training GMM for T31 | numFeatures: %d dimension: %d', y, x/2);
disp(prog6);
[means_t31, covariances_t31, priors_t31] = vl_gmm(new_t31', k);
toc;

tic;
prog7 = sprintf('    Training GMM for T32 | numFeatures: %d dimension: %d', y, x/2);
disp(prog7);
[means_t32, covariances_t32, priors_t32] = vl_gmm(new_t32', k);
toc;

tic;
prog8 = sprintf('    Training GMM for T41 | numFeatures: %d dimension: %d', y, x/2);
disp(prog8);
[means_t41, covariances_t41, priors_t41] = vl_gmm(new_t41', k);
toc;

tic;
prog9 = sprintf('    Training GMM for T42 | numFeatures: %d dimension: %d', y, x/2);
disp(prog9);
[means_t42, covariances_t42, priors_t42] = vl_gmm(new_t42', k);
toc;

% Encode FV
for ii = 1 : totalVideo
   filename = strcat('C:\Users\didpurwanto\Documents\Dataset\DogCentric\TDD\',srcFiles(ii).name);
   name = srcFiles(ii).name(1: end-4); 
   load(filename);
   
%    Import IDT features    
   tdd_s41 = tdd_feature_spatial_conv4_norm_1;
   tdd_s42 = tdd_feature_spatial_conv4_norm_2;
   tdd_s51 = tdd_feature_spatial_conv5_norm_1;
   tdd_s52 = tdd_feature_spatial_conv5_norm_2;
   tdd_t31 = tdd_feature_temporal_conv3_norm_1;
   tdd_t32 = tdd_feature_temporal_conv3_norm_2;
   tdd_t41 = tdd_feature_temporal_conv4_norm_1;
   tdd_t42 = tdd_feature_temporal_conv4_norm_2;
      
   en_s41 = tdd_s41'*choose_s41;
   en_s42 = tdd_s42'*choose_s42;
   en_s51 = tdd_s52'*choose_s51;
   en_s52 = tdd_s52'*choose_s52;   
   en_t31 = tdd_t31'*choose_t31;
   en_t32 = tdd_t32'*choose_t32;
   en_t41 = tdd_t41'*choose_t41;
   en_t42 = tdd_t42'*choose_t42;
   
   prog10 = sprintf('     Encode %s (%d/%d)',srcFiles(ii).name, ii, totalVideo);
   disp(prog10);
   
   encoding_s41_a = vl_fisher(en_s41', means_s41, covariances_s41, priors_s41);
   encoding_s42_a = vl_fisher(en_s42', means_s42, covariances_s42, priors_s42);
   encoding_s51_a = vl_fisher(en_s51', means_s51, covariances_s51, priors_s51);
   encoding_s52_a = vl_fisher(en_s52', means_s52, covariances_s52, priors_s52);
   
   encoding_t31_a = vl_fisher(en_t31', means_t31, covariances_t31, priors_t31);
   encoding_t32_a = vl_fisher(en_t32', means_t32, covariances_t32, priors_t32);
   encoding_t41_a = vl_fisher(en_t41', means_t41, covariances_t41, priors_t41);
   encoding_t42_a = vl_fisher(en_t42', means_t42, covariances_t42, priors_t42);
   
   % Power Normalization
   encoding_s41 = power(encoding_s41_a,2);
   encoding_s42 = power(encoding_s42_a,2);
   encoding_s51 = power(encoding_s51_a,2);
   encoding_s52 = power(encoding_s52_a,2);
   
   encoding_t31 = power(encoding_t31_a,2);
   encoding_t32 = power(encoding_t32_a,2);
   encoding_t41 = power(encoding_t41_a,2);
   encoding_t42 = power(encoding_t42_a,2);
      
   % L2 Normalization
   norm_s41 = encoding_s41'/norm(encoding_s41');
   norm_s42 = encoding_s42'/norm(encoding_s42');
   norm_s51 = encoding_s51'/norm(encoding_s51');
   norm_s52 = encoding_s52'/norm(encoding_s52');
   norm_t31 = encoding_t31'/norm(encoding_t31');
   norm_t32 = encoding_t32'/norm(encoding_t32');
   norm_t41 = encoding_t41'/norm(encoding_t41');
   norm_t42 = encoding_t42'/norm(encoding_t42');
   
   % Concanate   
   fv_s41 = vertcat(fv_s41, norm_s41);
   fv_s42 = vertcat(fv_s42, norm_s42);
   fv_s51 = vertcat(fv_s51, norm_s51);
   fv_s52 = vertcat(fv_s52, norm_s52);
   fv_t31 = vertcat(fv_t31, norm_t31);
   fv_t32 = vertcat(fv_t32, norm_t32);
   fv_t41 = vertcat(fv_t41, norm_t41);
   fv_t42 = vertcat(fv_t42, norm_t42);
end
disp('     Encoding FV done!')


save('fv_tdd.mat', 'fv_s41', 'fv_s42','fv_s51','fv_s52','fv_t31','fv_t32','fv_t41','fv_t42');
disp('- DONE!')
