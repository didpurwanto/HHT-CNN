%% DOCUMENTATION


% OBJECTIVES
% We followed the standard evaluation
% setting of the DogCentric dataset: we performed repeated
% random training/testing splits 100 times, and averaged the
% performance. We randomly selected half of videos per activity
% class as training videos, and used the others for the
% testing. If the number of total videos per class is odd, we
% add one more video

% DEPENDENCIES
% 1. LIBSVM plus chiSquare
% File already stored in Dependencies Folder. Just add thats folder
% to your MATLAB system path

% EVALUATION SETTING
% Input     : features representation 
% Output    : 100 separated features for training and testing. 

%%
clc
clear

% load data
disp('import data');
load ..\data\dogcentric\feature_representation\pot_caffe_lv4
% load ..\data\dogcentric\feature_representation\hht_1_5\fv_tdd
% load ..\data\dogcentric\feature_representation\hht_1_5\darwin_cnn
% load ..\data\dogcentric\feature_representation\hht_1_5\fv_handcrafted
% load ..\data\dogcentric\feature_representation\hht_1_5\fv_idt_dogcentric
load ..\data\dogcentric\feature_representation\label_pot
load ..\data\dogcentric\feature_representation\label_pot2
% load ..\data\dogcentric\feature_representation\hht_1_5\spatial_s51_hht_1_5
% 
% do normalization for each descriptor
disp('do normalization');
norm_type = 'L2';

% Darwin TVMV
% data9b = darwin_tvmv_backward;
% data9b_power = power(data9b,2);
% data9b_norm = func_norm(data9b_power,2),norm_type);
% data9f = darwin_tvmv_forward;
% data9f_power = power(data9f,2);
% data9f_norm = func_norm(data9f_power,2),norm_type);

% Darwin MA
% data10b = darwin_ma_backward;
% data10b_power = power(data10b,2);
% data10b_norm = func_norm(data10b_power,2),norm_type);
% data10f = darwin_ma_forward;
% data10f_power = power(data10f,2);
% data10f_norm = func_norm(data10f_power,2),norm_type);

% POT
data_tmp11 = double(caffe_pot_sum)';
data11 = func_norm(data_tmp11, norm_type);
data_tmp12 = caffe_pot_grad2';
data12 = func_norm(data_tmp12, norm_type);

% IDT
% data13 = double(fv_mbh);
% data14 = double(fv_tra);
% data15 = double(fv_hog);
% data16 = double(fv_hof);

% HHT
% data171 = func_norm(power(hht_final_mean,2),norm_type);
% data191 = func_norm(power(hht_final_centroid_spectral,2),norm_type);
% data231 = func_norm(power(hht_final_mean_instan,2),norm_type);
% data251 = func_norm(power(hht_final_entropy_energy,2),norm_type);
% 
% load ..\data\dogcentric\feature_representation\hht_1_5\spatial_s52_hht_1_5
% data172 = func_norm(power(hht_final_mean,2),norm_type);
% data192 = func_norm(power(hht_final_centroid_spectral,2),norm_type);
% data232 = func_norm(power(hht_final_mean_instan,2),norm_type);
% data252 = func_norm(power(hht_final_entropy_energy,2),norm_type);
% 
% load ..\data\dogcentric\feature_representation\hht_1_5\temporal_t41_hht_1_5
% data173 = func_norm(power(hht_final_mean,2),norm_type);
% data193 = func_norm(power(hht_final_centroid_spectral,2),norm_type);
% data233 = func_norm(power(hht_final_mean_instan,2),norm_type);
% data253 = func_norm(power(hht_final_entropy_energy,2),norm_type);
% 
% load ..\data\dogcentric\feature_representation\hht_1_5\temporal_t42_hht_1_5
% data174 = func_norm(power(hht_final_mean,2),norm_type);
% data194 = func_norm(power(hht_final_centroid_spectral,2),norm_type);
% data234 = func_norm(power(hht_final_mean_instan,2),norm_type);
% data254 = func_norm(power(hht_final_entropy_energy,2),norm_type);

disp('combine data');
% choose data
% data_itf = [data14 data15 data13];
data_pot = [data11 data12];
data = data_pot;

% data_hht = [entropy_energy                spectral_centroid_freq          mean_instan                     mean_analytic]
% data_hht1 = [data251 data252 data253 data254];
% data_hht2 = [data191 data192 data193 data194];
% data_hht3 = [data231 data232 data233 data234];
% data_hht4 = [data171 data172 data173 data174];


% data_norm1 = func_norm(data_hht1,2),norm_type);
% data_norm2 = func_norm(data_hht2,2),norm_type);
% data_norm3 = func_norm(data_hht3,2),norm_type);
% data_norm4 = func_norm(data_hht4,2),norm_type);

% data_darwin = [data10b_norm data10f_norm];
% data = [data_hht1 data_hht2 data_hht3 data_hht4];
% data = [data_norm1 data_norm2 data_norm3 data_norm4];


[x,y] = size(data);

% randoming
label = label';
training_label = label2';

cls1 = 1:14;
cls2 = 15:40;
cls3 = 41:50;
cls4 = 51:76;
cls5 = 77:98;
cls6 = 99:116;
cls7 = 117:138;
cls8 = 139:158;
cls9 = 159:186;
cls10 = 187:212;
ori = 1:212;

training_set = [];
testing_set = [];
result = [];
index_train = [];
index_test = [];

disp('do randoming video');
for i = 1:100
    % random for training and testing
    index1 = randperm(numel(cls1));
    nums1 = 1:length(cls1)/2;
    c1 = cls1(index1);
    train1 = c1(1:length(cls1)/2);
    test1 = c1((length(cls1)/2+1):length(cls1));
    
    index2 = randperm(numel(cls2));
    nums2 = 1:length(cls2)/2;
    c2 = cls2(index2);
    train2 = c2(1:length(cls2)/2);
    test2 = c2(length(cls2)/2+1:length(cls2));
    
    index3 = randperm(numel(cls3));
    nums3 = 1:length(cls3)/2;
    c3 = cls3(index3);
    train3 = c3(1:length(cls3)/2);
    test3 = c3(length(cls3)/2+1:length(cls3));
    
    index4 = randperm(numel(cls4));
    nums4 = 1:length(cls4)/2;
    c4 = cls4(index4);
    train4 = c4(1:length(cls4)/2);
    test4 = c4(length(cls4)/2+1:length(cls4));
    
    index5 = randperm(numel(cls5));
    nums5 = 1:length(cls5)/2;
    c5 = cls5(index5);
    train5 = c5(1:length(cls5)/2);
    test5 = c5(length(cls5)/2+1:length(cls5));
    
    index6 = randperm(numel(cls6));
    nums6 = 1:length(cls6)/2;
    c6 = cls6(index6);
    train6 = c6(1:length(cls6)/2);
    test6 = c6(length(cls6)/2+1:length(cls6));
    
    index7 = randperm(numel(cls7));
    nums7 = 1:length(cls7)/2;
    c7 = cls7(index7);
    train7 = c7(1:length(cls7)/2);
    test7 = c7(length(cls7)/2+1:length(cls7));
    
    index8 = randperm(numel(cls8));
    nums8 = 1:length(cls8)/2;
    c8 = cls8(index8);
    train8 = c8(1:length(cls8)/2);
    test8 = c8(length(cls8)/2+1:length(cls8));
    
    index9 = randperm(numel(cls9));
    nums9 = 1:length(cls9)/2;
    c9 = cls9(index9);
    train9 = c9(1:length(cls9)/2);
    test9 = c9(length(cls9)/2+1:length(cls9));
    
    index10 = randperm(numel(cls10));
    nums10 = 1:length(cls10)/2;
    c10 = cls10(index10);
    train10 = c10(1:length(cls10)/2);
    test10 = c10(length(cls10)/2+1:length(cls10));
    
    % index training and testing
    index_training = [train1 train2 train3 train4 train5 train6 train7 train8 train9 train10];
    index_testing = [test1 test2 test3 test4 test5 test6 test7 test8 test9 test10];
    
    index_train = vertcat(index_train,index_training);
    index_test = vertcat(index_test,index_testing);
    
    for j = 1 : 106
        training_set(j,:) = data(index_training(j),:);
        testing_set(j,:) = data(index_testing(j),:);
    end
    
    
    % SVM training  
    aa = sprintf('do training ...........................................(%d/%d)',i,100);
    disp(aa);
    libsvm_options = '-s 0 -t 0 -c 100 -q';
    
    % SVM testing
    model = svmtrain(training_label, training_set, libsvm_options);
    [predicted_label, accuracy, decision_values_prob_estimates] = svmpredict(training_label, testing_set, model);
    
    predicted_label_result(:,i) = predicted_label;

    % reset
    training_set = [];
    testing_set = [];
    result(i) = accuracy(1);  
    % disp(result);
    disp('mean average:');
    disp(mean(result));
end

average_result = mean(result);
disp(average_result);

save('result.mat','result','predicted_label_result','index_train','index_test');