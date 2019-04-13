clc
clear 
close all

%% Testing Handcrafted 
% label(1:4) = 1; % ball
% label(5:24) = 2; % car
% label(25:30) = 3; % drink 
% label(31:51) = 4; % feed
% label(52:70) = 5; % lookL
% label(71:83) = 6; % lookR
% label(84:104) = 7; % Pet
% label(105:117) = 8; % Shake
% label(118:136) = 9; % Sniff
% label(137:153) = 10; % Walk
% % 
% % save('labelTraining.mat','label');
% 
% label(1:4) = 1; % ball
% label(5:8) = 2; % car
% label(9:12) = 3; % drink 
% label(11:16) = 4; % feed
% label(17:20) = 5; % lookL
% label(21:24) = 6; % lookR
% label(25:28) = 7; % Pet
% label(29:32) = 8; % Shake
% label(33:36) = 9; % Sniff
% label(37:40) = 10; % Walk
% 
% save('labelTesting.mat','label');

% %% Training Handcrafted 
% label(1:4) = 1; % ball
% label(5:16) = 2; % car
% label(17:21) = 3; % drink 
% label(22:34) = 4; % feed
% label(35:45) = 5; % lookL
% label(46:54) = 6; % lookR
% label(55:67) = 7; % Pet
% label(68:77) = 8; % Shake
% label(78:89) = 9; % Sniff
% label(90:100) = 10; % Walk
% 
% save('LabelHancraftedPCA.mat','label');

% % 
% % save('labelTraining.mat','label');
% 
% label(1:14) = 1; % ball
% label(15:40) = 2; % car
% label(41:50) = 3; % drink 
% label(51:76) = 4; % feed
% label(77:98) = 5; % lookL
% label(99:116) = 6; % lookR
% label(117:138) = 7; % Pet
% label(139:158) = 8; % Shake
% label(159:186) = 9; % Sniff
% label(187:212) = 10; % Walk
% 
% save('label_pot.mat','label');

% label2(1:7) = 1; % ball
% label2(8:20) = 2; % car
% label2(21:25) = 3; % drink 
% label2(26:38) = 4; % feed
% label2(39:49) = 5; % lookL
% label2(50:58) = 6; % lookR
% label2(59:69) = 7; % Pet
% label2(70:79) = 8; % Shake
% label2(80:93) = 9; % Sniff
% label2(94:106) = 10; % Walk
% 
% save('label_pot2.mat','label2');

% label(1:18) = 1; % ball
% label(21:38) = 2; % car
% label(41:58) = 3; % drink 
% label(61:78) = 4; % feed
% label(81:98) = 5; % lookL
% label(101:118) = 6; % lookR
% label(121:138) = 7; % Pet
% label(141:158) = 8; % Shake
% label(161:178) = 9; % Sniff
% label(181:198) = 10; % Walk
% label(201:220) = 11; % Walk
% label(221:240) = 12; % Walk
% label(241:260) = 13; % Walk
% 
% save('label_lena.mat','label');

% label2(1:2) = 1; % ball
% label2(3:4) = 2; % car
% label2(5:6) = 3; % drink 
% label2(7:8) = 4; % feed
% label2(9:10) = 5; % lookL
% label2(10:12) = 6; % lookR
% label2(13:14) = 7; % Pet
% label2(15:16) = 8; % Shake
% label2(17:18) = 9; % Sniff
% label2(19:20) = 10; % Walk
% label2(21:22) = 11; % Walk
% label2(23:24) = 12; % Walk
% label2(25:26) = 13; % Walk
% 
% save('label_lena2.mat','label2');

% label_train(1:14) = 1; % ball
% label_train(21:34) = 2; % car
% label_train(41:54) = 3; % drink 
% label_train(61:74) = 4; % feed
% label_train(81:94) = 5; % lookL
% label_train(101:114) = 6; % lookR
% label_train(121:134) = 7; % Pet
% label_train(141:154) = 8; % Shake
% label_train(161:174) = 9; % Sniff
% label_train(181:194) = 10; % Walk
% label_train(201:214) = 11; % Walk
% label_train(221:234) = 12; % Walk
% label_train(241:254) = 13; % Walk
% 
% save('label_lena_train.mat','label_train');


% label_test(15:20) = 1; % ball
% label_test(35:40) = 2; % car
% label_test(55:60) = 3; % drink 
% label_test(75:80) = 4; % feed
% label_test(95:100) = 5; % lookL
% label_test(115:120) = 6; % lookR
% label_test(135:140) = 7; % Pet
% label_test(155:160) = 8; % Shake
% label_test(175:180) = 9; % Sniff
% label_test(195:200) = 10; % Walk
% label_test(215:220) = 11; % Walk
% label_test(235:240) = 12; % Walk
% label_test(255:260) = 13; % Walk

% save('label_lena_test.mat','label_test');


% label2(1:72) = 1; 
% label2(73:90) = 2;
% label2(91:162) = 3;
% label2(163:180) = 4;
% label2(181:198) = 5;
% 
% save('label_lena2_level1.mat','label2');
% 
% 
% label3(1:8) = 1; 
% label3(9:12) = 2;
% label3(13:20) = 3;
% label3(21:24) = 4;
% label3(25:26) = 5;
% 
% save('label_lena3_level1.mat','label3');


label(1:12) = 1;
label(13:24) = 2;
label(25:36) = 3;
label(37:48) = 4;
label(49:60) = 5;
label(61:72) = 6;
label(73:84) = 7;
save('label_jpl1.mat','label');

% label2(1:6) = 1;
% label2(7:12) = 2;
% label2(13:18) = 3;
% label2(19:24) = 4;
% label2(25:30) = 5;
% label2(31:36) = 6;
% label2(37:42) = 7;
% save('label_jpl2.mat','label2');