clc
clear
close all

srcFiles = dir('C:\Users\didpurwanto\Documents\Dataset\JPL\movie\*.avi');  
totalVideo = length(srcFiles);
selected_imf = 1;

for i = 1:totalVideo
    prog0 = sprintf('Preprocessing ..................................... (%d/%d)', i, totalVideo);
    disp(prog0);
    
    filename = strcat('C:\Users\didpurwanto\Documents\Dataset\DogCentric\IMF\imf_tdd_s51\', srcFiles(i).name);   
    name = srcFiles(i).name(1: end-4); 
    
    disp('    load data!');     
    load(filename);
    data = imf_s51;    
    len = size(data,2);
    
    hht_mean = [];
    hht_std = [];
    hht_centroid_spectral = [];
    hht_varian_coeff = [];
    hht_entropy = [];
    hht_entropy_instan = [];
    hht_mean_instan = [];
    hht_mean_energy = [];
    hht_entropy_energy = [];
    hht_std_energy = [];
    
    disp('    find feature!');     
    tic
    for j = 1:len
        if ~isempty(data{j})  
            % prog0 = sprintf('       feature %d .... (%d/%d)', i, j, len);
            % disp(prog0);
             
            lend = size(data{j},1);
            tmp = data{j};
            
            if size(data{j},1) < selected_imf && size(data{j},1) > 0
               tmp = repmat(tmp,selected_imf,1);
            end
            
            x = size(tmp,1);
            tmp_mean = [];
            tmp_std = [];
            tmp_centroid_spectral = [];
            tmp_varian_coeff = [];
            tmp_entropy = [];
            tmp_entropy_instan = [];
            tmp_mean_instan = [];
            tmp_mean_energy = [];
            tmp_entropy_energy = [];
            tmp_std_energy = [];
            
            for k = 1 : 1 % selectef analytic signal of IMF
                %% analytic signal                
                tmp_hilbert = hilbert(tmp(k,:)); 
                
                % see equation in emd-based temporal features paper
                
                % fitur 1 : mean
                f_mean = mean(abs(tmp_hilbert));
                
                % fitur 2 : std deviasi
                f_std = std(abs(tmp_hilbert),1);
                
                %% power density and instfreq
                tmp_fft = fft(tmp_hilbert); % fft from analytic
                pw = power(abs(tmp_fft),2);                                               
                
                % fitur 3 : spectral centroid
                val = 0; sum_val = 0;
                for l = 1:length(pw)
                    v = l*pw(l);
                    val = val+v;
                    sum_val = sum_val+pw(l);
                end
                f_centroid_spectral = val/sum_val;
                
                % fitur 4 : varian coefficient
                vcc = 0;
                for m = 1: length(pw)
                    c = m - f_centroid_spectral;
                    cc = power(c,2);
                    vc = pw(m)*cc;
                    vcc = vcc+vc;
                end
                f_varian_coeff = vcc/sum_val;

                % fitur 5 : entropy
                f_entropy = entropy(pw);
                
                %% instantaneous frequency
                instfreq = 1/(2*pi) * diff(unwrap(angle(tmp_hilbert)));
                
                % fitur 6 : entropy (instan)
                f_entropy_instan = entropy(instfreq);
                
                % fitur 7 : mean (instan)
                f_mean_instan = mean(instfreq);
                
                %% energy
                energy_tmp = power(abs(tmp_hilbert),2);
                energy = log(energy_tmp);
                
                % fitur 8 : mean (energy)
                f_mean_energy = mean(energy);
                
                % fitur 9 : entropy (energy)
                f_entropy_energy = entropy(energy);
                
                % fitur 10 : std deviasi (energy)
                f_std_energy = std(energy,1);
                
                
                % --------------------------------------------------
                tmp_mean = [tmp_mean f_mean];
                tmp_std = [tmp_std f_std];                                
                tmp_centroid_spectral = [tmp_centroid_spectral f_centroid_spectral];
                tmp_varian_coeff = [tmp_varian_coeff f_varian_coeff];                                
                tmp_entropy = [tmp_entropy f_entropy];                                
                tmp_entropy_instan = [tmp_entropy_instan f_entropy_instan];                
                tmp_mean_instan = [tmp_mean_instan f_mean_instan];                
                tmp_mean_energy = [tmp_mean_energy f_mean_energy];
                tmp_entropy_energy = [tmp_entropy_energy f_entropy_energy];
                tmp_std_energy = [tmp_std_energy f_std_energy];
            end               
        end  
        
        hht_mean = [hht_mean tmp_mean];
        hht_std = [hht_std tmp_std];
        hht_centroid_spectral = [hht_centroid_spectral tmp_centroid_spectral];
        hht_varian_coeff = [hht_varian_coeff tmp_varian_coeff];                        
        hht_entropy = [hht_entropy tmp_entropy];
        hht_entropy_instan = [hht_entropy_instan tmp_entropy_instan];        
        hht_mean_instan = [hht_mean_instan tmp_mean_instan];
        hht_mean_energy = [hht_mean_energy tmp_mean_energy];
        hht_entropy_energy = [hht_entropy_energy tmp_entropy_energy];
        hht_std_energy = [hht_std_energy tmp_std_energy];
    end   
    toc
    
    hht_final_mean(i,:) = hht_mean;   
    hht_final_std(i,:) = hht_std;       
    hht_final_centroid_spectral(i,:) = hht_centroid_spectral;   
    hht_final_varian_coeff(i,:) = hht_varian_coeff;       
    hht_final_entropy(i,:) = hht_entropy;
    hht_final_entropy_instan(i,:) = hht_entropy_instan;
    hht_final_mean_instan(i,:) = hht_mean_instan;
    hht_final_mean_energy(i,:) = hht_mean_energy;
    hht_final_entropy_energy(i,:) = hht_entropy_energy;
    hht_final_std_energy(i,:) = hht_std_energy;
end

save('spatial_s51_hht_1.mat','-v7.3',...
    'hht_final_mean','hht_final_std',...
    'hht_final_centroid_spectral','hht_final_varian_coeff','hht_final_entropy',...
    'hht_final_entropy_instan','hht_final_mean_instan',...
    'hht_final_mean_energy','hht_final_entropy_energy','hht_final_std_energy');
