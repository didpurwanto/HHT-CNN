function [pot_grad1, pot_grad2, pot_max2, pot_sum2] = func_pyramid_pot(vector_sequence, level)
    %% DOCUMENTATION

    
    % OBJECTIVES
    % Temporal Pyramid Function

    % Input     : nxm 4096x1 caffe feature;  
    %             [vector_sequence]   
    % Output    : POT vector representation (using sum, max, and and two histogram of gradients poolings)
    %             [pot_sum2, pot_max2, pot_grad1, pot_grad1_s, pot_grad2, pot_grad2_s]
    % Important parameter : level of temporal pyramid.

    %%  
    filter = 2;
    
    pot_max = []; pot_max2 = [];
    pot_sum = []; pot_sum2 = [];           
    pot_grad1= []; pot_grad2 = []; 
    
    dimension_level = 0;
    for l = 1:level
        n = power(filter,l);
        frame = size(vector_sequence, 2) - mod(size(vector_sequence, 2), n);  
        % disp(frame);
        tmp = vector_sequence(:, 1:frame);
        
        stride = frame/n;

        sum_feature = sum(tmp,1);        
        for m = 1:n           
            dimension_level = dimension_level+1;        
            
            % max pooling
            [M, I] = max(sum_feature(:,(stride*(m-1))+1:stride*m));
            pot_max = [pot_max, tmp(:,(stride*(m-1))+I:(stride*(m-1))+I)];
            pot_max2 = vertcat(pot_max2, tmp(:,(stride*(m-1))+I:(stride*(m-1))+I));
          
            % sum pooling
            sum_tmp = sum(tmp(:,(stride*(m-1))+1:stride*m),2);
            pot_sum = [pot_sum, sum_tmp];
            pot_sum2 = vertcat(pot_sum2, sum_tmp);
            
            % gradient sigma 1 (count) pooling 
            dataaa = tmp(:,(stride*(m-1))+1:stride*m);  
            dataa = gradient(dataaa);
            
            sign_tmp = sign(dataa);            
            grad1_positive = sum(sign_tmp==1,2);
            grad1_negative = sum(sign_tmp==-1,2);
            
            grad1 = vertcat(grad1_positive, grad1_negative);
            pot_grad1 = vertcat(pot_grad1, grad1); 
           

            % gradient sigma 2 (sum) pooling            
            data_p = [];
            data_n = [];
            [x1, y1] = size(dataa);

            for i = 1 : x1
                for j = 1 : y1
                    if sign_tmp(i,j)== 0
                        data_p(i,j) = 0; 
                        data_n(i,j) = 0; 
                    elseif sign_tmp(i,j) == 1
                        data_p(i,j) = dataa(i,j); 
                        data_n(i,j) = 0; 
                    elseif sign_tmp(i,j) == -1
                        data_p(i,j) = 0; 
                        data_n(i,j) = dataa(i,j); 
                    end
                end
            end
            
            get_p = sum(data_p,2);
            get_n = sum(data_n,2);
 
            grad2 = vertcat(get_p, get_n);
            pot_grad2 = vertcat(pot_grad2, grad2);            
            
        end
    end
end