function [new_feature] = func_norm(input_feature, norm_type)
    new_feature = [];
    [x,y] = size(input_feature);
    for i = 1:x
        if strcmp(norm_type,'L1') == 1
             tmp = input_feature(i,:)/norm(input_feature(i,:),1);
             new_feature = vertcat(new_feature,tmp);
        elseif strcmp(norm_type,'L2') == 1
             tmp = input_feature(i,:)/norm(input_feature(i,:),2);
             new_feature = vertcat(new_feature,tmp);
        end  
    end
end 
