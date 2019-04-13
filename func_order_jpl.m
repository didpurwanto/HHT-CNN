function [ordered] = func_order_jpl(data)
    ordered(1:12,:) = data(1:7:84,:);    
    ordered(13:24,:) = data(2:7:84,:);    
    ordered(25:36,:) = data(3:7:84,:);    
    ordered(37:48,:) = data(4:7:84,:);    
    ordered(49:60,:) = data(5:7:84,:);    
    ordered(61:72,:) = data(6:7:84,:);    
    ordered(73:84,:) = data(7:7:84,:);    
end