%calculate conditional entropy
function [ Hx_y, splitValue ] = conditional_ent(attr, data, labels) 

uniqueValues = unique(data(:,attr));
defaultChunks = 5;
totalInstances = size(data,1);

if size(uniqueValues,1) <= defaultChunks   
    classLabels_leftValue = labels(data(:,attr)==uniqueValues(1));
    Hx_y_Left = cal_entropy(classLabels_leftValue);    
    
    if(length(uniqueValues) > 1)
        classLabels_rightValue = labels(data(:,attr)== uniqueValues(2));
        Hx_y_Right = cal_entropy(classLabels_rightValue);
        Hx_y = (size(classLabels_leftValue,1)/totalInstances) * Hx_y_Left + (size(classLabels_rightValue,1)/totalInstances) * Hx_y_Right;
    else
        Hx_y = (size(classLabels_leftValue,1)/totalInstances) * Hx_y_Left;
    end
    
    splitValue = min(uniqueValues);    
else
    chunks = defaultChunks;    
    minValue = min(data(:,attr));
    maxValue = max(data(:,attr));
    step = (maxValue-minValue)/chunks;

    uniqueValues = minValue:step:maxValue;
    Hx_y = zeros(size(uniqueValues,2),1);

    for value=1:size(uniqueValues,2)
        classLabels_leftValue = labels(data(:,attr)<=uniqueValues(value));
        classLabels_rightValue = labels(data(:,attr)> uniqueValues(value));
        Hx_y_Left = cal_entropy(classLabels_leftValue);
        Hx_y_Right = cal_entropy(classLabels_rightValue);
        Hx_y(value) = (size(classLabels_leftValue,1)/totalInstances) * Hx_y_Left + (size(classLabels_rightValue,1)/totalInstances) * Hx_y_Right;       
    end

    [min_entropy,index] = min(Hx_y);
    Hx_y = min_entropy;
    splitValue = uniqueValues(index);
end


return;


end
