function [ Hx ] = cal_entropy( labels )

classes = unique(labels);
nClasses = size(classes,1);
totalLabels = size(labels,1);

%Initialize Entropy
Hx = 0;

for i = 1:nClasses
    class = classes(i);
    instances = size(labels(labels==class),1);
    if instances==0
        Hx = Hx + 0;
    else
        Hx = Hx + -(instances/totalLabels) * log(instances/totalLabels);
    end
end

end