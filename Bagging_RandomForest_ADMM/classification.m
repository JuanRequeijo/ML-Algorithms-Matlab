function label = classification(data, tree)
    %label = 0;
    classify_feature = data(:,tree.feature);
    l_vector = zeros(size(data,1),1);
    for row=1:length(classify_feature)
        if classify_feature(row,:) <= tree.bestValue
            l_vector(row,:) = tree.left_label;
        else
            l_vector(row,:) = tree.right_label;
        end
    end
    label = l_vector;
end