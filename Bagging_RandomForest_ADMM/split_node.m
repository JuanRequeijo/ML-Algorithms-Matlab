function t = split_node(X, Y, feature, bestValue, right_label, left_label, level)
% Recursively splits nodes based on information gain

% Check if the current leaf is consistent
% if numel(unique(Y(inds{node}))) == 1
%     return;
% end
% 
% % Check if all inputs have the same features
% % We do this by seeing if there are multiple unique rows of X
% if size(unique(X(inds{node},:),'rows'),1) == 1
%     return;
% end

% Otherwise, we need to split the current node on some feature
t.feature = feature;
t.bestValue = bestValue;
t.right_label = right_label;
t.left_label = left_label;

best_ig = -inf; %best information gain
best_feature = 0; %best feature to split on
best_val = 0; % best value to split the best feature on
ig_feat_mat = zeros(34,3);

training_data = X;
training_labels = Y;
% Loop over each feature
for i = 1:(size(X,2) - 1)
    feat = training_data(:,i);
    
    % Deterimine the values to split on
    vals = unique(feat);
    
    %splits = 0.5*(vals(1:end-1) + vals(2:end));
    if numel(vals) == 0
        continue
    end

    %Get entropy
    H = cal_entropy(training_labels); %This is actually H(X) in slide
    [hx_y, split] = conditional_ent(i, training_data, training_labels);
    
    %Getting the information gain
    IG = H - hx_y;
    
    ig_feat_mat(i,1) = i;
    ig_feat_mat(i,2) = IG;
    ig_feat_mat(i,3) = split;
    ig_feat_mat;
    
    % Find the best split
    [val ind] = max(IG);
    if val > best_ig
        best_ig = val;
        best_feature = i;
        best_val = split;
    end
end

% Split the current node into two nodes
feat = training_data(:,best_feature);
lfeat = feat < best_val;
rfeat = feat > best_val;

left_label_feat = training_labels(lfeat,:);
right_label_feat = training_labels(rfeat,:);

%left_vector = curr_Y(left_label_feat,:);
%right_vector = curr_Y(right_label_feat,:);
t.left_label = mode(left_label_feat);
t.right_label = mode(right_label_feat);
if (size(left_label_feat,1)) == 0
    if t.right_label == 1
        t.left_label = 2;
    else
        t.left_label = 1;
    end
elseif (size(right_label_feat,1)) == 0
    if t.left_label ==1
        t.right_label = 2;
    else
        t.right_label = 1;
    end
end
t.feature = best_feature;
t.bestValue = best_val;

if (level > 1)
    %n = numel(p)-2;
    training_data(:,best_feature) = [];
    t = split_node(X, Y, feature, bestValue, right_label, left_label, level-1);
    %[inds p labels] = split_node(X, Y, inds, p, labels, cols, n+2, level-1, p_feature);
end