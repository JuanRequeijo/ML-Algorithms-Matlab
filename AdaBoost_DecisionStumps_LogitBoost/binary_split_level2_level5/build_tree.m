function t = build_tree(X,Y, level)
% Builds a decision tree to predict Y from X.  The tree is grown by
% recursively splitting each node using the feature which gives the best
% information gain until the leaf is consistent or all inputs have the same
% feature values.
%
% X is an nxm matrix, where n is the number of points and m is the
% number of features.
% Y is an nx1 vector of classes
% cols is a cell-vector of labels for each feature
%
% RETURNS t, a structure with three entries:
% t.p is a vector with the index of each node's parent node
% t.inds is the rows of X in each node (non-empty only for leaves)
% t.labels is a vector of labels showing the decision that was made to get
% to that node
cols = {'1', '2', '3', '4', '5','6', '7' '8', '9', '10', '11', '12', '13', '14', '15','16', '17' '18', '19', '20', '21'};

% Create an empty decision tree, which has one node and everything in it
inds = {1:size(X,1)}; % A cell per node containing indices of all data in that node
p = 0; % Vector contiaining the index of the parent node for each node
labels = {}; % A label for each node

% Create tree by splitting on the root
[inds p labels] = split_node(X, Y, inds, p,labels, cols, 1, level);


t.inds = inds;
t.p = p;
t.labels = labels;



function [inds p labels] = split_node(X, Y, inds, p, labels, cols, node, level)
% Recursively splits nodes based on information gain

% Check if the current leaf is consistent
if numel(unique(Y(inds{node}))) == 1
    return;
end

% Check if all inputs have the same features
% We do this by seeing if there are multiple unique rows of X
if size(unique(X(inds{node},:),'rows'),1) == 1
    return;
end

% Otherwise, we need to split the current node on some feature

best_ig = -inf; %best information gain
best_feature = 0; %best feature to split on
best_val = 0; % best value to split the best feature on

curr_X = X(inds{node},:);
curr_Y = Y(inds{node});
% Loop over each feature
for i = 1:(size(X,2) - 1)
    feat = curr_X(:,i);
    
    % Deterimine the values to split on
    vals = unique(feat);
    splits = 0.5*(vals(1:end-1) + vals(2:end));
    if numel(vals) < 2
        continue
    end
    
    % Get binary values for each split value
    bin_mat = double(repmat(feat, [1 numel(splits)]) < repmat(splits', [numel(feat) 1]));
    
    % Compute the information gains
    H = ent(curr_Y);
    H_cond = zeros(1, size(bin_mat,2));
    for j = 1:size(bin_mat,2)
        H_cond(j) = cond_ent(curr_Y, bin_mat(:,j));
    end
    IG = H - H_cond;
    
    % Find the best split
    [val ind] = max(IG);
    if val > best_ig
        best_ig = val;
        best_feature = i;
        best_val = splits(ind);
    end
end

% Split the current node into two nodes
feat = curr_X(:,best_feature);
feat = feat < best_val;
inds = [inds; inds{node}(feat); inds{node}(~feat)];
inds{node} = [];
p = [p; node; node];
labels = [labels; sprintf('%s < %2.2f', cols{best_feature}, best_val); ...
    sprintf('%s >= %2.2f', cols{best_feature}, best_val)];

% Recurse on newly-create nodes recursive stops if the level reaches.
if (level > 0)
    n = numel(p)-2;
    [inds p labels] = split_node(X, Y, inds, p, labels, cols, n+1, level-1);
    [inds p labels] = split_node(X, Y, inds, p, labels, cols, n+2, level-1);
end


