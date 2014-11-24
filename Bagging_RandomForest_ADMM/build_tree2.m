function t = build_tree2(X,Y, level)
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


% Create an empty decision tree, which has one node and everything in it

bestValue = 0; % Vector contiaining the index of the parent node for each node
right_label = 0; % A label for each node
left_label = 0;
feature = 0;

% Create tree by splitting on the root
t = split_node(X, Y, feature, bestValue, right_label, left_label, level);
% 
% t.feature = feature;
% t.bestValue = bestValue;
% t.right_label = right_label;
% t.left_label = left_label;
end