function [trainErr, testErr] = myDtree(fileName, level, fold)

%Call the function to get the 10 folds
tenFoldCell = GetTenFold(fileName, fold);

    %iterate over folds
    for foldNum = 1:length(tenFoldCell)
        %now the engineering begins, we need to recurse for each fold for
        %given levels

        %Accessing training set of each fold
        current_Cell_Train = tenFoldCell{foldNum,1};

        %Accessing test set of each fold
        current_Cell_Test = tenFoldCell{foldNum,2};

        %Get train labels from the current fold
        curr_Y_train = current_Cell_Train(:,size(current_Cell_Train,2));

        %Get test labels for the current fold
        curr_Y_test = current_Cell_Train(:,size(current_Cell_Test, 2));

        %set up is complete so call the split function to do the recursive
        %spilit
        %split_node(current_Cell_Train, curr_Y_train, current_Cell_Test, curr_Y_test)
        t = build_tree(current_Cell_Train, curr_Y_train, level);

        % Build the decision tree
        %t = build_tree(X,Y,cols);

        % Display the tree
        treeplot(t.p');
        title('Decision tree)');
        [xs,ys,h,s] = treelayout(t.p');

        for i = 2:numel(t.p)
            % Get my coordinate
            my_x = xs(i);
            my_y = ys(i);

            % Get parent coordinate
            parent_x = xs(t.p(i));
            parent_y = ys(t.p(i));

            % Calculate weight coordinate (midpoint)
            mid_x = (my_x + parent_x)/2;
            mid_y = (my_y + parent_y)/2;

            % Edge label
            text(mid_x,mid_y,t.labels{i-1});

            % Leaf label
            if ~isempty(t.inds{i})
                val = curr_Y_train(t.inds{i});
                if numel(unique(val))==1
                    text(my_x, my_y, sprintf('y=%2.2f\nn=%d', val(1), numel(val)));
                else
                    %inconsistent data
                    text(my_x, my_y, sprintf('**y=%2.2f\nn=%d', mode(val), numel(val)));
                end
            end
        end
    end
end