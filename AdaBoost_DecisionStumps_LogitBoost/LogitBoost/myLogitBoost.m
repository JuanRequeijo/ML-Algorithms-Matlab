function [train_error, test_error] = myLogitBoost(file, fold, stump)

train_error = [];
test_error = [];

%Call the function to get the 10 folds
tenFoldCell = GetTenFold(file, fold);

%storing split and error rate for each fold
split_and_error = zeros(length(tenFoldCell),2);

ten_fold_test_error = zeros(10,1);
ten_fold_train_error = zeros(10,1);

    %Iterate over training folds
    for n=1:length(tenFoldCell)
        %Acessing training selt of each fold
        currentCellTrain = tenFoldCell{n,1};

        %Test data
        current_test_set = tenFoldCell{n,2};

        %Get train labels from the currnt fold
        curr_Y = currentCellTrain(:,size(currentCellTrain,2));
        
        current_test_set_Y = current_test_set(:,size(current_test_set,2));

        W_t = zeros(size(currentCellTrain,1));

        %initializing the W(T=1)
        for k =1:size(currentCellTrain, 1)
            W_t(k,1) = 1/size(currentCellTrain,1);
        end

        stump_error = zeros(stump, 3);
        stump_storage = cell(stump,2);

        for T=1:stump


            %place holder for tree
            %stump_holder = {1:size(currentCellTrain,1)};

            %feature to split
            %results_cell = cell(10,2);
            feature_to_split_cell = cell(size(currentCellTrain,2)-1,4);

            %iterate over each feature to find the best split,
            %-1 is last column is label column
            for feature_idx=1:(size(currentCellTrain,2) - 1)

                %get current feature
                curr_X = currentCellTrain(:,feature_idx);

                %identify the unique values
                unique_values_in_feature = unique(curr_X);

                H = get_weighted_entropy(curr_Y, W_t); %This is actually H(X) in slides
                %temp entropy holder

                %Storage for feature element's class
                element_class = zeros(size(unique_values_in_feature,1),2);

                %conditional probability H(X|y)
                H_cond = zeros(size(unique_values_in_feature,1),1);

                for aUnique=1:size(unique_values_in_feature,1)
                    match = curr_X(:,1) == unique_values_in_feature(aUnique);
                    mat = curr_Y(match);
                    majority_class = mode(mat);
                    element_class(aUnique,1) = unique_values_in_feature(aUnique);
                    element_class(aUnique,2) = majority_class;
                    H_cond(aUnique,1) = (length(mat)/size((curr_X),1)) * get_weighted_entropy(mat, W_t);
                end

                %Getting the information gain
                IG = H - sum(H_cond);

                %Storing the IG of features
                feature_to_split_cell{feature_idx, 1} = feature_idx;
                feature_to_split_cell{feature_idx, 2} = max(IG);
                feature_to_split_cell{feature_idx, 3} = unique_values_in_feature;
                feature_to_split_cell{feature_idx, 4} = element_class;
            end
            feature_to_split_cell;
            %set feature to split zero for every fold
            feature_to_split = 0;

            %getting the max IG of the fold
            max_IG_of_fold = max([feature_to_split_cell{:,2:2}]);

            %vector to store values in the best feature
            values_of_best_feature = zeros(size(15,1));

            %Iterating over cell to get get the index and the values under best
            %splited feature.
            for i=1:length(feature_to_split_cell)
                if (max_IG_of_fold == feature_to_split_cell{i,2});
                    feature_to_split = i;
                    values_of_best_feature = feature_to_split_cell{i,4};
                    labels = values_of_best_feature;
                end
            end
            split_feature = feature_to_split;

            %get all the rows and iterate over the rows
            temp_error = 0;
            for j=1:size(current_test_set(:,1))
                %getting the test label
                test_label = current_test_set(j,size(current_test_set,2));
                % take the feature that we trained to be best split
                value_of_test_feature = current_test_set(j,feature_to_split);
                for k=1:length(element_class)
                    if (value_of_test_feature == values_of_best_feature(k,1))
                        if (values_of_best_feature(k,2) ~= test_label)
                            temp_error = temp_error + 1;
                        end
                    end
                end
            end
            error_rate = temp_error/size(current_test_set,1);


            %Building the update function 
            for point = 1:size(currentCellTrain,1)
                get_feat_column = currentCellTrain(point, feature_to_split);
                label_of_stump = values_of_best_feature(values_of_best_feature(:,1) == get_feat_column,2);
                y_i = curr_Y(point,:);
                W_t(point) = (W_t(point) .* 1/(1 + exp(y_i * label_of_stump)));
            end

            test_error = error_rate;
            stump_error(T,1) = T;
            stump_error(T,3) = test_error;
            stump_storage{T,1} = feature_to_split;
            stump_storage{T,2} = values_of_best_feature;

        end
        %at this point we hav already trainded the stumps
        train_error = 0;
        avg_error = 0;
        for i = 1:size(currentCellTrain,1)
            sign = zeros(size(currentCellTrain,1),1);
            class = [];
            for k=1:length(stump_storage)
                stump_root = currentCellTrain(:,stump_storage{k,1});
                val = stump_root(i);
                branches = stump_storage{k,2};
                label_of_stump = branches(branches(:,1) == val,2);
                sign(i,1) = stump_error(k,3) * label_of_stump;
            end
            
            %Now we send the point over sigmoid function and based on this
            %we get a probability. 
            %We assign Higher Probability to class 1 & the other to class
            %-1
            
            sigmoid = 1/(1 + exp(sum(sign)));
            class1Prob = sigmoid;
            class2Prob = 1 - class1Prob;
            %Getting the class for a given point to our classifier
            if class1Prob < class2Prob
                class = 1;
            else
                class = -1;
            end
            if curr_Y(i) ~= class
                train_error = train_error + 1;
            end
        end
        avg_error = train_error/size(currentCellTrain,1);

        test_error = 0;
        avg_test_error = 0;
        %calculating test error
        for i = 1:size(current_test_set,1)
            sign = zeros(size(current_test_set,1),1);
            class = [];
            for k=1:length(stump_storage)
                stump_root = current_test_set(:,stump_storage{k,1});
                val = stump_root(i);
                branches = stump_storage{k,2};
                label_of_stump = branches(branches(:,1) == val,2);
                sign(i,1) = stump_error(k,3) * label_of_stump;
            end
            
            %Now we send the point over sigmoid function and based on this
            %we get a probability. 
            %We assign Higher Probability to class 1 & the other to class
            %-1
            sigmoid = 1/(1 + exp(sum(sign)));
            class1Prob = sigmoid;
            class2Prob = 1 - class1Prob;
            %Getting the class for a given point to our classifier
            if class1Prob < class2Prob
                class = 1;
            else
                class = -1;
            end
            if curr_Y(i) ~= class
                train_error = train_error + 1;
            end
        end
        avg_test_error = test_error/size(current_test_set,1);

        ten_fold_test_error(n,1) = avg_test_error;
        ten_fold_train_error(n,1) = avg_error;
    end

    fprintf('Error Rates for each stump after boosting')
    train_error = ten_fold_train_error;
    test_error = ten_fold_test_error;
    display(test_error)
end