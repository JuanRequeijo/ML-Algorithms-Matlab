%Theory Behind Bagging: 
%Segement training set into sub-section lets say entire training set is S
%we create S_i's from S and send it to multiple classifiers and take the 
%majority vote to detemine the result of classification.
function testError = myRForest(fileName, features, fold, level)

    %Call the function to get the 10 folds
    tenFoldCell = GetTenFold(fileName, fold);    

    for foldNum = 1:length(tenFoldCell)
        
        %Accessing test set of each fold
        current_Cell_Test = tenFoldCell{foldNum,2};
        
        test_data = current_Cell_Test;
        
        %Accessing training set of each fold
        current_Cell_Train = tenFoldCell{foldNum,1};
        
        %Get train labels from the current fold
 
        train_labels = current_Cell_Train(:,size(current_Cell_Train,2));        
        test_labels = current_Cell_Test(:,size(current_Cell_Test,2)); 
        
        plot_results_train = zeros(size(features,2),fold);
        plot_results_test = zeros(size(features,2),fold);
        
        %Generating baggs
        original_training_data = current_Cell_Train;

        %m = randperm(34)
%         m=m(1:length(features));
%         mask = zeros(size(original_training_data));
%         mask(:,m)=1;
%         masked_training_data = original_training_data.*mask;
        results = zeros(size(features,1),3);
        for feat=1:length(features)
            m = randperm(34);
            m=m(1:feat);
            mask = zeros(size(original_training_data));
            mask(:,m)=1;
            masked_training_data = original_training_data.*mask;
            
            label_matrix = [];
            label_matrix_test = [];
            for i=1:30
                %random with replacement
                replacement_idx = randsample(size(original_training_data,1),size(original_training_data,1),'true');
                
                %picking data with replacement
                train_data_replacement = masked_training_data(replacement_idx,:);
                tr_labels = train_labels(replacement_idx);

                tree = build_tree2(train_data_replacement, tr_labels,level);
                label_matrix = [label_matrix classification(original_training_data, tree)];
                label_matrix_test = [label_matrix_test classification(test_data, tree)];
            end
            %calculating training error
            tr_error = 0;
            for row=1:size(label_matrix,1)
                model_class = mode(label_matrix(row,:));
                if (model_class ~= train_labels(row))
                    tr_error = tr_error + 1;
                end
            end
            training_error = tr_error/size(original_training_data,1);
            
            %Calculating test Error
            te_error = 0;
            for k = 1:size(label_matrix_test,1)
                model_class_test = mode(label_matrix_test(k,:));
                if(model_class_test ~= test_labels(k))
                    te_error = te_error + 1;
                end
            end

            test_error = te_error / size(current_Cell_Test,1);
            results(feat,1) = feat;
            results(feat,2) = training_error;
            results(feat,3) = test_error;            
        end
        plot_results_train(:,foldNum) = results(:,2);
        plot_results_test(:, foldNum) = results(:,3);
    end
    plot(features, mean(plot_results_train'),'Red');
    hold on;
    display('Mean Training Error');
    display(mean(plot_results_train')');
    plot(features, mean(plot_results_test'));
    hold on;
    display('Mean Test Error')
    display(mean(plot_results_test')');
    display('Standard Deviation for test-error')
    display(std(plot_results_test')');

%end of the function
end