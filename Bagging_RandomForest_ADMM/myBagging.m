%Theory Behind Bagging: 
%Segement training set into sub-section lets say entire training set is S
%we create S_i's from S and send it to multiple classifiers and take the 
%majority vote to detemine the result of classification.
function testError = myBagging(fileName, baseClassifier, fold, level)

    %Call the function to get the 10 folds
    tenFoldCell = GetTenFold(fileName, fold); 
    plot_cell = cell(10,10);
    plot_results_train = zeros(size(baseClassifier,2),fold);
    plot_results_test = zeros(size(baseClassifier,2),fold);

    for foldNum = 1:length(tenFoldCell)
        
        %Accessing test set of each fold
        current_Cell_Test = tenFoldCell{foldNum,2};         
        
        %Accessing training set of each fold
        current_Cell_Train = tenFoldCell{foldNum,1};
        
        %Get train labels from the current fold
        train_labels = current_Cell_Train(:,size(current_Cell_Train,2));        
        test_labels = current_Cell_Test(:,size(current_Cell_Test,2));
        
        %Generating baggs
        X = current_Cell_Train(:,1:34);
        test_data = current_Cell_Test(:,1:34);
        
        num_classifier = [];
        errorR = [];

        % Array that get passed in for the runs
        s = baseClassifier;
        r = cell(fold,3);
        data_plot = [];
        results = zeros(fold,3);
        for i = 1:length(s)
            %Trick to make sure it runs for 50 since last element is 46
            limit = s(i) + 4;
            replacement_idx = randsample(300,300,'true');
            label_matrix = [];
            label_matrix_test = [];
            for j =1:limit
                %display(j)
                %shuffle before every bag
                %shuffle_X = X(randperm(size(X,1)),:);
                %shuffle_X = X;

                train_data_replacement = X(replacement_idx);
                tr_labels = train_labels(replacement_idx);

                tree = build_tree2(train_data_replacement, tr_labels,level);
                label_matrix = [label_matrix classification(X, tree)];
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
            training_error = sum(tr_error)/size(X,1);
            
            %Calculating test Error
            te_error = 0;
            for k = 1:size(label_matrix_test,1)
                model_class_test = mode(label_matrix_test(k,:));
                if(model_class_test ~= test_labels(k))
                    te_error = te_error + 1;
                end
            end

            test_error = te_error / size(current_Cell_Test,1);
            results(i,1) = limit;
            results(i,2) = training_error;
            results(i,3) = test_error;
        end

          tr_data = results(:,2);
          tD_data = results(:,3);
          plot_results_train(:,foldNum) = tr_data(1:length(baseClassifier));
          plot_results_test(:, foldNum) = tD_data(1:length(baseClassifier));
    end
    
    plot(baseClassifier + 4, mean(plot_results_train'),'Red');
    hold on;
    display('Mean Training Error');
    display(mean(plot_results_train')');
    plot(baseClassifier + 4, mean(plot_results_test'));
    hold on;
    display('Mean Test Error')
    display(mean(plot_results_test')');
    display('Standard Deviation for test-error')
    display(std(plot_results_test')');    

%end of the function
end