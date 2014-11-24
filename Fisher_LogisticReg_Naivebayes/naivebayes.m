%Naive Bayse Classifier
%This function split data to 80:20 as data and test, then from 80
%We use incremental 5,10,15,20,30 as the test data to understand the error
%rate. 
%Goal is to compare the plots in stanford paper
%http://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf
%Author: Anuradha Uduwage

function[tPercent] = naivebayes(file, iter, percent)
dm = load(file);
    for i=1:iter
        
        %Getting the index common to test and train data
        idx = randperm(size(dm.data,1));
        
        %Using same idx for data and labels
        shuffledMatrix_data = dm.data(idx,:);
        shuffledMatrix_label = dm.labels(idx,:);
        
        percent_data_80 = round((0.8) * length(shuffledMatrix_data));
        
        
        %Doing 80-20 split
        train = shuffledMatrix_data(1:percent_data_80,:);
        
        test = shuffledMatrix_data(percent_data_80+1:length(shuffledMatrix_data),:);
        
        %Getting the label data from the 80:20 split
        train_labels = shuffledMatrix_label(1:percent_data_80,:);
        
        test_labels = shuffledMatrix_label(percent_data_80+1:length(shuffledMatrix_data),:);
        
        %Getting the array of percents [5 10 15..]
        percent_tracker = zeros(length(percent), 2);
        
        for pRows = 1:length(percent)
            
            percentOfRows = round((percent(pRows)/100) * length(train));
            new_train = train(1:percentOfRows,:);
            new_train_label = train_labels(1:percentOfRows);
            
            %get unique labels in training
            numClasses = size(unique(new_train_label),1);
            classMean = zeros(numClasses,size(new_train,2));
            classStd = zeros(numClasses, size(new_train,2));
            priorClass = zeros(numClasses, size(2,1));
            
            % Doing the K class mean and std with prior
            for kclass=1:numClasses
                classMean(kclass,:) = mean(new_train(new_train_label == kclass,:));
                classStd(kclass, :) = std(new_train(new_train_label == kclass,:));
                priorClass(kclass, :) = length(new_train(new_train_label == kclass))/length(new_train);
            end
            
            error = 0;
            
            p = zeros(numClasses,1);
            
            % Calculating the posterior for each test row for each k class
            for testRow=1:length(test)
                c=0; k=0;
                for class=1:numClasses
                    temp_p = normpdf(test(testRow,:),classMean(class,:), classStd(class,:));
                    p(class, 1) = sum(log(temp_p)) + (log(priorClass(class)));
                end
                %Take the max of posterior 
                [c,k] = max(p(1,:));
                if test_labels(testRow) ~= k
                    error = error +  1;
                end
            end
            avgError = error/length(test);
            percent_tracker(pRows,:) = [avgError percent(pRows)];
            tPercent = percent_tracker;
            plot(percent_tracker)
        end
    end
end