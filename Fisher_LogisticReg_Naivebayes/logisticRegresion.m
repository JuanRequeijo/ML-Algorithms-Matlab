function[classMean] = LogisticReg(file, iter, percent)
dm = load(file);
    for i=1:iter
        
        %normalize data
        dataMean = mean(dm.data(:,:));
        dataStd = std(dm.data(:,:));
        normalized_matrix = zeros(size(dm.data))
        
        for dSize=1:length(dm.data)
            normalized_matrix(dSize,:) = (dm.data(dSize) - dataMean)./dataStd;
        end
        display(normalized_matrix)
        %creating the index
        idx = randperm(size(normalized_matrix,1))
        
        %Using same idx for data and labels
        shuffledMatrix_data = normalized_matrix(idx,:);
        shuffledMatrix_label = dm.labels(idx,:);
        
        %Getting the 80% 20%
        percent_data_80 = round((0.8) * length(shuffledMatrix_data));
        
        %Taking percentage of 80% [5 10 15 25 30]
        trainData = shuffledMatrix_data(1:percent_data_80,:);
        %assiging test data for above traindata
        testData = shuffledMatrix_data(percent_data_80+1:length(shuffledMatrix_data),:);
        
        %taking percent of training labels
        train_labels = shuffledMatrix_label(1:percent_data_80,:);
        test_labels = shuffledMatrix_label(percent_data_80+1:length(shuffledMatrix_data),:);
        
        learn_rate = 0.0085 % percentage of learning
        numClasses = size(unique(train_labels),1);
        
        for pRows = 1:length(percent)
            percentOfRows = round((percent(pRows)/100) * length(trainData))
            new_train_data = trainData(1:percentOfRows,:);
            new_train_label = train_labels(1:percentOfRows);
            
            w = zeros(numClasses -1, size(new_train_data,2));
            beta = 0; %Initial guess of the parameter, (starting probability)
            iter = 0; iter_max = 0; % settting up the iterations
            
            gB = 1; %bias scala (Do I need this to changing?)
            gW = zeros(size(w)); % initializing the gradient setting up to one it will do 
            %atleast an update
            
            while sum(abs(gW)) + abs(gB) > 0.1 %check if the gradient is large else iterate
                iter = iter + 1;
                gB = 0; gW = 0 * gW; % reset the gradient
                for i = 1:size(new_train_data,1)
                    for k = 1:numClasses
                        %calculate the denominator for all classes
                        %This is a constant from clas to class.
                        const = (1+exp((beta+w*(new_train_data(k,:)'))));
                    end
                    for j = 1:i
                        %calculating probability
                        c = exp(beta+w*(new_train_data(j,:))')/const;
                        gB = gB + c;
                        gW = gW + c * (new_train_data(j,:)); %x1(:,d)
                    end
                end
                
                w = w + learn_rate*gW % update the weight vector
                beta = beta + learn_rate*gB % update the bias scalar
                if iter > iter_max 
                    break; 
                end
            end
            p = zeros(numClasses,1);
            lError=0;
            % calculate the probabilities p(c|x) for the training data :
            for tRow = 1:length(testData)
                testData(tRow,:)
                for ck = 1:numClasses
                    p(ck,1) = exp((beta+w*(testData(tRow,:)'))/1000)/((1 + exp((beta+w*(testData(tRow,:)')))/1000))
                end
                [ck,value] = max(p(1,:));
                if test_labels(tRow) > value
                    lError = lError +  1;
                end                
            end
            %plot(p,percent)
        end
    end