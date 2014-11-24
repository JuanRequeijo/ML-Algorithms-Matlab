function[foldsE, error] = diagFisher(dataFile, x)

    %Call the function to get the 10 folds
    tenFoldCell = GetTenFold(dataFile, x);
    stdrate = zeros(10,1);
    
    %iterate of the 10 folds
    for n=1:length(tenFoldCell)
        
        %Getting the training data
        currentCellTrain = tenFoldCell{n,1}
        %Gettomg test data
        currentCellTest = tenFoldCell{n,2};
        
        testData = currentCellTest(:,1:(size(currentCellTest,2)-1))
        testLabel = currentCellTest(:,size(currentCellTest,2))
        
        total_col_minus_one = size(currentCellTest,2) - 1;
        total_col = size(currentCellTest,2);
        
        %Constructing
        M1Train = currentCellTrain(currentCellTrain(:,total_col)==1,1:total_col_minus_one);
        M2Train = currentCellTrain(currentCellTrain(:,total_col)==2,1:total_col_minus_one);
        M3Train = currentCellTrain(currentCellTrain(:,total_col)==3,1:total_col_minus_one);
        
        %Calculating the mean
        meanOfM1 = mean(M1Train);
        meanOfM2 = mean(M2Train);
        meanOfM3 = mean(M3Train);
        
        M1Size = size(M1Train);
        M2Size = size(M2Train);
        M3Size = size(M3Train);
        M1DataPoints_train = M1Size(1,1);
        M2DataPoints_train = M2Size(1,1);
        M3DataPoints_train = M3Size(1,1);
        totalMean = (M1DataPoints_train*meanOfM1 + M2DataPoints_train*meanOfM2 + M3DataPoints_train*meanOfM3)/(M1DataPoints_train + M2DataPoints_train + M3DataPoints_train);

        sb1 = M1DataPoints_train*((meanOfM1 - totalMean)'*(meanOfM1 - totalMean));
        sb2 = M1DataPoints_train*((meanOfM2 - totalMean)'*(meanOfM2 - totalMean));
        sb3 = M1DataPoints_train*((meanOfM3 - totalMean)'*(meanOfM3 - totalMean));
        sb = sb1 + sb2 + sb3;
        
        %Calculating the covariance
        sw1 = cov(M1Train);
        sw2 = cov(M2Train);
        sw3 = cov(M3Train);
        sw = sw1 + sw2 + sw3;
        
        % Using the Identity matrix
        identityW = eye(size(sw));
        
        %Getting the projection
        [eigonVector,evalue] = eigs(identityW\sb,2);
        
        %Geting the target
        y1 = M1Train * eigonVector;
        y2 = M2Train * eigonVector;
        y3 = M3Train * eigonVector;
        
        %Mean of the target
        meanOfy1 = mean(y1);
        meanOfy2 = mean(y2);
        meanOfy3 = mean(y3);
        covy1 = cov(y1);
        covy2 = cov(y2);
        covy3 = cov(y3);
        
        %Doing the project to the D-1
        projected_testData = testData * eigonVector;
        gaus_y1 = mvnpdf(projected_testData,meanOfy1, covy1);
        gaus_y2 = mvnpdf(projected_testData,meanOfy2, covy2);
        gaus_y3 = mvnpdf(projected_testData,meanOfy3, covy3);
        gaus_results = [];
        gaus_results = [gaus_results gaus_y1];
        gaus_results = [gaus_results gaus_y2];
        gaus_results = [gaus_results gaus_y3];
        error = 0;
        match = 0;
        for m=1:length(gaus_results)
            [t,i] = max(gaus_results(m,:));
            if i ~= testLabel(m,:)
                error =+ 1;
            else
                match =+ 1;
            end
        end
        erate = (error/length(gaus_results));
        stdrate(n,1) = erate;
        tempEr = 0;
        for er = 1:length(stdrate)
            tempEr = tempEr + stdrate(er);
        end
        error = tempEr/length(stdrate);
        foldsE = stdrate;
    end
end