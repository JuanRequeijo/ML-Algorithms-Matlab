%Given a vector the function will calculate the entropy
%Since we know each vector is a feature we can pass 
%a vector as a parameter to calculate entropy
%Theory of entropy
%entropy = -sum((dBelongToClass/TotalPoints) .* log2(dBelongToClass/TotalPoints))
function entropy = get_entropy(label_vector)
    frequency_table = tabulate(label_vector);
    
    % Remove zero-entries
    frequency_table = frequency_table(frequency_table(:,3)~=0,:);
    
    prob = frequency_table(:,3) / 100;
    % Get entropy
    entropy = -sum((prob + 1e-100) .* log2(prob + 1e-100));
    %entropy = -sum(prob2 .* log2(prob2));1e-100
end

% This entropy callculation from adaboost.
% cant use the fancy tabulate function since we have to update the weights
% for all points.
function ent = get_weighted_entropy(label_vector, weight)
    positive = sum(label_vector(:,1) == 1);
    negative = sum(label_vector(:,1) == -1);
    p_weight = 0
    n_weight = 0
    all_weight = 0
    
    for p=1 : positive
        p_weight = sum(weight)
    end
    
    for n=1 : negative
        n_weight = sum(weight)
    end
    
    %calculating weights for all points
    for all = 1 : size(label_vector,1)
        all_weight = sum(weight)
    end
    prob = []
    prob(1,1) = n_weight/all_weight
    prob(2,1) = p_weight/all_weight
    ent = -sum(prob .* log2(prob))
end