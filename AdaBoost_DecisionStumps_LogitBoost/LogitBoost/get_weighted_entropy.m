% This entropy callculation from adaboost.
% cant use the fancy tabulate function since we have to update the weights
% for all points.
function ent = get_weighted_entropy(label_vector, weight)
    positive = sum(label_vector(:,1) == 1);
    negative = sum(label_vector(:,1) == -1);
    p_weight = 0;
    n_weight = 0;
    all_weight = 0;
    
    for p=1 : positive
        p_weight = p_weight + weight(p);
    end
    
    for n=1 : negative
        n_weight = n_weight + weight(n);
    end
    
    %calculating weights for all points
    for all = 1 : size(label_vector,1)
        all_weight = all_weight + weight(all);
    end
    prob = [];
    prob(1,1) = n_weight/all_weight;
    prob(2,1) = p_weight/all_weight;
    ent = -sum((prob + 1e-100) .* log2(prob + 1e-100));
end