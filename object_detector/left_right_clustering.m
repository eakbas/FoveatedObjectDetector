function [feats,total_var, to_left] = left_right_clustering(feats, fpfeats, id)

% initialize left and right means

if nargin==2 || isempty(id)
    %id = randi(size(feats,1));
    left_mean = mean(feats,1);
    right_mean = mean(fpfeats,1);
else
    left_mean = feats(id,:);
    right_mean = fpfeats(id,:);
end

prev_to_left = [];

iter = 0;
while true
    % step 1: go over all examples and assign to means
    dists = pdist2([left_mean ; right_mean], feats);
    [~,to_left] = min(dists);
    to_left = to_left==1;
    
    if isequal(to_left, prev_to_left)
        break
    else
        prev_to_left = to_left;
    end
    
    % step 2: compute new cluster means
    left_mean  = mean([feats(to_left,:); fpfeats(~to_left,:)], 1);
    right_mean = mean([feats(~to_left,:); fpfeats(to_left,:)], 1);
    
    iter = iter + 1;
end

% return final features
feats = [feats(to_left,:); fpfeats(~to_left,:)];

total_var = mean(dists(1,to_left))+mean(dists(2,~to_left));