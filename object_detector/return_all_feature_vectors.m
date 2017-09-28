function F = return_all_feature_vectors(feats,th,tw,skip)

F = zeros(tw*th*size(feats,3), ...
    (size(feats,1)-th+1)*(size(feats,2)-tw+1));

i=1;
for x=1:skip:size(feats,2)-tw+1
    for y=1:skip:size(feats,1)-th+1
        f = feats(y:y+th-1, x:x+tw-1, :);
        F(:,i) = f(:);
        i = i + 1;
    end
end
F = F(:,1:i-1);
