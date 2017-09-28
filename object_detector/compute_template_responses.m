function R = compute_template_responses(feats, padxy, template, ...
    flip)

if ~isempty(padxy)
    % pad the feature map so that bounding boxes that overflow image borders are allowed
    feats = padarray(feats, [padxy(2)+1 padxy(1)+1 0]);
end


if isfield(template, 'rotW')
    W = template.rotW;
else
    % transform and rotate the filter
    
    % transform the filter into template's height and width
    W = reshape(template.w, [template.height template.width ...
        size(feats,3)]);
    
    % rotate the filter
    for d=1:size(W,3)
        W(:,:,d) = rot90(W(:,:,d), 2);
    end
end




R = 0;
for c=1:size(W,3)
    R = R + conv2(feats(:,:,c), W(:,:,c), 'valid');
%     R = R + filter2(W(:,:,c), feats(:,:,c), 'valid');
end

% try left-right flipping?
if flip
    R2 = compute_template_responses(flipfeat(feats), [], template, ....
        false);
    R2 = R2(:,end:-1:1);
%     t = template;
%     foo = flipfeat(W);
%     t.weights = [foo(:) ; template.weights(end)];
% R = compute_template_responses(feats, t, bias, false);
    R = max(R,R2);
end


