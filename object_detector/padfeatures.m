function feats = padfeatures(feats, pady, padx, td)

feats = padarray(feats, [pady padx 0]);

if nargin>3
    % add boundary occlusion features to the borders
    td = 32;
    feats(1:pady, :, td) = 1;
    feats(end-pady+1:end, :, td) = 1;
    feats(:,1:padx, td) = 1;
    feats(:,end-padx+1:end, td) = 1;
end