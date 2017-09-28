function warped_pos_feats = get_warped_positives(pos_examples, ...
    eg2template_assigns, model, C0s, mu0s)
% Trains the model on warped positive examples

warped_pos_feats = cell(length(unique(eg2template_assigns)), 2); % num templates x {left,right}
for i=1:length(pos_examples)
    % to which aspect ratio this positive example is assigned?
    tid = eg2template_assigns(i);
    
    % size of the template: 
    w = model.templates(tid).width*model.bin_length;
    h = model.templates(tid).height*model.bin_length;
    
    % read in the image
    img = imread(pos_examples(i).imgfilename);
    bbox = pos_examples(i).gt_bbox;
    
    padx = model.bin_length * (bbox(3)-bbox(1)+1) / w;
    pady = model.bin_length * (bbox(4)-bbox(2)+1) / h;
    x1 = round(bbox(1)-padx);
    x2 = round(bbox(3)+padx);
    y1 = round(bbox(2)-pady);
    y2 = round(bbox(4)+pady);
    
    % warp
    window = subarray(img, y1, y2, x1, x2, 1);
    warped_img = imresize(window, [h w]+2*model.bin_length);
    
      
    % extract features
    x = features(double(warped_img), model.bin_length);
    x = x(:,:,1:31);
    fx = flipfeat(x);
    warped_pos_feats{tid,1} = [warped_pos_feats{tid,1} ; x(:)'];
    warped_pos_feats{tid,2} = [warped_pos_feats{tid,2} ; fx(:)'];
end
