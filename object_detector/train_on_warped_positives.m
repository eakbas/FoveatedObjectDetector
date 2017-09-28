function [model,aligned_feats] = train_on_warped_positives(pos_examples, ...
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

%% initial training of templates
aligned_feats = {};
for t=1:length(model.templates)
    if isfield(model.templates, 'foveal') && ~model.templates(t).foveal
        % peripheral template

        % find corresponding (in terms of shape) foveal template
        fov_tid = find([model.templates.width]==model.templates(t).width & ...
            [model.templates.height]==model.templates(t).height & ...
            [model.templates.foveal]);

        % apply peripheral feature pooling
        feats = aligned_feats{fov_tid}*full(model.templates(t).T);

        % train
        model.templates(t).w = C0s{t}\(mean(feats,1)' - mu0s{t});
        
        model.templates(t).A = 1;
        model.templates(t).B = 0;
    else
        % foveal template

        % initialization of latent orientation {
        
    %     % initialize with mean of left, right
    %     feats = left_right_clustering(warped_pos_feats{t,1}, ...
    %         warped_pos_feats{t,2},[]);
        
        % find best initialization (i.e. minimum total variation in clustering)
        tv = zeros(min(1000,size(warped_pos_feats{t,1},1)),1);
        for f=1:length(tv)
            [~,tv(f)] = left_right_clustering(warped_pos_feats{t,1}, ...
                warped_pos_feats{t,2},f);
        end
        [~,id] = min(tv);
        feats = left_right_clustering(warped_pos_feats{t,1}, ...
            warped_pos_feats{t,2},id);
        % }

        % if peripheral training is enabled, store a copy of these orientation
        % alinged features
        aligned_feats{t} = feats;
        

        % train
        model.templates(t).w = C0s{t}\(mean(feats,1)' - mu0s{t});
        
%         model.templates(t).A = 1;
%         model.templates(t).B = 0;

%         model.templates(t).A = 1/sqrt( ...
%             model.templates(t).w'*C0s{t}*model.templates(t).w);
%         model.templates(t).B = -mean(feats,1)* ...
%             model.templates(t).w;

%         model.templates(t).A = 1/sqrt( ...
%             model.templates(t).w'*C0s{t}*model.templates(t).w);
%         model.templates(t).B = -mu0s{t}'*model.templates(t).w;
        
%         model.templates(t).A = 1/sqrt( ...
%             model.templates(t).w'*C0s{t}*model.templates(t).w);
%         model.templates(t).B = -0.5*model.templates(t).A* ...
%             (mean(feats,1)*model.templates(t).w + ...
%             mu0s{t}'*model.templates(t).w);

%         % shift the means of positive and negative distributions to +1 and
%         % -1, respectively
%         model.templates(t).A = 2/(mean(feats,1)*model.templates(t).w - ...
%             mu0s{t}'*model.templates(t).w);
%         model.templates(t).B = 1-mean(feats,1)*model.templates(t).w* ...
%             model.templates(t).A;
        
        m1 = mean(feats,1)*model.templates(t).w;
        m0 = mu0s{t}'*model.templates(t).w;
        s = sqrt(model.templates(t).w'*C0s{t}*model.templates(t).w);
        fun = @(x) distribution_scaling_cost(m1,m0,s,x);
        best = fminsearch(fun, [.1 -1]);
        model.templates(t).A = best(1);
        model.templates(t).B = best(2);


        % threshold 
        resps = feats*model.templates(t).w;
        resps = sort(resps);
        model.templates(t).threshold = resps(round(.025*length(resps)));
    end
end
