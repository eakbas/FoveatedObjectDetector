function [aggregate_responses, num_responses, feats] = ...
    fixate_CALIB(img, feats, fx, fy, model, ...
    template_shapes, ...
    aggregate_responses, num_responses)

%% num cell to pad the feature map
padx = model.padx;
pady = model.pady;


%% extract features if not given
max_num_cells_x = 0;
max_num_cells_y = 0;
if isempty(feats)
    feats.padx = padx;
    feats.pady = pady;
    
    img = double(img);
    
    feats.at_scale = cell(length(model.scales), 2);
    for s=1:length(model.scales)
        rezimg = imresize(img, model.scales(s));
        fs = features(rezimg, model.bin_length);
        fs = fs(:,:,1:31);
        feats.at_scale{s,1} = padfeatures(fs, pady+1, padx+1);
        feats.at_scale{s,2} = flipfeat(feats.at_scale{s,1});
        
        if size(feats.at_scale{s,1},2)>max_num_cells_x
            max_num_cells_x =  size(feats.at_scale{s,1},2);
        end
        
        if size(feats.at_scale{s,1},1)>max_num_cells_y
            max_num_cells_y = size(feats.at_scale{s,1},1);
        end
    end
end

%% find the fixation cells in the scale space
fixation_cells = [];
for s=1:length(model.scales)    
    fixation_cells(s,:) = round([ ...
        (size(feats.at_scale{s,1},2)-2*padx)*fx + 0.5 ...
        (size(feats.at_scale{s,1},1)-2*pady)*fy + 0.5])+[padx pady];
end

if isempty(aggregate_responses)
    % create the responses array
        aggregate_responses = NaN(length(model.scales), ...
            max_num_cells_y, max_num_cells_x, ...
            size(template_shapes,1));
        num_responses = zeros(size(aggregate_responses));
end

% %% visualize fixation cells
% for s=1:length(model.scales)
%     a = false(size(feats{s},1), size(feats{s},2));
%     a(fixation_cells(s,2), fixation_cells(s,1)) = true;
%     figure, imagesc(a)
%     drawnow
%     pause
% end

bboxes_left_top = zeros(5*length(model.templates), 5);
pointer = 1;
for t=1:length(model.templates)
    bboxes_left_top(pointer,:) = [model.templates(t).bboxes_left_top(1,1:2) ...
       model.templates(t).bboxes_left_top(1,1:2)+ ...
       [model.templates(t).width-1 model.templates(t).height-1] t];
    pointer = pointer + 1;
    
    if model.templates(t).foveal
        for b=2:size(model.templates(t).bboxes_left_top,1)
            bboxes_left_top(pointer,:) = [ ...
                model.templates(t).bboxes_left_top(b,:) ...
                model.templates(t).bboxes_left_top(b,1:2)+ ...
                [model.templates(t).width-1 model.templates(t).height-1] t];
            pointer = pointer + 1;
        end
    end
end
bboxes_left_top = bboxes_left_top(1:pointer-1,:);

%% compute template responses
for s=1:length(model.scales)
    bx_l_t = bboxes_left_top;
    
    bx_l_t(:,1) = bx_l_t(:,1) + fixation_cells(s,1);
    bx_l_t(:,3) = bx_l_t(:,3) + fixation_cells(s,1);
    bx_l_t(:,2) = bx_l_t(:,2) + fixation_cells(s,2);
    bx_l_t(:,4) = bx_l_t(:,4) + fixation_cells(s,2);    
    
    valids = bx_l_t(:,1)>0 & bx_l_t(:,2)>0 & ...
        bx_l_t(:,3)<=size(feats.at_scale{s,1},2) & ...
        bx_l_t(:,4)<=size(feats.at_scale{s,1},1);
    
    bx_l_t = bx_l_t(valids,:);
    
    for b=1:size(bx_l_t,1)
        t = bx_l_t(b,5);
        
%         shape_id = find(model.templates(t).height==template_shapes(:,1) & ...
%             model.templates(t).width==template_shapes(:,2));
        shape_id = model.templates(t).shape_id;
        
        
        % compute response
        f = feats.at_scale{s,1}(bx_l_t(b,2):bx_l_t(b,4), ...
            bx_l_t(b,1):bx_l_t(b,3), :);
        flipped_f = feats.at_scale{s,2}(bx_l_t(b,2):bx_l_t(b,4), ...
            size(feats.at_scale{s,1},2)-(bx_l_t(b,1)+model.templates(t).width-1)+1+...
            (0:model.templates(t).width-1), :);
        
        f = f(:)';
        f2 = flipped_f(:)';
        
        r = max(f*model.templates(t).w, f2*model.templates(t).w);
        
        %                 prob = 1./(1+exp(templates(t).b(1)*r + ...
        %                     templates(t).b(2)));
        
        if r<model.templates(t).threshold
            prob = -Inf;
        else
            prob = model.templates(t).A*r+model.templates(t).B;
        end
        
        %                 id2 = sub2ind(size(aggregate_response), s, bbox_left_top(2), ...
        %                         bbox_left_top(1), shape_id);
        id = s + (bx_l_t(b,2)-1)*length(model.scales) + ...
            (bx_l_t(b,1)-1)*length(model.scales)*size(aggregate_responses,2) + ...
            (shape_id-1)*size(aggregate_responses,3)*size(aggregate_responses,2)*...
            length(model.scales);
        
        %                 assert(isequal(id, id2));
        if isnan(aggregate_responses(id))
            num_responses(id) = num_responses(id) + 1;
            aggregate_responses(id) = prob;
        else
            aggregate_responses(id) = ( ...
                num_responses(id)*aggregate_responses(id) + prob) / ...
                (num_responses(id) + 1);
            num_responses(id) = num_responses(id) + 1;
        end
    end
end




function f = flipfeat(f)
% Horizontally flip HOG features (or filters).
%   f = flipfeat(f)
% 
%   Used for learning models with mirrored filters.
%
% Return value
%   f   Output, flipped features
%
% Arguments
%   f   Input features

% flip permutation
p = [10  9  8  7  6  5  4  3  2 ... % 1st set of contrast sensitive features
      1 18 17 16 15 14 13 12 11 ... % 2nd set of contrast sensitive features
     19 27 26 25 24 23 22 21 20 ... % Contrast insensitive features
     30 31 28 29];                % Gradient/texture energy features
     
f = f(:,end:-1:1,p);
