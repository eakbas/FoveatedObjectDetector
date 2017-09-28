function templates = create_foveal_and_peripheral_templates( ...
    widths, heights, num_feat_dims, fovea_left_top, fovea_right_bottom, ...
    peripheral_filters, bin_length)

templates = [];

%% create the foveal templates
for t=1:length(widths)
    id = length(templates)+1;
    templates(id).foveal = true;

    % width and height (in terms of number of foveal cells)
    templates(id).width = widths(t);
    templates(id).height = heights(t);

    % weights
    templates(id).w = zeros(heights(t), widths(t), num_feat_dims);

    % bias weight
    templates(id).wb = 0;
    
    templates(id).pooling_regions = [];
    templates(id).foveal_cells = [];
    
    % bouding boxes
    xs = fovea_left_top(1)+ ...
        (0:abs(fovea_left_top(1))+1+fovea_right_bottom(1)-widths(t));
    ys = fovea_left_top(2)+ ...
        (0:abs(fovea_left_top(2))+1+fovea_right_bottom(2)-heights(t));
    [xs,ys] = meshgrid(xs, ys);
    templates(id).bboxes_left_top = [xs(:) ys(:)];
    
    % feature transform per bbox
    for b=1:size(templates(id).bboxes_left_top,1)
        templates(id).ftransform{b} = 1;
    end
end


%% create peripheral templates
for t=1:length(widths)
    [bboxes, pooling_region_sets] = identify_all_output_labels(...
        peripheral_filters, bin_length, ...
        heights(t), widths(t), false);
    
    % create a peripheral template for each pooling-region-set
    for s=1:length(pooling_region_sets)
        id = length(templates)+1;
        
        templates(id).foveal = false;
        
        % template size
        templates(id).width = widths(t);
        templates(id).height = heights(t);
        
       

        templates(id).pooling_regions = pooling_region_sets{s};
        templates(id).foveal_cells = [];
        
        % bounding boxes
        bbs = bboxes(bboxes(:,4)==s,:);
        templates(id).bboxes_left_top = sortrows(bbs,-3);
        
        % feature transform per bbox
        for b=1:size(templates(id).bboxes_left_top,1)
            templates(id).ftransform{b} = get_peripheral_pooling_coeffs( ...
                peripheral_filters, bin_length, ...
                pooling_region_sets{s}, ...
                templates(id).bboxes_left_top(b,:), ...
                heights(t),widths(t), fovea_left_top, fovea_right_bottom);
        end
        
        % weights
        templates(id).w = zeros(size(templates(id).ftransform{1},1), ...
            num_feat_dims);
        
%         % bias weight
%         templates(id).wb = 0;
    end
end
