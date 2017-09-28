function [bboxes, bbox_covered_regions, bad_bboxes] = ...
    identify_all_output_labels(filter, bin_length, t_h, t_w, visualize_boxes)
% Given the peripheral filters, the configuration params and the shape of
%   the template, returns the list of all possible output (bounding box
%   placements) values along with which pooling regions they cover. 

% (conf.bin_length x conf.bin_length) square regions are called cells. The
% pooling regions from Simoncelli's model are simply called regions.

%% top left of the central cell (0,0)
central_cell_x = round(size(filter.regions,4)/2 - bin_length/2);
central_cell_y = round(size(filter.regions,3)/2 - bin_length/2);


%% pooling region weights in one image
visual_field_width = size(filter.regions,3);
W = reshape(filter.regions, [numel(filter.offsets) ...
    visual_field_width visual_field_width]);
W = squeeze(max(W,[],1));
binaryW = W>0;

% % fill in the fovea, too
% L = bwlabel(~binaryW,4);
% binaryW = imdilate(binaryW, strel('disk', 5));
% binaryW(L==L(1,1)) = 0;
% %binaryW(L==L(central_cell_y, central_cell_x)) = 1;

%% investigate each possible bbox location
bboxes = [];
bbox_covered_regions = {};
bad_bboxes = [];

% identify the coordinates of the first (top-left) cell 
start_x = -floor(central_cell_x/bin_length);
start_y = -floor(central_cell_y/bin_length);

for x=start_x:-2*start_x
    for y=start_y:-2*start_y
        % pixel coordinates for this box
        pixs_x = central_cell_x + x*bin_length + ...
            (0:t_w*bin_length-1);
        pixs_y = central_cell_y + y*bin_length + ...
            (0:t_h*bin_length-1);
        
        if any(pixs_x>size(W,2)) || any(pixs_y>size(W,1)) || ...
                any(pixs_x<1) || any(pixs_y<1)
            break
        end
        
        % how much of this box overlaps with the pooling regions? 
        foo = binaryW(pixs_y, pixs_x);
        overlap = sum(foo(:)) / numel(foo);
        
        if overlap<.8
            continue
        end
        
        
        % the covered part
        part = filter.regions(:,:,pixs_y,pixs_x);
        
        coverage_weights = 0;
        total_weights = 0;
        covered_cells = [];
        for a=1:size(part,1)
            for e=1:size(part,2)
                if any(part(a,e,:))
                    covered_part = sum(part(a,e,:))/ ...
                        sum(filter.weights{a,e});
                    if covered_part>.5
                         coverage_weights = coverage_weights + ...
                            sum(part(a,e,:));
                        total_weights = total_weights + ...
                            sum(filter.weights{a,e});
                        covered_cells = [covered_cells ; a e];
                    end
                end
            end
        end        
        
        
        if ~isempty(covered_cells)
            % record this
            overlap_quality = coverage_weights/(total_weights+eps); 
            
            if overlap_quality<.75
                continue
            end
            
            covered_cells = sortrows(covered_cells, [1 2]);
            
            id = find(cellfun(@(x) isequal(x, covered_cells), ...
                bbox_covered_regions));
            
            if isempty(id)
                bbox_covered_regions{length(bbox_covered_regions)+1} = ...
                    covered_cells;
                bboxes = [bboxes ; x y overlap_quality ...
                    length(bbox_covered_regions)];
            else
                bboxes = [bboxes ; x y overlap_quality id];
            end
        end
    end
end


%% pick the best bbox per bbox_covered_regions
keep = false(size(bboxes,1),1);
for s=1:length(bbox_covered_regions)
    % retrieve all bboxes
    ids = find(bboxes(:,4)==s);
    [~,id] = max(bboxes(ids,3));
    keep(ids(id)) = true;
end
bboxes = bboxes(keep,:);


%% are there any individual regions which is not covered by any group in 
% bbox_covered_regions? If yes, include the central bbox for each of them 
for a=1:size(filter.regions,1)
    for e=1:size(filter.regions,2)
        if ~any(cellfun(@(x) any(x(:,1)==a & x(:,2)==e), ...
                bbox_covered_regions))
            % get the pooling region
            region = squeeze(filter.regions(a,e,:,:));
            [py,px] = find(region>0);
            
            % center of the region
            cntr = [mean(px) mean(py)];
            
            % best (centered) bbox
            x = round((cntr(1) - central_cell_x - t_w*bin_length/2)/ ...
                bin_length);
            y = round((cntr(2) - central_cell_y - t_h*bin_length/2)/ ...
                bin_length);
            
            pixs_x = central_cell_x + x*bin_length + ...
                (0:t_w*bin_length-1);
            pixs_y = central_cell_y + y*bin_length + ...
                (0:t_h*bin_length-1);
            
            if any(pixs_x>size(W,2)) || any(pixs_y>size(W,1)) || ...
                    any(pixs_x<1) || any(pixs_y<1)
                break
            end
            
            % add the pooling region and the bbox
            bbox_covered_regions{length(bbox_covered_regions)+1} = [a e];
            bboxes = [bboxes ; x y 1 length(bbox_covered_regions)];
        end
    end
end

if visualize_boxes
    figure, imagesc(W);
    hold on
    for i=1:size(bboxes,1)        
        h=rectangle('position', ...
            [bboxes(i,1)*bin_length+central_cell_x ...
            bboxes(i,2)*bin_length+central_cell_y ...
            bin_length*t_w bin_length*t_h], ...
            'edgecolor', 'w','linewidth',2);   
        drawnow
        %pause
        %delete(h);
    end
end
