function [widths, heights, bbox_asp_ratio_idxs] = ...
    determine_template_sizes(bboxes, N, ...
    bin_length, min_pixel_area, max_pixel_area, max_dim_in_cells, ...
    representative_area_percentile)

% compute aspect ratios of bboxes
bbox_aspects = (bboxes(:,4)-bboxes(:,2)+1)./(bboxes(:,3)-bboxes(:,1)+1);

sorted_aspects = sort(bbox_aspects);

bbox_asp_ratio_idxs = zeros(length(bbox_aspects),1);

template_aspects = zeros(N,1);

% split into N equalsize chunks
split_inds = round(linspace(1,length(sorted_aspects), N+1));
sorted_aspects(end) = Inf;
for t=1:N
    I = bbox_aspects>=sorted_aspects(split_inds(t)) & ...
        bbox_aspects<sorted_aspects(split_inds(t+1));
    bbox_asp_ratio_idxs(I) = t;
    template_aspects(t) = median(bbox_aspects(I));
end

% areas
bbox_areas = (bboxes(:,4)-bboxes(:,2)+1).*(bboxes(:,3)-bboxes(:,1)+1);

widths = [];
heights = [];
for t=1:N
    a = sort(bbox_areas(bbox_asp_ratio_idxs==t));
    area = a(floor((representative_area_percentile/100)*length(a)));
    area = max(min(area, max_pixel_area), min_pixel_area);
    
    widths(t) = min(max_dim_in_cells*bin_length, ...
        sqrt(area/template_aspects(t)));
    heights(t) = template_aspects(t)*widths(t);
    
    if max(widths(t),heights(t))>max_dim_in_cells*bin_length
         heights(t) = min(max_dim_in_cells*bin_length, ...
             sqrt(area*template_aspects(t)));
         widths(t) = heights(t)/template_aspects(t);    
    end
    
    widths(t) = round(widths(t)/bin_length);
    heights(t) = round(heights(t)/bin_length);
end


% %% determine the best set of scales
% scales = linspace(min_scaling_factor, max_scaling_factor, 1000);
% % scale_votes = zeros(length(bbox_areas), length(scales));
% bbox_widths = (bboxes(:,3)-bboxes(:,1)+1);
% bbox_heights = (bboxes(:,4)-bboxes(:,2)+1);
% allvotes = cell(length(scales),1);
% for s=1:length(scales)
%     num_votes = 0;
%     for t=1:N
%         inters = min(widths(t)*bin_length, scales(s)*bbox_widths).* ...
%             min(heights(t)*bin_length, scales(s)*bbox_heights);
%         overlaps = inters./((widths(t)*heights(t)*bin_length*bin_length) + ...
%             (scales(s)^2.*bbox_widths.*bbox_heights) - inters);
%         %         scale_votes(overlaps>.7*1.05,s) = ...
%         %             scale_votes(overlaps>.7*1.05,s) + 1;
%         num_votes = num_votes + sum(overlaps>.7*1.05);
%     end
%     allvotes{s} = ones(num_votes,1)*scales(s);
% end
% allvotes = cell2mat(allvotes);
% [~,scales] = kmeans(allvotes, num_scales, 'emptyaction', 'singleton');
% scales = sort(scales);
% % choose scales
% csum = cumsum(sum(scale_votes));
% stepsize = max(csum)/num_scales;
% current = stepsize/2;
% chosen_scales = false(size(scales));
% for i=1:num_scales
%     id = find(csum>current,1);
%     chosen_scales(id) = true;
%     
%     current = current + stepsize;
% end
% scales = scales(chosen_scales)';