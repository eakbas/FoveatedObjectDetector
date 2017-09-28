function W = get_peripheral_pooling_coeffs(filter, bin_length, regions, ...
    bbox, t_h, t_w, fovea_left_top, fovea_right_bottom)

%% top left of the central cell (0,0)
central_cell_x = round(size(filter.regions,4)/2 - bin_length/2);
central_cell_y = round(size(filter.regions,3)/2 - bin_length/2);


pixs_x = central_cell_x + bbox(1)*bin_length + ...
    (0:t_w*bin_length-1);
pixs_y = central_cell_y + bbox(2)*bin_length + ...
    (0:t_h*bin_length-1);

part = filter.regions(:,:,pixs_y,pixs_x);
        


W = [];

% discretize each active region to square cell
for r=1:size(regions,1)
    w = zeros(t_h, t_w);
    for i=1:t_h
        for j=1:t_w
            x=squeeze(part(regions(r,1),regions(r,2),:,:));
            w(i,j) = sum(sum(x((i-1)*bin_length+1:i*bin_length, ...
                (j-1)*bin_length+1:j*bin_length)));            
        end
        
    end
    w = w/sum(sum(w));
    W(r,:) = w(:)';
end

% % any foveal region included? 
% [cell_x,cell_y] = meshgrid(bbox(1)+(0:t_w-1), bbox(2)+(0:t_h-1));
% foveal = cell_x>=fovea_left_top(1) & cell_x<=fovea_right_bottom(1) & ...
%     cell_y>=fovea_left_top(2) & cell_y<=fovea_right_bottom(2);
% if any(foveal(:))
%     ids = find(foveal(:));
%     for i=1:length(ids)
%         w = false(size(foveal));
%         w(ids(i)) = 1;
%         W = [W ; w(:)'];
%     end
% end