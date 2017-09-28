function good_overlaps = find_good_overlaps(feats, padx, pady, bbox, ...
    tw, th, bin_length, threshold)



[x,y] = meshgrid(1:size(feats,2)-tw+1, 1:size(feats,1)-th+1);
lx = (x-1-padx)*bin_length+1; % left x
ty = (y-1-pady)*bin_length+1; % top y
rx = (x-1-padx+tw)*bin_length; % right x
by = (y-1-pady+th)*bin_length; % bottom y
inter_width = max(0, min(rx,bbox(3))-max(lx,bbox(1))+1);
inter_height = max(0, min(by,bbox(4))-max(ty,bbox(2))+1);
inter_area = inter_width.*inter_height;
union_area = (by-ty+1).*(rx-lx+1) + ...
    (bbox(4)-bbox(2)+1)*(bbox(3)-bbox(1)+1) - ...
    inter_area;

overlap = inter_area ./ union_area;
goods = overlap>=threshold;
good_overlaps = [x(goods) y(goods) overlap(goods)];



% good_overlaps1 = [];
% 
% for y=1:size(feats,1)
%     for x=1:size(feats,2)
%         % compute the template bounding box, whose top-left cell is at
%         % (x,y)
%         lx = (x-1-padx)*bin_length+1; % left x
%         ty = (y-1-pady)*bin_length+1; % top y
%         rx = (x-1-padx+tw)*bin_length; % right x
%         by = (y-1-pady+th)*bin_length; % bottom y
% 
% %         % clip the bounding box so that it doesn't overflow the image
% %         % borders
% %         lx = max(1,lx);
% %         ty = max(1,ty);
% %         rx = min(imsize(2), rx);
% %         by = min(imsize(1), by);
%         
%         % compute overlap
%         inter_width = max(0, min(rx,bbox(3))-max(lx,bbox(1))+1);
%         inter_height = max(0, min(by,bbox(4))-max(ty,bbox(2))+1);
%         inter_area = inter_width*inter_height;
%         
%         union_area = (by-ty+1)*(rx-lx+1) + ...
%             (bbox(4)-bbox(2)+1)*(bbox(3)-bbox(1)+1) - ...
%             inter_area;
%         
%         overlap = inter_area / union_area;
%         
%         if overlap>=threshold
%             good_overlaps1 = [good_overlaps1 ; x y overlap];
%         end
%     end
% end


% if (isempty(good_overlaps) && isempty(good_overlaps1)) || ...
%     isequal(sortrows(good_overlaps,[1 2]), good_overlaps1)
% else
%     error('not equal');
% end