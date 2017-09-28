function show_bounding_boxes(img, BB, scores)
% Visualizes the given bounding boxes and their confidence scores. 
%   SHOW_BOUNDING_BOXES(IMG, BB, C) 
if nargin==3
    show_scores = true;
else
    show_scores = false;
end

imshow(img);
hold on

for i=1:size(BB,2)
    b = BB(:,i);
    h=rectangle('position', [b(1) b(2) b(3)-b(1) b(4)-b(2)], ...
        'edgecolor', rand(1,3), 'linewidth',2);
    
    if show_scores
        h2=text(b(1)+.5, b(2)+12, num2str(scores(i)),'background', 'cyan');
    end
%     pause
%     delete(h)
%     delete(h2);
end