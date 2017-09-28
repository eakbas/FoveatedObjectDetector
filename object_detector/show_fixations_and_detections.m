function show_fixations_and_detections(img, BB, scores, f, draw_lines)

if ~isempty(scores)
    show_scores = true;
else
    show_scores = false;
end

if nargin<5
    draw_lines = true;
end

imshow(img);
hold on

for i=1:size(BB,2)
    b = BB(:,i);
    h=rectangle('position', ...
        [b(1) b(2) b(3)-b(1) b(4)-b(2)], ...
        'edgecolor', 'y', 'linewidth',3);
    
    if show_scores
        h2=text(b(1)+.5, b(2)+12, num2str(scores(i)),'background', 'cyan');
    end
%     pause
%     delete(h)
%     delete(h2);
end

hold on


%show fixations
plot(f(:,1)*size(img,2), f(:,2)*size(img,1), 'y.', ...
    'markersize',20, 'linewidth',3);
if draw_lines
    line(f(:,1)*size(img,2), f(:,2)*size(img,1), ...
        'linestyle', '--', 'color', 'y', 'linewidth', 3);
end
% plot(f(:,1)*size(img,2), f(:,2)*size(img,1), 'w--', ...
%     'linewidth',2);


for i=1:size(f,1)
    text(f(i,1)*size(img,2)+5, f(i,2)*size(img,1)-5, ...
     num2str(i),'color', 'y', 'fontsize',20)
end