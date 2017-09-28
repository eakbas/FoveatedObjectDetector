function [C,BB] = detect_SW(img, model, do_nms)
% Runs the given sliding window model on the given image. 
%
%   [C,BB] = detect_SW(IMG, MODEL) runs MODEL on IMG and returns the
%   results in C and BB. C is N-by-1 vector containing the confidence
%   scores of the detections. BB is a 4-by-N matrix containing the
%   coordinates of the bounding boxes of the detections. Use
%   "show_bounding_boxes" to visualize the detections. 
% 
C = [];
BB = [];

if nargin<3
    do_nms = true;
end

% search over scales
for s=1:length(model.scales)
    feats = features(double(imresize(img,model.scales(s))), ...
        model.bin_length);
    feats = feats(:,:,1:31);
    feats = padfeatures(feats, model.pady+1, model.padx+1);
    
    % apply all templates at all locations
    for t=1:length(model.templates)
        R = compute_template_responses(feats, [],...
            model.templates(t), ...
            true);
        
        % z-score normalization
        %R = (R-templates(t).mu)/templates(t).std;
        
        %         % percentile rank
        %         R = round(10*R);
        %         R(R<1) = 1;
        %         R(R>length(templates(t).prctile)) = ...
        %             length(templates(t).prctile);
        %         R = templates(t).prctile(R);
        
        
        
        % find high response locations
        idx = find(R>model.templates(t).threshold);
        idx = idx(:);
        
        [y,x] = ind2sub(size(R), idx);
        
        % record scores
        R = R(idx);        
        R = model.templates(t).A*R+model.templates(t).B;
        C = [C ; R(:)];
        
        % record bounding boxes
        bboxes = [ (x'-1-model.padx)*model.bin_length+1; ...
            (y'-1-model.pady)*model.bin_length+1; ...
            (x'-1-model.padx+model.templates(t).width)*...
            model.bin_length ; ...
            (y'-1-model.pady+model.templates(t).height)*...
            model.bin_length] ...
            ./model.scales(s);
        BB = [BB , bboxes];
    end
end


%% clip detection boxes that overflow image borders
BB(1,:) = max(BB(1,:), 1);
BB(2,:) = max(BB(2,:), 1);
BB(3,:) = min(BB(3,:), size(img, 2));
BB(4,:) = min(BB(4,:), size(img, 1));

% remove invalid detections
w = BB(3,:)-BB(1,:)+1;
h = BB(4,:)-BB(2,:)+1;
valids = find((w > 0) & (h > 0));
BB = BB(:,valids);
C = C(valids);


%% non-maxima supression
if do_nms
    pick = nms([BB' C], .5);
    C = C(pick);
    BB = BB(:,pick);
end



