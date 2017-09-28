function [confidences, bboxes, fixation_locations] = ...
    detect_FOD(img, model, strategy, ...
    initial_fixation, num_fixations)
% Runs the given foveated model on the given image. 
%
%   [C,BB,F] = detect_FOD(IMG, MODEL, STRATEGY, INITIAL_FIXATION,
%   NUM_FIXATIONS) runs MODEL on IMG and returns the results in C, BB and
%   F. C is N-by-1 vector containing the confidence scores of the
%   detections. BB is a 4-by-N matrix containing the coordinates of the
%   bounding boxes of the detections. F is a NUM_FIXATIONS-by-2 matrix
%   containing the coordinates of the locations the model fixated. Use
%   "show_fixations_and_detections" to visualize the detections.
%
%   STRATEGY is one of 'MAP', 'MAP_IOR' or 'RAND'. Both 'MAP' and 'MAP_IOR'
%   implement the MAP rule but the latter also does inhibition of return. A
%   2 degree-radius area is inhibited around each fixation location. 'RAND'
%   tells the model to start with a randomly selected initial fixation
%   location. 
%
%   INITIAL_FIXATION is a 1-by-2 vector specifiying the initial fixation
%   location in normalized image coordinates. For example, [.5 .5] means
%   the center of the image. The top-left of the image is [0 0] and
%   bottom-right is [1 1]. 
%
%   NUM_FIXATIONS is an integer that specifies the number of fixations
%   (including the initial fixation) you want the model to make. 
% 
fixation_locations = NaN(num_fixations, 2);

if isempty(initial_fixation)     
    if strcmp(strategy, 'RAND')
        fixation_locations(1,:) = rand(1,2);
    elseif strcmp(strategy, 'RAND-C')
        fixation_locations(1,:) = [.5 .5];
    elseif strcmp(strategy, 'RAND-E')
        if rand>.5
            fixation_locations(1,:) = [.1 .5];
        else
            fixation_locations(1,:) = [.9 .5];
        end
        strategy = 'RAND';
    elseif strcmp(strategy, 'MAP_IOR_L')
        fixation_locations(1,:) = [.1 .5];
        strategy = 'MAP_IOR';
    elseif strcmp(strategy, 'MAP_IOR_R')
        fixation_locations(1,:) = [.9 .5];
        strategy = 'MAP_IOR';
    else
        % default initial fixation is at the center
        fixation_locations(1,:) = [.5 .5];
    end
else
    fixation_locations(1,:) = initial_fixation;
end

% template shapes
template_shapes = unique([model.templates.height; model.templates.width]','rows');

feats = []; 
aggregate_responses = [];
num_responses = [];

for f=1:num_fixations
    % fixate and collect template responses
    [aggregate_responses, num_responses, feats] = fixate_FOD(img, feats, ...
        fixation_locations(f,1), fixation_locations(f,2), ...
        model, template_shapes, ...
        aggregate_responses, num_responses);
    
    if f==num_fixations
        break
    end
    
    % choose next fixation point
    if strcmp(strategy, 'MAP')
        % find the location with maximal posterior prob
        [~,ind] = max(aggregate_responses(:));
        
        [s,ly,lx,sh] = ind2sub(size(aggregate_responses), ind);
        
        % best responding bbox location
        bbx = [ (lx-1-feats.padx)*model.bin_length+1; ...
            (ly-1-feats.pady)*model.bin_length+1; ...
            (lx-1-feats.padx+template_shapes(sh,2))*model.bin_length ; ...
            (ly-1-feats.pady+template_shapes(sh,1))*model.bin_length] ...
            ./model.scales(s);
        
        % find this bbox center
        center = [0.5*(bbx(1)+bbx(3)) 0.5*(bbx(2)+bbx(4))];
        
        % compute its normalized coordinates
        fixation_locations(f+1,:) = center./[size(img,2) size(img,1)];
    elseif strcmp(strategy, 'MAP_IOR')
        % sort the posteriors
        [~,inds] = sort(-aggregate_responses(:));

        [s,ly,lx,sh] = ind2sub(size(aggregate_responses), inds);

        for ii=1:length(s)
            % bbox location
            bbx = [ (lx(ii)-1-feats.padx)*model.bin_length+1; ...
                (ly(ii)-1-feats.pady)*model.bin_length+1; ...
                (lx(ii)-1-feats.padx+template_shapes(sh(ii),2))*model.bin_length ; ...
                (ly(ii)-1-feats.pady+template_shapes(sh(ii),1))*model.bin_length] ...
                ./model.scales(s(ii));
            
            % find this bbox center
            center = [0.5*(bbx(1)+bbx(3)) 0.5*(bbx(2)+bbx(4))];
            
            % compute its normalized coordinates
            loc = center./[size(img,2) size(img,1)];

            % is this location inhibited?            
            dists = sqrt( ...   % compute distance in terms of img pixels
                (size(img,2)*(fixation_locations(1:f,1)-loc(1))).^2+...
                (size(img,1)*(fixation_locations(1:f,2)-loc(2))).^2);
            if ~any(dists<6*model.bin_length)
                % not inhibited => good, new fixation point found, break.
                break
            end
        end
        if ii==length(s)
            % all locations are inhibited!
            break
        end
        fixation_locations(f+1,:) = loc;
    elseif strcmp(strategy, 'approxIS')
    elseif strcmp(strategy, 'RAND') || strcmp(strategy, 'RAND-C')
        fixation_locations(f+1,:) = rand(1,2);
    elseif strcmp(strategy, 'Systematic')
    else
        error(['Unknown eye movement strategy: ''' strategy '''']);
    end
end


%% transform aggregate responses to bboxes with confidences
inds = find(~isnan(aggregate_responses));
[s,ly,lx,sh] = ind2sub(size(aggregate_responses), inds);

confidences = aggregate_responses(inds);
bboxes = bsxfun(@times, [(lx'-1-feats.padx)*model.bin_length+1; ...
            (ly'-1-feats.pady)*model.bin_length+1; ...
            (lx'-1-feats.padx+template_shapes(sh,2)')*model.bin_length ; ...
            (ly'-1-feats.pady+template_shapes(sh,1)')*model.bin_length], ...
            1./model.scales(s)');

% bboxes2 = zeros(4, length(s));
% for i=1:length(inds)
%     bboxes2(:,i) = [(lx(i)-1-feats.padx)*conf.bin_length+1; ...
%             (ly(i)-1-feats.pady)*conf.bin_length+1; ...
%             (lx(i)-1-feats.padx+template_shapes(sh(i),2))*conf.bin_length ; ...
%             (ly(i)-1-feats.pady+template_shapes(sh(i),1))*conf.bin_length]./ ...
%             conf.scales(s(i));
% end
% max(abs(bboxes(:)-bboxes2(:))) % difference is too small

%% apply the detection threshold
valids = confidences>-Inf;
confidences = confidences(valids);
bboxes = bboxes(:,valids);

%% clip detection boxes that overflow image borders
bboxes(1,:) = max(bboxes(1,:), 1);
bboxes(2,:) = max(bboxes(2,:), 1);
bboxes(3,:) = min(bboxes(3,:), size(img, 2));
bboxes(4,:) = min(bboxes(4,:), size(img, 1));

% remove invalid detections
w = bboxes(3,:)-bboxes(1,:)+1;
h = bboxes(4,:)-bboxes(2,:)+1;
valids = find((w > 0) & (h > 0));
bboxes = bboxes(:,valids);
confidences = confidences(valids);


%% non-maxima supression
pick = nms([bboxes' confidences], .5);
confidences = confidences(pick);
bboxes = bboxes(:,pick);
%[confidences,idx] = sort(confidences, 'descend');
%bboxes = bboxes(:,idx);
