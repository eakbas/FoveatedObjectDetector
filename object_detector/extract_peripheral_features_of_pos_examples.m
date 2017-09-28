function [pos_examples, no_overlaps] = ...
    extract_peripheral_features_of_pos_examples(pos_examples, ...
    model, min_gt_bbox_overlap, verbosity)
% Extracts features of positive examples. Bounding box space (scales and
%   locations) is searched over to find good overlapping ones with the ground
%   truth bounding box. At each (fixation) location, peripheral bounding boxes
%   are also considered. 


% identify unique aspect ratios
asp_ratios = unique([model.templates.height; model.templates.width]','rows');

no_overlaps = false(length(pos_examples),1);

% number of cells to pad the features image
padx = model.padx;
pady = model.pady;

npos = length(pos_examples);

if verbosity>0
    textprogressbar('Extracting features of positive examples...');
end
for i=1:length(pos_examples)
    img = imread(pos_examples(i).imgfilename);

    pos_examples(i).features = [];
    pos_examples(i).features_id = [];
    pos_examples(i).features_metadata.templates = [];
    pos_examples(i).features_metadata.scales = [];
    pos_examples(i).features_metadata.flipped = [];
    pos_examples(i).features_metadata.locations = [];
    pos_examples(i).features_metadata.overlaps = [];
    pos_examples(i).features_metadata.fixation = [];
  
    ref_width = round(2*padx + size(img,2)/model.bin_length);
    ref_height = round(2*pady + size(img,1)/model.bin_length);
    
    for s=1:length(model.scales)
        if verbosity>0
            % update progress bar
            textprogressbar(100*((i-1)*length(model.scales)+s)/...
                (length(model.scales)*npos));        
        end

       
        % resize image and extract features
        rezimg = imresize(img, model.scales(s));        
        orig_feats = features(double(rezimg), model.bin_length);
        orig_feats = orig_feats(:,:,1:31);

        % pad the feature map with zeros w.r.t. the template's size
        img_feats = padfeatures(orig_feats, pady+1, padx+1);


        % see if there is any good overlap with the ground truth bounding box        
        for a=1:size(asp_ratios,1) % {
            th = asp_ratios(a,1);
            tw = asp_ratios(a,2);

            gos = find_good_overlaps(img_feats, ...
                padx, pady, ...
                round(pos_examples(i).gt_bbox*model.scales(s)), ...
                tw, th, ...
                model.bin_length, ...
                min_gt_bbox_overlap);

            if isempty(gos)
                continue
            end

            % for each good overlap, see which templates' bboxes align
            for g=1:size(gos,1); for t=1:length(model.templates)
                if model.templates(t).height~=th || model.templates(t).width~=tw
                    continue
                end

                % original features
                ys = gos(g,2)+(0:model.templates(t).height-1);
                xs = gos(g,1)+(0:model.templates(t).width-1);                                                       
                feat = img_feats(ys,xs,:);

                if model.templates(t).foveal
                    % this is a foveal template
                    
                    % store the feature vector
                    features_id = length(pos_examples(i).features)+1;
                    pos_examples(i).features{ ...
                        length(pos_examples(i).features)+1,1} = feat(:)';
                    pos_examples(i).features_id = [ ...
                        pos_examples(i).features_id ; ...
                        features_id];
                    pos_examples(i).features_metadata.templates = [ ...
                        pos_examples(i).features_metadata.templates ; t];
                    pos_examples(i).features_metadata.scales = [ ...
                        pos_examples(i).features_metadata.scales ; s];
                    pos_examples(i).features_metadata.locations = [ ...
                        pos_examples(i).features_metadata.locations ; ...
                        gos(g,1) gos(g,2)];
                    pos_examples(i).features_metadata.overlaps = [ ...
                        pos_examples(i).features_metadata.overlaps ; ...
                        gos(g,3)];
                    pos_examples(i).features_metadata.flipped = [ ...
                        pos_examples(i).features_metadata.flipped ; ...
                        false];
                    pos_examples(i).features_metadata.fixation = [ ...
                        pos_examples(i).features_metadata.fixation ; ...
                        -1 -1];

                    if model.use_left_right_flipping
                        ffeat = flipfeat(feat);

                        flipped_features_id = length(pos_examples(i).features)+1;
                        pos_examples(i).features{ ...
                            length(pos_examples(i).features)+1,1} = ffeat(:)';
                        pos_examples(i).features_id = [ ...
                            pos_examples(i).features_id ; ...
                            flipped_features_id];
                        pos_examples(i).features_metadata.templates = [ ...
                            pos_examples(i).features_metadata.templates ; t];
                        pos_examples(i).features_metadata.scales = [ ...
                            pos_examples(i).features_metadata.scales ; s];
                        pos_examples(i).features_metadata.locations = [ ...
                            pos_examples(i).features_metadata.locations ; ...
                            gos(g,1) gos(g,2)];
                        pos_examples(i).features_metadata.overlaps = [ ...
                            pos_examples(i).features_metadata.overlaps ; ...
                            gos(g,3)];
                        pos_examples(i).features_metadata.flipped = [ ...
                            pos_examples(i).features_metadata.flipped ; ...
                            true];
                        pos_examples(i).features_metadata.fixation = [ ...
                            pos_examples(i).features_metadata.fixation ; ...
                            -1 -1];
                    end
                else
                    % peripheral template

                    % fixation location:
                    fx = gos(g,1) - model.templates(t).bboxes_left_top(1,1);
                    fy = gos(g,2) - model.templates(t).bboxes_left_top(1,2);

                    % is the fixation location outside the image? 
                    if fx<1 || fy<1 || fx>size(img_feats,2) || ...
                        fy>size(img_feats,1)
                        continue
                    end

                    % store
                    tfeat = model.templates(t).ftransform{1}*reshape( ...
                        feat, [size(feat,1)*size(feat,2) size(feat,3)]);               
                   
                    % store the feature vector
                    pos_examples(i).features{ ...
                        length(pos_examples(i).features)+1,1} = [];
                    pos_examples(i).features_id = [ ...
                        pos_examples(i).features_id ; ...
                        features_id];
                    pos_examples(i).features_metadata.templates = [ ...
                        pos_examples(i).features_metadata.templates ; t];
                    pos_examples(i).features_metadata.scales = [ ...
                        pos_examples(i).features_metadata.scales ; s];
                    pos_examples(i).features_metadata.locations = [ ...
                        pos_examples(i).features_metadata.locations ; ...
                        gos(g,1) gos(g,2)];
                    pos_examples(i).features_metadata.overlaps = [ ...
                        pos_examples(i).features_metadata.overlaps ; ...
                        gos(g,3)];
                    pos_examples(i).features_metadata.flipped = [ ...
                        pos_examples(i).features_metadata.flipped ; ...
                        false];
                    pos_examples(i).features_metadata.fixation = [ ...
                        pos_examples(i).features_metadata.fixation ; ...
                        round(ref_width*(fx+0.5)/size(img_feats,2)-0.5) ...
                        round(ref_height*(fy+0.5)/size(img_feats,1)-0.5)];

                    if model.use_left_right_flipping
                        pos_examples(i).features{ ...
                            length(pos_examples(i).features)+1,1} = [];
                        pos_examples(i).features_id = [ ...
                            pos_examples(i).features_id ; ...
                            flipped_features_id];
                        pos_examples(i).features_metadata.templates = [ ...
                            pos_examples(i).features_metadata.templates ; t];
                        pos_examples(i).features_metadata.scales = [ ...
                            pos_examples(i).features_metadata.scales ; s];
                        pos_examples(i).features_metadata.locations = [ ...
                            pos_examples(i).features_metadata.locations ; ...
                            gos(g,1) gos(g,2)];
                        pos_examples(i).features_metadata.overlaps = [ ...
                            pos_examples(i).features_metadata.overlaps ; ...
                            gos(g,3)];
                        pos_examples(i).features_metadata.flipped = [ ...
                            pos_examples(i).features_metadata.flipped ; ...
                            true];
                        pos_examples(i).features_metadata.fixation = [ ...
                            pos_examples(i).features_metadata.fixation ; ...
                            round(ref_width*(fx+0.5)/size(img_feats,2)-0.5) ...
                            round(ref_height*(fy+0.5)/size(img_feats,1)-0.5)];                        
                    end
                end
            end; end
        end % }     
    end
    
    if isempty(pos_examples(i).features)
        % no good overlaps
        no_overlaps(i) = true;
    end
end
if verbosity>0
    textprogressbar(-1);
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
