function [pos_examples, no_overlaps] = ...
    extract_features_of_pos_examples(pos_examples, model, ...
    min_gt_bbox_overlap, verbosity)
% Extracts features of positive examples. Bounding box space (scales and
%   locations) is searched over to find good overlapping ones with the ground
%   truth bounding box. 

% THIS VERSION DOES NOT USE THE BOUNDARY TRUNCATION DIM (31 features)

no_overlaps = false(length(pos_examples),1);

times = zeros(size(model.scales));

if verbosity>0
    gt0 = clock;
    textprogressbar('Extracting features of positive examples...');
end
for i=1:length(pos_examples)
    img = imread(pos_examples(i).imgfilename);
    
    pos_examples(i).features = [];
    pos_examples(i).features_metadata.templates = [];
    pos_examples(i).features_metadata.scales = [];
    pos_examples(i).features_metadata.flipped = [];
    pos_examples(i).features_metadata.locations = [];
    pos_examples(i).features_metadata.overlaps = [];
        
    for s=1:length(model.scales)
        t0 = clock;
        rezimg = imresize(img, model.scales(s));
        
        orig_feats = features(double(rezimg), model.bin_length);
        orig_feats = orig_feats(:,:,1:31);
        
        for t=1:length(model.templates)            
            th = model.templates(t).height;
            tw = model.templates(t).width;

            % pad the feature map with zeros w.r.t. the template's size
            feats = padfeatures(orig_feats, model.pady+1, model.padx+1);

            % compute overlap scores for all possible bounding box
            % locations
            gos = find_good_overlaps(feats, model.padx, model.pady, ...
                round(model.scales(s)*pos_examples(i).gt_bbox), ...
                tw,th, model.bin_length, ...
                min_gt_bbox_overlap);
            
            if ~isempty(gos)                
                % go over all overlaps
                for p=1:size(gos,1)
                    ys = gos(p,2)+(0:model.templates(t).height-1);
                    xs = gos(p,1)+(0:model.templates(t).width-1);                                                       
                    
                    f = feats(ys,xs,:);
                    
                    % store the feature vector
                    pos_examples(i).features{ ...
                        length(pos_examples(i).features)+1,1} = f(:)';
                    pos_examples(i).features_metadata.templates = [ ...
                        pos_examples(i).features_metadata.templates ; t];
                    pos_examples(i).features_metadata.scales = [ ...
                        pos_examples(i).features_metadata.scales ; s];
                    pos_examples(i).features_metadata.locations = [ ...
                        pos_examples(i).features_metadata.locations ; ...
                        gos(p,1) gos(p,2)];
                    pos_examples(i).features_metadata.overlaps = [ ...
                        pos_examples(i).features_metadata.overlaps ; ...
                        gos(p,3)];
                    pos_examples(i).features_metadata.flipped = [ ...
                        pos_examples(i).features_metadata.flipped ; ...
                        false];

                    if model.use_left_right_flipping
                        f = flipfeat(f);

                        pos_examples(i).features{ ...
                            length(pos_examples(i).features)+1,1} = f(:)';
                        pos_examples(i).features_metadata.templates = [ ...
                            pos_examples(i).features_metadata.templates ; t];
                        pos_examples(i).features_metadata.scales = [ ...
                            pos_examples(i).features_metadata.scales ; s];
                        pos_examples(i).features_metadata.locations = [ ...
                            pos_examples(i).features_metadata.locations ; ...
                            gos(p,1) gos(p,2)];
                        pos_examples(i).features_metadata.overlaps = [ ...
                            pos_examples(i).features_metadata.overlaps ; ...
                            gos(p,3)];
                        pos_examples(i).features_metadata.flipped = [ ...
                            pos_examples(i).features_metadata.flipped ; ...
                            true];
                    end
                end
            end
        end
        times(s) = times(s) + etime(clock, t0);
    end
    
    if isempty(pos_examples(i).features)
        % no good overlaps
        no_overlaps(i) = true;
    end
    
    if verbosity>0
        textprogressbar(100*i/length(pos_examples));
    end
end

if verbosity>0
    textprogressbar(-1);
    fprintf(1,'took %.2f minutes.\n', etime(clock,gt0)/60);
end
