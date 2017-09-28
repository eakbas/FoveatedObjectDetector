function model = calibrate_model_given_pos_features(model, dataset, num_neg_images, pos_features)

% choose a subset of dataset
ids = [find(~dataset.imglabels,num_neg_images)];


% use a small portion of this subset to estimate the maximal responses for
% each template

% run model on this subset
all_max_scores = zeros(length(model.templates), round(.1*length(ids)));
parfor i=1:round(.1*length(ids))
    img = imread(fullfile(dataset.dir, dataset.imgnames{ids(i)}));
    
    max_scores = -Inf(length(model.templates),1);
    for s=1:length(model.scales)
        % extract features
        feats = features(double(imresize(img,model.scales(s))), ...
            model.bin_length);
        feats = feats(:,:,1:31);
        feats = padfeatures(feats, model.pady+1, model.padx+1);

        for t=1:length(model.templates)
            % evaluate template
            R = compute_template_responses(feats, [], ...
                model.templates(t), true);

            if ~isempty(R)
                max_scores(t) = max(max_scores(t), max(R(:)));
            end
        end % t, templates        
    end % s, scales    
    
    all_max_scores(:,i) = max_scores;
end % i, ids
all_max_scores = max(all_max_scores,[],2);

% create response histogram centers for each template
centers = zeros(length(model.templates), 1000);
for t=1:length(model.templates)
    centers(t, :) = linspace(model.templates(t).threshold, ...
        all_max_scores(t), 1000);
end


% run model on the subset
all_pos_scores = cell(length(ids),1);
all_neg_scores = cell(length(ids),1);
parfor i=1:length(ids)
    all_pos_scores{i} = cell(length(model.templates),1);
    all_neg_scores{i} = cell(length(model.templates),2);
    
    img = imread(fullfile(dataset.dir, dataset.imgnames{ids(i)}));

    posh = zeros(size(centers));
    negh = zeros(size(centers));
    
    for s=1:length(model.scales)
        % extract features
        feats = features(double(imresize(img,model.scales(s))), ...
            model.bin_length);
        feats = feats(:,:,1:31);
        feats = padfeatures(feats, model.pady+1, model.padx+1);

        for t=1:length(model.templates)
            % evaluate template
            R = compute_template_responses(feats, [], ...
                model.templates(t), true);

            idx = find(R>model.templates(t).threshold);
            idx = idx(:);
            
            [y,x] = ind2sub(size(R), idx);
            R = R(idx);
            C = R;

            BB = [ (x'-1-model.padx)*model.bin_length+1; ...
                (y'-1-model.pady)*model.bin_length+1; ...
                (x'-1-model.padx+model.templates(t).width)*...
                model.bin_length ; ...
                (y'-1-model.pady+model.templates(t).height)*...
                model.bin_length] ...
                ./model.scales(s);

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

            % evaluate boxes
            gt = dataset.bboxes{ids(i)};

            if isempty(gt)
                labels = false(length(C),1);
            else
                labels = false(length(C),1);
                for g=1:size(gt,1)
                    inter_width = max(0, min(BB(3,:),gt(g,3))-max(BB(1,:),gt(g,1)));
                    inter_height = max(0, min(BB(4,:),gt(g,4))-max(BB(2,:),gt(g,2)));
                    inter_area = inter_width.*inter_height;
                    
                    union_area = (BB(4,:)-BB(2,:)+1).*(BB(3,:)-BB(1,:)+1) + ...
                        (gt(g,4)-gt(g,2)+1)*(gt(g,3)-gt(g,1)+1) - ...
                        inter_area;
                    
                    overlaps = inter_area ./ union_area;
                    
                    labels(overlaps>.5) = true;
                end
            end
           
            % store the responses 
            posh(t,:) = posh(t,:) + hist(C(labels), centers(t,:));
            negh(t,:) = negh(t,:) + hist(C(~labels), centers(t,:));
        end % t, templates        
    end % s, scales    

   % store the responses and their labels
    for t=1:length(model.templates)
        all_pos_scores{i}{t} = posh(t,:);
        all_neg_scores{i}{t} = negh(t,:);
    end
end % i, ids


% responses of positive features
posh = zeros(size(centers));
for t=1:length(model.templates)
    resps = pos_features{t}*model.templates(t).w;
    posh(t,:) = hist(resps, centers(t,:));
end


% combine scores of negative examples
negh = zeros(size(centers));
for i=1:length(ids)
    for t=1:length(model.templates)
        negh(t,:) = negh(t,:) + all_neg_scores{i}{t};
    end
end


% fit sigmoid
for t=1:length(model.templates)
%     cost = @(A) sum(posh.* ...                                                                     
%          (1./(1+exp(-(A(1)*centers+A(2)))) - 1).^2) + ...                                              
%          sum(negh.* ...                                                                             
%          (1./(1+exp(-(A(1)*centers+A(2)))) - 0).^2);   
cost = @(A) sigmoid_cost(posh(t,:),negh(t,:),centers(t,:), A);

     xstar = fminsearch(cost, [0.1 -1]);%, optimset('display','final')); 
     model.templates(t).A = xstar(1);
     model.templates(t).B = xstar(2);
end


% % modify threshold
% T=0;
% for t=1:length(model.templates)
%     T = T +  (model.templates(t).A*model.threshold + ...
%         model.templates(t).B);
% end
% model.threshold = T/length(model.templates);


% #################################################################
function cost = sigmoid_cost(posh, negh, centers, A)
if A(1)<=0 
    cost = Inf;
else
    cost = sum(posh.* ...                                                                     
         (1./(1+exp(-(A(1)*centers+A(2)))) - 1).^2) + ...                                              
         sum(negh.* ...                                                                             
         (1./(1+exp(-(A(1)*centers+A(2)))) - 0).^2);                                                   
end
