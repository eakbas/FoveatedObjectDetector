function model = calibrate_model2(model, dataset)
% Joint calibration of templates.
% 
% %%
% % choose a subset of dataset
% ids = [find(dataset.imglabels,50) ; find(~dataset.imglabels,50)];
% X = cell(length(ids),1);
% Y = cell(length(ids),1);
% parfor i=1:length(ids)   
%     img = imread(fullfile(dataset.dir, dataset.imgnames{ids(i)}));
%     [X{i},Y{i}] = analyze_detections(img, dataset.bboxes{ids(i)}, model,true);
% end
% 
% %%
% X = cell2mat(X);
% Y = cell2mat(Y);
% fun = @(w) logistic_cost(X,Y, w);
% w=[.1 .1 -1 -1]';
% wstar = fmincon(fun,w,[],[],[],[],[0 0 -Inf -Inf],Inf(1,4), [], ...
%     optimset('display', 'iter', 'gradobj','on', ...
%     'algorithm','interior-point'));
% [fun(w) fun(wstar)]
% model.templates(1).A = wstar(1);
% model.templates(2).A = wstar(2);
% model.templates(1).B = wstar(3);
% model.templates(2).B = wstar(4);
% 
% 
% 
% %%
% X = cell2mat(X);
% Y = cell2mat(Y);
% fun = @(w) logistic_cost(X,Y, w);
% w=[.1*ones(1,80) -1 -1]';
% wstar = fmincon(fun,w,[],[],[],[],[eps*ones(1,80) -Inf -Inf],Inf(1,80), [], ...
%     optimset('display', 'iter', 'gradobj','on', ...
%     'algorithm','interior-point'));
% [fun(w) fun(wstar)]
% 
% i=1;
% for t=1:length(model.templates)
%     for s=1:length(model.scales)
%         model.templates(t).alpha(s) = wstar(i);
%         i = i + 1;
%     end
% end
% for t=1:length(model.templates)
%     model.templates(t).beta = wstar(i);
%     i = i + 1;
% end

%%
% choose a subset of dataset
ids = [find(dataset.imglabels,50) ; find(~dataset.imglabels,50)];


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


% combine scores
posh = zeros(size(centers));
negh = zeros(size(centers));
for i=1:length(ids)
    for t=1:length(model.templates)
        posh(t,:) = posh(t,:) + all_pos_scores{i}{t};
        negh(t,:) = negh(t,:) + all_neg_scores{i}{t};
    end
end


% create data for optimization
num_centers = size(centers,2);
num_templates = length(model.templates);
X = cell(num_templates,1);
Y = cell(num_templates,1);
W = cell(num_templates,1);
for t=1:length(model.templates)
    x = zeros(num_centers, 2*length(model.templates));
    x(:,t) = centers(t,:)';
    x(:,num_templates+t) = 1;

    % true-positives
    nonzero = posh(t,:)>0;
    tp = x(nonzero, :);
    tp_weights = posh(t,nonzero)';

    % false-positives
    nonzero = negh(t,:)>0;
    fp = x(nonzero, :);
    fp_weights = negh(t,nonzero)';

    X{t} = [tp;fp];
    Y{t} = [ones(length(tp),1) ; -ones(length(fp),1)];
    W{t} = [tp_weights ; fp_weights];
end

% relative constraints
for t1=1:length(model.templates)
    nonzeros_t1 = posh(t1,:)>0;
    
    for t2=1:length(model.templates)
        if t2==t1
            continue
        end
        
        nonzeros_t2 = negh(t2,:)>0;
        
        % make sure t1's true-positives score higher than t2's
        % false-positives
        [c1,c2] = meshgrid(centers(t1,nonzeros_t1), ...
            centers(t2, nonzeros_t2));
        [w1,w2] = meshgrid(posh(t1, nonzeros_t1), ...
            negh(t2, nonzeros_t2));
        x = zeros(numel(c1), 2*num_templates);
        x(:,t1) = c1(:);
        x(:,t2) = -c2(:);
        x(:,num_templates + t1) = 1;
        x(:,num_templates + t2) = -1;
        X{t1} = [X{t1} ; x];
        Y{t1} = [Y{t1} ; ones(size(x,1),1)];
        W{t1} = [W{t1} ; w1(:).*w2(:)];
    end
end



X = cell2mat(X);
Y = cell2mat(Y);
W = cell2mat(W);

fun = @(w) weighted_logistic_cost(X,Y,W,w);

w=[.1 .1 -1 -1]';
wstar = fmincon(fun,w,[],[],[],[],[eps eps -Inf -Inf],Inf(1,4), [], ...
    optimset('display', 'iter', 'gradobj','on', ...
    'algorithm','interior-point'));
%[fun(w) fun(wstar)]

for t=1:length(model.templates)
    model.templates(t).A = wstar(t);
    model.templates(t).B = wstar(t+length(model.templates));
end

% % fit sigmoid
% for t=1:length(model.templates)
% %     cost = @(A) sum(posh.* ...                                                                     
% %          (1./(1+exp(-(A(1)*centers+A(2)))) - 1).^2) + ...                                              
% %          sum(negh.* ...                                                                             
% %          (1./(1+exp(-(A(1)*centers+A(2)))) - 0).^2);   
% cost = @(A) sigmoid_cost(posh(t,:),negh(t,:),centers(t,:), A);
% 
%      xstar = fminsearch(cost, [0.1 -1]);%, optimset('display','final')); 
%      model.templates(t).A = xstar(1);
%      model.templates(t).B = xstar(2);
% end


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
