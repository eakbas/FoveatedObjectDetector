function model = train_templates(model, pos_examples, C0s, mu0s, ...
    regC, verbosity)


if isfield(model.templates, 'foveal')
    peripheral_system = true;
else
    peripheral_system = false;
end

prev_pos_beliefs = [];

%estimated_cache_sizes = NaN(length(templates),1);

if verbosity>0
    fprintf(1,'Training templates...\n');
end


if isfield(model.templates, 'foveal')
    % transform trained templates so that there is no need to transform
    % the fetaure vectors in evaluation
    for i=1:length(model.templates)
        if ~model.templates(i).foveal
            model.templates(i).w = single(model.templates(i).T* ...
                double(model.templates(i).w));
        end
    end
end
    

num_templates = length(model.templates);


costs = [];
for epoch=1:50   
    % compute beliefs for positive examples using the current templates
    if peripheral_system
        [pos_beliefs, pos_features_per_template] = ...
            compute_belief_for_pos_examples_peripheral(model, ....
            pos_examples, verbosity);
    else
        [pos_beliefs, pos_features_per_template, ~] = ...
            compute_belief_for_pos_examples(model, ....
            pos_examples, verbosity);
    end
    
    
    if isequal(pos_beliefs, prev_pos_beliefs)
        fprintf(1,'Beliefs for positive examples did not change. stopping at epoch %d.\n', epoch);
        break
    else
        prev_pos_beliefs = pos_beliefs;
    end
    
    
    % re-train templates with the new beliefs
    for t=1:length(model.templates)
        if isempty(pos_features_per_template{t})
            warning('Positive examples of template %d evaporated!',t);
            continue;
        end
        
        model.templates(t).w = C0s{t}\ ...
            (mean(pos_features_per_template{t},1)' - mu0s{t});
        
        % calibration
%         model.templates(t).A = 1;
%         model.templates(t).B = 0;
        
%         model.templates(t).A = 1/sqrt( ...
%             model.templates(t).w'*C0s{t}*model.templates(t).w);
%         model.templates(t).B = -mean(pos_features_per_template{t},1)* ...
%             model.templates(t).w;
        
        
%         model.templates(t).A = 1/sqrt( ...
%             model.templates(t).w'*C0s{t}*model.templates(t).w);
%         model.templates(t).B = -mu0s{t}'* ...
%             model.templates(t).w;
        
%         model.templates(t).A = 1/sqrt( ...
%             model.templates(t).w'*C0s{t}*model.templates(t).w);
%         model.templates(t).B = -0.5*model.templates(t).A* ...
%             (mean(pos_features_per_template{t},1)* ...
%             model.templates(t).w + mu0s{t}'*model.templates(t).w);
        
        % shift the means of positive and negative distributions to +1 and
        % -1, respectively
        model.templates(t).A = 2/(mean(pos_features_per_template{t},1)*model.templates(t).w - ...
            mu0s{t}'*model.templates(t).w);
        model.templates(t).B = 1-mean(pos_features_per_template{t},1)*model.templates(t).w* ...
            model.templates(t).A;

%         m1 = mean(pos_features_per_template{t},1)*model.templates(t).w;
%         m0 = mu0s{t}'*model.templates(t).w;
%         s = sqrt(model.templates(t).w'*C0s{t}*model.templates(t).w);
%         fun = @(x) distribution_scaling_cost(m1,m0,s,x);
%         best = fminsearch(fun, [.1 -1]);
%         model.templates(t).A = best(1);
%         model.templates(t).B = best(2);

              
        % threshold 
        resps = pos_features_per_template{t}*model.templates(t).w;
        resps = sort(resps);
        model.templates(t).threshold = resps(max(1,round(.025*length(resps))));
    end
    
    if isfield(model.templates, 'foveal')
        % transform trained templates so that there is no need to transform
        % the fetaure vectors in evaluation
        for i=1:length(model.templates)
            if ~model.templates(i).foveal
                model.templates(i).w = single(model.templates(i).T* ...
                    double(model.templates(i).w));
            end
        end
    end
    
    
    if isfield(model.templates, 'foveal')
        num_foveal_templates = sum([model.templates.foveal]==1);
        
        % compute optimal A and Bs
        fun = @(P) latent_LDA_cost_function(model, ...
            pos_features_per_template(1:num_foveal_templates), ...
            mu0s, regC, ...
            P(1:num_foveal_templates), ...
            P(num_foveal_templates+1:end));
        Pstar = fminsearch(fun, ...
            double([[model.templates(1:num_foveal_templates).A]'; ...
            [model.templates(1:num_foveal_templates).B]']));
        for t=1:num_foveal_templates
            model.templates(t).A = Pstar(t);
            model.templates(t).B = Pstar(t+num_foveal_templates);
        end
    else
        % compute optimal A and Bs
        fun = @(P) latent_LDA_cost_function(model, ...
            pos_features_per_template, mu0s, regC, ...
            P(1:length(model.templates)), ...
            P(length(model.templates)+1:end));
        Pstar = fminsearch(fun, double([[model.templates.A]'; ...
            [model.templates.B]']));
%         Pstar = fminunc(fun, [.1*ones(num_templates,1); ...
%             -1*ones(num_templates,1)], optimset('largescale','off', ...
%             'gradobj','on'));
        for t=1:num_templates
            model.templates(t).A = Pstar(t);
            model.templates(t).B = Pstar(t+num_templates);
        end
    end
    
    % compute cost
    costs(epoch) = latent_LDA_cost_function(model, ...
        pos_features_per_template, mu0s, regC, []);
    
    
    if epoch>1
        if costs(epoch)>costs(epoch-1)
            %model = prev_model;
            %fprintf(1,'cost increased, using previous model and stopping\n');
            if verbosity>0
                fprintf(1,'cost increased, stopping\n');
            end
            model = prev_model;
            break;
        elseif costs(epoch)>.95*costs(epoch-1)
            if verbosity>0
                fprintf(1,'cost not decreased sufficiently, stopping\n');
            end
            break
        end
    end
    
    prev_model = model;
end
costs
%[snr mean(snr,2)]
% 
% if peripheral_system
%     [pos_beliefs, pos_features_per_template, high_recall_threshold] = ...
%         compute_belief_for_pos_examples_peripheral(model, ....
%         pos_examples, verbosity);
% else
%     [pos_beliefs, pos_features_per_template, high_recall_threshold] = ...
%         compute_belief_for_pos_examples(model, ....
%         pos_examples, verbosity);
%     
%     % set a threshold per template
%     for t=1:length(model.templates)
%         resps = pos_features_per_template{t}*model.templates(t).w;
%         model.templates(t).threshold = min(resps);
%     end
% end
% 
% 
% model.threshold = high_recall_threshold;


%% #####################################################################
function [beliefs, pos_features_per_template, high_recall_threshold] = ...
    compute_belief_for_pos_examples(model, pos_examples, verbosity)
% Computes beliefs for positive examples given the current templates.  Returned
%   variable 'beliefs' is a N-by-2 matrix where N is the number of positive
%   examples. First column indicates the template to choice, the second
%   column gives the preffered scale. And the third column is the actual
%   belief score.

if verbosity>1
    fprintf(1,'>> Computing beliefs for the positive examples...\n');
end

beliefs = cell(length(pos_examples),1);
pos_features_per_template = cell(length(model.templates),1);

belief_responses = NaN(length(pos_examples),1);
for i=1:length(pos_examples)
    if isempty(pos_examples(i).features)
        beliefs{i} = [];
        continue;
    end
    
    beliefs{i} = false(size(pos_examples(i).features));
    
    % compute responses
    responses = zeros(size(pos_examples(i).features));
    for k=1:length(pos_examples(i).features)
        t_id = pos_examples(i).features_metadata.templates(k);
       
        responses(k) = model.templates(t_id).A* ...
            pos_examples(i).features{k}*model.templates(t_id).w + ...
            model.templates(t_id).B;

    end
    
    [belief_responses(i),best] = max(responses);
    t_id = pos_examples(i).features_metadata.templates(best);
    
    pos_features_per_template{t_id} = [ ...
        pos_features_per_template{t_id} ; ...
        pos_examples(i).features{best}];
    
    beliefs{i}(best) = true;    
end

if verbosity>1   
   for t=1:length(templates)
       fprintf(1,'%6d positive examples assigned to template #%2d\n', ...
           size(pos_features_per_template{t},1), t);
   end
   fprintf(1,'\n');
end


% compute threshold for high recall
high_recall_threshold = min(belief_responses);
% belief_responses = sort(belief_responses);
% high_recall_threshold = belief_responses(...
%     round(.02*length(belief_responses)));




%% #####################################################################
function [beliefs, pos_features_per_template, high_recall_threshold] = ...
    compute_belief_for_pos_examples_peripheral(model, ...
    pos_examples, verbosity)
% Computes beliefs for positive examples given the current templates.  Returned
%   variable 'beliefs' is a N-by-2 matrix where N is the number of positive
%   examples. First column indicates the template to choice, the second
%   column gives the preffered scale. And the third column is the actual
%   belief score.

if verbosity>1
    fprintf(1,'>> Computing beliefs for the positive examples...\n');
end

beliefs = cell(length(pos_examples),1);
belief_responses = cell(length(pos_examples),1);
pos_features_per_template = cell(length(model.templates),1);

for i=1:length(pos_examples)
    if isempty(pos_examples(i).features)
        beliefs{i} = [];
        continue;
    end
    
    beliefs{i} = false(size(pos_examples(i).features));
    
    % compute responses
    responses = zeros(size(pos_examples(i).features));
    fixations = cell(max(pos_examples(i).features_metadata.fixation)+1);
    for k=1:length(pos_examples(i).features)
        f_id = pos_examples(i).features_id(k);
        t_id = pos_examples(i).features_metadata.templates(k);
        fix = pos_examples(i).features_metadata.fixation(k,:);
        
%         responses(k) = model.templates(t_id).A* ...
%             pos_examples(i).features{f_id}* ...
%             single(model.templates(t_id).T* ...
%             double(model.templates(t_id).w)) + ...
%             model.templates(t_id).B;
        responses(k) = model.templates(t_id).A* ...
            pos_examples(i).features{f_id}* ...            
            model.templates(t_id).w + ...
            model.templates(t_id).B;
        
        if fix(1)==-1
            fixations{end} = [fixations{end} ; k responses(k)];
        else
            fixations{fix(1), fix(2)} = [fixations{fix(1), fix(2)} ; ...
                k responses(k)];
        end
    end

    % for each different fixation, choose the best template and select
    % other latents (scale, left-right) that are valid
    idx = find(~cellfun(@isempty, fixations));
    for f=1:length(idx)
        [mx,best] = max(fixations{idx(f)}(:,2));
        belief_responses{i} = [belief_responses{i} ; mx];
        feat_id = fixations{idx(f)}(best,1);
        t_id = pos_examples(i).features_metadata.templates(feat_id);
        
%         pos_features_per_template{t_id} = [ ...
%             pos_features_per_template{t_id} ; ...
%             single( ...
%             double(pos_examples(i).features{ ...
%             pos_examples(i).features_id(feat_id)})* ...
%             model.templates(t_id).T)];
        pos_features_per_template{t_id}{length( ...
            pos_features_per_template{t_id})+1} = ...
            single( ...
            double(pos_examples(i).features{ ...
            pos_examples(i).features_id(feat_id)})* ...
            model.templates(t_id).T);
        
        beliefs{i}(feat_id) = true;
    end
end

for t=1:length(model.templates)
    pos_features_per_template{t} = cell2mat( ...
        pos_features_per_template{t}');
end

if verbosity>1   
   for t=1:length(model.templates)
       fprintf(1,'%6d positive examples assigned to template #%2d\n', ...
           size(pos_features_per_template{t},1), t);
   end
   fprintf(1,'\n');
end

% compute threshold for high recall
high_recall_threshold = sort(cell2mat(belief_responses));
% belief_responses = sort(cell2mat(belief_responses));
% high_recall_threshold = belief_responses(...
%     round(.02*length(belief_responses)));
