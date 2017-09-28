function model = train_model(varargin)
% Trains the object detector model on the given dataset. 
%   MODEL = TRAIN_MODEL('argument1','val1', 'argument2', 'val2', ...)
%
%   The list of arguments and their possible values are listed below.
%   Argument names are case-sensitive.
%   
%   'descriptor': The value for this argument should be descriptive string
%       specifing the dataset and object class (and other things like which
%       training split is being used.) The string will be used as a prefix
%       in the filenames of the intermediately saved files as well as for
%       the final trained model. (REQUIRED)
%
%   'trainset_filename': Relative path of file containing the list of
%       training images. (REQUIRED)
%
%   'num_aspect_ratios': Number of different template shapes, i.e. aspect
%       ratios. This is for addressing pose variance of objects.
%       (REQUIRED)
%
%   'peripheral_training': A boolean indicating whether peripheral
%       templates should be trained or not. Set this to false, if you want
%       to train a sliding window (SW) model; to true, if you want to train
%       a foveated model (FOD). (REQUIRED)
%
%   'cache_dir': Relative path of a directory where intermediate and final
%       results will be saved. (REQUIRED)
%
%   'verbosity': 0,1 or 2. This controls the amount of log messages
%       produced. 0 means "be quite." 1 means "print info messages," and
%       2 means "print debugging and info messages." (Might not work as
%       expected :) )  (OPTIONAL)
%
%   'covariance_regularizer': Constant for regularization of the covariance
%       matrices. Recommended value is 0.1. (OPTIONAL)
%   
%   'min_pixel_area': Minimal area that a template can have in terms of
%       pixels. This is useful when training for small objects like
%       computer mouse. (OPTIONAL)
%
%   'latent_LDA_C': The C parameter in the latent-LDA cost function.
%       Default value is 1. (OPTIONAL)
%   
%   NOTE: There are other arguments that I haven't documented yet. I will
%   complete them as we go. 

% parse input arguments  {{{
inp = struct( ...
     'descriptor',                      [] ...
    ,'trainset_filename',               [] ...
    ,'pos_examples_and_dataset_filename', [] ...
    ,'num_aspect_ratios',               [] ...
    ,'covariance_regularizer',          0.1 ...
    ,'peripheral_training',             [] ...   
    ,'peripheral_filters_filename',     [] ...
    ,'verbosity',                       0 ...      
    ,'cache_dir',                       [] ...
    ,'initial_model',                   [] ...    
    ,'min_pixel_area',                  3000 ...
    ,'max_pixel_area',                  5000 ...
    ,'max_dim_in_cells',                13 ...
    ,'background_stats_filename',       [] ...
    ,'num_images_for_covariance_estimation', 2000 ...
    ,'latent_clustering',               false ...
    ,'latent_LDA_C',                    1 ...
    ,'representative_area_percentile',  20 ...
 );

for i=1:2:length(varargin)
    if isfield(inp, varargin{i})
        inp.(varargin{i}) = varargin{i+1};
    else
        error(['Unrecognized input argument: ' varargin{i}]);
    end
end
% done }}}

% Either 'trainset_filename' or 'pos_examples_and_dataset_filename' should be
% provided. For our own dataset format (i.e. imagefile.yes or imagefile.no
% files), 'trainset_filename' is recommended. For PASCAL datasets, the
% other one. 

%% create cache directory
if ~exist(inp.cache_dir, 'dir')
    mkdir(inp.cache_dir);
end

%% initialize model
if isempty(inp.initial_model)
    model = init_model();
else
    model = inp.initial_model;
end

%% Sliding window or the foveated object detector
if inp.peripheral_training
    model.search_model = 'FOD';
else
    model.search_model = 'SW';
end


%% collect positive examples: (no feature extraction here)
if isempty(inp.pos_examples_and_dataset_filename)
    pos_examples_file = fullfile(inp.cache_dir, ...
        sprintf('%s_training_set.mat', inp.descriptor));
    
    if exist(pos_examples_file, 'file')
        tmp = load(pos_examples_file);
        pos_examples = tmp.pos_examples;
        dataset = tmp.dataset;
    else
        [pos_examples,dataset] = get_positive_examples(...
            inp.trainset_filename, inp.verbosity);
        save(pos_examples_file, 'pos_examples', 'dataset');
    end
else
    tmp = load(inp.pos_examples_and_dataset_filename);
    pos_examples = tmp.pos_examples;
    dataset = tmp.dataset;
end




%% sanity check
if inp.verbosity>1
    %randomly choose a pos example, and show it
    id = randi(length(pos_examples));
    img = imread(pos_examples(id).imgfilename);

    % show the 5th scale
    img = imresize(img, model.scales(5));
    figure, imshow(img);
    bbox = pos_examples(id).gt_bbox*model.scales(5);
    rectangle('position', [bbox(1) bbox(2) bbox(3)-bbox(1) bbox(4)-bbox(2)]);
    drawnow

    % show another image with random scale
    id = randi(length(pos_examples));
    img = imread(pos_examples(id).imgfilename);
    sid = randi(length(model.scales));
    img = imresize(img, model.scales(sid));
    figure, imshow(img);
    bbox = pos_examples(id).gt_bbox*model.scales(sid);
    rectangle('position', [bbox(1) bbox(2) bbox(3)-bbox(1) bbox(4)-bbox(2)]);
    drawnow
end



%% choose template sizes and shapes
all_bboxes = cell2mat({pos_examples.gt_bbox}');
[t_widths, t_heights, eg2template_assigns] = ...
    determine_template_sizes( ...
    all_bboxes, inp.num_aspect_ratios, ...
    model.bin_length, inp.min_pixel_area, inp.max_pixel_area, ...
    inp.max_dim_in_cells, inp.representative_area_percentile);
% for mouse/monitor: 425,5000,13
% for pascal experiments these numbers are 3000,5000,13


if length(unique(t_widths))==1 && length(unique(t_heights))==1
    warning('duplicate size templates');
end

%% create templates
if inp.peripheral_training
    pf = load(inp.peripheral_filters_filename);
    num_angle_bins = size(pf.peripheral_filters.regions,1);
    num_eccentricity_bins = size(pf.peripheral_filters.regions,2);
    
    % create templates
    initial_templates_file = fullfile(inp.cache_dir, ...
        sprintf('%s_%d_FOD_initial_templates_%dx%d_visual_field.mat', ...
        inp.descriptor, inp.num_aspect_ratios, num_angle_bins, ...
        num_eccentricity_bins));
    if exist(initial_templates_file,'file')
        tmp = load(initial_templates_file);
        model.templates = tmp.model.templates;
    else
        
        
        model.templates = create_foveal_and_peripheral_templates( ...
            t_widths, t_heights, model.num_feat_dims, [-6 -6], [6 6], ...
            pf.peripheral_filters, model.bin_length);
        


        % compute peripheral feature pooling transformations (1xM to 1xN
        % transformations)
        for t=1:length(model.templates)
            if model.templates(t).foveal
                model.templates(t).T = 1;
            else
                T =[];
                F = model.templates(t).ftransform{1};
                th = model.templates(t).height;
                tw = model.templates(t).width;
                
                for f=1:model.num_feat_dims
                    for r=1:size(F,1)
                        T = [T, [zeros((f-1)*th*tw,1) ; ...
                            reshape(F(r,:),tw*th,[]) ; ...
                            zeros((model.num_feat_dims-f)*tw*th,1)]];
                    end
                end
                
                model.templates(t).T = sparse(T);
            end
        end

        
        save(initial_templates_file, 'model');
    end
    
    % clear visual field from memory
    clear pf
else
    for t=1:inp.num_aspect_ratios
        model.templates(t).width = t_widths(t);
        model.templates(t).height = t_heights(t);
        model.templates(t).T = 1; % this is feature pooling transformation
    end
end


%% Extract features of positive examples and identify their scales 
% that has a good overlap with the current templates
t0 = clock;
if inp.peripheral_training
     pos_features_filename = fullfile(inp.cache_dir, ...
        sprintf('%s_%d_FOD_pos_features_%dx%d_visual_field.mat', ...
        inp.descriptor, inp.num_aspect_ratios, ...
        num_angle_bins, num_eccentricity_bins));
    
    if exist(pos_features_filename, 'file')
        tmp = load(pos_features_filename);
        pos_examples = tmp.pos_examples;
        no_overlaps = tmp.no_overlaps;
    else
        [pos_examples, no_overlaps] = ...
            extract_peripheral_features_of_pos_examples(...
            pos_examples, model, 0.7, inp.verbosity);
        save(pos_features_filename, 'pos_examples', 'no_overlaps');
    end
else
    pos_features_filename = fullfile(inp.cache_dir, ...
        sprintf('%s_%d_SW_pos_features.mat', inp.descriptor, ...
        inp.num_aspect_ratios));
    
    if exist(pos_features_filename, 'file')
        tmp = load(pos_features_filename);
        pos_examples = tmp.pos_examples;
        no_overlaps = tmp.no_overlaps;
    else
        [pos_examples, no_overlaps] = extract_features_of_pos_examples(...
            pos_examples, model, 0.7, inp.verbosity);
        save(pos_features_filename, 'pos_examples', 'no_overlaps');
    end
end

fprintf(1,'%d (%.2f%%) examples don''t have good overlapping bounding boxes, hence feature vectors.\n', ...
    sum(no_overlaps), 100*sum(no_overlaps)/length(no_overlaps));
t1 = clock;
if inp.verbosity>0
    fprintf(1,'took %.2f minutes.\n', etime(t1, t0)/60);
end


%% compute background statistics (cov and mu of example feature vector)
C0s = cell(length(model.templates),1);
mu0s = C0s;

% bckgrnd = nonfov_HOG_covariance(...
%     inp.background_stats_filename, dataset, ...
%          model.padx, model.pady, model.bin_length,  ...
%          inp.max_dim_in_cells+1, inp.max_dim_in_cells+1, model.scales, ...
%          min(inp.num_images_for_covariance_estimation, ...
%          length(dataset.imgnames)));


for t=1:length(model.templates)
   mu0cov0file = fullfile(inp.cache_dir, sprintf('%02d_%02d_%dscales.mat', ...
       model.templates(t).height, model.templates(t).width, ...
       length(model.scales)));

   if exist(mu0cov0file, 'file')
%         if verbosity>0
%             fprintf(1,'Using precomputed %s\n', mu0cov0file);
%         end
       tmp = load(mu0cov0file);
       C0s{t} = tmp.cov0;
       mu0s{t} = tmp.mu0;
   else
       cov0_h = model.templates(t).height;
       cov0_w = model.templates(t).width;
       [mu0,cov0] = compute_mu_cov_of_background_examples(dataset, ...
           inp.num_images_for_covariance_estimation, cov0_h, cov0_w, ...
           model);

       save(mu0cov0file, 'mu0', 'cov0', 'cov0_h', 'cov0_w');

       C0s{t} = cov0;
       mu0s{t} = mu0;
   end
%     [mu0s{t} , C0s{t}] = bckgrnd.construct_mu_sigma( ...
%         model.templates(t).height, model.templates(t).width);

    % peripheral transfromation
    C0s{t} = model.templates(t).T'*C0s{t}*model.templates(t).T;
    mu0s{t} = model.templates(t).T'*mu0s{t};
    
    % regularize cov matrix
    C0s{t} = (1-inp.covariance_regularizer)*C0s{t} + ...
        inp.covariance_regularizer*diag(diag(C0s{t}));%(trace(C0s{t})/size(C0s{t},1))* ...
    %eye(size(C0s{t}));
end




%% initial training of templates (on warped positive examples)
[model,feats_per_template] = ...
    train_on_warped_positives_whitened_features(pos_examples, ...
    eg2template_assigns, model, C0s, mu0s);

if length(model.templates)>length(feats_per_template)
    % this is peripheral system, skip optimizing for A and B.
else
    % compute optimal A and B and compute latent LDA cost
    fun = @(P) latent_LDA_cost_function(model, ...
        feats_per_template, mu0s, inp.latent_LDA_C, ...
        P(1:length(model.templates)), ...
        P(length(model.templates)+1:end));
    num_templates = length(model.templates);
    Pstar = fminsearch(fun, double([[model.templates.A]'; ...
            [model.templates.B]']));
%     Pstar = fminunc(fun, [.1*ones(num_templates,1); ...
%             -1*ones(num_templates,1)], optimset('largescale','off', ...
%             'gradobj','on'));
    for t=1:min(num_templates, length(feats_per_template))
        model.templates(t).A = Pstar(t);
        model.templates(t).B = Pstar(t+num_templates);
    end
    
    if inp.verbosity>0
        cost = latent_LDA_cost_function(model, feats_per_template, ...
            mu0s, inp.latent_LDA_C, []);
        fprintf(1,'Cost after training on warped_positives: %f\n', cost);
    end
end
% model = calibrate_model_given_pos_features(model, dataset, 100, ...
%     feats_per_template);
%  %% visualize the initial templates
% if inp.verbosity>0
% for t=1:length(model.templates)
%     figure
%     visualizeHOG(foldHOG(reshape(model.templates(t).w, ...
%         [model.templates(t).height ...
%         model.templates(t).width model.num_feat_dims])));
%     drawnow
% end
% end



%% the main training
t0 = clock;
model = train_templates(model, pos_examples, C0s, mu0s, ...
    inp.latent_LDA_C, inp.verbosity);
% model = train_templates_with_calibration(model, pos_examples, ...
%     C0s, mu0s, dataset, inp.verbosity);
t1 = clock;
fprintf(1,'took %.2f minutes.\n', etime(t1, t0)/60);



if isfield(model.templates, 'foveal')
    % template shapes
    template_shapes = unique([model.templates.height ; ...
        model.templates.width]','rows');
    for t=1:length(model.templates)
        model.templates(t).shape_id = find( ...
            model.templates(t).height==template_shapes(:,1) & ...
            model.templates(t).width==template_shapes(:,2));
    end
else
    % rotate template weights for conv2
    for t=1:length(model.templates)
        W = reshape(model.templates(t).w, ...
            model.templates(t).height, model.templates(t).width, []);
        
        for d=1:size(W,3)
            W(:,:,d) = rot90(W(:,:,d), 2);
        end
        
        model.templates(t).rotW = W;
    end
end

% %% visualize the initial templates
% if inp.verbosity>0
% for t=1:length(model.templates)
%     figure
%     visualizeHOG(foldHOG(reshape(model.templates(t).w, ...
%         [model.templates(t).height ...
%         model.templates(t).width model.num_feat_dims])));
%     drawnow
% end
% end

%% calibration of raw LDA scores
if inp.verbosity>0
    fprintf(1,'Calibrating model...');
    t0 = clock;
end
if inp.peripheral_training
    % then, use less number of images as this calibration is taking sooo
    % long
    model = calibrate_model(model, dataset, 50, 25);
else
    model = calibrate_model(model, dataset);
end
if inp.verbosity>0
    fprintf(1,'done in %.2f minutes.\n', etime(clock, t0)/60);
end


%
% [ap,rec,prec] = evaluate_model(subdataset, model, [], @detect_SW, 1);
% VOC2007_ap(rec,prec)
% f = (2*rec.*prec)./(rec+prec);
% max(f)
%
% m1 = model;
% m1.templates = m1.templates(1);
% [ap,rec,prec] = evaluate_model(subdataset, m1, [], @detect_SW, 1);
% VOC2007_ap(rec,prec)
% f = (2*rec.*prec)./(rec+prec);
% max(f)
% m1 = model;
% m1.templates = m1.templates(2);
% [ap,rec,prec] = evaluate_model(subdataset, m1, [], @detect_SW, 1);
% VOC2007_ap(rec,prec)
% f = (2*rec.*prec)./(rec+prec);
% max(f)


%% save trained model
if inp.peripheral_training
    save(fullfile(inp.cache_dir, ...
        sprintf('%s_%d_%.3f_FOD_trained_model.mat', inp.descriptor, ...
        inp.num_aspect_ratios, inp.covariance_regularizer )), 'model');
else
    save(fullfile(inp.cache_dir, ...
        sprintf('%s_%d_%.3f_SW_trained_model.mat', inp.descriptor, ...
        inp.num_aspect_ratios, inp.covariance_regularizer )), 'model');
end
