%% README: 
% This is the main script that trains and evaluates SW and FOD models.


%% DATASETS
% First, prepare your dataset. To do that, open
% "./data/monitor_dataset/monitor_train01.txt" with a text editor and edit
% the first line to be the absolute path of the parent directory of the
% images in the dataset. For example, if "C:\\Users\XYZ\Desktop\FOD" is the
% directory where you unzipped the FOD package, then the first line of
% monitor_train01.txt should read as
% "C:\\Users\XYZ\Desktop\FOD\data\monitor_dataset\database". You should do
% the same for other dataset files like monitor_test01.txt. 
 


%% PATHS
% The following paths are necessary to run the model. Add them to your
% workspace. 
addpath object_detector/                % object detector
addpath foveated_visual_field/          % Freeman-Simoncelli model
addpath code_from_DPM/                  % for HOG feature extraction and 
                                        % HOG visualization


%% TRAINING A SLIDING WINDOW (SW) MODEL
% The following calls trains a SW model on the monitor_train01 dataset. For
% help with the input arguments, look at "help train_model". On my machine,
% this process takes about 15 minutes. 
t0=clock;
model = train_model('descriptor', 'monitor01', ...
    'trainset_filename', 'data/monitor_dataset/monitor_train01.txt', ...
    'num_aspect_ratios', 2, ...
    'peripheral_training', false, 'verbosity', 1, ...
    'cache_dir', 'cache');
fprintf('Training took %.2f minutes.\n', etime(clock(), t0)/60);


% Now you have trained a model, you can visualize one of the templates for
% sanity check. If everything went smoothly, you should be seeing the rough
% shape of your target object in the HOG features. 
figure, visualizeHOG(foldHOG(reshape( model.templates(1).w, ...
    model.templates(1).height, model.templates(1).width, [])));


% To run the trained model on an image, use "detect_SW" function, and use
% "show_bounding_boxes" on the output of "detect_SW" to visualize the
% detection results. Try this with your own images or images dowloaded
% from the web. 
img = imread('data/monitor_dataset/database/ZxsobICq.jpg');
[c,bb] = detect_SW(img, model);
figure, show_bounding_boxes(img, bb(:,1), c(1));


%% TESTING THE MODEL
% To run the trained model on the test set (which contains hundreds of
% images), use "evaluate_model." This will run the trained model on all of
% the 547 images in the testing set and will return you the
% precision-recall curve. "evaluate_model" runs "detect_SW" on many images
% in parallel, so don't forget the turn on the matlabpool by calling
% "matlabpool open" to speed up the evaluation. 
detect_fun = @detect_SW;
[ap,rec,prec] = evaluate_model('data/monitor_dataset/monitor_test01.txt', ...
    model, 'cache', detect_fun, 1);
figure, plot(rec,prec);
xlabel('recall');
ylabel('precision');




%% TRAINING A FOVEATED MODEL (FOD)

% Train the model. Notice that we set "peripheral_training" to true. 
t0=clock;
model = train_model('descriptor', 'monitor01', ...
    'trainset_filename', 'data/monitor_dataset/monitor_train01.txt', ...
    'num_aspect_ratios', 2, ...
    'peripheral_training', true, 'verbosity', 1, ...
    'peripheral_filters_filename', 'foveated_visual_field/peripheral_filters_30x4.mat', ...
    'cache_dir', 'cache', 'min_pixel_area', 425);
fprintf('Training took %.2f minutes.\n', etime(clock(), t0)/60);


% Visualize the first template: 
figure, visualizeHOG(foldHOG(reshape( model.templates(1).w, ...
    model.templates(1).height, model.templates(1).width, [])));


% Run the trained model on an image
img = imread('../data/monitor_dataset/database/89nXYzxK.jpg');
[c,bb,f] = detect_FOD(img, model,'MAP_IOR',[.8 .8],2);
figure, show_fixations_and_detections(img, bb(:,1:2), [], f, true);


% Run the trained model on the test set and retrieve the precision-recall
% curve. 
% (don't forget to turn on matlabpool:  "matlabpool open")
detect_fun = @(img,model)detect_FOD(img,model,'MAP_IOR',[],5);
[ap,rec,prec] = evaluate_model('../data/monitor_dataset/monitor_test01.txt', ...
    model, 'cache', detect_fun, 1);
figure, plot(rec,prec);
xlabel('recall');
ylabel('precision');
