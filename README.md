# Foveated Object Detector 

This repository contains the MATLAB source for the Foveated Object Detector
(FOD) described in [our paper](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005743) [1] and [technical report](https://arxiv.org/abs/1408.0814) [2].  The code was 
tested on (K)ubuntu Linux 14.04 (and 16.04) using MATLAB R2014a and above. 

The script `main.m` contains example calls that show how to  train and run the FOD
(and also its sliding window (SW) version) on a sample dataset provided under
`data/`.


[1] Akbas, E., & Eckstein, M. P. (2017). Object detection through search with a foveated visual system. PLoS Computational Biology, 13(10), e1005743.

[2] Emre Akbas, Miguel P. Eckstein, "Object Detection Through Exploration
With A Foveated Visual Field," Technical report, Vision and Image Understanding Lab, University of
California Santa-Barbara. [Link](http://arxiv.org/abs/1408.0814)



##  Dataset
First, prepare your dataset. To do that, open
`./data/monitor_dataset/monitor_train01.txt` with a text editor and edit the
first line to be **the absolute path** of the parent directory of the images in
the dataset. For example, suppose `/home/userX/FOD/` is the directory of this
repository, then the first line of `monitor_train01.txt` should read as
`/home/userX/FOD/data/monitor_dataset\database`. You should do the same for
other dataset file  `monitor_test01.txt`.
 


## Paths
Add the following paths to your MATLAB workspace by running 

```Matlab
addpath object_detector/                % object detector
addpath foveated_visual_field/          % Freeman-Simoncelli model
addpath code_from_DPM/                  % for HOG feature extraction and HOG visualization
```


## Training a Sliding Window (SW) Model 
The following calls trains a SW model on the `monitor_train01` dataset. For
help with the input arguments, run `help train_model`. Typically, on a i7
processor, this process takes about 15 minutes. 

```Matlab
t0=clock;
model = train_model('descriptor', 'monitor01', ...
    'trainset_filename', 'data/monitor_dataset/monitor_train01.txt', ...
    'num_aspect_ratios', 2, ...
    'peripheral_training', false, 'verbosity', 1, ...
    'cache_dir', 'cache');
fprintf('Training took %.2f minutes.\n', etime(clock(), t0)/60);
```


Now you have trained a model, you can visualize one of the learned templates for
sanity check. If everything went smoothly, you should be seeing the rough
shape of your target object in the HOG features. 

```Matlab
figure, visualizeHOG(foldHOG(reshape( model.templates(1).w, ...
    model.templates(1).height, model.templates(1).width, [])));
```


To run the trained model on an image, use the `detect_SW` function for
detection, and use the `show_bounding_boxes` on the output of `detect_SW` to visualize the
detection results. Try this with your own images or images dowloaded
from the web. 

```Matlab
img = imread('data/monitor_dataset/database/ZxsobICq.jpg');
[c,bb] = detect_SW(img, model);
figure, show_bounding_boxes(img, bb(:,1), c(1));
```


## Testing the Model 
To evaluate the trained model on the test set, use `evaluate_model`. This will
run the trained model on all of the 547 images in the testing set and will
return you the precision-recall curve. `evaluate_model` runs `detect_SW` on many images
in parallel, so don't forget the turn on the matlabpool by calling `matlabpool
open` to speed up the evaluation (it also works without matlabpool). 

```Matlab
detect_fun = @detect_SW;
[ap,rec,prec] = evaluate_model('data/monitor_dataset/monitor_test01.txt', ...
    model, 'cache', detect_fun, 1);
figure, plot(rec,prec);
xlabel('recall');
ylabel('precision');
```




## Training a Foveated Model (FOD)
To train a FOD model, we simply set `peripheral_training` to true and provide
the `peripheral_filters_filename` parameter. 

```Matlab
t0=clock;
model = train_model('descriptor', 'monitor01', ...
    'trainset_filename', 'data/monitor_dataset/monitor_train01.txt', ...
    'num_aspect_ratios', 2, ...
    'peripheral_training', true, 'verbosity', 1, ...
    'peripheral_filters_filename', 'foveated_visual_field/peripheral_filters_30x4.mat', ...
    'cache_dir', 'cache', 'min_pixel_area', 425);
fprintf('Training took %.2f minutes.\n', etime(clock(), t0)/60);
```


Visualize the first template: 

```Matlab
figure, visualizeHOG(foldHOG(reshape( model.templates(1).w, ...
    model.templates(1).height, model.templates(1).width, [])));
```


Run the trained model on an image:

```Matlab
img = imread('../data/monitor_dataset/database/89nXYzxK.jpg');
[c,bb,f] = detect_FOD(img, model,'MAP_IOR',[.8 .8],2);
figure, show_fixations_and_detections(img, bb(:,1:2), [], f, true);
```


Evaluate the trained model on the test set and retrieve the precision-recall
 curve:

```Matlab
% (don't forget to turn on matlabpool:  "matlabpool open")
detect_fun = @(img,model)detect_FOD(img,model,'MAP_IOR',[],5);
[ap,rec,prec] = evaluate_model('../data/monitor_dataset/monitor_test01.txt', ...
    model, 'cache', detect_fun, 1);
figure, plot(rec,prec);
xlabel('recall');
ylabel('precision');
```

