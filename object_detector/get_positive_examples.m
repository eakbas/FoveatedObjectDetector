function [pos_examples, dataset] = get_positive_examples(trainset_filename, verbosity)
% Reads in and returns all positive examples

if verbosity>0
    fprintf(1,'Collecting positive examples...');
end

pos_examples = [];
dataset = [];

fp = fopen(trainset_filename,'r');
imgnames = textscan(fp, '%s');
fclose(fp);
imgnames = imgnames{1};

% first line is the directory that contains images
pathstr = imgnames{1};
imgnames = imgnames(2:end);

dataset.dir = pathstr;
dataset.imgnames = imgnames;
dataset.imglabels = false(length(imgnames),1);
dataset.bboxes = cell(length(imgnames),1);
dataset.imsize = zeros(length(imgnames),2);

for i=1:length(imgnames)
    % read in image
    img = imread(fullfile(pathstr, imgnames{i}));

    % see if this image has recorded bounding boxes or not
    gt_file = fullfile(pathstr, [imgnames{i} '.yes']);
    if exist(gt_file, 'file')
        dataset.imglabels(i) = true;

        bboxes = load(gt_file);
        for b=1:size(bboxes)
            pos_examples(length(pos_examples)+1,1).imgfilename = ...
                fullfile(pathstr, imgnames{i});

            pos_examples(end).gt_bbox = round([bboxes(b,1) bboxes(b,2) ...
                bboxes(b,1)+bboxes(b,3)-1  ...
                bboxes(b,2)+bboxes(b,4)-1]);

            pos_examples(end).imsize = [size(img,2) size(img,1)];
        end
        
        dataset.bboxes{i} = round([bboxes(:,1) bboxes(:,2) ...
                bboxes(:,1)+bboxes(:,3)-1  ...
                bboxes(:,2)+bboxes(:,4)-1]);
        dataset.imsize(i,:) = [size(img,2) size(img,1)];
    else
        dataset.imglabels(i) = false;
    end

    %textprogressbar(100*i/length(imgids));
end
%textprogressbar(-1);

if verbosity>0
    fprintf(1,'done.\n');
end
