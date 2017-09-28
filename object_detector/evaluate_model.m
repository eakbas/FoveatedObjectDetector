function [ap,rec,prec] = evaluate_model(testset_filename, model, ...
    cache_dir, detect_fun, verbosity)
% First argument is either testset_filename or the 'dataset' struct. 

t0=clock;

% read in dataset
if isstruct(testset_filename)    
    dataset = testset_filename;
else
    [~, name, ~] = fileparts(testset_filename);
    gt_file = fullfile(cache_dir, [name '_dataset.mat']);
    if exist(gt_file, 'file')
        tmp = load(gt_file);
        pos_examples = tmp.pos_examples;
        dataset = tmp.dataset;
    else
        [pos_examples, dataset] = get_positive_examples( ...
            testset_filename, verbosity);
        save(gt_file, 'pos_examples', 'dataset');
    end
end


% apply detector to each image
C = cell(length(dataset.imgnames), 1);
BB = C;
TP = C; % for storing true-positives
FP = C; % for storing false-positives
parfor i=1:length(dataset.imgnames)
    % display progress
    if mod(i,round(length(dataset.imgnames)/10))==0
        fprintf('test: %d/%d\n',i,length(dataset.imgnames));
    end        
    
    % read in the image
    img = imread(fullfile(dataset.dir, dataset.imgnames{i}));
            
    try 
    % compute confidence of positive classification and bounding boxes
    [C{i},BB{i}] = detect_fun(img,model);
    catch err
        C{i} = single([]);
        BB{i} = single([]);
        fprintf(1,'error with image %s\n', dataset.imgnames{i});
    end
    
    % evaluate
    gt = dataset.bboxes{i};
    if isempty(gt)
        FP{i} = true(length(C{i}),1);
        TP{i} = false(length(C{i}),1);
    else
        labels = false(length(C{i}),1);
        for g=1:size(gt,1)
            inter_width = max(0, min(BB{i}(3,:),gt(g,3))-max(BB{i}(1,:),gt(g,1)));
            inter_height = max(0, min(BB{i}(4,:),gt(g,4))-max(BB{i}(2,:),gt(g,2)));
            inter_area = inter_width.*inter_height;
            
            union_area = (BB{i}(4,:)-BB{i}(2,:)+1).*(BB{i}(3,:)-BB{i}(1,:)+1) + ...
                (gt(g,4)-gt(g,2)+1)*(gt(g,3)-gt(g,1)+1) - ...
                inter_area;
            
            overlaps = inter_area ./ union_area;
            
            labels(overlaps>.5) = true;
        end
        TP{i} = labels;
        FP{i} = ~labels;
    end
end


% compute AP
FP = cell2mat(FP);
TP = cell2mat(TP);
C = cell2mat(C);
[~,ids] = sort(-C);

TP = cumsum(TP(ids));
FP = cumsum(FP(ids));

npos = sum(cellfun(@(x) size(x,1), dataset.bboxes));
rec = TP/npos;
prec = TP./(TP+FP);

ap = VOCap(rec, prec);

fprintf(1,'Evaluation took %.2f minutes. AP: %.3f\n', etime(clock, t0)/60, ...
    100*ap);
