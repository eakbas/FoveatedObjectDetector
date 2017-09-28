function [mu,cov] = compute_mu_cov_of_background_examples(dataset, N, ...
    th, tw, model)

% choose N images randomly
N = min(N, length(dataset.imgnames));
ids = randperm(length(dataset.imgnames));
imgids = ids(1:N);


rseed = RandStream('mt19937ar','Seed',sum(clock()));
RandStream.setGlobalStream(rseed);


% first pass: compute mu
textprogressbar(sprintf('Computing %dx%d mu: ',th,tw));
mu = zeros(model.num_feat_dims*th*tw,1);
n = 0;
for i=1:length(imgids)  
    % read image
    img = imread(fullfile(dataset.dir, dataset.imgnames{imgids(i)}));
    
    feats = features(double(imresize(img,model.scales(randi(length(model.scales))))), ...
         model.bin_length);
    feats = feats(:,:,1:31);
    feats = padfeatures(feats, model.pady+1, model.padx+1);
    
    % collect all sub-windows
    F = return_all_feature_vectors(feats, th, tw,2);
    F2 = return_all_feature_vectors(flipfeat(feats), th, tw,2);
%         F = zeros(tw*th*size(feats,3), ...
%             (size(feats,1)-th+1)*(size(feats,2)-tw+1));
%         kk =  1;
%         for x=1:size(feats,2)-tw+1
%             for y=1:size(feats,1)-th+1
%                 f = feats(y:y+th-1, x:x+tw-1, :);
%                 F(:,kk) = double(f(:));
%                 kk=kk+1;
% %                 f = double(f(:));
% %                 c = f*f';
% %                 valid = ~isnan(c);
% %                 cov(valid) = cov(valid) + c(valid);
% %                 n = n + valid;
%             end
%         end
     
    mu = mu + sum(F,2) + sum(F2,2);
    n = n + size(F,2) + size(F2,2);
    
    textprogressbar(100*i/length(imgids));
end
mu = mu/n;
textprogressbar(-1);




% make sure same images and scales will be used for both MU and
% SIGMA calculation
RandStream.setGlobalStream(rseed);


% second pass: compute cov
% parallel version: causes out of memory problems
textprogressbar('Computing covariance: ');
cov = zeros(th*tw*model.num_feat_dims,th*tw*model.num_feat_dims);
for i=1:length(imgids)
    % read image
    img = imread(fullfile(dataset.dir, dataset.imgnames{imgids(i)}));
    
    feats = features(double(imresize(img,model.scales(randi(length(model.scales))))), ...
         model.bin_length);
    feats = feats(:,:,1:31);
    feats = padfeatures(feats, model.pady+1, model.padx+1);
    
%         % slide window and compute covariance
%         F = zeros(tw*th*size(feats,3), ...
%             (size(feats,1)-th+1)*(size(feats,2)-tw+1));
%         kk =  1;
%         for x=1:size(feats,2)-tw+1
%             for y=1:size(feats,1)-th+1
%                 f = feats(y:y+th-1, x:x+tw-1, :) - mu_full;
%                 F(:,kk) = double(f(:));
%                 kk=kk+1;
% %                 f = double(f(:));
% %                 c = f*f';
% %                 valid = ~isnan(c);
% %                 cov(valid) = cov(valid) + c(valid);
% %                 n = n + valid;
%             end
%         end
    F = return_all_feature_vectors(feats, th, tw,2);
    F = bsxfun(@minus, F, mu);
    F2 = return_all_feature_vectors(flipfeat(feats), th, tw,2);
    F2 = bsxfun(@minus, F2, mu);
   
    cov = cov + F*F' + F2*F2';              
    
    textprogressbar(100*i/length(imgids));
end
cov = cov./n;
textprogressbar(-1);



