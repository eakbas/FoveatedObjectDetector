classdef nonfov_HOG_covariance
    properties
        max_half_width
        max_half_height
        SIGMA
        MU
        count_cov
        padx
        pady
        bin_len
        outfile
        num_images
        scales
    end
    
    methods
        function obj = nonfov_HOG_covariance(outfile, dataset, ...
                padx, pady, bin_len, max_half_width, max_half_height, ...
                scales, num_images)
            % Constructor method. Estimates the  covariances of all
            % cell-pairs.
            
            
             % has this covariance object been computed before?
            if exist(outfile, 'file')
                % yes
                fprintf(1,'%s exists. Loading it from the disk...\n', ...
                    outfile);
                
                tmp = load(outfile);
                obj = tmp.obj;
                return
            else
                % no. compute it.
                fprintf(1,'%s doesn''t exist. Computing it...\n', ...
                    outfile);
            end
            
            
            obj.outfile = outfile;
            obj.max_half_width = max_half_width;
            obj.max_half_height = max_half_height;
            obj.padx = padx;
            obj.pady = pady;
            obj.bin_len = bin_len;
            obj.num_images = num_images;
            obj.scales = scales;
            
            
            rseed = RandStream('mt19937ar','Seed',sum(clock()));
            RandStream.setGlobalStream(rseed);
            
            
            % Estimates the  covariances of all cell-pairs.
            t0 = clock();
            sum_v1r = 0;
            count_v1r = 0;
            textprogressbar('Computing MU: ');
            for i=1:num_images
                img = imread(fullfile(dataset.dir, ...
                    dataset.imgnames{i}));
                
                
                % extract features
                feats = features(imresize(double(img), ...
                    scales(randi(length(scales)))), bin_len);
                feats = feats(:,:,1:31);
                feats = padfeatures(feats, padx, pady);
                flipped_feats = flipfeat(feats);
                
                feats = reshape(feats, size(feats,1)*size(feats,2),[]);
                sum_v1r = sum_v1r + double(sum(feats, 1));
                count_v1r = count_v1r + size(feats,1);
                
                
                flipped_feats = reshape(flipped_feats, size(flipped_feats,1)*...
                    size(flipped_feats,2), []);
                sum_v1r = sum_v1r + double(sum(flipped_feats, 1));
                count_v1r = count_v1r + size(flipped_feats, 1);
                
                
                textprogressbar(100*i/num_images);
            end
            obj.MU = bsxfun(@times, 1./count_v1r, sum_v1r);
            textprogressbar(-1);
            
            fprintf(1,'MU computed in %.2f minutes.\n', etime(clock(), t0)/60);
            
            
            % make sure same images and scales will be used for both MU and
            % SIGMA calculation
            RandStream.setGlobalStream(rseed);
            
            
            t0 = clock();
            sum_cov = cell(2*max_half_height+1, max_half_width+1);
            for i=1:numel(sum_cov); sum_cov{i} = 0; end
            count_cov = zeros(size(sum_cov));
            textprogressbar('Computing SIGMA: ');
            for i=1:num_images
                img = imread(fullfile(dataset.dir, ...
                    dataset.imgnames{i}));
                
                
                % extract features                
                feats = features(imresize(double(img), ...
                    scales(randi(length(scales)))), bin_len);
                feats = feats(:,:,1:31);
                feats = padfeatures(feats, padx, pady);
                flipped_feats = flipfeat(feats);
                nrows = size(feats,1);
                ncols = size(feats,2);
                
                feats = reshape(feats, nrows*ncols,[]);
                flipped_feats = reshape(flipped_feats, nrows*ncols, []);
                
                feats = bsxfun(@minus, feats, obj.MU);
                flipped_feats = bsxfun(@minus, flipped_feats, obj.MU);
                
                for x=1:5:ncols  % every N cells is an effort to get more 'independent' data
                    for y=1:5:nrows
                        for dx=0:max_half_width
                            for dy=-max_half_height:max_half_height
                                nx = x + dx;
                                ny = y + dy;
                                
                                if nx<1 || ny<1 || nx>ncols || ny>nrows
                                    continue
                                end
                                
                                sum_cov{dy+max_half_height+1, dx+1} = ...
                                    sum_cov{dy+max_half_height+1, dx+1} + ...
                                    feats(y+(x-1)*nrows,:)'* ...
                                    feats(ny+(nx-1)*nrows,:) + ...
                                    feats(ny+(nx-1)*nrows,:)'* ...
                                    feats(y+(x-1)*nrows,:) + ...
                                    flipped_feats(y+(x-1)*nrows,:)'* ... % flipped features
                                    flipped_feats(ny+(nx-1)*nrows,:) + ...
                                    flipped_feats(ny+(nx-1)*nrows,:)'* ...
                                    flipped_feats(y+(x-1)*nrows,:);
                                
                                count_cov(dy+max_half_height+1, dx+1) = ...
                                    count_cov(dy+max_half_height+1, dx+1) + 4;
                            end
                        end
                    end
                end
                
                textprogressbar(100*i/num_images);
            end
           
          
            for i=1:numel(count_cov)
                if count_cov(i)>0
                    sum_cov{i} = sum_cov{i}/count_cov(i);
                end
            end
            obj.count_cov = count_cov;
            obj.SIGMA = sum_cov;
            clear sum_cov;
            textprogressbar(-1);
            fprintf(1,'SIGMA computed in %.2f minutes.\n', etime(clock(), t0)/60);
            
            % save the result
            save(outfile, 'obj');
        end
        
        
        
        
        function [mu,cov] = construct_mu_sigma(obj, h, w)
            % Constructs the MU and SIGMA for given width w and height h.
            
            % convert indices to coordinates
            [~,~,fs] = ind2sub([h w length(obj.MU)], ...
                1:(w*h*length(obj.MU)));
            
            % mu
            mu = obj.MU(fs)';
            
            
            % cov
%             tic
%             cov = zeros(h*w*length(obj.MU));
%             for r=1:h*w*length(obj.MU)
%                 for c=1:h*w*length(obj.MU)
%                     %[y_r,x_r,f_r] = ind2sub([h w length(obj.MU)], r);
%                     [y_r,x_r,f_r] = my_ind2sub([h w length(obj.MU)], r);
%                     %assert(y_r==q1 & x_r==q2 & f_r==q3);
%                     
%                     %[y_c,x_c,f_c] = ind2sub([h w length(obj.MU)], c);
%                     [y_c,x_c,f_c] = my_ind2sub([h w length(obj.MU)], c);
%                     
%                     dy = y_c - y_r;
%                     dx = x_c - x_r;
%                     
%                     if dx<0
%                         dx = -dx;
%                         dy = -dy;
%                     end
%                     
%                     cov(r,c) = obj.SIGMA{dy+obj.max_half_height+1, ...
%                         dx+1}(f_r, f_c);
%                 end
%             end
%             toc
%             
%             cov2 = cov;
            
            num_dims = length(obj.MU);
            
            %tic
            cov = zeros(h*w*num_dims, h*w*num_dims);
            fs = (1:num_dims)';
            for r=1:h
                rs = ones(num_dims,1)*r;
                
                for c=1:w
                    cs = ones(num_dims,1)*c;
                    
                    for nr=1:h
                        nrs = ones(num_dims,1)*nr;
                        
                        for nc=1:w
                            ncs = ones(num_dims,1)*nc;
                            
                            dy = nr-r;
                            dx = nc-c;
                            
                            if dx<0
                                dx = -dx;
                                dy = -dy;
                            elseif dx==0
                                if dy<0
                                    dy = -dy;
                                end
                            end
                            
                            ids1 = sub2ind([h w num_dims], ...
                                rs, cs, fs);
                            ids2 = sub2ind([h w num_dims], ...
                                nrs, ncs, fs);
                            
                            [ids2, ids1] = meshgrid(ids2, ids1);
                            
                            ids = sub2ind(size(cov), ids1(:), ids2(:));
                            
                            
                            y = min(dy+obj.max_half_height+1, ...
                                size(obj.SIGMA,1));
                            x = min(dx+1, size(obj.SIGMA,2));
                            
                            cov(ids) = obj.SIGMA{y,x}(:);
                        end
                    end
                end
            end
            %toc
        end
    end
end



function [r,c,f] = my_ind2sub(dims, ind)
r = rem(ind-1,dims(1))+1;
term = (ind-r)/dims(1);
c = rem(term, dims(2))+ 1;
f = term/dims(2) - (c-1)/dims(2) + 1;
end
