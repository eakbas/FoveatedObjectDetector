function peripheral_filters = generate_pooling_regions(deg_per_pixel, ...
    N_e, N_theta, visual_field_radius_in_deg, visual)
%% 
% smallest eccentricity
e0_in_deg = .49;
e_max = visual_field_radius_in_deg;

%% 
visual_field_width = round( ...
    2*(visual_field_radius_in_deg./deg_per_pixel));

center_r = round(visual_field_width/2);
center_c = center_r;

regions = zeros(visual_field_width,visual_field_width, N_theta, N_e);

%% generate pooling regions
h=waitbar(0,'');
for raw_r=1:visual_field_width
    waitbar( raw_r/visual_field_width, h);
    parfor raw_c=1:visual_field_width       
        r = raw_r - center_r;
        c = raw_c - center_c;
        
        % convert (r,c) to polar: (eccentricity, angle)
        e = sqrt(r^2+c^2)*deg_per_pixel;
        a = mod(atan2(r,c),2*pi);
        
        for nt=1:N_theta
            for ne=1:N_e
                regions(raw_r, raw_c, nt, ne) = ...
                    FS_hntheta(nt-1,a,N_theta) * ...
                    FS_gne(ne-1,e,N_e,e0_in_deg, e_max);
            end
        end
    end
end
close(h)

regions = permute(regions,[3 4 1 2]);

%% compute centers & sizes of pooling regions
centers = zeros(2, N_theta, N_e);
areas = zeros(N_theta, N_e);
for nt=1:N_theta
    for ne=1:N_e
        mask = squeeze(regions(nt, ne, :, :));
        [r,c] = find(mask);
        centers(:, nt, ne) = [mean(r) mean(c)];
        areas(nt,ne) = length(r);
    end
end


%% 
filters = [];
filters.regions = regions;
filters.centers = centers;
filters.areas = areas;

%% offset coordinates for pooling regions
center_r = round(size(filters.regions,3)/2);
center_c = round(size(filters.regions,4)/2);
for nt=1:size(filters.regions,1)
    for ne=1:size(filters.regions,2)
        [rs,cs] = find(squeeze(filters. regions(nt,ne,:,:))>0);
        filters.offsets{nt,ne} = [rs-center_r cs-center_c];
        idxs = sub2ind(size(filters.regions), nt*ones(size(rs)), ...
            ne*ones(size(rs)), rs, cs);
        filters.weights{nt,ne} = filters.regions(idxs);
        
        % unique pixels for this cell
        [rs,cs] = find(squeeze(filters. regions(nt,ne,:,:))>=0.5);
        filters.uniq_pix{nt,ne} = [rs cs];
    end
end

%% visualize it
W = reshape(filters.regions, [N_e*N_theta  visual_field_width visual_field_width]);
W = squeeze(max(W,[],1));

if visual
    figure, imagesc(W), colorbar
end


%% discard the last eccentricity cells (b/c they are overflowing the visual field)
orig_filters = filters; 
filters.regions = filters.regions(:,1:end-1,:,:);
filters.centers = filters.centers(:,:,1:end-1);
filters.areas = filters.areas(:,1:end-1);
filters.offsets = filters.offsets(:,1:end-1);
filters.weights = filters.weights(:,1:end-1);
filters.uniq_pix = filters.uniq_pix(:,1:end-1);

%% visualize it
W = reshape(filters.regions, [(N_e-1)*N_theta  visual_field_width visual_field_width]);
W = squeeze(max(W,[],1));

if visual
    figure, imagesc(W), colorbar
end


%% discard the cells within the fovea
fovea_radius = 1.8;

peripheral_filters = filters;

% decide upto which cell to discard (along the radial axis)
for n_e=1:size(peripheral_filters.regions,2)
    offsets = abs(peripheral_filters.offsets{1,n_e});
    
    % see how much of this cell is within the fovea
    within = offsets(:,1)<=fovea_radius/deg_per_pixel & ...
        offsets(:,2)<=fovea_radius/deg_per_pixel;
    
    if sum(within)/length(within)<.5
        break
    end
end

% discard upto (not including) n_e
peripheral_filters.regions = peripheral_filters.regions(:,n_e:end,:,:);
peripheral_filters.centers = peripheral_filters.centers(:,:,n_e:end);
peripheral_filters.areas = peripheral_filters.areas(:,n_e:end);
peripheral_filters.offsets = peripheral_filters.offsets(:,n_e:end);
peripheral_filters.weights = peripheral_filters.weights(:,n_e:end);
peripheral_filters.uniq_pix = peripheral_filters.uniq_pix(:,n_e:end);

% discard low weight pixels
weight_threshold = .3;

peripheral_filters.regions(peripheral_filters.regions(:)<=.3) = 0;

for i=1:size(peripheral_filters.offsets,1)
    for j=1:size(peripheral_filters.offsets,2)
        valids = peripheral_filters.weights{i,j}>weight_threshold;
        
        peripheral_filters.offsets{i,j} = ...
            peripheral_filters.offsets{i,j}(valids,:);
        
        peripheral_filters.weights{i,j} = ...
            peripheral_filters.weights{i,j}(valids);
        
        peripheral_filters.areas(i,j) = sum(valids);        
    end
end

peripheral_filters = rmfield(peripheral_filters, 'uniq_pix');

%% visualize it
if visual
    visual_field_width = size(peripheral_filters.regions,3);
    W = reshape(peripheral_filters.regions, [numel(peripheral_filters.offsets) ...
        visual_field_width visual_field_width]);
    W = squeeze(max(W,[],1));
    figure, imagesc(W), colorbar
    hold on
    plot(visual_field_width/2 +.5, visual_field_width/2 +.5, 'w+');
    
    x = get(gca, 'xtick');
    y = get(gca, 'ytick');
    
    % convert to degrees
    x = (x - visual_field_width/2)*deg_per_pixel;
    y = (y - visual_field_width/2)*deg_per_pixel;
    
    set(gca, 'xticklabel', arrayfun(@(x) sprintf('%.1f',x), x, ...
        'uniformoutput',false));
    set(gca, 'yticklabel', arrayfun(@(x) sprintf('%.1f',x), y, ...
        'uniformoutput',false));
    
    % % draw foveal region
    % rectangle('position', [-fovea_radius/deg_per_pixel + visual_field_width/2 ...
    %     -fovea_radius/deg_per_pixel+visual_field_width/2 ...
    %     2*fovea_radius/deg_per_pixel 2*fovea_radius/deg_per_pixel], ...
    %     'edgecolor', 'k', 'linewidth', 2)
    %
    % for i=3:8:visual_field_width
    %     line([i i], [1 visual_field_width], 'color', 'w');
    %     line([1 visual_field_width],[i i], 'color', 'w');
    % end
    
    for x=219:8:323
        for y=219:8:323
            center = [x+4 - visual_field_width/2, y+4 - visual_field_width/2];
            if sqrt(sum(center.^2))<55
                
                rectangle('position', [x y 8 8], 'edgecolor', 'w');
            end
        end
    end
end