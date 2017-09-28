%% configuration  (see "determine_scaling.m" for the settings here)
deg_per_pixel = 0.037;

visual_field_radius_in_deg = 10;

N_theta = 30;
% constants for Freeman-Simoncelli model: 
N_e = 8; % for V1 (see determine_scaling.m)
N_e = 4; % for V2 (see determine_scaling.m)


% generate pooling regions
peripheral_filters = generate_pooling_regions(deg_per_pixel, ...
    N_e, N_theta, visual_field_radius_in_deg, true);

save('peripheral_filters_30x4.mat', 'peripheral_filters', ...
    '-v7.3');
