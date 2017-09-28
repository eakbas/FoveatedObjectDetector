function model = init_model(num_scales)

if nargin<1
    num_scales = 40;
end

model = [];
model.templates = [];
model.bin_length = 8;
model.padx = 4;
model.pady = 4;
model.threshold = [];
model.use_left_right_flipping  = true;
model.num_feat_dims = 31;


% scales
model.scales = exp( linspace(log(0.1), log(1.5), num_scales)' );
