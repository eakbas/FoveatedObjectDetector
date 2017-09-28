function F = features(img, bin_length)
% This is a wrapper script for DPM's HOG feature extractor, which expects 
%   a color image. This wrapper checks if the image is grayscale and if so,
%   converts it to a 3-channel image. 

if size(img,3)==1
    img = cat(3, img, img, img);
end

F = HOGfeatures(img, bin_length);