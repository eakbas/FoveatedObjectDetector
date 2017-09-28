function f = flipfeat(f)
% Horizontally flip HOG features (or filters).
%   f = flipfeat(f)
% 
%   Used for learning models with mirrored filters.
%
%   THIS VERSION DOES NOT USE THE BOUNDARY TRUNCTATION DIM (31 features)
%
% Return value
%   f   Output, flipped features
%
% Arguments
%   f   Input features

% flip permutation
p = [10  9  8  7  6  5  4  3  2 ... % 1st set of contrast sensitive features
      1 18 17 16 15 14 13 12 11 ... % 2nd set of contrast sensitive features
     19 27 26 25 24 23 22 21 20 ... % Contrast insensitive features
     30 31 28 29];                % Gradient/texture energy features
     
f = f(:,end:-1:1,p);


