function f = FS_f(x, t)
% Implements the function f in Eq. (9) of Freeman&Simoncelli paper. 

if nargin<2
    t = 0.5;
end

f = zeros(size(x));

% case 1
idx = x>-(1+t)/2 & x<=(t-1)/2;
f(idx) = (cos(0.5*pi*((x(idx)-(t-1)/2)/t))).^2;

% case 2
idx = x>(t-1)/2 & x<=(1-t)/2;
f(idx) = 1;

% case 3
idx = x>(1-t)/2 & x<=(1+t)/2;
f(idx) = -(cos(0.5*pi*((x(idx)-(1+t)/2)/t))).^2+1;
