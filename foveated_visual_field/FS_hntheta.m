function h = FS_hntheta(n,theta,N,t)
% Implements Eq.(10) in Freeman & Simoncelli paper.

if nargin==2
    N = 9;
    t=1/2;
elseif nargin==3
    t=1/2;
end



w = (2*pi)/N;

theta = theta + w/4;

if n==0 && theta>(3/2)*pi
    theta = theta - 2*pi;
end
h = FS_f((theta - (w*n + 0.5*w*(1-t)))/w);
