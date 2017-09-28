function g = FS_gne(n,e,N,e0, e_max)

if nargin==2
    N = 10;
    e0 = .5;
elseif nargin==3
    e0 = .5;
end


w = (log(e_max) - log(e0))/N;
g = FS_f((log(e)-log(e0)-w*(n+1))/w);
