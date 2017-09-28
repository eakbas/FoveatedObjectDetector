function a = VOCap(rec,prec)
% This is the VOCap.m in the VOCdevkit of PASCAL 2012.

mrec=[0 ; rec ; 1];
mpre=[0 ; prec ; 0];
for i=numel(mpre)-1:-1:1
    mpre(i)=max(mpre(i),mpre(i+1));
end
i=find(mrec(2:end)~=mrec(1:end-1))+1;
a=sum((mrec(i)-mrec(i-1)).*mpre(i));
