function textprogressbar(x)


backstr = repmat('\b',1,10);
if ischar(x)
    % string argument given
    fprintf(1,'%s%10s',x,'');
elseif isnumeric(x)
    if x==-1
        % end
        fprintf(1,'\n');
    else
        % assuming x is between 0-100        
        fprintf(backstr);
        fprintf(1,'%9.2f%%', x);
    end
end