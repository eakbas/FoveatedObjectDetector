function [cost,g] = latent_LDA_cost_function(model, pos_feats, mu0s, regC, A, B)


if isempty(A)
    A = [model.templates.A]';
    B = [model.templates.B]';
end


% for FOD, compute only the foveal templates
if isfield(model.templates, 'foveal')
    model.templates = model.templates([model.templates.foveal]);
end

cost = 0;
g = 0;
for t=1:length(model.templates)    
    % LDA weights
    w = [A(t)*model.templates(t).w ; B(t)];
    cost = cost + 0.5*sum(w.^2)/length(w);
    
    % positive examples
    pos_resps = [pos_feats{t} ones(size(pos_feats{t},1),1)]*w;    
    %pos_exprw = exp(-pos_resps);
    %pos_margin_violations = log(1+pos_exprw);%max(0,1-pos_resps);
    pos_margin_violations = max(0,1-pos_resps);
    cost = cost + regC*sum(pos_margin_violations);
    
    % negative background
    neg_resp = [mu0s{t};1]'*w;
    %neg_exprw = exp(neg_resp);
    %neg_margin_violation = log(1+neg_exprw); %max(0,1+neg_resp);
    neg_margin_violation = max(0,1+neg_resp);
    cost = cost + regC*1000*neg_margin_violation;
    
    % gradient
    if nargout>1
        % compute gradient
        % A
%         g(t) = A(t)*sum(model.templates(t).w.^2) ...
%             - regC*sum((pos_feats{t}*model.templates(t).w).*(pos_exprw./(1+pos_exprw))) ...
%             + regC*1000*sum((mu0s{t}'*model.templates(t).w).*(neg_exprw./(1+neg_exprw)));
         g(t) = (1/length(w))*A(t)*sum(model.templates(t).w.^2) ...
            - regC*sum((pos_margin_violations>0).*(pos_feats{t}*model.templates(t).w)) ...
            + regC*1000*(neg_margin_violation>0)*(mu0s{t}'*model.templates(t).w);
        
        % B
%         g(t+length(model.templates)) = B(t) ...
%             - regC*sum((pos_exprw./(1+pos_exprw))) ...
%             + regC*1000*sum((neg_exprw./(1+neg_exprw)));
        g(t+length(model.templates)) = (1/length(w))*B(t) ...
            - regC*sum(pos_margin_violations>0) ...
            + regC*1000*(neg_margin_violation>0);
    end
end