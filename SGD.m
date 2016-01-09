function [sae1OptTheta] = SGD(funObj,theta,data,labels,options,varargin)
   
        
       for i=1:options.maxIter
             var=['iteration::',num2str(i)];
             disp(var);
             data = data(:,randperm(size(data,2)));
             r = randi([1 size(data,2)],1,1);
            [cost,grad] = funObj(theta,data(:,r),labels(:,r));
             cost
             theta = theta - 0.01 .* grad;
       end
       sae1OptTheta = theta;
end