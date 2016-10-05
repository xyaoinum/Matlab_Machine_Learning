function [] = q3solution_b()
load q3.mat;
xtrain = q3x_train.';
ttrain = q3t_train;
xtest = q3x_test.';
ttest = q3t_test;

iterations = [5,50,100,1000,5000,6000];
for i=1:length(iterations)
   numIterations = iterations(i);
   w = zeros(size(xtrain,1),1);
   b = 0;

   for j=1:numIterations
      wgrad = w_grad(w,b,xtrain,ttrain);
      bgrad = b_grad(w,b,xtrain,ttrain);
      w = w - alpha(j)*wgrad;
      b = b - 0.01*alpha(j)*bgrad;
   end
   
   numIterations
   w
   b
   
   total = size(xtest,2);
   correct = 0;
   
   
   
   for k=1:total
       tk=ttest(k);
       xk = xtest(:,k);
       if tk*(w.'*xk+b) > 0
          correct = correct + 1;
       end
   end
   
   accuracy = correct/total
   
    
    
end


end

function [result] = w_grad(w,b,x,t)

sum = 0;
n = size(x,2);
for i=1:n
   xi = x(:,i);
   ti = t(i);
   if ti*(w.'*xi + b) < 1
       sum = sum + ti*xi;
   end
end

result = w - 5*sum;

end


function [result] = b_grad(w,b,x,t)

sum = 0;
n = size(x,2);
for i=1:n
   xi = x(:,i);
   ti = t(i);
   if ti*(w.'*xi + b) < 1
       sum = sum + ti;
   end
end

result = -5*sum;

end

function [result] = alpha(j)
result = 0.5/(1+j*0.5);
end


