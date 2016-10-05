function [] = q2solution()
load('q2_data.mat');

xtrain = q2x_train.';
ytrain = q2t_train;
dimx = size(xtrain);
tmp = ones(1,dimx(2));
xtrain = [tmp;xtrain];
w = rand(dimx(1)+1,2);

t = 0;
while(true)
    
   t=t+1;
   w = w + 0.0005*gradall(w,xtrain,ytrain);

   if t>400000
       break;
   end
   
end

xtest = q2x_test.';
ytest = q2t_test;
dimx = size(xtest);
tmp = ones(1,dimx(2));
xtest = [tmp;xtest];

accuracy = 0;
correct = 0;
total = dimx(2);


for i=1:dimx(2)
    xi = xtest(:,i);
    yi = ytest(i);
    
    p1 = prob(1,xi,w);
    p2 = prob(2,xi,w);
    p3 = 1-p1-p2;

    if p1>=p2 && p1>=p3
        if yi == 1
           correct = correct + 1;
        end
    else
        if p2>=p1 && p2>=p3
            if yi == 2
                correct = correct + 1; 
            end
        else
            if yi == 3
                correct = correct + 1; 
            end
        end
    end
    
end

accuracy = correct/total

end

function [result] = gradall(w,x,y)
result = w;
dimw = size(w);
bigk = dimw(2) + 1;

for m=1:(bigk-1)
    result(:,m) = grad(m,w,x,y);
end

end



function [result] = grad(m,w,x,y)

dimx = size(x);

n = dimx(2);

result = zeros(dimx(1),1);


for i=1:n
    xi = x(:,i);

    if y(i) == m
       indicator = 1;
    else
       indicator = 0;
    end

    result = result + xi*(indicator-prob(m,xi,w));

end

end


function [result] = prob(k,x,w)

dim = size(w);
bigk = dim(2) + 1;

denominator = 1;
for j=1:(bigk-1)
   wj = w(:,j);
   wjt = wj.';
   denominator = denominator + exp(wjt*x);
end

if k<bigk
   wk = w(:,k);
   wkt = wk.';
   result = exp(wkt*x)/denominator;
else
   result = 1/denominator;
end

end


