function [] = q1solution()
load('q1_data.mat');
phi=length(q1y_train(q1y_train==1))/length(q1y_train)

tmp=q1x_train(q1y_train==0,:);
tmp=[sum(tmp(:,1)) sum(tmp(:,2))];
tmp=tmp.';
mu0=tmp/length(q1y_train(q1y_train==0))

tmp=q1x_train(q1y_train==1,:);
tmp=[sum(tmp(:,1)) sum(tmp(:,2))];
tmp=tmp.';
mu1=tmp/length(q1y_train(q1y_train==1))

sigma=0;



for i=1:length(q1y_train)
    if q1y_train(i)==1
        xi=q1x_train(i,:);
        xi=xi.';
        tmp=xi-mu1;
        tmpt=tmp.';
        sigma = sigma + tmp*tmpt;
        
    else
        xi=q1x_train(i,:);
        xi=xi.';
        tmp=xi-mu0;
        tmpt=tmp.';
        sigma = sigma + tmp*tmpt;
    end

end

sigma=sigma/length(q1y_train)

w0=log(phi/(1-phi)-1/2*(mu1.'/sigma*mu1)+1/2*(mu0.'/sigma*mu0));

w=sigma\mu1-sigma\mu0;

w=[w0;w];
wt=w.';

total = length(q1y_test);
correct = 0;
for i=1:total
    xi=q1x_test(i,:);
    xi=[1 xi];
    xi=xi.';
    if wt*xi <= 0
       if q1y_test(i)==0
          correct = correct + 1; 
       end
    else
       if q1y_test(i)==1
          correct = correct + 1; 
       end
    end
    
end

accuracy = correct/total

end

