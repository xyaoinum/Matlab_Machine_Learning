function [] = q1solution_b()
A = double(imread('mandrill-small.tiff'));
imshow(uint8(round(A)));
K = 5;
N = size(A,1)*size(A,2);

x_set = squeeze(reshape(A,1,N,3))';

mu_set = ones(3,K);
rand_k = randperm(N);
for k=1:K
	mu_set(:,k) = x_set(:,rand_k(k));
end

pi_set = ones(1,K)*1/K;
N_set = ones(1,K)*N/K;
gamma_matrix = ones(N,K)/K;

sigma_set = zeros(3,3,K);
for k=1:K
	for n=1:N
		tmp = x_set(:,n)-mu_set(:,k);
		sigma_set(:,:,k) = sigma_set(:,:,k) + tmp*tmp';
	end
	sigma_set(:,:,k) = sigma_set(:,:,k)/N;	
end

llh = 0;

while(true)
	for n=1:N
		denominator = 0;
		for j=1:K
			denominator = denominator + pi_set(j)*normal(x_set(:,n),mu_set(:,j),sigma_set(:,:,j));
		end

		for k=1:K
			gamma_matrix(n,k) = pi_set(k)*normal(x_set(:,n),mu_set(:,k),sigma_set(:,:,k))/denominator;
		end
	end

	for k=1:K
		gamma_k = gamma_matrix(:,k);
		N_set(k) = sum(gamma_k);
		
		sum_muk = zeros(3,1);
		for n=1:N
			sum_muk = sum_muk + x_set(:,n).*gamma_k(n);
		end
		mu_set(:,k) = sum_muk/N_set(k);
		
		sigma_set(:,:,k) = zeros(3,3);
		for n=1:N
			tmp = x_set(:,n)-mu_set(:,k);
			sigma_set(:,:,k) = sigma_set(:,:,k) + gamma_matrix(n,k)*tmp*tmp';
		end
		sigma_set(:,:,k) = sigma_set(:,:,k)/N_set(k);
		
		pi_set(k) = N_set(k)/N;
	end
	
	prev_llh = llh;
	llh = 0;
	for n=1:N
		tmp = 0;
		for k=1:K
			tmp = tmp + pi_set(k)*normal(x_set(:,n),mu_set(:,k),sigma_set(:,:,k));
		end
		llh = llh + log(tmp);
	end

	if(abs(llh-prev_llh) < 2)
		break;
	end
	fprintf('difference of log-likelihood: %d (<2 means convergence)\n',abs(llh-prev_llh));
end

mu1_mu2_mu3_mu4_mu5 = mu_set
sigma_1 = sigma_set(:,:,1)
sigma_2 = sigma_set(:,:,2)
sigma_3 = sigma_set(:,:,3)
sigma_4 = sigma_set(:,:,4)
sigma_5 = sigma_set(:,:,5)
log_likelihood = llh


A = double(imread('mandrill-large.tiff'));
for i=1:size(A,1)
	for j=1:size(A,2)
		 A(i,j,:)=classify(squeeze(A(i,j,:)),pi_set,mu_set,sigma_set);
	end
end

figure
imshow(uint8(round(A)));

end


function [result] = normal(x,mu,sigma)
result = 1/sqrt(det(sigma)*(2*pi)^3)*exp(-1/2*(x-mu)'/sigma*(x-mu));
end

function [result] = classify(x,pi_set,mu_set,sigma_set)
	index = 0;
	prob = -1;
	for k=1:5
		tmp_prob = pi_set(k)*normal(x,mu_set(:,k),sigma_set(:,:,k));
		if tmp_prob > prob
			prob = tmp_prob;
			index = k;
		end
	end
	result = mu_set(:,index);
end

