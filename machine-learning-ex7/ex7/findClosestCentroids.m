function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%		centeroids=KXn; idx=mX1; X=mXn;

% Set K
K = size(centroids, 1); % K

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);% mX1

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
[m n] = size(X);

for i=1:m
	for j=1:K
		temp=sum((X(i,:)-centroids(j,:)).^2)/n;	
		if (j==1)	
			temp_old=temp;	
		endif
		if (temp <= temp_old)
			idx(i)=j;
			temp_old=temp;	
		endif
	endfor
endfor
% =============================================================
end

