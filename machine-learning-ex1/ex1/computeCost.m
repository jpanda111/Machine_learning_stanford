function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
% X is 97X2 array, theta is 2X1 array, y is 97X1 array
% J=(X*theta-y)'*(X*theta-y)/(2*m);

h = theta'*X';% 1X97 array, 97 is the training #
Error= h-y'; % 1X97 array
J=sum(Error.^2)/(2*m); % final result

% =========================================================================

end
