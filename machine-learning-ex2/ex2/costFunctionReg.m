function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta)); % 28X1 array;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
% theta: 28X1; X: 118X28; y: 118X1;

h = sigmoid((X*theta)'); % h: 1Xm array;
theta_2=theta([2:end],:); % remove theta0: 27X1;
J = -(log(h)*y+log(1-h)*(1-y))/m + lambda*theta_2'*theta_2/(2*m); % a number
grad(1,:) = (h-y')*X(:,1)/m; % grad: 1X1 array;
grad([2:end],:) = ((h-y')*X(:,[2:end]))'/m + lambda*theta_2/m; % grad: 27X1 array;

% =============================================================

end
