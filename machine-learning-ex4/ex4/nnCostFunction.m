function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
% nn_params = [Theta1(:) ; Theta2(:)]; 10285X1 array;
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1)); % 25X401

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1)); % 10X26

% Setup some useful variables, m=5000;
m = size(X, 1);

% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1)); 
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
% m/i: sampling #; l: layer #;
% inputlayer to hiddenlayer: j=hiddenlayer #; k=inputlayer #;
% hiddenlayer to outputlayer: j=outputlayer #; k=hiddenlayer #;
% y=5000X1; theta1=25X401; theta2= 10X26; x=5000X400; h(x)=g(z3)=g(theta2*g(z2))=g(theta2*g(theta1*x))

X=[ones(m,1) X]; % 5000X401;
y_new=zeros(m,num_labels); % 5000X10;
J_m=zeros(m,1); % 5000X1;

% Theta1_grad=25X401; Theta2_grad=10X26; X=5000X401; 
triangle_1=zeros(size(Theta1));
triangle_2=zeros(size(Theta2));

for t=1:m
	% merge cost function together
	label=y(t);
	y_new(t,label)=1;
	
	% step1
	a_1=X(t,:); % 1X401
	z_2= a_1*Theta1'; % 1X25;
	a_2= [1 sigmoid(z_2)]; % 1X26;
	z_3= a_2*Theta2'; % 1X10;
	a_3= sigmoid(z_3); % 1X10;
	
	J_m(t)=-y_new(t,:)*log(a_3')-(1-y_new)(t,:)*log(1-a_3');
	
	% step2
	delta_3= a_3 - y_new(t,:); % 1X10;
	% step3
	delta_2= (delta_3*Theta2(:,2:end)).*sigmoidGradient(z_2); % 1X25;
	% step4
	triangle_2=triangle_2+delta_3'*a_2; % 10X26;
	triangle_1=triangle_1+delta_2'*a_1; % 25X401;
	
endfor

J=sum(J_m)/m + (lambda/(2*m))*(sum(diag(Theta1(:,2:end)*Theta1(:,2:end)'))+sum(diag(Theta2(:,2:end)*Theta2(:,2:end)')));

% step5

Theta1_grad(:,1)=triangle_1(:,1)/m ; % 25X401
Theta1_grad(:,2:end)=(triangle_1(:,2:end)+lambda*Theta1(:,2:end))/m;
Theta2_grad(:,1)=triangle_2(:,1)/m; % 10X26
Theta2_grad(:,2:end)=(triangle_2(:,2:end)+lambda*Theta2(:,2:end))/m;
% -------------------------------------------------------------
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
