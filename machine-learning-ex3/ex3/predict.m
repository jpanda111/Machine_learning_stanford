function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1); % m X1 array;
X = [ones(m,1) X]; % mX(401) array;

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
% Theta1: 25X401(input layer 401, hidden layer 26) Theta2: 10X26 (hidden layer 26, output laryer 10)

h2=sigmoid(X*Theta1'); % mX25
h2=[ones(m,1) h2]; % mX26
h3=sigmoid(h2*Theta2'); % mX10
[h3_max, idx]=max(h3,[],2); % mX1
idx(idx==10)=0;
p=idx; % mX1

% =========================================================================

end
