function visualizeBoundary(X, y, model, varargin)
%VISUALIZEBOUNDARY plots a non-linear decision boundary learned by the SVM
%   VISUALIZEBOUNDARYLINEAR(X, y, model) plots a non-linear decision 
%   boundary learned by the SVM and overlays the data on it

% Plot the training data on top of the boundary
plotData(X, y)

% Make classification predictions over a grid of values
x1plot = linspace(min(X(:,1)), max(X(:,1)), 100)'; % 100X1 row of X interpolation
x2plot = linspace(min(X(:,2)), max(X(:,2)), 100)'; % 100X1 column of X interpolation
[X1, X2] = meshgrid(x1plot, x2plot); % 100X100 
%X1=(x1plot rows copy as each row, copy 100 times); X2=(x2plot rows copy as each columns);
vals = zeros(size(X1)); % 100X100
for i = 1:size(X1, 2)
   this_X = [X1(:, i), X2(:, i)];
   vals(:, i) = svmPredict(model, this_X);
end

% Plot the SVM boundary
hold on
contour(X1, X2, vals, [0.5 0.5], 'b');
hold off;

end
