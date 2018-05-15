function plotDecisionBoundary(theta, X, y)
%PLOTDECISIONBOUNDARY Plots the data points X and y into a new figure with
%the decision boundary defined by theta
%   PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the 
%   positive examples and o for the negative examples. X is assumed to be 
%   a either 
%   1) Mx3 matrix, where the first column is an all-ones column for the                           
%      intercept.
%   2) MxN, N>3 matrix, where the first column is all-ones

% Plot Data
plotData(X(:,2:3), y);
hold on

if size(X, 2) <= 3 % 2D plot is okay; size(X,2) = # of columns= n+1 = # of features
    % Only need 2 points to define a line, so choose two endpoints, set the range
    plot_x = [min(X(:,2))-2,  max(X(:,2))+2]; % find X2's plot range;

    % Calculate the decision boundary line
    plot_y = (-1./theta(3)).*(theta(2).*plot_x + theta(1));
		% plot_y is X3, which make theta1+theta2*x2+theta2*x3=0
    % Plot, and adjust axes for better viewing
    plot(plot_x, plot_y)
    
    % Legend, specific for the exercise
    legend('Admitted', 'Not admitted', 'Decision Boundary')
    axis([30, 100, 30, 100])
else
    % Here is the grid range
    u = linspace(-1, 1.5, 50);
    v = linspace(-1, 1.5, 50);

    z = zeros(length(u), length(v)); % 50X50 array;
    % Evaluate z = theta*x over the grid
		% mapFeature(x,y)=[1 x y x^2 x*y y^2 x^3 x^2*y x*y^2 y^3 x^4 x^3*y x^2*y^2 x*y^3 y^4 ... y^6]; create quatric features 
    for i = 1:length(u)
        for j = 1:length(v)
            z(i,j) = mapFeature(u(i), v(j))*theta; % 1X28 * (n+1)X1 array 
						fprintf(' %f \n', z(i,j));
						fprintf(' %f \n', theta);
        end
    end
    z = z'; % important to transpose z before calling contour

    % Plot z = 0
    % Notice you need to specify the range [0, 0]
    contour(u, v, z, [0, 0], 'LineWidth', 2)
end
hold off

end
