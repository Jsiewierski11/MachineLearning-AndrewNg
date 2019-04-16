function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

numFeatures = size(X, 2);
tempTheta = zeros(size(theta));
%disp(sprintf('numFeatures = %f', numFeatures));

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. z
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %


    
    
    for i=1:numFeatures,
      J_Theta = sum( ((X * theta) - y)' * X(:, i) );
      J_Theta = J_Theta / m;
      newTheta = theta(i, 1) - (alpha * J_Theta);
      tempTheta(i, 1) = newTheta;  
    endfor
    
    theta = tempTheta;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
