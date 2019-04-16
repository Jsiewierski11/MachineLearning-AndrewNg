function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

J_Theta1 = 0;
J_Theta2 = 0;

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
  
    % Summing up values for J's 
    J_Theta1 = sum( (X * theta) - y);
    J_Theta2 = sum( ((X * theta) - y)' * X(:, 2) );
    
    
    J_Theta1 = J_Theta1 / m;
    J_Theta2 = J_Theta2 / m;
    
    
    newTheta1 = theta(1, 1) - (alpha * J_Theta1);
    newTheta2 = theta(2, 1) - (alpha * J_Theta2);
    
    theta(1, 1) = newTheta1;
    theta(2, 1) = newTheta2;
    
    
    %Resetting temp variables
    J_Theta1 = 0;
    J_Theta2 = 0;
    
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    
    #fprintf("J_history = %d\n", J_history(iter));

end

end