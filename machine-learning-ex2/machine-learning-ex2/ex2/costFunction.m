function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples
disp(sprintf("Number of training examples: %f\n", m));
numFeatures = size(X, 2);
tempTheta = zeros(size(theta));

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

hypothesis = X * theta;
hypothesis = sigmoid(hypothesis);

% calculating cost
for i=1:numFeatures,
  J = sum ( ( -y .* log(hypothesis) ) .- (1 .- y)  .* log(1 .- hypothesis) );  
endfor
J = J / m;

% calculating gradient
for i=1:numFeatures,
  J_Theta = sum( ((hypothesis) - y)' * X(:, i) );
  J_Theta = J_Theta / m;
  grad(i) = J_Theta;  
endfor




% =============================================================

end
