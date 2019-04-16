function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
numFeatures = size(X, 2);
thetaN = size(theta, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

hypothesis = X * theta;
hypothesis = sigmoid(hypothesis);

% calculating cost
for i=1:numFeatures,
  J = sum ( ( -y .* log(hypothesis) ) .- (1 .- y)  .* log(1 .- hypothesis) );  
endfor
J = J * (1/ m);

% calculating regularization
regSum = 0;
for i=2:numFeatures,
  regSum = regSum + theta(i)^2;
endfor
regSum = (lambda / (2*m)) * regSum;

% summing together
J = J + regSum;

% calculating gradient
for i=1:thetaN,
  if i == 1,
    J_Theta = sum( ((hypothesis) - y)' * X(:, i) );
    J_Theta = J_Theta / m;
    grad(i) = J_Theta;
  else
    J_Theta = sum( ((hypothesis) - y)' * X(:, i) );
    J_Theta = J_Theta / m;
    J_Theta = J_Theta + ( (lambda/m) * theta(i) );
    grad(i) = J_Theta;  
  endif
  
endfor




% =============================================================

end
