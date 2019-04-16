disp(sprintf("test case 1: LrCostFunction(Unregularized) \ntheta = [-2; -1; 1; 2] \nX = [ones(5,1) reshape(1:15,5,3)/10] \ny = [1;0;1;0;1] >= 0.5"));
% input
theta = [-2; -1; 1; 2];
X = [ones(5,1) reshape(1:15,5,3)/10];
y = [1;0;1;0;1] >= 0.5;       % creates a logical array

% test the unregularized results
[J grad] = lrCostFunction(theta, X, y, 0)


disp(sprintf("test case 2: LrCostFunction(Regularized) \ntheta = [-2; -1; 1; 2] \nX = [ones(5,1) reshape(1:15,5,3)/10] \ny = [1;0;1;0;1] >= 0.5"));
% test the regularized results
lambda = 3;
[J grad] = lrCostFunction(theta, X, y, lambda)

disp(sprintf("test case 3: oneVsAll \nX = [magic(3) ; sin(1:3); cos(1:3)] \ny = [1; 2; 2; 1; 3] \nnum_labels = 3 \nlambda = 0.1\n"));
%input:
X = [magic(3) ; sin(1:3); cos(1:3)];
y = [1; 2; 2; 1; 3];
num_labels = 3;
lambda = 0.1;
[all_theta] = oneVsAll(X, y, num_labels, lambda)

disp(sprintf("test case 4: predictOneVsAll \nX = [1 7; 4 5; 7 8; 1 4] \nall_theta = [1 -6 3; -2 4 -3]\n"));
% input:
all_theta = [1 -6 3; -2 4 -3];
X = [1 7; 4 5; 7 8; 1 4];
predictOneVsAll(all_theta, X)

disp(sprintf("test case 5: predict \nX = reshape(sin(1:16), 8, 2 \nTheta1 = reshape(sin(0 : 0.5 : 5.9), 4, 3)) \nTheta2 = reshape(sin(0 : 0.3 : 5.9), 4, 5)"));
% input:
Theta1 = reshape(sin(0 : 0.5 : 5.9), 4, 3);
Theta2 = reshape(sin(0 : 0.3 : 5.9), 4, 5);
X = reshape(sin(1:16), 8, 2);
p = predict(Theta1, Theta2, X);