disp(sprintf("test case 1 sigmoid(-5)\n"));
sigmoid(-5)

disp(sprintf("test case 2 sigmoid(0)\n"));
sigmoid(0)

disp(sprintf("test case 3 sigmoid([4 5 6])\n"));
sigmoid([4 5 6])

disp(sprintf("test case 4 sigmoid([-1; 0; 1])\n"));
sigmoid([-1; 0; 1])

V = reshape(-1:.1:.9, 4, 5);
disp(sprintf("test case 5 sigmoid(V)\n"));
sigmoid(V)
 

disp(sprintf("test case 1 costFunction(theta, X, y)\n"));
theta = [-2 -1 1 2]';
X = [ones(3,1) magic(3)];
y = [1 0 1]';
% un-regularized
[j g] = costFunction(theta, X, y);
disp(sprintf("j = \n"));
disp(j)
disp(sprintf("\n"));
disp(sprintf("g = \n"));
disp(g)
 
 
% regularized
theta = [-2 -1 1 2]';
X = [ones(3,1) magic(3)];
y = [1 0 1]';
[j g] = costFunctionReg(theta, X, y, 4);
% note: also works for ex3 lrCostFunction(theta, X, y, 4)
disp(sprintf("test case 2 costFunctionReg(theta, X, y, 4)\n"));
disp(sprintf("j = \n"));
disp(j)
disp(sprintf("\n"));
disp(sprintf("g = \n"));
disp(g)
  
 
%{
X = [1 1 ; 1 2.5 ; 1 3 ; 1 4];
theta = [-3.5 ; 1.3];
disp(sprintf("test case 1 predict(X, theta)\n"));
predict(X, theta)
%}
 
 
 
 