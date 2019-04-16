## Copyright (C) 2018 Jarid
## 
## This program is free software: you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
## 
## This program is distributed in the hope that it will be useful, but
## WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see
## <https://www.gnu.org/licenses/>.

## -*- texinfo -*- 
## @deftypefn {} {@var{retval} =} testCases (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: Jarid <Jarid@DESKTOP-05UUISS>
## Created: 2018-11-12

cost = computeCost( [1 2; 1 3; 1 4; 1 5], [7;6;5;4], [0.1;0.2] );
disp(sprintf('Case 1\n computeCost = %0.9f\n', cost));

cost = computeCost( [1 2 3; 1 3 4; 1 4 5; 1 5 6], [7;6;5;4], [0.1;0.2;0.3])
disp(sprintf('Case 2\n computeCost = %0.9f\n', cost));

[theta J_hist] = gradientDescent([1 5; 1 2; 1 4; 1 5],[1 6 4 2]',[0 0]',0.01,1000);
disp(sprintf('theta = %0.9f\n', theta));

[Xn mu sigma] = featureNormalize([1 ; 2 ; 3]);
disp(sprintf('Case 1 featureNormalize() = \n'));
disp(Xn)
disp(sprintf('mu = %f\n', mu));
disp(sprintf('sigma = %f\n', sigma));

[Xn mu sigma] = featureNormalize(magic(3));
disp(sprintf('Case 2 featureNormalize() = \n'));
disp(Xn)
disp(sprintf('mu = \n'));
disp(mu)
disp(sprintf('sigma = \n'));
disp(sigma)

X = [ 2 1 3; 7 1 9; 1 8 1; 3 7 4 ];
y = [2 ; 5 ; 5 ; 6];
[theta J_hist] = gradientDescentMulti(X, y, zeros(3,1), 0.01, 10);
disp(sprintf('\nCase 1 gradientDescentMulti\n'));
disp(sprintf('theta = \n'));
disp(theta);
disp(sprintf('\nJ_hist = \n'));
disp(J_hist);


X = [ 2 1 3; 7 1 9; 1 8 1; 3 7 4 ];
y = [2 ; 5 ; 5 ; 6];
[theta J_hist] = gradientDescentMulti(X, y, [0.1 ; -0.2 ; 0.3], 0.01, 10);
disp(sprintf('\nCase 2 gradientDescentMulti\n'));
disp(sprintf('theta = \n'));
disp(theta);
disp(sprintf('\nJ_hist = \n'));
disp(J_hist);


X = [ 2 1 3; 7 1 9; 1 8 1; 3 7 4 ];
y = [2 ; 5 ; 5 ; 6];
disp(sprintf('\nCase 1 normalEqn\n'));
theta = normalEqn(X,y)