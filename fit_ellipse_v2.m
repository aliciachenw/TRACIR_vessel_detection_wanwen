function ellipse = fit_ellipse_v2(x)
% https://www.mathworks.com/matlabcentral/fileexchange/15125-fitellipse-m
%FITELLIPSE   least squares fit of ellipse to 2D data
%
%   [Z, A, B, ALPHA] = FITELLIPSE(X)
%       Fit an ellipse to the 2D points in the 2xN array X. The ellipse is
%       returned in parametric form such that the equation of the ellipse
%       parameterised by 0 <= theta < 2*pi is:
%           X = Z + Q(ALPHA) * [A * cos(theta); B * sin(theta)]
%       where Q(ALPHA) is the rotation matrix
%           Q(ALPHA) = [cos(ALPHA), -sin(ALPHA); 
%                       sin(ALPHA), cos(ALPHA)]
%
%       Fitting is performed by nonlinear least squares, optimising the
%       squared sum of orthogonal distances from the points to the fitted
%       ellipse. The initial guess is calculated by a linear least squares
%       routine, by default using the Bookstein constraint (see below)
%
%   [...]            = FITELLIPSE(X, 'linear')
%       Fit an ellipse using linear least squares. The conic to be fitted
%       is of the form
%           x'Ax + b'x + c = 0
%       and the algebraic error is minimised by least squares with the
%       Bookstein constraint (lambda_1^2 + lambda_2^2 = 1, where 
%       lambda_i are the eigenvalues of A)
%
%   [...]            = FITELLIPSE(..., 'Property', 'value', ...)
%       Specify property/value pairs to change problem parameters
%          Property                  Values
%          =================================
%          'constraint'              {|'bookstein'|, 'trace'}
%                                    For the linear fit, the following
%                                    quadratic form is considered
%                                    x'Ax + b'x + c = 0. Different
%                                    constraints on the parameters yield
%                                    different fits. Both 'bookstein' and
%                                    'trace' are Euclidean-invariant
%                                    constraints on the eigenvalues of A,
%                                    meaning the fit will be invariant
%                                    under Euclidean transformations
%                                    'bookstein': lambda1^2 + lambda2^2 = 1
%                                    'trace'    : lambda1 + lambda2     = 1
%
%           Nonlinear Fit Property   Values
%           ===============================
%           'maxits'                 positive integer, default 200
%                                    Maximum number of iterations for the
%                                    Gauss Newton step
%
%           'tol'                    positive real, default 1e-5
%                                    Relative step size tolerance
%   Example:
%       % A set of points
%       x = [1 2 5 7 9 6 3 8; 
%            7 6 8 7 5 7 2 4];
% 
%       % Fit an ellipse using the Bookstein constraint
%       [zb, ab, bb, alphab] = fitellipse(x, 'linear');
%
%       % Find the least squares geometric estimate       
%       [zg, ag, bg, alphag] = fitellipse(x);
%       
%       % Plot the results
%       plot(x(1,:), x(2,:), 'ro')
%       hold on
%       % plotellipse(zb, ab, bb, alphab, 'b--')
%       % plotellipse(zg, ag, bg, alphag, 'k')
% 
%   See also PLOTELLIPSE
% Copyright Richard Brown, this code can be freely used and modified so
% long as this line is retained
x = x';
% Default parameters
params.constraint = 'bookstein';
params.maxits     = 200;
params.tol        = 1e-5;

% Constraints are Euclidean-invariant, so improve conditioning by removing
% centroid
centroid = mean(x, 2);
x        = x - repmat(centroid, 1, size(x, 2));

[z, a, b, alpha] = fitbookstein(x);

% Add the centroid back on
if ~isempty(z)
    z = z + centroid;
    ellipse.xc = z(1);
    ellipse.yc = z(2);
    ellipse.a = a;
    ellipse.b = b;
    ellipse.alpha = alpha;
else
    ellipse = [];
end % fitellipse

% ----END MAIN FUNCTION-----------%
function [z, a, b, alpha] = fitbookstein(x)
%FITBOOKSTEIN   Linear ellipse fit using bookstein constraint
%   lambda_1^2 + lambda_2^2 = 1, where lambda_i are the eigenvalues of A
% Convenience variables
m  = size(x, 2);
x1 = x(1, :)';
x2 = x(2, :)';
% Define the coefficient matrix B, such that we solve the system
% B *[v; w] = 0, with the constraint norm(w) == 1
B = [x1, x2, ones(m, 1), x1.^2, sqrt(2) * x1 .* x2, x2.^2];
% To enforce the constraint, we need to take the QR decomposition
[Q, R] = qr(B);
% Decompose R into blocks
R11 = R(1:3, 1:3);
R12 = R(1:3, 4:6);
R22 = R(4:6, 4:6);
% Solve R22 * w = 0 subject to norm(w) == 1
[U, S, V] = svd(R22);
w = V(:, 3);
% Solve for the remaining variables
v = -R11 \ R12 * w;
% Fill in the quadratic form
A        = zeros(2);
A(1)     = w(1);
A([2 3]) = 1 / sqrt(2) * w(2);
A(4)     = w(3);
bv       = v(1:2);
c        = v(3);
% Find the parameters
[z, a, b, alpha] = conic2parametric(A, bv, c);
end % fitellipse

function [z, a, b, alpha] = conic2parametric(A, bv, c)
% Diagonalise A - find Q, D such at A = Q' * D * Q
[Q, D] = eig(A);
Q = Q';
% If the determinant < 0, it's not an ellipse
if prod(diag(D)) <= 0 
    z = [];
    a = [];
    b = [];
    alpha = [];
else
% We have b_h' = 2 * t' * A + b'
t = -0.5 * (A \ bv);
c_h = t' * A * t + bv' * t + c;
z = t;
a = sqrt(-c_h / D(1,1));
b = sqrt(-c_h / D(2,2));
alpha = atan2(Q(1,2), Q(1,1));
end
end % conic2parametric

end