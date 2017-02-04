function [ boundary ] = Boundary( labels, features, model )
%UNTITLED Return val to plot the classification boundary 
%   Detailed explanation goes here


xplot = linspace(min(features(:,1)), max(features(:,1)), 100)';
yplot = linspace(min(features(:,2)), max(features(:,2)), 100)';
[X, Y] = meshgrid(xplot, yplot);
vals = zeros(size(X));
for i = 1:size(X, 2)
   x = [X(:,i),Y(:,i)];
   % Need to use evalc here to suppress LIBSVM accuracy printouts
   [T,predicted_labels, accuracy, decision_values] = ...
       evalc('svmpredict(ones(size(x(:,1))), x, model)');
   clear T;
   vals(:,i) = decision_values;
end

boundary.X      = X;
boundary.Y      = Y;
boundary.vals   = vals;

end

