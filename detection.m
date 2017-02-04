%% Scenario:
% x_targetg =       [ 500;  -500;   0;  900; 1800];
% y_targetg =       [-600;  -300;   0; 1000; 2000];
% TargetAltitudeg = [ 500;   500; 500;  600;  700];
%
% x_target =       [ 500;  -500;   0; 1000; 2000];
% y_target =       [-600;  -300;   0; 1000; 2000];
% TargetAltitude = [ 500;   500; 500;  500;  500];
%% Area parameters
% Name                        Value                 Unit    Type     Comment
% AreaWidth                   = single(2000);         % [m]   - single - Larghezza area target
% AreaLenght                  = single(2000);         % [m]   - single - Lunghezza area target
% BarycentreLat               = single(41.122533);    % [deg] - single - Latitudine del baricentro dell'area target
% BarycentreLon               = single(14.169928);    % [deg] - single - Longitudine del baricentro dell'area target
% BarycentreAltitude          = single(0);            % [m]   - single - Quota del baricentro dell'area target
% 
% %% Fixed target parameters
% % Name                        Value                 Unit    Type     Comment
% TargetNumber                = uint8(5);             % []    - uint8  - Quantizzazione numero di target
% MinTargetAltitude           = single(0);            % [m]   - single - Numero minimo di target
% MaxTargetAltitude           = single(10);           % [m]   - single - Numero massimo di target
% MinDistAmongTarget          = single(20);           % [m]   - single - Distanza minima tra i target
% MinTargetSurveillanceTime   = single(0);            % [s]   - single - Minimo tempo di sorveglianza dei target
% MaxTargetSurveillanceTime   = single(0);            % [s]   - single - Massimo tempo di sorveglianza dei target
% MaxECIcost                  = single(1);            % []    - single - Massimo valore dell'ECI cost dei target
% 
% %% Aircraft parameters
% % Name                        Value                         Unit    Type     Comment
% AircraftHorSpeed            = single(15);                   % [m/s] - single - Velocità orizzontale minima degli aircraft
% AircraftAltitude            = single(50);                   % [m]   - single - Distanza minima in quota tra le traiettorie degli aircraft
% XTakeoffAircraft            = single(-AreaWidth/2);         % [m]   - uint8  - Ascissa degli aircraft al decollo
% YTakeoffAircraft            = single(-AreaLenght/2);        % [m]   - uint8  - Ordinata degli aircraft al decollo
%
%%

load('allData.mat');

%% Initial parameters
%% set normalize to 1 if you want to normalize the data
init.normalize      = 1;
%% set tryTestSet to 1 if you want to test the testSet instead of all the data
init.tryTestSet     = 1;
%% set plotSet to 1 if you want to plot the data
init.plotSet        = 0;
%% Time when the safe zone is ended
init.endTraining    = 13000;
%% Time when the attack starts
init.startAttack    = 21061;
init.endTime        = numel(allData(:,2));
%% Type of solver: 2 for Schölkopf one-class, 5 for Tax and Duin
init.solver         = 5;
%% type of kernel: 2 RBF gaussian
init.kernel         = 2;
%% set crossValidate to 1 if you want to calculate best c and gamma value
init.crossValidate  = 0;
%% set plotcv to 1 if you want to plot cross validation results
init.plotcv         = 0;

%% Normalize Data
if init.normalize == 1
    data = mat2gray(allData(1:end,1:2));
else
    data = allData;
end

%% label and test set 
% Labels are necessary for accurancy estimation.
% Undetectable data are set to 1
testSet                 = data(init.endTraining:end,1:end);
badValues               = testSet(:,2) > 0.01; %%approximation of boundaries
numBadValues            = numel(testSet(badValues,2));
labelTestGood           = ones(init.startAttack-init.endTraining+1,1);
labelTestBad            = -ones(numBadValues,1);
labelTestUndetectable   = ones(init.endTime-init.startAttack-numBadValues,1);
labelTest               = [labelTestGood;labelTestBad;labelTestUndetectable];

%% label and training set 
trainSet        = data(1:17200,1:end);
labelTrain      = ones(length(trainSet),1);

%% Label and data prepared for training and test
labelData       = [labelTrain;labelTest];
preparedData    = [trainSet;testSet];


disp(['Time of evaluation   : ',    num2str(init.endTime),      ' ms']);
disp(['Time of training     : ',    num2str(init.endTraining),  ' ms']);
disp(['The attack starts at : ',    num2str(init.startAttack),  ' ms']);


if init.crossValidate == 1
    parameters = CrossValidation( labelTrain, trainSet, init.solver, init.kernel, init.plotcv);
end

%% Construct one-class SVM with RDF kernel (Gaussian)
if      init.solver == 2
    
    %one-class SVM according to Schölkopf with nu parameter 0.001 and gamma 100 kernel RBF (Gaussian -t 2)
    if init.crossValidate == 0
        model = svmtrain(labelTrain, trainSet(1:end,1:2), '-s 2 -t 2 -n 0.001 -g 100');
    else
        model = svmtrain(labelTrain, trainSet(1:end,1:2), ...
            sprintf('-s %d -t %d -c %f -g %f',init.solver,init.kernel, parameters.C, parameters.gamma));
    end
elseif  init.solver == 5
    %one-class SVM according to Tax and Duin with C parameter 0.0033 and gamma 100 kernel RBF (Gaussian -t 2)
    if init.crossValidate == 0
        model = svmtrain(labelTrain, trainSet(1:end,1:2), '-s 5 -t 2  -h 0 -c 0.0033 -g 100');
    else
        model = svmtrain(labelTrain, trainSet(1:end,1:2), ...
            sprintf('-s %d -t %d -c %f -g %f',init.solver,init.kernel, parameters.C, parameters.gamma));
    end
end

%Legacy
%model = svmtrain(labels, allData(1:end,1:2), '-s 2 -t 2 -n 0.001 -g 0.001 ');
%% one-class SVM according to Tax and Duin (SVDD) with penalty C 0-9 and gamma 0.2, RBF kernel (Gaussian -t 2)
%model = svmtrain(labelTrain, trainSet(1:end,1:2), '-s 5 -t 2  -c 0.9 -g 0.2');
%-n 0.001 -g 0.001
% Use the same data for label prediction and the test data
%[predicted_labels] = svmpredict(labels2, allData(1:end,1:2), model);%
%[predicted_labels] = svmpredict(labels, allData(1:end,1:2), model);

%% Predict the Anomaly
if init.tryTestSet == 1
    %predict the test set
    [predicted_labels] = svmpredict(labelTest, testSet(1:end,1:2), model);
else
    %predict all the set *bad way*
    [predicted_labels] = svmpredict(labelData, preparedData(1:end,1:2), model);
end

%% Predicted indices
inside_indices  = find(predicted_labels > 0);
out_indices     = find(predicted_labels == -1);

%% Plot the data sets
if init.plotSet == 1
    % PLOT DATA:
    % blue  = all data prepared
    % green = training set
    % red   = test set
    figure; hold on;
    subplot(311);
    scatter(trainSet(:,1), trainSet(:,2), 30, 'green');
    subplot(312);
    scatter(testSet(:,1), testSet(:,2), 30, 'red');
    subplot(313)
    scatter(preparedData(:,1), preparedData(:,2), 30, 'blue');
end
% Scatterplot of all support vectors, small red circles
%scatter(model.SVs(:,1), model.SVs(:,2), 20, 'red');

%% Calculate boundary
boundary = Boundary(labelData, preparedData, model);

%% Plot the Results
if init.tryTestSet == 1
    % PLOT DATA:
    % blue      = all the prepared data
    % green     = test data in the one class
    % red       = test data out of the one class
    % yellow    = training set of the one class
    figure; hold on;
    title('Support Vector Machine Classifier -- GPS Spoofing Data')
    xlabel('Position Error (m)')
    ylabel('Velocity Error (m)')
    scatter(preparedData(:,1),preparedData(:,2),30,'blue');
    scatter(testSet(inside_indices,1), testSet(inside_indices,2), 10, 'green');
    scatter(testSet(out_indices,1), testSet(out_indices,2), 10, 'red');
    scatter(trainSet(:,1), trainSet(:,2), 10, 'yellow');
    contour(boundary.X,boundary.Y, boundary.vals, [0 0], 'LineWidth', 1.5, 'Color', 'r');
    hold off;
else
    % PLOT DATA:
    % blue      = all the prepared data
    % green     = test data in the one class
    % red       = test data out of the one class
    % yellow    = training set of the one class
    figure; hold on;
    title('Support Vector Machine Classifier -- GPS Spoofing Data')
    xlabel('Position Error (m)')
    ylabel('Velocity Error (m)')
    scatter(preparedData(:,1),preparedData(:,2),30,'blue');
    scatter(preparedData(inside_indices,1), preparedData(inside_indices,2), 10, 'green');
    scatter(preparedData(out_indices,1), preparedData(out_indices,2), 10, 'red');
    scatter(trainSet(:,1), trainSet(:,2), 10, 'yellow');
    contour(boundary.X,boundary.Y, boundary.vals, [0 0], 'LineWidth', 1.5, 'Color', 'r'); 
    hold off;
end






%%legacy
%w = model.SVs' * model.sv_coef;
%b = -model.rho;
%plot_x = linspace(min(trainSet(:,1)), max(trainSet(:,1)), 30);
%plot_y = (-1/w(2))*(w(1)*plot_x + b);
%plot(plot_x, plot_y, 'k-', 'LineWidth', 1)
%plotboundary(labelData,data,model);