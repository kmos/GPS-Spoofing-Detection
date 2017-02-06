%% define MaxSimTime
MaxSimTime = 300;

%% load data
folder = ['scenario-',num2str(MaxSimTime)];
pathdata = [folder, '/dataSimulation'];
load(pathdata);

%% Time when the safe zone is ended
init.endTraining    = dataSimulation.time.endSafeZone;
%% Time when the attack starts
init.startAttack    = dataSimulation.time.attackTime;
init.endTime        = dataSimulation.time.endSimulation;
%% Evaluation window
init.step_size      = 1000; %[step] buffer evaluation

%% type of evaluation: E,horiz,z
init.typeEval       = 'E';

switch init.typeEval
    case 'E'
        disp('Evaluation of Euclidean Distance x y z');
        d = dataSimulation.E;
    case 'horiz'
        disp('Evaluation of horizontal plane');
        d = dataSimulation.horiz;
    case 'z'
        disp('Evaluation of Altitude');
        d = dataSimulation.z;
    otherwise
        disp('Error');
end

%% set normalize to 1 if you want to normalize the data
init.normalize      = 1;
%% Type of solver: 2 for Schölkopf one-class, 5 for Tax and Duin
init.solver         = 2;
%% type of kernel: 2 RBF gaussian
init.kernel         = 2;
%% set crossValidate to 1 if you want to calculate best c and gamma value
init.crossValidate  = 0;
%% set plotcv to 1 if you want to plot cross validation results
init.plotcv         = 0;

%% Normalize Data
if init.normalize == 1
    data = mat2gray(d(1:end,1:2));
else
    data = d;
end

%% label and test set 
% Labels are necessary for accurancy estimation.
testSet                 = data(init.endTraining+1:end,1:end);
labelTest               = d(init.endTraining+1:end,3);

%% label and training set 
trainSet        = data(1:init.endTraining,1:2);
labelTrain      = d(1:init.endTraining,3);


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
        model = svmtrain(labelTrain, trainSet(1:end,1:2), '-s 5 -t 2  -h 0 -c 0.0033 -g 100 ');
    else
        model = svmtrain(labelTrain, trainSet(1:end,1:2), ...
            sprintf('-s %d -t %d -c %f -g %f',init.solver,init.kernel, parameters.C, parameters.gamma));
    end
end

%% Predict the Anomaly
%[predicted_labels, accurancy] = svmpredict(labelTest, testSet(1:end,1:2), model);

clearvars results;

w = model.SVs' * model.sv_coef;
b = -model.rho;
absw = sqrt(sum(w.^2));

results.predicted_labels = zeros(size(testSet,1),init.step_size);
results.accuracy         = zeros(size(testSet,1),3);
results.dec_values       = zeros(size(testSet,1),init.step_size);
results.meandistance     = zeros(size(testSet,1),1);

j=1;


for i = init.step_size: init.step_size : size(testSet, 1)
    % Extract new points from data 
    new_points = testSet(i-init.step_size + 1 :i,1:2);
    new_labels = labelTest(i-init.step_size + 1:i);
    
    [predict_label, accuracy, dec_values] = svmpredict(new_labels, new_points, model);
    
    results.predicted_labels(j,:) = predict_label;
    results.accuracy(j,:)         = accuracy;
    results.dec_values(j,:)       = dec_values;
    results.meandistance(j,:)     = mean(dec_values / (absw-b));
    
    j=j+1;
end
%% Predicted indices
%inside_indices  = find(predicted_labels > 0);
%out_indices     = find(predicted_labels == -1);

%% Calculate boundary
boundary = Boundary(labelData, preparedData, model);

% %% Plot the Results
% % PLOT DATA:
%     % blue      = all the data
%     % green     = test data in the one class
%     % red       = test data out of the one class
%     % yellow    = training set of the one class
     figure; 
     hold on;
     title('Support Vector Machine Classifier -- GPS Spoofing Data')
     xlabel('Position Error (m)')
     ylabel('Velocity Error (m)')
     scatter(data(:,1),data(:,2),30,'blue');
%     scatter(testSet(inside_indices,1), testSet(inside_indices,2), 10, 'green');
%     scatter(testSet(out_indices,1), testSet(out_indices,2), 10, 'red');
     scatter(trainSet(:,1), trainSet(:,2), 10, 'yellow');
     contour(boundary.X,boundary.Y, boundary.vals, [0 0], 'LineWidth', 1.5, 'Color', 'r');
     hold off;

clearvars labelTestGood labelTestBad init d pathdata