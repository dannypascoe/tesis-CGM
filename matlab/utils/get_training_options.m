    function options = get_training_options(XVal, YVal, numTrain)
%GET_TRAINING_OPTIONS Retorna las opciones de entrenamiento para la red
%
%   options = get_training_options(XVal, YVal, numTrain)
%
%   Inputs:
%       XVal     - Cell array de entradas de validación
%       YVal     - Vector de salidas de validación
%       numTrain - Número de muestras de entrenamiento
%
%   Output:
%       options  - Objeto trainingOptions configurado
%
%   Configuración:
%       - Optimizador: Adam
%       - Épocas máximas: 300
%       - Mini-batch: 32
%       - Early stopping: 30 épocas sin mejora
%       - Regularización L2: 0.001
%       - Gradient clipping: L2 norm = 1

    miniBatchSize = 32;
    
    options = trainingOptions('adam', ...
        'MaxEpochs', 300, ...
        'MiniBatchSize', miniBatchSize, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', 0, ...
        'ExecutionEnvironment', 'auto', ...  % 'gpu' si está disponible, sino 'cpu'
        'L2Regularization', 0.001, ...
        'Plots', 'training-progress', ...
        'ValidationData', {XVal, YVal}, ...
        'ValidationFrequency', floor(numTrain / miniBatchSize), ...
        'ValidationPatience', 30, ...
        'GradientThreshold', 1, ...
        'GradientThresholdMethod', 'l2norm');

end
