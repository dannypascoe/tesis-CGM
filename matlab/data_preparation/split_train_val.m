function [XTrain, YTrain, XVal, YVal, numTrain] = split_train_val(Xtrain, Ytrain, valRatio)
%SPLIT_TRAIN_VAL Divide el dataset en entrenamiento y validación
%
%   [XTrain, YTrain, XVal, YVal, numTrain] = split_train_val(Xtrain, Ytrain, valRatio)
%
%   Inputs:
%       Xtrain   - Cell array de entradas completo
%       Ytrain   - Vector de salidas completo
%       valRatio - Proporción de datos para validación (0 a 1)
%
%   Outputs:
%       XTrain   - Cell array de entradas de entrenamiento
%       YTrain   - Vector de salidas de entrenamiento
%       XVal     - Cell array de entradas de validación
%       YVal     - Vector de salidas de validación
%       numTrain - Número de muestras de entrenamiento
%
%   NOTA: La división es cronológica (no aleatoria) para preservar
%         la estructura temporal de las series de tiempo.

    numSamples = numel(Ytrain);
    numTrain   = floor((1 - valRatio) * numSamples);
    
    % División de entrenamiento
    XTrain = Xtrain(1:numTrain);
    YTrain = Ytrain(1:numTrain);
    
    % División de validación
    XVal = Xtrain(numTrain + 1 : end);
    YVal = Ytrain(numTrain + 1 : end);

end
