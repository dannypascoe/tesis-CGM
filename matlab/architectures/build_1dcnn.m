function [layers, lgraph] = build_1dcnn(inputSize, numResponses)
%BUILD_1DCNN Construye la arquitectura de red Convolucional 1D
%
%   [layers, lgraph] = build_1dcnn(inputSize, numResponses)
%
%   Inputs:
%       inputSize    - Número de entradas (N_PAST + 2)
%       numResponses - Número de salidas (típicamente 1)
%
%   Outputs:
%       layers - Array de capas de la red
%       lgraph - [] (no se usa para esta arquitectura)
%
%   La arquitectura utiliza:
%   - Bloques convolucionales 1D para extracción de características
%   - MaxPooling para reducción de dimensionalidad
%   - GlobalAveragePooling para compresión final
%   - Capas densas para regresión
%
%   NOTA: Esta arquitectura requiere entrada en formato fila (1×8)

    % Configuración de las capas
    layers = [
        % Capa de entrada secuencial (1 canal, longitud mínima 8)
        sequenceInputLayer(1, 'MinLength', 8, 'Name', 'input')

        % Primer bloque convolucional
        convolution1dLayer(5, 32, 'Padding', 'same', 'Name', 'conv1')
        reluLayer('Name', 'relu1')
        dropoutLayer(0.2, 'Name', 'dropout1')
        maxPooling1dLayer(2, 'Stride', 2, 'Name', 'maxpool1')

        % Segundo bloque convolucional
        convolution1dLayer(3, 32, 'Padding', 'same', 'Name', 'conv2')
        reluLayer('Name', 'relu2')

        % Bloque de compresión
        globalAveragePooling1dLayer('Name', 'gap')

        % Bloque denso de salida
        fullyConnectedLayer(64, 'Name', 'fc1')
        reluLayer('Name', 'relu_fc')
        fullyConnectedLayer(numResponses, 'Name', 'fc_out')
        regressionLayer('Name', 'regression')
    ];

    % Esta arquitectura no requiere layerGraph
    lgraph = [];

end
