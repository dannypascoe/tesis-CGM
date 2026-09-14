function [layers, lgraph] = build_cnn_lstm(inputSize, numResponses)
%BUILD_CNN_LSTM Construye la arquitectura híbrida CNN-LSTM
%
%   [layers, lgraph] = build_cnn_lstm(inputSize, numResponses)
%
%   Inputs:
%       inputSize    - Número de entradas (N_PAST + 2)
%       numResponses - Número de salidas (típicamente 1)
%
%   Outputs:
%       layers - Array de capas de la red
%       lgraph - [] (no se usa para esta arquitectura)
%
%   La arquitectura combina:
%   - Conv1D para extraer características locales
%   - LSTM para capturar dependencias temporales
%   - Capas densas para regresión

    % Hiperparámetros de la arquitectura
    F  = 16;    % Número de filtros de la CNN
    K  = 3;     % Tamaño del kernel (ventana) de la CNN
    H  = 32;    % Número de neuronas en la capa LSTM
    FC = 64;    % Tamaño de la capa densa intermedia
    D  = 0.3;   % Dropout rate (30%)

    % Configuración de las capas
    layers = [
        % Capa de entrada secuencial
        sequenceInputLayer(inputSize, 'Name', 'input')
        
        % Bloque Convolucional
        convolution1dLayer(K, F, 'Padding', 'causal', 'Name', 'conv1d')
        reluLayer('Name', 'relu_conv')
        
        % Bloque LSTM
        lstmLayer(H, 'OutputMode', 'last', 'Name', 'lstm')
        dropoutLayer(D, 'Name', 'dropout')
        
        % Bloque de Regresión
        fullyConnectedLayer(FC, 'Name', 'fc1')
        reluLayer('Name', 'relu_fc')
        fullyConnectedLayer(numResponses, 'Name', 'fc_out')
        regressionLayer('Name', 'regression')
    ];

    % Esta arquitectura no requiere layerGraph
    lgraph = [];

end
