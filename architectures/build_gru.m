function [layers, lgraph] = build_gru(inputSize, numResponses)
%BUILD_GRU Construye la arquitectura de red GRU
%
%   [layers, lgraph] = build_gru(inputSize, numResponses)
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
%   - GRU para modelado temporal eficiente
%   - Capas densas para regresión
%
%   Ventajas de GRU sobre LSTM:
%   - Menos parámetros, entrenamiento más rápido
%   - Mejor convergencia en datasets pequeños

    % Hiperparámetros de la arquitectura
    H  = 32;    % Número de unidades ocultas en la capa GRU
    FC = 64;    % Tamaño de la capa densa intermedia
    D  = 0.3;   % Dropout rate (30%)

    % Configuración de las capas
    layers = [
        % Capa de entrada secuencial
        sequenceInputLayer(inputSize, 'Name', 'input')
        
        % Bloque GRU
        gruLayer(H, 'OutputMode', 'last', 'Name', 'gru')
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
