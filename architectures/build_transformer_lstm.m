function [layers, lgraph] = build_transformer_lstm(inputSize, numResponses)
%BUILD_TRANSFORMER_LSTM Construye la arquitectura híbrida Transformer-LSTM
%
%   [layers, lgraph] = build_transformer_lstm(inputSize, numResponses)
%
%   Inputs:
%       inputSize    - Número de entradas (N_PAST + 2)
%       numResponses - Número de salidas (típicamente 1)
%
%   Outputs:
%       layers - Array de capas de la red (para referencia)
%       lgraph - LayerGraph con conexiones residuales
%
%   La arquitectura combina:
%   - Embedding de posición aprendible
%   - Dos bloques de Self-Attention con máscaras causales
%   - Bloques Feed-Forward position-wise
%   - Conexiones residuales y normalización
%   - LSTM para refinamiento secuencial
%
%   NOTA: Esta arquitectura requiere entrada en formato fila (1×8)
%         y usa layerGraph en lugar de array de capas simple.

    % Hiperparámetros
    numChannels    = 1;               % Canal de entrada
    maxPosition    = 8;               % Longitud de secuencia (N_PAST + 2)
    embeddingDim   = 32;              % Dimensión del embedding
    numHeads       = 4;               % Cabezas de atención
    numKeyChannels = numHeads * 32;   % Canales de clave (128 total)
    ffDim          = 128;             % Dimensión del feed-forward
    lstmUnits      = 32;              % Unidades LSTM
    dropoutRate    = 0.3;             % Tasa de dropout

    % Configuración de las capas
    layers = [
        % === Capa de Entrada ===
        sequenceInputLayer(numChannels, ...
            'MinLength', maxPosition, ...
            'Name', 'input')

        % === Bloque de Proyección y Embedding Posicional ===
        fullyConnectedLayer(embeddingDim, 'Name', 'proj')
        positionEmbeddingLayer(embeddingDim, maxPosition, 'Name', 'posEmb')
        additionLayer(2, 'Name', 'addEmb')

        % === Primer Bloque de Atención ===
        selfAttentionLayer(numHeads, numKeyChannels, ...
            'AttentionMask', 'causal', ...
            'Name', 'attn1')
        additionLayer(2, 'Name', 'res1')
        layerNormalizationLayer('Name', 'norm1')

        % === Bloque Feed-Forward Position-Wise ===
        fullyConnectedLayer(ffDim, 'Name', 'ff1')
        reluLayer('Name', 'relu_ff')
        fullyConnectedLayer(embeddingDim, 'Name', 'ff2')
        additionLayer(2, 'Name', 'resFF')
        layerNormalizationLayer('Name', 'normFF')

        % === Segundo Bloque de Atención ===
        selfAttentionLayer(numHeads, numKeyChannels, ...
            'AttentionMask', 'causal', ...
            'Name', 'attn2')
        additionLayer(2, 'Name', 'res2')
        layerNormalizationLayer('Name', 'norm2')

        % === Bloque LSTM ===
        indexing1dLayer('last', 'Name', 'idx')
        lstmLayer(lstmUnits, 'OutputMode', 'last', 'Name', 'lstm1')
        dropoutLayer(dropoutRate, 'Name', 'drop1')
        lstmLayer(lstmUnits, 'OutputMode', 'last', 'Name', 'lstm2')
        dropoutLayer(dropoutRate, 'Name', 'drop2')

        % === Capa de Salida ===
        fullyConnectedLayer(numResponses, 'Name', 'fc_out')
        regressionLayer('Name', 'regression')
    ];

    % === Construcción del LayerGraph con Conexiones Residuales ===
    lgraph = layerGraph(layers);

    % 1) Proyección → addEmb/in2 (rama del embedding)
    lgraph = connectLayers(lgraph, 'proj', 'addEmb/in2');

    % 2) Residual primera atención: addEmb → res1/in2
    lgraph = connectLayers(lgraph, 'addEmb', 'res1/in2');

    % 3) Residual feed-forward: norm1 → resFF/in2
    lgraph = connectLayers(lgraph, 'norm1', 'resFF/in2');

    % 4) Residual segunda atención: normFF → res2/in2
    lgraph = connectLayers(lgraph, 'normFF', 'res2/in2');

end
