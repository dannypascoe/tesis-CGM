function input_shape = get_input_shape(architecture)
%GET_INPUT_SHAPE Retorna el tipo de reshape según la arquitectura
%
%   input_shape = get_input_shape(architecture)
%
%   Input:
%       architecture - Nombre de la arquitectura:
%                      'cnn_lstm', 'gru', '1dcnn', 'transformer_lstm'
%
%   Output:
%       input_shape - 'column' o 'row'
%
%   Mapeo:
%       CNN-LSTM, GRU           → 'column' (vector columna 8×1)
%       1D-CNN, Transformer-LSTM → 'row'    (vector fila 1×8)
%
%   Esto es necesario porque:
%   - Las arquitecturas recurrentes (LSTM, GRU) esperan sequenceInputLayer
%     con inputSize canales, donde cada entrada es un vector columna.
%   - Las arquitecturas CNN 1D esperan 1 canal con secuencia de largo 8,
%     por lo que la entrada debe ser un vector fila.

    switch lower(architecture)
        case {'cnn_lstm', 'gru'}
            input_shape = 'column';
            
        case {'1dcnn', 'transformer_lstm'}
            input_shape = 'row';
            
        otherwise
            error('Arquitectura no reconocida: %s\nOpciones válidas: cnn_lstm, gru, 1dcnn, transformer_lstm', ...
                architecture);
    end

end
