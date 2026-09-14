function [Xtrain, Ytrain] = build_dataset(g_norm, u_norm, m_norm, N_PAST, input_shape)
%BUILD_DATASET Construye el conjunto de entrenamiento con ventana deslizante
%
%   [Xtrain, Ytrain] = build_dataset(g_norm, u_norm, m_norm, N_PAST, input_shape)
%
%   Inputs:
%       g_norm      - Glucosa normalizada [vector]
%       u_norm      - Insulina normalizada [vector]
%       m_norm      - Comida normalizada [vector]
%       N_PAST      - Número de pasos históricos de glucosa
%       input_shape - 'column' para CNN-LSTM/GRU, 'row' para 1D-CNN/Transformer
%
%   Outputs:
%       Xtrain - Cell array de entradas {N-N_PAST × 1}
%       Ytrain - Vector de salidas [(N-N_PAST) × 1]
%
%   Cada entrada contiene:
%       [g(k-N_PAST), g(k-N_PAST+1), ..., g(k-1), u(k), m(k)]
%
%   La salida correspondiente es:
%       g(k)
%
%   El formato de cada entrada depende de input_shape:
%       'column' → vector columna (inputSize × 1)
%       'row'    → vector fila (1 × inputSize)

    N = length(g_norm);
    numSamples = N - N_PAST;
    
    Xtrain = cell(numSamples, 1);
    Ytrain = zeros(numSamples, 1);
    
    for k = (N_PAST + 1) : N
        % Extraer valores históricos de glucosa
        past_g = g_norm(k - N_PAST : k - 1);
        
        % Construir vector de entrada
        x = [past_g; u_norm(k); m_norm(k)];
        
        % Aplicar reshape según arquitectura
        switch input_shape
            case 'column'
                % Para CNN-LSTM y GRU: vector columna (8×1)
                Xtrain{k - N_PAST} = reshape(x, [], 1);
                
            case 'row'
                % Para 1D-CNN y Transformer: vector fila (1×8)
                Xtrain{k - N_PAST} = reshape(x, 1, []);
                
            otherwise
                error('input_shape debe ser "column" o "row"');
        end
        
        % Salida: glucosa en el instante k (también normalizada)
        Ytrain(k - N_PAST) = g_norm(k);
    end

end
