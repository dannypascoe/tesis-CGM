function [g_norm, u_norm, m_norm, norm_params] = normalize_data(g, u, m)
%NORMALIZE_DATA Normaliza las señales usando z-score (media=0, std=1)
%
%   [g_norm, u_norm, m_norm, norm_params] = normalize_data(g, u, m)
%
%   Inputs:
%       g - Señal de glucosa [vector]
%       u - Señal de insulina [vector]
%       m - Señal de comida/meals [vector]
%
%   Outputs:
%       g_norm      - Glucosa normalizada
%       u_norm      - Insulina normalizada
%       m_norm      - Comida normalizada
%       norm_params - Estructura con parámetros de normalización:
%                     .media_g, .std_g
%                     .media_u, .std_u
%                     .media_m, .std_m
%
%   La normalización z-score se calcula como:
%       x_norm = (x - media) / std
%
%   IMPORTANTE: Los parámetros de normalización se calculan SOLO con datos
%   de entrenamiento y se guardan para aplicar la misma transformación
%   a datos de prueba.

    % Calcular estadísticas de glucosa
    media_g = mean(g);
    std_g   = std(g);
    
    % Calcular estadísticas de insulina
    media_u = mean(u);
    std_u   = std(u);
    
    % Calcular estadísticas de comida
    media_m = mean(m);
    std_m   = std(m);
    
    % Normalizar señales
    g_norm = (g - media_g) / std_g;
    u_norm = (u - media_u) / std_u;
    m_norm = (m - media_m) / std_m;
    
    % Empaquetar parámetros de normalización
    norm_params.media_g = media_g;
    norm_params.std_g   = std_g;
    norm_params.media_u = media_u;
    norm_params.std_u   = std_u;
    norm_params.media_m = media_m;
    norm_params.std_m   = std_m;

end
