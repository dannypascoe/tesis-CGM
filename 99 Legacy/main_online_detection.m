%% =========================================================================
%  MAIN_ONLINE_DETECTION.M - Detección de Fallos e Imputación en Línea
%  =========================================================================
%  Descripción: Ejecuta la predicción en línea, detecta fallos en la señal
%               CGM e imputa los datos corruptos usando las redes entrenadas.
%
%  Prerrequisitos: 
%    - Ejecutar main_training.m primero para generar el modelo
%    - O cargar un modelo previamente entrenado
%
%  Uso: Modificar la variable 'architecture' y ejecutar el script.
%  =========================================================================

clear; clc; close all;

%% ===================== AGREGAR CARPETAS AL PATH =====================
% Ruta base del proyecto (MODIFICAR SEGÚN TU UBICACIÓN)
scriptPath = 'D:\Documentos\02 MROI\07 Tesis\03 Extension Tesis\CGM_Project\CGM_Project';

% Agregar subcarpetas al path de MATLAB
addpath(fullfile(scriptPath, 'architectures'));
addpath(fullfile(scriptPath, 'data_preparation'));
addpath(fullfile(scriptPath, 'utils'));
addpath(fullfile(scriptPath, 'data'));

fprintf('📁 Carpetas agregadas al path de MATLAB\n');

%% ===================== CONFIGURACIÓN =====================

% Seleccionar arquitectura a evaluar
% Opciones: 'cnn_lstm', 'gru', '1dcnn', 'transformer_lstm'
architecture = 'transformer_lstm';

% Parámetros de detección de fallos
umbral_min = 10;    % Umbral fijo en mg/dL para detección
winit      = 30;    % Longitud de ventana para varianza local
alpha_base = 0.3;   % Peso base para suavizado adaptativo

% Rutas (relativas a la ubicación del script)
data_path   = fullfile(scriptPath, 'data', 'raw');
models_path = fullfile(scriptPath, 'data', 'models');

%% ===================== CARGAR MODELO ENTRENADO =====================

fprintf('📂 Cargando modelo entrenado...\n');

model_file = fullfile(models_path, sprintf('trained_%s_latest.mat', architecture));

if ~exist(model_file, 'file')
    error('No se encontró el modelo: %s\nEjecute main_training.m primero.', model_file);
end

load(model_file, 'nets', 'media_g', 'std_g', 'media_u', 'std_u', ...
    'media_m', 'std_m', 'N_PAST', 'input_shape');

fprintf('   ✅ Modelo cargado: %s\n', architecture);
fprintf('   ✅ Redes entrenadas: %d\n', numel(nets));

%% ===================== CARGAR DATOS DE PRUEBA =====================

fprintf('📂 Cargando datos de prueba...\n');

[~, g_test, g_test_real, u, m, true_f] = load_and_prepare_data(data_path);

T = length(g_test);
fprintf('   ✅ Muestras de prueba: %d\n', T);

%% ===================== PREDICCIÓN EN LÍNEA =====================

fprintf('\n🚀 Iniciando predicción en línea...\n');

nModels = numel(nets);

% Preallocación de resultados para todos los modelos
g_final_all      = zeros(nModels, T);
ypred_all        = zeros(nModels, T);
fallas_all       = false(nModels, T);
error_detect_all = zeros(nModels, T);
error_impute_all = zeros(nModels, T);

for i = 1:nModels
    fprintf('   [%d/%d] Evaluando red %d...\n', i, nModels, i);
    
    net_i = nets{i};
    
    % Inicialización para este modelo
    g_final      = zeros(1, T);
    ypred        = zeros(1, T);
    fallas_vec   = false(1, T);
    error_detect = zeros(1, T);
    error_impute = zeros(1, T);
    
    % Valores iniciales
    g_final(1:N_PAST) = g_test(1:N_PAST);
    ypred(1:N_PAST)   = g_test(1:N_PAST);
    
    % --- Bucle principal de imputación en línea ---
    for k = (N_PAST + 1) : T
        
        % 1) Construir ventana histórica y corregir valores <10
        past_g = g_test(k - N_PAST : k - 1);
        invalid_idx = find(past_g < 10);
        for j = invalid_idx'
            past_g(j) = ypred(k - N_PAST - 1 + j);
        end
        
        % 2) Normalizar entradas
        past_g_norm = (past_g - media_g) / std_g;
        u_norm_k    = (u(k) - media_u) / std_u;
        m_norm_k    = (m(k) - media_m) / std_m;
        
        % 3) Construir secuencia de entrada según arquitectura
        x = [past_g_norm; u_norm_k; m_norm_k];
        switch input_shape
            case 'column'
                xseq = {reshape(x, [], 1)};
            case 'row'
                xseq = {reshape(x, 1, [])};
        end
        
        % 4) Predecir y desnormalizar
        y_hat_norm = predict(net_i, xseq);
        y_hat = y_hat_norm * std_g + media_g;
        
        % 5) Calcular errores
        err_detect = abs(y_hat - g_test(k));      % vs señal corrupta
        err_impute = abs(y_hat - g_test_real(k)); % vs señal real
        error_detect(k) = err_detect;
        error_impute(k) = err_impute;
        
        % 6) Detectar fallo e imputar
        [ypred_k, fallo] = detect_and_impute(y_hat, ypred(k-1), err_detect, ...
            error_detect(1:k-1), umbral_min, winit, alpha_base);
        
        ypred(k)      = ypred_k;
        fallas_vec(k) = fallo;
        
        % 7) Señal final
        if fallo
            g_final(k) = ypred_k;
        else
            g_final(k) = g_test(k);
        end
    end
    
    % Guardar resultados de este modelo
    g_final_all(i, :)      = g_final;
    ypred_all(i, :)        = ypred;
    fallas_all(i, :)       = fallas_vec;
    error_detect_all(i, :) = error_detect;
    error_impute_all(i, :) = error_impute;
end

fprintf('   ✅ Predicción completada\n');

%% ===================== GUARDAR RESULTADOS =====================

fprintf('\n💾 Guardando resultados...\n');

timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
results_file = fullfile(models_path, sprintf('results_%s_%s.mat', architecture, timestamp));

save(results_file, ...
    'g_final_all', 'ypred_all', 'fallas_all', ...
    'error_detect_all', 'error_impute_all', ...
    'g_test', 'g_test_real', 'true_f', ...
    'umbral_min', 'winit', 'alpha_base', ...
    'architecture', 'N_PAST');

% Versión "latest"
results_latest = fullfile(models_path, sprintf('results_%s_latest.mat', architecture));
save(results_latest, ...
    'g_final_all', 'ypred_all', 'fallas_all', ...
    'error_detect_all', 'error_impute_all', ...
    'g_test', 'g_test_real', 'true_f', ...
    'umbral_min', 'winit', 'alpha_base', ...
    'architecture', 'N_PAST');

fprintf('   ✅ Resultados guardados en: %s\n', results_file);

%% ===================== RESUMEN =====================

fprintf('\n═══════════════════════════════════════════\n');
fprintf('✅ DETECCIÓN E IMPUTACIÓN COMPLETADA\n');
fprintf('   Arquitectura: %s\n', upper(architecture));
fprintf('   Modelos evaluados: %d\n', nModels);
fprintf('   Muestras procesadas: %d\n', T);
fprintf('═══════════════════════════════════════════\n');
fprintf('\n📊 Ejecute main_metrics.m para calcular métricas\n');


%% ===================== FUNCIÓN AUXILIAR =====================

function [ypred_k, fallo] = detect_and_impute(y_hat, prev_pred, err, error_hist, umbral_min, winit, alpha_base)
%DETECT_AND_IMPUTE Detecta fallo e imputa valor si es necesario
%
%   [ypred_k, fallo] = detect_and_impute(y_hat, prev_pred, err, error_hist, umbral_min, winit, alpha_base)
%
%   Estrategia:
%   - Si error > umbral_min → fallo detectado
%     - Si error < 60 (ruido blanco): suavizado adaptativo
%     - Si error >= 60 (desconexión): usar predicción directa
%   - Si error <= umbral_min → no hay fallo, combinar predicción con valor previo

    if err > umbral_min
        % Fallo detectado
        
        % Calcular alpha adaptativo según varianza local
        if numel(error_hist) >= winit
            var_local = var(error_hist(end - winit + 1 : end));
            alpha = 1 / (1 + var_local);
        else
            alpha = alpha_base;
        end
        
        if err < 60
            % Ruido blanco: aplicar suavizado adaptativo
            ypred_k = alpha * y_hat + (1 - alpha) * prev_pred;
        else
            % Desconexión: usar predicción directa
            ypred_k = y_hat;
        end
        
        fallo = true;
    else
        % No hay fallo: combinar predicción con valor previo
        ypred_k = 0.8 * y_hat + 0.2 * prev_pred;
        fallo = false;
    end
end
