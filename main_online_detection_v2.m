%% =========================================================================
%  MAIN_ONLINE_DETECTION.M - Detección de Fallos e Imputación en Línea
%  =========================================================================
%  Descripción: Ejecuta la predicción en línea, detecta fallos en la señal
%               CGM e imputa los datos corruptos usando las redes entrenadas.
%
%  Soporta dos modos de detección:
%    - 'fixed'    : Umbral fijo (comportamiento original, umbral_min = 10)
%    - 'adaptive' : Umbral adaptativo B+ROC basado en contexto glucémico
%
%  Prerrequisitos: 
%    - Ejecutar main_training.m primero para generar el modelo
%    - O cargar un modelo previamente entrenado
%
%  Uso: Modificar 'architecture' y 'threshold_mode', luego ejecutar.
%
%  Para comparar modos:
%    1. Ejecutar con threshold_mode = 'fixed'    → genera results_*_fixed_*
%    2. Ejecutar con threshold_mode = 'adaptive' → genera results_*_adaptive_*
%    3. Ejecutar main_metrics.m sobre cada conjunto de resultados
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

% Parámetros de imputación (sin cambios)
winit      = 30;    % Longitud de ventana para varianza local
alpha_base = 0.3;   % Peso base para suavizado adaptativo

% Rutas (relativas a la ubicación del script)
data_path   = fullfile(scriptPath, 'data', 'raw');
models_path = fullfile(scriptPath, 'data', 'models');

% -----------------------------------------------------------------------
% RUTAS DE SALIDA POR BLOQUE
%   threshold_mode = 'fixed'    → Bloque A → data/models/bloque_A/
%   threshold_mode = 'adaptive' → Bloque B → data/models/bloque_B/
% Los modelos siempre se cargan desde models_path (raíz).
% -----------------------------------------------------------------------

%% ============== [NUEVO] CONFIGURACIÓN DE UMBRAL DE DETECCIÓN ==============
%
%  Modo de umbral:
%    'fixed'    - Umbral fijo original (umbral_min = 10 mg/dL)
%    'adaptive' - Umbral adaptativo B+ROC:
%                 τ(k) = max(τ_min, β · ŷ(k)) · (1 + γ · |ROC(k)|)
%
%  Para comparar ambos modos, ejecutar el script dos veces cambiando este valor.
%  Los resultados se guardan con el modo en el nombre del archivo.

threshold_mode = 'adaptive';   % <<< CAMBIAR AQUÍ: 'fixed' o 'adaptive'

% --- Parámetros del umbral fijo (modo 'fixed') ---
umbral_fijo = 10;   % mg/dL (valor original del paper)

% --- Parámetros del umbral adaptativo (modo 'adaptive') ---
%  beta    : proporción del nivel de glucosa (alineado con ISO 15197/MARD)
%  tau_min : piso mínimo del umbral (seguridad en hipoglucemia severa)
%  gamma   : peso de la tasa de cambio (ROC) de la glucosa
%
%  Con beta=0.10: a 100 mg/dL → τ_base=10 (igual al fijo original)
%                 a  70 mg/dL → τ_base=7   (más sensible en hipo)
%                 a 200 mg/dL → τ_base=20  (más permisivo en hiper)
adapt_params.beta    = 0.10;
adapt_params.tau_min = 7.0;
adapt_params.gamma   = 0.05; %0.15

fprintf('⚙️  Modo de umbral: %s\n', upper(threshold_mode));
if strcmp(threshold_mode, 'adaptive')
    fprintf('   β=%.2f | τ_min=%.1f mg/dL | γ=%.2f\n', ...
        adapt_params.beta, adapt_params.tau_min, adapt_params.gamma);
else
    fprintf('   Umbral fijo: %.1f mg/dL\n', umbral_fijo);
end

% --- Determinar y crear carpeta de salida según bloque ---
switch threshold_mode
    case 'fixed'
        output_path = fullfile(models_path, 'bloque_A');
    case 'adaptive'
        output_path = fullfile(models_path, 'bloque_B');
    otherwise
        output_path = models_path;
end
if ~exist(output_path, 'dir')
    mkdir(output_path);
    fprintf('📁 Carpeta creada: %s\n', output_path);
end

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
threshold_all    = zeros(nModels, T);   % [NUEVO] Almacenar umbrales usados
roc_all          = zeros(nModels, T);   % [NUEVO] Almacenar ROC calculado

for i = 1:nModels
    fprintf('   [%d/%d] Evaluando red %d...\n', i, nModels, i);
    
    net_i = nets{i};
    
    % Inicialización para este modelo
    g_final      = zeros(1, T);
    ypred        = zeros(1, T);
    fallas_vec   = false(1, T);
    error_detect = zeros(1, T);
    error_impute = zeros(1, T);
    threshold_vec = zeros(1, T);   % [NUEVO]
    roc_vec       = zeros(1, T);   % [NUEVO]
    
    % Valores iniciales
    g_final(1:N_PAST) = g_test(1:N_PAST);
    ypred(1:N_PAST)   = g_test(1:N_PAST);
    
    % Umbral inicial para las primeras N_PAST muestras
    switch threshold_mode
        case 'fixed'
            threshold_vec(1:N_PAST) = umbral_fijo;
        case 'adaptive'
            % Usar umbral base con la glucosa inicial promedio
            g_ref_init = mean(g_test(1:N_PAST));
            threshold_vec(1:N_PAST) = max(adapt_params.tau_min, ...
                adapt_params.beta * g_ref_init);
    end
    
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
        err_impute = abs(y_hat - g_test_real(k));  % vs señal real
        error_detect(k) = err_detect;
        error_impute(k) = err_impute;
        
        % ========== [NUEVO] 5.5) Calcular umbral de detección ==========
        switch threshold_mode
            case 'fixed'
                tau_k = umbral_fijo;
                roc_k = 0;
                
            case 'adaptive'
                % g_ref = predicción del modelo (NO la lectura del sensor,
                % que podría estar corrupta)
                [tau_k, tau_info] = compute_adaptive_threshold( ...
                    y_hat, past_g, adapt_params);
                roc_k = tau_info.roc;
        end
        
        threshold_vec(k) = tau_k;
        roc_vec(k) = roc_k;
        % ================================================================
        
        % 6) Detectar fallo e imputar
        %    [MODIFICADO] Se pasa tau_k (adaptativo o fijo) en lugar de
        %    umbral_min fijo. La función detect_and_impute no requiere
        %    cambios internos.
        [ypred_k, fallo] = detect_and_impute(y_hat, ypred(k-1), err_detect, ...
            error_detect(1:k-1), tau_k, winit, alpha_base);
        
        ypred(k)      = ypred_k;
        fallas_vec(k) = fallo;
        
        % 7) Señal final
        if fallo
            g_final(k) = ypred_k;
        else
            g_final(k) = g_test(k);
        end
    end
    
    % Guardar resultados 
    % de este modelo (5 modelos entrenados 5x1400)
    g_final_all(i, :)      = g_final;
    ypred_all(i, :)        = ypred;
    fallas_all(i, :)       = fallas_vec;
    error_detect_all(i, :) = error_detect;
    error_impute_all(i, :) = error_impute;
    threshold_all(i, :)    = threshold_vec;   % [NUEVO]
    roc_all(i, :)          = roc_vec;         % [NUEVO]
end

fprintf('   ✅ Predicción completada\n');

%% ===================== GUARDAR RESULTADOS =====================

fprintf('\n💾 Guardando resultados...\n');

timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');

% Guardar en subcarpeta del bloque correspondiente
results_file = fullfile(output_path, ...
    sprintf('results_%s_%s_%s.mat', architecture, threshold_mode, timestamp));

% Guardar también datos del umbral
save(results_file, ...
    'g_final_all', 'ypred_all', 'fallas_all', ...
    'error_detect_all', 'error_impute_all', ...
    'threshold_all', 'roc_all', ...
    'g_test', 'g_test_real', 'true_f', ...
    'threshold_mode', 'umbral_fijo', 'adapt_params', ...
    'winit', 'alpha_base', ...
    'architecture', 'N_PAST');

% Versión "latest" también en la subcarpeta del bloque
results_latest = fullfile(output_path, ...
    sprintf('results_%s_latest.mat', architecture));
save(results_latest, ...
    'g_final_all', 'ypred_all', 'fallas_all', ...
    'error_detect_all', 'error_impute_all', ...
    'threshold_all', 'roc_all', ...
    'g_test', 'g_test_real', 'true_f', ...
    'threshold_mode', 'umbral_fijo', 'adapt_params', ...
    'winit', 'alpha_base', ...
    'architecture', 'N_PAST');

fprintf('   ✅ Resultados guardados en: %s\n', results_file);

%% ===================== RESUMEN =====================

fprintf('\n═══════════════════════════════════════════\n');
fprintf('✅ DETECCIÓN E IMPUTACIÓN COMPLETADA\n');
fprintf('   Arquitectura: %s\n', upper(architecture));
fprintf('   Modo umbral:  %s\n', upper(threshold_mode));
fprintf('   Modelos evaluados: %d\n', nModels);
fprintf('   Muestras procesadas: %d\n', T);
fprintf('═══════════════════════════════════════════\n');

%% ========= [NUEVO] ESTADÍSTICAS DEL UMBRAL ADAPTATIVO =========

if strcmp(threshold_mode, 'adaptive')
    fprintf('\n📊 ESTADÍSTICAS DEL UMBRAL ADAPTATIVO LSTM-TRANSFORMER\n');
    fprintf('───────────────────────────────────────────\n');
    
    % Estadísticas promedio entre todos los modelos
    tau_mean_all = mean(threshold_all(:));
    tau_std_all  = std(threshold_all(:));
    tau_min_all  = min(threshold_all(:));
    tau_max_all  = max(threshold_all(:));
    
    roc_mean_all = mean(roc_all(:));
    roc_max_all  = max(roc_all(:));
    
    fprintf('   Umbral τ:\n');
    fprintf('     Media:  %.2f mg/dL\n', tau_mean_all);
    fprintf('     Std:    %.2f mg/dL\n', tau_std_all);
    fprintf('     Min:    %.2f mg/dL\n', tau_min_all);
    fprintf('     Max:    %.2f mg/dL\n', tau_max_all);
    fprintf('   ROC:\n');
    fprintf('     Media:  %.2f mg/dL/muestra\n', roc_mean_all);
    fprintf('     Max:    %.2f mg/dL/muestra\n', roc_max_all);
    fprintf('───────────────────────────────────────────\n');
    
    % Comparación: ¿cuántas muestras habrían sido detectadas diferente?
    fallas_fijo = false(nModels, T);
    for i = 1:nModels
        fallas_fijo(i, :) = error_detect_all(i, :) > umbral_fijo;
    end
    
    % Muestras donde el umbral adaptativo NO detectó fallo pero el fijo SÍ
    solo_fijo = fallas_fijo & ~fallas_all;
    % Muestras donde el umbral adaptativo SÍ detectó fallo pero el fijo NO
    solo_adapt = fallas_all & ~fallas_fijo;
    % Muestras donde ambos coinciden
    ambos_detectan = fallas_all & fallas_fijo;
    
    fprintf('\n📊 COMPARACIÓN vs UMBRAL FIJO (τ=%d mg/dL)\n', umbral_fijo);
    fprintf('───────────────────────────────────────────\n');
    fprintf('   Fallos detectados (adaptativo): %d (%.1f%%)\n', ...
        sum(fallas_all(:)), 100*sum(fallas_all(:))/(nModels*T));
    fprintf('   Fallos detectados (fijo):       %d (%.1f%%)\n', ...
        sum(fallas_fijo(:)), 100*sum(fallas_fijo(:))/(nModels*T));
    fprintf('   Coinciden ambos:                %d\n', sum(ambos_detectan(:)));
    fprintf('   Solo adaptativo detecta:        %d\n', sum(solo_adapt(:)));
    fprintf('   Solo fijo detecta:              %d\n', sum(solo_fijo(:)));
    fprintf('───────────────────────────────────────────\n');
end

fprintf('\n📊 Ejecute main_metrics.m para calcular métricas completas\n');

%% ========= [NUEVO] GRÁFICA: EVOLUCIÓN DEL UMBRAL ADAPTATIVO =========

if strcmp(threshold_mode, 'adaptive')
    figure('Name', sprintf('%s - Umbral Adaptativo', upper(architecture)), ...
           'Position', [100, 100, 1200, 700], 'Color', 'w');
    
    % Usar el primer modelo como ejemplo representativo
    modelo_plot = 1;
    
    % --- Subplot 1: Señal de glucosa + umbral ---
    subplot(3, 1, 1);
    hold on;
    plot(g_test, 'r', 'LineWidth', 1.0, 'DisplayName', 'Señal corrupta');
    plot(g_test_real, 'b', 'LineWidth', 1.2, 'DisplayName', 'Señal real');
    plot(g_final_all(modelo_plot, :), 'k--', 'LineWidth', 1.0, ...
        'DisplayName', sprintf('Imputada (modelo %d)', modelo_plot));
    legend('Location', 'best');
    ylabel('Glucosa (mg/dL)');
    title(sprintf('%s - Señal de Glucosa (modo: %s)', ...
        upper(architecture), upper(threshold_mode)));
    grid on;
    hold off;
    
    % --- Subplot 2: Umbral adaptativo vs fijo ---
    subplot(3, 1, 2);
    hold on;
    plot(threshold_all(modelo_plot, :), 'Color', [0.2, 0.6, 0.2], ...
        'LineWidth', 1.5, 'DisplayName', 'Umbral adaptativo τ(k)');
    yline(umbral_fijo, 'r--', 'LineWidth', 1.5, ...
        'DisplayName', sprintf('Umbral fijo (τ=%d)', umbral_fijo));
    
    % Sombrear zonas donde se detectó fallo
    fallas_plot = fallas_all(modelo_plot, :);
    if any(fallas_plot)
        y_lim = ylim;
        for idx = find(fallas_plot)
            patch([idx-0.5, idx+0.5, idx+0.5, idx-0.5], ...
                  [y_lim(1), y_lim(1), y_lim(2), y_lim(2)], ...
                  [1, 0.8, 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.3, ...
                  'HandleVisibility', 'off');
        end
    end
    
    legend('Location', 'best');
    ylabel('Umbral (mg/dL)');
    title('Umbral de Detección: Adaptativo vs Fijo');
    grid on;
    hold off;
    
    % --- Subplot 3: ROC y error de detección ---
    subplot(3, 1, 3);
    hold on;
    plot(error_detect_all(modelo_plot, :), 'Color', [0.8, 0.4, 0], ...
        'LineWidth', 0.8, 'DisplayName', 'Error detección |ŷ-g|');
    plot(roc_all(modelo_plot, :), 'Color', [0, 0.4, 0.8], ...
        'LineWidth', 1.2, 'DisplayName', 'ROC (tasa de cambio)');
    legend('Location', 'best');
    ylabel('mg/dL');
    xlabel('Muestras');
    title('Error de Detección y Tasa de Cambio (ROC)');
    grid on;
    hold off;
    
    sgtitle(sprintf('%s - Análisis de Umbral Adaptativo', ...
        upper(architecture)), 'FontWeight', 'bold', 'FontSize', 14);
end


%% ===================== FUNCIÓN AUXILIAR =====================

function [ypred_k, fallo] = detect_and_impute(y_hat, prev_pred, err, error_hist, umbral, winit, alpha_base)
%DETECT_AND_IMPUTE Detecta fallo e imputa valor si es necesario
%
%   [ypred_k, fallo] = detect_and_impute(y_hat, prev_pred, err, error_hist, umbral, winit, alpha_base)
%
%   El parámetro 'umbral' puede ser:
%     - Un valor fijo (ej. 10 mg/dL) → comportamiento original
%     - Un valor adaptativo calculado por compute_adaptive_threshold()
%
%   Estrategia:
%   - Si error > umbral → fallo detectado
%     - Si error < 60 (ruido blanco): suavizado adaptativo
%     - Si error >= 60 (desconexión): usar predicción directa
%   - Si error <= umbral → no hay fallo, combinar predicción con valor previo

    if err > umbral
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
