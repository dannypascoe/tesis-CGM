%% =========================================================================
%  MAIN_PIPELINE_INTEGRADO.M - Pipeline Detector + Imputador Independientes
%  =========================================================================
%  Descripción: Evalúa un pipeline donde el modelo de DETECCIÓN de fallos
%               y el modelo de IMPUTACIÓN son arquitecturas independientes.
%
%  Concepto clave:
%    - El DETECTOR genera ŷ_detect y compara contra g(k) para decidir
%      si hay fallo (usando umbral adaptativo B+ROC).
%    - Si hay fallo, el IMPUTADOR genera ŷ_impute como valor de reemplazo.
%    - Si no hay fallo, se conserva la lectura original del sensor.
%
%  Combinaciones evaluadas (exhaustivas, 4 arquitecturas principales):
%    Detector:  CNN-LSTM, GRU, 1D-CNN, Transformer-LSTM
%    Imputador: CNN-LSTM, GRU, 1D-CNN, Transformer-LSTM
%    Total: 16 combinaciones (4 × 4)
%    Nota: Incluye 4 combinaciones "diagonales" donde detector=imputador,
%          equivalentes al pipeline unificado pero con umbral adaptativo.
%
%  Umbral: Solo adaptativo B+ROC (ya validado).
%
%  -----------------------------------------------------------------------
%  DEFINICIONES DE ERROR DE IMPUTACIÓN (importante para Secciones 4.4 y 4.5)
%  -----------------------------------------------------------------------
%    D1 (PRINCIPAL) error_impute_pred_all = |y_hat_imp - g_real| en TODAS las
%        muestras. Es el error de PREDICCIÓN del imputador. Coincide con la
%        definición del paper publicado y con el Bloque B (main_online_detection).
%        --> ÚSESE ESTE para MAE/RMSE en la tesis (comparable con el paper).
%    D2 (referencia) error_impute_out_all = |g_final - g_real| en todas las
%        muestras. Error de la SALIDA entregada; las muestras limpias aportan ~0,
%        por lo que NO es comparable con el paper. Se conserva solo como respaldo.
%    D3 (secundaria, se calcula en main_metrics_pipeline_v3) = |g_final - g_real|
%        SOLO en muestras con fallo real (true_f==1): calidad de reconstrucción
%        donde la imputación realmente actuó.
%
%  NOTA: la lógica operativa (g_final, detección, F1) NO cambió respecto a la
%  versión anterior; el imputador ahora solo predice también en muestras sin
%  fallo para poder reportar D1, pero esa predicción no altera la salida.
%  -----------------------------------------------------------------------
%
%  Prerrequisitos: 
%    - Modelos entrenados para las 4 arquitecturas principales
%      (trained_cnn_lstm_latest.mat, trained_gru_latest.mat, 
%       trained_1dcnn_latest.mat, trained_transformer_lstm_latest.mat)
%
%  Uso: Configurar rutas y ejecutar. Los resultados se guardan en
%       data/models/results_pipeline_<detector>_<imputador>_<timestamp>.mat
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

fprintf('🔍 Carpetas agregadas al path de MATLAB\n');

%% ===================== CONFIGURACIÓN =====================

% --- Combinaciones a evaluar (4 × 4 = 16 combinaciones exhaustivas) ---
% Cada fila: {detector, imputador}
% Las 4 arquitecturas principales del paper publicado
architectures = {'cnn_lstm', 'gru', '1dcnn', 'transformer_lstm'};

% Generar todas las combinaciones (incluye diagonales: mismo modelo para ambas tareas)
combinaciones = cell(numel(architectures)^2, 2);
idx = 0;
for d = 1:numel(architectures)
    for im = 1:numel(architectures)
        idx = idx + 1;
        combinaciones{idx, 1} = architectures{d};
        combinaciones{idx, 2} = architectures{im};
    end
end

nCombos = size(combinaciones, 1);

% --- Parámetros de imputación (sin cambios respecto al original) ---
winit      = 30;    % Longitud de ventana para varianza local
alpha_base = 0.3;   % Peso base para suavizado adaptativo

% --- Parámetros del umbral adaptativo B+ROC ---
adapt_params.beta    = 0.10;
adapt_params.tau_min = 7.0;
adapt_params.gamma   = 0.05;   % Valor optimizado en experimentos previos

% --- Rutas ---
data_path   = fullfile(scriptPath, 'data', 'raw');
models_path = fullfile(scriptPath, 'data', 'models');

% -----------------------------------------------------------------------
% CARPETA DE SALIDA: data/models/bloque_C/
% Todos los resultados del pipeline integrado (16 combinaciones) se guardan
% aquí. Los modelos se siguen cargando desde models_path (raíz).
% -----------------------------------------------------------------------
output_path = fullfile(models_path, 'bloque_C');
if ~exist(output_path, 'dir')
    mkdir(output_path);
    fprintf('📁 Carpeta creada: %s\n', output_path);
end

fprintf('⚙️  Umbral adaptativo B+ROC\n');
fprintf('   β=%.2f | τ_min=%.1f mg/dL | γ=%.2f\n', ...
    adapt_params.beta, adapt_params.tau_min, adapt_params.gamma);

%% ===================== CARGAR DATOS DE PRUEBA =====================

fprintf('\n📂 Cargando datos de prueba...\n');
[~, g_test, g_test_real, u, m, true_f] = load_and_prepare_data(data_path);
T = length(g_test);
fprintf('   ✅ Muestras de prueba: %d\n', T);

%% ===================== PRE-CARGAR TODOS LOS MODELOS =====================
% Cargar una sola vez cada arquitectura necesaria

fprintf('\n📂 Pre-cargando modelos...\n');

% Identificar arquitecturas únicas necesarias
archs_needed = unique([combinaciones(:,1); combinaciones(:,2)]);
model_cache = struct();

for a = 1:numel(archs_needed)
    arch = archs_needed{a};
    model_file = fullfile(models_path, sprintf('trained_%s_latest.mat', arch));
    
    if ~exist(model_file, 'file')
        error('No se encontró el modelo: %s\nEjecute main_training.m para %s primero.', ...
            model_file, arch);
    end
    
    data = load(model_file, 'nets', 'media_g', 'std_g', 'media_u', 'std_u', ...
        'media_m', 'std_m', 'N_PAST', 'input_shape');
    
    % Guardar en caché usando nombre seguro para struct
    safe_name = matlab.lang.makeValidName(arch);
    model_cache.(safe_name) = data;
    
    fprintf('   ✅ %s cargado: %d redes, N_PAST=%d, shape=%s\n', ...
        upper(arch), numel(data.nets), data.N_PAST, data.input_shape);
end

%% ===================== ESTRUCTURA PARA RESULTADOS GLOBALES =====================

% Almacenar resultados de TODAS las combinaciones
all_results = struct();
timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');

%% ===================== BUCLE PRINCIPAL POR COMBINACIÓN =====================

for combo = 1:nCombos
    arch_detect = combinaciones{combo, 1};
    arch_impute = combinaciones{combo, 2};
    
    fprintf('\n╔═══════════════════════════════════════════════════════════╗\n');
    fprintf('║ COMBINACIÓN %d/%d: Detector=%s | Imputador=%s\n', ...
        combo, nCombos, upper(arch_detect), upper(arch_impute));
    fprintf('╚═══════════════════════════════════════════════════════════╝\n');
    
    % --- Extraer modelos del caché ---
    det = model_cache.(matlab.lang.makeValidName(arch_detect));
    imp = model_cache.(matlab.lang.makeValidName(arch_impute));
    
    nNets_det = numel(det.nets);
    nNets_imp = numel(imp.nets);
    
    % El número de runs del pipeline es el mínimo entre detector e imputador
    % (típicamente ambos tienen 5 runs)
    nRuns = min(nNets_det, nNets_imp);
    
    fprintf('   Detector: %d redes (N_PAST=%d) | Imputador: %d redes (N_PAST=%d)\n', ...
        nNets_det, det.N_PAST, nNets_imp, imp.N_PAST);
    fprintf('   Runs a evaluar: %d\n', nRuns);
    
    % --- Determinar índice de inicio ---
    % Necesitamos suficientes muestras históricas para AMBOS modelos
    N_PAST_max = max(det.N_PAST, imp.N_PAST);
    k_start = N_PAST_max + 1;
    
    % --- Preallocación ---
    g_final_all           = zeros(nRuns, T);
    ypred_det_all         = zeros(nRuns, T);   % Predicciones del detector
    ypred_imp_all         = zeros(nRuns, T);   % Predicciones del imputador (semántica original)
    ypred_imp_raw_all     = zeros(nRuns, T);   % [NUEVO] Predicción CRUDA del imputador en CADA k (para D1)
    fallas_all            = false(nRuns, T);
    error_detect_all      = zeros(nRuns, T);
    % --- Dos definiciones de error de imputación (ver nota al inicio del script) ---
    %  D1 = |y_hat_imp - g_real|  en todas las muestras  → error de PREDICCIÓN del
    %       imputador. Coincide con la definición del paper y del Bloque B.
    %       Es la métrica COMPARABLE para las Secciones 4.4 y 4.5.
    %  D2 = |g_final - g_real|    en todas las muestras  → error de la SALIDA
    %       entregada (las muestras limpias contribuyen ~0). Se conserva como
    %       referencia interna; NO es comparable con el paper.
    error_impute_pred_all = zeros(nRuns, T);   % [D1]
    error_impute_out_all  = zeros(nRuns, T);   % [D2]
    threshold_all         = zeros(nRuns, T);
    roc_all               = zeros(nRuns, T);
    
    % --- Bucle por run (pareja detector_i, imputador_i) ---
    for i = 1:nRuns
        fprintf('   [%d/%d] Evaluando pareja de redes %d...\n', i, nRuns, i);
        
        net_det = det.nets{i};
        net_imp = imp.nets{i};
        
        % Inicialización
        g_final       = zeros(1, T);
        ypred_det     = zeros(1, T);
        ypred_imp     = zeros(1, T);
        ypred_imp_raw = zeros(1, T);   % [NUEVO] predicción cruda del imputador (D1)
        fallas_vec    = false(1, T);
        error_detect  = zeros(1, T);
        error_impute_pred = zeros(1, T);   % [D1] |y_hat_imp - g_real|
        error_impute_out  = zeros(1, T);   % [D2] |g_final  - g_real|
        threshold_vec = zeros(1, T);
        roc_vec       = zeros(1, T);
        
        % Valores iniciales
        g_final(1:N_PAST_max)       = g_test(1:N_PAST_max);
        ypred_det(1:N_PAST_max)     = g_test(1:N_PAST_max);
        ypred_imp(1:N_PAST_max)     = g_test(1:N_PAST_max);
        ypred_imp_raw(1:N_PAST_max) = g_test(1:N_PAST_max);
        
        % Umbral inicial
        g_ref_init = mean(g_test(1:N_PAST_max));
        threshold_vec(1:N_PAST_max) = max(adapt_params.tau_min, ...
            adapt_params.beta * g_ref_init);
        
        % ========== Bucle principal de detección + imputación ==========
        for k = k_start : T
            
            % ---- PASO 1: PREDICCIÓN DEL DETECTOR ----
            % Construir ventana histórica para el DETECTOR
            past_g_det = g_test(k - det.N_PAST : k - 1);
            % Corregir valores <10 con predicciones previas del detector
            invalid_idx = find(past_g_det < 10);
            for j = invalid_idx'
                past_g_det(j) = ypred_det(k - det.N_PAST - 1 + j);
            end
            
            % Normalizar con stats del DETECTOR
            past_g_det_norm = (past_g_det - det.media_g) / det.std_g;
            u_norm_det = (u(k) - det.media_u) / det.std_u;
            m_norm_det = (m(k) - det.media_m) / det.std_m;
            
            % Construir secuencia de entrada
            x_det = [past_g_det_norm; u_norm_det; m_norm_det];
            switch det.input_shape
                case 'column'
                    xseq_det = {reshape(x_det, [], 1)};
                case 'row'
                    xseq_det = {reshape(x_det, 1, [])};
            end
            
            % Predecir y desnormalizar
            y_hat_det_norm = predict(net_det, xseq_det);
            y_hat_det = y_hat_det_norm * det.std_g + det.media_g;
            ypred_det(k) = y_hat_det;
            
            % ---- PASO 2: CALCULAR ERROR Y UMBRAL ----
            err_detect = abs(y_hat_det - g_test(k));
            error_detect(k) = err_detect;
            
            % Umbral adaptativo B+ROC
            [tau_k, tau_info] = compute_adaptive_threshold( ...
                y_hat_det, past_g_det, adapt_params);
            threshold_vec(k) = tau_k;
            roc_vec(k) = tau_info.roc;
            
            % ---- PASO 3: PREDICCIÓN DEL IMPUTADOR (EN CADA MUESTRA) ----
            %  [NUEVO] El imputador ahora predice SIEMPRE, haya fallo o no.
            %  Esto NO cambia la señal entregada g_final (en muestras sin fallo
            %  se conserva la lectura del sensor, igual que antes), pero permite
            %  medir el error de PREDICCIÓN del imputador en todas las muestras
            %  (definición D1, comparable con el paper y con el Bloque B).
            %  En la rama de fallo se REUTILIZA esta misma y_hat_imp, por lo que
            %  la imputación es idéntica a la versión anterior.
            past_g_imp = g_test(k - imp.N_PAST : k - 1);
            % Corregir valores <10 con predicciones previas del imputador
            invalid_idx_imp = find(past_g_imp < 10);
            for j = invalid_idx_imp'
                past_g_imp(j) = ypred_imp(k - imp.N_PAST - 1 + j);
            end

            % Normalizar con stats del IMPUTADOR
            past_g_imp_norm = (past_g_imp - imp.media_g) / imp.std_g;
            u_norm_imp = (u(k) - imp.media_u) / imp.std_u;
            m_norm_imp = (m(k) - imp.media_m) / imp.std_m;

            % Construir secuencia de entrada
            x_imp = [past_g_imp_norm; u_norm_imp; m_norm_imp];
            switch imp.input_shape
                case 'column'
                    xseq_imp = {reshape(x_imp, [], 1)};
                case 'row'
                    xseq_imp = {reshape(x_imp, 1, [])};
            end

            % Predecir y desnormalizar (predicción cruda del imputador)
            y_hat_imp_norm = predict(net_imp, xseq_imp);
            y_hat_imp = y_hat_imp_norm * imp.std_g + imp.media_g;

            % Guardar predicción cruda y error D1 (en TODAS las muestras)
            ypred_imp_raw(k)     = y_hat_imp;
            error_impute_pred(k) = abs(y_hat_imp - g_test_real(k));   % [D1]

            % ---- PASO 4: DECISIÓN DE FALLO ----
            if err_detect > tau_k
                % === FALLO DETECTADO: usar IMPUTADOR ===
                fallas_vec(k) = true;
                ypred_imp(k)  = y_hat_imp;   % (semántica original: predicción del imputador)

                % Aplicar estrategia de imputación (igual que el original)
                if numel(error_detect(1:k-1)) >= winit
                    var_local = var(error_detect(k - winit : k - 1));
                    alpha = 1 / (1 + var_local);
                else
                    alpha = alpha_base;
                end
                
                if err_detect < 60
                    % Ruido blanco: suavizado adaptativo
                    g_final(k) = alpha * y_hat_imp + (1 - alpha) * g_final(k-1);
                else
                    % Desconexión: usar predicción directa del imputador
                    g_final(k) = y_hat_imp;
                end
                
            else
                % === NO HAY FALLO: conservar lectura del sensor ===
                fallas_vec(k) = false;
                g_final(k) = g_test(k);

                % La SALIDA no usa al imputador (se conserva el sensor), pero
                % ypred_imp mantiene su semántica original (predicción del
                % detector como referencia). La predicción cruda del imputador
                % de esta muestra ya quedó guardada en ypred_imp_raw(k) para D1.
                ypred_imp(k) = y_hat_det;
            end

            % Error de la SALIDA entregada vs señal real, todas las muestras [D2]
            error_impute_out(k) = abs(g_final(k) - g_test_real(k));

        end  % for k
        
        % Guardar resultados de este run
        g_final_all(i, :)           = g_final;
        ypred_det_all(i, :)         = ypred_det;
        ypred_imp_all(i, :)         = ypred_imp;
        ypred_imp_raw_all(i, :)     = ypred_imp_raw;     % [NUEVO]
        fallas_all(i, :)            = fallas_vec;
        error_detect_all(i, :)      = error_detect;
        error_impute_pred_all(i, :) = error_impute_pred;  % [D1]
        error_impute_out_all(i, :)  = error_impute_out;   % [D2]
        threshold_all(i, :)         = threshold_vec;
        roc_all(i, :)               = roc_vec;
        
    end  % for i (runs)
    
    % ===== GUARDAR RESULTADOS DE ESTA COMBINACIÓN =====
    combo_name = sprintf('%s_det_%s_imp', arch_detect, arch_impute);
    
    results_file = fullfile(output_path, ...
        sprintf('results_pipeline_%s_%s.mat', combo_name, timestamp));
    
    % También guardar versión "latest" para fácil acceso
    results_latest = fullfile(output_path, ...
        sprintf('results_pipeline_%s_latest.mat', combo_name));
    
    % Estructura de guardado (compatible con main_metrics_pipeline.m)
    pipeline_config = struct();
    pipeline_config.arch_detect    = arch_detect;
    pipeline_config.arch_impute    = arch_impute;
    pipeline_config.threshold_mode = 'adaptive';
    pipeline_config.adapt_params   = adapt_params;
    pipeline_config.winit          = winit;
    pipeline_config.alpha_base     = alpha_base;
    pipeline_config.N_PAST_detect  = det.N_PAST;
    pipeline_config.N_PAST_impute  = imp.N_PAST;
    pipeline_config.nRuns          = nRuns;
    
    save_vars = {'g_final_all', 'ypred_det_all', 'ypred_imp_all', ...
        'ypred_imp_raw_all', ...
        'fallas_all', 'error_detect_all', ...
        'error_impute_pred_all', 'error_impute_out_all', ...
        'threshold_all', 'roc_all', ...
        'g_test', 'g_test_real', 'true_f', ...
        'pipeline_config'};
    
    save(results_file, save_vars{:});
    save(results_latest, save_vars{:});
    
    fprintf('\n   ✅ Resultados guardados: %s\n', results_file);
    
    % ===== RESUMEN RÁPIDO DE ESTA COMBINACIÓN =====
    fprintf('\n   📊 RESUMEN RÁPIDO:\n');
    
    % Métricas de detección
    ytrue = true_f(:)';
    accs = zeros(nRuns, 1); precs = zeros(nRuns, 1);
    recs = zeros(nRuns, 1); f1s = zeros(nRuns, 1);
    
    for i = 1:nRuns
        yhat = fallas_all(i, :);
        TP = sum(yhat == 1 & ytrue == 1);
        FP = sum(yhat == 1 & ytrue == 0);
        FN = sum(yhat == 0 & ytrue == 1);
        TN = sum(yhat == 0 & ytrue == 0);
        
        accs(i) = (TP + TN) / (TP + TN + FP + FN);
        precs(i) = TP / max(TP + FP, 1);
        recs(i) = TP / max(TP + FN, 1);
        f1s(i) = 2 * (precs(i) * recs(i)) / max(precs(i) + recs(i), eps);
    end
    
    fprintf('      Detección:  Acc=%.3f±%.3f | Prec=%.3f±%.3f | Rec=%.3f±%.3f | F1=%.3f±%.3f\n', ...
        mean(accs), std(accs), mean(precs), std(precs), ...
        mean(recs), std(recs), mean(f1s), std(f1s));
    
    % Métricas de imputación
    %  D1 (paper-comparable): error de predicción del imputador, todas las muestras
    MAEs_d1  = mean(error_impute_pred_all, 2);
    RMSEs_d1 = sqrt(mean(error_impute_pred_all.^2, 2));

    %  D3 (solo muestras con fallo REAL): calidad de reconstrucción de la salida
    %  NOTA: g_test_real puede estar almacenado como vector columna; se fuerza
    %  a vector fila (y_real_row) para que coincida con la orientación de
    %  g_final_all(i, true_fault_idx), que SIEMPRE es fila (indexado 2D).
    %  Sin esto, MATLAB hace broadcasting fila-vs-columna y genera una matriz
    %  n×n en vez de un vector, rompiendo mean()/la asignación a MAEs_d3(i).
    y_real_row = g_test_real(:)';
    true_fault_idx = (true_f(:)' == 1);
    MAEs_d3  = zeros(nRuns, 1);
    RMSEs_d3 = zeros(nRuns, 1);
    for i = 1:nRuns
        e_corr = abs(g_final_all(i, true_fault_idx) - y_real_row(true_fault_idx));
        MAEs_d3(i)  = mean(e_corr);
        RMSEs_d3(i) = sqrt(mean(e_corr.^2));
    end

    fprintf('      Imputación D1 (pred., todas):    MAE=%.2f±%.2f | RMSE=%.2f±%.2f mg/dL\n', ...
        mean(MAEs_d1), std(MAEs_d1), mean(RMSEs_d1), std(RMSEs_d1));
    fprintf('      Imputación D3 (salida, fallos):  MAE=%.2f±%.2f | RMSE=%.2f±%.2f mg/dL\n', ...
        mean(MAEs_d3), std(MAEs_d3), mean(RMSEs_d3), std(RMSEs_d3));

    % (compatibilidad con el resumen global de abajo: usar D1 como métrica principal)
    MAEs  = MAEs_d1;
    RMSEs = RMSEs_d1;
    
    % Almacenar en estructura global
    all_results(combo).combo_name  = combo_name;
    all_results(combo).arch_detect = arch_detect;
    all_results(combo).arch_impute = arch_impute;
    all_results(combo).acc  = [mean(accs), std(accs)];
    all_results(combo).prec = [mean(precs), std(precs)];
    all_results(combo).rec  = [mean(recs), std(recs)];
    all_results(combo).f1   = [mean(f1s), std(f1s)];
    all_results(combo).mae  = [mean(MAEs), std(MAEs)];
    all_results(combo).rmse = [mean(RMSEs), std(RMSEs)];
    
end  % for combo

%% ===================== TABLA COMPARATIVA FINAL =====================

fprintf('\n\n');
fprintf('╔══════════════════════════════════════════════════════════════════════════════════╗\n');
fprintf('║                    COMPARACIÓN DE PIPELINES INTEGRADOS                         ║\n');
fprintf('╠══════════════════════════════════════════════════════════════════════════════════╣\n');
fprintf('║ %-25s │ %-12s │ %-12s │ %-12s │ %-12s ║\n', ...
    'Combinación', 'F1-Score', 'Precision', 'MAE(mg/dL)', 'RMSE(mg/dL)');
fprintf('╠══════════════════════════════════════════════════════════════════════════════════╣\n');

best_f1 = 0;
best_mae = Inf;
best_combo_f1 = '';
best_combo_mae = '';

for combo = 1:nCombos
    r = all_results(combo);
    
    fprintf('║ %-25s │ %.3f±%.3f │ %.3f±%.3f │ %.2f±%.2f  │ %.2f±%.2f  ║\n', ...
        r.combo_name, ...
        r.f1(1), r.f1(2), ...
        r.prec(1), r.prec(2), ...
        r.mae(1), r.mae(2), ...
        r.rmse(1), r.rmse(2));
    
    if r.f1(1) > best_f1
        best_f1 = r.f1(1);
        best_combo_f1 = r.combo_name;
    end
    if r.mae(1) < best_mae
        best_mae = r.mae(1);
        best_combo_mae = r.combo_name;
    end
end

fprintf('╠══════════════════════════════════════════════════════════════════════════════════╣\n');
fprintf('║ 🏆 Mejor F1:  %-30s (F1=%.3f)                  ║\n', best_combo_f1, best_f1);
fprintf('║ 🏆 Mejor MAE: %-30s (MAE=%.2f)                  ║\n', best_combo_mae, best_mae);
fprintf('╚══════════════════════════════════════════════════════════════════════════════════╝\n');

%% ===================== GUARDAR RESUMEN GLOBAL =====================

summary_file = fullfile(output_path, sprintf('pipeline_summary_%s.mat', timestamp));
save(summary_file, 'all_results', 'combinaciones', 'adapt_params', 'timestamp');
fprintf('\n💾 Resumen global guardado en: %s\n', summary_file);

fprintf('\n📊 Para métricas detalladas, ejecute main_metrics_v2.m con results_block = ''bloque_C''\n');
fprintf('🏁 Pipeline integrado completado.\n');