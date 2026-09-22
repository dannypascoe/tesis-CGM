%% =========================================================================
%  MAIN_ONLINE_DETECTION_OHIO.M - Deteccion+Imputacion en linea sobre datos
%  reales de OhioT1DM (validacion externa de los modelos entrenados en Sorensen)
%  =========================================================================
%  Descripcion: Corre el MISMO algoritmo de main_online_detection_v2.m
%               (identico, sin cambios en la logica de deteccion/imputacion)
%               pero cargando la ventana real de OhioT1DM en vez de los
%               datos sinteticos de Sorensen. true_f aqui solo marca los
%               huecos REALES de la ventana (sin inyeccion sintetica
%               todavia -- eso es un paso posterior, pendiente de decidir).
%
%  Prerrequisitos:
%    - Modelo ya entrenado: trained_<architecture>_latest.mat
%    - Ventana de Ohio ya seleccionada y preparada:
%        main_ohio_window_selection_v1.m  -> paciente_<ID>_ventana5d_latest.mat
%        load_and_prepare_data_ohio.m, build_ohio_insulin_signal.m,
%        build_ohio_meal_signal.m, subject2_params.m (en el path)
%
%  Salida:
%    data/ohio/results/results_<architecture>_<paciente>_<timestamp>.mat
%    data/ohio/results/results_<architecture>_<paciente>_latest.mat
%  =========================================================================

clear; clc; close all;

%% ===================== AGREGAR CARPETAS AL PATH =====================
scriptPath = 'D:\Documentos\02 MROI\07 Tesis\03 Extension Tesis\CGM_Project\CGM_Project\matlab';

%addpath(fullfile(scriptPath, 'matlab'));              % subject2_params, build_ohio_*, load_and_prepare_data_ohio
addpath(fullfile(scriptPath, 'architectures'));
addpath(fullfile(scriptPath, 'data_preparation'));
addpath(fullfile(scriptPath, 'utils'));                % compute_adaptive_threshold.m
addpath(fullfile(scriptPath, 'data'));

fprintf('📁 Carpetas agregadas al path de MATLAB\n');

%% ===================== CONFIGURACION =====================

% Arquitectura a evaluar (debe tener trained_<architecture>_latest.mat)
architecture = 'cnn_lstm';

% Ventana de Ohio ya seleccionada (paciente 559 por default)
window_mat_path = fullfile(scriptPath, 'data', 'ohio', 'prepared', 'paciente_559_ventana5d_latest.mat');

% Parametros de imputacion (identicos a main_online_detection_v2.m)
winit      = 30;
alpha_base = 0.3;

% Modo de umbral: 'fixed' o 'adaptive'
threshold_mode = 'adaptive';
umbral_fijo = 10;   % mg/dL

adapt_params.beta    = 0.10;
adapt_params.tau_min = 7.0;
adapt_params.gamma   = 0.05;

fprintf('⚙️  Modo de umbral: %s\n', upper(threshold_mode));

% Rutas
models_path = fullfile(scriptPath, 'data', 'models');
output_path = fullfile(scriptPath, 'data', 'ohio', 'results');   % PARALELO a bloque_A..D, no dentro
if ~exist(output_path, 'dir')
    mkdir(output_path);
    fprintf('📁 Carpeta creada: %s\n', output_path);
end

%% ===================== CARGAR MODELO ENTRENADO =====================

fprintf('📂 Cargando modelo entrenado...\n');

model_file = fullfile(models_path, sprintf('trained_%s_latest.mat', architecture));
if ~exist(model_file, 'file')
    error('No se encontró el modelo: %s', model_file);
end

load(model_file, 'nets', 'media_g', 'std_g', 'media_u', 'std_u', ...
    'media_m', 'std_m', 'N_PAST', 'input_shape');

fprintf('   ✅ Modelo cargado: %s (%d redes)\n', architecture, numel(nets));

%% ===================== CARGAR VENTANA DE OHIO =====================

fprintf('📂 Cargando ventana real de OhioT1DM...\n');

[g_test, g_test_real, u, m, true_f, grid_ts, meta] = load_and_prepare_data_ohio(window_mat_path);

T = length(g_test);
fprintf('   ✅ Paciente %s | %d muestras | %.2f%% huecos reales (true_f)\n', ...
    meta.patient_id, T, 100*mean(true_f));

%% ===================== PREDICCION EN LINEA (idéntico a main_online_detection_v2.m) =====================

fprintf('\n🚀 Iniciando predicción en línea sobre datos reales de Ohio...\n');

nModels = numel(nets);

g_final_all      = zeros(nModels, T);
ypred_all        = zeros(nModels, T);
fallas_all       = false(nModels, T);
error_detect_all = zeros(nModels, T);
error_impute_all = zeros(nModels, T);
threshold_all    = zeros(nModels, T);
roc_all          = zeros(nModels, T);

for i = 1:nModels
    fprintf('   [%d/%d] Evaluando red %d...\n', i, nModels, i);

    net_i = nets{i};

    g_final      = zeros(1, T);
    ypred        = zeros(1, T);
    fallas_vec   = false(1, T);
    error_detect = zeros(1, T);
    error_impute = zeros(1, T);
    threshold_vec = zeros(1, T);
    roc_vec       = zeros(1, T);

    g_final(1:N_PAST) = g_test(1:N_PAST);
    ypred(1:N_PAST)   = g_test(1:N_PAST);

    switch threshold_mode
        case 'fixed'
            threshold_vec(1:N_PAST) = umbral_fijo;
        case 'adaptive'
            g_ref_init = mean(g_test(1:N_PAST));
            threshold_vec(1:N_PAST) = max(adapt_params.tau_min, ...
                adapt_params.beta * g_ref_init);
    end

    for k = (N_PAST + 1) : T

        % 1) Ventana historica, corrigiendo valores <10 (huecos reales incluidos)
        past_g = g_test(k - N_PAST : k - 1);
        invalid_idx = find(past_g < 10);
        for j = invalid_idx'
            past_g(j) = ypred(k - N_PAST - 1 + j);
        end

        % 2) Normalizar (con media_g/std_g/media_u/std_u/media_m/std_m DEL MODELO, no recalculados)
        past_g_norm = (past_g - media_g) / std_g;
        u_norm_k    = (u(k) - media_u) / std_u;
        m_norm_k    = (m(k) - media_m) / std_m;

        % 3) Secuencia de entrada
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

        % 5) Errores (err_impute sera NaN en muestras con hueco real, ya
        %    que g_test_real es NaN ahi -- esperado, filtrar despues)
        err_detect = abs(y_hat - g_test(k));
        err_impute = abs(y_hat - g_test_real(k));
        error_detect(k) = err_detect;
        error_impute(k) = err_impute;

        % 5.5) Umbral de deteccion
        switch threshold_mode
            case 'fixed'
                tau_k = umbral_fijo;
                roc_k = 0;
            case 'adaptive'
                [tau_k, tau_info] = compute_adaptive_threshold(y_hat, past_g, adapt_params);
                roc_k = tau_info.roc;
        end
        threshold_vec(k) = tau_k;
        roc_vec(k) = roc_k;

        % 6) Detectar fallo e imputar
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

    g_final_all(i, :)      = g_final;
    ypred_all(i, :)        = ypred;
    fallas_all(i, :)       = fallas_vec;
    error_detect_all(i, :) = error_detect;
    error_impute_all(i, :) = error_impute;
    threshold_all(i, :)    = threshold_vec;
    roc_all(i, :)          = roc_vec;
end

fprintf('   ✅ Predicción completada\n');

%% ===================== GUARDAR RESULTADOS =====================

fprintf('\n💾 Guardando resultados...\n');

timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');

results_file = fullfile(output_path, ...
    sprintf('results_%s_%s_%s.mat', architecture, meta.patient_id, timestamp));
results_latest = fullfile(output_path, ...
    sprintf('results_%s_%s_latest.mat', architecture, meta.patient_id));

save_vars = {'g_final_all','ypred_all','fallas_all', ...
    'error_detect_all','error_impute_all','threshold_all','roc_all', ...
    'g_test','g_test_real','true_f','grid_ts','meta', ...
    'threshold_mode','umbral_fijo','adapt_params','winit','alpha_base', ...
    'architecture','N_PAST'};

save(results_file, save_vars{:});
save(results_latest, save_vars{:});

fprintf('   ✅ Resultados guardados en: %s\n', results_file);

%% ===================== RESUMEN =====================

fprintf('\n═══════════════════════════════════════════\n');
fprintf('✅ DETECCIÓN E IMPUTACIÓN COMPLETADA (OhioT1DM)\n');
fprintf('   Paciente: %s\n', meta.patient_id);
fprintf('   Arquitectura: %s\n', upper(architecture));
fprintf('   Modo umbral:  %s\n', upper(threshold_mode));
fprintf('   Muestras: %d | huecos reales (true_f=1): %d (%.2f%%)\n', ...
    T, sum(true_f), 100*mean(true_f));

% Recall sobre los huecos reales (conteo simple, NO se reporta como
% metrica formal por bajo tamano de muestra -- ver discusion en el chat)
idx_real_gap = find(true_f == 1);
if ~isempty(idx_real_gap)
    recall_por_modelo = mean(fallas_all(:, idx_real_gap), 2);
    fprintf('   Detectados correctamente en los %d huecos reales, por modelo:\n', numel(idx_real_gap));
    for i = 1:nModels
        fprintf('     Modelo %d: %d/%d (%.0f%%)\n', i, ...
            sum(fallas_all(i, idx_real_gap)), numel(idx_real_gap), 100*recall_por_modelo(i));
    end
end
fprintf('═══════════════════════════════════════════\n');

%% ===================== FIGURA DE DIAGNOSTICO =====================

figure('Name', sprintf('Ohio %s - %s', meta.patient_id, upper(architecture)), ...
       'Position', [100, 100, 1100, 600], 'Color', 'w');

modelo_plot = 1;
hold on;
plot(grid_ts, g_test, 'Color', 'k', 'LineWidth', 0.8, 'DisplayName', 'g_{test} (real, huecos=0)');
plot(grid_ts, g_final_all(modelo_plot,:), 'b', 'LineWidth', 1.2, 'DisplayName', 'g_{final} (salida pipeline)');

idx_fallo = find(fallas_all(modelo_plot,:));
if ~isempty(idx_fallo)
    scatter(grid_ts(idx_fallo), g_final_all(modelo_plot, idx_fallo), 25, 'r', 'filled', ...
        'DisplayName', 'Fallo detectado');
end

legend('Location','best');
ylabel('Glucosa [mg/dL]'); xlabel('Tiempo');
title(sprintf('Validación OhioT1DM — paciente %s — %s (modelo %d)', ...
    meta.patient_id, upper(architecture), modelo_plot));
grid on; hold off;


%% ===================== FUNCION AUXILIAR (identica a main_online_detection_v2.m) =====================

function [ypred_k, fallo] = detect_and_impute(y_hat, prev_pred, err, error_hist, umbral, winit, alpha_base)
%DETECT_AND_IMPUTE Detecta fallo e imputa valor si es necesario (sin cambios
%   respecto a main_online_detection_v2.m).
    if err > umbral
        if numel(error_hist) >= winit
            var_local = var(error_hist(end - winit + 1 : end));
            alpha = 1 / (1 + var_local);
        else
            alpha = alpha_base;
        end

        if err < 60
            ypred_k = alpha * y_hat + (1 - alpha) * prev_pred;
        else
            ypred_k = y_hat;
        end

        fallo = true;
    else
        ypred_k = 0.8 * y_hat + 0.2 * prev_pred;
        fallo = false;
    end
end