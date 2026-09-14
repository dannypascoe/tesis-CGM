    %% ========================================================================
%  MAIN_METRICS.M - Cálculo de métricas y visualización de resultados
%  
%  Este script calcula métricas de regresión (imputación) y clasificación
%  (detección de fallos) a partir de los resultados generados por 
%  main_online_detection_v2.m
%
%  Requiere: results_<architecture>_latest.mat en data/models/
%
%  [NUEVO] Sección de comparación fijo vs adaptativo al final.
%          Activar con compare_thresholds = true en CONFIGURACIÓN.
% ========================================================================

clear; clc; close all;

%% ===================== CONFIGURACIÓN =====================

% Ruta base del proyecto (MODIFICAR SEGÚN TU UBICACIÓN)
scriptPath = 'D:\Documentos\02 MROI\07 Tesis\03 Extension Tesis\CGM_Project\CGM_Project';

% Seleccionar arquitectura a evaluar
% Opciones: 'cnn_lstm', 'gru', '1dcnn', 'transformer_lstm'
architecture = 'gru';

% Configuración de gráficas
plot_config.show_all_models    = true;   % true: mostrar las 5 redes, false: solo las seleccionadas
plot_config.models_to_plot     = [1, 5]; % Índices de modelos a graficar si show_all_models=false
plot_config.show_imputation    = true;   % Gráfica de imputación
plot_config.show_fault_detect  = true;   % Gráfica de detección de fallos
plot_config.show_confusion_mat = true;   % Matriz de confusión promedio
plot_config.show_roc_mean      = true;   % Curva ROC promedio
plot_config.show_roc_subplots  = true;   % ROC en subplots 2x3

% Configuración de tabla detallada por modelo
show_detailed_table  = true;   % Mostrar tabla con métricas individuales por modelo
show_table_figure    = false;  % Mostrar tabla como figura (además de consola)
export_to_excel      = true;  % Exportar tabla a archivo Excel

% Configuración de guardado
save_results = false;  % true: guardar métricas en archivo .mat

% -----------------------------------------------------------------------
% [NUEVO] CONFIGURACIÓN: COMPARACIÓN FIJO vs ADAPTATIVO
%
%  compare_thresholds : true  → ejecuta la sección comparativa al final
%                       false → salta esa sección (comportamiento original)
%
%  compare_arch  : arquitectura a comparar (por defecto la misma que arriba)
%
%  Archivos a comparar:
%    - Dejar en '' para auto-detectar el archivo con timestamp más reciente
%      en la carpeta models_path (busca patrón results_<arch>_fixed_*.mat)
%    - O bien especificar el nombre de archivo exacto (sin ruta completa),
%      por ejemplo: 'results_cnn_lstm_fixed_2026-01-31_14-28-56.mat'
% -----------------------------------------------------------------------
compare_thresholds = true;              % <<< ACTIVAR AQUÍ
compare_arch       = architecture;      % arquitectura a comparar
file_fixed         = '';               % '' = auto-detectar más reciente
file_adaptive      = '';               % '' = auto-detectar más reciente

% Rutas
models_path = fullfile(scriptPath, 'data', 'models');

% -----------------------------------------------------------------------
% CARPETA DE ENTRADA/SALIDA SEGÚN BLOQUE
%   Para Bloque A (fijo)      → apuntar a bloque_A/
%   Para Bloque B (adaptativo)→ apuntar a bloque_B/
%   Para Bloque C (pipeline)  → apuntar a bloque_C/  (ver main_pipeline)
%
%   Cambiar 'bloque_A' o 'bloque_B' según qué bloque estás evaluando.
%   Debe coincidir con el threshold_mode que usaste en main_online_detection.
% -----------------------------------------------------------------------
results_block = 'bloque_B';   % <<< CAMBIAR SEGÚN BLOQUE: 'bloque_A' o 'bloque_B'
output_path   = fullfile(models_path, results_block);

if ~exist(output_path, 'dir')
    error('❌ No se encontró la carpeta: %s\n   Ejecuta primero main_online_detection_v2.m', output_path);
end

%% ===================== CARGAR RESULTADOS =====================

fprintf('═══════════════════════════════════════════\n');
fprintf('📊 CÁLCULO DE MÉTRICAS - %s\n', upper(architecture));
fprintf('═══════════════════════════════════════════\n\n');

results_file = fullfile(output_path, sprintf('results_%s_latest.mat', architecture));
%results_file = fullfile(output_path, 'results_cnn_lstm_fixed_2026-01-31_14-28-56.mat');  % archivo específico

if ~isfile(results_file)
    error('❌ No se encontró el archivo de resultados: %s\n   Ejecuta primero main_online_detection_v2.m', results_file);
end

fprintf('📂 Cargando resultados desde:\n   %s\n\n', results_file);
load(results_file, ...
    'g_final_all', 'ypred_all', 'fallas_all', ...
    'error_detect_all', 'error_impute_all', ...
    'g_test', 'g_test_real', 'true_f', ...
    'umbral_min', 'N_PAST');

% Obtener dimensiones
nModels = size(g_final_all, 1);
N = size(g_final_all, 2);

fprintf('   Modelos evaluados: %d\n', nModels);
fprintf('   Muestras totales: %d\n\n', N);

%% ===================== MÉTRICAS DE REGRESIÓN (IMPUTACIÓN) =====================

fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
fprintf('📈 MÉTRICAS DE IMPUTACIÓN (mean ± std)\n');
fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n');

% ----- MAE (Mean Absolute Error) -----
MAEs = mean(error_impute_all, 2);
mean_MAE = mean(MAEs);
std_MAE  = std(MAEs);
fprintf('MAE:  %.2f ± %.2f mg/dL\n', mean_MAE, std_MAE);

% ----- MSE (Mean Squared Error) -----
MSEs = mean(error_impute_all.^2, 2);
mean_MSE = mean(MSEs);
std_MSE  = std(MSEs);
fprintf('MSE:  %.4f ± %.4f mg²/dL²\n', mean_MSE, std_MSE);

% ----- RMSE (Root Mean Squared Error) -----
RMSEs = sqrt(MSEs);
mean_RMSE = mean(RMSEs);
std_RMSE  = std(RMSEs);
fprintf('RMSE: %.2f ± %.2f mg/dL\n', mean_RMSE, std_RMSE);

% ----- MARD (Mean Absolute Relative Difference) -----
MARDs_global = zeros(nModels, 1);
MARDs_fallas = zeros(nModels, 1);
y_true = g_test_real(:)';  % Señal real (sin fallas)

for i = 1:nModels
    y_imp    = g_final_all(i, :);
    fallas_i = fallas_all(i, :);
    
    % MARD global (solo donde y_true >= 20 para evitar división por valores muy pequeños)
    valid_idx_global = (y_true >= 20);
    rel_err_global = abs(y_imp(valid_idx_global) - y_true(valid_idx_global)) ./ y_true(valid_idx_global);
    MARDs_global(i) = 100 * mean(rel_err_global);
    
    % MARD solo en puntos imputados (donde se detectó falla)
    valid_idx_fallas = fallas_i & (y_true >= 20);
    if any(valid_idx_fallas)
        rel_err_fallas = abs(y_imp(valid_idx_fallas) - y_true(valid_idx_fallas)) ./ y_true(valid_idx_fallas);
        MARDs_fallas(i) = 100 * mean(rel_err_fallas);
    else
        MARDs_fallas(i) = NaN;
    end
end

mean_MARD_global = mean(MARDs_global, 'omitnan');
std_MARD_global  = std(MARDs_global, 'omitnan');
mean_MARD_fallas = mean(MARDs_fallas, 'omitnan');
std_MARD_fallas  = std(MARDs_fallas, 'omitnan');

fprintf('MARD (global, ≥20 mg/dL):   %.2f ± %.2f %%\n', mean_MARD_global, std_MARD_global);
fprintf('MARD (solo imputados):      %.2f ± %.2f %%\n', mean_MARD_fallas, std_MARD_fallas);

% ----- R² (Coeficiente de determinación) -----
R2s = zeros(nModels, 1);
y_true = g_test_real(:)';
y_mean = mean(y_true);

for i = 1:nModels
    y_pred = g_final_all(i, :);
    ss_res = sum((y_true - y_pred).^2);
    ss_tot = sum((y_true - y_mean).^2);
    R2s(i) = 1 - (ss_res / ss_tot);
end

mean_R2 = mean(R2s);
std_R2  = std(R2s);
fprintf('R²:   %.4f ± %.4f\n\n', mean_R2, std_R2);

%% ===================== MÉTRICAS DE CLASIFICACIÓN (DETECCIÓN) =====================

fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
fprintf('📈 MÉTRICAS DE DETECCIÓN DE FALLOS (mean ± std)\n');
fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n');

% Preallocate
accuracies = zeros(nModels, 1);
precisions = zeros(nModels, 1);
recalls    = zeros(nModels, 1);
f1scores   = zeros(nModels, 1);
aucs       = zeros(nModels, 1);

ytrue = true_f(:)';  % Etiquetas verdaderas

for i = 1:nModels
    yhat = fallas_all(i, :);  % Predicciones binarias
    
    % Componentes de la matriz de confusión
    TP = sum(yhat == 1 & ytrue == 1);
    FP = sum(yhat == 1 & ytrue == 0);
    FN = sum(yhat == 0 & ytrue == 1);
    TN = sum(yhat == 0 & ytrue == 0);
    
    % Métricas
    accuracies(i) = (TP + TN) / (TP + TN + FP + FN);
    precisions(i) = TP / max(TP + FP, 1);
    recalls(i)    = TP / max(TP + FN, 1);
    f1scores(i)   = 2 * (precisions(i) * recalls(i)) / max(precisions(i) + recalls(i), eps);
    
    % AUC-ROC (usando error de detección como score continuo)
    [~, ~, ~, AUROC] = perfcurve(ytrue, error_detect_all(i, :), 1);
    aucs(i) = AUROC;
end

% Estadísticos finales
mean_acc  = mean(accuracies);  std_acc  = std(accuracies);
mean_prec = mean(precisions);  std_prec = std(precisions);
mean_rec  = mean(recalls);     std_rec  = std(recalls);
mean_f1   = mean(f1scores);    std_f1   = std(f1scores);
mean_auc  = mean(aucs);        std_auc  = std(aucs);

fprintf('Accuracy:  %.3f ± %.3f\n', mean_acc, std_acc);
fprintf('Precision: %.3f ± %.3f\n', mean_prec, std_prec);
fprintf('Recall:    %.3f ± %.3f\n', mean_rec, std_rec);
fprintf('F1-Score:  %.3f ± %.3f\n', mean_f1, std_f1);
fprintf('AUC:       %.3f ± %.3f\n\n', mean_auc, std_auc);

%% ===================== PREPARAR DATOS PARA ROC =====================

% Calcular curvas ROC para cada modelo (necesario para gráficas)
roc_X = cell(nModels, 1);
roc_Y = cell(nModels, 1);
roc_T = cell(nModels, 1);
confusion_matrices = zeros(2, 2, nModels);

ytrue_double = double(true_f(:)');

for i = 1:nModels
    ypred_bin  = double(fallas_all(i, :));
    ypred_cont = error_detect_all(i, :);
    
    % Matriz de confusión
    [C, ~] = confusionmat(ytrue_double, ypred_bin, 'Order', [0 1]);
    confusion_matrices(:, :, i) = C;
    
    % Curva ROC
    [X, Y, T, ~] = perfcurve(ytrue_double, ypred_cont, 1);
    roc_X{i} = X;
    roc_Y{i} = Y;
    roc_T{i} = T;
end

%% ===================== GRÁFICA: IMPUTACIÓN =====================

if plot_config.show_imputation
    figure('Name', sprintf('%s - Imputación', upper(architecture)), ...
           'Position', [100, 100, 1000, 500]);
    hold on;
    
    % Señal corrupta (con fallas)
    plot(g_test, 'r', 'LineWidth', 1.2, 'DisplayName', 'Corrupted Signal');
    
    % Colores y estilos para cada modelo
    colors = {'k', 'g', 'c', 'b', 'm'};
    lineStyles = {'--', '--', '--', '--', '--'};
    
    % Determinar qué modelos graficar
    if plot_config.show_all_models
        models_to_plot = 1:nModels;
    else
        models_to_plot = plot_config.models_to_plot;
    end
    
    % Graficar predicciones de los modelos seleccionados
    for idx = 1:length(models_to_plot)
        i = models_to_plot(idx);
        if i <= nModels
            plot(g_final_all(i, :), ...
                'Color', colors{i}, ...
                'LineStyle', lineStyles{i}, ...
                'LineWidth', 1.3, ...
                'DisplayName', sprintf('Model %d', i));
        end
    end
    
    % Señal real (sin fallas)
    plot(g_test_real, 'b', 'LineWidth', 1.5, 'DisplayName', 'Real Signal');
    
    legend('Location', 'best');
    ylabel('Glucose (mg/dL)', 'FontSize', 12);
    xlabel('Samples', 'FontSize', 12);
    title(sprintf('%s - Online Glucose Imputation', upper(architecture)));
    grid on;
    hold off;
end

%% ===================== GRÁFICA: DETECCIÓN DE FALLOS =====================

if plot_config.show_fault_detect
    figure('Name', sprintf('%s - Detección de Fallos', upper(architecture)), ...
           'Position', [100, 150, 1000, 400]);
    hold on;
    
    colors = {'k', 'g', 'c', 'b', 'm'};
    lineStyles = {'-', '-', '-', '-', '-'};
    
    if plot_config.show_all_models
        models_to_plot = 1:nModels;
    else
        models_to_plot = plot_config.models_to_plot;
    end
    
    for idx = 1:length(models_to_plot)
        i = models_to_plot(idx);
        if i <= nModels
            stem(fallas_all(i, :), ...
                'filled', ...
                'Color', colors{i}, ...
                'LineStyle', lineStyles{i}, ...
                'LineWidth', 1.0, ...
                'MarkerSize', 3, ...
                'DisplayName', sprintf('Model %d', i));
        end
    end
    
    legend('Location', 'best');
    ylabel('Detected Fault', 'FontSize', 12);
    xlabel('Samples', 'FontSize', 12);
    title(sprintf('%s - Online Fault Detection', upper(architecture)));
    ylim([-0.1, 1.1]);
    grid on;
    hold off;
end

%% ===================== GRÁFICA: MATRIZ DE CONFUSIÓN =====================

if plot_config.show_confusion_mat
    C_mean = mean(confusion_matrices, 3);
    
    fprintf('Matriz de Confusión Promedio:\n');
    fprintf('   TN = %.1f    FP = %.1f\n', C_mean(1,1), C_mean(1,2));
    fprintf('   FN = %.1f    TP = %.1f\n\n', C_mean(2,1), C_mean(2,2));
    
    figure('Name', sprintf('%s - Matriz de Confusión', upper(architecture)));
    heatmap({'No Fault', 'Fault'}, {'No Fault', 'Fault'}, round(C_mean), ...
        'ColorbarVisible', 'on');
    title(sprintf('%s - Mean Confusion Matrix', upper(architecture)));
end

%% ===================== GRÁFICA: CURVA ROC PROMEDIO =====================

if plot_config.show_roc_mean
    % Interpolar todas las curvas a un eje común
    fpr_common = linspace(0, 1, 1000);
    tpr_all = zeros(nModels, numel(fpr_common));
    
    for i = 1:nModels
        x = roc_X{i};
        y = roc_Y{i};
        [xu, ia] = unique(x);
        yu = y(ia);
        tpr_all(i, :) = interp1(xu, yu, fpr_common, 'linear', 'extrap');
    end
    
    mean_tpr = mean(tpr_all, 1);
    std_tpr  = std(tpr_all, 0, 1);
    upper_tpr = min(1, mean_tpr + std_tpr);
    lower_tpr = max(0, mean_tpr - std_tpr);
    
    % Figura ROC promedio
    figure('Name', sprintf('%s - ROC Promedio', upper(architecture)), ...
           'Color', 'w', 'Position', [100, 100, 600, 500]);
    hold on;
    
    % Área bajo la curva (sombreado)
    fill([fpr_common, fliplr(fpr_common)], ...
         [mean_tpr, zeros(size(mean_tpr))], ...
         [1.00, 0.45, 0.00], 'FaceAlpha', 0.10, 'EdgeColor', 'none');
    
    % Banda ±1 std
    fill([fpr_common, fliplr(fpr_common)], ...
         [upper_tpr, fliplr(lower_tpr)], ...
         [0.5, 0.5, 0.5], 'FaceAlpha', 0.3, 'EdgeColor', 'none');
    
    % Línea diagonal (clasificador aleatorio)
    plot([0, 1], [0, 1], 'k--', 'LineWidth', 1);
    
    % Curva ROC promedio
    plot(fpr_common, mean_tpr, 'b', 'LineWidth', 2);
    
    axis([0, 1, 0, 1]); axis square; grid on;
    xlabel('False Positive Rate', 'FontSize', 12);
    ylabel('True Positive Rate', 'FontSize', 12);
    title(sprintf('%s - Mean ROC ± 1 SD', upper(architecture)));
    
    % Texto con AUC
    text(0.65, 0.25, sprintf('Mean AUC = %.3f ± %.3f', mean_auc, std_auc), ...
        'FontSize', 12, 'FontWeight', 'bold');
    
    hold off;
end

%% ===================== GRÁFICA: ROC SUBPLOTS 2x3 =====================

if plot_config.show_roc_subplots
    model_names = arrayfun(@(k) sprintf('Model %d', k), 1:nModels, 'UniformOutput', false);
    
    figure('Name', sprintf('%s - ROC por Modelo', upper(architecture)), ...
           'Color', 'w', 'Position', [50, 50, 900, 600]);
    tiledlayout(2, 3, 'Padding', 'compact', 'TileSpacing', 'compact');
    cols = lines(max(nModels, 6));
    
    % 5 subplots con curvas individuales
    for i = 1:min(nModels, 5)
        nexttile;
        
        x = roc_X{i}(:);
        y = roc_Y{i}(:);
        [x, ia] = unique(x, 'stable');
        y = y(ia);
        
        % Asegurar que empiece en (0,0) y termine en (1,1)
        if x(1) > 0 || y(1) > 0
            x = [0; x];
            y = [0; y];
        end
        if x(end) < 1
            x = [x; 1];
            y = [y; 1];
        end
        
        hold on;
        plot([0, 1], [0, 1], 'k--', 'LineWidth', 1);
        fill([x; flipud(x)], [zeros(size(y)); flipud(y)], cols(i, :), ...
            'FaceAlpha', 0.08, 'EdgeColor', 'none');
        plot(x, y, 'LineWidth', 2, 'Color', cols(i, :));
        hold off;
        
        axis([0, 1, 0, 1]); axis square; grid on;
        title(model_names{i}, 'FontWeight', 'bold');
        xlabel('FPR'); ylabel('TPR');
        
        text(0.6, 0.2, sprintf('AUC = %.3f', aucs(i)), ...
            'FontWeight', 'bold', 'FontSize', 10);
    end
    
    % 6to subplot: promedio
    nexttile;
    
    fpr_common = linspace(0, 1, 1000);
    tpr_all = zeros(nModels, numel(fpr_common));
    for j = 1:nModels
        x = roc_X{j}(:);
        y = roc_Y{j}(:);
        [xu, ia] = unique(x);
        yu = y(ia);
        tpr_all(j, :) = interp1(xu, yu, fpr_common, 'linear', 'extrap');
    end
    mean_tpr = mean(tpr_all, 1);
    std_tpr  = std(tpr_all, 0, 1);
    
    hold on;
    plot([0, 1], [0, 1], 'k--', 'LineWidth', 1);
    fill([fpr_common, fliplr(fpr_common)], ...
         [mean_tpr + std_tpr, fliplr(mean_tpr - std_tpr)], ...
         [0.5, 0.5, 0.5], 'FaceAlpha', 0.25, 'EdgeColor', 'none');
    plot(fpr_common, mean_tpr, 'LineWidth', 2);
    hold off;
    
    axis([0, 1, 0, 1]); axis square; grid on;
    title('Mean ROC ± 1 SD', 'FontWeight', 'bold');
    xlabel('FPR'); ylabel('TPR');
    text(0.5, 0.2, sprintf('AUC = %.3f ± %.3f', mean_auc, std_auc), ...
        'FontWeight', 'bold', 'FontSize', 9);
    
    sgtitle(sprintf('%s - ROC Curves', upper(architecture)), 'FontWeight', 'bold','Interpreter', 'none');
end

%% ===================== RESUMEN FINAL =====================

fprintf('═══════════════════════════════════════════\n');
fprintf('✅ RESUMEN DE MÉTRICAS - %s\n', upper(architecture));
fprintf('═══════════════════════════════════════════\n');

fprintf('\n📊 Imputación:\n');
fprintf('   MAE:  %.2f ± %.2f mg/dL\n', mean_MAE, std_MAE);
fprintf('   RMSE: %.2f ± %.2f mg/dL\n', mean_RMSE, std_RMSE);
fprintf('   MARD: %.2f ± %.2f %%\n', mean_MARD_global, std_MARD_global);
fprintf('   R²:   %.4f ± %.4f\n', mean_R2, std_R2);
fprintf('\n🔍 Detección:\n');
fprintf('   Accuracy:  %.3f ± %.3f\n', mean_acc, std_acc);
fprintf('   Precision: %.3f ± %.3f\n', mean_prec, std_prec);
fprintf('   Recall:    %.3f ± %.3f\n', mean_rec, std_rec);
fprintf('   F1-Score:  %.3f ± %.3f\n', mean_f1, std_f1);
fprintf('   AUC:       %.3f ± %.3f\n', mean_auc, std_auc);
fprintf('═══════════════════════════════════════════\n\n');

%% ===================== TABLA DETALLADA POR MODELO =====================

if show_detailed_table
    fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
    fprintf('📋 TABLA DETALLADA POR MODELO - %s\n', upper(architecture));
    fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n');
    
    % --- Crear nombres de columnas ---
    model_names_col = cell(1, nModels);
    for i = 1:nModels
        model_names_col{i} = sprintf('Modelo_%d', i);
    end
    col_names = [model_names_col, {'Mean', 'Std', 'Mean_pm_Std'}];
    
    % --- TABLA DE REGRESIÓN ---
    fprintf('📈 MÉTRICAS DE REGRESIÓN (Imputación)\n');
    fprintf('─────────────────────────────────────────────────────────────────────────────────\n');
    
    % Construir matriz de datos de regresión
    reg_metrics = {
        'MAE',          MAEs',           mean_MAE,         std_MAE;
        'MSE',          MSEs',           mean_MSE,         std_MSE;
        'RMSE',         RMSEs',          mean_RMSE,        std_RMSE;
        'MARD_global',  MARDs_global',   mean_MARD_global, std_MARD_global;
        'MARD_fallas',  MARDs_fallas',   mean_MARD_fallas, std_MARD_fallas;
        'R2',           R2s',            mean_R2,          std_R2;
    };
    
    % Crear tabla de regresión
    reg_data = zeros(size(reg_metrics, 1), nModels + 3);
    reg_row_names = cell(size(reg_metrics, 1), 1);
    
    for r = 1:size(reg_metrics, 1)
        reg_row_names{r} = reg_metrics{r, 1};
        reg_data(r, 1:nModels) = reg_metrics{r, 2};
        reg_data(r, nModels+1) = reg_metrics{r, 3};  % Mean
        reg_data(r, nModels+2) = reg_metrics{r, 4};  % Std
    end
    
    % Mostrar tabla de regresión en consola
    T_reg = array2table(reg_data(:, 1:nModels+2), ...
        'VariableNames', [model_names_col, {'Mean', 'Std'}], ...
        'RowNames', reg_row_names);
    disp(T_reg);
    
    % --- TABLA DE CLASIFICACIÓN ---
    fprintf('\n🔍 MÉTRICAS DE CLASIFICACIÓN (Detección de Fallos)\n');
    fprintf('─────────────────────────────────────────────────────────────────────────────────\n');
    
    % Construir matriz de datos de clasificación
    class_metrics = {
        'Accuracy',   accuracies',   mean_acc,   std_acc;
        'Precision',  precisions',   mean_prec,  std_prec;
        'Recall',     recalls',      mean_rec,   std_rec;
        'F1_Score',   f1scores',     mean_f1,    std_f1;
        'AUC',        aucs',         mean_auc,   std_auc;
    };
    
    % Crear tabla de clasificación
    class_data = zeros(size(class_metrics, 1), nModels + 2);
    class_row_names = cell(size(class_metrics, 1), 1);
    
    for r = 1:size(class_metrics, 1)
        class_row_names{r} = class_metrics{r, 1};
        class_data(r, 1:nModels) = class_metrics{r, 2};
        class_data(r, nModels+1) = class_metrics{r, 3};  % Mean
        class_data(r, nModels+2) = class_metrics{r, 4};  % Std
    end
    
    % Mostrar tabla de clasificación en consola
    T_class = array2table(class_data, ...
        'VariableNames', [model_names_col, {'Mean', 'Std'}], ...
        'RowNames', class_row_names);
    disp(T_class);
    
    fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n');
    
    % --- MOSTRAR COMO FIGURA (OPCIONAL) ---
    if show_table_figure
        % Figura para tabla de regresión
        fig_reg = figure('Name', sprintf('%s - Tabla Regresión', upper(architecture)), ...
                         'Position', [100, 400, 800, 250], 'Color', 'w');
        
        % Preparar datos con formato string para Mean±Std
        reg_display = cell(size(reg_metrics, 1), nModels + 1);
        for r = 1:size(reg_metrics, 1)
            for c = 1:nModels
                reg_display{r, c} = sprintf('%.4f', reg_data(r, c));
            end
            reg_display{r, nModels+1} = sprintf('%.4f ± %.4f', reg_data(r, nModels+1), reg_data(r, nModels+2));
        end
        
        uitable(fig_reg, 'Data', reg_display, ...
            'ColumnName', [model_names_col, {'Mean ± Std'}], ...
            'RowName', reg_row_names, ...
            'Units', 'normalized', ...
            'Position', [0.02, 0.1, 0.96, 0.85], ...
            'FontSize', 10);
        sgtitle(sprintf('%s - Métricas de Regresión', upper(architecture)), 'FontWeight', 'bold','Interpreter', 'none');
        
        % Figura para tabla de clasificación
        fig_class = figure('Name', sprintf('%s - Tabla Clasificación', upper(architecture)), ...
                           'Position', [100, 100, 800, 220], 'Color', 'w');
        
        class_display = cell(size(class_metrics, 1), nModels + 1);
        for r = 1:size(class_metrics, 1)
            for c = 1:nModels
                class_display{r, c} = sprintf('%.4f', class_data(r, c));
            end
            class_display{r, nModels+1} = sprintf('%.4f ± %.4f', class_data(r, nModels+1), class_data(r, nModels+2));
        end
        
        uitable(fig_class, 'Data', class_display, ...
            'ColumnName', [model_names_col, {'Mean ± Std'}], ...
            'RowName', class_row_names, ...
            'Units', 'normalized', ...
            'Position', [0.02, 0.1, 0.96, 0.85], ...
            'FontSize', 10);
        sgtitle(sprintf('%s - Métricas de Clasificación', upper(architecture)), 'FontWeight', 'bold','Interpreter', 'none');
    end
    
    % --- EXPORTAR A EXCEL (OPCIONAL) ---
    if export_to_excel
        excel_filename = fullfile(output_path, sprintf('metrics_table_%s.xlsx', architecture));
        
        % Agregar columna Mean±Std como string
        mean_std_reg = cell(size(reg_metrics, 1), 1);
        for r = 1:size(reg_metrics, 1)
            mean_std_reg{r} = sprintf('%.4f ± %.4f', reg_data(r, nModels+1), reg_data(r, nModels+2));
        end
        T_reg_export = [T_reg, table(mean_std_reg, 'VariableNames', {'Mean_pm_Std'})];
        
        mean_std_class = cell(size(class_metrics, 1), 1);
        for r = 1:size(class_metrics, 1)
            mean_std_class{r} = sprintf('%.4f ± %.4f', class_data(r, nModels+1), class_data(r, nModels+2));
        end
        T_class_export = [T_class, table(mean_std_class, 'VariableNames', {'Mean_pm_Std'})];
        
        % Escribir a Excel
        writetable(T_reg_export, excel_filename, 'Sheet', 'Regresion', 'WriteRowNames', true);
        writetable(T_class_export, excel_filename, 'Sheet', 'Clasificacion', 'WriteRowNames', true);
        
        fprintf('📊 Tabla exportada a Excel: %s\n\n', excel_filename);
    end
end

%% ===================== GUARDAR RESULTADOS (OPCIONAL) =====================

if save_results
    fprintf('💾 Guardando métricas...\n');
    
    metrics = struct();
    
    % Métricas de regresión
    metrics.MAE  = struct('mean', mean_MAE, 'std', std_MAE, 'individual', MAEs);
    metrics.MSE  = struct('mean', mean_MSE, 'std', std_MSE, 'individual', MSEs);
    metrics.RMSE = struct('mean', mean_RMSE, 'std', std_RMSE, 'individual', RMSEs);
    metrics.MARD_global = struct('mean', mean_MARD_global, 'std', std_MARD_global, 'individual', MARDs_global);
    metrics.MARD_fallas = struct('mean', mean_MARD_fallas, 'std', std_MARD_fallas, 'individual', MARDs_fallas);
    metrics.R2   = struct('mean', mean_R2, 'std', std_R2, 'individual', R2s);
    
    % Métricas de clasificación
    metrics.accuracy  = struct('mean', mean_acc, 'std', std_acc, 'individual', accuracies);
    metrics.precision = struct('mean', mean_prec, 'std', std_prec, 'individual', precisions);
    metrics.recall    = struct('mean', mean_rec, 'std', std_rec, 'individual', recalls);
    metrics.f1score   = struct('mean', mean_f1, 'std', std_f1, 'individual', f1scores);
    metrics.auc       = struct('mean', mean_auc, 'std', std_auc, 'individual', aucs);
    
    % Guardar en la misma subcarpeta del bloque
    timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
    metrics_file = fullfile(output_path, sprintf('metrics_%s_%s.mat', architecture, timestamp));
    save(metrics_file, 'metrics', 'architecture', 'confusion_matrices', 'roc_X', 'roc_Y');
    
    fprintf('   ✅ Métricas guardadas en: %s\n\n', metrics_file);
end

fprintf('🏁 Análisis completado.\n');

%% ===================================================================
%  [NUEVO] COMPARACIÓN: UMBRAL FIJO vs UMBRAL ADAPTATIVO
%
%  Carga los archivos results_<arch>_fixed_*.mat y
%  results_<arch>_adaptive_*.mat, calcula todas las métricas para
%  cada uno y genera una tabla comparativa con deltas y tendencias,
%  más una figura con barras lado a lado.
%
%  Requisitos:
%    - compare_thresholds = true  (en la sección CONFIGURACIÓN)
%    - Haber ejecutado main_online_detection_v2.m en modo 'fixed' Y
%      en modo 'adaptive' para la arquitectura en compare_arch.
% ===================================================================

if compare_thresholds

    fprintf('\n');
    fprintf('╔══════════════════════════════════════════════════════════════╗\n');
    fprintf('║       COMPARACIÓN: UMBRAL FIJO vs UMBRAL ADAPTATIVO          ║\n');
    fprintf('║                  Arquitectura: %-10s                    ║\n', upper(compare_arch));
    fprintf('╚══════════════════════════════════════════════════════════════╝\n\n');

    % Carpetas donde viven los resultados de cada modo
    path_fixed    = fullfile(models_path, 'bloque_A');
    path_adaptive = fullfile(models_path, 'bloque_B');

    % ---- Resolver ruta del archivo FIJO (busca en bloque_A/) ----
    run_comparison = true;

    if isempty(file_fixed)
        d_fix = dir(fullfile(path_fixed, sprintf('results_%s_fixed_*.mat', compare_arch)));
        if isempty(d_fix)
            warning('⚠️  No se encontró ningún archivo fixed para "%s" en bloque_A/.\n   Ejecuta main_online_detection_v2.m con threshold_mode = ''fixed''.', compare_arch);
            run_comparison = false;
        else
            [~, idx_fix] = max([d_fix.datenum]);
            file_fixed_path = fullfile(path_fixed, d_fix(idx_fix).name);
            fprintf('   Auto-detectado (fixed):      %s\n', d_fix(idx_fix).name);
        end
    else
        file_fixed_path = fullfile(path_fixed, file_fixed);
        if ~isfile(file_fixed_path)
            warning('⚠️  Archivo fixed no encontrado: %s', file_fixed_path);
            run_comparison = false;
        end
    end

    % ---- Resolver ruta del archivo ADAPTATIVO (busca en bloque_B/) ----
    if run_comparison
        if isempty(file_adaptive)
            d_ada = dir(fullfile(path_adaptive, sprintf('results_%s_adaptive_*.mat', compare_arch)));
            if isempty(d_ada)
                warning('⚠️  No se encontró ningún archivo adaptive para "%s" en bloque_B/.\n   Ejecuta main_online_detection_v2.m con threshold_mode = ''adaptive''.', compare_arch);
                run_comparison = false;
            else
                [~, idx_ada] = max([d_ada.datenum]);
                file_adaptive_path = fullfile(path_adaptive, d_ada(idx_ada).name);
                fprintf('   Auto-detectado (adaptive):   %s\n', d_ada(idx_ada).name);
            end
        else
            file_adaptive_path = fullfile(path_adaptive, file_adaptive);
            if ~isfile(file_adaptive_path)
                warning('⚠️  Archivo adaptive no encontrado: %s', file_adaptive_path);
                run_comparison = false;
            end
        end
    end

    % ---- Calcular métricas y mostrar resultados ----
    if run_comparison

        fprintf('\n📂 Cargando y evaluando ambos modos...\n\n');
        data_fixed    = load(file_fixed_path);
        data_adaptive = load(file_adaptive_path);

        m_fix = compute_all_metrics(data_fixed);
        m_ada = compute_all_metrics(data_adaptive);

        % ============================================================
        % TABLA 1: MÉTRICAS DE DETECCIÓN
        % ============================================================
        fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
        fprintf('🔍 MÉTRICAS DE DETECCIÓN DE FALLOS %-10s\n',upper(compare_arch));
        fprintf('%-14s  %12s  %12s  %10s  %s\n', ...
            'Métrica', 'Fijo (mean)', 'Adapt (mean)', 'Delta', 'Tendencia');
        fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');

        det_names  = {'Accuracy',  'Precision', 'Recall',   'F1-Score', 'AUC'};
        det_fix    = [m_fix.acc,   m_fix.prec,  m_fix.rec,  m_fix.f1,  m_fix.auc];
        det_ada    = [m_ada.acc,   m_ada.prec,  m_ada.rec,  m_ada.f1,  m_ada.auc];
        det_hb     = [true, true, true, true, true];  % mayor = mejor

        for r = 1:numel(det_names)
            delta = det_ada(r) - det_fix(r);
            if det_hb(r),  sym = trend_symbol(delta);
            else,          sym = trend_symbol(-delta); end
            fprintf('%-14s  %12.4f  %12.4f  %+10.4f  %s\n', ...
                det_names{r}, det_fix(r), det_ada(r), delta, sym);
        end
        fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n');

        % ============================================================
        % TABLA 2: MÉTRICAS DE IMPUTACIÓN
        % ============================================================
        fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
        fprintf('📈 MÉTRICAS DE IMPUTACIÓN %-10s\n',upper(compare_arch));
        fprintf('%-14s  %12s  %12s  %10s  %s\n', ...
            'Métrica', 'Fijo (mean)', 'Adapt (mean)', 'Delta', 'Tendencia');
        fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');

        imp_names  = {'MAE', 'RMSE', 'MARD', 'R^2'};
        imp_fix    = [m_fix.mae, m_fix.rmse, m_fix.mard_g, m_fix.r2];
        imp_ada    = [m_ada.mae, m_ada.rmse, m_ada.mard_g, m_ada.r2];
        imp_hb     = [false, false, false, true];  % R2: mayor=mejor; resto: menor=mejor

        for r = 1:numel(imp_names)
            delta = imp_ada(r) - imp_fix(r);
            if imp_hb(r),  sym = trend_symbol(delta);
            else,          sym = trend_symbol(-delta); end
            fprintf('%-14s  %12.4f  %12.4f  %+10.4f  %s\n', ...
                imp_names{r}, imp_fix(r), imp_ada(r), delta, sym);
        end
        fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n');

        % ============================================================
        % FIGURA: Barras lado a lado (2 subplots)
        % ============================================================
        fig_cmp = figure('Name', sprintf('%s - Fijo vs Adaptativo', upper(compare_arch)), ...
                         'Position', [80, 80, 1150, 530], 'Color', 'w');

        color_fix = [0.25, 0.45, 0.75];   % azul
        color_ada = [0.15, 0.62, 0.38];   % verde

        % --- Subplot 1: Detección ---
        subplot(1, 2, 1);
        x = 1:numel(det_names);
        bar_det = [det_fix; det_ada]';
        b1 = bar(x, bar_det, 0.65);
        b1(1).FaceColor = color_fix;
        b1(2).FaceColor = color_ada;
        set(gca, 'XTick', x, 'XTickLabel', det_names, 'FontSize', 10);
        set(gca, 'TickLabelInterpreter', 'none');
        ylim([0, 1.12]);
        ylabel('Valor', 'FontSize', 11);
        title('Métricas de Detección', 'FontWeight', 'bold', 'FontSize', 12);
        legend({'Umbral Fijo', 'Umbral Adaptativo'}, 'Location', 'south', 'FontSize', 9);
        grid on; box on;

        % Etiquetas sobre las barras
        for i = 1:numel(det_names)
            text(i - 0.165, det_fix(i) + 0.022, sprintf('%.3f', det_fix(i)), ...
                'FontSize', 7.5, 'HorizontalAlignment', 'center', 'Color', color_fix*0.75);
            text(i + 0.165, det_ada(i) + 0.022, sprintf('%.3f', det_ada(i)), ...
                'FontSize', 7.5, 'HorizontalAlignment', 'center', 'Color', color_ada*0.75);
        end

        % --- Subplot 2: Imputación ---
        subplot(1, 2, 2);
        x = 1:numel(imp_names);
        bar_imp = [imp_fix; imp_ada]';
        b2 = bar(x, bar_imp, 0.65);
        b2(1).FaceColor = color_fix;
        b2(2).FaceColor = color_ada;
        set(gca, 'XTick', x, 'XTickLabel', imp_names, 'FontSize', 10);
        set(gca, 'TickLabelInterpreter', 'none');
        ylabel('Valor', 'FontSize', 11);
        title('Métricas de Imputación', 'FontWeight', 'bold', 'FontSize', 12);
        legend({'Umbral Fijo', 'Umbral Adaptativo'}, 'Location', 'northeast', 'FontSize', 9);
        grid on; box on;
        ylim([0, max([imp_fix, imp_ada]) * 1.15]);

        % Etiquetas sobre las barras
        ax2 = gca;
        y_top = ax2.YLim(2);
        for i = 1:numel(imp_names)
            offset = y_top * 0.018;
            text(i - 0.165, imp_fix(i) + offset, sprintf('%.3f', imp_fix(i)), ...
                'FontSize', 7.5, 'HorizontalAlignment', 'center', 'Color', color_fix*0.75);
            text(i + 0.165, imp_ada(i) + offset, sprintf('%.3f', imp_ada(i)), ...
                'FontSize', 7.5, 'HorizontalAlignment', 'center', 'Color', color_ada*0.75);
        end

        sgtitle(sprintf('%s  |  Fijo vs Adaptativo', upper(compare_arch)), ...
            'FontWeight', 'bold', 'FontSize', 13,'Interpreter', 'none');

        fprintf('📊 Figura comparativa generada.\n');
        fprintf('   Azul  = Umbral Fijo\n');
        fprintf('   Verde = Umbral Adaptativo\n\n');

    else
        fprintf('⚠️  Comparación omitida por falta de archivos.\n\n');
    end  % run_comparison

end  % compare_thresholds

fprintf('🏁 Script finalizado.\n');


%% ===================================================================
%  FUNCIONES LOCALES
% ===================================================================

function m = compute_all_metrics(data)
%COMPUTE_ALL_METRICS  Calcula métricas de detección e imputación
%   a partir de una estructura cargada desde results_*.mat.
%
%   Campos de salida (m):
%     Detección : acc, prec, rec, f1, auc
%     Imputación: mae, rmse, mard_g, mard_f, r2

    nM     = size(data.g_final_all, 1);
    ytrue  = data.true_f(:)';
    y_real = data.g_test_real(:)';
    y_mean_real = mean(y_real);

    % --- Métricas de detección ---
    acc_v  = zeros(nM, 1);
    prec_v = zeros(nM, 1);
    rec_v  = zeros(nM, 1);
    f1_v   = zeros(nM, 1);
    auc_v  = zeros(nM, 1);

    for i = 1:nM
        yhat = data.fallas_all(i, :);
        TP = sum(yhat == 1 & ytrue == 1);
        FP = sum(yhat == 1 & ytrue == 0);
        FN = sum(yhat == 0 & ytrue == 1);
        TN = sum(yhat == 0 & ytrue == 0);

        acc_v(i)  = (TP + TN) / (TP + TN + FP + FN);
        prec_v(i) = TP / max(TP + FP, 1);
        rec_v(i)  = TP / max(TP + FN, 1);
        f1_v(i)   = 2 * (prec_v(i) * rec_v(i)) / max(prec_v(i) + rec_v(i), eps);
        [~, ~, ~, auc_v(i)] = perfcurve(ytrue, data.error_detect_all(i, :), 1);
    end

    m.acc  = mean(acc_v);
    m.prec = mean(prec_v);
    m.rec  = mean(rec_v);
    m.f1   = mean(f1_v);
    m.auc  = mean(auc_v);

    % --- Métricas de imputación ---
    mae_v  = mean(abs(data.error_impute_all), 2);
    mse_v  = mean(data.error_impute_all .^ 2, 2);
    rmse_v = sqrt(mse_v);

    mard_g_v = zeros(nM, 1);
    mard_f_v = zeros(nM, 1);
    r2_v     = zeros(nM, 1);

    for i = 1:nM
        y_imp    = data.g_final_all(i, :);
        fallas_i = data.fallas_all(i, :);

        valid_g = (y_real >= 20);
        mard_g_v(i) = 100 * mean(abs(y_imp(valid_g) - y_real(valid_g)) ./ y_real(valid_g));

        valid_f = fallas_i & (y_real >= 20);
        if any(valid_f)
            mard_f_v(i) = 100 * mean(abs(y_imp(valid_f) - y_real(valid_f)) ./ y_real(valid_f));
        else
            mard_f_v(i) = NaN;
        end

        ss_res = sum((y_real - y_imp) .^ 2);
        ss_tot = sum((y_real - y_mean_real) .^ 2);
        r2_v(i) = 1 - ss_res / ss_tot;
    end

    m.mae    = mean(mae_v);
    m.rmse   = mean(rmse_v);
    m.mard_g = mean(mard_g_v, 'omitnan');
    m.mard_f = mean(mard_f_v, 'omitnan');
    m.r2     = mean(r2_v);
end


function sym = trend_symbol(delta)
%TREND_SYMBOL  Devuelve una cadena indicando la dirección del cambio.
%   delta > 0  →  mejora (adaptativo supera al fijo en esa dirección)
%   delta < 0  →  empeora
%   delta ≈ 0  →  sin cambio relevante

    tol = 1e-4;
    if delta > tol
        sym = '(+) MEJORA';
    elseif delta < -tol
        sym = '(-) EMPEORA';
    else
        sym = '(=) SIN CAMBIO';
    end
end
