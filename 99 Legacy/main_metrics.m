%% ========================================================================
%  MAIN_METRICS.M - Cálculo de métricas y visualización de resultados
%  
%  Este script calcula métricas de regresión (imputación) y clasificación
%  (detección de fallos) a partir de los resultados generados por 
%  main_online_detection.m
%
%  Requiere: results_<architecture>_latest.mat en data/models/
% ========================================================================

clear; clc; close all;

%% ===================== CONFIGURACIÓN =====================

% Ruta base del proyecto (MODIFICAR SEGÚN TU UBICACIÓN)
scriptPath = 'D:\Documentos\02 MROI\07 Tesis\03 Extension Tesis\CGM_Project\CGM_Project';

% Seleccionar arquitectura a evaluar
% Opciones: 'cnn_lstm', 'gru', '1dcnn', 'transformer_lstm'
architecture = 'cnn_lstm';

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
export_to_excel      = false;  % Exportar tabla a archivo Excel

% Configuración de guardado
save_results = true;  % true: guardar métricas en archivo .mat

% Rutas
models_path = fullfile(scriptPath, 'data', 'models');

%% ===================== CARGAR RESULTADOS =====================

fprintf('═══════════════════════════════════════════\n');
fprintf('📊 CÁLCULO DE MÉTRICAS - %s\n', upper(architecture));
fprintf('═══════════════════════════════════════════\n\n');

results_file = fullfile(models_path, sprintf('results_%s_latest.mat', architecture));
%results_file = 'D:\Documentos\02 MROI\07 Tesis\03 Extension Tesis\CGM_Project\CGM_Project\data\models\results_cnn_lstm_fixed_2026-01-31_14-28-56.mat';

if ~isfile(results_file)
    error('❌ No se encontró el archivo de resultados: %s\n   Ejecuta primero main_online_detection.m', results_file);
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
    
    sgtitle(sprintf('%s - ROC Curves', upper(architecture)), 'FontWeight', 'bold');
end

%% ===================== RESUMEN FINAL =====================

fprintf('═══════════════════════════════════════════\n');
fprintf('✅ RESUMEN DE MÉTRICAS - %s - ADAPTIVE\n', upper(architecture));
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
        sgtitle(sprintf('%s - Métricas de Regresión', upper(architecture)), 'FontWeight', 'bold');
        
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
        sgtitle(sprintf('%s - Métricas de Clasificación', upper(architecture)), 'FontWeight', 'bold');
    end
    
    % --- EXPORTAR A EXCEL (OPCIONAL) ---
    if export_to_excel
        excel_filename = fullfile(models_path, sprintf('metrics_table_%s.xlsx', architecture));
        
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
    
    % Guardar
    timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
    metrics_file = fullfile(models_path, sprintf('metrics_%s_%s.mat', architecture, timestamp));
    save(metrics_file, 'metrics', 'architecture', 'confusion_matrices', 'roc_X', 'roc_Y');
    
    fprintf('   ✅ Métricas guardadas en: %s\n\n', metrics_file);
end

fprintf('🏁 Análisis completado.\n');