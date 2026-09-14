%% =========================================================================
%  MAIN_METRICS_PIPELINE_V3.M - Métricas Detalladas del Pipeline Desacoplado
%  =========================================================================
%  Descripción: Calcula y compara métricas de todas las combinaciones
%               detector+imputador del pipeline desacoplado (main_pipeline_integrado_v2.m)
%
%  Genera:
%    - Métricas de regresión (MAE, RMSE, MARD, R²) por combinación
%    - Métricas de clasificación (Accuracy, Precision, Recall, F1, AUC)
%    - Tabla comparativa global
%    - Comparación vs esquema unificado (misma arquitectura detect+impute)
%    - Exportación opcional a Excel
%
%  Requiere: results_pipeline_*_latest.mat en data/models/
%  =========================================================================

clear; clc; close all;

%% ===================== CONFIGURACIÓN =====================

% Ruta base del proyecto (MODIFICAR SEGÚN TU UBICACIÓN)
scriptPath = 'D:\Documentos\02 MROI\07 Tesis\03 Extension Tesis\CGM_Project\CGM_Project';

% Rutas
models_path = fullfile(scriptPath, 'data', 'models');

% -----------------------------------------------------------------------
% CARPETA DE ENTRADA Y SALIDA: data/models/bloque_C/
% Este script lee los results_pipeline_*_latest.mat generados por
% main_pipeline_integrado_v3.m y guarda metricas en la misma carpeta.
% -----------------------------------------------------------------------
output_path = fullfile(models_path, 'bloque_C');
if ~exist(output_path, 'dir')
    error('No se encontro bloque_C/. Ejecuta primero main_pipeline_integrado_v2.m');
end

% Combinaciones del pipeline desacoplado (4 x 4 = 16 exhaustivas)
architectures = {'cnn_lstm', 'gru', '1dcnn', 'transformer_lstm'};
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

% Comparacion desacoplado vs. esquema unificado (opcional)
% El esquema unificado se extrae del DIAGONAL de la tabla de 16 combinaciones
% (misma arquitectura para detección e imputación), garantizando comparación válida.
compare_vs_unified = true;

% Configuración de salida
export_to_excel  = true;     % Exportar tabla a Excel (pipeline_comparison.xlsx en bloque_C/)
show_plots       = true;    % Mostrar gráficas comparativas

%% ===================== CARGAR Y CALCULAR MÉTRICAS =====================

fprintf('╔═══════════════════════════════════════════════════════════════╗\n');
fprintf('║     MÉTRICAS DEL PIPELINE INTEGRADO DETECTOR+IMPUTADOR      ║\n');
fprintf('╚═══════════════════════════════════════════════════════════════╝\n\n');

% Estructura para almacenar todas las métricas
metrics_all = struct();

for combo = 1:nCombos
    arch_det = combinaciones{combo, 1};
    arch_imp = combinaciones{combo, 2};
    combo_name = sprintf('%s_det_%s_imp', arch_det, arch_imp);
    
    % Cargar resultados
    results_file = fullfile(output_path, ...
        sprintf('results_pipeline_%s_latest.mat', combo_name));
    
    if ~isfile(results_file)
        fprintf('⚠️  No se encontró: %s (saltando)\n', combo_name);
        metrics_all(combo).valid = false;
        metrics_all(combo).combo_name = combo_name;
        continue;
    end
    
    fprintf('📂 Cargando: %s\n', combo_name);
    load(results_file);

    % --- Verificación de compatibilidad (definiciones D1/D3) ---
    if ~exist('error_impute_pred_all', 'var')
        error(['El archivo %s no contiene ''error_impute_pred_all''.\n' ...
               'Re-ejecuta main_pipeline_integrado_v2.m (versión con D1/D2/D3) ' ...
               'para regenerar los results_pipeline_*_latest.mat.'], combo_name);
    end

    nModels = size(g_final_all, 1);
    N = size(g_final_all, 2);
    ytrue = true_f(:)';
    y_real = g_test_real(:)';
    true_fault_idx = (ytrue == 1);   % muestras con fallo REAL (para D3)

    % ===== MÉTRICAS DE REGRESIÓN =====
    %  Métrica PRINCIPAL = D1: error de predicción del imputador en todas las
    %  muestras (comparable con el paper y con el Bloque B).

    % MAE / MSE / RMSE  [D1]
    MAEs  = mean(error_impute_pred_all, 2);
    MSEs  = mean(error_impute_pred_all.^2, 2);
    RMSEs = sqrt(MSEs);

    % --- [D3] MAE/RMSE de la SALIDA, solo en muestras con fallo real ---
    MAEs_corrupt  = zeros(nModels, 1);
    RMSEs_corrupt = zeros(nModels, 1);
    for i = 1:nModels
        e_corr = abs(g_final_all(i, true_fault_idx) - y_real(true_fault_idx));
        MAEs_corrupt(i)  = mean(e_corr);
        RMSEs_corrupt(i) = sqrt(mean(e_corr.^2));
    end

    % --- [D2] MAE de la salida en todas las muestras (solo referencia) ---
    MAEs_out = mean(error_impute_out_all, 2);


    % MARD (global y solo en fallas)
    MARDs_global = zeros(nModels, 1);
    MARDs_fallas = zeros(nModels, 1);
    
    for i = 1:nModels
        y_imp = g_final_all(i, :);
        fallas_i = fallas_all(i, :);
        
        valid_global = (y_real >= 20);
        rel_err = abs(y_imp(valid_global) - y_real(valid_global)) ./ y_real(valid_global);
        MARDs_global(i) = 100 * mean(rel_err);
        
        valid_fallas = fallas_i & (y_real >= 20);
        if any(valid_fallas)
            rel_err_f = abs(y_imp(valid_fallas) - y_real(valid_fallas)) ./ y_real(valid_fallas);
            MARDs_fallas(i) = 100 * mean(rel_err_f);
        else
            MARDs_fallas(i) = NaN;
        end
    end
    
    % R²
    R2s = zeros(nModels, 1);
    y_mean = mean(y_real);
    for i = 1:nModels
        y_pred = g_final_all(i, :);
        ss_res = sum((y_real - y_pred).^2);
        ss_tot = sum((y_real - y_mean).^2);
        R2s(i) = 1 - (ss_res / ss_tot);
    end
    
    % ===== MÉTRICAS DE CLASIFICACIÓN =====
    
    accuracies = zeros(nModels, 1);
    precisions = zeros(nModels, 1);
    recalls    = zeros(nModels, 1);
    f1scores   = zeros(nModels, 1);
    aucs       = zeros(nModels, 1);
    
    for i = 1:nModels
        yhat = fallas_all(i, :);
        
        TP = sum(yhat == 1 & ytrue == 1);
        FP = sum(yhat == 1 & ytrue == 0);
        FN = sum(yhat == 0 & ytrue == 1);
        TN = sum(yhat == 0 & ytrue == 0);
        
        accuracies(i) = (TP + TN) / (TP + TN + FP + FN);
        precisions(i) = TP / max(TP + FP, 1);
        recalls(i)    = TP / max(TP + FN, 1);
        f1scores(i)   = 2 * (precisions(i) * recalls(i)) / max(precisions(i) + recalls(i), eps);
        
        % AUC-ROC
        [~, ~, ~, AUROC] = perfcurve(ytrue, error_detect_all(i, :), 1);
        aucs(i) = AUROC;
    end
    
    % ===== ALMACENAR =====
    metrics_all(combo).valid       = true;
    metrics_all(combo).combo_name  = combo_name;
    metrics_all(combo).arch_detect = arch_det;
    metrics_all(combo).arch_impute = arch_imp;
    metrics_all(combo).nModels     = nModels;
    
    % Regresión
    metrics_all(combo).MAE   = struct('mean', mean(MAEs), 'std', std(MAEs), 'ind', MAEs);
    metrics_all(combo).MSE   = struct('mean', mean(MSEs), 'std', std(MSEs), 'ind', MSEs);
    metrics_all(combo).RMSE  = struct('mean', mean(RMSEs), 'std', std(RMSEs), 'ind', RMSEs);
    metrics_all(combo).MARD_global = struct('mean', mean(MARDs_global,'omitnan'), 'std', std(MARDs_global,'omitnan'), 'ind', MARDs_global);
    metrics_all(combo).MARD_fallas = struct('mean', mean(MARDs_fallas,'omitnan'), 'std', std(MARDs_fallas,'omitnan'), 'ind', MARDs_fallas);
    metrics_all(combo).R2    = struct('mean', mean(R2s), 'std', std(R2s), 'ind', R2s);
    % [D3] Reconstrucción de la salida solo en fallos reales
    metrics_all(combo).MAE_corrupt  = struct('mean', mean(MAEs_corrupt),  'std', std(MAEs_corrupt),  'ind', MAEs_corrupt);
    metrics_all(combo).RMSE_corrupt = struct('mean', mean(RMSEs_corrupt), 'std', std(RMSEs_corrupt), 'ind', RMSEs_corrupt);
    % [D2] Salida en todas las muestras (referencia)
    metrics_all(combo).MAE_out = struct('mean', mean(MAEs_out), 'std', std(MAEs_out), 'ind', MAEs_out);
    
    % Clasificación
    metrics_all(combo).accuracy  = struct('mean', mean(accuracies), 'std', std(accuracies), 'ind', accuracies);
    metrics_all(combo).precision = struct('mean', mean(precisions), 'std', std(precisions), 'ind', precisions);
    metrics_all(combo).recall    = struct('mean', mean(recalls), 'std', std(recalls), 'ind', recalls);
    metrics_all(combo).f1score   = struct('mean', mean(f1scores), 'std', std(f1scores), 'ind', f1scores);
    metrics_all(combo).auc       = struct('mean', mean(aucs), 'std', std(aucs), 'ind', aucs);
    
    % Imprimir resumen individual
    fprintf('   Detección:  Acc=%.3f±%.3f | Prec=%.3f±%.3f | Rec=%.3f±%.3f | F1=%.3f±%.3f | AUC=%.3f±%.3f\n', ...
        mean(accuracies), std(accuracies), mean(precisions), std(precisions), ...
        mean(recalls), std(recalls), mean(f1scores), std(f1scores), ...
        mean(aucs), std(aucs));
    fprintf('   Imputación D1 (pred., todas):   MAE=%.2f±%.2f | RMSE=%.2f±%.2f | MARD=%.2f±%.2f%% | R²=%.4f±%.4f\n', ...
        mean(MAEs), std(MAEs), mean(RMSEs), std(RMSEs), ...
        mean(MARDs_fallas,'omitnan'), std(MARDs_fallas,'omitnan'), ...
        mean(R2s), std(R2s));
    fprintf('   Imputación D3 (salida, fallos): MAE=%.2f±%.2f | RMSE=%.2f±%.2f mg/dL\n\n', ...
        mean(MAEs_corrupt), std(MAEs_corrupt), mean(RMSEs_corrupt), std(RMSEs_corrupt));
end

%% ===================== TABLA COMPARATIVA GLOBAL =====================

fprintf('\n');
fprintf('┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n');
fprintf('┃                          TABLA COMPARATIVA - PIPELINE INTEGRADO                    ┃\n');
fprintf('┣━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━┫\n');
fprintf('┃ %-25s ┃ %-12s ┃ %-12s ┃ %-12s ┃ %-8s ┃\n', ...
    'Detector → Imputador', 'F1-Score', 'Precision', 'MAE(mg/dL)', 'R²');
fprintf('┣━━━━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━╋━━━━━━━━━━┫\n');

for combo = 1:nCombos
    m = metrics_all(combo);
    if ~m.valid, continue; end
    
    label = sprintf('%s → %s', upper(m.arch_detect), upper(m.arch_impute));
    
    fprintf('┃ %-25s ┃ %.3f±%.3f  ┃ %.3f±%.3f  ┃ %.2f±%.2f   ┃ %.4f  ┃\n', ...
        label, ...
        m.f1score.mean, m.f1score.std, ...
        m.precision.mean, m.precision.std, ...
        m.MAE.mean, m.MAE.std, ...
        m.R2.mean);
end

fprintf('┗━━━━━━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━┻━━━━━━━━━━┛\n');

%% ===================== COMPARACIÓN VS PIPELINE UNIFICADO =====================

if compare_vs_unified
    % =========================================================================
    %  COMPARACIÓN: PIPELINE INTEGRADO vs PIPELINE UNIFICADO
    %
    %  NOTA METODOLÓGICA:
    %  El esquema unificado (misma arquitectura para detectar e imputar)
    %  se obtiene directamente del DIAGONAL de la tabla de 16 combinaciones,
    %  es decir, las entradas donde arch_detect == arch_impute.
    %
    %  Esto garantiza que AMBAS métricas (desacoplado y unificado) se calculan
    %  con la MISMA definición de error (D1: error de predicción del imputador
    %  en todas las muestras) y sobre el mismo conjunto, haciendo la comparación
    %  válida y consistente.
    %
    %  El esquema unificado se toma de la DIAGONAL (arch_det == arch_imp) de esta
    %  misma corrida, no de los results_<arch>_latest.mat del Bloque B, para
    %  evitar cualquier mezcla de fuentes. (Con la definición D1, el MAE de la
    %  diagonal aquí coincide con el del Bloque B salvo variación numérica menor.)
    % =========================================================================

    fprintf('\n');
    fprintf('┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n');
    fprintf('┃      COMPARACIÓN: PIPELINE DESACOPLADO vs ESQUEMA UNIFICADO (diagonal)             ┃\n');
    fprintf('┃  Unificado = misma arquitectura para detección e imputación (diagonal de las 16 combos)   ┃\n');
    fprintf('┃  Ambas métricas calculadas sobre el mismo conjunto → comparación válida                  ┃\n');
    fprintf('┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫\n');
    fprintf('┃ %-34s ┃ %-16s ┃ %-14s ┃ %-6s ┃\n', ...
        'Configuración', 'Tipo', 'MAE (mg/dL)', 'R²');
    fprintf('┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━╋━━━━━━━━┫\n');

    % Identificar arquitecturas únicas en el orden original
    arch_display_order = architectures;

    for a = 1:numel(arch_display_order)
        arch = arch_display_order{a};

        % --- Fila UNIFICADO: extraída del diagonal (arch_det == arch_imp == arch) ---
        diag_idx = find(strcmp({metrics_all.arch_detect}, arch) & ...
                        strcmp({metrics_all.arch_impute}, arch) & ...
                        [metrics_all.valid]);

        if ~isempty(diag_idx)
            md = metrics_all(diag_idx);
            label_u = sprintf('%s → %s', upper(arch), upper(arch));
            fprintf('┃ %-34s ┃ %-16s ┃ %.2f ± %.2f      ┃ %.4f ┃\n', ...
                label_u, 'Esq. Unificado', md.MAE.mean, md.MAE.std, md.R2.mean);
        end

        % --- Filas ESPECIALIZADAS: misma arquitectura como detector, distintos imputadores ---
        for combo = 1:nCombos
            m = metrics_all(combo);
            if ~m.valid, continue; end
            if ~strcmp(m.arch_detect, arch), continue; end
            if strcmp(m.arch_impute, arch), continue; end  % saltar el diagonal (ya impreso)

            label_e = sprintf('%s → %s', upper(m.arch_detect), upper(m.arch_impute));
            fprintf('┃ %-34s ┃ %-16s ┃ %.2f ± %.2f      ┃ %.4f ┃\n', ...
                label_e, 'Desacoplado', m.MAE.mean, m.MAE.std, m.R2.mean);
        end

        fprintf('┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━╋━━━━━━━━┫\n');
    end

    fprintf('┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━┻━━━━━━━━┛\n');

    % --- Resumen ejecutivo: mejor desacoplado vs esquema unificado por detector ---
    fprintf('\n  RESUMEN — Mejor desacoplado vs. esquema unificado por detector:\n');
    fprintf('  %-20s %-12s %-12s %-12s\n', 'Detector', 'MAE Unif.', 'MAE Mejor Esp.', 'Mejora');
    fprintf('  %s\n', repmat('-', 1, 58));

    for a = 1:numel(arch_display_order)
        arch = arch_display_order{a};

        diag_idx = find(strcmp({metrics_all.arch_detect}, arch) & ...
                        strcmp({metrics_all.arch_impute}, arch) & ...
                        [metrics_all.valid]);
        if isempty(diag_idx), continue; end
        mae_unif = metrics_all(diag_idx).MAE.mean;

        % Buscar mínimo MAE fuera del diagonal con el mismo detector
        best_mae_esp = Inf;
        best_imp     = '';
        for combo = 1:nCombos
            m = metrics_all(combo);
            if ~m.valid, continue; end
            if ~strcmp(m.arch_detect, arch), continue; end
            if strcmp(m.arch_impute, arch), continue; end
            if m.MAE.mean < best_mae_esp
                best_mae_esp = m.MAE.mean;
                best_imp     = upper(m.arch_impute);
            end
        end

        mejora_pct = 100 * (mae_unif - best_mae_esp) / mae_unif;
        fprintf('  %-20s %-12.4f %-12.4f %.1f%% (mejor imp: %s)\n', ...
            upper(arch), mae_unif, best_mae_esp, mejora_pct, best_imp);
    end
end

%% ===================== GRÁFICAS COMPARATIVAS =====================

if show_plots
    
    % Recopilar datos válidos para gráficas
    valid_combos = find([metrics_all.valid]);
    nValid = numel(valid_combos);
    
    if nValid == 0
        fprintf('⚠️  No hay combinaciones válidas para graficar.\n');
    else
        combo_labels = cell(nValid, 1);
        f1_means = zeros(nValid, 1); f1_stds = zeros(nValid, 1);
        prec_means = zeros(nValid, 1); prec_stds = zeros(nValid, 1);
        rec_means = zeros(nValid, 1); rec_stds = zeros(nValid, 1);
        mae_means = zeros(nValid, 1); mae_stds = zeros(nValid, 1);
        rmse_means = zeros(nValid, 1); rmse_stds = zeros(nValid, 1);
        
        for idx = 1:nValid
            c = valid_combos(idx);
            m = metrics_all(c);
            combo_labels{idx} = sprintf('%s→%s', upper(m.arch_detect), upper(m.arch_impute));
            f1_means(idx)   = m.f1score.mean;   f1_stds(idx)   = m.f1score.std;
            prec_means(idx) = m.precision.mean;  prec_stds(idx) = m.precision.std;
            rec_means(idx)  = m.recall.mean;     rec_stds(idx)  = m.recall.std;
            mae_means(idx)  = m.MAE.mean;        mae_stds(idx)  = m.MAE.std;
            rmse_means(idx) = m.RMSE.mean;       rmse_stds(idx) = m.RMSE.std;
        end
        
        % --- Figura 1: Métricas de Detección ---
        figure('Name', 'Pipeline - Métricas de Detección', ...
               'Position', [50, 400, 900, 400], 'Color', 'w');
        
        x = 1:nValid;
        bar_data = [f1_means, prec_means, rec_means];
        bar_err  = [f1_stds, prec_stds, rec_stds];
        
        b = bar(x, bar_data, 'grouped');
        hold on;
        
        % Colores
        b(1).FaceColor = [0.20, 0.47, 0.73];  % F1 - azul
        b(2).FaceColor = [0.30, 0.69, 0.29];  % Precision - verde
        b(3).FaceColor = [0.89, 0.47, 0.16];  % Recall - naranja
        
        % Error bars
        for j = 1:3
            x_bar = b(j).XEndPoints;
            errorbar(x_bar, bar_data(:,j), bar_err(:,j), '.k', 'LineWidth', 1);
        end
        
        set(gca, 'XTick', x, 'XTickLabel', combo_labels, 'XTickLabelRotation', 15);
        ylabel('Score');
        title('Métricas de Detección — Pipeline desacoplado (16 combinaciones)', 'Interpreter', 'none');
        legend({'F1-Score', 'Precision', 'Recall'}, 'Location', 'best');
        grid on;
        ylim([0, 1.15]);
        hold off;
        
        % --- Figura 2: Métricas de Imputación ---
        figure('Name', 'Pipeline - Métricas de Imputación', ...
               'Position', [50, 50, 900, 400], 'Color', 'w');
        
        subplot(1, 2, 1);
        bar(x, mae_means, 'FaceColor', [0.55, 0.25, 0.64]);
        hold on;
        errorbar(x, mae_means, mae_stds, '.k', 'LineWidth', 1.2);
        set(gca, 'XTick', x, 'XTickLabel', combo_labels, 'XTickLabelRotation', 15);
        ylabel('MAE (mg/dL)');
        title('MAE por Combinación');
        grid on;
        hold off;
        
        subplot(1, 2, 2);
        bar(x, rmse_means, 'FaceColor', [0.80, 0.36, 0.36]);
        hold on;
        errorbar(x, rmse_means, rmse_stds, '.k', 'LineWidth', 1.2);
        set(gca, 'XTick', x, 'XTickLabel', combo_labels, 'XTickLabelRotation', 15);
        ylabel('RMSE (mg/dL)');
        title('RMSE por Combinación');
        grid on;
        hold off;
        
        sgtitle('Métricas de Imputación — Pipeline desacoplado (16 combinaciones)', ...
            'FontWeight', 'bold', 'FontSize', 14);
    end
end

%% ===================== EXPORTAR A EXCEL (OPCIONAL) =====================

if export_to_excel
    fprintf('\n📊 Exportando a Excel...\n');
    
    excel_file = fullfile(output_path, 'pipeline_comparison.xlsx');
    valid_combos = find([metrics_all.valid]);
    nValid = numel(valid_combos);
    arch_labels = {'CNN-LSTM', 'GRU', '1D-CNN', 'Transformer-LSTM'};
    arch_keys   = {'cnn_lstm', 'gru', '1dcnn', 'transformer_lstm'};
    nA = numel(arch_keys);

    % ── Hoja 1: Lista completa (16 filas) ──────────────────────────────
    T_data = table();
    for idx = 1:nValid
        c = valid_combos(idx);
        m = metrics_all(c);
        row = table();
        row.Detector        = {upper(m.arch_detect)};
        row.Imputador       = {upper(m.arch_impute)};
        row.F1_mean         = m.f1score.mean;
        row.F1_std          = m.f1score.std;
        row.Precision_mean  = m.precision.mean;
        row.Precision_std   = m.precision.std;
        row.Recall_mean     = m.recall.mean;
        row.Recall_std      = m.recall.std;
        row.AUC_mean        = m.auc.mean;
        % --- D1 (principal, comparable con el paper) ---
        row.MAE_mean        = m.MAE.mean;
        row.MAE_std         = m.MAE.std;
        row.RMSE_mean       = m.RMSE.mean;
        row.RMSE_std        = m.RMSE.std;
        row.MARD_mean       = m.MARD_global.mean;
        row.R2_mean         = m.R2.mean;
        % --- D3 (solo fallos reales, secundaria) ---
        row.MAE_corrupt_mean  = m.MAE_corrupt.mean;
        row.MAE_corrupt_std   = m.MAE_corrupt.std;
        row.RMSE_corrupt_mean = m.RMSE_corrupt.mean;
        T_data = [T_data; row];
    end
    writetable(T_data, excel_file, 'Sheet', 'Lista_16_combinaciones');

    % ── Hoja 2: Tabla 4x4 — F1 (mean±std) ────────────────────────────
    header_row = [{'Detector \ Imputador'}, arch_labels];
    f1_table   = cell(nA + 1, nA + 1);
    f1_table(1,:) = header_row;
    for r = 1:nA
        f1_table{r+1, 1} = arch_labels{r};
        for cc = 1:nA
            idx_c = find(strcmp({metrics_all.arch_detect}, arch_keys{r}) & ...
                         strcmp({metrics_all.arch_impute}, arch_keys{cc}) & ...
                         [metrics_all.valid]);
            if ~isempty(idx_c)
                m = metrics_all(idx_c);
                f1_table{r+1, cc+1} = sprintf('%.3f ± %.3f', m.f1score.mean, m.f1score.std);
            end
        end
    end
    writecell(f1_table, excel_file, 'Sheet', 'Tabla4x4_F1');

    % ── Hoja 3: Tabla 4x4 — MAE (mean±std) ───────────────────────────
    mae_table = cell(nA + 1, nA + 1);
    mae_table(1,:) = header_row;
    for r = 1:nA
        mae_table{r+1, 1} = arch_labels{r};
        for cc = 1:nA
            idx_c = find(strcmp({metrics_all.arch_detect}, arch_keys{r}) & ...
                         strcmp({metrics_all.arch_impute}, arch_keys{cc}) & ...
                         [metrics_all.valid]);
            if ~isempty(idx_c)
                m = metrics_all(idx_c);
                mae_table{r+1, cc+1} = sprintf('%.2f ± %.2f', m.MAE.mean, m.MAE.std);
            end
        end
    end
    writecell(mae_table, excel_file, 'Sheet', 'Tabla4x4_MAE');

    % ── Hoja 3b: Tabla 4x4 — MAE D3 (salida, solo fallos reales) ─────
    mae3_table = cell(nA + 1, nA + 1);
    mae3_table(1,:) = header_row;
    for r = 1:nA
        mae3_table{r+1, 1} = arch_labels{r};
        for cc = 1:nA
            idx_c = find(strcmp({metrics_all.arch_detect}, arch_keys{r}) & ...
                         strcmp({metrics_all.arch_impute}, arch_keys{cc}) & ...
                         [metrics_all.valid]);
            if ~isempty(idx_c)
                m = metrics_all(idx_c);
                mae3_table{r+1, cc+1} = sprintf('%.2f ± %.2f', m.MAE_corrupt.mean, m.MAE_corrupt.std);
            end
        end
    end
    writecell(mae3_table, excel_file, 'Sheet', 'Tabla4x4_MAE_D3');

    % ── Hoja 4: Co-optimalidad — comparación directa ─────────────────
    co_labels  = {'Configuracion', 'Tipo', 'F1_mean', 'F1_std', ...
                  'MAE_D1_mean', 'MAE_D1_std', 'RMSE_D1_mean', 'R2_mean', ...
                  'MAE_D3_mean', 'MAE_D3_std'};
    co_data    = {};
    for a = 1:nA
        % Diagonal (esquema unificado)
        diag_idx = find(strcmp({metrics_all.arch_detect}, arch_keys{a}) & ...
                        strcmp({metrics_all.arch_impute}, arch_keys{a}) & ...
                        [metrics_all.valid]);
        if ~isempty(diag_idx)
            md = metrics_all(diag_idx);
            co_data(end+1,:) = {sprintf('%s -> %s', arch_labels{a}, arch_labels{a}), ...
                'Esq. Unificado', md.f1score.mean, md.f1score.std, ...
                md.MAE.mean, md.MAE.std, md.RMSE.mean, md.R2.mean, ...
                md.MAE_corrupt.mean, md.MAE_corrupt.std};
        end
        % Combinaciones desacopladas con ese detector
        for b = 1:nA
            if a == b, continue; end
            idx_c = find(strcmp({metrics_all.arch_detect}, arch_keys{a}) & ...
                         strcmp({metrics_all.arch_impute}, arch_keys{b}) & ...
                         [metrics_all.valid]);
            if ~isempty(idx_c)
                m = metrics_all(idx_c);
                co_data(end+1,:) = {sprintf('%s -> %s', arch_labels{a}, arch_labels{b}), ...
                    'Desacoplado', m.f1score.mean, m.f1score.std, ...
                    m.MAE.mean, m.MAE.std, m.RMSE.mean, m.R2.mean, ...
                    m.MAE_corrupt.mean, m.MAE_corrupt.std};
            end
        end
    end
    writecell([co_labels; co_data], excel_file, 'Sheet', 'Co_optimalidad');

    fprintf('   ✅ Excel guardado en: %s\n', excel_file);
    fprintf('      Hojas: Lista_16_combinaciones | Tabla4x4_F1 | Tabla4x4_MAE | Tabla4x4_MAE_D3 | Co_optimalidad\n');
end

%% ===================== GUARDAR MÉTRICAS =====================

metrics_file = fullfile(output_path, 'metrics_pipeline_all.mat');
save(metrics_file, 'metrics_all', 'combinaciones');
fprintf('\n💾 Métricas guardadas en: %s\n', metrics_file);
fprintf('🏁 Análisis completado.\n');