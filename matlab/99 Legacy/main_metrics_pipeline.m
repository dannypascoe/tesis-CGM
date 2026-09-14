%% =========================================================================
%  MAIN_METRICS_PIPELINE.M - Métricas Detalladas del Pipeline Integrado
%  =========================================================================
%  Descripción: Calcula y compara métricas de todas las combinaciones
%               detector+imputador generadas por main_pipeline_integrado.m
%
%  Genera:
%    - Métricas de regresión (MAE, RMSE, MARD, R²) por combinación
%    - Métricas de clasificación (Accuracy, Precision, Recall, F1, AUC)
%    - Tabla comparativa global
%    - Comparación vs pipeline unificado (misma arquitectura detect+impute)
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

% Combinaciones del pipeline integrado
combinaciones = {
    'cnn_lstm', '1dcnn';
    'cnn_lstm', 'gru';
    'gru',      '1dcnn';
    'gru',      'gru';
};
nCombos = size(combinaciones, 1);

% Resultados del pipeline unificado para comparación (opcional)
% Si existen, se compararán contra el pipeline integrado
unified_archs = {'cnn_lstm', 'gru', '1dcnn', 'transformer_lstm'};
compare_vs_unified = true;

% Configuración de salida
export_to_excel  = false;    % Exportar tabla a Excel
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
    results_file = fullfile(models_path, ...
        sprintf('results_pipeline_%s_latest.mat', combo_name));
    
    if ~isfile(results_file)
        fprintf('⚠️  No se encontró: %s (saltando)\n', combo_name);
        metrics_all(combo).valid = false;
        metrics_all(combo).combo_name = combo_name;
        continue;
    end
    
    fprintf('📂 Cargando: %s\n', combo_name);
    load(results_file);
    
    nModels = size(g_final_all, 1);
    N = size(g_final_all, 2);
    ytrue = true_f(:)';
    y_real = g_test_real(:)';
    
    % ===== MÉTRICAS DE REGRESIÓN =====
    
    % MAE
    MAEs = mean(error_impute_all, 2);
    
    % MSE y RMSE
    MSEs = mean(error_impute_all.^2, 2);
    RMSEs = sqrt(MSEs);
    
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
    fprintf('   Imputación: MAE=%.2f±%.2f | RMSE=%.2f±%.2f | MARD=%.2f±%.2f%% | R²=%.4f±%.4f\n\n', ...
        mean(MAEs), std(MAEs), mean(RMSEs), std(RMSEs), ...
        mean(MARDs_fallas,'omitnan'), std(MARDs_fallas,'omitnan'), ...
        mean(R2s), std(R2s));
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
    fprintf('\n');
    fprintf('┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n');
    fprintf('┃             COMPARACIÓN: PIPELINE INTEGRADO vs UNIFICADO                        ┃\n');
    fprintf('┣━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━┫\n');
    fprintf('┃ %-25s ┃ Tipo  ┃ F1      ┃ Prec     ┃ MAE      ┃ RMSE       ┃\n', 'Configuración');
    fprintf('┣━━━━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━╋━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━━━┫\n');
    
    % Primero: pipelines integrados
    for combo = 1:nCombos
        m = metrics_all(combo);
        if ~m.valid, continue; end
        
        label = sprintf('%s→%s', upper(m.arch_detect), upper(m.arch_impute));
        fprintf('┃ %-25s ┃ Integ ┃ %.3f   ┃ %.3f    ┃ %.2f    ┃ %.2f      ┃\n', ...
            label, m.f1score.mean, m.precision.mean, m.MAE.mean, m.RMSE.mean);
    end
    
    fprintf('┣━━━━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━╋━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━━━┫\n');
    
    % Segundo: pipelines unificados (si existen los resultados)
    for a = 1:numel(unified_archs)
        arch = unified_archs{a};
        
        % Intentar cargar resultados del pipeline unificado
        % Buscar primero versión adaptativa, luego versión genérica
        unified_file = fullfile(models_path, sprintf('results_%s_latest.mat', arch));
        
        if ~isfile(unified_file)
            continue;
        end
        
        try
            udata = load(unified_file, 'g_final_all', 'fallas_all', ...
                'error_detect_all', 'error_impute_all', ...
                'g_test_real', 'true_f');
            
            nM = size(udata.g_final_all, 1);
            ytrue_u = udata.true_f(:)';
            
            % Calcular F1, Precision, MAE, RMSE
            u_f1s = zeros(nM, 1); u_precs = zeros(nM, 1);
            u_maes = zeros(nM, 1); u_rmses = zeros(nM, 1);
            
            for i = 1:nM
                yhat_u = udata.fallas_all(i, :);
                TP = sum(yhat_u == 1 & ytrue_u == 1);
                FP = sum(yhat_u == 1 & ytrue_u == 0);
                FN = sum(yhat_u == 0 & ytrue_u == 1);
                
                p = TP / max(TP + FP, 1);
                r = TP / max(TP + FN, 1);
                u_precs(i) = p;
                u_f1s(i) = 2 * (p * r) / max(p + r, eps);
                
                u_maes(i) = mean(udata.error_impute_all(i, :));
                u_rmses(i) = sqrt(mean(udata.error_impute_all(i, :).^2));
            end
            
            fprintf('┃ %-25s ┃ Unif  ┃ %.3f   ┃ %.3f    ┃ %.2f    ┃ %.2f      ┃\n', ...
                upper(arch), mean(u_f1s), mean(u_precs), mean(u_maes), mean(u_rmses));
            
        catch
            % Si el formato no es compatible, saltar
            continue;
        end
    end
    
    fprintf('┗━━━━━━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━┻━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━━━┛\n');
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
        title('Métricas de Detección por Combinación');
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
        
        sgtitle('Métricas de Imputación - Pipeline Integrado', ...
            'FontWeight', 'bold', 'FontSize', 14);
    end
end

%% ===================== EXPORTAR A EXCEL (OPCIONAL) =====================

if export_to_excel
    fprintf('\n📊 Exportando a Excel...\n');
    
    excel_file = fullfile(models_path, 'pipeline_comparison.xlsx');
    
    % Construir tabla
    valid_combos = find([metrics_all.valid]);
    nValid = numel(valid_combos);
    
    T_data = table();
    for idx = 1:nValid
        c = valid_combos(idx);
        m = metrics_all(c);
        
        row = table();
        row.Detector  = {upper(m.arch_detect)};
        row.Imputador = {upper(m.arch_impute)};
        row.F1_Score  = m.f1score.mean;
        row.F1_Std    = m.f1score.std;
        row.Precision = m.precision.mean;
        row.Recall    = m.recall.mean;
        row.AUC       = m.auc.mean;
        row.MAE       = m.MAE.mean;
        row.MAE_Std   = m.MAE.std;
        row.RMSE      = m.RMSE.mean;
        row.MARD_pct  = m.MARD_fallas.mean;
        row.R2        = m.R2.mean;
        
        T_data = [T_data; row];
    end
    
    writetable(T_data, excel_file, 'Sheet', 'Pipeline_Integrado');
    fprintf('   ✅ Exportado a: %s\n', excel_file);
end

%% ===================== GUARDAR MÉTRICAS =====================

metrics_file = fullfile(models_path, 'metrics_pipeline_all.mat');
save(metrics_file, 'metrics_all', 'combinaciones');
fprintf('\n💾 Métricas guardadas en: %s\n', metrics_file);
fprintf('🏁 Análisis completado.\n');
