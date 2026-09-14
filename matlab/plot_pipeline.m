%% =========================================================================
%  PLOT_PIPELINE_TESIS.M - Gráficas para Tesis del Pipeline Integrado
%  =========================================================================
%  Genera 3 figuras de calidad para tesis:
%    1. Heatmap 4×4 de MAE (Detector × Imputador)
%    2. Barras agrupadas: Pipeline Integrado vs Unificado
%    3. Señal temporal del mejor pipeline (GRU → 1D-CNN)
%
%  Requiere: resultados del pipeline integrado y unificado en data/models/
%  =========================================================================

clear; clc; close all;

%% ===================== CONFIGURACIÓN =====================

% Ruta base del proyecto (MODIFICAR SEGÚN TU UBICACIÓN)
scriptPath = 'D:\Documentos\02 MROI\07 Tesis\03 Extension Tesis\CGM_Project\CGM_Project';
models_path = fullfile(scriptPath, 'data', 'models');

% Arquitecturas
architectures = {'cnn_lstm', 'gru', '1dcnn', 'transformer_lstm'};
arch_labels   = {'CNN-LSTM', 'GRU', '1D-CNN', 'Trans-LSTM'};
nArch = numel(architectures);

% Guardar figuras como .png (opcional)
save_figures = false;  % Cambiar a true para exportar
fig_path = fullfile(scriptPath, 'figures');
if save_figures && ~exist(fig_path, 'dir')
    mkdir(fig_path);
end

%% ===================== CARGAR DATOS =====================

fprintf('📂 Cargando resultados del pipeline integrado...\n');

% Matrices 4×4 para métricas
MAE_matrix  = NaN(nArch, nArch);
RMSE_matrix = NaN(nArch, nArch);
R2_matrix   = NaN(nArch, nArch);
F1_matrix   = NaN(nArch, nArch);

for d = 1:nArch
    for im = 1:nArch
        combo_name = sprintf('%s_det_%s_imp', architectures{d}, architectures{im});
        results_file = fullfile(models_path, ...
            sprintf('results_pipeline_%s_latest.mat', combo_name));
        
        if ~isfile(results_file)
            fprintf('   ⚠️  No encontrado: %s\n', combo_name);
            continue;
        end
        
        data = load(results_file);
        nModels = size(data.g_final_all, 1);
        y_real = data.g_test_real(:)';
        ytrue = data.true_f(:)';
        
        % MAE promedio
        MAEs = mean(data.error_impute_all, 2);
        MAE_matrix(d, im) = mean(MAEs);
        
        % RMSE promedio
        RMSEs = sqrt(mean(data.error_impute_all.^2, 2));
        RMSE_matrix(d, im) = mean(RMSEs);
        
        % R² promedio
        y_mean = mean(y_real);
        R2s = zeros(nModels, 1);
        for i = 1:nModels
            ss_res = sum((y_real - data.g_final_all(i,:)).^2);
            ss_tot = sum((y_real - y_mean).^2);
            R2s(i) = 1 - (ss_res / ss_tot);
        end
        R2_matrix(d, im) = mean(R2s);
        
        % F1 promedio
        f1s = zeros(nModels, 1);
        for i = 1:nModels
            yhat = data.fallas_all(i,:);
            TP = sum(yhat == 1 & ytrue == 1);
            FP = sum(yhat == 1 & ytrue == 0);
            FN = sum(yhat == 0 & ytrue == 1);
            p = TP / max(TP + FP, 1);
            r = TP / max(TP + FN, 1);
            f1s(i) = 2 * (p * r) / max(p + r, eps);
        end
        F1_matrix(d, im) = mean(f1s);
    end
end

fprintf('   ✅ Matrices 4×4 construidas\n');

% Cargar datos del pipeline unificado
fprintf('📂 Cargando resultados unificados...\n');

unified_MAE  = NaN(nArch, 1);
unified_RMSE = NaN(nArch, 1);
unified_F1   = NaN(nArch, 1);
unified_R2   = NaN(nArch, 1);

for a = 1:nArch
    unified_file = fullfile(models_path, ...
        sprintf('results_%s_latest.mat', architectures{a}));
    
    if ~isfile(unified_file)
        fprintf('   ⚠️  No encontrado unificado: %s\n', architectures{a});
        continue;
    end
    
    udata = load(unified_file);
    nM = size(udata.g_final_all, 1);
    y_real_u = udata.g_test_real(:)';
    ytrue_u = udata.true_f(:)';
    
    unified_MAE(a) = mean(mean(udata.error_impute_all, 2));
    unified_RMSE(a) = mean(sqrt(mean(udata.error_impute_all.^2, 2)));
    
    y_mean_u = mean(y_real_u);
    r2s = zeros(nM, 1);
    f1s = zeros(nM, 1);
    for i = 1:nM
        ss_res = sum((y_real_u - udata.g_final_all(i,:)).^2);
        ss_tot = sum((y_real_u - y_mean_u).^2);
        r2s(i) = 1 - (ss_res / ss_tot);
        
        yhat = udata.fallas_all(i,:);
        TP = sum(yhat == 1 & ytrue_u == 1);
        FP = sum(yhat == 1 & ytrue_u == 0);
        FN = sum(yhat == 0 & ytrue_u == 1);
        p = TP / max(TP + FP, 1);
        r = TP / max(TP + FN, 1);
        f1s(i) = 2 * (p * r) / max(p + r, eps);
    end
    unified_R2(a) = mean(r2s);
    unified_F1(a) = mean(f1s);
end

fprintf('   ✅ Datos unificados cargados\n\n');

%% =========================================================================
%  FIGURA 1: HEATMAP 4×4 DE MAE (Detector × Imputador)
%  =========================================================================

fig1 = figure('Name', 'Heatmap MAE', 'Position', [50, 400, 620, 500], 'Color', 'w');

imagesc(MAE_matrix);
colormap(flipud(summer));   % Verde claro = bajo (bueno), oscuro = alto
cb = colorbar;
cb.Label.String = 'MAE (mg/dL)';
cb.Label.FontSize = 11;

% Etiquetas
set(gca, 'XTick', 1:nArch, 'XTickLabel', arch_labels, 'XTickLabelRotation', 0, ...
         'YTick', 1:nArch, 'YTickLabel', arch_labels, ...
         'FontSize', 11, 'TickLength', [0 0]);

xlabel('Imputador', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Detector', 'FontSize', 12, 'FontWeight', 'bold');
title('MAE por combinación Detector × Imputador', 'FontSize', 13);

% Escribir valores dentro de cada celda
for d = 1:nArch
    for im = 1:nArch
        val = MAE_matrix(d, im);
        if ~isnan(val)
            % Texto blanco si el fondo es oscuro, negro si es claro
            if val > 1.2
                txt_color = 'w';
            else
                txt_color = 'k';
            end
            text(im, d, sprintf('%.2f', val), ...
                'HorizontalAlignment', 'center', ...
                'FontSize', 12, 'FontWeight', 'bold', ...
                'Color', txt_color);
        end
    end
end

% Resaltar la mejor celda (mínimo MAE)
[min_val, min_idx] = min(MAE_matrix(:));
[best_d, best_im] = ind2sub(size(MAE_matrix), min_idx);
hold on;
rectangle('Position', [best_im-0.5, best_d-0.5, 1, 1], ...
    'EdgeColor', 'r', 'LineWidth', 2.5, 'LineStyle', '-');
hold off;

if save_figures
    exportgraphics(fig1, fullfile(fig_path, 'fig_heatmap_mae.png'), 'Resolution', 300);
    fprintf('💾 Figura 1 guardada\n');
end

%% =========================================================================
%  FIGURA 2: BARRAS AGRUPADAS - INTEGRADO vs UNIFICADO
%  =========================================================================

fig2 = figure('Name', 'Integrado vs Unificado', 'Position', [50, 50, 800, 450], 'Color', 'w');

% Para cada arquitectura como detector, tomar su mejor pipeline integrado
% (siempre será con 1D-CNN como imputador)
best_integrated_MAE  = NaN(nArch, 1);
best_integrated_RMSE = NaN(nArch, 1);
best_integrated_label = cell(nArch, 1);

for d = 1:nArch
    [min_mae, best_imp] = min(MAE_matrix(d, :));
    best_integrated_MAE(d) = min_mae;
    best_integrated_RMSE(d) = RMSE_matrix(d, best_imp);
    best_integrated_label{d} = sprintf('%s→%s', arch_labels{d}, arch_labels{best_imp});
end

% --- Subplot 1: MAE ---
subplot(1, 2, 1);

x = 1:nArch;
bar_data_mae = [unified_MAE, best_integrated_MAE];
b1 = bar(x, bar_data_mae, 'grouped');

b1(1).FaceColor = [0.75, 0.75, 0.75];   % Gris - Unificado
b1(2).FaceColor = [0.20, 0.47, 0.73];   % Azul - Integrado

set(gca, 'XTick', x, 'XTickLabel', arch_labels, 'FontSize', 10);
ylabel('MAE (mg/dL)', 'FontSize', 11);
title('MAE: Integrado vs Unificado', 'FontSize', 12);
legend({'Unificado', 'Mejor integrado (→1D-CNN)'}, ...
    'Location', 'northeast', 'FontSize', 9);
grid on;
set(gca, 'GridAlpha', 0.15);

% Agregar etiquetas de valor sobre cada barra
hold on;
for j = 1:2
    x_bar = b1(j).XEndPoints;
    y_bar = b1(j).YEndPoints;
    for k = 1:nArch
        text(x_bar(k), y_bar(k) + 0.08, sprintf('%.2f', bar_data_mae(k,j)), ...
            'HorizontalAlignment', 'center', 'FontSize', 8, 'FontWeight', 'bold');
    end
end
hold off;

% --- Subplot 2: RMSE ---
subplot(1, 2, 2);

bar_data_rmse = [unified_RMSE, best_integrated_RMSE];
b2 = bar(x, bar_data_rmse, 'grouped');

b2(1).FaceColor = [0.75, 0.75, 0.75];   % Gris
b2(2).FaceColor = [0.80, 0.36, 0.36];   % Rojo - Integrado

set(gca, 'XTick', x, 'XTickLabel', arch_labels, 'FontSize', 10);
ylabel('RMSE (mg/dL)', 'FontSize', 11);
title('RMSE: Integrado vs Unificado', 'FontSize', 12);
legend({'Unificado', 'Mejor integrado (→1D-CNN)'}, ...
    'Location', 'northeast', 'FontSize', 9);
grid on;
set(gca, 'GridAlpha', 0.15);

% Etiquetas de valor
hold on;
for j = 1:2
    x_bar = b2(j).XEndPoints;
    y_bar = b2(j).YEndPoints;
    for k = 1:nArch
        text(x_bar(k), y_bar(k) + 0.15, sprintf('%.2f', bar_data_rmse(k,j)), ...
            'HorizontalAlignment', 'center', 'FontSize', 8, 'FontWeight', 'bold');
    end
end
hold off;

sgtitle('Pipeline Integrado vs Unificado: Métricas de Imputación', ...
    'FontSize', 14, 'FontWeight', 'bold');

if save_figures
    exportgraphics(fig2, fullfile(fig_path, 'fig_integrado_vs_unificado.png'), 'Resolution', 300);
    fprintf('💾 Figura 2 guardada\n');
end

%% =========================================================================
%  FIGURA 3: SEÑAL TEMPORAL - MEJOR PIPELINE (GRU → 1D-CNN)
%  =========================================================================

% Cargar resultados del mejor pipeline
best_combo = 'gru_det_1dcnn_imp';
best_file = fullfile(models_path, sprintf('results_pipeline_%s_latest.mat', best_combo));

if isfile(best_file)
    best_data = load(best_file);
    
    fig3 = figure('Name', 'Señal Temporal GRU→1DCNN', ...
        'Position', [50, 50, 1100, 650], 'Color', 'w');
    
    % Usar el primer modelo como ejemplo representativo
    modelo = 1;
    T = length(best_data.g_test);
    t = 1:T;
    
    g_corrupta = best_data.g_test;
    g_real     = best_data.g_test_real;
    g_final    = best_data.g_final_all(modelo, :);
    fallas_det = best_data.fallas_all(modelo, :);
    true_faults = best_data.true_f;
    
    % --- Subplot 1: Señales de glucosa ---
    subplot(3, 1, 1);
    hold on;
    
    plot(t, g_real, 'b-', 'LineWidth', 1.2, 'DisplayName', 'Señal real');
    plot(t, g_corrupta, 'Color', [0.8, 0.2, 0.2, 0.5], 'LineWidth', 0.8, ...
        'DisplayName', 'Señal corrupta');
    plot(t, g_final, 'k--', 'LineWidth', 1.0, 'DisplayName', 'Señal imputada (GRU→1D-CNN)');
    
    % Sombrear zonas de fallo real
    fault_starts = find(diff([0, true_faults]) == 1);
    fault_ends   = find(diff([true_faults, 0]) == -1);
    yl = ylim;
    for f = 1:numel(fault_starts)
        patch([fault_starts(f), fault_ends(f), fault_ends(f), fault_starts(f)], ...
              [yl(1), yl(1), yl(2), yl(2)], ...
              [1, 0.85, 0.85], 'EdgeColor', 'none', 'FaceAlpha', 0.4, ...
              'HandleVisibility', 'off');
    end
    
    legend('Location', 'best', 'FontSize', 9);
    ylabel('Glucosa (mg/dL)', 'FontSize', 11);
    title('Señal de glucosa: real, corrupta e imputada', 'FontSize', 12);
    grid on; set(gca, 'GridAlpha', 0.15);
    hold off;
    
    % --- Subplot 2: Umbral adaptativo y error de detección ---
    subplot(3, 1, 2);
    hold on;
    
    err_det = best_data.error_detect_all(modelo, :);
    tau_vec = best_data.threshold_all(modelo, :);
    
    % Error de detección
    plot(t, err_det, 'Color', [0.85, 0.45, 0.1], 'LineWidth', 0.7, ...
        'DisplayName', 'Error de detección |ŷ_{detect} - g(k)|');
    
    % Umbral adaptativo
    plot(t, tau_vec, 'Color', [0.15, 0.55, 0.15], 'LineWidth', 1.5, ...
        'DisplayName', 'Umbral adaptativo τ(k)');
    
    % Línea de umbral fijo como referencia
    yline(10, 'r--', 'LineWidth', 1.0, 'DisplayName', 'Umbral fijo (τ=10)');
    
    legend('Location', 'best', 'FontSize', 9);
    ylabel('mg/dL', 'FontSize', 11);
    title('Error de detección vs Umbral adaptativo B+ROC', 'FontSize', 12);
    grid on; set(gca, 'GridAlpha', 0.15);
    ylim([0, max(max(err_det)*1.1, max(tau_vec)*1.5)]);
    hold off;
    
    % --- Subplot 3: Detección de fallos (real vs detectado) ---
    subplot(3, 1, 3);
    hold on;
    
    % Fallos reales (ground truth)
    stem(t(true_faults == 1), ones(sum(true_faults == 1), 1), ...
        'Color', [0.2, 0.5, 0.8], 'Marker', 'none', 'LineWidth', 0.5, ...
        'DisplayName', 'Fallos reales');
    
    % Fallos detectados
    stem(t(fallas_det), 0.5 * ones(sum(fallas_det), 1), ...
        'Color', [0.8, 0.2, 0.2], 'Marker', 'none', 'LineWidth', 0.5, ...
        'DisplayName', 'Fallos detectados');
    
    legend('Location', 'best', 'FontSize', 9);
    ylabel('Indicador', 'FontSize', 11);
    xlabel('Muestras', 'FontSize', 11);
    title('Detección de fallos: Ground truth vs Detectados', 'FontSize', 12);
    ylim([-0.1, 1.2]);
    set(gca, 'YTick', [0, 0.5, 1], 'YTickLabel', {'Normal', 'Detectado', 'Real'});
    grid on; set(gca, 'GridAlpha', 0.15);
    hold off;
    
    sgtitle('Mejor pipeline: GRU (detector) → 1D-CNN (imputador) — Umbral adaptativo B+ROC', ...
        'FontSize', 14, 'FontWeight', 'bold');
    
    if save_figures
        exportgraphics(fig3, fullfile(fig_path, 'fig_senal_gru_1dcnn.png'), 'Resolution', 300);
        fprintf('💾 Figura 3 guardada\n');
    end
    
else
    fprintf('⚠️  No se encontró el archivo del mejor pipeline: %s\n', best_file);
end

%% ===================== RESUMEN =====================

fprintf('\n✅ Gráficas generadas:\n');
fprintf('   Fig 1: Heatmap 4×4 MAE (Detector × Imputador)\n');
fprintf('   Fig 2: Barras agrupadas Integrado vs Unificado (MAE + RMSE)\n');
fprintf('   Fig 3: Señal temporal del mejor pipeline GRU→1D-CNN\n');

if save_figures
    fprintf('\n💾 Figuras exportadas en: %s\n', fig_path);
else
    fprintf('\n💡 Para exportar las figuras, cambie save_figures = true\n');
end

fprintf('🏁 Completado.\n');
