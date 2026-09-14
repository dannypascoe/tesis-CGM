%% ========================================================================
%  FIGURA - Reconstruccion de senal CGM: GRU->GRU vs GRU->1D-CNN
%
%  Compara, dentro de las MISMAS ventanas de fallo detectado (mismo
%  detector GRU, misma semilla en ambos pipelines), las reconstrucciones
%  g_final de las dos configuraciones co-optimas.
%
%  Criterio de seleccion de la corrida (confirmado): la semilla cuyas
%  metricas agregadas (F1, MAE) esten mas cercanas al promedio de las 5,
%  usando la MISMA semilla para ambos pipelines.
%
%  Genera 1 figura con 2 paneles (vista completa + zoom) y la exporta en
%  .eps vectorial.
%
%  Prerrequisito: haber ejecutado main_pipeline_integrado_v2.m. Se cargan
%  directamente (sin volver a entrenar ni re-ejecutar nada):
%    data/models/bloque_C/results_pipeline_gru_det_gru_imp_latest.mat
%    data/models/bloque_C/results_pipeline_gru_det_1dcnn_imp_latest.mat
% ========================================================================

clear; clc; close all;

%% --------------------------- CONFIGURACION ------------------------------
scriptPath = 'D:\Documentos\02 MROI\07 Tesis\03 Extension Tesis\CGM_Project\CGM_Project';   % ajustar si es necesario

archivo_gg    = fullfile(scriptPath, 'data', 'models', 'bloque_C', ...
    'results_pipeline_gru_det_gru_imp_latest.mat');
archivo_g1    = fullfile(scriptPath, 'data', 'models', 'bloque_C', ...
    'results_pipeline_gru_det_1dcnn_imp_latest.mat');
archivo_excel = 'pipeline_comparison_nuevo.xlsx';   % misma convencion que el script de heatmaps/pareto

output_dir  = 'figuras_4_4';
if ~exist(output_dir, 'dir'); mkdir(output_dir); end
archivo_fig = fullfile(output_dir, 'fig_reconstruccion_GRUGRU_vs_GRU1DCNN.eps');

DT_MINUTES           = 5;      % intervalo de muestreo CGM (min)
UMBRAL_TIPO_FALLO    = 60;     % mg/dL - mismo criterio interno del pipeline (linea ~284 de main_pipeline_integrado_v2.m)
MARGEN_ZOOM_H        = 1.0;    % horas de contexto antes/despues de los 2 episodios, en el panel de zoom
CONTEXTO_COMPLETO_H  = 6.0;    % horas extra a cada lado del zoom, en el panel completo
AVISO_ZOOM_LARGO_H   = 12;     % si la ventana auto-detectada supera esto, se avisa (episodios muy separados)

% Para forzar manualmente la ventana de zoom (en horas desde el inicio de
% la senal), define aqui [h_ini h_fin]; deja [] para autodetectar.
ventana_manual_h = [];

%% --------------------------- CARGA Y VERIFICACION ------------------------------
fprintf('Cargando resultados...\n');
% NOTA: main_pipeline_integrado_v2.m ahora guarda error_impute_pred_all
% (basada en la prediccion cruda del imputador, ~D1) y error_impute_out_all
% (basada en g_final, con el suavizado alpha ya aplicado). Se usa pred_all
% por ser la equivalente a D1 (metrica estandar de la tesis). El chequeo
% cruzado de mas abajo confirma esto contra pipeline_comparison_nuevo.xlsx;
% si da warning, cambia aqui a 'error_impute_out_all'.
gg = load(archivo_gg, 'g_test', 'g_test_real', 'true_f', 'g_final_all', ...
                       'fallas_all', 'error_detect_all', 'error_impute_pred_all', 'pipeline_config');
g1 = load(archivo_g1, 'g_test', 'g_test_real', 'true_f', 'g_final_all', ...
                       'fallas_all', 'error_detect_all', 'error_impute_pred_all', 'pipeline_config');

fprintf('GRU->GRU:    N_PAST_det=%d | N_PAST_imp=%d | nRuns=%d\n', ...
    gg.pipeline_config.N_PAST_detect, gg.pipeline_config.N_PAST_impute, gg.pipeline_config.nRuns);
fprintf('GRU->1D-CNN: N_PAST_det=%d | N_PAST_imp=%d | nRuns=%d\n', ...
    g1.pipeline_config.N_PAST_detect, g1.pipeline_config.N_PAST_impute, g1.pipeline_config.nRuns);

if max(abs(gg.g_test - g1.g_test)) > 1e-9 || max(abs(gg.g_test_real - g1.g_test_real)) > 1e-9
    warning(['g_test/g_test_real no son identicos entre los dos archivos. ' ...
             'Verifica que ambos vengan de la misma ejecucion de main_pipeline_integrado_v2.m.']);
else
    fprintf('OK: g_test y g_test_real identicos entre ambos archivos.\n');
end

g_test      = gg.g_test(:)';
g_test_real = gg.g_test_real(:)';
true_f      = double(gg.true_f(:)');
T           = length(g_test);
nRuns       = size(gg.g_final_all, 1);
t_h         = (0:T-1) * DT_MINUTES / 60;

fprintf('T=%d muestras (%.1f dias) | nRuns=%d\n', T, T*DT_MINUTES/1440, nRuns);

%% --------------------------- SELECCION DE LA CORRIDA REPRESENTATIVA ------------------------------
F1_runs     = zeros(nRuns, 1);
MAE_GG_runs = zeros(nRuns, 1);
MAE_G1_runs = zeros(nRuns, 1);

for i = 1:nRuns
    yhat = double(gg.fallas_all(i, :));
    TP = sum(yhat==1 & true_f==1); FP = sum(yhat==1 & true_f==0);
    FN = sum(yhat==0 & true_f==1);
    prec = TP / max(TP+FP, 1);
    rec  = TP / max(TP+FN, 1);
    F1_runs(i) = 2*prec*rec / max(prec+rec, eps);

    MAE_GG_runs(i) = mean(gg.error_impute_pred_all(i, :));
    MAE_G1_runs(i) = mean(g1.error_impute_pred_all(i, :));
end

% Verificacion cruzada contra la tabla ya aprobada en la tesis (opcional)
if exist(archivo_excel, 'file')
    Tco   = readtable(archivo_excel, 'Sheet', 'Lista_16_combinaciones');
    det_c = strtrim(string(Tco.Detector));
    imp_c = strtrim(string(Tco.Imputador));
    row_gg = det_c=="GRU" & imp_c=="GRU";
    row_g1 = det_c=="GRU" & imp_c=="1DCNN";
    ok_chk = abs(mean(F1_runs)     - Tco.F1_mean(row_gg)) < 1e-3 && ...
             abs(mean(MAE_GG_runs) - Tco.MAE_mean(row_gg)) < 1e-2 && ...
             abs(mean(MAE_G1_runs) - Tco.MAE_mean(row_g1)) < 1e-2;
    if ok_chk
        fprintf('OK: promedios recalculados coinciden con %s.\n', archivo_excel);
    else
        warning(['Los promedios recalculados NO coinciden con la tabla ya aprobada (%s). ' ...
                 'Revisa que bloque_C/ no se haya regenerado despues de consolidar resultados.'], archivo_excel);
    end
else
    warning('No se encontro %s para la verificacion cruzada (se continua sin ella).', archivo_excel);
end

% Puntaje combinado estandarizado: mas cercano a 0 = mas representativo
z = @(v) (v - mean(v)) / (std(v) + eps);
score = abs(z(F1_runs)) + abs(z(MAE_GG_runs)) + abs(z(MAE_G1_runs));
[~, i_star] = min(score);

fprintf('\n%-6s %-8s %-10s %-10s %-8s\n', 'Run', 'F1', 'MAE_GG', 'MAE_G1', 'Score');
for i = 1:nRuns
    marca = ''; if i==i_star; marca = '  <-- elegida'; end
    fprintf('%-6d %-8.4f %-10.4f %-10.4f %-8.3f%s\n', i, F1_runs(i), MAE_GG_runs(i), MAE_G1_runs(i), score(i), marca);
end
fprintf('Corrida seleccionada: %d de %d\n', i_star, nRuns);

% Verificar (empiricamente, no solo por construccion) que la mascara de
% fallo y el error del detector son identicos entre pipelines en esta corrida
mask_gg = gg.fallas_all(i_star, :);
mask_g1 = g1.fallas_all(i_star, :);
err_gg  = gg.error_detect_all(i_star, :);
err_g1  = g1.error_detect_all(i_star, :);

if isequal(mask_gg, mask_g1)
    fprintf('OK: mascara de fallo detectado identica entre GRU->GRU y GRU->1D-CNN (corrida %d).\n', i_star);
else
    warning('Las mascaras de fallo difieren en %d muestras entre pipelines (corrida %d).', ...
        sum(mask_gg ~= mask_g1), i_star);
end
if max(abs(err_gg - err_g1)) > 1e-6
    warning('error_detect difiere entre pipelines en la corrida %d (max diff = %.4f).', ...
        i_star, max(abs(err_gg - err_g1)));
end

mask       = mask_gg;             % ventanas de fallo detectado (identica en ambos)
err_detect = err_gg;
g_final_gg = gg.g_final_all(i_star, :);
g_final_g1 = g1.g_final_all(i_star, :);

%% --------------------------- SEGMENTAR EPISODIOS DE FALLO Y CLASIFICAR ------------------------------
mp = [false, mask, false];
starts = find(diff(mp) == 1);        % primer indice (en mask) de cada episodio
ends   = find(diff(mp) == -1) - 1;   % ultimo indice (en mask) de cada episodio
nEp    = numel(starts);

tipo_ep = strings(nEp, 1);
for e = 1:nEp
    if mean(err_detect(starts(e):ends(e))) < UMBRAL_TIPO_FALLO
        tipo_ep(e) = "ruido_blanco";
    else
        tipo_ep(e) = "desconexion";
    end
end

fprintf('\nEpisodios de fallo detectados: %d (%d ruido blanco, %d desconexion)\n', ...
    nEp, sum(tipo_ep=="ruido_blanco"), sum(tipo_ep=="desconexion"));
fprintf('%-4s %-10s %-10s %-14s %-10s\n', '#', 'Inicio(h)', 'Fin(h)', 'Tipo', 'Dur(min)');
for e = 1:nEp
    fprintf('%-4d %-10.2f %-10.2f %-14s %-10.1f\n', ...
        e, t_h(starts(e)), t_h(ends(e)), tipo_ep(e), (ends(e)-starts(e)+1)*DT_MINUTES);
end

%% --------------------------- ELEGIR VENTANA CON AMBOS TIPOS DE FALLO ------------------------------
idx_rb = find(tipo_ep == "ruido_blanco");
idx_dc = find(tipo_ep == "desconexion");

if isempty(ventana_manual_h)
    if isempty(idx_rb) || isempty(idx_dc)
        error(['No se encontraron episodios de ambos tipos en esta corrida ' ...
               '(ruido blanco=%d, desconexion=%d). Revisa la tabla de arriba y ' ...
               'define ventana_manual_h manualmente.'], numel(idx_rb), numel(idx_dc));
    end

    mejor_span = Inf; mejor_par = [idx_rb(1) idx_dc(1)];
    for a = 1:numel(idx_rb)
        for b = 1:numel(idx_dc)
            e1 = idx_rb(a); e2 = idx_dc(b);
            span = max(ends(e1), ends(e2)) - min(starts(e1), starts(e2));
            if span < mejor_span
                mejor_span = span;
                mejor_par  = [e1 e2];
            end
        end
    end
    e1 = mejor_par(1); e2 = mejor_par(2);

    margen_muestras = round(MARGEN_ZOOM_H * 60 / DT_MINUTES);
    k_zoom_ini = max(1, min(starts(e1), starts(e2)) - margen_muestras);
    k_zoom_fin = min(T, max(ends(e1),   ends(e2))   + margen_muestras);

    fprintf('\nVentana de zoom auto-seleccionada:\n');
    fprintf('  Episodio %s: %.2f-%.2f h\n', tipo_ep(e1), t_h(starts(e1)), t_h(ends(e1)));
    fprintf('  Episodio %s: %.2f-%.2f h\n', tipo_ep(e2), t_h(starts(e2)), t_h(ends(e2)));
else
    k_zoom_ini = max(1, round(ventana_manual_h(1) * 60 / DT_MINUTES) + 1);
    k_zoom_fin = min(T, round(ventana_manual_h(2) * 60 / DT_MINUTES));
    fprintf('\nUsando ventana de zoom manual: %.2f - %.2f h\n', ventana_manual_h(1), ventana_manual_h(2));
end

dur_zoom_h = (k_zoom_fin - k_zoom_ini) * DT_MINUTES / 60;
fprintf('Duracion de la ventana de zoom: %.2f h\n', dur_zoom_h);
if dur_zoom_h > AVISO_ZOOM_LARGO_H
    warning(['La ventana de zoom abarca %.1f h (los episodios mas cercanos de cada tipo ' ...
             'estan lejos entre si). Revisa la tabla de episodios impresa arriba y considera ' ...
             'fijar ventana_manual_h con un par que prefieras.'], dur_zoom_h);
end

contexto_muestras = round(CONTEXTO_COMPLETO_H * 60 / DT_MINUTES);
k_full_ini = max(1, k_zoom_ini - contexto_muestras);
k_full_fin = min(T, k_zoom_fin + contexto_muestras);

%% --------------------------- FIGURA: 2 PANELES (COMPLETO + ZOOM) ------------------------------
fig = figure('Color', 'w', 'Position', [50 50 1300 780]);
tlo = tiledlayout(fig, 2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

ax1 = nexttile(tlo);
dibujar_panel_senales(ax1, t_h, k_full_ini, k_full_fin, g_test_real, g_test, ...
    g_final_gg, g_final_g1, mask, err_detect, UMBRAL_TIPO_FALLO, false);
title(ax1, 'Ventana de contexo', 'FontSize', 12, 'FontWeight', 'bold');
%title(ax1, sprintf('Ventana de contexto: %.2f - %.2f h', t_h(k_full_ini), t_h(k_full_fin)), ...
%    'FontSize', 12, 'FontWeight', 'bold');

yl1 = ylim(ax1);
hold(ax1, 'on');
plot(ax1, [t_h(k_zoom_ini) t_h(k_zoom_fin) t_h(k_zoom_fin) t_h(k_zoom_ini) t_h(k_zoom_ini)], ...
          [yl1(1) yl1(1) yl1(2) yl1(2) yl1(1)], 'k--', 'LineWidth', 1.2, 'HandleVisibility', 'off');

ax2 = nexttile(tlo);
dibujar_panel_senales(ax2, t_h, k_zoom_ini, k_zoom_fin, g_test_real, g_test, ...
    g_final_gg, g_final_g1, mask, err_detect, UMBRAL_TIPO_FALLO, true);
title(ax2, sprintf('Zoom: %.2f - %.2f h', t_h(k_zoom_ini), t_h(k_zoom_fin)), ...
    'FontSize', 12, 'FontWeight', 'bold');

% Leyenda compartida (una sola, fuera de ambos paneles)
lgd = legend(ax1, 'Orientation', 'horizontal', 'FontSize', 10);
lgd.Layout.Tile = 'north';

title(tlo, 'Reconstrucción de señal CGM: GRU \rightarrow GRU vs GRU \rightarrow 1D-CNN', ...
    'FontSize', 14, 'FontWeight', 'bold');

exportgraphics(fig, archivo_fig, 'ContentType', 'vector');
fprintf('\nFigura guardada en: %s\n', archivo_fig);


%% ========================================================================
%  FUNCIONES LOCALES
% ========================================================================
function dibujar_panel_senales(ax, t_h, k_ini, k_fin, g_real, g_crudo, g_gg, g_g1, ...
                                mask, err_detect, umbral, con_etiquetas)
    % Dibuja senal real, cruda y las 2 reconstrucciones en la ventana
    % [k_ini,k_fin], con sombreado de episodios de fallo por tipo.
    idx = k_ini:k_fin;
    hold(ax, 'on'); box(ax, 'on'); grid(ax, 'on');

    % Escala del eje Y basada en las 3 senales relevantes para la
    % comparacion (real y las 2 reconstrucciones). La senal cruda se
    % dibuja igual, pero no se deja que sus caidas/picos durante una
    % desconexion aplasten la escala de lo que realmente se compara.
    ylims_data = [min([g_real(idx) g_gg(idx) g_g1(idx)]) - 10, ...
                  max([g_real(idx) g_gg(idx) g_g1(idx)]) + 10];

    % --- Sombreado de episodios de fallo dentro de la ventana visible ---
    mp = [false, mask(idx), false];
    st = find(diff(mp)==1); en = find(diff(mp)==-1) - 1;
    for e = 1:numel(st)
        rango = idx(st(e)):idx(en(e));
        if mean(err_detect(rango)) < umbral
            color_ep = [1.00 0.92 0.55];   % amarillo claro: ruido blanco
            etiqueta = 'Ruido blanco';
        else
            color_ep = [1.00 0.75 0.75];   % rojo claro: desconexion
            etiqueta = 'Desconexión';
        end
        xs = [t_h(idx(st(e))) t_h(idx(en(e))) t_h(idx(en(e))) t_h(idx(st(e)))];
        ys = [ylims_data(1) ylims_data(1) ylims_data(2) ylims_data(2)];
        patch(ax, xs, ys, color_ep, 'EdgeColor', 'none', 'FaceAlpha', 0.55, 'HandleVisibility', 'off');
        if con_etiquetas
            text(ax, mean(t_h(idx([st(e) en(e)]))), ylims_data(2)-8, etiqueta, ...
                'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
        end
    end

    % --- Senales ---
    plot(ax, t_h(idx), g_real(idx),  '-',  'Color', [0.15 0.15 0.15], 'LineWidth', 1.6, ...
        'DisplayName', 'Señal real (referencia)');
    plot(ax, t_h(idx), g_crudo(idx), ':',  'Color', [0.55 0.55 0.55], 'LineWidth', 1.6, ...
        'DisplayName', 'Señal cruda (con fallo)');
    plot(ax, t_h(idx), g_gg(idx),    '-',  'Color', [0.90 0.45 0.00], 'LineWidth', 1.6, ...
        'DisplayName', 'GRU \rightarrow GRU');
    plot(ax, t_h(idx), g_g1(idx),    '--', 'Color', [0.49 0.18 0.56], 'LineWidth', 1.6, ...
        'DisplayName', 'GRU \rightarrow 1D-CNN');

    xlim(ax, [t_h(idx(1)) t_h(idx(end))]);
    ylim(ax, ylims_data);
    xlabel(ax, 'Tiempo (horas)',  'FontSize', 11, 'FontWeight', 'bold');
    ylabel(ax, 'Glucosa (mg/dL)', 'FontSize', 11, 'FontWeight', 'bold');
    set(ax, 'FontSize', 10);
end