%% ========================================================================
%  FIGURA 4.3 - Deteccion de fallos: Umbral Fijo vs Adaptativo
%  ========================================================================
%  Compara, para una arquitectura y semilla dadas, la deteccion de fallos
%  bajo umbral fijo (Bloque A) vs umbral adaptativo (Bloque B). Distingue
%  verdaderos positivos de falsos positivos para mostrar donde el umbral
%  fijo genera falsas alarmas y por que el umbral adaptativo las elimina.
%
%  Configurable (ver seccion CONFIGURACION):
%    - architecture    : 'cnn_lstm' | 'gru' | '1dcnn' | 'transformer_lstm'
%    - seed_idx_manual : indice de semilla (1-5), o [] para automatico
%                        (la mas cercana a la precision media bajo
%                        umbral fijo)
%    - zoom_window     : [inicio, fin] en muestras, o [] para graficar
%                        la señal completa en ambos paneles -- util para
%                        revisar si hay falsos positivos aislados, lejos
%                        de cualquier fallo real (ver reporte en consola;
%                        patron analogo al de GRU en la Figura 10 del
%                        paper, muestras ~100-400)
%
%  Fuente de datos:
%    data/models/bloque_A/results_<architecture>_latest.mat  (fijo)
%    data/models/bloque_B/results_<architecture>_latest.mat  (adaptativo)
%
%  Salida: figuras/fig_4_3_umbral_<architecture>.eps (vectorial)
%  ========================================================================

clear; clc; close all;

%% ===================== CONFIGURACION =====================

scriptPath  = 'D:\Documentos\02 MROI\07 Tesis\03 Extension Tesis\CGM_Project\CGM_Project';
models_path = fullfile(scriptPath, 'data', 'models');
fig_path    = fullfile(scriptPath, 'figuras');
if ~exist(fig_path, 'dir'), mkdir(fig_path); end

architecture = 'transformer_lstm';   % 'cnn_lstm' | 'gru' | '1dcnn' | 'transformer_lstm'

file_fixed    = fullfile(models_path, 'bloque_A', sprintf('results_%s_latest.mat', architecture));
file_adaptive = fullfile(models_path, 'bloque_B', sprintf('results_%s_latest.mat', architecture));

seed_idx_manual = [];   % ej. 3 para forzar una semilla; [] = automatica (mas cercana a la media)

% Ventana a mostrar. [] = señal completa en ambos paneles (util para
% revisar si hay falsos positivos aislados, lejos de cualquier fallo
% real -- como el patron reportado para GRU en la Figura 10 del paper,
% muestras ~100-400). [inicio, fin] = esa ventana exacta.
% Conversion hora -> muestra (Ts=5 min/muestra -> 12 muestras/hora),
% equivalentes de la Figura 4.2:
%   Contexto (56-72 h)  -> [672, 864]
%   Zoom (61.5-66.0 h)  -> [738, 792]
zoom_window = [];   % ej. [533, 613] para el zoom cercano a un FP ya identificado

arch_names = containers.Map( ...
    {'cnn_lstm', 'gru', '1dcnn', 'transformer_lstm'}, ...
    {'CNN-LSTM', 'GRU', '1D-CNN', 'Transformer-LSTM'});
if isKey(arch_names, architecture)
    arch_display = arch_names(architecture);
else
    arch_display = upper(architecture);
end

%% ===================== CARGAR DATOS =====================

fprintf('📂 Cargando resultados de %s...\n', arch_display);

S_fixed = load(file_fixed,    'error_detect_all', 'fallas_all', 'true_f', 'umbral_fijo');
S_adap  = load(file_adaptive, 'error_detect_all', 'threshold_all', 'fallas_all', ...
                              'true_f', 'adapt_params');

true_f  = double(S_fixed.true_f(:)');
nModels = size(S_fixed.fallas_all, 1);
T       = size(S_fixed.fallas_all, 2);

assert(isequal(true_f, double(S_adap.true_f(:)')), ...
    'true_f difiere entre bloque_A y bloque_B: revisar que ambos usen el mismo conjunto de prueba.');

%% ===================== SELECCION DE SEMILLA =====================
%  Por defecto (seed_idx_manual = []): se elige, entre las 5 semillas, el
%  modelo cuya precision bajo umbral FIJO este mas cerca de la media (no
%  el mejor ni el peor caso). Si se especifica seed_idx_manual, se usa
%  esa semilla directamente.

precisions = zeros(nModels, 1);
for i = 1:nModels
    yhat = S_fixed.fallas_all(i, :);
    TP = sum(yhat == 1 & true_f == 1);
    FP = sum(yhat == 1 & true_f == 0);
    precisions(i) = TP / max(TP + FP, 1);
end
mean_precision = mean(precisions);

fprintf('Precisiones por semilla (umbral fijo): %s\n', mat2str(precisions, 3));

if isempty(seed_idx_manual)
    [~, seed_idx] = min(abs(precisions - mean_precision));
    fprintf('✅ Semilla seleccionada automáticamente: %d (precision=%.3f, media=%.3f)\n', ...
        seed_idx, precisions(seed_idx), mean_precision);
else
    seed_idx = seed_idx_manual;
    fprintf('✅ Semilla especificada manualmente: %d (precision=%.3f, media=%.3f)\n', ...
        seed_idx, precisions(seed_idx), mean_precision);
end

%% ===================== CLASIFICAR TP / FP =====================

fallas_fijo = S_fixed.fallas_all(seed_idx, :);
fallas_adap = S_adap.fallas_all(seed_idx, :);

tp_fijo = (fallas_fijo == 1) & (true_f == 1);
fp_fijo = (fallas_fijo == 1) & (true_f == 0);
tp_adap = (fallas_adap == 1) & (true_f == 1);
fp_adap = (fallas_adap == 1) & (true_f == 0);

fprintf('\nUmbral fijo       -> TP=%d | FP=%d\n', sum(tp_fijo), sum(fp_fijo));
fprintf('Umbral adaptativo -> TP=%d | FP=%d\n', sum(tp_adap), sum(fp_adap));

%% ===================== LOCALIZAR SEGMENTOS DE FALLO REAL =====================

d_true      = diff([0, true_f, 0]);
true_starts = find(d_true(1:end-1) == 1);
true_ends   = find(d_true(2:end) == -1);

fprintf('\nSegmentos de fallo real en la señal completa: %d\n', numel(true_starts));
for s = 1:numel(true_starts)
    fprintf('   [%d] muestras %d-%d\n', s, true_starts(s), true_ends(s));
end

%% ===================== REPORTE DE FALSOS POSITIVOS (umbral fijo) =====================
%  Se lista cada falso positivo con su distancia al fallo real mas
%  cercano, marcando como "aislado" los que quedan lejos de cualquier
%  episodio real -- analogos al patron reportado para GRU en la Figura
%  10 del paper (falsas alarmas dispersas en tramos sin fallo inyectado).

ISLA_UMBRAL = 50;   % muestras: mas alla de esto se considera "aislado"

fp_idx = find(fp_fijo);
fprintf('\nFalsos positivos bajo umbral fijo: %d\n', numel(fp_idx));
if ~isempty(fp_idx)
    if isempty(true_starts)
        for c = 1:numel(fp_idx)
            fprintf('   muestra %d\n', fp_idx(c));
        end
    else
        for c = 1:numel(fp_idx)
            cand = fp_idx(c);
            dist = min(min(abs(cand - true_starts)), min(abs(cand - true_ends)));
            tag = '';
            if dist > ISLA_UMBRAL
                tag = '  <-- aislado (lejos de cualquier fallo real)';
            end
            fprintf('   muestra %d (distancia al fallo real mas cercano: %d)%s\n', cand, dist, tag);
        end
    end
end

%% ===================== SELECCIONAR VENTANA A GRAFICAR =====================
%  zoom_window = []    -> se grafica la señal completa en ambos paneles.
%  zoom_window = [a,b] -> se usa esa ventana exacta.

if isempty(zoom_window)
    win_start = 1;
    win_end   = T;
    fprintf('\n📌 Mostrando la señal completa: muestras [%d, %d]\n', win_start, win_end);
else
    win_start = max(1, zoom_window(1));
    win_end   = min(T, zoom_window(2));
    fprintf('\n📌 Ventana especificada: muestras [%d, %d]\n', win_start, win_end);
end

win = win_start:win_end;

tiene_fallo_real = any(true_f(win) == 1);
n_fp_en_ventana   = sum(fp_fijo(win));
fprintf('Ventana final: muestras [%d, %d] | contiene fallo real: %d | FP (fijo) en ventana: %d\n', ...
    win_start, win_end, tiene_fallo_real, n_fp_en_ventana);

%% ===================== FIGURA =====================

fig = figure('Name', sprintf('%s: Umbral Fijo vs Adaptativo', arch_display), ...
             'Position', [80, 80, 1000, 680], 'Color', 'w');
tl = tiledlayout(fig, 2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

col_tp  = [0.20, 0.55, 0.20];   % verde:   verdadero positivo
col_fp  = [0.80, 0.10, 0.10];   % rojo:    falso positivo
col_err = [0.35, 0.35, 0.35];   % gris:    señal de error
col_thr = [0.85, 0.33, 0.10];   % naranja: umbral

% --- Panel superior: umbral fijo -------------------------------------
ax1 = nexttile(tl);

err_win = S_fixed.error_detect_all(seed_idx, win);
y_max1  = max([err_win, S_fixed.umbral_fijo]) * 1.15;

hold(ax1, 'on');
shade_fault_windows(ax1, win, true_f(win), y_max1);
plot(ax1, win, err_win, '-', 'Color', col_err, 'LineWidth', 1.0, ...
    'DisplayName', 'Error');
yline(ax1, S_fixed.umbral_fijo, '--', 'Color', col_thr, 'LineWidth', 1.6, ...
    'DisplayName', sprintf('Umbral fijo'));
plot_markers(ax1, win, tp_fijo, S_fixed.error_detect_all(seed_idx,:), col_tp, 'Verdadero Positivo');
plot_markers(ax1, win, fp_fijo, S_fixed.error_detect_all(seed_idx,:), col_fp, 'Falso Positivo');
hold(ax1, 'off');

ylim(ax1, [0, y_max1]);
ylabel(ax1, 'mg/dL');
title(ax1, 'Umbral fijo (\tau = 10 mg/dL)');
legend(ax1, 'Location', 'northwest', 'FontSize', 9);
grid(ax1, 'on'); box(ax1, 'on');

if ~tiene_fallo_real
    text(ax1, win_start + 0.02*(win_end-win_start), y_max1*0.92, ...
        '(sin fallo real inyectado en este tramo)', ...
        'FontSize', 8, 'FontAngle', 'italic', 'Color', [0.4 0.4 0.4]);
end

% --- Panel inferior: umbral adaptativo --------------------------------
ax2 = nexttile(tl);

err_win2 = S_adap.error_detect_all(seed_idx, win);
thr_win2 = S_adap.threshold_all(seed_idx, win);
y_max2   = max([err_win2, thr_win2]) * 1.15;

hold(ax2, 'on');
shade_fault_windows(ax2, win, true_f(win), y_max2);
plot(ax2, win, err_win2, '-', 'Color', col_err, 'LineWidth', 1.0, ...
    'DisplayName', 'Error');
plot(ax2, win, thr_win2, '-', 'Color', col_thr, 'LineWidth', 1.6, ...
    'DisplayName', 'Umbral Adaptativo');
plot_markers(ax2, win, tp_adap, S_adap.error_detect_all(seed_idx,:), col_tp, 'Verdadero Positivo');
plot_markers(ax2, win, fp_adap, S_adap.error_detect_all(seed_idx,:), col_fp, 'Falso Positivo');
hold(ax2, 'off');

ylim(ax2, [0, y_max2]);
ylabel(ax2, 'mg/dL');
xlabel(ax2, 'Muestras');
p = S_adap.adapt_params;
title(ax2, sprintf('Umbral adaptativo (\\beta = %.2f, \\tau_{min} = %.1f, \\gamma = %.2f)', ...
    p.beta, p.tau_min, p.gamma));
legend(ax2, 'Location', 'northwest', 'FontSize', 9);
grid(ax2, 'on'); box(ax2, 'on');

linkaxes([ax1, ax2], 'x');
xlim(ax1, [win_start, win_end]);

title(tl, sprintf('%s: Detección de fallos, Umbral Fijo vs. Adaptativo', arch_display), ...
    'FontWeight', 'bold', 'FontSize', 13);

%% ===================== EXPORTAR =====================

out_file = fullfile(fig_path, sprintf('fig_4_3_umbral_%s.eps', architecture));
exportgraphics(fig, out_file, 'ContentType', 'vector');
fprintf('\n✅ Figura exportada en: %s\n', out_file);

%% ===================== FUNCIONES AUXILIARES =====================

function shade_fault_windows(ax, win, true_f_win, y_top)
%SHADE_FAULT_WINDOWS Sombrea en gris claro las ventanas de fallo real
%   (ground truth) dentro del rango 'win'.
    d = diff([0, true_f_win, 0]);
    starts = win(d(1:end-1) == 1);
    ends   = win(d(2:end) == -1);
    for j = 1:numel(starts)
        patch(ax, [starts(j), ends(j), ends(j), starts(j)], ...
            [0, 0, y_top, y_top], [0.90, 0.90, 0.90], ...
            'EdgeColor', 'none', 'FaceAlpha', 0.6, 'HandleVisibility', 'off');
    end
end

function plot_markers(ax, win, mask_full, error_full, color, label)
%PLOT_MARKERS Grafica marcadores circulares en los instantes donde
%   mask_full==1, restringido a la ventana 'win'.
    idx = win(mask_full(win));
    if isempty(idx), return; end
    plot(ax, idx, error_full(idx), 'o', 'MarkerSize', 6, ...
        'MarkerFaceColor', color, 'MarkerEdgeColor', color, ...
        'LineStyle', 'none', 'DisplayName', label);
end