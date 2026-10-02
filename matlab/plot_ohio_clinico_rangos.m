%% =========================================================================
%  PLOT_OHIO_CLINICO_RANGOS.M
%  Figura de tiempo en rango de la Seccion 4.6.2 (OhioT1DM), version
%  corregida de la figura "ohio_clinical_AGP" que dibuja
%  main_clinical_metrics_ohio_v1.m.
%  =========================================================================
%  No recalcula nada: lee ohio_clinical_metrics<res_tag>.mat (lo guarda
%  main_clinical_metrics_ohio_v1.m) y dibuja, por paciente, una barra
%  apilada por escenario con el porcentaje de tiempo en cada rango del
%  consenso ADA/EASD 2019. Los valores son los de la hoja PorPaciente del
%  .xlsx (media de las 5 redes), asi que coinciden con la tabla.
%
%  Cambios respecto a la figura original:
%    - Una sola leyenda arriba y paneles del mismo ancho (antes la leyenda
%      a la derecha angostaba el panel del 596 y cortaba las etiquetas).
%    - Cinco colores distintos: TBR nivel 2 / nivel 1 y TAR nivel 1 / nivel 2
%      ya no se confunden de dos en dos.
%    - Letra (a), (b), (c) en negritas bajo cada panel; el paciente va como
%      titulo de columna (misma excepcion que la Figura 2 de episodios).
%    - Nombres de escenario con el mismo formato que el resto de la tesis.
%    - show_locf = false omite la barra del LOCF.
%
%  Con res_tag = '_synth_s5p6_exc' el TIR que se imprime en la barra y en la
%  consola debe ser (Referencia, Sin imp., LOCF, GRU->GRU, GRU->1D-CNN):
%      540: 72.1  75.1  72.4  76.1  77.2
%      559: 62.3  63.4  63.3  66.3  67.1
%      596: 80.1  83.4  80.1  84.7  84.9
%
%  Salida (figures/ohio): fig_ohio_clinico_rangos<out_suffix>.eps (vectorial)
%  y .png (300 dpi).
%  =========================================================================

clear; clc; close all;

%% ===================== CONFIGURACION =====================
scriptPath = 'D:\Documentos\02 MROI\07 Tesis\03 Extension Tesis\CGM_Project\CGM_Project\matlab';  % <<< tu raiz

res_tag    = '_synth_s5p6_exc';       % corrida cuyas metricas se dibujan
out_suffix = '';                      % p. ej. '_s15' si dibujas '_synth_s15_xode'
show_locf  = false;                    % false: sin la barra del LOCF
show_col_titles = true;               % 'Paciente NNN' como titulo de cada columna

legend_fs       = 10;
axis_label_fs   = 11;
axis_label_bold = true;
tick_fs         = 10;
value_fs        = 9;                  % tamano del numero de TIR dentro de la barra
letters         = 'abc';

% Colores de las cinco bandas, de abajo hacia arriba:
% TBR nivel 2 (<54), TBR nivel 1 (54-69), TIR (70-180), TAR nivel 1 (181-250), TAR nivel 2 (>250)
zone_colors = [0.60 0.10 0.10;
               0.88 0.30 0.20;
               0.25 0.70 0.35;
               0.97 0.80 0.25;
               0.93 0.55 0.10];
zone_labels = {'TBR nivel 2 (<54)', 'TBR nivel 1 (54-69)', 'TIR (70-180)', ...
               'TAR nivel 1 (181-250)', 'TAR nivel 2 (>250)'};

res_dir = fullfile(scriptPath, 'data', 'ohio', 'results');
fig_dir = fullfile(scriptPath, 'figures', 'ohio');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

lab_style = struct('fs', axis_label_fs, 'weight', 'normal');
if axis_label_bold, lab_style.weight = 'bold'; end

%% ===================== CARGA =====================
f = fullfile(res_dir, ['ohio_clinical_metrics' res_tag '.mat']);
if ~exist(f, 'file')
    error('plot_ohio_clinico_rangos: no existe %s (corre antes main_clinical_metrics_ohio_v1.m)', f);
end
C  = load(f, 'PM', 'sc_names', 'metric_names', 'patients');
PM = C.PM;                                   % pacientes x escenarios x metricas (media de las 5 redes)
mi = @(nm) find(strcmp(C.metric_names, nm));

expected = {'Referencia', 'Sin imputacion', 'LOCF', 'GRU-GRU', 'GRU-1DCNN'};
if ~isequal(C.sc_names(:)', expected)
    error('plot_ohio_clinico_rangos: el orden de escenarios del .mat no es el esperado: %s', ...
        strjoin(C.sc_names(:)', ', '));
end
arrow     = char(8594);
sc_label  = {'Referencia', 'Sin imp.', 'LOCF', ['GRU ' arrow ' GRU'], ['GRU ' arrow ' 1D-CNN']};
sel = 1:numel(expected);
if ~show_locf, sel(3) = []; end

nP = numel(C.patients);
nS = numel(sel);

%% ===================== FIGURA: 1 FILA x 3 COLUMNAS (pacientes) =====================
fig = figure('Color', 'w', 'Position', [100 100 1300 560]);
tl = tiledlayout(1, nP, 'TileSpacing', 'compact', 'Padding', 'compact'); %#ok<NASGU>

fprintf('%-6s %-16s %8s %12s %7s %9s %9s\n', 'Pac.', 'Escenario', 'TBR<54', 'TBR 54-69', 'TIR', 'TAR 181-250', 'TAR>250');
for p = 1:nP
    ax = nexttile;  hold(ax, 'on');

    bd = zeros(nS, 5);
    for j = 1:nS
        s    = sel(j);
        tbr2 = PM(p, s, mi('TBR_L2'));
        tbr1 = PM(p, s, mi('TBR_L1'));                       % TBR_L1 (<70) incluye a TBR_L2 (<54)
        bd(j, :) = [tbr2, tbr1 - tbr2, PM(p, s, mi('TIR')), PM(p, s, mi('TAR_L1')), PM(p, s, mi('TAR_L2'))];
        fprintf('%-6s %-16s %8.1f %12.1f %7.1f %11.1f %9.1f\n', C.patients{p}, expected{s}, bd(j, :));
    end

    bh = bar(ax, bd, 'stacked', 'BarWidth', 0.65);
    for z = 1:5
        bh(z).FaceColor = zone_colors(z, :);
        bh(z).EdgeColor = 'w';
    end
    for j = 1:nS
        text(ax, j, bd(j, 1) + bd(j, 2) + bd(j, 3) / 2, sprintf('%.1f', bd(j, 3)), ...
            'HorizontalAlignment', 'center', 'Color', 'w', 'FontWeight', 'bold', 'FontSize', value_fs);
    end

    ylim(ax, [0 100]);
    set(ax, 'YTick', 0:20:100, 'XTick', 1:nS, 'XTickLabel', sc_label(sel), ...
        'XTickLabelRotation', 30, 'FontSize', tick_fs, 'Box', 'on');
    grid(ax, 'on');

    h = xlabel(ax, ['\bf(' letters(p) ')']);               % letra del panel bajo las etiquetas del eje X
    set(h, 'FontSize', lab_style.fs, 'FontWeight', lab_style.weight);
    if p == 1
        h = ylabel(ax, 'Porcentaje de tiempo (%)');
        set(h, 'FontSize', lab_style.fs, 'FontWeight', lab_style.weight);
        bh1 = bh;  ax1 = ax;
    end
    if show_col_titles
        title(ax, ['Paciente ' C.patients{p}], 'FontSize', 11, 'FontWeight', 'bold');
    end
end

% Leyenda comun arriba (las cinco bandas, de abajo hacia arriba en la barra)
lg = legend(ax1, bh1, zone_labels, 'Orientation', 'horizontal', 'NumColumns', 5, 'Box', 'off');
lg.FontSize   = legend_fs;
lg.Layout.Tile = 'north';

name = ['fig_ohio_clinico_rangos' out_suffix];
exportgraphics(fig, fullfile(fig_dir, [name '.eps']), 'ContentType', 'vector');
exportgraphics(fig, fullfile(fig_dir, [name '.png']), 'Resolution', 300);
fprintf('\nGuardado: %s (.eps/.png)\n', fullfile(fig_dir, name));