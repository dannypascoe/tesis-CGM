%% =========================================================================
%  PLOT_OHIO_SENSIBILIDAD_RUIDO.M
%  Figura de la Seccion 4.5.3 (sensibilidad al nivel de ruido en OhioT1DM).
%  =========================================================================
%  No corre redes ni modifica resultados: lee los .mat que guarda
%  main_pipeline_ohio_v4 de las dos corridas y dibuja, para una rafaga de
%  ruido, el mismo episodio con sigma = 5.6 (fila 1) y sigma = 15 (fila 2):
%
%    fila 1  protocolo del articulo   (etiqueta '_synth_s5p6_exc')
%    fila 2  sensibilidad al ruido    (etiqueta '_synth_s15_xode')
%    columnas: pacientes 540, 559 y 596.
%
%  Mismo estilo que Figura 2 de plot_ohio_reconstruccion_coopt.m (referencia
%  negra, lectura con fallo gris punteada, GRU->GRU naranja, GRU->1D-CNN morado
%  discontinuo; la salida se dibuja solo donde difiere de la lectura). Las dos
%  filas de cada columna comparten el eje Y, para que la amplitud del ruido se
%  compare a simple vista. El pie de figura (en el .tex) explica que la fila 1
%  es sigma = 5.6 y la fila 2 sigma = 15; no hay texto dentro de los paneles
%  salvo el paciente como titulo de cada columna (igual que la Figura 2).
%
%  burst_k elige la rafaga: 1 = primera (muestras 250-279, 30 muestras, 2.5 h);
%  2 = segunda (muestras 850-884, 35 muestras, 2.9 h). El texto de 4.5.3 esta
%  escrito para burst_k = 1 (en el 540 la referencia baja de 141 a 47 mg/dL
%  durante esa rafaga).
%
%  Al correr imprime, por panel, cuantas muestras de la rafaga senalo al menos
%  una de las 5 redes y cuantas senala en promedio cada red. Con burst_k = 1
%  deberian salir estos valores (union / media por red):
%      540: sigma 5.6 -> 5 / 3.8     sigma 15 -> 13 / 12.4
%      559: sigma 5.6 -> 9 / 8.8     sigma 15 -> 20 / 19.0
%      596: sigma 5.6 -> 5 / 3.4     sigma 15 -> 20 / 17.2
%
%  Salidas (figures/ohio): fig_ohio_sensibilidad_ruido.eps (vectorial) y .png
%  (300 dpi).
%  =========================================================================

clear; clc; close all;

%% ===================== CONFIGURACION =====================
scriptPath = 'D:\Documentos\02 MROI\07 Tesis\03 Extension Tesis\CGM_Project\CGM_Project\matlab';  % <<< tu raiz

tags       = {'_synth_s5p6_exc', '_synth_s15_xode'};   % fila 1 y fila 2
sigma_txt  = {'5.6', '15'};                            % solo para la consola
patients   = {'540', '559', '596'};
imp_names  = {'gru', '1dcnn'};         % imputador de cada configuracion (detector = GRU)
Ts_min     = 5;

burst_k     = 1;                       % 1 = primera rafaga de ruido, 2 = segunda
ctx_episode = 18;                      % muestras de contexto a cada lado de la rafaga (1.5 h)

xlab_time       = 'Tiempo (horas)';
legend_fs       = 10;
axis_label_fs   = 11;
axis_label_bold = true;
letters         = 'abcdef';
show_col_titles = true;                % 'Paciente NNN' como titulo de cada columna
show_samples    = false;               % true: circulo en cada muestra (para revisar cortes de linea)

% Colores (mismos que plot_ohio_reconstruccion_coopt.m)
col_ref   = [0 0 0];
col_raw   = [0.55 0.55 0.55];
col_gru   = [0.91 0.45 0.05];
col_cnn   = [0.48 0.18 0.62];
col_disc  = [1.00 0.85 0.85];
col_noise = [1.00 0.95 0.75];
col_gap   = [0.88 0.88 0.88];

res_dir = fullfile(scriptPath, 'data', 'ohio', 'results');
fig_dir = fullfile(scriptPath, 'figures', 'ohio');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

mk = 'none';  if show_samples, mk = 'o'; end

lab_style = struct('fs', axis_label_fs, 'weight', 'normal');
if axis_label_bold, lab_style.weight = 'bold'; end

arrow   = char(8594);
lab_gru = ['GRU ' arrow ' GRU'];
lab_cnn = ['GRU ' arrow ' 1D-CNN'];
lab_disc = ['Desconexi' char(243) 'n inyectada'];   % no se muestra en esta figura

%% ===================== CARGA =====================
nP = numel(patients);
PA = cell(1, nP);                      % protocolo del articulo (sigma = 5.6)
PB = cell(1, nP);                      % sensibilidad (sigma = 15)
for p = 1:nP
    PA{p} = load_patient(res_dir, patients{p}, tags{1}, imp_names);
    PB{p} = load_patient(res_dir, patients{p}, tags{2}, imp_names);
    if ~isequal(PA{p}.ft, PB{p}.ft)
        error('plot_ohio_sensibilidad_ruido: paciente %s, las fallas de las dos corridas no coinciden.', patients{p});
    end
end
T  = numel(PA{1}.ref);
t  = (0:T-1)' * Ts_min / 60;           % horas desde el inicio de la ventana
dt = Ts_min / 60;

%% ===================== FIGURA: 2 FILAS (sigma) x 3 COLUMNAS (pacientes) =====================
fig = figure('Color', 'w', 'Position', [100 100 1300 780]);
tl = tiledlayout(2, nP, 'TileSpacing', 'compact', 'Padding', 'compact'); %#ok<NASGU>
k = 0;
for r = 1:2
    for p = 1:nP
        k  = k + 1;
        ax = nexttile;
        if r == 1, Pp = PA{p}; else, Pp = PB{p}; end

        [lo, hi, a, b] = burst_range(PA{p}.inj, burst_k, ctx_episode, T);
        yl = y_limits_pair(PA{p}, PB{p}, lo:hi);

        draw_panel(ax, t, dt, Pp, [t(lo) t(hi)], yl, col_ref, col_raw, col_gru, col_cnn, ...
                   col_disc, col_noise, col_gap, mk);
        panel_xlabel(ax, letters(k), r == 2, xlab_time, lab_style);
        if p == 1, panel_ylabel(ax, lab_style); end
        if r == 1 && show_col_titles
            title(ax, ['Paciente ' Pp.pid], 'FontSize', 11, 'FontWeight', 'bold');
        end
        if k == 1, ax1 = ax; end

        fl = Pp.flag(:, a:b);          % 5 redes x muestras de la rafaga
        fprintf('(%s) paciente %s | sigma %s | rafaga %d: muestras %d-%d (%d) | senaladas por >=1 red: %d | media por red: %.1f\n', ...
            letters(k), Pp.pid, sigma_txt{r}, burst_k, a, b, b - a + 1, ...
            sum(any(fl, 1)), mean(sum(fl, 2)));
    end
end
% Leyenda comun: referencia, lectura con fallo, las dos configuraciones y el ruido
add_legend(ax1, col_ref, col_raw, col_gru, col_cnn, col_disc, col_noise, col_gap, ...
           lab_gru, lab_cnn, lab_disc, logical([1 1 1 1 0 1 0]), legend_fs);
save_fig(fig, fig_dir, 'fig_ohio_sensibilidad_ruido');

fprintf('\nListo. Figura en %s\n', fig_dir);

%% ===================== FUNCIONES LOCALES =====================
function Pp = load_patient(res_dir, pid, res_tag, imp_names)
%LOAD_PATIENT Senales de un paciente para una corrida (etiqueta res_tag):
%   lectura con fallo, referencia, tipo de muestra, salida media de las 5
%   redes de cada configuracion y marcas de falla de la combinacion GRU->GRU
%   (el detector es el mismo en las dos combinaciones).
    out = cell(1, 2);
    for c = 1:2
        f = fullfile(res_dir, sprintf('results_ohio_%s_gru_det_%s_imp%s_latest.mat', ...
            pid, imp_names{c}, res_tag));
        if ~exist(f, 'file')
            error('plot_ohio_sensibilidad_ruido: no existe %s', f);
        end
        S = load(f, 'g_test', 'g_test_real', 'g_final_all', 'fallas_all', 'fault_type', 'inj_all');
        out{c} = mean(S.g_final_all, 1)';                 % media de las 5 redes
        if c == 1
            Pp.raw_full = S.g_test(:);                    % lectura con fallos (ceros incluidos)
            Pp.ref      = S.g_test_real(:);               % lectura original (NaN en huecos reales)
            Pp.ft       = double(S.fault_type(:));        % 0 limpio, 1 hueco, 2 ruido, 3 desconexion
            Pp.flag     = logical(S.fallas_all);          % nRedes x T
            ia = S.inj_all;                               % celda (una por colocacion)
            if iscell(ia), Pp.inj = ia{1}; else, Pp.inj = ia(1); end
        end
    end
    Pp.pid     = pid;
    Pp.gru     = out{1};
    Pp.cnn     = out{2};
    Pp.raw     = Pp.raw_full;
    Pp.raw(Pp.raw < 10) = NaN;                            % enmascara ceros de desconexion y huecos
    Pp.gru_imp = imputed_only(Pp.raw_full, Pp.gru);
    Pp.cnn_imp = imputed_only(Pp.raw_full, Pp.cnn);
end

function [lo, hi, a, b] = burst_range(inj, k, ctx, T)
%BURST_RANGE Tramo a dibujar alrededor de la k-esima rafaga de ruido (por
%   orden de aparicion): [lo hi] con contexto, y [a b] de la rafaga.
    st = inj.start(:);  sp = inj.stop(:);  ty = inj.type(:);
    idx = find(ty == 2);
    [~, o] = sort(st(idx));
    idx = idx(o);
    if k > numel(idx)
        error('burst_range: el paciente solo tiene %d rafagas de ruido.', numel(idx));
    end
    a  = st(idx(k));  b = sp(idx(k));
    lo = max(a - ctx, 1);
    hi = min(b + ctx, T);
end

function yl = y_limits_pair(Pa, Pb, idx)
%Y_LIMITS_PAIR Limites verticales comunes a las dos corridas de un paciente.
    v = [Pa.ref(idx); Pa.raw(idx); Pa.gru_imp(idx); Pa.cnn_imp(idx); ...
                      Pb.raw(idx); Pb.gru_imp(idx); Pb.cnn_imp(idx)];
    v = v(~isnan(v));
    yl = [max(0, floor((min(v) - 10) / 10) * 10), ceil((max(v) + 15) / 10) * 10];
end

function y = imputed_only(raw_full, out)
%IMPUTED_ONLY Salida solo donde difiere de la lectura (con una muestra de
%   margen a cada lado para que las lineas se conecten con la senal).
    m  = abs(out - raw_full) > 1e-6;
    me = m;
    me(2:end)   = me(2:end)   | m(1:end-1);
    me(1:end-1) = me(1:end-1) | m(2:end);
    y = nan(size(out));
    y(me) = out(me);
end

function r = runs_of(mask)
%RUNS_OF Tramos consecutivos de una mascara logica: filas [inicio fin].
    mask = mask(:)';
    d = diff([0 double(mask) 0]);
    r = [find(d == 1)', find(d == -1)' - 1];
end

function draw_panel(ax, t, dt, Pp, xl, yl, col_ref, col_raw, col_gru, col_cnn, col_disc, col_noise, col_gap, mk)
%DRAW_PANEL Un panel: sombreado de episodios y las cuatro curvas (mk: marcador
%   de muestra, 'none' u 'o').
    hold(ax, 'on');
    set(ax, 'XLim', xl, 'YLim', yl, 'Box', 'on', 'FontSize', 10);
    grid(ax, 'on');
    shade_runs(ax, t, dt, Pp.ft == 1, col_gap,   yl);
    shade_runs(ax, t, dt, Pp.ft == 2, col_noise, yl);
    shade_runs(ax, t, dt, Pp.ft == 3, col_disc,  yl);
    plot(ax, t, Pp.raw,     '--', 'Color', col_raw, 'LineWidth', 0.9, 'Marker', mk, 'MarkerSize', 3.5);
    plot(ax, t, Pp.ref,     '-',  'Color', col_ref, 'LineWidth', 1.3, 'Marker', mk, 'MarkerSize', 3.5);
    plot(ax, t, Pp.gru_imp, '-',  'Color', col_gru, 'LineWidth', 1.7, 'Marker', mk, 'MarkerSize', 3.5);
    plot(ax, t, Pp.cnn_imp, '--', 'Color', col_cnn, 'LineWidth', 1.7, 'Marker', mk, 'MarkerSize', 3.5);
    set(ax, 'Layer', 'top');
end

function shade_runs(ax, t, dt, mask, col, yl)
%SHADE_RUNS Sombrea cada tramo consecutivo de la mascara.
    r = runs_of(mask);
    for k = 1:size(r, 1)
        x0 = t(r(k, 1)) - dt / 2;
        x1 = t(r(k, 2)) + dt / 2;
        patch('Parent', ax, 'XData', [x0 x1 x1 x0], 'YData', [yl(1) yl(1) yl(2) yl(2)], ...
              'FaceColor', col, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    end
end

function panel_xlabel(ax, letter, with_time, time_text, lab)
%PANEL_XLABEL Etiqueta del eje X: el tiempo (opcional) y debajo la letra del
%   panel en negritas entre parentesis.
    if with_time
        h = xlabel(ax, {time_text; ['\bf(' letter ')']});
    else
        h = xlabel(ax, ['\bf(' letter ')']);
    end
    set(h, 'FontSize', lab.fs, 'FontWeight', lab.weight);
end

function panel_ylabel(ax, lab)
%PANEL_YLABEL Etiqueta del eje Y con el tamano y grosor de lab.
    h = ylabel(ax, 'Glucosa (mg/dL)');
    set(h, 'FontSize', lab.fs, 'FontWeight', lab.weight);
end

function add_legend(ax, col_ref, col_raw, col_gru, col_cnn, col_disc, col_noise, col_gap, lab_gru, lab_cnn, lab_disc, keep, fs)
%ADD_LEGEND Leyenda comun en la parte superior, solo con las entradas de keep.
    h(1) = plot(ax, NaN, NaN, '-',  'Color', col_ref, 'LineWidth', 1.3);
    h(2) = plot(ax, NaN, NaN, '--', 'Color', col_raw, 'LineWidth', 0.9);
    h(3) = plot(ax, NaN, NaN, '-',  'Color', col_gru, 'LineWidth', 1.7);
    h(4) = plot(ax, NaN, NaN, '--', 'Color', col_cnn, 'LineWidth', 1.7);
    h(5) = patch('Parent', ax, 'XData', NaN, 'YData', NaN, 'FaceColor', col_disc,  'EdgeColor', 'none');
    h(6) = patch('Parent', ax, 'XData', NaN, 'YData', NaN, 'FaceColor', col_noise, 'EdgeColor', 'none');
    h(7) = patch('Parent', ax, 'XData', NaN, 'YData', NaN, 'FaceColor', col_gap,   'EdgeColor', 'none');
    labels = {'Referencia', 'Lectura con fallo', lab_gru, lab_cnn, ...
              lab_disc, 'Ruido blanco inyectado', 'Hueco real'};
    n  = nnz(keep);
    nc = n;  if n > 4, nc = ceil(n / 2); end
    lg = legend(ax, h(keep), labels(keep), 'Orientation', 'horizontal', 'NumColumns', nc, 'Box', 'off');
    lg.FontSize = fs;
    lg.Layout.Tile = 'north';
end

function save_fig(fig, fig_dir, name)
%SAVE_FIG Exporta en EPS vectorial y PNG de 300 dpi.
    exportgraphics(fig, fullfile(fig_dir, [name '.eps']), 'ContentType', 'vector');
    exportgraphics(fig, fullfile(fig_dir, [name '.png']), 'Resolution', 300);
    fprintf('Guardado: %s (.eps/.png)\n', fullfile(fig_dir, name));
end