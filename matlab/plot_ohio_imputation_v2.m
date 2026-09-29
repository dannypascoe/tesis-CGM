%% =========================================================================
%  PLOT_OHIO_IMPUTATION_V2.M - Figuras de deteccion + imputacion en OhioT1DM
%  =========================================================================
%  Lee los resultados guardados por main_pipeline_ohio_v3.m (no vuelve a
%  correr las redes). Sirve para el escenario con fallas sinteticas y para
%  el de solo huecos reales.
%
%  Figura 1 (ventana completa, por paciente)
%    Panel superior: senal original (gris), lectura que ve el pipeline
%      (negro; ruido en naranja), salida imputada (azul, solo donde el
%      pipeline sustituye la lectura) y falsas alarmas (circulos naranja).
%      Sombreado: ruido (naranja), desconexion (rojo), hueco real (gris).
%    Panel inferior: error de deteccion |prediccion - lectura| y umbral
%      adaptativo (media de las 5 corridas).
%
%  Figura 2 (zoom, por paciente)
%    Escenario sintetico: un recuadro por segmento inyectado, con la senal
%      original, las 5 redes por separado (lineas celestes) y su media
%      (azul). El titulo da el MAE de la imputacion, el MAE de LOCF (mantener
%      el ultimo valor valido) y el sesgo (imputado - original).
%    Escenario solo real: zoom al hueco real mas largo (>= min_gap).
%
%  Tambien imprime en consola una tabla por segmento (MAE, LOCF, sesgo).
%
%  Salida: <scriptPath>/figures/ohio/ohio_imputacion_<ID>_<combo><tag>[_zoom].png
%  =========================================================================

clear; clc; close all;

%% ===================== CONFIGURACION =====================
scriptPath  = 'D:\Documentos\02 MROI\07 Tesis\03 Extension Tesis\CGM_Project\CGM_Project\matlab';
results_dir = fullfile(scriptPath, 'data', 'ohio', 'results');
fig_dir     = fullfile(scriptPath, 'figures', 'ohio');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

patients   = {'540', '559', '596'};
combo_name = 'gru_det_gru_imp';     % 'gru_det_gru_imp' | 'gru_det_1dcnn_imp'
mode_tag   = '_synth_s5p6';              % '' (solo huecos reales) | '_synth' | '_synth_s5p6' ...
zoom_pad   = 30;                    % muestras de contexto a cada lado del segmento
min_gap    = 3;                     % hueco real minimo para el zoom (escenario real)

%% ===================== BUCLE POR PACIENTE =====================
for p = 1:numel(patients)
    pid = patients{p};
    f = fullfile(results_dir, sprintf('results_ohio_%s_%s%s_latest.mat', pid, combo_name, mode_tag));
    if ~exist(f, 'file')
        warning('No se encontro: %s', f);
        continue;
    end
    S = load(f);

    T   = numel(S.g_test);
    t_h = (0:T-1) * 5 / 60;          % horas desde el inicio de la ventana
    ft  = get_ft(S);
    is_synth = any(ft >= 2);

    combo_label = sprintf('%s -> %s', upper(S.pipeline_config.arch_detect), ...
        upper(S.pipeline_config.arch_impute));
    ttl = sprintf('Paciente %s | %s | FPR %.1f%% | MAE (D1) %.2f mg/dL', ...
        pid, combo_label, 100*mean(S.M.fpr), mean(S.M.mae_d1));
    if is_synth
        ttl = sprintf('%s | D3 desconexion %.1f (LOCF %.1f) mg/dL', ttl, ...
            mean(S.M.mae_d3_disc, 'omitnan'), mean(S.M.mae_locf_disc, 'omitnan'));
    else
        ttl = sprintf('%s | huecos reales %.1f%%', ttl, 100*mean(S.true_f));
    end

    % --- Ventana completa ---
    out_full = fullfile(fig_dir, sprintf('ohio_imputacion_%s_%s%s.png', pid, combo_name, mode_tag));
    plot_window(S, t_h, 1:T, ft, ttl, out_full);

    % --- Zoom ---
    out_zoom = fullfile(fig_dir, sprintf('ohio_imputacion_%s_%s%s_zoom.png', pid, combo_name, mode_tag));
    if is_synth
        zoom_segments(S, t_h, ft, pid, combo_label, out_zoom, zoom_pad);
    else
        [gs, ge] = find_runs(ft == 1);
        if isempty(gs)
            fprintf('Paciente %s: sin huecos reales, no hay zoom.\n', pid);
            continue;
        end
        [Lmax, im] = max(ge - gs + 1);
        if Lmax < min_gap
            fprintf('Paciente %s: el hueco mas largo tiene %d muestra(s); no se genera zoom.\n', pid, Lmax);
            continue;
        end
        i1 = max(1, gs(im) - 36);  i2 = min(T, ge(im) + 36);
        plot_window(S, t_h, i1:i2, ft, sprintf('%s | zoom: hueco de %d muestras', ttl, Lmax), out_zoom);
    end
end


%% ===================== FUNCIONES LOCALES =====================

function ft = get_ft(S)
%GET_FT Tipo de muestra: 0 limpia | 1 hueco real | 2 ruido | 3 desconexion.
    if isfield(S, 'fault_type')
        ft = double(S.fault_type(:).');
    else
        ft = double(S.true_f(:).');          % archivos de v2: solo huecos reales
    end
end

function C = colors()
    C.ref   = [0.70 0.70 0.70];
    C.clean = [0.10 0.10 0.10];
    C.noise = [0.93 0.55 0.10];
    C.disc  = [0.80 0.15 0.15];
    C.gap   = [0.40 0.40 0.40];
    C.imp   = [0.00 0.45 0.74];
    C.runs  = [0.62 0.80 0.95];
end

function plot_window(S, t_h, idx, ft, ttl, out_png)
    C      = colors();
    dt_h   = 5 / 60;
    t      = t_h(idx);
    ref    = S.g_test_real(:).';                  % NaN en huecos reales
    read   = S.g_test(:).';  read(read < 10) = NaN;   % lo que ve el pipeline
    repl   = mean(S.fallas_all, 1) >= 0.5;        % la salida sustituye a la lectura
    g_imp  = mean(S.g_final_all, 1);  g_imp(~repl) = NaN;
    fa     = repl & (ft == 0);                    % falsa alarma sobre lectura limpia

    ymax = max(ref(idx), [], 'omitnan');
    yl1  = [-8, max(350, ymax) + 110];            % holgura arriba para la leyenda

    fig = figure('Color', 'w', 'Position', [80 80 1250 680]);

    % ---------------- Panel superior ----------------
    ax1 = subplot(3, 1, [1 2]);  hold(ax1, 'on');  box(ax1, 'on');
    ylim(ax1, yl1);
    shade(ax1, t, dt_h, ft(idx) == 2, yl1, C.noise);
    shade(ax1, t, dt_h, ft(idx) == 3, yl1, C.disc);
    shade(ax1, t, dt_h, ft(idx) == 1, yl1, C.gap);

    plot(ax1, [t(1) t(end)], [250 250], '--', 'Color', [0.55 0.55 0.55], 'LineWidth', 0.8);
    h_ref = plot(ax1, t, ref(idx), '-', 'Color', C.ref, 'LineWidth', 1.4);
    y = read;  y(ft ~= 0) = NaN;
    h_cl  = plot(ax1, t, y(idx), '-', 'Color', C.clean, 'LineWidth', 1.0);
    y = read;  y(ft ~= 2) = NaN;
    h_no  = plot(ax1, t, y(idx), '-', 'Color', C.noise, 'LineWidth', 1.4);
    h_imp = plot(ax1, t, g_imp(idx), '-', 'Color', C.imp, 'LineWidth', 1.6, ...
        'Marker', '.', 'MarkerSize', 9);
    h_fa  = plot(ax1, t(fa(idx)), read(idx(fa(idx))), 'o', 'MarkerSize', 5, ...
        'MarkerEdgeColor', [0.85 0.40 0.00], 'LineWidth', 1.0);

    hs = [h_ref, h_cl, h_imp, h_fa];
    nm = {'Senal original', 'Lectura del pipeline', 'Salida imputada (media de corridas)', 'Falsa alarma'};
    if any(ft(idx) == 2), hs(end+1) = h_no; nm{end+1} = 'Ruido blanco'; end %#ok<AGROW>
    legend(ax1, hs, nm, 'Location', 'north', 'Orientation', 'horizontal', 'Box', 'off');

    ylabel(ax1, 'Glucosa (mg/dL)');
    title(ax1, ttl, 'FontWeight', 'normal', 'Interpreter', 'none');
    xlim(ax1, [t(1) t(end)]);

    % ---------------- Panel inferior ----------------
    ax2 = subplot(3, 1, 3);  hold(ax2, 'on');  box(ax2, 'on');
    yl2 = [0 80];
    ylim(ax2, yl2);
    shade(ax2, t, dt_h, ft(idx) == 2, yl2, C.noise);
    shade(ax2, t, dt_h, ft(idx) == 3, yl2, C.disc);
    shade(ax2, t, dt_h, ft(idx) == 1, yl2, C.gap);
    ed = mean(S.error_detect_all, 1);
    th = mean(S.threshold_all, 1);
    h_e = plot(ax2, t, ed(idx), '-', 'Color', [0.45 0.45 0.45], 'LineWidth', 0.9);
    h_t = plot(ax2, t, th(idx), '-', 'Color', [0.85 0.10 0.10], 'LineWidth', 1.3);
    legend(ax2, [h_e, h_t], {'|prediccion - lectura|', 'Umbral adaptativo'}, ...
        'Location', 'northeast', 'Box', 'off');
    ylabel(ax2, 'Error de deteccion (mg/dL)');
    xlabel(ax2, 'Tiempo desde el inicio de la ventana (h)');
    xlim(ax2, [t(1) t(end)]);

    linkaxes([ax1 ax2], 'x');
    print(fig, out_png, '-dpng', '-r300');
    fprintf('Figura guardada: %s\n', out_png);
end

function zoom_segments(S, t_h, ft, pid, combo_label, out_png, pad)
%ZOOM_SEGMENTS Un recuadro por segmento sintetico, con las 5 redes y su media.
    C   = colors();
    inj = S.inj_all{1};
    ref = S.g_test_real(:).';
    T   = numel(ref);
    G   = S.g_final_all;                      % nRuns x T
    gm  = mean(G, 1);
    nS  = numel(inj.start);
    nc  = 2;  nr = ceil(nS / nc);

    fig = figure('Color', 'w', 'Position', [80 60 1200 270 * nr + 90]);

    fprintf('\n Paciente %s | %s | reconstruccion por segmento (media de %d redes)\n', ...
        pid, combo_label, size(G, 1));
    fprintf('   %-12s %-12s %6s | %8s %8s %9s\n', 'tipo', 'muestras', 'largo', 'MAE', 'LOCF', 'sesgo');

    for j = 1:nS
        s = inj.start(j);  e = inj.stop(j);  L = e - s + 1;
        i1 = max(1, s - pad);  i2 = min(T, e + pad);  idx = i1:i2;
        seg = s:e;
        ax = subplot(nr, nc, j);  hold(ax, 'on');  box(ax, 'on');

        if inj.type(j) == 2, col = C.noise;  tname = 'Ruido';
        else,                col = C.disc;   tname = 'Desconexion';
        end

        % metricas del segmento
        mae_runs = mean(abs(G(:, seg) - ref(seg)), 2);
        mae   = mean(mae_runs);
        bias  = mean(mean(G(:, seg), 1) - ref(seg));
        locf  = ref(max(1, s - 1));
        mae_l = mean(abs(locf - ref(seg)));
        fprintf('   %-12s %4d-%-7d %6d | %8.1f %8.1f %+9.1f\n', tname, s, e, L, mae, mae_l, bias);

        % dibujo
        vals = [ref(idx).'; reshape(G(:, idx), [], 1)];
        lo = min(vals, [], 'omitnan');
        hi = max(vals, [], 'omitnan');
        ylim(ax, [min(-10, lo - 5), hi + 0.12 * (hi - lo) + 5]);
        yl = ylim(ax);
        shade(ax, t_h(idx), 5/60, (idx >= s) & (idx <= e), yl, col);

        plot(ax, t_h(idx), ref(idx), '-', 'Color', C.ref, 'LineWidth', 2.0);
        for r = 1:size(G, 1)
            plot(ax, t_h(idx), G(r, idx), '-', 'Color', C.runs, 'LineWidth', 0.8);
        end
        h_m = plot(ax, t_h(idx), gm(idx), '-', 'Color', C.imp, 'LineWidth', 2.0);
        y = S.g_test(:).';  y(y < 10) = NaN;  y(ft ~= 0) = NaN;
        plot(ax, t_h(idx), y(idx), '-', 'Color', C.clean, 'LineWidth', 1.0);
        if inj.type(j) == 2
            y = S.g_test(:).';  y(ft ~= 2) = NaN;
            plot(ax, t_h(idx), y(idx), '-', 'Color', C.noise, 'LineWidth', 1.6);
        else
            plot(ax, t_h(seg), zeros(1, L), '-', 'Color', C.disc, 'LineWidth', 3);
        end
        plot(ax, t_h(idx), locf * ones(1, numel(idx)), ':', 'Color', [0.35 0.35 0.35], 'LineWidth', 1.0);

        xlim(ax, [t_h(i1) t_h(i2)]);
        title(ax, sprintf('%s | %d-%d (%d) | MAE %.1f (LOCF %.1f) | sesgo %+.1f', ...
            tname, s, e, L, mae, mae_l, bias), 'FontWeight', 'normal');
        ylabel(ax, 'mg/dL');
        if j > nS - nc, xlabel(ax, 'h desde el inicio'); end
        if j == 1
            legend(ax, h_m, {'Media de las redes'}, 'Location', 'best', 'Box', 'off');
        end
    end

    annotation(fig, 'textbox', [0.05 0.002 0.90 0.04], 'EdgeColor', 'none', ...
        'HorizontalAlignment', 'left', 'FontSize', 9, 'Interpreter', 'none', ...
        'String', sprintf(['Paciente %s | %s | gris = senal original; celeste = cada red; ' ...
        'azul = media; punteado = LOCF (ultimo valor valido); sesgo = imputado - original'], ...
        pid, combo_label));

    print(fig, out_png, '-dpng', '-r300');
    fprintf('Figura guardada: %s\n', out_png);
end

function shade(ax, t, dt_h, mask, yl, col)
%SHADE Sombrea cada tramo de unos consecutivos de la mascara.
    [s, e] = find_runs(mask);
    for r = 1:numel(s)
        x0 = t(s(r)) - dt_h / 2;  x1 = t(e(r)) + dt_h / 2;
        patch(ax, [x0 x1 x1 x0], [yl(1) yl(1) yl(2) yl(2)], col, ...
            'FaceAlpha', 0.16, 'EdgeColor', 'none');
    end
end

function [s, e] = find_runs(mask)
%FIND_RUNS Inicio y fin de cada tramo de unos consecutivos.
    mask = double(mask(:).');
    d = diff([0, mask, 0]);
    s = find(d == 1);
    e = find(d == -1) - 1;
end