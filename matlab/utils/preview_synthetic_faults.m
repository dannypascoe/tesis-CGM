function preview_synthetic_faults(pid, g_test, g_test_real, true_f, k_min, cfg, seeds, adapt_params, zoom_pl)
%PREVIEW_SYNTHETIC_FAULTS  Muestra las fallas sinteticas ANTES de correr el pipeline.
%   Llama a inject_synthetic_faults con la misma configuracion y las mismas
%   semillas que usa main_pipeline_ohio_v3, asi que lo que se ve aqui es
%   exactamente lo que se va a inyectar. No usa ninguna red.
%
%   Figura 1 (una por paciente): una fila por colocacion, con la senal
%       original (gris), la lectura limpia (negro), las rafagas de ruido
%       (naranja), las desconexiones (rojo, sobre el eje en 0) y los huecos
%       reales (gris oscuro). Los tramos afectados van sombreados.
%   Figura 2 (una por paciente): zoom a cada segmento de la colocacion
%       zoom_pl, con +/-30 muestras de contexto. En las rafagas de ruido se
%       dibuja la banda de umbral aproximada del detector, ref +/- max(tau_min,
%       beta*ref), para comparar el ruido contra lo que el detector tolera.
%   Consola: tabla de segmentos con glucosa de referencia (media, rango,
%       variacion) y estadisticas del ruido realmente inyectado.
%
% Entradas:
%   pid          - identificador del paciente (texto)
%   g_test       - Nx1, lectura con 0 en huecos reales
%   g_test_real  - Nx1, referencia (NaN en huecos reales)
%   true_f       - huecos reales (1 = hueco)
%   k_min        - primera muestra evaluada (N_PAST + 1)
%   cfg          - configuracion de fallas (ver inject_synthetic_faults)
%   seeds        - vector de semillas, una por colocacion
%   adapt_params - struct con .beta y .tau_min (para la banda de umbral)
%   zoom_pl      - colocacion para el zoom (por defecto 1)

    if nargin < 9 || isempty(zoom_pl), zoom_pl = 1; end
    g_test = g_test(:);
    g_ref  = g_test_real(:);
    N      = numel(g_test);
    nPl    = numel(seeds);
    t_h    = (0:N-1) * 5 / 60;
    dt_h   = 5 / 60;

    C.ref   = [0.70 0.70 0.70];
    C.clean = [0.10 0.10 0.10];
    C.noise = [0.93 0.55 0.10];
    C.disc  = [0.80 0.15 0.15];
    C.gap   = [0.40 0.40 0.40];

    % ---- Inyectar (misma funcion y semillas que el pipeline) ----
    G = cell(nPl, 1);  FT = cell(nPl, 1);  INJ = cell(nPl, 1);
    for k = 1:nPl
        [g_syn, ~, ft, inj] = inject_synthetic_faults(g_test, g_ref, true_f, k_min, cfg, seeds(k));
        G{k} = g_syn;  FT{k} = ft(:);  INJ{k} = inj;
    end

    % ---- Tabla en consola ----
    fprintf('\n=====================================================\n');
    fprintf(' PREVISUALIZACION DE FALLAS | paciente %s\n', pid);
    fprintf('=====================================================\n');
    for k = 1:nPl
        ft = FT{k};  inj = INJ{k};
        n_hi = sum(ft >= 2 & g_ref >= 250);
        fprintf('\n Fallas sinteticas (colocacion %d/%d, semilla %d): %.1f%% de la ventana | muestras de falla con glucosa >=250: %d\n', ...
            k, nPl, seeds(k), 100 * mean(ft >= 2), n_hi);
        fprintf('   %-12s %-14s %6s | %-24s %8s | %s\n', ...
            'tipo', 'muestras', 'largo', 'glucosa ref: media [min max]', 'variacion', 'ruido inyectado');
        for j = 1:numel(inj.start)
            s = inj.start(j);  e = inj.stop(j);  L = e - s + 1;
            r = g_ref(s:e);
            if inj.type(j) == 2
                tname = 'ruido';
                nz = G{k}(s:e) - g_test(s:e);
                extra = sprintf('sigma real = %.1f | max |ruido| = %.1f mg/dL', std(nz), max(abs(nz)));
            else
                tname = 'desconexion';
                extra = '-';
            end
            fprintf('   %-12s %5d - %-6d %6d | %5.0f [%4.0f %4.0f]           %8.0f | %s\n', ...
                tname, s, e, L, mean(r), min(r), max(r), max(r) - min(r), extra);
        end
    end

    % ---- Figura 1: vista general, una fila por colocacion ----
    ymax = max(g_ref, [], 'omitnan') * 1.08;
    figure('Color', 'w', 'Name', sprintf('Fallas sinteticas - paciente %s', pid), ...
        'Position', [60 40 1250 min(980, 190 * nPl + 90)]);
    for k = 1:nPl
        ax = subplot(nPl, 1, k);  hold(ax, 'on');  box(ax, 'on');
        ft = FT{k};
        ytop = ymax;  if k == 1, ytop = ymax * 1.30; end     % holgura para la leyenda
        ylim(ax, [-8, ytop]);  xlim(ax, [t_h(1) t_h(end)]);

        shade(ax, t_h, dt_h, ft == 2, ylim(ax), C.noise);
        shade(ax, t_h, dt_h, ft == 3, ylim(ax), C.disc);
        shade(ax, t_h, dt_h, ft == 1, ylim(ax), C.gap);

        h_ref = plot(ax, t_h, g_ref, '-', 'Color', C.ref, 'LineWidth', 0.8);
        y = G{k};  y(ft ~= 0) = NaN;
        h_cl = plot(ax, t_h, y, '-', 'Color', C.clean, 'LineWidth', 1.0);
        y = G{k};  y(ft ~= 2) = NaN;
        h_no = plot(ax, t_h, y, '-', 'Color', C.noise, 'LineWidth', 1.5);
        y = zeros(N, 1);  y(ft ~= 3) = NaN;
        h_di = plot(ax, t_h, y, '-', 'Color', C.disc, 'LineWidth', 3);
        y = zeros(N, 1);  y(ft ~= 1) = NaN;
        h_ga = plot(ax, t_h, y, '-', 'Color', C.gap, 'LineWidth', 3);

        ylabel(ax, 'mg/dL');
        if nPl == 1
            title(ax, sprintf('Paciente %s | %.1f%% de la ventana con fallas sinteticas', ...
                pid, 100 * mean(ft >= 2)), 'FontWeight', 'normal');
        else
            title(ax, sprintf('Colocacion %d | semilla %d | %.1f%% de la ventana con fallas sinteticas', ...
                k, seeds(k), 100 * mean(ft >= 2)), 'FontWeight', 'normal');
        end
        if k == 1
            legend(ax, [h_ref, h_cl, h_no, h_di, h_ga], ...
                {'Senal original (referencia)', 'Lectura limpia', 'Ruido blanco', ...
                 'Desconexion', 'Hueco real'}, ...
                'Location', 'north', 'Orientation', 'horizontal', 'Box', 'off');
        end
        if k == nPl, xlabel(ax, 'Tiempo desde el inicio de la ventana (h)'); end
    end

    % ---- Figura 2: zoom a cada segmento de la colocacion zoom_pl ----
    zoom_pl = min(max(1, zoom_pl), nPl);
    inj = INJ{zoom_pl};  ft = FT{zoom_pl};  g_syn = G{zoom_pl};
    nS = numel(inj.start);
    nc = 2;  nr = ceil(nS / nc);
    fig2 = figure('Color', 'w', 'Name', sprintf('Zoom de segmentos - paciente %s', pid), ...
        'Position', [80 60 1150 250 * nr + 80]);
    pad = 30;
    for j = 1:nS
        s = inj.start(j);  e = inj.stop(j);
        i1 = max(1, s - pad);  i2 = min(N, e + pad);  idx = i1:i2;
        ax = subplot(nr, nc, j);  hold(ax, 'on');  box(ax, 'on');

        if inj.type(j) == 2
            col = C.noise;  tname = 'Ruido blanco';
            tau = max(adapt_params.tau_min, adapt_params.beta * g_ref(s:e));
            tt = t_h(s:e);
            fill(ax, [tt, fliplr(tt)], [(g_ref(s:e) + tau).', fliplr((g_ref(s:e) - tau).')], ...
                [0.75 0.85 0.95], 'EdgeColor', 'none', 'FaceAlpha', 0.7);
        else
            col = C.disc;   tname = 'Desconexion';
        end
        plot(ax, t_h(idx), g_ref(idx), '-', 'Color', C.ref, 'LineWidth', 1.2);
        y = g_syn(idx);  y(ft(idx) ~= 0) = NaN;
        plot(ax, t_h(idx), y, '-', 'Color', C.clean, 'LineWidth', 1.2);
        seg = (s:e);
        if inj.type(j) == 3
            plot(ax, t_h(seg), zeros(numel(seg), 1), '-', 'Color', col, 'LineWidth', 3);
        else
            plot(ax, t_h(seg), g_syn(seg), '-', 'Color', col, 'LineWidth', 2);
        end
        xlim(ax, [t_h(i1) t_h(i2)]);
        title(ax, sprintf('%s | muestras %d-%d (%d) | ref %.0f-%.0f mg/dL', ...
            tname, s, e, e - s + 1, min(g_ref(s:e)), max(g_ref(s:e))), 'FontWeight', 'normal');
        ylabel(ax, 'mg/dL');
        if j > nS - nc, xlabel(ax, 'h desde el inicio'); end
    end
    if any(inj.type == 2)
        annotation(fig2, 'textbox', [0.55 0.005 0.44 0.04], 'EdgeColor', 'none', ...
            'HorizontalAlignment', 'right', 'FontSize', 9, ...
            'String', 'Banda celeste: umbral aproximado del detector, ref +/- max(tau_min, beta*ref)');
    end

    drawnow;
end

function shade(ax, t_h, dt_h, mask, yl, col)
%SHADE Sombrea cada tramo de unos consecutivos de la mascara.
    mask = double(mask(:).');
    d = diff([0, mask, 0]);
    s = find(d == 1);
    e = find(d == -1) - 1;
    for r = 1:numel(s)
        x0 = t_h(s(r)) - dt_h / 2;  x1 = t_h(e(r)) + dt_h / 2;
        patch(ax, [x0 x1 x1 x0], [yl(1) yl(1) yl(2) yl(2)], col, ...
            'FaceAlpha', 0.16, 'EdgeColor', 'none');
    end
end