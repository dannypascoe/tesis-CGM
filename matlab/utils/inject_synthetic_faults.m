function [g_syn, true_f_syn, fault_type, info] = inject_synthetic_faults(g_test, g_test_real, true_f, k_min, cfg, seed)
%INJECT_SYNTHETIC_FAULTS  Suma fallas sinteticas a una ventana de Ohio.
%   Sigue el protocolo del articulo (Seccion 2.3):
%     F1 - rafagas de ruido blanco: ruido gaussiano de desviacion sigma,
%          recortado a +/- noise_max, en segmentos de 20 a 40 muestras.
%     F2 - desconexiones: la lectura pasa a 0 (mismo centinela que los
%          huecos reales) en segmentos de longitud fija.
%   Dos formas de ubicar las fallas:
%     - ALEATORIA (por defecto): posiciones al azar con semilla fija, SOLO
%       sobre tramos validos: sin solaparse con huecos reales, con las
%       primeras muestras (arranque) ni entre si, y con un margen de
%       muestras limpias a cada lado.
%     - MANUAL: si cfg.manual existe, se usan exactamente esos rangos
%       (cfg.manual.noise y cfg.manual.disc, matrices Kx2 con [inicio fin]
%       en indices de muestra). La semilla solo fija los valores del ruido.
%       Si un rango cae sobre un hueco real, sobre el arranque o cerca de
%       otro rango, se detiene con un mensaje que dice cual.
%   La referencia g_test_real NO se modifica, asi que D3 se puede medir.
%
% Entradas:
%   g_test      - Nx1, lectura con 0 en huecos reales
%   g_test_real - Nx1, referencia (NaN en huecos reales)
%   true_f      - huecos reales (1 = hueco), fila o columna
%   k_min       - primera muestra evaluada por el pipeline (N_PAST + 1)
%   cfg         - struct con campos:
%                   .n_noise      numero de rafagas de ruido
%                   .noise_len    [min max] de muestras por rafaga
%                   .noise_sigma  desviacion estandar del ruido [mg/dL]
%                   .noise_max    amplitud maxima [mg/dL]
%                   .disc_lens    vector con la longitud de cada desconexion
%                   .margin       muestras limpias a cada lado de cada falla
%                 y, opcionalmente (modo manual):
%                   .manual.noise  Kx2 [inicio fin] de rafagas de ruido
%                   .manual.disc   Kx2 [inicio fin] de desconexiones
%   seed        - semilla entera (reproducible; no altera el generador global)
%
% Salidas:
%   g_syn       - Nx1, lectura corrupta (ruido sumado; desconexion = 0)
%   true_f_syn  - 1xN, 1 en huecos reales y fallas sinteticas
%   fault_type  - 1xN, 0 limpia | 1 hueco real | 2 ruido | 3 desconexion
%   info        - struct con .start, .stop, .type (un elemento por falla) y .seed

    g_test      = g_test(:);
    g_test_real = g_test_real(:);
    real_gap    = logical(true_f(:));
    N           = numel(g_test);

    rs = RandStream('mt19937ar', 'Seed', seed);

    % ------------------------------------------------------------------
    % MODO MANUAL: rangos definidos por el usuario
    % ------------------------------------------------------------------
    if isfield(cfg, 'manual') && ~isempty(cfg.manual)
        man = cfg.manual;
        seg = zeros(0, 3);                                  % [inicio fin tipo]
        if isfield(man, 'noise') && ~isempty(man.noise)
            seg = [seg; man.noise(:, 1:2), 2 * ones(size(man.noise, 1), 1)];
        end
        if isfield(man, 'disc') && ~isempty(man.disc)
            seg = [seg; man.disc(:, 1:2), 3 * ones(size(man.disc, 1), 1)];
        end
        seg = sortrows(seg, 1);

        occupied   = real_gap;
        g_syn      = g_test;
        fault_type = zeros(N, 1);
        fault_type(real_gap) = 1;

        for n = 1:size(seg, 1)
            s = seg(n, 1);  e = seg(n, 2);  ty = seg(n, 3);
            tname = 'ruido';  if ty == 3, tname = 'desconexion'; end

            if s < k_min + cfg.margin || e > N || e < s
                error('Rango manual de %s [%d %d] invalido: debe cumplir %d <= inicio <= fin <= %d.', ...
                    tname, s, e, k_min + cfg.margin, N);
            end
            if any(isnan(g_test_real(s:e))) || any(real_gap(s:e))
                error('Rango manual de %s [%d %d] cae sobre un hueco real de la ventana. Muevelo.', tname, s, e);
            end
            lo = max(1, s - cfg.margin);  hi = min(N, e + cfg.margin);
            if any(occupied(lo:hi))
                error('Rango manual de %s [%d %d] queda a menos de %d muestras de un hueco real u otro rango. Muevelo.', ...
                    tname, s, e, cfg.margin);
            end
            occupied(s:e) = true;

            L = e - s + 1;
            if ty == 2
                noise = cfg.noise_sigma * randn(rs, L, 1);
                noise = max(min(noise, cfg.noise_max), -cfg.noise_max);
                g_syn(s:e) = max(g_test(s:e) + noise, 11);
                fault_type(s:e) = 2;
            else
                g_syn(s:e) = 0;
                fault_type(s:e) = 3;
            end
        end

        true_f_syn = double(fault_type.' > 0);
        fault_type = fault_type.';
        info = struct('start', seg(:, 1), 'stop', seg(:, 2), 'type', seg(:, 3), 'seed', seed);
        return;
    end

    % ------------------------------------------------------------------
    % MODO ALEATORIO
    % ------------------------------------------------------------------
    % --- Segmentos a colocar: desconexiones (tipo 3) y rafagas (tipo 2) ---
    disc_lens = cfg.disc_lens(:);
    noise_lens = zeros(cfg.n_noise, 1);
    for j = 1:cfg.n_noise
        noise_lens(j) = randi(rs, cfg.noise_len);   % entero en [min max]
    end
    seg_len  = [disc_lens; noise_lens];
    seg_type = [3 * ones(numel(disc_lens), 1); 2 * ones(cfg.n_noise, 1)];

    % Los mas largos primero: son los mas dificiles de ubicar
    [~, ord] = sort(seg_len, 'descend');

    % --- Estado de ocupacion ---
    occupied = real_gap;
    occupied(1 : min(N, k_min - 1 + cfg.margin)) = true;

    g_syn      = g_test;
    fault_type = zeros(N, 1);
    fault_type(real_gap) = 1;

    starts = zeros(numel(ord), 1);
    stops  = zeros(numel(ord), 1);
    types  = zeros(numel(ord), 1);

    for n = 1:numel(ord)
        q = ord(n);
        L = seg_len(q);

        lo_s = k_min + cfg.margin;
        hi_s = N - L + 1 - cfg.margin;
        if hi_s < lo_s
            error('inject_synthetic_faults: la ventana es demasiado corta para un segmento de %d muestras.', L);
        end

        placed = false;
        for attempt = 1:5000
            s  = randi(rs, [lo_s, hi_s]);
            e  = s + L - 1;
            lo = max(1, s - cfg.margin);
            hi = min(N, e + cfg.margin);
            if ~any(occupied(lo:hi)) && ~any(isnan(g_test_real(s:e)))
                placed = true;
                break;
            end
        end
        if ~placed
            error('inject_synthetic_faults: no se pudo ubicar un segmento de %d muestras (semilla %d).', L, seed);
        end

        occupied(s:e) = true;

        if seg_type(q) == 2
            noise = cfg.noise_sigma * randn(rs, L, 1);
            noise = max(min(noise, cfg.noise_max), -cfg.noise_max);
            % piso de 11 mg/dL para que el ruido nunca cruce el centinela (<10)
            g_syn(s:e) = max(g_test(s:e) + noise, 11);
            fault_type(s:e) = 2;
        else
            g_syn(s:e) = 0;
            fault_type(s:e) = 3;
        end

        starts(n) = s;  stops(n) = e;  types(n) = seg_type(q);
    end

    [starts, o2] = sort(starts);
    stops = stops(o2);  types = types(o2);

    true_f_syn = double(fault_type.' > 0);
    fault_type = fault_type.';

    info = struct('start', starts, 'stop', stops, 'type', types, 'seed', seed);
end