function C = find_excursion_ranges(patients, prep_path, window_tag, fault_cfg, manual_faults, k_min)
%FIND_EXCURSION_RANGES  Candidatos de [inicio fin] para desconexiones en hiper/hipoglucemia.
%   Lee las ventanas de Ohio ya preparadas (las mismas del pipeline), busca
%   los tramos donde la glucosa de referencia esta en hiperglucemia (>=250 y
%   >180) o hipoglucemia (<70) y propone donde colocar las dos desconexiones
%   de fault_cfg.disc_lens para que caigan lo mas dentro posible de esas
%   zonas. Solo mira la referencia; no usa redes ni resultados del pipeline.
%
%   Respeta las mismas reglas que inject_synthetic_faults en modo manual:
%     - fault_cfg.margin muestras limpias a cada lado del segmento
%     - nada sobre huecos reales ni sobre el arranque (k_min + margin)
%     - lejos de las rafagas de ruido de manual_faults.p<ID>.noise
%
%   Uso (desde la ventana de comandos, con AGREGAR CARPETAS AL PATH y
%   CONFIGURACION de main_pipeline_ohio_v3.m ya ejecutadas):
%       C = find_excursion_ranges(patients, prep_path, window_tag, ...
%                                 fault_cfg, manual_faults);
%
%   Entradas:
%     patients, prep_path, window_tag, fault_cfg, manual_faults - como en v3
%     k_min - (opcional) primera muestra evaluada por el pipeline
%             (N_PAST_all + 1). Por defecto 7.
%
%   Salida C.p<ID>: .disc (propuesta [inicio fin], una fila por desconexion,
%   en el orden de disc_lens), .zona (zona elegida por fila), .cobertura
%   (fraccion de muestras de la desconexion dentro de la zona).

    if nargin < 6 || isempty(k_min), k_min = 7; end

    margin = fault_cfg.margin;
    lens   = fault_cfg.disc_lens(:).';
    nL     = numel(lens);

    MIN_VALID = 20;          % mismo criterio de referencia valida del pipeline
    K         = 3;           % candidatos a listar por caso
    MIN_COV   = 0.5;         % cobertura minima para proponer una zona
    DO_PLOT   = true;

    zone_names = {'>=250', '>180', '<70'};
    zone_fun   = {@(g) g >= 250, @(g) g > 180, @(g) g < 70};
    zone_sgn   = [1 1 -1];   % +1: interesa el maximo, -1: el minimo
    nZ         = numel(zone_names);
    prio_long  = [1 2];      % desconexion larga: hiper >=250, luego >180
    prio_short = [3 1 2];    % desconexion corta: hipo, luego hiper

    C = struct();
    store = cell(numel(patients), 1);

    for p = 1:numel(patients)
        pid = patients{p};
        wf  = fullfile(prep_path, sprintf('paciente_%s_%s_latest.mat', pid, window_tag));
        if ~exist(wf, 'file')
            error('No se encontro la ventana: %s', wf);
        end
        [~, g, ~, ~, tf, grid_ts] = load_and_prepare_data_ohio(wf);
        g = g(:).';  N = numel(g);
        gap = logical(tf(:).') | isnan(g);

        % --- muestras donde NO puede haber una falla (ni su margen) ---
        bad = gap | ~(g >= MIN_VALID);
        nz  = zeros(0, 2);
        fld = ['p' pid];
        if isfield(manual_faults, fld) && isfield(manual_faults.(fld), 'noise')
            nz = manual_faults.(fld).noise;
            for r = 1:size(nz, 1)
                bad(max(1, nz(r, 1)):min(N, nz(r, 2))) = true;
            end
        end
        cb = [0 cumsum(bad)];

        g0 = g;  g0(isnan(g0)) = 0;
        cg = [0 cumsum(g0)];

        % --- por longitud y zona: validez, conteo y puntaje de cada inicio ---
        VL = cell(1, nL);  CN = cell(nL, nZ);  SC = cell(nL, nZ);
        for li = 1:nL
            L = lens(li);
            s = 1:(N - L + 1);  e = s + L - 1;
            a = s - margin;     b = e + margin;
            inr = a >= max(1, k_min) & b <= N;
            aa = max(a, 1);  bb = min(b, N);
            ok = inr & ((cb(bb + 1) - cb(aa)) == 0);
            VL{li} = ok;
            mu = (cg(e + 1) - cg(s)) / L;
            for zi = 1:nZ
                z  = zone_fun{zi}(g);  z(isnan(g)) = false;
                cz = [0 cumsum(z)];
                cnt = cz(e + 1) - cz(s);
                CN{li, zi} = cnt;
                SC{li, zi} = cnt * 1e4 + zone_sgn(zi) * mu;
            end
        end

        % --- resumen de la ventana ---
        fprintf('\n=====================================================================\n');
        fprintf(' PACIENTE %s | %d muestras | huecos reales %.1f%% | %s a %s\n', pid, N, ...
            100 * mean(gap), datestr(grid_ts(1), 'dd-mmm HH:MM'), datestr(grid_ts(end), 'dd-mmm HH:MM'));
        fprintf('=====================================================================\n');
        for zi = 1:nZ
            z = zone_fun{zi}(g);  z(isnan(g)) = false;
            [rs, re] = find_runs(z);
            [~, ord] = sort(re - rs, 'descend');
            fprintf(' Zona %-5s: %4d muestras (%.1f%%) en %d tramos', zone_names{zi}, ...
                sum(z), 100 * mean(z), numel(rs));
            if ~isempty(rs)
                fprintf(' | mas largos: ');
                for j = 1:min(3, numel(ord))
                    r = ord(j);
                    fprintf('[%d %d] (%d)  ', rs(r), re(r), re(r) - rs(r) + 1);
                end
            end
            fprintf('\n');
        end

        % --- candidatos por longitud y zona ---
        for li = 1:nL
            L = lens(li);
            for zi = 1:nZ
                idx = pick_top(SC{li, zi}, VL{li}, L, margin, K, zeros(0, 2));
                fprintf('\n Desconexion de %d muestras | zona %s\n', L, zone_names{zi});
                if isempty(idx) || CN{li, zi}(idx(1)) == 0
                    fprintf('   (ninguna posicion valida toca esta zona)\n');
                    continue;
                end
                fprintf('   %-3s %-11s %-13s %-8s %5s %6s %5s\n', '#', '[inicio fin]', ...
                    'inicio', 'en zona', 'min', 'media', 'max');
                for j = 1:numel(idx)
                    s0 = idx(j);  seg = g(s0:s0 + L - 1);
                    fprintf('   %-3d [%4d %4d] %-13s %3d/%-4d %5.0f %6.0f %5.0f\n', j, s0, s0 + L - 1, ...
                        datestr(grid_ts(s0), 'dd-mmm HH:MM'), CN{li, zi}(s0), L, ...
                        min(seg), mean(seg), max(seg));
                end
            end
        end

        % --- propuesta: la larga a hiper, la corta a hipo (o a otra hiper) ---
        old_disc = zeros(0, 2);
        if isfield(manual_faults, fld) && isfield(manual_faults.(fld), 'disc')
            old_disc = manual_faults.(fld).disc;
        end
        prop = zeros(nL, 2);  zsel = cell(nL, 1);  cov = nan(nL, 1);
        taken = zeros(0, 2);
        for li = 1:nL
            if li == 1, prio = prio_long; else, prio = prio_short; end
            best = [];  bz = 0;  bcov = 0;
            for zj = 1:numel(prio)
                zi  = prio(zj);
                idx = pick_top(SC{li, zi}, VL{li}, lens(li), margin, 1, taken);
                if isempty(idx), continue; end
                c = CN{li, zi}(idx) / lens(li);
                if c >= MIN_COV
                    best = idx;  bz = zi;  bcov = c;  break;
                elseif c > bcov                                  % la de mayor cobertura
                    best = idx;  bz = zi;  bcov = c;
                end
            end
            if isempty(best)
                if li <= size(old_disc, 1), prop(li, :) = old_disc(li, :); end
                zsel{li} = 'ninguna (se conserva el rango actual)';
                taken = [taken; prop(li, :)]; %#ok<AGROW>
                continue;
            end
            prop(li, :) = [best, best + lens(li) - 1];
            zsel{li}    = zone_names{bz};
            cov(li)     = CN{li, bz}(best) / lens(li);
            taken       = [taken; prop(li, :)]; %#ok<AGROW>
        end

        fprintf('\n Propuesta para %s (revisala en la vista previa antes de correr):\n', pid);
        for li = 1:nL
            fprintf('   desconexion %2d muestras -> [%4d %4d] | zona %-5s | cobertura %s\n', ...
                lens(li), prop(li, 1), prop(li, 2), zsel{li}, cov_txt(cov(li)));
        end
        if ~isempty(nz)
            fprintf('   manual_faults.p%s.noise = %s;   %% sin cambios\n', pid, mat2str(nz));
        end
        fprintf('   manual_faults.p%s.disc  = %s;\n', pid, mat2str(prop));

        C.(fld).disc      = prop;
        C.(fld).zona      = zsel;
        C.(fld).cobertura = cov;
        store{p} = struct('pid', pid, 'g', g, 'gap', gap, 'nz', nz, 'prop', prop, 'zsel', {zsel});
    end

    % --- figura: referencia, ruido actual y desconexiones propuestas ---
    if DO_PLOT
        nP = numel(patients);
        figure('Color', 'w', 'Position', [60 60 1300 250 * nP]);
        for p = 1:nP
            S = store{p};  N = numel(S.g);
            subplot(nP, 1, p);  hold on;
            gp = S.g;  gp(S.gap) = NaN;
            plot(1:N, gp, '-', 'Color', [0.15 0.15 0.15], 'LineWidth', 0.9);
            yline(70,  '--', 'Color', [0.55 0.55 0.55]);
            yline(180, '--', 'Color', [0.55 0.55 0.55]);
            yline(250, '--', 'Color', [0.55 0.55 0.55]);
            yl = [0 max(400, max(gp))];
            for r = 1:size(S.nz, 1)
                patch([S.nz(r, 1) S.nz(r, 2) S.nz(r, 2) S.nz(r, 1)], [yl(1) yl(1) yl(2) yl(2)], ...
                    [0.93 0.55 0.10], 'FaceAlpha', 0.25, 'EdgeColor', 'none');
            end
            for r = 1:size(S.prop, 1)
                patch([S.prop(r, 1) S.prop(r, 2) S.prop(r, 2) S.prop(r, 1)], [yl(1) yl(1) yl(2) yl(2)], ...
                    [0.80 0.15 0.15], 'FaceAlpha', 0.25, 'EdgeColor', 'none');
                text(mean(S.prop(r, :)), yl(2) * 0.96, sprintf('%d-%d', S.prop(r, 1), S.prop(r, 2)), ...
                    'HorizontalAlignment', 'center', 'FontSize', 9);
            end
            ylim(yl);  xlim([1 N]);
            ylabel(sprintf('Paciente %s (mg/dL)', S.pid));
            if p == nP, xlabel('Muestra dentro de la ventana'); end
            box on;
        end
        sgtitle('Naranja: ruido actual | Rojo: desconexiones propuestas');
    end
end


%% ===================== FUNCIONES LOCALES =====================

function idx = pick_top(score, valid, L, margin, K, excl)
%PICK_TOP  Inicios con mayor puntaje, validos, sin solaparse entre si y a
%   distancia >= margin de los segmentos de excl ([inicio fin] por fila).
    sc = score;  sc(~valid) = -Inf;
    s  = 1:numel(sc);  e = s + L - 1;
    for r = 1:size(excl, 1)
        sc(s <= excl(r, 2) + margin & e >= excl(r, 1) - margin) = -Inf;
    end
    idx = [];
    for k = 1:K
        [best, i] = max(sc);
        if ~isfinite(best), break; end
        idx(end + 1) = i; %#ok<AGROW>
        sc(abs(s - i) < L) = -Inf;
    end
end

function [gs, ge] = find_runs(mask)
%FIND_RUNS  Inicio y fin de cada tramo contiguo de 1 en mask.
    mask = logical(mask(:).');
    d  = diff([0 mask 0]);
    gs = find(d == 1);
    ge = find(d == -1) - 1;
end

function t = cov_txt(c)
    if isnan(c), t = 'n/d'; else, t = sprintf('%.0f%%', 100 * c); end
end