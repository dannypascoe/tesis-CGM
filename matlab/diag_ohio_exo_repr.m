%% =========================================================================
%  DIAG_OHIO_EXO_REPR.M - Etapa 1: representacion de u y m, prediccion a un paso
%  =========================================================================
%  Pregunta: si a los modelos congelados (entrenados en Sorensen) les damos u(k)
%  y m(k) construidas de distinta forma, cambia su error de prediccion a un paso
%  sobre glucosa REAL de Ohio? Es la version barata y sin fallas del experimento
%  (mismo protocolo que diag_semillas_y_ablacion.m, bloque 1): teacher forcing,
%  cada prediccion usa las 6 lecturas anteriores reales mas u(k), m(k).
%
%  Que se compara (filas = representacion de u, columnas = la de m):
%    impulse         bolo [U] y carbohidratos [g] como impulsos crudos
%    kernel          gamma de literatura (insulina 70/270 min, CHO 45/210 min)
%    kernel_matched  gamma con pico y cola medidos sobre el propio ODE
%    ode             sub-modelos Dalla Man/Sorensen (lo que usa hoy el pipeline)
%    neutral         control: el canal se fija a su media de Sorensen (z = 0)
%  La matriz completa separa el efecto de cada canal (p. ej. u con ODE y m con
%  impulsos). La diagonal es "misma representacion en los dos canales".
%
%  Como leerlo:
%    - Referencia superior: MAE del mismo modelo sobre Sorensen (su dominio) y
%      persistencia g(k-1) en cada ventana.
%    - Una diferencia solo cuenta si supera la dispersion entre las 5 redes
%      (se imprime); con 3 pacientes no hay base para una prueba estadistica.
%    - NO afines los parametros de los kernels mirando estos resultados: las
%      ventanas de Ohio son tambien las de evaluacion final.
%
%  No modifica nada salvo guardar la tabla en data/ohio/results.
%  =========================================================================

clear; clc; close all;

%% ===================== CONFIGURACION =====================
scriptPath = 'D:\Documentos\02 MROI\07 Tesis\03 Extension Tesis\CGM_Project\CGM_Project\matlab';
addpath(fullfile(scriptPath, 'architectures'));
addpath(fullfile(scriptPath, 'data_preparation'));
addpath(fullfile(scriptPath, 'utils'));
addpath(fullfile(scriptPath, 'data'));

models_path = fullfile(scriptPath, 'data', 'models');
data_path   = fullfile(scriptPath, 'data', 'raw');
prep_path   = fullfile(scriptPath, 'data', 'ohio', 'prepared');
out_path    = fullfile(scriptPath, 'data', 'ohio', 'results');
if ~exist(out_path, 'dir'), mkdir(out_path); end

archs    = {'gru', '1dcnn'};
patients = {'540', '559', '596'};
reprs    = {'impulse', 'kernel', 'kernel_matched', 'ode'};
labels   = [reprs, {'neutral'}];        % neutral = control (z = 0)
nR = numel(labels);

%% ===================== MODELOS =====================
MD = struct();
for a = 1:numel(archs)
    MD.(matlab.lang.makeValidName(archs{a})) = load( ...
        fullfile(models_path, sprintf('trained_%s_latest.mat', archs{a})), ...
        'nets', 'media_g', 'std_g', 'media_u', 'std_u', 'media_m', 'std_m', ...
        'N_PAST', 'input_shape');
end
nA = numel(archs);
N  = MD.(matlab.lang.makeValidName(archs{1})).N_PAST;

%% ===================== REFERENCIA: SORENSEN (dominio de entrenamiento) =====================
[~, ~, g_sor, u_sor, m_sor, ~] = load_and_prepare_data(data_path);
mae_sor = zeros(nA, 1);
for aI = 1:nA
    d = MD.(matlab.lang.makeValidName(archs{aI}));
    [e, ~, ~] = one_step_mae(d, g_sor(:), u_sor(:), m_sor(:), N);
    mae_sor(aI) = mean(e);
end

%% ===================== OHIO: TODAS LAS COMBINACIONES (u_repr x m_repr) =====================
nP = numel(patients);
E   = nan(nP, nA, nR, nR);        % media entre redes
SD  = nan(nP, nA, nR, nR);        % dispersion entre redes
pers = nan(nP, 1);
zstat = nan(nP, numel(reprs), 4); % media/std de z para u y m por representacion (diagnostico)

for p = 1:nP
    wf = fullfile(prep_path, sprintf('paciente_%s_ventana5d_latest.mat', patients{p}));
    U = struct();  M = struct();  g = [];
    for r = 1:numel(reprs)
        [~, g_real, uu, mm] = load_and_prepare_data_ohio(wf, [], reprs{r});
        U.(reprs{r}) = uu(:);  M.(reprs{r}) = mm(:);  g = g_real(:);
    end
    ref = MD.(matlab.lang.makeValidName(archs{1}));
    for r = 1:numel(reprs)
        zu = (U.(reprs{r}) - ref.media_u) / ref.std_u;
        zm = (M.(reprs{r}) - ref.media_m) / ref.std_m;
        zstat(p, r, :) = [mean(zu) std(zu) mean(zm) std(zm)];
    end

    for aI = 1:nA
        d = MD.(matlab.lang.makeValidName(archs{aI}));
        for iu = 1:nR
            for im = 1:nR
                if iu <= numel(reprs), uu = U.(reprs{iu}); else, uu = repmat(d.media_u, size(g)); end
                if im <= numel(reprs), mm = M.(reprs{im}); else, mm = repmat(d.media_m, size(g)); end
                [e, ~, pe] = one_step_mae(d, g, uu, mm, N);
                E(p, aI, iu, im)  = mean(e);
                SD(p, aI, iu, im) = std(e);
                pers(p) = pe;
            end
        end
    end
end

%% ===================== REPORTE =====================
fprintf('\n=====================================================\n');
fprintf(' Z de las entradas (u, m normalizadas con Sorensen): media / std\n');
fprintf(' Entrenamiento: media 0, std 1 por construccion\n');
fprintf('=====================================================\n');
for p = 1:nP
    fprintf(' Paciente %s\n   %-16s %10s %10s %10s %10s\n', patients{p}, 'repr', 'z_u media', 'z_u std', 'z_m media', 'z_m std');
    for r = 1:numel(reprs)
        fprintf('   %-16s %10.2f %10.2f %10.2f %10.2f\n', reprs{r}, zstat(p, r, 1), zstat(p, r, 2), zstat(p, r, 3), zstat(p, r, 4));
    end
end

fprintf('\n=====================================================\n');
fprintf(' MAE a un paso (mg/dL), senal limpia, media de %d pacientes\n', nP);
fprintf(' filas = representacion de u | columnas = representacion de m\n');
fprintf('=====================================================\n');
for aI = 1:nA
    Ea = squeeze(mean(E(:, aI, :, :), 1, 'omitnan'));           % nR x nR
    sda = squeeze(mean(SD(:, aI, :, :), 1, 'omitnan'));
    fprintf('\n %s | referencia Sorensen: %.2f | persistencia Ohio (media): %.2f\n', ...
        upper(archs{aI}), mae_sor(aI), mean(pers));
    fprintf('   %-16s', 'u \ m');
    fprintf(' %14s', labels{:});  fprintf('\n');
    for iu = 1:nR
        fprintf('   %-16s', labels{iu});
        for im = 1:nR, fprintf(' %14.2f', Ea(iu, im)); end
        fprintf('\n');
    end
    fprintf('   dispersion entre las 5 redes (media de la matriz): %.2f mg/dL\n', mean(sda(:)));
    io = find(strcmp(labels, 'ode'));
    fprintf('   diagonal vs ode/ode:');
    for r = 1:numel(reprs)
        fprintf(' %s %+.2f |', labels{r}, Ea(r, r) - Ea(io, io));
    end
    fprintf('\n');
end

fprintf('\n=====================================================\n');
fprintf(' Por paciente, diagonal (misma representacion en u y m)\n');
fprintf('=====================================================\n');
for aI = 1:nA
    fprintf('\n %s\n   %-8s %10s', upper(archs{aI}), 'paciente', 'persist.');
    fprintf(' %14s', labels{1:numel(reprs)});  fprintf('  %14s\n', 'neutral');
    for p = 1:nP
        fprintf('   %-8s %10.2f', patients{p}, pers(p));
        for r = 1:numel(reprs), fprintf(' %14.2f', E(p, aI, r, r)); end
        fprintf('  %14.2f\n', E(p, aI, nR, nR));
    end
end

%% ===================== GUARDAR =====================
rows = {};
for p = 1:nP
    for aI = 1:nA
        for iu = 1:nR
            for im = 1:nR
                rows(end+1, :) = {patients{p}, archs{aI}, labels{iu}, labels{im}, ...
                    E(p, aI, iu, im), SD(p, aI, iu, im), pers(p)}; %#ok<SAGROW>
            end
        end
    end
end
T = cell2table(rows, 'VariableNames', {'paciente', 'arq', 'u_repr', 'm_repr', 'mae_1paso', 'sd_redes', 'persistencia'});
ts = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
save(fullfile(out_path, sprintf('diag_exo_repr_%s.mat', ts)), 'T', 'E', 'SD', 'pers', 'mae_sor', 'labels', 'patients', 'archs', 'zstat');
writetable(T, fullfile(out_path, sprintf('diag_exo_repr_%s.xlsx', ts)));
fprintf('\nGuardado en %s\n', out_path);

%% ===================== FUNCIONES LOCALES =====================
function [e, kv, pers_mae] = one_step_mae(d, g, u, m, N)
%ONE_STEP_MAE MAE a un paso por red (teacher forcing) sobre las muestras con
%   las N+1 lecturas validas (sin huecos ni NaN). Devuelve un MAE por red.
    g = g(:);  u = u(:);  m = m(:);
    T = numel(g);
    k = (N + 1):T;
    P = zeros(N, numel(k));
    for j = 1:N, P(j, :) = g(k - N - 1 + j).'; end
    valid = ~any(isnan(P), 1) & ~isnan(g(k)).' & ~isnan(g(k - 1)).';
    kv = k(valid);
    Pv = P(:, valid);
    pers_mae = mean(abs(g(kv) - g(kv - 1)));
    X = [ (Pv - d.media_g) / d.std_g; ...
          ((u(kv) - d.media_u) / d.std_u).'; ...
          ((m(kv) - d.media_m) / d.std_m).' ];
    e = zeros(1, numel(d.nets));
    for i = 1:numel(d.nets)
        yn = predict_batch(d.nets{i}, X, d.input_shape);
        e(i) = mean(abs(yn * d.std_g + d.media_g - g(kv)));
    end
end

function y = predict_batch(net, X, input_shape)
%PREDICT_BATCH Igual que en diag_semillas_y_ablacion.m: una llamada con celda
%   de secuencias; si el tamano no coincide, bucle muestra por muestra.
    n = size(X, 2);
    xseq = cell(n, 1);
    for i = 1:n
        if strcmp(input_shape, 'column')
            xseq{i} = X(:, i);
        else
            xseq{i} = X(:, i).';
        end
    end
    try
        y = predict(net, xseq);
        y = y(:);
        if numel(y) ~= n
            error('tamano de salida inesperado');
        end
    catch
        y = zeros(n, 1);
        for i = 1:n
            y(i) = predict(net, xseq(i));
        end
    end
end
