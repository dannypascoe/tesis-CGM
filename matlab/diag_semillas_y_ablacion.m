%% =========================================================================
%  DIAG_SEMILLAS_Y_ABLACION.M - Diagnostico de los modelos congelados
%  =========================================================================
%  Responde tres preguntas sin fallas de por medio (senal limpia, un paso):
%    1) Los modelos usan u y m? Se mide el MAE de prediccion a un paso con
%       u,m reales, neutros (z = 0) y barajados en el tiempo, tanto en
%       Sorensen (su dominio) como en cada ventana de Ohio.
%    2) Como se compara el MAE de las redes con la persistencia g(k-1) en
%       cada dominio? (Si tambien pierden en Sorensen, no es un fallo de
%       transferencia.)
%    3) Hay semillas mejores o peores? Se imprime el MAE por semilla (redes
%       1..5) en cada dominio y, a partir de los resultados guardados del
%       pipeline, D1, D3 de desconexion, recall de ruido y FPR por semilla.
%
%  Teacher forcing: cada prediccion usa las 6 lecturas ANTERIORES reales
%  (limpias) mas u(k), m(k). No hay deteccion ni imputacion recursiva.
%  En Ohio solo se evaluan muestras con las 7 lecturas validas (sin huecos).
%
%  No modifica nada: solo lee modelos, datos y resultados guardados.
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
results_dir = fullfile(scriptPath, 'data', 'ohio', 'results');

archs    = {'gru', '1dcnn'};
patients = {'540', '559', '596'};
modes    = {'real', 'neutral', 'shuffle'};
res_tag  = '_synth_s5p6';          % sufijo de los resultados del pipeline a leer
seed_shuffle = 12345;

%% ===================== MODELOS =====================
MD = struct();
for a = 1:numel(archs)
    d = load(fullfile(models_path, sprintf('trained_%s_latest.mat', archs{a})), ...
        'nets', 'media_g', 'std_g', 'media_u', 'std_u', 'media_m', 'std_m', ...
        'N_PAST', 'input_shape');
    MD.(matlab.lang.makeValidName(archs{a})) = d;
end

%% ===================== DOMINIOS =====================
[~, ~, g_sor, u_sor, m_sor, ~] = load_and_prepare_data(data_path);
dom_name = {'Sorensen'};
dom_g = {g_sor(:)};  dom_u = {u_sor(:)};  dom_m = {m_sor(:)};
for p = 1:numel(patients)
    wf = fullfile(prep_path, sprintf('paciente_%s_ventana5d_latest.mat', patients{p}));
    [~, g_real, uu, mm] = load_and_prepare_data_ohio(wf);
    dom_name{end+1} = ['Ohio ' patients{p}];      %#ok<SAGROW>
    dom_g{end+1} = g_real(:);  dom_u{end+1} = uu(:);  dom_m{end+1} = mm(:); %#ok<SAGROW>
end
nD = numel(dom_name);
nA = numel(archs);
nM = numel(modes);

%% ===================== BLOQUE 1: ABLACION Y PERSISTENCIA =====================
fprintf('\n=====================================================\n');
fprintf(' BLOQUE 1 | MAE de prediccion a un paso (mg/dL), senal limpia\n');
fprintf('=====================================================\n');

mae_seed = cell(nD, nA, nM);      % cada celda: vector 1 x nRedes
pers = zeros(nD, 1);

for dI = 1:nD
    g = dom_g{dI};  T = numel(g);
    N = MD.(matlab.lang.makeValidName(archs{1})).N_PAST;

    k = (N + 1):T;
    P = zeros(N, numel(k));
    for j = 1:N
        P(j, :) = g(k - N - 1 + j).';
    end
    valid = ~any(isnan(P), 1) & ~isnan(g(k)).' & ~isnan(g(k - 1)).';
    kv = k(valid);
    Pv = P(:, valid);
    pers(dI) = mean(abs(g(kv) - g(kv - 1)));

    fprintf('\n %s | muestras evaluadas: %d | persistencia g(k-1): MAE = %.2f\n', ...
        dom_name{dI}, numel(kv), pers(dI));

    for aI = 1:nA
        d = MD.(matlab.lang.makeValidName(archs{aI}));
        for mI = 1:nM
            uu = dom_u{dI};  mm = dom_m{dI};
            switch modes{mI}
                case 'neutral'
                    uu = repmat(d.media_u, size(uu));
                    mm = repmat(d.media_m, size(mm));
                case 'shuffle'
                    rs = RandStream('mt19937ar', 'Seed', seed_shuffle);
                    perm = randperm(rs, T);
                    uu = uu(perm);  mm = mm(perm);
            end
            X = [ (Pv - d.media_g) / d.std_g; ...
                  ((uu(kv) - d.media_u) / d.std_u).'; ...
                  ((mm(kv) - d.media_m) / d.std_m).' ];

            nets = d.nets;
            e = zeros(1, numel(nets));
            for i = 1:numel(nets)
                yn = predict_batch(nets{i}, X, d.input_shape);
                yhat = yn * d.std_g + d.media_g;
                e(i) = mean(abs(yhat - g(kv)));
            end
            mae_seed{dI, aI, mI} = e;
        end

        e_r = mae_seed{dI, aI, 1};  e_n = mae_seed{dI, aI, 2};  e_s = mae_seed{dI, aI, 3};
        fprintf('   %-6s real %.2f+/-%.2f | neutral %.2f+/-%.2f (%+.2f) | barajado %.2f+/-%.2f (%+.2f) | razon real/persistencia = %.2f\n', ...
            upper(archs{aI}), mean(e_r), std(e_r), mean(e_n), std(e_n), mean(e_n) - mean(e_r), ...
            mean(e_s), std(e_s), mean(e_s) - mean(e_r), mean(e_r) / pers(dI));
    end
end

%% ===================== BLOQUE 2: MAE POR SEMILLA (u,m reales) =====================
fprintf('\n=====================================================\n');
fprintf(' BLOQUE 2 | MAE a un paso por semilla (u,m reales)\n');
fprintf('=====================================================\n');
for aI = 1:nA
    fprintf('\n %s\n   %-10s', upper(archs{aI}), 'semilla');
    for dI = 1:nD, fprintf(' %10s', dom_name{dI}); end
    fprintf('\n');
    nR = numel(mae_seed{1, aI, 1});
    for i = 1:nR
        fprintf('   %-10d', i);
        for dI = 1:nD, fprintf(' %10.2f', mae_seed{dI, aI, 1}(i)); end
        fprintf('\n');
    end
    fprintf('   %-10s', 'rango');
    for dI = 1:nD
        v = mae_seed{dI, aI, 1};  fprintf(' %10.2f', max(v) - min(v));
    end
    fprintf('\n   %-10s', 'mejor red');
    for dI = 1:nD
        [~, ib] = min(mae_seed{dI, aI, 1});  fprintf(' %10d', ib);
    end
    fprintf('\n');
end

%% ===================== BLOQUE 3: SEMILLAS EN EL PIPELINE (resultados guardados) =====================
fprintf('\n=====================================================\n');
fprintf(' BLOQUE 3 | Resultados del pipeline por semilla (%s)\n', res_tag);
fprintf('=====================================================\n');
combos = {'gru_det_gru_imp', 'gru_det_1dcnn_imp'};
for p = 1:numel(patients)
    for c = 1:numel(combos)
        f = fullfile(results_dir, sprintf('results_ohio_%s_%s%s_latest.mat', patients{p}, combos{c}, res_tag));
        if ~exist(f, 'file')
            fprintf('\n (no se encontro %s)\n', f);
            continue;
        end
        S = load(f, 'M');
        M = S.M;
        fprintf('\n Paciente %s | %s\n', patients{p}, strrep(combos{c}, '_', ' '));
        fprintf('   %-8s %9s %11s %11s %8s\n', 'semilla', 'D1 MAE', 'D3 desc.', 'rec ruido', 'FPR%');
        for i = 1:numel(M.mae_d1)
            fprintf('   %-8d %9.2f %11.2f %11.3f %8.1f\n', i, M.mae_d1(i), ...
                M.mae_d3_disc(i), M.rec_noise(i), 100 * M.fpr(i));
        end
        fprintf('   %-8s %9.2f %11.2f %11.3f %8.1f\n', 'rango', ...
            max(M.mae_d1) - min(M.mae_d1), max(M.mae_d3_disc) - min(M.mae_d3_disc), ...
            max(M.rec_noise) - min(M.rec_noise), 100 * (max(M.fpr) - min(M.fpr)));
    end
end


%% ===================== FUNCIONES LOCALES =====================
function y = predict_batch(net, X, input_shape)
%PREDICT_BATCH Prediccion normalizada para todas las columnas de X (nFeat x n).
%   Intenta una sola llamada con una celda de secuencias; si el tamano de
%   la salida no coincide, cae a un bucle muestra por muestra (igual que el pipeline).
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