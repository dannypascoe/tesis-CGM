function T = summarize_exo_arms(S, names, ref_name)
%SUMMARIZE_EXO_ARMS  Compara brazos del experimento de representacion de u y m.
%   S       - celda de structs cargados de ohio_pipeline_summary*.mat
%             (campos: pat_mean, pat_std, metric_names, patients, combinaciones)
%   names   - celda con el nombre de cada brazo (misma longitud que S)
%   ref_name- (opcional) brazo contra el que se calculan las diferencias
%             (default 'ode'; si no existe, el primero)
%
%   Para cada combinacion detector->imputador imprime media +/- DE entre
%   pacientes (n = numero de pacientes) de las metricas principales, la
%   diferencia contra el brazo de referencia y la dispersion entre las 5 redes
%   (piso de ruido: una diferencia menor a eso no se distingue de la semilla).
%   Devuelve una tabla larga (brazo, combinacion, metrica, media, DE, dif).

    if nargin < 3 || isempty(ref_name), ref_name = 'ode'; end
    iref = find(strcmp(names, ref_name), 1);
    if isempty(iref), iref = 1; ref_name = names{1}; end

    show   = {'mae_d1', 'rmse_d1', 'mae_d3_disc', 'rec', 'fpr', 'f1', 'auc'};
    header = {'D1 MAE', 'D1 RMSE', 'D3 desc.', 'Rec', 'FPR%', 'F1', 'AUC'};
    scale  = [1 1 1 1 100 1 1];

    ref = S{iref};
    nP = numel(ref.patients);
    nC = size(ref.combinaciones, 1);
    rows = {};

    for c = 1:nC
        fprintf('\n=====================================================\n');
        fprintf(' %s -> %s | media +/- DE entre %d pacientes | dif = brazo - %s\n', ...
            upper(ref.combinaciones{c, 1}), upper(ref.combinaciones{c, 2}), nP, ref_name);
        fprintf('=====================================================\n');
        fprintf(' %-16s', 'brazo');
        for k = 1:numel(show), fprintf(' %14s', header{k}); end
        fprintf(' | %8s %8s\n', 'dif MAE', 'ruido');

        for a = 1:numel(S)
            Sa = S{a};
            fprintf(' %-16s', names{a});
            for k = 1:numel(show)
                v = metric_by_patient(Sa, c, show{k}) * scale(k);
                mu = nmean(v);  sd = nsd(v);
                fprintf(' %6.2f+/-%-5.2f', mu, sd);
                vr = metric_by_patient(ref, c, show{k}) * scale(k);
                rows(end+1, :) = {names{a}, sprintf('%s->%s', ref.combinaciones{c,1}, ref.combinaciones{c,2}), ...
                    show{k}, mu, sd, mu - nmean(vr)}; %#ok<AGROW>
            end
            d1  = metric_by_patient(Sa, c, 'mae_d1');
            d1r = metric_by_patient(ref, c, 'mae_d1');
            sdn = metric_by_patient(Sa, c, 'mae_d1', 'std');      % dispersion entre redes
            fprintf(' | %+8.2f %8.2f\n', nmean(d1) - nmean(d1r), nmean(sdn));
        end

        fprintf('\n D1 MAE por paciente (mg/dL)\n %-16s', 'brazo');
        fprintf(' %8s', ref.patients{:});  fprintf('\n');
        for a = 1:numel(S)
            fprintf(' %-16s', names{a});
            fprintf(' %8.2f', metric_by_patient(S{a}, c, 'mae_d1'));  fprintf('\n');
        end
    end

    T = cell2table(rows, 'VariableNames', {'brazo', 'combinacion', 'metrica', 'media', 'de', 'dif_vs_ref'});
    fprintf('\nLectura: ''ruido'' es la DE del D1 MAE entre las 5 redes de un mismo paciente (media de pacientes).\n');
    fprintf('Una dif menor que el ruido, o con signo distinto entre pacientes, no permite concluir ventaja.\n');
end

function v = metric_by_patient(S, combo, name, which)
%METRIC_BY_PATIENT  Vector (pacientes x 1) de una metrica para una combinacion.
    if nargin < 4, which = 'mean'; end
    j = find(strcmp(S.metric_names, name), 1);
    if isempty(j), error('summarize_exo_arms: metrica %s no existe en el resumen.', name); end
    if strcmp(which, 'std'), A = S.pat_std; else, A = S.pat_mean; end
    v = squeeze(A(:, combo, j));
    v = v(:);
end

function m = nmean(v)
%NMEAN Media ignorando NaN (NaN si no queda ningun valor).
    v = v(~isnan(v));
    if isempty(v), m = NaN; else, m = mean(v); end
end

function s = nsd(v)
%NSD Desviacion estandar ignorando NaN (NaN si quedan menos de 2 valores).
    v = v(~isnan(v));
    if numel(v) < 2, s = NaN; else, s = std(v); end
end
