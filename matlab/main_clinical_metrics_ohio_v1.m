%% =========================================================================
%  MAIN_CLINICAL_METRICS_OHIO.M - Metricas clinicas de la validacion en OhioT1DM
%  =========================================================================
%  Calcula los indicadores del Consensus Report ADA/EASD 2019 (Battelino et
%  al.) sobre los resultados guardados por main_pipeline_ohio_v3.m
%  (fallas sinteticas, sigma = 5.6, rangos manuales), para los pacientes
%  540, 559 y 596 y las dos combinaciones co-optimas.
%
%  Escenarios (todos evaluados sobre las MISMAS muestras de referencia):
%    1. Referencia     : g_test_real, lectura limpia del sensor
%    2. Sin imputacion : referencia con las fallas sinteticas -> NaN
%    3. LOCF           : fallas sinteticas sustituidas por la ultima lectura
%                        limpia previa (conoce donde estan las fallas)
%    4. Pipeline       : g_final_all de cada combinacion, UNA CORRIDA A LA VEZ
%
%  Muestras evaluadas: con referencia valida (>= 20 mg/dL, no NaN) y despues
%  del arranque (k >= k_start). Los huecos reales del sensor quedan fuera
%  porque no hay referencia contra la cual medir.
%
%  Orden de agregacion (regla de trabajo):
%    corrida -> paciente (media de las 5 corridas) -> agregado (media +/- DE, n = 3)
%  La senal reconstruida no se promedia entre corridas ni entre pacientes
%  antes de calcular los indicadores. (main_clinical_metrics_v2 de Sorensen
%  si promedia la senal entre semillas.)
%
%  Ademas de los indicadores globales, se reporta la concordancia de
%  categoria de rango dentro de las propias fallas: porcentaje de muestras
%  de falla cuya salida cae en la misma categoria que la referencia
%  (<54 | 54-69 | 70-180 | 181-250 | >250). El TIR global diluye el error
%  porque las fallas son ~9% de la ventana.
%
%  Prerrequisito: results_ohio_<pid>_<combo>_synth_s5p6_latest.mat en
%                 data/ohio/results/ (con fault_type, g_final_all, k_start)
%
%  Salidas (data/ohio/results/ y figures/ohio/):
%    - ohio_clinical_metrics_synth_s5p6.xlsx / .mat
%    - ohio_clinical_AGP_synth_s5p6.eps / .png
%  =========================================================================

clear; clc; close all;

%% ===================== CONFIGURACION =====================
scriptPath  = 'D:\Documentos\02 MROI\07 Tesis\03 Extension Tesis\CGM_Project\CGM_Project\matlab';
results_dir = fullfile(scriptPath, 'data', 'ohio', 'results');
fig_dir     = fullfile(scriptPath, 'figures', 'ohio');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

patients = {'540', '559', '596'};
combos   = { 'gru_det_gru_imp',   'GRU-GRU'; ...
             'gru_det_1dcnn_imp', 'GRU-1DCNN' };
res_tag  = '_synth_s5p6_exc';          % corrida final (sigma = 5.6, rangos manuales)

% Umbrales ADA/EASD 2019 (mg/dL). Igual que en Sorensen: TBR_L1 (<70) incluye
% a TBR_L2 (<54); TAR_L1 (181-250) y TAR_L2 (>250) son excluyentes.
thr.L2 = 54;  thr.L1 = 70;  thr.H1 = 180;  thr.H2 = 250;
MIN_VALID = 20;                    % mismo criterio que compute_metrics del pipeline

metric_names  = {'TIR', 'TBR_L1', 'TBR_L2', 'TAR_L1', 'TAR_L2', 'Media', 'SD', 'CV', 'GMI'};
metric_labels = {'TIR 70-180 %', 'TBR L1 <70 %', 'TBR L2 <54 %', 'TAR L1 181-250 %', ...
                 'TAR L2 >250 %', 'Media mg/dL', 'SD mg/dL', 'CV %', 'GMI %'};
fmt_abs = {'%.1f', '%.1f', '%.1f', '%.1f', '%.1f', '%.1f', '%.1f', '%.1f', '%.2f'};
fmt_dlt = {'%+.2f', '%+.2f', '%+.2f', '%+.2f', '%+.2f', '%+.2f', '%+.2f', '%+.2f', '%+.3f'};
nMet = numel(metric_names);
mi   = @(nm) find(strcmp(metric_names, nm));
put  = @(v) reshape(v, 1, 1, []);

nP = numel(patients);
nC = size(combos, 1);
sc_names = [{'Referencia', 'Sin imputacion', 'LOCF'}, combos(:, 2)'];
nSc = numel(sc_names);

%% ===================== RESERVA DE MEMORIA =====================
PM   = nan(nP, nSc, nMet);     % media sobre corridas, por paciente
PS   = nan(nP, nSc, nMet);     % DE sobre corridas (solo escenarios de pipeline)
PD   = nan(nP, nSc, nMet);     % delta contra la referencia (media sobre corridas)
NEV  = nan(nP, nSc);           % muestras con dato en cada escenario
CONC = nan(nP, nSc, 2);        % concordancia de categoria [desconexion, ruido]
COMP = nan(nP, 7);             % composicion de las fallas
chk  = {};                     % verificacion contra M guardado

%% ===================== BUCLE POR PACIENTE =====================
for p = 1:nP
    pid = patients{p};

    % ---- Cargar las combinaciones del paciente ----
    S = cell(nC, 1);
    for c = 1:nC
        f = fullfile(results_dir, sprintf('results_ohio_%s_%s%s_latest.mat', pid, combos{c,1}, res_tag));
        if ~exist(f, 'file')
            error('No se encontro: %s', f);
        end
        S{c} = load(f);
        req  = {'g_final_all', 'g_test_real', 'fault_type', 'k_start'};
        miss = req(~isfield(S{c}, req));
        if ~isempty(miss)
            error('%s\nFaltan campos: %s\nCampos disponibles: %s', f, ...
                strjoin(miss, ', '), strjoin(fieldnames(S{c})', ', '));
        end
    end

    % ---- Referencia, tipo de muestra y mascara de evaluacion ----
    ref = double(S{1}.g_test_real(:).');
    ft  = double(S{1}.fault_type(:).');           % 0 limpia | 1 hueco | 2 ruido | 3 desconexion
    T   = numel(ref);
    for c = 2:nC
        if size(S{c}.g_final_all, 2) ~= T || ~isequal(ft, double(S{c}.fault_type(:).'))
            error('Paciente %s: las combinaciones no comparten ventana o fallas.', pid);
        end
    end

    k0 = max(cellfun(@(s) double(s.k_start), S));
    idx_eval = ~isnan(ref) & (ref >= MIN_VALID);
    idx_eval(1:k0 - 1) = false;

    fmask     = (ft >= 2);                        % fallas sinteticas (ruido y desconexion)
    idx_noise = idx_eval & (ft == 2);
    idx_disc  = idx_eval & (ft == 3);

    % ---- Senales de cada escenario (filas = corridas) ----
    sig = cell(nSc, 1);
    sig{1} = ref;
    sig{2} = ref;  sig{2}(fmask) = NaN;
    sig{3} = locf_fill(ref, fmask);
    for c = 1:nC
        sig{3 + c} = double(S{c}.g_final_all);
    end
    for s = 1:nSc
        sig{s}(:, ~idx_eval) = NaN;               % misma base de muestras para todos
    end

    % ---- Indicadores por corrida -> media/DE por paciente ----
    Mref = clin_rows(sig{1}, thr);
    for s = 1:nSc
        Ms = clin_rows(sig{s}, thr);
        PM(p, s, :) = put(mean(Ms, 1));
        if size(Ms, 1) > 1
            PS(p, s, :) = put(std(Ms, 0, 1));
        end
        PD(p, s, :) = put(mean(Ms - Mref, 1));
        NEV(p, s)   = mean(sum(~isnan(sig{s}), 2));
    end

    % ---- Concordancia de categoria dentro de las fallas ----
    cls_d = range_class(ref(idx_disc), thr);
    cls_n = range_class(ref(idx_noise), thr);
    for s = 3:nSc
        if any(idx_disc)
            CONC(p, s, 1) = mean(100 * mean(range_class(sig{s}(:, idx_disc), thr) == cls_d, 2));
        end
        if any(idx_noise)
            CONC(p, s, 2) = mean(100 * mean(range_class(sig{s}(:, idx_noise), thr) == cls_n, 2));
        end
    end

    % ---- Composicion de las fallas ----
    ref_f = ref(idx_disc | idx_noise);
    COMP(p, :) = [sum(idx_eval), sum(idx_noise), sum(idx_disc), ...
                  100 * mean(double(fmask(idx_eval))), ...
                  sum(ref_f > thr.H1), sum(ref_f > thr.H2), mean(ref_f)];

    % ---- Verificacion: MAE en desconexion contra lo guardado en M ----
    for s = 3:nSc
        mine   = mean(mean(abs(sig{s}(:, idx_disc) - ref(idx_disc)), 2));
        stored = NaN;
        if s == 3
            if isfield(S{1}, 'M') && isfield(S{1}.M, 'mae_locf_disc')
                stored = mean(S{1}.M.mae_locf_disc, 'omitnan');
            end
        else
            if isfield(S{s - 3}, 'M') && isfield(S{s - 3}.M, 'mae_d3_disc')
                stored = mean(S{s - 3}.M.mae_d3_disc, 'omitnan');
            end
        end
        chk(end + 1, :) = {pid, sc_names{s}, mine, stored}; %#ok<SAGROW>
    end
end

%% ===================== AGREGADO ENTRE PACIENTES =====================
AM  = squeeze(mean(PM, 1));      % nSc x nMet
AS  = squeeze(std(PM, 0, 1));
AD  = squeeze(mean(PD, 1));
ADS = squeeze(std(PD, 0, 1));
ACM = squeeze(mean(CONC, 1));    % nSc x 2
ACS = squeeze(std(CONC, 0, 1));

%% ===================== IMPRESION EN CONSOLA =====================
d_rows = [mi('TIR'), mi('TBR_L1'), mi('TAR_L1'), mi('TAR_L2'), mi('CV'), mi('GMI')];

fprintf('\n=====================================================================\n');
fprintf(' COMPOSICION DE LAS FALLAS (muestras evaluadas de cada ventana)\n');
fprintf('=====================================================================\n');
fprintf('%-6s %8s %7s %9s %9s %10s %10s %11s\n', 'Pac.', 'n_eval', 'ruido', 'desconex', ...
    'fallas %', 'fallas>180', 'fallas>250', 'g ref media');
for p = 1:nP
    fprintf('%-6s %8d %7d %9d %9.1f %10d %10d %11.1f\n', patients{p}, COMP(p, 1), ...
        COMP(p, 2), COMP(p, 3), COMP(p, 4), COMP(p, 5), COMP(p, 6), COMP(p, 7));
end

for p = 1:nP
    fprintf('\n=====================================================================\n');
    fprintf(' PACIENTE %s | pipeline = media de %d corridas\n', patients{p}, size(S{1}.g_final_all, 1));
    fprintf('=====================================================================\n');
    print_table('Indicadores', sc_names, metric_labels, squeeze(PM(p, :, :))', [], fmt_abs);
    Vd = squeeze(PD(p, :, :))';                   % nMet x nSc
    print_table('Delta contra referencia (pp)', sc_names, ...
        metric_labels(d_rows), Vd(d_rows, :), [], fmt_dlt(d_rows));
    print_table('Concordancia de categoria dentro de las fallas (%)', sc_names(3:end), ...
        {'Desconexion', 'Ruido'}, squeeze(CONC(p, 3:end, :))', [], {'%.1f', '%.1f'});
end

fprintf('\n=====================================================================\n');
fprintf(' AGREGADO ENTRE PACIENTES (media +/- DE de las medias por paciente, n = %d)\n', nP);
fprintf('=====================================================================\n');
print_table('Indicadores', sc_names, metric_labels, AM', AS', fmt_abs);
Vd  = AD';   Vds = ADS';                      % nMet x nSc
print_table('Delta contra referencia (pp)', sc_names, metric_labels(d_rows), ...
    Vd(d_rows, :), Vds(d_rows, :), fmt_dlt(d_rows));
print_table('Concordancia de categoria dentro de las fallas (%)', sc_names(3:end), ...
    {'Desconexion', 'Ruido'}, ACM(3:end, :)', ACS(3:end, :)', {'%.1f', '%.1f'});

fprintf('\nVERIFICACION: MAE en desconexion (mg/dL), calculado aqui | guardado en M\n');
for i = 1:size(chk, 1)
    flag = '';
    if ~isnan(chk{i, 4}) && abs(chk{i, 3} - chk{i, 4}) > 0.05
        flag = '   <-- REVISAR';
    end
    fprintf('  %-4s %-10s %7.2f | %7.2f%s\n', chk{i, 1}, chk{i, 2}, chk{i, 3}, chk{i, 4}, flag);
end

%% ===================== FIGURA: BARRAS APILADAS (AGP) =====================
zone_colors = [0.85 0.20 0.20; 0.95 0.60 0.10; 0.25 0.70 0.35; 0.95 0.60 0.10; 0.85 0.20 0.20];
sc_short    = strrep(sc_names, 'Sin imputacion', 'Sin imp.');

fig = figure('Color', 'w', 'Position', [60 60 1400 480]);
for p = 1:nP
    ax = subplot(1, nP, p);  hold(ax, 'on');
    bd = zeros(nSc, 5);
    for s = 1:nSc
        bd(s, :) = [PM(p, s, mi('TBR_L2')), PM(p, s, mi('TBR_L1')) - PM(p, s, mi('TBR_L2')), ...
                    PM(p, s, mi('TIR')), PM(p, s, mi('TAR_L1')), PM(p, s, mi('TAR_L2'))];
    end
    bh = bar(ax, bd, 'stacked', 'BarWidth', 0.65);
    for z = 1:5
        bh(z).FaceColor = zone_colors(z, :);
        bh(z).EdgeColor = 'w';
    end
    for s = 1:nSc
        text(ax, s, bd(s, 1) + bd(s, 2) + bd(s, 3) / 2, sprintf('%.1f', bd(s, 3)), ...
            'HorizontalAlignment', 'center', 'Color', 'w', 'FontWeight', 'bold', 'FontSize', 9);
    end
    ylim(ax, [0 100]);
    set(ax, 'XTick', 1:nSc, 'XTickLabel', sc_short, 'XTickLabelRotation', 30, 'FontSize', 10);
    title(ax, sprintf('Paciente %s', patients{p}));
    if p == 1, ylabel(ax, 'Porcentaje de tiempo (%)'); end
    grid(ax, 'on');  box(ax, 'on');
    if p == nP
        legend(bh, {'TBR L2 (<54)', 'TBR L1 (54-69)', 'TIR (70-180)', ...
            'TAR L1 (181-250)', 'TAR L2 (>250)'}, 'Location', 'eastoutside');
    end
end
exportgraphics(fig, fullfile(fig_dir, ['ohio_clinical_AGP' res_tag '.eps']), 'ContentType', 'vector');
exportgraphics(fig, fullfile(fig_dir, ['ohio_clinical_AGP' res_tag '.png']), 'Resolution', 300);

%% ===================== GUARDAR RESULTADOS =====================
% ---- Por paciente y escenario ----
rows = {};  rows_d = {};
for p = 1:nP
    for s = 1:nSc
        rows(end + 1, :)   = [{patients{p}, sc_names{s}, NEV(p, s)}, ...
            num2cell(squeeze(PM(p, s, :))'), num2cell(squeeze(PS(p, s, :))')]; %#ok<SAGROW>
        rows_d(end + 1, :) = [{patients{p}, sc_names{s}}, ...
            num2cell(squeeze(PD(p, s, :))')]; %#ok<SAGROW>
    end
end
T_pat = cell2table(rows, 'VariableNames', [{'paciente', 'escenario', 'n_muestras'}, ...
    metric_names, strcat('DE_', metric_names)]);
T_del = cell2table(rows_d, 'VariableNames', [{'paciente', 'escenario'}, strcat('delta_', metric_names)]);

% ---- Agregado entre pacientes ----
rows_a = {};
for s = 1:nSc
    rows_a(end + 1, :) = [{sc_names{s}}, num2cell(AM(s, :)), num2cell(AS(s, :)), ...
        num2cell(AD(s, :)), num2cell(ADS(s, :))]; %#ok<SAGROW>
end
T_agg = cell2table(rows_a, 'VariableNames', [{'escenario'}, metric_names, ...
    strcat('DE_', metric_names), strcat('delta_', metric_names), strcat('DEdelta_', metric_names)]);

% ---- Concordancia y composicion ----
rows_c = {};
for p = 1:nP
    for s = 3:nSc
        rows_c(end + 1, :) = {patients{p}, sc_names{s}, CONC(p, s, 1), CONC(p, s, 2)}; %#ok<SAGROW>
    end
end
for s = 3:nSc
    rows_c(end + 1, :) = {'MEDIA', sc_names{s}, ACM(s, 1), ACM(s, 2)}; %#ok<SAGROW>
    rows_c(end + 1, :) = {'DE',    sc_names{s}, ACS(s, 1), ACS(s, 2)}; %#ok<SAGROW>
end
T_con = cell2table(rows_c, 'VariableNames', {'paciente', 'escenario', 'conc_desconexion_pct', 'conc_ruido_pct'});

T_com = array2table(COMP, 'VariableNames', {'n_eval', 'n_ruido', 'n_desconexion', 'fallas_pct', ...
    'fallas_ref_gt180', 'fallas_ref_gt250', 'g_ref_media_fallas'});
T_com = addvars(T_com, patients(:), 'Before', 1, 'NewVariableNames', 'paciente');

xlsx_file = fullfile(results_dir, ['ohio_clinical_metrics' res_tag '.xlsx']);
if exist(xlsx_file, 'file'), delete(xlsx_file); end
writetable(T_pat, xlsx_file, 'Sheet', 'PorPaciente');
writetable(T_del, xlsx_file, 'Sheet', 'DeltaPorPaciente');
writetable(T_agg, xlsx_file, 'Sheet', 'Agregado');
writetable(T_con, xlsx_file, 'Sheet', 'Concordancia');
writetable(T_com, xlsx_file, 'Sheet', 'ComposicionFallas');

mat_file = fullfile(results_dir, ['ohio_clinical_metrics' res_tag '.mat']);
save(mat_file, 'PM', 'PS', 'PD', 'AM', 'AS', 'AD', 'ADS', 'CONC', 'ACM', 'ACS', 'COMP', 'NEV', ...
    'sc_names', 'metric_names', 'patients', 'combos', 'res_tag', 'thr', 'MIN_VALID');

fprintf('\nGuardado:\n   %s\n   %s\n   %s\n', xlsx_file, mat_file, fullfile(fig_dir, ['ohio_clinical_AGP' res_tag '.eps']));


%% ===================== FUNCIONES LOCALES =====================

function M = clin_rows(X, thr)
%CLIN_ROWS Indicadores clinicos de cada fila de X (NaN = sin dato).
%   Columnas: TIR, TBR_L1, TBR_L2, TAR_L1, TAR_L2, Media, SD, CV, GMI.
    n = size(X, 1);
    M = nan(n, 9);
    for i = 1:n
        x = X(i, ~isnan(X(i, :)));
        if isempty(x), continue; end
        mu = mean(x);
        sd = std(x);
        M(i, :) = [100 * mean(x >= thr.L1 & x <= thr.H1), ...
                   100 * mean(x <  thr.L1), ...
                   100 * mean(x <  thr.L2), ...
                   100 * mean(x >  thr.H1 & x <= thr.H2), ...
                   100 * mean(x >  thr.H2), ...
                   mu, sd, 100 * sd / mu, 3.31 + 0.02392 * mu];
    end
end

function c = range_class(x, thr)
%RANGE_CLASS Categoria de rango: 1 <54 | 2 54-69 | 3 70-180 | 4 181-250 | 5 >250.
    c = 1 + (x >= thr.L2) + (x >= thr.L1) + (x > thr.H1) + (x > thr.H2);
end

function g = locf_fill(ref, fmask)
%LOCF_FILL Sustituye cada segmento de falla por la ultima lectura limpia previa.
    g = ref;
    [gs, ge] = find_runs(fmask);
    for r = 1:numel(gs)
        j = gs(r) - 1;
        while j >= 1 && (isnan(ref(j)) || fmask(j))
            j = j - 1;
        end
        if j >= 1
            g(gs(r):ge(r)) = ref(j);
        end
    end
end

function [gs, ge] = find_runs(mask)
%FIND_RUNS Inicio y fin de cada tramo contiguo de 1 en mask.
    mask = logical(mask(:).');
    d  = diff([0 mask 0]);
    gs = find(d == 1);
    ge = find(d == -1) - 1;
end

function print_table(ttl, col_names, row_labels, V, SD, fmts)
%PRINT_TABLE Tabla en consola: filas = indicadores, columnas = escenarios.
%   SD vacio -> solo valores; SD con datos -> "valor +/- DE".
    w = 20;
    fprintf('\n%s\n', ttl);
    fprintf('%-18s', '');
    for s = 1:numel(col_names)
        fprintf('%*s', w, col_names{s});
    end
    fprintf('\n%s\n', repmat('-', 1, 18 + w * numel(col_names)));
    for r = 1:size(V, 1)
        fprintf('%-18s', row_labels{r});
        for s = 1:size(V, 2)
            if isnan(V(r, s))
                str = 'n/d';
            elseif isempty(SD) || isnan(SD(r, s))
                str = sprintf(fmts{r}, V(r, s));
            else
                str = sprintf([fmts{r} ' +/- ' strrep(fmts{r}, '+', '')], V(r, s), SD(r, s));
            end
            fprintf('%*s', w, str);
        end
        fprintf('\n');
    end
end