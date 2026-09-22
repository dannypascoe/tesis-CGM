%% =========================================================================
%  SCAN_OHIO_CANDIDATES.M - Escaneo diagnostico de pacientes OhioT1DM
%  =========================================================================
%  Descripcion: Corre la MISMA logica de parseo, union y busqueda de
%               ventana de main_ohio_window_selection_v3.m sobre una lista
%               de pacientes candidatos, SIN guardar ningun .mat. Solo
%               produce una tabla comparativa (% de huecos de la mejor
%               ventana de 5 dias, dias totales de registro, densidad de
%               comidas/bolos) para decidir cuales pacientes preparar de
%               verdad con main_ohio_window_selection_v3.m.
%
%  Demografia (Marling & Bunescu, 2020 -- "The OhioT1DM Dataset for Blood
%  Glucose Level Prediction: Update 2020"), incluida aqui como referencia
%  para la decision, NO leida de ningun archivo del dataset:
%    2020: 540(M,20-40) 544(M,40-60) 552(M,20-40) 567(F,20-40) 584(M,40-60) 596(M,60-80)
%    2018: 559(F,40-60) 563(M,40-60) 570(M,40-60) 575(F,40-60) 588(F,40-60) 591(F,40-60)
%    (2018 no tiene ningun paciente fuera de 40-60; el rango 60-80 SOLO
%    existe en 596. 2020 tiene sesgo de genero 5H/1M; 2018 esta balanceado
%    2H/4M pero sin variedad de edad.)
%
%  Ya preparados (main_ohio_window_selection_v3.m, con pre-roll y
%  calibracion causal): 559 (2018, F, 40-60), 540 (2020, M, 20-40).
%  =========================================================================

clear; clc; close all;

%% ===================== CONFIGURACION =====================

% Candidatos a escanear (ajusta la lista si solo quieres probar algunos)
candidates = { ...
    '596', '2020', 'M', '60-80'; ...   % unico paciente en 60-80
    '544', '2020', 'M', '40-60'; ...
    '552', '2020', 'M', '20-40'; ...
    '567', '2020', 'F', '20-40'; ...
    '584', '2020', 'M', '40-60'; ...
    '563', '2018', 'M', '40-60'; ...
    '570', '2018', 'M', '40-60'; ...
    '575', '2018', 'F', '40-60'; ...
    '588', '2018', 'F', '40-60'; ...
    '591', '2018', 'F', '40-60'; ...
};

window_days = 5;
GRID_TOL_MIN = 2.5;
DT_MINUTES = 5;

ohio_root = 'D:\Documentos\02 MROI\07 Tesis\03 Extension Tesis\OhioT1DM\OhioT1DM';  % <<< tu ruta

%% ===================== ESCANEAR CADA CANDIDATO =====================

nC = size(candidates, 1);
patient_id_col = candidates(:,1);
cohort_col     = candidates(:,2);
gender_col     = candidates(:,3);
age_col        = candidates(:,4);
pct_missing_best = nan(nC,1);
window_start   = NaT(nC,1);
window_end     = NaT(nC,1);
total_days     = nan(nC,1);
pct_missing_total = nan(nC,1);
n_meals        = nan(nC,1);
n_bolus        = nan(nC,1);
n_basal_chg    = nan(nC,1);
found          = false(nC,1);

for i = 1:nC
    pid = candidates{i,1};
    yr  = candidates{i,2};
    train_file = fullfile(ohio_root, yr, 'train', sprintf('%s-ws-training.xml', pid));
    test_file  = fullfile(ohio_root, yr, 'test',  sprintf('%s-ws-testing.xml',  pid));

    if ~exist(train_file, 'file') || ~exist(test_file, 'file')
        fprintf('[%s] archivos no encontrados, se omite.\n', pid);
        continue;
    end

    fprintf('[%s] parseando...\n', pid);
    d_train = parse_ohio_patient_xml(train_file);
    d_test  = parse_ohio_patient_xml(test_file);
    merged  = merge_ohio_records(d_train, d_test);

    [grid_ts, ~, gap_mask] = build_glucose_grid(merged.glucose, DT_MINUTES, GRID_TOL_MIN);
    [best, ~] = find_cleanest_window(grid_ts, gap_mask, window_days, DT_MINUTES);

    found(i)             = true;
    pct_missing_best(i)  = best.pct_missing;
    window_start(i)      = best.start_ts;
    window_end(i)        = best.end_ts;
    total_days(i)        = days(grid_ts(end) - grid_ts(1));
    pct_missing_total(i) = 100*sum(gap_mask)/numel(gap_mask);
    n_meals(i)            = numel(merged.meal.ts);
    n_bolus(i)             = numel(merged.bolus.ts_begin);
    n_basal_chg(i)          = numel(merged.basal.ts);
end

%% ===================== TABLA COMPARATIVA =====================

T = table(patient_id_col, cohort_col, gender_col, age_col, ...
    pct_missing_best, window_start, window_end, ...
    total_days, pct_missing_total, n_meals, n_bolus, n_basal_chg, ...
    'VariableNames', {'paciente','cohorte','sexo','edad', ...
    'pct_huecos_mejor_ventana','ventana_inicio','ventana_fin', ...
    'dias_totales','pct_huecos_total','n_comidas','n_bolos','n_cambios_basal'});

T = T(found, :);
T = sortrows(T, 'pct_huecos_mejor_ventana');

fprintf('\n=== CANDIDATOS ORDENADOS POR %% DE HUECOS (mejor ventana de %d dias) ===\n', window_days);
disp(T);

fprintf('\nYa preparados: 559 (2018,F,40-60) y 540 (2020,M,20-40) -- no incluidos en este escaneo.\n');

%% ===================== FUNCIONES AUXILIARES (identicas a v3, self-contenido) =====================

function data = parse_ohio_patient_xml(filepath)
%PARSE_OHIO_PATIENT_XML Lee un XML de OhioT1DM y extrae las secciones
%   necesarias para el pipeline: glucose_level, basal, temp_basal, bolus, meal.

    doc  = xmlread(filepath);
    root = doc.getDocumentElement();

    data = struct();
    data.patient_id = char(root.getAttribute('id'));

    data.glucose    = parse_simple_events(root, 'glucose_level', 'value');
    data.basal      = parse_simple_events(root, 'basal',         'value');
    data.meal       = parse_meal_events(root);
    data.temp_basal = parse_ranged_events(root, 'temp_basal', {'value'});
    data.bolus      = parse_bolus_events(root);
end

function out = parse_simple_events(root, section_name, value_attr)
%PARSE_SIMPLE_EVENTS Parsea eventos <event ts="..." <value_attr>="..."/>
%   dentro de <section_name>...</section_name>. Sirve para glucose_level y basal.
    out.ts    = datetime.empty(0, 1);
    out.value = [];

    sections = root.getElementsByTagName(section_name);
    if sections.getLength() == 0
        return;
    end
    events = sections.item(0).getElementsByTagName('event');
    n = events.getLength();

    ts    = NaT(n, 1);
    value = nan(n, 1);

    for i = 0:n-1
        ev = events.item(i);
        ts(i+1)    = parse_ohio_datetime(char(ev.getAttribute('ts')));
        value(i+1) = str2double(char(ev.getAttribute(value_attr)));
    end

    [ts, order] = sort(ts);
    value = value(order);

    out.ts    = ts;
    out.value = value;
end

function out = parse_meal_events(root)
%PARSE_MEAL_EVENTS Igual que parse_simple_events pero conserva 'type' y 'carbs'.
    out.ts    = datetime.empty(0, 1);
    out.carbs = [];
    out.type  = {};

    sections = root.getElementsByTagName('meal');
    if sections.getLength() == 0
        return;
    end
    events = sections.item(0).getElementsByTagName('event');
    n = events.getLength();

    ts    = NaT(n, 1);
    carbs = nan(n, 1);
    type  = cell(n, 1);

    for i = 0:n-1
        ev = events.item(i);
        ts(i+1)    = parse_ohio_datetime(char(ev.getAttribute('ts')));
        carbs(i+1) = str2double(char(ev.getAttribute('carbs')));
        type{i+1}  = char(ev.getAttribute('type'));
    end

    [ts, order] = sort(ts);
    out.ts    = ts;
    out.carbs = carbs(order);
    out.type  = type(order);
end

function out = parse_ranged_events(root, section_name, extra_attrs)
%PARSE_RANGED_EVENTS Parsea eventos con ts_begin/ts_end (ej. temp_basal).
    out.ts_begin = datetime.empty(0, 1);
    out.ts_end   = datetime.empty(0, 1);
    for a = 1:numel(extra_attrs)
        out.(extra_attrs{a}) = [];
    end

    sections = root.getElementsByTagName(section_name);
    if sections.getLength() == 0
        return;
    end
    events = sections.item(0).getElementsByTagName('event');
    n = events.getLength();

    ts_begin   = NaT(n, 1);
    ts_end     = NaT(n, 1);
    extra_vals = nan(n, numel(extra_attrs));

    for i = 0:n-1
        ev = events.item(i);
        ts_begin(i+1) = parse_ohio_datetime(char(ev.getAttribute('ts_begin')));
        ts_end(i+1)   = parse_ohio_datetime(char(ev.getAttribute('ts_end')));
        for a = 1:numel(extra_attrs)
            extra_vals(i+1, a) = str2double(char(ev.getAttribute(extra_attrs{a})));
        end
    end

    [ts_begin, order] = sort(ts_begin);
    out.ts_begin = ts_begin;
    out.ts_end   = ts_end(order);
    for a = 1:numel(extra_attrs)
        out.(extra_attrs{a}) = extra_vals(order, a);
    end
end

function out = parse_bolus_events(root)
%PARSE_BOLUS_EVENTS Igual que parse_ranged_events pero conserva type, dose
%   y bwz_carb_input (estimado de carbohidratos usado en el bolus wizard;
%   puede diferir del campo 'carbs' reportado en <meal>).
    out.ts_begin       = datetime.empty(0, 1);
    out.ts_end         = datetime.empty(0, 1);
    out.type           = {};
    out.dose           = [];
    out.bwz_carb_input = [];

    sections = root.getElementsByTagName('bolus');
    if sections.getLength() == 0
        return;
    end
    events = sections.item(0).getElementsByTagName('event');
    n = events.getLength();

    ts_begin = NaT(n, 1);
    ts_end   = NaT(n, 1);
    type     = cell(n, 1);
    dose     = nan(n, 1);
    carbin   = nan(n, 1);

    for i = 0:n-1
        ev = events.item(i);
        ts_begin(i+1) = parse_ohio_datetime(char(ev.getAttribute('ts_begin')));
        ts_end(i+1)   = parse_ohio_datetime(char(ev.getAttribute('ts_end')));
        type{i+1}     = char(ev.getAttribute('type'));
        dose(i+1)     = str2double(char(ev.getAttribute('dose')));
        carb_str = char(ev.getAttribute('bwz_carb_input'));
        if ~isempty(carb_str)
            carbin(i+1) = str2double(carb_str);
        end
    end

    [ts_begin, order] = sort(ts_begin);
    out.ts_begin       = ts_begin;
    out.ts_end         = ts_end(order);
    out.type           = type(order);
    out.dose           = dose(order);
    out.bwz_carb_input = carbin(order);
end

function dt = parse_ohio_datetime(ts_str)
%PARSE_OHIO_DATETIME Convierte 'dd-MM-yyyy HH:mm:ss' (formato confirmado
%   en el XML de OhioT1DM) a datetime de MATLAB.
    dt = datetime(ts_str, 'InputFormat', 'dd-MM-yyyy HH:mm:ss');
end

function merged = merge_ohio_records(d_train, d_test)
%MERGE_OHIO_RECORDS Concatena train+test en un solo registro cronologico,
%   eliminando duplicados exactos de timestamp en la frontera train/test.
    merged = struct();
    merged.patient_id = d_train.patient_id;

    % --- Campos simples (ts + value): glucose, basal ---
    fields_simple = {'glucose', 'basal'};
    for f = 1:numel(fields_simple)
        fn = fields_simple{f};
        ts_all    = [d_train.(fn).ts;    d_test.(fn).ts];
        value_all = [d_train.(fn).value; d_test.(fn).value];

        [ts_dedup, uidx]    = unique(ts_all, 'stable');
        [ts_sorted, order]  = sort(ts_dedup);
        idx = uidx(order);

        merged.(fn).ts    = ts_sorted;
        merged.(fn).value = value_all(idx);
    end

    % --- meal (conserva type y carbs) ---
    ts_all    = [d_train.meal.ts;    d_test.meal.ts];
    carbs_all = [d_train.meal.carbs; d_test.meal.carbs];
    type_all  = [d_train.meal.type;  d_test.meal.type];

    [ts_dedup, uidx]   = unique(ts_all, 'stable');
    [ts_sorted, order] = sort(ts_dedup);
    idx = uidx(order);

    merged.meal.ts    = ts_sorted;
    merged.meal.carbs = carbs_all(idx);
    merged.meal.type  = type_all(idx);

    % --- temp_basal y bolus (ts_begin/ts_end) ---
    merged.temp_basal = merge_ranged(d_train.temp_basal, d_test.temp_basal, {'value'}, {});
    merged.bolus       = merge_ranged(d_train.bolus, d_test.bolus, {'dose', 'bwz_carb_input'}, {'type'});
end

function out = merge_ranged(a, b, numeric_fields, text_fields)
%MERGE_RANGED Concatena y deduplica dos structs con ts_begin/ts_end
%   (temp_basal o bolus), conservando campos numericos y de texto asociados.
    ts_begin_all = [a.ts_begin; b.ts_begin];
    ts_end_all   = [a.ts_end;   b.ts_end];

    [ts_dedup, uidx]    = unique(ts_begin_all, 'stable');
    [ts_sorted, order]  = sort(ts_dedup);
    idx = uidx(order);

    out.ts_begin = ts_sorted;
    out.ts_end   = ts_end_all(idx);
    for f = 1:numel(numeric_fields)
        vals_all = [a.(numeric_fields{f}); b.(numeric_fields{f})];
        out.(numeric_fields{f}) = vals_all(idx);
    end
    for f = 1:numel(text_fields)
        vals_all = [a.(text_fields{f}); b.(text_fields{f})];
        out.(text_fields{f}) = vals_all(idx);
    end
end

function [grid_ts, grid_glucose, gap_mask] = build_glucose_grid(glucose, dt_minutes, tol_minutes)
%BUILD_GLUCOSE_GRID Reindexa los eventos de glucosa a una rejilla uniforme
%   de dt_minutes (alineada a medianoche), dejando NaN donde no hay lectura
%   real suficientemente cercana. gap_mask == true marca un hueco.
    t0 = dateshift(glucose.ts(1), 'start', 'day');   % alinear a medianoche
    t1 = glucose.ts(end);

    grid_ts = (t0 : minutes(dt_minutes) : t1)';
    n = numel(grid_ts);
    grid_glucose = nan(n, 1);

    offsets_min = minutes(glucose.ts - t0);
    idx = round(offsets_min / dt_minutes) + 1;
    residual = abs(offsets_min - (idx - 1) * dt_minutes);

    valid = idx >= 1 & idx <= n & residual <= tol_minutes;
    grid_glucose(idx(valid)) = glucose.value(valid);

    gap_mask = isnan(grid_glucose);
end

function report_longest_gaps(grid_ts, gap_mask, dt_minutes, top_n)
%REPORT_LONGEST_GAPS Imprime los N huecos mas largos (estilo Tabla 5 del
%   paper publicado), util para inspeccionar antes de elegir la ventana.
    d = diff([0; gap_mask; 0]);
    starts = find(d == 1);
    ends   = find(d == -1) - 1;

    if isempty(starts)
        fprintf('No se detectaron huecos en el registro.');
        return;
    end

    durations_min = (ends - starts + 1) * dt_minutes;
    [durations_sorted, order] = sort(durations_min, 'descend');
    n_show = min(top_n, numel(durations_sorted));

    fprintf('Top %d huecos mas largos:', n_show);
    for i = 1:n_show
        k = order(i);
        fprintf('  %s -> %s | %.1f h', ...
            datestr(grid_ts(starts(k))), datestr(grid_ts(ends(k))), durations_sorted(i) / 60);
    end
end

function [best, ranking] = find_cleanest_window(grid_ts, gap_mask, window_days, dt_minutes)
%FIND_CLEANEST_WINDOW Evalua todas las ventanas de window_days alineadas a
%   medianoche y regresa la de menor % de huecos, junto con el ranking
%   completo (ordenado de mejor a peor) para revision manual.
    samples_per_day = 24 * 60 / dt_minutes;
    window_samples  = window_days * samples_per_day;

    day_starts = find(grid_ts.Hour == 0 & grid_ts.Minute == 0 & grid_ts.Second == 0);

    start_ts    = NaT(0, 1);
    end_ts      = NaT(0, 1);
    pct_missing = [];

    for i = 1:numel(day_starts)
        idx0 = day_starts(i);
        idx1 = idx0 + window_samples - 1;
        if idx1 > numel(gap_mask)
            continue;
        end
        start_ts(end+1, 1)    = grid_ts(idx0);                                        %#ok<AGROW>
        end_ts(end+1, 1)      = grid_ts(idx1);                                        %#ok<AGROW>
        pct_missing(end+1, 1) = 100 * sum(gap_mask(idx0:idx1)) / window_samples;       %#ok<AGROW>
    end

    if isempty(start_ts)
        error('El registro es mas corto que la ventana solicitada (%d dias).', window_days);
    end

    ranking = table(start_ts, end_ts, pct_missing);
    ranking = sortrows(ranking, 'pct_missing');

    best = table2struct(ranking(1, :));
end