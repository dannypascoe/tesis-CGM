%% =========================================================================
%  MAIN_OHIO_WINDOW_SELECTION_V1.M
%  =========================================================================
%  Descripcion: Carga los archivos train+test de OhioT1DM para UN paciente,
%               los une en un solo registro cronologico, detecta huecos
%               reales en la senal de glucosa (CGM) y selecciona la ventana
%               de N dias consecutivos con menor proporcion de datos
%               faltantes. Guarda el subconjunto crudo (glucosa, basal,
%               temp_basal, bolus, meal) de esa ventana para uso posterior.
%
%  IMPORTANTE: este script NO inyecta fallos ni construye u(k)/m(k) todavia.
%              Es un paso previo a load_and_prepare_data_ohio.m: aqui solo
%              se elige y recorta el tramo de datos reales a usar.
%
%  Formato de fecha confirmado en el XML: 'dd-MM-yyyy HH:mm:ss'
%
%  Prerrequisitos:
%    - Archivos <ID>-ws-training.xml y <ID>-ws-testing.xml del dataset
%      OhioT1DM (requiere Data Use Agreement con Ohio University).
%
%  Salida:
%    - data/ohio/prepared/paciente_<ID>_ventana<N>d_<timestamp>.mat
%    - data/ohio/prepared/paciente_<ID>_ventana<N>d_latest.mat
%  =========================================================================

clear; clc; close all;

%% ===================== CONFIGURACION =====================

% ID del paciente a procesar (elegir manualmente)
patient_id = '559';        % <<< CAMBIAR AQUI

% Duracion de la ventana a seleccionar, en dias
window_days = 5;

% Tolerancia para considerar un evento de glucosa "en su lugar" en la rejilla
% de 5 minutos (minutos). Si el timestamp real se desvia mas que esto del
% punto de rejilla mas cercano, ese punto de rejilla queda como hueco.
GRID_TOL_MIN = 2.5;

DT_MINUTES = 5;   % cadencia nominal del CGM

% Rutas base del dataset OhioT1DM (MODIFICAR SEGUN TU UBICACION)
ohio_root = 'D:\Documentos\02 MROI\07 Tesis\03 Extension Tesis\OhioT1DM\OhioT1DM';
years_to_try = {'2018', '2020'};

% Ruta de salida dentro del proyecto (PARALELA a bloque_A..D, no dentro de ellos)
scriptPath  = 'D:\Documentos\02 MROI\07 Tesis\03 Extension Tesis\CGM_Project\CGM_Project';
output_path = fullfile(scriptPath, 'data', 'ohio', 'prepared');
if ~exist(output_path, 'dir')
    mkdir(output_path);
    fprintf('Carpeta creada: %s\n', output_path);
end

%% ===================== LOCALIZAR ARCHIVOS DEL PACIENTE =====================

train_file   = '';
test_file    = '';
patient_year = '';

for y = 1:numel(years_to_try)
    yr = years_to_try{y};
    candidate_train = fullfile(ohio_root, yr, 'train', sprintf('%s-ws-training.xml', patient_id));
    candidate_test  = fullfile(ohio_root, yr, 'test',  sprintf('%s-ws-testing.xml',  patient_id));
    if exist(candidate_train, 'file') && exist(candidate_test, 'file')
        train_file   = candidate_train;
        test_file    = candidate_test;
        patient_year = yr;
        break;
    end
end

if isempty(train_file)
    error('No se encontraron los archivos del paciente %s en 2018 ni 2020. Revisa el ID o las rutas.', patient_id);
end

fprintf('Paciente %s localizado en cohorte %s\n', patient_id, patient_year);
fprintf('  Train: %s\n', train_file);
fprintf('  Test:  %s\n', test_file);

%% ===================== PARSEAR AMBOS ARCHIVOS =====================

fprintf('\nParseando XML...\n');
d_train = parse_ohio_patient_xml(train_file);
d_test  = parse_ohio_patient_xml(test_file);

fprintf('  Train: %d muestras glucosa (%s a %s)\n', numel(d_train.glucose.ts), ...
    datestr(d_train.glucose.ts(1)), datestr(d_train.glucose.ts(end)));
fprintf('  Test:  %d muestras glucosa (%s a %s)\n', numel(d_test.glucose.ts), ...
    datestr(d_test.glucose.ts(1)), datestr(d_test.glucose.ts(end)));

%% ===================== UNIR TRAIN + TEST =====================

merged = merge_ohio_records(d_train, d_test);

fprintf('\nRegistro unido: %d muestras de glucosa | %s a %s (%.1f dias)\n', ...
    numel(merged.glucose.ts), datestr(merged.glucose.ts(1)), datestr(merged.glucose.ts(end)), ...
    days(merged.glucose.ts(end) - merged.glucose.ts(1)));

%% ===================== CONSTRUIR REJILLA DE 5 MIN Y DETECTAR HUECOS =====================

[grid_ts, grid_glucose, gap_mask] = build_glucose_grid(merged.glucose, DT_MINUTES, GRID_TOL_MIN);

pct_missing_total = 100 * sum(gap_mask) / numel(gap_mask);
fprintf('\nCadencia esperada: %d min | Muestras en rejilla: %d\n', DT_MINUTES, numel(grid_ts));
fprintf('Datos faltantes totales: %.2f%%\n', pct_missing_total);

% Reportar los huecos mas largos (estilo Tabla 5 del paper publicado)
report_longest_gaps(grid_ts, gap_mask, DT_MINUTES, 5);

%% ===================== BUSCAR LA VENTANA DE N DIAS MAS LIMPIA =====================

[best_window, ranking] = find_cleanest_window(grid_ts, gap_mask, window_days, DT_MINUTES);

fprintf('\n=== TOP CANDIDATAS: ventanas de %d dias (menor %% de huecos) ===\n', window_days);
n_show = min(5, height(ranking));
for i = 1:n_show
    fprintf('  %d) %s -> %s | huecos: %.2f%%\n', i, ...
        datestr(ranking.start_ts(i)), datestr(ranking.end_ts(i)), ranking.pct_missing(i));
end

fprintf('\n>>> Ventana seleccionada automaticamente: %s -> %s (huecos: %.2f%%)\n', ...
    datestr(best_window.start_ts), datestr(best_window.end_ts), best_window.pct_missing);
fprintf('    Para tomar otra candidata de la lista de arriba, cambia "window_choice" abajo.\n');

% --- Permite anular la seleccion automatica y tomar otra fila del ranking ---
window_choice = 1;   % <<< 1 = la mejor (menor %% huecos); cambiar a 2, 3... para otra candidata
best_window = table2struct(ranking(window_choice, :));

%% ===================== RECORTAR Y GUARDAR EL SUBCONJUNTO CRUDO =====================

window_data = extract_ohio_window(merged, best_window.start_ts, best_window.end_ts);

meta = struct();
meta.patient_id   = patient_id;
meta.cohort_year  = patient_year;
meta.window_days  = window_days;
meta.start_ts     = best_window.start_ts;
meta.end_ts       = best_window.end_ts;
meta.pct_missing  = best_window.pct_missing;
meta.train_file   = train_file;
meta.test_file    = test_file;
meta.generated_on = datestr(now, 'yyyy-mm-dd HH:MM:SS');

timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
out_file_ts     = fullfile(output_path, sprintf('paciente_%s_ventana%dd_%s.mat', patient_id, window_days, timestamp));
out_file_latest = fullfile(output_path, sprintf('paciente_%s_ventana%dd_latest.mat', patient_id, window_days));

save(out_file_ts, 'window_data', 'meta');
save(out_file_latest, 'window_data', 'meta');

fprintf('\nGuardado: %s\n', out_file_ts);
fprintf('Guardado: %s\n', out_file_latest);


%% =========================================================================
%  FUNCIONES AUXILIARES
%  =========================================================================

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
        fprintf('\nNo se detectaron huecos en el registro.\n');
        return;
    end

    durations_min = (ends - starts + 1) * dt_minutes;
    [durations_sorted, order] = sort(durations_min, 'descend');
    n_show = min(top_n, numel(durations_sorted));

    fprintf('\nTop %d huecos mas largos:\n', n_show);
    for i = 1:n_show
        k = order(i);
        fprintf('  %s -> %s | %.1f h\n', ...
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

function window_data = extract_ohio_window(merged, start_ts, end_ts)
%EXTRACT_OHIO_WINDOW Recorta los eventos crudos de todas las senales al
%   rango [start_ts, end_ts]. Basal incluye tambien el ultimo evento ANTES
%   del inicio, para saber la tasa vigente al comenzar la ventana.
    window_data = struct();
    window_data.start_ts = start_ts;
    window_data.end_ts   = end_ts;

    window_data.glucose = crop_simple(merged.glucose, start_ts, end_ts);
    window_data.meal    = crop_simple(merged.meal,    start_ts, end_ts);

    idx_before = find(merged.basal.ts <= start_ts, 1, 'last');
    idx_in     = find(merged.basal.ts > start_ts & merged.basal.ts <= end_ts);
    idx_basal  = unique([idx_before; idx_in]);
    window_data.basal.ts    = merged.basal.ts(idx_basal);
    window_data.basal.value = merged.basal.value(idx_basal);

    window_data.temp_basal = crop_ranged(merged.temp_basal, start_ts, end_ts);
    window_data.bolus      = crop_ranged(merged.bolus, start_ts, end_ts);
end

function out = crop_simple(s, start_ts, end_ts)
%CROP_SIMPLE Recorta un struct con campo 'ts' a [start_ts, end_ts].
    idx = s.ts >= start_ts & s.ts <= end_ts;
    fns = fieldnames(s);
    for f = 1:numel(fns)
        out.(fns{f}) = s.(fns{f})(idx);
    end
end

function out = crop_ranged(s, start_ts, end_ts)
%CROP_RANGED Recorta un struct con ts_begin/ts_end, conservando cualquier
%   evento que se traslape con la ventana (no solo los contenidos por completo).
    idx = s.ts_end >= start_ts & s.ts_begin <= end_ts;
    fns = fieldnames(s);
    for f = 1:numel(fns)
        out.(fns{f}) = s.(fns{f})(idx);
    end
end
