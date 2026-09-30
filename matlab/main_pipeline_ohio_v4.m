%% =========================================================================
%  MAIN_PIPELINE_OHIO_V4.M - Validacion externa en OhioT1DM (modelos congelados)
%  =========================================================================
%  Descripcion: Corre el pipeline desacoplado detector + imputador sobre las
%               ventanas reales de OhioT1DM YA preparadas, usando los modelos
%               entrenados en Sorensen SIN reentrenar y con su normalizacion
%               original (media_*/std_* de trained_<arq>_latest.mat).
%
%  Logica operativa: identica a main_pipeline_integrado_v3.m (misma ventana
%  deslizante, umbral adaptativo B+ROC, suavizado adaptativo, D1 y D3).
%
%  Cambios respecto a main_pipeline_ohio_v2.m:
%    - fault_mode = 'synthetic': ademas de los huecos reales, se SUMAN fallas
%      sinteticas con el protocolo del articulo (Seccion 2.3) mediante
%      inject_synthetic_faults.m: 2 rafagas de ruido blanco (sigma = 5.6,
%      amplitud maxima 30 mg/dL, 20-40 muestras) y 2 desconexiones de 50 y
%      20 muestras (lectura = 0), ~9% de la ventana. Se repite en
%      n_placements posiciones aleatorias (semillas fijas) por paciente.
%    - D3 (error de la salida contra la referencia en muestras con falla) ya
%      se puede medir, en total y por tipo de falla y rango de glucosa
%      (<250 / >=250 mg/dL).
%    - Linea base de D3: LOCF (mantener el ultimo valor valido), que es la
%      alternativa causal sin modelo durante una falla.
%    - Recall por tipo de falla sintetica (ruido y desconexion).
%    - LOCF por tipo de falla: la comparacion justa de la imputacion es la
%      de desconexion (D3 desc. vs LOCF desc.); el D3 total mezcla ambos
%      tipos y las rafagas de ruido no detectadas pasan sin corregir.
%    - fault_mode = 'real' reproduce el escenario de v2 (solo huecos reales).
%    - fault_placement = 'manual' (por defecto): defines tu los rangos de las
%      fallas en manual_faults (una sola senal corrupta por paciente).
%      'random' reparte las fallas al azar y permite varias ubicaciones
%      (n_placements) para no depender de una posicion concreta.
%    - preview_faults: antes de correr las redes muestra las fallas que se
%      van a inyectar (preview_synthetic_faults.m) y pregunta si continuar;
%      con preview_only = true solo las muestra y termina.
%
%  Cambios respecto a v3 (v4, experimento de representacion de u y m):
%    - exo_repr: como se construyen u(k) y m(k) a partir de los eventos de Ohio
%      ('ode' | 'kernel' | 'kernel_matched' | 'impulse' | {repr_u, repr_m};
%      ver load_and_prepare_data_ohio.m). Con 'ode' la logica es la de v3.
%    - Si existe el appdata 'ohio_exo_arm' (lo fija run_ohio_exo_experiment.m),
%      input_mode y exo_repr se toman de ahi; sin el, valen los de este archivo.
%    - Los archivos de salida llevan el sufijo _x<exo_repr> (no pisan los de v3).
%    - Al terminar guarda la ruta del resumen en el appdata 'ohio_last_summary'.
%    - Nada mas cambia: mismas fallas, mismo detector/imputador, mismas metricas.
%
%  Definiciones (iguales a v2): las primeras N_PAST muestras se excluyen de
%  todas las metricas; D1 exige referencia real en k y en k-1 y se calcula
%  sobre TODAS las muestras con referencia (incluidas las de fallas
%  sinteticas, como en el articulo); FPR = FP/(FP+TN) sobre muestras limpias.
%  Las estadisticas por paciente agrupan colocaciones x corridas (media y DE);
%  luego se agrega entre pacientes (media +/- DE de las medias por paciente).
%
%  Prerrequisitos:
%    - trained_gru_latest.mat y trained_1dcnn_latest.mat en data/models
%    - paciente_<ID>_ventana5d_latest.mat en data/ohio/prepared
%    - inject_synthetic_faults.m en utils (o en el path)
%
%  Salidas (data/ohio/results), sufijo _synth en modo sintetico:
%    - results_ohio_<ID>_<det>_det_<imp>_imp[_synth]_latest.mat
%        (senales de la 1a colocacion + M_all e inj_all de todas)
%    - ohio_pipeline_summary[_synth]_<timestamp>.mat / .xlsx
%  =========================================================================

clear; clc; close all;

%% ===================== AGREGAR CARPETAS AL PATH =====================
scriptPath = 'D:\Documentos\02 MROI\07 Tesis\03 Extension Tesis\CGM_Project\CGM_Project\matlab';

addpath(fullfile(scriptPath, 'architectures'));
addpath(fullfile(scriptPath, 'data_preparation'));
addpath(fullfile(scriptPath, 'utils'));
addpath(fullfile(scriptPath, 'data'));

fprintf('Carpetas agregadas al path de MATLAB\n');

%% ===================== CONFIGURACION =====================

% Pacientes (uno o varios). Para una primera prueba: patients = {'559'};
patients = {'540', '559', '596'};
window_tag = 'ventana5d';

% Modo de entradas exogenas: 'real' (u, m construidas desde el XML) o
% 'neutral' (ablacion: u = media_u, m = media_m -> z = 0)
input_mode = 'real';

% v4: representacion de u y m (ver load_and_prepare_data_ohio.m)
%   'ode' | 'kernel' | 'kernel_matched' | 'impulse' | {repr_u, repr_m}
exo_repr = 'ode';
if isappdata(0, 'ohio_exo_arm')             % lo fija run_ohio_exo_experiment.m
    arm = getappdata(0, 'ohio_exo_arm');
    input_mode = arm.input_mode;
    exo_repr   = arm.exo_repr;
end
if iscell(exo_repr), exo_name = strjoin(exo_repr, '+'); else, exo_name = exo_repr; end
if strcmp(input_mode, 'neutral'), exo_tag = ''; else, exo_tag = ['_x' exo_name]; end

if ~ismember(input_mode, {'real', 'neutral'})
    error('input_mode debe ser ''real'' o ''neutral''.');
end
if strcmp(input_mode, 'real'), input_tag = ''; else, input_tag = ['_' input_mode]; end

% Escenario de fallas:
%   'real'      -> solo los huecos reales de la ventana (igual que v2)
%   'synthetic' -> huecos reales + fallas sinteticas (protocolo del articulo)
fault_mode = 'synthetic';
if ~ismember(fault_mode, {'real', 'synthetic'})
    error('fault_mode debe ser ''real'' o ''synthetic''.');
end
if strcmp(fault_mode, 'real'), fault_tag = ''; else, fault_tag = '_synth'; end
% Etiqueta libre para distinguir corridas y que NO se sobrescriban los _latest.mat
% (ej. '_s5p6' para el protocolo del articulo, '_s15' para la sensibilidad).
run_label = '_s5p6_exc';
mode_tag = [input_tag fault_tag run_label exo_tag];
fprintf('Entradas exogenas: %s (u,m: %s) | Escenario de fallas: %s\n', input_mode, exo_name, fault_mode);

% Fallas sinteticas (solo si fault_mode = 'synthetic')
%   fault_placement = 'manual' -> TU defines los rangos (manual_faults, abajo)
%   fault_placement = 'random' -> posiciones aleatorias con semilla fija
fault_placement = 'manual';
if ~ismember(fault_placement, {'manual', 'random'})
    error('fault_placement debe ser ''manual'' o ''random''.');
end
n_placements          = 1;          % solo si fault_placement = 'random': ubicaciones aleatorias por paciente
fault_cfg.n_noise     = 2;          % rafagas de ruido blanco
fault_cfg.noise_len   = [20 40];    % muestras por rafaga
fault_cfg.noise_sigma = 5.6;        % mg/dL
fault_cfg.noise_max   = 30;         % mg/dL (amplitud maxima)
fault_cfg.disc_lens   = [50 20];    % muestras por desconexion
fault_cfg.margin      = 6;          % muestras limpias a cada lado de cada falla
fault_cfg.seed_base   = 2026;       % semilla = seed_base + 1000*paciente + colocacion

% Rangos manuales (fault_placement = 'manual'): indices de muestra dentro de la
% ventana de 1440 muestras (muestra = hora*12 + 1). Una fila [inicio fin] por falla.
% Deben estar sobre tramos validos, lejos de huecos reales (>= margin muestras)
% y del arranque; si no, el script se detiene y dice cual rango mover.
% Puntos de partida iguales para los tres pacientes; ajustalos viendo la vista previa.
manual_faults.p540.noise = [ 250 279;  850 884];
manual_faults.p540.disc  = [  69 118;  607 626];
manual_faults.p559.noise = [ 250 279;  850 884];
manual_faults.p559.disc  = [ 677 726; 1186 1205];
manual_faults.p596.noise = [ 250 279;  850 884];
manual_faults.p596.disc  = [ 514 563;  663 682];

% Previsualizacion de las fallas ANTES de correr las redes (usa la misma
% funcion y las mismas semillas que el pipeline; ver preview_synthetic_faults.m)
preview_faults       = false;        % mostrar figuras y tabla de segmentos
preview_only         = false;       % true = solo mirar y salir (sin correr nada)
preview_zoom_placement = 1;         % colocacion cuyo detalle se hace zoom

% Cada fila: {detector, imputador}  (co-optimos de Sorensen)
combinaciones = { 'gru', 'gru'; ...
                  'gru', '1dcnn' };
nCombos = size(combinaciones, 1);

% Parametros de imputacion (identicos a main_pipeline_integrado_v3.m)
winit      = 30;
alpha_base = 0.3;

% Umbral adaptativo B+ROC (identico a main_pipeline_integrado_v3.m)
adapt_params.beta    = 0.10;
adapt_params.tau_min = 7.0;
adapt_params.gamma   = 0.05;
adapt_params.tau_D   = 60;

% Rutas
models_path = fullfile(scriptPath, 'data', 'models');
prep_path   = fullfile(scriptPath, 'data', 'ohio', 'prepared');
output_path = fullfile(scriptPath, 'data', 'ohio', 'results');
if ~exist(output_path, 'dir')
    mkdir(output_path);
    fprintf('Carpeta creada: %s\n', output_path);
end

timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');

%% ===================== PRE-CARGAR MODELOS (CONGELADOS) =====================

fprintf('\nPre-cargando modelos...\n');
archs_needed = unique(combinaciones(:));
model_cache = struct();

for a = 1:numel(archs_needed)
    arch = archs_needed{a};
    model_file = fullfile(models_path, sprintf('trained_%s_latest.mat', arch));
    if ~exist(model_file, 'file')
        error('No se encontro el modelo: %s', model_file);
    end
    d = load(model_file, 'nets', 'media_g', 'std_g', 'media_u', 'std_u', ...
        'media_m', 'std_m', 'N_PAST', 'input_shape');
    model_cache.(matlab.lang.makeValidName(arch)) = d;
    fprintf('   %s: %d redes, N_PAST=%d, shape=%s | media_g=%.2f std_g=%.2f\n', ...
        upper(arch), numel(d.nets), d.N_PAST, d.input_shape, d.media_g, d.std_g);
end

%% ===================== BUCLE POR PACIENTE Y COMBINACION =====================

nP = numel(patients);
metric_names = {'acc', 'prec', 'rec', 'f1', 'auc', 'fpr', 'fp_day', 'fp_ep', ...
                'mae_d1', 'rmse_d1', 'mard_d1', 'mae_d3', 'rmse_d3', ...
                'rec_noise', 'rec_disc', 'mae_d3_noise', 'mae_d3_disc', ...
                'mae_d3_lo', 'mae_d3_hi', 'mae_locf', ...
                'mae_locf_noise', 'mae_locf_disc'};
nMet = numel(metric_names);
mi = @(nm) find(strcmp(metric_names, nm), 1);

band_edges = [20 70 180 250 Inf];
band_names = {'<70', '70-180', '180-250', '>250'};
nB = numel(band_names);

% N_PAST maximo entre los modelos cargados (para el arranque de la inyeccion)
N_PAST_all = 0;
for a = 1:numel(archs_needed)
    N_PAST_all = max(N_PAST_all, ...
        model_cache.(matlab.lang.makeValidName(archs_needed{a})).N_PAST);
end

if strcmp(fault_mode, 'synthetic') && strcmp(fault_placement, 'random')
    nPl = n_placements;
else
    nPl = 1;                 % modo manual o solo huecos reales: una sola senal por paciente
end

%% ===================== PREVISUALIZACION DE FALLAS (opcional) =====================
% Muestra la senal con las fallas que se van a inyectar, sin usar las redes,
% y pregunta si se continua. Sirve para ajustar fault_cfg antes de gastar
% tiempo de calculo.
if strcmp(fault_mode, 'synthetic') && preview_faults
    for p = 1:nP
        pid = patients{p};
        window_file = fullfile(prep_path, ...
            sprintf('paciente_%s_%s_latest.mat', pid, window_tag));
        if ~exist(window_file, 'file')
            error('No se encontro la ventana: %s', window_file);
        end
        [g_prev, g_prev_real, ~, ~, tf_prev] = load_and_prepare_data_ohio(window_file);
        seeds_prev = fault_cfg.seed_base + 1000 * p + (1:nPl);
        cfg_prev = fault_cfg;
        if strcmp(fault_placement, 'manual')
            cfg_prev.manual = manual_faults.(['p' pid]);
        end
        preview_synthetic_faults(pid, g_prev, g_prev_real, tf_prev, ...
            N_PAST_all + 1, cfg_prev, seeds_prev, adapt_params, preview_zoom_placement);
    end
    drawnow;
    if preview_only
        fprintf('\npreview_only = true: se muestran las fallas y se termina aqui.\n');
        return;
    end
    resp = input(sprintf('\nContinuar con el pipeline con estas fallas? [s/n]: '), 's');
    if ~strcmpi(strtrim(resp), 's')
        fprintf('Detenido. Ajusta fault_cfg / n_placements y vuelve a correr.\n');
        return;
    end
end

pat_mean  = nan(nP, nCombos, nMet);   % media sobre colocaciones x corridas
pat_std   = nan(nP, nCombos, nMet);   % DE sobre colocaciones x corridas
band_mean = nan(nP, nCombos, nB);     % FPR por rango de glucosa
band_n    = nan(nP, nB);              % muestras limpias por rango (media)
base_mat  = nan(nP, 3);               % persistencia: MAE, RMSE, MARD

for p = 1:nP
    pid = patients{p};

    fprintf('\n=====================================================\n');
    fprintf(' PACIENTE %s\n', pid);
    fprintf('=====================================================\n');

    window_file = fullfile(prep_path, ...
        sprintf('paciente_%s_%s_latest.mat', pid, window_tag));
    if ~exist(window_file, 'file')
        error('No se encontro la ventana: %s', window_file);
    end

    [g_test, g_test_real, u, m, true_f, grid_ts, meta] = ...
        load_and_prepare_data_ohio(window_file, [], exo_repr);

    g_test = g_test(:);  g_test_real = g_test_real(:);
    u = u(:);            m = m(:);
    T = numel(g_test);

    % Revision de rangos de entrada contra la normalizacion de Sorensen
    ref_model = model_cache.(matlab.lang.makeValidName(combinaciones{1, 1}));
    report_input_ranges(pid, g_test, u, m, ref_model);

    if strcmp(input_mode, 'neutral')
        u = repmat(ref_model.media_u, size(u));
        m = repmat(ref_model.media_m, size(m));
        fprintf('   [ABLACION] u y m fijadas a su media de Sorensen (z = 0).\n');
    end

    if any(g_test(1:N_PAST_all) < 10)
        fprintf('   Aviso: la ventana empieza con hueco; las primeras %d muestras se rellenan para arrancar.\n', N_PAST_all);
    end

    pool      = cell(nCombos, 1);     % filas = colocaciones x corridas, cols = metricas
    band_pool = cell(nCombos, 1);
    R1        = cell(nCombos, 1);     % resultados de la 1a colocacion (para guardar/graficar)
    M_all     = cell(nPl, nCombos);
    inj_all   = cell(nPl, 1);

    for pl = 1:nPl
        if strcmp(fault_mode, 'synthetic')
            seed = fault_cfg.seed_base + 1000 * p + pl;
            cfg_use = fault_cfg;
            if strcmp(fault_placement, 'manual')
                cfg_use.manual = manual_faults.(['p' pid]);
            end
            [g_run, tf_run, ft_run, inj] = inject_synthetic_faults( ...
                g_test, g_test_real, true_f, N_PAST_all + 1, cfg_use, seed);
            fprintf('\n   Fallas sinteticas (%s, semilla %d): %.1f%% de la ventana\n', ...
                fault_placement, seed, 100 * mean(ft_run >= 2));
        else
            g_run = g_test;  tf_run = true_f;  ft_run = double(true_f(:).');  inj = [];
        end
        inj_all{pl} = inj;
        if pl == 1
            g_run1 = g_run;  tf1 = tf_run;  ft1 = ft_run;
        end

        for combo = 1:nCombos
            arch_detect = combinaciones{combo, 1};
            arch_impute = combinaciones{combo, 2};

            det = model_cache.(matlab.lang.makeValidName(arch_detect));
            imp = model_cache.(matlab.lang.makeValidName(arch_impute));

            if pl == 1
                fprintf('\n   --- %s | Detector=%s | Imputador=%s ---\n', ...
                    pid, upper(arch_detect), upper(arch_impute));
            end

            R = run_combo(det, imp, g_run, g_test_real, u, m, adapt_params, ...
                winit, alpha_base, pl == 1);
            M = compute_metrics(R, tf_run, g_test_real, band_edges, ft_run);

            M_all{pl, combo} = M;
            if pl == 1, R1{combo} = R; end

            V = zeros(numel(M.acc), nMet);
            for k_met = 1:nMet
                V(:, k_met) = M.(metric_names{k_met});
            end
            pool{combo}      = [pool{combo}; V];
            band_pool{combo} = [band_pool{combo}; M.fpr_band];
        end
        if nPl > 1 && pl > 1
            fprintf('   Colocacion %d/%d lista.\n', pl, nPl);
        end
    end

    % ---- Estadisticas por paciente y combinacion ----
    nb = zeros(nPl, nB);
    for pl = 1:nPl, nb(pl, :) = M_all{pl, 1}.n_band; end
    band_n(p, :) = mean(nb, 1);
    base_mat(p, :) = [M_all{1,1}.base_mae, M_all{1,1}.base_rmse, M_all{1,1}.base_mard];

    for combo = 1:nCombos
        arch_detect = combinaciones{combo, 1};
        arch_impute = combinaciones{combo, 2};
        combo_name  = sprintf('%s_det_%s_imp', arch_detect, arch_impute);
        det = model_cache.(matlab.lang.makeValidName(arch_detect));
        imp = model_cache.(matlab.lang.makeValidName(arch_impute));

        V = pool{combo};
        pat_mean(p, combo, :)  = mean(V, 1, 'omitnan');
        pat_std(p, combo, :)   = std(V, 0, 1, 'omitnan');
        band_mean(p, combo, :) = mean(band_pool{combo}, 1, 'omitnan');

        fprintf('\n   >>> %s | %s -> %s  (%d colocacion(es) x 5 corridas)\n', ...
            pid, upper(arch_detect), upper(arch_impute), nPl);
        fprintf('      Deteccion (huecos reales + sinteticas): Rec=%.3f | FPR=%.1f%% (%.0f falsas alarmas/dia, %.0f episodios) | AUC=%.3f\n', ...
            pat_mean(p,combo,mi('rec')), 100*pat_mean(p,combo,mi('fpr')), ...
            pat_mean(p,combo,mi('fp_day')), pat_mean(p,combo,mi('fp_ep')), ...
            pat_mean(p,combo,mi('auc')));
        if strcmp(fault_mode, 'synthetic')
            fprintf('      Recall por tipo: ruido=%.3f | desconexion=%.3f\n', ...
                pat_mean(p,combo,mi('rec_noise')), pat_mean(p,combo,mi('rec_disc')));
        end
        if combo == 1
            fprintf('      FPR por rango de glucosa (muestras limpias):');
            for b = 1:nB
                fprintf(' %s: %.1f%% (n~%.0f) |', band_names{b}, ...
                    100*band_mean(p,combo,b), band_n(p,b));
            end
            fprintf('\n');
        end
        fprintf('      Imputacion D1: MAE=%.2f | RMSE=%.2f mg/dL | MARD=%.2f%%   (persistencia MAE=%.2f, razon=%.2f)\n', ...
            pat_mean(p,combo,mi('mae_d1')), pat_mean(p,combo,mi('rmse_d1')), ...
            pat_mean(p,combo,mi('mard_d1')), base_mat(p,1), ...
            pat_mean(p,combo,mi('mae_d1')) / base_mat(p,1));
        if isnan(pat_mean(p, combo, mi('mae_d3')))
            fprintf('      Imputacion D3: n/d (sin fallas con referencia; usa fault_mode = ''synthetic'')\n');
        else
            fprintf('      Imputacion D3: MAE=%.2f | RMSE=%.2f mg/dL   (LOCF MAE=%.2f, razon=%.2f)\n', ...
                pat_mean(p,combo,mi('mae_d3')), pat_mean(p,combo,mi('rmse_d3')), ...
                pat_mean(p,combo,mi('mae_locf')), ...
                pat_mean(p,combo,mi('mae_d3')) / pat_mean(p,combo,mi('mae_locf')));
            fprintf('        ruido:       D3 MAE=%.2f | LOCF MAE=%.2f\n', ...
                pat_mean(p,combo,mi('mae_d3_noise')), pat_mean(p,combo,mi('mae_locf_noise')));
            fprintf('        desconexion: D3 MAE=%.2f | LOCF MAE=%.2f  (razon=%.2f)\n', ...
                pat_mean(p,combo,mi('mae_d3_disc')), pat_mean(p,combo,mi('mae_locf_disc')), ...
                pat_mean(p,combo,mi('mae_d3_disc')) / pat_mean(p,combo,mi('mae_locf_disc')));
            fprintf('        por glucosa: <250 D3 MAE=%.2f | >=250 D3 MAE=%.2f\n', ...
                pat_mean(p,combo,mi('mae_d3_lo')), pat_mean(p,combo,mi('mae_d3_hi')));
        end

        % ---- Guardar resultados de este paciente y combinacion ----
        pipeline_config = struct();
        pipeline_config.patient        = pid;
        pipeline_config.arch_detect    = arch_detect;
        pipeline_config.arch_impute    = arch_impute;
        pipeline_config.threshold_mode = 'adaptive';
        pipeline_config.adapt_params   = adapt_params;
        pipeline_config.winit          = winit;
        pipeline_config.alpha_base     = alpha_base;
        pipeline_config.N_PAST_detect  = det.N_PAST;
        pipeline_config.N_PAST_impute  = imp.N_PAST;
        pipeline_config.nRuns          = size(R1{combo}.fallas_all, 1);
        pipeline_config.window_file    = window_file;
        pipeline_config.models_frozen  = true;
        pipeline_config.script         = 'main_pipeline_ohio_v4';
        pipeline_config.input_mode     = input_mode;
        pipeline_config.exo_repr       = exo_name;
        pipeline_config.fault_mode     = fault_mode;
        pipeline_config.n_placements   = nPl;
        pipeline_config.fault_cfg      = fault_cfg;

        Sfile = R1{combo};                    % senales de la 1a colocacion
        Sfile.g_test          = g_run1;       % lectura (corrupta en modo sintetico)
        Sfile.g_test_real     = g_test_real;  % referencia (NaN en huecos reales)
        Sfile.true_f          = tf1;
        Sfile.fault_type      = ft1;          % 0 limpia | 1 hueco real | 2 ruido | 3 desconexion
        Sfile.u               = u;
        Sfile.m               = m;
        Sfile.grid_ts         = grid_ts;
        Sfile.meta            = meta;
        Sfile.pipeline_config = pipeline_config;
        Sfile.M               = M_all{1, combo};
        Sfile.M_all           = M_all(:, combo);
        Sfile.inj_all         = inj_all;

        f_ts     = fullfile(output_path, sprintf('results_ohio_%s_%s%s_%s.mat', pid, combo_name, mode_tag, timestamp));
        f_latest = fullfile(output_path, sprintf('results_ohio_%s_%s%s_latest.mat', pid, combo_name, mode_tag));
        for f_out = {f_ts, f_latest}
            save(f_out{1}, '-struct', 'Sfile');
        end
        fprintf('      Guardado: %s\n', f_latest);
    end
end

%% ===================== TABLA POR PACIENTE Y AGREGADO ENTRE PACIENTES =====================

fprintf('\n\n=====================================================\n');
fprintf(' RESUMEN (por paciente: media sobre colocaciones x corridas)\n');
fprintf('=====================================================\n');
fprintf('%-6s %-13s %6s %7s | %7s %7s | %9s %9s\n', ...
    'Pac.', 'Det->Imp', 'Rec', 'FPR%', 'D1 MAE', 'pers.', 'D3 desc.', 'LOCF desc');

var_names = [{'paciente', 'detector', 'imputador'}, metric_names, ...
             {'base_mae', 'base_rmse', 'base_mard'}];
rows = {};

for p = 1:nP
    for combo = 1:nCombos
        v = squeeze(pat_mean(p, combo, :))';
        fprintf('%-6s %-13s %6.3f %7.1f | %7.2f %7.2f | %9.2f %9.2f\n', ...
            patients{p}, [combinaciones{combo,1} '->' combinaciones{combo,2}], ...
            v(mi('rec')), 100*v(mi('fpr')), v(mi('mae_d1')), base_mat(p, 1), ...
            v(mi('mae_d3_disc')), v(mi('mae_locf_disc')));
        rows(end+1, :) = [{patients{p}, combinaciones{combo,1}, combinaciones{combo,2}}, ...
            num2cell(v), num2cell(base_mat(p, :))]; %#ok<SAGROW>
    end
end

fprintf('\nAgregado entre pacientes (media +/- DE de las medias por paciente, n=%d)\n', nP);
b_mu = mean(base_mat, 1, 'omitnan');  b_sd = std(base_mat, 0, 1, 'omitnan');
for combo = 1:nCombos
    mu = squeeze(mean(pat_mean(:, combo, :), 1, 'omitnan'))';
    sd = squeeze(std(pat_mean(:, combo, :), 0, 1, 'omitnan'))';
    fprintf('  %s->%s: Rec=%.3f+/-%.3f | FPR=%.1f+/-%.1f%% | D1 MAE=%.2f+/-%.2f | D3 desc.=%.2f+/-%.2f | LOCF desc.=%.2f+/-%.2f | (persistencia MAE=%.2f+/-%.2f)\n', ...
        combinaciones{combo,1}, combinaciones{combo,2}, ...
        mu(mi('rec')), sd(mi('rec')), 100*mu(mi('fpr')), 100*sd(mi('fpr')), ...
        mu(mi('mae_d1')), sd(mi('mae_d1')), mu(mi('mae_d3_disc')), sd(mi('mae_d3_disc')), ...
        mu(mi('mae_locf_disc')), sd(mi('mae_locf_disc')), b_mu(1), b_sd(1));
    rows(end+1, :) = [{'MEDIA', combinaciones{combo,1}, combinaciones{combo,2}}, ...
        num2cell(mu), num2cell(b_mu)]; %#ok<SAGROW>
    rows(end+1, :) = [{'DE',    combinaciones{combo,1}, combinaciones{combo,2}}, ...
        num2cell(sd), num2cell(b_sd)]; %#ok<SAGROW>
end

summary_table = cell2table(rows, 'VariableNames', var_names);

summary_mat  = fullfile(output_path, sprintf('ohio_pipeline_summary%s_%s.mat', mode_tag, timestamp));
summary_xlsx = fullfile(output_path, sprintf('ohio_pipeline_summary%s_%s.xlsx', mode_tag, timestamp));
save(summary_mat, 'summary_table', 'pat_mean', 'pat_std', 'band_mean', 'band_n', ...
    'band_names', 'base_mat', 'metric_names', 'patients', 'combinaciones', ...
    'adapt_params', 'timestamp', 'input_mode', 'exo_name', 'fault_mode', 'fault_cfg', 'n_placements');
writetable(summary_table, summary_xlsx);
fprintf('\nResumen guardado en:\n   %s\n   %s\n', summary_mat, summary_xlsx);
setappdata(0, 'ohio_last_summary', summary_mat);   % v4: lo lee run_ohio_exo_experiment.m


%% ===================== FUNCIONES LOCALES =====================

function R = run_combo(det, imp, g_test, g_test_real, u, m, adapt_params, winit, alpha_base, verbose)
%RUN_COMBO Pipeline en linea detector + imputador (logica de main_pipeline_integrado_v3).
    if nargin < 10, verbose = true; end
    T = numel(g_test);
    nRuns = min(numel(det.nets), numel(imp.nets));
    N_PAST_max = max(det.N_PAST, imp.N_PAST);
    k_start = N_PAST_max + 1;
    first_valid = g_test(find(g_test >= 10, 1));

    g_final_all           = zeros(nRuns, T);
    ypred_det_all         = zeros(nRuns, T);
    ypred_imp_all         = zeros(nRuns, T);
    ypred_imp_raw_all     = zeros(nRuns, T);
    fallas_all            = false(nRuns, T);
    error_detect_all      = zeros(nRuns, T);
    error_impute_pred_all = nan(nRuns, T);     % D1 (NaN donde no hay referencia)
    error_impute_out_all  = nan(nRuns, T);     % D2 (solo compatibilidad)
    threshold_all         = zeros(nRuns, T);
    roc_all               = zeros(nRuns, T);

    for i = 1:nRuns
        if verbose, fprintf('      [%d/%d] pareja de redes %d\n', i, nRuns, i); end

        net_det = det.nets{i};
        net_imp = imp.nets{i};

        g_final       = zeros(1, T);
        ypred_det     = zeros(1, T);
        ypred_imp     = zeros(1, T);
        ypred_imp_raw = zeros(1, T);
        fallas_vec    = false(1, T);
        error_detect  = zeros(1, T);
        err_pred      = nan(1, T);
        err_out       = nan(1, T);
        threshold_vec = zeros(1, T);
        roc_vec       = zeros(1, T);

        % Arranque: si hay hueco en las primeras muestras, se rellena con la
        % primera lectura valida (solo para arrancar; estas muestras no se evaluan)
        g_init = g_test(1:N_PAST_max).';
        g_init(g_init < 10) = first_valid;

        g_final(1:N_PAST_max)       = g_init;
        ypred_det(1:N_PAST_max)     = g_init;
        ypred_imp(1:N_PAST_max)     = g_init;
        ypred_imp_raw(1:N_PAST_max) = g_init;

        g_ref_init = mean(g_init);
        threshold_vec(1:N_PAST_max) = max(adapt_params.tau_min, ...
            adapt_params.beta * g_ref_init);

        for k = k_start : T

            % ---- PASO 1: prediccion del detector ----
            past_g_det = g_test(k - det.N_PAST : k - 1);
            invalid_idx = find(past_g_det < 10);
            for j = invalid_idx'
                past_g_det(j) = ypred_det(k - det.N_PAST - 1 + j);
            end

            x_det = [(past_g_det - det.media_g) / det.std_g; ...
                     (u(k) - det.media_u) / det.std_u; ...
                     (m(k) - det.media_m) / det.std_m];
            switch det.input_shape
                case 'column'
                    xseq_det = {reshape(x_det, [], 1)};
                case 'row'
                    xseq_det = {reshape(x_det, 1, [])};
            end
            y_hat_det = predict(net_det, xseq_det) * det.std_g + det.media_g;
            ypred_det(k) = y_hat_det;

            % ---- PASO 2: error de deteccion y umbral adaptativo ----
            err_detect = abs(y_hat_det - g_test(k));
            error_detect(k) = err_detect;

            [tau_k, tau_info] = compute_adaptive_threshold(y_hat_det, past_g_det, adapt_params);
            threshold_vec(k) = tau_k;
            roc_vec(k) = tau_info.roc;

            % ---- PASO 3: prediccion del imputador (en cada muestra, para D1) ----
            past_g_imp = g_test(k - imp.N_PAST : k - 1);
            invalid_idx_imp = find(past_g_imp < 10);
            for j = invalid_idx_imp'
                past_g_imp(j) = ypred_imp(k - imp.N_PAST - 1 + j);
            end

            x_imp = [(past_g_imp - imp.media_g) / imp.std_g; ...
                     (u(k) - imp.media_u) / imp.std_u; ...
                     (m(k) - imp.media_m) / imp.std_m];
            switch imp.input_shape
                case 'column'
                    xseq_imp = {reshape(x_imp, [], 1)};
                case 'row'
                    xseq_imp = {reshape(x_imp, 1, [])};
            end
            y_hat_imp = predict(net_imp, xseq_imp) * imp.std_g + imp.media_g;

            ypred_imp_raw(k) = y_hat_imp;
            err_pred(k) = abs(y_hat_imp - g_test_real(k));   % NaN en huecos reales

            % ---- PASO 4: decision de fallo ----
            if err_detect > tau_k
                fallas_vec(k) = true;
                ypred_imp(k)  = y_hat_imp;

                if k - 1 >= winit
                    var_local = var(error_detect(k - winit : k - 1));
                    alpha = 1 / (1 + var_local);
                else
                    alpha = alpha_base;
                end

                if err_detect < adapt_params.tau_D
                    g_final(k) = alpha * y_hat_imp + (1 - alpha) * g_final(k-1);
                else
                    g_final(k) = y_hat_imp;
                end
            else
                fallas_vec(k) = false;
                g_final(k)    = g_test(k);
                ypred_imp(k)  = y_hat_det;
            end

            err_out(k) = abs(g_final(k) - g_test_real(k));
        end

        g_final_all(i, :)           = g_final;
        ypred_det_all(i, :)         = ypred_det;
        ypred_imp_all(i, :)         = ypred_imp;
        ypred_imp_raw_all(i, :)     = ypred_imp_raw;
        fallas_all(i, :)            = fallas_vec;
        error_detect_all(i, :)      = error_detect;
        error_impute_pred_all(i, :) = err_pred;
        error_impute_out_all(i, :)  = err_out;
        threshold_all(i, :)         = threshold_vec;
        roc_all(i, :)               = roc_vec;
    end

    R = struct('g_final_all', g_final_all, 'ypred_det_all', ypred_det_all, ...
        'ypred_imp_all', ypred_imp_all, 'ypred_imp_raw_all', ypred_imp_raw_all, ...
        'fallas_all', fallas_all, 'error_detect_all', error_detect_all, ...
        'error_impute_pred_all', error_impute_pred_all, ...
        'error_impute_out_all', error_impute_out_all, ...
        'threshold_all', threshold_all, 'roc_all', roc_all, 'k_start', k_start);
end

function M = compute_metrics(R, true_f, g_test_real, band_edges, fault_type)
%COMPUTE_METRICS Metricas por corrida (vectores nRuns x 1) y lineas base.
%   Todas las metricas excluyen las primeras N_PAST muestras (arranque).
%   fault_type: 0 limpia | 1 hueco real | 2 ruido sintetico | 3 desconexion sintetica
%   Deteccion: TP/FP/FN/TN sobre true_f (huecos reales + sinteticas);
%              FPR = FP/(FP+TN) sobre muestras limpias.
%   D1: error de prediccion del imputador en muestras con referencia real
%       en k y en k-1 (misma mascara que la persistencia).
%   D3: error de la SALIDA (g_final) en muestras con falla sintetica (la
%       referencia sobrevive en g_test_real; los huecos reales no cuentan).
%   Linea base de D3: LOCF, el ultimo valor valido antes de la falla.
    ytrue  = double(true_f(:).');
    ftype  = double(fault_type(:).');
    y_real = g_test_real(:).';
    T      = numel(y_real);
    nRuns  = size(R.fallas_all, 1);
    ndays  = T * 5 / 1440;

    idx_det = true(1, T);
    idx_det(1:R.k_start - 1) = false;

    has_ref  = ~isnan(y_real) & (y_real >= 20);
    prev_ref = [false, has_ref(1:end-1)];
    idx_ref  = has_ref & prev_ref & idx_det;

    % Muestras con falla sintetica y referencia
    idx_f   = (ftype >= 2) & has_ref & idx_det;
    idx_fn  = idx_f & (ftype == 2);
    idx_fd  = idx_f & (ftype == 3);
    idx_flo = idx_f & (y_real <  250);
    idx_fhi = idx_f & (y_real >= 250);

    % Linea base: persistencia g(k-1) sobre las mismas muestras que D1
    pos       = find(idx_ref);
    e_base    = abs(y_real(pos) - y_real(pos - 1));
    base_mae  = mean(e_base);
    base_rmse = sqrt(mean(e_base.^2));
    base_mard = 100 * mean(e_base ./ y_real(pos));

    % Linea base de D3: LOCF (mantener el ultimo valor valido)
    mae_locf = NaN;  mae_locf_noise = NaN;  mae_locf_disc = NaN;
    if any(idx_f)
        locf = y_real;
        for k = 2:T
            if ytrue(k) == 1 || isnan(locf(k))
                locf(k) = locf(k-1);
            end
        end
        mae_locf = mean(abs(locf(idx_f) - y_real(idx_f)));
        if any(idx_fn), mae_locf_noise = mean(abs(locf(idx_fn) - y_real(idx_fn))); end
        if any(idx_fd), mae_locf_disc  = mean(abs(locf(idx_fd) - y_real(idx_fd))); end
    end

    nB = numel(band_edges) - 1;
    n_band = zeros(1, nB);
    idx_valid = (ytrue == 0) & has_ref & idx_det;
    for b = 1:nB
        n_band(b) = sum(idx_valid & y_real >= band_edges(b) & y_real < band_edges(b+1));
    end

    n_noise = sum(idx_det & ftype == 2);
    n_disc  = sum(idx_det & ftype == 3);

    acc = zeros(nRuns,1);  prec = zeros(nRuns,1);  rec = zeros(nRuns,1);
    f1  = zeros(nRuns,1);  auc  = nan(nRuns,1);
    fpr = zeros(nRuns,1);  fp_day = zeros(nRuns,1);  fp_ep = zeros(nRuns,1);
    fpr_band = nan(nRuns, nB);
    mae_d1 = zeros(nRuns,1);  rmse_d1 = zeros(nRuns,1);  mard_d1 = zeros(nRuns,1);
    mae_d3 = nan(nRuns,1);    rmse_d3 = nan(nRuns,1);
    rec_noise = nan(nRuns,1);  rec_disc = nan(nRuns,1);
    mae_d3_noise = nan(nRuns,1);  mae_d3_disc = nan(nRuns,1);
    mae_d3_lo = nan(nRuns,1);     mae_d3_hi = nan(nRuns,1);

    mabs = @(v, idx) mean(abs(v(idx) - y_real(idx)));

    for i = 1:nRuns
        yhat = R.fallas_all(i, :);
        yh = yhat(idx_det);  yt = ytrue(idx_det);
        TP = sum(yh == 1 & yt == 1);
        FP = sum(yh == 1 & yt == 0);
        FN = sum(yh == 0 & yt == 1);
        TN = sum(yh == 0 & yt == 0);

        acc(i)  = (TP + TN) / (TP + TN + FP + FN);
        prec(i) = TP / max(TP + FP, 1);
        rec(i)  = TP / max(TP + FN, 1);
        f1(i)   = 2 * prec(i) * rec(i) / max(prec(i) + rec(i), eps);
        fpr(i)  = FP / max(FP + TN, 1);
        fp_day(i) = FP / ndays;

        fp_mask = (yhat == 1) & (ytrue == 0) & idx_det;
        fp_ep(i) = sum(diff([0, fp_mask]) == 1);

        if n_noise > 0, rec_noise(i) = sum(yhat == 1 & idx_det & ftype == 2) / n_noise; end
        if n_disc  > 0, rec_disc(i)  = sum(yhat == 1 & idx_det & ftype == 3) / n_disc;  end

        for b = 1:nB
            in_b = idx_valid & y_real >= band_edges(b) & y_real < band_edges(b+1);
            if any(in_b)
                fpr_band(i, b) = sum(yhat(in_b) == 1) / sum(in_b);
            end
        end

        try
            [~, ~, ~, auc(i)] = perfcurve(yt, R.error_detect_all(i, idx_det), 1);
        catch
            auc(i) = NaN;
        end

        e = abs(R.ypred_imp_raw_all(i, idx_ref) - y_real(idx_ref));
        mae_d1(i)  = mean(e);
        rmse_d1(i) = sqrt(mean(e.^2));
        mard_d1(i) = 100 * mean(e ./ y_real(idx_ref));

        if any(idx_f)
            g_out = R.g_final_all(i, :);
            e3 = abs(g_out(idx_f) - y_real(idx_f));
            mae_d3(i)  = mean(e3);
            rmse_d3(i) = sqrt(mean(e3.^2));
            if any(idx_fn),  mae_d3_noise(i) = mabs(g_out, idx_fn);  end
            if any(idx_fd),  mae_d3_disc(i)  = mabs(g_out, idx_fd);  end
            if any(idx_flo), mae_d3_lo(i)    = mabs(g_out, idx_flo); end
            if any(idx_fhi), mae_d3_hi(i)    = mabs(g_out, idx_fhi); end
        end
    end

    M = struct('acc', acc, 'prec', prec, 'rec', rec, 'f1', f1, 'auc', auc, ...
        'fpr', fpr, 'fp_day', fp_day, 'fp_ep', fp_ep, ...
        'mae_d1', mae_d1, 'rmse_d1', rmse_d1, 'mard_d1', mard_d1, ...
        'mae_d3', mae_d3, 'rmse_d3', rmse_d3, ...
        'rec_noise', rec_noise, 'rec_disc', rec_disc, ...
        'mae_d3_noise', mae_d3_noise, 'mae_d3_disc', mae_d3_disc, ...
        'mae_d3_lo', mae_d3_lo, 'mae_d3_hi', mae_d3_hi, ...
        'mae_locf', repmat(mae_locf, nRuns, 1), ...
        'mae_locf_noise', repmat(mae_locf_noise, nRuns, 1), ...
        'mae_locf_disc', repmat(mae_locf_disc, nRuns, 1), ...
        'fpr_band', fpr_band, 'n_band', n_band, ...
        'base_mae', base_mae, 'base_rmse', base_rmse, 'base_mard', base_mard);
end

function report_input_ranges(pid, g_test, u, m, mdl)
%REPORT_INPUT_RANGES Compara los rangos de las entradas de Ohio con la
%   normalizacion de Sorensen (z-score con media_*/std_* del modelo).
    gv = g_test(g_test >= 10);
    zg = (gv - mdl.media_g) / mdl.std_g;
    zu = (u   - mdl.media_u) / mdl.std_u;
    zm = (m   - mdl.media_m) / mdl.std_m;

    fprintf('   Entradas del paciente %s frente a la normalizacion de Sorensen:\n', pid);
    fprintf('     g: %6.0f a %6.0f mg/dL (media %.0f) | |z| max = %.1f | %%|z|>3: %.1f\n', ...
        min(gv), max(gv), mean(gv), max(abs(zg)), 100*mean(abs(zg) > 3));
    fprintf('     u: %6.1f a %6.1f mU/L  (Sorensen 0-77.5)  | |z| max = %.1f | %%|z|>3: %.1f\n', ...
        min(u), max(u), max(abs(zu)), 100*mean(abs(zu) > 3));
    fprintf('     m: %6.1f a %6.1f mg/min (Sorensen 0-578)  | |z| max = %.1f | %%|z|>3: %.1f\n', ...
        min(m), max(m), max(abs(zm)), 100*mean(abs(zm) > 3));
end