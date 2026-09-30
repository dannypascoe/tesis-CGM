function [g_test, g_test_real, u, m, true_f, grid_ts, meta] = load_and_prepare_data_ohio(window_mat_path, std_m_ref, exo_repr, kernel_opts)
%LOAD_AND_PREPARE_DATA_OHIO  Carga la ventana de datos reales de OhioT1DM
%   ya seleccionada (main_ohio_window_selection_v5.m) y construye las
%   señales necesarias para el pipeline de deteccion+imputacion: g_test,
%   g_test_real, u, m, true_f -- mismo espiritu que load_and_prepare_data.m
%   (Sorensen), pero sin g_train (Ohio no se usa para reentrenar).
%
%   CAMBIOS RESPECTO A LA VERSION ANTERIOR:
%   - Pre-roll: u(k) y m(k) se integran sobre una rejilla extendida que
%     empieza en window_data.preroll_start_ts (24 h antes de la ventana) y
%     despues se recortan a [start_ts, end_ts]. Asi el transitorio inicial
%     de los sub-modelos (insulina arranca en el estado estacionario del
%     Subject 2; comidas/bolos de la noche previa ignorados) ocurre fuera
%     de la ventana evaluada. Si el .mat es de la version v1 (sin
%     pre-roll), se avisa y se sigue como antes.
%   - Calibracion de m(k) con meta.std_raw_calib (calculada en
%     main_ohio_window_selection_v5.m fuera de la ventana evaluada). Si el
%     .mat no la trae, se avisa y se calibra con la propia rejilla.
%   - std_m_ref opcional, que se pasa a build_ohio_meal_signal. Si se
%     omite, se usa su default (153.8637, de trained_cnn_lstm_latest.mat).
%
%   IMPORTANTE (sin cambios):
%   - NO normaliza. La normalizacion reutiliza media_*/std_* guardados en
%     trained_<arquitectura>_latest.mat, nunca recalculados sobre Ohio.
%   - true_f marca 1 SOLO en los huecos reales de la ventana. La inyeccion
%     de fallos sinteticos debe SUMARSE a este true_f, no reemplazarlo.
%   - g_test usa 0 como centinela en los huecos reales (criterio
%     "past_g < 10" del pipeline).
%   - g_test_real queda en NaN en los huecos reales (no hay referencia).
%
% Entradas:
%   window_mat_path - .mat de main_ohio_window_selection_v5.m
%   std_m_ref       - (opcional) std de referencia para calibrar m(k)
%   exo_repr        - (opcional) como se construyen u(k) y m(k). Default 'ode'.
%       'ode'            sub-modelos de Dalla Man/Sorensen (comportamiento original)
%       'kernel'         kernels gamma de literatura (insulina 70/270 min, CHO 45/210 min)
%       'kernel_matched' kernels gamma ajustados a la respuesta del propio ODE
%       'impulse'        impulsos crudos: bolo [U] y carbohidratos [g]
%       'custom'         kernel_opts.custom_builder = @(ctx) -> [u_full, m_full]; ctx trae
%                        grid_ts, window_data, Ts_min, subj y calib_factor (ver abajo)
%       {'ode','impulse'} celda de dos nombres: representacion de u y de m por separado
%     Todos comparten ventana, pre-roll, huecos, true_f, g_test y normalizacion; solo
%     cambia la construccion de u y m. En 'kernel*' y 'custom' la escala de m usa el
%     mismo calib_factor que 'ode'. NO confundir con input_mode de
%     main_pipeline_ohio_v4.m ('real'/'neutral'), que es un control aparte.
%   kernel_opts     - (opcional) struct para build_ohio_kernel_signals / 'custom'
%
% Salidas:
%   g_test, g_test_real, u, m - vectores columna, longitud N (solo ventana)
%   true_f  - vector FILA (misma convencion que load_and_prepare_data.m)
%   grid_ts - Nx1 datetime, rejilla de la ventana evaluada
%   meta    - struct meta del archivo + n_preroll (muestras descartadas)

    if nargin < 2
        std_m_ref = [];
    end
    if nargin < 3 || isempty(exo_repr), exo_repr = 'ode'; end
    if nargin < 4 || isempty(kernel_opts), kernel_opts = struct(); end

    % Dos representaciones distintas: u de la primera y m de la segunda
    if iscell(exo_repr)
        if numel(exo_repr) ~= 2
            error('exo_repr en celda debe ser {repr_u, repr_m}.');
        end
        [g_test, g_test_real, u, ~, true_f, grid_ts, meta] = ...
            load_and_prepare_data_ohio(window_mat_path, std_m_ref, exo_repr{1}, kernel_opts);
        [~, ~, ~, m] = load_and_prepare_data_ohio(window_mat_path, std_m_ref, exo_repr{2}, kernel_opts);
        meta.exo_repr = [exo_repr{1} '+' exo_repr{2}];
        return;
    end

    Ts_min = 5;

    %% 1) Cargar la ventana ya seleccionada
    S = load(window_mat_path, 'window_data', 'meta');
    window_data = S.window_data;
    meta = S.meta;

    %% 2) Rejilla evaluada y rejilla extendida (con pre-roll)
    grid_ts = (window_data.start_ts : minutes(Ts_min) : window_data.end_ts)';
    N = numel(grid_ts);

    if isfield(window_data, 'preroll_start_ts') && ~isempty(window_data.preroll_start_ts)
        t_pre = window_data.preroll_start_ts;
    else
        warning(['load_and_prepare_data_ohio: el .mat no trae pre-roll ' ...
                 '(generado con v1). Regenera la ventana con main_ohio_window_selection_v5.m.']);
        t_pre = window_data.start_ts;
    end

    grid_full = (t_pre : minutes(Ts_min) : window_data.end_ts)';
    n_pre = numel(grid_full) - N;

    % La rejilla extendida debe caer exactamente sobre start_ts
    if n_pre < 0 || abs(minutes(grid_full(n_pre+1) - window_data.start_ts)) > 1e-6
        error('Pre-roll desalineado con la rejilla de %d min.', Ts_min);
    end

    %% 3) Parametros fisiologicos del Subject 2
    subj = subject2_params();

    %% 4) Insulina y comida sobre la rejilla extendida, luego recorte
    if isfield(meta, 'std_raw_calib') && ~isempty(meta.std_raw_calib)
        std_raw_calib = meta.std_raw_calib;
    else
        warning(['load_and_prepare_data_ohio: el .mat no trae std_raw_calib; ' ...
                 'm(k) se calibra con la propia ventana. ' ...
                 'Regenera con main_ohio_window_selection_v5.m.']);
        std_raw_calib = [];
    end
    switch lower(exo_repr)
        case 'ode'
            u_full = build_ohio_insulin_signal(grid_full, window_data.basal, ...
                window_data.temp_basal, window_data.bolus, Ts_min, subj);
            m_full = build_ohio_meal_signal(grid_full, window_data.meal, ...
                window_data.bolus, Ts_min, subj, std_m_ref, std_raw_calib);

        case {'kernel', 'kernel_matched', 'custom'}
            % El factor de calibracion de m(k) sale del brazo ODE para que la escala
            % sea identica; solo cambia la dinamica de absorcion.
            [~, ~, ~, ~, ~, calib_factor] = build_ohio_meal_signal(grid_full, ...
                window_data.meal, window_data.bolus, Ts_min, subj, std_m_ref, std_raw_calib);
            if strcmpi(exo_repr, 'custom')
                if ~isfield(kernel_opts, 'custom_builder')
                    error('exo_repr ''custom'' requiere kernel_opts.custom_builder.');
                end
                ctx = struct('grid_ts', grid_full, 'window_data', window_data, ...
                             'Ts_min', Ts_min, 'subj', subj, 'calib_factor', calib_factor);
                [u_full, m_full] = kernel_opts.custom_builder(ctx);
            else
                if strcmpi(exo_repr, 'kernel_matched'), kernel_opts.preset = 'matched'; end
                [u_full, m_full, kinfo] = build_ohio_kernel_signals(grid_full, ...
                    window_data.basal, window_data.temp_basal, window_data.bolus, ...
                    window_data.meal, Ts_min, subj, calib_factor, kernel_opts);
                meta.kernel_info = kinfo;
            end

        case 'impulse'
            [u_full, m_full] = build_ohio_impulse_signals(grid_full, ...
                window_data.bolus, window_data.meal);

        otherwise
            error('load_and_prepare_data_ohio: exo_repr ''%s'' desconocido.', exo_repr);
    end
    meta.exo_repr = lower(exo_repr);

    u = u_full(n_pre+1:end);
    m = m_full(n_pre+1:end);

    %% 5) Glucosa real sobre la rejilla evaluada + huecos reales
    [g_raw, gap_mask] = build_glucose_on_grid(grid_ts, window_data.glucose, Ts_min);

    %% 6) g_test (centinela 0), g_test_real (NaN), true_f
    g_test = g_raw;
    g_test(gap_mask) = 0;

    g_test_real = g_raw;
    g_test_real(gap_mask) = NaN;

    true_f = double(gap_mask(:))';

    %% 7) Validaciones
    if numel(u) ~= N || numel(m) ~= N || numel(g_test) ~= N
        error('Longitudes inconsistentes: g_test=%d, u=%d, m=%d (N esperado=%d)', ...
            numel(g_test), numel(u), numel(m), N);
    end

    meta.n_preroll = n_pre;

    fprintf(['load_and_prepare_data_ohio: paciente %s | %d muestras | ' ...
             '%.2f%% huecos reales | pre-roll %.1f h | u,m: %s\n'], ...
        meta.patient_id, N, 100*sum(gap_mask)/N, n_pre*Ts_min/60, meta.exo_repr);
end

function [g_raw, gap_mask] = build_glucose_on_grid(grid_ts, glucose, Ts_min)
%BUILD_GLUCOSE_ON_GRID Reindexa la glucosa cruda a la rejilla uniforme
%   (misma logica que build_glucose_grid en main_ohio_window_selection).
    N = numel(grid_ts);
    g_raw = nan(N,1);

    t0 = grid_ts(1);
    tol_min = Ts_min/2;

    offsets_min = minutes(glucose.ts - t0);
    idx = round(offsets_min/Ts_min) + 1;
    residual = abs(offsets_min - (idx-1)*Ts_min);

    valid = idx>=1 & idx<=N & residual<=tol_min;
    g_raw(idx(valid)) = glucose.value(valid);

    gap_mask = isnan(g_raw);
end