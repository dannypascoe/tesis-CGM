function [u_mU_L, m_mg, info] = build_ohio_kernel_signals(grid_ts, basal, temp_basal, bolus, meal, Ts_min, subj, calib_factor, opts)
%BUILD_OHIO_KERNEL_SIGNALS  u(k) y m(k) a partir de kernels de absorcion
%   parametricos (gamma), como alternativa a integrar los sub-modelos ODE de
%   Dalla Man/Sorensen (build_ohio_insulin_signal / build_ohio_meal_signal).
%
%   Es la variante "convolucion con un kernel simple" de la propuesta de
%   dar forma fisiologica a los impulsos de Ohio, con dos decisiones para que
%   la comparacion contra el brazo ODE sea limpia:
%
%   1) MISMAS UNIDADES Y MISMA ESCALA que la senal de entrenamiento.
%      - Insulina: kernel de area unitaria multiplicado por la ganancia
%        estatica del sub-modelo de insulina (ohio_insulin_dc_gain). La
%        infusion basal (basal + temp_basal) pasa por el mismo kernel, por lo
%        que el nivel medio de u(k) sale de la fisiologia, igual que en el
%        brazo ODE. Salida en mU/L.
%      - Comida: el area del kernel se fija a 1/kabs_gut, que es la integral
%        de Qgut por unidad de masa ingerida en el modelo gastrointestinal
%        (toda la masa termina absorbida). Luego se divide entre el MISMO
%        calib_factor que usa build_ohio_meal_signal, de modo que la escala
%        de m(k) sea identica entre brazos.
%   2) Solo cambia la DINAMICA (forma y retardo del kernel). Eventos,
%      conversiones de unidades, resolucion de comidas (resolve_meal_carbs) y
%      reconstruccion de IIR (build_ohio_iir_signal) son los mismos.
%
%   LIMITE CONOCIDO: el modelo gastrico es no lineal. Para una comida aislada
%   su forma normalizada no depende del tamano (pico ~30 min, 95% en ~280 min),
%   asi que un kernel lineal la reproduce; con comidas separadas menos de ~3 h
%   el modelo ODE se aparta de cualquier kernel lineal (ver test_kernel_core.m).
%
% Entradas:
%   grid_ts      - Nx1 datetime, rejilla uniforme (extendida con pre-roll)
%   basal        - struct .ts, .value (U/h)
%   temp_basal   - struct .ts_begin, .ts_end, .value (U/h)
%   bolus        - struct .ts_begin, .dose (U), .bwz_carb_input (g)
%   meal         - struct .ts, .carbs (g)
%   Ts_min       - cadencia de la rejilla [min]
%   subj         - subject2_params()
%   calib_factor - escalar; el mismo que devuelve build_ohio_meal_signal
%   opts         - (opcional) struct:
%       .preset               'literature' (default) | 'matched'
%       .ins_peak_min, .ins_t95_min, .cho_peak_min, .cho_t95_min
%                             sobrescriben el preset
%       .horizon_min          longitud del kernel [min], default 720
%       .ins_kernel_override  vector w [1/min] (sum(w)*Ts=1) en lugar de gamma
%       .cho_kernel_override  idem para comida (uso: tests)
%     Presets:
%       'literature' - insulina rapida: pico 70 min, 95% a 270 min;
%                      CHO: pico 45 min, 95% a 210 min (valores de la propuesta)
%       'matched'    - moda y percentil 95 medidos sobre la respuesta al impulso
%                      del propio modelo ODE del Subject 2: insulina 46/259 min,
%                      CHO 30/280 min (ver test_kernel_core.m, seccion 3)
%
% Salidas:
%   u_mU_L - Nx1 [mU/L]      m_mg - Nx1 (misma escala calibrada que el brazo ODE)
%   info   - struct con parametros de los kernels y ganancias

    if nargin < 9 || isempty(opts), opts = struct(); end
    opts = fill_opts(opts);
    N = numel(grid_ts);

    %% 1) Insulina: IIR [pmol/min] -> kernel -> mU/L
    IIR = build_ohio_iir_signal(grid_ts, basal, temp_basal, bolus, Ts_min);
    if isempty(opts.ins_kernel_override)
        [w_ins, ki] = ohio_gamma_kernel(opts.ins_peak_min, opts.ins_t95_min, Ts_min, opts.horizon_min);
    else
        w_ins = opts.ins_kernel_override(:);
        ki = struct('override', true);
    end
    G_dc   = ohio_insulin_dc_gain(subj);
    u_mU_L = G_dc * apply_absorption_kernel(IIR, w_ins, Ts_min);

    %% 2) Comida: pulsos de masa [mg/min] -> kernel -> Qgut equivalente
    [meal_ts, meal_g] = resolve_meal_carbs(meal, bolus);
    Dt_mgmin = zeros(N,1);
    for i = 1:numel(meal_ts)
        D0_mg = meal_g(i) * 1000;                 % g -> mg
        if D0_mg <= 0, continue; end
        [~, k] = min(abs(grid_ts - meal_ts(i)));
        Dt_mgmin(k) = Dt_mgmin(k) + D0_mg / Ts_min;
    end
    if isempty(opts.cho_kernel_override)
        [w_cho, kc] = ohio_gamma_kernel(opts.cho_peak_min, opts.cho_t95_min, Ts_min, opts.horizon_min);
    else
        w_cho = opts.cho_kernel_override(:);
        kc = struct('override', true);
    end
    Qgut_raw = apply_absorption_kernel(Dt_mgmin, w_cho, Ts_min) / subj.kabs_gut;
    m_mg     = Qgut_raw / calib_factor;

    info = struct('preset', opts.preset, 'ins_kernel', ki, 'cho_kernel', kc, ...
                  'G_dc', G_dc, 'calib_factor', calib_factor);
end

function opts = fill_opts(opts)
    if ~isfield(opts,'preset') || isempty(opts.preset), opts.preset = 'literature'; end
    switch lower(opts.preset)
        case 'literature', p = [70 270 45 210];
        case 'matched',    p = [46 259 30 280];
        otherwise, error('build_ohio_kernel_signals: preset "%s" desconocido.', opts.preset);
    end
    def = struct('ins_peak_min',p(1), 'ins_t95_min',p(2), 'cho_peak_min',p(3), ...
                 'cho_t95_min',p(4), 'horizon_min',720, ...
                 'ins_kernel_override',[], 'cho_kernel_override',[]);
    f = fieldnames(def);
    for i = 1:numel(f)
        if ~isfield(opts, f{i}) || (isempty(opts.(f{i})) && ~contains(f{i},'override'))
            opts.(f{i}) = def.(f{i});
        end
    end
end
