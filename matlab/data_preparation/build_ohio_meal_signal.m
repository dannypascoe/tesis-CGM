function [m_mg, Qsto1, Qsto2, Qgut_raw, D0_used, calib_factor] = build_ohio_meal_signal(grid_ts, meal, bolus, Ts_min, subj, std_m_ref, std_raw_calib)
%BUILD_OHIO_MEAL_SIGNAL  Construye m(k) a partir de eventos crudos de
%   comida de OhioT1DM, integrando el sub-modelo no lineal de absorcion
%   gastrointestinal de Dalla Man (Glucose Abs Model1), verificado
%   directamente en el codigo fuente, con los parametros reales del
%   Subject 2 ("Subject 5" en el codigo), usando los gramos reales
%   reportados (sin ningun factor de conversion inventado sobre D0).
%
%   CALIBRACION DE ESCALA (necesaria, documentada como tal):
%   Las unidades absolutas exactas que produjo el pipeline original de
%   Sorensen para generar meal.mat no se pudieron recuperar (ver
%   discusion previa sobre el factor "/15" del glosario de Simulink,
%   descartado aqui por no ser physicamente justificable). En su lugar,
%   se aplica UN SOLO factor multiplicativo (sin desplazar la media) que
%   iguala la desviacion estandar de Qgut_raw a la de la referencia real
%   (std_m_ref, tomada de meal.mat / trained_<arch>_latest.mat). Esto
%   preserva la validez de normalizar despues con media_m/std_m de
%   Sorensen (a diferencia de un ajuste de media+std, que se cancelaria
%   algebraicamente con esa normalizacion y equivaldria a normalizar
%   Ohio contra si mismo -- ver discusion en el chat).
%
%   CAMBIO: el factor ya no tiene que salir de la propia ventana. Si se
%   pasa std_raw_calib (std de Qgut calculada fuera de la ventana evaluada,
%   ver compute_meal_calibration en main_ohio_window_selection_v5.m), el
%   factor es std_raw_calib / std_m_ref y la calibracion no usa ningun
%   dato de la ventana. Sin std_raw_calib se conserva el comportamiento
%   anterior (std de la propia rejilla) solo por compatibilidad.
%
% Entradas:
%   grid_ts    - Nx1 datetime, rejilla uniforme
%   meal       - struct .ts (datetime), .carbs (gramos)
%   bolus      - struct .ts_begin (datetime), .bwz_carb_input (gramos)
%   Ts_min     - cadencia de la rejilla (min)
%   subj       - struct de subject2_params() (usa kmax_gut, kmin_gut, kabs_gut,
%                kgri_gut, b_gut, d_gut)
%   std_m_ref  - (opcional) desviacion estandar de referencia para calibrar
%                la escala. Default = 153.8637 (std_m real de meal.mat /
%                trained_cnn_lstm_latest.mat). Pasa el std_m de la
%                arquitectura que vayas a usar si difiere.
%   std_raw_calib - (opcional) std de Qgut sin calibrar calculada fuera de
%                la ventana evaluada (meta.std_raw_calib).
%
% Salidas:
%   m_mg         - Nx1, señal YA CALIBRADA -> esto es lo que se usa como m(k)
%   Qsto1, Qsto2, Qgut_raw - Nx1 cada uno, estados SIN calibrar (diagnostico)
%   D0_used      - Nx1, tamano de comida activo en cada muestra (diagnostico)
%   calib_factor - escalar, factor aplicado (m_mg = Qgut_raw / calib_factor)

    if nargin < 6 || isempty(std_m_ref)
        std_m_ref = 153.8637;   % std_m real, de meal.mat / trained_cnn_lstm_latest.mat
    end

    N = numel(grid_ts);

    %% 1) Fuente de carbohidratos: meal.carbs + bolus.bwz_carb_input como respaldo/complemento
    [meal_ts, meal_g] = resolve_meal_carbs(meal, bolus);

    %% 2) D0(t): tamaño de comida activo (gramos reales, sin escalar), se
    %      actualiza en cada evento y se mantiene hasta la siguiente comida
    %      (convencion estandar para simulacion multi-comida con este modelo)
    D0_used  = ones(N,1);     % valor minimo inocuo antes de la primera comida
    Dt_mgmin = zeros(N,1);

    for i = 1:numel(meal_ts)
        D0_mg = meal_g(i) * 1000;         % g -> mg (sin ningun otro factor)
        if D0_mg <= 0, continue; end
        [~, k] = min(abs(grid_ts - meal_ts(i)));
        D0_used(k:end) = D0_mg;
        Dt_mgmin(k) = Dt_mgmin(k) + D0_mg / Ts_min;   % pulso de ingestion en esa muestra
    end

    %% 3) Integracion numerica (sistema no lineal -> RK4 con sub-pasos)
    Qsto1 = zeros(N,1); Qsto2 = zeros(N,1); Qgut_raw = zeros(N,1);
    x = [0; 0; 0];
    n_sub = 10;                 % sub-pasos por muestra de 5 min (h=0.5 min, estable)
    h = Ts_min / n_sub;

    for k = 2:N
        Dt_k = Dt_mgmin(k-1);
        D0_k = D0_used(k-1);
        for s = 1:n_sub
            x = rk4_step(@(xx) gut_ode(xx, Dt_k, D0_k, subj), x, h);
        end
        Qsto1(k) = x(1); Qsto2(k) = x(2); Qgut_raw(k) = x(3);
    end

    %% 4) Calibracion de escala: SOLO factor multiplicativo, sin tocar la media
    if nargin >= 7 && ~isempty(std_raw_calib)
        std_raw = std_raw_calib;          % calculada fuera de la ventana
        origen  = 'fuera de la ventana';
    else
        std_raw = std(Qgut_raw);          % compatibilidad: usa la propia rejilla
        origen  = 'propia rejilla';
    end
    if std_raw > 0
        calib_factor = std_raw / std_m_ref;
    else
        calib_factor = 1;
    end
    m_mg = Qgut_raw / calib_factor;

    fprintf('build_ohio_meal_signal: factor de calibracion = %.3f (std_raw=%.1f [%s] -> std objetivo=%.1f)\n', ...
        calib_factor, std_raw, origen, std_m_ref);
end

function dx = gut_ode(x, Dt, D0, subj)
    Qsto1 = x(1); Qsto2 = x(2); Qgut = x(3);
    Qsto  = Qsto1 + Qsto2;

    a = 5 / (2*D0*(1-subj.b_gut));
    c = 5 / (2*D0*subj.d_gut);
    kempt = subj.kmin_gut + (subj.kmax_gut - subj.kmin_gut)/2 * ...
            ( tanh(a*(Qsto - subj.b_gut*D0)) - tanh(c*(Qsto - subj.d_gut*D0)) + 2 );

    dQsto1 = -subj.kgri_gut*Qsto1 + Dt;
    dQsto2 = -kempt*Qsto2 + subj.kgri_gut*Qsto1;
    dQgut  = -subj.kabs_gut*Qgut + kempt*Qsto2;

    dx = [dQsto1; dQsto2; dQgut];
end

function x_next = rk4_step(f, x, h)
    k1 = f(x);
    k2 = f(x + h/2*k1);
    k3 = f(x + h/2*k2);
    k4 = f(x + h*k3);
    x_next = x + h/6*(k1 + 2*k2 + 2*k3 + k4);
end

function [ts, grams] = resolve_meal_carbs(meal, bolus)
%RESOLVE_MEAL_CARBS  meal.carbs como fuente principal. Ademas:
%   (a) si una comida reportada trae carbs=0, se completa con el
%       bwz_carb_input del bolo mas cercano (<=10 min);
%   (b) si un bolo trae bwz_carb_input>0 sin ninguna comida cercana
%       reportada, se agrega como un evento de comida adicional
%       (bolo dado por alimento sin registro explicito de <meal>).
    ts    = meal.ts(:);
    grams = meal.carbs(:);

    has_bolus_carbs = ~isempty(bolus.ts_begin) && isfield(bolus,'bwz_carb_input');

    if has_bolus_carbs
        for i = 1:numel(bolus.ts_begin)
            c = bolus.bwz_carb_input(i);
            if isnan(c) || c <= 0, continue; end
            if isempty(ts)
                near = false;
            else
                near = any(abs(ts - bolus.ts_begin(i)) <= minutes(10) & grams > 0);
            end
            if ~near
                ts(end+1,1)    = bolus.ts_begin(i);   %#ok<AGROW>
                grams(end+1,1) = c;                    %#ok<AGROW>
            end
        end
    end

    idx0 = find(grams == 0);
    if has_bolus_carbs
        for ii = idx0'
            [dt, j] = min(abs(bolus.ts_begin - ts(ii)));
            if dt <= minutes(10) && bolus.bwz_carb_input(j) > 0
                grams(ii) = bolus.bwz_carb_input(j);
            end
        end
    end

    [ts, order] = sort(ts);
    grams = grams(order);
end