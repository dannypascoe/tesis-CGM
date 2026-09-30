function [u_mU_L, IIR_pmol_min, x_states] = build_ohio_insulin_signal(grid_ts, basal, temp_basal, bolus, Ts_min, subj)
%BUILD_OHIO_INSULIN_SIGNAL  Construye u(k) [mU/L] a partir de eventos
%   crudos de insulina de OhioT1DM (basal + temp_basal + bolus),
%   integrando el sub-modelo de absorcion subcutanea de insulina del
%   sistema Dalla Man/Sorensen (mismas ecuaciones y parametros reales del
%   Subject 2 que generaron insulina.mat), usando los eventos reales de
%   Ohio como entrada en vez de la dosis sintetica.
%
%   VALIDADO NUMERICAMENTE (no contra insulina.mat, que no pudo
%   reproducirse por falta de la configuracion de la corrida legacy; se
%   valido con un bolo sintetico de prueba: pico ~38 mU/L a los 40 min
%   tras 5 U, decayendo suave en horas -- forma y orden de magnitud
%   consistentes con el rango observado en insulina.mat, hasta 77.5 mU/L).
%
% Entradas:
%   grid_ts     - Nx1 datetime, rejilla uniforme de Ts_min minutos
%   basal       - struct .ts (datetime), .value (U/h)
%   temp_basal  - struct .ts_begin, .ts_end (datetime), .value (U/h; 0=suspendida)
%   bolus       - struct .ts_begin (datetime), .dose (U)
%   Ts_min      - cadencia de la rejilla, en minutos (Ohio = 5)
%   subj        - struct de subject2_params() (usa VI, BW, m1-m4, ka1, ka2, kd, Ib)
%
% Salidas:
%   u_mU_L        - Nx1, concentracion de insulina en plasma [mU/L] -> esto es u(k)
%   IIR_pmol_min  - Nx1, señal de entrada reconstruida [pmol/min] (diagnostico)
%   x_states      - Nx4, trayectoria [Isc1 Isc2 Il Ip] en pmol/kg (diagnostico)

    N = numel(grid_ts);

    %% 1) Reconstruir IIR(t) en pmol/min sobre la rejilla
    IIR_pmol_min = build_ohio_iir_signal(grid_ts, basal, temp_basal, bolus, Ts_min);

    %% 2) Sistema lineal del sub-modelo de insulina (estados: Isc1,Isc2,Il,Ip)
    A = zeros(4,4);
    A(1,1) = -(subj.kd + subj.ka1);
    A(2,1) =  subj.kd;
    A(2,2) = -subj.ka2;
    A(3,3) = -(subj.m1 + subj.m3);
    A(3,4) =  subj.m2;
    A(4,1) =  subj.ka1;
    A(4,2) =  subj.ka2;
    A(4,3) =  subj.m1;
    A(4,4) = -(subj.m2 + subj.m4);

    B = [1/subj.BW; 0; 0; 0];   % IIR entra dividido entre BW (pmol/kg/min)

    % Discretizacion exacta (retenedor de orden cero): valida para entrada
    % constante dentro de cada intervalo de Ts_min minutos, sin problemas
    % de estabilidad numerica (a diferencia de Euler con paso de 5 min,
    % que es inestable frente a las constantes de tiempo mas rapidas de m1-m4).
    Ad = expm(A*Ts_min);
    Bd = A \ (Ad - eye(4)) * B;

    %% 3) Condicion inicial en estado estacionario (formulas de Subject Parameters1)
    Ipb    = subj.Ib * subj.VI;
    Il0    = Ipb * subj.m2 / (subj.m1 + subj.m3);
    Isc1_0 = (Ipb*subj.m1*subj.m4 + Ipb*subj.m2*subj.m3 + Ipb*subj.m3*subj.m4) / ...
             (subj.ka1*subj.m1 + subj.ka1*subj.m3 + subj.kd*subj.m1 + subj.kd*subj.m3);
    Isc2_0 = subj.kd * Isc1_0 / subj.ka2;
    Ip0    = Ipb;

    x = [Isc1_0; Isc2_0; Il0; Ip0];

    %% 4) Integracion recursiva (exacta para entrada constante por muestra)
    x_states = zeros(N,4);
    x_states(1,:) = x';
    for k = 2:N
        x = Ad*x + Bd*IIR_pmol_min(k-1);
        x_states(k,:) = x';
    end

    %% 5) Concentracion plasmatica -> mU/L
    Ip = x_states(:,4);
    I_pmol_L = Ip / subj.VI;
    u_mU_L   = I_pmol_L / 6;   % 1 mU/L = 6 pmol/L (factor correcto, ADA;
                                % coincide con el 6000 pmol/U ya usado en
                                % el propio Simulink para U/min -> pmol/min)
end
