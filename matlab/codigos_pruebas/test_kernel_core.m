%% TEST_KERNEL_CORE.M - Verificacion numerica del nucleo de los kernels
%  Corre en MATLAB y en Octave (no usa datetime). Comprueba:
%   1) Ganancia estatica en forma cerrada = ganancia numerica del modelo.
%   2) Convolucion con la respuesta al impulso EXACTA del modelo de insulina
%      == recursion ODE de build_ohio_insulin_signal (tras el pre-roll).
%   3) Parametros de los kernels gamma (moda y percentil 95 pedidos).
%   4) Meal: con la forma normalizada del propio modelo gastrico, el kernel
%      reproduce una comida aislada de cualquier tamano, y se mide cuanto se
%      desvia cuando dos comidas se solapan (limite de cualquier kernel lineal).
clear; clc;
subj = subject2_params();
Ts = 5;
scriptPath = 'D:\Documentos\02 MROI\07 Tesis\03 Extension Tesis\CGM_Project\CGM_Project\matlab';
addpath(genpath(scriptPath));
%% ---- 1) Ganancia estatica ------------------------------------------------
A = zeros(4,4);
A(1,1) = -(subj.kd + subj.ka1);  A(2,1) = subj.kd;  A(2,2) = -subj.ka2;
A(3,3) = -(subj.m1 + subj.m3);   A(3,4) = subj.m2;
A(4,1) = subj.ka1; A(4,2) = subj.ka2; A(4,3) = subj.m1; A(4,4) = -(subj.m2 + subj.m4);
B = [1/subj.BW; 0; 0; 0];
C = [0 0 0 1/subj.VI/6];
G_num = -C * (A \ B);
G_cf  = ohio_insulin_dc_gain(subj);
fprintf('1) Ganancia estatica: numerica = %.6f | forma cerrada = %.6f | dif = %.2e\n', G_num, G_cf, abs(G_num-G_cf));
assert(abs(G_num-G_cf) < 1e-9);

%% ---- 2) Equivalencia con el modelo de insulina ---------------------------
N   = 24*12*3;                                  % 3 dias a 5 min
IIR = 80*ones(N,1);                             % basal ~0.8 U/h en pmol/min
IIR(24*12 + 100)  = IIR(24*12 + 100)  + 6000*4/Ts;   % bolo 4 U
IIR(24*12 + 220)  = IIR(24*12 + 220)  + 6000*2.5/Ts; % bolo 2.5 U
IIR(2*24*12 + 40) = IIR(2*24*12 + 40) + 6000*6/Ts;   % bolo 6 U

Ad = expm(A*Ts);  Bd = A \ (Ad - eye(4)) * B;
Ipb = subj.Ib*subj.VI;
Il0 = Ipb*subj.m2/(subj.m1+subj.m3);
Isc1_0 = (Ipb*subj.m1*subj.m4 + Ipb*subj.m2*subj.m3 + Ipb*subj.m3*subj.m4) / ...
         (subj.ka1*subj.m1 + subj.ka1*subj.m3 + subj.kd*subj.m1 + subj.kd*subj.m3);
x = [Isc1_0; subj.kd*Isc1_0/subj.ka2; Il0; Ipb];
u_ode = zeros(N,1);  u_ode(1) = C*x;
for k = 2:N
    x = Ad*x + Bd*IIR(k-1);
    u_ode(k) = C*x;
end

Nk = 288;                                       % 24 h de kernel
h = zeros(Nk,1);  v = Bd;
for n = 1:Nk, h(n) = C*v; v = Ad*v; end         % h[n] = C*Ad^(n-1)*Bd
w_exact = h / (G_cf * Ts);                      % normalizado: sum(w)*Ts ~ 1
fprintf('2) Kernel exacto: sum(w)*Ts = %.6f (truncado a 24 h)\n', sum(w_exact)*Ts);
w_exact = w_exact / (sum(w_exact)*Ts);
u_ker = G_cf * apply_absorption_kernel(IIR, w_exact, Ts);
sel = (24*12+1):N;                              % descarta 24 h de pre-roll
err = max(abs(u_ker(sel) - u_ode(sel)));
fprintf('   max|u_kernel_exacto - u_ODE| tras pre-roll = %.3e mU/L (u_ODE max = %.1f)\n', err, max(u_ode(sel)));
assert(err < 1e-4 * max(u_ode(sel)));

%% ---- 3) Parametros de los kernels gamma ----------------------------------
presets = {'insulina, literatura', 70, 270; 'CHO, literatura', 45, 210; ...
           'insulina, ajustado al ODE', 46, 259; 'CHO, ajustado al ODE', 30, 280};
for i = 1:size(presets,1)
    [w, kinf] = ohio_gamma_kernel(presets{i,2}, presets{i,3}, Ts, 720);
    lag = ((1:numel(w))' - 0.5)*Ts;
    [~, imax] = max(w);
    cdf = cumsum(w)*Ts;
    t95_num = lag(find(cdf >= 0.95, 1));
    fprintf('3) %-26s k=%.3f theta=%.2f min | moda pedida %3d -> %3.0f | t95 pedido %3d -> %3.0f | media %.0f min\n', ...
        presets{i,1}, kinf.shape_k, kinf.scale_min, presets{i,2}, lag(imax), presets{i,3}, t95_num, kinf.mean_min);
end

%% ---- 4) Meal: modelo gastrico vs kernel con su propia forma --------------
kabs = subj.kabs_gut;
nT = 24*12;
k0 = 20;  D_ref = 60e3;
P = zeros(nT,1);  P(k0) = D_ref;
D0 = ones(nT,1);  D0(k0:end) = D_ref;
Q_ref = ohio_gut_sim(P, D0, Ts, subj);
g_ode = Q_ref(k0+1:end) * kabs / D_ref;            % forma normalizada [1/min]
fprintf('4) Forma del modelo gastrico (60 g): pico en %d min, area normalizada = %.4f (ideal 1)\n', ...
    (find(g_ode==max(g_ode),1))*Ts - Ts/2, sum(g_ode)*Ts);
w_gut = g_ode / (sum(g_ode)*Ts);

% (a) comida aislada de 10 g con el kernel de la forma de 60 g
P10 = zeros(nT,1);  P10(k0) = 10e3;  D010 = ones(nT,1);  D010(k0:end) = 10e3;
Q10_ode = ohio_gut_sim(P10, D010, Ts, subj);
Q10_ker = apply_absorption_kernel(P10/Ts, w_gut, Ts) / kabs;
e_iso = max(abs(Q10_ker - Q10_ode)) / max(Q10_ode);
fprintf('   (a) comida aislada 10 g: error relativo max = %.2e\n', e_iso);

% (b) dos comidas solapadas (60 g y 40 g, separadas 2 h)
kb = k0 + 24;
P2 = zeros(nT,1);  P2(k0) = 60e3;  P2(kb) = 40e3;
D02 = ones(nT,1);  D02(k0:end) = 60e3;  D02(kb:end) = 40e3;
Q2_ode = ohio_gut_sim(P2, D02, Ts, subj);
Q2_ker = apply_absorption_kernel(P2/Ts, w_gut, Ts) / kabs;
e_ov = max(abs(Q2_ker - Q2_ode)) / max(Q2_ode);
fprintf('   (b) dos comidas solapadas (2 h): error relativo max = %.3f (%.1f%% del pico)\n', e_ov, 100*e_ov);

fprintf('\nVerificaciones numericas (sin datetime) OK.\n');

%% ---- 5) Integracion con los builders reales (solo MATLAB: usa datetime) ----
if exist('datetime') == 0 %#ok<EXIST>
    fprintf('5) Omitida: este entorno no tiene datetime (correr en MATLAB).\n');
    return;
end
Ts_min = 5;
nG     = 24*12*3;
grid_ts = datetime(2026,1,1) + minutes(Ts_min*(0:nG-1))';
basal      = struct('ts', grid_ts(1), 'value', 0.8);
temp_basal = struct('ts_begin', datetime.empty(0,1), 'ts_end', datetime.empty(0,1), 'value', []);
bolus = struct('ts_begin', grid_ts([24*12+100; 24*12+220; 2*24*12+40]), ...
               'ts_end',   grid_ts([24*12+100; 24*12+220; 2*24*12+40]), ...
               'dose',     [4; 2.5; 6], 'type', {{'normal';'normal';'normal'}}, ...
               'bwz_carb_input', [NaN; NaN; NaN]);
meal  = struct('ts', grid_ts([24*12+110; 2*24*12+60]), 'carbs', [60; 25]);   % comidas aisladas

% 5a) Insulina: kernel exacto del ODE dentro del builder real == build_ohio_insulin_signal
u_ode5 = build_ohio_insulin_signal(grid_ts, basal, temp_basal, bolus, Ts_min, subj);
[~, ~, ~, ~, ~, cf] = build_ohio_meal_signal(grid_ts, meal, bolus, Ts_min, subj, 153.8637, 1000);
[u_k5, ~] = build_ohio_kernel_signals(grid_ts, basal, temp_basal, bolus, meal, Ts_min, subj, cf, ...
    struct('ins_kernel_override', w_exact));
sel5 = (24*12+1):nG;
e5a = max(abs(u_k5(sel5) - u_ode5(sel5)));
fprintf('5a) Insulina, builder real: max|u_kernel_exacto - u_ODE| = %.3e mU/L\n', e5a);
assert(e5a < 1e-3);

% 5b) Comida aislada: kernel con la forma normalizada del ODE == build_ohio_meal_signal
m_ode5 = build_ohio_meal_signal(grid_ts, meal, bolus, Ts_min, subj, 153.8637, 1000);
[~, m_k5] = build_ohio_kernel_signals(grid_ts, basal, temp_basal, bolus, meal, Ts_min, subj, cf, ...
    struct('cho_kernel_override', w_gut));
e5b = max(abs(m_k5 - m_ode5)) / max(m_ode5);
fprintf('5b) Comida aislada, builder real: error relativo max = %.2e\n', e5b);
assert(e5b < 1e-2);

% 5c) Los modos del loader deben producir vectores del mismo largo y sin NaN
[uk, mk, info] = build_ohio_kernel_signals(grid_ts, basal, temp_basal, bolus, meal, Ts_min, subj, cf);
assert(all(isfinite(uk)) && all(isfinite(mk)));
fprintf('5c) Kernel de literatura: k_ins=%.2f, k_cho=%.2f, G_dc=%.4f | u max=%.1f mU/L | m max=%.1f\n', ...
    info.ins_kernel.shape_k, info.cho_kernel.shape_k, info.G_dc, max(uk), max(mk));
fprintf('\nTodas las verificaciones pasaron.\n');
