%% =========================================================================
%  PLOT_DATOS_SORENSEN.M - Figura de datos simulados (modelo de Sorensen)
%  =========================================================================
%  Regenera la figura de 3 paneles:
%    (a) glucosa G_CGM(t), (b) insulina plasmatica I(t), (c) absorcion de
%    carbohidratos CHO(t).
%  Usa los mismos .mat que main_training.m a traves de load_and_prepare_data.m
%
%  NOTA: en el codigo, u (insulina.mat) y m (meal.mat) son los mismos
%  vectores en entrenamiento y en prueba; al cambiar de conjunto solo
%  cambia el panel (a).
%  =========================================================================

clear; clc; close all;

%% ===================== RUTAS =====================
scriptPath = 'D:\Documentos\02 MROI\07 Tesis\03 Extension Tesis\CGM_Project\CGM_Project\matlab';
addpath(fullfile(scriptPath, 'data_preparation'));
addpath(fullfile(scriptPath, 'utils'));

data_path = fullfile(scriptPath, 'data', 'raw');
out_file  = fullfile(scriptPath, 'figures', 'datos_sinteticos_sorensen.eps');

%% ===================== CONFIGURACION =====================
conjunto = 'train';     % 'train' -> glucose.mat     (figura actual)
                        % 'test'  -> glucose_sim.mat (prueba, sin fallas)

% Colores [R G B]
col_g = [0.00 0.00 0.00];   % negro
col_u = [0.00 0.20 0.55];   % azul oscuro
col_m = [0.35 0.10 0.45];   % morado oscuro
lw    = 1.2;            % grosor de linea

% Tamanos de letra
fs_ticks  = 9;          % numeros de los ejes
fs_labels = 11;         % nombres de los ejes
fs_letra  = 12;         % (a), (b), (c)

xlabel_txt = 'Muestras';   % solo en el panel inferior

% Tamano de la figura (cm)
fig_w = 16;
fig_h = 15;

% Limites del eje Y (los de la figura actual); [] = automatico
ylim_g = [0 250];
ylim_u = [0 80];
ylim_m = [0 600];

%% ===================== DATOS =====================
[g_train, ~, g_test_real, u, m, ~] = load_and_prepare_data(data_path);

switch conjunto
    case 'train'
        g = g_train;
    case 'test'
        g = g_test_real;
    otherwise
        error('conjunto debe ser ''train'' o ''test''');
end

T = numel(g);
k = 1:T;
u = u(1:T);             % en la prueba, el lazo en linea usa u(k), m(k) con k = 1..T
m = m(1:T);

%% ===================== FIGURA =====================
fig = figure('Units', 'centimeters', 'Position', [2 2 fig_w fig_h], 'Color', 'w');
tl  = tiledlayout(fig, 3, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

senales = {g, u, m};
colores = {col_g, col_u, col_m};
ylabels = {'Glucosa (mg/dL)', 'Insulina (mU/L)', 'Carbohidratos (mg/min)'};
ylims   = {ylim_g, ylim_u, ylim_m};
letras  = {'(a)', '(b)', '(c)'};

for p = 1:3
    ax = nexttile(tl);
    plot(ax, k, senales{p}, 'Color', colores{p}, 'LineWidth', lw);
    grid(ax, 'on');
    box(ax, 'on');
    xlim(ax, [0 T]);
    if ~isempty(ylims{p})
        ylim(ax, ylims{p});
    end

    % Primero el tamano de los ticks: si se cambia despues, MATLAB
    % reescala tambien las etiquetas y se pierde fs_labels
    ax.FontSize = fs_ticks;

    ylabel(ax, ylabels{p}, 'FontSize', fs_labels, 'FontWeight', 'bold');
    letra = ['\fontsize{' num2str(fs_letra) '}\bf' letras{p}];
    if p == 3
        xlabel(ax, {xlabel_txt, letra}, 'FontSize', fs_labels, 'FontWeight', 'bold');
    else
        xlabel(ax, letra, 'FontSize', fs_labels);   % paneles superiores: solo la letra
    end
end

%% ===================== EXPORTAR =====================
if ~exist(fileparts(out_file), 'dir')
    mkdir(fileparts(out_file));
end
exportgraphics(fig, out_file, 'ContentType', 'vector');
fprintf('Figura guardada en: %s\n', out_file);