%% =========================================================================
%  PLOT_CONJUNTO_PRUEBA.M - Figura del conjunto de prueba con fallas
%  =========================================================================
%  Replica la figura del conjunto de prueba sintetico (Sorensen): senal
%  limpia de referencia, lectura con fallo y regiones sombreadas por tipo
%  de fallo. Estilo alineado con las figuras de resultados (Figura 4.3):
%  referencia en negro, lectura con fallo gris punteada, desconexion en
%  rosa, ruido blanco en amarillo y leyenda arriba, fuera del eje.
%
%  Tipo de fallo por muestra (true_f solo es binario):
%    desconexion  -> true_f == 1 y lectura < 10 mg/dL (o NaN)
%    ruido blanco -> true_f == 1 y el resto
%  =========================================================================

clear; clc; close all;

%% ===================== RUTAS =====================
scriptPath = 'D:\Documentos\02 MROI\07 Tesis\03 Extension Tesis\CGM_Project\CGM_Project\matlab';
addpath(fullfile(scriptPath, 'data_preparation'));
addpath(fullfile(scriptPath, 'utils'));

data_path = fullfile(scriptPath, 'data', 'raw');
out_file  = fullfile(scriptPath, 'figures', 'conjunto_prueba.eps');

%% ===================== CONFIGURACION =====================
eje_x       = 'horas';  % 'horas' (como la Figura 4.3) o 'muestras'
DT_MIN      = 5;        % minutos por muestra
UMBRAL_DESC = 10;       % mg/dL, mismo criterio "< 10" del pipeline

% Colores [R G B]
col_ref   = [0.00 0.00 0.00];   % referencia (senal limpia)
col_fallo = [0.60 0.60 0.60];   % lectura con fallo
col_desc  = [1.00 0.82 0.82];   % desconexion inyectada
col_ruido = [1.00 0.93 0.70];   % ruido blanco inyectado

lw_ref   = 1.3;
lw_fallo = 1.1;

% Tamanos de letra
fs_ticks   = 10;
fs_labels  = 13;
fs_leyenda = 11;

% Tamano de la figura (cm) y limites del eje Y
fig_w  = 16;
fig_h  = 8.5;
ylim_g = [0 180];

%% ===================== DATOS =====================
[~, g_test, g_test_real, ~, ~, true_f] = load_and_prepare_data(data_path);

T      = numel(g_test);
true_f = logical(true_f(:));
g_test = g_test(:);

switch eje_x
    case 'horas'
        x    = (0:T-1)' * DT_MIN / 60;
        dx   = DT_MIN / 60;
        xlab = 'Tiempo (horas)';
    case 'muestras'
        x    = (1:T)';
        dx   = 1;
        xlab = 'Muestras';
    otherwise
        error('eje_x debe ser ''horas'' o ''muestras''');
end

mask_desc  = true_f & (g_test < UMBRAL_DESC | isnan(g_test));
mask_ruido = true_f & ~mask_desc;

% La lectura con fallo solo se dibuja donde hay fallo
g_fallo = g_test;
g_fallo(~true_f) = NaN;

fprintf('Muestras con desconexion:  %d\n', sum(mask_desc));
fprintf('Muestras con ruido blanco: %d\n', sum(mask_ruido));

%% ===================== FIGURA =====================
fig = figure('Units', 'centimeters', 'Position', [2 2 fig_w fig_h], 'Color', 'w');
ax  = axes(fig);
hold(ax, 'on');
xlim(ax, [0 x(end)]);
ylim(ax, ylim_g);

% Regiones sombreadas (primero, para que queden debajo de las curvas)
h_desc  = dibujar_regiones(ax, x, mask_desc,  dx, ylim_g, col_desc);
h_ruido = dibujar_regiones(ax, x, mask_ruido, dx, ylim_g, col_ruido);

% Curvas
h_ref   = plot(ax, x, g_test_real, '-',  'Color', col_ref,   'LineWidth', lw_ref);
h_fallo = plot(ax, x, g_fallo,     '--', 'Color', col_fallo, 'LineWidth', lw_fallo);

% Formato de ejes (FontSize de ticks antes que las etiquetas)
grid(ax, 'on');
box(ax, 'on');
ax.Layer    = 'top';
ax.FontSize = fs_ticks;
xlabel(ax, xlab,              'FontSize', fs_labels, 'FontWeight', 'bold');
ylabel(ax, 'Glucosa (mg/dL)', 'FontSize', fs_labels, 'FontWeight', 'bold');

% Leyenda dentro del eje, esquina inferior derecha (zona libre: despues de
% la segunda desconexion la glucosa no baja de ~80 mg/dL)
lgd = legend(ax, [h_ref, h_fallo, h_desc, h_ruido], ...
    {'Referencia', 'Lectura con fallo', 'Desconexión inyectada', 'Ruido blanco inyectado'}, ...
    'Location', 'southeast', 'NumColumns', 1, 'Box', 'on', 'FontSize', fs_leyenda);
lgd.Color       = 'w';
lgd.EdgeColor   = [0.7 0.7 0.7];
lgd.ItemTokenSize = [20 10];   % muestras de linea/parche mas cortas

%% ===================== EXPORTAR =====================
if ~exist(fileparts(out_file), 'dir')
    mkdir(fileparts(out_file));
end
exportgraphics(fig, out_file, 'ContentType', 'vector');
fprintf('Figura guardada en: %s\n', out_file);

%% ===================== FUNCIONES LOCALES =====================
function h = dibujar_regiones(ax, x, mask, dx, yl, col)
%DIBUJAR_REGIONES Sombrea cada tramo contiguo de mask. Devuelve el handle
%   del primer tramo (para la leyenda); los demas no aparecen en ella.
    d   = diff([0; mask(:); 0]);
    ini = find(d == 1);
    fin = find(d == -1) - 1;
    if isempty(ini)
        error('No se encontraron tramos para una de las regiones de fallo.');
    end
    for r = 1:numel(ini)
        x1 = x(ini(r)) - dx/2;
        x2 = x(fin(r)) + dx/2;
        hp = patch(ax, [x1 x2 x2 x1], [yl(1) yl(1) yl(2) yl(2)], col, ...
            'EdgeColor', 'none');
        if r == 1
            h = hp;
        else
            hp.HandleVisibility = 'off';
        end
    end
end