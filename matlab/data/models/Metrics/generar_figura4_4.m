%% ========================================================================
%  FIGURAS - Seccion 4.4 (Pipeline desacoplado, 16 combinaciones)
%
%  Genera y exporta en .eps (vectorial):
%    FIGURA A: Mapas de calor 4x4 F1 y MAE lado a lado
%              -> muestra que F1 depende solo del detector (filas)
%                 y MAE depende solo del imputador (columnas)
%    FIGURA B: Dispersion F1 vs RMSE, 16 combinaciones,
%              con GRU->GRU y GRU->1D-CNN destacados
%
%  Fuente de datos: pipeline_comparison_nuevo.xlsx
%                    hoja "Lista_16_combinaciones"
%
%  NOTA: la Figura A se construye con imagesc + texto (no con heatmap())
%  porque dos objetos heatmap() nativos en una misma figura chocan en su
%  layout interno (uno termina invisible). imagesc + tiledlayout es mas
%  robusto y ademas exporta como vectorial real en el .eps.
% ========================================================================

clear; clc; close all;

%% --------------------------- CONFIGURACION ------------------------------
data_file  = 'pipeline_comparison_nuevo.xlsx';   % ajustar ruta si es necesario
hoja_datos = 'Lista_16_combinaciones';
output_dir = 'figuras_4_4';
if ~exist(output_dir, 'dir'); mkdir(output_dir); end

archivo_fig_heatmaps = fullfile(output_dir, 'fig_4_4_heatmaps_F1_MAE.eps');
archivo_fig_pareto   = fullfile(output_dir, 'fig_4_4_pareto_F1_RMSE.eps');

%% --------------------------- CARGA DE DATOS ------------------------------
T = readtable(data_file, 'Sheet', hoja_datos);

det_col = strtrim(string(T.Detector));
imp_col = strtrim(string(T.Imputador));

% Orden fijo de arquitecturas (coincide con las tablas ya redactadas en 4.4)
arch_keys   = ["CNN_LSTM", "GRU", "1DCNN", "TRANSFORMER_LSTM"];
arch_labels = {'CNN-LSTM', 'GRU', '1D-CNN', 'Transformer-LSTM'};
n = numel(arch_keys);

F1_mat   = nan(n);   % filas = Detector, columnas = Imputador
MAE_mat  = nan(n);
RMSE_mat = nan(n);

for i = 1:n
    for j = 1:n
        mask = (det_col == arch_keys(i)) & (imp_col == arch_keys(j));
        if any(mask)
            F1_mat(i,j)   = T.F1_mean(mask);
            MAE_mat(i,j)  = T.MAE_mean(mask);
            RMSE_mat(i,j) = T.RMSE_mean(mask);
        else
            warning('No se encontro la combinacion %s -> %s en %s', ...
                     arch_keys(i), arch_keys(j), hoja_datos);
        end
    end
end

% Verificacion de independencia (informativo, no se grafica)
fprintf('--- Verificacion de independencia detector-imputador ---\n');
fprintf('Desv. estandar de F1 entre columnas, por fila (esperado ~0):\n');
disp(std(F1_mat, 0, 2)');
fprintf('Desv. estandar de MAE entre filas, por columna (esperado ~0):\n');
disp(std(MAE_mat, 0, 1));

%% ========================================================================
%  FIGURA A - Mapas de calor 4x4 (F1 y MAE) lado a lado
% ========================================================================
figA = figure('Color', 'w', 'Position', [100 100 1150 480]);
tlo  = tiledlayout(figA, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

axF1 = nexttile(tlo);
dibujar_heatmap_manual(axF1, F1_mat, arch_labels, parula, '%.3f', ...
                        'F1-score');

axMAE = nexttile(tlo);
dibujar_heatmap_manual(axMAE, MAE_mat, arch_labels, flipud(parula), '%.2f', ...
                        'MAE (mg/dL)');

title(tlo, ['Independencia entre Detección (F1) e Imputación (MAE) ' ...
            'en el Pipeline Desacoplado'], 'FontWeight', 'bold', 'FontSize', 12);

exportgraphics(figA, archivo_fig_heatmaps, 'ContentType', 'vector');
fprintf('\nFigura A (heatmaps) guardada en: %s\n', archivo_fig_heatmaps);

%% ========================================================================
%  FIGURA B - Dispersion F1 vs RMSE, co-optimos destacados
% ========================================================================
F1_all   = T.F1_mean;
RMSE_all = T.RMSE_mean;

idx_gru_gru   = (det_col == "GRU") & (imp_col == "GRU");
idx_gru_1dcnn = (det_col == "GRU") & (imp_col == "1DCNN");
idx_resto     = ~(idx_gru_gru | idx_gru_1dcnn);

% Optimo global en el plano F1-RMSE: GRU->1D-CNN.
% Esto ocurre por construccion: F1 depende solo del detector (GRU es el
% detector con mayor F1) y RMSE depende solo del imputador (1D-CNN es el
% imputador con menor RMSE) -- ver Figura A. Por lo tanto GRU->1D-CNN es
% el UNICO punto no dominado en este plano.
F1_opt   = F1_all(idx_gru_1dcnn);
RMSE_opt = RMSE_all(idx_gru_1dcnn);

figB = figure('Name', 'F1 vs RMSE - Co-optimos', 'Color', 'w', ...
              'Position', [100 100 920 560]);
hold on; box on; grid on;

xl = [min(RMSE_all)-0.6, max(RMSE_all)+1.0];
yl = [min(F1_all)-0.02, max(F1_all)+0.02];

% Region donde ninguna combinacion iguala o supera a GRU->1D-CNN en ambos ejes
patch([xl(1) RMSE_opt RMSE_opt xl(1)], [F1_opt F1_opt yl(2) yl(2)], ...
      [0.90 0.95 0.90], 'EdgeColor', 'none', 'FaceAlpha', 0.6, ...
      'HandleVisibility', 'off');
xline(RMSE_opt, '--', 'Color', [0.4 0.4 0.4], 'HandleVisibility', 'off');
yline(F1_opt,   '--', 'Color', [0.4 0.4 0.4], 'HandleVisibility', 'off');

% Co-optimos
scatter(RMSE_all(idx_gru_gru), F1_all(idx_gru_gru), 170, [0.85 0.10 0.10], ...
        'p', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1, ...
        'DisplayName', 'GRU \rightarrow GRU (MAE global mínimo)');
scatter(RMSE_all(idx_gru_1dcnn), F1_all(idx_gru_1dcnn), 170, [0.10 0.55 0.20], ...
        'd', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1, ...
        'DisplayName', 'GRU \rightarrow 1D-CNN (óptimo F1-RMSE)');

% Resto de combinaciones (14 puntos)
scatter(RMSE_all(idx_resto), F1_all(idx_resto), 65, [0.5 0.5 0.5], 'filled', ...
        'MarkerFaceAlpha', 0.5, 'DisplayName', 'Otras combinaciones');

text(RMSE_all(idx_gru_gru)+0.10, F1_all(idx_gru_gru), 'GRU\rightarrowGRU', ...
     'FontSize', 10, 'VerticalAlignment', 'bottom');
text(RMSE_all(idx_gru_1dcnn)+0.10, F1_all(idx_gru_1dcnn), 'GRU\rightarrow1D-CNN', ...
     'FontSize', 10, 'VerticalAlignment', 'bottom');

xlabel('RMSE (mg/dL)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('F1-score', 'FontSize', 12, 'FontWeight', 'bold');
title('F1 (Detección) vs RMSE (Imputación) - 16 Combinaciones', ...
      'FontSize', 13, 'FontWeight', 'bold');
legend('Location', 'eastoutside', 'FontSize', 10);
set(gca, 'FontSize', 11);
xlim(xl); ylim(yl);

exportgraphics(figB, archivo_fig_pareto, 'ContentType', 'vector');
fprintf('Figura B (F1 vs RMSE) guardada en: %s\n', archivo_fig_pareto);

fprintf('\nListo. Punto no dominado en el plano F1-RMSE: GRU->1D-CNN ');
fprintf('(F1=%.3f, RMSE=%.3f mg/dL)\n', F1_opt, RMSE_opt);


%% ========================================================================
%  FUNCIONES LOCALES
% ========================================================================
function dibujar_heatmap_manual(ax, datos, etiquetas, cmap, fmt, titulo)
    % Dibuja un heatmap 100% vectorial (imagesc + texto) en el eje ax.
    n = size(datos, 1);
    imagesc(ax, datos);
    set(ax, 'YDir', 'reverse');
    colormap(ax, cmap);
    axis(ax, 'square');
    caxis(ax, [min(datos(:)) max(datos(:))]); %#ok<CAXIS>
    hold(ax, 'on');

    % Bordes entre celdas
    for k = 0.5:1:(n+0.5)
        plot(ax, [k k], [0.5 n+0.5], 'Color', [0.25 0.25 0.25], 'LineWidth', 0.75);
        plot(ax, [0.5 n+0.5], [k k], 'Color', [0.25 0.25 0.25], 'LineWidth', 0.75);
    end

    % Etiquetas numericas con color de texto contrastante
    clims = caxis(ax); %#ok<CAXIS>
    for i = 1:n
        for j = 1:n
            col = color_contraste(datos(i,j), clims, cmap);
            text(ax, j, i, sprintf(fmt, datos(i,j)), ...
                 'HorizontalAlignment', 'center', 'FontSize', 10, 'Color', col);
        end
    end

    set(ax, 'XTick', 1:n, 'XTickLabel', etiquetas, 'XTickLabelRotation', 20, ...
            'YTick', 1:n, 'YTickLabel', etiquetas, 'FontSize', 10, 'TickLength', [0 0]);
    xlabel(ax, 'Arquitectura Imputadora', 'FontSize', 11, 'FontWeight', 'bold');
    ylabel(ax, 'Arquitectura Detectora', 'FontSize', 11, 'FontWeight', 'bold');
    title(ax, titulo, 'FontSize', 12, 'FontWeight', 'bold');
    colorbar(ax);
    box(ax, 'on');
end

function col = color_contraste(valor, clims, cmap)
    % Elige texto blanco o negro segun la luminancia del color de fondo.
    frac = (valor - clims(1)) / max(clims(2) - clims(1), eps);
    frac = min(max(frac, 0), 1);
    idx = 1 + round(frac * (size(cmap,1) - 1));
    rgb = cmap(idx, :);
    luminancia = 0.299*rgb(1) + 0.587*rgb(2) + 0.114*rgb(3);
    if luminancia > 0.55
        col = [0 0 0];
    else
        col = [1 1 1];
    end
end