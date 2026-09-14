%% =========================================================================
%  MAIN_CLINICAL_METRICS.M - Análisis de Impacto Clínico del Pipeline CGM
%  =========================================================================
%  Descripción: Calcula los indicadores clínicos del Consensus Report
%               ADA/EASD 2019 (Battelino et al.) para cuantificar el impacto
%               clínico de la imputación propuesta en 3 escenarios.
%
%  Escenarios comparados:
%    1. Referencia    : g_test_real  (señal real limpia, gold standard)
%    2. Sin imputación: g_test_real  con instantes de fallo → NaN
%    3. Con pipeline  : g_final_all  (salida del detector + imputador)
%
%  Indicadores (Consensus Report ADA/EASD 2019):
%    - TIR   : Time In Range        70–180 mg/dL  (objetivo >= 70%)
%    - TBR L1: Time Below Range L1  < 70 mg/dL    (objetivo < 4%)
%    - TBR L2: Time Below Range L2  < 54 mg/dL    (objetivo < 1%)
%    - TAR L1: Time Above Range L1  180–250 mg/dL (objetivo < 25%)
%    - TAR L2: Time Above Range L2  > 250 mg/dL   (objetivo < 5%)
%    - Mean  : Glucosa media (mg/dL)
%    - SD    : Desviación estándar (mg/dL)
%    - CV    : Coeficiente de variación, CV = SD/Mean*100 (objetivo <= 36%)
%    - GMI   : Glucose Management Indicator, estimador de HbA1c
%              GMI(%) = 3.31 + 0.02392 * Mean(mg/dL)   (Bergenstal 2018)
%
%  Combos evaluados (co-óptimos del Punto 2):
%    - GRU -> 1D-CNN     (mejor F1/Recall — prioriza no perder fallos)
%    - CNN-LSTM -> 1D-CNN (mejor Precisión — prioriza no reemplazar válidos)
%
%  Prerrequisitos:
%    - Haber ejecutado main_pipeline_integrado_v2.m
%    - Archivos results_pipeline_<det>_<imp>_latest.mat disponibles
%
%  Salidas:
%    - Tabla comparativa de 9 indicadores (3 escenarios x 2 combos)
%    - Figura de perfil glucémico con zonas de riesgo
%    - Figura de barras apiladas estilo AGP
%    - Archivo clinical_metrics_results_<timestamp>.mat
%  =========================================================================

clear; clc; close all;

%% ===================== AGREGAR CARPETAS AL PATH =====================
% MODIFICAR SEGÚN TU UBICACIÓN
scriptPath = 'D:\Documentos\02 MROI\07 Tesis\03 Extension Tesis\CGM_Project\CGM_Project';

addpath(fullfile(scriptPath, 'architectures'));
addpath(fullfile(scriptPath, 'data_preparation'));
addpath(fullfile(scriptPath, 'utils'));
addpath(fullfile(scriptPath, 'data'));

fprintf('Carpetas agregadas al path de MATLAB\n');

%% ===================== CONFIGURACIÓN =====================

% Combos co-óptimos a analizar
combos_to_analyze = {
    'gru', 'gru';      % GRU -> GRU      (mínimo MAE global)
    'gru', '1dcnn';    % GRU -> 1D-CNN   (mejor RMSE/R²/MAE en fallo)
};

% Ruta raíz de modelos/resultados
models_path = fullfile(scriptPath, 'data', 'models');

% -----------------------------------------------------------------------
% CARPETAS DE ENTRADA Y SALIDA
%   Los resultados del pipeline (bloque_C) son la ENTRADA de este script.
%   Las figuras y métricas clínicas (bloque_D) son la SALIDA.
% -----------------------------------------------------------------------
input_path  = fullfile(models_path, 'bloque_C');   % resultados pipeline integrado
output_path = fullfile(models_path, 'bloque_D');   % salida: métricas clínicas Sorensen

if ~exist(input_path, 'dir')
    error('❌ No se encontró bloque_C/. Ejecuta primero main_pipeline_integrado_v2.m');
end
if ~exist(output_path, 'dir')
    mkdir(output_path);
    fprintf('📁 Carpeta creada: %s\n', output_path);
end

% Intervalo de muestreo CGM (minutos)
DT_MINUTES = 5;

% Umbrales clínicos (Consensus Report ADA/EASD 2019) — mg/dL
THR_HYPO_L2  =  54;
THR_HYPO_L1  =  70;
THR_NORMAL_H = 180;
THR_HYPER_L2 = 250;

% Objetivos de variabilidad y control glucémico (Battelino 2019; Bergenstal 2018)
CV_TARGET    = 36;     % CV objetivo (%)  - estabilidad glucémica
GMI_TARGET   =  7.0;   % GMI objetivo (%) - control glucémico equivalente HbA1c

% Límite fisiológico mínimo (filtrar outliers, no relacionado con fallos)
MIN_VALID = 20;

timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
all_clinical = struct();

fprintf('Indicadores: TIR | TBR L1/L2 | TAR L1/L2 | Mean | SD | CV | GMI\n');
fprintf('Combos a evaluar: %d\n\n', size(combos_to_analyze, 1));

%% ===================== BUCLE PRINCIPAL POR COMBO =====================

for c = 1:size(combos_to_analyze, 1)

    arch_det   = combos_to_analyze{c, 1};
    arch_imp   = combos_to_analyze{c, 2};
    combo_name = sprintf('%s_det_%s_imp', arch_det, arch_imp);

    fprintf('COMBO %d/%d: %s -> %s\n', c, size(combos_to_analyze,1), ...
        upper(arch_det), upper(arch_imp));

    % --- Cargar resultados del pipeline (desde bloque_C/) ---
    result_file = fullfile(input_path, ...
        sprintf('results_pipeline_%s_latest.mat', combo_name));

    if ~exist(result_file, 'file')
        warning('No encontrado: %s\nSaltando.', result_file);
        continue;
    end

    fprintf('Cargando: %s\n', result_file);
    d = load(result_file, 'g_test_real', 'g_final_all', 'true_f');

    T     = length(d.g_test_real);
    nRuns = size(d.g_final_all, 1);

    fprintf('T=%d muestras | nRuns=%d | %.1f dias\n', T, nRuns, T * DT_MINUTES / 1440);

    n_faults   = sum(d.true_f);
    pct_faults = 100 * n_faults / T;
    fprintf('Fallos: %d muestras (%.1f%%)\n\n', n_faults, pct_faults);

    % ─────────────────────────────────────────────────────────────
    %  PREPARAR LAS 3 SEÑALES
    % ─────────────────────────────────────────────────────────────

    % 1. Referencia: señal limpia completa
    g_ref = d.g_test_real(:)';

    % 2. Sin imputación: señal limpia con instantes de fallo -> NaN
    g_no_imp = d.g_test_real(:)';
    g_no_imp(d.true_f == 1) = NaN;

    % 3. Con pipeline: media de los nRuns runs
    g_pipe = mean(d.g_final_all, 1);

    signals   = {g_ref, g_no_imp, g_pipe};
    scenarios = {'Referencia', 'Sin imputación', 'Con pipeline'};

    % ─────────────────────────────────────────────────────────────
    %  CALCULAR INDICADORES POR ESCENARIO
    % ─────────────────────────────────────────────────────────────

    clear metrics_table;

    for s = 1:3
        sig = signals{s};

        valid = ~isnan(sig) & sig >= MIN_VALID & sig <= 400;
        sv    = sig(valid);
        nv    = sum(valid);

        m.scenario   = scenarios{s};
        m.n_valid    = nv;
        m.data_avail = 100 * nv / T;

        % --- Tiempo en rangos clínicos (Battelino 2019) ---
        m.TIR    = 100 * sum(sv >= THR_HYPO_L1 & sv <= THR_NORMAL_H) / nv;
        m.TBR_L1 = 100 * sum(sv < THR_HYPO_L1)                       / nv;
        m.TBR_L2 = 100 * sum(sv < THR_HYPO_L2)                       / nv;
        m.TAR_L1 = 100 * sum(sv > THR_NORMAL_H & sv <= THR_HYPER_L2) / nv;
        m.TAR_L2 = 100 * sum(sv > THR_HYPER_L2)                      / nv;

        % --- Estadística descriptiva y variabilidad ---
        % Mean: glucosa media (mg/dL)
        % SD:   desviación estándar (mg/dL)
        % CV:   coeficiente de variación (%) — Battelino 2019, objetivo <= 36%
        % GMI:  Glucose Management Indicator (%) — Bergenstal 2018
        %       fórmula: GMI(%) = 3.31 + 0.02392 * Mean_glucose(mg/dL)
        m.mean_g = mean(sv);
        m.sd_g   = std(sv);
        m.CV     = 100 * m.sd_g / m.mean_g;
        m.GMI    = 3.31 + 0.02392 * m.mean_g;

        metrics_table(s) = m;
    end

    % ─────────────────────────────────────────────────────────────
    %  IMPRIMIR TABLA
    % ─────────────────────────────────────────────────────────────

    fprintf('\n');
    fprintf('+-----------------------+------------------+------------------+------------------+-----------------+\n');
    fprintf('| METRICAS CLINICAS     |   Referencia     |  Sin imputacion  |   Con pipeline   |    Objetivo     |\n');
    fprintf('+-----------------------+------------------+------------------+------------------+-----------------+\n');
    fprintf('| Datos disponibles     | %14.1f%% | %14.1f%% | %14.1f%% |      100%%       |\n', ...
        metrics_table(1).data_avail, metrics_table(2).data_avail, metrics_table(3).data_avail);
    fprintf('+-----------------------+------------------+------------------+------------------+-----------------+\n');
    fprintf('| TIR    (70-180)       | %14.1f%% | %14.1f%% | %14.1f%% |      >= 70%%     |\n', ...
        metrics_table(1).TIR,    metrics_table(2).TIR,    metrics_table(3).TIR);
    fprintf('| TBR L1 (<70)          | %14.1f%% | %14.1f%% | %14.1f%% |       < 4%%      |\n', ...
        metrics_table(1).TBR_L1, metrics_table(2).TBR_L1, metrics_table(3).TBR_L1);
    fprintf('| TBR L2 (<54)          | %14.1f%% | %14.1f%% | %14.1f%% |       < 1%%      |\n', ...
        metrics_table(1).TBR_L2, metrics_table(2).TBR_L2, metrics_table(3).TBR_L2);
    fprintf('| TAR L1 (180-250)      | %14.1f%% | %14.1f%% | %14.1f%% |      < 25%%      |\n', ...
        metrics_table(1).TAR_L1, metrics_table(2).TAR_L1, metrics_table(3).TAR_L1);
    fprintf('| TAR L2 (>250)         | %14.1f%% | %14.1f%% | %14.1f%% |       < 5%%      |\n', ...
        metrics_table(1).TAR_L2, metrics_table(2).TAR_L2, metrics_table(3).TAR_L2);
    fprintf('+-----------------------+------------------+------------------+------------------+-----------------+\n');
    fprintf('| Glucosa media (mg/dL) | %15.1f  | %15.1f  | %15.1f  |   informativo   |\n', ...
        metrics_table(1).mean_g, metrics_table(2).mean_g, metrics_table(3).mean_g);
    fprintf('| SD (mg/dL)            | %15.1f  | %15.1f  | %15.1f  |   informativo   |\n', ...
        metrics_table(1).sd_g,   metrics_table(2).sd_g,   metrics_table(3).sd_g);
    fprintf('| CV (%%)                | %14.1f%% | %14.1f%% | %14.1f%% |     <= 36%%      |\n', ...
        metrics_table(1).CV,     metrics_table(2).CV,     metrics_table(3).CV);
    fprintf('| GMI (%%, ~HbA1c)       | %14.2f%% | %14.2f%% | %14.2f%% |      < 7%%       |\n', ...
        metrics_table(1).GMI,    metrics_table(2).GMI,    metrics_table(3).GMI);
    fprintf('+-----------------------+------------------+------------------+------------------+-----------------+\n');

    dTIR_no  = metrics_table(2).TIR - metrics_table(1).TIR;
    dTIR_pip = metrics_table(3).TIR - metrics_table(1).TIR;
    dCV_no   = metrics_table(2).CV  - metrics_table(1).CV;
    dCV_pip  = metrics_table(3).CV  - metrics_table(1).CV;
    dGMI_no  = metrics_table(2).GMI - metrics_table(1).GMI;
    dGMI_pip = metrics_table(3).GMI - metrics_table(1).GMI;
    fprintf('\n   DELTA TIR sin imputar: %+.1f pp | DELTA TIR con pipeline: %+.1f pp\n', ...
        dTIR_no, dTIR_pip);
    fprintf('   DELTA CV  sin imputar: %+.2f pp | DELTA CV  con pipeline: %+.2f pp\n', ...
        dCV_no, dCV_pip);
    fprintf('   DELTA GMI sin imputar: %+.3f pp | DELTA GMI con pipeline: %+.3f pp\n\n', ...
        dGMI_no, dGMI_pip);

    % ─────────────────────────────────────────────────────────────
    %  FIGURA 1: PERFIL GLUCÉMICO (primeros 5 días)
    % ─────────────────────────────────────────────────────────────

    N_SHOW = min(T, 5 * 1440 / DT_MINUTES);
    t_h    = (0:N_SHOW-1) * DT_MINUTES / 60;  % conversión samples -> horas 5 min/muestra
    % muestra × 5 min/muestra ÷ 60 min/hora = horas

    fig1 = figure('Name', sprintf('Perfil - %s->%s', upper(arch_det), upper(arch_imp)), ...
        'Position', [50, 100, 1400, 500], 'Color', 'white');
    hold on;

    % Zonas de riesgo (fondo)
    fill([t_h(1) t_h(end) t_h(end) t_h(1)], [0 0 THR_HYPO_L2 THR_HYPO_L2], ...
        [0.9 0.3 0.3], 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    fill([t_h(1) t_h(end) t_h(end) t_h(1)], [THR_HYPO_L2 THR_HYPO_L2 THR_HYPO_L1 THR_HYPO_L1], ...
        [1.0 0.6 0.2], 'FaceAlpha', 0.12, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    fill([t_h(1) t_h(end) t_h(end) t_h(1)], [THR_HYPO_L1 THR_HYPO_L1 THR_NORMAL_H THR_NORMAL_H], ...
        [0.3 0.8 0.3], 'FaceAlpha', 0.08, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    fill([t_h(1) t_h(end) t_h(end) t_h(1)], [THR_NORMAL_H THR_NORMAL_H THR_HYPER_L2 THR_HYPER_L2], ...
        [1.0 0.6 0.2], 'FaceAlpha', 0.10, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    fill([t_h(1) t_h(end) t_h(end) t_h(1)], [THR_HYPER_L2 THR_HYPER_L2 400 400], ...
        [0.9 0.3 0.3], 'FaceAlpha', 0.12, 'EdgeColor', 'none', 'HandleVisibility', 'off');

    % Señales
    plot(t_h, g_ref(1:N_SHOW), 'Color', [0.2 0.2 0.2], 'LineWidth', 1.5, ...
        'DisplayName', 'Señal real (referencia)');
    plot(t_h, g_no_imp(1:N_SHOW), 'Color', [0.85 0.3 0.3], 'LineWidth', 1.0, ...
        'LineStyle', '--', 'DisplayName', 'Sin imputación (gaps)');
    plot(t_h, g_pipe(1:N_SHOW), 'Color', [0.0 0.45 0.74], 'LineWidth', 1.5, ...
        'DisplayName', sprintf('Pipeline %s->%s', upper(arch_det), upper(arch_imp)));

    fi = find(d.true_f(1:N_SHOW));
    if ~isempty(fi)
        plot(t_h(fi), g_ref(fi), 'r.', 'MarkerSize', 5, ...
            'DisplayName', sprintf('Fallos (n=%d)', numel(fi)));
    end

    yline(THR_HYPO_L1, '--', 'Color', [0.8 0.4 0], 'LineWidth', 1.0, ...
        'Label', '70 mg/dL', 'LabelHorizontalAlignment', 'left', 'HandleVisibility', 'off');
    yline(THR_NORMAL_H, '--', 'Color', [0.8 0.4 0], 'LineWidth', 1.0, ...
        'Label', '180 mg/dL', 'LabelHorizontalAlignment', 'left', 'HandleVisibility', 'off');

    hold off;
    xlim([0 t_h(end)]); ylim([0 400]);
    xlabel('Tiempo (horas)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Glucosa (mg/dL)',  'FontSize', 12, 'FontWeight', 'bold');
    title(sprintf('Perfil Glucémico CGM — Pipeline %s -> %s', ...
        upper(arch_det), upper(arch_imp)), 'FontSize', 13, 'FontWeight', 'bold', 'Interpreter', 'none');
    legend('Location', 'northeast', 'FontSize', 10, 'Interpreter', 'none');
    grid on; set(gca, 'FontSize', 11);

    % ─────────────────────────────────────────────────────────────
    %  FIGURA 2: BARRAS APILADAS AGP
    % ─────────────────────────────────────────────────────────────

    fig2 = figure('Name', sprintf('AGP - %s->%s', upper(arch_det), upper(arch_imp)), ...
        'Position', [100, 100, 780, 520], 'Color', 'white');

    bd = zeros(3, 5);
    for s = 1:3
        bd(s,1) = metrics_table(s).TBR_L2;
        bd(s,2) = metrics_table(s).TBR_L1 - metrics_table(s).TBR_L2;
        bd(s,3) = metrics_table(s).TIR;
        bd(s,4) = metrics_table(s).TAR_L1;
        bd(s,5) = metrics_table(s).TAR_L2;
    end

    zone_colors = [
        0.85 0.20 0.20;
        0.95 0.60 0.10;
        0.25 0.70 0.35;
        0.95 0.60 0.10;
        0.85 0.20 0.20;
    ];

    bh = bar(bd, 'stacked', 'BarWidth', 0.55);
    for z = 1:5
        bh(z).FaceColor = zone_colors(z,:);
        bh(z).EdgeColor = 'white';
        bh(z).LineWidth = 1.5;
    end

    for s = 1:3
        text(s, 50, sprintf('TIR\n%.1f%%', metrics_table(s).TIR), ...
            'HorizontalAlignment', 'center', 'FontSize', 11, ...
            'FontWeight', 'bold', 'Color', 'white');
    end

    yline(70, '--k', 'TIR objetivo >=70%', 'LineWidth', 1.2, ...
        'LabelHorizontalAlignment', 'right', 'FontSize', 9);

    set(gca, 'XTickLabel', {'Referencia', 'Sin imputación', 'Con pipeline'}, ...
        'FontSize', 12);
    ylabel('Porcentaje de tiempo (%)', 'FontSize', 12, 'FontWeight', 'bold');
    title(sprintf('Tiempo en Rango (AGP) — %s -> %s', upper(arch_det), upper(arch_imp)), ...
        'FontSize', 13, 'FontWeight', 'bold', 'Interpreter', 'none');
    legend({'TBR L2 (<54)', 'TBR L1 (54-70)', 'TIR (70-180)', ...
        'TAR L1 (180-250)', 'TAR L2 (>250)'}, ...
        'Location', 'eastoutside', 'FontSize', 10);
    ylim([0 100]); grid on;

    % ─────────────────────────────────────────────────────────────
    %  GUARDAR FIGURAS
    % ─────────────────────────────────────────────────────────────

    f1_path = fullfile(output_path, sprintf('clinical_profile_%s_%s.png', combo_name, timestamp));
    f2_path = fullfile(output_path, sprintf('clinical_AGP_%s_%s.png',     combo_name, timestamp));
    exportgraphics(fig1, f1_path, 'Resolution', 300);
    exportgraphics(fig2, f2_path, 'Resolution', 300);
    fprintf('Perfil guardado: %s\n',   f1_path);
    fprintf('AGP guardado:    %s\n\n', f2_path);

    % Almacenar para resumen final
    all_clinical(c).combo_name    = combo_name;
    all_clinical(c).arch_det      = arch_det;
    all_clinical(c).arch_imp      = arch_imp;
    all_clinical(c).metrics       = metrics_table;
    all_clinical(c).dTIR_no_imp   = dTIR_no;
    all_clinical(c).dTIR_pipeline = dTIR_pip;
    all_clinical(c).dCV_no_imp    = dCV_no;
    all_clinical(c).dCV_pipeline  = dCV_pip;
    all_clinical(c).dGMI_no_imp   = dGMI_no;
    all_clinical(c).dGMI_pipeline = dGMI_pip;
    all_clinical(c).n_faults      = n_faults;
    all_clinical(c).pct_faults    = pct_faults;

end  % for c

%% ===================== TABLA RESUMEN FINAL (ambos combos) =====================

fprintf('\n');
fprintf('+-------------------+--------+--------+----------+--------+--------+----------+-----------+\n');
fprintf('|                   |     GRU -> GRU             |        GRU -> 1D-CNN       |           |\n');
fprintf('| Indicador         |  Ref.  | NoImp  | Pipeline |  Ref.  | NoImp  | Pipeline | Objetivo  |\n');
fprintf('+-------------------+--------+--------+----------+--------+--------+----------+-----------+\n');

% --- Bloque 1: indicadores de tiempo en rango (%) ---
metrics_names = {'TIR', 'TBR_L1', 'TBR_L2', 'TAR_L1', 'TAR_L2'};
row_labels    = {'TIR  (70-180)    ', 'TBR L1 (<70)    ', 'TBR L2 (<54)    ', ...
                 'TAR L1(180-250)  ', 'TAR L2 (>250)   '};
objectives    = {'>=70%', '<4%', '<1%', '<25%', '<5%'};

for i = 1:numel(metrics_names)
    mn = metrics_names{i};
    v  = zeros(2, 3);
    for c = 1:min(2, numel(all_clinical))
        if isfield(all_clinical(c), 'metrics') && numel(all_clinical(c).metrics) >= 3
            for s = 1:3
                v(c,s) = all_clinical(c).metrics(s).(mn);
            end
        end
    end
    fprintf('| %s | %5.1f%% | %5.1f%% |   %5.1f%%  | %5.1f%% | %5.1f%% |   %5.1f%%  |  %-8s |\n', ...
        row_labels{i}, v(1,1), v(1,2), v(1,3), v(2,1), v(2,2), v(2,3), objectives{i});
end

fprintf('+-------------------+--------+--------+----------+--------+--------+----------+-----------+\n');

% --- Bloque 2: estadística descriptiva y variabilidad ---
% mean_g y sd_g se imprimen en mg/dL (sin %), CV y GMI se imprimen en %.
extra_names  = {'mean_g', 'sd_g',   'CV',     'GMI'};
extra_labels = {'Media (mg/dL)    ', 'SD (mg/dL)      ', 'CV (variabilid.)', 'GMI (~HbA1c)    '};
extra_obj    = {'inform.', 'inform.', '<= 36%', '< 7%'};
extra_isPct  = [false,    false,    true,    true];
extra_fmt2   = {'%.1f', '%.1f', '%.2f'};  % formato base

for i = 1:numel(extra_names)
    mn = extra_names{i};
    v  = zeros(2, 3);
    for c = 1:min(2, numel(all_clinical))
        if isfield(all_clinical(c), 'metrics') && numel(all_clinical(c).metrics) >= 3
            for s = 1:3
                v(c,s) = all_clinical(c).metrics(s).(mn);
            end
        end
    end
    if extra_isPct(i)
        % CV y GMI: con símbolo de %
        fprintf('| %s | %5.1f%% | %5.1f%% |   %5.1f%%  | %5.1f%% | %5.1f%% |   %5.1f%%  |  %-8s |\n', ...
            extra_labels{i}, v(1,1), v(1,2), v(1,3), v(2,1), v(2,2), v(2,3), extra_obj{i});
    else
        % mean_g y sd_g: sin símbolo de %
        fprintf('| %s | %6.1f | %6.1f |  %7.1f | %6.1f | %6.1f |  %7.1f |  %-8s |\n', ...
            extra_labels{i}, v(1,1), v(1,2), v(1,3), v(2,1), v(2,2), v(2,3), extra_obj{i});
    end
end

fprintf('+-------------------+--------+--------+----------+--------+--------+----------+-----------+\n');

%% ===================== GUARDAR RESULTADOS =====================

out_file = fullfile(output_path, sprintf('clinical_metrics_results_%s.mat', timestamp));
save(out_file, 'all_clinical', 'timestamp');
fprintf('\nResultados guardados: %s\n', out_file);
fprintf('Analisis clinico completado.\n');