%% CHECK_OHIO_INPUTS.M - Verificaciones previas (puntos 3 y 4)
%  1) Compara media/std de u y m guardadas en los modelos que entran en
%     Ohio (GRU, 1D-CNN) contra el default std_m_ref = 153.8637 de
%     build_ohio_meal_signal (valor copiado en su momento de CNN-LSTM).
%  2) Cuenta bolos extendidos (ts_end > ts_begin) en cada ventana de Ohio
%     preparada, para decidir si vale la pena repartirlos en el tiempo.
%  Solo diagnostico: no modifica ningun archivo.

clear; clc;

scriptPath  = 'D:\Documentos\02 MROI\07 Tesis\03 Extension Tesis\CGM_Project\CGM_Project\matlab'; % <<< tu raiz unificada
models_path = fullfile(scriptPath, 'data', 'models');
prep_path   = fullfile(scriptPath, 'data', 'ohio', 'prepared');

%% 1) Estadisticas de normalizacion por arquitectura
archs = {'gru', '1dcnn'};
STD_M_DEFAULT = 153.8637;
fprintf('=== Estadisticas de normalizacion guardadas en cada modelo ===\n');
fprintf('%-10s %12s %12s %12s %12s\n', 'arq', 'media_u', 'std_u', 'media_m', 'std_m');
for a = 1:numel(archs)
    f = fullfile(models_path, sprintf('trained_%s_latest.mat', archs{a}));
    d = load(f, 'media_u', 'std_u', 'media_m', 'std_m');
    fprintf('%-10s %12.4f %12.4f %12.4f %12.4f\n', archs{a}, ...
        d.media_u, d.std_u, d.media_m, d.std_m);
    fprintf('           diferencia std_m vs default: %+.4f\n', d.std_m - STD_M_DEFAULT);
end

%% 2) Bolos extendidos por paciente (solo ventana evaluada, sin pre-roll)
files = dir(fullfile(prep_path, 'paciente_*_latest.mat'));
fprintf('\n=== Bolos extendidos por ventana preparada ===\n');
for i = 1:numel(files)
    S = load(fullfile(files(i).folder, files(i).name), 'window_data', 'meta');
    b = S.window_data.bolus;
    if isempty(b.ts_begin)
        fprintf('%s: sin bolos\n', S.meta.patient_id);
        continue;
    end
    in_win = b.ts_begin >= S.window_data.start_ts;
    b.ts_begin = b.ts_begin(in_win); b.ts_end = b.ts_end(in_win);
    b.dose = b.dose(in_win); b.type = b.type(in_win);
    dur_min = minutes(b.ts_end - b.ts_begin);
    ext = dur_min > 5;
    fprintf('%s: %d bolos, %d extendidos (%.1f%% de la dosis total de bolo)\n', ...
        S.meta.patient_id, numel(b.ts_begin), sum(ext), ...
        100*sum(b.dose(ext))/max(sum(b.dose), eps));
    if any(ext)
        tipos = unique(b.type(ext));
        fprintf('   tipos: %s | duracion: %.0f a %.0f min\n', strjoin(tipos, ', '), ...
            min(dur_min(ext)), max(dur_min(ext)));
    end
end