%% CHECK_OHIO_INPUT_ARMS.M - Diagnostico previo al experimento de entradas
%  Construye u(k) y m(k) de cada ventana de Ohio con los cuatro valores de exo_repr de
%  load_and_prepare_data_ohio (impulse, kernel, kernel_matched, ode) y muestra:
%    - estadisticos en unidades fisicas y en el espacio normalizado con los
%      media_*/std_* de ENTRENAMIENTO (lo que realmente ve la red);
%    - correlacion de cada brazo con el brazo ODE;
%    - una figura por paciente para verificar a simple vista que los kernels
%      producen curvas suaves con niveles comparables al brazo ODE.
%  No corre ningun modelo ni modifica archivos (salvo guardar las figuras).

clear; clc; close all;

scriptPath  = 'D:\Documentos\02 MROI\07 Tesis\03 Extension Tesis\CGM_Project\CGM_Project\matlab'; % <<< tu raiz unificada
addpath(fullfile(scriptPath, 'architectures'));
addpath(fullfile(scriptPath, 'data_preparation'));
addpath(fullfile(scriptPath, 'utils'));
addpath(fullfile(scriptPath, 'data'));
models_path = fullfile(scriptPath, 'data', 'models');
prep_path   = fullfile(scriptPath, 'data', 'ohio', 'prepared');
fig_path    = fullfile(scriptPath, 'figures');
if ~exist(fig_path, 'dir'), mkdir(fig_path); end

patients = {'540', '559', '596'};
modes    = {'impulse', 'kernel', 'kernel_matched', 'ode'};   % exo_repr
arch     = 'gru';                     % estadisticos de normalizacion de referencia

ref = load(fullfile(models_path, sprintf('trained_%s_latest.mat', arch)), ...
    'media_u', 'std_u', 'media_m', 'std_m');
fprintf('Referencia de entrenamiento (%s): media_u=%.3f std_u=%.3f | media_m=%.3f std_m=%.3f\n\n', ...
    arch, ref.media_u, ref.std_u, ref.media_m, ref.std_m);

for p = 1:numel(patients)
    pid  = patients{p};
    file = fullfile(prep_path, sprintf('paciente_%s_ventana5d_latest.mat', pid));

    U = cell(1, numel(modes));  M = cell(1, numel(modes));  ts = [];
    for j = 1:numel(modes)
        [~, ~, U{j}, M{j}, ~, ts] = load_and_prepare_data_ohio(file, ref.std_m, modes{j});
    end
    iode = find(strcmp(modes, 'ode'));

    fprintf('=== Paciente %s ===\n', pid);
    fprintf('%-15s | %-31s | %-31s | %-13s\n', 'modo', 'u: media / std / max (fisico)', ...
        'u normalizada: media / std / max', 'corr(u,m) vs ODE');
    for j = 1:numel(modes)
        un = (U{j} - ref.media_u) / ref.std_u;
        mn = (M{j} - ref.media_m) / ref.std_m;
        cu = corrcoef(U{j}, U{iode});  cm = corrcoef(M{j}, M{iode});
        fprintf('%-15s | %9.3f %9.3f %9.3f | %9.3f %9.3f %9.3f | %5.2f  %5.2f\n', modes{j}, ...
            mean(U{j}), std(U{j}), max(U{j}), mean(un), std(un), max(un), cu(1,2), cm(1,2));
        fprintf('%-15s | m: %9.3f %9.3f %9.3f | m norm: %9.3f %9.3f %9.3f |\n', '', ...
            mean(M{j}), std(M{j}), max(M{j}), mean(mn), std(mn), max(mn));
    end
    fprintf('\n');

    % Figura: brazos con unidades fisicas comparables (sin 'impulse')
    fig = figure('Position', [100 100 1100 600], 'Name', ['Ohio ' pid]);
    subplot(2,1,1); hold on;
    for j = 2:numel(modes), plot(ts, U{j}, 'DisplayName', modes{j}); end
    ylabel('u(k) [mU/L]'); legend('Location', 'best'); grid on;
    title(sprintf('Paciente %s: entradas de insulina y comida por brazo', pid));
    subplot(2,1,2); hold on;
    for j = 2:numel(modes), plot(ts, M{j}, 'DisplayName', modes{j}); end
    ylabel('m(k) [escala calibrada]'); grid on;
    exportgraphics(fig, fullfile(fig_path, sprintf('ohio_input_arms_%s.png', pid)), 'Resolution', 200);
end
