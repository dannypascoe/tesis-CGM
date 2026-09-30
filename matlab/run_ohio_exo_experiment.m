function T = run_ohio_exo_experiment(arms, ref_name)
%RUN_OHIO_EXO_EXPERIMENT  Etapa 2: pipeline completo por representacion de u y m.
%   Corre main_pipeline_ohio_v4.m una vez por brazo (todo lo demas identico:
%   mismos modelos congelados, mismas ventanas, mismas fallas) y compara.
%
%   T = run_ohio_exo_experiment()                     % brazos por defecto
%   T = run_ohio_exo_experiment({'ode','kernel'})     % subconjunto
%   T = run_ohio_exo_experiment({'ode', {'ode','impulse'}})   % u con ODE, m con impulsos
%
%   Brazos: 'neutral' (control, u = media_u y m = media_m), 'impulse',
%   'kernel', 'kernel_matched', 'ode', o una celda {repr_u, repr_m}.
%
%   Requisitos: scriptPath dentro de main_pipeline_ohio_v4.m y los archivos de
%   esta carpeta en el path. La configuracion de fallas (manual_faults,
%   run_label, etc.) se edita en main_pipeline_ohio_v4.m y vale para todos los
%   brazos. Cada corrida guarda sus propios archivos (sufijo _x<brazo>).
%
%   Nota tecnica: el script del pipeline hace 'clear' sobre el workspace base;
%   por eso este archivo es una funcion (su workspace no se toca) y el brazo se
%   pasa por appdata.

    if nargin < 1 || isempty(arms)
        arms = {'neutral', 'impulse', 'kernel', 'kernel_matched', 'ode'};
    end
    if nargin < 2, ref_name = 'ode'; end
    cleanup = onCleanup(@() clear_appdata()); %#ok<NASGU>

    nA = numel(arms);
    S = cell(nA, 1);  names = cell(nA, 1);
    for a = 1:nA
        arm = arms{a};
        if ischar(arm) && strcmp(arm, 'neutral')
            cfg = struct('input_mode', 'neutral', 'exo_repr', 'ode');
            names{a} = 'neutral';
        else
            cfg = struct('input_mode', 'real', 'exo_repr', []);
            cfg.exo_repr = arm;
            if iscell(arm), names{a} = strjoin(arm, '+'); else, names{a} = arm; end
        end
        fprintf('\n#################################################\n');
        fprintf('# BRAZO %d/%d: %s\n', a, nA, names{a});
        fprintf('#################################################\n');
        if isappdata(0, 'ohio_last_summary'), rmappdata(0, 'ohio_last_summary'); end
        setappdata(0, 'ohio_exo_arm', cfg);
        evalin('base', 'main_pipeline_ohio_v4');
        f = getappdata(0, 'ohio_last_summary');
        S{a} = load(f);
        S{a}.file = f;
    end

    clear_appdata();
    T = summarize_exo_arms(S, names, ref_name);

    out_dir = fileparts(S{1}.file);
    ts = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
    writetable(T, fullfile(out_dir, sprintf('ohio_exo_experiment_%s.xlsx', ts)));
    save(fullfile(out_dir, sprintf('ohio_exo_experiment_%s.mat', ts)), 'T', 'names', 'arms');
    fprintf('\nTabla guardada en %s\n', out_dir);
end

function clear_appdata()
    if isappdata(0, 'ohio_exo_arm'),     rmappdata(0, 'ohio_exo_arm');     end
    if isappdata(0, 'ohio_last_summary'), rmappdata(0, 'ohio_last_summary'); end
end
