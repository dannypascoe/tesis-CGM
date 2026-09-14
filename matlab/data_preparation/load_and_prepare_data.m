function [g_train, g_test, g_test_real, u, m, true_f] = load_and_prepare_data(data_path)
%LOAD_AND_PREPARE_DATA Carga los archivos .mat y prepara las variables
%
%   [g_train, g_test, g_test_real, u, m, true_f] = load_and_prepare_data(data_path)
%
%   Input:
%       data_path - Ruta a la carpeta con los archivos .mat
%
%   Outputs:
%       g_train     - Glucosa sin fallos (entrenamiento) [vector columna]
%       g_test      - Glucosa con fallos (prueba) [vector columna]
%       g_test_real - Glucosa sin fallos (referencia prueba) [vector columna]
%       u           - Insulina [vector columna]
%       m           - Comida/meals [vector columna]
%       true_f      - Etiquetas reales de fallas [vector fila]
%
%   Archivos requeridos en data_path:
%       - glucose.mat       (variable: glucosa)
%       - glucosaerror.mat  (variable: glucosaerror)
%       - glucose_sim.mat   (variable: glucose_sim)
%       - insulina.mat      (variable: insulina)
%       - meal.mat          (variable: meal)
%       - fallas_correct.mat (variable: true_f)

    % Verificar que la ruta existe
    if ~exist(data_path, 'dir')
        error('La ruta de datos no existe: %s', data_path);
    end

    % Cargar archivos de datos
    try
        % Glucosa de entrenamiento (sin fallos)
        data = load(fullfile(data_path, 'glucose.mat'));
        g_train = data.glucosa(:);
        
        % Glucosa de prueba (con fallos)
        data = load(fullfile(data_path, 'glucosaerror.mat'));
        g_test = data.glucosaerror(:);
        
        % Glucosa de referencia (sin fallos, para evaluación)
        data = load(fullfile(data_path, 'glucose_sim.mat'));
        g_test_real = data.glucose_sim(:);
        
        % Insulina
        data = load(fullfile(data_path, 'insulina.mat'));
        u = data.insulina(:);
        
        % Comida (meals)
        data = load(fullfile(data_path, 'meal.mat'));
        m = data.meal(:);
        
        % Etiquetas de fallas reales
        data = load(fullfile(data_path, 'fallas_correct.mat'));
        true_f = data.true_f(:)';
        
    catch ME
        error('Error al cargar datos: %s\nVerifique que todos los archivos .mat existen en: %s', ...
            ME.message, data_path);
    end

    % Validar que todas las señales tengan la misma longitud
    N = length(g_train);
    if length(u) ~= N || length(m) ~= N
        error('Las señales de entrenamiento tienen longitudes diferentes');
    end
    
    T = length(g_test);
    if length(g_test_real) ~= T || length(true_f) ~= T
        error('Las señales de prueba tienen longitudes diferentes');
    end

end
